//! Expression lowering: operators (trapping ones at their operator token's site),
//! casts, literals, paths, calls, methods (receiver adjustments of typeck), fields, indexing, `?`,
//! struct and variant literals, closures, f-strings (`String.new`, `push_str`,
//! `Display.fmt`), value `if`, ranges, and the implicit coercions typeck recorded.

use stave_ir::{CastKind, Op, Operand, PanicKind};
use stave_syntax::{NodeKind, Term};

use crate::lower_fn::{Base, E, FnCx, Place, Proj, Ty};
use crate::mono::{InstKey, Req, Resolved};
use crate::resolve::{DeferKind, Res};
use crate::sigs::TraitRef;
use crate::traits::Adjust;
use crate::typeck::{Coercion, Target};
use crate::types::{ConstArg, FloatBits, GArg, Interner, TypeId, TypeKind};

fn passthrough_kind(k: NodeKind) -> bool {
    matches!(
        k,
        NodeKind::Expr
            | NodeKind::RangeExpr
            | NodeKind::OrExpr
            | NodeKind::AndExpr
            | NodeKind::CmpExpr
            | NodeKind::BitOr
            | NodeKind::BitXor
            | NodeKind::BitAnd
            | NodeKind::Shift
            | NodeKind::Add
            | NodeKind::Mul
            | NodeKind::Cast
    )
}

/// A postfix chain's state between steps.
enum Cur {
    Path(Res, u32),
    /// A type-relative step (`T.m`); the call's target says what it is.
    TypeRel,
    Val(E),
}

/// Built-in binary operator of a token.
pub(crate) fn bin_op(t: u8) -> Option<(Op, bool)> {
    let m = |x: Term| x as u8 == t;
    Some(if m(Term::Plus) {
        (Op::Add, true)
    } else if m(Term::Minus) {
        (Op::Sub, true)
    } else if m(Term::Star) {
        (Op::Mul, true)
    } else if m(Term::Slash) {
        (Op::Div, true)
    } else if m(Term::Percent) {
        (Op::Rem, true)
    } else if m(Term::PlusPercent) {
        (Op::AddWrap, false)
    } else if m(Term::MinusPercent) {
        (Op::SubWrap, false)
    } else if m(Term::StarPercent) {
        (Op::MulWrap, false)
    } else if m(Term::PlusPipe) {
        (Op::AddSat, false)
    } else if m(Term::MinusPipe) {
        (Op::SubSat, false)
    } else if m(Term::StarPipe) {
        (Op::MulSat, false)
    } else if m(Term::LtLt) {
        (Op::Shl, true)
    } else if m(Term::GtGt) {
        (Op::Shr, true)
    } else if m(Term::LtLtPercent) {
        (Op::ShlWrap, false)
    } else if m(Term::Amp) {
        (Op::And, false)
    } else if m(Term::Pipe) {
        (Op::Or, false)
    } else if m(Term::Caret) {
        (Op::Xor, false)
    } else {
        return None;
    })
}

/// The built-in op of an operator trait name.
fn trait_op(name: &str) -> Option<(Op, bool)> {
    Some(match name {
        "Add" => (Op::Add, true),
        "Sub" => (Op::Sub, true),
        "Mul" => (Op::Mul, true),
        "Div" => (Op::Div, true),
        "Rem" => (Op::Rem, true),
        "WrapAdd" => (Op::AddWrap, false),
        "WrapSub" => (Op::SubWrap, false),
        "WrapMul" => (Op::MulWrap, false),
        "SatAdd" => (Op::AddSat, false),
        "SatSub" => (Op::SubSat, false),
        "SatMul" => (Op::MulSat, false),
        "Shl" => (Op::Shl, true),
        "WrapShl" => (Op::ShlWrap, false),
        "Shr" => (Op::Shr, true),
        "BitAnd" => (Op::And, false),
        "BitOr" => (Op::Or, false),
        "BitXor" => (Op::Xor, false),
        "Neg" => (Op::Neg, true),
        "Not" => (Op::Not, false),
        _ => return None,
    })
}

/// Splits an f-string token into literal segments and holes `(byte offset, name)*`.
pub(crate) enum FmtPart {
    Lit(Vec<u8>),
    Hole(Vec<(u32, String)>),
}

pub(crate) fn fmt_parts(text: &[u8], base: u32) -> Option<Vec<FmtPart>> {
    let mut out = Vec::new();
    let mut raw: Vec<u8> = Vec::new();
    let mut j = 2;
    let flush = |raw: &mut Vec<u8>, out: &mut Vec<FmtPart>| -> Option<()> {
        if !raw.is_empty() {
            let mut q = Vec::with_capacity(raw.len() + 2);
            q.push(b'"');
            q.extend_from_slice(raw);
            q.push(b'"');
            let s = std::str::from_utf8(&q).ok()?;
            out.push(FmtPart::Lit(crate::literals::parse_str(s)?));
            raw.clear();
        }
        Some(())
    };
    while j < text.len() {
        match text[j] {
            b'"' => break,
            b'\\' => {
                raw.extend_from_slice(text.get(j..j + 2)?);
                j += 2;
            }
            b'{' => {
                flush(&mut raw, &mut out)?;
                let mut k = j + 1;
                let mut segs = Vec::new();
                loop {
                    let s = k;
                    while k < text.len() && (text[k].is_ascii_alphanumeric() || text[k] == b'_') {
                        k += 1;
                    }
                    if k == s {
                        return None;
                    }
                    segs.push((base + s as u32, String::from_utf8_lossy(&text[s..k]).into_owned()));
                    match text.get(k) {
                        Some(b'.') => k += 1,
                        Some(b'}') => {
                            k += 1;
                            break;
                        }
                        _ => return None,
                    }
                }
                out.push(FmtPart::Hole(segs));
                j = k;
            }
            c => {
                raw.push(c);
                j += 1;
            }
        }
    }
    flush(&mut raw, &mut out)?;
    Some(out)
}

impl<'a> FnCx<'a> {
    /// The single child a pass-through node forwards to.
    pub fn passthrough(&self, n: u32) -> Option<u32> {
        let icx = self.icx;
        let k = icx.kind(n)?;
        if passthrough_kind(k) {
            return icx.only_child(n);
        }
        if k == NodeKind::Unary {
            let c = icx.first_child(n)?;
            if icx.kind(c) == Some(NodeKind::Postfix) && icx.a(c) == icx.a(n) {
                return Some(c);
            }
        }
        None
    }

    pub fn core(&self, n: u32) -> u32 {
        let mut cur = n;
        for _ in 0..1_000_000 {
            match self.passthrough(cur) {
                Some(c) => cur = c,
                None => break,
            }
        }
        cur
    }

    fn coercion_of(&self, n: u32) -> Option<&'a Coercion> {
        let tb = self.tb();
        // a coercion of `n`'s chain is keyed at `n` or a descendant: none in that range, none at
        // all (most expressions have none, and the chain is not walked for them)
        let lo = self.icx.ast.descendants(n).start;
        let k = tb.coercions.partition_point(|c| c.0 < lo);
        if tb.coercions.get(k).is_none_or(|c| c.0 > n) {
            return None;
        }
        let mut cur = n;
        for _ in 0..1_000_000 {
            if let Some(c) = tb.coercion(cur) {
                return Some(c);
            }
            match self.passthrough(cur) {
                Some(c) => cur = c,
                None => break,
            }
        }
        // `(e)` grouping
        let icx = self.icx;
        if icx.kind(cur) == Some(NodeKind::Postfix)
            && let Some(p) = icx.first_child(cur)
            && icx.postfix_end(cur) == icx.primary_end(p)
            && let Some(c) = tb.coercion(p)
        {
            return Some(c);
        }
        None
    }

    /// Lowers expression `n`; `want` is the expected type of a coercion site.
    pub fn expr(&mut self, n: u32, want: Option<Ty>) -> E {
        if !self.open() {
            return E::Never;
        }
        self.depth += 1;
        // The type checker's limit: a deeper body is E0005 and never reaches lowering.
        if self.depth > crate::typeck::MAX_DEPTH {
            self.depth -= 1;
            return self.fail(n, "expression nesting too deep");
        }
        let c = self.core(n);
        let e = self.expr_node(c, want);
        let e = match self.coercion_of(n) {
            Some(co) => self.coerce(e, co, want),
            None => e,
        };
        self.depth -= 1;
        e
    }

    /// The value of expression `n`.
    pub fn val(&mut self, n: u32, want: Option<Ty>) -> Option<Operand> {
        let e = self.expr(n, want);
        self.operand_of(e)
    }

    pub fn operand_of(&mut self, e: E) -> Option<Operand> {
        match e {
            E::V(o, _) => Some(o),
            E::Unit => Some(self.unit_val()),
            E::Never => None,
            E::P(p) => self.read(&p),
        }
    }

    pub fn e_ty(&mut self, e: &E) -> Ty {
        match e {
            E::V(_, t) => *t,
            E::P(p) => p.ty,
            E::Unit => self.t_unit(),
            E::Never => TypeId::NONE,
        }
    }

    pub fn place_of(&mut self, e: E) -> Option<Place> {
        match e {
            E::P(p) => Some(p),
            E::V(o, t) => Some(Place { base: Base::Val(o, t), proj: Vec::new(), ty: t, root: None }),
            E::Unit => {
                let u = self.unit_val();
                let t = self.t_unit();
                Some(Place { base: Base::Val(u, t), proj: Vec::new(), ty: t, root: None })
            }
            E::Never => None,
        }
    }

    /// An rvalue base of a droppable type becomes a statement temporary (so its unmoved parts
    /// are dropped at the end of the statement).
    fn own_temp(&mut self, p: Place) -> Place {
        if let Base::Val(v, t) = p.base
            && p.proj.is_empty()
            && self.needs_drop(t)
        {
            let (s, st) = self.spill(v, t, true);
            return Place { base: Base::Mem(s, st), proj: Vec::new(), ty: t, root: None };
        }
        p
    }

    fn coerce(&mut self, e: E, c: &Coercion, want: Option<Ty>) -> E {
        match c {
            Coercion::MutToShared => {
                let t = self.e_ty(&e);
                let Some(v) = self.operand_of(e) else { return E::Never };
                let Some((inner, _, raw)) = self.pointee(t) else { return E::V(v, t) };
                let w = match want {
                    Some(w) => w,
                    None if raw => self.t_ptr(false, inner),
                    None => self.t_ref(false, inner),
                };
                let r = self.convert_ptr(v, t, w);
                E::V(r, w)
            }
            Coercion::ArrayToSlice => {
                let t = self.e_ty(&e);
                let Some(v) = self.operand_of(e) else { return E::Never };
                let Some((inner, m, _)) = self.pointee(t) else { return E::V(v, t) };
                let TypeKind::Array { elem, len } = self.kind(inner) else { return E::V(v, t) };
                let n = len.value().unwrap_or(0);
                let w = match want {
                    Some(w) => w,
                    None => {
                        let s = self.tys.intern(TypeKind::Slice { elem, align: 0 });
                        self.t_ref(m, s)
                    }
                };
                let ep = self.t_ref(m, elem);
                let z = self.int(0);
                let p = self.op2(Op::ElemAddr, ep, v, z);
                let n = self.int(n as i128);
                let r = self.op2(Op::SliceMake, w, p, n);
                E::V(r, w)
            }
            Coercion::Unalign => {
                let t = self.e_ty(&e);
                let Some(v) = self.operand_of(e) else { return E::Never };
                let Some((inner, m, _)) = self.pointee(t) else { return E::V(v, t) };
                let TypeKind::Slice { elem, .. } = self.kind(inner) else { return E::V(v, t) };
                let w = match want {
                    Some(w) => w,
                    None => {
                        let s = self.tys.intern(TypeKind::Slice { elem, align: 0 });
                        self.t_ref(m, s)
                    }
                };
                let r = self.convert_ptr(v, t, w);
                E::V(r, w)
            }
            Coercion::StringToStr => {
                let t = self.e_ty(&e);
                let Some(v) = self.operand_of(e) else { return E::Never };
                let s = self.string_ty();
                let want_ref = self.t_ref(false, s);
                let v = self.convert_ptr(v, t, want_ref);
                self.as_str(v)
            }
            Coercion::FmtToStr => {
                let Some(p) = self.place_of(e) else { return E::Never };
                let s = self.string_ty();
                let ptr = match (&p.base, p.proj.is_empty()) {
                    (Base::Mem(m, _), true) if p.root.is_none() => {
                        self.f.temps.push((*m, s));
                        let r = self.t_ref(false, s);
                        let mt = self.t_ref(true, s);
                        self.convert_ptr(*m, mt, r)
                    }
                    _ => {
                        let Some(v) = self.read(&p) else { return E::Never };
                        let (m, mt) = self.spill(v, s, true);
                        let r = self.t_ref(false, s);
                        self.convert_ptr(m, mt, r)
                    }
                };
                self.as_str(ptr)
            }
            Coercion::Reborrow => {
                // `&mut [T; N]` reborrowed for a `&mut [T]` param: typeck records one coercion
                // per node, so the array-to-slice part is applied here
                let t = self.e_ty(&e);
                let to_slice = want.is_some_and(|w| w != t)
                    && self
                        .pointee(t)
                        .is_some_and(|(i, _, _)| matches!(self.kind(i), TypeKind::Array { .. }))
                    && want
                        .and_then(|w| self.pointee(w))
                        .is_some_and(|(i, _, _)| matches!(self.kind(i), TypeKind::Slice { .. }));
                // and `&mut [T align A]` for a `&mut [T]` param
                let unalign = want.is_some_and(|w| w != t)
                    && self.pointee(t).is_some_and(
                        |(i, _, _)| matches!(self.kind(i), TypeKind::Slice { align, .. } if align != 0),
                    );
                if to_slice {
                    self.coerce(e, &Coercion::ArrayToSlice, want)
                } else if unalign {
                    self.coerce(e, &Coercion::Unalign, want)
                } else {
                    e
                }
            }
            Coercion::NeverToAny | Coercion::FnItem => e,
        }
    }

    pub fn string_ty(&mut self) -> Ty {
        let s = self.g.sigs.lang.string;
        self.tys.mk_adt(s, Vec::new())
    }

    fn as_str(&mut self, p: Operand) -> E {
        let Some(m) = self.g.lang.string_as_str else {
            return self.fail_at(None, "std.string.String.as_str not found");
        };
        let tok = self.cur_tok;
        self.call_inst(InstKey { decl: m, args: Vec::new() }, &[p], tok)
    }

    fn expr_node(&mut self, n: u32, want: Option<Ty>) -> E {
        let Some(k) = self.nkind(n) else { return self.fail(n, "bad node") };
        match k {
            NodeKind::IfExpr => self.if_expr(n),
            NodeKind::MatchExpr => self.match_expr(n),
            NodeKind::RangeExpr => self.range_expr(n),
            NodeKind::OrExpr | NodeKind::AndExpr => self.logic(n),
            NodeKind::CmpExpr => self.compare(n),
            NodeKind::BitOr
            | NodeKind::BitXor
            | NodeKind::BitAnd
            | NodeKind::Shift
            | NodeKind::Add
            | NodeKind::Mul => self.binary_chain(n),
            NodeKind::Cast => self.cast_chain(n),
            NodeKind::Unary => self.unary(n, want),
            NodeKind::Postfix => self.postfix(n, want),
            NodeKind::Primary => self.primary(n, want),
            NodeKind::Expr => match self.icx.first_child(n) {
                Some(c) => self.expr(c, want),
                None => self.fail(n, "empty expression"),
            },
            _ => self.fail(n, "unexpected expression node"),
        }
    }

    // ---------- operators ----------

    fn logic(&mut self, n: u32) -> E {
        let kids = self.kids(n);
        let is_or = self.nkind(n) == Some(NodeKind::OrExpr);
        let bt = self.t_bool();
        let Some(&first) = kids.first() else { return self.fail(n, "empty") };
        // blocks of the operators, outermost (last) first: right, join
        let mut blocks = Vec::new();
        for _ in 1..kids.len() {
            let right = self.f.b.create_block();
            let j = self.join(n, &[bt]);
            blocks.push((right, j));
        }
        blocks.reverse();
        let Some(mut acc) = self.val(first, Some(bt)) else { return E::Never };
        for (k, &c) in kids.iter().enumerate().skip(1) {
            let (right, j) = blocks[k - 1];
            let short = self.int(if is_or { 1 } else { 0 });
            let args = self.edge(j, &[short]);
            let jb = self.f.joins[j].block;
            if is_or {
                self.f.b.cbr(acc, jb, &args, right, &[]);
            } else {
                self.f.b.cbr(acc, right, &[], jb, &args);
            }
            self.f.b.switch_to(right);
            if let Some(r) = self.val(c, Some(bt)) {
                self.br_join(j, &[r]);
            }
            self.enter(j);
            acc = Operand::Value(self.f.joins[j].prefix[0]);
        }
        E::V(acc, bt)
    }

    fn compare(&mut self, n: u32) -> E {
        let kids = self.kids(n);
        let (Some(&l), Some(&opn), Some(&r)) = (kids.first(), kids.get(1), kids.get(2)) else {
            return self.fail(n, "bad comparison");
        };
        let icx = self.icx;
        let op = icx.tk(icx.a(opn));
        let Some(a) = self.val(l, None) else { return E::Never };
        let Some(b) = self.val(r, None) else { return E::Never };
        let o = if op == Term::EqEq as u8 {
            Op::Eq
        } else if op == Term::BangEq as u8 {
            Op::Ne
        } else if op == Term::Lt as u8 {
            Op::Lt
        } else if op == Term::LtEq as u8 {
            Op::Le
        } else if op == Term::Gt as u8 {
            Op::Gt
        } else {
            Op::Ge
        };
        let rt = self.nty(n);
        if !matches!(self.kind(rt), TypeKind::Bool) {
            let ot = self.nty(l);
            return self.vec_compare(icx.a(opn), o, a, b, ot, rt);
        }
        E::V(self.op2(o, rt, a, b), rt)
    }

    fn binary_chain(&mut self, n: u32) -> E {
        let kids = self.kids(n);
        let ty = self.nty(n);
        let Some(&first) = kids.first() else { return self.fail(n, "empty") };
        let Some(mut acc) = self.val(first, None) else { return E::Never };
        for &c in &kids[1..] {
            let icx = self.icx;
            let optok = icx.a(c).saturating_sub(1);
            let op = icx.tk(optok);
            let Some(r) = self.val(c, None) else { return E::Never };
            match self.binop(optok, op, acc, r, ty) {
                Some(v) => acc = v,
                None => return E::Never,
            }
        }
        E::V(acc, ty)
    }

    /// `a op b` of type `ty` at operator token `tok`.
    pub fn binop(&mut self, tok: u32, op: u8, a: Operand, b: Operand, ty: Ty) -> Option<Operand> {
        if let Some(Target::Op { method, args }) = self.target(tok) {
            match self.resolve_method(method, args) {
                Some(Resolved::Inst(k)) => {
                    let e = self.call_inst(k, &[a, b], tok);
                    return self.operand_of(e);
                }
                Some(Resolved::BuiltinOp(name)) => {
                    let (o, trap) = trait_op(name)?;
                    return if trap {
                        self.trapping(tok, o, Some(ty), &[a, b])
                    } else {
                        Some(self.op2(o, ty, a, b))
                    };
                }
                None => {
                    self.fail_at(Some(tok), "unresolved operator method");
                    return None;
                }
            }
        }
        let Some((o, trap)) = bin_op(op) else {
            self.fail_at(Some(tok), "unknown operator");
            return None;
        };
        if self.is_simd(ty) {
            return self.vec_binop(tok, o, a, b, ty);
        }
        let float = matches!(self.kind(ty), TypeKind::Float(_));
        if trap && !float { self.trapping(tok, o, Some(ty), &[a, b]) } else { Some(self.op2(o, ty, a, b)) }
    }

    fn unary(&mut self, n: u32, want: Option<Ty>) -> E {
        let icx = self.icx;
        let a = icx.a(n);
        let t = icx.tk(a);
        let Some(c) = icx.last_child(n) else { return self.fail(n, "bad unary") };
        if t == Term::KwUnsafe as u8 {
            return self.expr(c, want);
        }
        if t == Term::Minus as u8 {
            if let Some(p) = self.down_to_primary(c)
                && icx.a(p) == a + 1
                && (icx.is(a + 1, Term::IntLit) || icx.is(a + 1, Term::FloatLit))
            {
                let ty = self.nty(n);
                return self.literal(a + 1, true, ty);
            }
            let ty = self.nty(n);
            let Some(v) = self.val(c, None) else { return E::Never };
            return self.unop(a, "Neg", Op::Neg, v, ty);
        }
        if t == Term::Bang as u8 {
            let bt = self.t_bool();
            let Some(v) = self.val(c, Some(bt)) else { return E::Never };
            return E::V(self.op1(Op::Not, bt, v), bt);
        }
        if t == Term::Tilde as u8 {
            let ty = self.nty(n);
            let Some(v) = self.val(c, None) else { return E::Never };
            return self.unop(a, "Not", Op::Not, v, ty);
        }
        if t == Term::Star as u8 {
            let e = self.expr(c, None);
            let pt = self.e_ty(&e);
            let Some(v) = self.operand_of(e) else { return E::Never };
            return E::P(self.deref(v, pt));
        }
        if t == Term::Amp as u8 || t == Term::AmpAmp as u8 {
            let rt = self.nty(n);
            let e = self.expr(c, None);
            let Some(p) = self.place_of(e) else { return E::Never };
            let p = match p.base {
                Base::Val(..) => self.own_temp(p),
                _ => p,
            };
            let Some(v) = self.addr(&p, Some(rt)) else { return E::Never };
            return E::V(v, rt);
        }
        self.expr(c, want)
    }

    fn unop(&mut self, tok: u32, trait_name: &str, op: Op, v: Operand, ty: Ty) -> E {
        if let Some(Target::Op { method, args }) = self.target(tok) {
            return match self.resolve_method(method, args) {
                Some(Resolved::Inst(k)) => self.call_inst(k, &[v], tok),
                Some(Resolved::BuiltinOp(_)) => self.builtin_unop(tok, op, v, ty),
                None => self.fail_at(Some(tok), "unresolved operator method"),
            };
        }
        let _ = trait_name;
        self.builtin_unop(tok, op, v, ty)
    }

    fn builtin_unop(&mut self, tok: u32, op: Op, v: Operand, ty: Ty) -> E {
        if self.is_simd(ty) {
            return self.vec_unop(tok, op, v, ty);
        }
        let r = if op == Op::Neg && matches!(self.kind(ty), TypeKind::Int { .. }) {
            match self.trapping(tok, op, Some(ty), &[v]) {
                Some(r) => r,
                None => return E::Never,
            }
        } else {
            self.op1(op, ty, v)
        };
        E::V(r, ty)
    }

    /// The Primary below pass-through wrappers and suffix-less Postfix nodes.
    pub fn down_to_primary(&self, n: u32) -> Option<u32> {
        let icx = self.icx;
        let mut cur = n;
        for _ in 0..1_000_000 {
            if let Some(c) = self.passthrough(cur) {
                cur = c;
                continue;
            }
            match icx.kind(cur)? {
                NodeKind::Postfix => {
                    let p = icx.first_child(cur)?;
                    if icx.postfix_end(cur) != icx.primary_end(p) {
                        return None;
                    }
                    cur = p;
                }
                NodeKind::Primary => {
                    let a = icx.a(cur);
                    if icx.is(a, Term::LParen) {
                        let kids = icx.kids_vec(cur);
                        let close = icx.close(a);
                        if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
                            cur = kids[0];
                            continue;
                        }
                        return None;
                    }
                    return Some(cur);
                }
                _ => return None,
            }
        }
        None
    }

    fn cast_chain(&mut self, n: u32) -> E {
        let kids = self.kids(n);
        let Some(&first) = kids.first() else { return self.fail(n, "bad cast") };
        let e = self.expr(first, None);
        let mut from = self.e_ty(&e);
        let Some(mut acc) = self.operand_of(e) else { return E::Never };
        let last = kids.len().saturating_sub(1);
        for (k, &tn) in kids.iter().enumerate().skip(1) {
            let icx = self.icx;
            let optok = icx.a(tn).saturating_sub(1);
            let op = icx.tk(optok);
            let kind = if op == Term::AsPercent as u8 {
                CastKind::Wrap
            } else if op == Term::AsPipe as u8 {
                CastKind::Sat
            } else if op == Term::AsTilde as u8 {
                CastKind::Round
            } else if op == Term::AsQuestion as u8 {
                CastKind::Try
            } else {
                CastKind::Exact
            };
            let rt = if k == last {
                self.nty(n)
            } else {
                let to = self.lower_type(tn);
                if kind == CastKind::Try { self.option_of(to) } else { to }
            };
            if rt.is_none() {
                return self.fail(tn, "cast type");
            }
            if kind == CastKind::Exact && from == rt && self.is_simd(from) {
                continue;
            }
            if matches!(self.kind(from), TypeKind::Vector { .. }) {
                let Some(r) = self.vec_cast(optok, kind, acc, rt) else { return E::Never };
                acc = r;
                from = rt;
                continue;
            }
            // `&x as *const T` (thin ref to raw pointer) and fat refs to fat raw pointers
            let fat_from = self
                .pointee(from)
                .is_some_and(|(i, _, _)| matches!(self.kind(i), TypeKind::Slice { .. } | TypeKind::Str));
            acc = if fat_from { self.convert_ptr(acc, from, rt) } else { self.op1(Op::Cast(kind), rt, acc) };
            from = rt;
        }
        E::V(acc, from)
    }

    pub fn option_of(&mut self, t: Ty) -> Ty {
        let o = self.g.sigs.lang.option;
        self.tys.mk_adt(o, vec![GArg::Type(t)])
    }

    fn range_expr(&mut self, n: u32) -> E {
        let kids = self.kids(n);
        let (Some(&l), Some(&r)) = (kids.first(), kids.get(1)) else { return self.fail(n, "bad range") };
        let ty = self.nty(n);
        let Some(a) = self.val(l, None) else { return E::Never };
        let Some(b) = self.val(r, None) else { return E::Never };
        E::V(self.opn(Op::Make, ty, &[a, b]), ty)
    }

    fn if_expr(&mut self, n: u32) -> E {
        let kids = self.kids(n);
        let (Some(&c), Some(&a), Some(&b)) = (kids.first(), kids.get(1), kids.get(2)) else {
            return self.fail(n, "bad if");
        };
        let ty = self.nty(n);
        let then = self.f.b.create_block();
        let els = self.f.b.create_block();
        let vt = if ty.is_none() || self.is_never(ty) { self.t_unit() } else { ty };
        let j = self.join(n, &[vt]);
        let bt = self.t_bool();
        let Some(cv) = self.val(c, Some(bt)) else { return E::Never };
        self.f.b.cbr(cv, then, &[], els, &[]);
        let snap = self.f.env.clone();
        self.f.b.switch_to(then);
        if let Some(v) = self.val(a, Some(vt)) {
            self.br_join(j, &[v]);
        }
        self.f.env = snap;
        self.f.b.switch_to(els);
        if let Some(v) = self.val(b, Some(vt)) {
            self.br_join(j, &[v]);
        }
        let reached = self.f.joins[j].st.is_some();
        self.enter(j);
        if !reached {
            self.f.b.unreachable();
            return E::Never;
        }
        E::V(Operand::Value(self.f.joins[j].prefix[0]), vt)
    }

    // ---------- literals and primaries ----------

    /// The literal at token `tok` (`neg`: folded unary minus) of type `ty`.
    pub fn literal(&mut self, tok: u32, neg: bool, ty: Ty) -> E {
        let icx = self.icx;
        let text = icx.tok_str(tok);
        let k = self.kind(ty);
        if icx.is(tok, Term::IntLit) && !matches!(k, TypeKind::Float(_)) {
            let Some((mag, _)) = crate::literals::parse_int(text) else {
                return self.fail_at(Some(tok), "malformed literal");
            };
            let v = if neg { -(mag as i128) } else { mag as i128 };
            return E::V(self.int(v), ty);
        }
        // float (or an int literal with a float suffix)
        let body: String = text.chars().filter(|&c| c != '_').collect();
        let cut = body.find(['f', 'b']).unwrap_or(body.len());
        let digits = &body[..cut];
        match k {
            TypeKind::Float(FloatBits::F32) => {
                let Ok(x) = digits.parse::<f32>() else {
                    return self.fail_at(Some(tok), "malformed literal");
                };
                let x = if neg { -x } else { x };
                E::V(self.f.b.f32(x), ty)
            }
            TypeKind::Float(FloatBits::F64) => {
                let Ok(x) = digits.parse::<f64>() else {
                    return self.fail_at(Some(tok), "malformed literal");
                };
                let x = if neg { -x } else { x };
                E::V(self.f.b.f64(x), ty)
            }
            _ => self.unsupported_at(Some(tok), "f16/bf16 literals are not supported by the IR yet"),
        }
    }

    fn primary(&mut self, n: u32, want: Option<Ty>) -> E {
        let icx = self.icx;
        let a = icx.a(n);
        let t = icx.tk(a);
        if t == Term::IntLit as u8 || t == Term::FloatLit as u8 {
            let ty = self.nty(n);
            return self.literal(a, false, ty);
        }
        if t == Term::KwTrue as u8 || t == Term::KwFalse as u8 {
            let bt = self.t_bool();
            return E::V(self.int(i128::from(t == Term::KwTrue as u8)), bt);
        }
        if t == Term::CharLit as u8 {
            let Some(c) = crate::literals::parse_char(icx.tok_str(a)) else {
                return self.fail(n, "malformed char");
            };
            let ct = self.tys.intern(TypeKind::Char);
            return E::V(self.int(c as i128), ct);
        }
        if t == Term::StrLit as u8 {
            let Some(bytes) = crate::literals::parse_str(icx.tok_str(a)) else {
                return self.fail(n, "malformed string");
            };
            let s = self.string(&bytes);
            let st = self.tys.intern(TypeKind::Str);
            let rt = self.t_ref(false, st);
            return E::V(Operand::Str(s), rt);
        }
        if t == Term::FmtLit as u8 {
            return self.fstring(n, a);
        }
        if t == Term::KwSelf as u8 {
            return match icx.uses(n).first().map(|u| u.res).and_then(|r| self.local_of(r)) {
                Some(l) => E::P(self.local_place(l)),
                None => self.fail(n, "unresolved self"),
            };
        }
        if t == Term::LParen as u8 {
            return self.paren(n, want);
        }
        if t == Term::LBracket as u8 {
            return self.array_lit(n);
        }
        if t == Term::KwFn as u8 {
            return self.closure(n);
        }
        // a path without suffix is a Postfix; this is a bare primary used directly
        self.postfix_primary(n)
    }

    fn postfix_primary(&mut self, n: u32) -> E {
        let icx = self.icx;
        let a = icx.a(n);
        match icx.use_at(n, a) {
            Some(u) => self.path_val(u.res, a, n),
            None => self.fail(n, "unresolved name"),
        }
    }

    fn paren(&mut self, n: u32, want: Option<Ty>) -> E {
        let icx = self.icx;
        let a = icx.a(n);
        let close = icx.close(a);
        let kids = self.kids(n);
        if kids.is_empty() {
            return E::Unit;
        }
        if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
            return self.expr(kids[0], want);
        }
        let ty = self.nty(n);
        let ets: Vec<Ty> = match self.kind(ty) {
            TypeKind::Tuple(ts) => ts.to_vec(),
            _ => vec![TypeId::NONE; kids.len()],
        };
        let mut vs = Vec::with_capacity(kids.len());
        for (k, &c) in kids.iter().enumerate() {
            let w = ets.get(k).copied().filter(|t| !t.is_none());
            let Some(v) = self.val(c, w) else { return E::Never };
            vs.push(v);
        }
        E::V(self.opn(Op::Make, ty, &vs), ty)
    }

    fn array_lit(&mut self, n: u32) -> E {
        let icx = self.icx;
        let kids = self.kids(n);
        let ty = self.nty(n);
        let TypeKind::Array { elem, len } = self.kind(ty) else { return self.fail(n, "array type") };
        if kids.len() == 2 && icx.is(icx.end_tok(kids[0]) + 1, Term::Semi) {
            let Some(v) = self.val(kids[0], Some(elem)) else { return E::Never };
            let Some(len) = len.value() else { return self.fail(n, "array length") };
            if len > (1 << 20) {
                return self.fail(n, "array repeat literal too long for the IR (no splat op)");
            }
            let vs = vec![v; len as usize];
            return E::V(self.opn(Op::Make, ty, &vs), ty);
        }
        let mut vs = Vec::with_capacity(kids.len());
        for &c in &kids {
            let Some(v) = self.val(c, Some(elem)) else { return E::Never };
            vs.push(v);
        }
        E::V(self.opn(Op::Make, ty, &vs), ty)
    }

    // ---------- paths ----------

    /// The value of a path that ends at `tok` (the chain's node `node`).
    fn path_val(&mut self, res: Res, tok: u32, node: u32) -> E {
        let icx = self.icx;
        if let Some(l) = self.local_of(res) {
            return E::P(self.local_place(l));
        }
        match res {
            Res::Generic(g) if g.is_const => {
                let sig_np = self.cur_n_parent;
                let idx = if g.level == 0 { g.index as u32 } else { sig_np + g.index as u32 };
                let ct = self.cur_generic_const_ty(idx);
                match self.args.get(idx as usize) {
                    Some(GArg::Const(ConstArg::Value(v))) => {
                        let v = *v as i128;
                        E::V(self.int(v), ct)
                    }
                    _ => self.fail_at(Some(tok), "unbound const generic"),
                }
            }
            Res::Decl(d) => match self.decl_kind(d) {
                Some(crate::decls::DeclKind::Const) => {
                    let Some(c) = self.g.sigs.konst(d) else { return self.fail_at(Some(tok), "const") };
                    let (ty, Some(v)) = (c.ty, c.value.clone()) else {
                        return self.fail_at(Some(tok), "const without a value");
                    };
                    self.const_val(&v, ty, tok)
                }
                Some(crate::decls::DeclKind::Static) => self.static_place(d, tok),
                Some(crate::decls::DeclKind::Fn | crate::decls::DeclKind::Method) => {
                    let end = icx.end_tok(node);
                    let tgt = self.target(end).or_else(|| self.target(tok)).cloned();
                    match tgt {
                        Some(Target::Fn { decl, args }) => {
                            let args: Vec<GArg> = args.iter().map(|a| self.sub_garg(*a)).collect();
                            let ty = self.nty(node);
                            let f = self.req(Req::Inst(InstKey { decl, args }));
                            E::V(Operand::Func(f), ty)
                        }
                        _ => self.fail_at(Some(tok), "fn value without a target"),
                    }
                }
                Some(crate::decls::DeclKind::Variant) => {
                    let end = icx.end_tok(node);
                    let idx = match self.target(end).or_else(|| self.target(tok)) {
                        Some(Target::Variant { index, .. }) => *index,
                        _ => self.g.prog.decl(d).map_or(0, |x| x.index),
                    };
                    let ty = self.nty(node);
                    if !matches!(self.kind(ty), TypeKind::Adt { .. }) {
                        return self.fail_at(Some(tok), "variant value type");
                    }
                    let i = self.int(idx as i128);
                    E::V(self.opn(Op::EnumMake, ty, &[i]), ty)
                }
                _ => self.fail_at(Some(tok), "not a value"),
            },
            _ => self.fail_at(Some(tok), "not a value"),
        }
    }

    fn static_place(&mut self, d: crate::ids::DeclId, tok: u32) -> E {
        let Some(&idx) = self.g.statics.get(&d) else {
            return self.fail_at(Some(tok), "static without a value");
        };
        let Some(c) = self.g.sigs.konst(d) else { return self.fail_at(Some(tok), "static") };
        let ty = c.ty;
        let pt = self.t_ptr(true, ty);
        let id = self.intrinsic("static_addr");
        let i = self.int(idx as i128);
        let p = Operand::Value(self.f.b.op(stave_ir::Op::Intrinsic(id), FnCx::ir(pt), &[i]));
        E::P(Place { base: Base::Mem(p, pt), proj: Vec::new(), ty, root: None })
    }

    /// A const value of type `ty`.
    pub fn const_val(&mut self, v: &crate::consteval::Val, ty: Ty, tok: u32) -> E {
        use crate::consteval::Val;
        match v {
            Val::Int(i) => match self.kind(ty) {
                TypeKind::Float(FloatBits::F32) => E::V(self.f.b.f32(*i as f32), ty),
                TypeKind::Float(FloatBits::F64) => E::V(self.f.b.f64(*i as f64), ty),
                _ => E::V(self.int(*i), ty),
            },
            Val::Float(x) => match self.kind(ty) {
                TypeKind::Float(FloatBits::F32) => E::V(self.f.b.f32(*x as f32), ty),
                TypeKind::Float(FloatBits::F64) => E::V(self.f.b.f64(*x), ty),
                _ => self.unsupported_at(Some(tok), "f16/bf16 constants are not supported"),
            },
            Val::Bool(b) => E::V(self.int(i128::from(*b)), ty),
            Val::Char(c) => E::V(self.int(*c as i128), ty),
            Val::Str(bytes) => {
                let s = self.string(bytes);
                E::V(Operand::Str(s), ty)
            }
            Val::Agg(vs) => {
                if vs.is_empty() && self.is_unit(ty) {
                    return E::Unit;
                }
                let ets: Vec<Ty> = match self.kind(ty) {
                    TypeKind::Tuple(ts) => ts.to_vec(),
                    TypeKind::Array { elem, .. } => vec![elem; vs.len()],
                    TypeKind::Adt { .. } => {
                        self.adt_fields(ty).and_then(|v| v.into_iter().next()).unwrap_or_default()
                    }
                    _ => return self.fail_at(Some(tok), "const aggregate type"),
                };
                let mut ops = Vec::with_capacity(vs.len());
                for (x, &t) in vs.iter().zip(&ets) {
                    let e = self.const_val(x, t, tok);
                    let Some(o) = self.operand_of(e) else { return E::Never };
                    ops.push(o);
                }
                E::V(self.opn(Op::Make, ty, &ops), ty)
            }
            Val::Variant(vi, fs) => {
                let fts = self.adt_fields(ty).and_then(|v| v.get(*vi as usize).cloned()).unwrap_or_default();
                let mut ops = vec![self.int(*vi as i128)];
                for (x, &t) in fs.iter().zip(&fts) {
                    let e = self.const_val(x, t, tok);
                    let Some(o) = self.operand_of(e) else { return E::Never };
                    ops.push(o);
                }
                E::V(self.opn(Op::EnumMake, ty, &ops), ty)
            }
        }
    }

    // ---------- postfix chains ----------

    fn postfix(&mut self, n: u32, want: Option<Ty>) -> E {
        let icx = self.icx;
        let kids = self.kids(n);
        let Some(&prim) = kids.first() else { return self.fail(n, "empty postfix") };
        let pa = icx.a(prim);
        let pend = icx.postfix_end(n);
        let prim_end = icx.primary_end(prim);
        let only = pend == prim_end;
        let mut kid = 1usize;
        let mut cur = if icx.is(pa, Term::Ident) {
            match icx.use_at(prim, pa) {
                Some(u) => Cur::Path(u.res, pa),
                None => match icx.deferred_at(prim, pa) {
                    Some(_) => Cur::TypeRel,
                    None => return self.fail(n, "unresolved name"),
                },
            }
        } else if icx.is(pa, Term::KwSelfType) {
            Cur::Path(Res::SelfType, pa)
        } else {
            Cur::Val(self.expr(prim, if only { want } else { None }))
        };
        let mut p = prim_end + 1;
        for _ in 0..10_000_000 {
            if !self.open() {
                return E::Never;
            }
            let t = icx.tk(p);
            if t == Term::Dot as u8 && icx.is(p + 1, Term::Ident) {
                let seg = p + 1;
                let mut q = p + 2;
                if icx.is(q, Term::ColonColon) {
                    kid += 1;
                    let (e, partial) = icx.angle_end(q + 1);
                    q = if partial { e - 1 } else { e };
                }
                if !matches!(cur, Cur::Val(_))
                    && let Some(u) = icx.use_at(prim, seg)
                {
                    cur = Cur::Path(u.res, seg);
                    p = q;
                    continue;
                }
                if !matches!(cur, Cur::Val(_))
                    && let Some(d) = icx.deferred_at(prim, seg)
                    && d.kind == DeferKind::TypeRelative
                {
                    cur = Cur::TypeRel;
                    p = q;
                    continue;
                }
                let recv = self.cur_value(cur, prim);
                if matches!(recv, E::Never) {
                    return E::Never;
                }
                if icx.is(q, Term::LParen) {
                    let close = icx.close(q);
                    let args = if close == q + 1 {
                        crate::itemcx::Kids::default()
                    } else {
                        let a = kids.get(kid).copied();
                        kid += 1;
                        a.map(|a| icx.kids_vec(a)).unwrap_or_default()
                    };
                    let last = close == pend;
                    cur = Cur::Val(self.method_call(n, recv, seg, &args, last));
                    p = close + 1;
                } else {
                    cur = Cur::Val(self.field_step(recv, seg));
                    p = q;
                }
                continue;
            }
            if t == Term::Dot as u8 && icx.is(p + 1, Term::IntLit) {
                let recv = self.cur_value(cur, prim);
                cur = Cur::Val(self.field_step(recv, p + 1));
                p += 2;
                continue;
            }
            if t == Term::DotLBrace as u8 {
                let close = icx.close(p);
                let sl = kids.get(kid).copied();
                kid += 1;
                let last = close == pend;
                cur = Cur::Val(self.struct_lit(n, sl, p, last));
                p = close + 1;
                continue;
            }
            if t == Term::LParen as u8 {
                let close = icx.close(p);
                let args = if close == p + 1 {
                    crate::itemcx::Kids::default()
                } else {
                    let a = kids.get(kid).copied();
                    kid += 1;
                    a.map(|a| icx.kids_vec(a)).unwrap_or_default()
                };
                let last = close == pend;
                cur = Cur::Val(self.call(n, cur, prim, p, &args, last));
                p = close + 1;
                continue;
            }
            if t == Term::LBracket as u8 {
                let close = icx.close(p);
                let ie = kids.get(kid).copied();
                kid += 1;
                let base = self.cur_value(cur, prim);
                cur = Cur::Val(self.index_step(base, ie, p));
                p = close + 1;
                continue;
            }
            if t == Term::Question as u8 {
                let base = self.cur_value(cur, prim);
                cur = Cur::Val(self.try_step(n, base, p));
                p += 1;
                continue;
            }
            break;
        }
        match cur {
            Cur::Val(e) => e,
            Cur::Path(res, tok) => self.path_val(res, tok, if only { prim } else { n }),
            Cur::TypeRel => self.fail(n, "type-relative path is not a value"),
        }
    }

    fn cur_value(&mut self, cur: Cur, prim: u32) -> E {
        match cur {
            Cur::Val(e) => e,
            Cur::Path(res, tok) => self.path_val(res, tok, prim),
            Cur::TypeRel => self.fail(prim, "type-relative path is not a value"),
        }
    }

    /// Applies `derefs` auto-derefs to a place.
    fn derefs(&mut self, mut p: Place, k: u32) -> Option<Place> {
        for _ in 0..k {
            let v = self.read(&p)?;
            let t = p.ty;
            p = self.deref(v, t);
        }
        Some(p)
    }

    fn field_step(&mut self, recv: E, tok: u32) -> E {
        let (index, derefs) = match self.target(tok) {
            Some(Target::Field { index, derefs, .. }) => (*index, *derefs),
            Some(Target::TupleField { index, derefs }) => (*index, *derefs),
            _ => return self.fail_at(Some(tok), "field without a target"),
        };
        let Some(p) = self.place_of(recv) else { return E::Never };
        let p = self.own_temp(p);
        let Some(mut p) = self.derefs(p, derefs) else { return E::Never };
        let ft = match self.kind(p.ty) {
            TypeKind::Tuple(ts) => ts.get(index as usize).copied(),
            TypeKind::Range { elem, .. } => Some(elem),
            TypeKind::Adt { .. } => {
                self.adt_fields(p.ty).and_then(|v| v.first().and_then(|f| f.get(index as usize).copied()))
            }
            _ => None,
        };
        let Some(ft) = ft else {
            let k = self.kind(p.ty);
            return self.fail_at(Some(tok), format!("field type of {k:?}"));
        };
        p.proj.push(Proj::Field(index, ft));
        p.ty = ft;
        E::P(p)
    }

    fn index_step(&mut self, base: E, ie: Option<u32>, tok: u32) -> E {
        let derefs = match self.target(tok) {
            Some(Target::Index { derefs }) => *derefs,
            Some(Target::Slice { derefs }) => {
                let d = *derefs;
                return self.slice_step(base, ie, tok, d);
            }
            _ => return self.fail_at(Some(tok), "index without a target"),
        };
        let Some(ie) = ie else { return self.fail_at(Some(tok), "index") };
        let Some(p) = self.place_of(base) else { return E::Never };
        let p = self.own_temp(p);
        let Some(mut p) = self.derefs(p, derefs) else { return E::Never };
        let us = self.t_usize();
        let Some(i) = self.val(ie, Some(us)) else { return E::Never };
        let (et, len) = match self.kind(p.ty) {
            TypeKind::Array { elem, len } => (elem, len.value()),
            TypeKind::Slice { elem, .. } => (elem, None),
            _ => return self.fail_at(Some(tok), "indexing a non-array"),
        };
        if let (Some(k), Some(n)) = (self.kval(i), len)
            && (k < 0 || k as u128 >= n as u128)
        {
            self.trapping(tok, Op::Panic(PanicKind::Bounds), None, &[]);
            self.f.b.unreachable();
            return E::Never;
        }
        p.proj.push(Proj::Elem(i, et, tok));
        p.ty = et;
        E::P(p)
    }

    /// `xs[a..b]` (`xs[a..=b]`: end `b + 1`): the slice place of elements `a..end` of an array
    /// or slice of length `n`, after the tests `end <= n` (inclusive: `b < n`, so `b + 1` cannot
    /// overflow) and `a <= end`, each a branch to `panic.bounds`. Constant operands are tested
    /// here: a range out of a constant length is an unconditional `panic.bounds`.
    fn slice_step(&mut self, base: E, ie: Option<u32>, tok: u32, derefs: u32) -> E {
        let Some(ie) = ie else { return self.fail_at(Some(tok), "slice") };
        let Some(p) = self.place_of(base) else { return E::Never };
        let p = self.own_temp(p);
        let Some(p) = self.derefs(p, derefs) else { return E::Never };
        let rt = self.nty(ie);
        let inclusive = matches!(self.kind(rt), TypeKind::Range { inclusive: true, .. });
        let core = self.core(ie);
        if self.icx.kind(core) != Some(NodeKind::RangeExpr) {
            return self.fail_at(Some(tok), "slice without a range");
        }
        let rk = self.kids(core);
        let (Some(&l), Some(&r)) = (rk.first(), rk.get(1)) else {
            return self.fail_at(Some(tok), "slice range");
        };
        let us = self.t_usize();
        let Some(a) = self.val(l, Some(us)) else { return E::Never };
        let Some(b) = self.val(r, Some(us)) else { return E::Never };
        let (elem, len) = match self.kind(p.ty) {
            TypeKind::Array { elem, len } => (elem, len.value()),
            TypeKind::Slice { elem, .. } => (elem, None),
            _ => return self.fail_at(Some(tok), "slicing a non-array"),
        };
        let (ka, kb, kn) = (self.kval(a), self.kval(b), len.map(i128::from));
        let ke = kb.map(|k| k + i128::from(inclusive));
        // `xs[a..=usize::MAX]` never fits: its end is not a `usize`
        let out = matches!((ke, kn), (Some(e), Some(n)) if e > n)
            || matches!((ka, ke), (Some(a), Some(e)) if a > e)
            || ke.is_some_and(|e| e > i128::from(u64::MAX));
        if out {
            self.trapping(tok, Op::Panic(PanicKind::Bounds), None, &[]);
            self.f.b.unreachable();
            return E::Never;
        }
        let Some((bp, bt)) = self.addr_t(&p, None) else { return E::Never };
        let n = match kn {
            Some(k) => self.int(k),
            None => self.op1(Op::SliceLen, us, bp),
        };
        let end = if inclusive {
            if kb.is_none() || kn.is_none() {
                self.trap_if(Op::Ge, b, n, tok);
            }
            match ke {
                Some(k) => self.int(k),
                None => {
                    let one = self.int(1);
                    let Some(e) = self.trapping(tok, Op::Add, Some(us), &[b, one]) else { return E::Never };
                    e
                }
            }
        } else {
            if kb.is_none() || kn.is_none() {
                self.trap_if(Op::Gt, b, n, tok);
            }
            b
        };
        if ka.is_none() || ke.is_none() {
            self.trap_if(Op::Gt, a, end, tok);
        }
        // a constant start keeps the length `end - a` a term of the bounds facts
        let n = match (ka, ke) {
            (Some(0), _) => end,
            (Some(x), Some(e)) => self.int(e - x),
            (Some(_), None) => match self.trapping(tok, Op::Sub, Some(us), &[end, a]) {
                Some(v) => v,
                None => return E::Never,
            },
            (None, _) => self.op2(Op::SubWrap, us, end, a),
        };
        let mutable = self.pointee(bt).is_some_and(|x| x.1);
        let st = self.tys.intern(TypeKind::Slice { elem, align: 0 });
        let ept = self.ptr_like(bt, elem);
        let ep = self.op2(Op::ElemAddr, ept, bp, a);
        let srt = self.t_ref(mutable, st);
        let s = self.op2(Op::SliceMake, srt, ep, n);
        E::P(Place { base: Base::Mem(s, srt), proj: Vec::new(), ty: st, root: None })
    }

    /// Branches to `panic.bounds` (site `tok`) when `op x y` (an unsigned compare) holds.
    fn trap_if(&mut self, op: Op, x: Operand, y: Operand, tok: u32) {
        let bt = self.t_bool();
        let c = self.op2(op, bt, x, y);
        let (trap, ok) = (self.f.b.create_block(), self.f.b.create_block());
        self.f.b.cbr(c, trap, &[], ok, &[]);
        self.f.b.switch_to(trap);
        self.trapping(tok, Op::Panic(PanicKind::Bounds), None, &[]);
        self.f.b.unreachable();
        self.f.b.switch_to(ok);
    }

    fn try_step(&mut self, n: u32, base: E, tok: u32) -> E {
        let bt = self.e_ty(&base);
        let Some(v) = self.operand_of(base) else { return E::Never };
        let TypeKind::Adt { def, args } = self.kind(bt) else {
            return self.fail_at(Some(tok), "`?` operand");
        };
        let is_opt = def == self.g.sigs.lang.option;
        let (ok, err) =
            if is_opt { (self.g.opt_some, self.g.opt_none) } else { (self.g.res_ok, self.g.res_err) };
        let payload = match args.first() {
            Some(GArg::Type(t)) => *t,
            _ => return self.fail_at(Some(tok), "`?` payload"),
        };
        let err_ret = self.f.b.create_block();
        let cont = self.f.b.create_block();
        let u32t = self.t_u32();
        let tag = self.op1(Op::EnumTag, u32t, v);
        let okc = self.int(ok as i128);
        let bt2 = self.t_bool();
        let c = self.op2(Op::Eq, bt2, tag, okc);
        self.f.b.cbr(c, cont, &[], err_ret, &[]);
        let snap = self.f.env.clone();
        // error path
        self.f.b.switch_to(err_ret);
        let rt = self.f.ret;
        let r = if is_opt {
            let i = self.int(self.g.opt_none as i128);
            self.opn(Op::EnumMake, rt, &[i])
        } else {
            let et = match args.get(1) {
                Some(GArg::Type(t)) => *t,
                _ => return self.fail_at(Some(tok), "`?` error type"),
            };
            let (a, b) = (self.int(err as i128), self.int(0));
            let e = self.opn(Op::EnumGet, et, &[v, a, b]);
            let ri = match self.kind(rt) {
                TypeKind::Adt { .. } => self.g.res_err,
                _ => err,
            };
            let i = self.int(ri as i128);
            self.opn(Op::EnumMake, rt, &[i, e])
        };
        self.drop_temps_keep();
        self.drop_scopes(0);
        self.f.b.ret(Some(r));
        // continue
        self.f.env = snap;
        self.f.b.switch_to(cont);
        let _ = n;
        if self.is_unit(payload) {
            return E::Unit;
        }
        let (a, b) = (self.int(ok as i128), self.int(0));
        E::V(self.opn(Op::EnumGet, payload, &[v, a, b]), payload)
    }

    /// Drops the statement temporaries on an exit path without forgetting them for the
    /// fallthrough path.
    pub fn drop_temps_keep(&mut self) {
        let ts = self.f.temps.clone();
        for &(p, t) in ts.iter().rev() {
            self.drop_ptr(p, t);
        }
    }

    // ---------- calls ----------

    /// Lowers call arguments against parameter types.
    fn args(&mut self, args: &[u32], params: &[Ty]) -> Option<Vec<Operand>> {
        let mut out = Vec::with_capacity(args.len());
        for (k, &a) in args.iter().enumerate() {
            let w = params.get(k).copied();
            out.push(self.val(a, w)?);
        }
        Some(out)
    }

    fn call(&mut self, n: u32, cur: Cur, prim: u32, paren: u32, args: &[u32], last: bool) -> E {
        let icx = self.icx;
        let ctok = {
            let mut t = paren.saturating_sub(1);
            if icx.is(t, Term::Gt) || icx.is(t, Term::GtGt) {
                let mut k = t;
                while k > icx.a(n) && !icx.is(k, Term::ColonColon) {
                    k -= 1;
                }
                t = k.saturating_sub(1);
            }
            t
        };
        self.cur_tok = ctok;
        let tgt = self.target(paren).cloned();
        match tgt {
            Some(Target::Fn { decl, args: ga }) => {
                let ga: Vec<GArg> = ga.iter().map(|a| self.sub_garg(*a)).collect();
                let k = InstKey { decl, args: ga };
                let Some((ps, _)) = self.inst_sig(&k) else { return self.fail_at(Some(ctok), "callee") };
                let Some(vs) = self.args(args, &ps) else { return E::Never };
                self.call_inst(k, &vs, ctok)
            }
            Some(Target::Intrinsic { decl }) => {
                let name = self.g.prog.decl(decl).map_or("", |d| self.g.prog.strings.get(d.name)).to_string();
                let Some(sig) = self.g.sigs.fn_sig(decl) else {
                    return self.fail_at(Some(ctok), "intrinsic");
                };
                let (ps, ret) = (sig.params.clone(), sig.ret);
                let Some(vs) = self.args(args, &ps) else { return E::Never };
                let id = self.intrinsic(&name);
                let rt = self.ret_of(ret);
                let r = self.f.b.intrinsic(id, rt.map(FnCx::ir), &vs);
                self.call_result(r, ret)
            }
            Some(Target::Variant { decl, index }) => {
                let adt = self.g.prog.decl(decl).map_or(crate::ids::DeclId::NONE, |d| d.parent);
                self.variant_ctor(n, adt, index, args, last)
            }
            Some(Target::TraitMethod { method, self_ty, args: ga }) => {
                let st = self.sub(self_ty);
                let ga: Vec<GArg> = ga.iter().map(|a| self.sub_garg(*a)).collect();
                let np = self.g.sigs.fn_sig(method).map_or(0, |s| s.n_parent as usize);
                let ta = ga.get(..np.min(ga.len())).unwrap_or(&[]).to_vec();
                let own = ga.get(np..).unwrap_or(&[]).to_vec();
                let w = self.g.world();
                match crate::mono::resolve_trait_method(&w, &mut self.tys, method, st, &ta, &own) {
                    Some(Resolved::Inst(k)) => {
                        let Some((ps, _)) = self.inst_sig(&k) else {
                            return self.fail_at(Some(ctok), "callee");
                        };
                        let Some(vs) = self.args(args, &ps) else { return E::Never };
                        self.call_inst(k, &vs, ctok)
                    }
                    Some(Resolved::BuiltinOp(name)) => {
                        let Some(vs) = self.args(args, &[st, st]) else { return E::Never };
                        let Some((o, trap)) = trait_op(name) else { return self.fail_at(Some(ctok), "op") };
                        let r = if vs.len() == 1 {
                            return self.builtin_unop(ctok, o, vs[0], st);
                        } else if trap {
                            match self.trapping(ctok, o, Some(st), &vs) {
                                Some(r) => r,
                                None => return E::Never,
                            }
                        } else {
                            self.opn(o, st, &vs)
                        };
                        E::V(r, st)
                    }
                    None => self.fail_at(Some(ctok), "no impl for a trait method call"),
                }
            }
            Some(Target::FnValue) => {
                let fe = self.cur_value(cur, prim);
                let ft = self.e_ty(&fe);
                let Some(fv) = self.operand_of(fe) else { return E::Never };
                let TypeKind::Fn { params, ret, .. } = self.kind(ft) else {
                    return self.fail_at(Some(ctok), "call of a non-fn value");
                };
                let Some(vs) = self.args(args, &params) else { return E::Never };
                let rt = self.ret_of(ret);
                let s = self.site(ctok);
                self.f.b.set_site(s);
                self.f.last_site = None;
                let r = self.f.b.call_ind(fv, rt.map(FnCx::ir), &vs);
                self.f.b.set_site(stave_ir::Site::NONE);
                self.call_result(r, ret)
            }
            Some(Target::SizeOf { ty, align }) => {
                let t = self.sub(ty);
                let Some((s, a)) = self.layout(t) else { return self.fail_at(Some(ctok), "layout") };
                let us = self.t_usize();
                E::V(self.int(if align { a } else { s } as i128), us)
            }
            Some(_) => self.vector_unsupported(Some(ctok), "vector built-in"),
            // A type-relative built-in (`f32x4.splat(x)`) has its target at the name token.
            None => match self.target(ctok) {
                Some(Target::Builtin { name, recv, ret, .. }) => {
                    let (name, recv, ret) = (*name, *recv, *ret);
                    let (t, rt) = (self.sub(recv), self.sub(ret));
                    self.vec_assoc(rt, t, name, args, ctok)
                }
                _ => self.fail_at(Some(ctok), "call without a target"),
            },
        }
    }

    /// The ADT type of a literal or constructor: the node type when it is the last step, else
    /// inferred from the field values.
    fn adt_type(
        &mut self,
        n: u32,
        adt: crate::ids::DeclId,
        last: bool,
        vi: usize,
        vals: &[(usize, Ty)],
    ) -> Ty {
        if last {
            let t = self.nty(n);
            if matches!(self.kind(t), TypeKind::Adt { def, .. } if def == adt) {
                return t;
            }
        }
        let Some(ad) = self.g.sigs.adt(adt) else { return TypeId::NONE };
        let ng = ad.generics.len();
        if ng == 0 {
            return self.tys.mk_adt(adt, Vec::new());
        }
        let mut binds: Vec<Option<GArg>> = vec![None; ng];
        let w = self.g.world();
        if let Some(v) = ad.variants.get(vi) {
            for &(fi, t) in vals {
                if let Some(f) = v.fields.get(fi) {
                    let _ = w.match_ty(&self.tys, f.ty, t, &mut binds);
                }
            }
        }
        if binds.iter().any(|b| b.is_none()) {
            return TypeId::NONE;
        }
        let args: Vec<GArg> = binds.into_iter().flatten().collect();
        self.tys.mk_adt(adt, args)
    }

    fn variant_ctor(&mut self, n: u32, adt: crate::ids::DeclId, index: u32, args: &[u32], last: bool) -> E {
        let mut ty = if last { self.nty(n) } else { TypeId::NONE };
        if !matches!(self.kind(ty), TypeKind::Adt { def, .. } if def == adt) {
            ty = TypeId::NONE;
        }
        let fts: Option<Vec<Ty>> = if ty.is_none() {
            None
        } else {
            self.adt_fields(ty).and_then(|v| v.get(index as usize).cloned())
        };
        let mut vs = Vec::with_capacity(args.len());
        let mut tys = Vec::with_capacity(args.len());
        for (k, &a) in args.iter().enumerate() {
            let w = fts.as_ref().and_then(|f| f.get(k).copied());
            let e = self.expr(a, w);
            let t = self.e_ty(&e);
            let Some(v) = self.operand_of(e) else { return E::Never };
            vs.push(v);
            tys.push((k, t));
        }
        if ty.is_none() {
            ty = self.adt_type(n, adt, last, index as usize, &tys);
        }
        if ty.is_none() {
            return self.fail(n, "cannot infer the type of a variant constructor");
        }
        let mut ops = vec![self.int(index as i128)];
        ops.extend(vs);
        E::V(self.opn(Op::EnumMake, ty, &ops), ty)
    }

    fn struct_lit(&mut self, n: u32, sl: Option<u32>, tok: u32, last: bool) -> E {
        let (adt, vidx) = match self.target(tok) {
            Some(Target::Struct { decl }) => (*decl, None),
            Some(Target::Variant { decl, index }) => {
                (self.g.prog.decl(*decl).map_or(crate::ids::DeclId::NONE, |d| d.parent), Some(*index))
            }
            _ => return self.fail_at(Some(tok), "struct literal without a target"),
        };
        let vi = vidx.unwrap_or(0) as usize;
        let Some(ad) = self.g.sigs.adt(adt) else { return self.fail_at(Some(tok), "adt") };
        let Some(var) = ad.variants.get(vi) else { return self.fail_at(Some(tok), "variant") };
        let field_names: Vec<crate::ids::Sym> = var.fields.iter().map(|f| f.name).collect();
        let icx = self.icx;
        let exprs = sl.map(|s| icx.kids_vec(s)).unwrap_or_default();
        let mut ty = if last { self.nty(n) } else { TypeId::NONE };
        if !matches!(self.kind(ty), TypeKind::Adt { def, .. } if def == adt) {
            ty = TypeId::NONE;
        }
        let fts: Option<Vec<Ty>> =
            if ty.is_none() { None } else { self.adt_fields(ty).and_then(|v| v.get(vi).cloned()) };
        let mut vals: Vec<Option<Operand>> = vec![None; field_names.len()];
        let mut tys = Vec::new();
        for &e in &exprs {
            let name = icx.sym(icx.a(e).saturating_sub(2));
            let Some(fi) = field_names.iter().position(|&f| f == name) else {
                return self.fail(e, "unknown field");
            };
            let w = fts.as_ref().and_then(|f| f.get(fi).copied());
            let ex = self.expr(e, w);
            let t = self.e_ty(&ex);
            let Some(v) = self.operand_of(ex) else { return E::Never };
            vals[fi] = Some(v);
            tys.push((fi, t));
        }
        if ty.is_none() {
            ty = self.adt_type(n, adt, last, vi, &tys);
        }
        if ty.is_none() {
            let k = self.nty(n);
            let kk = self.kind(k);
            let raw = self.tb().ty(n);
            let rk = self.kind(raw);
            let args = self.args.clone();
            let ak: Vec<TypeKind> = args
                .iter()
                .filter_map(|a| if let GArg::Type(t) = a { Some(self.kind(*t)) } else { None })
                .collect();
            return self.fail_at(
                Some(tok),
                format!(
                    "cannot infer the type of a struct literal ({kk:?} raw {rk:?} args {ak:?}, last {last})"
                ),
            );
        }
        let mut ops = Vec::with_capacity(vals.len() + 1);
        if let Some(v) = vidx {
            ops.push(self.int(v as i128));
        }
        for v in vals {
            match v {
                Some(v) => ops.push(v),
                None => return self.fail_at(Some(tok), "missing field"),
            }
        }
        let op = if vidx.is_some() { Op::EnumMake } else { Op::Make };
        E::V(self.opn(op, ty, &ops), ty)
    }

    /// The receiver after typeck's adjustment.
    fn receiver(&mut self, recv: E, adj: Adjust, want: Option<Ty>) -> Option<Operand> {
        let p = self.place_of(recv)?;
        let p = if adj.autoref.is_some() { self.own_temp(p) } else { p };
        let p = self.derefs(p, adj.derefs)?;
        match adj.autoref {
            Some(m) => {
                let w = match want {
                    Some(w) => w,
                    None => self.t_ref(m, p.ty),
                };
                self.addr(&p, Some(w))
            }
            None => self.read(&p),
        }
    }

    fn method_call(&mut self, n: u32, recv: E, seg: u32, args: &[u32], last: bool) -> E {
        let _ = last;
        self.cur_tok = seg;
        match self.target(seg).cloned() {
            Some(Target::Method { method, args: ga, adjust }) => match self.resolve_method(&method, &ga) {
                Some(Resolved::Inst(k)) => {
                    let Some((ps, _)) = self.inst_sig(&k) else { return self.fail_at(Some(seg), "callee") };
                    let Some(rv) = self.receiver(recv, adjust, ps.first().copied()) else { return E::Never };
                    let Some(mut vs) = self.args(args, ps.get(1..).unwrap_or(&[])) else { return E::Never };
                    vs.insert(0, rv);
                    self.call_inst(k, &vs, seg)
                }
                Some(Resolved::BuiltinOp(name)) => {
                    let Some(rv) = self.receiver(recv, adjust, None) else { return E::Never };
                    let ty = self.nty(n);
                    let Some(mut vs) = self.args(args, &[ty]) else { return E::Never };
                    vs.insert(0, rv);
                    let Some((o, trap)) = trait_op(name) else { return self.fail_at(Some(seg), "op") };
                    if vs.len() == 1 {
                        return self.builtin_unop(seg, o, vs[0], ty);
                    }
                    let r =
                        if trap { self.trapping(seg, o, Some(ty), &vs) } else { Some(self.opn(o, ty, &vs)) };
                    match r {
                        Some(r) => E::V(r, ty),
                        None => E::Never,
                    }
                }
                None => self.fail_at(Some(seg), "unresolved method"),
            },
            Some(Target::Builtin { name, adjust, ret, consts, .. }) => {
                let rt = self.sub(ret);
                self.builtin_method(n, recv, name, adjust, args, seg, (rt, &consts))
            }
            _ => self.fail_at(Some(seg), "method call without a target"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn builtin_method(
        &mut self,
        n: u32,
        recv: E,
        name: &str,
        adj: Adjust,
        args: &[u32],
        seg: u32,
        (rt, consts): (Ty, &[u64]),
    ) -> E {
        let Some(p) = self.place_of(recv) else { return E::Never };
        let Some(p) = self.derefs(p, adj.derefs) else { return E::Never };
        if self.is_simd(p.ty) {
            return self.vec_method(rt, p, name, args, seg, consts);
        }
        match name {
            "align_to" => match &p.base {
                Base::Mem(v, _) if p.proj.is_empty() => {
                    let v = *v;
                    let st = self.t_ref(false, p.ty);
                    self.align_to(rt, v, st, seg)
                }
                _ => self.fail_at(Some(seg), "align_to of an unsized place"),
            },
            "len" => {
                let us = self.t_usize();
                match self.kind(p.ty) {
                    TypeKind::Array { len, .. } => E::V(self.int(len.value().unwrap_or(0) as i128), us),
                    TypeKind::Slice { .. } | TypeKind::Str => match &p.base {
                        Base::Mem(v, _) if p.proj.is_empty() => {
                            let v = *v;
                            E::V(self.op1(Op::SliceLen, us, v), us)
                        }
                        _ => self.fail_at(Some(seg), "len of an unsized place"),
                    },
                    _ => self.fail_at(Some(seg), "len"),
                }
            }
            "mul_wide" => {
                let t = p.ty;
                let Some(a) = self.read(&p) else { return E::Never };
                let Some(b) = self.val(args.first().copied().unwrap_or(n), Some(t)) else { return E::Never };
                let tt = self.tys.mk_tuple(vec![t, t]);
                E::V(self.op2(Op::MulWide, tt, a, b), tt)
            }
            "popcount" | "clz" | "ctz" => {
                let t = p.ty;
                let Some(a) = self.read(&p) else { return E::Never };
                let o = match name {
                    "popcount" => Op::Popcount,
                    "clz" => Op::Clz,
                    _ => Op::Ctz,
                };
                E::V(self.op1(o, t, a), t)
            }
            "read" => {
                let pt = p.ty;
                let Some(v) = self.read(&p) else { return E::Never };
                let Some((inner, _, _)) = self.pointee(pt) else { return self.fail_at(Some(seg), "read") };
                if self.is_unit(inner) {
                    return E::Unit;
                }
                E::V(Operand::Value(self.f.b.load(FnCx::ir(inner), v)), rt)
            }
            "write" | "write_stream" => {
                let pt = p.ty;
                let Some(v) = self.read(&p) else { return E::Never };
                let inner = self.pointee(pt).map_or(TypeId::NONE, |x| x.0);
                let Some(x) = self.val(args.first().copied().unwrap_or(n), Some(inner)) else {
                    return E::Never;
                };
                self.f.b.store(v, x);
                E::Unit
            }
            "select" => {
                if !matches!(self.kind(p.ty), TypeKind::Bool) {
                    return self.vector_unsupported(Some(seg), "mask select");
                }
                // the operands' type: the checker records no result type for this built-in
                let rt = self.nty(n);
                let Some(c) = self.read(&p) else { return E::Never };
                let (Some(&an), Some(&bn)) = (args.first(), args.get(1)) else {
                    return self.fail_at(Some(seg), "select operands");
                };
                // both operands are evaluated, left to right, whatever `c` is
                let Some(a) = self.val(an, Some(rt)) else { return E::Never };
                let Some(b) = self.val(bn, Some(rt)) else { return E::Never };
                if self.is_unit(rt) {
                    return E::Unit;
                }
                let r = self.opn(Op::Select, rt, &[c, a, b]);
                if self.needs_drop(rt) {
                    // the operand not selected is a statement temporary, picked without a branch
                    let u = self.opn(Op::Select, rt, &[c, b, a]);
                    self.spill(u, rt, true);
                }
                E::V(r, rt)
            }
            s if s.starts_with("prefetch_") => {
                // a hint: no effect on results, no IR op yet
                for &a in args {
                    if self.val(a, None).is_none() {
                        return E::Never;
                    }
                }
                E::Unit
            }
            _ => self.unsupported_at(Some(seg), format!("built-in method `{name}` is not supported yet")),
        }
    }

    // ---------- closures ----------

    fn closure(&mut self, n: u32) -> E {
        let Some(k) = self.closure_nodes.iter().position(|&c| c == n) else {
            return self.fail(n, "closure");
        };
        let ty = self.nty(n);
        match self.closure_value(n, k, ty) {
            Some(v) => E::V(v, ty),
            None => E::Never,
        }
    }

    // ---------- f-strings ----------

    fn fstring(&mut self, n: u32, tok: u32) -> E {
        let icx = self.icx;
        let text = icx.tok_text(tok);
        let base = icx.toks.get(tok as usize).map_or(0, |t| t.start);
        let Some(parts) = fmt_parts(text, base) else { return self.fail(n, "malformed f-string") };
        let (Some(new), Some(push)) = (self.g.lang.string_new, self.g.lang.string_push_str) else {
            return self.fail(n, "std.string.String not found");
        };
        let st = self.string_ty();
        let ts = self.new_slot(st);
        let e = self.call_inst(InstKey { decl: new, args: Vec::new() }, &[], tok);
        let Some(sv) = self.operand_of(e) else { return E::Never };
        self.f.b.store(ts, sv);
        for part in parts {
            match part {
                FmtPart::Lit(bytes) => {
                    if bytes.is_empty() {
                        continue;
                    }
                    let s = self.string(&bytes);
                    let e =
                        self.call_inst(InstKey { decl: push, args: Vec::new() }, &[ts, Operand::Str(s)], tok);
                    if matches!(e, E::Never) {
                        return E::Never;
                    }
                }
                FmtPart::Hole(segs) => {
                    if !self.fmt_hole(n, tok, &segs, ts, st) {
                        return E::Never;
                    }
                }
            }
        }
        let mt = self.t_ref(true, st);
        E::P(Place { base: Base::Mem(ts, mt), proj: Vec::new(), ty: st, root: None })
    }

    fn fmt_hole(&mut self, n: u32, tok: u32, segs: &[(u32, String)], ts: Operand, st: Ty) -> bool {
        let icx = self.icx;
        let Some((off0, s0)) = segs.first() else { return false };
        let mut k = 1;
        let mut e = if s0 == "self" {
            match icx.use_at_offset(n, *off0).map(|u| u.res).and_then(|r| self.local_of(r)) {
                Some(l) => E::P(self.local_place(l)),
                None => {
                    self.fail_at(Some(tok), "unresolved self in f-string");
                    return false;
                }
            }
        } else {
            let Some(u) = icx.use_at_offset(n, *off0) else {
                self.fail_at(Some(tok), "unresolved f-string hole");
                return false;
            };
            let mut res = u.res;
            while k < segs.len() {
                let is_module =
                    matches!(res, Res::Decl(d) if self.decl_kind(d) == Some(crate::decls::DeclKind::Module));
                match icx.use_at_offset(n, segs[k].0) {
                    Some(u2) if is_module => {
                        res = u2.res;
                        k += 1;
                    }
                    _ => break,
                }
            }
            match res {
                Res::Decl(d) if self.decl_kind(d) == Some(crate::decls::DeclKind::Const) => {
                    let Some(c) = self.g.sigs.konst(d) else { return false };
                    let (ty, Some(v)) = (c.ty, c.value.clone()) else { return false };
                    self.const_val(&v, ty, tok)
                }
                Res::Decl(d) if self.decl_kind(d) == Some(crate::decls::DeclKind::Static) => {
                    self.static_place(d, tok)
                }
                r => match self.local_of(r) {
                    Some(l) => E::P(self.local_place(l)),
                    None => {
                        self.fail_at(Some(tok), "f-string hole is not a value");
                        return false;
                    }
                },
            }
        };
        // field steps (auto-deref through references)
        for (_, name) in &segs[k..] {
            let Some(p) = self.place_of(e) else { return false };
            let mut p = p;
            while let TypeKind::Ref { .. } = self.kind(p.ty) {
                let Some(v) = self.read(&p) else { return false };
                let t = p.ty;
                p = self.deref(v, t);
            }
            let TypeKind::Adt { def, .. } = self.kind(p.ty) else {
                self.fail_at(Some(tok), "f-string field of a non-struct");
                return false;
            };
            let sym = self.g.prog.strings.lookup(name);
            let Some(fi) = self
                .g
                .sigs
                .adt(def)
                .and_then(|a| a.variants.first())
                .and_then(|v| v.fields.iter().position(|f| Some(f.name) == sym))
            else {
                self.fail_at(Some(tok), "f-string field");
                return false;
            };
            let Some(ft) = self.adt_fields(p.ty).and_then(|v| v.first().and_then(|f| f.get(fi).copied()))
            else {
                return false;
            };
            p.proj.push(Proj::Field(fi as u32, ft));
            p.ty = ft;
            e = E::P(p);
        }
        // a pointer to the Display type (references peeled)
        let Some(p) = self.place_of(e) else { return false };
        let mut refs = 0;
        let mut d = p.ty;
        while let TypeKind::Ref { inner, .. } = self.kind(d) {
            d = inner;
            refs += 1;
        }
        let rd = self.t_ref(false, d);
        let ptr = if refs == 0 {
            match &p.base {
                Base::Mem(..) => {
                    let Some(a) = self.addr(&p, Some(rd)) else { return false };
                    a
                }
                _ => {
                    // a copy of the value, borrowed
                    let Some(v) = self.read(&p) else { return false };
                    let (s, mt) = self.spill(v, d, false);
                    self.convert_ptr(s, mt, rd)
                }
            }
        } else {
            let Some(mut v) = self.read(&p) else { return false };
            let mut t = p.ty;
            for _ in 1..refs {
                let Some((inner, _, _)) = self.pointee(t) else { return false };
                v = Operand::Value(self.f.b.load(FnCx::ir(inner), v));
                t = inner;
            }
            self.convert_ptr(v, t, rd)
        };
        let disp = self.g.sigs.lang.display;
        let w = self.g.world();
        let Some((imp, iargs)) =
            w.find_impl(&mut self.tys, d, &TraitRef { trait_decl: disp, args: Vec::new() }, &[], 0)
        else {
            self.fail_at(Some(tok), "no Display impl for an f-string hole");
            return false;
        };
        let fmt_sym = self.g.prog.strings.lookup("fmt");
        let Some(m) = self.g.sigs.impl_def(imp).and_then(|i| {
            i.methods.iter().copied().find(|&m| self.g.prog.decl(m).is_some_and(|x| Some(x.name) == fmt_sym))
        }) else {
            return false;
        };
        let mut args = iargs;
        args.push(GArg::Type(st));
        let e = self.call_inst(InstKey { decl: m, args }, &[ptr, ts], tok);
        !matches!(e, E::Never)
    }

    // ---------- types ----------

    /// Lowers a `Type` node of the current item (substituted).
    pub fn lower_type(&mut self, tn: u32) -> Ty {
        let env =
            crate::sigs::TyEnv { icx: self.icx, n_parent: self.cur_n_parent, self_ty: self.raw_self_ty };
        let mut h = TyHost { tys: &mut self.tys, prog: self.g.prog, sigs: self.g.sigs };
        let t = crate::sigs::lower_type(&mut h, &env, tn, false);
        self.sub(t)
    }

    fn cur_generic_const_ty(&mut self, idx: u32) -> Ty {
        let d = self.tb().decl;
        match self.g.sigs.fn_sig(d).and_then(|s| s.generics.get(idx as usize)) {
            Some(g) => g.const_ty,
            None => self.t_usize(),
        }
    }
}

/// Type lowering host for cast types (the signature-phase lowering over the body's overlay).
struct TyHost<'x, 'a> {
    tys: &'x mut crate::types::Overlay<'a>,
    prog: &'a crate::Program,
    sigs: &'a crate::sigs::Sigs,
}

impl Interner for TyHost<'_, '_> {
    fn kind(&self, t: TypeId) -> &TypeKind {
        self.tys.kind(t)
    }
    fn intern(&mut self, k: TypeKind) -> TypeId {
        self.tys.intern(k)
    }
    fn fixed_ty(&self, f: TypeId) -> TypeId {
        self.tys.fixed_ty(f)
    }
}

impl crate::sigs::LowerHost for TyHost<'_, '_> {
    fn prog(&self) -> &crate::Program {
        self.prog
    }
    fn sigs_generics_len(&mut self, d: crate::ids::DeclId) -> Option<Vec<bool>> {
        Some(self.sigs.generics(d).iter().map(|g| g.is_const).collect())
    }
    fn adt_generics(&mut self, d: crate::ids::DeclId) -> Vec<crate::sigs::GenericParam> {
        self.sigs.generics(d).to_vec()
    }
    fn alias_target(&mut self, d: crate::ids::DeclId) -> Option<(Vec<crate::sigs::GenericParam>, TypeId)> {
        match self.sigs.get(d) {
            crate::sigs::DeclSig::Alias(a) => Some((a.generics.clone(), a.target)),
            _ => None,
        }
    }
    fn const_u64(&mut self, env: &crate::sigs::TyEnv<'_>, n: u32, ty: TypeId) -> ConstArg {
        crate::consteval::eval_u64(self, env, n, ty)
    }
    fn report(&mut self, _d: Option<stave_diag::Diagnostic>) {}
}

impl crate::consteval::CEnv for TyHost<'_, '_> {
    fn const_value(&mut self, d: crate::ids::DeclId) -> Option<(TypeId, crate::consteval::Val)> {
        match self.sigs.get(d) {
            crate::sigs::DeclSig::Const(c) => Some((c.ty, c.value.clone()?)),
            _ => None,
        }
    }
    fn adt_def(&mut self, d: crate::ids::DeclId) -> Option<crate::sigs::AdtDef> {
        self.sigs.adt(d).cloned()
    }
    fn layout(&mut self, t: TypeId) -> Option<(u64, u64)> {
        crate::traits::layout_of(self.tys, self.sigs, t, 0)
    }
    fn lang(&self) -> &crate::sigs::LangItems {
        &self.sigs.lang
    }
    fn fn_types(&mut self, d: crate::ids::DeclId) -> Option<(Vec<TypeId>, TypeId)> {
        self.sigs.fn_sig(d).map(|f| (f.params.clone(), f.ret))
    }
}
