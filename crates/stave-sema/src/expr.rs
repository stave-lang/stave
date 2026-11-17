//! Expressions of the body checker ([`crate::typeck`]): operators, casts, literals, paths,
//! calls, methods, fields, indexing, `?`, struct and variant literals, closures, f-strings,
//! `if then else`, `match`, ranges, and `for` loops.
//!
//! Literal typing is bidirectional: a context type (hint) flows down to operands that
//! share the result's type, arms, and parentheses; a literal without one joins a pending group
//! fixed by the other operand or arm ([`crate::typeck::Ty::Lit`]). Coercion sites (annotations,
//! arguments, fields, returns, assignments) check the final type.
//!
//! [`Target`]s are keyed by the token of the step they describe: the name token of a field or
//! method, the `(` of a call, the `[` of an index, the `?`, the `.{` of a struct literal, the
//! operator token of a user-type operator, and the name token of a path step.

use stave_diag::Span;
use stave_syntax::{NodeKind, Term};

use crate::builtins::{self, ArgTy};
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::{DeclId, Sym};
use crate::prim::{Builtin, Lanes, Prim};
use crate::resolve::{DeferKind, Res};
use crate::sigs::{GenericParam, TraitRef, VariantShape};
use crate::traits::{Adjust, Lookup, MethodRef};
use crate::typeck::{Ck, Coercion, PendingCheck, Target, Ty};
use crate::types::{self, ConstArg, GArg, Interner, TypeId, TypeKind, ids};

/// A postfix chain's state between steps.
#[derive(Clone, Debug)]
enum St {
    Val(Ty),
    /// A module (no value).
    NoVal,
    Fn {
        decl: DeclId,
        targs: Option<Vec<GArg>>,
    },
    /// A method path `Type.m` (impl method decl).
    Assoc {
        method: DeclId,
        impl_args: Option<Vec<GArg>>,
        targs: Option<Vec<GArg>>,
    },
    Adt {
        def: DeclId,
        args: Option<Vec<GArg>>,
    },
    TypeVal(TypeId),
    Variant {
        decl: DeclId,
        adt: DeclId,
        index: u32,
        args: Option<Vec<GArg>>,
    },
    Trait,
    TraitMethod {
        method: DeclId,
        trait_decl: DeclId,
        targs: Option<Vec<GArg>>,
    },
    /// `T.m` for a generic `T` (or `Self` in a trait) through a bound.
    BoundAssoc {
        method: DeclId,
        trait_args: Vec<GArg>,
        self_ty: TypeId,
        targs: Option<Vec<GArg>>,
    },
    Builtin(Builtin, Option<Vec<GArg>>),
    /// `f32x4.splat` (a built-in type-relative fn).
    TypeAssoc {
        ty: TypeId,
        name: Sym,
        tok: u32,
    },
    Err,
}

const OPS: [(Term, &str); 13] = [
    (Term::Plus, "Add"),
    (Term::PlusPercent, "WrapAdd"),
    (Term::PlusPipe, "SatAdd"),
    (Term::Minus, "Sub"),
    (Term::MinusPercent, "WrapSub"),
    (Term::MinusPipe, "SatSub"),
    (Term::Star, "Mul"),
    (Term::StarPercent, "WrapMul"),
    (Term::StarPipe, "SatMul"),
    (Term::Slash, "Div"),
    (Term::Percent, "Rem"),
    (Term::LtLt, "Shl"),
    (Term::LtLtPercent, "WrapShl"),
];

fn op_trait(op: u8) -> Option<&'static str> {
    if op == Term::GtGt as u8 {
        return Some("Shr");
    }
    if op == Term::Amp as u8 {
        return Some("BitAnd");
    }
    if op == Term::Pipe as u8 {
        return Some("BitOr");
    }
    if op == Term::Caret as u8 {
        return Some("BitXor");
    }
    OPS.iter().find(|(t, _)| *t as u8 == op).map(|(_, n)| *n)
}

fn is_plain(op: u8) -> bool {
    op == Term::Plus as u8 || op == Term::Minus as u8 || op == Term::Star as u8
}

fn is_wrap_sat(op: u8) -> bool {
    [
        Term::PlusPercent,
        Term::MinusPercent,
        Term::StarPercent,
        Term::PlusPipe,
        Term::MinusPipe,
        Term::StarPipe,
        Term::LtLtPercent,
    ]
    .iter()
    .any(|t| *t as u8 == op)
}

fn is_bit(op: u8) -> bool {
    op == Term::Amp as u8 || op == Term::Pipe as u8 || op == Term::Caret as u8
}

fn is_shift(op: u8) -> bool {
    op == Term::LtLt as u8 || op == Term::GtGt as u8 || op == Term::LtLtPercent as u8
}

impl<'a> Ck<'a> {
    /// Types expression node `n` with context type `hint` and records the type on `n`.
    pub fn expr(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        if !self.enter(n) {
            self.set(n, Ty::ERR);
            return Ty::ERR;
        }
        let t = self.expr_node(n, hint);
        self.set(n, t);
        self.leave();
        t
    }

    fn expr_node(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let Some(k) = self.icx.kind(n) else { return Ty::ERR };
        match k {
            NodeKind::IfExpr => self.if_expr(n, hint),
            NodeKind::MatchExpr => self.match_expr(n, hint),
            NodeKind::RangeExpr => self.range_expr(n, hint),
            NodeKind::OrExpr | NodeKind::AndExpr => self.logic(n),
            NodeKind::CmpExpr => self.compare(n),
            NodeKind::BitOr
            | NodeKind::BitXor
            | NodeKind::BitAnd
            | NodeKind::Shift
            | NodeKind::Add
            | NodeKind::Mul => self.binary_chain(n, hint),
            NodeKind::Cast => self.cast_chain(n),
            NodeKind::Unary => self.unary(n, hint),
            NodeKind::Postfix => self.postfix(n, hint),
            // a bare name is a path with no steps
            NodeKind::Primary if self.icx.is_path_head(n) => self.postfix(n, hint),
            NodeKind::Primary => self.primary(n, hint),
            _ => Ty::ERR,
        }
    }

    // ---------- operators ----------

    fn logic(&mut self, n: u32) -> Ty {
        let kids = self.kids(n);
        let mut div = false;
        for (i, &c) in kids.iter().enumerate() {
            let before = self.diverges;
            self.diverges = false;
            self.check_expr(c, ids::BOOL, None);
            if i == 0 {
                div = self.diverges;
            }
            self.diverges = before || div;
        }
        self.free_kids(kids);
        Ty::T(ids::BOOL)
    }

    fn compare(&mut self, n: u32) -> Ty {
        let icx = self.icx;
        let kids = self.kids(n);
        let (Some(&l), Some(&opn), Some(&r)) = (kids.first(), kids.get(1), kids.get(2)) else {
            self.free_kids(kids);
            return Ty::ERR;
        };
        self.free_kids(kids);
        let op = icx.tk(icx.a(opn));
        let lt = self.expr(l, None);
        let lh = match self.resolve(lt) {
            Ty::T(t) => Some(t),
            _ => None,
        };
        let rt = self.expr(r, lh);
        let t = self.unify_operands(r, lt, rt);
        let t = self.resolve(t);
        let Ty::T(t) = t else { return Ty::T(ids::BOOL) };
        if t == ids::ERROR || self.tys.has_error(t) {
            return Ty::T(ids::BOOL);
        }
        let eq_only = op == Term::EqEq as u8 || op == Term::BangEq as u8;
        match self.tys.kind(t).clone() {
            TypeKind::Int { .. } | TypeKind::Float(_) | TypeKind::Bool | TypeKind::Char => Ty::T(ids::BOOL),
            TypeKind::Ptr { .. } | TypeKind::Fn { .. } if eq_only => Ty::T(ids::BOOL),
            TypeKind::Vector { lane, n: lanes } => {
                self.gate_compare(icx.a(opn), op, t);
                let w = types::lane_bits(self.tys.kind(lane)) as u8;
                Ty::T(self.tys.intern(TypeKind::Mask { width: w, n: lanes }))
            }
            TypeKind::Mask { .. } if eq_only => {
                self.gate_compare(icx.a(opn), op, t);
                Ty::T(t)
            }
            _ => {
                self.e0206(icx.a(opn), t);
                Ty::T(ids::BOOL)
            }
        }
    }

    fn e0206(&mut self, tok: u32, t: TypeId) {
        let op = self.icx.tok_str(tok).to_string();
        let tn = self.show(t);
        let d = mk("E0206", None, &[("op", &op), ("type", &tn)], self.icx.tok_span(tok), &[]);
        self.report(d);
    }

    /// Unifies two operands of one type (rule 4): a pending literal takes the other's type.
    /// Mismatches are E0201 at `r` (E0602/E0603 for vectors).
    fn unify_operands(&mut self, r: u32, lt: Ty, rt: Ty) -> Ty {
        let (lt, rt) = (self.resolve(lt), self.resolve(rt));
        match (lt, rt) {
            (Ty::T(a), Ty::T(b)) => {
                if a == b || b == ids::NEVER {
                    return Ty::T(a);
                }
                if a == ids::NEVER {
                    return Ty::T(b);
                }
                if a == ids::ERROR || b == ids::ERROR || self.tys.has_error(a) || self.tys.has_error(b) {
                    return Ty::ERR;
                }
                if a.is_none() || b.is_none() {
                    return Ty::ERR;
                }
                let (va, vb) = (
                    matches!(self.tys.kind(a), TypeKind::Vector { .. }),
                    matches!(self.tys.kind(b), TypeKind::Vector { .. }),
                );
                if va && vb {
                    let (ls, rs) = (self.show(a), self.show(b));
                    let d = mk("E0603", None, &[("left", &ls), ("right", &rs)], self.icx.node_span(r), &[]);
                    self.report(d);
                    return Ty::ERR;
                }
                if va != vb {
                    let (s, v) = if va { (b, a) } else { (a, b) };
                    let (ss, vs) = (self.show(s), self.show(v));
                    let operand = self.icx.node_text(r).to_string();
                    let d = mk(
                        "E0602",
                        None,
                        &[("scalar", &ss), ("vector", &vs), ("operand", &operand)],
                        self.icx.node_span(r),
                        &[],
                    );
                    self.report(d);
                    return Ty::ERR;
                }
                self.mismatch(r, Ty::T(b), a, None);
                Ty::ERR
            }
            (Ty::Lit(g), Ty::T(b)) => {
                if b == ids::ERROR {
                    let _ = self.fix(g, ids::ERROR);
                    return Ty::ERR;
                }
                if b == ids::NEVER {
                    return Ty::Lit(g);
                }
                if matches!(self.tys.kind(b), TypeKind::Vector { .. }) {
                    return self.lit_with_vector(r, g, b);
                }
                if self.fix(g, b) {
                    Ty::T(b)
                } else {
                    self.mismatch(r, Ty::T(b), self.lit_kind_type(g), None);
                    Ty::ERR
                }
            }
            (Ty::T(a), Ty::Lit(g)) => {
                if a == ids::ERROR {
                    let _ = self.fix(g, ids::ERROR);
                    return Ty::ERR;
                }
                if a == ids::NEVER {
                    return Ty::Lit(g);
                }
                if matches!(self.tys.kind(a), TypeKind::Vector { .. }) {
                    return self.lit_with_vector(r, g, a);
                }
                if self.fix(g, a) {
                    Ty::T(a)
                } else {
                    self.mismatch(r, Ty::Lit(g), a, None);
                    Ty::ERR
                }
            }
            (Ty::Lit(g1), Ty::Lit(g2)) => {
                if self.unite(g1, g2) {
                    Ty::Lit(g1)
                } else {
                    let (r1, _) = self.lit_root(g1);
                    let _ = self.fix(r1, ids::ERROR);
                    let (r2, _) = self.lit_root(g2);
                    let _ = self.fix(r2, ids::ERROR);
                    let d = mk(
                        "E0201",
                        None,
                        &[
                            ("expected", if self.lit_root(g1).1.float { "{float}" } else { "{integer}" }),
                            ("found", if self.lit_root(g2).1.float { "{float}" } else { "{integer}" }),
                            ("operand", self.icx.node_text(r)),
                        ],
                        self.icx.node_span(r),
                        &[],
                    );
                    self.report(d);
                    Ty::ERR
                }
            }
        }
    }

    /// E0602 at `r`: a literal operand of a vector operator (a scalar; `TxN.splat` broadcasts it).
    fn lit_with_vector(&mut self, r: u32, g: u32, v: TypeId) -> Ty {
        let float = self.lit_root(g).1.float;
        let _ = self.fix(g, ids::ERROR);
        let vs = self.show(v);
        let operand = self.icx.node_text(r).to_string();
        let scalar = if float { "{float}" } else { "{integer}" };
        let d = mk(
            "E0602",
            None,
            &[("scalar", scalar), ("vector", &vs), ("operand", &operand)],
            self.icx.node_span(r),
            &[],
        );
        self.report(d);
        Ty::ERR
    }

    /// A stand-in for printing a pending group's kind in E0201 (`{integer}` is printed by
    /// `show_ty`; this only selects a type the group cannot take).
    fn lit_kind_type(&self, _g: u32) -> TypeId {
        ids::ERROR
    }

    fn binary_chain(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let kids = self.kids(n);
        let Some(&first) = kids.first() else {
            self.free_kids(kids);
            return Ty::ERR;
        };
        let mut acc = self.expr(first, hint);
        for (i, &c) in kids.iter().enumerate().skip(1) {
            let optok = icx.a(c).saturating_sub(1);
            let op = icx.tk(optok);
            let h = match self.resolve(acc) {
                Ty::T(t) if !t.is_none() && t != ids::NEVER => Some(t),
                _ => hint,
            };
            let rt = self.expr(c, h);
            let lnode = if i == 1 { first } else { n };
            acc = self.binop_types(optok, op, lnode, acc, c, rt);
        }
        self.free_kids(kids);
        acc
    }

    /// The type of `l op r`. `at`: the operator token (an `AssignOp` node's first token works too).
    pub fn binop_types(&mut self, at: u32, op: u8, l: u32, lt: Ty, r: u32, rt: Ty) -> Ty {
        let icx = self.icx;
        let at_tok = if icx.kind(at) == Some(NodeKind::AssignOp)
            && at < self.lo + self.node_ty.len() as u32
            && at >= self.lo
        {
            icx.a(at)
        } else {
            at
        };
        // a literal left operand of a vector operator: E0602 at the literal
        if let (Ty::Lit(g), Ty::T(v)) = (self.resolve(lt), self.resolve(rt))
            && matches!(self.tys.kind(v), TypeKind::Vector { .. })
        {
            return self.lit_with_vector(l, g, v);
        }
        if is_shift(op) {
            // a literal amount of a vector shift is a lane value: `v << 3`
            if let (Ty::T(vt), Ty::Lit(g)) = (self.resolve(lt), self.resolve(rt))
                && let TypeKind::Vector { lane, .. } = self.tys.kind(vt).clone()
            {
                if self.fix(g, lane) {
                    if op != Term::LtLtPercent as u8 {
                        self.const_shift_amount(r, vt);
                    }
                    return self.op_on_gated(at_tok, op, vt, "const");
                }
                return Ty::ERR;
            }
            // the amount has the left operand's type
            let t = self.unify_operands(r, lt, rt);
            match self.resolve(t) {
                Ty::Lit(g) => {
                    if op != Term::LtLtPercent as u8
                        && let Some(v) = self.lit_value(r)
                    {
                        self.add_check(g, PendingCheck::Shift(r, v));
                    }
                    return Ty::Lit(g);
                }
                Ty::T(tt) => {
                    if op != Term::LtLtPercent as u8 {
                        self.const_shift_amount(r, tt);
                    }
                    return self.op_on_gated(at_tok, op, tt, "vector");
                }
            }
        }
        let t = self.unify_operands(r, lt, rt);
        match self.resolve(t) {
            Ty::Lit(g) => {
                let float = self.lit_root(g).1.float;
                if float && (is_wrap_sat(op) || is_bit(op)) {
                    let opt = icx.tok_str(at_tok).to_string();
                    let d =
                        mk("E0206", None, &[("op", &opt), ("type", "{float}")], icx.tok_span(at_tok), &[]);
                    self.report(d);
                    let _ = self.fix(g, ids::ERROR);
                    return Ty::ERR;
                }
                Ty::Lit(g)
            }
            Ty::T(tt) => self.op_on_gated(at_tok, op, tt, ""),
        }
    }

    /// [`Ck::op_on`], then the lowering gate of a vector or mask operator (crate::gates).
    fn op_on_gated(&mut self, tok: u32, op: u8, t: TypeId, form: &str) -> Ty {
        let r = self.op_on(tok, op, t);
        if r == Ty::T(t) {
            self.gate_binop(tok, op, t, form);
        }
        r
    }

    /// Validity of a binary operator on operand type `t` (both operands `t`): the result type.
    fn op_on(&mut self, tok: u32, op: u8, t: TypeId) -> Ty {
        if t == ids::ERROR || t.is_none() || self.tys.has_error(t) {
            return Ty::ERR;
        }
        if t == ids::NEVER {
            return Ty::T(t);
        }
        let k = self.tys.kind(t).clone();
        let bad = |s: &mut Self| {
            s.e0206(tok, t);
            Ty::ERR
        };
        match k {
            TypeKind::Int { .. } => Ty::T(t),
            TypeKind::Float(_) => {
                if is_wrap_sat(op) || is_bit(op) || is_shift(op) {
                    return bad(self);
                }
                Ty::T(t)
            }
            TypeKind::Bool => {
                if is_bit(op) {
                    Ty::T(t)
                } else {
                    bad(self)
                }
            }
            TypeKind::Vector { lane, .. } => {
                let int_lane = self.tys.int(lane).is_some();
                if int_lane {
                    if is_plain(op) {
                        let o = self.icx.tok_str(tok).to_string();
                        let tn = self.show(t);
                        let d = mk("E0604", None, &[("op", &o), ("type", &tn)], self.icx.tok_span(tok), &[]);
                        self.report(d);
                        return Ty::ERR;
                    }
                    if op == Term::Slash as u8 {
                        let tn = self.show(t);
                        let d = mk("E0607", None, &[("type", &tn)], self.icx.tok_span(tok), &[]);
                        self.report(d);
                        return Ty::ERR;
                    }
                    if op == Term::Percent as u8 {
                        return bad(self);
                    }
                    Ty::T(t)
                } else {
                    if is_wrap_sat(op) || is_bit(op) || is_shift(op) || op == Term::Percent as u8 {
                        return bad(self);
                    }
                    Ty::T(t)
                }
            }
            TypeKind::Mask { .. } => {
                if is_bit(op) {
                    Ty::T(t)
                } else {
                    bad(self)
                }
            }
            _ => match op_trait(op) {
                Some(name) => self.user_op(tok, name, t),
                None => bad(self),
            },
        }
    }

    /// A user-type operator through its trait (`impl Add for S`) or a bound (`T: Add`).
    fn user_op(&mut self, tok: u32, trait_name: &str, t: TypeId) -> Ty {
        let Some(&tr) = self.sigs.lang.ops.get(trait_name) else {
            self.e0206(tok, t);
            return Ty::ERR;
        };
        let method_name = crate::sigs::OP_TRAITS.iter().find(|x| x.0 == trait_name).map_or("", |x| x.1);
        let msym = self.prog.strings.lookup(method_name);
        if let TypeKind::Param(p) = self.tys.kind(t).clone() {
            let has =
                self.genv().get(p as usize).is_some_and(|g| g.bounds.iter().any(|b| b.trait_decl == tr));
            if has {
                if let Some(m) = msym.and_then(|s| self.prog.member(tr, s)) {
                    self.target(
                        tok,
                        Target::Op {
                            method: MethodRef::Bound { method: m, trait_args: Vec::new(), self_ty: t },
                            args: Vec::new(),
                        },
                    );
                }
                return Ty::T(t);
            }
            self.e0901(tok, p);
            return Ty::ERR;
        }
        let genv = self.genv();
        let tref = TraitRef { trait_decl: tr, args: Vec::new() };
        match self.w.find_impl(&mut self.tys, t, &tref, genv, 0) {
            Some((imp, args)) => {
                let m = self.sigs.impl_def(imp).and_then(|i| {
                    i.methods
                        .iter()
                        .copied()
                        .find(|&m| msym.is_some_and(|s| self.prog.decl(m).is_some_and(|d| d.name == s)))
                });
                if let Some(m) = m {
                    self.target(
                        tok,
                        Target::Op {
                            method: MethodRef::Impl { method: m, impl_decl: imp, impl_args: args.clone() },
                            args: args.clone(),
                        },
                    );
                    self.site(tok, m, &args);
                }
                Ty::T(t)
            }
            None => {
                self.e0206(tok, t);
                Ty::ERR
            }
        }
    }

    pub fn e0901(&mut self, tok: u32, param: u32) {
        let name = self.icx.tok_str(tok).to_string();
        let (pn, span) = match self.genv().get(param as usize) {
            Some(g) => (self.prog.strings.get(g.name).to_string(), Some(g.name_span)),
            None => ("T".to_string(), None),
        };
        let roles: Vec<(&str, Span)> = span.map(|s| ("param", s)).into_iter().collect();
        let d = mk("E0901", None, &[("name", &name), ("param", &pn)], self.icx.tok_span(tok), &roles);
        self.report(d);
    }

    /// The value of a constant shift amount at `r` (a literal or a const), checked against `t`.
    pub fn const_shift_amount(&mut self, r: u32, t: TypeId) {
        let bits = match self.tys.kind(t) {
            TypeKind::Int { bits, .. } => bits.bits(),
            TypeKind::Vector { lane, .. } => types::lane_bits(self.tys.kind(*lane)),
            _ => return,
        };
        let lane_t = match self.tys.kind(t) {
            TypeKind::Vector { lane, .. } => *lane,
            _ => t,
        };
        if let Some(v) = self.lit_value(r).or_else(|| self.const_value_of(r)) {
            self.e0205(r, v, bits, lane_t);
        }
    }

    /// The magnitude of a non-negative int literal at `n` (through parentheses).
    pub fn lit_value(&self, n: u32) -> Option<u128> {
        let p = self.down_to_primary(n)?;
        let a = self.icx.a(p);
        if !self.icx.is(a, Term::IntLit) {
            return None;
        }
        crate::literals::parse_int(self.icx.tok_str(a)).map(|x| x.0)
    }

    /// The value of a non-negative int const that `n` names by a plain path.
    fn const_value_of(&self, n: u32) -> Option<u128> {
        let u = self.icx.uses(self.icx.chain_head(n)?).last()?;
        let Res::Decl(d) = u.res else { return None };
        if self.icx.postfix_end(n) != self.icx.chain_end(self.icx.a(n)) {
            return None;
        }
        let c = self.sigs.konst(d)?;
        match c.value.as_ref()? {
            crate::consteval::Val::Int(i) if *i >= 0 => Some(*i as u128),
            _ => None,
        }
    }

    /// Whether `n` is an f-string literal (through parentheses).
    pub fn fstring_node(&self, n: u32) -> bool {
        self.down_to_primary(n).is_some_and(|p| self.icx.is(self.icx.a(p), Term::FmtLit))
    }

    /// The Primary that `n` is, through parentheses.
    fn down_to_primary(&self, n: u32) -> Option<u32> {
        let mut cur = n;
        for _ in 0..1_000_000 {
            match self.icx.kind(cur)? {
                NodeKind::Primary => {
                    let a = self.icx.a(cur);
                    if self.icx.is(a, Term::LParen) {
                        let kids = self.icx.kids_vec(cur);
                        let close = self.icx.close(a);
                        if kids.len() == 1 && !self.icx.is(close - 1, Term::Comma) {
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

    fn unary(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let a = icx.a(n);
        let t = icx.tk(a);
        let Some(c) = icx.last_child(n) else { return Ty::ERR };
        if t == Term::KwUnsafe as u8 {
            self.unsafe_depth += 1;
            let r = self.expr(c, hint);
            self.unsafe_depth -= 1;
            return r;
        }
        if t == Term::Minus as u8 {
            // `-` directly on a literal is folded into it.
            if let Some(p) = self.down_to_primary(c)
                && icx.a(p) == a + 1
                && (icx.is(a + 1, Term::IntLit) || icx.is(a + 1, Term::FloatLit))
            {
                let r = self.literal(n, a + 1, true, hint);
                self.mark_chain(c, r);
                return r;
            }
            let ct = self.expr(c, hint);
            return match self.resolve(ct) {
                Ty::Lit(g) => {
                    self.add_check(g, PendingCheck::Neg(n));
                    Ty::Lit(g)
                }
                Ty::T(ct) => self.neg_on(a, ct),
            };
        }
        if t == Term::Bang as u8 {
            self.check_expr(c, ids::BOOL, None);
            return Ty::T(ids::BOOL);
        }
        if t == Term::Tilde as u8 {
            let ct = self.expr(c, hint);
            return match self.resolve(ct) {
                Ty::Lit(g) => {
                    if self.lit_root(g).1.float {
                        let d = mk("E0206", None, &[("op", "~"), ("type", "{float}")], icx.tok_span(a), &[]);
                        self.report(d);
                        let _ = self.fix(g, ids::ERROR);
                        return Ty::ERR;
                    }
                    Ty::Lit(g)
                }
                Ty::T(ct) => self.not_on(a, ct),
            };
        }
        if t == Term::Star as u8 {
            let ct = self.expr(c, None);
            let ct = self.force(ct);
            return match self.tys.kind(ct).clone() {
                TypeKind::Ref { inner, .. } => Ty::T(inner),
                TypeKind::Ptr { inner, .. } => {
                    self.unsafe_op(icx.tok_span(a));
                    Ty::T(inner)
                }
                TypeKind::Error => Ty::ERR,
                _ => {
                    self.e0206(a, ct);
                    Ty::ERR
                }
            };
        }
        if t == Term::Amp as u8 || t == Term::AmpAmp as u8 {
            // `&&x` = `& &x`: the outer half when the child Unary starts at the same token.
            let outer_of_split =
                t == Term::AmpAmp as u8 && icx.kind(c) == Some(NodeKind::Unary) && icx.a(c) == a;
            let mutable = !outer_of_split && icx.is(a + 1, Term::KwMut);
            let inner_hint = hint.and_then(|h| match self.tys.kind(h) {
                TypeKind::Ref { inner, .. } => Some(*inner),
                _ => None,
            });
            let ct = self.expr(c, inner_hint);
            let ct = self.force(ct);
            if mutable {
                self.static_mutation(c, icx.tok_span(a));
            }
            if ct == ids::ERROR {
                return Ty::ERR;
            }
            return Ty::T(self.tys.mk_ref(mutable, ct));
        }
        self.expr(c, hint)
    }

    fn mark_chain(&mut self, top: u32, t: Ty) {
        let mut cur = top;
        for _ in 0..1_000_000 {
            self.set(cur, t);
            match self.icx.kind(cur) {
                Some(NodeKind::Primary) => {
                    let a = self.icx.a(cur);
                    if self.icx.is(a, Term::LParen)
                        && let Some(k) = self.icx.first_child(cur)
                    {
                        cur = k;
                        continue;
                    }
                    return;
                }
                _ => return,
            }
        }
    }

    fn neg_on(&mut self, tok: u32, t: TypeId) -> Ty {
        if t == ids::ERROR || self.tys.has_error(t) {
            return Ty::ERR;
        }
        match self.tys.kind(t).clone() {
            TypeKind::Int { signed: true, .. } | TypeKind::Float(_) => Ty::T(t),
            TypeKind::Vector { lane, .. } if self.tys.is_float(lane) => {
                self.gate(tok, "vec.neg", t, "");
                Ty::T(t)
            }
            TypeKind::Int { .. } | TypeKind::Vector { .. } | TypeKind::Bool | TypeKind::Char => {
                self.e0206(tok, t);
                Ty::ERR
            }
            _ => self.user_unary(tok, "Neg", t),
        }
    }

    fn not_on(&mut self, tok: u32, t: TypeId) -> Ty {
        if t == ids::ERROR || self.tys.has_error(t) {
            return Ty::ERR;
        }
        match self.tys.kind(t).clone() {
            TypeKind::Int { .. } => Ty::T(t),
            TypeKind::Mask { .. } => {
                self.gate(tok, "mask.not", t, "");
                Ty::T(t)
            }
            TypeKind::Vector { lane, .. } if self.tys.int(lane).is_some() => {
                self.gate(tok, "vec.not", t, "");
                Ty::T(t)
            }
            TypeKind::Float(_) | TypeKind::Bool | TypeKind::Char | TypeKind::Vector { .. } => {
                self.e0206(tok, t);
                Ty::ERR
            }
            _ => self.user_unary(tok, "Not", t),
        }
    }

    fn user_unary(&mut self, tok: u32, name: &str, t: TypeId) -> Ty {
        self.user_op(tok, name, t)
    }

    fn cast_chain(&mut self, n: u32) -> Ty {
        let icx = self.icx;
        let kids = self.kids(n);
        let Some(&first) = kids.first() else {
            self.free_kids(kids);
            return Ty::ERR;
        };
        let t0 = self.expr(first, None);
        let mut acc = self.force(t0);
        for &tn in &kids[1..] {
            let optok = icx.a(tn).saturating_sub(1);
            let op = icx.tk(optok);
            let plain = op == Term::KwAs as u8;
            let to = self.lower_ty(tn, false);
            if acc == ids::ERROR || to == ids::ERROR || self.tys.has_error(acc) {
                acc = if op == Term::AsQuestion as u8 { self.option_of(to) } else { to };
                if to == ids::ERROR {
                    acc = ids::ERROR;
                }
                continue;
            }
            match self.cast_result(op, acc, to) {
                Some(r) => {
                    let identity = plain && acc == to;
                    if let Some(name) = crate::gates::cast_name(op)
                        && !identity
                    {
                        self.gate_cast(optok, name, acc, to);
                    }
                    acc = r;
                }
                None => {
                    let (f, t) = (self.show(acc), self.show(to));
                    let sp = icx.range_span(optok, icx.end_tok(tn));
                    let d = mk("E0204", None, &[("from", &f), ("to", &t)], sp, &[]);
                    self.report(d);
                    acc = ids::ERROR;
                }
            }
        }
        self.free_kids(kids);
        Ty::T(acc)
    }

    pub fn option_of(&mut self, t: TypeId) -> TypeId {
        let o = self.sigs.lang.option;
        if o.is_none() || t == ids::ERROR {
            return ids::ERROR;
        }
        self.tys.mk_adt(o, vec![GArg::Type(t)])
    }

    /// Cast table: the result type of `from <op> to`, `None` if not allowed (E0204).
    fn cast_result(&mut self, op: u8, from: TypeId, to: TypeId) -> Option<TypeId> {
        let plain = op == Term::KwAs as u8;
        let wrap = op == Term::AsPercent as u8;
        let sat = op == Term::AsPipe as u8;
        let round = op == Term::AsTilde as u8;
        let checked = op == Term::AsQuestion as u8;
        let fk = self.tys.kind(from).clone();
        let tk = self.tys.kind(to).clone();
        // a plain cast to the identical type is a bit copy, for any type
        if plain && from == to {
            return Some(to);
        }
        // vectors: lane-wise, equal lane counts
        if let (TypeKind::Vector { lane: fl, n: fnn }, TypeKind::Vector { lane: tl, n: tn }) = (&fk, &tk) {
            if fnn != tn {
                return None;
            }
            let ok = self.scalar_cast_ok(op, *fl, *tl);
            if !ok {
                return None;
            }
            return Some(if checked { self.option_of(to) } else { to });
        }
        if from == to {
            return Some(if checked { self.option_of(to) } else { to });
        }
        // references to raw pointers: plain `as` only
        if let TypeKind::Ref { mutable, inner } = fk {
            if let TypeKind::Ptr { mutable: pm, inner: pi } = tk
                && plain
                && pi == inner
                && (mutable || !pm)
            {
                return Some(to);
            }
            return None;
        }
        if !self.scalar_cast_ok(op, from, to) {
            return None;
        }
        let _ = (wrap, sat, round);
        Some(if checked { self.option_of(to) } else { to })
    }

    fn scalar_cast_ok(&mut self, op: u8, from: TypeId, to: TypeId) -> bool {
        let fk = self.tys.kind(from).clone();
        let tk = self.tys.kind(to).clone();
        let is64 = |t: TypeId| matches!(t, ids::U64 | ids::USIZE | ids::I64 | ids::ISIZE);
        let sig_bits = |k: &TypeKind| -> u32 {
            match k {
                TypeKind::Float(types::FloatBits::F16) => 11,
                TypeKind::Float(types::FloatBits::B16) => 8,
                TypeKind::Float(types::FloatBits::F32) => 24,
                TypeKind::Float(types::FloatBits::F64) => 53,
                _ => 0,
            }
        };
        let int_mag = |k: &TypeKind| -> Option<(bool, u32)> {
            match k {
                TypeKind::Int { signed, bits } => Some((*signed, bits.bits())),
                _ => None,
            }
        };
        let is_ptr = |k: &TypeKind| matches!(k, TypeKind::Ptr { .. });
        let fint = int_mag(&fk);
        let tint = int_mag(&tk);
        let ffloat = matches!(fk, TypeKind::Float(_));
        let tfloat = matches!(tk, TypeKind::Float(_));
        if op == Term::KwAs as u8 {
            if let (Some((fs, fb)), Some((ts, tb))) = (fint, tint) {
                return (fs == ts && tb >= fb) || (!fs && ts && tb > fb);
            }
            if matches!(fk, TypeKind::Bool) && tint.is_some() {
                return true;
            }
            if matches!(fk, TypeKind::Char) {
                return matches!(to, ids::U32 | ids::U64 | ids::I64 | ids::USIZE | ids::ISIZE);
            }
            if from == ids::U8 && matches!(tk, TypeKind::Char) {
                return true;
            }
            if let (Some((fs, fb)), true) = (fint, tfloat) {
                let need = if fs { fb - 1 } else { fb };
                return need <= sig_bits(&tk);
            }
            if ffloat && tfloat {
                return matches!(
                    (&fk, &tk),
                    (
                        TypeKind::Float(types::FloatBits::F16 | types::FloatBits::B16),
                        TypeKind::Float(types::FloatBits::F32 | types::FloatBits::F64)
                    ) | (TypeKind::Float(types::FloatBits::F32), TypeKind::Float(types::FloatBits::F64))
                );
            }
            if is_ptr(&fk) && is_ptr(&tk) {
                return true;
            }
            if (is_ptr(&fk) && is64(to)) || (is64(from) && is_ptr(&tk)) {
                return true;
            }
            return false;
        }
        if op == Term::AsPercent as u8 {
            return tint.is_some()
                && (fint.is_some() || matches!(fk, TypeKind::Bool | TypeKind::Char) || is_ptr(&fk));
        }
        if op == Term::AsPipe as u8 {
            return tint.is_some() && (fint.is_some() || ffloat || matches!(fk, TypeKind::Char));
        }
        if op == Term::AsTilde as u8 {
            return tfloat && (fint.is_some() || ffloat);
        }
        if op == Term::AsQuestion as u8 {
            if tint.is_some() {
                return fint.is_some() || ffloat || matches!(fk, TypeKind::Char);
            }
            if matches!(tk, TypeKind::Char) || tfloat {
                return fint.is_some() || (ffloat && tfloat);
            }
            return false;
        }
        false
    }

    fn range_expr(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let kids = self.kids(n);
        let (Some(&l), Some(&r)) = (kids.first(), kids.get(1)) else {
            self.free_kids(kids);
            return Ty::ERR;
        };
        self.free_kids(kids);
        let inclusive = icx.is(icx.a(r).saturating_sub(1), Term::DotDotEq);
        let eh = hint.and_then(|h| match self.tys.kind(h) {
            TypeKind::Range { elem, .. } => Some(*elem),
            _ => None,
        });
        let lt = self.expr(l, eh);
        let lh = match self.resolve(lt) {
            Ty::T(t) if t != ids::NEVER => Some(t),
            _ => eh,
        };
        let rt = self.expr(r, lh);
        let t = self.unify_operands(r, lt, rt);
        let t = self.force(t);
        if t == ids::ERROR {
            return Ty::ERR;
        }
        if !matches!(self.tys.kind(t), TypeKind::Int { .. } | TypeKind::Char) {
            self.e0206(icx.a(r).saturating_sub(1), t);
            return Ty::ERR;
        }
        Ty::T(self.tys.intern(TypeKind::Range { elem: t, inclusive }))
    }

    fn if_expr(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let kids = self.kids(n);
        let (Some(&c), Some(&a), Some(&b)) = (kids.first(), kids.get(1), kids.get(2)) else {
            self.free_kids(kids);
            return Ty::ERR;
        };
        self.free_kids(kids);
        self.check_expr(c, ids::BOOL, None);
        let before = self.diverges;
        self.diverges = false;
        let at = self.expr(a, hint);
        let da = self.diverges;
        self.diverges = false;
        let bh = match self.resolve(at) {
            Ty::T(t) if !t.is_none() && t != ids::NEVER && t != ids::ERROR => Some(t),
            _ => hint,
        };
        let bt = self.expr(b, bh);
        let db = self.diverges;
        self.diverges = before || (da && db);
        self.arms_unify(b, at, bt, hint)
    }

    /// Unifies two arm types (rule 4), with the expected type when given.
    fn arms_unify(&mut self, b: u32, at: Ty, bt: Ty, hint: Option<TypeId>) -> Ty {
        let (at, bt) = (self.resolve(at), self.resolve(bt));
        if at == Ty::T(ids::NEVER) {
            return bt;
        }
        if bt == Ty::T(ids::NEVER) {
            return at;
        }
        if let Some(h) = hint
            && let (Ty::T(x), Ty::T(y)) = (at, bt)
            && x != y
        {
            // each arm is coerced at the site; keep the expected type if both can take it
            if (x == h || self.coercible(x, h).is_some()) && (y == h || self.coercible(y, h).is_some()) {
                return Ty::T(h);
            }
        }
        self.unify_operands(b, at, bt)
    }

    // ---------- literals and primaries ----------

    /// A literal at token `tok` (node `n`), `neg`: folded unary minus.
    fn literal(&mut self, n: u32, tok: u32, neg: bool, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let text = icx.tok_str(tok);
        if icx.is(tok, Term::IntLit) {
            let Some((mag, suf)) = crate::literals::parse_int(text) else {
                return self.malformed(n);
            };
            return match suf {
                crate::literals::Suffix::Int(t) => {
                    if !crate::literals::int_fits(mag, neg, t) {
                        let g = self.new_lit(n, mag, neg, false);
                        let _ = self.fix_quiet(g, t);
                    }
                    Ty::T(t)
                }
                crate::literals::Suffix::Float(t) => Ty::T(t),
                crate::literals::Suffix::None => {
                    let g = self.new_lit(n, mag, neg, false);
                    if let Some(h) = hint
                        && self.tys.int(h).is_some()
                    {
                        let _ = self.fix(g, h);
                        return Ty::T(h);
                    }
                    Ty::Lit(g)
                }
            };
        }
        let Some((v, suf)) = crate::literals::parse_float(text) else { return self.malformed(n) };
        match suf {
            crate::literals::Suffix::Float(t) => {
                if !crate::literals::float_fits(v, t) {
                    let g = self.new_lit(n, 0, neg, true);
                    let _ = self.fix(g, t);
                }
                Ty::T(t)
            }
            _ => {
                let g = self.new_lit(n, 0, neg, true);
                if let Some(h) = hint
                    && self.tys.is_float(h)
                {
                    let _ = self.fix(g, h);
                    return Ty::T(h);
                }
                Ty::Lit(g)
            }
        }
    }

    fn fix_quiet(&mut self, g: u32, t: TypeId) -> bool {
        self.fix(g, t)
    }

    fn malformed(&mut self, n: u32) -> Ty {
        let text = self.icx.node_text(n).to_string();
        let d = mk("E0203", Some("malformed"), &[("literal", &text)], self.icx.node_span(n), &[]);
        self.report(d);
        Ty::ERR
    }

    fn primary(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let a = icx.a(n);
        let t = icx.tk(a);
        if t == Term::IntLit as u8 || t == Term::FloatLit as u8 {
            return self.literal(n, a, false, hint);
        }
        if t == Term::KwTrue as u8 || t == Term::KwFalse as u8 {
            return Ty::T(ids::BOOL);
        }
        if t == Term::CharLit as u8 {
            if crate::literals::parse_char(icx.tok_str(a)).is_none() {
                return self.malformed(n);
            }
            return Ty::T(ids::CHAR);
        }
        if t == Term::StrLit as u8 {
            if crate::literals::parse_str(icx.tok_str(a)).is_none() {
                return self.malformed(n);
            }
            return Ty::T(ids::REF_STR);
        }
        if t == Term::FmtLit as u8 {
            return self.fstring(n, a);
        }
        if t == Term::KwSelf as u8 {
            return match icx.uses(n).first().map(|u| u.res) {
                Some(Res::SelfValue(b)) => Ty::T(self.locals.get(&b.node).copied().unwrap_or(ids::ERROR)),
                _ => Ty::ERR,
            };
        }
        if t == Term::LParen as u8 {
            return self.paren(n, hint);
        }
        if t == Term::LBracket as u8 {
            return self.array_lit(n, hint);
        }
        if t == Term::KwFn as u8 {
            return self.closure(n);
        }
        // identifiers and `Self` are paths: handled by `postfix`
        Ty::ERR
    }

    fn paren(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let a = icx.a(n);
        let close = icx.close(a);
        let kids = self.kids(n);
        if kids.is_empty() {
            self.free_kids(kids);
            return Ty::T(ids::UNIT);
        }
        if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
            let k = kids[0];
            self.free_kids(kids);
            return self.expr(k, hint);
        }
        let hints: Vec<Option<TypeId>> = match hint.map(|h| self.tys.kind(h).clone()) {
            Some(TypeKind::Tuple(ts)) if ts.len() == kids.len() => ts.iter().map(|&x| Some(x)).collect(),
            Some(TypeKind::Tuple(_)) => {
                // a tuple of another length: E0201 here, elements without context stay silent
                let mut found = Vec::new();
                for &k in &kids {
                    let t = self.expr(k, None);
                    found.push(self.show_ty(t));
                    if let Ty::Lit(g) = self.resolve(t) {
                        let _ = self.fix(g, ids::ERROR);
                    }
                }
                self.free_kids(kids);
                if let Some(h) = hint {
                    let expected = self.show(h);
                    let found = format!("({}{})", found.join(", "), if found.len() == 1 { "," } else { "" });
                    let operand = icx.node_text(n).to_string();
                    let d = mk(
                        "E0201",
                        None,
                        &[("expected", &expected), ("found", &found), ("operand", &operand)],
                        icx.node_span(n),
                        &[],
                    );
                    self.report(d);
                }
                return Ty::ERR;
            }
            _ => vec![None; kids.len()],
        };
        let mut tys = Vec::with_capacity(kids.len());
        for (i, &k) in kids.iter().enumerate() {
            let t = match hints[i] {
                Some(h) => {
                    let t = self.expr(k, Some(h));
                    match self.resolve(t) {
                        Ty::Lit(_) => self.coerce(k, t, h, None),
                        Ty::T(x) => x,
                    }
                }
                None => {
                    let t = self.expr(k, None);
                    self.force(t)
                }
            };
            tys.push(t);
        }
        self.free_kids(kids);
        Ty::T(self.tys.mk_tuple(tys))
    }

    fn array_lit(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let kids = self.kids(n);
        let eh = hint.and_then(|h| match self.tys.kind(h) {
            TypeKind::Array { elem, .. } => Some(*elem),
            _ => None,
        });
        // `[e; n]`
        if kids.len() == 2 && icx.is(icx.end_tok(kids[0]) + 1, Term::Semi) {
            let (e, cnt) = (kids[0], kids[1]);
            self.free_kids(kids);
            let et = self.expr(e, eh);
            let et = match (self.resolve(et), eh) {
                (Ty::Lit(_), Some(h)) => self.coerce(e, et, h, None),
                (t, _) => self.force(t),
            };
            self.check_expr(cnt, ids::USIZE, None);
            let env = self.env();
            let len = crate::consteval::eval_u64(self, &env, cnt, ids::USIZE);
            if et == ids::ERROR || len == ConstArg::Error {
                return Ty::ERR;
            }
            return Ty::T(self.tys.intern(TypeKind::Array { elem: et, len }));
        }
        if kids.is_empty() {
            self.free_kids(kids);
            return match hint.map(|h| (h, self.tys.kind(h).clone())) {
                Some((h, TypeKind::Array { len: ConstArg::Value(0), .. })) => Ty::T(h),
                _ => {
                    let text = icx.node_text(n).to_string();
                    let d = mk("E0202", Some("infer"), &[("literal", &text)], icx.node_span(n), &[]);
                    self.report(d);
                    Ty::ERR
                }
            };
        }
        let mut elem = eh;
        let mut tys = Vec::with_capacity(kids.len());
        for &k in &kids {
            let t = self.expr(k, elem);
            if elem.is_none()
                && let Ty::T(x) = self.resolve(t)
                && x != ids::NEVER
            {
                elem = Some(x);
            }
            tys.push((k, t));
        }
        let elem_t = match elem {
            Some(e) => e,
            None => {
                // all pending: unite, then no context
                let mut acc = tys[0].1;
                for &(k, t) in &tys[1..] {
                    acc = self.unify_operands(k, acc, t);
                }
                self.force(acc)
            }
        };
        for &(k, t) in &tys {
            self.coerce(k, t, elem_t, None);
        }
        let len = kids.len() as u64;
        self.free_kids(kids);
        if elem_t == ids::ERROR {
            return Ty::ERR;
        }
        Ty::T(self.tys.intern(TypeKind::Array { elem: elem_t, len: ConstArg::Value(len) }))
    }

    fn closure(&mut self, n: u32) -> Ty {
        let icx = self.icx;
        let mut params = Vec::new();
        // A closure has the `needs` of its fn: its body is checked and compiled under them.
        let features = self.features;
        if let Some(ps) = icx.child(n, NodeKind::Params) {
            for p in icx.kids_vec(ps) {
                let t = match icx.child(p, NodeKind::Type) {
                    Some(tn) => {
                        let t = self.lower_ty(tn, false);
                        self.sig_width_at(icx.node_span(tn), t, features);
                        t
                    }
                    None => ids::ERROR,
                };
                self.locals.insert(p, t);
                params.push(t);
            }
        }
        let ret_node = icx.child(n, NodeKind::Type);
        let ret = match ret_node {
            Some(r) => {
                let t = self.lower_ty(r, true);
                self.sig_width_at(icx.node_span(r), t, features);
                t
            }
            None => ids::UNIT,
        };
        let Some(block) = icx.child(n, NodeKind::Block) else { return Ty::ERR };
        let (reach, div, scope, cis) =
            (self.reachable, self.diverges, self.scope_depth, self.closure_in_scope);
        self.reachable = true;
        if scope > 0 {
            self.closure_in_scope = Some(icx.a(n));
        }
        self.scope_depth = 0;
        self.rets.push(crate::typeck::RetCx {
            ty: ret,
            span: ret_node.map(|r| icx.node_span(r)),
            loops: Vec::new(),
            is_never: ret == ids::NEVER,
        });
        let d = self.block(block);
        self.rets.pop();
        self.end_of_body(block, d, ret);
        self.reachable = reach;
        self.diverges = div;
        self.scope_depth = scope;
        self.closure_in_scope = cis;
        if params.contains(&ids::ERROR) || ret == ids::ERROR {
            return Ty::ERR;
        }
        Ty::T(self.tys.intern(TypeKind::Fn { params: params.into_boxed_slice(), ret, features }))
    }

    fn fstring(&mut self, n: u32, tok: u32) -> Ty {
        let icx = self.icx;
        let text = icx.tok_text(tok);
        let base = icx.toks.get(tok as usize).map_or(0, |t| t.start);
        let holes = crate::scan::fmt_holes(text, base);
        for h in holes {
            let Some(&(off0, s0)) = h.segs.first() else { continue };
            let mut ty = if s0 == "self" {
                match icx.use_at_offset(n, off0).map(|u| u.res) {
                    Some(Res::SelfValue(b)) => self.locals.get(&b.node).copied().unwrap_or(ids::ERROR),
                    _ => ids::ERROR,
                }
            } else {
                let Some(u) = icx.use_at_offset(n, off0) else { continue };
                let mut st = self.path_state(u.res, None, u.token, None);
                let mut k = 1;
                // module/type steps resolved by the resolver
                while k < h.segs.len() {
                    let (off, _) = h.segs[k];
                    match icx.use_at_offset(n, off) {
                        Some(u2) if !matches!(st, St::Val(_)) => {
                            st = self.step_path(st, u2.res, None, u2.token);
                            k += 1;
                        }
                        _ => break,
                    }
                }
                let v = self.value_of(st, n, None);
                let mut t = self.force(v);
                for &(off, seg) in &h.segs[k..] {
                    let sym = self.prog.strings.lookup(seg).unwrap_or(Sym::NONE);
                    t = self.field_of(t, sym, Span::new(icx.fid(), off, seg.len() as u32), None);
                }
                t
            };
            if s0 == "self" {
                for &(off, seg) in &h.segs[1..] {
                    let sym = self.prog.strings.lookup(seg).unwrap_or(Sym::NONE);
                    ty = self.field_of(ty, sym, Span::new(icx.fid(), off, seg.len() as u32), None);
                }
            }
            if ty == ids::ERROR || self.tys.has_error(ty) {
                continue;
            }
            let (inner, _) = self.tys.peel_refs(ty);
            let disp = self.sigs.lang.display;
            if disp.is_none() {
                continue;
            }
            let genv = self.genv();
            let ok = self.w.implements(
                &mut self.tys,
                inner,
                &TraitRef { trait_decl: disp, args: Vec::new() },
                genv,
            );
            if !ok {
                let last = h.segs.last().copied().unwrap_or((off0, s0));
                let sp = Span::new(icx.fid(), off0, last.0 + last.1.len() as u32 - off0);
                if let TypeKind::Param(p) = self.tys.kind(inner).clone() {
                    let pn = self
                        .genv()
                        .get(p as usize)
                        .map_or("T".to_string(), |g| self.prog.strings.get(g.name).to_string());
                    let roles: Vec<(&str, Span)> =
                        self.genv().get(p as usize).map(|g| ("param", g.name_span)).into_iter().collect();
                    let d = mk("E0901", None, &[("name", "fmt"), ("param", &pn)], sp, &roles);
                    self.report(d);
                } else {
                    let tn = self.show(inner);
                    let d = mk("E0208", None, &[("name", "fmt"), ("type", &tn)], sp, &[]);
                    self.report(d);
                }
                continue;
            }
            // lowering calls `Display.fmt::<String>` on the hole (references peeled)
            let fmt = self.prog.strings.lookup("fmt");
            let string = self.sigs.lang.string;
            if let Some(m) = self.sigs.trait_def(disp).and_then(|t| {
                t.methods.iter().copied().find(|&m| self.prog.decl(m).is_some_and(|x| Some(x.name) == fmt))
            }) && string.is_some()
            {
                let w = self.tys.mk_adt(string, Vec::new());
                self.implicit.push((
                    tok,
                    Target::TraitMethod { method: m, self_ty: inner, args: vec![GArg::Type(w)] },
                ));
            }
        }
        let s = self.sigs.lang.string;
        if s.is_none() {
            self.has_error = true;
            return Ty::ERR;
        }
        Ty::T(self.tys.mk_adt(s, Vec::new()))
    }

    // ---------- paths ----------

    /// The state of a resolved path segment `res` (`targs`: its turbofish).
    fn path_state(&mut self, res: Res, targs: Option<u32>, tok: u32, prev: Option<&St>) -> St {
        let icx = self.icx;
        match res {
            Res::Local(b) | Res::SelfValue(b) => match self.locals.get(&b.node) {
                Some(&t) => St::Val(Ty::T(t)),
                None => St::Err,
            },
            Res::Generic(g) => {
                let idx = if g.level == 0 { g.index as u32 } else { self.n_parent + g.index as u32 };
                if g.is_const {
                    let t = self.genv().get(idx as usize).map_or(ids::ERROR, |p| p.const_ty);
                    St::Val(Ty::T(t))
                } else {
                    St::TypeVal(self.tys.mk_param(idx))
                }
            }
            Res::SelfType => match self.self_ty {
                Some(t) => St::TypeVal(t),
                None => {
                    // `Self` in a trait: a type-relative base
                    St::TypeVal(ids::SELF_TY)
                }
            },
            Res::Prim(p) => match p {
                Prim::Scalar(s) => St::TypeVal(types::scalar_type(s)),
                Prim::Vector { lane, n } => {
                    let l = types::scalar_type(lane);
                    let t = self.tys.intern(TypeKind::Vector { lane: l, n });
                    self.gate_width(icx.tok_span(tok), t);
                    St::TypeVal(t)
                }
                Prim::Mask { width, n } => {
                    let t = self.tys.intern(TypeKind::Mask { width, n });
                    self.gate_width(icx.tok_span(tok), t);
                    St::TypeVal(t)
                }
                _ => St::Err,
            },
            Res::Builtin(b) => {
                if b == Builtin::FenceStream {
                    self.unsafe_op(icx.tok_span(tok));
                }
                let ga = targs.and_then(|g| {
                    // `fence_stream` has no generic params: any turbofish is E0201
                    let k = if b == Builtin::FenceStream { 0 } else { icx.kids_vec(g).len() };
                    self.turbofish(g, &vec![false; k], g)
                });
                St::Builtin(b, ga)
            }
            Res::Unresolved => St::Err,
            Res::Decl(d) => {
                let Some(decl) = self.prog.decl(d).copied() else { return St::Err };
                let shape = |s: &Self, x: DeclId| -> Vec<bool> {
                    s.sigs.generics(x).iter().map(|g| g.is_const).collect()
                };
                match decl.kind {
                    DeclKind::Module => St::NoVal,
                    DeclKind::Fn => {
                        let ga = targs.and_then(|g| {
                            let sh = shape(self, d);
                            self.turbofish(g, &sh, g)
                        });
                        St::Fn { decl: d, targs: ga }
                    }
                    DeclKind::Const => {
                        self.target(tok, Target::Global { decl: d });
                        St::Val(Ty::T(self.sigs.konst(d).map_or(ids::ERROR, |c| c.ty)))
                    }
                    DeclKind::Static => {
                        if decl.flags.has(DeclFlags::MUT) {
                            self.unsafe_op(icx.tok_span(tok));
                        }
                        self.target(tok, Target::Global { decl: d });
                        St::Val(Ty::T(self.sigs.konst(d).map_or(ids::ERROR, |c| c.ty)))
                    }
                    DeclKind::Struct | DeclKind::Enum => {
                        let ga = targs.and_then(|g| {
                            let sh = shape(self, d);
                            self.turbofish(g, &sh, g)
                        });
                        St::Adt { def: d, args: ga }
                    }
                    DeclKind::Variant => {
                        let adt = decl.parent;
                        let args = match prev {
                            Some(St::Adt { def, args }) if *def == adt => args.clone(),
                            Some(St::TypeVal(t)) => match self.tys.kind(*t) {
                                TypeKind::Adt { def, args } if *def == adt => Some(args.to_vec()),
                                _ => None,
                            },
                            _ => None,
                        };
                        St::Variant { decl: d, adt, index: decl.index, args }
                    }
                    DeclKind::Method => {
                        let impl_decl = decl.parent;
                        let impl_args = match prev {
                            Some(St::Adt { def, args: Some(args) }) => {
                                let t = self.tys.mk_adt(*def, args.clone());
                                self.impl_args_for(impl_decl, t)
                            }
                            Some(St::TypeVal(t)) => self.impl_args_for(impl_decl, *t),
                            _ => None,
                        };
                        let ga = targs.and_then(|g| {
                            let all = shape(self, d);
                            let np = self.sigs.fn_sig(d).map_or(0, |s| s.n_parent as usize);
                            let own: Vec<bool> = all.get(np..).unwrap_or(&[]).to_vec();
                            self.turbofish(g, &own, g)
                        });
                        St::Assoc { method: d, impl_args, targs: ga }
                    }
                    DeclKind::TraitMethod => {
                        let ga = targs.and_then(|g| {
                            let all = shape(self, d);
                            let np = self.sigs.fn_sig(d).map_or(0, |s| s.n_parent as usize);
                            let own: Vec<bool> = all.get(np..).unwrap_or(&[]).to_vec();
                            self.turbofish(g, &own, g)
                        });
                        St::TraitMethod { method: d, trait_decl: decl.parent, targs: ga }
                    }
                    DeclKind::Trait => St::Trait,
                    DeclKind::TypeAlias => match self.sigs.get(d) {
                        crate::sigs::DeclSig::Alias(al) => {
                            let target = al.target;
                            let sh: Vec<bool> = al.generics.iter().map(|g| g.is_const).collect();
                            if sh.is_empty() {
                                St::TypeVal(target)
                            } else {
                                match targs.and_then(|g| self.turbofish(g, &sh, g)) {
                                    Some(args) => St::TypeVal(self.tys.subst(target, &args, None)),
                                    None => St::Err,
                                }
                            }
                        }
                        _ => St::Err,
                    },
                    _ => St::Err,
                }
            }
        }
    }

    fn step_path(&mut self, prev: St, res: Res, targs: Option<u32>, tok: u32) -> St {
        let p = prev.clone();
        self.path_state(res, targs, tok, Some(&p))
    }

    /// The generic args of impl `imp` when its self type matches `t`.
    fn impl_args_for(&mut self, imp: DeclId, t: TypeId) -> Option<Vec<GArg>> {
        let idef = self.sigs.impl_def(imp)?;
        let genv = self.genv();
        self.w.impl_applies(&mut self.tys, idef, t, genv, 0)
    }

    /// The value of a finished path state.
    fn value_of(&mut self, st: St, n: u32, hint: Option<TypeId>) -> Ty {
        match st {
            St::Val(t) => t,
            St::Err => Ty::ERR,
            St::Variant { decl, adt, index, args } => {
                let shape = self.sigs.adt(adt).and_then(|a| a.variants.get(index as usize)).map(|v| v.shape);
                if shape != Some(VariantShape::Unit) {
                    return self.not_value(n);
                }
                let ng = self.sigs.adt(adt).map_or(0, |a| a.generics.len());
                let args = match args {
                    Some(a) => Some(a),
                    None if ng == 0 => Some(Vec::new()),
                    None => hint.and_then(|h| match self.tys.kind(h) {
                        TypeKind::Adt { def, args } if *def == adt => Some(args.to_vec()),
                        _ => None,
                    }),
                };
                let tok = self.icx.end_tok(n);
                self.target(tok, Target::Variant { decl, index });
                match args {
                    Some(a) => {
                        let t = self.tys.mk_adt(adt, a);
                        self.owned(tok, t);
                        Ty::T(t)
                    }
                    None => {
                        self.e0202_infer(n);
                        Ty::ERR
                    }
                }
            }
            St::Fn { decl, targs } => self.fn_item_value(n, decl, targs, hint),
            St::NoVal | St::Adt { .. } | St::TypeVal(_) | St::Trait => self.not_value(n),
            St::Assoc { method, impl_args, targs } => {
                let Some(sig) = self.sigs.fn_sig(method) else { return Ty::ERR };
                let mut args: Vec<GArg> = impl_args.unwrap_or_default();
                if args.len() != sig.n_parent as usize {
                    self.e0202_infer(n);
                    return Ty::ERR;
                }
                args.extend(targs.unwrap_or_default());
                self.fn_value_of(n, method, sig, args)
            }
            _ => self.not_value(n),
        }
    }

    fn not_value(&mut self, n: u32) -> Ty {
        let text = self.icx.node_text(n).to_string();
        let d = mk("E0201", Some("not-value"), &[("operand", &text)], self.icx.node_span(n), &[]);
        self.report(d);
        Ty::ERR
    }

    pub fn e0202_infer(&mut self, n: u32) {
        let text = self.icx.node_text(n).to_string();
        let d = mk("E0202", Some("infer"), &[("literal", &text)], self.icx.node_span(n), &[]);
        self.report(d);
    }

    fn fn_item_value(&mut self, n: u32, decl: DeclId, targs: Option<Vec<GArg>>, hint: Option<TypeId>) -> Ty {
        let Some(sig) = self.sigs.fn_sig(decl) else { return Ty::ERR };
        let ng = sig.generics.len();
        let args: Vec<GArg> = match targs {
            Some(a) => a,
            None if ng == 0 => Vec::new(),
            None => {
                // infer from the expected fn type
                let mut binds: Vec<Option<GArg>> = vec![None; ng];
                if let Some(h) = hint {
                    let ft = self.sig_fn_type(sig);
                    let _ = self.w.match_ty(&self.tys, ft, h, &mut binds);
                }
                if binds.iter().any(|b| b.is_none()) {
                    self.e0202_infer(n);
                    return Ty::ERR;
                }
                binds.into_iter().flatten().collect()
            }
        };
        self.fn_value_of(n, decl, sig, args)
    }

    fn sig_fn_type(&mut self, sig: &crate::sigs::FnSig) -> TypeId {
        let k = TypeKind::Fn {
            params: sig.params.clone().into_boxed_slice(),
            ret: sig.ret,
            features: sig.features,
        };
        if sig.params.contains(&ids::ERROR) || sig.ret == ids::ERROR {
            return ids::ERROR;
        }
        self.tys.intern(k)
    }

    fn fn_value_of(&mut self, n: u32, decl: DeclId, sig: &crate::sigs::FnSig, args: Vec<GArg>) -> Ty {
        let ft = self.sig_fn_type(sig);
        let t = self.tys.subst(ft, &args, None);
        let tok = self.icx.end_tok(n);
        self.target(tok, Target::Fn { decl, args: args.clone() });
        self.site(tok, decl, &args);
        self.coerce_note(n, Coercion::FnItem);
        if sig.is_unsafe {
            // E0807: fn types carry no `unsafe` marker, so a call through the value
            // would skip E0803 (also for `extern` imports, whose calls are unsafe).
            let path = self.prog.decl_path(decl);
            let d = mk("E0807", None, &[("name", &path)], self.icx.node_span(n), &[]);
            self.report(d);
        } else if crate::threads::is_checked_thread_fn(self.prog, decl) {
            // a call through the value would skip the capture checks of crate::threads
            let path = self.prog.decl_path(decl);
            let d = mk("E0807", Some("thread"), &[("name", &path)], self.icx.node_span(n), &[]);
            self.report(d);
        }
        Ty::T(t)
    }

    /// A postfix chain (see the module docs): a `Postfix`, or a bare `Primary` path (no steps).
    fn postfix(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let kids = self.kids(n);
        let Some(prim) = icx.chain_head(n) else {
            self.free_kids(kids);
            return Ty::ERR;
        };
        let pa = icx.a(prim);
        let pend = icx.postfix_end(n);
        let prim_end = icx.primary_end(prim);
        let only = pend == prim_end;
        let mut kid = 1usize;
        let mut st = if icx.is(pa, Term::Ident) {
            let targs = icx.child(prim, NodeKind::GenericArgs);
            match icx.use_at(prim, pa) {
                Some(u) => {
                    let s = self.path_state(u.res, targs, pa, None);
                    if let St::Val(t) = &s {
                        self.set(prim, *t);
                    }
                    s
                }
                None => St::Err,
            }
        } else if icx.is(pa, Term::KwSelfType) {
            match self.self_ty {
                Some(t) => St::TypeVal(t),
                None => St::TypeVal(ids::SELF_TY),
            }
        } else {
            let t = self.expr(prim, if only { hint } else { None });
            St::Val(t)
        };
        let mut p = prim_end + 1;
        for _ in 0..10_000_000 {
            let t = icx.tk(p);
            if t == Term::Dot as u8 && icx.is(p + 1, Term::Ident) {
                let seg = p + 1;
                let mut q = p + 2;
                let mut targs_node = None;
                if icx.is(q, Term::ColonColon) {
                    targs_node = kids.get(kid).copied();
                    kid += 1;
                    let (e, partial) = icx.angle_end(q + 1);
                    q = if partial { e - 1 } else { e };
                }
                if let Some(u) = icx.use_at(prim, seg) {
                    st = self.step_path(st, u.res, targs_node, seg);
                    p = q;
                    continue;
                }
                if let Some(d) = icx.deferred_at(prim, seg)
                    && d.kind == DeferKind::TypeRelative
                {
                    st = self.type_rel(st, seg, targs_node);
                    p = q;
                    continue;
                }
                // a member of a value (a step after a type the resolver left unresolved: already
                // diagnosed, or a method of a dropped impl)
                if matches!(st, St::Adt { .. } | St::TypeVal(_) | St::NoVal | St::Trait) {
                    st = St::Err;
                }
                let recv = self.value_of(st, prim, None);
                if icx.is(q, Term::LParen) {
                    let close = icx.close(q);
                    let args = if close == q + 1 {
                        crate::itemcx::Kids::default()
                    } else {
                        let a = kids.get(kid).copied();
                        kid += 1;
                        a.map(|a| icx.kids_vec(a)).unwrap_or_default()
                    };
                    let h = if close == pend { hint } else { None };
                    let r = self.method_call(n, recv, seg, targs_node, &args, h);
                    if self.resolve(r) == Ty::T(ids::NEVER) {
                        self.diverges = true;
                    }
                    st = St::Val(r);
                    p = close + 1;
                } else {
                    let recv = self.force(recv);
                    let sym = icx.sym(seg);
                    let r = self.field_of(recv, sym, icx.tok_span(seg), Some(seg));
                    st = St::Val(Ty::T(r));
                    p = q;
                }
                continue;
            }
            if t == Term::Dot as u8 && icx.is(p + 1, Term::IntLit) {
                let recv = self.value_of(st, prim, None);
                let recv = self.force(recv);
                let idx: Option<usize> = icx.tok_str(p + 1).parse().ok();
                let (inner, derefs) = self.tys.peel_refs(recv);
                let r = match (self.tys.kind(inner).clone(), idx) {
                    (TypeKind::Tuple(ts), Some(i)) if i < ts.len() => {
                        self.target(p + 1, Target::TupleField { index: i as u32, derefs });
                        ts[i]
                    }
                    (TypeKind::Error, _) => ids::ERROR,
                    _ => {
                        let tn = self.show(recv);
                        let nm = icx.tok_str(p + 1).to_string();
                        let d = mk("E0208", None, &[("name", &nm), ("type", &tn)], icx.tok_span(p + 1), &[]);
                        self.report(d);
                        ids::ERROR
                    }
                };
                st = St::Val(Ty::T(r));
                p += 2;
                continue;
            }
            if t == Term::DotLBrace as u8 {
                let close = icx.close(p);
                let sl = kids.get(kid).copied();
                kid += 1;
                let h = if close == pend { hint } else { None };
                let r = match sl {
                    Some(sl) => self.struct_lit(st, sl, p, h, n),
                    None => Ty::ERR,
                };
                st = St::Val(r);
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
                let h = if close == pend { hint } else { None };
                let r = self.call(n, st, p, &args, h);
                if self.resolve(r) == Ty::T(ids::NEVER) {
                    self.diverges = true;
                }
                st = St::Val(r);
                p = close + 1;
                continue;
            }
            if t == Term::LBracket as u8 {
                let close = icx.close(p);
                let ie = kids.get(kid).copied();
                kid += 1;
                let base = self.value_of(st, prim, None);
                let base = self.force(base);
                let r = self.index(base, ie, p);
                st = St::Val(Ty::T(r));
                p = close + 1;
                continue;
            }
            if t == Term::Question as u8 {
                let base = self.value_of(st, prim, None);
                let base = self.force(base);
                let r = self.try_op(base, p);
                st = St::Val(Ty::T(r));
                p += 1;
                continue;
            }
            break;
        }
        self.free_kids(kids);
        self.value_of(st, if only { prim } else { n }, hint)
    }

    /// A type-relative step `T.m` (deferred by the resolver).
    fn type_rel(&mut self, st: St, seg: u32, targs: Option<u32>) -> St {
        let icx = self.icx;
        let base = match st {
            St::TypeVal(t) => t,
            St::Adt { def, args: Some(a) } => self.tys.mk_adt(def, a),
            St::Adt { def, args: None } if self.sigs.generics(def).is_empty() => {
                self.tys.mk_adt(def, Vec::new())
            }
            St::Err => return St::Err,
            _ => return St::Err,
        };
        let name = icx.sym(seg);
        let text = self.prog.strings.get(name);
        if base == ids::ERROR {
            return St::Err;
        }
        if matches!(self.tys.kind(base), TypeKind::Vector { .. } | TypeKind::Mask { .. })
            && builtins::builtin_assoc(&mut self.tys, base, text).is_some()
        {
            return St::TypeAssoc { ty: base, name, tok: seg };
        }
        let genv = self.genv();
        // trait `Self` in a trait body: methods of the trait itself
        match self.w.lookup_method(&mut self.tys, base, name, genv) {
            Lookup::Found { m: MethodRef::Impl { method, impl_args, .. }, level: 0, .. } => {
                let ga = targs.and_then(|g| {
                    let sig = self.sigs.fn_sig(method);
                    let own: Vec<bool> = sig.map_or(Vec::new(), |s| {
                        s.generics[s.n_parent as usize..].iter().map(|g| g.is_const).collect()
                    });
                    self.turbofish(g, &own, g)
                });
                St::Assoc { method, impl_args: Some(impl_args), targs: ga }
            }
            Lookup::Found { m: MethodRef::Bound { method, trait_args, self_ty }, level: 0, .. } => {
                let ga = targs.and_then(|g| {
                    let sig = self.sigs.fn_sig(method);
                    let own: Vec<bool> = sig.map_or(Vec::new(), |s| {
                        s.generics[s.n_parent as usize..].iter().map(|g| g.is_const).collect()
                    });
                    self.turbofish(g, &own, g)
                });
                St::BoundAssoc { method, trait_args, self_ty, targs: ga }
            }
            Lookup::Ambiguous { traits, methods, recv } => {
                self.e0908(seg, recv, traits, methods);
                St::Err
            }
            Lookup::NotInBounds { param } => {
                self.e0901(seg, param);
                St::Err
            }
            _ => {
                if base == ids::SELF_TY {
                    // `Self.m` inside a trait: a trait method of the enclosing trait
                    return St::Err;
                }
                let tn = self.show(base);
                let d = mk("E0208", None, &[("name", text), ("type", &tn)], icx.tok_span(seg), &[]);
                self.report(d);
                St::Err
            }
        }
    }

    fn e0908(&mut self, seg: u32, recv: TypeId, traits: [DeclId; 2], methods: [DeclId; 2]) {
        let name = self.icx.tok_str(seg).to_string();
        let tn = self.show(recv);
        let (p1, p2) = (self.prog.decl_path(traits[0]), self.prog.decl_path(traits[1]));
        let mut roles: Vec<(&str, Span)> = Vec::new();
        if let Some(d) = self.prog.decl(methods[0]) {
            roles.push(("first", d.name_span));
        }
        if let Some(d) = self.prog.decl(methods[1]) {
            roles.push(("second", d.name_span));
        }
        let d = mk(
            "E0908",
            None,
            &[("name", &name), ("type", &tn), ("first", &p1), ("second", &p2)],
            self.icx.tok_span(seg),
            &roles,
        );
        self.report(d);
    }

    // ---------- fields, index, try ----------

    /// Field `name` of a value of type `t` (auto-deref through references).
    pub fn field_of(&mut self, t: TypeId, name: Sym, at: Span, key: Option<u32>) -> TypeId {
        if t == ids::ERROR || self.tys.has_error(t) {
            return ids::ERROR;
        }
        let (inner, derefs) = self.tys.peel_refs(t);
        if let TypeKind::Adt { def, args } = self.tys.kind(inner).clone()
            && let Some(adt) = self.sigs.adt(def)
            && !adt.is_enum
            && let Some(v) = adt.variants.first()
            && let Some((i, f)) = v.fields.iter().enumerate().find(|(_, f)| f.name == name)
        {
            let ft = self.tys.subst(f.ty, &args, None);
            if let Some(fd) = self.prog.decl(f.decl)
                && !fd.is_pub()
                && fd.module != self.icx.res_module()
            {
                let module = self.prog.module(fd.module).map_or(String::new(), |m| m.path_text.clone());
                let nm = self.prog.strings.get(name).to_string();
                let d =
                    mk("E0106", None, &[("name", &nm), ("module", &module)], at, &[("decl", fd.name_span)]);
                self.report(d);
            }
            if let Some(k) = key {
                self.target(k, Target::Field { decl: def, index: i as u32, derefs });
            }
            return ft;
        }
        let tn = self.show(t);
        let nm = self.prog.strings.get(name);
        let nm = if nm.is_empty() { "_" } else { nm };
        let d = mk("E0208", None, &[("name", nm), ("type", &tn)], at, &[]);
        self.report(d);
        ids::ERROR
    }

    /// `Some(inclusive)` when index expression `e` is a range written in the brackets.
    fn range_index(&self, e: u32) -> Option<bool> {
        let icx = self.icx;
        if icx.kind(e) != Some(NodeKind::RangeExpr) {
            return None;
        }
        let r = icx.kids_vec(e).get(1).copied()?;
        Some(icx.is(icx.a(r).saturating_sub(1), Term::DotDotEq))
    }

    fn index(&mut self, base: TypeId, ie: Option<u32>, tok: u32) -> TypeId {
        let range = ie.and_then(|e| self.range_index(e));
        if let Some(e) = ie {
            let want = match range {
                Some(inclusive) => self.tys.intern(TypeKind::Range { elem: ids::USIZE, inclusive }),
                None => ids::USIZE,
            };
            self.check_expr(e, want, None);
        }
        if base == ids::ERROR || self.tys.has_error(base) {
            return ids::ERROR;
        }
        let (inner, derefs) = self.tys.peel_refs(base);
        match self.tys.kind(inner).clone() {
            TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } if range.is_some() => {
                self.target(tok, Target::Slice { derefs });
                self.tys.intern(TypeKind::Slice { elem, align: 0 })
            }
            TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } => {
                self.target(tok, Target::Index { derefs });
                elem
            }
            _ => {
                let d = mk(
                    "E0206",
                    None,
                    &[("op", "[]"), ("type", &self.show(base))],
                    self.icx.tok_span(tok),
                    &[],
                );
                self.report(d);
                ids::ERROR
            }
        }
    }

    fn try_op(&mut self, base: TypeId, tok: u32) -> TypeId {
        if base == ids::ERROR || self.tys.has_error(base) {
            return ids::ERROR;
        }
        let lang = &self.sigs.lang;
        let (opt, res) = (lang.option, lang.result);
        let ret = self.rets.last().map_or(ids::ERROR, |r| r.ty);
        let rspan = self.rets.last().and_then(|r| r.span);
        let TypeKind::Adt { def, args } = self.tys.kind(base).clone() else {
            let d =
                mk("E0206", None, &[("op", "?"), ("type", &self.show(base))], self.icx.tok_span(tok), &[]);
            self.report(d);
            return ids::ERROR;
        };
        let inner = match args.first() {
            Some(GArg::Type(t)) => *t,
            _ => ids::ERROR,
        };
        if def != opt && def != res || def.is_none() {
            let d =
                mk("E0206", None, &[("op", "?"), ("type", &self.show(base))], self.icx.tok_span(tok), &[]);
            self.report(d);
            return ids::ERROR;
        }
        if ret == ids::ERROR {
            return inner;
        }
        let ok = match self.tys.kind(ret).clone() {
            TypeKind::Adt { def: rd, args: ra } if rd == def => def == opt || ra.get(1) == args.get(1),
            _ => false,
        };
        if !ok {
            let expected = self.show(ret);
            let found = self.show(base);
            let roles: Vec<(&str, Span)> = rspan.map(|s| ("expected", s)).into_iter().collect();
            let d = mk(
                "E0201",
                None,
                &[("expected", &expected), ("found", &found), ("operand", "?")],
                self.icx.tok_span(tok),
                &roles,
            );
            self.report(d);
        }
        inner
    }

    // ---------- calls ----------

    /// Whether `t` (callee numbering) mentions a generic param without a binding.
    fn unresolved(&self, t: TypeId, binds: &[Option<GArg>]) -> bool {
        let mut stack = vec![t];
        let mut kids = Vec::new();
        let mut steps = 0;
        while let Some(x) = stack.pop() {
            steps += 1;
            if steps > 100_000 {
                return true;
            }
            let k = self.tys.kind(x);
            match k {
                TypeKind::Param(i) => {
                    if binds.get(*i as usize).is_none_or(|b| b.is_none()) {
                        return true;
                    }
                }
                TypeKind::Array { len: ConstArg::Param(i), .. } => {
                    if binds.get(*i as usize).is_none_or(|b| b.is_none()) {
                        return true;
                    }
                }
                TypeKind::Adt { args, .. } => {
                    for a in args.iter() {
                        if let GArg::Const(ConstArg::Param(i)) = a
                            && binds.get(*i as usize).is_none_or(|b| b.is_none())
                        {
                            return true;
                        }
                    }
                }
                _ => {}
            }
            kids.clear();
            k.children(&mut kids);
            stack.extend_from_slice(&kids);
        }
        false
    }

    fn subst_binds(&mut self, t: TypeId, binds: &[Option<GArg>]) -> TypeId {
        let args: Vec<GArg> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| b.unwrap_or(GArg::Type(self.tys.mk_param_placeholder(i))))
            .collect();
        self.tys.subst(t, &args, None)
    }

    /// Infers bindings from `t` against pattern `p` (lenient about coercions).
    fn infer(&mut self, p: TypeId, t: TypeId, binds: &mut [Option<GArg>]) {
        let mut b2 = binds.to_vec();
        if self.w.match_ty(&self.tys, p, t, &mut b2) {
            binds.copy_from_slice(&b2);
            return;
        }
        // a fn value with other features: the coercion of the argument decides (E0707)
        if let (TypeKind::Fn { features: pf, .. }, TypeKind::Fn { params, ret, features }) =
            (self.tys.kind(p).clone(), self.tys.kind(t).clone())
            && pf != features
        {
            let t2 = self.tys.intern(TypeKind::Fn { params, ret, features: pf });
            let mut b5 = binds.to_vec();
            if self.w.match_ty(&self.tys, p, t2, &mut b5) {
                binds.copy_from_slice(&b5);
            }
            return;
        }
        // `&mut T` for `&T`, `&[T; N]` for `&[T]`
        if let TypeKind::Ref { inner, .. } = self.tys.kind(t).clone() {
            let shared = self.tys.mk_ref(false, inner);
            let mut b3 = binds.to_vec();
            if self.w.match_ty(&self.tys, p, shared, &mut b3) {
                binds.copy_from_slice(&b3);
                return;
            }
            if let TypeKind::Array { elem, .. } = self.tys.kind(inner).clone()
                && let TypeKind::Ref { mutable, .. } = self.tys.kind(p).clone()
            {
                let s = self.tys.intern(TypeKind::Slice { elem, align: 0 });
                let r = self.tys.mk_ref(mutable, s);
                let mut b4 = binds.to_vec();
                if self.w.match_ty(&self.tys, p, r, &mut b4) {
                    binds.copy_from_slice(&b4);
                }
            }
        }
    }

    /// Checks argument nodes against parameter types `params` (callee numbering, bindings in
    /// `binds`): two passes so literals take the types inferred from the other arguments
    /// (literal typing), then the expected type `hint` against `ret`. Unresolved bindings after
    /// that are E0202 case `infer` at `at`.
    #[allow(clippy::too_many_arguments)]
    fn check_args(
        &mut self,
        at: u32,
        params: &[TypeId],
        spans: &[Span],
        binds: &mut [Option<GArg>],
        args: &[u32],
        ret: TypeId,
        hint: Option<TypeId>,
    ) {
        let mut pending: Vec<(usize, Ty)> = Vec::new();
        for (i, &a) in args.iter().enumerate() {
            let Some(&p) = params.get(i) else {
                let t = self.expr(a, None);
                self.force(t);
                continue;
            };
            let why = spans.get(i).copied();
            if !self.unresolved(p, binds) {
                let want = self.subst_binds(p, binds);
                let t = self.expr(a, Some(want));
                let t = self.coerce(a, t, want, why);
                self.note_reborrow(a, t, want);
            } else {
                let t = self.expr(a, None);
                match self.resolve(t) {
                    Ty::Lit(_) => pending.push((i, t)),
                    Ty::T(x) => {
                        if x != ids::ERROR && x != ids::NEVER {
                            self.infer(p, x, binds);
                        }
                        pending.push((i, Ty::T(x)));
                    }
                }
            }
        }
        if let Some(h) = hint
            && binds.iter().any(|b| b.is_none())
            && !self.unresolved(ret, &vec![Some(GArg::Type(ids::ERROR)); binds.len()])
        {
            self.infer(ret, h, binds);
        }
        let mut lit_err = false;
        // generic positions: literals last
        pending.sort_by_key(|(_, t)| matches!(t, Ty::Lit(_)));
        for (i, t) in pending {
            let a = args[i];
            let p = params[i];
            let why = spans.get(i).copied();
            if self.unresolved(p, binds) {
                if let Ty::Lit(g) = self.resolve(t) {
                    self.e0202(g);
                    lit_err = true;
                }
                continue;
            }
            let want = self.subst_binds(p, binds);
            let t2 = self.coerce(a, t, want, why);
            self.note_reborrow(a, t2, want);
        }
        if binds.iter().any(|b| b.is_none()) {
            if !lit_err {
                self.e0202_infer(at);
            }
            for b in binds.iter_mut() {
                if b.is_none() {
                    *b = Some(GArg::Type(ids::ERROR));
                }
            }
        }
    }

    fn note_reborrow(&mut self, a: u32, t: TypeId, want: TypeId) {
        if t == want && matches!(self.tys.kind(want), TypeKind::Ref { mutable: true, .. }) {
            self.coerce_note(a, Coercion::Reborrow);
        }
    }

    /// Bound (E0901 `unsatisfied`, E1407 for `Flat`) and view (E0411/E0412/E0414,
    /// [`crate::kinds::view_arg`]) checks of a call's generic args.
    fn check_bounds(&mut self, at: Span, generic: DeclId, gens: &[GenericParam], args: &[GArg]) {
        let genv = self.genv();
        for (i, g) in gens.iter().enumerate() {
            let Some(GArg::Type(a)) = args.get(i).copied() else { continue };
            if a == ids::ERROR || self.tys.has_error(a) {
                continue;
            }
            // A view is never Flat: under a `Flat` bound, E1407 (naming the reference) replaces
            // the view diagnostics, whose fix (`T: view`) would not help.
            let flat = self.sigs.lang.flat;
            let flat_bound = flat.is_some() && g.bounds.iter().any(|b| b.trait_decl == flat);
            if !g.view && !flat_bound && self.w.is_view(&self.tys, a, genv) {
                let tn = self.show(a);
                let d = crate::kinds::view_arg(self.prog, generic, g, &tn, at);
                self.report(d);
                continue;
            }
            for b in &g.bounds {
                let bargs: Vec<GArg> = b
                    .args
                    .iter()
                    .map(|x| match x {
                        GArg::Type(t) => GArg::Type(self.tys.subst(*t, args, None)),
                        c => *c,
                    })
                    .collect();
                let tr = TraitRef { trait_decl: b.trait_decl, args: bargs };
                if b.trait_decl == self.sigs.lang.flat && b.trait_decl.is_some() {
                    if let Some(wit) = self.w.flat_witness(&mut self.tys, a, genv) {
                        self.e1407(at, a, &wit, g.name_span);
                    }
                    continue;
                }
                if !self.w.implements(&mut self.tys, a, &tr, genv) {
                    let tn = self.show(a);
                    let trn = self
                        .prog
                        .decl(b.trait_decl)
                        .map_or("_".to_string(), |d| self.prog.strings.get(d.name).to_string());
                    let d = mk(
                        "E0901",
                        Some("unsatisfied"),
                        &[("name", &tn), ("param", &trn)],
                        at,
                        &[("param", g.name_span)],
                    );
                    self.report(d);
                }
            }
        }
    }

    /// E1407: `ty` (given for a `Flat`-bound param declared at `param`) is not Flat; `wit` names
    /// its first non-flat component.
    fn e1407(&mut self, at: Span, ty: TypeId, wit: &crate::traits::FlatWitness, param: Span) {
        use crate::traits::NotFlat;
        let tn = self.show(ty);
        let d = match wit.why {
            NotFlat::Unknown => return,
            NotFlat::Depth => {
                let limit = crate::traits::QUERY_DEPTH.to_string();
                mk("E1407", Some("depth"), &[("type", &tn), ("limit", &limit)], at, &[("param", param)])
            }
            why if wit.path.is_empty() => {
                mk("E1407", Some("self"), &[("type", &tn), ("reason", why.text())], at, &[("param", param)])
            }
            why => {
                let pn = self.show(wit.part);
                let holes =
                    [("type", tn.as_str()), ("path", &wit.path), ("part", &pn), ("reason", why.text())];
                mk("E1407", None, &holes, at, &[("param", param)])
            }
        };
        self.report(d);
    }

    fn call_text_span(&self, n: u32, paren: u32) -> Span {
        self.icx.range_span(self.icx.a(n), paren.saturating_sub(1))
    }

    /// The callee name token of a call whose `(` is at `paren` (the token before `(`, or before
    /// its `::<..>`).
    fn callee_tok(&self, n: u32, paren: u32) -> u32 {
        let icx = self.icx;
        let mut t = paren.saturating_sub(1);
        if icx.is(t, Term::Gt) || icx.is(t, Term::GtGt) {
            // back over `::<..>`
            let mut k = t;
            while k > icx.a(n) && !icx.is(k, Term::ColonColon) {
                k -= 1;
            }
            t = k.saturating_sub(1);
        }
        t
    }

    /// A call `(args)` on state `st`.
    fn call(&mut self, n: u32, st: St, paren: u32, args: &[u32], hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let ctok = self.callee_tok(n, paren);
        let cspan = icx.tok_span(ctok);
        match st {
            St::Fn { decl, targs } => {
                let Some(sig) = self.sigs.fn_sig(decl) else { return self.args_only(args) };
                if !self.arity(n, paren, sig.params.len(), args.len(), sig.variadic) {
                    return self.args_only(args);
                }
                if sig.is_unsafe {
                    self.unsafe_op(cspan);
                }
                self.features_ok(cspan, decl, sig.features);
                if sig.abi == crate::sigs::Abi::Intrinsic {
                    self.intrinsic_arch(cspan, decl);
                    self.target(paren, Target::Intrinsic { decl });
                }
                let mut binds: Vec<Option<GArg>> = vec![None; sig.generics.len()];
                if let Some(t) = targs {
                    for (i, a) in t.into_iter().enumerate() {
                        if let Some(b) = binds.get_mut(i) {
                            *b = Some(a);
                        }
                    }
                }
                let params = sig.params.clone();
                self.check_args(n, &params, &sig.param_spans, &mut binds, args, sig.ret, hint);
                let gargs: Vec<GArg> = binds.into_iter().flatten().collect();
                self.check_bounds(cspan, decl, &sig.generics, &gargs);
                if sig.abi != crate::sigs::Abi::Intrinsic {
                    self.target(paren, Target::Fn { decl, args: gargs.clone() });
                }
                self.site(paren, decl, &gargs);
                Ty::T(self.tys.subst(sig.ret, &gargs, None))
            }
            St::Assoc { method, impl_args, targs } => {
                let Some(sig) = self.sigs.fn_sig(method) else { return self.args_only(args) };
                if !self.arity(n, paren, sig.params.len(), args.len(), false) {
                    return self.args_only(args);
                }
                self.method_checks(method, cspan, ctok);
                let np = sig.n_parent as usize;
                let mut binds: Vec<Option<GArg>> = vec![None; sig.generics.len()];
                if let Some(ia) = impl_args {
                    for (i, a) in ia.into_iter().enumerate().take(np) {
                        binds[i] = Some(a);
                    }
                }
                if let Some(t) = targs {
                    for (i, a) in t.into_iter().enumerate() {
                        if let Some(b) = binds.get_mut(np + i) {
                            *b = Some(a);
                        }
                    }
                }
                let params = sig.params.clone();
                self.check_args(n, &params, &sig.param_spans, &mut binds, args, sig.ret, hint);
                let gargs: Vec<GArg> = binds.into_iter().flatten().collect();
                if let Some(imp) = self.prog.decl(method).map(|d| d.parent) {
                    let ig = self.sigs.impl_def(imp).map(|i| i.generics.clone()).unwrap_or_default();
                    self.check_bounds(cspan, imp, &ig, &gargs[..np.min(gargs.len())]);
                }
                let own: Vec<GenericParam> = sig.generics[np..].to_vec();
                let own_args: Vec<GArg> = gargs.get(np..).unwrap_or(&[]).to_vec();
                self.check_bounds_own(cspan, method, &own, &gargs, &own_args);
                self.target(paren, Target::Fn { decl: method, args: gargs.clone() });
                self.site(paren, method, &gargs);
                Ty::T(self.tys.subst(sig.ret, &gargs, None))
            }
            St::TraitMethod { method, trait_decl, targs } => {
                self.trait_call(n, paren, method, trait_decl, None, Vec::new(), targs, args, hint)
            }
            St::BoundAssoc { method, trait_args, self_ty, targs } => {
                let td = self.prog.decl(method).map_or(DeclId::NONE, |d| d.parent);
                self.trait_call(n, paren, method, td, Some(self_ty), trait_args, targs, args, hint)
            }
            St::Variant { decl, adt, index, args: vargs } => {
                let Some(adef) = self.sigs.adt(adt) else { return self.args_only(args) };
                let Some(v) = adef.variants.get(index as usize) else { return self.args_only(args) };
                if v.shape != VariantShape::Tuple {
                    let text = self.icx.node_text(n).to_string();
                    let ph = self.tys_adt_placeholder(adt);
                    let found = self.show(ph);
                    let d = mk(
                        "E0201",
                        Some("not-callable"),
                        &[("operand", &text), ("found", &found)],
                        cspan,
                        &[],
                    );
                    self.report(d);
                    return self.args_only(args);
                }
                if !self.arity(n, paren, v.fields.len(), args.len(), false) {
                    return self.args_only(args);
                }
                let params: Vec<TypeId> = v.fields.iter().map(|f| f.ty).collect();
                let spans: Vec<Span> = v.fields.iter().map(|f| f.span).collect();
                let mut binds: Vec<Option<GArg>> = vec![None; adef.generics.len()];
                if let Some(va) = vargs {
                    for (i, a) in va.into_iter().enumerate() {
                        if let Some(b) = binds.get_mut(i) {
                            *b = Some(a);
                        }
                    }
                } else if let Some(h) = hint
                    && let TypeKind::Adt { def, args: hargs } = self.tys.kind(h).clone()
                    && def == adt
                {
                    // the expected type gives the ADT's args
                    for (i, a) in hargs.iter().enumerate() {
                        if let Some(b) = binds.get_mut(i)
                            && !matches!(a, GArg::Type(t) if self.tys.has_error(*t))
                        {
                            *b = Some(*a);
                        }
                    }
                }
                let ret = self.adt_self_type(adt, adef.generics.len());
                self.check_args(n, &params, &spans, &mut binds, args, ret, hint);
                let gargs: Vec<GArg> = binds.into_iter().flatten().collect();
                let gens = adef.generics.clone();
                self.check_bounds(cspan, adt, &gens, &gargs);
                self.target(paren, Target::Variant { decl, index });
                let t = self.tys.mk_adt(adt, gargs);
                self.owned(paren, t);
                Ty::T(t)
            }
            St::Builtin(Builtin::FenceStream, _) => {
                if !args.is_empty() {
                    self.arity(n, paren, 0, args.len(), false);
                    return self.args_only(args);
                }
                self.target(paren, Target::FenceStream);
                Ty::T(ids::UNIT)
            }
            St::Builtin(b, targs) => {
                if !args.is_empty() {
                    self.arity(n, paren, 0, args.len(), false);
                    return self.args_only(args);
                }
                let t = match targs.as_deref() {
                    Some([GArg::Type(t)]) => *t,
                    _ => {
                        self.e0202_infer(n);
                        return Ty::ERR;
                    }
                };
                self.target(paren, Target::SizeOf { ty: t, align: b == Builtin::AlignOf });
                Ty::T(ids::USIZE)
            }
            St::TypeAssoc { ty, name, tok } => {
                let text = self.prog.strings.get(name);
                let Some((ps, ret)) = builtins::builtin_assoc(&mut self.tys, ty, text) else {
                    return self.args_only(args);
                };
                if !self.arity(n, paren, ps.len(), args.len(), false) {
                    return self.args_only(args);
                }
                let ats = self.builtin_args(&ps, args);
                let nm = builtins::builtin_assoc_name(text);
                self.target(
                    tok,
                    Target::Builtin {
                        name: nm,
                        recv: ty,
                        adjust: Adjust::default(),
                        ret,
                        consts: Vec::new(),
                    },
                );
                self.gate_assoc(tok, nm, ty, ret, &ats);
                Ty::T(ret)
            }
            St::Val(t) => {
                let t = self.force(t);
                if t == ids::ERROR || self.tys.has_error(t) {
                    return self.args_only(args);
                }
                match self.tys.kind(t).clone() {
                    TypeKind::Fn { params, ret, features } => {
                        if !self.arity(n, paren, params.len(), args.len(), false) {
                            return self.args_only(args);
                        }
                        if features & !self.proven_features() != 0 {
                            self.features_ok(cspan, DeclId::NONE, features);
                        }
                        for (i, &a) in args.iter().enumerate() {
                            let want = params[i];
                            let t2 = self.check_expr(a, want, None);
                            self.note_reborrow(a, t2, want);
                        }
                        self.target(paren, Target::FnValue { ty: t });
                        Ty::T(ret)
                    }
                    _ => {
                        let text = self.icx.node_text(n).to_string();
                        let found = self.show(t);
                        let d = mk(
                            "E0201",
                            Some("not-callable"),
                            &[("operand", &text), ("found", &found)],
                            cspan,
                            &[],
                        );
                        self.report(d);
                        self.args_only(args)
                    }
                }
            }
            St::Err => self.args_only(args),
            _ => {
                let text = self.icx.node_text(n).to_string();
                let d = mk("E0201", Some("not-callable"), &[("operand", &text), ("found", "_")], cspan, &[]);
                self.report(d);
                self.args_only(args)
            }
        }
    }

    fn tys_adt_placeholder(&mut self, adt: DeclId) -> TypeId {
        let n = self.sigs.adt(adt).map_or(0, |a| a.generics.len());
        let args: Vec<GArg> = (0..n).map(|_| GArg::Type(ids::ERROR)).collect();
        if args.is_empty() { self.tys.mk_adt(adt, args) } else { ids::ERROR }
    }

    /// `Adt<Param(0), .., Param(n-1)>` (an ADT's own generic instance, callee numbering).
    fn adt_self_type(&mut self, adt: DeclId, n: usize) -> TypeId {
        let gens = self.sigs.adt(adt).map(|a| a.generics.clone()).unwrap_or_default();
        let args: Vec<GArg> = (0..n)
            .map(|i| {
                if gens.get(i).is_some_and(|g| g.is_const) {
                    GArg::Const(ConstArg::Param(i as u32))
                } else {
                    GArg::Type(self.tys.mk_param(i as u32))
                }
            })
            .collect();
        self.tys.mk_adt(adt, args)
    }

    fn check_bounds_own(
        &mut self,
        at: Span,
        method: DeclId,
        own: &[GenericParam],
        all: &[GArg],
        _own_args: &[GArg],
    ) {
        // bounds of a method's own generics, args indexed in the flat list
        let np = all.len().saturating_sub(own.len());
        let mut padded: Vec<GenericParam> = Vec::with_capacity(all.len());
        for _ in 0..np {
            padded.push(GenericParam {
                name: Sym::NONE,
                item: crate::ids::ItemId::NONE,
                node: 0,
                name_span: at,
                is_const: false,
                const_ty: ids::ERROR,
                bounds: Vec::new(),
                view: true,
            });
        }
        padded.extend(own.iter().cloned());
        self.check_bounds(at, method, &padded, all);
    }

    fn arity(&mut self, n: u32, paren: u32, want: usize, got: usize, variadic: bool) -> bool {
        if got == want || (variadic && got >= want) {
            return true;
        }
        let text = self.icx.span_text(self.call_text_span(n, paren)).to_string();
        let d = mk(
            "E0201",
            Some("arity"),
            &[("operand", &text), ("count", &want.to_string()), ("given", &got.to_string())],
            self.icx.tok_span(paren),
            &[],
        );
        self.report(d);
        false
    }

    fn args_only(&mut self, args: &[u32]) -> Ty {
        for &a in args {
            let t = self.expr(a, None);
            if let Ty::Lit(g) = self.resolve(t) {
                let _ = self.fix(g, ids::ERROR);
            }
        }
        Ty::ERR
    }

    fn builtin_args(&mut self, ps: &[ArgTy], args: &[u32]) -> Vec<TypeId> {
        let mut out = Vec::new();
        for (i, &a) in args.iter().enumerate() {
            match ps.get(i) {
                Some(&ArgTy::Is(t)) => {
                    // a `&mut` argument is reborrowed, as for fn params
                    let t2 = self.check_expr(a, t, None);
                    self.note_reborrow(a, t2, t);
                    out.push(t2);
                }
                Some(ArgTy::Mask(m)) => out.push(self.mask_arg(a, *m)),
                Some(&ArgTy::Index(want)) => out.push(self.index_arg(a, want)),
                Some(&ArgTy::Aligned { elem, mutable, bytes }) => {
                    let t = self.aligned_arg(a, elem, mutable, bytes);
                    self.note_reborrow(a, t, t);
                    out.push(t);
                }
                _ => {
                    let t = self.expr(a, None);
                    out.push(self.force(t));
                }
            }
        }
        out
    }

    /// The operands of `c.select(a, b)` on a `bool` `c`, in order, and the result type: `a` and
    /// `b` have one type of any kind (E0201 at `b` otherwise), which is the result's. `a` takes
    /// the expected type, `b` takes `a`'s (so `(i < n).select(xs[i], 0)` types the literal).
    fn select_operands(&mut self, args: &[u32], hint: Option<TypeId>) -> Ty {
        let (Some(&a), Some(&b)) = (args.first(), args.get(1)) else { return Ty::ERR };
        let at = self.expr(a, hint);
        let bh = match self.resolve(at) {
            Ty::T(t) if !t.is_none() && t != ids::NEVER && t != ids::ERROR => Some(t),
            _ => hint,
        };
        let bt = self.expr(b, bh);
        self.unify_operands(b, at, bt)
    }

    /// A mask argument of type `want`: a mask of another width or lane count is E0605.
    fn mask_arg(&mut self, a: u32, want: TypeId) -> TypeId {
        let t = self.expr(a, Some(want));
        let t = self.force(t);
        if t != want
            && let (TypeKind::Mask { .. }, TypeKind::Mask { width, n }) =
                (self.tys.kind(t).clone(), self.tys.kind(want).clone())
        {
            self.e0605(a, t, width, n);
            return ids::ERROR;
        }
        self.coerce(a, Ty::T(t), want, None)
    }

    /// A gather / scatter index ([`ArgTy::Index`]): an unsigned-lane vector of `want`'s fixed lane
    /// count as is, anything else coerced to `want`.
    fn index_arg(&mut self, a: u32, want: TypeId) -> TypeId {
        let t = self.expr(a, Some(want));
        let t = self.force(t);
        if let (TypeKind::Vector { lane, n: Lanes::Fixed(k) }, TypeKind::Vector { n: Lanes::Fixed(wk), .. }) =
            (self.tys.kind(t).clone(), self.tys.kind(want).clone())
            && k == wk
            && matches!(self.tys.kind(lane), TypeKind::Int { signed: false, .. })
        {
            return t;
        }
        self.coerce(a, Ty::T(t), want, None)
    }

    /// E0605 at `a`: mask `found` where `mask{width}x{n}` lanes are needed.
    fn e0605(&mut self, a: u32, found: TypeId, width: u8, n: Lanes) {
        let lane = match width {
            8 => ids::U8,
            16 => ids::U16,
            64 => ids::U64,
            _ => ids::U32,
        };
        let v = self.tys.intern(TypeKind::Vector { lane, n });
        let (ms, vs) = (self.show(found), self.show(v));
        let w = width.to_string();
        let d =
            mk("E0605", None, &[("mask", &ms), ("vector", &vs), ("width", &w)], self.icx.node_span(a), &[]);
        self.report(d);
    }

    /// An `&[elem align A]` argument (`&mut` when `mutable`) with A ≥ `bytes` (E0606).
    fn aligned_arg(&mut self, a: u32, elem: TypeId, mutable: bool, bytes: u32) -> TypeId {
        let t = self.expr(a, None);
        let t = self.force(t);
        if t == ids::ERROR || self.tys.has_error(t) {
            return ids::ERROR;
        }
        if let TypeKind::Ref { mutable: m, inner } = self.tys.kind(t).clone()
            && (m || !mutable)
            && let TypeKind::Slice { elem: e, align } = self.tys.kind(inner).clone()
            && e == elem
        {
            if align < bytes {
                let tn = self.show(t);
                let b = bytes.to_string();
                let d = mk("E0606", None, &[("type", &tn), ("align", &b)], self.icx.node_span(a), &[]);
                self.report(d);
                return ids::ERROR;
            }
            return t;
        }
        let s = self.tys.intern(TypeKind::Slice { elem, align: bytes });
        let want = self.tys.mk_ref(mutable, s);
        self.mismatch(a, Ty::T(t), want, None);
        ids::ERROR
    }

    /// The type of `m.select(a, b)`: `a` and `b` of one vector (or mask) type whose lanes match the
    /// mask `m` (E0603, E0605).
    fn select_ty(&mut self, m: TypeId, ats: &[TypeId], args: &[u32]) -> TypeId {
        let (Some(&a), Some(&b)) = (ats.first(), ats.get(1)) else { return ids::ERROR };
        if a == ids::ERROR || b == ids::ERROR || self.tys.has_error(a) || self.tys.has_error(b) {
            return ids::ERROR;
        }
        let TypeKind::Mask { width, n } = self.tys.kind(m).clone() else { return ids::ERROR };
        let shape = |s: &Self, t: TypeId| match s.tys.kind(t) {
            TypeKind::Vector { lane, n } => Some((types::lane_bits(s.tys.kind(*lane)), *n)),
            TypeKind::Mask { width, n } => Some((*width as u32, *n)),
            _ => None,
        };
        let Some(sa) = shape(self, a) else {
            self.mismatch(args[0], Ty::T(a), m, None);
            return ids::ERROR;
        };
        if a != b {
            if shape(self, b).is_some() {
                let (ls, rs) = (self.show(a), self.show(b));
                let d = mk("E0603", None, &[("left", &ls), ("right", &rs)], self.icx.node_span(args[1]), &[]);
                self.report(d);
            } else {
                self.mismatch(args[1], Ty::T(b), a, None);
            }
            return ids::ERROR;
        }
        if sa != (width as u32, n) {
            let at = self.icx.node_span(args[0]);
            let ms = self.show(m);
            let vs = self.show(a);
            let w = sa.0.to_string();
            let d = mk("E0605", None, &[("mask", &ms), ("vector", &vs), ("width", &w)], at, &[]);
            self.report(d);
            return ids::ERROR;
        }
        a
    }

    /// The lane indices of `v.permute::<[i0, ..]>()` on vector `v` (E0608 for an index past the
    /// lanes); `None` after a reported error.
    fn permute_indices(&mut self, g: u32, v: TypeId) -> Option<Vec<u64>> {
        let icx = self.icx;
        let n = match self.tys.kind(v) {
            TypeKind::Vector { n: Lanes::Fixed(k), .. } => *k as u64,
            _ => return Some(Vec::new()),
        };
        let kids = icx.kids_vec(g);
        let mut out = Vec::new();
        let bad = |s: &mut Self| {
            let text = icx.node_text(g).to_string();
            let d = mk("E0201", Some("const-arg"), &[("operand", &text)], icx.node_span(g), &[]);
            s.report(d);
            None
        };
        let [x] = kids[..] else { return bad(self) };
        let a = icx.a(x);
        if !icx.is(a, Term::LBracket) {
            return bad(self);
        }
        let close = icx.close(a);
        for t in a + 1..close {
            if icx.is(t, Term::Comma) {
                continue;
            }
            if !icx.is(t, Term::IntLit) {
                return bad(self);
            }
            let Some((k, _)) = crate::literals::parse_int(icx.tok_str(t)) else { return bad(self) };
            if k >= n as u128 {
                let (ks, ns) = (k.to_string(), n.to_string());
                let d = mk("E0608", None, &[("index", &ks), ("lanes", &ns)], icx.tok_span(t), &[]);
                self.report(d);
                return None;
            }
            out.push(k as u64);
        }
        if out.len() as u64 != n {
            return bad(self);
        }
        Some(out)
    }

    fn intrinsic_arch(&mut self, at: Span, decl: DeclId) {
        let name = self.prog.decl(decl).map_or("", |d| self.prog.strings.get(d.name));
        let Some((arch, _, _)) = builtins::intrinsic(name) else { return };
        let ok = match arch {
            builtins::IArch::Both => true,
            builtins::IArch::X86 => self.prog.target.arch() == stave_target::Arch::X86_64,
            builtins::IArch::Ptx => false,
        };
        if !ok {
            let path = self.prog.decl_path(decl);
            let f = match arch {
                builtins::IArch::X86 => "x86_64",
                _ => "ptx",
            };
            let roles: Vec<(&str, Span)> =
                self.prog.decl(decl).map(|d| ("callee", d.name_span)).into_iter().collect();
            let d = mk("E0704", None, &[("callee", &path), ("feature", f)], at, &roles);
            self.report(d);
        }
    }

    /// Unsafe, visibility, `Drop.drop` (E0909), and feature checks of a method call.
    fn method_checks(&mut self, method: DeclId, at: Span, tok: u32) {
        let Some(sig) = self.sigs.fn_sig(method) else { return };
        if sig.is_unsafe {
            self.unsafe_op(at);
        }
        self.features_ok(at, method, sig.features);
        let Some(md) = self.prog.decl(method).copied() else { return };
        let imp = self.sigs.impl_def(md.parent);
        let drop = self.sigs.lang.drop;
        let is_drop = drop.is_some()
            && (md.parent == drop
                || imp.and_then(|i| i.trait_ref.as_ref()).is_some_and(|t| t.trait_decl == drop));
        if is_drop {
            let d = mk("E0909", None, &[], at, &[]);
            self.report(d);
        }
        // visibility: inherent methods by their own `pub`, trait impl methods by the trait's
        let owner = match imp.and_then(|i| i.trait_ref.as_ref()) {
            Some(t) => self.prog.decl(t.trait_decl).copied(),
            None if md.kind == DeclKind::Method => Some(md),
            None => None,
        };
        if let Some(o) = owner
            && !o.is_pub()
            && o.module != self.icx.res_module()
        {
            let module = self.prog.module(o.module).map_or(String::new(), |m| m.path_text.clone());
            let nm = self.prog.strings.get(md.name).to_string();
            let d = mk(
                "E0106",
                None,
                &[("name", &nm), ("module", &module)],
                self.icx.tok_span(tok),
                &[("decl", o.name_span)],
            );
            self.report(d);
        }
    }

    /// `Trait.m(x, ..)` (qualified) or `T.m(x)` (through a bound): `Self` inferred from the
    /// arguments when not given.
    #[allow(clippy::too_many_arguments)]
    fn trait_call(
        &mut self,
        n: u32,
        paren: u32,
        method: DeclId,
        trait_decl: DeclId,
        self_ty: Option<TypeId>,
        trait_args: Vec<GArg>,
        targs: Option<Vec<GArg>>,
        args: &[u32],
        hint: Option<TypeId>,
    ) -> Ty {
        let ctok = self.callee_tok(n, paren);
        let cspan = self.icx.tok_span(ctok);
        let Some(sig) = self.sigs.fn_sig(method) else { return self.args_only(args) };
        if !self.arity(n, paren, sig.params.len(), args.len(), false) {
            return self.args_only(args);
        }
        self.method_checks(method, cspan, ctok);
        let ng = sig.generics.len();
        let np = sig.n_parent as usize;
        // `Self` is variable `ng`.
        let self_var = self.tys.mk_param(ng as u32);
        let params: Vec<TypeId> =
            sig.params.iter().map(|&p| self.tys.subst(p, &[], Some(self_var))).collect();
        let ret = self.tys.subst(sig.ret, &[], Some(self_var));
        let mut binds: Vec<Option<GArg>> = vec![None; ng + 1];
        for (i, a) in trait_args.into_iter().enumerate().take(np) {
            binds[i] = Some(a);
        }
        if let Some(t) = targs {
            for (i, a) in t.into_iter().enumerate() {
                if let Some(b) = binds.get_mut(np + i) {
                    *b = Some(a);
                }
            }
        }
        if let Some(s) = self_ty {
            binds[ng] = Some(GArg::Type(s));
        }
        let spans = sig.param_spans.clone();
        self.check_args(n, &params, &spans, &mut binds, args, ret, hint);
        let all: Vec<GArg> = binds.into_iter().flatten().collect();
        let s = match all.get(ng) {
            Some(GArg::Type(t)) => *t,
            _ => ids::ERROR,
        };
        let targs_all: Vec<GArg> = all[..ng.min(all.len())].to_vec();
        if s != ids::ERROR && !self.tys.has_error(s) && trait_decl.is_some() {
            let tr = TraitRef { trait_decl, args: targs_all[..np.min(targs_all.len())].to_vec() };
            let genv = self.genv();
            if !self.w.implements(&mut self.tys, s, &tr, genv) {
                let tn = self.show(s);
                let trn = self
                    .prog
                    .decl(trait_decl)
                    .map_or("_".to_string(), |d| self.prog.strings.get(d.name).to_string());
                let d = mk("E0901", Some("unsatisfied"), &[("name", &tn), ("param", &trn)], cspan, &[]);
                self.report(d);
            }
        }
        self.target(paren, Target::TraitMethod { method, self_ty: s, args: targs_all.clone() });
        let mut site_args = targs_all.clone();
        site_args.push(GArg::Type(s));
        self.site(paren, method, &site_args);
        let r = self.tys.subst(ret, &all, None);
        Ty::T(r)
    }

    /// `recv.name(args)` (method lookup and receiver adjustment).
    fn method_call(
        &mut self,
        n: u32,
        recv: Ty,
        seg: u32,
        targs_node: Option<u32>,
        args: &[u32],
        hint: Option<TypeId>,
    ) -> Ty {
        let icx = self.icx;
        let recv = self.force(recv);
        let name = icx.sym(seg);
        let sspan = icx.tok_span(seg);
        let paren = if let Some(g) = targs_node { icx.angle_end(icx.a(g)).0 } else { seg + 1 };
        if recv == ids::ERROR || self.tys.has_error(recv) {
            return self.args_only(args);
        }
        let genv = self.genv();
        let lk = self.w.lookup_method(&mut self.tys, recv, name, genv);
        match lk {
            Lookup::Found { m, level, recv: c } => match m {
                MethodRef::Builtin(b) => {
                    let mut consts: Vec<u64> = Vec::new();
                    let gargs: Vec<GArg> = match targs_node {
                        Some(g) if b.name == "permute" => {
                            match self.permute_indices(g, c) {
                                Some(ks) => consts = ks,
                                None => return self.args_only(args),
                            }
                            Vec::new()
                        }
                        Some(g) => {
                            let k = icx.kids_vec(g);
                            let shape: Vec<bool> = k
                                .iter()
                                .map(|&x| {
                                    !icx.child(x, NodeKind::TypeBase).is_some()
                                        && !icx.child(x, NodeKind::ArrOrIdx).is_some()
                                })
                                .collect();
                            self.turbofish(g, &shape, g).unwrap_or_default()
                        }
                        None => Vec::new(),
                    };
                    let opt = self.sigs.lang.option;
                    let Some((ps, ret)) = builtins::builtin_method_sig(&mut self.tys, opt, &b, c, &gargs)
                    else {
                        let tn = self.show(recv);
                        let nm = self.prog.strings.get(name).to_string();
                        let d = mk("E0208", None, &[("name", &nm), ("type", &tn)], sspan, &[]);
                        self.report(d);
                        return self.args_only(args);
                    };
                    if b.is_unsafe {
                        self.unsafe_op(sspan);
                    }
                    if !self.arity(n, paren, ps.len(), args.len(), false) {
                        return self.args_only(args);
                    }
                    // `select` on a `bool` takes its type from its operands; on a mask from the
                    // vector operands.
                    let mut ats = Vec::new();
                    let (ret, out) = if b.name == "select" && matches!(self.tys.kind(c), TypeKind::Bool) {
                        (ret, self.select_operands(args, hint))
                    } else {
                        ats = self.builtin_args(&ps, args);
                        let ret = if b.name == "select" { self.select_ty(c, &ats, args) } else { ret };
                        if let [GArg::Const(ConstArg::Value(k))] = gargs[..]
                            && b.name == "div_const"
                        {
                            consts.push(k);
                        }
                        (ret, Ty::T(ret))
                    };
                    let adjust = Adjust {
                        derefs: level,
                        autoref: if b.by_ref { Some(false) } else { None },
                        reborrow: false,
                    };
                    self.target(seg, Target::Builtin { name: b.name, recv: c, adjust, ret, consts });
                    self.gate_method(seg, b.name, c, ret, &ats);
                    out
                }
                MethodRef::Impl { method, ref impl_args, .. } => {
                    let Some(sig) = self.sigs.fn_sig(method) else { return self.args_only(args) };
                    let Some(sk) = sig.self_kind else {
                        let tn = self.show(recv);
                        let nm = self.prog.strings.get(name).to_string();
                        let d = mk("E0208", None, &[("name", &nm), ("type", &tn)], sspan, &[]);
                        self.report(d);
                        return self.args_only(args);
                    };
                    if !self.arity(n, paren, sig.params.len() - 1, args.len(), false) {
                        return self.args_only(args);
                    }
                    self.method_checks(method, sspan, seg);
                    let np = sig.n_parent as usize;
                    let mut binds: Vec<Option<GArg>> = vec![None; sig.generics.len()];
                    for (i, a) in impl_args.iter().enumerate().take(np) {
                        binds[i] = Some(*a);
                    }
                    if let Some(g) = targs_node {
                        let own: Vec<bool> = sig.generics[np..].iter().map(|g| g.is_const).collect();
                        if let Some(t) = self.turbofish(g, &own, g) {
                            for (i, a) in t.into_iter().enumerate() {
                                binds[np + i] = Some(a);
                            }
                        }
                    }
                    let params: Vec<TypeId> = sig.params[1..].to_vec();
                    let spans: Vec<Span> = sig.param_spans.get(1..).unwrap_or(&[]).to_vec();
                    self.check_args(n, &params, &spans, &mut binds, args, sig.ret, hint);
                    let gargs: Vec<GArg> = binds.into_iter().flatten().collect();
                    let own: Vec<GenericParam> = sig.generics[np..].to_vec();
                    self.check_bounds_own(sspan, method, &own, &gargs, &[]);
                    let adjust = self.adjust_for(level, sk);
                    if adjust.autoref == Some(true) && level == 0 {
                        self.static_mutation_upto(n, Some(seg - 1), sspan);
                    }
                    self.target(seg, Target::Method { method: m.clone(), args: gargs.clone(), adjust });
                    self.site(seg, method, &gargs);
                    Ty::T(self.tys.subst(sig.ret, &gargs, None))
                }
                MethodRef::Bound { method, ref trait_args, self_ty } => {
                    let Some(sig) = self.sigs.fn_sig(method) else { return self.args_only(args) };
                    let Some(sk) = sig.self_kind else {
                        let tn = self.show(recv);
                        let nm = self.prog.strings.get(name).to_string();
                        let d = mk("E0208", None, &[("name", &nm), ("type", &tn)], sspan, &[]);
                        self.report(d);
                        return self.args_only(args);
                    };
                    if !self.arity(n, paren, sig.params.len() - 1, args.len(), false) {
                        return self.args_only(args);
                    }
                    self.method_checks(method, sspan, seg);
                    let np = sig.n_parent as usize;
                    let mut binds: Vec<Option<GArg>> = vec![None; sig.generics.len()];
                    for (i, a) in trait_args.iter().enumerate().take(np) {
                        binds[i] = Some(*a);
                    }
                    if let Some(g) = targs_node {
                        let own: Vec<bool> = sig.generics[np..].iter().map(|g| g.is_const).collect();
                        if let Some(t) = self.turbofish(g, &own, g) {
                            for (i, a) in t.into_iter().enumerate() {
                                binds[np + i] = Some(a);
                            }
                        }
                    }
                    let params: Vec<TypeId> =
                        sig.params[1..].iter().map(|&p| self.tys.subst(p, &[], Some(self_ty))).collect();
                    let ret = self.tys.subst(sig.ret, &[], Some(self_ty));
                    let spans: Vec<Span> = sig.param_spans.get(1..).unwrap_or(&[]).to_vec();
                    self.check_args(n, &params, &spans, &mut binds, args, ret, hint);
                    let gargs: Vec<GArg> = binds.into_iter().flatten().collect();
                    let adjust = self.adjust_for(level, sk);
                    if adjust.autoref == Some(true) && level == 0 {
                        self.static_mutation_upto(n, Some(seg - 1), sspan);
                    }
                    self.target(seg, Target::Method { method: m.clone(), args: gargs.clone(), adjust });
                    let mut site_args = gargs.clone();
                    site_args.push(GArg::Type(self_ty));
                    self.site(seg, method, &site_args);
                    Ty::T(self.tys.subst(ret, &gargs, None))
                }
            },
            Lookup::Ambiguous { traits, methods, recv: c } => {
                self.e0908(seg, c, traits, methods);
                self.args_only(args)
            }
            Lookup::NotInBounds { param } => {
                self.e0901(seg, param);
                self.args_only(args)
            }
            Lookup::NotFound => {
                let tn = self.show(recv);
                let nm = self.prog.strings.get(name).to_string();
                let d = mk("E0208", None, &[("name", &nm), ("type", &tn)], sspan, &[]);
                self.report(d);
                self.args_only(args)
            }
        }
    }

    /// The immutable `static` whose storage the place expression `n` names: the static itself,
    /// or a field, tuple field, or element of it, through parentheses. `end` cuts the postfix
    /// chain of `n` at that token (exclusive): the receiver of a method call.
    fn static_place(&self, n: u32, end: Option<u32>) -> Option<DeclId> {
        let icx = self.icx;
        let (prim, limit) = match icx.kind(n)? {
            NodeKind::Postfix => (icx.first_child(n)?, end.unwrap_or(icx.postfix_end(n) + 1)),
            NodeKind::Primary => (n, icx.primary_end(n) + 1),
            _ => return None,
        };
        let pa = icx.a(prim);
        let (d, mut p) = if icx.is(pa, Term::LParen) {
            let close = icx.close(pa);
            let inner = icx.first_child(prim)?;
            if icx.ast.children_rev(prim).count() != 1 || icx.is(close - 1, Term::Comma) {
                return None;
            }
            (self.static_place(inner, None)?, close + 1)
        } else {
            let u = icx.uses(prim).last()?;
            let Res::Decl(d) = u.res else { return None };
            let decl = self.prog.decl(d)?;
            if decl.kind != DeclKind::Static || decl.flags.has(DeclFlags::MUT) {
                return None;
            }
            (d, u.token + 1)
        };
        while p < limit {
            if icx.is(p, Term::Dot) && (icx.is(p + 1, Term::Ident) || icx.is(p + 1, Term::IntLit)) {
                if icx.is(p + 2, Term::LParen) || icx.is(p + 2, Term::ColonColon) {
                    return None; // a method call: its result is not the static's storage
                }
                p += 2;
            } else if icx.is(p, Term::LBracket) {
                p = icx.close(p) + 1;
            } else {
                return None;
            }
        }
        Some(d)
    }

    /// E0503 case `static` at `at` when `n` names the storage of an immutable `static`.
    pub fn static_mutation(&mut self, n: u32, at: Span) {
        self.static_mutation_upto(n, None, at);
    }

    fn static_mutation_upto(&mut self, n: u32, end: Option<u32>, at: Span) {
        let Some(d) = self.static_place(n, end) else { return };
        let name = self.prog.decl(d).map_or("_", |x| self.prog.strings.get(x.name)).to_string();
        let diag = mk("E0503", Some("static"), &[("place", &name)], at, &[]);
        self.report(diag);
    }

    fn adjust_for(&self, level: u32, sk: crate::sigs::SelfKind) -> Adjust {
        use crate::sigs::SelfKind;
        match sk {
            SelfKind::Value => Adjust { derefs: level, autoref: None, reborrow: false },
            SelfKind::Ref => Adjust { derefs: level, autoref: Some(false), reborrow: false },
            SelfKind::RefMut => Adjust { derefs: level, autoref: Some(true), reborrow: level > 0 },
        }
    }

    // ---------- struct literals ----------

    fn struct_lit(&mut self, st: St, sl: u32, tok: u32, hint: Option<TypeId>, n: u32) -> Ty {
        let icx = self.icx;
        let (adt, vidx, preset): (DeclId, Option<u32>, Option<Vec<GArg>>) = match st {
            St::Adt { def, args } => (def, None, args),
            St::Variant { adt, index, args, .. } => (adt, Some(index), args),
            St::TypeVal(t) => match self.tys.kind(t).clone() {
                TypeKind::Adt { def, args } => (def, None, Some(args.to_vec())),
                _ => {
                    self.not_value(n);
                    return self.struct_fields_only(sl);
                }
            },
            St::Err => return self.struct_fields_only(sl),
            _ => {
                self.not_value(n);
                return self.struct_fields_only(sl);
            }
        };
        let Some(adef) = self.sigs.adt(adt) else { return self.struct_fields_only(sl) };
        let vi = match vidx {
            Some(i) => i as usize,
            None => {
                if adef.is_enum {
                    self.not_value(n);
                    return self.struct_fields_only(sl);
                }
                0
            }
        };
        let Some(v) = adef.variants.get(vi) else { return self.struct_fields_only(sl) };
        if v.shape != VariantShape::Record {
            self.not_value(n);
            return self.struct_fields_only(sl);
        }
        let ng = adef.generics.len();
        let mut binds: Vec<Option<GArg>> = vec![None; ng];
        if let Some(p) = preset {
            for (i, a) in p.into_iter().enumerate() {
                if let Some(b) = binds.get_mut(i) {
                    *b = Some(a);
                }
            }
        } else if let Some(h) = hint
            && let TypeKind::Adt { def, args } = self.tys.kind(h).clone()
            && def == adt
        {
            for (i, a) in args.iter().enumerate() {
                if let Some(b) = binds.get_mut(i) {
                    *b = Some(*a);
                }
            }
        }
        let exprs = icx.kids_vec(sl);
        let mut seen: Vec<usize> = Vec::new();
        let mut params = Vec::new();
        let mut spans = Vec::new();
        let mut nodes = Vec::new();
        let my_module = icx.res_module();
        for &e in &exprs {
            let name_tok = icx.a(e).saturating_sub(2);
            let name = icx.sym(name_tok);
            match v.fields.iter().position(|f| f.name == name) {
                Some(fi) => {
                    if seen.contains(&fi) {
                        let nm = self.prog.strings.get(name).to_string();
                        let scope = self.prog.decl_path(v.decl);
                        let first = exprs
                            .iter()
                            .find(|&&x| icx.sym(icx.a(x).saturating_sub(2)) == name)
                            .map_or(icx.tok_span(name_tok), |&x| icx.tok_span(icx.a(x).saturating_sub(2)));
                        let d = mk(
                            "E0105",
                            None,
                            &[("name", &nm), ("scope", &scope)],
                            icx.tok_span(name_tok),
                            &[("first", first)],
                        );
                        self.report(d);
                        self.args_only(&[e]);
                        continue;
                    }
                    seen.push(fi);
                    let f = &v.fields[fi];
                    if let Some(fd) = self.prog.decl(f.decl)
                        && !fd.is_pub()
                        && self.prog.decl(fd.parent).is_some_and(|p| p.kind == DeclKind::Struct)
                        && fd.module != my_module
                    {
                        let module =
                            self.prog.module(fd.module).map_or(String::new(), |m| m.path_text.clone());
                        let nm = self.prog.strings.get(name).to_string();
                        let d = mk(
                            "E0106",
                            None,
                            &[("name", &nm), ("module", &module)],
                            icx.tok_span(name_tok),
                            &[("decl", fd.name_span)],
                        );
                        self.report(d);
                    }
                    params.push(f.ty);
                    spans.push(f.span);
                    nodes.push(e);
                }
                None => {
                    let tn =
                        self.prog.strings.get(self.prog.decl(adt).map_or(Sym::NONE, |d| d.name)).to_string();
                    let nm = self.prog.strings.get(name);
                    let nm = if nm.is_empty() { "_" } else { nm };
                    let d = mk("E0208", None, &[("name", nm), ("type", &tn)], icx.tok_span(name_tok), &[]);
                    self.report(d);
                    self.args_only(&[e]);
                }
            }
        }
        let ret = self.adt_self_type(adt, ng);
        self.check_args(n, &params, &spans, &mut binds, &nodes, ret, hint);
        let gargs: Vec<GArg> = binds.into_iter().flatten().collect();
        let t = self.tys.mk_adt(adt, gargs.clone());
        for (fi, f) in v.fields.iter().enumerate() {
            if !seen.contains(&fi) {
                let nm = self.prog.strings.get(f.name).to_string();
                let tn = self.show(t);
                let d = mk(
                    "E0208",
                    Some("missing-field"),
                    &[("name", &nm), ("type", &tn)],
                    icx.tok_span(tok),
                    &[],
                );
                self.report(d);
            }
        }
        let gens = adef.generics.clone();
        self.check_bounds(icx.tok_span(tok), adt, &gens, &gargs);
        match vidx {
            Some(i) => self.target(tok, Target::Variant { decl: v.decl, index: i }),
            None => self.target(tok, Target::Struct { decl: adt }),
        }
        self.owned(tok, t);
        Ty::T(t)
    }

    fn struct_fields_only(&mut self, sl: u32) -> Ty {
        let exprs = self.icx.kids_vec(sl);
        self.args_only(&exprs)
    }

    // ---------- match and for ----------

    fn match_expr(&mut self, n: u32, hint: Option<TypeId>) -> Ty {
        let icx = self.icx;
        let kids = self.kids(n);
        let Some(&scrut) = kids.first() else {
            self.free_kids(kids);
            return Ty::ERR;
        };
        let st = self.expr(scrut, None);
        let st = self.force(st);
        let before = self.diverges;
        let mut all_div = true;
        let mut result: Option<Ty> = None;
        let mut rows = Vec::new();
        let mut arm_ts: Vec<(u32, Ty)> = Vec::new();
        for (k, &arm) in kids[1..].iter().enumerate() {
            // [ArmPattern, guard?, value]
            let ak = icx.kids_vec(arm);
            let Some(&ap) = ak.first() else { continue };
            let guard = if ak.len() >= 3 { Some(ak[1]) } else { None };
            crate::patterns::check_arm(self, ap, st, guard.is_some(), k as u32, &mut rows);
            if let Some(g) = guard {
                self.check_expr(g, ids::BOOL, None);
            }
            let Some(&e) = ak.last() else { continue };
            self.diverges = false;
            let h = match result.map(|r| self.resolve(r)) {
                Some(Ty::T(t)) if t != ids::NEVER && t != ids::ERROR && hint.is_none() => Some(t),
                _ => hint,
            };
            let t = self.expr(e, h);
            all_div &= self.diverges;
            arm_ts.push((e, t));
            result = Some(match result {
                None => t,
                Some(r) => self.arms_unify(e, r, t, hint),
            });
        }
        self.free_kids(kids);
        self.diverges = before || (all_div && !arm_ts.is_empty());
        crate::patterns::check_match(self, scrut, st, &rows);
        result.unwrap_or(Ty::T(ids::NEVER))
    }

    /// A `match` statement: whether it diverges (exhaustive and every arm diverges).
    pub fn match_stmt(&mut self, _s: u32, kids: &[u32]) -> bool {
        let icx = self.icx;
        let Some(&scrut) = kids.first() else { return false };
        self.diverges = false;
        let st = self.expr(scrut, None);
        let st = self.force(st);
        let scrut_div = std::mem::take(&mut self.diverges);
        let mut rows = Vec::new();
        let mut all_div = true;
        let reach = self.reachable;
        for (k, &arm) in kids[1..].iter().enumerate() {
            // [ArmPattern, guard?, block]
            let ak = icx.kids_vec(arm);
            let Some(&ap) = ak.first() else { continue };
            let guard = if ak.len() >= 3 { Some(ak[1]) } else { None };
            crate::patterns::check_arm(self, ap, st, guard.is_some(), k as u32, &mut rows);
            if let Some(g) = guard {
                self.check_expr(g, ids::BOOL, None);
            }
            if let Some(&b) = ak.last() {
                let d = self.block(b);
                self.reachable = reach;
                all_div &= d;
            }
        }
        let exhaustive = crate::patterns::check_match(self, scrut, st, &rows);
        scrut_div || (exhaustive && all_div && !rows.is_empty())
    }

    /// A `for` loop (element or SIMD chunk loop).
    pub fn for_stmt(&mut self, f: u32) {
        let icx = self.icx;
        let kids = icx.kids_vec(f);
        let Some(&pat) = kids.first() else { return };
        let bind_ty = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Type));
        let Some(iter) = kids.iter().copied().find(|&c| icx.is_expr(c)) else { return };
        let body = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Block));
        let tail = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::TailClause));
        // `for i: T in a..b`: the binding type is the context of the range's literals
        let bt_pre = bind_ty.map(|b| self.lower_ty(b, false));
        let it_hint = match bt_pre {
            Some(b) if !matches!(self.tys.kind(b), TypeKind::Vector { .. } | TypeKind::Ref { .. }) => {
                Some(self.tys.intern(TypeKind::Range { elem: b, inclusive: false }))
            }
            _ => None,
        };
        let it = self.expr(iter, it_hint);
        let it = self.force(it);
        let (base_elem, arr_len) = {
            let (inner, _) = self.tys.peel_refs(it);
            match self.tys.kind(inner).clone() {
                TypeKind::Array { elem, len } => (Some(elem), len.value()),
                TypeKind::Slice { elem, .. } => (Some(elem), None),
                _ => (None, None),
            }
        };
        let elem_t = if let Some(b) = bt_pre {
            // chunk loop: the binding type is a vector (or `&mut` vector)
            let vec_t = match self.tys.kind(b).clone() {
                TypeKind::Ref { inner, .. } => inner,
                _ => b,
            };
            if let TypeKind::Vector { lane, n } = self.tys.kind(vec_t).clone() {
                if base_elem.is_some_and(|e| e != lane) && it != ids::ERROR {
                    let s = self.tys.intern(TypeKind::Slice { elem: lane, align: 0 });
                    let want = self.tys.mk_ref(false, s);
                    self.mismatch(iter, Ty::T(it), want, None);
                }
                let multiple = match (arr_len, n) {
                    (Some(len), Lanes::Fixed(k)) => len % (k as u64) == 0,
                    (_, Lanes::Native) => true,
                    _ => false,
                };
                if tail.is_none() && !multiple && it != ids::ERROR {
                    let tn = self.show(it);
                    let d = mk("E0601", None, &[("type", &tn)], icx.node_span(iter), &[]);
                    self.report(d);
                }
                self.gate(icx.a(f), "loop.chunk", vec_t, "");
                if let Some(tc) = tail {
                    self.tail_clause(tc, b, lane, n);
                    let op = if icx.is(icx.a(tc) + 1, Term::KwMasked) {
                        if matches!(self.tys.kind(b), TypeKind::Ref { mutable: true, .. }) {
                            "loop.tail_masked_mut"
                        } else {
                            "loop.tail_masked"
                        }
                    } else if icx.is(icx.a(tc) + 1, Term::KwScalar) {
                        "loop.tail_scalar"
                    } else {
                        "loop.tail_exact"
                    };
                    self.gate(icx.a(tc), op, vec_t, "");
                }
            } else if b != ids::ERROR {
                // a plain annotated element binding: it must be the element type
                let et = self.element_type(icx.a(f), iter, it);
                if et != b && et != ids::ERROR && !self.tys.has_error(et) {
                    let want = self.tys_range_of(b);
                    self.mismatch(iter, Ty::T(it), want, None);
                }
                if let Some(tc) = tail {
                    self.tail_clause(tc, b, ids::ERROR, Lanes::Fixed(1));
                }
            }
            b
        } else {
            if let Some(tc) = tail {
                self.tail_clause(tc, ids::ERROR, ids::ERROR, Lanes::Fixed(1));
            }
            self.element_type(icx.a(f), iter, it)
        };
        let pp = self.pattern(pat, elem_t, crate::patterns::Mode::Move);
        crate::patterns::irrefutable(self, pat, &pp, elem_t);
        if let Some(b) = body {
            if let Some(r) = self.rets.last_mut() {
                r.loops.push(crate::typeck::LoopCx { has_break: false });
            }
            let reach = self.reachable;
            self.block(b);
            self.reachable = reach;
            if let Some(r) = self.rets.last_mut() {
                r.loops.pop();
            }
        }
    }

    fn tys_range_of(&mut self, b: TypeId) -> TypeId {
        self.tys.intern(TypeKind::Range { elem: b, inclusive: false })
    }

    /// The element type of `for .. in iter` (`it`: the type of `iter`); an iterator's `next` is
    /// recorded as an implicit call at `for_tok`.
    fn element_type(&mut self, for_tok: u32, iter: u32, it: TypeId) -> TypeId {
        if it == ids::ERROR || self.tys.has_error(it) {
            return ids::ERROR;
        }
        match self.tys.kind(it).clone() {
            TypeKind::Range { elem, .. } => return elem,
            TypeKind::Array { elem, .. } => return elem,
            TypeKind::Ref { mutable, inner } => match self.tys.kind(inner).clone() {
                TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } => {
                    return self.tys.mk_ref(mutable, elem);
                }
                _ => {}
            },
            _ => {}
        }
        // an iterator: `fn next(&mut self) -> Option<X>`
        if let Some(next) = self.prog.strings.lookup("next") {
            let genv = self.genv();
            if let Lookup::Found { m, .. } = self.w.lookup_method(&mut self.tys, it, next, genv)
                && let Some((sig, args, self_ty)) = self.w.method_sig(&m)
            {
                let ret = self.tys.subst(sig.ret, &args, self_ty);
                // lowering resolves `next` with the impl's args (none of its own)
                let call_args = match &m {
                    MethodRef::Impl { impl_args, .. } => impl_args.clone(),
                    _ => Vec::new(),
                };
                self.implicit.push((
                    for_tok,
                    Target::Method { method: m, args: call_args, adjust: Adjust::default() },
                ));
                if let TypeKind::Adt { def, args } = self.tys.kind(ret).clone()
                    && def == self.sigs.lang.option
                    && let Some(GArg::Type(x)) = args.first()
                {
                    return *x;
                }
            }
        }
        let tn = self.show(it);
        let d = mk("E0208", None, &[("name", "next"), ("type", &tn)], self.icx.node_span(iter), &[]);
        self.report(d);
        ids::ERROR
    }

    fn tail_clause(&mut self, tc: u32, bind: TypeId, lane: TypeId, n: Lanes) {
        let icx = self.icx;
        let a = icx.a(tc);
        let kids = icx.kids_vec(tc);
        let binds: Vec<u32> =
            kids.iter().copied().filter(|&c| icx.kind(c) == Some(NodeKind::TailBind)).collect();
        if icx.is(a + 1, Term::KwMasked) {
            if let Some(&v) = binds.first() {
                self.locals.insert(v, bind);
            }
            if let Some(&m) = binds.get(1) {
                let mt = if lane == ids::ERROR {
                    ids::ERROR
                } else {
                    let w = types::lane_bits(self.tys.kind(lane)) as u8;
                    self.tys.intern(TypeKind::Mask { width: w, n })
                };
                self.locals.insert(m, mt);
            }
        } else if icx.is(a + 1, Term::KwScalar)
            && let Some(&x) = binds.first()
        {
            self.locals.insert(x, lane);
        }
        if let Some(b) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Block)) {
            let reach = self.reachable;
            self.block(b);
            self.reachable = reach;
        }
    }
}

impl crate::itemcx::ItemCx<'_> {
    /// The module of this item.
    pub fn res_module(&self) -> crate::ids::ModuleId {
        self.prog.item(self.item).map_or(crate::ids::ModuleId::NONE, |i| i.module)
    }
}

trait ParamPlaceholder {
    fn mk_param_placeholder(&mut self, i: usize) -> TypeId;
}

impl<I: Interner> ParamPlaceholder for I {
    fn mk_param_placeholder(&mut self, i: usize) -> TypeId {
        self.mk_param(i as u32)
    }
}
