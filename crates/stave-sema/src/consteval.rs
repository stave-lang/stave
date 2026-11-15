//! Const-expr evaluation: literals, consts, arithmetic, comparison, the `as`
//! family, `size_of::<T>()` / `align_of::<T>()`, `if then else`, and aggregate literals (array
//! `[a, b]` / `[e; n]`, tuple, struct, variant); also indexing and field access of const
//! aggregates (an extension LANG.md does not list), and a call of a constructor fixed by path
//! ([`LangItems::const_ctors`], contracts/intrinsics.md "Paths fixed by the compiler"), which is
//! evaluated as the struct literal its body returns, with the parameters bound to the
//! arguments. Anything else (calls, loops, non-const items such as fns,
//! statics, locals, references, f-strings, ranges, `match`) is E0904. A const that depends on
//! itself is E0904 case `cycle` (reported by the signature builder, which evaluates consts on
//! demand). Evaluation is linear in the expression size: every node is visited once.
//!
//! Type errors inside a const initializer are reported by the body checker (which also types
//! const and `static` initializers for the `types` dump), so evaluation fails silently on them.
//! Arithmetic overflow and division by zero in a const are errors (const contexts keep
//! their errors): E0203 at the operation (no dedicated code). A shift amount at or above
//! the bit width is E0205, reported by the body checker.

use stave_syntax::{NodeKind, Term};

use crate::decls::DeclKind;
use crate::diag::mk;
use crate::ids::DeclId;
use crate::itemcx::{Brackets, ItemCx};
use crate::literals::{self, Suffix};
use crate::prim::Builtin;
use crate::resolve::Res;
use crate::sigs::{
    AdtDef, LangItems, LowerHost, NameCx, SigBuilder, TyEnv, VariantShape, lower_garg, lower_type,
};
use crate::types::{self, ConstArg, FloatBits, GArg, IntBits, Interner, TypeId, TypeKind, ids};

/// A const value (its type is kept alongside).
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Int(i128),
    Float(f64),
    Bool(bool),
    Char(u32),
    Str(Box<[u8]>),
    /// Tuples, arrays, structs (fields in declaration order); unit is `Agg([])`.
    Agg(Vec<Val>),
    /// An enum value: variant index, fields.
    Variant(u32, Vec<Val>),
}

/// What a const evaluation needs from its context.
pub(crate) trait CEnv: LowerHost {
    fn const_value(&mut self, d: DeclId) -> Option<(TypeId, Val)>;
    fn adt_def(&mut self, d: DeclId) -> Option<AdtDef>;
    fn layout(&mut self, t: TypeId) -> Option<(u64, u64)>;
    fn lang(&self) -> &LangItems;
    /// The parameter types and return type of fn `d`.
    fn fn_types(&mut self, d: DeclId) -> Option<(Vec<TypeId>, TypeId)>;
}

impl CEnv for SigBuilder<'_> {
    fn const_value(&mut self, d: DeclId) -> Option<(TypeId, Val)> {
        SigBuilder::const_value(self, d)
    }

    fn adt_def(&mut self, d: DeclId) -> Option<AdtDef> {
        SigBuilder::adt_def(self, d)
    }

    fn layout(&mut self, t: TypeId) -> Option<(u64, u64)> {
        SigBuilder::layout(self, t)
    }

    fn lang(&self) -> &LangItems {
        &self.lang
    }

    fn fn_types(&mut self, d: DeclId) -> Option<(Vec<TypeId>, TypeId)> {
        SigBuilder::fn_types(self, d)
    }
}

/// A partial result: the type (`NONE` = an unsuffixed literal still pending) and the value.
type R = Option<(TypeId, Val)>;

struct Ev<'e, 'a, E: CEnv> {
    env: &'e mut E,
    te: &'e TyEnv<'a>,
    depth: u32,
    /// Inside a constructor body: its parameters (`Param` node, value).
    params: Vec<(u32, (TypeId, Val))>,
}

const MAX_DEPTH: u32 = 2000;

/// Evaluates the initializer `e` of a const or `static` of type `ty`.
pub(crate) fn eval_item_const<E: CEnv>(env: &mut E, te: &TyEnv<'_>, e: u32, ty: TypeId) -> Option<Val> {
    let mut ev = Ev { env, te, depth: 0, params: Vec::new() };
    let (t, v) = ev.eval(e, Some(ty))?;
    ev.fix(e, t, v, ty)
}

/// Evaluates an array length / const generic argument to a `u64` (or a const generic param).
pub(crate) fn eval_u64<E: CEnv>(env: &mut E, te: &TyEnv<'_>, n: u32, ty: TypeId) -> ConstArg {
    // A bare const generic parameter.
    if let Some(p) = const_param_path(te, n) {
        return ConstArg::Param(p);
    }
    let mut ev = Ev { env, te, depth: 0, params: Vec::new() };
    let Some((t, v)) = ev.eval(n, Some(ty)) else { return ConstArg::Error };
    match ev.fix(n, t, v, ty) {
        Some(Val::Int(i)) if i >= 0 && i <= u64::MAX as i128 => ConstArg::Value(i as u64),
        _ => ConstArg::Error,
    }
}

/// If `n` is (wrappers around) a path naming a const generic param: its flat index.
fn const_param_path(te: &TyEnv<'_>, n: u32) -> Option<u32> {
    let icx = te.icx;
    let mut n = n;
    for _ in 0..64 {
        let kids = icx.kids_vec(n);
        match icx.kind(n)? {
            NodeKind::Primary => {
                let u = icx.uses(n).first()?;
                return match u.res {
                    Res::Generic(g) if g.is_const && icx.primary_end(n) == icx.a(n) => {
                        Some(if g.level == 0 { g.index as u32 } else { te.n_parent + g.index as u32 })
                    }
                    _ => None,
                };
            }
            NodeKind::Postfix => {
                if kids.len() != 1 || icx.postfix_end(n) != icx.primary_end(kids[0]) {
                    return None;
                }
                n = kids[0];
            }
            NodeKind::Unary => {
                if kids.len() != 1 || icx.a(kids[0]) != icx.a(n) {
                    return None;
                }
                n = kids[0];
            }
            _ => {
                if kids.len() != 1 {
                    return None;
                }
                n = kids[0];
            }
        }
    }
    None
}

fn wrap_int(v: i128, signed: bool, bits: u32) -> i128 {
    let m = (v as u128) & (if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 });
    if signed && bits < 128 && (m >> (bits - 1)) & 1 == 1 { (m as i128) - (1i128 << bits) } else { m as i128 }
}

impl<'e, 'a, E: CEnv> Ev<'e, 'a, E> {
    /// Well-known type `f` ([`ids`]) in the environment's table (it may be canonical).
    fn fx(&self, f: TypeId) -> TypeId {
        self.env.fixed_ty(f)
    }

    fn int_range(&self, t: TypeId) -> Option<(i128, i128)> {
        literals::int_range_kind(self.env.kind(t))
    }

    fn e0904(&mut self, n: u32, what: &str) -> R {
        let icx = self.te.icx;
        let d = mk("E0904", None, &[("what", what)], icx.node_span(n), &[]);
        self.env.report(d);
        None
    }

    fn overflow(&mut self, n: u32, t: TypeId) -> R {
        let icx = self.te.icx;
        let text = icx.node_text(n).to_string();
        let tn = types::display(self.env, &NameCx { prog: self.env.prog(), generics: &[] }, t);
        let (lo, hi) = self.int_range(t).unwrap_or((0, 0));
        let d = mk(
            "E0203",
            None,
            &[("literal", &text), ("type", &tn), ("min", &lo.to_string()), ("max", &hi.to_string())],
            icx.node_span(n),
            &[],
        );
        self.env.report(d);
        None
    }

    /// Gives a pending literal its type (range-checked); checks a typed value against `want`.
    fn fix(&mut self, n: u32, t: TypeId, v: Val, want: TypeId) -> Option<Val> {
        if t.is_none() {
            return self.fix_lit(n, v, want);
        }
        if t == want || t == self.fx(ids::NEVER) { Some(v) } else { None }
    }

    fn fix_lit(&mut self, n: u32, v: Val, want: TypeId) -> Option<Val> {
        match v {
            Val::Int(i) => {
                if let Some((lo, hi)) = self.int_range(want) {
                    if i < lo || i > hi {
                        return self.overflow(n, want).map(|x| x.1);
                    }
                    Some(Val::Int(i))
                } else {
                    None
                }
            }
            Val::Float(f) => {
                if self.env.is_float(want) {
                    Some(Val::Float(round_float(f, self.env.kind(want))))
                } else {
                    None
                }
            }
            other => Some(other),
        }
    }

    fn eval(&mut self, n: u32, want: Option<TypeId>) -> R {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            return None;
        }
        let r = self.eval_inner(n, want);
        self.depth -= 1;
        r
    }

    fn eval_inner(&mut self, n: u32, want: Option<TypeId>) -> R {
        let icx = self.te.icx;
        let kids = icx.kids_vec(n);
        let k = icx.kind(n)?;
        match k {
            NodeKind::Expr => self.eval(*kids.first()?, want),
            NodeKind::IfExpr => {
                let (ct, cv) = self.eval(kids[0], Some(self.fx(ids::BOOL)))?;
                if ct != self.fx(ids::BOOL) {
                    return None;
                }
                let pick = if cv == Val::Bool(true) { kids.get(1)? } else { kids.get(2)? };
                // both arms must be const-exprs
                let other = if cv == Val::Bool(true) { kids.get(2)? } else { kids.get(1)? };
                let r = self.eval(*pick, want)?;
                let want2 = if r.0.is_none() { want } else { Some(r.0) };
                let _ = self.eval(*other, want2)?;
                Some(r)
            }
            NodeKind::MatchExpr => self.e0904(n, "a non-const item"),
            NodeKind::RangeExpr => {
                if kids.len() > 1 {
                    return self.e0904(n, "a non-const item");
                }
                self.eval(kids[0], want)
            }
            NodeKind::OrExpr | NodeKind::AndExpr => {
                if kids.len() == 1 {
                    return self.eval(kids[0], want);
                }
                let is_or = k == NodeKind::OrExpr;
                let mut acc = !is_or;
                for &c in &kids {
                    let (t, v) = self.eval(c, Some(self.fx(ids::BOOL)))?;
                    if t != self.fx(ids::BOOL) {
                        return None;
                    }
                    let b = v == Val::Bool(true);
                    acc = if is_or { acc || b } else { acc && b };
                }
                Some((self.fx(ids::BOOL), Val::Bool(acc)))
            }
            NodeKind::CmpExpr => {
                if kids.len() == 1 {
                    return self.eval(kids[0], want);
                }
                let op_node = *kids.get(1)?;
                let op = icx.tk(icx.a(op_node));
                let (lt, lv, rt, rv) = self.pair(kids[0], *kids.get(2)?, None)?;
                let _ = (lt, rt);
                let ord = match (&lv, &rv) {
                    (Val::Int(a), Val::Int(b)) => a.partial_cmp(b),
                    (Val::Float(a), Val::Float(b)) => a.partial_cmp(b),
                    (Val::Bool(a), Val::Bool(b)) => a.partial_cmp(b),
                    (Val::Char(a), Val::Char(b)) => a.partial_cmp(b),
                    _ => return None,
                };
                use std::cmp::Ordering::*;
                let b = match ord {
                    None => op == Term::BangEq as u8,
                    Some(o) => {
                        if op == Term::EqEq as u8 {
                            o == Equal
                        } else if op == Term::BangEq as u8 {
                            o != Equal
                        } else if op == Term::Lt as u8 {
                            o == Less
                        } else if op == Term::LtEq as u8 {
                            o != Greater
                        } else if op == Term::Gt as u8 {
                            o == Greater
                        } else {
                            o != Less
                        }
                    }
                };
                Some((self.fx(ids::BOOL), Val::Bool(b)))
            }
            NodeKind::BitOr
            | NodeKind::BitXor
            | NodeKind::BitAnd
            | NodeKind::Shift
            | NodeKind::Add
            | NodeKind::Mul => {
                if kids.len() == 1 {
                    return self.eval(kids[0], want);
                }
                let mut acc = self.eval(kids[0], want)?;
                for &c in &kids[1..] {
                    let op = icx.tk(icx.a(c).saturating_sub(1));
                    let is_shift = k == NodeKind::Shift;
                    let w = if acc.0.is_none() { want } else { Some(acc.0) };
                    let r = self.eval(c, w)?;
                    // unify the operand types (pending literals take the other's)
                    let (t, a, b) = if is_shift {
                        let t = if acc.0.is_none() { r.0 } else { acc.0 };
                        (t, acc.1.clone(), r.1)
                    } else {
                        self.unify(acc, r)?
                    };
                    let t = if t.is_none() { want.unwrap_or(TypeId::NONE) } else { t };
                    let v = self.arith(n, op, t, a, b)?;
                    acc = (t, v);
                }
                Some(acc)
            }
            NodeKind::Cast => {
                if kids.len() == 1 {
                    return self.eval(kids[0], want);
                }
                let mut acc = self.eval(kids[0], None)?;
                for &c in &kids[1..] {
                    let op = icx.tk(icx.a(c).saturating_sub(1));
                    let to = lower_type(self.env, self.te, c, false);
                    acc = self.cast(op, acc, to)?;
                }
                Some(acc)
            }
            NodeKind::Unary => {
                let a = icx.a(n);
                let t = icx.tk(a);
                let &c = kids.first()?;
                if icx.kind(c) == Some(NodeKind::Postfix) && icx.a(c) == a {
                    return self.eval(c, want);
                }
                if t == Term::Minus as u8 {
                    let (ct, cv) = self.eval(c, want)?;
                    return match cv {
                        Val::Int(i) => {
                            if !ct.is_none() {
                                match self.env.int(ct) {
                                    Some((true, _)) => {}
                                    _ => return None,
                                }
                                if let Some((lo, _)) = self.int_range(ct)
                                    && i == lo
                                {
                                    return self.overflow(n, ct);
                                }
                            }
                            Some((ct, Val::Int(-i)))
                        }
                        Val::Float(f) => Some((ct, Val::Float(-f))),
                        _ => None,
                    };
                }
                if t == Term::Bang as u8 {
                    let (ct, cv) = self.eval(c, Some(self.fx(ids::BOOL)))?;
                    return match cv {
                        Val::Bool(b) if ct == self.fx(ids::BOOL) => Some((ct, Val::Bool(!b))),
                        _ => None,
                    };
                }
                if t == Term::Tilde as u8 {
                    let (ct, cv) = self.eval(c, want)?;
                    return match cv {
                        Val::Int(i) => {
                            let tt = if ct.is_none() { want? } else { ct };
                            let (signed, bits) = self.env.int(tt)?;
                            Some((tt, Val::Int(wrap_int(!i, signed, bits.bits()))))
                        }
                        _ => None,
                    };
                }
                self.e0904(n, "a non-const item")
            }
            NodeKind::Postfix => self.postfix(n, &kids, want),
            NodeKind::Primary => self.primary(n, &kids, want),
            _ => None,
        }
    }

    /// Evaluates two operands of one type.
    fn pair(&mut self, l: u32, r: u32, want: Option<TypeId>) -> Option<(TypeId, Val, TypeId, Val)> {
        let a = self.eval(l, want)?;
        let w = if a.0.is_none() { want } else { Some(a.0) };
        let b = self.eval(r, w)?;
        let (t, av, bv) = self.unify(a, b)?;
        Some((t, av, t, bv))
    }

    fn unify(&mut self, a: (TypeId, Val), b: (TypeId, Val)) -> Option<(TypeId, Val, Val)> {
        match (a.0.is_none(), b.0.is_none()) {
            (false, false) => {
                if a.0 != b.0 {
                    return None;
                }
                Some((a.0, a.1, b.1))
            }
            (true, false) => Some((b.0, round_lit(a.1, b.0, self.env), b.1)),
            (false, true) => Some((a.0, a.1, round_lit(b.1, a.0, self.env))),
            (true, true) => Some((TypeId::NONE, a.1, b.1)),
        }
    }

    fn arith(&mut self, at: u32, op: u8, t: TypeId, a: Val, b: Val) -> Option<Val> {
        let n = at;
        match (a, b) {
            (Val::Int(x), Val::Int(y)) => {
                let (signed, bits) = if t.is_none() { (true, IntBits::B64) } else { self.env.int(t)? };
                let bw = bits.bits();
                let range = self.int_range(t);
                let check = |v: Option<i128>| -> Option<i128> {
                    let v = v?;
                    match range {
                        Some((lo, hi)) if v < lo || v > hi => None,
                        _ => Some(v),
                    }
                };
                let sat = |v: Option<i128>, neg_hint: bool| -> i128 {
                    let (lo, hi) = range.unwrap_or((i128::MIN, i128::MAX));
                    match v {
                        Some(v) => v.clamp(lo, hi),
                        None => {
                            if neg_hint {
                                lo
                            } else {
                                hi
                            }
                        }
                    }
                };
                let p = |o: Term| op == o as u8;
                let r = if p(Term::Plus) {
                    check(x.checked_add(y))
                } else if p(Term::Minus) {
                    check(x.checked_sub(y))
                } else if p(Term::Star) {
                    check(x.checked_mul(y))
                } else if p(Term::Slash) || p(Term::Percent) {
                    if y == 0 {
                        return self.overflow(n, t).map(|v| v.1);
                    }
                    let q = if p(Term::Slash) { x.checked_div(y) } else { x.checked_rem(y) };
                    check(q)
                } else if p(Term::PlusPercent) {
                    Some(wrap_int(x.wrapping_add(y), signed, bw))
                } else if p(Term::MinusPercent) {
                    Some(wrap_int(x.wrapping_sub(y), signed, bw))
                } else if p(Term::StarPercent) {
                    Some(wrap_int(x.wrapping_mul(y), signed, bw))
                } else if p(Term::PlusPipe) {
                    Some(sat(x.checked_add(y), y < 0))
                } else if p(Term::MinusPipe) {
                    Some(sat(x.checked_sub(y), y > 0))
                } else if p(Term::StarPipe) {
                    Some(sat(x.checked_mul(y), (x < 0) != (y < 0)))
                } else if p(Term::Amp) {
                    Some(wrap_int(x & y, signed, bw))
                } else if p(Term::Pipe) {
                    Some(wrap_int(x | y, signed, bw))
                } else if p(Term::Caret) {
                    Some(wrap_int(x ^ y, signed, bw))
                } else if p(Term::LtLt) || p(Term::LtLtPercent) || p(Term::GtGt) {
                    let amt = if p(Term::LtLtPercent) { y & (bw as i128 - 1) } else { y };
                    if amt < 0 || amt >= bw as i128 {
                        return None;
                    }
                    if p(Term::GtGt) {
                        let u = if signed { x } else { (x as u128 & ((1u128 << bw) - 1)) as i128 };
                        Some(wrap_int(u >> amt, signed, bw))
                    } else {
                        Some(wrap_int(x.wrapping_shl(amt as u32), signed, bw))
                    }
                } else {
                    return None;
                };
                match r {
                    Some(v) => Some(Val::Int(v)),
                    None => self.overflow(n, t).map(|v| v.1),
                }
            }
            (Val::Float(x), Val::Float(y)) => {
                let f32_ = t == self.fx(ids::F32);
                let v = if op == Term::Plus as u8 {
                    if f32_ { ((x as f32) + (y as f32)) as f64 } else { x + y }
                } else if op == Term::Minus as u8 {
                    if f32_ { ((x as f32) - (y as f32)) as f64 } else { x - y }
                } else if op == Term::Star as u8 {
                    if f32_ { ((x as f32) * (y as f32)) as f64 } else { x * y }
                } else if op == Term::Slash as u8 {
                    if f32_ { ((x as f32) / (y as f32)) as f64 } else { x / y }
                } else if op == Term::Percent as u8 {
                    if f32_ { ((x as f32) % (y as f32)) as f64 } else { x % y }
                } else {
                    return None;
                };
                Some(Val::Float(v))
            }
            (Val::Bool(x), Val::Bool(y)) => {
                if op == Term::Amp as u8 {
                    Some(Val::Bool(x & y))
                } else if op == Term::Pipe as u8 {
                    Some(Val::Bool(x | y))
                } else if op == Term::Caret as u8 {
                    Some(Val::Bool(x ^ y))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn cast(&mut self, op: u8, (t, v): (TypeId, Val), to: TypeId) -> R {
        if to == self.env.error_ty() {
            return None;
        }
        // A pending literal has no type: `5 as u8` is invalid (no context), left to the checker.
        if t.is_none() {
            return None;
        }
        let tk = self.env.kind(to).clone();
        let from_int = self.env.int(t);
        let wrapping = op == Term::AsPercent as u8;
        let saturating = op == Term::AsPipe as u8;
        let checked = op == Term::AsQuestion as u8;
        let rounding = op == Term::AsTilde as u8;
        let int_of = |v: &Val| -> Option<i128> {
            match v {
                Val::Int(i) => Some(*i),
                Val::Bool(b) => Some(*b as i128),
                Val::Char(c) => Some(*c as i128),
                _ => None,
            }
        };
        let _ = from_int;
        match tk {
            TypeKind::Int { signed, bits } => {
                let (lo, hi) = self.int_range(to)?;
                let res: Option<i128> = match &v {
                    Val::Float(f) => {
                        if f.is_nan() {
                            if saturating { Some(0) } else { None }
                        } else if saturating {
                            Some((f.trunc() as i128).clamp(lo, hi))
                        } else if checked {
                            if f.fract() == 0.0 && *f >= lo as f64 && *f <= hi as f64 {
                                Some(*f as i128)
                            } else {
                                return self.none_of(to);
                            }
                        } else {
                            None
                        }
                    }
                    other => {
                        let i = int_of(other)?;
                        if wrapping {
                            Some(wrap_int(i, signed, bits.bits()))
                        } else if saturating {
                            Some(i.clamp(lo, hi))
                        } else if checked {
                            if i < lo || i > hi {
                                return self.none_of(to);
                            }
                            Some(i)
                        } else {
                            Some(i)
                        }
                    }
                };
                let r = res?;
                if checked { self.some_of(to, Val::Int(r)) } else { Some((to, Val::Int(r))) }
            }
            TypeKind::Float(fb) => {
                let f = match &v {
                    Val::Float(f) => *f,
                    other => int_of(other)? as f64,
                };
                let r = match fb {
                    FloatBits::F32 => (f as f32) as f64,
                    _ => f,
                };
                let _ = rounding;
                if checked {
                    if r != f && !f.is_nan() {
                        return self.none_of(to);
                    }
                    return self.some_of(to, Val::Float(r));
                }
                Some((to, Val::Float(r)))
            }
            TypeKind::Char => {
                let i = int_of(&v)?;
                let ok = char::from_u32(u32::try_from(i).ok()?).is_some();
                if checked {
                    if !ok {
                        return self.none_of(to);
                    }
                    return self.some_of(to, Val::Char(i as u32));
                }
                if !ok {
                    return None;
                }
                Some((to, Val::Char(i as u32)))
            }
            TypeKind::Bool => match v {
                Val::Bool(b) => Some((to, Val::Bool(b))),
                _ => None,
            },
            _ => None,
        }
    }

    fn option_of(&mut self, inner: TypeId) -> Option<TypeId> {
        let opt = self.env.lang().option;
        if opt.is_none() {
            return None;
        }
        Some(self.env.mk_adt(opt, vec![GArg::Type(inner)]))
    }

    fn none_of(&mut self, inner: TypeId) -> R {
        Some((self.option_of(inner)?, Val::Variant(0, Vec::new())))
    }

    fn some_of(&mut self, inner: TypeId, v: Val) -> R {
        Some((self.option_of(inner)?, Val::Variant(1, vec![v])))
    }

    fn primary(&mut self, n: u32, kids: &[u32], want: Option<TypeId>) -> R {
        let icx = self.te.icx;
        let a = icx.a(n);
        let t = icx.tk(a);
        if t == Term::IntLit as u8 {
            let (v, suf) = literals::parse_int(icx.tok_str(a))?;
            if v > i128::MAX as u128 {
                return match suf {
                    Suffix::Int(t) => {
                        let t = self.fx(t);
                        self.overflow(n, t)
                    }
                    _ => None,
                };
            }
            return match suf {
                Suffix::None => match want {
                    Some(w) if self.env.int(w).is_some() => Some((TypeId::NONE, Val::Int(v as i128))),
                    _ => Some((TypeId::NONE, Val::Int(v as i128))),
                },
                Suffix::Int(st) => Some((self.fx(st), Val::Int(v as i128))),
                Suffix::Float(st) => Some((self.fx(st), Val::Float(v as f64))),
            };
        }
        if t == Term::FloatLit as u8 {
            let (v, suf) = literals::parse_float(icx.tok_str(a))?;
            return match suf {
                Suffix::Float(st) => Some((self.fx(st), Val::Float(v))),
                _ => Some((TypeId::NONE, Val::Float(v))),
            };
        }
        if t == Term::KwTrue as u8 || t == Term::KwFalse as u8 {
            return Some((self.fx(ids::BOOL), Val::Bool(t == Term::KwTrue as u8)));
        }
        if t == Term::CharLit as u8 {
            return Some((self.fx(ids::CHAR), Val::Char(literals::parse_char(icx.tok_str(a))?)));
        }
        if t == Term::StrLit as u8 {
            let b = literals::parse_str(icx.tok_str(a))?;
            return Some((self.fx(ids::REF_STR), Val::Str(b.into_boxed_slice())));
        }
        if t == Term::FmtLit as u8 || t == Term::KwSelf as u8 {
            return self.e0904(n, "a non-const item");
        }
        if t == Term::KwFn as u8 {
            return self.e0904(n, "a loop");
        }
        if t == Term::LParen as u8 {
            let close = icx.close(a);
            if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
                return self.eval(kids[0], want);
            }
            let wants: Vec<Option<TypeId>> = match want.map(|w| self.env.kind(w).clone()) {
                Some(TypeKind::Tuple(ts)) if ts.len() == kids.len() => ts.iter().map(|&x| Some(x)).collect(),
                _ => vec![None; kids.len()],
            };
            let mut tys = Vec::new();
            let mut vals = Vec::new();
            for (i, &c) in kids.iter().enumerate() {
                let (ct, cv) = self.eval(c, wants[i])?;
                let (ct, cv) = if ct.is_none() {
                    let w = wants[i]?;
                    (w, self.fix_lit(c, cv, w)?)
                } else {
                    (ct, cv)
                };
                tys.push(ct);
                vals.push(cv);
            }
            let tt = self.env.mk_tuple(tys);
            return Some((tt, Val::Agg(vals)));
        }
        if t == Term::LBracket as u8 {
            let elem_want = match want.map(|w| self.env.kind(w).clone()) {
                Some(TypeKind::Array { elem, .. }) => Some(elem),
                _ => None,
            };
            // `[e; n]`
            if kids.len() == 2 && icx.is(icx.end_tok(kids[0]) + 1, Term::Semi) {
                let (et, ev) = self.eval(kids[0], elem_want)?;
                let (et, ev) = if et.is_none() {
                    let w = elem_want?;
                    (w, self.fix_lit(kids[0], ev, w)?)
                } else {
                    (et, ev)
                };
                let us = self.fx(ids::USIZE);
                let len = eval_u64(self.env, self.te, kids[1], us);
                let ConstArg::Value(len) = len else { return None };
                if len > 1 << 24 {
                    return None;
                }
                let at = self.env.intern(TypeKind::Array { elem: et, len: ConstArg::Value(len) });
                return Some((at, Val::Agg(vec![ev; len as usize])));
            }
            let mut et = elem_want;
            let mut vals = Vec::with_capacity(kids.len());
            let mut pending = Vec::new();
            for &c in kids {
                let (ct, cv) = self.eval(c, et)?;
                if ct.is_none() {
                    pending.push(vals.len());
                } else if et.is_none() {
                    et = Some(ct);
                } else if et != Some(ct) {
                    return None;
                }
                vals.push((c, cv));
            }
            let elem = et?;
            let mut out = Vec::with_capacity(vals.len());
            for (i, (c, v)) in vals.into_iter().enumerate() {
                if pending.contains(&i) {
                    out.push(self.fix_lit(c, v, elem)?);
                } else {
                    out.push(v);
                }
            }
            let at = self.env.intern(TypeKind::Array { elem, len: ConstArg::Value(out.len() as u64) });
            return Some((at, Val::Agg(out)));
        }
        if t == Term::Ident as u8 {
            // A single name: a const, a unit variant (resolved in the chain), a generic const.
            let u = icx.uses(n).first()?;
            return self.path_value(n, u.res, want);
        }
        None
    }

    fn path_value(&mut self, n: u32, res: Res, want: Option<TypeId>) -> R {
        match res {
            Res::Decl(d) => {
                let decl = *self.env.prog().decl(d)?;
                match decl.kind {
                    DeclKind::Const => self.env.const_value(d),
                    DeclKind::Variant => self.variant_value(n, d, &[], want),
                    DeclKind::Error => None,
                    DeclKind::Fn | DeclKind::Method => self.e0904(n, "a non-const item"),
                    _ => self.e0904(n, "a non-const item"),
                }
            }
            Res::Unresolved => None,
            Res::Local(b) => match self.params.iter().find(|p| p.0 == b.node) {
                Some(p) => Some(p.1.clone()),
                None => self.e0904(n, "a non-const item"),
            },
            _ => self.e0904(n, "a non-const item"),
        }
    }

    /// A call of the constructor `m` ([`LangItems::const_ctors`]): the value of the struct
    /// literal its body returns, with its parameters bound to the argument values.
    fn ctor_call(&mut self, n: u32, m: DeclId, args: &[u32], want: Option<TypeId>) -> R {
        let prog = self.te.icx.prog;
        let decl = *prog.decl(m)?;
        let (ptys, ret) = self.env.fn_types(m)?;
        if ptys.len() != args.len() {
            return None;
        }
        // The type arguments come from the expected type: the return type is the constructed
        // ADT over the constructor's own generic params.
        let gargs: Option<Vec<GArg>> =
            match (want.map(|w| self.env.kind(w).clone()), self.env.kind(ret).clone()) {
                (Some(TypeKind::Adt { def, args }), TypeKind::Adt { def: rd, .. }) if def == rd => {
                    Some(args.to_vec())
                }
                _ => None,
            };
        let mut vals = Vec::with_capacity(args.len());
        for (&a, &pt) in args.iter().zip(&ptys) {
            let pw = match &gargs {
                Some(g) => Some(self.env.subst(pt, g, None)),
                None if !self.env.has_params(pt) => Some(pt),
                None => None,
            };
            let (t, v) = self.eval(a, pw)?;
            vals.push(match pw {
                Some(w) if t.is_none() => (w, self.fix_lit(a, v, w)?),
                _ => (t, v),
            });
        }
        let want = match want {
            Some(w) => Some(w),
            None if !self.env.has_params(ret) => Some(ret),
            None => None,
        };
        let item = prog.item(decl.item)?;
        let file = prog.file(item.file)?;
        let info = file.parsed.items.get(item.index as usize)?;
        let br = Brackets::new(&file.parsed.tokens, info.start, info.end);
        let icx = ItemCx::new(prog, decl.item, &br)?;
        let pnodes: Vec<u32> = icx
            .child(decl.node, NodeKind::Params)
            .map(|p| icx.kids_vec(p))
            .unwrap_or_default()
            .into_iter()
            .filter(|&c| icx.kind(c) == Some(NodeKind::Param))
            .collect();
        if pnodes.len() != vals.len() {
            return None;
        }
        // The body is exactly `return e;`.
        let body = icx.child(decl.node, NodeKind::Block)?;
        let stmts = icx.kids_vec(body);
        let e = match stmts[..] {
            [s] if icx.is(icx.a(s), Term::KwReturn) => icx.first_child(s),
            _ => None,
        };
        let Some(e) = e else { return self.e0904(n, "a call") };
        let te = TyEnv { icx, n_parent: 0, self_ty: None };
        let mut inner = Ev {
            env: &mut *self.env,
            te: &te,
            depth: self.depth,
            params: pnodes.into_iter().zip(vals).collect(),
        };
        inner.eval(e, want)
    }

    fn variant_value(&mut self, n: u32, v: DeclId, args: &[u32], want: Option<TypeId>) -> R {
        let decl = *self.env.prog().decl(v)?;
        let adt = self.env.adt_def(decl.parent)?;
        let idx = adt.variants.iter().position(|x| x.decl == v)?;
        let vd = adt.variants[idx].clone();
        let gargs = self.adt_args_from(adt.decl, want, adt.generics.len())?;
        if vd.fields.len() != args.len() {
            return None;
        }
        let mut vals = Vec::new();
        for (f, &a) in vd.fields.iter().zip(args) {
            let ft = self.env.subst(f.ty, &gargs, None);
            let (at, av) = self.eval(a, Some(ft))?;
            vals.push(self.fix(a, at, av, ft)?);
        }
        let _ = n;
        let t = self.env.mk_adt(adt.decl, gargs);
        Some((t, Val::Variant(idx as u32, vals)))
    }

    /// The type args of ADT `d` from the expected type (consts have declared types).
    fn adt_args_from(&mut self, d: DeclId, want: Option<TypeId>, count: usize) -> Option<Vec<GArg>> {
        if count == 0 {
            return Some(Vec::new());
        }
        match want.map(|w| self.env.kind(w).clone()) {
            Some(TypeKind::Adt { def, args }) if def == d => Some(args.to_vec()),
            _ => None,
        }
    }

    fn postfix(&mut self, n: u32, kids: &[u32], want: Option<TypeId>) -> R {
        let icx = self.te.icx;
        let prim = *kids.first()?;
        let pa = icx.a(prim);
        let mut kid = 1usize;
        // Builtins: `size_of::<T>()` / `align_of::<T>()`.
        if icx.is(pa, Term::Ident)
            && let Some(u) = icx.uses(prim).first()
            && let Res::Builtin(b) = u.res
        {
            let pk = icx.kids_vec(prim);
            let ga = pk.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::GenericArgs))?;
            let args = icx.kids_vec(ga);
            if args.len() != 1 {
                return None;
            }
            let GArg::Type(t) = lower_garg(self.env, self.te, args[0], false) else { return None };
            if t == self.env.error_ty() || self.env.has_params(t) {
                return None;
            }
            let (size, align) = self.env.layout(t)?;
            let v = if b == Builtin::SizeOf { size } else { align };
            return Some((self.fx(ids::USIZE), Val::Int(v as i128)));
        }
        // The head chain: uses keyed at the Primary.
        let mut p = icx.primary_end(prim) + 1;
        let mut cur: Option<(TypeId, Val)> = None;
        let mut path: Option<Res> = None;
        if icx.is(pa, Term::Ident) {
            path = icx.use_at(prim, pa).map(|u| u.res);
            // a single name is a value right away
        } else {
            cur = Some(self.primary(prim, &icx.kids_vec(prim), want)?);
        }
        loop {
            let t = icx.tk(p);
            if t == Term::Dot as u8 && icx.is(p + 1, Term::Ident) {
                let seg = p + 1;
                if let Some(u) = icx.use_at(prim, seg) {
                    path = Some(u.res);
                    p += 2;
                    if icx.is(p, Term::ColonColon) {
                        p = icx.angle_end(p + 1).0;
                        kid += 1;
                    }
                    continue;
                }
                // a field of a value
                let base = match cur.take() {
                    Some(c) => c,
                    None => self.path_value(prim, path.take()?, None)?,
                };
                let name = icx.sym(seg);
                cur = Some(self.field(base, name)?);
                p += 2;
                continue;
            }
            if t == Term::Dot as u8 && icx.is(p + 1, Term::IntLit) {
                let base = match cur.take() {
                    Some(c) => c,
                    None => self.path_value(prim, path.take()?, None)?,
                };
                let i: usize = icx.tok_str(p + 1).parse().ok()?;
                cur = Some(match (self.env.kind(base.0).clone(), base.1) {
                    (TypeKind::Tuple(ts), Val::Agg(vs)) => (*ts.get(i)?, vs.get(i)?.clone()),
                    _ => return None,
                });
                p += 2;
                continue;
            }
            if t == Term::LParen as u8 {
                // A variant constructor is const; any other call is not.
                let close = icx.close(p);
                let args = if close == p + 1 {
                    crate::itemcx::Kids::default()
                } else {
                    let a = *kids.get(kid)?;
                    kid += 1;
                    icx.kids_vec(a)
                };
                if cur.is_none()
                    && let Some(Res::Decl(d)) = path
                    && self.env.prog().decl(d).is_some_and(|x| x.kind == DeclKind::Variant)
                {
                    path = None;
                    cur = Some(self.variant_value(n, d, &args, want)?);
                    p = close + 1;
                    continue;
                }
                if cur.is_none()
                    && let Some(Res::Decl(d)) = path
                    && self.env.lang().const_ctors.contains(&d)
                {
                    path = None;
                    cur = Some(self.ctor_call(n, d, &args, want)?);
                    p = close + 1;
                    continue;
                }
                return self.e0904(n, "a call");
            }
            if t == Term::DotLBrace as u8 {
                let sl = *kids.get(kid)?;
                kid += 1;
                let Some(Res::Decl(d)) = path.take() else { return self.e0904(n, "a non-const item") };
                cur = Some(self.struct_lit(d, sl, want)?);
                p = icx.close(p) + 1;
                continue;
            }
            if t == Term::LBracket as u8 {
                let ie = *kids.get(kid)?;
                kid += 1;
                let base = match cur.take() {
                    Some(c) => c,
                    None => self.path_value(prim, path.take()?, None)?,
                };
                let (it, iv) = self.eval(ie, Some(self.fx(ids::USIZE)))?;
                if !(it.is_none() || it == self.fx(ids::USIZE)) {
                    return None;
                }
                let Val::Int(i) = iv else { return None };
                cur = Some(match (self.env.kind(base.0).clone(), base.1) {
                    (TypeKind::Array { elem, .. }, Val::Agg(vs)) => {
                        (elem, vs.get(usize::try_from(i).ok()?)?.clone())
                    }
                    _ => return None,
                });
                p = icx.close(p) + 1;
                continue;
            }
            if t == Term::Question as u8 {
                return self.e0904(n, "a non-const item");
            }
            break;
        }
        match cur {
            Some(c) => Some(c),
            None => self.path_value(prim, path?, want),
        }
    }

    fn field(&mut self, (t, v): (TypeId, Val), name: crate::ids::Sym) -> R {
        let (inner, _) = self.env.peel_refs(t);
        let TypeKind::Adt { def, args } = self.env.kind(inner).clone() else { return None };
        let adt = self.env.adt_def(def)?;
        if adt.is_enum {
            return None;
        }
        let vd = adt.variants.first()?;
        let i = vd.fields.iter().position(|f| f.name == name)?;
        let ft = self.env.subst(vd.fields[i].ty, &args, None);
        match v {
            Val::Agg(vs) => Some((ft, vs.get(i)?.clone())),
            _ => None,
        }
    }

    fn struct_lit(&mut self, d: DeclId, sl: u32, want: Option<TypeId>) -> R {
        let icx = self.te.icx;
        let decl = *self.env.prog().decl(d)?;
        let (adt_decl, vidx) = match decl.kind {
            DeclKind::Struct => (d, None),
            DeclKind::Variant => (decl.parent, Some(d)),
            _ => return None,
        };
        let adt = self.env.adt_def(adt_decl)?;
        let vi = match vidx {
            Some(v) => adt.variants.iter().position(|x| x.decl == v)?,
            None => 0,
        };
        let vd = adt.variants.get(vi)?.clone();
        if vd.shape != VariantShape::Record {
            return None;
        }
        let gargs = self.adt_args_from(adt_decl, want, adt.generics.len())?;
        let mut vals: Vec<Option<Val>> = vec![None; vd.fields.len()];
        for e in icx.kids_vec(sl) {
            let name_tok = icx.a(e).checked_sub(2)?;
            let name = icx.sym(name_tok);
            let fi = vd.fields.iter().position(|f| f.name == name)?;
            let ft = self.env.subst(vd.fields[fi].ty, &gargs, None);
            let (et, ev) = self.eval(e, Some(ft))?;
            vals[fi] = Some(self.fix(e, et, ev, ft)?);
        }
        let vals: Option<Vec<Val>> = vals.into_iter().collect();
        let t = self.env.mk_adt(adt_decl, gargs);
        let v = if adt.is_enum { Val::Variant(vi as u32, vals?) } else { Val::Agg(vals?) };
        Some((t, v))
    }
}

fn round_float(f: f64, k: &TypeKind) -> f64 {
    if matches!(k, TypeKind::Float(FloatBits::F32)) { (f as f32) as f64 } else { f }
}

fn round_lit<E: Interner + ?Sized>(v: Val, t: TypeId, env: &E) -> Val {
    match v {
        Val::Float(f) if env.is_float(t) => Val::Float(round_float(f, env.kind(t))),
        other => other,
    }
}

/// Reads a const value as a `u64` (array lengths, patterns).
pub fn as_u64(v: &Val) -> Option<u64> {
    match v {
        Val::Int(i) if *i >= 0 && *i <= u64::MAX as i128 => Some(*i as u64),
        _ => None,
    }
}

impl ConstArg {
    pub fn value(self) -> Option<u64> {
        match self {
            ConstArg::Value(v) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping() {
        assert_eq!(wrap_int(256, false, 8), 0);
        assert_eq!(wrap_int(128, true, 8), -128);
        assert_eq!(wrap_int(-1, false, 8), 255);
        assert_eq!(wrap_int(-1, true, 64), -1);
    }
}
