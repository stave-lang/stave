//! Const-expr evaluation: literals, consts, arithmetic, comparison, the `as`
//! family, `size_of::<T>()` / `align_of::<T>()`, `if then else`, and aggregate literals (array
//! `[a, b]` / `[e; n]`, tuple, struct, variant); also indexing and field access of const
//! aggregates (an extension LANG.md does not list), and a call of a constructor fixed by path
//! ([`LangItems::const_ctors`], contracts/intrinsics.md "Paths fixed by the compiler"), which is
//! evaluated as the struct literal its body returns, with the parameters bound to the
//! arguments. Anything else (calls, loops, non-const items such as fns,
//! statics, locals, references, f-strings, ranges, `match`) is E0904. A const that depends on
//! itself is E0904 case `cycle` (reported by the signature builder, which evaluates consts on
//! demand; a const's own type is part of it).
//!
//! **Two passes.** `decide` gives every node its type first, by the body checker's rules
//! (`expr.rs`): an unsuffixed literal takes the type of the other operand of its operator, of the
//! expected type, or of the chain it is in (`200 + 100 - 100u8` is `u8` throughout); a cast
//! operand, a tuple element without an expected type, and a comparison of literals only have
//! no type (E0202). Then the rules of literals (range) and operators (defined on the type) are
//! checked, and `value` evaluates each node in its decided type, never in a wider one; a node
//! without a decided type there is an internal error. `value` is lazy where the run time is
//! (`if` arms, `&&`, `||`), so a trap is an error exactly when the expression traps at run time.
//! The decided types of the literal and operator nodes go into [`Evaluated::types`]; the body
//! checker's types for the same nodes must be the same (else an internal error).
//!
//! Type errors inside a const initializer are reported by the body checker (which also types
//! const and `static` initializers for the `types` dump), so evaluation reports nothing for them.
//! Every failure that reports nothing leaves a deferred diagnostic instead (the innermost one): a
//! precise one where the form is known (a method call or a user operator is E0904 "a call"),
//! else E0904 at the innermost node without a value. The owner of the expression reports it
//! unless it reported an error itself: the body checker for a const initializer, [`eval_u64`]
//! itself for an array length or const argument. So an initializer without a value always has
//! a diagnostic.
//! Arithmetic overflow, division by zero, `MIN / -1`, and `MIN % -1` in a const are errors
//! (const contexts keep their errors): E0203 at the operation (no dedicated code). A shift
//! amount outside `0..bits` is E0205 (the body checker reports a literal or const amount).
//! An array literal of more than [`MAX_DATA_BYTES`] bytes is E0904 case `size`; `[e; n]` is one
//! value repeated ([`Val::Rep`]), never `n` values.

use rustc_hash::FxHashMap;
use stave_diag::Diagnostic;
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
use crate::types::{self, ConstArg, FloatBits, GArg, TypeId, TypeKind, ids};
use std::sync::Arc;

/// A const value (its type is kept alongside).
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Int(i128),
    Float(f64),
    Bool(bool),
    Char(u32),
    Str(Box<[u8]>),
    /// Tuples, arrays, structs (fields in declaration order); unit is `Agg([])`. Shared: a value
    /// used twice (`(C, C)`) is one value, so evaluation memory grows with its steps, not with
    /// the value's leaves.
    Agg(Arc<[Val]>),
    /// An array of `n` copies of one value (`[e; n]`): one value, not `n`, so compile-time data
    /// costs what it is written with.
    Rep(Arc<Val>, u64),
    /// An enum value: variant index, fields.
    Variant(u32, Arc<[Val]>),
}

/// What a const evaluation needs from its context.
pub(crate) trait CEnv: LowerHost {
    fn const_value(&mut self, d: DeclId) -> Option<(TypeId, Val)>;
    fn adt_def(&mut self, d: DeclId) -> Option<AdtDef>;
    /// [`crate::traits::type_bytes`] in this environment's table.
    fn layout(&mut self, t: TypeId) -> Result<(u64, u64), crate::traits::NoLayout>;
    fn lang(&self) -> &LangItems;
    /// The parameter types and return type of fn `d`.
    fn fn_types(&mut self, d: DeclId) -> Option<(Vec<TypeId>, TypeId)>;
    /// An internal error of the evaluation: the compilation stops with it.
    fn internal(&mut self, msg: String);
}

impl CEnv for SigBuilder<'_> {
    fn const_value(&mut self, d: DeclId) -> Option<(TypeId, Val)> {
        SigBuilder::const_value(self, d)
    }

    fn adt_def(&mut self, d: DeclId) -> Option<AdtDef> {
        SigBuilder::adt_def(self, d)
    }

    fn layout(&mut self, t: TypeId) -> Result<(u64, u64), crate::traits::NoLayout> {
        SigBuilder::layout(self, t)
    }

    fn lang(&self) -> &LangItems {
        &self.lang
    }

    fn fn_types(&mut self, d: DeclId) -> Option<(Vec<TypeId>, TypeId)> {
        SigBuilder::fn_types(self, d)
    }

    fn internal(&mut self, msg: String) {
        self.internal.push(msg);
    }
}

/// A value with its type.
type R = Option<(TypeId, Val)>;

/// The decided type of a node: a type, or the group of unsuffixed literals that share one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    T(TypeId),
    Lit(u32),
}

/// A group of unsuffixed literals of one type (union-find): `first` is the node of its first
/// literal, where E0202 points.
struct Grp {
    parent: u32,
    float: bool,
    fixed: Option<TypeId>,
    first: u32,
}

/// A literal: magnitude (ints) or value (floats), and a folded unary minus.
#[derive(Clone, Copy)]
struct Lit {
    mag: u128,
    fval: f64,
    neg: bool,
    float: bool,
}

/// A rule an operator node needs of its operand type, checked once every group has its type.
#[derive(Clone, Copy)]
enum Rule {
    Bin { tok: u32, ty: Ty },
    Un { tok: u32, ty: Ty },
    Cmp { tok: u32, ty: Ty },
}

/// One step of a postfix chain, as decided; `value` runs them in order.
enum Step {
    /// A value known when the chain is decided: a const, a constructor parameter.
    Known(Val),
    /// The value of a head expression node.
    Node(u32),
    /// Field `i` of a struct value, or element `i` of a tuple value.
    Part(usize),
    /// Element at the index node of an array value of the type.
    Index(u32, TypeId),
    /// Variant `idx` of an enum with argument nodes.
    Variant(u32, Vec<u32>),
    /// A struct literal (variant `idx` of an enum if `is_enum`) with `(field index, node)`.
    Struct { idx: u32, is_enum: bool, n: usize, fields: Vec<(usize, u32)> },
    /// A call of the constructor decl with argument nodes.
    Ctor(DeclId, Vec<u32>),
}

struct Ev<'e, 'a, E: CEnv> {
    env: &'e mut E,
    te: &'e TyEnv<'a>,
    depth: u32,
    /// Inside a constructor body: its parameters (`Param` node, value).
    params: Vec<(u32, (TypeId, Val))>,
    /// Diagnostics reported, plus failures explained by an error reported elsewhere.
    nrep: u32,
    /// The deferred diagnostic of the innermost failure that reported nothing.
    stuck: Option<Diagnostic>,
    /// The decided type of every node.
    nty: FxHashMap<u32, Ty>,
    grps: Vec<Grp>,
    lits: FxHashMap<u32, Lit>,
    /// Literal nodes in decide order (range checks).
    lit_order: Vec<u32>,
    rules: Vec<(u32, Rule)>,
    steps: FxHashMap<u32, Vec<Step>>,
    casts: FxHashMap<u32, Vec<(u8, TypeId)>>,
    /// The first internal error: a node `value` reached without a decided type.
    internal: Option<String>,
}

const MAX_DEPTH: u32 = 2000;

/// The most bytes of compile-time data a value of a const-expr may have, and the most bytes of
/// a `const` or `static` (contracts/lang-details.md "Size limits"): const evaluation holds the
/// value and the image holds its bytes.
pub const MAX_DATA_BYTES: u64 = 1 << 26;

/// The most bytes every `static` and `const` of a program may take together, file data and zero
/// fill alike, each rounded up to its alignment (contracts/lang-details.md "Size limits"). Both
/// targets lay an image out as text, then read-only data, data, zero fill, and reach a static
/// PC-relatively: x86-64 by a `rel32` (`S - P` within ±2^31, so the code and all data before the
/// last static's start must stay below 2 GiB), AArch64 by `ADRP` (±2^32), and the ELF writer
/// takes at most 2^32 bytes in all. x86-64 binds: half of its 2 GiB is kept for the code, string
/// literals, the runtime's own data, and padding, so 2^30 bytes on both targets.
pub const MAX_PROGRAM_DATA_BYTES: u64 = 1 << 30;

/// The compile-time data limit on type `t` ([`MAX_DATA_BYTES`]): `Err(bytes)` past it, with
/// `u64::MAX` for a size past `u64`. A type with params or an error passes (its instances, or
/// the error's diagnostic, decide); a type without a layout otherwise is rejected.
pub(crate) fn data_size<E: CEnv + ?Sized>(env: &mut E, t: TypeId) -> Result<(), u64> {
    if env.has_params(t) || env.has_error(t) {
        return Ok(());
    }
    match env.layout(t) {
        Ok((b, _)) if b <= MAX_DATA_BYTES => Ok(()),
        Ok((b, _)) => Err(b),
        Err(_) => Err(u64::MAX),
    }
}

/// The most bytes a value of a fn may have (a parameter, binding, temporary, or return value,
/// per instance; contracts/lang-details.md "Size limits"): the x86-64 back end keeps a value
/// in one frame slot of at most 256 MiB.
pub const MAX_VALUE_BYTES: u64 = 1 << 28;

/// The most bytes the values of one fn instance may take together (its frame sum: parameters,
/// bindings, every expression value but a place read in place, two slots for a value `if` /
/// `match` (its result and the live arm), each call's result, by-value receiver and arguments, the return value;
/// contracts/lang-details.md "Size limits"). A back end's frame holds at most 1 GiB. Its frame
/// measures at most 1.5 times that sum (a filled local, a whole aggregate loaded to read a part
/// of it) on both targets, debug and release; 16 MiB are kept for the back end's own slots and
/// spills: (1 GiB - 16 MiB) / 1.5.
pub const MAX_FRAME_BYTES: u64 = ((1 << 30) - (1 << 24)) / 3 * 2;

/// The `what` of the E0904 at the innermost node without a value.
const OUTSIDE: &str = "an expression outside the const-expr forms";

/// The result of evaluating a const or `static` initializer.
pub(crate) struct Evaluated {
    pub value: Option<Val>,
    /// Without a value and without a diagnostic of the evaluation: the one the owner reports if
    /// the initializer is otherwise well typed (module doc).
    pub deferred: Option<Diagnostic>,
    /// The decided type of every literal and operator node of a primitive scalar type: the
    /// body checker's types for these nodes must be the same (else an internal error).
    pub types: Vec<(u32, TypeId)>,
}

/// Evaluates the initializer `e` of a const or `static` of type `ty`.
pub(crate) fn eval_item_const<E: CEnv>(env: &mut E, te: &TyEnv<'_>, e: u32, ty: TypeId) -> Evaluated {
    let mut ev = Ev::new(env, te);
    let v = ev.eval(e, Some(ty));
    ev.flush_internal();
    let types = ev.scalar_types();
    match v {
        Some((_, v)) => Evaluated { value: Some(v), deferred: None, types },
        None => Evaluated { value: None, deferred: ev.deferred(e), types },
    }
}

/// Evaluates an array length / const generic argument to a `u64` (or a const generic param).
/// No other checker types these expressions, so a deferred diagnostic is reported here.
pub(crate) fn eval_u64<E: CEnv>(env: &mut E, te: &TyEnv<'_>, n: u32, ty: TypeId) -> ConstArg {
    let (c, d) = eval_u64_deferred(env, te, n, ty);
    if d.is_some() {
        env.report(d);
    }
    c
}

/// [`eval_u64`] returning the deferred diagnostic, for a length the body checker also types
/// (`[e; n]`): reported only if that found no error.
pub(crate) fn eval_u64_deferred<E: CEnv>(
    env: &mut E,
    te: &TyEnv<'_>,
    n: u32,
    ty: TypeId,
) -> (ConstArg, Option<Diagnostic>) {
    // A bare const generic parameter.
    if let Some(p) = const_param_path(te, n) {
        if ty.is_none() {
            return (ConstArg::Param(p), None);
        }
        return (crate::sigs::const_param_arg(env, &te.icx, n, p, ty), None);
    }
    let mut ev = Ev::new(env, te);
    let want = if ty.is_none() { None } else { Some(ty) };
    let r = ev.eval(n, want);
    ev.flush_internal();
    match r {
        Some((_, Val::Int(i))) if i >= 0 && i <= u64::MAX as i128 => (ConstArg::Value(i as u64), None),
        // a value of an unsigned type is in range: only a signed const param type is left
        Some(_) => (ConstArg::Error, ev.e0904_at(n, OUTSIDE)),
        None => (ConstArg::Error, ev.deferred(n)),
    }
}

/// If `n` is a bare name of a const generic param: its flat index.
fn const_param_path(te: &TyEnv<'_>, n: u32) -> Option<u32> {
    let icx = te.icx;
    if icx.kind(n)? != NodeKind::Primary {
        return None;
    }
    let u = icx.uses(n).first()?;
    match u.res {
        Res::Generic(g) if g.is_const && icx.primary_end(n) == icx.a(n) => {
            Some(if g.level == 0 { g.index as u32 } else { te.n_parent + g.index as u32 })
        }
        _ => None,
    }
}

fn wrap_int(v: i128, signed: bool, bits: u32) -> i128 {
    let m = (v as u128) & (if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 });
    if signed && bits < 128 && (m >> (bits - 1)) & 1 == 1 { (m as i128) - (1i128 << bits) } else { m as i128 }
}

/// The value cells of concrete type `t`: one per scalar, reference, or fn value; the fields of a
/// tuple, struct, or enum variant added; an array `[T; n]` n times `T`'s. Saturates at `u64::MAX`;
/// `None` for a type without a known shape (generic, erroneous, nested past the type limit).
/// Memoised per type: types share components.
pub(crate) fn cells<E: CEnv + ?Sized>(env: &mut E, t: TypeId) -> Option<u64> {
    let mut memo = FxHashMap::default();
    cells_memo(env, t, 0, &mut memo)
}

fn cells_memo<E: CEnv + ?Sized>(
    env: &mut E,
    t: TypeId,
    depth: u32,
    memo: &mut FxHashMap<TypeId, Option<u64>>,
) -> Option<u64> {
    if depth >= crate::typedepth::MAX_TYPE_DEPTH {
        return None;
    }
    if let Some(&c) = memo.get(&t) {
        return c;
    }
    // a cycle through a by-value field has no layout (E0005 / E0911 elsewhere)
    memo.insert(t, None);
    let k = env.kind(t).clone();
    let sum = |v: &[u64]| v.iter().fold(0u64, |a, &b| a.saturating_add(b)).max(1);
    let r = match k {
        TypeKind::Array { elem, len: ConstArg::Value(n) } => {
            cells_memo(env, elem, depth + 1, memo).map(|c| c.saturating_mul(n))
        }
        TypeKind::Array { .. } | TypeKind::Param(_) | TypeKind::Error => None,
        TypeKind::Tuple(ts) => {
            let mut v = Vec::with_capacity(ts.len());
            for &x in ts.iter() {
                v.push(cells_memo(env, x, depth + 1, memo)?);
            }
            Some(sum(&v))
        }
        TypeKind::Adt { def, args } => {
            let adt = env.adt_def(def)?;
            let mut most = 0u64;
            for vd in &adt.variants {
                let mut v = Vec::with_capacity(vd.fields.len());
                for f in &vd.fields {
                    let ft = env.subst(f.ty, &args, None);
                    v.push(cells_memo(env, ft, depth + 1, memo)?);
                }
                most = most.max(sum(&v));
            }
            Some(most.max(1))
        }
        _ => Some(1),
    };
    memo.insert(t, r);
    r
}

impl<'e, 'a, E: CEnv> Ev<'e, 'a, E> {
    fn new(env: &'e mut E, te: &'e TyEnv<'a>) -> Self {
        Ev {
            env,
            te,
            depth: 0,
            params: Vec::new(),
            nrep: 0,
            stuck: None,
            nty: FxHashMap::default(),
            grps: Vec::new(),
            lits: FxHashMap::default(),
            lit_order: Vec::new(),
            rules: Vec::new(),
            steps: FxHashMap::default(),
            casts: FxHashMap::default(),
            internal: None,
        }
    }

    fn report(&mut self, d: Option<Diagnostic>) {
        self.nrep += 1;
        self.env.report(d);
    }

    /// A failure explained by an error reported elsewhere (an unresolved name, an erroneous
    /// type, a const without a value).
    fn explained<T>(&mut self) -> Option<T> {
        self.nrep += 1;
        None
    }

    /// A failure whose diagnostic `d` holds if the expression is otherwise well typed.
    fn defer<T>(&mut self, d: Option<Diagnostic>) -> Option<T> {
        if self.stuck.is_none() {
            self.stuck = d;
        }
        None
    }

    /// An internal error: the evaluation stops without a value.
    fn ice<T>(&mut self, n: u32, what: &str) -> Option<T> {
        if self.internal.is_none() {
            let text = self.te.icx.node_text(n);
            self.internal = Some(format!("const evaluation: `{text}`: {what}"));
        }
        None
    }

    fn flush_internal(&mut self) {
        if let Some(m) = self.internal.take() {
            self.nrep += 1;
            self.env.internal(m);
        }
    }

    fn e0904_at(&self, n: u32, what: &str) -> Option<Diagnostic> {
        mk("E0904", None, &[("what", what)], self.te.icx.node_span(n), &[])
    }

    /// After a failed evaluation of `top`: the deferred diagnostic, unless one was reported.
    fn deferred(&mut self, top: u32) -> Option<Diagnostic> {
        if self.nrep > 0 {
            return None;
        }
        if let Some(d) = self.stuck.take() {
            return Some(d);
        }
        self.e0904_at(top, OUTSIDE)
    }

    /// Well-known type `f` ([`ids`]) in the environment's table (it may be canonical).
    fn fx(&self, f: TypeId) -> TypeId {
        self.env.fixed_ty(f)
    }

    fn int_range(&self, t: TypeId) -> Option<(i128, i128)> {
        literals::int_range_kind(self.env.kind(t))
    }

    fn e0904<T>(&mut self, n: u32, what: &str) -> Option<T> {
        let d = self.e0904_at(n, what);
        self.report(d);
        None
    }

    fn show(&mut self, t: TypeId) -> String {
        types::display(self.env, &NameCx { prog: self.env.prog(), generics: &[] }, t)
    }

    /// E0203 at the operation that ends at node `n` (the operands from `start`): its value
    /// is outside type `t` or it divides by zero, so it traps at run time.
    fn overflow<T>(&mut self, start: u32, n: u32, t: TypeId) -> Option<T> {
        let icx = self.te.icx;
        let sp = icx.range_span(icx.a(start), icx.end_tok(n));
        let text = icx.span_text(sp).to_string();
        let tn = self.show(t);
        let (lo, hi) = self.int_range(t).unwrap_or((0, 0));
        let d = mk(
            "E0203",
            None,
            &[("literal", &text), ("type", &tn), ("min", &lo.to_string()), ("max", &hi.to_string())],
            sp,
            &[],
        );
        self.report(d);
        None
    }

    // ---------- literal groups ----------

    fn find(&mut self, g: u32) -> u32 {
        let mut r = g;
        while self.grps[r as usize].parent != r {
            r = self.grps[r as usize].parent;
        }
        let mut x = g;
        while self.grps[x as usize].parent != r {
            let p = self.grps[x as usize].parent;
            self.grps[x as usize].parent = r;
            x = p;
        }
        r
    }

    fn resolve(&mut self, t: Ty) -> Ty {
        match t {
            Ty::Lit(g) => {
                let r = self.find(g);
                match self.grps[r as usize].fixed {
                    Some(t) => Ty::T(t),
                    None => Ty::Lit(r),
                }
            }
            t => t,
        }
    }

    /// The type the hint gives a later operand: a decided type, else the outer hint.
    fn hint_of(&mut self, t: Ty, outer: Option<TypeId>) -> Option<TypeId> {
        match self.resolve(t) {
            Ty::T(t) => Some(t),
            Ty::Lit(_) => outer,
        }
    }

    /// Fixes group `g` to `t`; `false` if `t` is not a type of the group's literal kind.
    fn fix(&mut self, g: u32, t: TypeId) -> bool {
        let r = self.find(g);
        if let Some(prev) = self.grps[r as usize].fixed {
            return prev == t;
        }
        let float = self.grps[r as usize].float;
        let ok = if float { self.env.is_float(t) } else { self.env.int(t).is_some() };
        if ok {
            self.grps[r as usize].fixed = Some(t);
        }
        ok
    }

    fn kind_name(&self, g: u32) -> &'static str {
        if self.grps[g as usize].float { "{float}" } else { "{integer}" }
    }

    /// E0201 at node `n`: found `found` where `expected` is required.
    fn mismatch<T>(&mut self, n: u32, expected: &str, found: &str) -> Option<T> {
        let icx = self.te.icx;
        let text = icx.node_text(n).to_string();
        let d = mk(
            "E0201",
            Some("no-cast"),
            &[("expected", expected), ("found", found), ("operand", &text)],
            icx.node_span(n),
            &[],
        );
        self.defer(d)
    }

    fn is_err(&self, t: TypeId) -> bool {
        self.env.is_error_ty(t) || self.env.has_error(t)
    }

    /// Node `n` of type `t` where a `want` is required (a pending group takes it).
    fn coerce(&mut self, n: u32, t: Ty, want: TypeId) -> Option<TypeId> {
        match self.resolve(t) {
            Ty::T(x) if x == want => Some(want),
            Ty::T(x) if self.is_err(x) || self.is_err(want) => self.explained(),
            Ty::T(x) => {
                let (e, f) = (self.show(want), self.show(x));
                self.mismatch(n, &e, &f)
            }
            Ty::Lit(g) => {
                if self.fix(g, want) {
                    return Some(want);
                }
                if self.is_err(want) {
                    return self.explained();
                }
                let (e, f) = (self.show(want), self.kind_name(g));
                self.mismatch(n, &e, f)
            }
        }
    }

    /// The two operand types of one operator, `r` the right operand.
    fn unify(&mut self, r: u32, a: Ty, b: Ty) -> Option<Ty> {
        match (self.resolve(a), self.resolve(b)) {
            (Ty::T(x), Ty::T(y)) if x == y => Some(Ty::T(x)),
            (Ty::T(x), Ty::T(y)) if self.is_err(x) || self.is_err(y) => self.explained(),
            (Ty::T(x), Ty::T(y)) => {
                let (e, f) = (self.show(x), self.show(y));
                self.mismatch(r, &e, &f)
            }
            (Ty::Lit(g), Ty::T(y)) | (Ty::T(y), Ty::Lit(g)) => {
                if self.fix(g, y) {
                    return Some(Ty::T(y));
                }
                if self.is_err(y) {
                    return self.explained();
                }
                let (e, f) = (self.show(y), self.kind_name(g));
                self.mismatch(r, &e, f)
            }
            (Ty::Lit(g1), Ty::Lit(g2)) => {
                if self.grps[g1 as usize].float != self.grps[g2 as usize].float {
                    let (e, f) = (self.kind_name(g1), self.kind_name(g2));
                    return self.mismatch(r, e, f);
                }
                let (lo, hi) = if g1 < g2 { (g1, g2) } else { (g2, g1) };
                self.grps[hi as usize].parent = lo;
                Some(Ty::Lit(lo))
            }
        }
    }

    /// E0202 at the first literal of pending group `g`: nothing gives it a type.
    fn no_type<T>(&mut self, g: u32) -> Option<T> {
        let icx = self.te.icx;
        let n = self.grps[g as usize].first;
        let text = icx.node_text(n).to_string();
        let case = if self.grps[g as usize].float { Some("float") } else { None };
        let d = mk("E0202", case, &[("literal", &text)], icx.node_span(n), &[]);
        self.defer(d)
    }

    /// The type of node `n` of a closed context (a cast operand, a tuple element without an
    /// expected type): a pending group has no type (E0202).
    fn force(&mut self, t: Ty) -> Option<TypeId> {
        match self.resolve(t) {
            Ty::T(t) => Some(t),
            Ty::Lit(g) => self.no_type(g),
        }
    }

    // ---------- evaluation ----------

    /// Decides the type of every node of `e` (expected type `want`), checks the rules of its
    /// literals and operators, then evaluates it.
    fn eval(&mut self, e: u32, want: Option<TypeId>) -> R {
        let (g0, l0, r0) = (self.grps.len(), self.lit_order.len(), self.rules.len());
        let t = self.decide(e, want)?;
        match want {
            Some(w) => self.coerce(e, t, w)?,
            None => self.force(t)?,
        };
        self.settle(g0, l0, r0)?;
        self.value(e)
    }

    /// The checks of the groups, literals, and rules decided since the given marks, once every
    /// group has its type: a group without one is E0202, a literal outside its type E0203, an
    /// operator on a type without it E0206 (a user operator: E0904, a call).
    fn settle(&mut self, g0: usize, l0: usize, r0: usize) -> Option<()> {
        for g in g0..self.grps.len() {
            let r = self.find(g as u32);
            if self.grps[r as usize].fixed.is_none() {
                return self.no_type(r);
            }
        }
        for i in l0..self.lit_order.len() {
            let n = self.lit_order[i];
            let t = self.ty_of(n)?;
            let l = self.lits[&n];
            self.range_check(n, l, t)?;
        }
        for i in r0..self.rules.len() {
            let (n, rule) = self.rules[i];
            self.rule_check(n, rule)?;
        }
        Some(())
    }

    fn range_check(&mut self, n: u32, l: Lit, t: TypeId) -> Option<()> {
        let icx = self.te.icx;
        if l.float {
            if self.env.is_float(t) {
                let f32_ = matches!(self.env.kind(t), TypeKind::Float(FloatBits::F32));
                if !l.fval.is_finite() || f32_ && !(l.fval as f32).is_finite() {
                    let (text, tn) = (icx.node_text(n).to_string(), self.show(t));
                    let d = mk(
                        "E0203",
                        Some("float"),
                        &[("literal", &text), ("type", &tn)],
                        icx.node_span(n),
                        &[],
                    );
                    return self.defer(d);
                }
            }
            return Some(());
        }
        if self.env.is_float(t) {
            return Some(());
        }
        let Some((lo, hi)) = self.int_range(t) else { return Some(()) };
        let v = if l.mag > i128::MAX as u128 {
            None
        } else {
            Some(if l.neg { -(l.mag as i128) } else { l.mag as i128 })
        };
        if v.is_some_and(|v| v >= lo && v <= hi) {
            return Some(());
        }
        let (text, tn) = (icx.node_text(n).to_string(), self.show(t));
        let d = mk(
            "E0203",
            None,
            &[("literal", &text), ("type", &tn), ("min", &lo.to_string()), ("max", &hi.to_string())],
            icx.node_span(n),
            &[],
        );
        self.defer(d)
    }

    fn rule_check(&mut self, n: u32, rule: Rule) -> Option<()> {
        let (tok, ty, ok) = match rule {
            Rule::Bin { tok, ty } => {
                let op = self.te.icx.tk(tok);
                let t = self.rty(n, ty)?;
                let ok = match self.env.kind(t) {
                    TypeKind::Int { .. } => true,
                    TypeKind::Float(_) => [Term::Plus, Term::Minus, Term::Star, Term::Slash, Term::Percent]
                        .iter()
                        .any(|&o| op == o as u8),
                    TypeKind::Bool => [Term::Amp, Term::Pipe, Term::Caret].iter().any(|&o| op == o as u8),
                    _ => false,
                };
                (tok, t, ok)
            }
            Rule::Un { tok, ty } => {
                let t = self.rty(n, ty)?;
                let minus = self.te.icx.is(tok, Term::Minus);
                let ok = match self.env.kind(t) {
                    TypeKind::Int { signed, .. } => *signed || !minus,
                    TypeKind::Float(_) => minus,
                    _ => false,
                };
                (tok, t, ok)
            }
            Rule::Cmp { tok, ty } => {
                let t = self.rty(n, ty)?;
                let ok = matches!(
                    self.env.kind(t),
                    TypeKind::Int { .. } | TypeKind::Float(_) | TypeKind::Bool | TypeKind::Char
                );
                (tok, t, ok)
            }
        };
        if ok {
            return Some(());
        }
        if self.is_err(ty) {
            return self.explained();
        }
        if matches!(self.env.kind(ty), TypeKind::Adt { .. }) {
            let d = self.e0904_at(n, "a call");
            return self.defer(d);
        }
        let icx = self.te.icx;
        let (op, tn) = (icx.tok_str(tok).to_string(), self.show(ty));
        let d = mk("E0206", None, &[("op", &op), ("type", &tn)], icx.tok_span(tok), &[]);
        self.defer(d)
    }

    /// A decided type that must be fixed by now.
    fn rty(&mut self, n: u32, t: Ty) -> Option<TypeId> {
        match self.resolve(t) {
            Ty::T(t) => Some(t),
            Ty::Lit(_) => self.ice(n, "an operand has no type after typing"),
        }
    }

    /// The decided type of node `n`.
    fn ty_of(&mut self, n: u32) -> Option<TypeId> {
        match self.nty.get(&n).copied() {
            Some(t) => self.rty(n, t),
            None => self.ice(n, "the node was not typed"),
        }
    }

    /// The decided types of the literal and operator nodes of a primitive scalar type.
    fn scalar_types(&mut self) -> Vec<(u32, TypeId)> {
        let mut nodes: Vec<(u32, Ty)> = self.nty.iter().map(|(&n, &t)| (n, t)).collect();
        nodes.sort_unstable_by_key(|x| x.0);
        let mut out = Vec::new();
        for (n, t) in nodes {
            if let Ty::T(t) = self.resolve(t)
                && matches!(
                    self.env.kind(t),
                    TypeKind::Int { .. } | TypeKind::Float(_) | TypeKind::Bool | TypeKind::Char
                )
            {
                out.push((n, t));
            }
        }
        out
    }

    // ---------- typing ----------

    fn decide(&mut self, n: u32, hint: Option<TypeId>) -> Option<Ty> {
        if let Some(&t) = self.nty.get(&n) {
            return Some(t);
        }
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            let d = mk("E0904", Some("depth"), &[], self.te.icx.node_span(n), &[]);
            self.report(d);
            return None;
        }
        let before = self.nrep;
        let r = self.decide_inner(n, hint);
        self.depth -= 1;
        match r {
            Some(t) => {
                self.nty.insert(n, t);
            }
            None if self.nrep == before && self.stuck.is_none() => self.stuck = self.e0904_at(n, OUTSIDE),
            None => {}
        }
        r
    }

    fn decide_inner(&mut self, n: u32, hint: Option<TypeId>) -> Option<Ty> {
        let icx = self.te.icx;
        let kids = icx.kids_vec(n);
        let k = icx.kind(n)?;
        let bool_t = self.fx(ids::BOOL);
        match k {
            NodeKind::IfExpr => {
                let (&c, &a, &b) = (kids.first()?, kids.get(1)?, kids.get(2)?);
                let ct = self.decide(c, Some(bool_t))?;
                self.coerce(c, ct, bool_t)?;
                let at = self.decide(a, hint)?;
                let bh = self.hint_of(at, hint);
                let bt = self.decide(b, bh)?;
                self.unify(b, at, bt)
            }
            NodeKind::MatchExpr => self.e0904(n, "a non-const item"),
            NodeKind::RangeExpr => {
                if kids.len() > 1 {
                    return self.e0904(n, "a non-const item");
                }
                self.decide(kids[0], hint)
            }
            NodeKind::OrExpr | NodeKind::AndExpr => {
                if kids.len() == 1 {
                    return self.decide(kids[0], hint);
                }
                for &c in &kids {
                    let t = self.decide(c, Some(bool_t))?;
                    self.coerce(c, t, bool_t)?;
                }
                Some(Ty::T(bool_t))
            }
            NodeKind::CmpExpr => {
                if kids.len() == 1 {
                    return self.decide(kids[0], hint);
                }
                let (&l, &opn, &r) = (kids.first()?, kids.get(1)?, kids.get(2)?);
                let lt = self.decide(l, None)?;
                let lh = self.hint_of(lt, None);
                let rt = self.decide(r, lh)?;
                let t = self.unify(r, lt, rt)?;
                self.rules.push((n, Rule::Cmp { tok: icx.a(opn), ty: t }));
                Some(Ty::T(bool_t))
            }
            NodeKind::BitOr
            | NodeKind::BitXor
            | NodeKind::BitAnd
            | NodeKind::Shift
            | NodeKind::Add
            | NodeKind::Mul => {
                if kids.len() == 1 {
                    return self.decide(kids[0], hint);
                }
                // every operand has the chain's type: a literal takes the other operand's
                let mut acc = self.decide(kids[0], hint)?;
                let mut toks = Vec::with_capacity(kids.len() - 1);
                for &c in &kids[1..] {
                    toks.push(icx.a(c).saturating_sub(1));
                    let h = self.hint_of(acc, hint);
                    let rt = self.decide(c, h)?;
                    acc = self.unify(c, acc, rt)?;
                }
                for tok in toks {
                    self.rules.push((n, Rule::Bin { tok, ty: acc }));
                }
                Some(acc)
            }
            NodeKind::Cast => {
                if kids.len() == 1 {
                    return self.decide(kids[0], hint);
                }
                let t0 = self.decide(kids[0], None)?;
                let mut from = self.force(t0)?;
                let mut steps = Vec::with_capacity(kids.len() - 1);
                for &c in &kids[1..] {
                    let optok = icx.a(c).saturating_sub(1);
                    let op = icx.tk(optok);
                    let to = lower_type(self.env, self.te, c, false);
                    if self.is_err(to) || self.is_err(from) {
                        return self.explained();
                    }
                    if !self.castable(op, from, to) {
                        let (f, t) = (self.show(from), self.show(to));
                        let sp = icx.range_span(optok, icx.end_tok(c));
                        let d = mk("E0204", None, &[("from", &f), ("to", &t)], sp, &[]);
                        return self.defer(d);
                    }
                    steps.push((op, to));
                    from = if op == Term::AsQuestion as u8 { self.option_of(to)? } else { to };
                }
                self.casts.insert(n, steps);
                Some(Ty::T(from))
            }
            NodeKind::Unary => {
                let a = icx.a(n);
                let t = icx.tk(a);
                let &c = kids.first()?;
                if t == Term::Minus as u8 {
                    // `-` directly on a literal is part of the literal
                    if let Some(p) = down_to_primary(&icx, c)
                        && icx.a(p) == a + 1
                        && (icx.is(a + 1, Term::IntLit) || icx.is(a + 1, Term::FloatLit))
                    {
                        let r = self.literal(n, a + 1, true, hint)?;
                        let mut cur = c;
                        while cur != p {
                            self.nty.insert(cur, r);
                            cur = icx.kids_vec(cur)[0];
                        }
                        self.nty.insert(p, r);
                        return Some(r);
                    }
                    let ct = self.decide(c, hint)?;
                    self.rules.push((n, Rule::Un { tok: a, ty: ct }));
                    return Some(ct);
                }
                if t == Term::Bang as u8 {
                    let ct = self.decide(c, Some(bool_t))?;
                    self.coerce(c, ct, bool_t)?;
                    return Some(Ty::T(bool_t));
                }
                if t == Term::Tilde as u8 {
                    let ct = self.decide(c, hint)?;
                    self.rules.push((n, Rule::Un { tok: a, ty: ct }));
                    return Some(ct);
                }
                self.e0904(n, "a non-const item")
            }
            NodeKind::Postfix => self.decide_postfix(n, &kids, hint),
            // a bare name is a path with no steps
            NodeKind::Primary if icx.is_path_head(n) => self.decide_postfix(n, &[n], hint),
            NodeKind::Primary => self.decide_primary(n, &kids, hint),
            // a const named in a generic argument (`sync.Mutex<u32, RANK>`): its last path step
            NodeKind::TypeBase => {
                let res = icx.uses(n).last()?.res;
                let mut steps = Vec::new();
                let t = self.head_path(n, res, hint, &mut steps)?;
                self.steps.insert(n, steps);
                Some(Ty::T(t))
            }
            _ => None,
        }
    }

    /// Whether the evaluator casts a value of `from` to `to` with operator `op`: the
    /// checker's rule (`expr::scalar_cast_ok`) on scalar types.
    fn castable(&mut self, op: u8, from: TypeId, to: TypeId) -> bool {
        let (fk, tk) = (self.env.kind(from).clone(), self.env.kind(to).clone());
        let scalar = |k: &TypeKind| {
            matches!(k, TypeKind::Int { .. } | TypeKind::Float(_) | TypeKind::Bool | TypeKind::Char)
        };
        if !scalar(&fk) || !scalar(&tk) {
            return false;
        }
        from == to || crate::expr::scalar_cast_ok(op, &fk, &tk)
    }

    /// A literal at token `tok` (node `n`), `neg`: a folded unary minus.
    fn literal(&mut self, n: u32, tok: u32, neg: bool, hint: Option<TypeId>) -> Option<Ty> {
        let icx = self.te.icx;
        let text = icx.tok_str(tok);
        let malformed = |ev: &mut Self| {
            let t = icx.node_text(n).to_string();
            let d = mk("E0203", Some("malformed"), &[("literal", &t)], icx.node_span(n), &[]);
            ev.defer(d)
        };
        let (lit, suffix) = if icx.is(tok, Term::IntLit) {
            let Some((mag, suf)) = literals::parse_int(text) else { return malformed(self) };
            (Lit { mag, fval: mag as f64, neg, float: false }, suf)
        } else {
            let Some((v, suf)) = literals::parse_float(text) else { return malformed(self) };
            (Lit { mag: 0, fval: v, neg, float: true }, suf)
        };
        let mut lit = lit;
        let t = match suffix {
            Suffix::Int(_) if lit.float => return malformed(self),
            Suffix::Int(t) => Ty::T(self.fx(t)),
            Suffix::Float(t) => {
                lit.float = true;
                Ty::T(self.fx(t))
            }
            Suffix::None => {
                let g = self.grps.len() as u32;
                self.grps.push(Grp { parent: g, float: lit.float, fixed: None, first: n });
                if let Some(h) = hint {
                    let _ = self.fix(g, h);
                }
                Ty::Lit(g)
            }
        };
        self.lits.insert(n, lit);
        self.lit_order.push(n);
        Some(t)
    }

    fn decide_primary(&mut self, n: u32, kids: &[u32], hint: Option<TypeId>) -> Option<Ty> {
        let icx = self.te.icx;
        let a = icx.a(n);
        let t = icx.tk(a);
        if t == Term::IntLit as u8 || t == Term::FloatLit as u8 {
            return self.literal(n, a, false, hint);
        }
        if t == Term::KwTrue as u8 || t == Term::KwFalse as u8 {
            return Some(Ty::T(self.fx(ids::BOOL)));
        }
        if t == Term::CharLit as u8 {
            return Some(Ty::T(self.fx(ids::CHAR)));
        }
        if t == Term::StrLit as u8 {
            return Some(Ty::T(self.fx(ids::REF_STR)));
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
                return self.decide(kids[0], hint);
            }
            let wants: Vec<Option<TypeId>> = match hint.map(|w| self.env.kind(w).clone()) {
                Some(TypeKind::Tuple(ts)) if ts.len() == kids.len() => ts.iter().map(|&x| Some(x)).collect(),
                _ => vec![None; kids.len()],
            };
            let mut tys = Vec::with_capacity(kids.len());
            for (i, &c) in kids.iter().enumerate() {
                let ct = self.decide(c, wants[i])?;
                tys.push(match wants[i] {
                    Some(w) => self.coerce(c, ct, w)?,
                    None => self.force(ct)?,
                });
            }
            let tt = self.env.mk_tuple(tys);
            if !self.size_ok(n, tt) {
                return None;
            }
            return Some(Ty::T(tt));
        }
        if t == Term::LBracket as u8 {
            let eh = match hint.map(|w| self.env.kind(w).clone()) {
                Some(TypeKind::Array { elem, .. }) => Some(elem),
                _ => None,
            };
            let (elem, len) = if kids.len() == 2 && icx.is(icx.end_tok(kids[0]) + 1, Term::Semi) {
                // `[e; n]`
                let et = self.decide(kids[0], eh)?;
                let elem = match (self.resolve(et), eh) {
                    (Ty::Lit(_), Some(h)) => self.coerce(kids[0], et, h)?,
                    (t, _) => self.force(t)?,
                };
                let us = self.fx(ids::USIZE);
                let (_, lv) = self.eval(kids[1], Some(us))?;
                let Val::Int(len) = lv else { return self.ice(kids[1], "a length is not an integer") };
                (elem, u64::try_from(len).ok()?)
            } else if kids.is_empty() {
                return match hint.map(|h| (h, self.env.kind(h).clone())) {
                    Some((h, TypeKind::Array { len: ConstArg::Value(0), .. })) => Some(Ty::T(h)),
                    _ => {
                        let text = icx.node_text(n).to_string();
                        let d = mk("E0202", Some("infer"), &[("literal", &text)], icx.node_span(n), &[]);
                        self.defer(d)
                    }
                };
            } else {
                let mut et = eh;
                let mut tys = Vec::with_capacity(kids.len());
                for &c in kids {
                    let ct = self.decide(c, et)?;
                    if et.is_none()
                        && let Ty::T(x) = self.resolve(ct)
                    {
                        et = Some(x);
                    }
                    tys.push((c, ct));
                }
                let elem = match et {
                    Some(e) => e,
                    None => {
                        let mut acc = tys[0].1;
                        for &(c, ct) in &tys[1..] {
                            acc = self.unify(c, acc, ct)?;
                        }
                        self.force(acc)?
                    }
                };
                for &(c, ct) in &tys {
                    self.coerce(c, ct, elem)?;
                }
                (elem, kids.len() as u64)
            };
            let at = self.env.intern(TypeKind::Array { elem, len: ConstArg::Value(len) });
            if !self.size_ok(n, at) {
                return None;
            }
            return Some(Ty::T(at));
        }
        if t == Term::Ident as u8 {
            // A single name: a const, a unit variant (resolved in the chain), a generic const.
            let u = icx.uses(n).first()?;
            let mut steps = Vec::new();
            let ty = self.head_path(n, u.res, hint, &mut steps)?;
            self.steps.insert(n, steps);
            return Some(Ty::T(ty));
        }
        None
    }

    /// The limit on compile-time data ([`data_size`]) on the type of a value const evaluation
    /// builds at `n`: an array, tuple, struct, or variant literal, or a const ctor call. Checked
    /// on the type before the value is built, so no evaluation holds more data than the limit.
    fn size_ok(&mut self, n: u32, at: TypeId) -> bool {
        match data_size(self.env, at) {
            Ok(()) => true,
            Err(bytes) => {
                let tn = self.show(at);
                let bytes = bytes.to_string();
                let d = mk(
                    "E0904",
                    Some("size"),
                    &[("type", &tn), ("bytes", &bytes)],
                    self.te.icx.node_span(n),
                    &[],
                );
                let _: Option<()> = self.defer(d);
                false
            }
        }
    }

    /// The head of a path: pushes its step, returns its type.
    fn head_path(&mut self, n: u32, res: Res, want: Option<TypeId>, steps: &mut Vec<Step>) -> Option<TypeId> {
        match res {
            Res::Decl(d) => {
                let decl = *self.env.prog().decl(d)?;
                match decl.kind {
                    // a const without a value has its own error
                    DeclKind::Const => match self.env.const_value(d) {
                        Some((t, v)) => {
                            steps.push(Step::Known(v));
                            Some(t)
                        }
                        None => self.explained(),
                    },
                    DeclKind::Variant => self.decide_variant(d, &[], want, None, steps),
                    DeclKind::Error => self.explained(),
                    _ => self.e0904(n, "a non-const item"),
                }
            }
            Res::Unresolved => self.explained(),
            Res::Local(b) => match self.params.iter().find(|p| p.0 == b.node) {
                Some(p) => {
                    let (t, v) = p.1.clone();
                    steps.push(Step::Known(v));
                    Some(t)
                }
                None => self.e0904(n, "a non-const item"),
            },
            _ => self.e0904(n, "a non-const item"),
        }
    }

    fn decide_variant(
        &mut self,
        v: DeclId,
        args: &[u32],
        want: Option<TypeId>,
        targs: Option<u32>,
        steps: &mut Vec<Step>,
    ) -> Option<TypeId> {
        let decl = *self.env.prog().decl(v)?;
        let adt = self.env.adt_def(decl.parent)?;
        let idx = adt.variants.iter().position(|x| x.decl == v)?;
        let vd = adt.variants[idx].clone();
        let gargs = self.adt_args_from(adt.decl, want, adt.generics.len(), targs)?;
        if vd.fields.len() != args.len() {
            return None;
        }
        for (f, &a) in vd.fields.iter().zip(args) {
            let ft = self.env.subst(f.ty, &gargs, None);
            let at = self.decide(a, Some(ft))?;
            self.coerce(a, at, ft)?;
        }
        steps.push(Step::Variant(idx as u32, args.to_vec()));
        Some(self.env.mk_adt(adt.decl, gargs))
    }

    /// The type args of ADT `d` from the expected type (consts have declared types).
    /// The type args of ADT `d`: those of the expected type, else its turbofish `targs`
    /// (`W::<u16>.{ .. }` with nothing expected) when they are concrete. The expected type comes
    /// first: in a const ctor's body evaluated for a call it is the instance, while the
    /// turbofish names the body's own params.
    fn adt_args_from(
        &mut self,
        d: DeclId,
        want: Option<TypeId>,
        count: usize,
        targs: Option<u32>,
    ) -> Option<Vec<GArg>> {
        if count == 0 {
            return Some(Vec::new());
        }
        if let Some(TypeKind::Adt { def, args }) = want.map(|w| self.env.kind(w).clone())
            && def == d
        {
            return Some(args.to_vec());
        }
        let ga = targs?;
        let icx = self.te.icx;
        let params: Vec<Option<TypeId>> =
            self.env.adt_def(d)?.generics.iter().map(|g| g.is_const.then_some(g.const_ty)).collect();
        let nodes = icx.kids_vec(ga);
        if nodes.len() != count || params.len() != count {
            return None;
        }
        let args: Vec<GArg> =
            nodes.iter().zip(&params).map(|(&a, &c)| lower_garg(self.env, self.te, a, c)).collect();
        let concrete = args.iter().all(|a| match a {
            GArg::Type(t) => !self.env.has_params(*t) && !self.is_err(*t),
            GArg::Const(c) => matches!(c, ConstArg::Value(_)),
        });
        concrete.then_some(args)
    }

    fn decide_postfix(&mut self, n: u32, kids: &[u32], want: Option<TypeId>) -> Option<Ty> {
        let icx = self.te.icx;
        let prim = *kids.first()?;
        let pa = icx.a(prim);
        let mut kid = 1usize;
        // Builtins: `size_of::<T>()` / `align_of::<T>()`.
        if icx.is(pa, Term::Ident)
            && let Some(u) = icx.uses(prim).first()
            && let Res::Builtin(b) = u.res
            && matches!(b, Builtin::SizeOf | Builtin::AlignOf)
        {
            let pk = icx.kids_vec(prim);
            let ga = pk.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::GenericArgs))?;
            let args = icx.kids_vec(ga);
            if args.len() != 1 {
                return None;
            }
            let GArg::Type(t) = lower_garg(self.env, self.te, args[0], None) else { return None };
            if self.is_err(t) {
                return self.explained();
            }
            if self.env.has_params(t) {
                // as written: a type parameter has no name of its own here
                let tn = icx.node_spelled(args[0]);
                let f = if b == Builtin::SizeOf { "size_of" } else { "align_of" };
                let d = mk("E0904", Some("generic"), &[("name", f), ("type", &tn)], icx.node_span(n), &[]);
                return self.defer(d);
            }
            let Ok((size, align)) = self.env.layout(t) else {
                let tn = self.show(t);
                let d = mk("E0911", None, &[("type", &tn)], icx.node_span(n), &[]);
                self.report(d);
                return None;
            };
            let v = if b == Builtin::SizeOf { size } else { align };
            let us = self.fx(ids::USIZE);
            self.steps.insert(n, vec![Step::Known(Val::Int(v as i128))]);
            return Some(Ty::T(us));
        }
        // The head chain: uses keyed at the Primary.
        let mut p = icx.primary_end(prim) + 1;
        let mut steps: Vec<Step> = Vec::new();
        let mut cur: Option<TypeId> = None;
        let mut path: Option<Res> = None;
        // the turbofish of the current path segment (`W::<u16>.{ .. }`): the ADT's type args
        let mut targs: Option<u32> = None;
        if icx.is(pa, Term::Ident) {
            path = icx.use_at(prim, pa).map(|u| u.res);
            targs = icx.kids_vec(prim).iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::GenericArgs));
        } else {
            let t = self.decide(prim, None)?;
            cur = Some(self.force(t)?);
            steps.push(Step::Node(prim));
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
                        targs =
                            kids.get(kid).copied().filter(|&g| icx.kind(g) == Some(NodeKind::GenericArgs));
                        kid += 1;
                    }
                    continue;
                }
                // the receiver first: a const it names is evaluated (a cycle is found there)
                let base = match cur {
                    Some(c) => c,
                    None => self.head_path(prim, path.take()?, None, &mut steps)?,
                };
                // a method call on a value (`G.len()`, `x.select(a, b)`)
                if icx.is(p + 2, Term::LParen) || icx.is(p + 2, Term::ColonColon) {
                    let d = self.e0904_at(n, "a call");
                    return self.defer(d);
                }
                // a field of a struct value
                let (inner, _) = self.env.peel_refs(base);
                let TypeKind::Adt { def, args } = self.env.kind(inner).clone() else { return None };
                let adt = self.env.adt_def(def)?;
                if adt.is_enum {
                    return None;
                }
                let vd = adt.variants.first()?;
                let name = icx.sym(seg);
                let i = vd.fields.iter().position(|f| f.name == name)?;
                let ft = self.env.subst(vd.fields[i].ty, &args, None);
                steps.push(Step::Part(i));
                cur = Some(ft);
                p += 2;
                continue;
            }
            if t == Term::Dot as u8 && icx.is(p + 1, Term::IntLit) {
                let base = match cur {
                    Some(c) => c,
                    None => self.head_path(prim, path.take()?, None, &mut steps)?,
                };
                let i: usize = icx.tok_str(p + 1).parse().ok()?;
                let TypeKind::Tuple(ts) = self.env.kind(base).clone() else { return None };
                cur = Some(*ts.get(i)?);
                steps.push(Step::Part(i));
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
                    let vt = self.decide_variant(d, &args, want, targs, &mut steps)?;
                    if !self.size_ok(n, vt) {
                        return None;
                    }
                    cur = Some(vt);
                    p = close + 1;
                    continue;
                }
                if cur.is_none()
                    && let Some(Res::Decl(d)) = path
                    && self.env.lang().const_ctors.contains(&d)
                {
                    path = None;
                    let ct = self.decide_ctor(d, &args, want, &mut steps)?;
                    if !self.size_ok(n, ct) {
                        return None;
                    }
                    cur = Some(ct);
                    p = close + 1;
                    continue;
                }
                return self.e0904(n, "a call");
            }
            if t == Term::DotLBrace as u8 {
                let sl = *kids.get(kid)?;
                kid += 1;
                let Some(Res::Decl(d)) = path.take() else { return self.e0904(n, "a non-const item") };
                let st = self.decide_struct(d, sl, want, targs, &mut steps)?;
                if !self.size_ok(n, st) {
                    return None;
                }
                cur = Some(st);
                p = icx.close(p) + 1;
                continue;
            }
            if t == Term::LBracket as u8 {
                let ie = *kids.get(kid)?;
                kid += 1;
                let base = match cur {
                    Some(c) => c,
                    None => self.head_path(prim, path.take()?, None, &mut steps)?,
                };
                let TypeKind::Array { elem, .. } = self.env.kind(base).clone() else { return None };
                let us = self.fx(ids::USIZE);
                let it = self.decide(ie, Some(us))?;
                self.coerce(ie, it, us)?;
                steps.push(Step::Index(ie, base));
                cur = Some(elem);
                p = icx.close(p) + 1;
                continue;
            }
            if t == Term::Question as u8 {
                return self.e0904(n, "a non-const item");
            }
            break;
        }
        let ty = match cur {
            Some(c) => c,
            None => self.head_path(prim, path?, want, &mut steps)?,
        };
        self.steps.insert(n, steps);
        Some(Ty::T(ty))
    }

    /// A call of the constructor `m` ([`LangItems::const_ctors`]): typed as the struct literal
    /// its body returns; evaluated with its parameters bound to the argument values.
    fn decide_ctor(
        &mut self,
        m: DeclId,
        args: &[u32],
        want: Option<TypeId>,
        steps: &mut Vec<Step>,
    ) -> Option<TypeId> {
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
        for (&a, &pt) in args.iter().zip(&ptys) {
            let pw = match &gargs {
                Some(g) => Some(self.env.subst(pt, g, None)),
                None if !self.env.has_params(pt) => Some(pt),
                None => None,
            };
            let at = self.decide(a, pw)?;
            match pw {
                Some(w) => self.coerce(a, at, w)?,
                None => self.force(at)?,
            };
        }
        let ty = match want {
            Some(w) if gargs.is_some() => w,
            _ if !self.env.has_params(ret) => ret,
            _ => return None,
        };
        steps.push(Step::Ctor(m, args.to_vec()));
        Some(ty)
    }

    fn decide_struct(
        &mut self,
        d: DeclId,
        sl: u32,
        want: Option<TypeId>,
        targs: Option<u32>,
        steps: &mut Vec<Step>,
    ) -> Option<TypeId> {
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
        let gargs = self.adt_args_from(adt_decl, want, adt.generics.len(), targs)?;
        let mut fields = Vec::new();
        for e in icx.kids_vec(sl) {
            let name_tok = icx.a(e).checked_sub(2)?;
            let name = icx.sym(name_tok);
            let fi = vd.fields.iter().position(|f| f.name == name)?;
            let ft = self.env.subst(vd.fields[fi].ty, &gargs, None);
            let et = self.decide(e, Some(ft))?;
            self.coerce(e, et, ft)?;
            fields.push((fi, e));
        }
        if (0..vd.fields.len()).any(|i| !fields.iter().any(|f| f.0 == i)) {
            return None;
        }
        steps.push(Step::Struct { idx: vi as u32, is_enum: adt.is_enum, n: vd.fields.len(), fields });
        Some(self.env.mk_adt(adt_decl, gargs))
    }

    fn option_of(&mut self, inner: TypeId) -> Option<TypeId> {
        let opt = self.env.lang().option;
        if opt.is_none() {
            return None;
        }
        Some(self.env.mk_adt(opt, vec![GArg::Type(inner)]))
    }

    // ---------- values ----------

    /// The value of node `n`, every type decided. Lazy where the run time is: only the taken
    /// arm of an `if`, and `&&` / `||` stop at the first operand that decides them, so a trap
    /// is reported exactly when the same expression traps at run time.
    fn value(&mut self, n: u32) -> R {
        let icx = self.te.icx;
        let kids = icx.kids_vec(n);
        let k = icx.kind(n)?;
        let bool_t = self.fx(ids::BOOL);
        if let Some(&l) = self.lits.get(&n) {
            let t = self.ty_of(n)?;
            let v = if self.env.is_float(t) {
                let f = if l.neg { -l.fval } else { l.fval };
                Val::Float(round_float(f, self.env.kind(t)))
            } else {
                Val::Int(if l.neg { -(l.mag as i128) } else { l.mag as i128 })
            };
            return Some((t, v));
        }
        match k {
            NodeKind::IfExpr => {
                let (&c, &a, &b) = (kids.first()?, kids.get(1)?, kids.get(2)?);
                let (_, cv) = self.value(c)?;
                let (_, v) = self.value(if cv == Val::Bool(true) { a } else { b })?;
                Some((self.ty_of(n)?, v))
            }
            NodeKind::RangeExpr => self.value(*kids.first()?),
            NodeKind::OrExpr | NodeKind::AndExpr => {
                if kids.len() == 1 {
                    return self.value(kids[0]);
                }
                let is_or = k == NodeKind::OrExpr;
                for &c in &kids {
                    let (_, v) = self.value(c)?;
                    if (v == Val::Bool(true)) == is_or {
                        return Some((bool_t, Val::Bool(is_or)));
                    }
                }
                Some((bool_t, Val::Bool(!is_or)))
            }
            NodeKind::CmpExpr => {
                if kids.len() == 1 {
                    return self.value(kids[0]);
                }
                let op = icx.tk(icx.a(*kids.get(1)?));
                let (_, lv) = self.value(kids[0])?;
                let (_, rv) = self.value(*kids.get(2)?)?;
                let ord = match (&lv, &rv) {
                    (Val::Int(a), Val::Int(b)) => a.partial_cmp(b),
                    (Val::Float(a), Val::Float(b)) => a.partial_cmp(b),
                    (Val::Bool(a), Val::Bool(b)) => a.partial_cmp(b),
                    (Val::Char(a), Val::Char(b)) => a.partial_cmp(b),
                    _ => return self.ice(n, "a comparison of values that are not scalars"),
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
                Some((bool_t, Val::Bool(b)))
            }
            NodeKind::BitOr
            | NodeKind::BitXor
            | NodeKind::BitAnd
            | NodeKind::Shift
            | NodeKind::Add
            | NodeKind::Mul => {
                if kids.len() == 1 {
                    return self.value(kids[0]);
                }
                let t = self.ty_of(n)?;
                let (_, mut acc) = self.value(kids[0])?;
                for &c in &kids[1..] {
                    let op = icx.tk(icx.a(c).saturating_sub(1));
                    let (_, b) = self.value(c)?;
                    acc = self.arith(kids[0], c, op, t, acc, b)?;
                }
                Some((t, acc))
            }
            NodeKind::Cast => {
                if kids.len() == 1 {
                    return self.value(kids[0]);
                }
                let mut acc = self.value(kids[0])?;
                let steps = self.casts.remove(&n)?;
                for (op, to) in steps {
                    acc = self.cast(n, op, acc, to)?;
                }
                Some(acc)
            }
            NodeKind::Unary => {
                let a = icx.a(n);
                let t = icx.tk(a);
                let (ct, cv) = self.value(*kids.first()?)?;
                if t == Term::Minus as u8 {
                    return match cv {
                        Val::Int(i) => {
                            if self.int_range(ct).is_some_and(|(lo, _)| i == lo) {
                                return self.overflow(n, n, ct);
                            }
                            Some((ct, Val::Int(-i)))
                        }
                        Val::Float(f) => Some((ct, Val::Float(-f))),
                        _ => self.ice(n, "`-` of a value that is not a number"),
                    };
                }
                if t == Term::Bang as u8 {
                    return match cv {
                        Val::Bool(b) => Some((ct, Val::Bool(!b))),
                        _ => self.ice(n, "`!` of a value that is not a bool"),
                    };
                }
                match (cv, self.env.int(ct)) {
                    (Val::Int(i), Some((signed, bits))) => {
                        Some((ct, Val::Int(wrap_int(!i, signed, bits.bits()))))
                    }
                    _ => self.ice(n, "`~` of a value that is not an integer"),
                }
            }
            NodeKind::Postfix | NodeKind::TypeBase => self.run_steps(n),
            NodeKind::Primary if icx.is_path_head(n) => self.run_steps(n),
            NodeKind::Primary => self.value_primary(n, &kids),
            _ => self.ice(n, "a form without a value"),
        }
    }

    fn value_primary(&mut self, n: u32, kids: &[u32]) -> R {
        let icx = self.te.icx;
        let a = icx.a(n);
        let t = icx.tk(a);
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
        if t == Term::Ident as u8 {
            return self.run_steps(n);
        }
        let ty = self.ty_of(n)?;
        if t == Term::LParen as u8 {
            let close = icx.close(a);
            if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
                return self.value(kids[0]);
            }
            let mut vals = Vec::with_capacity(kids.len());
            for &c in kids {
                vals.push(self.value(c)?.1);
            }
            return Some((ty, Val::Agg(vals.into())));
        }
        if t == Term::LBracket as u8 {
            if kids.len() == 2 && icx.is(icx.end_tok(kids[0]) + 1, Term::Semi) {
                let (_, v) = self.value(kids[0])?;
                let TypeKind::Array { len: ConstArg::Value(len), .. } = self.env.kind(ty).clone() else {
                    return self.ice(n, "an array literal without a length");
                };
                return Some((ty, Val::Rep(Arc::new(v), len)));
            }
            let mut vals = Vec::with_capacity(kids.len());
            for &c in kids {
                vals.push(self.value(c)?.1);
            }
            return Some((ty, Val::Agg(vals.into())));
        }
        self.ice(n, "a form without a value")
    }

    fn run_steps(&mut self, n: u32) -> R {
        let ty = self.ty_of(n)?;
        let Some(steps) = self.steps.remove(&n) else { return self.ice(n, "a path without its steps") };
        let mut cur: Option<Val> = None;
        for s in steps {
            cur = Some(match s {
                Step::Known(v) => v,
                Step::Node(p) => self.value(p)?.1,
                Step::Part(i) => match cur? {
                    Val::Agg(vs) if i < vs.len() => vs[i].clone(),
                    _ => return self.ice(n, "a field of a value without it"),
                },
                Step::Index(ie, at) => {
                    let (_, iv) = self.value(ie)?;
                    let Val::Int(i) = iv else { return self.ice(ie, "an index that is not an integer") };
                    match cur? {
                        Val::Agg(vs) => match usize::try_from(i).ok().filter(|&i| i < vs.len()) {
                            Some(i) => vs[i].clone(),
                            None => return self.index_range(ie, at, vs.len()),
                        },
                        Val::Rep(v, len) => match u64::try_from(i).ok().filter(|&i| i < len) {
                            Some(_) => Arc::unwrap_or_clone(v),
                            None => {
                                return self.index_range(ie, at, usize::try_from(len).unwrap_or(usize::MAX));
                            }
                        },
                        _ => return self.ice(n, "an index into a value that is not an array"),
                    }
                }
                Step::Variant(idx, args) => {
                    let mut vals = Vec::with_capacity(args.len());
                    for a in args {
                        vals.push(self.value(a)?.1);
                    }
                    Val::Variant(idx, vals.into())
                }
                Step::Struct { idx, is_enum, n: count, fields } => {
                    let mut vals: Vec<Option<Val>> = vec![None; count];
                    for (fi, e) in fields {
                        vals[fi] = Some(self.value(e)?.1);
                    }
                    let vals: Vec<Val> = vals.into_iter().collect::<Option<_>>()?;
                    if is_enum { Val::Variant(idx, vals.into()) } else { Val::Agg(vals.into()) }
                }
                Step::Ctor(m, args) => self.ctor_call(n, m, &args, ty)?,
            });
        }
        Some((ty, cur?))
    }

    /// E0203 case `index`: the index `ie` of a const array of length `len` is out of range (it
    /// would trap `Bounds` at run time; a const keeps the error).
    fn index_range(&mut self, ie: u32, at: TypeId, len: usize) -> R {
        let icx = self.te.icx;
        let text = icx.node_text(ie).to_string();
        let tn = self.show(at);
        let d = if len == 0 {
            mk("E0203", Some("index-empty"), &[("literal", &text), ("type", &tn)], icx.node_span(ie), &[])
        } else {
            let max = (len - 1).to_string();
            let holes = [("literal", text.as_str()), ("type", &tn), ("min", "0"), ("max", &max)];
            mk("E0203", Some("index"), &holes, icx.node_span(ie), &[])
        };
        self.report(d);
        None
    }

    /// The value of the constructor `m`'s body with its parameters bound to the argument values.
    fn ctor_call(&mut self, n: u32, m: DeclId, args: &[u32], ty: TypeId) -> Option<Val> {
        let prog = self.te.icx.prog;
        let decl = *prog.decl(m)?;
        let (ptys, _) = self.env.fn_types(m)?;
        let mut vals = Vec::with_capacity(args.len());
        if ptys.len() != args.len() {
            return None;
        }
        for &a in args {
            vals.push(self.value(a)?);
        }
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
        let mut inner = Ev::new(&mut *self.env, &te);
        inner.depth = self.depth;
        inner.params = pnodes.into_iter().zip(vals).collect();
        let r = inner.eval(e, Some(ty));
        // a failure inside the body is the call's (the body's nodes are another item's)
        self.nrep += inner.nrep;
        if let Some(m) = inner.internal.take() {
            self.internal.get_or_insert(m);
        }
        if r.is_none() && inner.nrep == 0 && self.stuck.is_none() {
            self.stuck = self.e0904_at(n, OUTSIDE);
        }
        r.map(|x| x.1)
    }

    fn arith(&mut self, start: u32, at: u32, op: u8, t: TypeId, a: Val, b: Val) -> Option<Val> {
        match (a, b) {
            (Val::Int(x), Val::Int(y)) => {
                let Some((signed, bits)) = self.env.int(t) else {
                    return self.ice(at, "integer operands of a type that is not an integer");
                };
                let bw = bits.bits();
                let (lo, hi) = self.int_range(t)?;
                let check = |v: Option<i128>| v.filter(|v| *v >= lo && *v <= hi);
                let sat = |v: Option<i128>, neg: bool| match v {
                    Some(v) => v.clamp(lo, hi),
                    None if neg => lo,
                    None => hi,
                };
                let p = |o: Term| op == o as u8;
                let r = if p(Term::Plus) {
                    check(x.checked_add(y))
                } else if p(Term::Minus) {
                    check(x.checked_sub(y))
                } else if p(Term::Star) {
                    check(x.checked_mul(y))
                } else if p(Term::Slash) || p(Term::Percent) {
                    // `MIN / -1` and `MIN % -1` trap like division by zero
                    if y == 0 || signed && x == lo && y == -1 {
                        None
                    } else if p(Term::Slash) {
                        check(x.checked_div(y))
                    } else {
                        check(x.checked_rem(y))
                    }
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
                        // the body checker reports E0205 for a literal or const amount itself
                        let tn = self.show(t);
                        let case = if amt < 0 { Some("negative") } else { None };
                        let holes = [
                            ("amount", amt.to_string()),
                            ("bits", bw.to_string()),
                            ("type", tn),
                            ("last", (bw - 1).to_string()),
                        ];
                        let holes: Vec<(&str, &str)> = holes.iter().map(|(k, v)| (*k, v.as_str())).collect();
                        let d = mk("E0205", case, &holes, self.te.icx.node_span(at), &[]);
                        return self.defer(d);
                    }
                    if p(Term::GtGt) {
                        let u = if signed { x } else { (x as u128 & ((1u128 << bw) - 1)) as i128 };
                        Some(wrap_int(u >> amt, signed, bw))
                    } else {
                        Some(wrap_int(x.wrapping_shl(amt as u32), signed, bw))
                    }
                } else {
                    return self.ice(at, "an integer operator without a rule");
                };
                match r {
                    Some(v) => Some(Val::Int(v)),
                    None => self.overflow(start, at, t),
                }
            }
            (Val::Float(x), Val::Float(y)) => {
                let f32_ = matches!(self.env.kind(t), TypeKind::Float(FloatBits::F32));
                let f = |a: f64, b: f64, o: fn(f64, f64) -> f64, o32: fn(f32, f32) -> f32| {
                    if f32_ { o32(a as f32, b as f32) as f64 } else { o(a, b) }
                };
                let v = if op == Term::Plus as u8 {
                    f(x, y, |a, b| a + b, |a, b| a + b)
                } else if op == Term::Minus as u8 {
                    f(x, y, |a, b| a - b, |a, b| a - b)
                } else if op == Term::Star as u8 {
                    f(x, y, |a, b| a * b, |a, b| a * b)
                } else if op == Term::Slash as u8 {
                    f(x, y, |a, b| a / b, |a, b| a / b)
                } else if op == Term::Percent as u8 {
                    f(x, y, |a, b| a % b, |a, b| a % b)
                } else {
                    return self.ice(at, "a float operator without a rule");
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
                    self.ice(at, "a bool operator without a rule")
                }
            }
            _ => self.ice(at, "operands that are not scalars"),
        }
    }

    fn cast(&mut self, n: u32, op: u8, (t, v): (TypeId, Val), to: TypeId) -> R {
        let tk = self.env.kind(to).clone();
        let saturating = op == Term::AsPipe as u8;
        let checked = op == Term::AsQuestion as u8;
        let wrapping = op == Term::AsPercent as u8;
        let int_of = |v: &Val| -> Option<i128> {
            match v {
                Val::Int(i) => Some(*i),
                Val::Bool(b) => Some(*b as i128),
                Val::Char(c) => Some(*c as i128),
                _ => None,
            }
        };
        let _ = t;
        match tk {
            TypeKind::Int { signed, bits } => {
                let (lo, hi) = self.int_range(to)?;
                let res: i128 = match &v {
                    Val::Float(f) => {
                        if f.is_nan() {
                            if checked {
                                return self.none_of(to);
                            }
                            0
                        } else if checked {
                            if f.fract() == 0.0 && *f >= lo as f64 && *f <= hi as f64 {
                                *f as i128
                            } else {
                                return self.none_of(to);
                            }
                        } else {
                            // `as|`: the checker allows no other cast from a float to an int
                            (f.trunc() as i128).clamp(lo, hi)
                        }
                    }
                    other => {
                        let Some(i) = int_of(other) else {
                            return self.ice(n, "a cast of a value without one");
                        };
                        if wrapping {
                            wrap_int(i, signed, bits.bits())
                        } else if saturating {
                            i.clamp(lo, hi)
                        } else if checked {
                            if i < lo || i > hi {
                                return self.none_of(to);
                            }
                            i
                        } else if i < lo || i > hi {
                            // a plain `as` is lossless by its types
                            return self.ice(n, "a plain `as` that loses the value");
                        } else {
                            i
                        }
                    }
                };
                if checked { self.some_of(to, Val::Int(res)) } else { Some((to, Val::Int(res))) }
            }
            TypeKind::Float(fb) => {
                let (f, from_int) = match &v {
                    Val::Float(f) => (*f, None),
                    other => match int_of(other) {
                        Some(i) => (i as f64, Some(i)),
                        None => return self.ice(n, "a cast of a value without one"),
                    },
                };
                let r = match fb {
                    FloatBits::F32 => (f as f32) as f64,
                    _ => f,
                };
                if checked {
                    let lost = match from_int {
                        Some(i) => r as i128 != i || r >= 2f64.powi(127),
                        None => r != f && !f.is_nan(),
                    };
                    if lost {
                        return self.none_of(to);
                    }
                    return self.some_of(to, Val::Float(r));
                }
                Some((to, Val::Float(r)))
            }
            TypeKind::Char => {
                let Some(i) = int_of(&v) else { return self.ice(n, "a cast of a value without one") };
                let ok = u32::try_from(i).ok().and_then(char::from_u32).is_some();
                if checked {
                    if !ok {
                        return self.none_of(to);
                    }
                    return self.some_of(to, Val::Char(i as u32));
                }
                if !ok {
                    return self.ice(n, "a plain `as char` of a value that is no char");
                }
                Some((to, Val::Char(i as u32)))
            }
            TypeKind::Bool => match v {
                Val::Bool(b) => Some((to, Val::Bool(b))),
                _ => self.ice(n, "a cast to bool of a value that is not a bool"),
            },
            _ => self.ice(n, "a cast to a type that is not a scalar"),
        }
    }

    fn none_of(&mut self, inner: TypeId) -> R {
        Some((self.option_of(inner)?, Val::Variant(0, Arc::new([]))))
    }

    fn some_of(&mut self, inner: TypeId, v: Val) -> R {
        Some((self.option_of(inner)?, Val::Variant(1, Arc::new([v]))))
    }
}

/// The Primary that `n` is, through parentheses.
fn down_to_primary(icx: &ItemCx<'_>, n: u32) -> Option<u32> {
    let mut cur = n;
    loop {
        if icx.kind(cur)? != NodeKind::Primary {
            return None;
        }
        let a = icx.a(cur);
        if !icx.is(a, Term::LParen) {
            return Some(cur);
        }
        let kids = icx.kids_vec(cur);
        if kids.len() != 1 || icx.is(icx.close(a) - 1, Term::Comma) {
            return None;
        }
        cur = kids[0];
    }
}

fn round_float(f: f64, k: &TypeKind) -> f64 {
    if matches!(k, TypeKind::Float(FloatBits::F32)) { (f as f32) as f64 } else { f }
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
