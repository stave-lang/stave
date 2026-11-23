//! Traits and impls for non-generic code and core generics: Copy, Send/Sync, and Flat (the view
//! kind and [`crate::kinds::Kind`] are in [`crate::kinds`]), layouts, impl matching, trait-bound
//! checks, method lookup, and impl conformance (E0905 E0906 E0907, Copy E0415, Drop
//! E0416, an operator trait on a non-Copy type E0422, overlapping impls E0903, any impl of Flat
//! E1408).
//!
//! **Impl matching** unifies an impl's self type (its generic params are variables) with a
//! concrete or rigid type (a generic param of the checked fn is rigid); the impl applies when its
//! params' bounds hold for the bound arguments. **Method lookup**: the receiver type,
//! then each type reached by peeling one `&`/`&mut`; at the first type that has a method of that
//! name: an inherent impl's method first, else the trait impls' (two traits: E0908); a generic
//! param supplies the methods of its bounds. The receiver is then auto-referenced to the method's
//! `&self` / `&mut self` ([`Adjust`]).
//!
//! **Built-in trait impls**: `Copy` for scalars, `&T`, raw pointers, fn values (fn values
//! are Copy; their captures are by reference), arrays and tuples of Copy, vectors and masks; the
//! operator traits for the scalar types the operator is defined on (checked through bounds such
//! as `T: Add`); `Send`/`Sync` structurally (an ADT is Send/Sync if its fields are), except for
//! an ADT with an explicit `unsafe impl` of that trait: then only its impls decide (a failing
//! bound, as in `unsafe impl<T: Sync> Sync for G<T>`, makes the instance not Sync). Recursive
//! types are Send/Sync coinductively ([`QUERY_DEPTH`]: every query ends).
//!
//! **Flat** is derived, never implemented (an impl is E1408 and is not consulted): a type is
//! Flat iff it holds no reference, view, raw pointer, fn value, `std.cell.UnsafeCell`, or ADT
//! with a `Drop` impl, recursively through struct fields, enum payloads, the type arguments of
//! an ADT instance, and array, tuple, and range elements. `str` and slices are not Flat
//! (unsized); a generic param is Flat only through a `Flat` bound. Recursive types are Flat
//! coinductively, like Send/Sync.
//!
//! **Frozen** is derived the same way (an impl is E1408): a type is Frozen iff nothing reachable
//! from a value of it can change while the value is only shared: no `std.cell.UnsafeCell` and no
//! fn value (its captures are not in its type) is reachable through its fields, enum payloads,
//! type arguments, array, slice, tuple, and range elements, or the pointees of its references
//! and raw pointers. Views, `Drop` types, `str`, and slices may be Frozen. A generic param is
//! Frozen through a `Frozen` or a `Flat` bound (Flat implies Frozen). A rule that relies on "`&T`
//! is read-only" asks for it: the contract operand rule ([`crate::contracts`]).

use rustc_hash::FxHashMap;
use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind, Term};

use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::{DeclId, Sym};
use crate::itemcx::{Brackets, ItemCx};
use crate::resolve::TypeHead;
use crate::sigs::{FnSig, GenericParam, ImplDef, NameCx, Sigs, TraitRef};
use crate::types::{self, ConstArg, FloatBits, GArg, Interner, TypeId, TypeKind, TypeTable};
use crate::{Program, prim::Prim};

/// Why a type has no layout ([`type_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoLayout {
    /// Its size does not fit in a `u64`: no value of it can exist, and every reader rejects it.
    Overflow,
    /// Not concrete, the error type, a by-value cycle, or nested past the type depth limit: each
    /// of these has its own diagnostic.
    Unknown,
}

/// The ADT definitions a layout reads: the checker's ([`Sigs`]) or the signature builder's.
pub trait LayoutEnv: Interner {
    /// Is it an enum, and the field types of each variant (unsubstituted).
    fn adt_shape(&mut self, d: crate::ids::DeclId) -> Option<(bool, Vec<Vec<TypeId>>)>;
}

/// A table with the checker's ADT definitions.
pub struct WithSigs<'a, I: Interner + ?Sized> {
    pub cx: &'a mut I,
    pub sigs: &'a Sigs,
}

impl<I: Interner + ?Sized> Interner for WithSigs<'_, I> {
    fn kind(&self, t: TypeId) -> &TypeKind {
        self.cx.kind(t)
    }
    fn intern(&mut self, k: TypeKind) -> TypeId {
        self.cx.intern(k)
    }
    fn refusals(&self) -> u32 {
        self.cx.refusals()
    }
    fn claimed_mut(&mut self) -> &mut u32 {
        self.cx.claimed_mut()
    }
    fn fixed_ty(&self, f: TypeId) -> TypeId {
        self.cx.fixed_ty(f)
    }
}

impl<I: Interner + ?Sized> LayoutEnv for WithSigs<'_, I> {
    fn adt_shape(&mut self, d: crate::ids::DeclId) -> Option<(bool, Vec<Vec<TypeId>>)> {
        let adt = self.sigs.adt(d)?;
        Some((adt.is_enum, adt.variants.iter().map(|v| v.fields.iter().map(|f| f.ty).collect()).collect()))
    }
}

/// `(size, align)` of `t`: the one answer to "how many bytes" (equal to stave-ir's layout rules),
/// with a size past `u64` as [`NoLayout::Overflow`]. Memoised by type and depth: types share
/// components, so without it a `(A, A)` nested k deep costs 2^k.
pub fn type_bytes<E: LayoutEnv + ?Sized>(env: &mut E, t: TypeId) -> Result<(u64, u64), NoLayout> {
    bytes_memo(env, t, 0, &mut FxHashMap::default())
}

/// A layout or why there is none ([`type_bytes`]).
type Bytes = Result<(u64, u64), NoLayout>;

fn bytes_memo<E: LayoutEnv + ?Sized>(
    env: &mut E,
    t: TypeId,
    depth: u32,
    memo: &mut FxHashMap<(TypeId, u32), Bytes>,
) -> Bytes {
    if depth >= crate::typedepth::MAX_TYPE_DEPTH {
        return Err(NoLayout::Unknown);
    }
    if let Some(&l) = memo.get(&(t, depth)) {
        return l;
    }
    let k = env.kind(t).clone();
    let r = match k {
        TypeKind::Adt { def, args } => match env.adt_shape(def) {
            None => Err(NoLayout::Unknown),
            Some((is_enum, variants)) => (|| {
                let mut vls = Vec::with_capacity(variants.len());
                for v in &variants {
                    let mut fs = Vec::with_capacity(v.len());
                    for &f in v {
                        let ft = env.subst(f, &args, None);
                        fs.push(bytes_memo(env, ft, depth + 1, memo)?);
                    }
                    vls.push(fs);
                }
                adt_layout(is_enum, &vls)
            })(),
        },
        TypeKind::Ref { inner, .. } | TypeKind::Ptr { inner, .. } => {
            Ok(if matches!(env.kind(inner), TypeKind::Str | TypeKind::Slice { .. }) {
                (16, 8)
            } else {
                (8, 8)
            })
        }
        _ => scalar_layout(&k, &mut |x| bytes_memo(env, x, depth + 1, memo)),
    };
    memo.insert((t, depth), r);
    r
}

/// Layout of a scalar or a type built from components (arrays, tuples, ranges, vectors, masks);
/// `sub` lays out a component.
pub fn scalar_layout(
    k: &TypeKind,
    sub: &mut dyn FnMut(TypeId) -> Result<(u64, u64), NoLayout>,
) -> Result<(u64, u64), NoLayout> {
    Ok(match k {
        TypeKind::Int { bits, .. } => {
            let b = (bits.bits() / 8) as u64;
            (b, b)
        }
        TypeKind::Float(FloatBits::F16 | FloatBits::B16) => (2, 2),
        TypeKind::Float(FloatBits::F32) => (4, 4),
        TypeKind::Float(FloatBits::F64) => (8, 8),
        TypeKind::Bool => (1, 1),
        TypeKind::Char => (4, 4),
        TypeKind::Never => (0, 1),
        // fat pointers to slices and `str` are handled by the callers: (16, 8)
        TypeKind::Ref { .. } | TypeKind::Ptr { .. } => (8, 8),
        // a fn value is `(code, environment)`, as the IR lays it out
        TypeKind::Fn { .. } => (16, 8),
        TypeKind::Array { elem, len: ConstArg::Value(n) } => {
            let (s, a) = sub(*elem)?;
            (s.checked_mul(*n).ok_or(NoLayout::Overflow)?, a)
        }
        TypeKind::Tuple(ts) => {
            let mut fields = Vec::new();
            for &t in ts.iter() {
                fields.push(sub(t)?);
            }
            struct_layout(&fields, 0)?
        }
        TypeKind::Range { elem, .. } => {
            let f = sub(*elem)?;
            struct_layout(&[f, f], 0)?
        }
        TypeKind::Vector { lane, n: crate::prim::Lanes::Fixed(n) } => {
            let (s, _) = sub(*lane)?;
            let total = s.checked_mul(*n as u64).ok_or(NoLayout::Overflow)?;
            (total, total)
        }
        TypeKind::Mask { width, n: crate::prim::Lanes::Fixed(n) } => {
            let total = (*width as u64 / 8) * (*n as u64);
            (total, total)
        }
        _ => return Err(NoLayout::Unknown),
    })
}

/// Layout of `t` in table `cx` (concrete types only): [`type_bytes`] without the reason.
pub fn layout_of<I: Interner + ?Sized>(cx: &mut I, sigs: &Sigs, t: TypeId) -> Option<(u64, u64)> {
    type_bytes(&mut WithSigs { cx, sigs }, t).ok()
}

fn struct_layout(fields: &[(u64, u64)], start: u64) -> Result<(u64, u64), NoLayout> {
    struct_offsets(fields, start).map(|(_, size, align)| (size, align)).ok_or(NoLayout::Overflow)
}

/// C-like record from `start`: each field's offset, and the record's `(size, align)`.
fn struct_offsets(fields: &[(u64, u64)], start: u64) -> Option<(Vec<u64>, u64, u64)> {
    let mut off = start;
    let mut align = 1u64;
    let mut offs = Vec::with_capacity(fields.len());
    for &(s, a) in fields {
        let a = a.max(1);
        off = off.checked_add(a - 1)? / a * a;
        offs.push(off);
        off = off.checked_add(s)?;
        align = align.max(a);
    }
    let size = off.checked_add(align - 1)? / align * align;
    Some((offs, size, align))
}

/// The field offsets of a concrete struct, enum (per variant, after the `u32` tag), or tuple, by
/// the rules of [`adt_layout`]; `None` for other types and types without a layout.
pub fn field_offsets<I: Interner + ?Sized>(cx: &mut I, sigs: &Sigs, t: TypeId) -> Option<Vec<Vec<u64>>> {
    let (variants, start) = match cx.kind(t).clone() {
        TypeKind::Adt { def, args } => {
            let adt = sigs.adt(def)?;
            let fs: Vec<Vec<TypeId>> = adt
                .variants
                .iter()
                .map(|v| v.fields.iter().map(|f| cx.subst(f.ty, &args, None)).collect())
                .collect();
            (fs, if adt.is_enum { 4 } else { 0 })
        }
        TypeKind::Tuple(ts) => (vec![ts.to_vec()], 0),
        _ => return None,
    };
    let mut out = Vec::with_capacity(variants.len());
    for v in variants {
        let mut ls = Vec::with_capacity(v.len());
        for f in v {
            ls.push(layout_of(cx, sigs, f)?);
        }
        out.push(struct_offsets(&ls, start)?.0);
    }
    Some(out)
}

/// ADT layout: a struct is C-like; an enum has a `u32` tag at 0, then each variant's fields from
/// offset 4, size the largest variant rounded up.
pub fn adt_layout(is_enum: bool, variants: &[Vec<(u64, u64)>]) -> Result<(u64, u64), NoLayout> {
    if !is_enum {
        return struct_layout(variants.first().map_or(&[][..], |v| &v[..]), 0);
    }
    let mut size = 4u64;
    let mut align = 4u64;
    for v in variants {
        let (s, a) = struct_layout(v, 4)?;
        size = size.max(s);
        align = align.max(a);
    }
    let size = size.checked_add(align - 1).ok_or(NoLayout::Overflow)? / align * align;
    Ok((size, align))
}

/// The longest part path a Flat / Frozen witness names ([`World::derived_witness`]): the type
/// depth limit, which no type of a checked program passes (E0005 case `type`).
///
/// A trait query itself has no depth limit: every step goes to a component of its type (a type
/// argument, an element, a field) or to another goal on the same type (a blanket impl's bound),
/// a goal reached again while in progress is cut ([`Query`]), and the types are finitely many
/// ([`crate::types::MAX_TYPE_HEIGHT`]). So every query ends, with the answer the rules give.
pub(crate) const QUERY_DEPTH: u32 = crate::typedepth::MAX_TYPE_DEPTH;

/// State of one top-level trait query: the Send/Sync goals in progress (a goal reached again is
/// assumed to hold), the lowest in-progress index an assumption used below the current goal
/// (`usize::MAX`: none), and the results already known. The other goals (Copy, a trait with
/// impls) are inductive: one reached again while in progress does not hold through that path
/// (`goals`, `goal_low`: the same bookkeeping, so a negative answer that used it is not
/// memoized). `any_kind`: impl matching skips the
/// view-kind rule (a query that asks whether an impl could apply to some instance, or names the
/// impl a view was refused by; never one that selects an impl).
struct Query {
    /// Coinductive goals in progress, and the index of each.
    active: Vec<(TypeId, DeclId)>,
    active_at: FxHashMap<(TypeId, DeclId), usize>,
    low: usize,
    /// Inductive goals in progress, and the index of each.
    goals: Vec<(TypeId, TraitRef)>,
    goals_at: FxHashMap<(TypeId, TraitRef), usize>,
    goal_low: usize,
    memo: FxHashMap<(TypeId, DeclId), bool>,
    /// Answers that used a goal still in progress further out: a positive one an assumed
    /// coinductive goal, a negative one a cut inductive goal. The goal they used settles them
    /// when it ends ([`Query::settle`]): kept if its answer is the one they assumed.
    pending: Vec<((TypeId, DeclId), bool)>,
    any_kind: bool,
}

impl Default for Query {
    fn default() -> Query {
        Query {
            active: Vec::new(),
            active_at: FxHashMap::default(),
            low: usize::MAX,
            goals: Vec::new(),
            goals_at: FxHashMap::default(),
            goal_low: usize::MAX,
            memo: FxHashMap::default(),
            pending: Vec::new(),
            any_kind: false,
        }
    }
}

impl Query {
    /// Enters inductive goal `g`: `None` if it is in progress (it does not hold through this
    /// path), else the mark [`Query::leave_goal`] takes.
    fn enter_goal(&mut self, g: (TypeId, TraitRef)) -> Option<(usize, usize)> {
        if let Some(&i) = self.goals_at.get(&g) {
            self.goal_low = self.goal_low.min(i);
            return None;
        }
        let me = self.goals.len();
        self.goals_at.insert(g.clone(), me);
        self.goals.push(g);
        Some((me, std::mem::replace(&mut self.goal_low, usize::MAX)))
    }

    /// Leaves the goal entered with `mark`: whether its answer used no goal still in progress
    /// further out (so a negative answer is final).
    fn leave_goal(&mut self, mark: (usize, usize)) -> bool {
        let (me, saved) = mark;
        for g in self.goals.drain(me..) {
            self.goals_at.remove(&g);
        }
        let low = self.goal_low;
        self.goal_low = saved.min(if low < me { low } else { usize::MAX });
        low >= me
    }

    /// Enters coinductive goal `key`: `Err` (it holds through this path) if it is in progress.
    fn enter_active(&mut self, key: (TypeId, DeclId)) -> Result<(), ()> {
        if let Some(&i) = self.active_at.get(&key) {
            self.low = self.low.min(i);
            return Err(());
        }
        self.active_at.insert(key, self.active.len());
        self.active.push(key);
        Ok(())
    }

    fn leave_active(&mut self) {
        if let Some(k) = self.active.pop() {
            self.active_at.remove(&k);
        }
    }

    /// For a goal kept in no stack (a tuple's Copy): the mark [`Query::own_low`] takes.
    fn open_low(&mut self) -> usize {
        std::mem::replace(&mut self.goal_low, usize::MAX)
    }

    /// Whether the answer since `open_low` used no inductive goal in progress.
    fn own_low(&mut self, saved: usize) -> bool {
        let low = self.goal_low;
        self.goal_low = saved.min(low);
        low == usize::MAX
    }

    /// Records answer `r` of a goal that ended (`key`: `None` for one not memoized): final
    /// (`free_pos`: it used no assumed coinductive goal further out, `free_neg`: no cut
    /// inductive goal further out; a positive answer needs the first, a negative one the second)
    /// or pending. The pending answers recorded since `start` (below this goal) that assumed
    /// this goal's answer wrongly are dropped: positives once a coinductive goal fails,
    /// negatives once an inductive goal holds. A goal free of both is the outermost goal the rest
    /// used, so they are final now: kept if they assumed its answer, else dropped (dropping only
    /// costs a recomputation). Returns the final answers.
    fn settle(
        &mut self,
        key: Option<(TypeId, DeclId)>,
        r: bool,
        free_pos: bool,
        free_neg: bool,
        start: usize,
        coinductive: bool,
    ) -> Vec<((TypeId, DeclId), bool)> {
        let start = start.min(self.pending.len());
        let mut below: Vec<((TypeId, DeclId), bool)> = self.pending.drain(start..).collect();
        if coinductive && !r {
            below.retain(|e| !e.1);
        }
        if !coinductive && r {
            below.retain(|e| e.1);
        }
        let mut out = Vec::new();
        if free_pos && free_neg {
            out.extend(below.into_iter().filter(|e| if e.1 { coinductive && r } else { !coinductive && !r }));
        } else {
            self.pending.extend(below);
        }
        if let Some(key) = key {
            if (r && free_pos) || (!r && free_neg) {
                out.push((key, r));
            } else {
                self.pending.push((key, r));
            }
        }
        for &(k, v) in &out {
            self.memo.insert(k, v);
        }
        out
    }
}

impl Query {
    fn any_kind() -> Query {
        Query { any_kind: true, ..Query::default() }
    }
}

/// Final answers of trait queries about concrete types of the global type table, shared by
/// every query of a checking run, so each (type, trait) is answered once ([`World::implements`]).
/// Armed for a table by [`QueryCache::arm`]: only ids below the table's length at that time are
/// kept (ids appended later may differ between later phases), and arming again (the table was
/// renumbered) forgets every answer.
#[derive(Default)]
pub struct QueryCache {
    limit: std::sync::atomic::AtomicU32,
    map: std::sync::Mutex<FxHashMap<(TypeId, DeclId, bool), bool>>,
}

impl std::fmt::Debug for QueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("QueryCache")
    }
}

impl QueryCache {
    /// Keeps answers for the ids of a table of `len` types from now on, forgetting all before.
    pub fn arm(&self, len: usize) {
        self.lock().clear();
        self.limit.store(len.min(u32::MAX as usize) as u32, std::sync::atomic::Ordering::Release);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FxHashMap<(TypeId, DeclId, bool), bool>> {
        self.map.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn holds<I: Interner + ?Sized>(&self, cx: &I, t: TypeId) -> bool {
        !t.is_local()
            && !t.is_none()
            && (t.idx() as u64) < u64::from(self.limit.load(std::sync::atomic::Ordering::Acquire))
            && !cx.has_params(t)
    }

    fn get<I: Interner + ?Sized>(&self, cx: &I, t: TypeId, tr: DeclId, any_kind: bool) -> Option<bool> {
        if !self.holds(cx, t) {
            return None;
        }
        self.lock().get(&(t, tr, any_kind)).copied()
    }

    fn put<I: Interner + ?Sized>(&self, cx: &I, answers: &[((TypeId, DeclId), bool)], any_kind: bool) {
        let keep: Vec<_> = answers.iter().filter(|((t, _), _)| self.holds(cx, *t)).collect();
        if keep.is_empty() {
            return;
        }
        let mut m = self.lock();
        for &&((t, d), v) in &keep {
            m.insert((t, d, any_kind), v);
        }
    }
}

/// The traits the compiler derives from a type's structure, never from impls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derived {
    Flat,
    Frozen,
}

/// Why a type is not `Flat` / `Frozen` (E1407 / E1410 `reason`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotFlat {
    Ref,
    Ptr,
    FnValue,
    View,
    UnsafeCell,
    Drop,
    /// A generic param without a `Flat` bound (or `Self`).
    Param,
    /// `str` or a slice: no by-value copy exists.
    Unsized,
    /// The witness path passed [`QUERY_DEPTH`] parts (no type of a checked program has one).
    Depth,
    /// An ADT without a built signature: only after an earlier error, never reported.
    Unknown,
}

impl NotFlat {
    /// The E1407 `reason` text (a `choices` entry of the registry).
    pub fn text(self) -> &'static str {
        self.text_for(Derived::Flat)
    }

    /// The `reason` text for trait `d` (E1407, E1410, E0211 case `frozen`).
    pub fn text_for(self, d: Derived) -> &'static str {
        if d == Derived::Frozen {
            match self {
                NotFlat::FnValue => return "a fn value (its captures are not in its type)",
                NotFlat::Param => return "a generic parameter without a `Frozen` bound",
                _ => {}
            }
        }
        match self {
            NotFlat::Ref => "a reference",
            NotFlat::Ptr => "a raw pointer",
            NotFlat::FnValue => "a fn value",
            NotFlat::View => "a view type",
            NotFlat::UnsafeCell => "`UnsafeCell` (interior mutability)",
            NotFlat::Drop => "a type with `Drop`",
            NotFlat::Param => "a generic parameter without a `Flat` bound",
            NotFlat::Unsized => "unsized",
            NotFlat::Depth | NotFlat::Unknown => "",
        }
    }
}

/// A component of a type in the Flat rule (rendered into the E1407 path).
#[derive(Clone, Copy, Debug)]
enum FlatSeg {
    /// Field `.1` of the struct.
    Field(DeclId, u32),
    /// Field `.2` of variant `.1` of the enum.
    VariantField(DeclId, u32, u32),
    /// Type argument `.1` of the ADT instance.
    Arg(DeclId, u32),
    Index(u32),
    Elem,
    Start,
    /// The pointee of a reference or raw pointer (Frozen only).
    Deref,
}

/// The first non-flat component of a type: its path (`a.p`, `[_]`, `0`, `Some.0`, `<T>`; empty
/// for the type itself), its type, and why it is not flat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlatWitness {
    pub path: String,
    pub part: TypeId,
    pub why: NotFlat,
}

/// Read-only program view for trait queries.
#[derive(Clone, Copy)]
pub struct World<'a> {
    pub prog: &'a Program,
    pub sigs: &'a Sigs,
}

/// A resolved method.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodRef {
    /// A method of an impl (inherent or trait): the method decl, the impl, its generic args.
    Impl { method: DeclId, impl_decl: DeclId, impl_args: Vec<GArg> },
    /// A trait method reached through a generic param's bound: trait method decl, the bound's
    /// trait args, the receiver type (`Self`).
    Bound { method: DeclId, trait_args: Vec<GArg>, self_ty: TypeId },
    /// A built-in method ([`crate::builtins`]).
    Builtin(crate::builtins::BuiltinMethod),
}

/// Why a method is no `for` loop's `next` ([`World::for_next_elem`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotNext {
    /// Not `&mut self` (`&self`, `self`, or no `self`).
    Receiver,
    /// Params besides `self`.
    Params,
    /// Generic params of its own.
    Generics,
    /// A result other than `Option<X>`.
    Result,
    /// A built-in method (none is named `next`).
    Builtin,
}

impl NotNext {
    /// The E0219 `next-shape` reason (a `why` choice of the registry).
    pub fn why(self) -> &'static str {
        match self {
            NotNext::Receiver => "it does not take `&mut self`",
            NotNext::Params => "it takes params besides `&mut self`",
            NotNext::Generics => "it has generic params of its own",
            NotNext::Result | NotNext::Builtin => "it does not return an `Option`",
        }
    }
}

/// The receiver adjustment of a method call (made explicit for lowering and borrowck).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Adjust {
    /// `*` applied this many times (auto-deref through `&`/`&mut`).
    pub derefs: u32,
    /// Then `&` (false) or `&mut` (true), if the method takes `&self` / `&mut self` and the
    /// receiver place was reached by value.
    pub autoref: Option<bool>,
    /// A `&mut` receiver passed on to `&mut self` is reborrowed (`&mut *r`).
    pub reborrow: bool,
}

/// Method lookup outcome.
pub enum Lookup {
    Found {
        m: MethodRef,
        level: u32,
        recv: TypeId,
    },
    /// Two traits, or two instances of one trait, supply it (E0908): the trait decls, their
    /// method decls, and the two methods as resolved.
    Ambiguous {
        traits: [DeclId; 2],
        methods: [DeclId; 2],
        refs: Box<[MethodRef; 2]>,
        recv: TypeId,
    },
    /// A generic param without a bound providing it (E0901).
    NotInBounds {
        param: u32,
    },
    NotFound,
}

impl<'a> World<'a> {
    pub fn adt_view(&self, d: DeclId) -> bool {
        self.prog.decl(d).is_some_and(|x| x.flags.has(DeclFlags::VIEW))
    }

    /// The type head of a type (for impl tables).
    pub fn head<I: Interner + ?Sized>(&self, cx: &I, t: TypeId) -> Option<TypeHead> {
        match cx.kind(t) {
            TypeKind::Adt { def, .. } => Some(TypeHead::Adt(*def)),
            k => types::type_scalar(k).map(|s| TypeHead::Prim(Prim::Scalar(s))).or_else(|| match k {
                TypeKind::Vector { lane, n } => types::type_scalar(cx.kind(*lane))
                    .map(|l| TypeHead::Prim(Prim::Vector { lane: l, n: *n })),
                TypeKind::Mask { width, n } => Some(TypeHead::Prim(Prim::Mask { width: *width, n: *n })),
                _ => None,
            }),
        }
    }

    /// Unifies impl pattern `pat` (generic params = variables in `binds`) with `t`. Past its
    /// step limit the answer is "no match".
    pub fn match_ty<I: Interner + ?Sized>(
        &self,
        cx: &I,
        pat: TypeId,
        t: TypeId,
        binds: &mut [Option<GArg>],
    ) -> bool {
        self.match_ty_in(cx, pat, t, binds)
    }

    /// Exact: each pair of (pattern, type) components is matched once. A pair met again gives
    /// the same answer (a param binds once, later occurrences compare), and types share
    /// components, so the walk is linear in the distinct pairs, not in the leaves.
    fn match_ty_in<I: Interner + ?Sized>(
        &self,
        cx: &I,
        pat: TypeId,
        t: TypeId,
        binds: &mut [Option<GArg>],
    ) -> bool {
        let mut stack = vec![(pat, t)];
        let mut seen: rustc_hash::FxHashSet<(TypeId, TypeId)> = rustc_hash::FxHashSet::default();
        while let Some((p, x)) = stack.pop() {
            if !seen.insert((p, x)) {
                continue;
            }
            if p == x && !cx.has_params(p) {
                continue;
            }
            if cx.is_error_ty(x) {
                continue;
            }
            match (cx.kind(p), cx.kind(x)) {
                (TypeKind::Param(i), _) => {
                    let Some(slot) = binds.get_mut(*i as usize) else { return false };
                    match slot {
                        None => *slot = Some(GArg::Type(x)),
                        Some(GArg::Type(prev)) => {
                            if *prev != x {
                                return false;
                            }
                        }
                        Some(GArg::Const(_)) => return false,
                    }
                }
                (TypeKind::Ref { mutable: m1, inner: i1 }, TypeKind::Ref { mutable: m2, inner: i2 })
                | (TypeKind::Ptr { mutable: m1, inner: i1 }, TypeKind::Ptr { mutable: m2, inner: i2 }) => {
                    if m1 != m2 {
                        return false;
                    }
                    stack.push((*i1, *i2));
                }
                (TypeKind::Array { elem: e1, len: l1 }, TypeKind::Array { elem: e2, len: l2 }) => {
                    if !match_const(*l1, *l2, binds) {
                        return false;
                    }
                    stack.push((*e1, *e2));
                }
                (TypeKind::Slice { elem: e1, align: a1 }, TypeKind::Slice { elem: e2, align: a2 }) => {
                    if a1 != a2 {
                        return false;
                    }
                    stack.push((*e1, *e2));
                }
                (
                    TypeKind::Range { elem: e1, inclusive: i1 },
                    TypeKind::Range { elem: e2, inclusive: i2 },
                ) => {
                    if i1 != i2 {
                        return false;
                    }
                    stack.push((*e1, *e2));
                }
                (TypeKind::Tuple(a), TypeKind::Tuple(b)) => {
                    if a.len() != b.len() {
                        return false;
                    }
                    for (p, x) in a.iter().zip(b.iter()) {
                        stack.push((*p, *x));
                    }
                }
                (TypeKind::Adt { def: d1, args: a1 }, TypeKind::Adt { def: d2, args: a2 }) => {
                    if d1 != d2 || a1.len() != a2.len() {
                        return false;
                    }
                    for (p, x) in a1.iter().zip(a2.iter()) {
                        match (p, x) {
                            (GArg::Type(p), GArg::Type(x)) => stack.push((*p, *x)),
                            (GArg::Const(p), GArg::Const(x)) => {
                                if !match_const(*p, *x, binds) {
                                    return false;
                                }
                            }
                            _ => return false,
                        }
                    }
                }
                (
                    TypeKind::Fn { params: p1, ret: r1, features: f1, deny: d1 },
                    TypeKind::Fn { params: p2, ret: r2, features: f2, deny: d2 },
                ) => {
                    if f1 != f2 || d1 != d2 || p1.len() != p2.len() {
                        return false;
                    }
                    for (p, x) in p1.iter().zip(p2.iter()) {
                        stack.push((*p, *x));
                    }
                    stack.push((*r1, *r2));
                }
                (TypeKind::Vector { lane: l1, n: n1 }, TypeKind::Vector { lane: l2, n: n2 }) => {
                    if n1 != n2 {
                        return false;
                    }
                    stack.push((*l1, *l2));
                }
                _ => {
                    if p != x {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Whether `t` implements `tr` in a context whose generic params are `env` (their bounds hold).
    pub fn implements<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        env: &[GenericParam],
    ) -> bool {
        self.implements_q(cx, t, tr, env, &mut Query::default())
    }

    fn implements_q<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        env: &[GenericParam],
        q: &mut Query,
    ) -> bool {
        if cx.is_error_ty(t) {
            return true;
        }
        let lang = &self.sigs.lang;
        if let TypeKind::Param(i) = cx.kind(t) {
            let Some(g) = env.get(*i as usize) else { return false };
            if g.bounds.iter().any(|b| b.trait_decl == tr.trait_decl && b.args == tr.args) {
                return true;
            }
            // Flat implies Frozen: no pointer and no `UnsafeCell` leaves nothing to reach.
            if tr.trait_decl == lang.frozen
                && lang.frozen.is_some()
                && lang.flat.is_some()
                && g.bounds.iter().any(|b| b.trait_decl == lang.flat)
            {
                return true;
            }
            // Copy bound implied by nothing else; Send/Sync are not implied either.
            return false;
        }
        if tr.trait_decl == lang.copy && lang.copy.is_some() {
            return self.is_copy_q(cx, t, env, q);
        }
        let key = (t, tr.trait_decl);
        if tr.args.is_empty() {
            if let Some(&r) = q.memo.get(&key) {
                return r;
            }
            if let Some(r) = self.sigs.queries.get(&*cx, t, tr.trait_decl, q.any_kind) {
                return r;
            }
        }
        let derived = if tr.trait_decl.is_none() {
            None
        } else if tr.trait_decl == lang.flat {
            Some(Derived::Flat)
        } else if tr.trait_decl == lang.frozen {
            Some(Derived::Frozen)
        } else {
            None
        };
        let auto = (tr.trait_decl == lang.send || tr.trait_decl == lang.sync || derived.is_some())
            && tr.trait_decl.is_some();
        // Send/Sync/Flat/Frozen are coinductive: a type reached again while its own query is in
        // progress (a recursive type) is assumed to implement them.
        if auto && q.enter_active(key).is_err() {
            return true;
        }
        let me = if auto { q.active.len() - 1 } else { q.active.len() };
        let saved_low = std::mem::replace(&mut q.low, usize::MAX);
        let start = q.pending.len();
        let mark = if auto {
            (q.goals.len(), q.open_low())
        } else {
            match q.enter_goal((t, tr.clone())) {
                Some(m) => m,
                None => {
                    q.low = saved_low;
                    return false;
                }
            }
        };
        let r = if let Some(d) = derived {
            self.auto_derived(cx, t, tr, d, env, q)
        } else if auto {
            self.send_sync(cx, t, tr, env, q)
        } else if let Some(name) = lang.ops.iter().find(|(_, d)| **d == tr.trait_decl).map(|(n, _)| *n)
            && builtin_op_impl(cx, name, t)
        {
            true
        } else {
            self.find_impl_q(cx, t, tr, env, q).is_some()
        };
        if auto {
            q.leave_active();
        }
        let free_neg = if auto { q.own_low(mark.1) } else { q.leave_goal(mark) };
        let low = q.low;
        let memo_key = tr.args.is_empty().then_some(key);
        let done = q.settle(memo_key, r, low >= me, free_neg, start, auto);
        self.sigs.queries.put(&*cx, &done, q.any_kind);
        q.low = saved_low.min(if low < me { low } else { usize::MAX });
        r
    }

    /// Send/Sync: an explicit impl decides for the types of its head (an ADT with an `unsafe impl`
    /// of the trait is not structural: a bound of the impl that fails makes the type not
    /// implement it); other types are structural.
    fn send_sync<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        env: &[GenericParam],
        q: &mut Query,
    ) -> bool {
        if self.find_impl_q(cx, t, tr, env, q).is_some() {
            return true;
        }
        if let TypeKind::Adt { def, .. } = cx.kind(t)
            && self.has_impl_for_head(cx, *def, tr.trait_decl)
        {
            return false;
        }
        self.auto_send_sync(cx, t, tr, env, q)
    }

    /// Whether trait `trait_decl` has an impl whose self type is an instance of ADT `def`.
    fn has_impl_for_head<I: Interner + ?Sized>(&self, cx: &I, def: DeclId, trait_decl: DeclId) -> bool {
        let Some(list) = self.sigs.impls_by_trait.get(&trait_decl) else { return false };
        list.iter().any(|&i| {
            self.sigs
                .impl_def(i)
                .is_some_and(|imp| matches!(cx.kind(imp.self_ty), TypeKind::Adt { def: d, .. } if *d == def))
        })
    }

    fn auto_send_sync<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        env: &[GenericParam],
        q: &mut Query,
    ) -> bool {
        let is_sync = tr.trait_decl == self.sigs.lang.sync;
        match cx.kind(t).clone() {
            TypeKind::Ptr { .. } | TypeKind::Fn { .. } => false,
            TypeKind::Ref { mutable, inner } => {
                // &T: Send iff T: Sync; &mut T: Send iff T: Send. Both Sync iff T: Sync.
                let want = if !is_sync && !mutable { self.sigs.lang.sync } else { tr.trait_decl };
                self.implements_q(cx, inner, &TraitRef { trait_decl: want, args: Vec::new() }, env, q)
            }
            TypeKind::Adt { def, args } => {
                // UnsafeCell<T>: Send iff T: Send (its field), never Sync (crate::kinds)
                if is_sync && def == self.sigs.lang.unsafe_cell && def.is_some() {
                    return false;
                }
                let Some(adt) = self.sigs.adt(def) else { return false };
                for v in &adt.variants {
                    for f in &v.fields {
                        let ft = cx.subst(f.ty, &args, None);
                        if !self.implements_q(cx, ft, tr, env, q) {
                            return false;
                        }
                    }
                }
                true
            }
            k => {
                let mut kids = Vec::new();
                k.children(&mut kids);
                kids.into_iter().all(|c| self.implements_q(cx, c, tr, env, q))
            }
        }
    }

    /// Flat / Frozen: structural only (impls of them are rejected, E1408, and never consulted).
    #[allow(clippy::too_many_arguments)]
    fn auto_derived<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        d: Derived,
        env: &[GenericParam],
        q: &mut Query,
    ) -> bool {
        match self.derived_parts(cx, t, d) {
            Ok(parts) => parts.into_iter().all(|(_, c)| self.implements_q(cx, c, tr, env, q)),
            Err(_) => false,
        }
    }

    /// One step of the Flat rule: the components of `t` (fields of every variant, then the type
    /// arguments of an ADT instance; array, tuple, and range elements), or why `t` itself is not
    /// flat. A generic param is the caller's to decide (its bounds); `Self` and a const param
    /// used as a type are not flat.
    fn derived_parts<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        d: Derived,
    ) -> Result<Vec<(FlatSeg, TypeId)>, NotFlat> {
        if d == Derived::Frozen {
            return self.frozen_parts(cx, t);
        }
        let lang = &self.sigs.lang;
        match cx.kind(t).clone() {
            TypeKind::Int { .. }
            | TypeKind::Float(_)
            | TypeKind::Bool
            | TypeKind::Char
            | TypeKind::Vector { .. }
            | TypeKind::Mask { .. }
            | TypeKind::Never
            | TypeKind::Error => Ok(Vec::new()),
            TypeKind::Ref { .. } => Err(NotFlat::Ref),
            TypeKind::Ptr { .. } => Err(NotFlat::Ptr),
            TypeKind::Fn { .. } => Err(NotFlat::FnValue),
            TypeKind::Str | TypeKind::Slice { .. } => Err(NotFlat::Unsized),
            TypeKind::Param(_) | TypeKind::ConstParam(_) | TypeKind::SelfTy => Err(NotFlat::Param),
            TypeKind::Array { elem, .. } => Ok(vec![(FlatSeg::Elem, elem)]),
            TypeKind::Range { elem, .. } => Ok(vec![(FlatSeg::Start, elem)]),
            TypeKind::Tuple(ts) => {
                Ok(ts.iter().enumerate().map(|(i, &x)| (FlatSeg::Index(i as u32), x)).collect())
            }
            TypeKind::Adt { def, args } => {
                if def == lang.unsafe_cell && def.is_some() {
                    return Err(NotFlat::UnsafeCell);
                }
                if self.adt_view(def) {
                    return Err(NotFlat::View);
                }
                if lang.drop.is_some() && self.has_impl_for_head(cx, def, lang.drop) {
                    return Err(NotFlat::Drop);
                }
                let Some(adt) = self.sigs.adt(def) else { return Err(NotFlat::Unknown) };
                let mut out = Vec::new();
                for (vi, v) in adt.variants.iter().enumerate() {
                    for (fi, f) in v.fields.iter().enumerate() {
                        let ft = cx.subst(f.ty, &args, None);
                        let seg = if adt.is_enum {
                            FlatSeg::VariantField(def, vi as u32, fi as u32)
                        } else {
                            FlatSeg::Field(def, fi as u32)
                        };
                        out.push((seg, ft));
                    }
                }
                for (i, a) in args.iter().enumerate() {
                    if let GArg::Type(x) = a {
                        out.push((FlatSeg::Arg(def, i as u32), *x));
                    }
                }
                Ok(out)
            }
        }
    }

    /// One step of the Frozen rule: the components of `t` (as for Flat, plus the pointee of a
    /// reference or raw pointer and the element of a slice), or why `t` itself is not Frozen.
    fn frozen_parts<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
    ) -> Result<Vec<(FlatSeg, TypeId)>, NotFlat> {
        let lang = &self.sigs.lang;
        match cx.kind(t).clone() {
            TypeKind::Int { .. }
            | TypeKind::Float(_)
            | TypeKind::Bool
            | TypeKind::Char
            | TypeKind::Str
            | TypeKind::Vector { .. }
            | TypeKind::Mask { .. }
            | TypeKind::Never
            | TypeKind::Error => Ok(Vec::new()),
            TypeKind::Ref { inner, .. } | TypeKind::Ptr { inner, .. } => Ok(vec![(FlatSeg::Deref, inner)]),
            TypeKind::Fn { .. } => Err(NotFlat::FnValue),
            TypeKind::Param(_) | TypeKind::ConstParam(_) | TypeKind::SelfTy => Err(NotFlat::Param),
            TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } => Ok(vec![(FlatSeg::Elem, elem)]),
            TypeKind::Range { elem, .. } => Ok(vec![(FlatSeg::Start, elem)]),
            TypeKind::Tuple(ts) => {
                Ok(ts.iter().enumerate().map(|(i, &x)| (FlatSeg::Index(i as u32), x)).collect())
            }
            TypeKind::Adt { def, args } => {
                if def == lang.unsafe_cell && def.is_some() {
                    return Err(NotFlat::UnsafeCell);
                }
                let Some(adt) = self.sigs.adt(def) else { return Err(NotFlat::Unknown) };
                let mut out = Vec::new();
                for (vi, v) in adt.variants.iter().enumerate() {
                    for (fi, f) in v.fields.iter().enumerate() {
                        let ft = cx.subst(f.ty, &args, None);
                        let seg = if adt.is_enum {
                            FlatSeg::VariantField(def, vi as u32, fi as u32)
                        } else {
                            FlatSeg::Field(def, fi as u32)
                        };
                        out.push((seg, ft));
                    }
                }
                for (i, a) in args.iter().enumerate() {
                    if let GArg::Type(x) = a {
                        out.push((FlatSeg::Arg(def, i as u32), *x));
                    }
                }
                Ok(out)
            }
        }
    }

    /// Why `t` is not `Flat` in the scope of `env` (`None`: it is): the first non-flat component
    /// in [`World::flat_parts`] order, reached by descending into the first failing component at
    /// each level. [`NotFlat::Depth`] when the path passes [`QUERY_DEPTH`] parts.
    pub fn flat_witness<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        env: &[GenericParam],
    ) -> Option<FlatWitness> {
        self.derived_witness(cx, t, env, Derived::Flat)
    }

    /// [`World::flat_witness`] for either derived trait.
    pub fn derived_witness<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        env: &[GenericParam],
        d: Derived,
    ) -> Option<FlatWitness> {
        let decl = match d {
            Derived::Flat => self.sigs.lang.flat,
            Derived::Frozen => self.sigs.lang.frozen,
        };
        if decl.is_none() {
            return None;
        }
        let tr = TraitRef { trait_decl: decl, args: Vec::new() };
        // one query for the whole walk: each part's answer is computed once
        let mut q = Query::default();
        if self.implements_q(cx, t, &tr, env, &mut q) {
            return None;
        }
        let mut path = String::new();
        let mut cur = t;
        for _ in 0..=QUERY_DEPTH {
            let parts = match self.derived_parts(cx, cur, d) {
                Ok(p) => p,
                Err(why) => return Some(FlatWitness { path, part: cur, why }),
            };
            let Some((seg, next)) =
                parts.into_iter().find(|&(_, c)| !self.implements_q(cx, c, &tr, env, &mut q))
            else {
                break;
            };
            self.push_seg(&mut path, seg);
            cur = next;
        }
        Some(FlatWitness { path, part: cur, why: NotFlat::Depth })
    }

    fn push_seg(&self, path: &mut String, seg: FlatSeg) {
        let name =
            |s: Sym, i: u32| if s.is_some() { self.prog.strings.get(s).to_string() } else { i.to_string() };
        let text = match seg {
            FlatSeg::Elem => {
                path.push_str("[_]");
                return;
            }
            FlatSeg::Start => "start".to_string(),
            FlatSeg::Deref => "*".to_string(),
            FlatSeg::Index(i) => i.to_string(),
            FlatSeg::Field(d, fi) => self
                .sigs
                .adt(d)
                .and_then(|a| a.variants.first())
                .and_then(|v| v.fields.get(fi as usize))
                .map_or(fi.to_string(), |f| name(f.name, fi)),
            FlatSeg::VariantField(d, vi, fi) => {
                let v = self.sigs.adt(d).and_then(|a| a.variants.get(vi as usize));
                let vn = v.map_or(vi.to_string(), |v| name(v.name, vi));
                let f =
                    v.and_then(|v| v.fields.get(fi as usize)).map_or(fi.to_string(), |f| name(f.name, fi));
                format!("{vn}.{f}")
            }
            FlatSeg::Arg(d, i) => {
                let g = self.sigs.adt(d).and_then(|a| a.generics.get(i as usize));
                path.push('<');
                path.push_str(&g.map_or(i.to_string(), |g| name(g.name, i)));
                path.push('>');
                return;
            }
        };
        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(&text);
    }

    /// The impl of `tr` for `t` and its generic args.
    pub fn find_impl<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        env: &[GenericParam],
    ) -> Option<(DeclId, Vec<GArg>)> {
        self.find_impl_q(cx, t, tr, env, &mut Query::default())
    }

    fn find_impl_q<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        env: &[GenericParam],
        q: &mut Query,
    ) -> Option<(DeclId, Vec<GArg>)> {
        let list = self.sigs.impls_by_trait.get(&tr.trait_decl)?;
        // only the impls with the type's head or with none can apply: their positions, in list
        // order (so the first that applies is the same as in the whole list)
        let by_head = &self.sigs.trait_impls_by_head;
        let none = by_head.get(&(tr.trait_decl, None)).map_or(&[][..], |v| &v[..]);
        let own = match self.head(&*cx, t) {
            Some(h) => by_head.get(&(tr.trait_decl, Some(h))).map_or(&[][..], |v| &v[..]),
            None => &[][..],
        };
        let mut positions: Vec<u32> = own.iter().chain(none).copied().collect();
        positions.sort_unstable();
        for i in positions.into_iter().filter_map(|k| list.get(k as usize).copied()) {
            let Some(imp) = self.sigs.impl_def(i) else { continue };
            if let Some(args) = self.impl_applies_q(cx, imp, t, env, q) {
                // the trait args must match after substitution
                let Some(itr) = &imp.trait_ref else { continue };
                let ok = itr.args.len() == tr.args.len()
                    && itr.args.iter().zip(&tr.args).all(|(a, b)| match (a, b) {
                        (GArg::Type(x), GArg::Type(y)) => cx.subst(*x, &args, Some(t)) == *y,
                        (GArg::Const(x), GArg::Const(y)) => {
                            crate::mono::subst_garg(cx, GArg::Const(*x), &args) == GArg::Const(*y)
                        }
                        _ => false,
                    });
                if ok {
                    return Some((i, args));
                }
            }
        }
        None
    }

    /// If impl `imp` applies to `t`: its generic args.
    pub fn impl_applies<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        imp: &ImplDef,
        t: TypeId,
        env: &[GenericParam],
    ) -> Option<Vec<GArg>> {
        self.impl_applies_q(cx, imp, t, env, &mut Query::default())
    }

    /// [`World::impl_applies`] without the view-kind rule: the impl a view was refused by, for
    /// the diagnostic that names its param. Never selects an impl.
    pub fn impl_applies_but_views<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        imp: &ImplDef,
        t: TypeId,
        env: &[GenericParam],
    ) -> Option<Vec<GArg>> {
        self.impl_applies_q(cx, imp, t, env, &mut Query::any_kind())
    }

    /// The impl with a method `name` that would apply to `recv` (references peeled) if its params
    /// took views, and its args: why a method call on a view-holding type found nothing.
    pub fn method_refused_view<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        recv: TypeId,
        name: Sym,
        env: &[GenericParam],
    ) -> Option<(DeclId, Vec<GArg>)> {
        let mut c = recv;
        loop {
            let head = self.head(cx, c);
            let lists: [&[DeclId]; 2] = [
                head.and_then(|h| self.sigs.impls_by_head.get(&h)).map_or(&[][..], |v| &v[..]),
                &self.sigs.headless_impls[..],
            ];
            for list in lists {
                for &i in list {
                    let Some(imp) = self.sigs.impl_def(i) else { continue };
                    if !imp.methods.iter().any(|&m| self.prog.decl(m).is_some_and(|d| d.name == name)) {
                        continue;
                    }
                    if let Some(args) = self.impl_applies_but_views(cx, imp, c, env) {
                        return Some((i, args));
                    }
                }
            }
            match cx.kind(c) {
                TypeKind::Ref { inner, .. } => c = *inner,
                _ => return None,
            }
        }
    }

    /// The generic args of impl `imp` when its self type matches `t`, its bounds unchecked (the
    /// call that uses them reports the unmet ones).
    pub fn impl_head_args<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        imp: &ImplDef,
        t: TypeId,
    ) -> Option<Vec<GArg>> {
        if cx.is_error_ty(imp.self_ty) {
            return None;
        }
        let mut binds: Vec<Option<GArg>> = vec![None; imp.generics.len()];
        if !self.match_ty(cx, imp.self_ty, t, &mut binds) {
            return None;
        }
        binds.into_iter().collect()
    }

    fn impl_applies_q<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        imp: &ImplDef,
        t: TypeId,
        env: &[GenericParam],
        q: &mut Query,
    ) -> Option<Vec<GArg>> {
        if cx.is_error_ty(imp.self_ty) {
            return None;
        }
        let mut binds: Vec<Option<GArg>> = vec![None; imp.generics.len()];
        if !self.match_ty(cx, imp.self_ty, t, &mut binds) {
            return None;
        }
        let err = cx.error_ty();
        let args: Vec<GArg> = binds
            .iter()
            .enumerate()
            .map(|(k, b)| {
                b.unwrap_or(if imp.generics[k].is_const {
                    GArg::Const(ConstArg::Error)
                } else {
                    GArg::Type(err)
                })
            })
            .collect();
        // Bounds of the impl's params.
        for (k, g) in imp.generics.iter().enumerate() {
            let GArg::Type(a) = args[k] else { continue };
            // The impl's body is checked with a param without `view` holding no reference, so
            // the impl does not exist for a view.
            if !q.any_kind && !g.view && !cx.is_error_ty(a) && self.is_view(&*cx, a, env) {
                return None;
            }
            for b in &g.bounds {
                let b2 = TraitRef {
                    trait_decl: b.trait_decl,
                    args: b.args.iter().map(|&x| crate::mono::subst_garg(cx, x, &args)).collect(),
                };
                if !self.implements_q(cx, a, &b2, env, q) {
                    return None;
                }
            }
        }
        Some(args)
    }

    pub fn is_copy<I: Interner + ?Sized>(&self, cx: &mut I, t: TypeId, env: &[GenericParam]) -> bool {
        self.is_copy_q(cx, t, env, &mut Query::default())
    }

    fn is_copy_q<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        env: &[GenericParam],
        q: &mut Query,
    ) -> bool {
        match cx.kind(t).clone() {
            TypeKind::Int { .. }
            | TypeKind::Float(_)
            | TypeKind::Bool
            | TypeKind::Char
            | TypeKind::Ptr { .. }
            | TypeKind::Fn { .. }
            | TypeKind::Never
            | TypeKind::Error
            | TypeKind::Vector { .. }
            | TypeKind::Mask { .. } => true,
            TypeKind::Ref { mutable, .. } => !mutable,
            TypeKind::Str | TypeKind::Slice { .. } | TypeKind::SelfTy | TypeKind::ConstParam(_) => false,
            TypeKind::Array { elem, .. } | TypeKind::Range { elem, .. } => self.is_copy_q(cx, elem, env, q),
            TypeKind::Tuple(ts) => {
                let copy = self.sigs.lang.copy;
                if let Some(r) = self.copy_known(cx, t, q) {
                    return r;
                }
                let low = std::mem::replace(&mut q.low, usize::MAX);
                let mark = q.open_low();
                let start = q.pending.len();
                let r = ts.iter().all(|&x| self.is_copy_q(cx, x, env, q));
                let own = q.own_low(mark);
                // not a goal of its own: what it used is settled by the goals that set it
                let free_pos = q.low == usize::MAX;
                let done = q.settle(Some((t, copy)), r, free_pos, own, start, false);
                self.sigs.queries.put(&*cx, &done, q.any_kind);
                q.low = q.low.min(low);
                r
            }
            TypeKind::Param(i) => {
                let copy = self.sigs.lang.copy;
                env.get(i as usize).is_some_and(|g| g.bounds.iter().any(|b| b.trait_decl == copy))
            }
            TypeKind::Adt { .. } => {
                let copy = self.sigs.lang.copy;
                if copy.is_none() {
                    return false;
                }
                if let Some(r) = self.copy_known(cx, t, q) {
                    return r;
                }
                let tr = TraitRef { trait_decl: copy, args: Vec::new() };
                let Some(mark) = q.enter_goal((t, tr.clone())) else { return false };
                let low = std::mem::replace(&mut q.low, usize::MAX);
                let start = q.pending.len();
                let r = self.find_impl_q(cx, t, &tr, env, q).is_some();
                let own = q.leave_goal(mark);
                let done = q.settle(Some((t, copy)), r, q.low == usize::MAX, own, start, false);
                self.sigs.queries.put(&*cx, &done, q.any_kind);
                q.low = q.low.min(low);
                r
            }
        }
    }

    /// The known Copy answer for `t`: this query's, or the run's ([`QueryCache`]).
    fn copy_known<I: Interner + ?Sized>(&self, cx: &I, t: TypeId, q: &Query) -> Option<bool> {
        let copy = self.sigs.lang.copy;
        q.memo.get(&(t, copy)).copied().or_else(|| self.sigs.queries.get(cx, t, copy, q.any_kind))
    }

    /// [`crate::kinds::is_view_in`]: a view in the scope of `env` (a `T: view` param counts).
    pub fn is_view<I: Interner + ?Sized>(&self, cx: &I, t: TypeId, env: &[GenericParam]) -> bool {
        crate::kinds::is_view_in(self.prog, cx, t, env)
    }

    /// Trait methods named `name` reachable from a generic param's bounds:
    /// (trait decl, method decl, trait args). Two bounds of one trait with other args
    /// (`T: Nx<u8> + Nx<u16>`) are two entries, as two impls of one trait are.
    fn bound_methods(&self, env: &[GenericParam], param: u32, name: Sym) -> Vec<(DeclId, DeclId, Vec<GArg>)> {
        let mut out = Vec::new();
        let Some(g) = env.get(param as usize) else { return out };
        for b in &g.bounds {
            if let Some(m) = self.prog.member(b.trait_decl, name)
                && self.prog.decl(m).is_some_and(|x| x.kind == DeclKind::TraitMethod)
                && !out.iter().any(|(t, _, a)| *t == b.trait_decl && *a == b.args)
            {
                out.push((b.trait_decl, m, b.args.clone()));
            }
        }
        out
    }

    /// Method lookup for receiver type `recv`.
    /// `from`: the package of the call; only impls of packages it sees take part (see
    /// [`incoherent`]).
    pub fn lookup_method<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        recv: TypeId,
        name: Sym,
        env: &[GenericParam],
        from: Option<crate::PackageId>,
    ) -> Lookup {
        let mut c = recv;
        for level in 0..64u32 {
            if cx.is_error_ty(c) {
                return Lookup::NotFound;
            }
            if let TypeKind::Param(p) = cx.kind(c).clone() {
                let ms = self.bound_methods(env, p, name);
                match ms.as_slice() {
                    [(_, m, args)] => {
                        return Lookup::Found {
                            m: MethodRef::Bound { method: *m, trait_args: args.clone(), self_ty: c },
                            level,
                            recv: c,
                        };
                    }
                    [a, b, ..] => {
                        let r = |x: &(DeclId, DeclId, Vec<GArg>)| MethodRef::Bound {
                            method: x.1,
                            trait_args: x.2.clone(),
                            self_ty: c,
                        };
                        return Lookup::Ambiguous {
                            traits: [a.0, b.0],
                            methods: [a.1, b.1],
                            refs: Box::new([r(a), r(b)]),
                            recv: c,
                        };
                    }
                    [] => {
                        if !matches!(cx.kind(c), TypeKind::Ref { .. }) {
                            return Lookup::NotInBounds { param: p };
                        }
                    }
                }
            }
            if let Some(r) = self.lookup_at(cx, c, name, env, from) {
                return match r {
                    Ok(m) => Lookup::Found { m, level, recv: c },
                    Err((traits, methods, refs)) => Lookup::Ambiguous { traits, methods, refs, recv: c },
                };
            }
            if let Some(b) = crate::builtins::builtin_method(cx, c, self.prog.strings.get(name)) {
                return Lookup::Found { m: MethodRef::Builtin(b), level, recv: c };
            }
            match cx.kind(c) {
                TypeKind::Ref { inner, .. } => c = *inner,
                _ => return Lookup::NotFound,
            }
        }
        Lookup::NotFound
    }

    /// The methods of the inherent impls that apply to exactly type `c` (impls of the packages
    /// `from` sees, as in method lookup), in declaration order.
    pub fn inherent_methods<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        c: TypeId,
        env: &[GenericParam],
        from: Option<crate::PackageId>,
    ) -> Vec<DeclId> {
        let mut out = Vec::new();
        let Some(head) = self.head(cx, c) else { return out };
        let Some(list) = self.sigs.impls_by_head.get(&head) else { return out };
        for &i in list {
            let Some(imp) = self.sigs.impl_def(i) else { continue };
            let seen = match (from, package_of(self.prog, i)) {
                (Some(f), Some(p)) => package_sees(self.prog, f, p),
                _ => true,
            };
            if imp.trait_ref.is_some() || !seen || self.impl_applies(cx, imp, c, env).is_none() {
                continue;
            }
            out.extend(imp.methods.iter().copied());
        }
        out
    }

    /// Methods named `name` of exactly type `c`: inherent first, then trait impls.
    #[allow(clippy::type_complexity)]
    fn lookup_at<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        c: TypeId,
        name: Sym,
        env: &[GenericParam],
        from: Option<crate::PackageId>,
    ) -> Option<Result<MethodRef, ([DeclId; 2], [DeclId; 2], Box<[MethodRef; 2]>)>> {
        let seen = |i: DeclId| match (from, package_of(self.prog, i)) {
            (Some(f), Some(p)) => package_sees(self.prog, f, p),
            _ => true,
        };
        let head = self.head(cx, c);
        let lists: [&[DeclId]; 2] = [
            head.and_then(|h| self.sigs.impls_by_head.get(&h)).map_or(&[][..], |v| &v[..]),
            &self.sigs.headless_impls[..],
        ];
        // inherent
        for list in lists {
            for &i in list {
                let Some(imp) = self.sigs.impl_def(i) else { continue };
                if imp.trait_ref.is_some() || !seen(i) {
                    continue;
                }
                let Some(m) =
                    imp.methods.iter().copied().find(|&m| self.prog.decl(m).is_some_and(|d| d.name == name))
                else {
                    continue;
                };
                if let Some(args) = self.impl_applies(cx, imp, c, env) {
                    return Some(Ok(MethodRef::Impl { method: m, impl_decl: i, impl_args: args }));
                }
            }
        }
        // trait impls
        let mut found: Vec<(DeclId, MethodRef, DeclId)> = Vec::new();
        for list in lists {
            for &i in list {
                let Some(imp) = self.sigs.impl_def(i) else { continue };
                let Some(tr) = &imp.trait_ref else { continue };
                if !seen(i) {
                    continue;
                }
                let Some(m) =
                    imp.methods.iter().copied().find(|&m| self.prog.decl(m).is_some_and(|d| d.name == name))
                else {
                    continue;
                };
                if let Some(args) = self.impl_applies(cx, imp, c, env) {
                    // two impls of one trait (other trait args) for this type: the call names
                    // neither, so it is ambiguous; the labels then show the two impls' methods
                    let same = found.iter().any(|(t, _, _)| *t == tr.trait_decl);
                    if same {
                        // the same trait args again: overlapping impls, E0903 at the impl
                        let mine: Vec<GArg> =
                            tr.args.iter().map(|&a| crate::mono::subst_garg(cx, a, &args)).collect();
                        let dup = found.iter().any(|(t, f, _)| {
                            *t == tr.trait_decl
                                && match f {
                                    MethodRef::Impl { impl_decl, impl_args, .. } => self
                                        .sigs
                                        .impl_def(*impl_decl)
                                        .and_then(|x| x.trait_ref.as_ref())
                                        .is_some_and(|ftr| {
                                            ftr.args
                                                .iter()
                                                .map(|&a| crate::mono::subst_garg(cx, a, impl_args))
                                                .eq(mine.iter().copied())
                                        }),
                                    _ => false,
                                }
                        });
                        if dup {
                            continue;
                        }
                    }
                    let tm = if same { m } else { self.prog.member(tr.trait_decl, name).unwrap_or(m) };
                    if same
                        && let Some(first) = found.iter_mut().find(|(t, _, _)| *t == tr.trait_decl)
                        && let MethodRef::Impl { method, .. } = first.1
                    {
                        first.2 = method;
                    }
                    found.push((
                        tr.trait_decl,
                        MethodRef::Impl { method: m, impl_decl: i, impl_args: args },
                        tm,
                    ));
                }
            }
        }
        match found.len() {
            0 => None,
            1 => found.pop().map(|(_, m, _)| Ok(m)),
            _ => {
                found.sort_by_key(|(t, _, _)| *t);
                let refs = Box::new([found[0].1.clone(), found[1].1.clone()]);
                Some(Err(([found[0].0, found[1].0], [found[0].2, found[1].2], refs)))
            }
        }
    }

    /// The signature of a method ref, instantiated: params, ret (with `Self` replaced), and the
    /// generic args of the container (method generics stay as params, offset after them).
    pub fn method_sig(&self, m: &MethodRef) -> Option<(&'a FnSig, Vec<GArg>, Option<TypeId>)> {
        match m {
            MethodRef::Impl { method, impl_args, .. } => {
                Some((self.sigs.fn_sig(*method)?, impl_args.clone(), None))
            }
            MethodRef::Bound { method, trait_args, self_ty } => {
                Some((self.sigs.fn_sig(*method)?, trait_args.clone(), Some(*self_ty)))
            }
            MethodRef::Builtin(_) => None,
        }
    }

    /// `m` as the `next` a `for` loop calls, which passes the source and nothing else: exactly
    /// `fn next(&mut self) -> Option<X>` (no other param, no generic param of its own); its
    /// element type `X`. Every other shape is no iterator's `next`: the first part that differs.
    /// The shape only: the loop's call is then checked as `recv.next()` is (features, privacy,
    /// `unsafe`, ..).
    pub fn for_next_elem<I: Interner + ?Sized>(&self, cx: &mut I, m: &MethodRef) -> Result<TypeId, NotNext> {
        let (sig, args, self_ty) = self.method_sig(m).ok_or(NotNext::Builtin)?;
        if sig.self_kind != Some(crate::sigs::SelfKind::RefMut) {
            return Err(NotNext::Receiver);
        }
        if sig.params.len() != 1 {
            return Err(NotNext::Params);
        }
        if sig.generics.len() != sig.n_parent as usize {
            return Err(NotNext::Generics);
        }
        let ret = cx.subst(sig.ret, &args, self_ty);
        match cx.kind(ret) {
            TypeKind::Adt { def, args } if *def == self.sigs.lang.option => match args.first() {
                Some(GArg::Type(x)) => Ok(*x),
                _ => Err(NotNext::Result),
            },
            _ => Err(NotNext::Result),
        }
    }

    /// The trait args of the trait instance that supplies `m` (an impl's, its args substituted,
    /// or a bound's); `None` for an inherent or built-in method.
    pub fn trait_args_of<I: Interner + ?Sized>(&self, cx: &mut I, m: &MethodRef) -> Option<Vec<GArg>> {
        match m {
            MethodRef::Impl { impl_decl, impl_args, .. } => {
                let tr = self.sigs.impl_def(*impl_decl)?.trait_ref.as_ref()?;
                Some(tr.args.iter().map(|&a| crate::mono::subst_garg(cx, a, impl_args)).collect())
            }
            MethodRef::Bound { trait_args, .. } => Some(trait_args.clone()),
            MethodRef::Builtin(_) => None,
        }
    }
}

fn match_const(p: ConstArg, x: ConstArg, binds: &mut [Option<GArg>]) -> bool {
    match p {
        ConstArg::Param(i) => match binds.get_mut(i as usize) {
            Some(slot @ None) => {
                *slot = Some(GArg::Const(x));
                true
            }
            Some(Some(GArg::Const(prev))) => *prev == x,
            _ => false,
        },
        _ => p == x || x == ConstArg::Error,
    }
}

/// Whether the scalar operator trait `name` is built in for `t`.
pub fn builtin_op_impl<I: Interner + ?Sized>(cx: &I, name: &str, t: TypeId) -> bool {
    let k = cx.kind(t);
    let int = matches!(k, TypeKind::Int { .. });
    let signed_int = matches!(k, TypeKind::Int { signed: true, .. });
    let float = matches!(k, TypeKind::Float(_));
    let boolean = matches!(k, TypeKind::Bool);
    match name {
        "Add" | "Sub" | "Mul" | "Div" | "Rem" => int || float,
        "WrapAdd" | "WrapSub" | "WrapMul" | "SatAdd" | "SatSub" | "SatMul" | "Shl" | "WrapShl" | "Shr" => int,
        "BitAnd" | "BitOr" | "BitXor" => int || boolean,
        "Neg" => signed_int || float,
        "Not" => int,
        _ => false,
    }
}

/// Impl conformance and marker checks over every impl (sequential, decl order).
pub(crate) fn check_impls(
    prog: &Program,
    sigs: &Sigs,
    tys: &mut TypeTable,
    brackets: &[Brackets],
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let w = World { prog, sigs };
    let mut impls: Vec<DeclId> =
        (0..sigs.decls.len() as u32).map(DeclId).filter(|&d| sigs.impl_def(d).is_some()).collect();
    impls.sort();
    // the impls seen so far, per trait and head (`None`: no head), each with its position: two
    // impls with different heads never overlap, so each impl meets only those with its head or
    // none (all of them for one without a head)
    let mut seen: FxHashMap<(DeclId, Option<TypeHead>), Vec<(u32, DeclId)>> = FxHashMap::default();
    let mut seen_all: FxHashMap<DeclId, Vec<(u32, DeclId)>> = FxHashMap::default();
    for d in impls {
        let Some(imp) = sigs.impl_def(d) else { continue };
        let Some(decl) = prog.decl(d) else { continue };
        let Some(icx) = brackets.get(decl.item.idx()).and_then(|b| ItemCx::new(prog, decl.item, b)) else {
            continue;
        };
        let impl_span = decl.name_span;
        let names = NameCx { prog, generics: &imp.generics };
        let self_text = types::display(tys, &names, imp.self_ty);
        // E0421: a use of the impl binds its params from the self type alone (method lookup
        // knows only the receiver), so a param that does not occur there would get no type.
        let free = unconstrained_params(tys, imp);
        if !free.is_empty() {
            for k in free {
                let g = &imp.generics[k as usize];
                let pn = prog.strings.get(g.name).to_string();
                out.extend(mk(
                    "E0421",
                    None,
                    &[("param", &pn), ("type", &self_text)],
                    impl_span,
                    &[("param", g.name_span)],
                ));
            }
            continue;
        }
        let Some(tr) = &imp.trait_ref else {
            if let Some(dg) = incoherent(prog, sigs, tys, d, imp, &self_text) {
                out.push(dg);
            }
            continue;
        };
        // E1408: Flat and Frozen come from the type's structure only, so the impl is rejected
        // whole and no other impl check applies to it.
        let lang = &sigs.lang;
        if tr.trait_decl.is_some() && (tr.trait_decl == lang.flat || tr.trait_decl == lang.frozen) {
            let name = if tr.trait_decl == lang.flat { "Flat" } else { "Frozen" };
            out.extend(mk("E1408", None, &[("trait", name), ("type", &self_text)], impl_span, &[]));
            continue;
        }
        let Some(trait_def) = sigs.trait_def(tr.trait_decl) else { continue };
        let trait_path = prog.decl_path(tr.trait_decl);
        // a `Drop` impl has the stricter E0419 below
        if tr.trait_decl != lang.drop
            && let Some(dg) = incoherent(prog, sigs, tys, d, imp, &self_text)
        {
            out.push(dg);
            continue;
        }
        // E0810: an unsafe trait (Send and Sync are declared unsafe in std.core) is implemented
        // only by an `unsafe("..") impl`; otherwise safe code could declare any type Sync.
        if let Some(td) = prog.decl(tr.trait_decl)
            && td.flags.has(DeclFlags::UNSAFE)
            && !decl.flags.has(DeclFlags::UNSAFE)
        {
            out.extend(mk("E0810", None, &[("trait", &trait_path)], impl_span, &[("trait", td.name_span)]));
        }
        // E0907: `pub` on a trait impl method.
        for &m in &imp.methods {
            let Some(md) = prog.decl(m) else { continue };
            if md.is_pub()
                && let Some(t) = pub_token(&icx, md.node)
            {
                let n = prog.strings.get(md.name);
                // the edit ends where the next token starts: no blank is left behind
                let at = icx.tok_span(t);
                let next = icx.tok_span(t + 1).offset;
                let cut = Span::new(at.file, at.offset, next.saturating_sub(at.offset).max(at.len));
                out.extend(mk("E0907", None, &[("name", n)], at, &[("cut", cut)]));
            }
        }
        // E0906: missing and extra methods; E0905: signature mismatch.
        for &tm in &trait_def.methods {
            let Some(tmd) = prog.decl(tm) else { continue };
            let found =
                imp.methods.iter().copied().find(|&m| prog.decl(m).is_some_and(|x| x.name == tmd.name));
            match found {
                None => {
                    let n = prog.strings.get(tmd.name);
                    out.extend(mk(
                        "E0906",
                        None,
                        &[("name", n), ("trait", &trait_path)],
                        impl_span,
                        &[("trait", tmd.name_span)],
                    ));
                }
                Some(m) => {
                    if !sig_matches(&w, tys, imp, tr, tm, m)
                        && let Some(md) = prog.decl(m)
                    {
                        let n = prog.strings.get(md.name);
                        let sig_text = trait_sig_text(prog, brackets, tm);
                        out.extend(mk(
                            "E0905",
                            None,
                            &[("name", n), ("signature", &sig_text)],
                            md.name_span,
                            &[("trait", tmd.name_span)],
                        ));
                    }
                }
            }
        }
        for &m in &imp.methods {
            let Some(md) = prog.decl(m) else { continue };
            if !trait_def.methods.iter().any(|&tm| prog.decl(tm).is_some_and(|x| x.name == md.name)) {
                let n = prog.strings.get(md.name);
                out.extend(mk(
                    "E0906",
                    Some("extra"),
                    &[("name", n), ("trait", &trait_path)],
                    md.name_span,
                    &[],
                ));
            }
        }
        let lang = &sigs.lang;
        // E0415: Copy needs Copy fields.
        if tr.trait_decl == lang.copy
            && let TypeKind::Adt { def, args } = tys.kind(imp.self_ty).clone()
            && let Some(adt) = sigs.adt(def)
        {
            'fields: for v in &adt.variants {
                for (fi, f) in v.fields.iter().enumerate() {
                    let ft = tys.subst(f.ty, &args, None);
                    if tys.is_error_ty(ft) || w.is_copy(tys, ft, &imp.generics) {
                        continue;
                    }
                    let fname =
                        if f.name.is_some() { prog.strings.get(f.name).to_string() } else { fi.to_string() };
                    let ftext = types::display(tys, &names, ft);
                    out.extend(mk(
                        "E0415",
                        None,
                        &[("type", &self_text), ("field", &fname), ("field_type", &ftext)],
                        impl_span,
                        &[("field", f.span)],
                    ));
                    break 'fields;
                }
            }
        }
        // E0422: an operator is a call not written as one, so its operands must never move: an
        // operator trait's self type (also its rhs and result) is Copy in every instance the impl
        // covers, i.e. Copy under the impl's own bounds.
        if lang.ops.values().any(|&t| t == tr.trait_decl)
            && !tys.is_error_ty(imp.self_ty)
            && !w.is_copy(tys, imp.self_ty, &imp.generics)
        {
            let name = lang.ops.iter().find(|&(_, &t)| t == tr.trait_decl).map_or("", |(n, _)| *n);
            let forms = sigs.op_forms(prog, name);
            out.extend(mk(
                "E0422",
                None,
                &[("trait", &trait_path), ("type", &self_text), ("sig", &forms.sig), ("call", &forms.call)],
                impl_span,
                &[],
            ));
        }
        // E0419: drop glue runs the destructors of ADTs only, and a package adds none to the
        // types of another (their values would drop differently in each package).
        // E0416: Drop on a Copy type. Copy and Drop exclude each other on every instance, so
        // the impls conflict when one instance of the Drop impl's head may be Copy: a Copy impl
        // whose head unifies with it.
        if tr.trait_decl == lang.drop && lang.drop.is_some() && !tys.is_error_ty(imp.self_ty) {
            let package_of = |m| prog.module(m).map(|x| x.package);
            let own_adt = match tys.kind(imp.self_ty) {
                TypeKind::Adt { def, .. } => {
                    prog.decl(*def).is_some_and(|x| package_of(x.module) == package_of(decl.module))
                }
                _ => false,
            };
            if !own_adt {
                out.extend(mk("E0419", None, &[("type", &self_text)], impl_span, &[]));
                // the impl takes part in nothing else: no overlap reports for or against it
                continue;
            } else if let Some(ci) = copy_impl_overlapping(&w, tys, imp) {
                let roles: Vec<(&str, Span)> =
                    prog.decl(ci).map(|x| ("copy", x.name_span)).into_iter().collect();
                out.extend(mk("E0416", None, &[("type", &self_text)], impl_span, &roles));
            }
        }
        // E0414 case `drop`: the destructor of a type runs for every instance, so its impl must
        // take a view wherever the type does.
        if tr.trait_decl == lang.drop
            && lang.drop.is_some()
            && let TypeKind::Adt { def, args } = tys.kind(imp.self_ty).clone()
        {
            let adt_gens = sigs.generics(def);
            let mut named: Vec<u32> = Vec::new();
            for (i, a) in args.iter().enumerate() {
                let GArg::Type(a) = *a else { continue };
                if !adt_gens.get(i).is_some_and(|g| g.view) {
                    continue;
                }
                let mut stack = vec![a];
                let mut kids = Vec::new();
                let mut seen = rustc_hash::FxHashSet::default();
                while let Some(x) = stack.pop() {
                    if !seen.insert(x) {
                        continue;
                    }
                    if let TypeKind::Param(k) = *tys.kind(x) {
                        if imp.generics.get(k as usize).is_some_and(|g| !g.view) && !named.contains(&k) {
                            named.push(k);
                        }
                        continue;
                    }
                    kids.clear();
                    tys.kind(x).children(&mut kids);
                    stack.extend_from_slice(&kids);
                }
            }
            named.sort_unstable();
            let adt_path = prog.decl_path(def);
            for k in named {
                let g = &imp.generics[k as usize];
                let pn = prog.strings.get(g.name).to_string();
                out.extend(mk(
                    "E0414",
                    Some("drop"),
                    &[("param", &pn), ("type", &self_text), ("generic", &adt_path)],
                    impl_span,
                    &[("param", g.name_span)],
                ));
            }
        }
        // E0903: overlapping impls of one trait; a built-in operator impl counts as one.
        if let Some(op) = lang.ops.iter().find(|(_, t)| **t == tr.trait_decl).map(|(n, _)| *n)
            && builtin_op_impl(tys, op, imp.self_ty)
        {
            out.extend(mk("E0903", None, &[("trait", &trait_path), ("type", &self_text)], impl_span, &[]));
            continue;
        }
        let mut prev: Vec<(u32, DeclId)> = match imp.head {
            Some(h) => {
                let own = seen.get(&(tr.trait_decl, Some(h))).map_or(&[][..], |v| &v[..]);
                let none = seen.get(&(tr.trait_decl, None)).map_or(&[][..], |v| &v[..]);
                own.iter().chain(none).copied().collect()
            }
            None => seen_all.get(&tr.trait_decl).cloned().unwrap_or_default(),
        };
        prev.sort_unstable();
        for &(_, o) in prev.iter() {
            let Some(other) = sigs.impl_def(o) else { continue };
            if tys.is_error_ty(other.self_ty) || tys.is_error_ty(imp.self_ty) {
                continue;
            }
            if overlap(tys, other, imp) {
                let od = prog.decl(o).map_or(impl_span, |x| x.name_span);
                out.extend(mk(
                    "E0903",
                    None,
                    &[("trait", &trait_path), ("type", &self_text)],
                    impl_span,
                    &[("other", od)],
                ));
                break;
            }
        }
        let all = seen_all.entry(tr.trait_decl).or_default();
        let pos = all.len() as u32;
        all.push((pos, d));
        seen.entry((tr.trait_decl, imp.head)).or_default().push((pos, d));
    }
    out
}

/// The generic params of `imp` (type and const) that do not occur in its self type.
fn unconstrained_params(tys: &TypeTable, imp: &ImplDef) -> Vec<u32> {
    if tys.is_error_ty(imp.self_ty) {
        return Vec::new();
    }
    let mut used = vec![false; imp.generics.len()];
    let mut mark = |i: u32| {
        if let Some(u) = used.get_mut(i as usize) {
            *u = true;
        }
    };
    let mut stack = vec![imp.self_ty];
    let mut kids = Vec::new();
    // each distinct component once: types share them
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        let k = tys.kind(t);
        match k {
            TypeKind::Param(i) | TypeKind::ConstParam(i) => mark(*i),
            TypeKind::Array { len: ConstArg::Param(i), .. } => mark(*i),
            TypeKind::Adt { args, .. } => {
                for a in args.iter() {
                    if let GArg::Const(ConstArg::Param(i)) = a {
                        mark(*i);
                    }
                }
            }
            _ => {}
        }
        kids.clear();
        k.children(&mut kids);
        stack.extend_from_slice(&kids);
    }
    (0..imp.generics.len() as u32).filter(|&i| !used[i as usize]).collect()
}

/// The package that declares `d`.
pub(crate) fn package_of(prog: &Program, d: DeclId) -> Option<crate::PackageId> {
    prog.decl(d).and_then(|x| prog.module(x.module)).map(|m| m.package)
}

/// Whether package `from` sees the items of package `to`: itself or a dependency, at any depth.
pub(crate) fn package_sees(prog: &Program, from: crate::PackageId, to: crate::PackageId) -> bool {
    if from == to {
        return true;
    }
    let mut seen = vec![from];
    let mut stack = vec![from];
    while let Some(p) = stack.pop() {
        let Some(pk) = prog.packages.get(p.idx()) else { continue };
        for &(_, q) in &pk.deps {
            if q == to {
                return true;
            }
            if !seen.contains(&q) {
                seen.push(q);
                stack.push(q);
            }
        }
    }
    false
}

/// E0420, coherence by package: an inherent impl is in the package that declares its self type;
/// a trait impl is in the package of the trait or in that of the self type's head. A struct or
/// enum belongs to the package that declares it; the built-in types (scalars, vectors, masks,
/// `str`, references, raw pointers, slices, arrays, tuples, fn types, ranges) belong to `std`; a
/// generic param belongs to none. So for one type and one trait the impls a package may see come
/// from that package and its dependencies only, and one package never changes what a method
/// call in another one means.
fn incoherent(
    prog: &Program,
    sigs: &Sigs,
    tys: &TypeTable,
    d: DeclId,
    imp: &ImplDef,
    self_text: &str,
) -> Option<Diagnostic> {
    if tys.is_error_ty(imp.self_ty) {
        return None;
    }
    let at = prog.decl(d)?.name_span;
    let pkg = package_of(prog, d);
    let std_pkg = if sigs.lang.copy.is_some() { package_of(prog, sigs.lang.copy) } else { None };
    let head_local = match tys.kind(imp.self_ty) {
        TypeKind::Adt { def, .. } => package_of(prog, *def) == pkg,
        TypeKind::Param(_) | TypeKind::ConstParam(_) | TypeKind::SelfTy | TypeKind::Error => false,
        _ => pkg.is_some() && std_pkg == pkg,
    };
    match &imp.trait_ref {
        None if !head_local => mk("E0420", Some("inherent"), &[("type", self_text)], at, &[]),
        Some(tr) if !head_local && package_of(prog, tr.trait_decl) != pkg => {
            let path = prog.decl_path(tr.trait_decl);
            mk("E0420", None, &[("trait", &path), ("type", self_text)], at, &[])
        }
        _ => None,
    }
}

/// Whether two impls of one trait may both apply to one type and one list of trait args: their
/// self types and trait args unify together, the params of both impls as variables (bounds are
/// not consulted).
fn overlap(tys: &mut TypeTable, a: &ImplDef, b: &ImplDef) -> bool {
    let (Some(ta), Some(tb)) = (&a.trait_ref, &b.trait_ref) else { return false };
    heads_unify(tys, a, b, Some((&ta.args, &tb.args))).is_some()
}

/// A most general unifier of the heads of impls `a` and `b`: `a`'s generic params are the
/// variables `0..na`, `b`'s are renumbered to `na..`. `trait_args`: the two impls' trait args,
/// unified along with the self types. `None`: no type is an instance of both heads. Past the step
/// limit the answer is a unifier with the bindings found so far ("may unify").
fn heads_unify(
    tys: &mut TypeTable,
    a: &ImplDef,
    b: &ImplDef,
    trait_args: Option<(&[GArg], &[GArg])>,
) -> Option<Vec<Option<GArg>>> {
    let na = a.generics.len() as u32;
    let shift: Vec<GArg> = b
        .generics
        .iter()
        .enumerate()
        .map(|(i, g)| {
            if g.is_const {
                GArg::Const(ConstArg::Param(na + i as u32))
            } else {
                GArg::Type(tys.mk_param(na + i as u32))
            }
        })
        .collect();
    let mut u = Unifier { binds: vec![None; a.generics.len() + b.generics.len()], done: Default::default() };
    let bt = tys.subst(b.self_ty, &shift, None);
    if !u.unify(tys, a.self_ty, bt) {
        return None;
    }
    if let Some((xa, xb)) = trait_args {
        if xa.len() != xb.len() {
            return None;
        }
        for (&p, &x) in xa.iter().zip(xb) {
            let ok = match (p, crate::mono::subst_garg(tys, x, &shift)) {
                (GArg::Type(p), GArg::Type(x)) => u.unify(tys, p, x),
                (GArg::Const(p), GArg::Const(x)) => u.unify_const(p, x),
                _ => false,
            };
            if !ok {
                return None;
            }
        }
    }
    Some(u.binds)
}

/// A Copy impl whose head unifies with the head of Drop impl `drop`. Bounds are not consulted:
/// the rule is about the heads, so it does not depend on which types implement what.
fn copy_impl_overlapping(w: &World<'_>, tys: &mut TypeTable, drop: &ImplDef) -> Option<DeclId> {
    let list = w.sigs.impls_by_trait.get(&w.sigs.lang.copy)?;
    list.iter().copied().find(|&ci| {
        w.sigs.impl_def(ci).is_some_and(|cimp| {
            !tys.is_error_ty(cimp.self_ty) && heads_unify(tys, drop, cimp, None).is_some()
        })
    })
}

/// Syntactic unification with an occurs check over the variables `binds` (type and const
/// params; a variable is bound at most once and bindings never form a cycle). Exact and linear
/// in the distinct components: a pair once unified stays unified (bindings only grow), so it is
/// not unified again (`done`), and the occurs check visits each component once.
struct Unifier {
    binds: Vec<Option<GArg>>,
    done: rustc_hash::FxHashSet<(TypeId, TypeId)>,
}

impl Unifier {
    fn resolve(&self, cx: &TypeTable, mut t: TypeId) -> TypeId {
        while let TypeKind::Param(i) = cx.kind(t) {
            match self.binds.get(*i as usize) {
                Some(Some(GArg::Type(x))) => t = *x,
                _ => break,
            }
        }
        t
    }

    fn resolve_const(&self, mut c: ConstArg) -> ConstArg {
        while let ConstArg::Param(i) = c {
            match self.binds.get(i as usize) {
                Some(Some(GArg::Const(x))) => c = *x,
                _ => break,
            }
        }
        c
    }

    /// Whether variable `v` occurs in `t` (bindings followed).
    fn occurs(&mut self, cx: &TypeTable, v: u32, t: TypeId) -> bool {
        let mut stack = vec![t];
        let mut kids = Vec::new();
        let mut seen = rustc_hash::FxHashSet::default();
        while let Some(x) = stack.pop() {
            let x = self.resolve(cx, x);
            if !seen.insert(x) {
                continue;
            }
            if let TypeKind::Param(i) = cx.kind(x) {
                if *i == v {
                    return true;
                }
                continue;
            }
            kids.clear();
            cx.kind(x).children(&mut kids);
            stack.extend_from_slice(&kids);
        }
        false
    }

    fn bind(&mut self, cx: &TypeTable, v: u32, t: TypeId) -> bool {
        // a param outside both impls' generics cannot be bound: compare it as a rigid type
        if !matches!(self.binds.get(v as usize), Some(None)) {
            return false;
        }
        if self.occurs(cx, v, t) {
            return false;
        }
        self.binds[v as usize] = Some(GArg::Type(t));
        true
    }

    fn unify_const(&mut self, p: ConstArg, x: ConstArg) -> bool {
        let (p, x) = (self.resolve_const(p), self.resolve_const(x));
        if p == x || p == ConstArg::Error || x == ConstArg::Error {
            return true;
        }
        for (v, other) in [(p, x), (x, p)] {
            if let ConstArg::Param(i) = v
                && let Some(slot @ None) = self.binds.get_mut(i as usize)
            {
                *slot = Some(GArg::Const(other));
                return true;
            }
        }
        false
    }

    /// `false`: `a` and `b` have no common instance.
    fn unify(&mut self, cx: &TypeTable, a: TypeId, b: TypeId) -> bool {
        let mut stack = vec![(a, b)];
        while let Some((p, x)) = stack.pop() {
            if !self.done.insert((p, x)) {
                continue;
            }
            let (p, x) = (self.resolve(cx, p), self.resolve(cx, x));
            if p == x || cx.is_error_ty(p) || cx.is_error_ty(x) {
                continue;
            }
            match (cx.kind(p), cx.kind(x)) {
                (TypeKind::Param(i), _) if (*i as usize) < self.binds.len() => {
                    if !self.bind(cx, *i, x) {
                        return false;
                    }
                }
                (_, TypeKind::Param(i)) if (*i as usize) < self.binds.len() => {
                    if !self.bind(cx, *i, p) {
                        return false;
                    }
                }
                (TypeKind::Ref { mutable: m1, inner: i1 }, TypeKind::Ref { mutable: m2, inner: i2 })
                | (TypeKind::Ptr { mutable: m1, inner: i1 }, TypeKind::Ptr { mutable: m2, inner: i2 }) => {
                    if m1 != m2 {
                        return false;
                    }
                    stack.push((*i1, *i2));
                }
                (TypeKind::Array { elem: e1, len: l1 }, TypeKind::Array { elem: e2, len: l2 }) => {
                    let (e1, e2, l1, l2) = (*e1, *e2, *l1, *l2);
                    if !self.unify_const(l1, l2) {
                        return false;
                    }
                    stack.push((e1, e2));
                }
                (TypeKind::Slice { elem: e1, align: a1 }, TypeKind::Slice { elem: e2, align: a2 }) => {
                    if a1 != a2 {
                        return false;
                    }
                    stack.push((*e1, *e2));
                }
                (
                    TypeKind::Range { elem: e1, inclusive: i1 },
                    TypeKind::Range { elem: e2, inclusive: i2 },
                ) => {
                    if i1 != i2 {
                        return false;
                    }
                    stack.push((*e1, *e2));
                }
                (TypeKind::Tuple(ta), TypeKind::Tuple(tb)) => {
                    if ta.len() != tb.len() {
                        return false;
                    }
                    stack.extend(ta.iter().copied().zip(tb.iter().copied()));
                }
                (TypeKind::Adt { def: d1, args: a1 }, TypeKind::Adt { def: d2, args: a2 }) => {
                    if d1 != d2 || a1.len() != a2.len() {
                        return false;
                    }
                    let pairs: Vec<(GArg, GArg)> = a1.iter().copied().zip(a2.iter().copied()).collect();
                    for (ga, gb) in pairs {
                        match (ga, gb) {
                            (GArg::Type(ta), GArg::Type(tb)) => stack.push((ta, tb)),
                            (GArg::Const(ca), GArg::Const(cb)) => {
                                if !self.unify_const(ca, cb) {
                                    return false;
                                }
                            }
                            _ => return false,
                        }
                    }
                }
                (
                    TypeKind::Fn { params: p1, ret: r1, features: f1, deny: d1 },
                    TypeKind::Fn { params: p2, ret: r2, features: f2, deny: d2 },
                ) => {
                    if f1 != f2 || d1 != d2 || p1.len() != p2.len() {
                        return false;
                    }
                    stack.extend(p1.iter().copied().zip(p2.iter().copied()));
                    stack.push((*r1, *r2));
                }
                (TypeKind::Vector { lane: l1, n: n1 }, TypeKind::Vector { lane: l2, n: n2 }) => {
                    if n1 != n2 {
                        return false;
                    }
                    stack.push((*l1, *l2));
                }
                _ => return false,
            }
        }
        true
    }
}

/// The `pub` token before a method's `fn` (or its `unsafe(..)`).
fn pub_token(icx: &ItemCx<'_>, fn_node: u32) -> Option<u32> {
    let a = icx.a(fn_node);
    // `pub fn`, `pub unsafe("..") fn`
    for back in 1..=5u32 {
        let t = a.checked_sub(back)?;
        if icx.is(t, Term::KwPub) {
            return Some(t);
        }
        if !(icx.is(t, Term::KwUnsafe)
            || icx.is(t, Term::LParen)
            || icx.is(t, Term::RParen)
            || icx.is(t, Term::StrLit))
        {
            return None;
        }
    }
    None
}

/// The source text of a trait method's signature (without the `;`).
fn trait_sig_text(prog: &Program, brackets: &[Brackets], tm: DeclId) -> String {
    let Some(d) = prog.decl(tm) else { return String::new() };
    let Some(icx) = brackets.get(d.item.idx()).and_then(|b| ItemCx::new(prog, d.item, b)) else {
        return String::new();
    };
    let a = icx.a(d.node);
    // the TraitFn ends at its `;`
    let mut e = a;
    for t in a..(a + 100_000) {
        if icx.is(t, Term::Semi) || icx.is(t, Term::Eof) {
            e = t.saturating_sub(1);
            break;
        }
    }
    icx.span_text(icx.range_span(a, e)).to_string()
}

/// E0905 comparison: the impl method against the trait method with `Self` = the impl's self type,
/// the trait's params = the impl's trait args, and the trait method's own generics renumbered.
fn sig_matches(
    w: &World<'_>,
    tys: &mut TypeTable,
    imp: &ImplDef,
    tr: &TraitRef,
    tm: DeclId,
    m: DeclId,
) -> bool {
    let (Some(ts), Some(ms)) = (w.sigs.fn_sig(tm), w.sigs.fn_sig(m)) else { return true };
    let nt = ts.n_parent as usize;
    let own_t = ts.generics.len() - nt;
    let ni = ms.n_parent as usize;
    let own_m = ms.generics.len() - ni;
    if own_t != own_m {
        return false;
    }
    let mut args: Vec<GArg> = tr.args.clone();
    args.truncate(nt);
    while args.len() < nt {
        args.push(GArg::Type(tys.error_ty()));
    }
    for j in 0..own_t {
        let g = &ts.generics[nt + j];
        args.push(if g.is_const {
            GArg::Const(ConstArg::Param((ni + j) as u32))
        } else {
            GArg::Type(tys.mk_param((ni + j) as u32))
        });
    }
    let sub = |tys: &mut TypeTable, t: TypeId| tys.subst(t, &args, Some(imp.self_ty));
    if ts.params.len() != ms.params.len() || ts.self_kind != ms.self_kind {
        return false;
    }
    for (a, b) in ts.params.iter().zip(&ms.params) {
        let a2 = sub(tys, *a);
        if a2 != *b && !tys.has_error(a2) && !tys.has_error(*b) {
            return false;
        }
    }
    let r2 = sub(tys, ts.ret);
    if r2 != ms.ret && !tys.has_error(r2) && !tys.has_error(ms.ret) {
        return false;
    }
    if ts.origins != ms.origins
        || ts.param_origins != ms.param_origins
        || ts.features != ms.features
        || ts.is_unsafe != ms.is_unsafe
    {
        return false;
    }
    // The method's own generics, position by position: kind, `view`, a const param's type, and
    // the bounds as a set of complete trait refs (trait and args) after the same substitution.
    // A caller instantiates them by the trait's declaration and the instance runs the impl's
    // body, so any difference reaches a body with arguments it was not checked for.
    let sub_arg = |tys: &mut TypeTable, a: GArg| match a {
        GArg::Type(t) => GArg::Type(tys.subst(t, &args, Some(imp.self_ty))),
        c => crate::mono::subst_garg(tys, c, &args),
    };
    for j in 0..own_t {
        let (a, b) = (&ts.generics[nt + j], &ms.generics[ni + j]);
        if a.is_const != b.is_const || a.view != b.view {
            return false;
        }
        if a.is_const {
            let at = sub(tys, a.const_ty);
            if at != b.const_ty && !tys.has_error(at) && !tys.has_error(b.const_ty) {
                return false;
            }
        }
        let mut ta: Vec<(DeclId, Vec<GArg>)> = Vec::with_capacity(a.bounds.len());
        for x in &a.bounds {
            let xs = x.args.iter().map(|&g| sub_arg(tys, g)).collect();
            ta.push((x.trait_decl, xs));
        }
        let mut tb: Vec<(DeclId, Vec<GArg>)> =
            b.bounds.iter().map(|x| (x.trait_decl, x.args.clone())).collect();
        ta.sort();
        ta.dedup();
        tb.sort();
        tb.dedup();
        let erroneous = ta.iter().chain(&tb).flat_map(|(_, xs)| xs).any(|x| match x {
            GArg::Type(t) => tys.has_error(*t),
            GArg::Const(c) => *c == ConstArg::Error,
        });
        if ta != tb && !erroneous {
            return false;
        }
    }
    let _ = NodeKind::File;
    true
}
