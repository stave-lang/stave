//! Traits and impls for non-generic code and core generics: Copy, Send/Sync, and Flat (the view
//! kind and [`crate::kinds::Kind`] are in [`crate::kinds`]), layouts, impl matching, trait-bound
//! checks, method lookup, and impl conformance (E0905 E0906 E0907, Copy E0415, Drop
//! E0416, overlapping impls E0903, any impl of Flat E1408).
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
//! types are Send/Sync coinductively; a query nested past a fixed depth answers "no".
//!
//! **Flat** is derived, never implemented (an impl is E1408 and is not consulted): a type is
//! Flat iff it holds no reference, view, raw pointer, fn value, `std.cell.UnsafeCell`, or ADT
//! with a `Drop` impl, recursively through struct fields, enum payloads, the type arguments of
//! an ADT instance, and array, tuple, and range elements. `str` and slices are not Flat
//! (unsized); a generic param is Flat only through a `Flat` bound. Recursive types are Flat
//! coinductively, like Send/Sync; past the depth cutoff the answer is "no" (E1407 case `depth`).
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

/// C-like layout of a non-ADT type: `(size, align)`; `sub` gives the layout
/// of component types. `None` for unsized and generic types.
pub fn scalar_layout(k: &TypeKind, sub: &mut dyn FnMut(TypeId) -> Option<(u64, u64)>) -> Option<(u64, u64)> {
    Some(match k {
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
            (s.checked_mul(*n)?, a)
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
            let total = s * (*n as u64);
            (total, total)
        }
        TypeKind::Mask { width, n: crate::prim::Lanes::Fixed(n) } => {
            let total = (*width as u64 / 8) * (*n as u64);
            (total, total)
        }
        _ => return None,
    })
}

/// Layout of `t` in table `cx` (concrete types only). `depth`: the parts above `t`; a part
/// [`crate::typedepth::MAX_TYPE_DEPTH`] parts down has no layout.
pub fn layout_of<I: Interner + ?Sized>(cx: &mut I, sigs: &Sigs, t: TypeId, depth: u32) -> Option<(u64, u64)> {
    if depth >= crate::typedepth::MAX_TYPE_DEPTH {
        return None;
    }
    let k = cx.kind(t).clone();
    match k {
        TypeKind::Adt { def, args } => {
            let adt = sigs.adt(def)?;
            let mut vls = Vec::new();
            for v in &adt.variants {
                let mut fs = Vec::new();
                for f in &v.fields {
                    let ft = cx.subst(f.ty, &args, None);
                    fs.push(layout_of(cx, sigs, ft, depth + 1)?);
                }
                vls.push(fs);
            }
            adt_layout(adt.is_enum, &vls)
        }
        TypeKind::Ref { inner, .. } | TypeKind::Ptr { inner, .. } => {
            Some(if matches!(cx.kind(inner), TypeKind::Str | TypeKind::Slice { .. }) {
                (16, 8)
            } else {
                (8, 8)
            })
        }
        _ => scalar_layout(&k, &mut |x| layout_of(cx, sigs, x, depth + 1)),
    }
}

fn struct_layout(fields: &[(u64, u64)], start: u64) -> Option<(u64, u64)> {
    let mut off = start;
    let mut align = 1u64;
    for &(s, a) in fields {
        let a = a.max(1);
        off = off.checked_add(a - 1)? / a * a;
        off = off.checked_add(s)?;
        align = align.max(a);
    }
    let size = off.checked_add(align - 1)? / align * align;
    Some((size, align))
}

/// ADT layout: a struct is C-like; an enum has a `u32` tag at 0, then each variant's fields from
/// offset 4, size the largest variant rounded up.
pub fn adt_layout(is_enum: bool, variants: &[Vec<(u64, u64)>]) -> Option<(u64, u64)> {
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
    let size = size.checked_add(align - 1)? / align * align;
    Some((size, align))
}

/// Deepest type nesting a trait query descends: the type depth limit. Past it a type does not
/// implement the trait: the conservative answer (not Copy: moved; not Send/Sync: rejected at the
/// thread boundary; not Flat: rejected at the bound). Within one type the limit is not reached
/// (E0005 case `type` rejects deeper types); a query that follows references through distinct
/// types can reach it. Recursive types end earlier, by the coinductive Send/Sync rule ([`Query`]).
pub(crate) const QUERY_DEPTH: u32 = crate::typedepth::MAX_TYPE_DEPTH;

/// State of one top-level trait query: the Send/Sync goals in progress (a goal reached again is
/// assumed to hold), the lowest in-progress index an assumption used below the current goal
/// (`usize::MAX`: none), and the results already known. `any_kind`: impl matching skips the
/// view-kind rule (a query that asks whether an impl could apply to some instance, or names the
/// impl a view was refused by; never one that selects an impl).
struct Query {
    active: Vec<(TypeId, DeclId)>,
    low: usize,
    memo: FxHashMap<(TypeId, DeclId), bool>,
    any_kind: bool,
}

impl Default for Query {
    fn default() -> Query {
        Query { active: Vec::new(), low: usize::MAX, memo: FxHashMap::default(), any_kind: false }
    }
}

impl Query {
    fn any_kind() -> Query {
        Query { any_kind: true, ..Query::default() }
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
    /// The query reached its depth cutoff (fails closed).
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
    /// Two traits supply it (E0908): the trait decls and their method decls.
    Ambiguous {
        traits: [DeclId; 2],
        methods: [DeclId; 2],
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
        self.match_ty_in(cx, pat, t, binds, &mut false)
    }

    fn match_ty_in<I: Interner + ?Sized>(
        &self,
        cx: &I,
        pat: TypeId,
        t: TypeId,
        binds: &mut [Option<GArg>],
        exceeded: &mut bool,
    ) -> bool {
        let mut stack = vec![(pat, t)];
        let mut steps = 0u32;
        while let Some((p, x)) = stack.pop() {
            steps += 1;
            if steps > 100_000 {
                *exceeded = true;
                return false;
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
        self.implements_q(cx, t, tr, env, 0, &mut Query::default())
    }

    fn implements_q<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        env: &[GenericParam],
        depth: u32,
        q: &mut Query,
    ) -> bool {
        if cx.is_error_ty(t) {
            return true;
        }
        if depth > QUERY_DEPTH {
            return false;
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
            return self.is_copy_q(cx, t, env, depth + 1, q);
        }
        let key = (t, tr.trait_decl);
        if tr.args.is_empty()
            && let Some(&r) = q.memo.get(&key)
        {
            return r;
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
        if auto {
            // Send/Sync/Flat are coinductive: a type reached again while its own query is in
            // progress (a recursive type) is assumed to implement them.
            if let Some(i) = q.active.iter().position(|a| *a == key) {
                q.low = q.low.min(i);
                return true;
            }
        }
        let me = q.active.len();
        let saved_low = std::mem::replace(&mut q.low, usize::MAX);
        if auto {
            q.active.push(key);
        }
        let r = if let Some(d) = derived {
            self.auto_derived(cx, t, tr, d, env, depth + 1, q)
        } else if auto {
            self.send_sync(cx, t, tr, env, depth + 1, q)
        } else if let Some(name) = lang.ops.iter().find(|(_, d)| **d == tr.trait_decl).map(|(n, _)| *n)
            && builtin_op_impl(cx, name, t)
        {
            true
        } else {
            self.find_impl_q(cx, t, tr, env, depth + 1, q).is_some()
        };
        if auto {
            q.active.pop();
        }
        // A positive result that used the assumption of a query further out holds only if that
        // query succeeds: it is not memoized. Negative results never depend on an assumption.
        let low = q.low;
        if tr.args.is_empty() && (!r || low >= me) {
            q.memo.insert(key, r);
        }
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
        depth: u32,
        q: &mut Query,
    ) -> bool {
        if self.find_impl_q(cx, t, tr, env, depth, q).is_some() {
            return true;
        }
        if let TypeKind::Adt { def, .. } = cx.kind(t)
            && self.has_impl_for_head(cx, *def, tr.trait_decl)
        {
            return false;
        }
        self.auto_send_sync(cx, t, tr, env, depth, q)
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
        depth: u32,
        q: &mut Query,
    ) -> bool {
        let is_sync = tr.trait_decl == self.sigs.lang.sync;
        match cx.kind(t).clone() {
            TypeKind::Ptr { .. } | TypeKind::Fn { .. } => false,
            TypeKind::Ref { mutable, inner } => {
                // &T: Send iff T: Sync; &mut T: Send iff T: Send. Both Sync iff T: Sync.
                let want = if !is_sync && !mutable { self.sigs.lang.sync } else { tr.trait_decl };
                self.implements_q(cx, inner, &TraitRef { trait_decl: want, args: Vec::new() }, env, depth, q)
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
                        if !self.implements_q(cx, ft, tr, env, depth, q) {
                            return false;
                        }
                    }
                }
                true
            }
            k => {
                let mut kids = Vec::new();
                k.children(&mut kids);
                kids.into_iter().all(|c| self.implements_q(cx, c, tr, env, depth, q))
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
        depth: u32,
        q: &mut Query,
    ) -> bool {
        match self.derived_parts(cx, t, d) {
            Ok(parts) => parts.into_iter().all(|(_, c)| self.implements_q(cx, c, tr, env, depth, q)),
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
    /// each level. [`NotFlat::Depth`] when the query failed at its depth cutoff only.
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
        if self.implements(cx, t, &tr, env) {
            return None;
        }
        let mut path = String::new();
        let mut cur = t;
        for _ in 0..=QUERY_DEPTH {
            let parts = match self.derived_parts(cx, cur, d) {
                Ok(p) => p,
                Err(why) => return Some(FlatWitness { path, part: cur, why }),
            };
            let Some((seg, next)) = parts.into_iter().find(|&(_, c)| !self.implements(cx, c, &tr, env))
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
        depth: u32,
    ) -> Option<(DeclId, Vec<GArg>)> {
        self.find_impl_q(cx, t, tr, env, depth, &mut Query::default())
    }

    fn find_impl_q<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        tr: &TraitRef,
        env: &[GenericParam],
        depth: u32,
        q: &mut Query,
    ) -> Option<(DeclId, Vec<GArg>)> {
        let list = self.sigs.impls_by_trait.get(&tr.trait_decl)?;
        for &i in list {
            let Some(imp) = self.sigs.impl_def(i) else { continue };
            if let Some(args) = self.impl_applies_q(cx, imp, t, env, depth, q) {
                // the trait args must match after substitution
                let Some(itr) = &imp.trait_ref else { continue };
                let ok = itr.args.len() == tr.args.len()
                    && itr.args.iter().zip(&tr.args).all(|(a, b)| match (a, b) {
                        (GArg::Type(x), GArg::Type(y)) => cx.subst(*x, &args, Some(t)) == *y,
                        (GArg::Const(x), GArg::Const(y)) => x == y,
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
        depth: u32,
    ) -> Option<Vec<GArg>> {
        self.impl_applies_q(cx, imp, t, env, depth, &mut Query::default())
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
        self.impl_applies_q(cx, imp, t, env, 0, &mut Query::any_kind())
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
        depth: u32,
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
                    args: b
                        .args
                        .iter()
                        .map(|x| match x {
                            GArg::Type(t) => GArg::Type(cx.subst(*t, &args, None)),
                            c => *c,
                        })
                        .collect(),
                };
                if !self.implements_q(cx, a, &b2, env, depth + 1, q) {
                    return None;
                }
            }
        }
        Some(args)
    }

    pub fn is_copy<I: Interner + ?Sized>(&self, cx: &mut I, t: TypeId, env: &[GenericParam]) -> bool {
        self.is_copy_q(cx, t, env, 0, &mut Query::default())
    }

    fn is_copy_q<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        t: TypeId,
        env: &[GenericParam],
        depth: u32,
        q: &mut Query,
    ) -> bool {
        if depth > QUERY_DEPTH {
            return false;
        }
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
            TypeKind::Array { elem, .. } | TypeKind::Range { elem, .. } => {
                self.is_copy_q(cx, elem, env, depth + 1, q)
            }
            TypeKind::Tuple(ts) => {
                let copy = self.sigs.lang.copy;
                if let Some(&r) = q.memo.get(&(t, copy)) {
                    return r;
                }
                let low = std::mem::replace(&mut q.low, usize::MAX);
                let r = ts.iter().all(|&x| self.is_copy_q(cx, x, env, depth + 1, q));
                if !r || q.low == usize::MAX {
                    q.memo.insert((t, copy), r);
                }
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
                if let Some(&r) = q.memo.get(&(t, copy)) {
                    return r;
                }
                let low = std::mem::replace(&mut q.low, usize::MAX);
                let tr = TraitRef { trait_decl: copy, args: Vec::new() };
                let r = self.find_impl_q(cx, t, &tr, env, depth + 1, q).is_some();
                if !r || q.low == usize::MAX {
                    q.memo.insert((t, copy), r);
                }
                q.low = q.low.min(low);
                r
            }
        }
    }

    /// [`crate::kinds::is_view_in`]: a view in the scope of `env` (a `T: view` param counts).
    pub fn is_view<I: Interner + ?Sized>(&self, cx: &I, t: TypeId, env: &[GenericParam]) -> bool {
        crate::kinds::is_view_in(self.prog, cx, t, env)
    }

    /// Trait methods named `name` reachable from a generic param's bounds:
    /// (trait decl, method decl, trait args).
    fn bound_methods(&self, env: &[GenericParam], param: u32, name: Sym) -> Vec<(DeclId, DeclId, Vec<GArg>)> {
        let mut out = Vec::new();
        let Some(g) = env.get(param as usize) else { return out };
        for b in &g.bounds {
            if let Some(m) = self.prog.member(b.trait_decl, name)
                && self.prog.decl(m).is_some_and(|x| x.kind == DeclKind::TraitMethod)
                && !out.iter().any(|(t, _, _)| *t == b.trait_decl)
            {
                out.push((b.trait_decl, m, b.args.clone()));
            }
        }
        out
    }

    /// Method lookup for receiver type `recv`.
    pub fn lookup_method<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        recv: TypeId,
        name: Sym,
        env: &[GenericParam],
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
                        return Lookup::Ambiguous { traits: [a.0, b.0], methods: [a.1, b.1], recv: c };
                    }
                    [] => {
                        if !matches!(cx.kind(c), TypeKind::Ref { .. }) {
                            return Lookup::NotInBounds { param: p };
                        }
                    }
                }
            }
            if let Some(r) = self.lookup_at(cx, c, name, env) {
                return match r {
                    Ok(m) => Lookup::Found { m, level, recv: c },
                    Err((traits, methods)) => Lookup::Ambiguous { traits, methods, recv: c },
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

    /// Methods named `name` of exactly type `c`: inherent first, then trait impls.
    #[allow(clippy::type_complexity)]
    fn lookup_at<I: Interner + ?Sized>(
        &self,
        cx: &mut I,
        c: TypeId,
        name: Sym,
        env: &[GenericParam],
    ) -> Option<Result<MethodRef, ([DeclId; 2], [DeclId; 2])>> {
        let head = self.head(cx, c);
        let lists: [&[DeclId]; 2] = [
            head.and_then(|h| self.sigs.impls_by_head.get(&h)).map_or(&[][..], |v| &v[..]),
            &self.sigs.headless_impls[..],
        ];
        // inherent
        for list in lists {
            for &i in list {
                let Some(imp) = self.sigs.impl_def(i) else { continue };
                if imp.trait_ref.is_some() {
                    continue;
                }
                let Some(m) =
                    imp.methods.iter().copied().find(|&m| self.prog.decl(m).is_some_and(|d| d.name == name))
                else {
                    continue;
                };
                if let Some(args) = self.impl_applies(cx, imp, c, env, 0) {
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
                let Some(m) =
                    imp.methods.iter().copied().find(|&m| self.prog.decl(m).is_some_and(|d| d.name == name))
                else {
                    continue;
                };
                if found.iter().any(|(t, _, _)| *t == tr.trait_decl) {
                    continue;
                }
                if let Some(args) = self.impl_applies(cx, imp, c, env, 0) {
                    let tm = self.prog.member(tr.trait_decl, name).unwrap_or(m);
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
                Some(Err(([found[0].0, found[1].0], [found[0].2, found[1].2])))
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
    let mut seen: FxHashMap<DeclId, Vec<DeclId>> = FxHashMap::default();
    for d in impls {
        let Some(imp) = sigs.impl_def(d) else { continue };
        let Some(decl) = prog.decl(d) else { continue };
        let Some(icx) = brackets.get(decl.item.idx()).and_then(|b| ItemCx::new(prog, decl.item, b)) else {
            continue;
        };
        let impl_span = decl.name_span;
        let names = NameCx { prog, generics: &imp.generics };
        let self_text = types::display(tys, &names, imp.self_ty);
        let Some(tr) = &imp.trait_ref else { continue };
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
        // E0416: Drop on a Copy type. Copy and Drop exclude each other on every instance, so
        // the impls conflict when one instance of the Drop impl's head may be Copy: a Copy impl
        // whose head unifies with it, or a built-in Copy type.
        if tr.trait_decl == lang.drop && lang.drop.is_some() && !tys.is_error_ty(imp.self_ty) {
            let copy_impl = copy_impl_overlapping(&w, tys, imp);
            let builtin_copy = match tys.kind(imp.self_ty) {
                TypeKind::Adt { .. } => false,
                TypeKind::Param(_) => true,
                _ => {
                    // every type param stands for a Copy type too
                    let mut env = imp.generics.clone();
                    for g in env.iter_mut().filter(|g| !g.is_const) {
                        g.bounds.push(TraitRef { trait_decl: lang.copy, args: Vec::new() });
                    }
                    w.is_copy(tys, imp.self_ty, &env)
                }
            };
            if copy_impl.is_some() || builtin_copy {
                let roles: Vec<(&str, Span)> = copy_impl
                    .and_then(|ci| prog.decl(ci))
                    .map(|x| ("copy", x.name_span))
                    .into_iter()
                    .collect();
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
                while let Some(x) = stack.pop() {
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
        // E0903: overlapping impls of one trait.
        let prev = seen.entry(tr.trait_decl).or_default();
        for &o in prev.iter() {
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
        prev.push(d);
    }
    out
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
    let mut u = Unifier { binds: vec![None; a.generics.len() + b.generics.len()], steps: 0 };
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

/// A Copy impl that may apply to an instance of Drop impl `drop`'s head. A unifier rules the
/// pair out only when it binds a param of either impl to a type without params that fails one
/// of that param's bounds; any other unifier counts (a bound on a type with params may hold for
/// some instance).
fn copy_impl_overlapping(w: &World<'_>, tys: &mut TypeTable, drop: &ImplDef) -> Option<DeclId> {
    let copy = w.sigs.lang.copy;
    let list = w.sigs.impls_by_trait.get(&copy)?;
    let na = drop.generics.len();
    for &ci in list {
        let Some(cimp) = w.sigs.impl_def(ci) else { continue };
        if tys.is_error_ty(cimp.self_ty) {
            continue;
        }
        let Some(binds) = heads_unify(tys, drop, cimp, None) else { continue };
        let n = binds.len();
        let args: Vec<GArg> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| match b {
                Some(a) => *a,
                None if i >= na && cimp.generics[i - na].is_const => GArg::Const(ConstArg::Param(i as u32)),
                None if i < na && drop.generics[i].is_const => GArg::Const(ConstArg::Param(i as u32)),
                None => GArg::Type(tys.mk_param(i as u32)),
            })
            .collect();
        // the bindings are acyclic: n rounds of substitution reach the fixed point
        let resolve = |tys: &mut TypeTable, a: GArg| {
            let mut a = a;
            for _ in 0..=n {
                let b = crate::mono::subst_garg(tys, a, &args);
                if b == a {
                    break;
                }
                a = b;
            }
            a
        };
        let shift: Vec<GArg> = (0..cimp.generics.len())
            .map(|i| {
                if cimp.generics[i].is_const {
                    GArg::Const(ConstArg::Param((na + i) as u32))
                } else {
                    GArg::Type(tys.mk_param((na + i) as u32))
                }
            })
            .collect();
        let mut excluded = false;
        let params =
            drop.generics.iter().map(|g| (g, None)).chain(cimp.generics.iter().map(|g| (g, Some(&shift))));
        for (k, (g, sh)) in params.enumerate() {
            if g.is_const || g.bounds.is_empty() {
                continue;
            }
            let p = tys.mk_param(k as u32);
            let GArg::Type(t) = resolve(tys, GArg::Type(p)) else { continue };
            if tys.has_params(t) {
                continue;
            }
            for b in &g.bounds {
                let mut bargs = Vec::with_capacity(b.args.len());
                for &x in &b.args {
                    let x = match sh {
                        Some(sh) => crate::mono::subst_garg(tys, x, sh),
                        None => x,
                    };
                    bargs.push(resolve(tys, x));
                }
                let ground = bargs.iter().all(|x| match x {
                    GArg::Type(x) => !tys.has_params(*x),
                    GArg::Const(c) => !matches!(c, ConstArg::Param(_)),
                });
                if ground && !w.implements(tys, t, &TraitRef { trait_decl: b.trait_decl, args: bargs }, &[]) {
                    excluded = true;
                    break;
                }
            }
            if excluded {
                break;
            }
        }
        if !excluded {
            return Some(ci);
        }
    }
    None
}

/// Most steps of one [`heads_unify`].
const UNIFY_STEPS: u32 = 100_000;

/// Syntactic unification with an occurs check over the variables `binds` (type and const
/// params; a variable is bound at most once and bindings never form a cycle).
struct Unifier {
    binds: Vec<Option<GArg>>,
    steps: u32,
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
        while let Some(x) = stack.pop() {
            self.steps += 1;
            if self.steps > UNIFY_STEPS {
                return false;
            }
            let x = self.resolve(cx, x);
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

    /// `false`: `a` and `b` have no common instance. Past the step limit: `true`.
    fn unify(&mut self, cx: &TypeTable, a: TypeId, b: TypeId) -> bool {
        let mut stack = vec![(a, b)];
        while let Some((p, x)) = stack.pop() {
            self.steps += 1;
            if self.steps > UNIFY_STEPS {
                return true;
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
