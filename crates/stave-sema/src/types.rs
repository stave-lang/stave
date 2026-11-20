//! Types: the type keys ([`TypeKind`]), interning, substitution, printing,
//! kinds, and canonical type ids.
//!
//! **Interning.** Equal keys get equal ids, so type equality is an id compare. The signature
//! phase interns into the global [`TypeTable`]; each function body interns into an [`Overlay`]
//! over the frozen global table (ids with the high bit set are body-local), so bodies are checked
//! in parallel without locks. After all bodies, the overlays are merged into the global table and
//! every id is renumbered canonically ([`TypeTable::canonicalize`]).
//!
//! **Canonical type id**: the rank in the canonical type-key order:
//! the variant in the order of [`TypeKind`] (the extensions [`TypeKind::Range`]
//! and [`TypeKind::SelfTy`] sort after `Never`), then the fields in declaration order compared
//! lexicographically: a type field by its canonical id, a decl by its (canonical) decl id, a number
//! or flag by value, an enumerated field in its listed order, a list element by element (a prefix
//! first). The error type has no key: it is ranked last, after every keyed type, and is never
//! printed.
//!
//! **Extensions** (not in LANG.md, see the module docs of [`crate::typeck`]): `Range{elem, inclusive}`
//! for `a..b` / `a..=b` (`Range<T>` / `RangeInclusive<T>` have no decl; the
//! runtime, which has no std, uses ranges too), printed `range t<T>` / `range incl t<T>`; `SelfTy`,
//! the `Self` of a trait's method signatures (never part of a checked body's types).

use std::cmp::Ordering;

use rustc_hash::FxHashMap;

use crate::ids::DeclId;
use crate::prim::{Lanes, Scalar};

/// A type id. Global ids index the [`TypeTable`]; ids with [`TypeId::LOCAL`] set index the
/// current body's [`Overlay`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(pub u32);

impl TypeId {
    /// "No type" (a node that is not an expression or denotes no value).
    pub const NONE: TypeId = TypeId(u32::MAX);
    pub const LOCAL: u32 = 1 << 31;

    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    pub fn is_local(self) -> bool {
        self.0 != u32::MAX && self.0 & Self::LOCAL != 0
    }

    #[inline]
    pub fn idx(self) -> usize {
        (self.0 & !Self::LOCAL) as usize
    }
}

/// Integer width. `Size` is 64 bits on every target but a distinct type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntBits {
    B8,
    B16,
    B32,
    B64,
    Size,
}

impl IntBits {
    pub fn bits(self) -> u32 {
        match self {
            IntBits::B8 => 8,
            IntBits::B16 => 16,
            IntBits::B32 => 32,
            IntBits::B64 | IntBits::Size => 64,
        }
    }
}

/// Float format (`B16` = bf16; sorts last).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatBits {
    F16,
    F32,
    F64,
    B16,
}

/// A const value in a type (array length, const generic argument).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstArg {
    Value(u64),
    /// A const generic parameter (flat index).
    Param(u32),
    /// Unknown because of an earlier error.
    Error,
}

/// A generic argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GArg {
    Type(TypeId),
    Const(ConstArg),
}

/// The type key. The variant order is the canonical key order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Int {
        signed: bool,
        bits: IntBits,
    },
    Float(FloatBits),
    Bool,
    Char,
    Str,
    Ref {
        mutable: bool,
        inner: TypeId,
    },
    Ptr {
        mutable: bool,
        inner: TypeId,
    },
    Array {
        elem: TypeId,
        len: ConstArg,
    },
    /// `align` 0: no alignment clause.
    Slice {
        elem: TypeId,
        align: u32,
    },
    Tuple(Box<[TypeId]>),
    Adt {
        def: DeclId,
        args: Box<[GArg]>,
    },
    /// `features`: the stave-target `FeatureSet` bits (closed under implication). `deny`: the
    /// effects of the deny clause ([`crate::effects::Effects`] bits).
    Fn {
        params: Box<[TypeId]>,
        ret: TypeId,
        features: u64,
        deny: u8,
    },
    Vector {
        lane: TypeId,
        n: Lanes,
    },
    Mask {
        width: u8,
        n: Lanes,
    },
    /// A generic type parameter (flat index: container params, then the fn's own).
    Param(u32),
    /// The type of a const generic parameter used as a value is its declared type; this key is
    /// the parameter itself where a type is required (kept for the canonical variant order).
    ConstParam(u32),
    Never,
    /// Extension: `Range<T>` / `RangeInclusive<T>`.
    Range {
        elem: TypeId,
        inclusive: bool,
    },
    /// Extension: `Self` in a trait's signatures.
    SelfTy,
    /// The error type (no key; compatible with everything).
    Error,
}

impl TypeKind {
    fn tag(&self) -> u8 {
        match self {
            TypeKind::Int { .. } => 0,
            TypeKind::Float(_) => 1,
            TypeKind::Bool => 2,
            TypeKind::Char => 3,
            TypeKind::Str => 4,
            TypeKind::Ref { .. } => 5,
            TypeKind::Ptr { .. } => 6,
            TypeKind::Array { .. } => 7,
            TypeKind::Slice { .. } => 8,
            TypeKind::Tuple(_) => 9,
            TypeKind::Adt { .. } => 10,
            TypeKind::Fn { .. } => 11,
            TypeKind::Vector { .. } => 12,
            TypeKind::Mask { .. } => 13,
            TypeKind::Param(_) => 14,
            TypeKind::ConstParam(_) => 15,
            TypeKind::Never => 16,
            TypeKind::Range { .. } => 17,
            TypeKind::SelfTy => 18,
            TypeKind::Error => 19,
        }
    }

    /// The component types, in key order.
    pub fn children(&self, out: &mut Vec<TypeId>) {
        match self {
            TypeKind::Ref { inner, .. } | TypeKind::Ptr { inner, .. } => out.push(*inner),
            TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } | TypeKind::Range { elem, .. } => {
                out.push(*elem)
            }
            TypeKind::Tuple(ts) => out.extend_from_slice(ts),
            TypeKind::Adt { args, .. } => {
                for a in args.iter() {
                    if let GArg::Type(t) = a {
                        out.push(*t);
                    }
                }
            }
            TypeKind::Fn { params, ret, .. } => {
                out.extend_from_slice(params);
                out.push(*ret);
            }
            TypeKind::Vector { lane, .. } => out.push(*lane),
            _ => {}
        }
    }

    /// The same key with every component type mapped by `f`.
    pub fn map(&self, mut f: impl FnMut(TypeId) -> TypeId) -> TypeKind {
        match self {
            TypeKind::Ref { mutable, inner } => TypeKind::Ref { mutable: *mutable, inner: f(*inner) },
            TypeKind::Ptr { mutable, inner } => TypeKind::Ptr { mutable: *mutable, inner: f(*inner) },
            TypeKind::Array { elem, len } => TypeKind::Array { elem: f(*elem), len: *len },
            TypeKind::Slice { elem, align } => TypeKind::Slice { elem: f(*elem), align: *align },
            TypeKind::Range { elem, inclusive } => TypeKind::Range { elem: f(*elem), inclusive: *inclusive },
            TypeKind::Tuple(ts) => TypeKind::Tuple(ts.iter().map(|&t| f(t)).collect()),
            TypeKind::Adt { def, args } => TypeKind::Adt {
                def: *def,
                args: args
                    .iter()
                    .map(|a| match a {
                        GArg::Type(t) => GArg::Type(f(*t)),
                        c => *c,
                    })
                    .collect(),
            },
            TypeKind::Fn { params, ret, features, deny } => TypeKind::Fn {
                params: params.iter().map(|&t| f(t)).collect(),
                ret: f(*ret),
                features: *features,
                deny: *deny,
            },
            TypeKind::Vector { lane, n } => TypeKind::Vector { lane: f(*lane), n: *n },
            other => other.clone(),
        }
    }
}

/// The well-known types of a fresh [`TypeTable`] (interned first, in this order). These ids hold
/// only until [`TypeTable::canonicalize`] renumbers the table: code that may run on a canonical
/// table (mono, lowering, the public helpers) maps them with [`Interner::fixed_ty`] (or uses
/// [`Interner::error_ty`] / [`Interner::is_error_ty`] / [`Interner::scalar_ty`]).
pub mod ids {
    use super::TypeId;
    pub const ERROR: TypeId = TypeId(0);
    pub const NEVER: TypeId = TypeId(1);
    pub const UNIT: TypeId = TypeId(2);
    pub const BOOL: TypeId = TypeId(3);
    pub const CHAR: TypeId = TypeId(4);
    pub const STR: TypeId = TypeId(5);
    pub const I8: TypeId = TypeId(6);
    pub const I16: TypeId = TypeId(7);
    pub const I32: TypeId = TypeId(8);
    pub const I64: TypeId = TypeId(9);
    pub const ISIZE: TypeId = TypeId(10);
    pub const U8: TypeId = TypeId(11);
    pub const U16: TypeId = TypeId(12);
    pub const U32: TypeId = TypeId(13);
    pub const U64: TypeId = TypeId(14);
    pub const USIZE: TypeId = TypeId(15);
    pub const F16: TypeId = TypeId(16);
    pub const BF16: TypeId = TypeId(17);
    pub const F32: TypeId = TypeId(18);
    pub const F64: TypeId = TypeId(19);
    pub const SELF_TY: TypeId = TypeId(20);
    /// `&str`.
    pub const REF_STR: TypeId = TypeId(21);
    pub(super) const COUNT: u32 = 22;
}

/// Number of well-known types ([`ids`]).
const FIXED: usize = ids::COUNT as usize;

/// Interning access shared by the global table and body overlays.
pub trait Interner {
    fn kind(&self, t: TypeId) -> &TypeKind;
    fn intern(&mut self, k: TypeKind) -> TypeId;

    /// The id in this table of the well-known type `f` (one of [`ids`]; any other id is returned
    /// unchanged). Identity for a table that was never canonicalized; [`TypeTable`] and
    /// [`Overlay`] track the renumbering, and every wrapper must forward to its table.
    fn fixed_ty(&self, f: TypeId) -> TypeId {
        f
    }

    /// The error type's id in this table.
    fn error_ty(&self) -> TypeId {
        self.fixed_ty(ids::ERROR)
    }

    /// Whether `t` is the error type (exactly; see [`Interner::has_error`] for "mentions").
    fn is_error_ty(&self, t: TypeId) -> bool {
        t == self.error_ty()
    }

    /// The id of a built-in scalar type (`str` included) in this table.
    fn scalar_ty(&self, s: Scalar) -> TypeId {
        self.fixed_ty(scalar_type(s))
    }

    fn mk_ref(&mut self, mutable: bool, inner: TypeId) -> TypeId {
        if self.is_error_ty(inner) {
            return inner;
        }
        self.intern(TypeKind::Ref { mutable, inner })
    }

    fn mk_ptr(&mut self, mutable: bool, inner: TypeId) -> TypeId {
        if self.is_error_ty(inner) {
            return inner;
        }
        self.intern(TypeKind::Ptr { mutable, inner })
    }

    fn mk_tuple(&mut self, ts: Vec<TypeId>) -> TypeId {
        let err = self.error_ty();
        if ts.contains(&err) {
            return err;
        }
        self.intern(TypeKind::Tuple(ts.into_boxed_slice()))
    }

    fn mk_adt(&mut self, def: DeclId, args: Vec<GArg>) -> TypeId {
        let err = self.error_ty();
        if args.contains(&GArg::Type(err)) {
            return err;
        }
        self.intern(TypeKind::Adt { def, args: args.into_boxed_slice() })
    }

    fn mk_param(&mut self, i: u32) -> TypeId {
        self.intern(TypeKind::Param(i))
    }

    /// Whether `t` mentions the error type anywhere.
    fn has_error(&self, t: TypeId) -> bool {
        // a type without components (most operands) needs no walk
        if matches!(
            self.kind(t),
            TypeKind::Int { .. }
                | TypeKind::Float(_)
                | TypeKind::Bool
                | TypeKind::Char
                | TypeKind::Str
                | TypeKind::Never
                | TypeKind::Mask { .. }
                | TypeKind::Param(_)
                | TypeKind::ConstParam(_)
                | TypeKind::SelfTy
        ) {
            return false;
        }
        let mut stack = vec![t];
        let mut kids = Vec::new();
        while let Some(x) = stack.pop() {
            let k = self.kind(x);
            if matches!(k, TypeKind::Error) {
                return true;
            }
            if let TypeKind::Array { len: ConstArg::Error, .. } = k {
                return true;
            }
            kids.clear();
            k.children(&mut kids);
            stack.extend_from_slice(&kids);
        }
        false
    }

    /// Whether `t` mentions a generic parameter (type or const) or `SelfTy`.
    fn has_params(&self, t: TypeId) -> bool {
        let mut stack = vec![t];
        let mut kids = Vec::new();
        while let Some(x) = stack.pop() {
            let k = self.kind(x);
            match k {
                TypeKind::Param(_) | TypeKind::ConstParam(_) | TypeKind::SelfTy => return true,
                TypeKind::Array { len: ConstArg::Param(_), .. } => return true,
                TypeKind::Adt { args, .. }
                    if args.iter().any(|a| matches!(a, GArg::Const(ConstArg::Param(_)))) =>
                {
                    return true;
                }
                _ => {}
            }
            kids.clear();
            k.children(&mut kids);
            stack.extend_from_slice(&kids);
        }
        false
    }

    /// `t` with `Param(i)` replaced by `args[i]` (type args; const args replace const params in
    /// array lengths and ADT args) and `SelfTy` by `self_ty` (if given). Params without an
    /// argument stay.
    fn subst(&mut self, t: TypeId, args: &[GArg], self_ty: Option<TypeId>) -> TypeId {
        if args.is_empty() && self_ty.is_none() {
            return t;
        }
        let mut memo: FxHashMap<TypeId, TypeId> = FxHashMap::default();
        subst_rec(self, t, args, self_ty, &mut memo, 0)
    }

    fn int(&self, t: TypeId) -> Option<(bool, IntBits)> {
        match self.kind(t) {
            TypeKind::Int { signed, bits } => Some((*signed, *bits)),
            _ => None,
        }
    }

    fn is_float(&self, t: TypeId) -> bool {
        matches!(self.kind(t), TypeKind::Float(_))
    }

    /// Peels references: `(inner, number of refs)`.
    fn peel_refs(&self, t: TypeId) -> (TypeId, u32) {
        let mut t = t;
        let mut n = 0;
        while let TypeKind::Ref { inner, .. } = self.kind(t) {
            t = *inner;
            n += 1;
            if n > 1_000_000 {
                break;
            }
        }
        (t, n)
    }
}

const SUBST_DEPTH: u32 = 100_000;

fn subst_rec<I: Interner + ?Sized>(
    cx: &mut I,
    t: TypeId,
    args: &[GArg],
    self_ty: Option<TypeId>,
    memo: &mut FxHashMap<TypeId, TypeId>,
    depth: u32,
) -> TypeId {
    if let Some(&r) = memo.get(&t) {
        return r;
    }
    if depth > SUBST_DEPTH {
        return cx.error_ty();
    }
    let k = cx.kind(t).clone();
    let r = match &k {
        TypeKind::Param(i) => match args.get(*i as usize) {
            Some(GArg::Type(a)) => *a,
            _ => t,
        },
        TypeKind::SelfTy => self_ty.unwrap_or(t),
        TypeKind::Int { .. }
        | TypeKind::Float(_)
        | TypeKind::Bool
        | TypeKind::Char
        | TypeKind::Str
        | TypeKind::Never
        | TypeKind::Error
        | TypeKind::Mask { .. }
        | TypeKind::ConstParam(_) => t,
        _ => {
            let mapped = k.map(|c| subst_rec(cx, c, args, self_ty, memo, depth + 1));
            let mapped = match mapped {
                TypeKind::Array { elem, len: ConstArg::Param(i) } => match args.get(i as usize) {
                    Some(GArg::Const(c)) => TypeKind::Array { elem, len: *c },
                    _ => TypeKind::Array { elem, len: ConstArg::Param(i) },
                },
                TypeKind::Adt { def, args: aa } => TypeKind::Adt {
                    def,
                    args: aa
                        .iter()
                        .map(|a| match a {
                            GArg::Const(ConstArg::Param(i)) => match args.get(*i as usize) {
                                Some(GArg::Const(c)) => GArg::Const(*c),
                                _ => *a,
                            },
                            other => *other,
                        })
                        .collect(),
                },
                m => m,
            };
            let err = cx.error_ty();
            if mapped.children_contain(err) { err } else { cx.intern(mapped) }
        }
    };
    memo.insert(t, r);
    r
}

impl TypeKind {
    fn children_contain(&self, x: TypeId) -> bool {
        let mut v = Vec::new();
        self.children(&mut v);
        v.contains(&x)
    }
}

/// The global type table.
#[derive(Clone, Debug)]
pub struct TypeTable {
    kinds: Vec<TypeKind>,
    map: FxHashMap<TypeKind, TypeId>,
    /// `fixed[i]`: the id of well-known type `ids::*` number `i` (identity until canonicalized).
    fixed: [TypeId; FIXED],
}

impl Default for TypeTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeTable {
    pub fn new() -> TypeTable {
        let mut t = TypeTable {
            kinds: Vec::new(),
            map: FxHashMap::default(),
            fixed: std::array::from_fn(|i| TypeId(i as u32)),
        };
        let int = |signed, bits| TypeKind::Int { signed, bits };
        let fixed = [
            TypeKind::Error,
            TypeKind::Never,
            TypeKind::Tuple(Box::new([])),
            TypeKind::Bool,
            TypeKind::Char,
            TypeKind::Str,
            int(true, IntBits::B8),
            int(true, IntBits::B16),
            int(true, IntBits::B32),
            int(true, IntBits::B64),
            int(true, IntBits::Size),
            int(false, IntBits::B8),
            int(false, IntBits::B16),
            int(false, IntBits::B32),
            int(false, IntBits::B64),
            int(false, IntBits::Size),
            TypeKind::Float(FloatBits::F16),
            TypeKind::Float(FloatBits::B16),
            TypeKind::Float(FloatBits::F32),
            TypeKind::Float(FloatBits::F64),
            TypeKind::SelfTy,
        ];
        for k in fixed {
            t.intern(k);
        }
        let rs = t.intern(TypeKind::Ref { mutable: false, inner: ids::STR });
        debug_assert_eq!(rs, ids::REF_STR);
        debug_assert_eq!(t.kinds.len() as u32, ids::COUNT);
        t
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn get(&self, t: TypeId) -> Option<&TypeKind> {
        if t.is_local() || t.is_none() {
            return None;
        }
        self.kinds.get(t.idx())
    }

    pub fn lookup(&self, k: &TypeKind) -> Option<TypeId> {
        self.map.get(k).copied()
    }

    /// Every type, by id.
    pub fn iter(&self) -> impl Iterator<Item = (TypeId, &TypeKind)> {
        self.kinds.iter().enumerate().map(|(i, k)| (TypeId(i as u32), k))
    }

    /// Merges an overlay's local types: returns the local index → global id map.
    pub fn merge(&mut self, local: &[TypeKind]) -> Vec<TypeId> {
        let mut out: Vec<TypeId> = Vec::with_capacity(local.len());
        for k in local {
            let err = self.error_ty();
            let g = k.map(|c| if c.is_local() { out.get(c.idx()).copied().unwrap_or(err) } else { c });
            let id = self.intern(g);
            out.push(id);
        }
        out
    }

    /// The canonical renumbering: old id → new id (canonical type-key order; the
    /// error type last). The table itself is permuted.
    pub fn canonicalize(&mut self) -> Vec<TypeId> {
        let n = self.kinds.len();
        // Depth (well-founded: components are strictly smaller).
        let mut order: Vec<u32> = (0..n as u32).collect();
        {
            let kinds = &self.kinds;
            let cmp = |a: u32, b: u32| cmp_types(kinds, TypeId(a), TypeId(b));
            use rayon::slice::ParallelSliceMut;
            order.par_sort_by(|&a, &b| cmp(a, b));
        }
        let mut old_to_new = vec![TypeId::NONE; n];
        for (new, &old) in order.iter().enumerate() {
            old_to_new[old as usize] = TypeId(new as u32);
        }
        let mut kinds: Vec<TypeKind> = Vec::with_capacity(n);
        for &old in &order {
            let k = self.kinds[old as usize].map(|c| old_to_new.get(c.idx()).copied().unwrap_or(c));
            kinds.push(k);
        }
        self.map = kinds.iter().enumerate().map(|(i, k)| (k.clone(), TypeId(i as u32))).collect();
        self.kinds = kinds;
        for f in &mut self.fixed {
            *f = old_to_new.get(f.idx()).copied().unwrap_or(*f);
        }
        old_to_new
    }

    /// The error type's id (after canonicalization it is the last id).
    pub fn error_id(&self) -> TypeId {
        self.error_ty()
    }
}

impl Interner for TypeTable {
    fn kind(&self, t: TypeId) -> &TypeKind {
        static ERR: TypeKind = TypeKind::Error;
        self.get(t).unwrap_or(&ERR)
    }

    #[inline]
    fn fixed_ty(&self, f: TypeId) -> TypeId {
        self.fixed.get(f.0 as usize).copied().unwrap_or(f)
    }

    fn intern(&mut self, k: TypeKind) -> TypeId {
        if let Some(&id) = self.map.get(&k) {
            return id;
        }
        let id = TypeId(self.kinds.len() as u32);
        self.kinds.push(k.clone());
        self.map.insert(k, id);
        id
    }
}

/// Structural comparison in the canonical key order (the error type last).
fn cmp_types(kinds: &[TypeKind], a: TypeId, b: TypeId) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    let (Some(ka), Some(kb)) = (kinds.get(a.idx()), kinds.get(b.idx())) else { return a.cmp(&b) };
    let (ta, tb) = (ka.tag(), kb.tag());
    if ta != tb {
        return ta.cmp(&tb);
    }
    let ct = |x: TypeId, y: TypeId| cmp_types(kinds, x, y);
    let cl = |xs: &[TypeId], ys: &[TypeId]| -> Ordering {
        for (x, y) in xs.iter().zip(ys) {
            let o = ct(*x, *y);
            if o != Ordering::Equal {
                return o;
            }
        }
        xs.len().cmp(&ys.len())
    };
    match (ka, kb) {
        (TypeKind::Int { signed: s1, bits: b1 }, TypeKind::Int { signed: s2, bits: b2 }) => {
            (s1, b1).cmp(&(s2, b2))
        }
        (TypeKind::Float(x), TypeKind::Float(y)) => x.cmp(y),
        (TypeKind::Ref { mutable: m1, inner: i1 }, TypeKind::Ref { mutable: m2, inner: i2 })
        | (TypeKind::Ptr { mutable: m1, inner: i1 }, TypeKind::Ptr { mutable: m2, inner: i2 }) => {
            m1.cmp(m2).then_with(|| ct(*i1, *i2))
        }
        (TypeKind::Array { elem: e1, len: l1 }, TypeKind::Array { elem: e2, len: l2 }) => {
            ct(*e1, *e2).then_with(|| l1.cmp(l2))
        }
        (TypeKind::Slice { elem: e1, align: a1 }, TypeKind::Slice { elem: e2, align: a2 }) => {
            ct(*e1, *e2).then_with(|| a1.cmp(a2))
        }
        (TypeKind::Tuple(x), TypeKind::Tuple(y)) => cl(x, y),
        (TypeKind::Adt { def: d1, args: a1 }, TypeKind::Adt { def: d2, args: a2 }) => {
            d1.cmp(d2).then_with(|| {
                for (x, y) in a1.iter().zip(a2.iter()) {
                    let o = match (x, y) {
                        (GArg::Type(p), GArg::Type(q)) => ct(*p, *q),
                        (GArg::Const(p), GArg::Const(q)) => p.cmp(q),
                        (GArg::Type(_), GArg::Const(_)) => Ordering::Less,
                        (GArg::Const(_), GArg::Type(_)) => Ordering::Greater,
                    };
                    if o != Ordering::Equal {
                        return o;
                    }
                }
                a1.len().cmp(&a2.len())
            })
        }
        (
            TypeKind::Fn { params: p1, ret: r1, features: f1, deny: d1 },
            TypeKind::Fn { params: p2, ret: r2, features: f2, deny: d2 },
        ) => cl(p1, p2).then_with(|| ct(*r1, *r2)).then_with(|| f1.cmp(f2)).then_with(|| d1.cmp(d2)),
        (TypeKind::Vector { lane: l1, n: n1 }, TypeKind::Vector { lane: l2, n: n2 }) => {
            ct(*l1, *l2).then_with(|| n1.cmp(n2))
        }
        (TypeKind::Mask { width: w1, n: n1 }, TypeKind::Mask { width: w2, n: n2 }) => (w1, n1).cmp(&(w2, n2)),
        (TypeKind::Param(x), TypeKind::Param(y)) | (TypeKind::ConstParam(x), TypeKind::ConstParam(y)) => {
            x.cmp(y)
        }
        (TypeKind::Range { elem: e1, inclusive: i1 }, TypeKind::Range { elem: e2, inclusive: i2 }) => {
            ct(*e1, *e2).then_with(|| i1.cmp(i2))
        }
        _ => Ordering::Equal,
    }
}

/// A body's type overlay over the frozen global table.
pub struct Overlay<'g> {
    pub global: &'g TypeTable,
    pub local: Vec<TypeKind>,
    map: FxHashMap<TypeKind, TypeId>,
}

impl<'g> Overlay<'g> {
    pub fn new(global: &'g TypeTable) -> Overlay<'g> {
        Overlay { global, local: Vec::new(), map: FxHashMap::default() }
    }
}

impl Interner for Overlay<'_> {
    fn kind(&self, t: TypeId) -> &TypeKind {
        static ERR: TypeKind = TypeKind::Error;
        if t.is_local() { self.local.get(t.idx()).unwrap_or(&ERR) } else { self.global.kind(t) }
    }

    #[inline]
    fn fixed_ty(&self, f: TypeId) -> TypeId {
        self.global.fixed_ty(f)
    }

    fn intern(&mut self, k: TypeKind) -> TypeId {
        if let Some(id) = self.global.lookup(&k) {
            return id;
        }
        if let Some(&id) = self.map.get(&k) {
            return id;
        }
        let id = TypeId(self.local.len() as u32 | TypeId::LOCAL);
        self.local.push(k.clone());
        self.map.insert(k, id);
        id
    }
}

/// The scalar type of a built-in scalar name (`str` included).
pub fn scalar_type(s: Scalar) -> TypeId {
    match s {
        Scalar::I8 => ids::I8,
        Scalar::I16 => ids::I16,
        Scalar::I32 => ids::I32,
        Scalar::I64 => ids::I64,
        Scalar::Isize => ids::ISIZE,
        Scalar::U8 => ids::U8,
        Scalar::U16 => ids::U16,
        Scalar::U32 => ids::U32,
        Scalar::U64 => ids::U64,
        Scalar::Usize => ids::USIZE,
        Scalar::F16 => ids::F16,
        Scalar::Bf16 => ids::BF16,
        Scalar::F32 => ids::F32,
        Scalar::F64 => ids::F64,
        Scalar::Bool => ids::BOOL,
        Scalar::Char => ids::CHAR,
        Scalar::Str => ids::STR,
    }
}

/// The scalar name of a scalar type.
pub fn type_scalar(k: &TypeKind) -> Option<Scalar> {
    Some(match k {
        TypeKind::Int { signed: true, bits: IntBits::B8 } => Scalar::I8,
        TypeKind::Int { signed: true, bits: IntBits::B16 } => Scalar::I16,
        TypeKind::Int { signed: true, bits: IntBits::B32 } => Scalar::I32,
        TypeKind::Int { signed: true, bits: IntBits::B64 } => Scalar::I64,
        TypeKind::Int { signed: true, bits: IntBits::Size } => Scalar::Isize,
        TypeKind::Int { signed: false, bits: IntBits::B8 } => Scalar::U8,
        TypeKind::Int { signed: false, bits: IntBits::B16 } => Scalar::U16,
        TypeKind::Int { signed: false, bits: IntBits::B32 } => Scalar::U32,
        TypeKind::Int { signed: false, bits: IntBits::B64 } => Scalar::U64,
        TypeKind::Int { signed: false, bits: IntBits::Size } => Scalar::Usize,
        TypeKind::Float(FloatBits::F16) => Scalar::F16,
        TypeKind::Float(FloatBits::B16) => Scalar::Bf16,
        TypeKind::Float(FloatBits::F32) => Scalar::F32,
        TypeKind::Float(FloatBits::F64) => Scalar::F64,
        TypeKind::Bool => Scalar::Bool,
        TypeKind::Char => Scalar::Char,
        TypeKind::Str => Scalar::Str,
        _ => return None,
    })
}

/// Lane width in bits of a vector lane type.
pub fn lane_bits(k: &TypeKind) -> u32 {
    match k {
        TypeKind::Int { bits, .. } => bits.bits(),
        TypeKind::Float(FloatBits::F16 | FloatBits::B16) => 16,
        TypeKind::Float(FloatBits::F32) => 32,
        TypeKind::Float(FloatBits::F64) => 64,
        _ => 0,
    }
}

/// Printing context for the surface spelling (diagnostics).
pub trait Names {
    /// The name of an ADT decl.
    fn adt_name(&self, d: DeclId) -> String;
    /// The name of generic parameter `i` in the current context.
    fn param_name(&self, i: u32) -> String;
}

/// The Stave surface spelling of `t` (`&mut [u8]`, `Option<u32>`, `fn(i32) -> i32`), used in
/// diagnostics. ADTs print by their declared name. Depth-limited.
pub fn display<I: Interner + ?Sized>(cx: &I, names: &dyn Names, t: TypeId) -> String {
    let mut s = String::new();
    display_into(cx, names, t, &mut s, 0);
    s
}

pub(crate) fn feature_names(bits: u64) -> Vec<&'static str> {
    let Some(fs) = stave_target::FeatureSet::from_bits(bits) else { return Vec::new() };
    let all: Vec<stave_target::Feature> = fs.iter().collect();
    // Only features not implied by another member of the set.
    all.iter().filter(|f| !all.iter().any(|g| g != *f && g.implies().contains(f))).map(|f| f.name()).collect()
}

fn display_into<I: Interner + ?Sized>(cx: &I, names: &dyn Names, t: TypeId, s: &mut String, depth: u32) {
    if depth > 64 {
        s.push_str("..");
        return;
    }
    let rec = |t: TypeId, s: &mut String| display_into(cx, names, t, s, depth + 1);
    let lanes = |n: &Lanes| match n {
        Lanes::Fixed(k) => k.to_string(),
        Lanes::Native => "n".to_string(),
    };
    match cx.kind(t) {
        k @ (TypeKind::Int { .. } | TypeKind::Float(_) | TypeKind::Bool | TypeKind::Char | TypeKind::Str) => {
            s.push_str(type_scalar(k).map_or("?", |x| x.name()))
        }
        TypeKind::Ref { mutable, inner } => {
            s.push_str(if *mutable { "&mut " } else { "&" });
            rec(*inner, s);
        }
        TypeKind::Ptr { mutable, inner } => {
            s.push_str(if *mutable { "*mut " } else { "*const " });
            rec(*inner, s);
        }
        TypeKind::Array { elem, len } => {
            s.push('[');
            rec(*elem, s);
            s.push_str("; ");
            match len {
                ConstArg::Value(v) => s.push_str(&v.to_string()),
                ConstArg::Param(i) => s.push_str(&names.param_name(*i)),
                ConstArg::Error => s.push('_'),
            }
            s.push(']');
        }
        TypeKind::Slice { elem, align } => {
            s.push('[');
            rec(*elem, s);
            if *align != 0 {
                s.push_str(&format!(" align {align}"));
            }
            s.push(']');
        }
        TypeKind::Tuple(ts) => {
            s.push('(');
            for (i, x) in ts.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                rec(*x, s);
            }
            if ts.len() == 1 {
                s.push(',');
            }
            s.push(')');
        }
        TypeKind::Adt { def, args } => {
            s.push_str(&names.adt_name(*def));
            if !args.is_empty() {
                s.push('<');
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    match a {
                        GArg::Type(x) => rec(*x, s),
                        GArg::Const(ConstArg::Value(v)) => s.push_str(&v.to_string()),
                        GArg::Const(ConstArg::Param(i)) => s.push_str(&names.param_name(*i)),
                        GArg::Const(ConstArg::Error) => s.push('_'),
                    }
                }
                s.push('>');
            }
        }
        TypeKind::Fn { params, ret, features, deny } => {
            s.push_str("fn(");
            for (i, x) in params.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                rec(*x, s);
            }
            s.push(')');
            if *ret != cx.fixed_ty(ids::UNIT) {
                s.push_str(" -> ");
                rec(*ret, s);
            }
            let f = feature_names(*features);
            if !f.is_empty() {
                s.push_str(&format!(" needs({})", f.join(", ")));
            }
            if *deny != 0 {
                s.push_str(&format!(
                    " without({})",
                    crate::effects::Effects(*deny).minimal().names().join(", ")
                ));
            }
        }
        TypeKind::Vector { lane, n } => {
            let l = type_scalar(cx.kind(*lane)).map_or("?", |x| x.name());
            s.push_str(&format!("{l}x{}", lanes(n)));
        }
        TypeKind::Mask { width, n } => s.push_str(&format!("mask{width}x{}", lanes(n))),
        TypeKind::Param(i) | TypeKind::ConstParam(i) => s.push_str(&names.param_name(*i)),
        TypeKind::Never => s.push_str("never"),
        TypeKind::Range { elem, inclusive } => {
            s.push_str(if *inclusive { "RangeInclusive<" } else { "Range<" });
            rec(*elem, s);
            s.push('>');
        }
        TypeKind::SelfTy => s.push_str("Self"),
        TypeKind::Error => s.push('_'),
    }
}

/// The checked-IR `TypeKey` text of `t` (canonical ids must be final): `int s 32`, `ref mut t4`.
pub fn type_key_text(table: &TypeTable, t: TypeId) -> String {
    let tid = |x: &TypeId| format!("t{}", x.0);
    let lanes = |n: &Lanes| match n {
        Lanes::Fixed(k) => k.to_string(),
        Lanes::Native => "n".to_string(),
    };
    let carg = |c: &ConstArg| match c {
        ConstArg::Value(v) => v.to_string(),
        ConstArg::Param(i) => format!("constparam {i}"),
        ConstArg::Error => "_".to_string(),
    };
    match table.kind(t) {
        TypeKind::Int { signed, bits } => {
            let b = match bits {
                IntBits::B8 => "8",
                IntBits::B16 => "16",
                IntBits::B32 => "32",
                IntBits::B64 => "64",
                IntBits::Size => "size",
            };
            format!("int {} {b}", if *signed { "s" } else { "u" })
        }
        TypeKind::Float(f) => format!(
            "float {}",
            match f {
                FloatBits::F16 => "16",
                FloatBits::F32 => "32",
                FloatBits::F64 => "64",
                FloatBits::B16 => "b16",
            }
        ),
        TypeKind::Bool => "bool".into(),
        TypeKind::Char => "char".into(),
        TypeKind::Str => "str".into(),
        TypeKind::Ref { mutable, inner } => {
            format!("ref {}{}", if *mutable { "mut " } else { "" }, tid(inner))
        }
        TypeKind::Ptr { mutable, inner } => {
            format!("ptr {} {}", if *mutable { "mut" } else { "const" }, tid(inner))
        }
        TypeKind::Array { elem, len } => format!("array {} {}", tid(elem), carg(len)),
        TypeKind::Slice { elem, align } => {
            if *align == 0 {
                format!("slice {}", tid(elem))
            } else {
                format!("slice {} align {align}", tid(elem))
            }
        }
        TypeKind::Tuple(ts) => {
            let mut s = "tuple".to_string();
            for x in ts.iter() {
                s.push(' ');
                s.push_str(&tid(x));
            }
            s
        }
        TypeKind::Adt { def, args } => {
            let mut s = format!("adt d{}", def.0);
            for a in args.iter() {
                s.push(' ');
                match a {
                    GArg::Type(x) => s.push_str(&tid(x)),
                    GArg::Const(c) => s.push_str(&carg(c)),
                }
            }
            s
        }
        // The deny clause is not part of the checked IR's fn type.
        TypeKind::Fn { params, ret, features, .. } => {
            let mut s = "fntype".to_string();
            for x in params.iter() {
                s.push(' ');
                s.push_str(&tid(x));
            }
            s.push_str(" -> ");
            s.push_str(&tid(ret));
            for f in feature_names(*features) {
                s.push(' ');
                s.push_str(f);
            }
            s
        }
        TypeKind::Vector { lane, n } => format!("vec {} {}", tid(lane), lanes(n)),
        TypeKind::Mask { width, n } => format!("mask {width} {}", lanes(n)),
        TypeKind::Param(i) => format!("param {i}"),
        TypeKind::ConstParam(i) => format!("constparam {i}"),
        TypeKind::Never => "never".into(),
        TypeKind::Range { elem, inclusive } => {
            if *inclusive {
                format!("range incl {}", tid(elem))
            } else {
                format!("range {}", tid(elem))
            }
        }
        TypeKind::SelfTy => "self".into(),
        TypeKind::Error => "error".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_ids_and_canonical_order() {
        let mut t = TypeTable::new();
        assert_eq!(t.kind(ids::U8), &TypeKind::Int { signed: false, bits: IntBits::B8 });
        let r = t.mk_ref(true, ids::I32);
        let tup = t.mk_tuple(vec![ids::BOOL, r]);
        let remap = t.canonicalize();
        // ints first (u before s), error last
        assert_eq!(remap[ids::U8.idx()], TypeId(0));
        assert_eq!(remap[ids::ERROR.idx()], TypeId(t.len() as u32 - 1));
        let nr = remap[r.idx()];
        assert_eq!(type_key_text(&t, nr), format!("ref mut t{}", remap[ids::I32.idx()].0));
        let nt = remap[tup.idx()];
        assert!(type_key_text(&t, nt).starts_with("tuple t"));
        assert_eq!(type_key_text(&t, remap[ids::ISIZE.idx()]), "int s size");
    }

    #[test]
    fn overlay_and_merge() {
        let g = TypeTable::new();
        let mut o = Overlay::new(&g);
        let a = o.mk_ref(false, ids::U8);
        let b = o.mk_ref(false, a);
        assert!(a.is_local() && b.is_local());
        assert_eq!(o.mk_ref(false, ids::U8), a);
        let local = o.local.clone();
        let mut g2 = g.clone();
        let m = g2.merge(&local);
        assert_eq!(g2.kind(m[1]), &TypeKind::Ref { mutable: false, inner: m[0] });
    }

    /// The helpers on a canonicalized table: canonical `u8` is id 0 (the fresh table's error id),
    /// so a helper that compared against a fixed id would turn `Vec<u8>` into `u8` (regression).
    #[test]
    fn helpers_on_a_canonical_table() {
        let mut t = TypeTable::new();
        let p0 = t.mk_param(0);
        let vec_t = t.mk_adt(DeclId(7), vec![GArg::Type(p0)]);
        let ref_vec = t.mk_ref(false, vec_t);
        let arr = t.intern(TypeKind::Array { elem: p0, len: ConstArg::Param(1) });
        let fresh = t.clone();
        let remap = t.canonicalize();
        let [p0, vec_t, ref_vec, arr] = [p0, vec_t, ref_vec, arr].map(|x| remap[x.idx()]);
        let u8t = t.scalar_ty(Scalar::U8);
        assert_eq!(u8t, TypeId(0));
        assert_eq!(t.kind(u8t), &TypeKind::Int { signed: false, bits: IntBits::B8 });
        let err = t.error_ty();
        assert_eq!(err, TypeId(t.len() as u32 - 1));
        assert_eq!(t.error_id(), err);
        assert!(t.is_error_ty(err) && !t.is_error_ty(u8t));
        struct N;
        impl Names for N {
            fn adt_name(&self, _: DeclId) -> String {
                "Vec".into()
            }
            fn param_name(&self, i: u32) -> String {
                format!("T{i}")
            }
        }
        for f in 0..ids::COUNT {
            let (a, b) = (display(&t, &N, t.fixed_ty(TypeId(f))), display(&fresh, &N, TypeId(f)));
            assert_eq!(a, b, "fixed {f}");
        }
        assert_eq!(display(&t, &N, t.fixed_ty(ids::REF_STR)), "&str");
        assert_eq!(t.fixed_ty(TypeId(1000)), TypeId(1000));
        // substitution
        let s = t.subst(vec_t, &[GArg::Type(u8t)], None);
        assert_eq!(t.kind(s), &TypeKind::Adt { def: DeclId(7), args: Box::new([GArg::Type(u8t)]) });
        let s = t.subst(ref_vec, &[GArg::Type(u8t)], None);
        let TypeKind::Ref { mutable: false, inner } = *t.kind(s) else { panic!("{:?}", t.kind(s)) };
        assert_eq!(t.kind(inner), &TypeKind::Adt { def: DeclId(7), args: Box::new([GArg::Type(u8t)]) });
        let s = t.subst(arr, &[GArg::Type(u8t), GArg::Const(ConstArg::Value(4))], None);
        assert_eq!(t.kind(s), &TypeKind::Array { elem: u8t, len: ConstArg::Value(4) });
        assert_eq!(t.subst(p0, &[GArg::Type(u8t)], None), u8t);
        // constructors
        let r = t.mk_ref(true, u8t);
        assert_eq!(t.kind(r), &TypeKind::Ref { mutable: true, inner: u8t });
        let r = t.mk_ptr(false, u8t);
        assert_eq!(t.kind(r), &TypeKind::Ptr { mutable: false, inner: u8t });
        let tup = t.mk_tuple(vec![u8t, u8t]);
        assert_eq!(t.kind(tup), &TypeKind::Tuple(Box::new([u8t, u8t])));
        let a = t.mk_adt(DeclId(7), vec![GArg::Type(u8t)]);
        assert!(matches!(t.kind(a), TypeKind::Adt { .. }));
        // the error type still absorbs
        assert_eq!(t.mk_ref(false, err), err);
        assert_eq!(t.mk_tuple(vec![u8t, err]), err);
        assert_eq!(t.mk_adt(DeclId(7), vec![GArg::Type(err)]), err);
        assert_eq!(t.subst(vec_t, &[GArg::Type(err)], None), err);
        // printing: `fn()` has no `-> ()` (the unit id moved too)
        let f = t.intern(TypeKind::Fn {
            params: Box::new([u8t]),
            ret: t.fixed_ty(ids::UNIT),
            features: 0,
            deny: 0,
        });
        assert_eq!(display(&t, &N, f), "fn(u8)");
        assert_eq!(display(&t, &N, s), "[u8; 4]");
        // overlays and merges over the canonical table
        let mut o = Overlay::new(&t);
        assert_eq!(o.error_ty(), err);
        assert_eq!(o.scalar_ty(Scalar::U8), u8t);
        let l = o.mk_ref(true, a);
        assert!(l.is_local());
        assert_eq!(o.mk_ref(false, o.error_ty()), err);
        let local = o.local.clone();
        let m = t.merge(&local);
        assert_eq!(t.kind(m[0]), &TypeKind::Ref { mutable: true, inner: a });
    }

    #[test]
    fn subst_params() {
        let mut t = TypeTable::new();
        let p0 = t.mk_param(0);
        let r = t.mk_ref(false, p0);
        let s = t.subst(r, &[GArg::Type(ids::U32)], None);
        assert_eq!(t.kind(s), &TypeKind::Ref { mutable: false, inner: ids::U32 });
    }
}
