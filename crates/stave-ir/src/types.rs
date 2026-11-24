//! IR types: the type key [`TypeKey`] (canonical type ids), ADT shapes, the TypeKey text printer
//! (contracts/checked-ir.md "TypeKey text"), and canonical renumbering.
//!
//! Every IR value carries a [`TypeId`]. When the module is printed, `TypeId(n)` prints as `t<n>`,
//! so ids must be canonical by then: either the front end interns in canonical order, or it calls
//! [`TypeTable::canonicalize`] (through `Module::canonicalize_types`) once the table holds every
//! type of the program.

use std::cmp::Ordering;
use std::fmt::Write as _;

use rustc_hash::FxHashMap;
use stave_target::{Feature, FeatureSet};

/// A type id: the index into [`TypeTable`]; printed as `t<n>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub u32);

/// A declaration id (canonical decl id D); printed as `d<n>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeclId(pub u32);

/// Integer width. Order is the canonical key order: `8 16 32 64 size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntBits {
    B8,
    B16,
    B32,
    B64,
    /// `isize` / `usize`: 64 bits on every Stave target.
    Size,
}

impl IntBits {
    /// Width in bits (`Size` = 64).
    pub fn width(self) -> u8 {
        match self {
            IntBits::B8 => 8,
            IntBits::B16 => 16,
            IntBits::B32 => 32,
            IntBits::B64 | IntBits::Size => 64,
        }
    }

    fn text(self) -> &'static str {
        match self {
            IntBits::B8 => "8",
            IntBits::B16 => "16",
            IntBits::B32 => "32",
            IntBits::B64 => "64",
            IntBits::Size => "size",
        }
    }
}

/// Float format. Order is the canonical key order: `16 32 64 b16` (bf16 last).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FloatBits {
    F16,
    F32,
    F64,
    /// bfloat16.
    B16,
}

impl FloatBits {
    fn text(self) -> &'static str {
        match self {
            FloatBits::F16 => "16",
            FloatBits::F32 => "32",
            FloatBits::F64 => "64",
            FloatBits::B16 => "b16",
        }
    }

    /// Storage width in bits.
    pub fn width(self) -> u8 {
        match self {
            FloatBits::F16 | FloatBits::B16 => 16,
            FloatBits::F32 => 32,
            FloatBits::F64 => 64,
        }
    }
}

/// Vector / mask lane count: fixed, or the sizeless SVE count (`n`). `Fixed` sorts first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lanes {
    Fixed(u8),
    Native,
}

/// A generic argument of an ADT: a type, or a const value (printed as a decimal integer).
/// Types sort before consts at the same position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GenericArg {
    Type(TypeId),
    Const(i128),
}

/// The type key. Variant order is the canonical order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeKey {
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
    RawPtr {
        mutable: bool,
        inner: TypeId,
    },
    Array {
        elem: TypeId,
        len: u64,
    },
    Slice {
        elem: TypeId,
        align: Option<u32>,
    },
    /// The unit type is the empty tuple.
    Tuple(Vec<TypeId>),
    Adt {
        def: DeclId,
        args: Vec<GenericArg>,
    },
    FnType {
        params: Vec<TypeId>,
        ret: TypeId,
        features: FeatureSet,
    },
    Vec {
        lane: TypeId,
        n: Lanes,
    },
    Mask {
        width: u8,
        n: Lanes,
    },
    Param(u32),
    ConstParam(u32),
}

impl TypeKey {
    fn variant_rank(&self) -> u8 {
        match self {
            TypeKey::Int { .. } => 0,
            TypeKey::Float(_) => 1,
            TypeKey::Bool => 2,
            TypeKey::Char => 3,
            TypeKey::Str => 4,
            TypeKey::Ref { .. } => 5,
            TypeKey::RawPtr { .. } => 6,
            TypeKey::Array { .. } => 7,
            TypeKey::Slice { .. } => 8,
            TypeKey::Tuple(_) => 9,
            TypeKey::Adt { .. } => 10,
            TypeKey::FnType { .. } => 11,
            TypeKey::Vec { .. } => 12,
            TypeKey::Mask { .. } => 13,
            TypeKey::Param(_) => 14,
            TypeKey::ConstParam(_) => 15,
        }
    }

    /// Component types, in key order.
    pub fn components(&self) -> Vec<TypeId> {
        let mut out = Vec::new();
        self.for_each_component(|t| out.push(t));
        out
    }

    /// Calls `f` on every component type, in key order.
    pub fn for_each_component(&self, mut f: impl FnMut(TypeId)) {
        match self {
            TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. } => f(*inner),
            TypeKey::Array { elem, .. } | TypeKey::Slice { elem, .. } => f(*elem),
            TypeKey::Tuple(ts) => ts.iter().copied().for_each(f),
            TypeKey::Adt { args, .. } => {
                for a in args {
                    if let GenericArg::Type(t) = a {
                        f(*t);
                    }
                }
            }
            TypeKey::FnType { params, ret, .. } => {
                params.iter().copied().for_each(&mut f);
                f(*ret);
            }
            TypeKey::Vec { lane, .. } => f(*lane),
            _ => {}
        }
    }

    /// Applies `f` to every component type id in place.
    pub fn map_components(&mut self, mut f: impl FnMut(TypeId) -> TypeId) {
        match self {
            TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. } => *inner = f(*inner),
            TypeKey::Array { elem, .. } | TypeKey::Slice { elem, .. } => *elem = f(*elem),
            TypeKey::Tuple(ts) => ts.iter_mut().for_each(|t| *t = f(*t)),
            TypeKey::Adt { args, .. } => {
                for a in args {
                    if let GenericArg::Type(t) = a {
                        *t = f(*t);
                    }
                }
            }
            TypeKey::FnType { params, ret, .. } => {
                params.iter_mut().for_each(|t| *t = f(*t));
                *ret = f(*ret);
            }
            TypeKey::Vec { lane, .. } => *lane = f(*lane),
            _ => {}
        }
    }

    /// True for the unit type (empty tuple).
    pub fn is_unit(&self) -> bool {
        matches!(self, TypeKey::Tuple(ts) if ts.is_empty())
    }
}

/// Struct or enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdtKind {
    Struct,
    Enum,
}

/// A declaration an ADT type or [`crate::ir::Module::cell`] refers to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclInfo {
    pub id: DeclId,
    pub kind: AdtKind,
    /// The declaration's dotted path (`std.cell.UnsafeCell`).
    pub path: String,
}

/// The monomorphized shape of an ADT instance type: field types per variant (a struct has one
/// variant). Needed by aggregate ops, layout, and the interpreter; not printed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdtShape {
    pub kind: AdtKind,
    pub variants: Vec<Vec<TypeId>>,
}

impl AdtShape {
    pub fn structure(fields: Vec<TypeId>) -> AdtShape {
        AdtShape { kind: AdtKind::Struct, variants: vec![fields] }
    }

    pub fn enumeration(variants: Vec<Vec<TypeId>>) -> AdtShape {
        AdtShape { kind: AdtKind::Enum, variants }
    }
}

/// Error of [`TypeTable::canonicalize`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonError {
    /// A component id is out of range.
    BadComponent { ty: TypeId, component: TypeId },
    /// The key graph has a cycle through this type (keys must be strictly larger than their parts).
    Cycle(TypeId),
    /// Nesting deeper than [`MAX_TYPE_DEPTH`].
    TooDeep(TypeId),
}

/// The deepest type nesting the recursive type algorithms (canonical compare, layout) accept.
pub const MAX_TYPE_DEPTH: usize = 1_000;

/// Interned type table. Equal keys get equal ids.
#[derive(Clone, Debug, Default)]
pub struct TypeTable {
    keys: Vec<TypeKey>,
    shapes: Vec<Option<AdtShape>>,
    map: FxHashMap<TypeKey, TypeId>,
}

/// Structural: the same keys and shapes by id (`map` is derived from `keys`).
impl PartialEq for TypeTable {
    fn eq(&self, other: &TypeTable) -> bool {
        self.keys == other.keys && self.shapes == other.shapes
    }
}

impl TypeTable {
    pub fn new() -> TypeTable {
        TypeTable::default()
    }

    /// Interns `key` (component ids are not checked here; the verifier checks them).
    pub fn intern(&mut self, key: TypeKey) -> TypeId {
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        let id = TypeId(self.keys.len() as u32);
        self.keys.push(key.clone());
        self.shapes.push(None);
        self.map.insert(key, id);
        id
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn key(&self, t: TypeId) -> Option<&TypeKey> {
        self.keys.get(t.0 as usize)
    }

    pub fn keys(&self) -> &[TypeKey] {
        &self.keys
    }

    /// Looks up an interned key.
    pub fn find(&self, key: &TypeKey) -> Option<TypeId> {
        self.map.get(key).copied()
    }

    /// Sets the shape of an ADT type. Returns false if `t` is not an `Adt` key.
    pub fn set_shape(&mut self, t: TypeId, shape: AdtShape) -> bool {
        match (self.keys.get(t.0 as usize), self.shapes.get_mut(t.0 as usize)) {
            (Some(TypeKey::Adt { .. }), Some(slot)) => {
                *slot = Some(shape);
                true
            }
            _ => false,
        }
    }

    pub fn shape(&self, t: TypeId) -> Option<&AdtShape> {
        self.shapes.get(t.0 as usize).and_then(|s| s.as_ref())
    }

    /// For an `Option`-shaped enum (two variants: one empty, one with one field): (none variant,
    /// some variant, payload type). The `cast.try` result and its layout follow this shape.
    pub fn option_shape(&self, t: TypeId) -> Option<(u32, u32, TypeId)> {
        let s = self.shape(t)?;
        if s.kind != AdtKind::Enum || s.variants.len() != 2 {
            return None;
        }
        match (s.variants[0].as_slice(), s.variants[1].as_slice()) {
            ([], [p]) => Some((0, 1, *p)),
            ([p], []) => Some((1, 0, *p)),
            _ => None,
        }
    }

    // Convenience constructors.
    pub fn int(&mut self, signed: bool, bits: IntBits) -> TypeId {
        self.intern(TypeKey::Int { signed, bits })
    }
    pub fn i8(&mut self) -> TypeId {
        self.int(true, IntBits::B8)
    }
    pub fn i16(&mut self) -> TypeId {
        self.int(true, IntBits::B16)
    }
    pub fn i32(&mut self) -> TypeId {
        self.int(true, IntBits::B32)
    }
    pub fn i64(&mut self) -> TypeId {
        self.int(true, IntBits::B64)
    }
    pub fn isize(&mut self) -> TypeId {
        self.int(true, IntBits::Size)
    }
    pub fn u8(&mut self) -> TypeId {
        self.int(false, IntBits::B8)
    }
    pub fn u16(&mut self) -> TypeId {
        self.int(false, IntBits::B16)
    }
    pub fn u32(&mut self) -> TypeId {
        self.int(false, IntBits::B32)
    }
    pub fn u64(&mut self) -> TypeId {
        self.int(false, IntBits::B64)
    }
    pub fn usize(&mut self) -> TypeId {
        self.int(false, IntBits::Size)
    }
    pub fn f32(&mut self) -> TypeId {
        self.intern(TypeKey::Float(FloatBits::F32))
    }
    pub fn f64(&mut self) -> TypeId {
        self.intern(TypeKey::Float(FloatBits::F64))
    }
    pub fn bool(&mut self) -> TypeId {
        self.intern(TypeKey::Bool)
    }
    pub fn char(&mut self) -> TypeId {
        self.intern(TypeKey::Char)
    }
    pub fn str(&mut self) -> TypeId {
        self.intern(TypeKey::Str)
    }
    pub fn unit(&mut self) -> TypeId {
        self.intern(TypeKey::Tuple(Vec::new()))
    }
    pub fn tuple(&mut self, elems: Vec<TypeId>) -> TypeId {
        self.intern(TypeKey::Tuple(elems))
    }
    pub fn ref_(&mut self, mutable: bool, inner: TypeId) -> TypeId {
        self.intern(TypeKey::Ref { mutable, inner })
    }
    pub fn ptr(&mut self, mutable: bool, inner: TypeId) -> TypeId {
        self.intern(TypeKey::RawPtr { mutable, inner })
    }
    pub fn array(&mut self, elem: TypeId, len: u64) -> TypeId {
        self.intern(TypeKey::Array { elem, len })
    }
    pub fn slice(&mut self, elem: TypeId) -> TypeId {
        self.intern(TypeKey::Slice { elem, align: None })
    }
    pub fn fn_type(&mut self, params: Vec<TypeId>, ret: TypeId, features: FeatureSet) -> TypeId {
        self.intern(TypeKey::FnType { params, ret, features })
    }
    /// Interns an ADT instance type and sets its shape.
    pub fn adt(&mut self, def: DeclId, args: Vec<GenericArg>, shape: AdtShape) -> TypeId {
        let t = self.intern(TypeKey::Adt { def, args });
        self.set_shape(t, shape);
        t
    }

    /// Canonical comparison of two types: variant in listed order,
    /// then fields in declaration order, components by their own canonical order.
    pub fn canonical_cmp(&self, a: TypeId, b: TypeId) -> Result<Ordering, CanonError> {
        self.cmp_depth(a, b, 0)
    }

    fn cmp_depth(&self, a: TypeId, b: TypeId, depth: usize) -> Result<Ordering, CanonError> {
        if a == b {
            return Ok(Ordering::Equal);
        }
        if depth > MAX_TYPE_DEPTH {
            return Err(CanonError::TooDeep(a));
        }
        let ka = self.key(a).ok_or(CanonError::BadComponent { ty: a, component: a })?;
        let kb = self.key(b).ok_or(CanonError::BadComponent { ty: b, component: b })?;
        let d = depth + 1;
        let o = ka.variant_rank().cmp(&kb.variant_rank());
        if o != Ordering::Equal {
            return Ok(o);
        }
        let list = |x: &[TypeId], y: &[TypeId]| -> Result<Ordering, CanonError> {
            for (p, q) in x.iter().zip(y) {
                let o = self.cmp_depth(*p, *q, d)?;
                if o != Ordering::Equal {
                    return Ok(o);
                }
            }
            Ok(x.len().cmp(&y.len()))
        };
        use TypeKey as K;
        Ok(match (ka, kb) {
            (K::Int { signed: s1, bits: b1 }, K::Int { signed: s2, bits: b2 }) => (s1, b1).cmp(&(s2, b2)),
            (K::Float(x), K::Float(y)) => x.cmp(y),
            (K::Ref { mutable: m1, inner: i1 }, K::Ref { mutable: m2, inner: i2 })
            | (K::RawPtr { mutable: m1, inner: i1 }, K::RawPtr { mutable: m2, inner: i2 }) => {
                m1.cmp(m2).then(self.cmp_depth(*i1, *i2, d)?)
            }
            (K::Array { elem: e1, len: l1 }, K::Array { elem: e2, len: l2 }) => {
                self.cmp_depth(*e1, *e2, d)?.then(l1.cmp(l2))
            }
            (K::Slice { elem: e1, align: a1 }, K::Slice { elem: e2, align: a2 }) => {
                self.cmp_depth(*e1, *e2, d)?.then(a1.cmp(a2))
            }
            (K::Tuple(x), K::Tuple(y)) => list(x, y)?,
            (K::Adt { def: d1, args: a1 }, K::Adt { def: d2, args: a2 }) => {
                let mut o = d1.cmp(d2);
                if o == Ordering::Equal {
                    for (p, q) in a1.iter().zip(a2) {
                        o = match (p, q) {
                            (GenericArg::Type(x), GenericArg::Type(y)) => self.cmp_depth(*x, *y, d)?,
                            (GenericArg::Const(x), GenericArg::Const(y)) => x.cmp(y),
                            (GenericArg::Type(_), GenericArg::Const(_)) => Ordering::Less,
                            (GenericArg::Const(_), GenericArg::Type(_)) => Ordering::Greater,
                        };
                        if o != Ordering::Equal {
                            break;
                        }
                    }
                    if o == Ordering::Equal {
                        o = a1.len().cmp(&a2.len());
                    }
                }
                o
            }
            (
                K::FnType { params: p1, ret: r1, features: f1 },
                K::FnType { params: p2, ret: r2, features: f2 },
            ) => {
                // the return type only on equal params: comparing it anyway costs 2^depth on
                // fn types that share components (`type A = fn(B, B) -> B`)
                match list(p1, p2)? {
                    Ordering::Equal => self.cmp_depth(*r1, *r2, d)?.then(f1.bits().cmp(&f2.bits())),
                    o => o,
                }
            }
            (K::Vec { lane: l1, n: n1 }, K::Vec { lane: l2, n: n2 }) => {
                self.cmp_depth(*l1, *l2, d)?.then(n1.cmp(n2))
            }
            (K::Mask { width: w1, n: n1 }, K::Mask { width: w2, n: n2 }) => (w1, n1).cmp(&(w2, n2)),
            (K::Param(x), K::Param(y)) | (K::ConstParam(x), K::ConstParam(y)) => x.cmp(y),
            _ => Ordering::Equal,
        })
    }

    /// Checks that every component id is in range, the key graph is acyclic, and nesting is at
    /// most [`MAX_TYPE_DEPTH`].
    pub fn check_well_formed(&self) -> Result<(), CanonError> {
        let n = self.keys.len();
        for (i, k) in self.keys.iter().enumerate() {
            let mut bad = None;
            k.for_each_component(|c| {
                if c.0 as usize >= n && bad.is_none() {
                    bad = Some(c);
                }
            });
            if let Some(c) = bad {
                return Err(CanonError::BadComponent { ty: TypeId(i as u32), component: c });
            }
        }
        // Iterative DFS: cycle check and nesting height. state 0 = new, 1 = on stack, 2 = done.
        let mut state = vec![0u8; n];
        let mut height = vec![0usize; n];
        for root in 0..n {
            if state[root] != 0 {
                continue;
            }
            let mut stack: Vec<(usize, Vec<TypeId>, usize)> = vec![(root, self.keys[root].components(), 0)];
            state[root] = 1;
            while let Some((node, comps, idx)) = stack.last_mut() {
                if *idx < comps.len() {
                    let c = comps[*idx].0 as usize;
                    *idx += 1;
                    match state[c] {
                        0 => {
                            state[c] = 1;
                            let cc = self.keys[c].components();
                            stack.push((c, cc, 0));
                        }
                        1 => return Err(CanonError::Cycle(TypeId(c as u32))),
                        _ => {}
                    }
                } else {
                    let node = *node;
                    let h = comps.iter().map(|c| height[c.0 as usize]).max().unwrap_or(0) + 1;
                    if h > MAX_TYPE_DEPTH {
                        return Err(CanonError::TooDeep(TypeId(node as u32)));
                    }
                    height[node] = h;
                    state[node] = 2;
                    stack.pop();
                }
            }
        }
        Ok(())
    }

    /// Renumbers the table to canonical ids (the rank of each key in canonical order) and returns
    /// the map old id → new id (index = old id). Shapes are carried and remapped.
    pub fn canonicalize(&mut self) -> Result<Vec<TypeId>, CanonError> {
        self.check_well_formed()?;
        let n = self.keys.len();
        let mut order: Vec<TypeId> = (0..n as u32).map(TypeId).collect();
        // Well-formed (in range, acyclic, bounded depth) and interned (distinct keys), so the
        // comparison cannot fail and is a total order.
        order.sort_by(|a, b| self.canonical_cmp(*a, *b).unwrap_or(Ordering::Equal));
        let mut remap = vec![TypeId(0); n];
        for (new, old) in order.iter().enumerate() {
            remap[old.0 as usize] = TypeId(new as u32);
        }
        let mut keys = Vec::with_capacity(n);
        let mut shapes = Vec::with_capacity(n);
        for old in &order {
            let mut k = self.keys[old.0 as usize].clone();
            k.map_components(|t| remap[t.0 as usize]);
            keys.push(k);
            let s = self.shapes[old.0 as usize].clone().map(|mut s| {
                for v in &mut s.variants {
                    for f in v.iter_mut() {
                        *f = remap.get(f.0 as usize).copied().unwrap_or(*f);
                    }
                }
                s
            });
            shapes.push(s);
        }
        self.map = keys.iter().cloned().enumerate().map(|(i, k)| (k, TypeId(i as u32))).collect();
        self.keys = keys;
        self.shapes = shapes;
        Ok(remap)
    }

    /// The TypeKey text of `t` (checked-ir.md "TypeKey text"), e.g. `int s 32`, `ref mut t4`.
    pub fn key_text(&self, t: TypeId) -> String {
        match self.key(t) {
            Some(k) => key_text(k),
            None => String::from("?"),
        }
    }
}

/// The features a `fntype` key prints: the members not implied by another member, in the
/// contract's feature order (so `needs(avx2)` prints `avx2`, not its closure).
pub fn feature_generators(fs: FeatureSet) -> Vec<Feature> {
    fs.iter()
        .filter(|f| {
            !fs.iter().any(|g| g != *f && FeatureSet::of(g).contains(*f) && !FeatureSet::of(*f).contains(g))
        })
        .collect()
}

/// TypeKey text of one key.
pub fn key_text(k: &TypeKey) -> String {
    let mut s = String::new();
    let lanes = |n: &Lanes| match n {
        Lanes::Fixed(k) => k.to_string(),
        Lanes::Native => String::from("n"),
    };
    match k {
        TypeKey::Int { signed, bits } => {
            let _ = write!(s, "int {} {}", if *signed { "s" } else { "u" }, bits.text());
        }
        TypeKey::Float(b) => {
            let _ = write!(s, "float {}", b.text());
        }
        TypeKey::Bool => s.push_str("bool"),
        TypeKey::Char => s.push_str("char"),
        TypeKey::Str => s.push_str("str"),
        TypeKey::Ref { mutable, inner } => {
            let _ = write!(s, "ref {}t{}", if *mutable { "mut " } else { "" }, inner.0);
        }
        TypeKey::RawPtr { mutable, inner } => {
            let _ = write!(s, "ptr {} t{}", if *mutable { "mut" } else { "const" }, inner.0);
        }
        TypeKey::Array { elem, len } => {
            let _ = write!(s, "array t{} {}", elem.0, len);
        }
        TypeKey::Slice { elem, align } => {
            let _ = write!(s, "slice t{}", elem.0);
            if let Some(a) = align {
                let _ = write!(s, " align {a}");
            }
        }
        TypeKey::Tuple(ts) => {
            s.push_str("tuple");
            for t in ts {
                let _ = write!(s, " t{}", t.0);
            }
        }
        TypeKey::Adt { def, args } => {
            let _ = write!(s, "adt d{}", def.0);
            for a in args {
                match a {
                    GenericArg::Type(t) => {
                        let _ = write!(s, " t{}", t.0);
                    }
                    GenericArg::Const(c) => {
                        let _ = write!(s, " {c}");
                    }
                }
            }
        }
        TypeKey::FnType { params, ret, features } => {
            s.push_str("fntype");
            for p in params {
                let _ = write!(s, " t{}", p.0);
            }
            let _ = write!(s, " -> t{}", ret.0);
            for f in feature_generators(*features) {
                let _ = write!(s, " {}", f.name());
            }
        }
        TypeKey::Vec { lane, n } => {
            let _ = write!(s, "vec t{} {}", lane.0, lanes(n));
        }
        TypeKey::Mask { width, n } => {
            let _ = write!(s, "mask {} {}", width, lanes(n));
        }
        TypeKey::Param(i) => {
            let _ = write!(s, "param {i}");
        }
        TypeKey::ConstParam(i) => {
            let _ = write!(s, "constparam {i}");
        }
    }
    s
}
