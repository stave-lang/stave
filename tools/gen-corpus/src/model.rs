//! The planned shape of every module: item signatures and line budgets. Bodies are generated
//! from the plans (phase 2) and never change a plan, so modules generate independently.

/// Scalar types used by the generator.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Sc {
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F64,
    Bool,
}

impl Sc {
    pub const INTS: [Sc; 6] = [Sc::I32, Sc::I64, Sc::U8, Sc::U16, Sc::U32, Sc::U64];
    pub const ALL: [Sc; 8] = [Sc::I32, Sc::I64, Sc::U8, Sc::U16, Sc::U32, Sc::U64, Sc::F64, Sc::Bool];
    /// Relative frequency of each type in `ALL` order.
    pub const WEIGHTS: [u32; 8] = [24, 14, 8, 5, 24, 12, 8, 5];

    pub fn name(self) -> &'static str {
        match self {
            Sc::I32 => "i32",
            Sc::I64 => "i64",
            Sc::U8 => "u8",
            Sc::U16 => "u16",
            Sc::U32 => "u32",
            Sc::U64 => "u64",
            Sc::F64 => "f64",
            Sc::Bool => "bool",
        }
    }

    pub fn is_int(self) -> bool {
        !matches!(self, Sc::F64 | Sc::Bool)
    }

    pub fn signed(self) -> bool {
        matches!(self, Sc::I32 | Sc::I64)
    }

    pub fn bits(self) -> u32 {
        match self {
            Sc::U8 => 8,
            Sc::U16 => 16,
            Sc::I32 | Sc::U32 => 32,
            Sc::I64 | Sc::U64 | Sc::F64 => 64,
            Sc::Bool => 1,
        }
    }

    /// Source types with a lossless `as` to `self`.
    pub fn lossless_from(self) -> &'static [Sc] {
        match self {
            Sc::I32 => &[Sc::U8, Sc::U16],
            Sc::I64 => &[Sc::I32, Sc::U8, Sc::U16, Sc::U32],
            Sc::U16 => &[Sc::U8],
            Sc::U32 => &[Sc::U8, Sc::U16],
            Sc::U64 => &[Sc::U8, Sc::U16, Sc::U32],
            Sc::F64 => &[Sc::I32, Sc::U8, Sc::U16, Sc::U32],
            Sc::U8 | Sc::Bool => &[],
        }
    }
}

/// A struct field type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FTy {
    Sc(Sc),
    /// An earlier, flat (scalar fields only) struct of the same module.
    S(u32),
    /// The type parameter of a generic struct.
    T,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: &'static str,
    pub ty: FTy,
}

#[derive(Clone, Debug)]
pub struct StructDef {
    pub name: String,
    /// `struct Name<T> { pub a: T, pub n: u32, .. }`.
    pub generic: bool,
    pub fields: Vec<Field>,
    /// Index of the inherent impl.
    pub imp: Option<u32>,
    /// `(trait, trait impl)` pairs implemented for this struct.
    pub traits: Vec<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub enum VKind {
    Unit,
    Tup(Vec<Sc>),
    Rec(Vec<(&'static str, Sc)>),
}

#[derive(Clone, Debug)]
pub struct VariantDef {
    pub name: &'static str,
    pub kind: VKind,
}

#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<VariantDef>,
}

/// A fn parameter type. Struct and enum indices are of the fn's own module.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PTy {
    Sc(Sc),
    Ref(u32),
    Mut(u32),
    Val(u32),
    EVal(u32),
    T,
    RefT,
}

/// A fn return type. Struct and enum indices are of the fn's own module.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ret {
    Unit,
    Sc(Sc),
    S(u32),
    E(u32),
    Opt(Sc),
    Res(Sc, u32),
    Tup(Sc, Sc),
    /// `&sc from p0` (p0: `&S`).
    RefSc(Sc),
    /// `&S from p0 + p1`.
    RefS(u32),
    /// `&mut sc from p0` (p0: `&mut S`).
    MutSc(Sc),
    T,
    /// `&T from p0 + p1`.
    RefT,
    /// `Cell<T>` for a generic struct.
    Cell(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FnKind {
    Plain,
    /// `<T>(p0: bool, p1: T, p2: T) -> T`.
    Pick,
    /// `<T>(p0: &T, p1: &T, p2: bool) -> &T from p0 + p1`.
    RefPick,
    /// `<T: Tr>(p0: &T, p1: sc) -> sc`, calling method 0 of trait `Tr` (own module).
    Bound(u32),
    /// `<const N: u32>(p0: u32) -> u32`.
    Const,
    /// `<T>(p0: T) -> Cell<T>`.
    GenCtor(u32),
}

#[derive(Clone, Debug)]
pub struct FnDef {
    pub name: String,
    pub public: bool,
    pub kind: FnKind,
    pub params: Vec<PTy>,
    pub ret: Ret,
    /// Body lines (statements, including the final `return`).
    pub body: u32,
    pub doc: bool,
    /// Call rank: a body calls only fns and methods of lower rank (bounded call depth).
    pub rank: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MKind {
    /// `pub fn make(p0.., ) -> Self` (one param per scalar field).
    Ctor,
    /// `pub fn get_f(&self) -> sc`.
    Get(u32),
    /// `pub fn bump_f(&mut self, p0: sc)` (int field).
    Bump(u32),
    /// `pub fn into_f(self) -> sc`.
    Consume(u32),
    /// `pub fn ref_f(&self) -> &sc from self`.
    RefGet(u32),
    /// `pub fn mix_f(&self, p0: &Self) -> sc` (int or float field).
    Mix(u32),
    /// Generic struct: `pub fn get_a(&self) -> &T from self`.
    GGet,
    /// Generic struct: `pub fn put(&mut self, p0: T)`.
    GPut,
    /// Generic struct: `pub fn count(&self) -> u32`.
    GCount,
}

#[derive(Clone, Debug)]
pub struct Method {
    pub name: String,
    pub kind: MKind,
    pub body: u32,
    pub rank: u8,
}

#[derive(Clone, Debug)]
pub struct ImplDef {
    pub target: u32,
    pub methods: Vec<Method>,
}

#[derive(Clone, Debug)]
pub struct TraitMethod {
    pub name: String,
    pub arg: Option<Sc>,
    /// Always an integer type.
    pub ret: Sc,
}

#[derive(Clone, Debug)]
pub struct TraitDef {
    pub name: String,
    pub methods: Vec<TraitMethod>,
}

#[derive(Clone, Debug)]
pub struct TraitImpl {
    pub tr: u32,
    pub target: u32,
    pub bodies: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ConstDef {
    pub name: String,
    pub ty: Sc,
    pub alias: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct AliasDef {
    pub name: String,
    pub ty: Sc,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Item {
    Fn(u32),
    Struct(u32),
    Enum(u32),
    Impl(u32),
    Trait(u32),
    TraitImpl(u32),
    Const(u32),
    Alias(u32),
}

#[derive(Clone, Debug, Default)]
pub struct ModPlan {
    /// Path segments: `src/a/b.stv` is `["a", "b"]`.
    pub path: Vec<String>,
    /// Imported modules (indices of earlier modules), sorted.
    pub imports: Vec<u32>,
    pub uses_io: bool,
    /// Total lines of the file.
    pub budget: u32,
    pub structs: Vec<StructDef>,
    pub enums: Vec<EnumDef>,
    pub fns: Vec<FnDef>,
    pub impls: Vec<ImplDef>,
    pub traits: Vec<TraitDef>,
    pub timpls: Vec<TraitImpl>,
    pub consts: Vec<ConstDef>,
    pub aliases: Vec<AliasDef>,
    pub items: Vec<Item>,
    /// One-line filler consts after the last item (exact line count).
    pub filler: u32,
}

impl ModPlan {
    pub fn dotted(&self) -> String {
        self.path.join(".")
    }
}

pub fn struct_lines(s: &StructDef) -> u32 {
    if s.fields.is_empty() { 1 } else { 2 + s.fields.len() as u32 }
}

pub fn enum_lines(e: &EnumDef) -> u32 {
    2 + e.variants.len() as u32
}

pub fn impl_lines(i: &ImplDef) -> u32 {
    2 + i.methods.iter().map(|m| 2 + m.body).sum::<u32>()
}

pub fn trait_lines(t: &TraitDef) -> u32 {
    2 + t.methods.len() as u32
}

pub fn timpl_lines(t: &TraitImpl) -> u32 {
    2 + t.bodies.iter().map(|b| 2 + b).sum::<u32>()
}

/// Lines of the header: comment, `import std.io;`, imports.
pub fn header_lines(p: &ModPlan) -> u32 {
    1 + u32::from(p.uses_io) + p.imports.len() as u32
}

/// Lines of `src/main.stv` importing `sinks` modules.
pub fn main_lines(sinks: u32, modules: u32) -> u32 {
    if modules == 0 { 8 } else { 2 * sinks + 8 }
}
