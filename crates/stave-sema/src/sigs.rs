//! Signatures: generics and bounds, ADT definitions, fn signatures (methods,
//! trait methods, extern imports, intrinsics, exports, dispatch), impl headers, traits, type
//! aliases, consts and `static mut`s with their const values, built from the syntax and
//! the resolutions alone, before any body is checked.
//!
//! The build is sequential and demand-driven (an alias target, a const value for an array length,
//! a layout for `size_of` are computed when first needed; a cycle is E0904 for consts and the error
//! type otherwise), then every decl is visited in canonical decl order, so the result never depends
//! on threads. Types are interned in the global [`TypeTable`], frozen afterwards.
//!
//! Generic parameters are numbered flat: the container's (impl or trait) first, then the fn's own
//! (`GenericRef.level` 1), so `Param(i)` indexes [`FnSig::generics`].

use rustc_hash::FxHashMap;
use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind, Term};

use crate::Program;
use crate::consteval::{self, Val};
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::{DeclId, ItemId, Sym};
use crate::itemcx::{Brackets, ItemCx};
use crate::prim::{Lanes, Prim};
use crate::resolve::{Res, TypeHead};
use crate::types::{self, ConstArg, GArg, Interner, Names, TypeId, TypeKind, TypeTable, ids};

/// A trait with its arguments (a bound, or the trait of an impl).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitRef {
    pub trait_decl: DeclId,
    pub args: Vec<GArg>,
}

/// One generic parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenericParam {
    pub name: Sym,
    /// Its `GenericParam` node and item.
    pub item: ItemId,
    pub node: u32,
    pub name_span: Span,
    pub is_const: bool,
    /// The declared type of a const parameter.
    pub const_ty: TypeId,
    pub bounds: Vec<TraitRef>,
    /// `T: view`.
    pub view: bool,
}

/// Calling convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    Stave,
    C,
    /// `extern "stave-intrinsic"` (contracts/intrinsics.md).
    Intrinsic,
}

/// How a method takes `self`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SelfKind {
    Value,
    Ref,
    RefMut,
}

/// A `from` origin: parameter index, `copy` mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Origin {
    pub param: u8,
    pub copy: bool,
}

/// One entry of a parameter's `from` clause: the callee may store what `source` borrows (`copy`:
/// the origins of the view `source` points to) into what `target` points to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamOrigin {
    pub target: u8,
    pub source: u8,
    pub copy: bool,
}

/// A fn signature. Methods have the receiver as `params[0]`.
#[derive(Clone, Debug)]
pub struct FnSig {
    pub decl: DeclId,
    /// Flat generics (container's first).
    pub generics: Vec<GenericParam>,
    /// How many of `generics` belong to the container (impl/trait).
    pub n_parent: u32,
    pub params: Vec<TypeId>,
    /// Span of each parameter's type (or the `self` parameter).
    pub param_spans: Vec<Span>,
    pub self_kind: Option<SelfKind>,
    pub ret: TypeId,
    /// Span of the return type (`None`: unit, no `->`).
    pub ret_span: Option<Span>,
    pub origins: Vec<Origin>,
    /// The parameters' `from` clauses: the valid entries, sorted, deduplicated.
    pub param_origins: Vec<ParamOrigin>,
    /// stave-target `FeatureSet` bits.
    pub features: u64,
    /// The deny clause ([`crate::effects::Effects`] bits) and the span of each effect it names.
    pub deny: u8,
    pub deny_spans: Vec<(u8, Span)>,
    pub is_unsafe: bool,
    pub abi: Abi,
    pub variadic: bool,
    /// The fn node in its item.
    pub node: u32,
}

/// A struct field or variant payload field (tuple payloads: fields named by index, `name` NONE).
#[derive(Clone, Debug)]
pub struct FieldDef {
    /// The field decl (`NONE` for a tuple-payload element).
    pub decl: DeclId,
    pub name: Sym,
    pub ty: TypeId,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariantShape {
    Unit,
    Tuple,
    Record,
}

#[derive(Clone, Debug)]
pub struct VariantDef {
    /// The variant decl (a struct: the struct decl).
    pub decl: DeclId,
    pub name: Sym,
    pub shape: VariantShape,
    pub fields: Vec<FieldDef>,
}

/// A struct or enum (layout comes after mono).
#[derive(Clone, Debug)]
pub struct AdtDef {
    pub decl: DeclId,
    pub is_enum: bool,
    pub is_view: bool,
    pub generics: Vec<GenericParam>,
    /// A struct has exactly one variant (record shape).
    pub variants: Vec<VariantDef>,
}

/// An impl block header.
#[derive(Clone, Debug)]
pub struct ImplDef {
    pub decl: DeclId,
    pub item: ItemId,
    pub generics: Vec<GenericParam>,
    pub self_ty: TypeId,
    pub head: Option<TypeHead>,
    pub trait_ref: Option<TraitRef>,
    pub methods: Vec<DeclId>,
    pub is_unsafe: bool,
}

/// A trait.
#[derive(Clone, Debug)]
pub struct TraitDef {
    pub decl: DeclId,
    pub generics: Vec<GenericParam>,
    pub methods: Vec<DeclId>,
    pub is_unsafe: bool,
}

/// A const or `static mut`.
#[derive(Clone, Debug)]
pub struct ConstDef {
    pub ty: TypeId,
    /// `None` if the initializer is not a valid const-expr (diagnosed).
    pub value: Option<Val>,
}

/// A type alias.
#[derive(Clone, Debug)]
pub struct AliasDef {
    pub generics: Vec<GenericParam>,
    pub target: TypeId,
}

/// The signature part of a decl.
#[derive(Clone, Debug, Default)]
pub enum DeclSig {
    #[default]
    None,
    Fn(Box<FnSig>),
    Adt(Box<AdtDef>),
    Impl(Box<ImplDef>),
    Trait(Box<TraitDef>),
    Const(ConstDef),
    Static(ConstDef),
    Alias(AliasDef),
}

/// The std language items the checker needs (found by path; absent without std).
#[derive(Clone, Debug)]
pub struct LangItems {
    pub option: DeclId,
    pub result: DeclId,
    pub string: DeclId,
    pub copy: DeclId,
    pub send: DeclId,
    pub sync: DeclId,
    /// `Flat`: derived from the type's structure only; no impl counts (crate::traits).
    pub flat: DeclId,
    /// `Frozen`: derived like `Flat`; no `UnsafeCell` or fn value reachable (crate::traits).
    pub frozen: DeclId,
    pub drop: DeclId,
    pub display: DeclId,
    pub write: DeclId,
    /// `std.cell.UnsafeCell` (fixed path): never Sync, so no type containing it is Sync without
    /// an explicit `unsafe(..) impl Sync` (crate::kinds).
    pub unsafe_cell: DeclId,
    /// `std.soa.Soa<T>` (fixed path): its fields, outside `std.soa`, are the columns of T, one per
    /// field of the struct T (crate::expr `field_of`).
    pub soa: DeclId,
    /// Operator traits by name (`Add`, `WrapAdd`, ...).
    pub ops: FxHashMap<&'static str, DeclId>,
    /// The constructors a const-expr may call (fixed paths): `new` of `std.cell.UnsafeCell`,
    /// `std.atomic.{U32, U64, Usize, Bool}`, `std.sync.{Mutex, Once}`.
    pub const_ctors: Vec<DeclId>,
    /// The ranked locks of `std.sync` (fixed paths, [`crate::ranks`]).
    pub locks: LockItems,
}

/// The `std.sync` items the rank check knows (`NONE` without std).
#[derive(Clone, Copy, Debug)]
pub struct LockItems {
    pub mutex: DeclId,
    pub guard: DeclId,
    pub once: DeclId,
    /// `Mutex.lock`, `Once.get_or_init`, `lock_pair`: the acquisitions of the rank that is their
    /// one const generic param.
    pub lock: DeclId,
    pub get_or_init: DeclId,
    pub lock_pair: DeclId,
}

impl Default for LockItems {
    fn default() -> Self {
        LockItems {
            mutex: DeclId::NONE,
            guard: DeclId::NONE,
            once: DeclId::NONE,
            lock: DeclId::NONE,
            get_or_init: DeclId::NONE,
            lock_pair: DeclId::NONE,
        }
    }
}

impl Default for LangItems {
    fn default() -> Self {
        LangItems {
            option: DeclId::NONE,
            result: DeclId::NONE,
            string: DeclId::NONE,
            copy: DeclId::NONE,
            send: DeclId::NONE,
            sync: DeclId::NONE,
            flat: DeclId::NONE,
            frozen: DeclId::NONE,
            drop: DeclId::NONE,
            display: DeclId::NONE,
            write: DeclId::NONE,
            unsafe_cell: DeclId::NONE,
            soa: DeclId::NONE,
            ops: FxHashMap::default(),
            const_ctors: Vec::new(),
            locks: LockItems::default(),
        }
    }
}

pub const OP_TRAITS: [(&str, &str); 19] = [
    ("Add", "add"),
    ("WrapAdd", "wrap_add"),
    ("SatAdd", "sat_add"),
    ("Sub", "sub"),
    ("WrapSub", "wrap_sub"),
    ("SatSub", "sat_sub"),
    ("Mul", "mul"),
    ("WrapMul", "wrap_mul"),
    ("SatMul", "sat_mul"),
    ("Div", "div"),
    ("Rem", "rem"),
    ("Shl", "shl"),
    ("WrapShl", "wrap_shl"),
    ("Shr", "shr"),
    ("BitAnd", "bitand"),
    ("BitOr", "bitor"),
    ("BitXor", "bitxor"),
    ("Neg", "neg"),
    ("Not", "not"),
];

/// The frozen signature table.
#[derive(Debug, Default)]
pub struct Sigs {
    /// By decl id.
    pub decls: Vec<DeclSig>,
    pub lang: LangItems,
    /// Impl decls per type head, in item order; impls with no head (blanket, refs, tuples).
    pub impls_by_head: FxHashMap<TypeHead, Vec<DeclId>>,
    pub headless_impls: Vec<DeclId>,
    /// Trait impls per trait decl.
    pub impls_by_trait: FxHashMap<DeclId, Vec<DeclId>>,
}

impl Sigs {
    pub fn get(&self, d: DeclId) -> &DeclSig {
        static NONE: DeclSig = DeclSig::None;
        self.decls.get(d.idx()).unwrap_or(&NONE)
    }

    pub fn fn_sig(&self, d: DeclId) -> Option<&FnSig> {
        match self.get(d) {
            DeclSig::Fn(f) => Some(f),
            _ => None,
        }
    }

    pub fn adt(&self, d: DeclId) -> Option<&AdtDef> {
        match self.get(d) {
            DeclSig::Adt(a) => Some(a),
            _ => None,
        }
    }

    pub fn impl_def(&self, d: DeclId) -> Option<&ImplDef> {
        match self.get(d) {
            DeclSig::Impl(i) => Some(i),
            _ => None,
        }
    }

    pub fn trait_def(&self, d: DeclId) -> Option<&TraitDef> {
        match self.get(d) {
            DeclSig::Trait(t) => Some(t),
            _ => None,
        }
    }

    pub fn konst(&self, d: DeclId) -> Option<&ConstDef> {
        match self.get(d) {
            DeclSig::Const(c) | DeclSig::Static(c) => Some(c),
            _ => None,
        }
    }

    /// The generics of a decl (fn, ADT, impl, trait, alias).
    pub fn generics(&self, d: DeclId) -> &[GenericParam] {
        match self.get(d) {
            DeclSig::Fn(f) => &f.generics,
            DeclSig::Adt(a) => &a.generics,
            DeclSig::Impl(i) => &i.generics,
            DeclSig::Trait(t) => &t.generics,
            DeclSig::Alias(a) => &a.generics,
            _ => &[],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Todo,
    InProgress,
    Done,
}

/// Names for diagnostics: ADT names from the program, params from a generics list.
pub(crate) struct NameCx<'a> {
    pub prog: &'a Program,
    pub generics: &'a [GenericParam],
}

impl Names for NameCx<'_> {
    fn adt_name(&self, d: DeclId) -> String {
        self.prog.decl(d).map_or_else(|| "_".to_string(), |x| self.prog.strings.get(x.name).to_string())
    }

    fn param_name(&self, i: u32) -> String {
        match self.generics.get(i as usize) {
            Some(g) => self.prog.strings.get(g.name).to_string(),
            None => format!("T{i}"),
        }
    }
}

/// Where a type is lowered.
#[derive(Clone, Copy)]
pub(crate) struct TyEnv<'a> {
    pub icx: ItemCx<'a>,
    pub n_parent: u32,
    pub self_ty: Option<TypeId>,
}

/// Lowering callbacks a type lowering needs from its context (signature phase or body phase).
pub(crate) trait LowerHost: Interner {
    fn prog(&self) -> &Program;
    /// Per generic param of `d`: `None` for a type param, the declared type of a const one.
    fn sigs_generics_len(&mut self, d: DeclId) -> Option<Vec<Option<TypeId>>>;
    /// The generic params of an ADT (view checks of type arguments).
    fn adt_generics(&mut self, d: DeclId) -> Vec<GenericParam>;
    fn alias_target(&mut self, d: DeclId) -> Option<(Vec<GenericParam>, TypeId)>;
    /// Evaluates the const-expr `n` (expected type `ty`) to a `u64` (array length, const arg).
    fn const_u64(&mut self, env: &TyEnv<'_>, n: u32, ty: TypeId) -> ConstArg;
    fn report(&mut self, d: Option<Diagnostic>);
    /// The generic params in scope (a `T: view` param counts as a view in type arguments,
    /// view rules); none by default.
    fn view_env(&self) -> &[GenericParam] {
        &[]
    }
    /// The declared type of const generic param `idx` (flat index) in scope; `None` for a host
    /// that reports nothing (a const arg is then not type-checked).
    fn const_param_ty(&self, idx: u32) -> Option<TypeId>;
    /// A written fixed vector or mask type `ty` of `bits` bits at `span` (E0702, crate::gates).
    fn vector_width(&mut self, _span: Span, _ty: &str, _bits: u32) {}
    /// Signature type `t` written at `span`, of a fn or fn type with features `features` (E0703,
    /// crate::gates). Only where E0702 does not gate written vectors (inside `needs(...)`), so a
    /// too wide vector gets one of the two codes.
    fn sig_width(&mut self, _span: Span, _t: TypeId, _features: u64) {}
}

/// Lowers a `Type` or `TypeBase` node. `ret_pos`: directly after `->` (where `never` is legal).
pub(crate) fn lower_type<H: LowerHost>(h: &mut H, env: &TyEnv<'_>, n: u32, ret_pos: bool) -> TypeId {
    let icx = env.icx;
    // Iterative over `&`/`*` prefixes (unbounded chains).
    let mut wraps: Vec<(bool, bool)> = Vec::new(); // (is_ref, mutable)
    let mut n = n;
    let mut ret_pos = ret_pos;
    let base = loop {
        if wraps.len() >= crate::typedepth::MAX_TYPE_DEPTH as usize {
            too_deep(h, &icx, n);
            return h.error_ty();
        }
        let a = icx.a(n);
        match icx.kind(n) {
            Some(NodeKind::Type) => {
                if icx.is(a, Term::LBracket) {
                    break lower_array(h, env, n);
                }
                match icx.first_child(n) {
                    Some(c) => n = c,
                    None => break h.error_ty(),
                }
            }
            Some(NodeKind::TypeBase) => {
                let t = icx.tk(a);
                if t == Term::Amp as u8 || t == Term::AmpAmp as u8 {
                    let Some(c) = icx.first_child(n) else { break h.error_ty() };
                    if t == Term::AmpAmp as u8 {
                        // `&&T` = `& &T`: the outer half when the child type starts at the same token.
                        let inner_tb = icx.first_child(c).filter(|&x| icx.a(x) == a);
                        if inner_tb.is_some() {
                            wraps.push((true, false));
                            n = c;
                            ret_pos = false;
                            // the inner `&` is the TypeBase at the same token
                            if let Some(tb) = inner_tb {
                                let mutable = icx.is(a + 1, Term::KwMut);
                                wraps.push((true, mutable));
                                match icx.first_child(tb) {
                                    Some(cc) => n = cc,
                                    None => break h.error_ty(),
                                }
                            }
                            continue;
                        }
                    }
                    let mutable = icx.is(a + 1, Term::KwMut);
                    wraps.push((true, mutable));
                    n = c;
                    ret_pos = false;
                } else if t == Term::Star as u8 {
                    let mutable = icx.is(a + 1, Term::KwMut);
                    let Some(c) = icx.first_child(n) else { break h.error_ty() };
                    wraps.push((false, mutable));
                    n = c;
                    ret_pos = false;
                } else {
                    break lower_base(h, env, n, ret_pos);
                }
            }
            _ => break h.error_ty(),
        }
    };
    let mut t = base;
    for &(is_ref, mutable) in wraps.iter().rev() {
        t = if is_ref { h.mk_ref(mutable, t) } else { h.mk_ptr(mutable, t) };
    }
    t
}

fn lower_array<H: LowerHost>(h: &mut H, env: &TyEnv<'_>, n: u32) -> TypeId {
    let icx = env.icx;
    let a = icx.a(n);
    let close = icx.close(a);
    let kids = icx.kids_vec(n);
    let Some(&elem_n) = kids.iter().find(|&&c| icx.kind(c) == Some(NodeKind::Type)) else {
        return h.error_ty();
    };
    let elem = lower_type(h, env, elem_n, false);
    if let Some(&e) = kids.iter().find(|&&c| icx.is_expr(c)) {
        let len = h.const_u64(env, e, h.fixed_ty(ids::USIZE));
        if h.is_error_ty(elem) {
            return h.error_ty();
        }
        if len == ConstArg::Error {
            return h.error_ty();
        }
        return h.intern(TypeKind::Array { elem, len });
    }
    let mut align = 0u32;
    if close >= 2 && icx.is(close - 2, Term::KwAlign) && icx.is(close - 1, Term::IntLit) {
        align =
            crate::literals::parse_int_text(icx.tok_str(close - 1)).map_or(0, |(v, _)| v.min(1 << 20) as u32);
    }
    if h.is_error_ty(elem) {
        return h.error_ty();
    }
    h.intern(TypeKind::Slice { elem, align })
}

fn lower_base<H: LowerHost>(h: &mut H, env: &TyEnv<'_>, n: u32, ret_pos: bool) -> TypeId {
    let icx = env.icx;
    let a = icx.a(n);
    let t = icx.tk(a);
    if t == Term::LParen as u8 {
        let close = icx.close(a);
        let kids = icx.kids_vec(n);
        let tys: Vec<TypeId> = kids
            .iter()
            .filter(|&&c| icx.kind(c) == Some(NodeKind::Type))
            .map(|&c| lower_type(h, env, c, false))
            .collect();
        if tys.len() == 1 && !icx.is(close - 1, Term::Comma) {
            return tys[0];
        }
        return h.mk_tuple(tys);
    }
    if t == Term::KwFn as u8 {
        let close = icx.close(a + 1);
        let has_ret = icx.is(close + 1, Term::MinusGt);
        let kids = icx.kids_vec(n);
        let mut tys: Vec<u32> =
            kids.iter().copied().filter(|&c| icx.kind(c) == Some(NodeKind::Type)).collect();
        let ret_node = if has_ret { tys.pop() } else { None };
        let ret = match ret_node {
            Some(r) => lower_type(h, env, r, true),
            None if has_ret => h.error_ty(),
            None => h.fixed_ty(ids::UNIT),
        };
        let params: Vec<TypeId> = tys.iter().map(|&c| lower_type(h, env, c, false)).collect();
        let features = match icx.child(n, NodeKind::Needs) {
            Some(nd) => needs_bits(h, &icx, nd),
            None => 0,
        };
        let deny = match icx.child(n, NodeKind::Without) {
            Some(nd) => crate::effects::Effects::fn_type_deny(deny_bits(h, &icx, nd).0),
            None => 0,
        };
        if params.contains(&h.error_ty()) || h.is_error_ty(ret) {
            return h.error_ty();
        }
        for (&t, &c) in params.iter().zip(&tys).chain(ret_node.iter().map(|c| (&ret, c))) {
            h.sig_width(icx.node_span(c), t, features);
        }
        return h.intern(TypeKind::Fn { params: params.into_boxed_slice(), ret, features, deny });
    }
    if t == Term::KwSelfType as u8 {
        return env.self_ty.unwrap_or(h.error_ty());
    }
    if t != Term::Ident as u8 {
        return h.error_ty();
    }
    // Ident chain: the last resolved segment names the type.
    let uses = icx.uses(n);
    let Some(last) = uses.last() else { return h.error_ty() };
    let gargs_node = icx.child(n, NodeKind::GenericArgs);
    match last.res {
        Res::Unresolved => h.error_ty(),
        Res::Prim(p) => lower_prim(h, env, n, p, gargs_node, ret_pos),
        Res::Generic(g) => {
            let idx = if g.level == 0 { g.index as u32 } else { env.n_parent + g.index as u32 };
            if g.is_const {
                not_a_type(h, &icx, n);
                return h.error_ty();
            }
            h.mk_param(idx)
        }
        Res::SelfType => env.self_ty.unwrap_or(h.error_ty()),
        Res::Decl(d) => {
            let Some(decl) = h.prog().decl(d).copied() else { return h.error_ty() };
            match decl.kind {
                DeclKind::Error => h.error_ty(),
                DeclKind::Struct | DeclKind::Enum => {
                    let Some(shape) = h.sigs_generics_len(d) else { return h.error_ty() };
                    let args = lower_gargs(h, env, gargs_node, &shape, n);
                    match args {
                        Some(args) => {
                            view_args(h, &icx, n, d, &args);
                            soa_row(h, &icx, n, d, &args);
                            h.mk_adt(d, args)
                        }
                        None => h.error_ty(),
                    }
                }
                DeclKind::TypeAlias => {
                    let Some((gens, target)) = h.alias_target(d) else { return h.error_ty() };
                    let shape: Vec<Option<TypeId>> =
                        gens.iter().map(|g| g.is_const.then_some(g.const_ty)).collect();
                    let Some(args) = lower_gargs(h, env, gargs_node, &shape, n) else { return h.error_ty() };
                    // Checked before substituting, which stops at the limit.
                    if crate::types::subst_height(&*h, target, &args) > crate::typedepth::MAX_TYPE_DEPTH {
                        too_deep(h, &icx, n);
                        return h.error_ty();
                    }
                    let t = h.subst(target, &args, None);
                    if !ret_pos && t == h.fixed_ty(ids::NEVER) {
                        e0209(h, &icx, n);
                        return h.error_ty();
                    }
                    t
                }
                _ => {
                    not_a_type(h, &icx, n);
                    h.error_ty()
                }
            }
        }
        Res::Local(_) | Res::SelfValue(_) | Res::Builtin(_) => {
            not_a_type(h, &icx, n);
            h.error_ty()
        }
    }
}

/// E0005 case `type`: the type written at node `n` nests deeper than the type depth limit.
fn too_deep<H: LowerHost>(h: &mut H, icx: &ItemCx<'_>, n: u32) {
    h.report(mk("E0005", Some("type"), &[], icx.node_span(n), &[]));
}

/// E0414 (case `container` for the std package's ADTs): a view type argument for a generic param
/// without the `view` bound ([`crate::kinds::view_arg`]).
fn view_args<H: LowerHost>(h: &mut H, icx: &ItemCx<'_>, n: u32, d: DeclId, args: &[GArg]) {
    let gens = h.adt_generics(d);
    let mut out = Vec::new();
    {
        let prog = h.prog();
        let env = h.view_env();
        let names = NameCx { prog, generics: env };
        for (g, a) in gens.iter().zip(args) {
            let GArg::Type(t) = a else { continue };
            if g.view || h.has_error(*t) || !crate::kinds::is_view_in(prog, h, *t, env) {
                continue;
            }
            let tn = types::display(h, &names, *t);
            out.extend(crate::kinds::view_arg(prog, d, g, &tn, icx.node_span(n)));
        }
    }
    for d in out {
        h.report(Some(d));
    }
}

/// Whether `m` is module `soa` of the built-in `std` package.
pub(crate) fn is_builtin_soa(p: &Program, m: &crate::modules::Module) -> bool {
    m.path_text == "std.soa"
        && p.packages.get(m.package.idx()).is_some_and(|k| k.kind == crate::PackageKind::Builtin)
}

/// E0910: a written `std.soa.Soa<T>` whose row type T is neither a struct nor a generic param.
fn soa_row<H: LowerHost>(h: &mut H, icx: &ItemCx<'_>, n: u32, d: DeclId, args: &[GArg]) {
    let Some(GArg::Type(t)) = args.first().copied() else { return };
    let diag = {
        let prog = h.prog();
        let is_soa = prog.decl(d).is_some_and(|x| {
            prog.strings.get(x.name) == "Soa"
                && prog.module(x.module).is_some_and(|m| is_builtin_soa(prog, m))
        });
        if !is_soa || h.has_error(t) {
            return;
        }
        let ok = match h.kind(t) {
            TypeKind::Adt { def, .. } => prog.decl(*def).is_some_and(|x| x.kind == DeclKind::Struct),
            TypeKind::Param(_) => true,
            _ => false,
        };
        if ok {
            return;
        }
        let names = NameCx { prog, generics: h.view_env() };
        let tn = types::display(h, &names, t);
        mk("E0910", None, &[("type", &tn)], icx.node_span(n), &[])
    };
    h.report(diag);
}

fn e0209<H: LowerHost>(h: &mut H, icx: &ItemCx<'_>, n: u32) {
    let d = mk("E0209", None, &[], icx.node_span(n), &[]);
    h.report(d);
}

fn not_a_type<H: LowerHost>(h: &mut H, icx: &ItemCx<'_>, n: u32) {
    let text = icx.node_text(n).to_string();
    let d = mk(
        "E0101",
        Some("not-type"),
        &[("name", if text.is_empty() { "_" } else { &text })],
        icx.node_span(n),
        &[],
    );
    h.report(d);
}

/// Feature bits of a `Needs` node (unknown names: E0708).
pub(crate) fn needs_bits<H: LowerHost + ?Sized>(h: &mut H, icx: &ItemCx<'_>, nd: u32) -> u64 {
    let mut fs = stave_target::FeatureSet::default();
    for f in icx.kids_vec(nd) {
        let t = icx.a(f);
        let name = icx.tok_str(t);
        match stave_target::FeatureSet::of_name(name) {
            Some(named) => fs = fs.union(named),
            None => {
                let d = mk("E0708", None, &[("name", name)], icx.tok_span(t), &[]);
                h.report(d);
            }
        }
    }
    fs.bits()
}

/// The effects a `without(..)` clause names, with the span of each; E1404 for an unknown name.
pub(crate) fn deny_bits<H: LowerHost + ?Sized>(
    h: &mut H,
    icx: &ItemCx<'_>,
    nd: u32,
) -> (u8, Vec<(u8, Span)>) {
    let mut bits = 0u8;
    let mut spans = Vec::new();
    for e in icx.kids_vec(nd) {
        let t = icx.a(e);
        let name = icx.tok_str(t);
        match crate::effects::Effects::deny_of_name(name) {
            Some(b) => {
                bits |= b.0;
                spans.push((b.0, icx.tok_span(t)));
            }
            None => {
                let d = mk("E1404", None, &[("name", name)], icx.tok_span(t), &[]);
                h.report(d);
            }
        }
    }
    (bits, spans)
}

/// Lowers generic args for a decl whose params are `shape` (true = const param). `None` on a
/// count mismatch (diagnosed).
pub(crate) fn lower_gargs<H: LowerHost>(
    h: &mut H,
    env: &TyEnv<'_>,
    gargs: Option<u32>,
    shape: &[Option<TypeId>],
    at: u32,
) -> Option<Vec<GArg>> {
    let icx = env.icx;
    let kids = match gargs {
        Some(g) => icx.kids_vec(g),
        None => crate::itemcx::Kids::default(),
    };
    if kids.len() != shape.len() {
        let text = icx.node_text(at).to_string();
        let d = mk(
            "E0201",
            Some("type-args"),
            &[("operand", &text), ("count", &shape.len().to_string()), ("given", &kids.len().to_string())],
            icx.node_span(at),
            &[],
        );
        h.report(d);
        return None;
    }
    let mut out = Vec::with_capacity(kids.len());
    for (k, &ga) in kids.iter().enumerate() {
        out.push(lower_garg(h, env, ga, shape[k]));
    }
    Some(out)
}

/// Const generic param `idx` written at node `n` as the argument of a const param of type
/// `want`: E0201 when its own declared type differs (a value of one type never stands for
/// another: `300` of a `usize` param is no `u8`).
pub(crate) fn const_param_arg<H: LowerHost + ?Sized>(
    h: &mut H,
    icx: &ItemCx<'_>,
    n: u32,
    idx: u32,
    want: TypeId,
) -> ConstArg {
    let Some(t) = h.const_param_ty(idx) else { return ConstArg::Param(idx) };
    if t == want || h.is_error_ty(t) || h.is_error_ty(want) {
        return ConstArg::Param(idx);
    }
    let names = NameCx { prog: h.prog(), generics: &[] };
    let (e, f) = (types::display(&*h, &names, want), types::display(&*h, &names, t));
    let text = icx.node_text(n).to_string();
    let d = mk("E0201", None, &[("expected", &e), ("found", &f), ("operand", &text)], icx.node_span(n), &[]);
    h.report(d);
    ConstArg::Error
}

/// One `GenericArg`: a type (`want_const` `None`) or a const of the param's declared type
/// (`usize` when it is unknown).
pub(crate) fn lower_garg<H: LowerHost>(
    h: &mut H,
    env: &TyEnv<'_>,
    ga: u32,
    want_const: Option<TypeId>,
) -> GArg {
    let icx = env.icx;
    let a = icx.a(ga);
    let child = icx.first_child(ga);
    if let Some(declared) = want_const {
        let cty =
            if declared.is_none() || h.is_error_ty(declared) { h.fixed_ty(ids::USIZE) } else { declared };
        if icx.is(a, Term::IntLit) {
            let Some((v, _)) = crate::literals::parse_int_text(icx.tok_str(a)) else {
                return GArg::Const(ConstArg::Error);
            };
            let (lo, hi) = crate::literals::int_range_kind(h.kind(cty)).unwrap_or((0, u64::MAX as i128));
            if v > hi as u128 {
                let tn = types::display(&*h, &NameCx { prog: h.prog(), generics: &[] }, cty);
                let (lo, hi) = (lo.to_string(), hi.to_string());
                let holes = [("literal", icx.tok_str(a)), ("type", tn.as_str()), ("min", &lo), ("max", &hi)];
                h.report(mk("E0203", None, &holes, icx.tok_span(a), &[]));
                return GArg::Const(ConstArg::Error);
            }
            return GArg::Const(ConstArg::Value(v as u64));
        }
        if let Some(c) = child {
            match icx.kind(c) {
                Some(k) if crate::typeck::is_expr_kind(k) => {
                    return GArg::Const(h.const_u64(env, c, cty));
                }
                Some(NodeKind::TypeBase) => {
                    // A const generic param or a const named like a type path.
                    if let Some(u) = icx.uses(c).last() {
                        match u.res {
                            Res::Generic(g) if g.is_const => {
                                let idx =
                                    if g.level == 0 { g.index as u32 } else { env.n_parent + g.index as u32 };
                                if declared.is_none() {
                                    return GArg::Const(ConstArg::Param(idx));
                                }
                                return GArg::Const(const_param_arg(h, &icx, c, idx, declared));
                            }
                            Res::Decl(d) if h.prog().decl(d).is_some_and(|x| x.kind == DeclKind::Const) => {
                                return GArg::Const(h.const_u64(env, c, cty));
                            }
                            Res::Unresolved => return GArg::Const(ConstArg::Error),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        let text = icx.node_text(ga).to_string();
        let d = mk("E0201", Some("const-arg"), &[("operand", &text)], icx.node_span(ga), &[]);
        h.report(d);
        return GArg::Const(ConstArg::Error);
    }
    match child.and_then(|c| icx.kind(c).map(|k| (c, k))) {
        Some((c, NodeKind::TypeBase)) => GArg::Type(lower_type(h, env, c, false)),
        Some((c, NodeKind::ArrOrIdx)) => {
            // `[T; N]` / `[T]` as a type argument.
            let kids = icx.kids_vec(c);
            let Some(&tn) = kids.iter().find(|&&x| icx.kind(x) == Some(NodeKind::Type)) else {
                return GArg::Type(h.error_ty());
            };
            let elem = lower_type(h, env, tn, false);
            if let Some(&e) = kids.iter().find(|&&x| icx.is_expr(x)) {
                let len = h.const_u64(env, e, h.fixed_ty(ids::USIZE));
                if h.is_error_ty(elem) || len == ConstArg::Error {
                    return GArg::Type(h.error_ty());
                }
                return GArg::Type(h.intern(TypeKind::Array { elem, len }));
            }
            let close = icx.close(a);
            let mut align = 0u32;
            if close >= 2 && icx.is(close - 2, Term::KwAlign) {
                align = crate::literals::parse_int_text(icx.tok_str(close - 1))
                    .map_or(0, |(v, _)| v.min(1 << 20) as u32);
            }
            if h.is_error_ty(elem) {
                return GArg::Type(h.error_ty());
            }
            GArg::Type(h.intern(TypeKind::Slice { elem, align }))
        }
        _ => {
            let text = icx.node_text(ga).to_string();
            let d = mk(
                "E0101",
                Some("not-type"),
                &[("name", if text.is_empty() { "_" } else { &text })],
                icx.node_span(ga),
                &[],
            );
            h.report(d);
            GArg::Type(h.error_ty())
        }
    }
}

fn lower_prim<H: LowerHost>(
    h: &mut H,
    env: &TyEnv<'_>,
    n: u32,
    p: Prim,
    gargs: Option<u32>,
    ret_pos: bool,
) -> TypeId {
    let icx = env.icx;
    match p {
        Prim::Scalar(s) => {
            if gargs.is_some() {
                let _ = lower_gargs(h, env, gargs, &[], n);
                return h.error_ty();
            }
            h.scalar_ty(s)
        }
        Prim::Never => {
            if !ret_pos {
                e0209(h, &icx, n);
                return h.error_ty();
            }
            h.fixed_ty(ids::NEVER)
        }
        Prim::Vector { lane, n: lanes } => {
            let l = h.scalar_ty(lane);
            if let Lanes::Fixed(k) = lanes {
                let bits = types::lane_bits(h.kind(l)) * k as u32;
                h.vector_width(icx.node_span(n), icx.tok_str(icx.a(n)), bits);
            }
            h.intern(TypeKind::Vector { lane: l, n: lanes })
        }
        Prim::Mask { width, n: lanes } => {
            if let Lanes::Fixed(k) = lanes {
                h.vector_width(icx.node_span(n), icx.tok_str(icx.a(n)), width as u32 * k as u32);
            }
            h.intern(TypeKind::Mask { width, n: lanes })
        }
        Prim::VecGeneric | Prim::MaskGeneric => {
            // `vec<T, N>` / `mask<W, N>` (E0609 when no argument is generic).
            let kids = gargs.map(|g| icx.kids_vec(g)).unwrap_or_default();
            if kids.len() != 2 {
                let us = h.fixed_ty(ids::USIZE);
                let _ = lower_gargs(h, env, gargs, &[None, Some(us)], n);
                return h.error_ty();
            }
            let us = Some(h.fixed_ty(ids::USIZE));
            let first = if p == Prim::VecGeneric {
                lower_garg(h, env, kids[0], None)
            } else {
                lower_garg(h, env, kids[0], us)
            };
            let second = lower_garg(h, env, kids[1], us);
            // A lane count and a mask width are fixed at the type: no const param stands for one
            // (an instance would have no layout).
            let not_fixed = |h: &mut H, at: u32| {
                let spelled = icx.node_text(at).to_string();
                let d = mk("E0609", Some("lanes"), &[("spelled", &spelled)], icx.node_span(at), &[]);
                h.report(d);
                h.error_ty()
            };
            let lanes = match second {
                GArg::Const(ConstArg::Value(v)) if matches!(v, 2 | 4 | 8 | 16 | 32 | 64) => {
                    Lanes::Fixed(v as u8)
                }
                GArg::Const(ConstArg::Error) => return h.error_ty(),
                _ => return not_fixed(h, kids[1]),
            };
            let t = match (p, first) {
                (Prim::VecGeneric, GArg::Type(lane)) => {
                    if h.is_error_ty(lane) {
                        return h.error_ty();
                    }
                    h.intern(TypeKind::Vector { lane, n: lanes })
                }
                (Prim::MaskGeneric, GArg::Const(ConstArg::Value(w))) if matches!(w, 8 | 16 | 32 | 64) => {
                    h.intern(TypeKind::Mask { width: w as u8, n: lanes })
                }
                (Prim::MaskGeneric, GArg::Const(ConstArg::Error)) => return h.error_ty(),
                (Prim::MaskGeneric, _) => return not_fixed(h, kids[0]),
                _ => return h.error_ty(),
            };
            // All arguments concrete: the concrete spelling is required.
            let concrete = types::display(h, &NameCx { prog: h.prog(), generics: &[] }, t);
            let spelled = icx.node_text(n).to_string();
            let d =
                mk("E0609", None, &[("spelled", &spelled), ("concrete", &concrete)], icx.node_span(n), &[]);
            h.report(d);
            t
        }
    }
}

/// The signature builder (see the module docs).
pub(crate) struct SigBuilder<'p> {
    pub prog: &'p Program,
    pub brackets: &'p [Brackets],
    pub tys: TypeTable,
    pub sigs: Vec<DeclSig>,
    state: Vec<State>,
    gen_cache: FxHashMap<DeclId, Vec<GenericParam>>,
    pub diags: Vec<Diagnostic>,
    pub lang: LangItems,
    /// Const evaluation stack (cycle reporting).
    const_stack: Vec<DeclId>,
    /// The generics of the decl being built ([`LowerHost::view_env`]).
    cur_generics: Vec<GenericParam>,
    /// While the bounds of a generics list are lowered: the params in scope there (their const
    /// types for [`LowerHost::const_param_ty`]; the view env stays empty).
    bound_scope: Option<Vec<GenericParam>>,
    layout_busy: Vec<TypeId>,
    /// Lowering the signature of a fn with `needs(...)`: no E0702 (crate::gates).
    ungated: bool,
    /// E0702 spans already reported.
    wide: Vec<Span>,
}

impl Interner for SigBuilder<'_> {
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

impl<'p> LowerHost for SigBuilder<'p> {
    fn prog(&self) -> &Program {
        self.prog
    }

    fn sigs_generics_len(&mut self, d: DeclId) -> Option<Vec<Option<TypeId>>> {
        Some(self.generics_of(d).iter().map(|g| g.is_const.then_some(g.const_ty)).collect())
    }

    fn adt_generics(&mut self, d: DeclId) -> Vec<GenericParam> {
        self.generics_of(d)
    }

    fn alias_target(&mut self, d: DeclId) -> Option<(Vec<GenericParam>, TypeId)> {
        self.ensure(d);
        match &self.sigs[d.idx()] {
            DeclSig::Alias(a) => Some((a.generics.clone(), a.target)),
            _ => None,
        }
    }

    fn const_u64(&mut self, env: &TyEnv<'_>, n: u32, ty: TypeId) -> ConstArg {
        consteval::eval_u64(self, env, n, ty)
    }

    fn report(&mut self, d: Option<Diagnostic>) {
        self.diags.extend(d);
    }

    fn view_env(&self) -> &[GenericParam] {
        &self.cur_generics
    }

    fn const_param_ty(&self, idx: u32) -> Option<TypeId> {
        let scope = self.bound_scope.as_deref().unwrap_or(&self.cur_generics);
        scope.get(idx as usize).filter(|g| g.is_const).map(|g| g.const_ty)
    }

    fn vector_width(&mut self, span: Span, ty: &str, bits: u32) {
        if self.ungated || self.wide.contains(&span) {
            return;
        }
        let d = crate::gates::too_wide(ty, bits, crate::gates::max_vector_bits(self.prog.target), span);
        if d.is_some() {
            self.wide.push(span);
        }
        self.diags.extend(d);
    }

    fn sig_width(&mut self, span: Span, t: TypeId, features: u64) {
        if !self.ungated || self.wide.contains(&span) {
            return;
        }
        let Some((v, bits)) = crate::gates::sig_wide_part(self, self.prog.target, t, features) else {
            return;
        };
        let name = types::display(self, &NameCx { prog: self.prog, generics: &[] }, v);
        self.wide.push(span);
        let d = crate::gates::sig_too_wide(self.prog.target, features, &name, bits, span);
        self.diags.extend(d);
    }
}

impl<'p> SigBuilder<'p> {
    pub fn new(prog: &'p Program, brackets: &'p [Brackets]) -> SigBuilder<'p> {
        let n = prog.decls.len();
        let mut b = SigBuilder {
            prog,
            brackets,
            tys: TypeTable::new(),
            sigs: vec![DeclSig::None; n],
            state: vec![State::Todo; n],
            gen_cache: FxHashMap::default(),
            diags: Vec::new(),
            lang: LangItems::default(),
            const_stack: Vec::new(),
            layout_busy: Vec::new(),
            cur_generics: Vec::new(),
            bound_scope: None,
            ungated: false,
            wide: Vec::new(),
        };
        b.find_lang_items();
        b
    }

    fn find_lang_items(&mut self) {
        let p = self.prog;
        let find = |module: &[&str], name: &str| -> DeclId {
            // compiler-known items live in the built-in packages only
            let Some(m) = p.modules.iter().find(|m| {
                m.path.len() == module.len()
                    && m.path.iter().zip(module).all(|(s, t)| p.strings.get(*s) == *t)
                    && p.packages.get(m.package.idx()).is_some_and(|k| k.kind == crate::PackageKind::Builtin)
            }) else {
                return DeclId::NONE;
            };
            match p.strings.lookup(name) {
                Some(s) => p.module_item(m.id, s).unwrap_or(DeclId::NONE),
                None => DeclId::NONE,
            }
        };
        let core = ["std", "core"];
        self.lang.option = find(&core, "Option");
        self.lang.result = find(&core, "Result");
        self.lang.copy = find(&core, "Copy");
        self.lang.send = find(&core, "Send");
        self.lang.sync = find(&core, "Sync");
        self.lang.flat = find(&core, "Flat");
        self.lang.frozen = find(&core, "Frozen");
        self.lang.drop = find(&core, "Drop");
        self.lang.display = find(&core, "Display");
        self.lang.write = find(&core, "Write");
        self.lang.string = find(&["std", "string"], "String");
        self.lang.unsafe_cell = find(&["std", "cell"], "UnsafeCell");
        // a user module may also be spelled `std.soa`: only the built-in package's counts
        self.lang.soa = match (p.modules.iter().find(|m| is_builtin_soa(p, m)), p.strings.lookup("Soa")) {
            (Some(m), Some(s)) => p.module_item(m.id, s).unwrap_or(DeclId::NONE),
            _ => DeclId::NONE,
        };
        for (t, _) in OP_TRAITS {
            let d = find(&core, t);
            if d.is_some() {
                self.lang.ops.insert(t, d);
            }
        }
        let ctors: [(&[&str], &str); 7] = [
            (&["std", "cell"], "UnsafeCell"),
            (&["std", "atomic"], "U32"),
            (&["std", "atomic"], "U64"),
            (&["std", "atomic"], "Usize"),
            (&["std", "atomic"], "Bool"),
            (&["std", "sync"], "Mutex"),
            (&["std", "sync"], "Once"),
        ];
        if let Some(new) = p.strings.lookup("new") {
            for (module, name) in ctors {
                let adt = find(module, name);
                let m = p.methods.inherent.get(&(TypeHead::Adt(adt), new)).and_then(|v| v.first());
                if let (true, Some(&m)) = (adt.is_some(), m) {
                    self.lang.const_ctors.push(m);
                }
            }
        }
        let sync = ["std", "sync"];
        let method = |adt: DeclId, name: &str| -> DeclId {
            let m = p.strings.lookup(name).and_then(|s| p.methods.inherent.get(&(TypeHead::Adt(adt), s)));
            match m.and_then(|v| v.first()) {
                Some(&m) if adt.is_some() => m,
                _ => DeclId::NONE,
            }
        };
        let l = &mut self.lang.locks;
        l.mutex = find(&sync, "Mutex");
        l.guard = find(&sync, "MutexGuard");
        l.once = find(&sync, "Once");
        l.lock = method(l.mutex, "lock");
        l.get_or_init = method(l.once, "get_or_init");
        l.lock_pair = find(&sync, "lock_pair");
    }

    pub fn icx(&self, item: ItemId) -> Option<ItemCx<'p>> {
        let br = self.brackets.get(item.idx())?;
        ItemCx::new(self.prog, item, br)
    }

    /// Builds every decl's signature (canonical decl order).
    pub fn build_all(&mut self) {
        for d in 0..self.prog.decls.len() as u32 {
            self.ensure(DeclId(d));
        }
    }

    pub fn finish(self) -> (TypeTable, Sigs, Vec<Diagnostic>) {
        let mut s = Sigs { decls: self.sigs, lang: self.lang, ..Default::default() };
        for (k, sig) in s.decls.iter().enumerate() {
            if let DeclSig::Impl(i) = sig {
                let d = DeclId(k as u32);
                match i.head {
                    Some(h) => s.impls_by_head.entry(h).or_default().push(d),
                    None => s.headless_impls.push(d),
                }
                if let Some(t) = &i.trait_ref {
                    s.impls_by_trait.entry(t.trait_decl).or_default().push(d);
                }
            }
        }
        // Item order within a head (impl decls of one item order are already canonical).
        let prog = self.prog;
        let item_of = |d: &DeclId| prog.decl(*d).map_or(ItemId::NONE, |x| x.item);
        for v in s.impls_by_head.values_mut() {
            v.sort_by_key(item_of);
        }
        s.headless_impls.sort_by_key(item_of);
        for v in s.impls_by_trait.values_mut() {
            v.sort_by_key(item_of);
        }
        (self.tys, s, self.diags)
    }

    /// The flat generics of decl `d` (container's first for methods).
    pub fn generics_of(&mut self, d: DeclId) -> Vec<GenericParam> {
        if let Some(g) = self.gen_cache.get(&d) {
            return g.clone();
        }
        let Some(decl) = self.prog.decl(d).copied() else { return Vec::new() };
        let mut out = Vec::new();
        let mut n_parent = 0u32;
        let mut self_ty = None;
        if matches!(decl.kind, DeclKind::Method | DeclKind::TraitMethod) && decl.parent.is_some() {
            out = self.generics_of(decl.parent);
            n_parent = out.len() as u32;
            // `Self` in a method's bounds: the impl's self type, or the trait's `Self`
            self_ty = self.self_type_of(decl.parent);
        }
        // Insert a placeholder to stop recursion through bounds that mention the decl.
        self.gen_cache.insert(d, out.clone());
        if let Some(icx) = self.icx(decl.item)
            && decl.node != u32::MAX
            && let Some(g) = icx.child(decl.node, NodeKind::Generics)
        {
            // bounds are lowered with no view env (their params are not this scope's)
            let saved = std::mem::take(&mut self.cur_generics);
            let own = self.lower_generics(&icx, g, n_parent, self_ty, &out);
            self.cur_generics = saved;
            out.extend(own);
        }
        self.gen_cache.insert(d, out.clone());
        out
    }

    fn lower_generics(
        &mut self,
        icx: &ItemCx<'p>,
        g: u32,
        n_parent: u32,
        self_ty: Option<TypeId>,
        parent: &[GenericParam],
    ) -> Vec<GenericParam> {
        let mut out: Vec<GenericParam> = Vec::new();
        let kids = icx.kids_vec(g);
        // Names first (bounds may mention later params).
        for &c in &kids {
            if icx.kind(c) != Some(NodeKind::GenericParam) {
                continue;
            }
            let a = icx.a(c);
            let is_const = icx.is(a, Term::KwConst);
            let nt = if is_const { a + 1 } else { a };
            out.push(GenericParam {
                name: icx.sym(nt),
                item: icx.item,
                node: c,
                name_span: icx.tok_span(nt),
                is_const,
                const_ty: ids::ERROR,
                bounds: Vec::new(),
                view: false,
            });
        }
        let env = TyEnv { icx: *icx, n_parent, self_ty };
        let params: Vec<u32> =
            kids.iter().copied().filter(|&c| icx.kind(c) == Some(NodeKind::GenericParam)).collect();
        for (k, &c) in params.iter().enumerate() {
            if out[k].is_const
                && let Some(t) = icx.child(c, NodeKind::Type)
            {
                let ct = lower_type(self, &env, t, false);
                // a const arg is a `u64` value, read back in the param's type
                let unsigned = matches!(self.tys.kind(ct), TypeKind::Int { signed: false, .. });
                out[k].const_ty = if unsigned || self.tys.is_error_ty(ct) {
                    ct
                } else {
                    let names = NameCx { prog: self.prog, generics: &[] };
                    let found = types::display(&self.tys, &names, ct);
                    let name = self.prog.strings.get(out[k].name).to_string();
                    let d = mk(
                        "E0201",
                        Some("const-param-type"),
                        &[("operand", &name), ("found", &found)],
                        icx.node_span(t),
                        &[],
                    );
                    self.diags.extend(d);
                    ids::ERROR
                };
            }
        }
        // a bound may pass any param of the list (or of the parent) as a const arg
        let scope: Vec<GenericParam> = parent.iter().chain(&out).cloned().collect();
        let saved = self.bound_scope.replace(scope);
        for (k, &c) in params.iter().enumerate() {
            if !out[k].is_const
                && let Some(b) = icx.child(c, NodeKind::Bound)
            {
                let (bounds, view) = self.lower_bound(&env, b);
                out[k].bounds = bounds;
                out[k].view = view;
            }
        }
        self.bound_scope = saved;
        out
    }

    /// A `Bound`: (trait refs, `view`).
    pub fn lower_bound(&mut self, env: &TyEnv<'p>, b: u32) -> (Vec<TraitRef>, bool) {
        let icx = env.icx;
        let mut out = Vec::new();
        let mut view = false;
        for bi in icx.kids_vec(b) {
            let a = icx.a(bi);
            if icx.is(a, Term::KwView) {
                view = true;
                continue;
            }
            if let Some(t) = self.trait_ref_at(env, bi) {
                out.push(t);
            }
        }
        (out, view)
    }

    /// The trait named by a `BoundItem` or `TypeBase` path node.
    fn trait_ref_at(&mut self, env: &TyEnv<'p>, n: u32) -> Option<TraitRef> {
        let icx = env.icx;
        let last = icx.uses(n).last()?;
        let Res::Decl(d) = last.res else { return None };
        let decl = self.prog.decl(d)?;
        if decl.kind != DeclKind::Trait {
            if decl.kind != DeclKind::Error {
                let text = icx.node_text(n).to_string();
                let dg = mk("E0101", Some("not-trait"), &[("name", &text)], icx.node_span(n), &[]);
                self.diags.extend(dg);
            }
            return None;
        }
        let shape: Vec<Option<TypeId>> =
            self.generics_of(d).iter().map(|g| g.is_const.then_some(g.const_ty)).collect();
        let gargs = icx.child(n, NodeKind::GenericArgs);
        let args = lower_gargs(self, env, gargs, &shape, n)?;
        Some(TraitRef { trait_decl: d, args })
    }

    /// Builds the signature of `d` if not done (cycle: the error state stays).
    pub fn ensure(&mut self, d: DeclId) {
        let Some(st) = self.state.get(d.idx()).copied() else { return };
        if st != State::Todo {
            if st == State::InProgress
                && let Some(decl) = self.prog.decl(d)
                && matches!(decl.kind, DeclKind::Const | DeclKind::Static)
            {
                self.report_const_cycle(d);
            }
            return;
        }
        self.state[d.idx()] = State::InProgress;
        let Some(decl) = self.prog.decl(d).copied() else { return };
        let env = match decl.kind {
            DeclKind::Fn
            | DeclKind::Method
            | DeclKind::TraitMethod
            | DeclKind::Struct
            | DeclKind::Enum
            | DeclKind::Impl
            | DeclKind::TypeAlias => self.generics_of(d),
            _ => Vec::new(),
        };
        let saved = std::mem::replace(&mut self.cur_generics, env);
        let saved_scope = self.bound_scope.take();
        let sig = match decl.kind {
            DeclKind::Fn | DeclKind::Method | DeclKind::TraitMethod => self.build_fn(d),
            DeclKind::Struct | DeclKind::Enum => self.build_adt(d),
            DeclKind::Impl => self.build_impl(d),
            DeclKind::Trait => self.build_trait(d),
            DeclKind::Const | DeclKind::Static => self.build_const(d),
            DeclKind::TypeAlias => self.build_alias(d),
            _ => DeclSig::None,
        };
        self.cur_generics = saved;
        self.bound_scope = saved_scope;
        self.sigs[d.idx()] = sig;
        self.state[d.idx()] = State::Done;
    }

    fn report_const_cycle(&mut self, d: DeclId) {
        let Some(pos) = self.const_stack.iter().position(|&x| x == d) else { return };
        let cycle: Vec<DeclId> = self.const_stack[pos..].to_vec();
        // Start at the smallest decl id, follow the dependency order.
        let start = cycle.iter().enumerate().min_by_key(|(_, x)| **x).map_or(0, |(i, _)| i);
        let rot: Vec<DeclId> = cycle[start..].iter().chain(cycle[..start].iter()).copied().collect();
        let Some(first) = rot.first().and_then(|&x| self.prog.decl(x)) else { return };
        let name = self.prog.strings.get(first.name).to_string();
        let roles: Vec<(&str, Span)> =
            rot[1..].iter().filter_map(|&x| self.prog.decl(x)).map(|x| ("cycle", x.name_span)).collect();
        let dg = mk("E0904", Some("cycle"), &[("name", &name)], first.name_span, &roles);
        self.diags.extend(dg);
    }

    fn self_type_of(&mut self, d: DeclId) -> Option<TypeId> {
        let decl = self.prog.decl(d)?;
        match decl.kind {
            DeclKind::Impl => {
                self.ensure(d);
                match &self.sigs[d.idx()] {
                    DeclSig::Impl(i) => Some(i.self_ty),
                    _ => Some(ids::ERROR),
                }
            }
            DeclKind::Trait => Some(ids::SELF_TY),
            _ => None,
        }
    }

    /// What fn `d` is when it may not declare `needs` (E0709): `main` of the root package, a
    /// method of a trait or of a trait impl, or an `extern` export.
    fn needs_forbidden(&mut self, d: DeclId, decl: &crate::decls::Decl) -> Option<&'static str> {
        if decl.flags.has(DeclFlags::EXTERN_EXPORT) {
            return Some("an `extern` export");
        }
        match decl.kind {
            DeclKind::TraitMethod => Some("a trait method"),
            DeclKind::Method => {
                self.ensure(decl.parent);
                match &self.sigs[decl.parent.idx()] {
                    DeclSig::Impl(i) if i.trait_ref.is_some() => Some("a trait method"),
                    _ => None,
                }
            }
            DeclKind::Fn => {
                let main = self.prog.strings.lookup("main")?;
                let is_main = self.prog.packages.iter().any(|p| {
                    p.is_root
                        && p.root_module.get().is_some_and(|m| self.prog.module_item(m, main) == Some(d))
                });
                is_main.then_some("`main`")
            }
            _ => None,
        }
    }

    fn build_fn(&mut self, d: DeclId) -> DeclSig {
        let Some(decl) = self.prog.decl(d).copied() else { return DeclSig::None };
        let Some(icx) = self.icx(decl.item) else { return DeclSig::None };
        let n = decl.node;
        let generics = self.generics_of(d);
        let n_parent = if matches!(decl.kind, DeclKind::Method | DeclKind::TraitMethod) {
            self.generics_of(decl.parent).len() as u32
        } else {
            0
        };
        let self_ty = if matches!(decl.kind, DeclKind::Method | DeclKind::TraitMethod) {
            self.self_type_of(decl.parent)
        } else {
            None
        };
        let env = TyEnv { icx, n_parent, self_ty };
        let kind = icx.kind(n);
        let ungated = std::mem::replace(&mut self.ungated, icx.child(n, NodeKind::Needs).is_some());
        let mut params = Vec::new();
        let mut param_spans = Vec::new();
        let mut param_nodes: Vec<u32> = Vec::new();
        let mut self_kind = None;
        let mut variadic = false;
        // Params (XParams nest for extern imports).
        let mut stack: Vec<u32> = Vec::new();
        if let Some(p) = icx.child(n, NodeKind::Params) {
            stack.extend(icx.kids_vec(p));
        }
        if let Some(p) = icx.child(n, NodeKind::XParams) {
            let mut cur = Some(p);
            while let Some(x) = cur {
                if icx.is(icx.a(x), Term::DotDotDot) {
                    variadic = true;
                }
                if let Some(pp) = icx.child(x, NodeKind::Param) {
                    stack.push(pp);
                }
                cur = icx.child(x, NodeKind::XParams);
            }
        }
        for p in stack {
            if icx.kind(p) != Some(NodeKind::Param) {
                continue;
            }
            param_nodes.push(p);
            if let Some(sp) = icx.child(p, NodeKind::SelfParam) {
                let a = icx.a(sp);
                let sk = if icx.is(a, Term::Amp) {
                    if icx.is(a + 1, Term::KwMut) { SelfKind::RefMut } else { SelfKind::Ref }
                } else {
                    SelfKind::Value
                };
                let st = self_ty.unwrap_or(ids::ERROR);
                let t = match sk {
                    SelfKind::Value => st,
                    SelfKind::Ref => self.mk_ref(false, st),
                    SelfKind::RefMut => self.mk_ref(true, st),
                };
                if params.is_empty() {
                    self_kind = Some(sk);
                }
                params.push(t);
                param_spans.push(icx.node_span(sp));
                continue;
            }
            match icx.child(p, NodeKind::Type) {
                Some(t) => {
                    params.push(lower_type(self, &env, t, false));
                    param_spans.push(icx.node_span(t));
                }
                None => {
                    params.push(ids::ERROR);
                    param_spans.push(icx.node_span(p));
                }
            }
        }
        // Return type.
        let (ret_node, from_node) = match icx.child(n, NodeKind::RetClause) {
            Some(rc) => (icx.child(rc, NodeKind::Type), icx.child(rc, NodeKind::From)),
            None => {
                // extern fns and exports: `-> Type` directly.
                let direct = match kind {
                    Some(NodeKind::ExternFn | NodeKind::ExternItem) => icx.child(n, NodeKind::Type),
                    _ => None,
                };
                (direct, None)
            }
        };
        let ret = match ret_node {
            Some(t) => lower_type(self, &env, t, true),
            None => ids::UNIT,
        };
        self.ungated = ungated;
        let ret_span = ret_node.map(|t| icx.node_span(t));
        let mut origins = Vec::new();
        // span and written name of each origin (crate::origins::check_sig)
        let mut origin_at: Vec<(Span, String)> = Vec::new();
        if let Some(f) = from_node {
            for o in icx.kids_vec(f) {
                let a = icx.a(o);
                let copy = icx.is(a, Term::KwCopy);
                let Some(u) = icx.uses(o).first() else { continue };
                let bnode = match u.res {
                    Res::Local(b) | Res::SelfValue(b) => b.node,
                    _ => continue,
                };
                // the binding node of a param is its Param node, of self its SelfParam node
                let idx = param_nodes
                    .iter()
                    .position(|&p| p == bnode || icx.child(p, NodeKind::SelfParam) == Some(bnode));
                if let Some(i) = idx {
                    origins.push(Origin { param: i.min(255) as u8, copy });
                    let name_tok = if copy { a + 1 } else { a };
                    origin_at.push((icx.node_span(o), icx.tok_str(name_tok).to_string()));
                }
            }
        }
        let needs_node = icx.child(n, NodeKind::Needs);
        let mut features = match needs_node {
            Some(nd) => needs_bits(self, &icx, nd),
            None => 0,
        };
        if let Some(nd) = needs_node
            && let Some(what) = self.needs_forbidden(d, &decl)
        {
            // E0709; the fn is checked as if it had no `needs` (no second error from the trait
            // impl signature compare).
            let dg = mk("E0709", None, &[("what", what)], icx.node_span(nd), &[]);
            self.diags.extend(dg);
            features = 0;
        } else if needs_node.is_some()
            && stave_target::FeatureSet::from_bits(features)
                .is_some_and(|fs| fs.is_owned_by(self.prog.target.arch()))
        {
            // E0703 (a fn needing another arch's features is left out of the build)
            let ungated = std::mem::replace(&mut self.ungated, true);
            for (&t, &sp) in params.iter().zip(&param_spans).chain(ret_span.iter().map(|s| (&ret, s))) {
                self.sig_width(sp, t, features);
            }
            self.ungated = ungated;
        }
        let (deny, deny_spans) = match icx.child(n, NodeKind::Without) {
            Some(nd) => deny_bits(self, &icx, nd),
            None => (0, Vec::new()),
        };
        let mut is_unsafe = decl.flags.has(DeclFlags::UNSAFE);
        let mut abi = Abi::Stave;
        if decl.flags.has(DeclFlags::EXTERN_IMPORT) {
            abi = Abi::C;
            // the ExternItem's ABI string
            if let Some(root) = icx.ast.root() {
                for x in icx.ast.descendants(root) {
                    if icx.kind(x) == Some(NodeKind::ExternItem) && icx.ast.descendants(x).contains(&n) {
                        let s = icx.tok_str(icx.a(x) + 1);
                        if s == "\"stave-intrinsic\"" {
                            abi = Abi::Intrinsic;
                        }
                    }
                }
            }
            is_unsafe = true;
        } else if decl.flags.has(DeclFlags::EXTERN_EXPORT) {
            abi = Abi::C;
        }
        // The parameters' `from` clauses as written (crate::origins::check_params).
        let mut clauses: Vec<crate::origins::ParamClause> = Vec::new();
        for (ti, &p) in param_nodes.iter().enumerate() {
            let Some(pf) = icx.child(p, NodeKind::ParamFrom) else { continue };
            let name = match icx.child(p, NodeKind::SelfParam) {
                Some(_) => "self".to_string(),
                None => icx.tok_str(icx.a(p)).to_string(),
            };
            let from = icx.child(pf, NodeKind::From).unwrap_or(pf);
            let mut entries = Vec::new();
            for o in icx.kids_vec(from) {
                if icx.kind(o) != Some(NodeKind::Origin) {
                    continue;
                }
                let a = icx.a(o);
                let copy = icx.is(a, Term::KwCopy);
                let Some(u) = icx.uses(o).first() else { continue };
                let bnode = match u.res {
                    Res::Local(b) | Res::SelfValue(b) => b.node,
                    _ => continue,
                };
                let idx = param_nodes
                    .iter()
                    .position(|&q| q == bnode || icx.child(q, NodeKind::SelfParam) == Some(bnode));
                if let Some(i) = idx {
                    let name_tok = if copy { a + 1 } else { a };
                    entries.push(crate::origins::ClauseEntry {
                        source: i.min(255) as u8,
                        copy,
                        span: icx.node_span(o),
                        name: icx.tok_str(name_tok).to_string(),
                    });
                }
            }
            clauses.push(crate::origins::ParamClause {
                target: ti.min(255) as u8,
                name,
                span: icx.node_span(pf),
                entries,
            });
        }
        let mut sig = FnSig {
            decl: d,
            generics,
            n_parent,
            params,
            param_spans,
            self_kind,
            ret,
            ret_span,
            origins,
            param_origins: Vec::new(),
            features,
            deny,
            deny_spans,
            is_unsafe,
            abi,
            variadic,
            node: n,
        };
        // E0520 E0521 (origin table).
        let ds = crate::origins::check_sig(self.prog, &self.tys, &sig, &origin_at);
        self.diags.extend(ds);
        // E0514 E0515 E0520 E0521 (the parameters' clauses).
        if !clauses.is_empty() {
            let kernel = kind == Some(NodeKind::KernelDecl);
            let mirrors_trait = decl.kind == DeclKind::Method && {
                self.ensure(decl.parent);
                matches!(&self.sigs[decl.parent.idx()], DeclSig::Impl(i) if i.trait_ref.is_some())
            };
            let (param_origins, ds) =
                crate::origins::check_params(self.prog, &self.tys, &sig, kernel, mirrors_trait, &clauses);
            self.diags.extend(ds);
            sig.param_origins = param_origins;
        }
        DeclSig::Fn(Box::new(sig))
    }

    fn build_adt(&mut self, d: DeclId) -> DeclSig {
        let Some(decl) = self.prog.decl(d).copied() else { return DeclSig::None };
        let Some(icx) = self.icx(decl.item) else { return DeclSig::None };
        let generics = self.generics_of(d);
        let env = TyEnv { icx, n_parent: 0, self_ty: None };
        let is_enum = decl.kind == DeclKind::Enum;
        let is_view = decl.flags.has(DeclFlags::VIEW);
        let mut variants = Vec::new();
        let prog = self.prog;
        let field_defs = |b: &mut SigBuilder<'p>, parent: DeclId| -> Vec<FieldDef> {
            let mut out = Vec::new();
            for &f in prog.members(parent) {
                let Some(fd) = prog.decl(f).copied() else { continue };
                if fd.kind != DeclKind::Field {
                    continue;
                }
                let ty = match icx.child(fd.node, NodeKind::Type) {
                    Some(t) => lower_type(b, &env, t, false),
                    None => ids::ERROR,
                };
                out.push(FieldDef { decl: f, name: fd.name, ty, is_pub: fd.is_pub(), span: fd.name_span });
            }
            out
        };
        if is_enum {
            for &v in prog.members(d) {
                let Some(vd) = prog.decl(v).copied() else { continue };
                if vd.kind != DeclKind::Variant {
                    continue;
                }
                let (shape, fields) = if vd.flags.has(DeclFlags::TUPLE) {
                    let mut fs = Vec::new();
                    for c in icx.kids_vec(vd.node) {
                        if icx.kind(c) == Some(NodeKind::Type) {
                            let ty = lower_type(self, &env, c, false);
                            fs.push(FieldDef {
                                decl: DeclId::NONE,
                                name: Sym::NONE,
                                ty,
                                is_pub: true,
                                span: icx.node_span(c),
                            });
                        }
                    }
                    (VariantShape::Tuple, fs)
                } else if vd.flags.has(DeclFlags::RECORD) {
                    (VariantShape::Record, field_defs(self, v))
                } else {
                    (VariantShape::Unit, Vec::new())
                };
                variants.push(VariantDef { decl: v, name: vd.name, shape, fields });
            }
        } else {
            let fields = field_defs(self, d);
            variants.push(VariantDef { decl: d, name: decl.name, shape: VariantShape::Record, fields });
        }
        // E0410: reference- or view-typed field in a non-view ADT.
        if !is_view {
            let adt_name = self.prog.strings.get(decl.name).to_string();
            let mut diags = Vec::new();
            for v in &variants {
                for f in &v.fields {
                    if f.ty != ids::ERROR && crate::kinds::is_view_concrete(self.prog, &self.tys, f.ty) {
                        let tname =
                            types::display(&self.tys, &NameCx { prog: self.prog, generics: &generics }, f.ty);
                        let fname = if f.name.is_some() {
                            self.prog.strings.get(f.name).to_string()
                        } else {
                            "_".to_string()
                        };
                        diags.extend(crate::kinds::e0410(&fname, &tname, &adt_name, f.span));
                    }
                }
            }
            self.diags.extend(diags);
        }
        DeclSig::Adt(Box::new(AdtDef { decl: d, is_enum, is_view, generics, variants }))
    }

    fn build_impl(&mut self, d: DeclId) -> DeclSig {
        let Some(decl) = self.prog.decl(d).copied() else { return DeclSig::None };
        let Some(icx) = self.icx(decl.item) else { return DeclSig::None };
        let Some(info) = self.prog.impl_info(d).cloned() else { return DeclSig::None };
        let generics = self.generics_of(d);
        let env = TyEnv { icx, n_parent: 0, self_ty: None };
        let self_ty =
            if info.self_ty == u32::MAX { ids::ERROR } else { lower_type(self, &env, info.self_ty, false) };
        let trait_ref = if info.trait_ty == u32::MAX {
            None
        } else {
            let tb = icx.child(info.trait_ty, NodeKind::TypeBase).unwrap_or(info.trait_ty);
            let env2 = TyEnv { icx, n_parent: 0, self_ty: Some(self_ty) };
            self.trait_ref_at(&env2, tb)
        };
        let head = match self.tys.kind(self_ty) {
            TypeKind::Adt { def, .. } => Some(TypeHead::Adt(*def)),
            k => types::type_scalar(k).map(|s| TypeHead::Prim(Prim::Scalar(s))).or_else(|| match k {
                TypeKind::Vector { lane, n } => types::type_scalar(self.tys.kind(*lane))
                    .map(|l| TypeHead::Prim(Prim::Vector { lane: l, n: *n })),
                TypeKind::Mask { width, n } => Some(TypeHead::Prim(Prim::Mask { width: *width, n: *n })),
                _ => None,
            }),
        };
        DeclSig::Impl(Box::new(ImplDef {
            decl: d,
            item: decl.item,
            generics,
            self_ty,
            head,
            trait_ref,
            methods: info.methods.clone(),
            is_unsafe: decl.flags.has(DeclFlags::UNSAFE),
        }))
    }

    fn build_trait(&mut self, d: DeclId) -> DeclSig {
        let Some(decl) = self.prog.decl(d).copied() else { return DeclSig::None };
        let generics = self.generics_of(d);
        let methods: Vec<DeclId> = self
            .prog
            .members(d)
            .iter()
            .copied()
            .filter(|&m| self.prog.decl(m).is_some_and(|x| x.kind == DeclKind::TraitMethod))
            .collect();
        DeclSig::Trait(Box::new(TraitDef {
            decl: d,
            generics,
            methods,
            is_unsafe: decl.flags.has(DeclFlags::UNSAFE),
        }))
    }

    fn build_alias(&mut self, d: DeclId) -> DeclSig {
        let Some(decl) = self.prog.decl(d).copied() else { return DeclSig::None };
        let Some(icx) = self.icx(decl.item) else { return DeclSig::None };
        let generics = self.generics_of(d);
        let env = TyEnv { icx, n_parent: 0, self_ty: None };
        // `type A = never;` is allowed (the alias is checked where it is used).
        let target = match icx.child(decl.node, NodeKind::Type) {
            Some(t) => lower_type(self, &env, t, true),
            None => ids::ERROR,
        };
        DeclSig::Alias(AliasDef { generics, target })
    }

    fn build_const(&mut self, d: DeclId) -> DeclSig {
        let Some(decl) = self.prog.decl(d).copied() else { return DeclSig::None };
        let Some(icx) = self.icx(decl.item) else { return DeclSig::None };
        let env = TyEnv { icx, n_parent: 0, self_ty: None };
        let kids = icx.kids_vec(decl.node);
        let ty_node = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Type));
        let init = kids.iter().copied().find(|&c| icx.is_expr(c));
        let ty = match ty_node {
            Some(t) => lower_type(self, &env, t, false),
            None => ids::ERROR,
        };
        let is_static = decl.kind == DeclKind::Static;
        // E0413: no reference or view, except `&str` from a string literal or such a const.
        let mut e0413 = false;
        if ty != ids::ERROR
            && crate::kinds::is_view_concrete(self.prog, &self.tys, ty)
            && !(ty == ids::REF_STR && init.is_some_and(|e| self.is_static_str_init(&icx, e)))
        {
            let name = self.prog.strings.get(decl.name).to_string();
            let tname = types::display(&self.tys, &NameCx { prog: self.prog, generics: &[] }, ty);
            let sp = ty_node.map_or(decl.name_span, |t| icx.node_span(t));
            self.diags.extend(crate::kinds::e0413(is_static, &name, &tname, sp));
            e0413 = true;
        }
        // The value: cycles report E0904 through `ensure` (InProgress).
        self.const_stack.push(d);
        // Publish the type before evaluating (a self-reference sees the type, not the value).
        let value = match init {
            Some(e) if ty != ids::ERROR && !e0413 => consteval::eval_item_const(self, &env, e, ty),
            _ => None,
        };
        self.const_stack.pop();
        let c = ConstDef { ty, value };
        if is_static { DeclSig::Static(c) } else { DeclSig::Const(c) }
    }

    /// Whether the const initializer `e` is a string literal or a path to a `&str` const.
    fn is_static_str_init(&mut self, icx: &ItemCx<'p>, e: u32) -> bool {
        let prim = match icx.kind(e) {
            Some(NodeKind::Primary) => {
                if icx.is(icx.a(e), Term::StrLit) {
                    return true;
                }
                e
            }
            // a dotted path `m.NAME` (no child but the Primary)
            Some(NodeKind::Postfix) => {
                let kids = icx.kids_vec(e);
                if kids.len() != 1 {
                    return false;
                }
                kids[0]
            }
            _ => return false,
        };
        // the path's uses are keyed at the Primary, the last one names the const
        if let Some(u) = icx.uses(prim).last()
            && let Res::Decl(d) = u.res
            && self.prog.decl(d).is_some_and(|x| x.kind == DeclKind::Const)
        {
            self.ensure(d);
            return matches!(&self.sigs[d.idx()], DeclSig::Const(c) if c.ty == ids::REF_STR);
        }
        false
    }

    /// The const value of decl `d` (demand-driven).
    pub fn const_value(&mut self, d: DeclId) -> Option<(TypeId, Val)> {
        self.ensure(d);
        match &self.sigs.get(d.idx())? {
            DeclSig::Const(c) => Some((c.ty, c.value.clone()?)),
            _ => None,
        }
    }

    /// The parameter types and return type of fn `d` (demand-driven).
    pub fn fn_types(&mut self, d: DeclId) -> Option<(Vec<TypeId>, TypeId)> {
        self.ensure(d);
        match &self.sigs.get(d.idx())? {
            DeclSig::Fn(f) => Some((f.params.clone(), f.ret)),
            _ => None,
        }
    }

    /// The ADT def of `d` (demand-driven), cloned.
    pub fn adt_def(&mut self, d: DeclId) -> Option<AdtDef> {
        self.ensure(d);
        match &self.sigs.get(d.idx())? {
            DeclSig::Adt(a) => Some((**a).clone()),
            _ => None,
        }
    }

    /// Layout `(size, align)` of a concrete type, demand-driven.
    pub fn layout(&mut self, t: TypeId) -> Option<(u64, u64)> {
        if self.layout_busy.contains(&t) || self.layout_busy.len() > 4096 {
            return None;
        }
        self.layout_busy.push(t);
        let r = self.layout_inner(t);
        self.layout_busy.pop();
        r
    }

    fn layout_inner(&mut self, t: TypeId) -> Option<(u64, u64)> {
        let k = self.tys.kind(t).clone();
        Some(match k {
            TypeKind::Adt { def, args } => {
                let adt = self.adt_def(def)?;
                let mut variant_layouts = Vec::new();
                for v in &adt.variants {
                    let mut fields = Vec::new();
                    for f in &v.fields {
                        let ft = self.tys.subst(f.ty, &args, None);
                        fields.push(self.layout(ft)?);
                    }
                    variant_layouts.push(fields);
                }
                crate::traits::adt_layout(adt.is_enum, &variant_layouts)?
            }
            TypeKind::Ref { inner, .. } | TypeKind::Ptr { inner, .. } => {
                if matches!(self.tys.kind(inner), TypeKind::Str | TypeKind::Slice { .. }) {
                    (16, 8)
                } else {
                    (8, 8)
                }
            }
            _ => {
                let mut sub = |x: TypeId| self.layout(x);
                crate::traits::scalar_layout(&k, &mut sub)?
            }
        })
    }
}
