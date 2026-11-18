//! Name resolution: scopes, locals, dotted chains (D4), E0101 E0103 E0105 E0106.
//!
//! Every item is resolved on its own (in parallel) against read-only program tables. The walk is
//! iterative (an explicit task stack), so no nesting depth can exhaust the compiler's stack.
//!
//! Lookup order for a single name:
//! - expression head: locals (innermost first; `let` rebinding shadows), generic params (a type
//!   param starts a type-relative path, `T.m(x)`), then the module scope (items, import
//!   bindings, error decls), the prelude (the `pub` items of `std.core`, in modules of packages
//!   that depend on `std` and in `std` itself), the built-in type names (`never` included), the
//!   built-in fns `size_of` / `align_of`;
//! - type, bound, and pattern-path head: generic params (types only for patterns: none), then
//!   module scope, prelude, built-ins.
//!
//! Dotted chains (D4) are resolved step by step: a module step looks up the module's items
//! (E0106 if private to another module); a type step looks up an enum variant, an inherent
//! method, then a trait-impl method (a trait step: a trait method). The first step after a value
//! (local, fn, const, variant) is a field or method needing types: it is recorded as
//! [`Deferred`] and left to the type checker, as is a step after a type whose members need types
//! (built-in types, generic params, aliases, `Self` in a trait).
//!
//! Recorded per item ([`ItemRes`]): every resolved Ident use, keyed by (node, byte offset) where
//! the node is the chain's `Primary` (expressions), `TypeBase` (types), `BoundItem`, `Pattern`,
//! `FieldPat`, `Name`, `Origin`, `Import`, or `AllocatorDecl` node; the deferred steps; and the
//! local bindings (keyed by their binding node: `Param`, `SelfParam`, `Pattern`, `FieldPat`,
//! `TailBind`; each binding node binds one name).

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::{Diagnostic, Span};
use stave_syntax::{ItemAst, NodeKind, Term, Token};

use crate::decls::{Decl, DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::{BindingRef, DeclId, GenericRef, ItemId, ModuleId, Sym};
use crate::intern::Strings;
use crate::modules::Module;
use crate::prim::{Builtin, Prim};
use crate::scan::{self, T_COLONCOLON, T_DOT, T_IDENT, is, kind, tok_span};
use crate::{File, Item};

/// What a name use refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Res {
    Decl(DeclId),
    /// A local binding of the same item (`names`: `local I N`).
    Local(BindingRef),
    /// A generic parameter of the same item (`names`: `local I N`, N = the `GenericParam`).
    Generic(GenericRef),
    /// `self` (the enclosing method's `self` parameter; keyword, no `names` line).
    SelfValue(BindingRef),
    /// `Self` (the enclosing impl's self type or the trait's `Self`; keyword, no line).
    SelfType,
    /// A built-in type name (no line).
    Prim(Prim),
    /// A built-in generic fn, `size_of` / `align_of` (no line).
    Builtin(Builtin),
    /// Not found (E0101/E0103 reported): the type checker gives it the error type.
    Unresolved,
}

/// One resolved name use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Use {
    /// The key node (see the module docs).
    pub node: u32,
    /// The file token index of the identifier (an f-string hole: the `FmtLit` token).
    pub token: u32,
    /// Byte offset of the identifier in the file.
    pub offset: u32,
    pub res: Res,
    /// The identifier is inside an f-string hole.
    pub in_fmt: bool,
}

/// What a deferred step needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeferKind {
    /// A field or method of a value: needs the value's type.
    Member,
    /// A member of a type whose members need type checking: a built-in type (`f32x4.splat`), a
    /// generic param (`T.m`), a type alias, `Self` in a trait, or a method name supplied by
    /// several trait impls (E0908 is the type checker's).
    TypeRelative,
}

/// A chain step left to the type checker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deferred {
    pub node: u32,
    pub token: u32,
    pub offset: u32,
    pub kind: DeferKind,
    /// What the step applies to (the resolution of the previous segment).
    pub base: Res,
}

/// How a local is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindKind {
    Param,
    SelfParam,
    ClosureParam,
    Let,
    /// `match` arm pattern.
    Arm,
    /// `for` pattern.
    For,
    /// `tail masked (v, m)` / `tail scalar (x)`.
    Tail,
}

/// One local binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub at: BindingRef,
    pub name: Sym,
    pub kind: BindKind,
    pub mutable: bool,
    /// Token of the bound identifier.
    pub token: u32,
}

/// Resolution results of one item.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ItemRes {
    /// Sorted by (node, offset).
    pub uses: Vec<Use>,
    /// Sorted by (node, offset).
    pub deferred: Vec<Deferred>,
    /// Sorted by binding node.
    pub bindings: Vec<Binding>,
    /// Or-pattern bindings: (binding node in a later alternative, the first alternative's
    /// binding node of that name), sorted by the first component. A later alternative's binding
    /// with no entry has a name the first alternative does not bind.
    pub aliases: Vec<(u32, u32)>,
    /// The first use of syntax that parses but has no semantics in stage0 yet: its first token
    /// and what it is. A program with one is not checked further.
    pub unimplemented: Option<(u32, &'static str)>,
}

impl ItemRes {
    /// The binding `at` (bindings are sorted by node).
    pub fn binding(&self, at: BindingRef) -> Option<&Binding> {
        self.bindings.binary_search_by(|b| b.at.cmp(&at)).ok().map(|i| &self.bindings[i])
    }

    /// The first alternative's binding that binding node `node` of a later or-pattern
    /// alternative stands for.
    pub fn alias(&self, node: u32) -> Option<u32> {
        if self.aliases.is_empty() {
            return None;
        }
        self.aliases.binary_search_by_key(&node, |a| a.0).ok().map(|i| self.aliases[i].1)
    }

    /// The binding node that owns the local of binding node `node`: itself, or the first
    /// alternative's binding when `node` is in a later or-pattern alternative.
    pub fn canonical(&self, node: u32) -> u32 {
        self.alias(node).unwrap_or(node)
    }

    /// The uses keyed at `node`, in offset order.
    pub fn uses_at(&self, node: u32) -> &[Use] {
        let lo = self.uses.partition_point(|u| u.node < node);
        let hi = self.uses.partition_point(|u| u.node <= node);
        &self.uses[lo..hi]
    }
}

/// The head of an impl's self type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TypeHead {
    Adt(DeclId),
    Prim(Prim),
}

/// An impl block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplInfo {
    pub decl: DeclId,
    pub item: ItemId,
    /// `Type` node of the self type in the item's AST.
    pub self_ty: u32,
    /// `Type` node of the trait (`u32::MAX` for an inherent impl).
    pub trait_ty: u32,
    /// The self type's head, when it names a struct, enum, or built-in type (`None`: a generic
    /// param, a reference, tuple, array, fn type, alias, or an unresolved name).
    pub head: Option<TypeHead>,
    /// The trait (`NONE` for an inherent impl or an unresolved trait).
    pub trait_decl: DeclId,
    /// Its methods, in declaration order.
    pub methods: Vec<DeclId>,
}

/// A module scope entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScopeEntry {
    pub decl: DeclId,
    /// Bound by an import (not visible through paths from other modules).
    pub import: bool,
}

/// Methods by (type head, name).
#[derive(Clone, Debug, Default)]
pub struct MethodTable {
    pub inherent: FxHashMap<(TypeHead, Sym), Vec<DeclId>>,
    pub via_trait: FxHashMap<(TypeHead, Sym), Vec<DeclId>>,
}

/// Read-only program tables for the per-item walk.
pub(crate) struct Tables<'a> {
    pub files: &'a [File],
    pub modules: &'a [Module],
    pub items: &'a [Item],
    pub decls: &'a [Decl],
    pub strings: &'a Strings,
    pub scopes: &'a [FxHashMap<Sym, ScopeEntry>],
    pub prelude: &'a FxHashMap<Sym, DeclId>,
    pub prims: &'a FxHashMap<Sym, Prim>,
    pub builtins: &'a FxHashMap<Sym, Builtin>,
    /// Per module: it sees the prelude (its package is `std` or depends on it).
    pub prelude_ok: &'a [bool],
    /// (parent, name) → member: variants of enums, fields, trait methods.
    pub members: &'a FxHashMap<(DeclId, Sym), DeclId>,
    pub methods: &'a MethodTable,
    pub impls: &'a [ImplInfo],
    pub impl_by_decl: &'a FxHashMap<DeclId, u32>,
    /// Identifiers of dropped `impl` headers: a failed method lookup on a type of that name is
    /// silent (no cascades).
    pub poisoned: &'a FxHashSet<Sym>,
    /// Variant name → enums declaring it (for E0103).
    pub variant_names: &'a FxHashMap<Sym, Vec<DeclId>>,
}

impl Tables<'_> {
    pub fn global(&self, module: ModuleId, sym: Sym) -> Option<Res> {
        if let Some(e) = self.scopes.get(module.idx()).and_then(|s| s.get(&sym)) {
            return Some(Res::Decl(e.decl));
        }
        if self.prelude_ok.get(module.idx()).copied().unwrap_or(false)
            && let Some(&d) = self.prelude.get(&sym)
        {
            return Some(Res::Decl(d));
        }
        if let Some(&p) = self.prims.get(&sym) {
            return Some(Res::Prim(p));
        }
        let b = *self.builtins.get(&sym)?;
        if b.soa_only() && self.modules.get(module.idx()).is_none_or(|m| m.path_text != "std.soa") {
            return None;
        }
        Some(Res::Builtin(b))
    }

    /// An item of `module` visible through a path (imports excluded).
    fn module_item(&self, module: ModuleId, sym: Sym) -> Option<DeclId> {
        let e = self.scopes.get(module.idx())?.get(&sym)?;
        if e.import { None } else { Some(e.decl) }
    }

    fn decl(&self, d: DeclId) -> Option<&Decl> {
        self.decls.get(d.idx())
    }

    /// Method `name` of `head`: inherent first, then the unique trait-impl method.
    fn method(&self, head: TypeHead, name: Sym) -> MethodLookup {
        if let Some(v) = self.methods.inherent.get(&(head, name))
            && let Some(&d) = v.first()
        {
            return MethodLookup::Found(d);
        }
        match self.methods.via_trait.get(&(head, name)).map(Vec::as_slice) {
            Some([d]) => MethodLookup::Found(*d),
            Some([_, _, ..]) => MethodLookup::Ambiguous,
            _ => MethodLookup::None,
        }
    }

    pub fn module_path(&self, m: ModuleId) -> &str {
        self.modules.get(m.idx()).map_or("", |m| &m.path_text)
    }
}

enum MethodLookup {
    Found(DeclId),
    Ambiguous,
    None,
}

/// One segment of a dotted chain.
#[derive(Clone, Copy, Debug)]
struct Seg {
    sym: Sym,
    tok: u32,
    off: u32,
    len: u32,
}

/// Where a chain is (what the next step applies to).
#[derive(Clone, Copy, Debug)]
enum St {
    Module(ModuleId),
    Type(DeclId),
    /// A type whose members need types.
    TypeRel(Res),
    Value(Res),
    /// Nothing more to resolve (an error decl, or after an error).
    Stop,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ns {
    Value,
    Type,
    /// Pattern paths and dispatch/`Name` paths: globals only, no locals or generics.
    Global,
}

#[derive(Clone, Copy, Debug)]
enum SelfCtx {
    None,
    Impl(u32),
    Trait,
}

enum Task {
    Visit(u32),
    PopLocals(u32),
    PopGenerics(u32),
    BindPattern(u32, BindKind),
    BindArm(u32),
    BindParam(u32, BindKind),
    EndFn { gmark: u32, lmark: u32, self_param: Option<BindingRef>, param_mark: u32, level: u8 },
    EndClosure { lmark: u32, param_mark: u32 },
    EndContainer { gmark: u32, ctx: SelfCtx, level: u8 },
    PopName,
    EndContract(Option<Clause>),
}

/// The contract clause being resolved: inside one, `old(..)` is the entry snapshot and, inside
/// an `ensures`, `result` is the returned value.
#[derive(Clone, Copy, Debug)]
struct Clause {
    ensures: bool,
    /// The binding node of `result`: the fn's first `Ensures` node.
    result: u32,
}

struct W<'t, 'a> {
    t: &'t Tables<'a>,
    file: &'a File,
    toks: &'a [Token],
    syms: &'a [Sym],
    ast: ItemAst<'a>,
    module: ModuleId,
    item: ItemId,
    locals: Vec<(Sym, BindingRef)>,
    generics: Vec<(Sym, GenericRef)>,
    self_param: Option<BindingRef>,
    self_ctx: SelfCtx,
    /// Generic level for the next `Generics` list.
    level: u8,
    /// Locals from here are the current fn's / closure's params (E0105 among them).
    param_mark: u32,
    /// Closure nesting depth.
    closures: u32,
    /// Enclosing named things (type, fn) for E0105 scope text.
    names: Vec<Sym>,
    res: ItemRes,
    diags: Vec<Diagnostic>,
    stack: Vec<Task>,
    segs: Vec<Seg>,
    chain: Vec<u32>,
    /// The first `Ensures` node of the fn being resolved.
    first_ensures: Option<u32>,
    clause: Option<Clause>,
}

impl<'t, 'a> W<'t, 'a> {
    fn a(&self, n: u32) -> u32 {
        self.ast.get(n).map_or(0, |x| x.a)
    }

    fn sym(&self, tok: u32) -> Sym {
        self.syms.get(tok as usize).copied().unwrap_or(Sym::NONE)
    }

    fn span(&self, tok: u32) -> Span {
        tok_span(self.file.id, &self.file.bytes, self.toks, tok as usize)
    }

    fn seg_span(&self, s: &Seg) -> Span {
        Span::new(self.file.id, s.off, s.len)
    }

    fn seg_of(&self, tok: u32) -> Seg {
        let sp = self.span(tok);
        Seg { sym: self.sym(tok), tok, off: sp.offset, len: sp.len }
    }

    fn name(&self, s: Sym) -> &'a str {
        self.t.strings.get(s)
    }

    fn record(&mut self, node: u32, seg: &Seg, res: Res, in_fmt: bool) {
        self.res.uses.push(Use { node, token: seg.tok, offset: seg.off, res, in_fmt });
    }

    fn defer(&mut self, node: u32, seg: &Seg, kind: DeferKind, base: Res) {
        self.res.deferred.push(Deferred { node, token: seg.tok, offset: seg.off, kind, base });
    }

    fn diag(&mut self, d: Option<Diagnostic>) {
        self.diags.extend(d);
    }

    fn unimplemented(&mut self, n: u32, what: &'static str) {
        let tok = self.a(n);
        if self.res.unimplemented.is_none_or(|(t, _)| tok < t) {
            self.res.unimplemented = Some((tok, what));
        }
    }

    fn e0101(&mut self, seg: &Seg) {
        let n = self.name(seg.sym);
        let n = if n.is_empty() { "_" } else { n };
        let d = mk("E0101", None, &[("name", n)], self.seg_span(seg), &[]);
        self.diag(d);
    }

    fn scope_text(&self) -> String {
        let mut s = self.t.module_path(self.module).to_string();
        for &n in self.names.iter().filter(|n| n.is_some()) {
            if !s.is_empty() {
                s.push('.');
            }
            s.push_str(self.name(n));
        }
        s
    }

    fn e0105(&mut self, name: Sym, primary: Span, first: Span) {
        let scope = self.scope_text();
        let d =
            mk("E0105", None, &[("name", self.name(name)), ("scope", &scope)], primary, &[("first", first)]);
        self.diag(d);
    }

    /// The contextual names of a contract clause, which win over every other binding there:
    /// `old` called directly ([`Builtin::Old`]), and in an `ensures` the returned value `result`
    /// (a local whose binding node is the fn's first `Ensures` node).
    fn contextual(&self, h: &Seg, nsegs: usize) -> Option<Res> {
        let c = self.clause?;
        let name = self.name(h.sym);
        if name == "old" && nsegs == 1 && is(self.toks, h.tok as usize + 1, Term::LParen) {
            return Some(Res::Builtin(Builtin::Old));
        }
        if name == "result" && c.ensures {
            return Some(Res::Local(BindingRef { node: c.result }));
        }
        None
    }

    fn lookup_local(&self, sym: Sym) -> Option<BindingRef> {
        self.locals.iter().rev().find(|(s, _)| *s == sym).map(|&(_, b)| b)
    }

    fn lookup_generic(&self, sym: Sym) -> Option<GenericRef> {
        self.generics.iter().rev().find(|(s, _)| *s == sym).map(|&(_, g)| g)
    }

    /// Visibility check of a decl reached through a path (E0106).
    fn check_vis(&mut self, d: DeclId, at: Span) {
        let Some(decl) = self.t.decl(d) else { return };
        let mut owner = *decl;
        if decl.kind == DeclKind::Method && decl.flags.has(DeclFlags::TRAIT_IMPL) {
            // Trait impl methods have the trait's visibility.
            let trait_decl = self
                .t
                .impl_by_decl
                .get(&decl.parent)
                .and_then(|&i| self.t.impls.get(i as usize))
                .map_or(DeclId::NONE, |i| i.trait_decl);
            match self.t.decl(trait_decl) {
                Some(td) => owner = *td,
                None => return,
            }
        }
        if owner.kind == DeclKind::Variant || owner.kind == DeclKind::TraitMethod {
            return;
        }
        if owner.module != self.module && !owner.is_pub() {
            let module = self.t.module_path(owner.module).to_string();
            let name = self.name(decl.name);
            let d =
                mk("E0106", None, &[("name", name), ("module", &module)], at, &[("decl", owner.name_span)]);
            self.diag(d);
        }
    }

    fn classify(&self, r: Res) -> St {
        match r {
            Res::Decl(d) => match self.t.decl(d).map(|d| (d.kind, d.module)) {
                Some((DeclKind::Module, m)) => St::Module(m),
                Some((DeclKind::Struct | DeclKind::Enum | DeclKind::Trait, _)) => St::Type(d),
                Some((DeclKind::TypeAlias, _)) => St::TypeRel(r),
                Some((DeclKind::Error | DeclKind::Impl, _)) | None => St::Stop,
                Some(_) => St::Value(r),
            },
            Res::Prim(_) => St::TypeRel(r),
            Res::Generic(g) if !g.is_const => St::TypeRel(r),
            Res::SelfType => match self.self_ctx {
                SelfCtx::Impl(i) => match self.t.impls.get(i as usize).and_then(|i| i.head) {
                    Some(TypeHead::Adt(d)) => St::Type(d),
                    _ => St::TypeRel(r),
                },
                _ => St::TypeRel(r),
            },
            Res::Unresolved => St::Stop,
            Res::Local(_) | Res::SelfValue(_) | Res::Generic(_) | Res::Builtin(_) => St::Value(r),
        }
    }

    /// An enum visible here (declared in this module, bound in its scope, or in the prelude)
    /// with a variant named `sym` (E0103).
    fn enum_with_variant(&self, sym: Sym) -> Option<DeclId> {
        let enums = self.t.variant_names.get(&sym)?;
        enums.iter().copied().find(|&e| {
            let Some(d) = self.t.decl(e) else { return false };
            matches!(self.t.global(self.module, d.name), Some(Res::Decl(x)) if x == e)
        })
    }

    /// Reports an unresolved head: E0103 when `variant_ok` and a visible enum has a variant of
    /// that name, else E0101.
    fn unresolved_head(&mut self, seg: &Seg, variant_ok: bool) {
        if variant_ok && let Some(e) = self.enum_with_variant(seg.sym) {
            let en = self.t.decl(e).map_or(Sym::NONE, |d| d.name);
            let d = mk(
                "E0103",
                None,
                &[("name", self.name(seg.sym)), ("enum", self.name(en))],
                self.seg_span(seg),
                &[],
            );
            self.diag(d);
            return;
        }
        self.e0101(seg);
    }

    /// Resolves the chain `segs` keyed at `node`. `head` is the head's resolution if the caller
    /// already has it (`Self`). `variant_ok`: an unresolved head may be E0103.
    fn chain(&mut self, node: u32, ns: Ns, head: Option<Res>, variant_ok: bool, in_fmt: bool) {
        let segs = std::mem::take(&mut self.segs);
        let Some(h) = segs.first().copied() else {
            self.segs = segs;
            return;
        };
        let contextual = if ns == Ns::Value { self.contextual(&h, segs.len()) } else { None };
        let head_res = match head.or(contextual) {
            Some(r) => Some(r),
            None => match ns {
                Ns::Value => self
                    .lookup_local(h.sym)
                    .map(Res::Local)
                    .or_else(|| self.lookup_generic(h.sym).map(Res::Generic))
                    .or_else(|| self.t.global(self.module, h.sym)),
                Ns::Type => {
                    self.lookup_generic(h.sym).map(Res::Generic).or_else(|| self.t.global(self.module, h.sym))
                }
                Ns::Global => self.t.global(self.module, h.sym),
            },
        };
        let Some(head_res) = head_res else {
            self.unresolved_head(&h, variant_ok);
            self.record(node, &h, Res::Unresolved, in_fmt);
            self.segs = segs;
            return;
        };
        self.record(node, &h, head_res, in_fmt);
        let mut st = self.classify(head_res);
        for s in &segs[1..] {
            match st {
                St::Stop => break,
                St::Value(base) => {
                    if ns == Ns::Type {
                        self.e0101(s);
                    } else {
                        self.defer(node, s, DeferKind::Member, base);
                    }
                    break;
                }
                St::TypeRel(base) => {
                    if ns == Ns::Type {
                        self.e0101(s);
                    } else {
                        self.defer(node, s, DeferKind::TypeRelative, base);
                    }
                    break;
                }
                St::Module(m) => match self.t.module_item(m, s.sym) {
                    Some(d) => {
                        self.check_vis(d, self.seg_span(s));
                        self.record(node, s, Res::Decl(d), in_fmt);
                        st = self.classify(Res::Decl(d));
                    }
                    None => {
                        self.e0101(s);
                        self.record(node, s, Res::Unresolved, in_fmt);
                        break;
                    }
                },
                St::Type(d) => {
                    if ns == Ns::Type {
                        // Types have no member types.
                        self.e0101(s);
                        break;
                    }
                    let Some(decl) = self.t.decl(d).copied() else { break };
                    let member = self.t.members.get(&(d, s.sym)).copied().filter(|&m| {
                        matches!(
                            self.t.decl(m).map(|x| x.kind),
                            Some(DeclKind::Variant | DeclKind::TraitMethod)
                        )
                    });
                    if let Some(m) = member {
                        self.record(node, s, Res::Decl(m), in_fmt);
                        st = St::Value(Res::Decl(m));
                        continue;
                    }
                    if decl.kind == DeclKind::Trait {
                        self.e0101(s);
                        self.record(node, s, Res::Unresolved, in_fmt);
                        break;
                    }
                    match self.t.method(TypeHead::Adt(d), s.sym) {
                        MethodLookup::Found(m) => {
                            self.check_vis(m, self.seg_span(s));
                            self.record(node, s, Res::Decl(m), in_fmt);
                            st = St::Value(Res::Decl(m));
                        }
                        MethodLookup::Ambiguous => {
                            self.defer(node, s, DeferKind::TypeRelative, Res::Decl(d));
                            break;
                        }
                        MethodLookup::None => {
                            if !self.t.poisoned.contains(&decl.name) {
                                self.e0101(s);
                                self.record(node, s, Res::Unresolved, in_fmt);
                            }
                            break;
                        }
                    }
                }
            }
        }
        self.segs = segs;
    }

    /// Fills `self.segs` from the identifier chain at token `first`.
    fn segs_from_tokens(&mut self, first: usize) {
        let mut chain = std::mem::take(&mut self.chain);
        scan::ident_chain(self.toks, first, &mut chain);
        self.segs.clear();
        for &t in &chain {
            let s = self.seg_of(t);
            self.segs.push(s);
        }
        self.chain = chain;
    }

    fn type_like_path(&mut self, n: u32, ns: Ns) {
        let a = self.a(n) as usize;
        if kind(self.toks, a) != T_IDENT {
            return;
        }
        self.segs_from_tokens(a);
        self.chain(n, ns, None, false, false);
    }

    /// A `Primary` (every postfix chain starts with one): resolve its head chain, the name and
    /// the `.Ident` steps that follow it.
    fn chain_head(&mut self, prim: u32) {
        let a = self.a(prim) as usize;
        let k = kind(self.toks, a);
        let (head, mut pos) = if k == T_IDENT {
            if kind(self.toks, a + 1) == T_COLONCOLON {
                let (end, partial) = scan::angle_end(self.toks, a + 2);
                if partial {
                    self.segs_from_tokens(a);
                    self.segs.truncate(1);
                    self.chain(prim, Ns::Value, None, true, false);
                    return;
                }
                (None, end)
            } else {
                (None, a + 1)
            }
        } else if k == Term::KwSelf as u8 {
            let seg = self.seg_of(a as u32);
            match self.self_param {
                Some(b) => {
                    self.record(prim, &seg, Res::SelfValue(b), false);
                    if kind(self.toks, a + 1) == T_DOT && kind(self.toks, a + 2) == T_IDENT {
                        let s = self.seg_of(a as u32 + 2);
                        self.defer(prim, &s, DeferKind::Member, Res::SelfValue(b));
                    }
                }
                None => {
                    let d = mk("E0101", None, &[("name", "self")], self.seg_span(&seg), &[]);
                    self.diag(d);
                    self.record(prim, &seg, Res::Unresolved, false);
                }
            }
            return;
        } else if k == Term::KwSelfType as u8 {
            (Some(()), a + 1)
        } else if k == Term::FmtLit as u8 {
            self.fmt(prim, a);
            return;
        } else {
            return;
        };
        // Head segment plus `.Ident (:: <..>)?` steps.
        self.segs.clear();
        let hs = self.seg_of(a as u32);
        self.segs.push(hs);
        while kind(self.toks, pos) == T_DOT && kind(self.toks, pos + 1) == T_IDENT {
            let s = self.seg_of(pos as u32 + 1);
            self.segs.push(s);
            pos += 2;
            if kind(self.toks, pos) == T_COLONCOLON {
                let (end, partial) = scan::angle_end(self.toks, pos + 1);
                if partial {
                    break;
                }
                pos = end;
            }
        }
        if head.is_some() {
            if matches!(self.self_ctx, SelfCtx::None) {
                let seg = self.segs[0];
                let d = mk("E0101", None, &[("name", "Self")], self.seg_span(&seg), &[]);
                self.diag(d);
                self.record(prim, &seg, Res::Unresolved, false);
                return;
            }
            self.chain(prim, Ns::Value, Some(Res::SelfType), false, false);
        } else {
            self.chain(prim, Ns::Value, None, true, false);
        }
    }

    /// f-string holes of the `FmtLit` token `tok` (Primary `prim`).
    fn fmt(&mut self, prim: u32, tok: usize) {
        let src: &'a [u8] = &self.file.bytes;
        let text = stave_syntax::token_text(src, self.toks, tok);
        let base = self.toks.get(tok).map_or(0, |t| t.start);
        for hole in scan::fmt_holes(text, base) {
            self.segs.clear();
            let mut is_self = false;
            for (k, &(off, s)) in hole.segs.iter().enumerate() {
                if k == 0 && s == "self" {
                    is_self = true;
                }
                let sym = self.t.strings.lookup(s).unwrap_or(Sym::NONE);
                self.segs.push(Seg { sym, tok: tok as u32, off, len: s.len() as u32 });
            }
            if is_self {
                let seg = self.segs[0];
                match self.self_param {
                    Some(b) => {
                        self.record(prim, &seg, Res::SelfValue(b), true);
                        if let Some(s) = self.segs.get(1).copied() {
                            self.defer(prim, &s, DeferKind::Member, Res::SelfValue(b));
                        }
                    }
                    None => {
                        let d = mk("E0101", None, &[("name", "self")], self.seg_span(&seg), &[]);
                        self.diag(d);
                        self.record(prim, &seg, Res::Unresolved, true);
                    }
                }
                continue;
            }
            self.chain(prim, Ns::Value, None, true, true);
        }
    }

    fn bind(&mut self, sym: Sym, at: BindingRef, kind: BindKind, mutable: bool, token: u32) {
        self.locals.push((sym, at));
        self.res.bindings.push(Binding { at, name: sym, kind, mutable, token });
    }

    /// Resolves the paths of the pattern rooted at `root` and binds its names (after the whole
    /// pattern, so bindings never affect its own paths). A name the pattern binds twice is E0105
    /// at each later binding.
    fn bind_pattern(&mut self, root: u32, bk: BindKind) {
        let lo = self.ast.descendants(root).start;
        let mut pending: Vec<(Sym, BindingRef, bool, u32)> = Vec::new();
        for n in lo..=root {
            match self.ast.kind(n) {
                Some(NodeKind::Pattern) => {
                    let a = self.a(n) as usize;
                    match kind(self.toks, a) {
                        k if k == Term::KwMut as u8 => {
                            if kind(self.toks, a + 1) == T_IDENT {
                                pending.push((
                                    self.sym(a as u32 + 1),
                                    BindingRef { node: n },
                                    true,
                                    a as u32 + 1,
                                ));
                            }
                        }
                        T_IDENT => {
                            self.segs_from_tokens(a);
                            let end = self.chain.last().map_or(a, |&l| l as usize) + 1;
                            let suffix =
                                is(self.toks, end, Term::LParen) || is(self.toks, end, Term::DotLBrace);
                            if self.segs.len() == 1 && !suffix {
                                pending.push((self.sym(a as u32), BindingRef { node: n }, false, a as u32));
                            } else {
                                let single = self.segs.len() == 1;
                                self.chain(n, Ns::Global, None, single, false);
                            }
                        }
                        _ => {}
                    }
                }
                Some(NodeKind::FieldPat) => {
                    let a = self.a(n) as usize;
                    if kind(self.toks, a) == T_IDENT && !is(self.toks, a + 1, Term::Colon) {
                        pending.push((self.sym(a as u32), BindingRef { node: n }, false, a as u32));
                    }
                }
                _ => {}
            }
        }
        for (i, &(s, at, m, t)) in pending.iter().enumerate() {
            if let Some(first) = pending[..i].iter().find(|p| p.0 == s) {
                let (ps, fs) = (self.span(t), self.span(first.3));
                self.e0105(s, ps, fs);
            }
            self.bind(s, at, bk, m, t);
        }
    }

    /// Binds the alternatives of arm pattern `ap` (`P1 | P2 | ..`). Only the names of the first
    /// alternative enter the scope: a name of a later alternative is another occurrence of the
    /// first alternative's binding of that name ([`ItemRes::aliases`]; the k-th occurrence of a
    /// name pairs with the k-th there).
    fn bind_arm(&mut self, ap: u32) {
        let mut rev = self.ast.children_rev(ap);
        let Some(last) = rev.next() else { return };
        if rev.next().is_none() {
            self.bind_pattern(last, BindKind::Arm);
            return;
        }
        let alts = self.ast.children(ap);
        let lo = self.locals.len();
        self.bind_pattern(alts[0], BindKind::Arm);
        let hi = self.locals.len();
        let mut taken: Vec<bool> = Vec::new();
        for &alt in &alts[1..] {
            let bmark = self.res.bindings.len();
            self.bind_pattern(alt, BindKind::Arm);
            self.locals.truncate(hi);
            taken.clear();
            taken.resize(hi - lo, false);
            for k in bmark..self.res.bindings.len() {
                let b = self.res.bindings[k];
                let free = (0..hi - lo).find(|&i| !taken[i] && self.locals[lo + i].0 == b.name);
                if let Some(i) = free {
                    taken[i] = true;
                    self.res.aliases.push((b.at.node, self.locals[lo + i].1.node));
                }
            }
        }
    }

    fn bind_param(&mut self, p: u32, bk: BindKind) {
        let a = self.a(p);
        if kind(self.toks, a as usize) != T_IDENT {
            return;
        }
        let sym = self.sym(a);
        let at = BindingRef { node: p };
        let prior = self.locals[self.param_mark as usize..].iter().find(|(s, _)| *s == sym).map(|&(_, b)| b);
        if let Some(first) = prior {
            let fa = self.a(first.node);
            let (ps, fs) = (self.span(a), self.span(fa));
            self.e0105(sym, ps, fs);
        }
        self.bind(sym, at, bk, false, a);
    }

    fn param(&mut self, p: u32, bk: BindKind) {
        if let Some(pf) = self.ast.child(p, NodeKind::ParamFrom) {
            self.unimplemented(pf, "a `from` clause on a parameter");
        }
        if let Some(sp) = self.ast.child(p, NodeKind::SelfParam) {
            let at = BindingRef { node: sp };
            self.self_param = Some(at);
            let a = self.a(sp);
            let mutable = is(self.toks, a as usize + 1, Term::KwMut);
            self.res.bindings.push(Binding {
                at,
                name: Sym::NONE,
                kind: BindKind::SelfParam,
                mutable,
                token: a,
            });
            return;
        }
        self.stack.push(Task::BindParam(p, bk));
        if let Some(t) = self.ast.child(p, NodeKind::Type) {
            self.stack.push(Task::Visit(t));
        }
    }

    fn generics(&mut self, g: u32) {
        let kids = self.ast.children(g);
        let start = self.generics.len();
        let mut index = 0u16;
        for &c in &kids {
            if self.ast.kind(c) != Some(NodeKind::GenericParam) {
                continue;
            }
            let a = self.a(c);
            let is_const = is(self.toks, a as usize, Term::KwConst);
            let nt = if is_const { a + 1 } else { a };
            let sym = self.sym(nt);
            if let Some(&(_, first)) = self.generics[start..].iter().find(|(s, _)| *s == sym) {
                let fa = self.a(first.node);
                let fnt = if first.is_const { fa + 1 } else { fa };
                let (ps, fs) = (self.span(nt), self.span(fnt));
                self.e0105(sym, ps, fs);
            }
            self.generics.push((sym, GenericRef { node: c, index, level: self.level, is_const }));
            index = index.saturating_add(1);
        }
        for &c in kids.iter().rev() {
            self.stack.push(Task::Visit(c));
        }
    }

    fn push_children(&mut self, n: u32) {
        for c in self.ast.children_rev(n) {
            self.stack.push(Task::Visit(c));
        }
    }

    fn fn_like(&mut self, n: u32) {
        let a = self.a(n) as usize;
        let name_tok = (a..(a + 8).min(self.toks.len()))
            .find(|&i| is(self.toks, i, Term::KwFn))
            .map_or(a + 1, |i| i + 1);
        let name = self.sym(name_tok as u32);
        self.names.push(name);
        let level = self.level;
        self.stack.push(Task::PopName);
        self.stack.push(Task::EndFn {
            gmark: self.generics.len() as u32,
            lmark: self.locals.len() as u32,
            self_param: self.self_param,
            param_mark: self.param_mark,
            level,
        });
        self.self_param = None;
        self.param_mark = self.locals.len() as u32;
        if !matches!(self.self_ctx, SelfCtx::None) {
            self.level = 1;
        }
        self.first_ensures = self.ast.child(n, NodeKind::Ensures);
        self.push_children(n);
    }

    fn visit(&mut self, n: u32) {
        let Some(kind_n) = self.ast.kind(n) else { return };
        match kind_n {
            NodeKind::FnDecl
            | NodeKind::KernelDecl
            | NodeKind::DispatchDecl
            | NodeKind::ExternFn
            | NodeKind::TraitFn => self.fn_like(n),
            NodeKind::ExternItem => {
                if self.ast.child(n, NodeKind::Block).is_some() {
                    self.fn_like(n);
                } else {
                    self.push_children(n);
                }
            }
            NodeKind::Params | NodeKind::XParams => {
                let bk = if self.closure_depth() { BindKind::ClosureParam } else { BindKind::Param };
                let kids = self.ast.children(n);
                for &c in kids.iter().rev() {
                    match self.ast.kind(c) {
                        Some(NodeKind::Param) => self.param(c, bk),
                        _ => self.stack.push(Task::Visit(c)),
                    }
                }
            }
            NodeKind::Param => {
                let bk = if self.closure_depth() { BindKind::ClosureParam } else { BindKind::Param };
                self.param(n, bk);
            }
            NodeKind::Generics => self.generics(n),
            NodeKind::Origin => {
                let a = self.a(n) as usize;
                let t = if is(self.toks, a, Term::KwCopy) { a + 1 } else { a };
                let seg = self.seg_of(t as u32);
                if is(self.toks, t, Term::KwSelf) {
                    match self.self_param {
                        Some(b) => self.record(n, &seg, Res::SelfValue(b), false),
                        None => {
                            let d = mk("E0101", None, &[("name", "self")], self.seg_span(&seg), &[]);
                            self.diag(d);
                        }
                    }
                } else if kind(self.toks, t) == T_IDENT {
                    match self.lookup_local(seg.sym) {
                        Some(b) => self.record(n, &seg, Res::Local(b), false),
                        None => {
                            self.e0101(&seg);
                            self.record(n, &seg, Res::Unresolved, false);
                        }
                    }
                }
            }
            NodeKind::Needs
            | NodeKind::Feature
            | NodeKind::Literal
            | NodeKind::AssignOp
            | NodeKind::CmpOp => {}
            NodeKind::Unsafe | NodeKind::SelfParam | NodeKind::Import | NodeKind::AllocatorDecl => {}
            NodeKind::Block => {
                self.stack.push(Task::PopLocals(self.locals.len() as u32));
                self.push_children(n);
            }
            NodeKind::Stmt => {
                let a = self.a(n) as usize;
                if is(self.toks, a, Term::KwLet) {
                    let kids = self.ast.children(n);
                    if let Some(&p) = kids.first() {
                        self.stack.push(Task::BindPattern(p, BindKind::Let));
                    }
                    for &c in kids.iter().skip(1).rev() {
                        self.stack.push(Task::Visit(c));
                    }
                } else {
                    self.push_children(n);
                }
            }
            NodeKind::ForStmt => {
                let kids = self.ast.children(n);
                let mark = self.locals.len() as u32;
                let mut it = kids.iter().copied();
                let pat = it.next();
                let rest: Vec<u32> = it.collect();
                // rest: [Type] Expr Block [TailClause]
                let block_pos = rest.iter().position(|&c| self.ast.kind(c) == Some(NodeKind::Block));
                if let Some(bp) = block_pos {
                    for &c in rest[bp + 1..].iter().rev() {
                        self.stack.push(Task::Visit(c));
                    }
                    self.stack.push(Task::PopLocals(mark));
                    self.stack.push(Task::Visit(rest[bp]));
                    if let Some(p) = pat {
                        self.stack.push(Task::BindPattern(p, BindKind::For));
                    }
                    for &c in rest[..bp].iter().rev() {
                        self.stack.push(Task::Visit(c));
                    }
                }
            }
            NodeKind::TailClause => {
                // `tail masked (v, m)` / `tail scalar (x)`: each name is its own TailBind node.
                let mark = self.locals.len() as u32;
                self.stack.push(Task::PopLocals(mark));
                self.push_children(n);
                for c in self.ast.children(n) {
                    if self.ast.kind(c) != Some(NodeKind::TailBind) {
                        continue;
                    }
                    let t = self.a(c);
                    if kind(self.toks, t as usize) == T_IDENT {
                        let s = self.sym(t);
                        self.bind(s, BindingRef { node: c }, BindKind::Tail, false, t);
                    }
                }
            }
            NodeKind::StmtArm | NodeKind::ExprArm => {
                let kids = self.ast.children(n);
                self.stack.push(Task::PopLocals(self.locals.len() as u32));
                for &c in kids.iter().skip(1).rev() {
                    self.stack.push(Task::Visit(c));
                }
                if let Some(&ap) = kids.first() {
                    self.stack.push(Task::BindArm(ap));
                }
            }
            NodeKind::Requires | NodeKind::Ensures => {
                let ensures = kind_n == NodeKind::Ensures;
                self.stack.push(Task::EndContract(self.clause));
                self.clause = Some(Clause { ensures, result: self.first_ensures.unwrap_or(n) });
                self.push_children(n);
            }
            NodeKind::Without | NodeKind::Effect => {}
            NodeKind::Primary => {
                self.chain_head(n);
                let a = self.a(n) as usize;
                if is(self.toks, a, Term::KwFn) {
                    self.stack.push(Task::EndClosure {
                        lmark: self.locals.len() as u32,
                        param_mark: self.param_mark,
                    });
                    self.param_mark = self.locals.len() as u32;
                    self.closures += 1;
                }
                self.push_children(n);
            }
            NodeKind::TypeBase => {
                let a = self.a(n) as usize;
                if is(self.toks, a, Term::KwSelfType) {
                    let seg = self.seg_of(a as u32);
                    if matches!(self.self_ctx, SelfCtx::None) {
                        let d = mk("E0101", None, &[("name", "Self")], self.seg_span(&seg), &[]);
                        self.diag(d);
                        self.record(n, &seg, Res::Unresolved, false);
                    } else {
                        self.record(n, &seg, Res::SelfType, false);
                    }
                } else {
                    self.type_like_path(n, Ns::Type);
                }
                self.push_children(n);
            }
            NodeKind::BoundItem => {
                self.type_like_path(n, Ns::Type);
                self.push_children(n);
            }
            NodeKind::Name => self.type_like_path(n, Ns::Global),
            NodeKind::Pattern | NodeKind::FieldPat => {}
            NodeKind::ImplDecl | NodeKind::TraitDecl => {
                let ctx = self.self_ctx;
                let level = self.level;
                self.stack.push(Task::EndContainer { gmark: self.generics.len() as u32, ctx, level });
                self.level = 0;
                let d = self.t.items.get(self.item_idx()).map_or(DeclId::NONE, |it| it.decl);
                if kind_n == NodeKind::ImplDecl {
                    let idx = self.t.impl_by_decl.get(&d).copied();
                    self.self_ctx = idx.map_or(SelfCtx::Trait, SelfCtx::Impl);
                    let owner = idx
                        .and_then(|i| self.t.impls.get(i as usize))
                        .and_then(|i| i.head)
                        .map(|h| match h {
                            TypeHead::Adt(d) => self.t.decl(d).map_or(Sym::NONE, |d| d.name),
                            TypeHead::Prim(_) => Sym::NONE,
                        })
                        .unwrap_or(Sym::NONE);
                    self.names.push(owner);
                } else {
                    self.self_ctx = SelfCtx::Trait;
                    let owner = self.t.decl(d).map_or(Sym::NONE, |d| d.name);
                    self.names.push(owner);
                }
                self.stack.push(Task::PopName);
                self.push_children(n);
            }
            NodeKind::AdtDecl | NodeKind::TypeAlias => {
                self.stack.push(Task::PopGenerics(self.generics.len() as u32));
                self.push_children(n);
            }
            _ => self.push_children(n),
        }
    }

    fn closure_depth(&self) -> bool {
        self.closures > 0
    }

    fn item_idx(&self) -> usize {
        self.item.idx()
    }

    fn run(&mut self, root: u32) {
        self.stack.push(Task::Visit(root));
        while let Some(t) = self.stack.pop() {
            match t {
                Task::Visit(n) => self.visit(n),
                Task::PopLocals(m) => self.locals.truncate(m as usize),
                Task::PopGenerics(m) => self.generics.truncate(m as usize),
                Task::BindPattern(p, bk) => self.bind_pattern(p, bk),
                Task::BindArm(ap) => self.bind_arm(ap),
                Task::BindParam(p, bk) => self.bind_param(p, bk),
                Task::EndFn { gmark, lmark, self_param, param_mark, level } => {
                    self.generics.truncate(gmark as usize);
                    self.locals.truncate(lmark as usize);
                    self.self_param = self_param;
                    self.param_mark = param_mark;
                    self.level = level;
                }
                Task::EndClosure { lmark, param_mark } => {
                    self.locals.truncate(lmark as usize);
                    self.param_mark = param_mark;
                    self.closures = self.closures.saturating_sub(1);
                }
                Task::EndContainer { gmark, ctx, level } => {
                    self.generics.truncate(gmark as usize);
                    self.self_ctx = ctx;
                    self.level = level;
                }
                Task::PopName => {
                    self.names.pop();
                }
                Task::EndContract(c) => self.clause = c,
            }
        }
    }
}

/// Resolves one ok item: (results, diagnostics).
pub(crate) fn resolve_item(t: &Tables<'_>, item: ItemId) -> (ItemRes, Vec<Diagnostic>) {
    let Some(it) = t.items.get(item.idx()) else { return Default::default() };
    let Some(file) = t.files.get(it.file.0 as usize) else { return Default::default() };
    if !it.ok {
        return Default::default();
    }
    let ast = file.parsed.item_ast(it.index as usize);
    let Some(root) = ast.root() else { return Default::default() };
    let mut w = W {
        t,
        file,
        toks: &file.parsed.tokens,
        syms: &file.syms,
        ast,
        module: it.module,
        item,
        locals: Vec::new(),
        generics: Vec::new(),
        self_param: None,
        self_ctx: SelfCtx::None,
        level: 0,
        param_mark: 0,
        closures: 0,
        names: Vec::new(),
        res: ItemRes::default(),
        diags: Vec::new(),
        stack: Vec::new(),
        segs: Vec::new(),
        chain: Vec::new(),
        first_ensures: None,
        clause: None,
    };
    w.run(root);
    let mut res = w.res;
    res.uses.sort_by_key(|u| (u.node, u.offset));
    res.deferred.sort_by_key(|d| (d.node, d.offset));
    res.bindings.sort_by_key(|b| b.at);
    res.aliases.sort_unstable();
    (res, w.diags)
}

/// The head of an impl's self type and its trait, resolved without diagnostics (the item walk
/// reports them): `(head, trait decl)`.
pub(crate) fn impl_header(
    t: &Tables<'_>,
    item: ItemId,
    self_ty: u32,
    trait_ty: u32,
) -> (Option<TypeHead>, DeclId) {
    let Some(it) = t.items.get(item.idx()) else { return (None, DeclId::NONE) };
    let Some(file) = t.files.get(it.file.0 as usize) else { return (None, DeclId::NONE) };
    let ast = file.parsed.item_ast(it.index as usize);
    let toks = &file.parsed.tokens;
    // Generic param names of the impl (a head naming one is a blanket impl).
    let mut gnames: Vec<Sym> = Vec::new();
    if let Some(root) = ast.root() {
        for n in ast.descendants(root) {
            if ast.kind(n) == Some(NodeKind::ImplDecl)
                && let Some(g) = ast.child(n, NodeKind::Generics)
            {
                for p in ast.children(g) {
                    let a = ast.get(p).map_or(0, |x| x.a) as usize;
                    let nt = if is(toks, a, Term::KwConst) { a + 1 } else { a };
                    gnames.push(file.syms.get(nt).copied().unwrap_or(Sym::NONE));
                }
            }
        }
    }
    let path_res = |ty: u32| -> Option<Res> {
        let tb = ast.child(ty, NodeKind::TypeBase)?;
        let a = ast.get(tb)?.a as usize;
        if kind(toks, a) != T_IDENT {
            return None;
        }
        let mut chain = Vec::new();
        scan::ident_chain(toks, a, &mut chain);
        let sym = |tk: u32| file.syms.get(tk as usize).copied().unwrap_or(Sym::NONE);
        let head = sym(*chain.first()?);
        if gnames.contains(&head) {
            return None;
        }
        let mut r = t.global(it.module, head)?;
        for &s in &chain[1..] {
            let m = match r {
                Res::Decl(d) => match t.decl(d) {
                    Some(dd) if dd.kind == DeclKind::Module => dd.module,
                    _ => return None,
                },
                _ => return None,
            };
            r = Res::Decl(t.module_item(m, sym(s))?);
        }
        Some(r)
    };
    let head = if self_ty == u32::MAX {
        None
    } else {
        match path_res(self_ty) {
            Some(Res::Decl(d)) => match t.decl(d).map(|x| x.kind) {
                Some(DeclKind::Struct | DeclKind::Enum) => Some(TypeHead::Adt(d)),
                _ => None,
            },
            Some(Res::Prim(p)) => Some(TypeHead::Prim(p)),
            _ => None,
        }
    };
    let trait_decl = if trait_ty == u32::MAX {
        DeclId::NONE
    } else {
        match path_res(trait_ty) {
            Some(Res::Decl(d)) if t.decl(d).map(|x| x.kind) == Some(DeclKind::Trait) => d,
            _ => DeclId::NONE,
        }
    };
    (head, trait_decl)
}

/// E0105 for duplicate methods: per type head over all inherent impls (in item order), and per
/// trait impl (or impl with an unknown head).
pub(crate) fn method_duplicates(
    decls: &[Decl],
    impls: &[ImplInfo],
    strings: &Strings,
    modules: &[Module],
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let head_text = |h: Option<TypeHead>, imp: &ImplInfo| -> String {
        match h {
            Some(TypeHead::Adt(d)) => {
                let dd = &decls[d.idx()];
                let m = modules.get(dd.module.idx()).map_or("", |m| m.path_text.as_str());
                let n = strings.get(dd.name);
                if m.is_empty() { n.to_string() } else { format!("{m}.{n}") }
            }
            Some(TypeHead::Prim(p)) => p.text(),
            None => {
                let m = decls.get(imp.decl.idx()).and_then(|d| modules.get(d.module.idx()));
                m.map_or(String::new(), |m| m.path_text.clone())
            }
        }
    };
    let report = |ids: &[DeclId], scope: &str, out: &mut Vec<Diagnostic>| {
        for (k, &d) in ids.iter().enumerate() {
            let name = decls[d.idx()].name;
            if let Some(&f) = ids[..k].iter().find(|&&e| decls[e.idx()].name == name) {
                let dd = mk(
                    "E0105",
                    None,
                    &[("name", strings.get(name)), ("scope", scope)],
                    decls[d.idx()].name_span,
                    &[("first", decls[f.idx()].name_span)],
                );
                out.extend(dd);
            }
        }
    };
    let mut by_head: Vec<(TypeHead, Vec<DeclId>, usize)> = Vec::new();
    let mut head_pos: FxHashMap<TypeHead, usize> = FxHashMap::default();
    for (k, imp) in impls.iter().enumerate() {
        match (imp.head, imp.trait_decl.is_none() && imp.trait_ty == u32::MAX) {
            (Some(h), true) => {
                let p = *head_pos.entry(h).or_insert_with(|| {
                    by_head.push((h, Vec::new(), k));
                    by_head.len() - 1
                });
                by_head[p].1.extend_from_slice(&imp.methods);
            }
            _ => {
                let scope = head_text(imp.head, imp);
                report(&imp.methods, &scope, &mut out);
            }
        }
    }
    for (h, ids, k) in &by_head {
        let scope = head_text(Some(*h), &impls[*k]);
        report(ids, &scope, &mut out);
    }
    out
}
