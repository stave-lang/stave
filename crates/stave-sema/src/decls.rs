//! Declarations: the decl table and its per-file collection.
//!
//! One [`Decl`] per module, item (fn, struct, enum, trait, const, static, type alias, impl), and member
//! (field, variant, method, trait method). A name declared only by a dropped item is an
//! [`DeclKind::Error`] decl: uses resolve to it silently (no cascades). Params, `let`s, pattern
//! bindings, and generic params are not decls: they are item-local bindings
//! ([`crate::BindingRef`], [`crate::GenericRef`]).
//!
//! **Canonical decl id**: the rank of
//! the key `(module path, steps)`, compared lexicographically, where the module path is the
//! decl's module path as canonical string ids (a module decl uses its parent path) and `steps`
//! is the chain of `(name, kind, tiebreak)` from the top-level decl down to the decl (a member's
//! key extends its parent's). `name` compares by canonical string id (an impl has no name and
//! sorts after every name), `kind` in [`DeclKind`] order, `tiebreak` is the package id for a
//! module, the item id for a top-level decl, and the declaration index among its siblings for a
//! member, then the name token's byte offset (two fns of one `extern` block). A prefix sorts
//! first, so members follow their parent.

use stave_diag::{Diagnostic, Span};
use stave_syntax::{ItemAst, ItemKind, NodeKind, Term};

use crate::File;
use crate::diag::mk;
use crate::ids::{DeclId, ItemId, ModuleId, Sym};
use crate::intern::Strings;
use crate::scan::{self, is, kind, tok_span};

/// What a declaration is. The order is the canonical key order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeclKind {
    Module,
    /// Fn items, including `test fn`, `kernel fn`, `dispatch fn`, extern imports and exports.
    Fn,
    Struct,
    Enum,
    Trait,
    Const,
    /// `static NAME: T = e;` or `static mut NAME: T = e;` ([`DeclFlags::MUT`]: every use is unsafe).
    Static,
    TypeAlias,
    Impl,
    /// Struct field, or field of a struct-form variant (parent: the variant).
    Field,
    Variant,
    /// Method of an impl (parent: the impl).
    Method,
    /// Signature in a trait (parent: the trait).
    TraitMethod,
    /// Declared only by a dropped item, or bound by an import that failed (no cascades).
    Error,
}

/// Declaration flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DeclFlags(pub u16);

impl DeclFlags {
    pub const PUB: u16 = 1;
    pub const UNSAFE: u16 = 1 << 1;
    /// `view struct` / `view enum`.
    pub const VIEW: u16 = 1 << 2;
    pub const TEST: u16 = 1 << 3;
    pub const KERNEL: u16 = 1 << 4;
    pub const DISPATCH: u16 = 1 << 5;
    /// A fn of an `extern "C" { }` block.
    pub const EXTERN_IMPORT: u16 = 1 << 6;
    /// `extern "C" fn f() { }`.
    pub const EXTERN_EXPORT: u16 = 1 << 7;
    /// An impl with `for` (trait impl); also set on its methods.
    pub const TRAIT_IMPL: u16 = 1 << 8;
    /// Variant with a `( .. )` payload.
    pub const TUPLE: u16 = 1 << 9;
    /// Variant with a `{ .. }` payload.
    pub const RECORD: u16 = 1 << 10;
    /// Method or trait method with a `self` parameter.
    pub const HAS_SELF: u16 = 1 << 11;
    /// `static mut` (a `static` without it is immutable).
    pub const MUT: u16 = 1 << 12;
    /// `rec fn`: the fn may be part of a call cycle.
    pub const REC: u16 = 1 << 13;

    pub fn has(self, f: u16) -> bool {
        self.0 & f != 0
    }
}

/// One declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decl {
    pub kind: DeclKind,
    pub name: Sym,
    /// The module declaring it (a module decl: the module itself).
    pub module: ModuleId,
    pub parent: DeclId,
    /// The item holding its syntax (`NONE` for modules and error decls of failed imports).
    pub item: ItemId,
    /// Its node in the item's AST: `FnDecl`, `KernelDecl`, `DispatchDecl`, `ExternFn`,
    /// `ExternItem` (export), `AdtDecl`, `TraitDecl`, `TraitFn`, `ImplDecl`, `ConstDecl`,
    /// `StaticDecl`, `TypeAlias`, `Field`, `Variant` (`u32::MAX` if none).
    pub node: u32,
    /// The name token's span (an impl: the `impl` keyword; a module: offset 0 of its file).
    pub name_span: Span,
    pub flags: DeclFlags,
    /// Declaration order among its siblings (field, variant, and method index).
    pub index: u32,
}

impl Decl {
    pub fn is_pub(&self) -> bool {
        self.flags.has(DeclFlags::PUB)
    }
}

/// An impl block found during collection.
#[derive(Clone, Debug)]
pub(crate) struct RawImpl {
    /// Local decl index of the impl.
    pub decl: u32,
    pub item: ItemId,
    /// `Type` node of the self type.
    pub self_ty: u32,
    /// `Type` node of the trait (`u32::MAX` for an inherent impl).
    pub trait_ty: u32,
    /// Local decl indices of its methods.
    pub methods: Vec<u32>,
}

/// An import found during collection.
#[derive(Clone, Debug)]
pub(crate) struct RawImport {
    pub item: ItemId,
    /// Token indices of the path segments.
    pub segs: Vec<u32>,
    /// Token of the bound name (last segment or the `as` name).
    pub bound_tok: u32,
    /// The import item (`import` .. `;`).
    pub span: Span,
}

/// What one file declares.
#[derive(Debug, Default)]
pub(crate) struct FileDecls {
    /// Parents are local indices (`DeclId(local)`) or `NONE`.
    pub decls: Vec<Decl>,
    /// (local item index, local decl index) of each item's main decl.
    pub item_decl: Vec<(u32, u32)>,
    pub impls: Vec<RawImpl>,
    pub imports: Vec<RawImport>,
    /// Names declared by dropped items: (name, name token span).
    pub dropped: Vec<(Sym, Span)>,
    /// Identifiers of dropped `impl` headers.
    pub poisoned: Vec<Sym>,
    /// `allocator` items: (item, first path token, path token count).
    pub allocators: Vec<(ItemId, u32, u32)>,
    pub diags: Vec<Diagnostic>,
}

struct Collector<'a> {
    file: &'a File,
    ast: ItemAst<'a>,
    item: ItemId,
    item_local: u32,
    module: ModuleId,
    module_path: &'a str,
    strings: &'a Strings,
    out: &'a mut FileDecls,
}

const NO_NODE: u32 = u32::MAX;

impl Collector<'_> {
    fn sym(&self, tok: u32) -> Sym {
        self.file.syms.get(tok as usize).copied().unwrap_or(Sym::NONE)
    }

    fn span(&self, tok: u32) -> Span {
        tok_span(self.file.id, &self.file.bytes, &self.file.parsed.tokens, tok as usize)
    }

    fn a(&self, n: u32) -> u32 {
        self.ast.get(n).map_or(0, |x| x.a)
    }

    fn toks(&self) -> &[stave_syntax::Token] {
        &self.file.parsed.tokens
    }

    /// The name token of the `FnDecl` whose first token is `a`: `rec`? `fn` Ident.
    fn fn_decl_name(&self, a: u32) -> u32 {
        if is(self.toks(), a as usize, Term::KwRec) { a + 2 } else { a + 1 }
    }

    /// [`DeclFlags::REC`] for the `FnDecl` whose first token is `a`.
    fn rec_flag(&self, a: u32) -> u16 {
        if is(self.toks(), a as usize, Term::KwRec) { DeclFlags::REC } else { 0 }
    }

    /// The first identifier at or after token `from` within the next few tokens (decl names
    /// follow their keyword, possibly after `view`/`unsafe(..)` prefixes).
    fn name_after(&self, from: u32, kw: Term) -> u32 {
        let toks = self.toks();
        let mut i = from as usize;
        let lim = (i + 8).min(toks.len());
        while i < lim {
            if is(toks, i, kw) {
                return if kind(toks, i + 1) == scan::T_IDENT { (i + 1) as u32 } else { u32::MAX };
            }
            if is(toks, i, Term::KwUnsafe) {
                // skip `unsafe ( StrLit )`
                i += 4;
                continue;
            }
            i += 1;
        }
        u32::MAX
    }

    fn push(&mut self, kind: DeclKind, name_tok: u32, node: u32, parent: u32, flags: u16, index: u32) -> u32 {
        let name = if name_tok == u32::MAX { Sym::NONE } else { self.sym(name_tok) };
        let name_span = if name_tok == u32::MAX { self.span(self.a(node)) } else { self.span(name_tok) };
        let id = self.out.decls.len() as u32;
        self.out.decls.push(Decl {
            kind,
            name,
            module: self.module,
            parent: if parent == u32::MAX { DeclId::NONE } else { DeclId(parent) },
            item: self.item,
            node,
            name_span,
            flags: DeclFlags(flags),
            index,
        });
        id
    }

    fn main(&mut self, d: u32) {
        self.out.item_decl.push((self.item_local, d));
    }

    /// E0105 for later duplicates among the members `ids` (in declaration order).
    fn dup_members(&mut self, ids: &[u32], scope: &str) {
        for (k, &d) in ids.iter().enumerate() {
            let name = self.out.decls[d as usize].name;
            if name.is_none() {
                continue;
            }
            if let Some(&first) = ids[..k].iter().find(|&&e| self.out.decls[e as usize].name == name) {
                let primary = self.out.decls[d as usize].name_span;
                let related = self.out.decls[first as usize].name_span;
                let n = self.strings.get(name);
                if let Some(dg) =
                    mk("E0105", None, &[("name", n), ("scope", scope)], primary, &[("first", related)])
                {
                    self.out.diags.push(dg);
                }
            }
        }
    }

    fn scoped(&self, name: Sym) -> String {
        let n = self.strings.get(name);
        if self.module_path.is_empty() { n.to_string() } else { format!("{}.{n}", self.module_path) }
    }

    fn has_self_param(&self, params: Option<u32>) -> bool {
        let Some(p) = params else { return false };
        self.ast.children_rev(p).any(|c| self.ast.child(c, NodeKind::SelfParam).is_some())
    }

    fn fields(&mut self, fields_node: u32, parent: u32, scope: &str) {
        let mut ids = Vec::new();
        for (k, f) in self.ast.children(fields_node).into_iter().enumerate() {
            if self.ast.kind(f) != Some(NodeKind::Field) {
                continue;
            }
            let a = self.a(f);
            let (flags, name_tok) =
                if is(self.toks(), a as usize, Term::KwPub) { (DeclFlags::PUB, a + 1) } else { (0, a) };
            ids.push(self.push(DeclKind::Field, name_tok, f, parent, flags, k as u32));
        }
        self.dup_members(&ids, scope);
    }

    fn adt(&mut self, n: u32, pub_flag: u16) {
        let a = self.a(n);
        let view = if is(self.toks(), a as usize, Term::KwView) { DeclFlags::VIEW } else { 0 };
        let kw = if view != 0 { a + 1 } else { a };
        let dk = if is(self.toks(), kw as usize, Term::KwEnum) { DeclKind::Enum } else { DeclKind::Struct };
        let name_tok = kw + 1;
        let d = self.push(dk, name_tok, n, u32::MAX, pub_flag | view, 0);
        self.main(d);
        let scope = self.scoped(self.out.decls[d as usize].name);
        if dk == DeclKind::Struct {
            if let Some(f) = self.ast.child(n, NodeKind::Fields) {
                self.fields(f, d, &scope);
            }
            return;
        }
        let mut vids = Vec::new();
        let mut k = 0u32;
        for v in self.ast.children(n) {
            if self.ast.kind(v) != Some(NodeKind::Variant) {
                continue;
            }
            let vt = self.a(v);
            let next = kind(self.toks(), vt as usize + 1);
            let shape = if next == Term::LParen as u8 {
                DeclFlags::TUPLE
            } else if next == Term::LBrace as u8 {
                DeclFlags::RECORD
            } else {
                0
            };
            let vd = self.push(DeclKind::Variant, vt, v, d, shape | DeclFlags::PUB, k);
            k += 1;
            vids.push(vd);
            if let Some(f) = self.ast.child(v, NodeKind::Fields) {
                let vscope = format!("{scope}.{}", self.strings.get(self.out.decls[vd as usize].name));
                self.fields(f, vd, &vscope);
            }
        }
        self.dup_members(&vids, &scope);
    }

    fn fn_flags(&self, n: u32) -> u16 {
        let params = self.ast.child(n, NodeKind::Params);
        if self.has_self_param(params) { DeclFlags::HAS_SELF } else { 0 }
    }

    fn item_body(&mut self, body: u32, pub_flag: u16, test: bool) {
        let kids = self.ast.children(body);
        let mut unsafe_flag = 0;
        for &c in &kids {
            let a = self.a(c);
            match self.ast.kind(c) {
                Some(NodeKind::Unsafe) => unsafe_flag = DeclFlags::UNSAFE,
                Some(NodeKind::FnDecl) => {
                    let t = if test { DeclFlags::TEST } else { 0 };
                    let d = self.push(
                        DeclKind::Fn,
                        self.fn_decl_name(a),
                        c,
                        u32::MAX,
                        pub_flag | unsafe_flag | t | self.rec_flag(a),
                        0,
                    );
                    self.main(d);
                }
                Some(NodeKind::KernelDecl) => {
                    let d = self.push(DeclKind::Fn, a + 2, c, u32::MAX, pub_flag | DeclFlags::KERNEL, 0);
                    self.main(d);
                }
                Some(NodeKind::DispatchDecl) => {
                    let d = self.push(DeclKind::Fn, a + 2, c, u32::MAX, pub_flag | DeclFlags::DISPATCH, 0);
                    self.main(d);
                }
                Some(NodeKind::ConstDecl) => {
                    let d = self.push(DeclKind::Const, a + 1, c, u32::MAX, pub_flag, 0);
                    self.main(d);
                }
                Some(NodeKind::StaticDecl) => {
                    let (name, m) = if is(self.toks(), a as usize + 1, Term::KwMut) {
                        (a + 2, DeclFlags::MUT)
                    } else {
                        (a + 1, 0)
                    };
                    let d = self.push(DeclKind::Static, name, c, u32::MAX, pub_flag | m, 0);
                    self.main(d);
                }
                Some(NodeKind::TypeAlias) => {
                    let d = self.push(DeclKind::TypeAlias, a + 1, c, u32::MAX, pub_flag, 0);
                    self.main(d);
                }
                Some(NodeKind::AdtDecl) => self.adt(c, pub_flag),
                Some(NodeKind::TraitDecl) => {
                    let d = self.push(DeclKind::Trait, a + 1, c, u32::MAX, pub_flag | unsafe_flag, 0);
                    self.main(d);
                    let scope = self.scoped(self.out.decls[d as usize].name);
                    let mut ids = Vec::new();
                    let mut k = 0;
                    for f in self.ast.children(c) {
                        if self.ast.kind(f) != Some(NodeKind::TraitFn) {
                            continue;
                        }
                        let fa = self.a(f);
                        let name_tok = self.name_after(fa, Term::KwFn);
                        let us =
                            if self.ast.child(f, NodeKind::Unsafe).is_some() { DeclFlags::UNSAFE } else { 0 };
                        let fl = self.fn_flags(f) | us | DeclFlags::PUB;
                        ids.push(self.push(DeclKind::TraitMethod, name_tok, f, d, fl, k));
                        k += 1;
                    }
                    self.dup_members(&ids, &scope);
                }
                Some(NodeKind::ImplDecl) => self.impl_decl(c, unsafe_flag),
                Some(NodeKind::ExternItem) => {
                    if self.ast.child(c, NodeKind::Block).is_some() {
                        let d = self.push(
                            DeclKind::Fn,
                            a + 3,
                            c,
                            u32::MAX,
                            pub_flag | DeclFlags::EXTERN_EXPORT,
                            0,
                        );
                        self.main(d);
                    } else {
                        for f in self.ast.children(c) {
                            if self.ast.kind(f) == Some(NodeKind::ExternFn) {
                                let fa = self.a(f);
                                let fl = pub_flag | DeclFlags::EXTERN_IMPORT | DeclFlags::UNSAFE;
                                self.push(DeclKind::Fn, fa + 1, f, u32::MAX, fl, 0);
                            }
                        }
                    }
                }
                Some(NodeKind::AllocatorDecl) => {
                    let mut chain = Vec::new();
                    scan::ident_chain(self.toks(), a as usize + 1, &mut chain);
                    if let Some(&f) = chain.first() {
                        self.out.allocators.push((self.item, f, chain.len() as u32));
                    }
                }
                _ => {}
            }
        }
    }

    fn impl_decl(&mut self, n: u32, unsafe_flag: u16) {
        let kids = self.ast.children(n);
        let types: Vec<u32> =
            kids.iter().copied().filter(|&c| self.ast.kind(c) == Some(NodeKind::Type)).collect();
        let (self_ty, trait_ty) = match types.as_slice() {
            [s] => (*s, NO_NODE),
            [t, s, ..] => (*s, *t),
            [] => (NO_NODE, NO_NODE),
        };
        let trait_flag = if trait_ty != NO_NODE { DeclFlags::TRAIT_IMPL } else { 0 };
        let d = self.push(DeclKind::Impl, u32::MAX, n, u32::MAX, unsafe_flag | trait_flag, 0);
        self.main(d);
        let mut methods = Vec::new();
        let mut prev_unsafe: Option<u32> = None;
        let mut k = 0;
        for &c in &kids {
            match self.ast.kind(c) {
                Some(NodeKind::Unsafe) => prev_unsafe = Some(c),
                Some(NodeKind::FnDecl) => {
                    let first = prev_unsafe.take().unwrap_or(c);
                    let fa = self.a(first);
                    let mut fl = self.fn_flags(c) | trait_flag;
                    if first != c {
                        fl |= DeclFlags::UNSAFE;
                    }
                    if fa > 0 && is(self.toks(), fa as usize - 1, Term::KwPub) {
                        fl |= DeclFlags::PUB;
                    }
                    let a = self.a(c);
                    fl |= self.rec_flag(a);
                    methods.push(self.push(DeclKind::Method, self.fn_decl_name(a), c, d, fl, k));
                    k += 1;
                }
                _ => prev_unsafe = None,
            }
        }
        self.out.impls.push(RawImpl { decl: d, item: self.item, self_ty, trait_ty, methods });
    }

    fn import(&mut self) {
        let toks = &self.file.parsed.tokens;
        let a = self.ast.root().and_then(|r| self.ast.get(r)).map_or(0, |x| x.a) as usize;
        let mut segs = Vec::new();
        scan::ident_chain(toks, a + 1, &mut segs);
        let after = segs.last().map_or(a + 1, |&l| l as usize + 1);
        let bound_tok = if is(toks, after, Term::KwAs) && kind(toks, after + 1) == scan::T_IDENT {
            (after + 1) as u32
        } else {
            segs.last().copied().unwrap_or(u32::MAX)
        };
        let it = &self.file.parsed.items[self.item_local as usize];
        let span = scan::range_span(
            self.file.id,
            &self.file.bytes,
            toks,
            it.start as usize,
            (it.end as usize).saturating_sub(1),
        );
        if bound_tok != u32::MAX && !segs.is_empty() {
            self.out.imports.push(RawImport { item: self.item, segs, bound_tok, span });
        }
    }
}

/// Collects the declarations of `file` (module `module`, items from `first_item`).
pub(crate) fn collect_file(file: &File, module: ModuleId, module_path: &str, strings: &Strings) -> FileDecls {
    let mut out = FileDecls::default();
    for (k, it) in file.parsed.items.iter().enumerate() {
        let item = ItemId(file.first_item + k as u32);
        if !it.ok {
            if it.kind == ItemKind::PreItem {
                continue;
            }
            let d = scan::dropped_names(&file.parsed.tokens, it.start as usize, it.end as usize);
            for t in d.names {
                let s = file.syms.get(t as usize).copied().unwrap_or(Sym::NONE);
                if s.is_some() {
                    out.dropped.push((s, tok_span(file.id, &file.bytes, &file.parsed.tokens, t as usize)));
                }
            }
            for t in d.impl_header_idents {
                if let Some(&s) = file.syms.get(t as usize)
                    && s.is_some()
                {
                    out.poisoned.push(s);
                }
            }
            continue;
        }
        let ast = file.parsed.item_ast(k);
        let Some(root) = ast.root() else { continue };
        let mut c =
            Collector { file, ast, item, item_local: k as u32, module, module_path, strings, out: &mut out };
        match ast.kind(root) {
            Some(NodeKind::Import) => c.import(),
            Some(NodeKind::Item) => {
                let a = c.a(root) as usize;
                let toks = &file.parsed.tokens;
                let pub_flag = if is(toks, a, Term::KwPub) { DeclFlags::PUB } else { 0 };
                let test = is(toks, a, Term::KwTest);
                for ch in ast.children(root) {
                    match ast.kind(ch) {
                        Some(NodeKind::ItemBody) => c.item_body(ch, pub_flag, false),
                        Some(NodeKind::FnDecl) if test => {
                            let fa = c.a(ch);
                            let fl = DeclFlags::TEST | c.rec_flag(fa);
                            let d = c.push(DeclKind::Fn, c.fn_decl_name(fa), ch, u32::MAX, fl, 0);
                            c.main(d);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    out
}
