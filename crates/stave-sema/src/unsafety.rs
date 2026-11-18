//! The unsafe inventory and the unsafe-operation rule. `stave unsafe-report` prints [`inventory`]; CI
//! diffs it per change.
//!
//! **Sites.** One per obligation string, plus one per call of an `extern "C"` import:
//!
//! | Kind | Source | `fn` | Obligation |
//! |---|---|---|---|
//! | `Block` | `unsafe("..") { }` statement | enclosing fn | its string |
//! | `Expr` | `unsafe("..") e` | enclosing fn | its string |
//! | `Fn` | `unsafe("..") fn` (fn, method, trait method signature) | that fn | the caller's obligation |
//! | `Trait` | `unsafe("..") trait` | the trait's path | the implementer's obligation |
//! | `Impl` | `unsafe("..") impl` | the impl's path (`<module>.<self-type head>`, as method paths) | the implementer's |
//! | `Extern` | `unsafe("..") extern "C" { }` (an import block) | the module's path | the declarer's: every import matches its C definition |
//! | `ExternCall` | each use of an `extern "C"` import | enclosing fn | the innermost enclosing `Block`/`Expr` site's string |
//!
//! `fn` is the checked-IR path (contracts/checked-ir.md `Path`; methods `<module>.<Type>.<name>`).
//! The span is the `unsafe` keyword (`ExternCall`: the callee's name token). Sites are in canonical
//! order: (file F, offset), then kind in the order above. Every package of the program is listed
//! (the root package, its dependencies, `std`, `runtime`), dropped items never (they have no
//! decls). An `ExternCall` outside every `unsafe(..)` is E0803 (the inventory is only meaningful for
//! a program without errors) and is not listed.
//!
//! **E0013** (missing obligation) is a syntax error found by the parser (contracts/parse-errors.md
//! row 1): the item is dropped, so every listed site has its string. An empty string
//! `unsafe("")` is E0013 too (parse-errors.md "Empty obligation"): its item is dropped,
//! so no listed obligation is empty.
//!
//! **E0803** (unsafe operation outside `unsafe(..)`) needs the operation's type, so the type
//! checker detects the operations, each through [`e0803`] when not inside an `unsafe(..)` block or
//! expression of the same body (an `unsafe fn` body is not an unsafe context): `*p` on a raw
//! pointer (expr.rs `unary`), the raw-pointer methods `read`/`write`/... (builtin method sigs with
//! `is_unsafe`), a call of an `unsafe` fn or method or of an `extern` import (`call`,
//! `method_checks`), and any use of a `static mut` (path resolution: read, write, `&`).

use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind, Term};

use crate::Program;
use crate::decls::{DeclFlags, DeclKind};
use crate::ids::{DeclId, ItemId};
use crate::resolve::Res;
use crate::typeck::Typeck;

/// What an inventory site is in report order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnsafeKind {
    Block,
    Expr,
    Fn,
    Trait,
    Impl,
    Extern,
    ExternCall,
}

impl UnsafeKind {
    /// The report name (`--json` `kind`).
    pub fn name(self) -> &'static str {
        match self {
            UnsafeKind::Block => "block",
            UnsafeKind::Expr => "expr",
            UnsafeKind::Fn => "fn",
            UnsafeKind::Trait => "trait",
            UnsafeKind::Impl => "impl",
            UnsafeKind::Extern => "extern",
            UnsafeKind::ExternCall => "extern-call",
        }
    }
}

/// One inventory entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsafeSite {
    pub span: Span,
    /// The report path: the diagnostics path of the file when it is inside the root package's
    /// directory, else `<package name>/<path in the package>` (e.g. `runtime/alloc.stv`), so the
    /// report never depends on where the library is installed.
    pub file: String,
    /// Checked-IR path of the enclosing fn (`Trait`/`Impl`: of the trait / impl).
    pub func: String,
    pub kind: UnsafeKind,
    /// The obligation text (string literal contents, escapes decoded; invalid UTF-8 replaced).
    pub obligation: String,
}

/// E0803 at `sp`: an unsafe operation outside `unsafe(..)` (the one constructor of the code).
pub(crate) fn e0803(sp: Span) -> Option<Diagnostic> {
    crate::diag::mk("E0803", None, &[], sp, &[])
}

/// The unsafe inventory of the whole program, in canonical (file, offset, kind) order.
pub fn inventory(prog: &Program, tc: &Typeck) -> Vec<UnsafeSite> {
    // Fn-like decls per item: (node, decl).
    let mut by_item: Vec<Vec<(u32, DeclId)>> = vec![Vec::new(); prog.items.len()];
    for (k, d) in prog.decls.iter().enumerate() {
        if d.node == u32::MAX
            || !matches!(
                d.kind,
                DeclKind::Fn | DeclKind::Method | DeclKind::TraitMethod | DeclKind::Trait | DeclKind::Impl
            )
        {
            continue;
        }
        if let Some(v) = by_item.get_mut(d.item.0 as usize) {
            v.push((d.node, DeclId(k as u32)));
        }
    }
    let mut out = Vec::new();
    for (i, it) in prog.items.iter().enumerate() {
        if !it.ok {
            continue;
        }
        item_sites(prog, tc, ItemId(i as u32), &by_item[i], &mut out);
    }
    out.sort_by_key(|a| (a.span.file, a.span.offset, a.kind));
    out
}

fn item_sites(prog: &Program, tc: &Typeck, item: ItemId, decls: &[(u32, DeclId)], out: &mut Vec<UnsafeSite>) {
    let Some(it) = prog.item(item) else { return };
    let Some(file) = prog.file(it.file) else { return };
    let ast = prog.item_ast(item);
    let toks = &file.parsed.tokens;
    let report_path = if file.display_path.starts_with('/') {
        let pkg = prog.packages.get(file.package.0 as usize).map_or("", |p| p.name.as_str());
        format!("{pkg}/{}", file.rel_path)
    } else {
        file.display_path.clone()
    };
    let parents = ast.parents();
    let decl_at = |n: u32| decls.iter().find(|(dn, _)| *dn == n).map(|&(_, d)| d);
    // The innermost fn-like decl whose node contains `n`.
    let enclosing = |n: u32| -> Option<DeclId> {
        let mut p = n;
        while p != u32::MAX {
            if let Some(d) = decl_at(p)
                && prog.decl(d).is_some_and(|x| {
                    matches!(x.kind, DeclKind::Fn | DeclKind::Method | DeclKind::TraitMethod)
                })
            {
                return Some(d);
            }
            p = parents.get(p as usize).copied().unwrap_or(u32::MAX);
        }
        None
    };
    let path_of = |d: Option<DeclId>| d.map_or(String::new(), |d| decl_path(prog, tc, d));
    let module_path = prog.module(it.module).map_or(String::new(), |m| m.path_text.clone());
    // Block/Expr regions: (first node, last node, obligation).
    let mut regions: Vec<(u32, u32, String)> = Vec::new();
    for n in 0..ast.len() as u32 {
        if ast.kind(n) != Some(NodeKind::Unsafe) {
            continue;
        }
        let Some(node) = ast.get(n) else { continue };
        let kw = node.a as usize;
        let span = crate::scan::tok_span(file.id, &file.bytes, toks, kw);
        let lit = kw + 2;
        let obligation = if crate::scan::kind(toks, lit) == Term::StrLit as u8 {
            let text = stave_syntax::token_text(&file.bytes, toks, lit);
            let bytes =
                std::str::from_utf8(text).ok().and_then(crate::literals::parse_str).unwrap_or_default();
            String::from_utf8_lossy(&bytes).into_owned()
        } else {
            String::new()
        };
        let parent = parents.get(n as usize).copied().unwrap_or(u32::MAX);
        // The next sibling of `n` under `parent`.
        let next = (parent != u32::MAX)
            .then(|| ast.children(parent).into_iter().skip_while(|&c| c != n).nth(1))
            .flatten();
        let (kind, func, region) = match ast.kind(parent) {
            Some(NodeKind::Unary) => (UnsafeKind::Expr, path_of(enclosing(parent)), Some(parent)),
            Some(NodeKind::TraitFn) => (UnsafeKind::Fn, path_of(decl_at(parent)), None),
            Some(NodeKind::ItemBody | NodeKind::ImplDecl | NodeKind::TraitDecl) => {
                let target = next.unwrap_or(u32::MAX);
                let kind = match ast.kind(target) {
                    Some(NodeKind::TraitDecl) => UnsafeKind::Trait,
                    Some(NodeKind::ImplDecl) => UnsafeKind::Impl,
                    Some(NodeKind::ExternItem) if decl_at(target).is_none() => UnsafeKind::Extern,
                    _ => UnsafeKind::Fn,
                };
                let func = match kind {
                    UnsafeKind::Extern => module_path.clone(),
                    _ => path_of(decl_at(target)),
                };
                (kind, func, None)
            }
            // Statement form `Unsafe Block`: the block is the next sibling.
            _ => (UnsafeKind::Block, path_of(enclosing(n)), next),
        };
        if let Some(r) = region {
            let lo = ast.descendants(r).start;
            regions.push((lo, r, obligation.clone()));
        }
        out.push(UnsafeSite { span, file: report_path.clone(), func, kind, obligation });
    }
    // Calls of extern imports: the innermost region holding the use.
    let Some(res) = prog.item_res(item) else { return };
    for u in &res.uses {
        let Res::Decl(d) = u.res else { continue };
        if !prog.decl(d).is_some_and(|x| x.flags.has(DeclFlags::EXTERN_IMPORT)) {
            continue;
        }
        let inner = regions
            .iter()
            .filter(|(lo, hi, _)| *lo <= u.node && u.node <= *hi)
            .min_by_key(|(lo, hi, _)| hi - lo);
        let Some((_, _, obligation)) = inner else { continue };
        let span = crate::scan::tok_span(file.id, &file.bytes, toks, u.token as usize);
        out.push(UnsafeSite {
            span,
            file: report_path.clone(),
            func: path_of(enclosing(u.node)),
            kind: UnsafeKind::ExternCall,
            obligation: obligation.clone(),
        });
    }
}

/// The report path of a fn-like, trait, or impl decl.
fn decl_path(prog: &Program, tc: &Typeck, d: DeclId) -> String {
    let Some(decl) = prog.decl(d) else { return String::new() };
    if decl.kind != DeclKind::Impl {
        return crate::lower::fn_path(prog, &tc.sigs, d);
    }
    let module = prog.module(decl.module).map_or(String::new(), |m| m.path_text.clone());
    let join = |s: String| if module.is_empty() { s } else { format!("{module}.{s}") };
    match tc.sigs.impl_def(d).and_then(|i| i.head) {
        Some(crate::resolve::TypeHead::Adt(a)) => prog.decl_path(a),
        Some(crate::resolve::TypeHead::Prim(p)) => join(p.text()),
        None => join("impl".to_string()),
    }
}
