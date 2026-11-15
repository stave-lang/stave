//! Modules and imports: module = file, import resolution, acyclic imports (E0102).
//!
//! **Module path** (display path, used in diagnostics, E0102 ordering, and decl keys): in the
//! package being built, the file path under `src/` (`src/a/b.stv` is `a.b`, `src/main.stv` is
//! `main`); in a dependency, the package name followed by that path, with `src/lib.stv` being the
//! package name alone (`prims`, `prims.host`); in a built-in package, the package name followed by
//! the path under its directory (`std.io`).
//!
//! **Import paths** (`import s0.s1...;`, from a module of package P): `s0` is P's
//! own name (the rest is a module path of P; nothing left = P's root module), else the binding
//! name of one of P's dependencies (`std`, `prims`: the rest is a module path of that package),
//! else the path is a module path of P (`import a.b;` = `src/a/b.stv`). If `s0` is both a
//! dependency name and the first segment of a module path of P (`src/x.stv` or `src/x/`), the
//! import is E0107 (ambiguous) and binds an error decl; `import P.x...;` names the module. The
//! import binds its last segment, or the `as` name. No re-exports: a path into module `m` sees
//! `m`'s items only, never its imports, and a module never reaches a submodule (`import geo;`
//! gives no `geo.shapes`).

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::{Diagnostic, Span};

use crate::diag::mk;
use crate::ids::{DeclId, ItemId, ModuleId, PackageId, Sym};
use crate::intern::Strings;

/// A module (one source file).
#[derive(Clone, Debug)]
pub struct Module {
    pub id: ModuleId,
    pub package: PackageId,
    pub file: stave_diag::FileId,
    /// Path relative to its package (`["a", "b"]`, `["main"]`, `["lib"]`).
    pub rel: Vec<Sym>,
    /// Module path segments (see the module docs).
    pub path: Vec<Sym>,
    /// Module path text (`a.b`, `std.io`).
    pub path_text: String,
    /// Its [`crate::DeclKind::Module`] decl.
    pub decl: DeclId,
    /// Imports in source order.
    pub imports: Vec<Import>,
}

/// One import of a module.
#[derive(Clone, Debug)]
pub struct Import {
    pub item: ItemId,
    /// The imported module (`NONE` if it does not exist: E0101).
    pub target: ModuleId,
    pub bound: Sym,
    pub bound_span: Span,
    /// The whole `import ...;`.
    pub span: Span,
    /// Path segment tokens.
    pub segs: Vec<u32>,
    /// Per segment, the module the path up to it names (`NONE`: a package name or a directory).
    pub seg_modules: Vec<ModuleId>,
}

/// Lookup of modules by (package, relative path).
pub(crate) struct ModuleIndex {
    by_rel: FxHashMap<(PackageId, Vec<Sym>), ModuleId>,
    dirs: FxHashSet<(PackageId, Vec<Sym>)>,
}

impl ModuleIndex {
    pub fn new(modules: &[Module]) -> ModuleIndex {
        let mut by_rel = FxHashMap::default();
        let mut dirs = FxHashSet::default();
        for m in modules {
            by_rel.insert((m.package, m.rel.clone()), m.id);
            for k in 1..m.rel.len() {
                dirs.insert((m.package, m.rel[..k].to_vec()));
            }
        }
        ModuleIndex { by_rel, dirs }
    }

    pub fn get(&self, p: PackageId, rel: &[Sym]) -> Option<ModuleId> {
        self.by_rel.get(&(p, rel.to_vec())).copied()
    }

    fn exists_prefix(&self, p: PackageId, rel: &[Sym]) -> bool {
        let k = (p, rel.to_vec());
        self.by_rel.contains_key(&k) || self.dirs.contains(&k)
    }
}

/// What resolving an import path gives.
pub(crate) struct PathResult {
    pub target: ModuleId,
    pub seg_modules: Vec<ModuleId>,
    /// Index of the segment that names nothing (E0101), if any.
    pub failed_at: Option<usize>,
    /// The first segment is both a dependency and a module of the package (E0107): the path
    /// resolves to nothing, and `failed_at` is `None`.
    pub ambiguous: bool,
}

/// Package facts the import resolver needs.
pub(crate) struct PkgView<'a> {
    pub name: Sym,
    pub deps: &'a [(Sym, PackageId)],
    pub root: ModuleId,
}

/// Resolves the import path `segs` (string ids) from package `from`.
pub(crate) fn resolve_path(
    idx: &ModuleIndex,
    pkgs: &[PkgView<'_>],
    from: PackageId,
    segs: &[Sym],
) -> PathResult {
    let mut seg_modules = vec![ModuleId::NONE; segs.len()];
    if segs.is_empty() {
        return PathResult { target: ModuleId::NONE, seg_modules, failed_at: None, ambiguous: false };
    }
    let Some(pv) = pkgs.get(from.idx()) else {
        return PathResult { target: ModuleId::NONE, seg_modules, failed_at: Some(0), ambiguous: false };
    };
    let s0 = segs.first().copied().unwrap_or(Sym::NONE);
    let (pkg, skip) = if s0 == pv.name {
        (from, 1)
    } else if let Some(&(_, d)) = pv.deps.iter().find(|(n, _)| *n == s0) {
        if idx.exists_prefix(from, &segs[..1]) {
            return PathResult { target: ModuleId::NONE, seg_modules, failed_at: None, ambiguous: true };
        }
        (d, 1)
    } else {
        (from, 0)
    };
    if skip == 1 {
        seg_modules[0] = pkgs.get(pkg.idx()).map_or(ModuleId::NONE, |p| p.root);
    }
    let rest = &segs[skip..];
    if rest.is_empty() {
        let target = seg_modules[0];
        let failed_at = if target.is_none() { Some(0) } else { None };
        return PathResult { target, seg_modules, failed_at, ambiguous: false };
    }
    let mut failed_at = None;
    for k in 0..rest.len() {
        let prefix = &rest[..=k];
        if let Some(m) = idx.get(pkg, prefix) {
            seg_modules[skip + k] = m;
        } else if !idx.exists_prefix(pkg, prefix) {
            failed_at = Some(skip + k);
            break;
        }
    }
    let target = if failed_at.is_none() { seg_modules[segs.len() - 1] } else { ModuleId::NONE };
    let failed_at = if failed_at.is_none() && target.is_none() { Some(segs.len() - 1) } else { failed_at };
    PathResult { target, seg_modules, failed_at, ambiguous: false }
}

/// Strongly connected components of the import graph (iterative Tarjan), each as a list of
/// modules; only components that contain a cycle (size > 1, or a self-import).
fn cyclic_sccs(modules: &[Module]) -> Vec<Vec<ModuleId>> {
    let n = modules.len();
    let succ = |m: usize| modules[m].imports.iter().filter(|i| i.target.is_some()).map(|i| i.target.idx());
    let mut index = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut next = 0u32;
    let mut out = Vec::new();
    // Frames: (node, position in its successor list).
    let mut call: Vec<(usize, usize)> = Vec::new();
    for root in 0..n {
        if index[root] != u32::MAX {
            continue;
        }
        call.push((root, 0));
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut pos)) = call.last_mut() {
            let s: Vec<usize> = succ(v).collect();
            if *pos < s.len() {
                let w = s[*pos];
                *pos += 1;
                if index[w] == u32::MAX {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            call.pop();
            if let Some(&(p, _)) = call.last() {
                low[p] = low[p].min(low[v]);
            }
            if low[v] == index[v] {
                let mut comp = Vec::new();
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    comp.push(ModuleId(w as u32));
                    if w == v {
                        break;
                    }
                }
                let self_loop = comp.len() == 1 && succ(v).any(|w| w == v);
                if comp.len() > 1 || self_loop {
                    out.push(comp);
                }
            }
        }
    }
    out
}

/// E0102, one per cyclic component: start at the component's smallest module
/// path, follow each module's first in-component import in source order until a module repeats,
/// rotate the loop to its smallest module path. Primary: the first module's import of the
/// second; related ("import"): the other imports of the loop, in order. Modules must be in
/// canonical order (ids ordered by module path text).
pub(crate) fn cycle_diagnostics(modules: &[Module]) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for comp in cyclic_sccs(modules) {
        let inside: FxHashSet<ModuleId> = comp.iter().copied().collect();
        let smallest = |ms: &[ModuleId]| -> Option<ModuleId> {
            ms.iter().copied().min_by(|a, b| modules[a.idx()].path_text.cmp(&modules[b.idx()].path_text))
        };
        let Some(start) = smallest(&comp) else { continue };
        // Walk: (module, the import taken from it).
        let mut walk: Vec<(ModuleId, usize)> = Vec::new();
        let mut cur = start;
        loop {
            if let Some(p) = walk.iter().position(|(m, _)| *m == cur) {
                walk.drain(..p);
                break;
            }
            let Some(k) = modules[cur.idx()].imports.iter().position(|i| inside.contains(&i.target)) else {
                walk.clear();
                break;
            };
            walk.push((cur, k));
            cur = modules[cur.idx()].imports[k].target;
        }
        let ms: Vec<ModuleId> = walk.iter().map(|w| w.0).collect();
        let Some(first) = smallest(&ms) else { continue };
        let rot = walk.iter().position(|w| w.0 == first).unwrap_or(0);
        walk.rotate_left(rot);
        let edge = |w: &(ModuleId, usize)| modules[w.0.idx()].imports[w.1].span;
        let primary = edge(&walk[0]);
        let related: Vec<(&str, Span)> = walk[1..].iter().map(|w| ("import", edge(w))).collect();
        let path = &modules[walk[0].0.idx()].path_text;
        if let Some(d) = mk("E0102", None, &[("module", path)], primary, &related) {
            out.push(d);
        }
    }
    out
}

/// The module path text of segments.
pub(crate) fn path_text(strings: &Strings, segs: &[Sym]) -> String {
    let mut s = String::new();
    for (k, &x) in segs.iter().enumerate() {
        if k > 0 {
            s.push('.');
        }
        s.push_str(strings.get(x));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use stave_diag::FileId;

    fn module(id: u32, path: &str, imports: &[(u32, u32)]) -> Module {
        Module {
            id: ModuleId(id),
            package: PackageId(0),
            file: FileId(id),
            rel: Vec::new(),
            path: Vec::new(),
            path_text: path.into(),
            decl: DeclId::NONE,
            imports: imports
                .iter()
                .map(|&(t, off)| Import {
                    item: ItemId(0),
                    target: ModuleId(t),
                    bound: Sym::NONE,
                    bound_span: Span::new(FileId(id), off, 1),
                    span: Span::new(FileId(id), off, 1),
                    segs: Vec::new(),
                    seg_modules: Vec::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn cycle_rule() {
        // a -> c, c -> b, b -> a (and c -> d, d -> c): one SCC {a, b, c, d}.
        // Walk from a: a -> c (first in-SCC import), c -> b, b -> a: loop [a, c, b].
        let ms = vec![
            module(0, "a", &[(2, 10)]),
            module(1, "b", &[(0, 20)]),
            module(2, "c", &[(1, 30), (3, 31)]),
            module(3, "d", &[(2, 40)]),
            module(4, "e", &[(4, 50)]),
        ];
        let ds = cycle_diagnostics(&ms);
        assert_eq!(ds.len(), 2);
        let d = ds.iter().find(|d| d.primary().unwrap().offset == 10).unwrap();
        let rel: Vec<u32> = d.related().iter().map(|r| r.span.offset).collect();
        assert_eq!(rel, vec![10, 30, 20], "primary label first, then the imports in cycle order");
        let e = ds.iter().find(|d| d.primary().unwrap().offset == 50).unwrap();
        assert_eq!(e.message(), "import cycle through module `e`");
    }

    #[test]
    fn rotation_to_smallest() {
        // Start "b" is smallest of {b, c, x}? No: SCC {c, b, x}: smallest "b". b -> x? walk.
        // b -> c, c -> x, x -> c: loop [c, x] (b not on it), rotated to start at "c".
        let ms =
            vec![module(0, "b", &[(1, 1)]), module(1, "c", &[(2, 2)]), module(2, "x", &[(1, 3), (0, 4)])];
        let ds = cycle_diagnostics(&ms);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds[0].primary().unwrap().offset, 2);
        assert_eq!(ds[0].message(), "import cycle through module `c`");
    }
}
