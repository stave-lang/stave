//! Image-level rules both backends follow when they lay a module out: the text order of the
//! functions, the symbol of each function and `static mut`, and how a site's file path splits
//! into a line-table directory and name.

use rustc_hash::FxHashMap;

use crate::ir::{FuncId, Module};

/// The image entry symbol (the per-arch stub that calls `runtime.start.entry`).
pub const START: &str = "_start";

/// Symbols the compiler defines itself (stubs, tables, helpers) all start with this prefix.
pub const RESERVED_PREFIX: &str = "__stave";

/// Whether a Stave path would clash with a compiler-defined symbol if used verbatim.
pub fn is_reserved(path: &str) -> bool {
    path.starts_with(RESERVED_PREFIX) || path == START
}

/// The functions with a body in text order: by instance, then by id.
pub fn text_order(m: &Module) -> Vec<FuncId> {
    let mut order: Vec<FuncId> =
        (0..m.funcs.len()).filter(|&i| m.funcs[i].body.is_some()).map(|i| FuncId(i as u32)).collect();
    order.sort_by(|x, y| m.funcs[x.index()].instance.cmp(&m.funcs[y.index()].instance).then(x.cmp(y)));
    order
}

/// Per `FuncId`: the symbol name of each function in `fns` (`None` for the others). An
/// `extern "C"` function is named by its [`crate::Func::symbol`] (a C name has no `.`, so it never
/// meets a path); any other function by its path, or `path@instance` when another function of
/// `fns` has the same path or the path is reserved ([`is_reserved`]).
pub fn func_symbols(m: &Module, fns: &[FuncId]) -> Vec<Option<String>> {
    let mut count: FxHashMap<&str, u32> = FxHashMap::default();
    for f in fns {
        if let Some(func) = m.funcs.get(f.index()) {
            *count.entry(func.path.as_str()).or_default() += 1;
        }
    }
    let mut out = vec![None; m.funcs.len()];
    for f in fns {
        let Some(func) = m.funcs.get(f.index()) else { continue };
        if let Some(sym) = &func.symbol {
            out[f.index()] = Some(sym.clone());
            continue;
        }
        let shared = count.get(func.path.as_str()).copied().unwrap_or(0) > 1;
        out[f.index()] = Some(if shared || is_reserved(&func.path) {
            format!("{}@{}", func.path, func.instance)
        } else {
            func.path.clone()
        });
    }
    out
}

/// The symbol of `static mut` number `i`: its path, or `__stave_static.<i>` when the path is
/// empty, reserved, or already `taken` by another symbol.
pub fn static_symbol(m: &Module, i: usize, taken: impl Fn(&str) -> bool) -> String {
    match m.statics.get(i) {
        Some(st) if !st.path.is_empty() && !is_reserved(&st.path) && !taken(&st.path) => st.path.clone(),
        _ => format!("{RESERVED_PREFIX}_static.{i}"),
    }
}

/// A site's file path as (directory, file name), split at the last `/`; a path without one is
/// in the compilation directory `.`.
pub fn split_file(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(i) => (&path[..i], &path[i + 1..]),
        None => (".", path),
    }
}
