//! The shape of checked IR, for comparing two lowerings of one program (debug and release
//! builds number instances, types, and strings differently, so their text differs).

use std::collections::BTreeMap;

use stave_ir::{Body, Module, Op, Term};

/// One function's shape: per block its op names and its terminator, in order. Two lowerings of
/// a function that agree here run the same ops and keep the same checks.
pub fn shape(b: &Body) -> String {
    let mut s = String::new();
    for bd in &b.blocks {
        for i in bd.insts.indices() {
            s.push_str(b.insts.op[i as usize].name().unwrap_or("intrinsic"));
            s.push(' ');
        }
        s.push_str(match bd.term {
            Term::Br(_) => "br",
            Term::Cbr { .. } => "cbr",
            Term::Switch { .. } => "switch",
            Term::Ret(_) => "ret",
            Term::Unreachable => "unreachable",
        });
        s.push('\n');
    }
    s
}

/// The shapes of the module's functions by path (the instances of one path sorted).
pub fn shapes(m: &Module) -> BTreeMap<String, Vec<String>> {
    let mut by_path: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in &m.funcs {
        if let Some(b) = &f.body {
            by_path.entry(f.path.clone()).or_default().push(shape(b));
        }
    }
    for v in by_path.values_mut() {
        v.sort();
    }
    by_path
}

/// The `check.bounds` left in the module, per function path (paths without one are absent).
pub fn kept_checks(m: &Module) -> BTreeMap<String, usize> {
    let mut by_path = BTreeMap::new();
    for f in &m.funcs {
        let n = f.body.as_ref().map_or(0, |b| {
            b.blocks
                .iter()
                .flat_map(|bd| bd.insts.indices())
                .filter(|&i| b.insts.op[i as usize] == Op::BoundsCheck)
                .count()
        });
        if n > 0 {
            *by_path.entry(f.path.clone()).or_default() += n;
        }
    }
    by_path
}

/// What `release` has that `debug` lacks: a function shape of a path, as `path` strings. Empty
/// when every function the release build lowers has the shape the debug build gives it.
pub fn not_in(release: &Module, debug: &Module) -> Vec<String> {
    let dbg = shapes(debug);
    let mut missing = Vec::new();
    for (path, want) in shapes(release) {
        let mut have = dbg.get(&path).cloned().unwrap_or_default();
        for w in want {
            match have.iter().position(|h| *h == w) {
                Some(k) => {
                    have.swap_remove(k);
                }
                None => missing.push(path.clone()),
            }
        }
    }
    missing
}
