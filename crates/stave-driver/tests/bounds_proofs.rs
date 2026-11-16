//! Release bounds-check elision on real front-end IR: `tests/conformance/run/bounds_proofs.stv`
//! lowered, put through the release passes before code generation (stave-opt, then
//! `stave_ir::bounds::elide`), keeps exactly the checks its look-alike functions need.

use std::path::Path;

use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_ir::bounds::{self, Proof};
use stave_ir::{Module, Op};

fn repo() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// The checked IR of the program after the release passes that run before code generation, and
/// the proofs recorded per function path.
fn released(file: &str) -> (Module, Vec<(String, Proof)>) {
    let mut opts = Options::new(Goal::Check).threads(Some(2));
    opts.lower = true;
    opts.release = true;
    opts.target = Some(stave_target::Triple::parse("x86_64-linux").unwrap());
    let out = pipeline::run(&repo().join(file), &Library::in_dir(repo()), &opts).unwrap();
    assert!(!out.has_errors(), "{file}: {:?}", out.diagnostics);
    let mut m = out.module.expect("checked IR");
    for b in m.funcs.iter_mut().filter_map(|f| f.body.as_mut()) {
        stave_opt::before_elision(b);
    }
    let e = bounds::elide(&mut m);
    stave_ir::verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    bounds::audit(&m).unwrap_or_else(|u| panic!("unguarded after elision: {u:?}"));
    let proofs = e.into_iter().map(|x| (m.funcs[x.func.index()].path.clone(), x.proof)).collect();
    (m, proofs)
}

fn kept(m: &Module, path: &str) -> usize {
    let f = m.funcs.iter().find(|f| f.path == path).unwrap_or_else(|| panic!("no fn {path}"));
    let b = f.body.as_ref().unwrap();
    b.blocks
        .iter()
        .flat_map(|bd| bd.insts.indices())
        .filter(|&i| b.insts.op[i as usize] == Op::BoundsCheck)
        .count()
}

#[test]
fn proofs_and_kept_checks() {
    let (m, proofs) = released("tests/conformance/run/bounds_proofs.stv");
    let of = |path: &str| -> Vec<Proof> { proofs.iter().filter(|p| p.0 == path).map(|p| p.1).collect() };
    for (path, n) in [
        ("main.dot", 0),
        ("main.dot_or_zero", 0),
        ("main.rises", 0),
        ("main.tail_sum", 0),
        ("main.pair_sum", 0),
        ("main.window_sum", 0),
        ("main.chunk_sum", 0),
        ("main.quad_sum", 0),
        ("main.le_only", 1),
        ("main.eq_on_one_path", 1),
        ("main.next_under_len", 1),
        ("main.prev_wrapping", 1),
        ("main.at_offset", 1),
        ("main.chunk_le_sat", 1),
        ("main.chunk_short_test", 1),
        ("main.chunk_wrapping", 1),
        ("main.chunk_other", 1),
        ("main.chunk_shrunk", 1),
        ("main.chunk_moved", 1),
    ] {
        assert_eq!(kept(&m, path), n, "{path}: {:?}", of(path));
    }
    assert!(
        matches!(of("main.dot")[..], [Proof::LoopBound { .. }, Proof::LenEq { .. }]),
        "{:?}",
        of("main.dot")
    );
    assert!(matches!(of("main.dot_or_zero")[..], [Proof::LoopBound { .. }, Proof::LenEq { .. }]));
    assert!(matches!(of("main.rises")[..], [Proof::LoopBound { .. }, Proof::Offset { k: -1, .. }]));
    assert!(matches!(of("main.tail_sum")[..], [Proof::Offset { k: 1, .. }]));
    assert!(matches!(of("main.pair_sum")[..], [Proof::Offset { k: 0, .. }, Proof::Offset { k: 1, .. }]));
    // `i + 4 <= xs.len()` proves the vector's last lane `i + 3`, and every scalar read below it
    assert!(matches!(of("main.chunk_sum")[..], [Proof::Offset { k: 3, .. }]), "{:?}", of("main.chunk_sum"));
    assert!(matches!(of("main.quad_sum")[..], [Proof::Offset { k: 0, .. }, .., Proof::Offset { k: 3, .. }]));
    // `&xs[0..n]` has length `n` and tests `n <= xs.len()`
    assert!(
        matches!(of("main.window_sum")[..], [Proof::Offset { k: 0, .. }, Proof::Offset { k: 0, .. }]),
        "{:?}",
        of("main.window_sum")
    );
}
