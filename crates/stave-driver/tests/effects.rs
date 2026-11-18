//! The effect pass in the driver: the checked IR it builds for `trap` maps back to every
//! call-graph node; `trap` is the same whether read in `stave check`, a debug build, or a release
//! build; a program that denies no `trap` lowers nothing for it.

use std::path::Path;

use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_sema::callgraph::NodeKind;
use stave_sema::effects::Effects;

fn repo() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn run(file: &str, goal: Goal, f: impl FnOnce(&mut Options)) -> Output {
    let mut opts = Options::new(goal).threads(Some(4));
    opts.target = Some(stave_target::Triple::parse("x86_64-linux").unwrap());
    f(&mut opts);
    pipeline::run(&repo().join(file), &Library::in_dir(repo()), &opts).unwrap()
}

/// The node named `name`.
fn node(out: &Output, name: &str) -> u32 {
    let tc = out.typeck.as_ref().unwrap();
    let g = &tc.call_graph;
    let all: Vec<u32> = (0..g.nodes.len() as u32).collect();
    let names = g.node_names(&out.program, tc, &all);
    names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no node `{name}`")) as u32
}

/// Every fn and closure node with a body gets its function of the lowered IR (std and runtime
/// included, generic instances and closures too), on programs with generics, closures, fn
/// values, threads, and f-strings.
#[test]
fn every_node_maps_to_its_ir() {
    for file in [
        "tests/conformance/run/fnval_coerce.stv",
        "tests/conformance/run/once_init.stv",
        "tests/conformance/run/select_words.stv",
        "tests/conformance/effects/block_neg.stv",
        "tests/conformance/effects/generic_instance.stv",
    ] {
        let out = run(file, Goal::Check, |_| {});
        assert!(out.failure.is_none(), "{file}: {:?}", out.failure);
        let tc = out.typeck.as_ref().unwrap();
        let g = &tc.call_graph;
        let all: Vec<u32> = (0..g.nodes.len() as u32).collect();
        let l = stave_driver::effects::node_ir(&out.program, tc, &all).unwrap();
        let unmapped = g.node_names(&out.program, tc, &l.unmapped);
        assert!(unmapped.is_empty(), "{file}: {unmapped:?}");
        let bodies = g
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Fn { body: true, .. } | NodeKind::Closure { .. }))
            .count();
        assert!(l.funcs >= bodies, "{file}: {} functions for {bodies} nodes", l.funcs);
    }
}

/// `trap` of a fn is what the checked IR after the removal step says, and that IR is the same
/// in a debug and a release build: `main.g` of trap_ok.stv has its bounds check proven and
/// removed in both, and the effect pass of `stave check` agrees.
#[test]
fn trap_is_the_same_in_check_debug_and_release() {
    let file = "tests/conformance/effects/trap_ok.stv";
    for release in [false, true] {
        let out = run(file, Goal::Check, |o| {
            o.lower = true;
            o.release = release;
        });
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let m = out.module.as_ref().unwrap();
        let g = m.funcs.iter().position(|f| f.path == "main.g").unwrap();
        let t = stave_ir::traps::scan(m, stave_ir::FuncId(g as u32));
        assert_eq!(t.trap, None, "release {release}");
    }
    let out = run(file, Goal::Check, |o| o.effects_all = true);
    let sets = out.effects.as_ref().unwrap();
    assert!(!sets.set[node(&out, "main.g") as usize].meets(Effects::TRAP));
    assert!(sets.trap_from_ir);
    // and a fn that indexes past what it tests keeps its check: trap
    let out = run("tests/conformance/effects/trap_neg.stv", Goal::Check, |_| {});
    let codes: Vec<&str> = out.diagnostics.iter().map(|d| d.code()).collect();
    assert_eq!(codes, ["E1401"]);
}

/// The checks run in `stave check` and in builds alike, with the same diagnostics.
#[test]
fn check_and_builds_report_the_same() {
    let key = |o: &Output| -> Vec<(String, u32, u32)> {
        o.diagnostics
            .iter()
            .map(|d| (d.code().to_string(), d.primary().map_or(0, |s| s.offset), d.related().len() as u32))
            .collect()
    };
    for file in ["tests/conformance/effects/trap_neg.stv", "tests/conformance/effects/fn_types.stv"] {
        let check = run(file, Goal::Check, |_| {});
        let debug = run(file, Goal::Build, |_| {});
        let release = run(file, Goal::Build, |o| o.release = true);
        assert!(!check.diagnostics.is_empty(), "{file}");
        assert_eq!(key(&check), key(&debug), "{file}");
        assert_eq!(key(&check), key(&release), "{file}");
    }
}

/// Without a deny clause naming `trap`, the pass builds no IR.
#[test]
fn no_trap_denied_lowers_nothing() {
    let out = run("tests/conformance/effects/io_neg.stv", Goal::Check, |_| {});
    let tc = out.typeck.as_ref().unwrap();
    let r = stave_driver::effects::run(&out.program, tc, false).unwrap();
    assert_eq!(r.lowered, 0);
    assert!(!r.sets.trap_from_ir);
    assert_eq!(r.diagnostics.len(), 1);
    let r = stave_driver::effects::run(&out.program, tc, true).unwrap();
    assert!(r.lowered > 0);
}
