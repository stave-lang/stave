//! The computed stack bound ([`stave_driver::stack`]) on built programs, both CPU architectures,
//! debug and release:
//! - a call chain without recursion and with large frames gets its bound on main and on a
//!   spawned thread, the bound wins over a much smaller `[run] stack`, and the program runs
//!   (`tests/conformance/run/stack_bound`);
//! - the same chain through a call cycle of a `rec fn` has no bound, keeps that `[run] stack`,
//!   and overflows (`stack_rec`): so the chain needs more than `[run] stack`, and the computed
//!   stack is what lets `stack_bound` run; with the `rec fn` on no cycle it is bounded and runs
//!   (`stack_rec_stale`);
//! - each `spawn` body is a root with its own bound (`stack_spawn.stv`);
//! - a trap in the deepest frame of a bounded chain, on main and on a spawned thread, still
//!   prints the whole panic report (`run/stack_trap_main`, `borrow/stack_trap_spawn`);
//! - a drop cycle of a recursive type has no bound;
//! - every run-suite program, debug and release, maps every fn value it makes to the call graph
//!   (no stack is unbounded for an unresolved fn-value call), and gets a bound unless it reaches
//!   a call cycle or a drop cycle.
//!
//! A leg that cannot run here (no qemu-aarch64, not Linux) builds only, or fails when
//! `STAVE_REQUIRE_LEGS` lists it.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::exec::{self, RunSpec, TempDir};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_driver::stack::{At, Depth, Why};
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run_dir() -> PathBuf {
    repo().join("tests/conformance/run")
}

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{}: diagnostics {:?}", p.display(), out.diagnostics);
    assert!(out.failure.is_none(), "{}: {}", p.display(), out.failure.as_ref().unwrap());
    assert!(out.image.is_some() && out.stack.is_some(), "{}: no image or no stack bound", p.display());
    out
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// `Some(None)` native, `Some(Some(cmd))` through `cmd`, `None` not runnable.
fn runner(target: Triple) -> Option<Option<&'static str>> {
    let how = if exec::can_run(target) {
        Some(None)
    } else if target.name() == "aarch64-linux" && cfg!(target_os = "linux") && have("qemu-aarch64") {
        Some(Some("qemu-aarch64"))
    } else {
        None
    };
    legs::gate(target.name(), how.is_some());
    how
}

fn targets() -> [Triple; 2] {
    ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap())
}

/// Runs the image of `out`; `None` when `target` cannot run here.
fn run(out: &Output, target: Triple, name: &str, cwd: &Path) -> Option<exec::Ran> {
    let how = runner(target)?;
    let tmp = TempDir::new("stkb").unwrap();
    let bin = tmp.path().join(name);
    out.image.as_ref().unwrap().write_file(&bin).unwrap();
    let spec = RunSpec { args: &[], runner: how, cwd: Some(cwd), inherit: false };
    Some(exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display())))
}

fn bytes(d: Depth, what: &str) -> u64 {
    d.unwrap_or_else(|w| panic!("{what}: no bound ({w:?})"))
}

/// Eight frames of 32 KiB: the bound covers them on main and on the spawned thread, it wins over
/// `[run] stack = "64KiB"`, and the program runs.
#[test]
fn the_bound_wins_over_run_stack() {
    let p = run_dir().join("stack_bound");
    let want = std::fs::read(p.join("expected.stdout")).unwrap();
    for target in targets() {
        for release in [false, true] {
            let out = build(&p, target, release);
            let r = out.stack.as_ref().unwrap();
            let what = format!("stack_bound [{target}, release {release}]");
            assert_eq!(r.run_stack, Some(64 << 10), "{what}");
            let main = bytes(r.main, &what);
            assert!(main > 8 * 32768, "{what}: main {main}");
            let spawn: Vec<u64> = r.threads.iter().map(|t| bytes(t.depth, &what)).collect();
            assert_eq!(spawn.len(), 1, "{what}: {:?}", r.threads);
            assert!(spawn[0] > 8 * 32768, "{what}: spawn {}", spawn[0]);
            let lines = out.stack_lines();
            assert_eq!(lines.len(), 2, "{what}: {lines:?}");
            assert_eq!(lines[0], format!("stack: main: {main} bytes"), "{what}");
            assert!(lines[1].starts_with("stack: spawn at src/main.stv:67:9: "), "{what}: {lines:?}");
            let Some(got) = run(&out, target, "stack_bound", &p) else { continue };
            assert_eq!(
                (got.exit, &got.stdout),
                (0, &want),
                "{what}: {}",
                String::from_utf8_lossy(&got.stderr)
            );
        }
    }
}

/// The same chain through a call cycle of a `rec fn`: no bound, `[run] stack` (64 KiB) applies,
/// and the chain overflows it at the guard page.
#[test]
fn a_rec_fn_keeps_run_stack() {
    let p = run_dir().join("stack_rec");
    for target in targets() {
        for release in [false, true] {
            let out = build(&p, target, release);
            let r = out.stack.as_ref().unwrap();
            let what = format!("stack_rec [{target}, release {release}]");
            assert!(matches!(r.main, Err(Why::Rec(At::Graph(_)))), "{what}: {:?}", r.main);
            assert_eq!(
                out.stack_lines(),
                vec![
                    "stack: main: unbounded (reaches rec fn `main.f4`), [run] stack 65536 bytes".to_string()
                ],
                "{what}"
            );
            let Some(got) = run(&out, target, "stack_rec", &p) else { continue };
            let err = String::from_utf8_lossy(&got.stderr);
            assert_eq!(got.exit, 101, "{what}: {err}");
            assert!(err.contains("panic: StackOverflow"), "{what}: {err}");
        }
    }
}

/// A bounds check fails in the deepest frame of the chain, on main and on a spawned thread: the
/// panic report runs on top of the chain inside the computed stack and prints in full.
#[test]
fn a_trap_at_the_bottom_of_the_bound_reports_in_full() {
    // the spawn one is outside run/: the interpreter runs a spawn body on the calling thread
    let borrow = repo().join("tests/conformance/borrow");
    for (dir, name, thread) in [(run_dir(), "stack_trap_main", 0), (borrow, "stack_trap_spawn", 1)] {
        let p = dir.join(name);
        let want = format!("panic: Bounds at src/main.stv:12:15 in main.f0 (thread {thread})\n");
        for target in targets() {
            for release in [false, true] {
                let out = build(&p, target, release);
                let r = out.stack.as_ref().unwrap();
                let what = format!("{name} [{target}, release {release}]");
                bytes(r.main, &what);
                for t in &r.threads {
                    bytes(t.depth, &what);
                }
                let Some(got) = run(&out, target, name, &p) else { continue };
                assert_eq!(got.exit, 101, "{what}");
                assert_eq!(String::from_utf8_lossy(&got.stderr), want, "{what}");
            }
        }
    }
}

/// Two `spawn` bodies, one deep and one shallow: two rows with their own bounds.
#[test]
fn spawn_bodies_get_their_own_bound() {
    let p = run_dir().join("stack_spawn.stv");
    let want = std::fs::read(run_dir().join("stack_spawn.stdout")).unwrap();
    for target in targets() {
        for release in [false, true] {
            let out = build(&p, target, release);
            let r = out.stack.as_ref().unwrap();
            let what = format!("stack_spawn [{target}, release {release}]");
            let mut b: Vec<u64> = r.threads.iter().map(|t| bytes(t.depth, &what)).collect();
            assert_eq!(b.len(), 2, "{what}: {:?}", r.threads);
            b.sort_unstable();
            assert!(b[0] < 32768 && b[1] > 3 * 16384, "{what}: {b:?}");
            let Some(got) = run(&out, target, "stack_spawn", &run_dir()) else { continue };
            assert_eq!(
                (got.exit, &got.stdout),
                (0, &want),
                "{what}: {}",
                String::from_utf8_lossy(&got.stderr)
            );
        }
    }
}

/// Dropping a recursive value is a cycle through drop glue: no bound.
#[test]
fn a_drop_cycle_has_no_bound() {
    let tmp = TempDir::new("stkd").unwrap();
    let p = tmp.path().join("drop_cycle.stv");
    std::fs::write(
        &p,
        "import std.io;\nimport std.vec;\n\nstruct N {\n    pub v: u32,\n    pub kids: vec.Vec<N>,\n}\n\n\
         fn main() {\n    let mut root = N.{ v: 1u32, kids: vec.Vec::<N>.new() };\n    \
         root.kids.push(N.{ v: 2u32, kids: vec.Vec::<N>.new() });\n    let n = root.kids.len();\n    \
         io.print(f\"kids {n}\\n\");\n}\n",
    )
    .unwrap();
    let out = build(&p, Triple::parse("x86_64-linux").unwrap(), false);
    let r = out.stack.as_ref().unwrap();
    assert!(matches!(r.main, Err(Why::Cycle(_))), "{:?}", r.main);
    let lines = out.stack_lines();
    assert!(
        lines[0].starts_with("stack: main: unbounded (reaches a call cycle through `drop_glue "),
        "{lines:?}"
    );
}

/// Every program of the run suite (and the thread programs): no stack is unbounded for a
/// fn-value call the call graph does not resolve, and one without `rec fn` gets a bound.
#[test]
fn run_suite_programs_are_resolved() {
    let mut programs: Vec<PathBuf> = Vec::new();
    for e in std::fs::read_dir(run_dir()).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with(['.', '_']) {
            continue;
        }
        if p.is_dir() || name.ends_with(".stv") {
            programs.push(p);
        }
    }
    let threads = repo().join("tests/conformance/borrow");
    for name in ["threads_scope_run", "threads_par_for_run", "threads_join_run", "threads_mutex_join"] {
        programs.push(threads.join(format!("{name}.stv")));
    }
    programs.sort();
    let target = Triple::parse("x86_64-linux").unwrap();
    let mut bad = Vec::new();
    let mut bounded = 0;
    for p in &programs {
        let src = if p.is_dir() { p.join("src/main.stv") } else { p.clone() };
        let text = std::fs::read_to_string(&src).unwrap_or_default();
        // programs that do not build (negative tests, other targets) are not this test's
        if text.contains("//~ ERROR") || text.contains("//~ TARGET") || text.contains("//~ CHECK-ONLY") {
            continue;
        }
        for release in [false, true] {
            let mut o = Options::new(Goal::Build).threads(Some(4));
            o.target = Some(target);
            o.release = release;
            let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
            let Some(r) = &out.stack else { continue };
            let all = std::iter::once(r.main).chain(r.threads.iter().map(|t| t.depth));
            for d in all {
                match d {
                    Ok(_) => bounded += 1,
                    Err(Why::Rec(_) | Why::Cycle(_))
                        if text.contains("rec fn") || text.contains("vec.Vec<") => {}
                    Err(w) => bad.push(format!(
                        "{} (release {release}): {w:?}: {:?}",
                        p.display(),
                        out.stack_lines()
                    )),
                }
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    assert!(bounded > 80, "{bounded} bounded stacks");
}
