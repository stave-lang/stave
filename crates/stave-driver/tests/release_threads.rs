//! Thread-starting programs build and run as static ELF executables on both CPU architectures,
//! debug and release: the `threads_*` run programs of `tests/conformance/borrow/` (`scope` /
//! `spawn`, `thread.par_for`, `thread.join`) run natively (x86-64) or under `qemu-aarch64`, and
//! their stdout equals the program's `.stdout` file with exit status 0. The executables run
//! directly: the reference interpreter runs thread bodies inline, so it cannot show that a thread
//! really starts.
//!
//! `runtime.thread.entry` is called only by the backend's `__stave_thread_start` stub, never by
//! IR, so in every built module (also a thread-free one) it has a body exactly when some body
//! uses `intrinsic.thread_clone` or `intrinsic.thread_start_addr`.
//!
//! A leg that cannot run here (no qemu-aarch64, not Linux) is skipped with a message, or fails
//! when `STAVE_REQUIRE_LEGS` lists it; the module checks always run.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::exec::{self, RunSpec, TempDir};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::{Module, Op};
use stave_target::Triple;

const THREAD_ENTRY: &str = "runtime.thread.entry";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const PROGRAMS: [&str; 4] =
    ["threads_scope_run", "threads_par_for_run", "threads_join_run", "threads_mutex_join"];

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{}: diagnostics {:?}", p.display(), out.diagnostics);
    let mode = if release { "release" } else { "debug" };
    assert!(out.failure.is_none(), "{} ({target}, {mode}): {}", p.display(), out.failure.as_ref().unwrap());
    assert!(out.image.is_some() && out.module.as_ref().is_some_and(|m| m.main.is_some()));
    out
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// The runner for `target` here: `Some(None)` native, `Some(Some(cmd))` through `cmd`, `None`
/// not runnable.
fn runner(target: Triple) -> Option<Option<&'static str>> {
    let how = if exec::can_run(target) {
        Some(None)
    } else if target == Triple::parse("aarch64-linux").unwrap()
        && cfg!(target_os = "linux")
        && have("qemu-aarch64")
    {
        Some(Some("qemu-aarch64"))
    } else {
        None
    };
    legs::gate(target.name(), how.is_some());
    how
}

fn starts_threads(m: &Module) -> bool {
    m.funcs.iter().filter_map(|f| f.body.as_ref()).any(|b| {
        b.insts.op.iter().any(|op| {
            matches!(op, Op::Intrinsic(i)
                if matches!(m.intrinsics.get(i.index()).map(String::as_str),
                    Some("thread_clone" | "thread_start_addr")))
        })
    })
}

fn has_thread_entry(m: &Module) -> bool {
    m.funcs.iter().any(|f| f.path == THREAD_ENTRY && f.body.is_some())
}

#[test]
fn thread_programs_run_debug_and_release_on_both_arches() {
    let dir = repo().join("tests/conformance/borrow");
    let tmp = TempDir::new("rthr").unwrap();
    let mut failures = Vec::new();
    let mut runs = 0;
    for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        let how = runner(target);
        if how.is_none() {
            eprintln!("note: {target} binaries cannot run here (no native CPU, no qemu-aarch64): build only");
        }
        for name in PROGRAMS {
            let p = dir.join(format!("{name}.stv"));
            let want = std::fs::read(dir.join(format!("{name}.stdout"))).unwrap();
            for release in [false, true] {
                let mode = if release { "release" } else { "debug" };
                let out = build(&p, target, release);
                let m = out.module.as_ref().unwrap();
                if !starts_threads(m) || !has_thread_entry(m) {
                    failures.push(format!(
                        "{name} [{target}, {mode}]: starts threads {}, has `{THREAD_ENTRY}` {}",
                        starts_threads(m),
                        has_thread_entry(m)
                    ));
                    continue;
                }
                let Some(runner) = how else { continue };
                let bin = tmp.path().join(format!("{name}-{target}-{mode}"));
                out.image.as_ref().unwrap().write_file(&bin).unwrap();
                let spec = RunSpec { args: &[], runner, cwd: Some(&dir), inherit: false };
                let got = exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
                runs += 1;
                if got.exit != 0 || got.stdout != want {
                    failures.push(format!(
                        "{name} [{target}, {mode}]: exit {}\n--- stdout\n{}--- want\n{}--- stderr\n{}",
                        got.exit,
                        String::from_utf8_lossy(&got.stdout),
                        String::from_utf8_lossy(&want),
                        String::from_utf8_lossy(&got.stderr)
                    ));
                }
            }
        }
    }
    eprintln!("{runs} native runs");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// A program that starts no thread has no `runtime.thread.entry`, debug or release.
#[test]
fn thread_entry_only_with_thread_start() {
    let p = repo().join("tests/conformance/run/hello");
    for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        for release in [false, true] {
            let out = build(&p, target, release);
            let m = out.module.as_ref().unwrap();
            assert_eq!(
                has_thread_entry(m),
                starts_threads(m),
                "hello [{target}, release {release}]: `{THREAD_ENTRY}` iff a thread-start intrinsic"
            );
        }
    }
}
