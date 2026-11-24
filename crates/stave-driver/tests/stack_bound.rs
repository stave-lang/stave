//! The computed stack bound ([`stave_driver::stack`]) on built programs, both CPU architectures,
//! debug and release:
//! - a call chain without recursion and with large frames gets its bound on main and on a
//!   spawned thread, the bound wins over a much smaller `[run] stack`, and the program runs
//!   (`tests/conformance/run/stack_bound`);
//! - the same chain through a call cycle of a `rec fn` has no bound; it gets that `[run] stack`
//!   plus the bound of the chain with the cycle collapsed, and runs (`stack_rec`); with the
//!   `rec fn` on no cycle it is bounded and runs (`stack_rec_stale`);
//! - a `rec fn` whose recursion leaves too little stack for the `without(trap)` chain below it
//!   overflows in its own entry, which probes the chain's reserve, not in the chain
//!   (`stack_rec_reserve`);
//! - each `spawn` body is a root with its own bound (`stack_spawn.stv`);
//! - a trap in the deepest frame of a bounded chain, on main and on a spawned thread, still
//!   prints the whole panic report (`run/stack_trap_main`, `borrow/stack_trap_spawn`);
//! - a `rec fn` whose only calls off its cycle are traps gets no probe; recursing without end
//!   with small, guard-sized, and larger frames, on main and on a spawned thread, it ends in its
//!   own `StackOverflow` report; a trap at the deepest level that still fits prints its own
//!   report (the report runs on the alternate stack), and one level deeper is `StackOverflow`;
//! - a cycle through a fn-type table: the table's targets count below its caller, which probes,
//!   so the overflow is raised in the cycle, never in a `without(trap)` target;
//! - a report that runs past its stack faults in the guard below it and ends the process at
//!   once (exit 101, no second report); a thread whose signal stack the kernel refuses fails at
//!   its start; a report to a pipe without a reader keeps exit 101; a fault in the signal
//!   handler itself (past its stack) ends the process with exit 101, never a loop; a fault that is
//!   no stack overflow is killed by SIGSEGV, on main and on a thread;
//! - the thread-block offsets the back ends read are the runtime's;
//! - a drop cycle of a recursive type has no bound;
//! - every run-suite program, debug and release, maps every fn value it makes to the call graph
//!   (no stack is unbounded for an unresolved fn-value call), gets a bound unless it reaches a
//!   call cycle or a drop cycle, has a panic report that fits the report stack, and a signal
//!   handler that fits half the signal stack.
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
    build_with(p, &repo(), target, release)
}

/// [`build`] with the std and runtime of library directory `lib`.
fn build_with(p: &Path, lib: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(lib), &o).unwrap();
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

/// `runtime.config` constant `name` (a `usize` literal).
fn config_const(name: &str) -> u64 {
    let cfg = std::fs::read_to_string(repo().join("runtime/config.stv")).unwrap();
    let prefix = format!("pub const {name}: usize = ");
    cfg.lines()
        .find_map(|l| l.strip_prefix(prefix.as_str()))
        .and_then(|v| v.trim_end_matches(';').parse().ok())
        .unwrap_or_else(|| panic!("runtime.config.{name}"))
}

/// Builds the one-file program `src` as `<tmp>/<name>.stv`.
fn build_src(tmp: &TempDir, name: &str, src: &str, target: Triple, release: bool) -> Output {
    let p = tmp.path().join(format!("{name}.stv"));
    std::fs::write(&p, src).unwrap();
    build(&p, target, release)
}

/// The paths of the functions that probe their reserve at entry.
fn probing(out: &Output) -> Vec<String> {
    let m = out.module.as_ref().unwrap();
    out.stack.as_ref().unwrap().probes.iter().map(|&f| m.func(f).unwrap().path.clone()).collect()
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

/// The same chain through a call cycle of a `rec fn`: no bound; the stack is `[run] stack`
/// (64 KiB) plus the chain's bound with the cycle collapsed, and the chain runs.
#[test]
fn a_rec_fn_gets_run_stack_plus_the_acyclic_bound() {
    let p = run_dir().join("stack_rec");
    let want = std::fs::read(p.join("expected.stdout")).unwrap();
    for target in targets() {
        for release in [false, true] {
            let out = build(&p, target, release);
            let r = out.stack.as_ref().unwrap();
            let what = format!("stack_rec [{target}, release {release}]");
            assert!(matches!(r.main, Err(Why::Rec(At::Graph(_)))), "{what}: {:?}", r.main);
            let acyclic = r.main_acyclic.unwrap_or_else(|| panic!("{what}: no acyclic bound"));
            assert!(acyclic > 8 * 32768, "{what}: {acyclic}");
            assert_eq!(
                out.stack_lines(),
                vec![format!(
                    "stack: main: unbounded (reaches rec fn `main.f4`), [run] stack 65536 + {acyclic} bytes"
                )],
                "{what}"
            );
            // `f4` probes what it calls outside its cycle: the four frames below it
            assert_eq!(r.reserve.len(), 1, "{what}: {:?}", r.reserve);
            assert!(r.reserve[0] > 4 * 32768, "{what}: {:?}", r.reserve);
            let Some(got) = run(&out, target, "stack_rec", &p) else { continue };
            assert_eq!(
                (got.exit, &got.stdout),
                (0, &want),
                "{what}: {}",
                String::from_utf8_lossy(&got.stderr)
            );
        }
    }
}

/// A recursion that leaves less than the chain below it needs: the recursing fn's entry faults,
/// never a frame of the chain.
#[test]
fn a_member_probes_its_reserve() {
    let p = run_dir().join("stack_rec_reserve");
    let want = "panic: StackOverflow at src/main.stv:33:8 in main.f4 (thread 0)\n";
    for target in targets() {
        for release in [false, true] {
            let out = build(&p, target, release);
            let r = out.stack.as_ref().unwrap();
            let what = format!("stack_rec_reserve [{target}, release {release}]");
            assert!(r.reserve.len() == 1 && r.reserve[0] > 4 << 20, "{what}: {:?}", r.reserve);
            let Some(got) = run(&out, target, "stack_rec_reserve", &p) else { continue };
            assert_eq!(got.exit, 101, "{what}");
            assert_eq!(String::from_utf8_lossy(&got.stderr), want, "{what}");
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

/// A reserve probe reads at most `PROBE_STRIDE` below the last read, which is at most 4 KiB below
/// a written stack byte: the runtime's guard below every stack must be at least that wide.
#[test]
fn the_probe_stride_cannot_pass_the_guard() {
    let guard = config_const("GUARD");
    assert!(stave_obj::stub::PROBE_STRIDE + 4096 <= guard, "guard {guard}");
}

const FIB_ACK: &str = r#"import std.io;

rec fn fib(n: u64) -> u64 {
    if n < 2u64 {
        return n;
    }
    return fib(n - 1u64) + fib(n - 2u64);
}

rec fn ack(m: u64, n: u64) -> u64 {
    if m == 0u64 {
        return n + 1u64;
    }
    if n == 0u64 {
        return ack(m - 1u64, 1u64);
    }
    return ack(m - 1u64, ack(m, n - 1u64));
}

fn main() {
    let a = fib(20u64);
    let b = ack(2u64, 3u64);
    io.print(f"{a} {b}\n");
}
"#;

/// A `rec fn` whose calls off its cycle are only traps: nothing runs below it but its own code and
/// the report, which runs on the alternate stack, so it gets no probe; `stack_rec`'s `f4`, which
/// calls a chain off its cycle, does.
#[test]
fn only_traps_below_a_cycle_need_no_probe() {
    let tmp = TempDir::new("stkp").unwrap();
    for target in targets() {
        for release in [false, true] {
            let what = format!("[{target}, release {release}]");
            let out = build_src(&tmp, "fib_ack", FIB_ACK, target, release);
            assert_eq!(probing(&out), Vec::<String>::new(), "fib_ack {what}");
            let out = build(&run_dir().join("stack_rec"), target, release);
            assert_eq!(probing(&out), vec!["main.f4".to_string()], "stack_rec {what}");
        }
    }
}

/// `down` recursing without end with a frame of `WORDS` words; `MAIN` calls it.
const DOWN: &str = r#"import std.io;

rec fn down(n: u64) -> u64 {
    let mut buf: [u64; WORDS] = [0u64; WORDS];
    let i = (n % WORDSu64) as usize;
    buf[i] = n;
    if n == 0u64 {
        return buf[i];
    }
    return down(n - 1u64) +% buf[i];
}

MAIN"#;

const ON_MAIN: &str = r#"fn main() {
    let r = DOWN;
    io.print(f"{r}\n");
}
"#;

const ON_SPAWN: &str = r#"fn main() {
    let mut b: u64 = 0;
    scope {
        spawn { b = DOWN; }
    }
    io.print(f"{b}\n");
}
"#;

/// A `rec fn` recursing without end, with frames of 64 bytes, about the guard (60 KiB), and
/// larger than it (100 KiB), on main and on a spawned thread: each run ends in the
/// `StackOverflow` report of the recursing fn, exit 101, never a signal.
#[test]
fn unbounded_recursion_reports_stack_overflow() {
    let tmp = TempDir::new("stko").unwrap();
    for words in [8u32, 7600, 12800] {
        for (place, thread, main) in [("main", 0, ON_MAIN), ("spawn", 1, ON_SPAWN)] {
            let src = DOWN
                .replace("WORDS", &words.to_string())
                .replace("MAIN", &main.replace("DOWN", "down(1000000000u64)"));
            let name = format!("ov_{words}_{place}");
            let want = format!(":3:8 in main.down (thread {thread})\n");
            for target in targets() {
                for release in [false, true] {
                    let what = format!("{name} [{target}, release {release}]");
                    let out = build_src(&tmp, &name, &src, target, release);
                    let Some(got) = run(&out, target, &name, tmp.path()) else { continue };
                    let err = String::from_utf8_lossy(&got.stderr);
                    assert_eq!(got.exit, 101, "{what}: {err}");
                    assert!(
                        err.starts_with("panic: StackOverflow at ")
                            && err.ends_with(&want)
                            && err.lines().count() == 1,
                        "{what}: {err}"
                    );
                }
            }
        }
    }
}

/// `down` traps Bounds at depth `k`, the first argument.
const TRAP_AT: &str = r#"import std.io;
import std.env;
import std.num;

rec fn down(n: u64, k: u64) -> u64 {
    let mut buf: [u64; 4] = [0u64; 4];
    let i = (n % 4u64) as usize;
    buf[i] = n;
    if n == k {
        return buf[i + 4];
    }
    return down(n + 1u64, k) +% buf[i];
}

fn depth() -> u64 {
    let a = env.arg(1);
    return match a {
        Option.Some(s) => match num.parse_u64(s.as_str()) {
            Option.Some(k) => k,
            Option.None => 0u64,
        },
        Option.None => 0u64,
    };
}

MAIN"#;

/// A Bounds trap at depth `k` of a small-frame `rec fn`, on main and on a spawned thread: at the
/// largest `k` that reports Bounds (bisection), with less than a frame of stack left, the report
/// prints in full; around it every run ends in one report line, Bounds up to that `k` and
/// `StackOverflow` past it, exit 101.
#[test]
fn a_trap_at_the_deepest_level_prints_its_report() {
    let tmp = TempDir::new("stkt").unwrap();
    for (place, thread, main) in [("main", 0, ON_MAIN), ("spawn", 1, ON_SPAWN)] {
        let src = TRAP_AT.replace("MAIN", &main.replace("DOWN", "down(0u64, depth())"));
        let name = format!("trap_at_{place}");
        let bounds = format!("panic: Bounds at {name}.stv:10:19 in main.down (thread {thread})\n");
        for target in targets() {
            for release in [false, true] {
                let what = format!("{name} [{target}, release {release}]");
                let out = build_src(&tmp, &name, &src, target, release);
                let Some(how) = runner(target) else { continue };
                let bin = tmp.path().join(&name);
                out.image.as_ref().unwrap().write_file(&bin).unwrap();
                let at = |k: u64| {
                    let args = [k.to_string()];
                    let spec = RunSpec { args: &args, runner: how, cwd: Some(tmp.path()), inherit: false };
                    let got = exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{what}: {e}"));
                    (got.exit, String::from_utf8_lossy(&got.stderr).into_owned())
                };
                let is_bounds = |k: u64| at(k).1.ends_with(&bounds);
                let (mut lo, mut hi) = (0u64, 1u64 << 20);
                assert!(is_bounds(lo) && !is_bounds(hi), "{what}: {:?}", at(lo));
                while hi - lo > 1 {
                    let mid = lo + (hi - lo) / 2;
                    if is_bounds(mid) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                for k in lo.saturating_sub(2)..=lo + 2 {
                    let (exit, err) = at(k);
                    let want = if k <= lo {
                        err.ends_with(&bounds)
                    } else {
                        err.starts_with("panic: StackOverflow at ")
                    };
                    assert!(
                        exit == 101 && err.lines().count() == 1 && want,
                        "{what}: k_max {lo}, k {k}: {exit} {err}"
                    );
                }
            }
        }
    }
}

/// A cycle through a fn-type table: `g` is chosen by a value `if`, so its call goes through the
/// table of `fn(u64) -> u64`, which holds `back` (so the table is in the cycle) and the
/// `without(trap)` `heavy` (240 KB frame) and `light`. The table stands for `a`'s call: `a` probes
/// for `heavy`, and the overflow is raised in `a`, never in `heavy`.
const TABLE_CYCLE: &str = r#"import std.io;

rec fn a(n: u64) -> u64 {
    let g: fn(u64) -> u64 = if n % 1000u64 == 999u64 then heavy else light;
    let h = g(n);
    return back(n + 1u64) +% (h -% h);
}

rec fn back(n: u64) -> u64 {
    let keep: fn(u64) -> u64 = back;
    return a(n);
}

fn light(n: u64) -> u64 without(trap) {
    return n;
}

fn heavy(n: u64) -> u64 without(trap) {
    let t: [u64; 30000] = [n; 30000];
    return t[0];
}

MAIN"#;

#[test]
fn a_table_in_a_cycle_counts_below_its_caller() {
    let tmp = TempDir::new("stkf").unwrap();
    for (place, thread, main) in [("main", 0, ON_MAIN), ("spawn", 1, ON_SPAWN)] {
        let src = TABLE_CYCLE.replace("MAIN", &main.replace("DOWN", "back(0u64)"));
        let name = format!("table_{place}");
        for target in targets() {
            for release in [false, true] {
                let what = format!("{name} [{target}, release {release}]");
                let out = build_src(&tmp, &name, &src, target, release);
                assert!(probing(&out).contains(&"main.a".to_string()), "{what}: {:?}", probing(&out));
                let Some(got) = run(&out, target, &name, tmp.path()) else { continue };
                let err = String::from_utf8_lossy(&got.stderr);
                let at_a = format!(":3:8 in main.a (thread {thread})\n");
                let at_b = format!(":9:8 in main.back (thread {thread})\n");
                assert_eq!(got.exit, 101, "{what}: {err}");
                assert!(
                    err.starts_with("panic: StackOverflow at ")
                        && (err.ends_with(&at_a) || err.ends_with(&at_b))
                        && err.lines().count() == 1,
                    "{what}: {err}"
                );
            }
        }
    }
}

/// A copy of the shipped library in `tmp`, with `old` replaced by `new` in `file` (relative to
/// the library directory).
fn patched_library(tmp: &TempDir, file: &str, edits: &[(&str, &str)]) -> PathBuf {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap() {
            let p = e.unwrap().path();
            let dst = to.join(p.file_name().unwrap());
            if p.is_dir() { copy(&p, &dst) } else { std::fs::copy(&p, &dst).map(|_| ()).unwrap() }
        }
    }
    let lib = tmp.path().join("lib");
    for d in ["std", "runtime"] {
        copy(&repo().join(d), &lib.join(d));
    }
    let p = lib.join(file);
    let mut text = std::fs::read_to_string(&p).unwrap();
    for (old, new) in edits {
        assert!(text.contains(old), "{file}: {old}");
        text = text.replacen(old, new, 1);
    }
    std::fs::write(&p, text).unwrap();
    lib
}

/// A Bounds trap in `pick`.
const BOUNDS: &str = r#"import std.io;

fn pick(xs: &[u64], i: usize) -> u64 {
    return xs[i];
}

fn at() -> u64 {
    let xs: [u64; 4] = [1u64; 4];
    let s = &xs[0..4];
    return pick(s, s.len() + 3);
}

MAIN"#;

/// The report itself recursing far past its stack (injected into `runtime.panic.head`), for a
/// Bounds trap and for a stack overflow, on main and on a spawned thread: the report faults in
/// the guard below its stack and the process ends at once, exit 101, with no second report (the
/// line was not written yet, so stderr is empty) and never a raw signal.
#[test]
fn a_report_past_its_stack_ends_the_process() {
    let tmp = TempDir::new("stkg").unwrap();
    let lib = patched_library(
        &tmp,
        "runtime/panic.stv",
        &[
            (
                "    l.lit(\"panic: \");\n",
                "    l.dec(burn(kind as u64 + 100000u64));\n    l.lit(\"panic: \");\n",
            ),
            (
                "fn head(",
                "rec fn burn(n: u64) -> u64 {\n    let mut t: [u64; 128] = [0u64; 128];\n    \
                 t[(n % 128u64) as usize] = n;\n    if n == 0u64 {\n        return t[0];\n    }\n    \
                 return burn(n - 1u64) +% t[1];\n}\n\nfn head(",
            ),
        ],
    );
    let cases = [
        ("bounds", BOUNDS.to_string(), "at()"),
        ("overflow", DOWN.replace("WORDS", "8"), "down(1000000000u64)"),
    ];
    for (kind, src, call) in &cases {
        for (place, main) in [("main", ON_MAIN), ("spawn", ON_SPAWN)] {
            let src = src.replace("MAIN", &main.replace("DOWN", call));
            let name = format!("deep_report_{kind}_{place}");
            let p = tmp.path().join(format!("{name}.stv"));
            std::fs::write(&p, &src).unwrap();
            for target in targets() {
                for release in [false, true] {
                    let what = format!("{name} [{target}, release {release}]");
                    let out = build_with(&p, &lib, target, release);
                    assert!(out.stack.as_ref().unwrap().alt.is_err(), "{what}: the report recurses");
                    let Some(got) = run(&out, target, &name, tmp.path()) else { continue };
                    let err = String::from_utf8_lossy(&got.stderr);
                    assert_eq!((got.exit, err.as_ref()), (101, ""), "{what}");
                }
            }
        }
    }
}

/// Runs `bin` (through `how`) with a time limit; `None` if it did not end within it (then killed).
fn run_limited(bin: &Path, how: Option<&str>, secs: u64) -> Option<std::process::ExitStatus> {
    // ETXTBSY: another test thread's fork still holds a just-written binary open.
    let mut tries = 0;
    let mut child = loop {
        let mut cmd = match how {
            Some(r) => {
                let mut c = Command::new(r);
                c.arg(bin);
                c
            }
            None => Command::new(bin),
        };
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        match cmd.spawn() {
            Err(e) if e.raw_os_error() == Some(26) && tries < 50 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            r => break r.unwrap(),
        }
    };
    let start = std::time::Instant::now();
    loop {
        if let Some(s) = child.try_wait().unwrap() {
            return Some(s);
        }
        if start.elapsed().as_secs() >= secs {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// The signal handler itself overrunning its stack (injected into `runtime.panic.on_signal`:
/// a 1 KiB-frame recursion 100 levels deep, past the 32 KiB signal stack), on a stack overflow,
/// main and spawned thread: the fault in the guard below the signal stack enters the handler
/// again, which ends the process at once with exit 101: no loop, no hang.
#[test]
fn a_fault_in_the_signal_handler_ends_the_process() {
    let tmp = TempDir::new("stkh").unwrap();
    let lib = patched_library(
        &tmp,
        "runtime/panic.stv",
        &[
            (
                "    // siginfo_t.si_addr is at offset 16",
                "    if burn(sig as u64 + 100u64) == 5u64 {\n        os.exit_group(3);\n    }\n    \
                 // siginfo_t.si_addr is at offset 16",
            ),
            (
                "fn head(",
                "rec fn burn(n: u64) -> u64 {\n    let mut t: [u64; 128] = [0u64; 128];\n    \
                 t[(n % 128u64) as usize] = n;\n    if n == 0u64 {\n        return t[0];\n    }\n    \
                 return burn(n - 1u64) +% t[1];\n}\n\nfn head(",
            ),
        ],
    );
    for (place, main) in [("main", ON_MAIN), ("spawn", ON_SPAWN)] {
        let src = DOWN.replace("WORDS", "8").replace("MAIN", &main.replace("DOWN", "down(1000000000u64)"));
        let name = format!("handler_fault_{place}");
        let p = tmp.path().join(format!("{name}.stv"));
        std::fs::write(&p, &src).unwrap();
        for target in targets() {
            for release in [false, true] {
                let what = format!("{name} [{target}, release {release}]");
                let out = build_with(&p, &lib, target, release);
                let Some(how) = runner(target) else { continue };
                let bin = tmp.path().join(&name);
                out.image.as_ref().unwrap().write_file(&bin).unwrap();
                let status = run_limited(&bin, how, 120).unwrap_or_else(|| panic!("{what}: no end in 120 s"));
                assert_eq!(status.code(), Some(101), "{what}: {status:?}");
            }
        }
    }
}

/// A fault that is not a stack overflow (a wild write) on main and on a spawned thread: the
/// default action, so the process is killed by SIGSEGV, never a normal exit (no race between
/// the re-raised signal and the handler's fallback exit).
#[test]
fn a_wild_write_is_killed_by_its_signal() {
    use std::os::unix::process::ExitStatusExt;
    let tmp = TempDir::new("stkw").unwrap();
    let poke = "import std.io;\n\nfn poke() {\n    unsafe(\"deliberate wild write\") {\n        \
                *(16usize as *mut u64) = 1u64;\n    }\n}\n\n";
    let mains = [
        ("main", "fn main() {\n    poke();\n    io.print(\"after\\n\");\n}\n"),
        (
            "spawn",
            "fn main() {\n    scope {\n        spawn { poke(); }\n    }\n    io.print(\"after\\n\");\n}\n",
        ),
    ];
    for (place, main) in mains {
        let name = format!("wild_{place}");
        for target in targets() {
            for release in [false, true] {
                let what = format!("{name} [{target}, release {release}]");
                let out = build_src(&tmp, &name, &format!("{poke}{main}"), target, release);
                let Some(how) = runner(target) else { continue };
                let bin = tmp.path().join(&name);
                out.image.as_ref().unwrap().write_file(&bin).unwrap();
                for _ in 0..10 {
                    let status =
                        run_limited(&bin, how, 120).unwrap_or_else(|| panic!("{what}: no end in 120 s"));
                    assert_eq!((status.code(), status.signal()), (None, Some(11)), "{what}: {status:?}");
                }
            }
        }
    }
}

/// A thread whose signal stack the kernel refuses (injected: size 0) fails at its start with a
/// panic, exit 101, before its body runs, so an overflow there can never be a raw SIGSEGV.
#[test]
fn a_thread_without_its_signal_stack_fails_at_start() {
    let tmp = TempDir::new("stks").unwrap();
    let lib = patched_library(
        &tmp,
        "runtime/thread.stv",
        &[("panic.set_altstack(alt, alt_len)", "panic.set_altstack(alt, 0)")],
    );
    let src = DOWN.replace("WORDS", "8").replace("MAIN", &ON_SPAWN.replace("DOWN", "down(10u64)"));
    let p = tmp.path().join("no_sigstack.stv");
    std::fs::write(&p, &src).unwrap();
    for target in targets() {
        for release in [false, true] {
            let what = format!("no_sigstack [{target}, release {release}]");
            let out = build_with(&p, &lib, target, release);
            let Some(got) = run(&out, target, "no_sigstack", tmp.path()) else { continue };
            let err = String::from_utf8_lossy(&got.stderr);
            assert_eq!(got.exit, 101, "{what}: {err}");
            assert!(err.ends_with("(thread 1)\ncannot install the thread's signal stack\n"), "{what}: {err}");
            assert!(got.stdout.is_empty(), "{what}: the body ran");
        }
    }
}

/// A trap whose report goes to a pipe without a reader: the write fails, and the exit status
/// is the panic's 101, never SIGPIPE.
#[test]
fn a_report_to_a_closed_pipe_keeps_the_panic_status() {
    let tmp = TempDir::new("stkq").unwrap();
    let src = BOUNDS.replace("MAIN", &ON_MAIN.replace("DOWN", "at()"));
    for target in targets() {
        for release in [false, true] {
            let what = format!("closed_pipe [{target}, release {release}]");
            let out = build_src(&tmp, "closed_pipe", &src, target, release);
            let Some(how) = runner(target) else { continue };
            let bin = tmp.path().join("closed_pipe");
            out.image.as_ref().unwrap().write_file(&bin).unwrap();
            let (reader, writer) = std::io::pipe().unwrap();
            drop(reader);
            let mut cmd = match how {
                Some(r) => {
                    let mut c = Command::new(r);
                    c.arg(&bin);
                    c
                }
                None => Command::new(&bin),
            };
            let status = cmd
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(writer)
                .status()
                .unwrap();
            assert_eq!(exec::status_code(status), 101, "{what}");
        }
    }
}

/// The back ends read a thread's alternate stack from its block at
/// `stave_obj::stub::BLOCK_ALT_TOP` and `BLOCK_ALT_SIZE`: the offsets of `runtime.state.Block`'s
/// `alt_top` and `alt_size` (fields laid out in order, each at its alignment).
#[test]
fn the_block_offsets_are_the_runtimes() {
    let state = std::fs::read_to_string(repo().join("runtime/state.stv")).unwrap();
    let nring: u64 = state
        .lines()
        .find_map(|l| l.strip_prefix("pub const NRING: usize = "))
        .and_then(|v| v.trim_end_matches(';').parse().ok())
        .expect("runtime.state.NRING");
    let body = state.split("pub struct Block {").nth(1).and_then(|b| b.split('}').next()).expect("Block");
    let mut off = 0u64;
    let mut at = std::collections::HashMap::new();
    for l in body.lines().map(str::trim).filter(|l| l.starts_with("pub ")) {
        let (name, ty) = l.trim_start_matches("pub ").trim_end_matches(',').split_once(": ").unwrap();
        let (size, align) = match ty {
            "usize" | "u64" => (8, 8),
            "u32" => (4, 4),
            "[u32; NRING]" => (4 * nring, 4),
            _ => panic!("Block field {name}: {ty}"),
        };
        off = off.div_ceil(align) * align;
        at.insert(name.to_string(), off);
        off += size;
    }
    assert_eq!(at.get("alt_top").copied(), Some(u64::from(stave_obj::stub::BLOCK_ALT_TOP)));
    assert_eq!(at.get("alt_size").copied(), Some(u64::from(stave_obj::stub::BLOCK_ALT_SIZE)));
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
    let report_room = config_const("REPORT_STACK");
    // the other half holds the kernel's signal frame
    let signal_room = config_const("ALTSTACK") / 2;
    let mut bad = Vec::new();
    let mut bounded = 0;
    let mut deepest = 0;
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
            match r.alt {
                Ok(a) if a <= report_room => deepest = deepest.max(a),
                a => bad.push(format!("{} (release {release}): report {a:?}", p.display())),
            }
            if !r.signal.is_ok_and(|d| d <= signal_room) {
                bad.push(format!("{} (release {release}): signal handler {:?}", p.display(), r.signal));
            }
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
    assert!(deepest > 0, "no report depth");
}
