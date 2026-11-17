//! What `runtime.cpu.init` does at start, end to end on static ELF executables, debug and
//! release (contracts/intrinsics.md "CPU feature detection"):
//!
//! - `STAVE_CPU=-name,..` removes features above the target baseline (and those implying them)
//!   from the word dispatch and `std.cpu.features()` read; a bad value exits 101 with one line;
//! - a binary whose target baseline the CPU lacks stops with `panic: UnsupportedCpu`, and every
//!   build above its arch's lowest baseline carries the check;
//! - `STAVE_DISPATCH_TRACE=1` prints one line per executed dispatch body, none for a call
//!   resolved at compile time.
//!
//! x86-64 runs natively and under `qemu-x86_64 -cpu Nehalem` (no avx); AArch64 under
//! `qemu-aarch64 -cpu cortex-a76` (no sve) and `-cpu max` (sve). Only features stable for a
//! model across qemu versions are asserted. A leg that cannot run here is skipped with a
//! message, or fails when `STAVE_REQUIRE_LEGS` lists it (the qemu-x86_64 runs are not a leg).

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::exec::{self, TempDir};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::Module;
use stave_target::{Arch, Triple};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Prints `pick <arm>`, `direct <arm>`, then `<feature> <0|1>` of the word `std.cpu.features()`
/// returns (`top`: its bit 63). `direct_user`'s call of `pick` is redirected at compile time.
const PROGRAM: &str = r#"import std.cpu;
import std.io;

fn on_avx2() -> u32 needs(avx2) {
    return 3;
}

fn on_avx() -> u32 needs(avx) {
    return 4;
}

fn on_sve() -> u32 needs(sve) {
    return 2;
}

fn on_base() -> u32 {
    return 0;
}

dispatch fn pick() -> u32 { avx2 => on_avx2, avx => on_avx, sve => on_sve, _ => on_base, }

fn direct_user() -> u32 needs(avx2) {
    return pick();
}

dispatch fn direct() -> u32 { avx2 => direct_user, _ => on_base, }

fn show(label: &str, k: u32) {
    io.print(label);
    if k == 4 { io.print(" avx\n"); return; }
    if k == 3 { io.print(" avx2\n"); return; }
    if k == 2 { io.print(" sve\n"); return; }
    io.print(" base\n");
}

fn bit(name: &str, w: u64, m: u64) {
    io.print(name);
    if w & m != 0 { io.print(" 1\n"); } else { io.print(" 0\n"); }
}

fn main() -> i32 {
    show("pick", pick());
    show("direct", direct());
    let w = cpu.features();
    bit("sse42", w, cpu.SSE42);
    bit("avx", w, cpu.AVX);
    bit("avx2", w, cpu.AVX2);
    bit("fma", w, cpu.FMA);
    bit("sve", w, cpu.SVE);
    bit("sve2", w, cpu.SVE2);
    bit("top", w, 1u64 << 63);
    return 0;
}
"#;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{}: diagnostics {:?}", p.display(), out.diagnostics);
    let mode = if release { "release" } else { "debug" };
    assert!(out.failure.is_none(), "{} ({target}, {mode}): {}", p.display(), out.failure.as_ref().unwrap());
    assert!(out.image.is_some());
    out
}

fn has_body(m: &Module, path: &str) -> bool {
    m.funcs.iter().any(|f| f.path == path && f.body.is_some())
}

fn cpu_fns(m: &Module) -> Vec<&str> {
    m.funcs
        .iter()
        .filter(|f| f.body.is_some() && f.path.starts_with("runtime.cpu."))
        .map(|f| f.path.as_str())
        .collect()
}

/// x86-64 features of the host from `/proc/cpuinfo`: (avx, avx2, fma).
fn host_avx() -> (bool, bool, bool) {
    let info = std::fs::read_to_string("/proc/cpuinfo").expect("/proc/cpuinfo");
    let line = info.lines().find(|l| l.starts_with("flags")).expect("/proc/cpuinfo: no flags line");
    let flags: Vec<&str> = line.split_once(':').unwrap().1.split_whitespace().collect();
    (flags.contains(&"avx"), flags.contains(&"avx2"), flags.contains(&"fma"))
}

struct Ran {
    exit: i32,
    stdout: String,
    stderr: String,
}

/// Runs `bin` (under `runner`, words split at spaces) with exactly the given `STAVE_CPU` /
/// `STAVE_DISPATCH_TRACE` values (`None`: unset).
fn run(bin: &Path, runner: Option<&str>, stave_cpu: Option<&str>, trace: bool) -> Ran {
    let mut tries = 0;
    loop {
        let mut words = runner.map(|r| r.split_whitespace().collect::<Vec<_>>()).unwrap_or_default();
        let mut cmd = if words.is_empty() {
            Command::new(bin)
        } else {
            let mut c = Command::new(words.remove(0));
            c.args(words).arg(bin);
            c
        };
        cmd.env_remove("STAVE_CPU").env_remove("STAVE_DISPATCH_TRACE");
        if let Some(v) = stave_cpu {
            cmd.env("STAVE_CPU", v);
        }
        if trace {
            cmd.env("STAVE_DISPATCH_TRACE", "1");
        }
        match cmd.stdin(std::process::Stdio::null()).output() {
            // ETXTBSY: another test thread's fork still holds a just-written binary open
            Err(e) if e.raw_os_error() == Some(26) && tries < 50 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => panic!("{}: {e}", bin.display()),
            Ok(o) => {
                return Ran {
                    exit: exec::status_code(o.status),
                    stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
                };
            }
        }
    }
}

/// The CPU a run sees after `STAVE_CPU`, and what [`PROGRAM`] must print for it.
#[derive(Clone, Copy, Debug, Default)]
struct Cpu {
    sse42: bool,
    avx: bool,
    avx2: bool,
    fma: bool,
    sve: bool,
}

/// [`PROGRAM`]'s stdout on `target` for the word `c`.
fn stdout_for(target: Triple, c: Cpu) -> String {
    let x86 = target.arch() == Arch::X86_64;
    let v3 = target.baseline().contains(stave_target::Feature::Avx2);
    let pick = if x86 && c.avx2 {
        "avx2"
    } else if x86 && c.avx {
        "avx"
    } else if !x86 && c.sve {
        "sve"
    } else {
        "base"
    };
    // `direct`'s avx2 arm is x86-64 only; its target calls on_avx2 directly
    let direct = if x86 && (c.avx2 || v3) { "avx2" } else { "base" };
    let b = |x: bool| u8::from(x);
    format!(
        "pick {pick}\ndirect {direct}\nsse42 {}\navx {}\navx2 {}\nfma {}\nsve {}\nsve2 0\ntop 0\n",
        b(c.sse42),
        b(c.avx),
        b(c.avx2),
        b(c.fma),
        b(c.sve)
    )
}

/// One run: STAVE_CPU, trace, and the expected (exit, stdout, stderr).
struct Case {
    stave_cpu: Option<&'static str>,
    trace: bool,
    exit: i32,
    stdout: String,
    stderr: String,
}

fn ok(stave_cpu: Option<&'static str>, trace: bool, stdout: String, stderr: String) -> Case {
    Case { stave_cpu, trace, exit: 0, stdout, stderr }
}

fn fail(stave_cpu: &'static str, line: &str) -> Case {
    Case {
        stave_cpu: Some(stave_cpu),
        trace: false,
        exit: 101,
        stdout: String::new(),
        stderr: format!("{line}\n"),
    }
}

/// The trace lines of [`PROGRAM`] on `target` for the word `c` (none where pruning or the
/// call-site redirect resolved the call).
fn trace_for(target: Triple, c: Cpu) -> String {
    match target.arch() {
        // `pick`'s and `direct`'s first arms are covered on v3: both calls are resolved
        Arch::X86_64 if target.baseline().contains(stave_target::Feature::Avx2) => String::new(),
        Arch::X86_64 => {
            let (p, d) = if c.avx2 {
                ("on_avx2", "direct_user")
            } else if c.avx {
                ("on_avx", "on_base")
            } else {
                ("on_base", "on_base")
            };
            // direct_user's `pick()` is redirected to on_avx2: no second pick line
            format!("dispatch: main.pick -> main.{p}\ndispatch: main.direct -> main.{d}\n")
        }
        // `pick`'s sve arm is covered on +sve: the call is resolved
        _ if target.baseline().contains(stave_target::Feature::Sve) => String::new(),
        // `direct` keeps only `_`: main calls on_base
        _ => format!("dispatch: main.pick -> main.{}\n", if c.sve { "on_sve" } else { "on_base" }),
    }
}

/// The cases of one CPU `c` on `target`.
fn cases_for(target: Triple, c: Cpu) -> Vec<Case> {
    let base = |c: Cpu| stdout_for(target, c);
    let mut v = vec![
        ok(None, false, base(c), String::new()),
        ok(None, true, base(c), trace_for(target, c)),
        fail("-avx3", "stave: STAVE_CPU: unknown feature 'avx3'"),
        fail("avx2", "stave: STAVE_CPU: bad syntax"),
        fail("-avx2,", "stave: STAVE_CPU: bad syntax"),
        fail("-avx2,-", "stave: STAVE_CPU: bad syntax"),
        ok(Some(""), false, base(c), String::new()),
    ];
    let v3 = target.baseline().contains(stave_target::Feature::Avx2);
    match target.arch() {
        Arch::X86_64 => {
            // removing a feature removes those implying it, never a baseline feature
            let no_avx2 = Cpu { avx2: c.avx2 && v3, ..c };
            let no_avx = Cpu { avx: c.avx && v3, avx2: c.avx2 && v3, fma: c.fma && v3, ..c };
            v.push(ok(Some("-avx2"), false, base(no_avx2), String::new()));
            v.push(ok(Some("-avx2"), true, base(no_avx2), trace_for(target, no_avx2)));
            v.push(ok(Some("-avx"), false, base(no_avx), String::new()));
            v.push(ok(Some("-fma,-avx2"), false, base(Cpu { fma: c.fma && v3, ..no_avx2 }), String::new()));
            v.push(ok(Some("-sse42"), false, base(c), String::new()));
            // another arch's feature: accepted, no effect
            v.push(ok(Some("-neon"), false, base(c), String::new()));
            v.push(ok(Some("-sve2,-neon"), false, base(c), String::new()));
        }
        _ => {
            let sve_base = target.baseline().contains(stave_target::Feature::Sve);
            let no_sve = Cpu { sve: c.sve && sve_base, ..c };
            v.push(ok(Some("-sve"), false, base(no_sve), String::new()));
            v.push(ok(Some("-sve"), true, base(no_sve), trace_for(target, no_sve)));
            v.push(ok(Some("-neon"), false, base(c), String::new()));
            v.push(ok(Some("-avx2"), false, base(c), String::new()));
        }
    }
    v
}

fn check_run(got: &Ran, want: &Case, skip_sve2: bool) -> Result<(), String> {
    let strip = |s: &str| -> String {
        s.lines().filter(|l| !(skip_sve2 && l.starts_with("sve2 "))).map(|l| format!("{l}\n")).collect()
    };
    if got.exit != want.exit || strip(&got.stdout) != strip(&want.stdout) || got.stderr != want.stderr {
        return Err(format!(
            "exit {} (want {})\n--- stdout\n{}--- want\n{}--- stderr\n{}--- want\n{}",
            got.exit, want.exit, got.stdout, want.stdout, got.stderr, want.stderr
        ));
    }
    Ok(())
}

#[test]
fn start_mask_baseline_and_trace() {
    let tmp = TempDir::new("cpu-start").unwrap();
    let src = tmp.path().join("cpu_start.stv");
    std::fs::write(&src, PROGRAM).unwrap();
    let v2 = Triple::parse("x86_64-linux").unwrap();
    let v3 = Triple::parse("x86_64-linux-v3").unwrap();
    let a64 = Triple::parse("aarch64-linux").unwrap();
    let a64_sve = Triple::parse("aarch64-linux+sve").unwrap();
    let native_v2 = legs::gate(v2.name(), exec::can_run(v2) && Path::new("/proc/cpuinfo").is_file());
    let native_v3 = legs::gate(v3.name(), exec::can_run(v3) && Path::new("/proc/cpuinfo").is_file());
    let qemu_a64 = legs::gate(a64.name(), cfg!(target_os = "linux") && have("qemu-aarch64"));
    let qemu_x86 = cfg!(target_os = "linux") && have("qemu-x86_64");

    let unsupported = "panic: UnsupportedCpu at <unknown>:0:0 in <unknown> (thread 0)\n";
    // (target, runner, label, CPU as detected; None: the CPU lacks the target's baseline)
    let mut legs: Vec<(Triple, Option<&str>, &str, Option<Cpu>)> = Vec::new();
    if native_v2 || native_v3 {
        let (avx, avx2, fma) = host_avx();
        let host = Cpu { sse42: true, avx, avx2, fma, sve: false };
        if native_v2 {
            legs.push((v2, None, "native", Some(host)));
        }
        if native_v3 {
            legs.push((v3, None, "native", Some(host)));
        }
    }
    if qemu_x86 {
        let nehalem = Cpu { sse42: true, ..Cpu::default() };
        legs.push((v2, Some("qemu-x86_64 -cpu Nehalem"), "Nehalem", Some(nehalem)));
        legs.push((v3, Some("qemu-x86_64 -cpu Nehalem"), "Nehalem", None));
    } else {
        eprintln!("note: no qemu-x86_64: the Nehalem runs are skipped");
    }
    if qemu_a64 {
        let a76 = Cpu::default();
        let max = Cpu { sve: true, ..Cpu::default() };
        legs.push((a64, Some("qemu-aarch64 -cpu cortex-a76"), "cortex-a76", Some(a76)));
        legs.push((a64, Some("qemu-aarch64 -cpu max"), "max", Some(max)));
        legs.push((a64_sve, Some("qemu-aarch64 -cpu cortex-a76"), "cortex-a76", None));
        legs.push((a64_sve, Some("qemu-aarch64 -cpu max"), "max", Some(max)));
    } else {
        eprintln!("note: aarch64-linux binaries cannot run under qemu-aarch64 here: build only");
    }

    let mut failures = Vec::new();
    let mut runs = 0;
    for target in [v2, v3, a64, a64_sve] {
        for release in [false, true] {
            let mode = if release { "release" } else { "debug" };
            let out = build(&src, target, release);
            let bin = tmp.path().join(format!("cpu-start-{target}-{mode}"));
            out.image.as_ref().unwrap().write_file(&bin).unwrap();
            for (t, runner, label, cpu) in &legs {
                if *t != target {
                    continue;
                }
                let cases = match cpu {
                    Some(c) => cases_for(target, *c),
                    // checked before STAVE_CPU is read
                    None => [None, Some("-avx3")]
                        .into_iter()
                        .map(|e| Case {
                            stave_cpu: e,
                            trace: false,
                            exit: 101,
                            stdout: String::new(),
                            stderr: String::new(),
                        })
                        .collect(),
                };
                for case in &cases {
                    let got = run(&bin, *runner, case.stave_cpu, case.trace);
                    runs += 1;
                    let r = if cpu.is_none() {
                        let ok = got.exit == 101
                            && got.stdout.is_empty()
                            && got.stderr.starts_with(unsupported)
                            && got.stderr.lines().count() == 2;
                        if ok {
                            Ok(())
                        } else {
                            Err(format!("exit {}\n--- stderr\n{}", got.exit, got.stderr))
                        }
                    } else {
                        check_run(&got, case, *label == "max")
                    };
                    if let Err(e) = r {
                        failures.push(format!(
                            "[{target}, {mode}, {label}, STAVE_CPU={:?}, trace {}]: {e}",
                            case.stave_cpu, case.trace
                        ));
                    }
                }
            }
        }
    }
    eprintln!("{runs} runs");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// A program above its arch's lowest baseline has start code that checks the CPU, even without
/// dispatch or `std.cpu`; one at the lowest baseline has no `runtime.cpu` code.
#[test]
fn baseline_check_only_above_the_lowest_baseline() {
    let hello = repo().join("tests/conformance/run/hello");
    for (t, check) in [
        ("x86_64-linux", false),
        ("x86_64-linux-v3", true),
        ("x86_64-linux-v4", true),
        ("aarch64-linux", false),
        ("aarch64-linux+sve", true),
        ("aarch64-linux+sve2", true),
    ] {
        let target = Triple::parse(t).unwrap();
        for release in [false, true] {
            let out = build(&hello, target, release);
            let m = out.module.as_ref().unwrap();
            if check {
                assert!(has_body(m, "runtime.cpu.init"), "{t}, release {release}: {:?}", cpu_fns(m));
                // nothing reads the word
                assert!(!has_body(m, "runtime.cpu.features"), "{t}, release {release}: {:?}", cpu_fns(m));
            } else {
                assert!(cpu_fns(m).is_empty(), "{t}, release {release}: has {:?}", cpu_fns(m));
            }
        }
    }
}
