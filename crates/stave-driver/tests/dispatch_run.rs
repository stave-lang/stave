//! `dispatch fn` end to end: static ELF executables, debug and release, run natively or under a
//! qemu `-cpu` model, pick the arm the CPU's feature word selects; the IR shows the static
//! pruning (target baseline, the dispatch's own needs, the caller's needs).
//!
//! The arms have scalar bodies and only report which arm ran. qemu models differ between qemu
//! versions, so only features stable for a model are asserted: Nehalem has no avx2;
//! cortex-a76 has no sve; max has sve.
//!
//! A leg that cannot run here is skipped with a message, or fails when `STAVE_REQUIRE_LEGS`
//! lists it (the qemu-x86_64 runs are not a leg: they are skipped without qemu-x86_64).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::exec::{self, RunSpec, TempDir};
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::{Module, Op, Operand};
use stave_target::{Arch, Triple};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Prints `<label> <arm>` per way of reaching a dispatch: `pick` (a call from `main`), `value`
/// (through a fn value), `word` (the arm `std.cpu.features()` selects), `own` (a dispatch with
/// its own needs, x86-64 only), `direct` (a call from a `needs(avx2)` fn).
const PROGRAM: &str = r#"import std.cpu;
import std.io;

fn on_avx2() -> u32 needs(avx2) {
    return 3;
}

fn on_sve() -> u32 needs(sve) {
    return 2;
}

fn on_dotprod() -> u32 needs(dotprod) {
    return 1;
}

fn on_base() -> u32 {
    return 0;
}

dispatch fn pick() -> u32 { avx2 => on_avx2, sve => on_sve, dotprod => on_dotprod, _ => on_base, }

fn from_word(w: u64) -> u32 {
    if w & cpu.AVX2 != 0 { return 3; }
    if w & cpu.SVE != 0 { return 2; }
    if w & cpu.DOTPROD != 0 { return 1; }
    return 0;
}

fn own_v3() -> u32 needs(avx2) {
    return 3;
}

fn own_avx() -> u32 needs(avx) {
    return 4;
}

dispatch fn own() -> u32 needs(avx) { avx2 => own_v3, _ => own_avx, }

fn via_avx() -> u32 needs(avx) {
    return own();
}

dispatch fn outer() -> u32 { avx => via_avx, _ => on_base, }

fn direct_user() -> u32 needs(avx2) {
    return pick();
}

dispatch fn direct() -> u32 { avx2 => direct_user, _ => on_base, }

fn show(label: &str, k: u32) {
    io.print(label);
    if k == 4 { io.print(" avx\n"); return; }
    if k == 3 { io.print(" avx2\n"); return; }
    if k == 2 { io.print(" sve\n"); return; }
    if k == 1 { io.print(" dotprod\n"); return; }
    io.print(" base\n");
}

fn main() -> i32 {
    show("pick", pick());
    let f: fn() -> u32 = pick;
    show("value", f());
    show("word", from_word(cpu.features()));
    show("own", outer());
    show("direct", direct());
    return 0;
}
"#;

/// A dispatch and nothing else that reads the feature word, called and used as a fn value.
const PRUNED: &str = r#"import std.io;

fn on_avx2() -> u32 needs(avx2) {
    return 3;
}

fn on_base() -> u32 {
    return 0;
}

dispatch fn pick() -> u32 { avx2 => on_avx2, _ => on_base, }

fn main() -> i32 {
    let f: fn() -> u32 = pick;
    if pick() == 3 && f() == 3 { io.print("avx2\n"); } else { io.print("base\n"); }
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

/// The paths of the fns `path`'s body calls directly, in instruction order, without the
/// dispatch trace calls (`runtime.cpu.trace`; `None`: no such fn with a body).
fn callees(m: &Module, path: &str) -> Option<Vec<String>> {
    let f = m.funcs.iter().find(|f| f.path == path)?;
    let b = f.body.as_ref()?;
    let mut out = Vec::new();
    for (i, op) in b.insts.op.iter().enumerate() {
        if *op == Op::Call
            && let Some(Operand::Func(g)) = b.args(i as u32).first()
            && let Some(g) = m.func(*g)
            && g.path != "runtime.cpu.trace"
        {
            out.push(g.path.clone());
        }
    }
    Some(out)
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

/// `<label> <arm>` lines as a map.
fn parse(out: &[u8]) -> BTreeMap<String, String> {
    String::from_utf8_lossy(out)
        .lines()
        .filter_map(|l| l.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
        .collect()
}

/// x86-64 features of the host from `/proc/cpuinfo`: (avx, avx2).
fn host_avx() -> (bool, bool) {
    let info = std::fs::read_to_string("/proc/cpuinfo").expect("/proc/cpuinfo");
    let line = info.lines().find(|l| l.starts_with("flags")).expect("/proc/cpuinfo: no flags line");
    let flags: Vec<&str> = line.split_once(':').unwrap().1.split_whitespace().collect();
    (flags.contains(&"avx"), flags.contains(&"avx2"))
}

/// What one run must print.
#[derive(Clone, Copy, Debug)]
enum Want {
    /// x86-64 with (avx, avx2).
    X86(bool, bool),
    /// AArch64: whether the model has sve.
    A64 { sve: bool },
}

fn check(got: &BTreeMap<String, String>, want: Want) -> Result<(), String> {
    let get = |k: &str| got.get(k).map(String::as_str).unwrap_or("?");
    let pick = get("pick");
    if get("value") != pick {
        return Err(format!("value {} != pick {pick}", get("value")));
    }
    // the feature word the dispatch read is std.cpu's
    if get("word") != pick {
        return Err(format!("word {} != pick {pick}", get("word")));
    }
    match want {
        Want::X86(avx, avx2) => {
            let a = if avx2 { "avx2" } else { "base" };
            let own = if avx2 {
                "avx2"
            } else if avx {
                "avx"
            } else {
                "base"
            };
            for (k, v) in [("pick", a), ("own", own), ("direct", a)] {
                if get(k) != v {
                    return Err(format!("{k}: {} (want {v})", get(k)));
                }
            }
        }
        Want::A64 { sve } => {
            if (pick == "sve") != sve {
                return Err(format!("pick {pick}: sve {}expected", if sve { "" } else { "not " }));
            }
            for k in ["own", "direct"] {
                if get(k) != "base" {
                    return Err(format!("{k}: {} (want base)", get(k)));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn dispatch_picks_by_cpu() {
    let tmp = TempDir::new("disp").unwrap();
    let src = tmp.path().join("dispatch.stv");
    std::fs::write(&src, PROGRAM).unwrap();
    let v2 = Triple::parse("x86_64-linux").unwrap();
    let v3 = Triple::parse("x86_64-linux-v3").unwrap();
    let a64 = Triple::parse("aarch64-linux").unwrap();
    let native_v2 = legs::gate(v2.name(), exec::can_run(v2) && Path::new("/proc/cpuinfo").is_file());
    let native_v3 = legs::gate(v3.name(), exec::can_run(v3) && Path::new("/proc/cpuinfo").is_file());
    let qemu_a64 = legs::gate(a64.name(), cfg!(target_os = "linux") && have("qemu-aarch64"));
    let qemu_x86 = cfg!(target_os = "linux") && have("qemu-x86_64");

    // (target, runner, expectation, label)
    let mut cases: Vec<(Triple, Option<String>, Want, &str)> = Vec::new();
    if native_v2 {
        let (avx, avx2) = host_avx();
        cases.push((v2, None, Want::X86(avx, avx2), "native"));
    }
    if native_v3 {
        cases.push((v3, None, Want::X86(true, true), "native"));
    }
    if qemu_x86 {
        cases.push((v2, Some("qemu-x86_64 -cpu Nehalem".into()), Want::X86(false, false), "Nehalem"));
    } else {
        eprintln!("note: no qemu-x86_64: the Nehalem runs are skipped");
    }
    if qemu_a64 {
        for (model, sve) in [("cortex-a76", false), ("max", true)] {
            cases.push((a64, Some(format!("qemu-aarch64 -cpu {model}")), Want::A64 { sve }, model));
        }
    } else {
        eprintln!("note: aarch64-linux binaries cannot run under qemu-aarch64 here: build only");
    }

    let mut failures = Vec::new();
    let mut runs = 0;
    let mut picks: BTreeMap<&str, String> = BTreeMap::new();
    for target in [v2, v3, a64] {
        for release in [false, true] {
            let mode = if release { "release" } else { "debug" };
            let out = build(&src, target, release);
            let m = out.module.as_ref().unwrap();
            check_ir(m, target).unwrap_or_else(|e| panic!("[{target}, {mode}] IR: {e}"));
            let bin = tmp.path().join(format!("disp-{target}-{mode}"));
            out.image.as_ref().unwrap().write_file(&bin).unwrap();
            for (t, runner, want, label) in &cases {
                if *t != target {
                    continue;
                }
                let spec = RunSpec { args: &[], runner: runner.as_deref(), cwd: None, inherit: false };
                let got = exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
                runs += 1;
                let lines = parse(&got.stdout);
                let r = if got.exit != 0 { Err(format!("exit {}", got.exit)) } else { check(&lines, *want) };
                if let Err(e) = r {
                    failures.push(format!(
                        "[{target}, {mode}, {label}]: {e}\n--- stdout\n{}--- stderr\n{}",
                        String::from_utf8_lossy(&got.stdout),
                        String::from_utf8_lossy(&got.stderr)
                    ));
                }
                if target == a64 {
                    picks.insert(label, lines.get("pick").cloned().unwrap_or_default());
                }
            }
        }
    }
    if let (Some(a), Some(b)) = (picks.get("cortex-a76"), picks.get("max")) {
        assert_ne!(a, b, "cortex-a76 and max pick the same arm");
    }
    eprintln!("{runs} runs");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// The lowering of [`PROGRAM`] on `target`.
fn check_ir(m: &Module, target: Triple) -> Result<(), String> {
    let calls = |p: &str| callees(m, p).ok_or_else(|| format!("no `{p}` with a body"));
    let expect = |p: &str, want: &[&str]| -> Result<(), String> {
        let got = calls(p)?;
        if got != want {
            return Err(format!("`{p}` calls {got:?}, want {want:?}"));
        }
        Ok(())
    };
    // the program reads the word (`std.cpu`), so start code fills it
    if !has_body(m, "runtime.cpu.init") || !has_body(m, "runtime.cpu.features") {
        return Err(format!("no runtime.cpu.init/features: {:?}", cpu_fns(m)));
    }
    // `main` calls a dispatch whose first live arm the baseline covers as that arm, directly: the
    // dispatch is then not built (`pick` is, as a fn value)
    let main_calls = calls("main.main")?;
    let main_expect = |calls_these: &[&str], not_these: &[&str]| -> Result<(), String> {
        for c in calls_these {
            if !main_calls.iter().any(|x| x == c) {
                return Err(format!("`main.main` does not call `{c}`: {main_calls:?}"));
            }
        }
        for c in not_these {
            if main_calls.iter().any(|x| x == c) || has_body(m, c) {
                return Err(format!("`{c}` is called or built: {main_calls:?}"));
            }
        }
        Ok(())
    };
    match target.arch() {
        Arch::X86_64 if target.baseline().contains(stave_target::Feature::Avx2) => {
            // v3: the baseline covers every arm the chains reach first: no word load
            expect("main.pick", &["main.on_avx2"])?;
            let direct = ["main.on_avx2", "main.via_avx", "main.direct_user"];
            main_expect(&direct, &["main.outer", "main.direct"])?;
            // via_avx (needs avx) and the v3 baseline cover own's first arm
            expect("main.via_avx", &["main.own_v3"])?;
            if has_body(m, "main.own") {
                return Err("`main.own` is built".into());
            }
        }
        Arch::X86_64 => {
            // sve and dotprod are AArch64 arms, dropped; `_` ends the chain
            expect("main.pick", &["runtime.cpu.features", "main.on_avx2", "main.on_base"])?;
            expect("main.own", &["runtime.cpu.features", "main.own_v3", "main.own_avx"])?;
            expect("main.outer", &["runtime.cpu.features", "main.via_avx", "main.on_base"])?;
            main_expect(&["main.pick", "main.outer", "main.direct"], &[])?;
            // a fn needing only avx does not cover own's first arm (avx2) on v2
            expect("main.via_avx", &["main.own"])?;
        }
        _ => {
            let all = ["runtime.cpu.features", "main.on_sve", "main.on_dotprod", "main.on_base"];
            expect("main.pick", &all)?;
            // every arm of `outer` and `direct` but `_` is dropped: their calls are `_`'s
            main_expect(&["main.pick", "main.on_base"], &["main.outer", "main.direct"])?;
            if has_body(m, "main.own") || has_body(m, "main.direct_user") {
                return Err("x86-64-only fns in an AArch64 build".into());
            }
        }
    }
    if target.arch() == Arch::X86_64 {
        // `needs(avx2)` covers pick's first arm: the call skips the dispatch
        expect("main.direct_user", &["main.on_avx2"])?;
    }
    Ok(())
}

/// A dispatch whose first arm the target baseline covers is one direct call: no feature-word
/// load (no `runtime.cpu.features`; start code still has `init`, which checks the v3 baseline),
/// and (release) the arms below it are not built.
#[test]
fn baseline_pruned_dispatch_has_no_cpu_code() {
    let tmp = TempDir::new("disp-pruned").unwrap();
    let src = tmp.path().join("pruned.stv");
    std::fs::write(&src, PRUNED).unwrap();
    let v3 = Triple::parse("x86_64-linux-v3").unwrap();
    let native = legs::gate(v3.name(), exec::can_run(v3));
    for release in [false, true] {
        let out = build(&src, v3, release);
        let m = out.module.as_ref().unwrap();
        assert_eq!(callees(m, "main.pick").unwrap(), ["main.on_avx2"], "release {release}");
        let main_calls = callees(m, "main.main").unwrap();
        assert!(main_calls.iter().any(|c| c == "main.on_avx2"), "release {release}: {main_calls:?}");
        assert!(!has_body(m, "runtime.cpu.features"), "release {release}: has {:?}", cpu_fns(m));
        if release {
            assert!(!has_body(m, "main.on_base"), "the pruned `_` arm is built");
        }
        if native {
            let bin = tmp.path().join(format!("pruned-{release}"));
            out.image.as_ref().unwrap().write_file(&bin).unwrap();
            let spec = RunSpec { args: &[], runner: None, cwd: None, inherit: false };
            let got = exec::run(&bin, &spec).unwrap();
            let stdout = String::from_utf8_lossy(&got.stdout).into_owned();
            assert_eq!((got.exit, stdout.as_str()), (0, "avx2\n"));
        }
    }
}

/// The IR interpreter stores no feature word: every dispatch that tests the word takes `_`.
#[test]
fn interpreter_takes_the_default_arm() {
    let tmp = TempDir::new("disp-interp").unwrap();
    let src = tmp.path().join("dispatch.stv");
    std::fs::write(&src, PROGRAM).unwrap();
    for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        let mut o = Options::new(Goal::Check).threads(Some(4));
        o.lower = true;
        o.target = Some(target);
        let out = pipeline::run(&src, &Library::in_dir(&repo()), &o).unwrap();
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let m = out.module.as_ref().unwrap();
        let r = interp::run(m, "main.main", target, &[], &[], Limits::default()).unwrap().ran;
        let want = "pick base\nvalue base\nword base\nown base\ndirect base\n";
        assert_eq!(String::from_utf8_lossy(&r.stdout), want, "{target}");
    }
}
