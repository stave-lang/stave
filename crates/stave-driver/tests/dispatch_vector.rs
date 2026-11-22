//! Dispatch arms with 256-bit vector bodies, end to end: `tests/conformance/dispatch/
//! run_vector_arms.stv` (`needs(avx2)` and `needs(avx2, fma)` arms over f32x8 / i32x8, closures
//! with 256-bit ops under a baseline fn and `thread.par_for_mut`, 256-bit values live across
//! calls into baseline code) and `run_default_arm_baseline.stv` (a `_` target that needs
//! `sse42`), debug and release, for x86_64-linux natively (the arm the host CPU has) and under
//! `qemu-x86_64 -cpu Nehalem` (the `_` arm), for x86_64-linux-v3 (dispatch pruned to the 256-bit
//! arm at compile time), and for aarch64-linux under qemu (the x86-64 arms dropped). Each program
//! checks its results against a scalar reference itself and, given an argument, names the arm
//! that ran on each line.
//!
//! qemu models differ between qemu versions: only features stable for a model are asserted
//! (Nehalem has no avx2). A leg that cannot run here is skipped with a message, or fails when
//! `STAVE_REQUIRE_LEGS` lists it (the qemu-x86_64 runs are not a leg: skipped without it).

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::exec::{self, RunSpec, TempDir};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::{Func, Module, Op, Operand};
use stave_target::{Arch, Triple};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn program(name: &str) -> PathBuf {
    repo().join("tests/conformance/dispatch").join(name)
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{}: diagnostics {:?}", p.display(), out.diagnostics);
    assert!(
        out.failure.is_none(),
        "{} ({target}, release {release}): {}",
        p.display(),
        out.failure.as_ref().unwrap()
    );
    out
}

/// The paths of the fns `path`'s body calls directly (`None`: no such fn with a body).
fn callees(m: &Module, path: &str) -> Option<Vec<String>> {
    calls_of(m, m.funcs.iter().find(|f| f.path == path)?)
}

/// The paths of the fns `f`'s body calls directly, the dispatch trace calls left out (`None`:
/// no body).
fn calls_of(m: &Module, f: &Func) -> Option<Vec<String>> {
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

/// (avx2, fma) of the host CPU.
fn host_x86() -> (bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        (std::arch::is_x86_feature_detected!("avx2"), std::arch::is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (false, false)
    }
}

/// The arm each line of `run_vector_arms` must name when the CPU has (avx2, fma).
fn vector_arms(avx2: bool, fma: bool) -> Vec<(&'static str, &'static str)> {
    let v = if avx2 { "avx2" } else { "base" };
    let d = if avx2 && fma { "avx2+fma" } else { v };
    vec![("fsum", v), ("istats", v), ("dot", d), ("axpy_clamp", v), ("rows", v), ("live", v)]
}

/// How the dispatches of `run_vector_arms` are lowered on `target`: on x86_64-linux a call goes
/// to the dispatch, which tests the feature word; on x86_64-linux-v3 the baseline covers the
/// first arm, which the caller calls directly; on AArch64 every arm but `_` is dropped.
fn check_ir(m: &Module, target: Triple) -> Result<(), String> {
    let calls = |p: &str| callees(m, p).ok_or_else(|| format!("no `{p}` with a body"));
    let calls_fn = |p: &str, c: &str| -> Result<bool, String> { Ok(calls(p)?.iter().any(|x| x == c)) };
    for (tester, disp, arms) in [
        ("main.test_fsum", "main.fsum", &["main.fsum_v3", "main.fsum_base"][..]),
        ("main.test_dot", "main.dot", &["main.dot_fma", "main.dot_v3", "main.dot_base"][..]),
        ("main.test_rows", "main.rows", &["main.rows_v3", "main.rows_base"][..]),
    ] {
        let (direct, not) = match target.arch() {
            Arch::X86_64 if target.baseline().contains(stave_target::Feature::Avx2) => (arms[0], disp),
            Arch::X86_64 => {
                let want: Vec<String> = arms.iter().map(|a| a.to_string()).collect();
                let got = calls(disp)?;
                if got != want {
                    return Err(format!("`{disp}` calls {got:?}, want {want:?}"));
                }
                (disp, arms[0])
            }
            _ => (*arms.last().unwrap(), disp),
        };
        if !calls_fn(tester, direct)? || calls_fn(tester, not)? {
            return Err(format!("`{tester}` calls {:?}: want `{direct}`, not `{not}`", calls(tester)?));
        }
    }
    Ok(())
}

/// Runs `bin` with an argument and checks every line: it ends in ` ok`, names the arm `want`
/// gives for its first word, and without the arm equals `golden`.
fn check_run(bin: &Path, runner: Option<&str>, want: &[(&str, &str)], golden: &str) -> Result<(), String> {
    let args = ["arms".to_string()];
    let spec = RunSpec { args: &args, runner, cwd: None, inherit: false };
    let got = exec::run(bin, &spec).map_err(|e| format!("{}: {e}", bin.display()))?;
    let out = String::from_utf8_lossy(&got.stdout).into_owned();
    let show = || {
        format!("exit {}\n--- stdout\n{out}--- stderr\n{}", got.exit, String::from_utf8_lossy(&got.stderr))
    };
    if got.exit != 0 {
        return Err(show());
    }
    let mut plain = String::new();
    let lines: Vec<&str> = out.lines().collect();
    if lines.len() != want.len() {
        return Err(format!("{} lines, want {}\n{}", lines.len(), want.len(), show()));
    }
    for (line, (name, arm)) in lines.iter().zip(want) {
        let w: Vec<&str> = line.split(' ').collect();
        if w.len() < 3 || w[0] != *name || w[w.len() - 2] != *arm || w[w.len() - 1] != "ok" {
            return Err(format!("line `{line}`: want `{name} .. {arm} ok`\n{}", show()));
        }
        let mut keep: Vec<&str> = w[..w.len() - 2].to_vec();
        keep.push("ok");
        plain.push_str(&keep.join(" "));
        plain.push('\n');
    }
    if plain != golden {
        return Err(format!("results differ from the golden file:\n{plain}--- want\n{golden}"));
    }
    Ok(())
}

/// One run of a binary: (runner, the arm each line must name, label).
type Run = (Option<String>, Vec<(&'static str, &'static str)>, String);

/// Every run of one program: `runs(target)` gives the runs of `target`'s binaries.
fn run_program(
    name: &str,
    targets: &[Triple],
    runs: &dyn Fn(Triple) -> Vec<Run>,
    ir: &dyn Fn(&Module, Triple) -> Result<(), String>,
) -> (usize, Vec<String>) {
    let src = program(&format!("{name}.stv"));
    let golden = std::fs::read_to_string(program(&format!("{name}.stdout"))).unwrap();
    let tmp = TempDir::new("disp-vec").unwrap();
    let mut failures = Vec::new();
    let mut n = 0;
    for &target in targets {
        for release in [false, true] {
            let mode = if release { "release" } else { "debug" };
            let out = build(&src, target, release);
            if let Err(e) = ir(out.module.as_ref().unwrap(), target) {
                failures.push(format!("{name} [{target} {mode}] IR: {e}"));
            }
            let bin = tmp.path().join(format!("{name}-{target}-{mode}"));
            out.image.as_ref().unwrap().write_file(&bin).unwrap();
            if target == Triple::parse("x86_64-linux").unwrap() {
                // a fn with 256-bit code calls and returns into baseline code: vzeroupper
                let bytes = std::fs::read(&bin).unwrap();
                if !bytes.windows(3).any(|w| w == [0xc5, 0xf8, 0x77]) {
                    failures.push(format!("{name} [{target} {mode}]: no vzeroupper in the image"));
                }
            }
            for (runner, want, label) in runs(target) {
                n += 1;
                if let Err(e) = check_run(&bin, runner.as_deref(), &want, &golden) {
                    failures.push(format!("{name} [{target} {mode} {label}]: {e}"));
                }
            }
        }
    }
    (n, failures)
}

#[test]
fn vector_arms_match_scalar_references() {
    let v2 = Triple::parse("x86_64-linux").unwrap();
    let v3 = Triple::parse("x86_64-linux-v3").unwrap();
    let a64 = Triple::parse("aarch64-linux").unwrap();
    let native_v2 = legs::gate(v2.name(), exec::can_run(v2));
    let native_v3 = legs::gate(v3.name(), exec::can_run(v3));
    let qemu_a64 = legs::gate(a64.name(), cfg!(target_os = "linux") && have("qemu-aarch64"));
    let qemu_x86 = cfg!(target_os = "linux") && have("qemu-x86_64");
    if !qemu_x86 {
        eprintln!("note: no qemu-x86_64: the Nehalem runs are skipped");
    }
    let (avx2, fma) = host_x86();

    let runs = |t: Triple| {
        let mut r = Vec::new();
        if t == v2 && native_v2 {
            r.push((None, vector_arms(avx2, fma), "native".to_string()));
        }
        if t == v2 && qemu_x86 {
            r.push((
                Some("qemu-x86_64 -cpu Nehalem".to_string()),
                vector_arms(false, false),
                "Nehalem".into(),
            ));
        }
        if t == v3 && native_v3 {
            r.push((None, vector_arms(true, true), "native".into()));
        }
        if t == a64 && qemu_a64 {
            r.push((Some("qemu-aarch64".to_string()), vector_arms(false, false), "qemu".into()));
        }
        r
    };
    let (n, mut failures) = run_program("run_vector_arms", &[v2, v3, a64], &runs, &check_ir);

    let base_runs = |t: Triple| {
        let arm = |avx2: bool| vec![("count", if avx2 { "avx2" } else { "sse42" })];
        let mut r = Vec::new();
        if t == v2 && native_v2 {
            r.push((None, arm(avx2), "native".to_string()));
        }
        if t == v2 && qemu_x86 {
            r.push((Some("qemu-x86_64 -cpu Nehalem".to_string()), arm(false), "Nehalem".into()));
        }
        r
    };
    let no_ir = |_: &Module, _: Triple| Ok(());
    let (m, more) = run_program("run_default_arm_baseline", &[v2], &base_runs, &no_ir);
    failures.extend(more);
    eprintln!("{} runs", n + m);
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// The features of each instance of the generics `run_generic_needs` uses at f32x8 and f32x4:
/// on x86_64-linux the f32x8 instance inherits `avx2` and the f32x4 one needs nothing; on
/// x86_64-linux-v3 the baseline covers them (nothing inherited); on AArch64 the `avx2` arm is
/// dropped with its f32x8 instances. `first` only borrows: no instance of it inherits.
fn check_generic_ir(m: &Module, target: Triple) -> Result<(), String> {
    let avx2 = |p: &str| -> Vec<bool> {
        let mut v: Vec<bool> = m
            .funcs
            .iter()
            .filter(|f| f.path == p && f.body.is_some())
            .map(|f| f.features.contains(stave_target::Feature::Avx2))
            .collect();
        v.sort();
        v
    };
    // x86_64-linux: one instance with avx2, one without; elsewhere none with (on -v3 a release
    // build keeps only the f32x8 instances: the dispatch always takes the avx2 arm)
    let inherits = target.arch() == Arch::X86_64 && !target.baseline().contains(stave_target::Feature::Avx2);
    let ok = |v: &[bool]| if inherits { v == [false, true] } else { !v.is_empty() && v.iter().all(|b| !b) };
    for p in ["main.id", "main.fold", "main.ping", "main.pong", "main.via", "std.mem.swap"] {
        if !ok(&avx2(p)) {
            return Err(format!("`{p}` instances with avx2: {:?}", avx2(p)));
        }
    }
    let first = avx2("main.first");
    if first.is_empty() || first.iter().any(|b| *b) {
        return Err(format!("`main.first` instances with avx2: {first:?}"));
    }
    Ok(())
}

/// Generic instances at f32x8 under a `needs(avx2)` arm and at f32x4 under the `_` arm
/// (`run_generic_needs`): the arm the CPU has runs, with the instance feature sets of
/// [`check_generic_ir`].
#[test]
fn generic_instances_inherit_arm_features() {
    let v2 = Triple::parse("x86_64-linux").unwrap();
    let v3 = Triple::parse("x86_64-linux-v3").unwrap();
    let a64 = Triple::parse("aarch64-linux").unwrap();
    let native_v2 = legs::gate(v2.name(), exec::can_run(v2));
    let native_v3 = legs::gate(v3.name(), exec::can_run(v3));
    let qemu_a64 = legs::gate(a64.name(), cfg!(target_os = "linux") && have("qemu-aarch64"));
    let qemu_x86 = cfg!(target_os = "linux") && have("qemu-x86_64");
    let (avx2, _) = host_x86();
    let names = [
        "id",
        "pair",
        "trait",
        "fold",
        "first",
        "recursion",
        "turbofish",
        "swap",
        "vec",
        "unwrap",
        "fn_value",
        "drop",
    ];
    let arms = |wide: bool| -> Vec<(&'static str, &'static str)> {
        names.iter().map(|n| (*n, if wide { "avx2" } else { "base" })).collect()
    };
    let runs = |t: Triple| {
        let mut r = Vec::new();
        if t == v2 && native_v2 {
            r.push((None, arms(avx2), "native".to_string()));
        }
        if t == v2 && qemu_x86 {
            r.push((Some("qemu-x86_64 -cpu Nehalem".to_string()), arms(false), "Nehalem".into()));
        }
        if t == v3 && native_v3 {
            r.push((None, arms(true), "native".into()));
        }
        if t == a64 && qemu_a64 {
            r.push((Some("qemu-aarch64".to_string()), arms(false), "qemu".into()));
        }
        r
    };
    let (n, failures) = run_program("run_generic_needs", &[v2, v3, a64], &runs, &check_generic_ir);
    eprintln!("{n} runs");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// The `std.slice` methods reach their 256-bit loops through dispatch fns: on x86_64-linux the
/// dispatch tests the feature word and picks the `needs(avx2)` loop or the 128-bit one; on
/// x86_64-linux-v3 the method calls the 256-bit loop directly; AArch64 builds no 256-bit loop.
/// (`interp_vs_native` runs `tests/conformance/simd/combinators_*.stv` natively, under
/// `-cpu Nehalem`, and in the interpreter.)
#[test]
fn slice_methods_dispatch_to_256_bit_loops() {
    let src = repo().join("tests/conformance/simd/combinators_wide.stv");
    let fns = [
        "sum_u8",
        "min_i32",
        "max_f32",
        "find_u64",
        "fill_u16",
        "eq_f64",
        "bytes_eq",
        "copy_lanes",
        "find_any_byte",
    ];
    for t in ["x86_64-linux", "x86_64-linux-v3", "aarch64-linux"] {
        let target = Triple::parse(t).unwrap();
        for release in [false, true] {
            let out = build(&src, target, release);
            let m = out.module.as_ref().unwrap();
            let callers_of = |g: &str| -> Vec<String> {
                m.funcs
                    .iter()
                    .filter(|h| calls_of(m, h).is_some_and(|c| c.iter().any(|x| x == g)))
                    .map(|h| h.path.clone())
                    .collect()
            };
            for f in fns {
                let (base, wide, disp) =
                    (format!("std.slice.{f}"), format!("std.slice.{f}_v3"), format!("std.slice.{f}_cpu"));
                let at = format!("{t} release {release}");
                match (target.arch(), target.baseline().contains(stave_target::Feature::Avx2)) {
                    (Arch::X86_64, false) => {
                        let want = vec![wide.clone(), base.clone()];
                        assert_eq!(callees(m, &disp), Some(want), "{at}: `{disp}`");
                        assert!(!callers_of(&disp).is_empty(), "{at}: nothing calls `{disp}`");
                    }
                    (Arch::X86_64, true) => {
                        assert!(callees(m, &disp).is_none(), "{at}: `{disp}` is built");
                        assert!(!callers_of(&wide).is_empty(), "{at}: nothing calls `{wide}`");
                    }
                    _ => assert!(callees(m, &wide).is_none(), "{at}: `{wide}` is built"),
                }
            }
        }
    }
}
