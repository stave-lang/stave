//! The `v4` leg end to end: AVX-512 code in static ELF executables, debug and release, run on
//! an AVX-512 CPU or under Intel SDE (`$STAVE_SDE -icx --`, tests/support/legs.rs):
//!
//! - a default-target `dispatch` with a 512-bit arm takes that arm under the v4 set and the
//!   scalar `_` arm elsewhere, with the same output;
//! - the same program built for `x86_64-linux-v4` (std and runtime too) prints the same;
//! - `std.cpu.features()` has the five v4 bits exactly where the CPU (or SDE's CPUID) does;
//! - a v4 binary on a CPU without the v4 set stops with `panic: UnsupportedCpu`.
//!
//! A leg that cannot run here is skipped with a message, or fails when `STAVE_REQUIRE_LEGS`
//! lists it. SDE emulates CPUID, not `/proc/cpuinfo`: the native expectations come from
//! `exec::can_run` and the standard library's detection.

use std::path::{Path, PathBuf};

#[path = "../../../tests/support/legs.rs"]
mod legs;

use legs::V4Run;
use stave_driver::exec::{self, Executable, RunSpec, TempDir};
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::{Module, Op, Operand};
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A `[f32]` of 37 elements (two full 16-lane chunks and a tail of 5) through `scale`: each
/// element above 0.5 doubled, the others less 1; prints the sum and the output array. The
/// inputs are odd quarters in -4.25..4.75, so every output and every partial sum is exact in
/// f32 whatever the summation order.
const KERNEL: &str = r#"import std.io;

const THR: f32 = 0.5;

fn scale_v4(xs: &[f32], out: &mut [f32], thr: f32) -> f32 needs(avx512f, avx512bw, avx512vl, avx512dq, avx512cd) {
    let t = f32x16.splat(thr);
    let two = f32x16.splat(2.0);
    let one = f32x16.splat(1.0);
    let zero = f32x16.splat(0.0);
    let mut acc = zero;
    let n = xs.len();
    let mut i: usize = 0;
    while i + 16 <= n {
        let v = f32x16.load(xs, i);
        let r = (v > t).select(v * two, v - one);
        r.store(out, i);
        acc = acc + r;
        i = i + 16;
    }
    if i < n {
        let m = mask32x16.first_n(n - i);
        let v = f32x16.load_masked(xs, i, m);
        let r = (v > t).select(v * two, v - one);
        r.store_masked(out, i, m);
        acc = acc + m.select(r, zero);
    }
    return acc.reduce_add();
}

fn scale_base(xs: &[f32], out: &mut [f32], thr: f32) -> f32 {
    let mut acc: f32 = 0.0;
    for i in 0..xs.len() {
        let x = xs[i];
        let r = if x > thr then x * 2.0 else x - 1.0;
        out[i] = r;
        acc = acc + r;
    }
    return acc;
}

dispatch fn scale(xs: &[f32], out: &mut [f32], thr: f32) -> f32 {
    avx512f + avx512bw + avx512vl + avx512dq + avx512cd => scale_v4,
    _ => scale_base,
}

fn main() -> i32 {
    let mut xs: [f32; 37] = [0.0; 37];
    let mut out: [f32; 37] = [0.0; 37];
    let mut k: u32 = 0;
    let mut x: f32 = -4.25;
    for i in 0usize..37 {
        xs[i] = x;
        k = k + 7;
        x = x + 3.5;
        if k >= 19 {
            k = k - 19;
            x = x - 9.5;
        }
    }
    let s = scale(&xs[0..37], &mut out[0..37], THR);
    io.print(f"sum {s}\n");
    io.print("out");
    for i in 0usize..37 {
        let y = out[i];
        io.print(f" {y}");
    }
    io.print("\n");
    return 0;
}
"#;

/// Prints `<feature> <0|1>` for the five features of the v4 set.
const FEATURES: &str = r#"import std.cpu;
import std.io;

fn bit(name: &str, w: u64, m: u64) {
    io.print(name);
    if w & m != 0 { io.print(" 1\n"); } else { io.print(" 0\n"); }
}

fn main() -> i32 {
    let w = cpu.features();
    bit("avx512f", w, cpu.AVX512F);
    bit("avx512bw", w, cpu.AVX512BW);
    bit("avx512vl", w, cpu.AVX512VL);
    bit("avx512dq", w, cpu.AVX512DQ);
    bit("avx512cd", w, cpu.AVX512CD);
    return 0;
}
"#;

const V4_SET: [&str; 5] = ["avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512cd"];

/// [`KERNEL`]'s stdout.
fn kernel_stdout() -> String {
    let mut xs = [0f32; 37];
    let (mut k, mut x) = (0u32, -4.25f32);
    for v in &mut xs {
        *v = x;
        k += 7;
        x += 3.5;
        if k >= 19 {
            k -= 19;
            x -= 9.5;
        }
    }
    let out: Vec<f32> = xs.iter().map(|&x| if x > 0.5 { x * 2.0 } else { x - 1.0 }).collect();
    let sum: f32 = out.iter().sum();
    // Stave's float text and Rust's `{:?}` agree on these values: a decimal point, no exponent
    let mut s = format!("sum {sum:?}\nout");
    for y in &out {
        s.push_str(&format!(" {y:?}"));
    }
    s.push('\n');
    s
}

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{} ({target}): diagnostics {:?}", p.display(), out.diagnostics);
    let mode = if release { "release" } else { "debug" };
    assert!(out.failure.is_none(), "{} ({target}, {mode}): {}", p.display(), out.failure.as_ref().unwrap());
    assert!(out.image.is_some());
    out
}

/// The paths of the fns `path`'s body calls directly, without the dispatch trace calls (`None`:
/// no such fn with a body).
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

struct Ran {
    exit: i32,
    stdout: String,
    stderr: String,
}

impl Ran {
    fn of(r: exec::Ran) -> Ran {
        Ran {
            exit: r.exit,
            stdout: String::from_utf8_lossy(&r.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&r.stderr).into_owned(),
        }
    }

    /// The `dispatch: ` lines of stderr (SDE may add its own lines).
    fn trace(&self) -> Vec<&str> {
        self.stderr.lines().filter(|l| l.starts_with("dispatch: ")).collect()
    }

    fn show(&self) -> String {
        format!("exit {}\n--- stdout\n{}--- stderr\n{}", self.exit, self.stdout, self.stderr)
    }
}

/// Runs the file `bin` (under `runner`) with `STAVE_DISPATCH_TRACE=1` when `trace`, without
/// `STAVE_CPU`. ETXTBSY (another test thread's fork still holds a just-written binary open) is
/// retried.
fn run(bin: &Path, runner: Option<&str>, trace: bool) -> Ran {
    let spec = RunSpec { args: &[], runner, cwd: None, inherit: false };
    let mut tries = 0;
    loop {
        let mut cmd = exec::command(bin, &spec).unwrap();
        cmd.env_remove("STAVE_CPU").env_remove("STAVE_DISPATCH_TRACE");
        if trace {
            cmd.env("STAVE_DISPATCH_TRACE", "1");
        }
        match cmd.stdin(std::process::Stdio::null()).output() {
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

fn triple(name: &str) -> Triple {
    Triple::parse(name).unwrap()
}

/// Whether default-target binaries run natively here (the `v2` leg).
fn native_v2() -> bool {
    let v2 = triple("x86_64-linux");
    legs::gate(v2.name(), exec::can_run(v2))
}

/// How the `v4` leg runs here (`None`: skipped, with a note).
fn v4_leg(what: &str) -> Option<V4Run> {
    let how = legs::v4(exec::can_run(triple("x86_64-linux-v4")));
    if how.is_none() {
        eprintln!("note: no AVX-512 CPU and no STAVE_SDE: {what} skipped");
    }
    how
}

fn runner_of(how: &V4Run) -> Option<&str> {
    match how {
        V4Run::Native => None,
        V4Run::Runner(r) => Some(r.as_str()),
    }
}

/// The default-target dispatch takes the 512-bit arm exactly where the CPU has the v4 set, and
/// both arms print the same.
#[test]
fn dispatch_v4_arm() {
    let tmp = TempDir::new("v4-disp").unwrap();
    let src = tmp.path().join("kernel.stv");
    std::fs::write(&src, KERNEL).unwrap();
    let want = kernel_stdout();
    let v4_cpu = exec::can_run(triple("x86_64-linux-v4"));
    // the IR interpreter sees no CPU features: it takes `_`
    let mut o = Options::new(Goal::Check).threads(Some(4));
    o.lower = true;
    o.target = Some(triple("x86_64-linux"));
    let checked = pipeline::run(&src, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!checked.has_errors(), "{:?}", checked.diagnostics);
    let m = checked.module.as_ref().unwrap();
    let r = interp::run(m, "main.main", triple("x86_64-linux"), &[], &[], Limits::default()).unwrap().ran;
    assert_eq!((r.exit, String::from_utf8_lossy(&r.stdout).into_owned()), (0, want.clone()), "interpreter");

    let arm =
        |v4: bool| format!("dispatch: main.scale -> main.{}", if v4 { "scale_v4" } else { "scale_base" });

    // (runner, whether the run sees the v4 set, label)
    let mut cases: Vec<(Option<String>, bool, &str)> = Vec::new();
    if native_v2() {
        cases.push((None, v4_cpu, "native"));
    } else {
        eprintln!("note: x86_64-linux binaries cannot run here: native runs skipped");
    }
    match v4_leg("the 512-bit arm's run") {
        Some(V4Run::Runner(r)) => cases.push((Some(r), true, "sde")),
        // the native case runs the 512-bit arm
        Some(V4Run::Native) => {}
        None => {}
    }

    let mut failures = Vec::new();
    let mut runs = 0;
    for release in [false, true] {
        let mode = if release { "release" } else { "debug" };
        let out = build(&src, triple("x86_64-linux"), release);
        let m = out.module.as_ref().unwrap();
        let calls = callees(m, "main.scale").expect("no `main.scale` with a body");
        assert_eq!(calls, ["main.scale_v4", "main.scale_base"], "{mode}");
        let bin = tmp.path().join(format!("kernel-{mode}"));
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        for (runner, v4, label) in &cases {
            let got = run(&bin, runner.as_deref(), true);
            runs += 1;
            let trace = arm(*v4);
            if got.exit != 0 || got.stdout != want || got.trace() != [trace.as_str()] {
                failures
                    .push(format!("[{mode}, {label}] want `{trace}`\n{}--- want stdout\n{want}", got.show()));
            }
        }
    }
    eprintln!("{runs} runs");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// [`KERNEL`] built for `x86_64-linux-v4`: the baseline covers the 512-bit arm, so `main` calls
/// it directly, and the program prints what the default-target build prints.
#[test]
fn whole_program_v4() {
    let tmp = TempDir::new("v4-whole").unwrap();
    let src = tmp.path().join("kernel.stv");
    std::fs::write(&src, KERNEL).unwrap();
    let want = kernel_stdout();
    let how = v4_leg("the x86_64-linux-v4 runs");
    let mut failures = Vec::new();
    for release in [false, true] {
        let mode = if release { "release" } else { "debug" };
        let out = build(&src, triple("x86_64-linux-v4"), release);
        let m = out.module.as_ref().unwrap();
        let main_calls = callees(m, "main.main").expect("no `main.main` with a body");
        assert!(main_calls.iter().any(|c| c == "main.scale_v4"), "{mode}: main calls {main_calls:?}");
        assert!(!has_body(m, "main.scale"), "{mode}: the resolved dispatch is built");
        let Some(how) = &how else { continue };
        let bin = tmp.path().join(format!("kernel-v4-{mode}"));
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        let got = run(&bin, runner_of(how), true);
        if got.exit != 0 || got.stdout != want || !got.trace().is_empty() {
            failures.push(format!("[{mode}, {how:?}]\n{}--- want stdout\n{want}", got.show()));
        }
    }
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// The v4 features of this CPU as the standard library detects them, in [`V4_SET`] order.
fn host_v4_bits() -> [bool; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        [
            std::arch::is_x86_feature_detected!("avx512f"),
            std::arch::is_x86_feature_detected!("avx512bw"),
            std::arch::is_x86_feature_detected!("avx512vl"),
            std::arch::is_x86_feature_detected!("avx512dq"),
            std::arch::is_x86_feature_detected!("avx512cd"),
        ]
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        [false; 5]
    }
}

fn features_stdout(bits: [bool; 5]) -> String {
    V4_SET.iter().zip(bits).map(|(f, b)| format!("{f} {}\n", u8::from(b))).collect()
}

/// `std.cpu.features()` of a default-target program: all five v4 bits under SDE; natively the
/// CPU's own (all five set iff `can_run(x86_64-linux-v4)`). The SDE run goes both through a
/// file and through an in-memory [`Executable`] (the runner gets `/dev/fd/N`).
#[test]
fn cpu_features_under_sde() {
    let tmp = TempDir::new("v4-feat").unwrap();
    let src = tmp.path().join("features.stv");
    std::fs::write(&src, FEATURES).unwrap();
    let native = native_v2();
    let how = v4_leg("the SDE runs");
    let host = host_v4_bits();
    assert_eq!(
        host.iter().all(|b| *b),
        exec::can_run(triple("x86_64-linux-v4")),
        "can_run(x86_64-linux-v4) disagrees with the detected features {host:?}"
    );
    let mut failures = Vec::new();
    let mut runs = Vec::new();
    for release in [false, true] {
        let mode = if release { "release" } else { "debug" };
        let out = build(&src, triple("x86_64-linux"), release);
        let image = out.image.as_ref().unwrap();
        let bin = tmp.path().join(format!("features-{mode}"));
        image.write_file(&bin).unwrap();
        let mut check = |label: &str, got: Ran, want: &str| {
            runs.push(format!("{mode}, {label}"));
            if got.exit != 0 || got.stdout != want {
                failures.push(format!("[{mode}, {label}]\n{}--- want stdout\n{want}", got.show()));
            }
        };
        if native {
            check("native", run(&bin, None, false), &features_stdout(host));
        }
        if let Some(V4Run::Runner(r)) = &how {
            let all = features_stdout([true; 5]);
            check("sde, file", run(&bin, Some(r), false), &all);
            let mut exe = Executable::new("features", image.bytes()).unwrap();
            let spec = RunSpec { args: &[], runner: Some(r), cwd: None, inherit: false };
            let label = if exe.in_memory() { "sde, memory" } else { "sde, executable on disk" };
            check(label, Ran::of(exe.run(&spec).unwrap()), &all);
        }
    }
    eprintln!("runs: {runs:?}");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// A v4 binary started on a CPU without the v4 set stops before `main`.
#[test]
fn v4_binary_on_plain_cpu() {
    if exec::can_run(triple("x86_64-linux-v4")) {
        eprintln!("note: this CPU has the v4 set: the UnsupportedCpu run is skipped");
        return;
    }
    if !native_v2() {
        eprintln!("note: x86_64-linux binaries cannot run here: the UnsupportedCpu run is skipped");
        return;
    }
    let tmp = TempDir::new("v4-plain").unwrap();
    let src = tmp.path().join("features.stv");
    std::fs::write(&src, FEATURES).unwrap();
    let unsupported = "panic: UnsupportedCpu at <unknown>:0:0 in <unknown> (thread 0)\n";
    for release in [false, true] {
        let out = build(&src, triple("x86_64-linux-v4"), release);
        let bin = tmp.path().join(format!("plain-{release}"));
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        let got = run(&bin, None, false);
        let ok = got.exit == 101
            && got.stdout.is_empty()
            && got.stderr.starts_with(unsupported)
            && got.stderr.lines().count() == 2;
        assert!(ok, "release {release}: {}", got.show());
    }
}
