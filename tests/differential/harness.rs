//! Differential test of hand-written SIMD kernels (`kernels/*.stv`) against their scalar
//! references. Each kernel file and `diff.stv` become one package; the cases of `gen.rs` go to
//! its `cases.bin`. The program prints one line per case, the vector kernels' tokens, ` =`, then
//! the references' tokens; every line must match itself, and every leg must print what the
//! interpreter prints. Legs: the stave-ir interpreter, then native debug and release builds for
//! x86_64-linux, x86_64-linux-v3 and aarch64-linux (under qemu-aarch64 on an x86-64 host);
//! a `_v3` kernel builds for x86_64-linux-v3 only. A `_disp` kernel (dispatch fns over
//! `needs(v4)`, `needs(avx2)` and `needs(sve)` arms and the kernels of `<k>.stv`) builds for every
//! target, and each binary runs several times with `STAVE_DISPATCH_TRACE=1`, under `STAVE_CPU`
//! masks, Intel SDE (the v4 leg) and qemu CPU models: each run must pick the arm that CPU allows
//! ([`gate_runs`]) and print what the interpreter prints. A leg that cannot run here is skipped
//! with a note, or fails when `STAVE_REQUIRE_LEGS` lists it.
//!
//! Env: `STAVE_DIFF_SEED` (u64), `STAVE_DIFF_FULL=1` (every misalignment), `STAVE_DIFF_RANDOM`
//! (random cases per kind, default 24), `STAVE_DIFF_KERNELS` (comma-separated names).
//! See README.md.

#[path = "gen.rs"]
mod generate;
#[path = "../support/legs.rs"]
mod legs;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use generate::{Elem, Kind, Param, Plan};
use stave_driver::build::Failure;
use stave_driver::exec::{self, Ran, RunSpec, TempDir};
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_target::Triple;

struct Kernel {
    name: &'static str,
    kinds: &'static [Kind],
}

const fn kind(arrays: &'static [Elem], lanes: usize, param: Param) -> Kind {
    Kind { arrays, lanes, param, palettes: &generate::PALETTES, extra: &[] }
}

/// The logical lane counts of std's float `sum` and `dot`, and the block of their two-level rule
/// (`SUM_BLOCK` in std/slice.stv), in elements.
const L_F32: usize = 32;
const L_F64: usize = 16;
const BLOCK: usize = generate::SUM_BLOCK;

/// Lengths at and past one block: two blocks, the second of one element; four blocks, the
/// last short; six blocks, a block count that is not a power of two. (The interpreter leg runs
/// them element by element, so they stay few.)
const BLOCK_LENS: &[usize] = &[BLOCK, BLOCK + 1, 3 * BLOCK + 5, 5 * BLOCK + L_F32 + 3];

const fn rkind(arrays: &'static [Elem], lanes: usize) -> Kind {
    Kind { arrays, lanes, param: Param::None, palettes: &generate::REDUCE_PALETTES, extra: BLOCK_LENS }
}

/// The cases of the float reduction kernels: f32 and f64 sums, then dot products. Lengths run
/// to 6L + 1, past the lengths (128 f32, 64 f64) where std's `sum` and `dot` go through
/// `dispatch`, so each std arm sees every tail length too, then [`BLOCK_LENS`].
const REDUCE_KINDS: &[Kind] = &[
    rkind(&[Elem::F32], 2 * L_F32),
    rkind(&[Elem::F64], 2 * L_F64),
    rkind(&[Elem::F32, Elem::F32], 2 * L_F32),
    rkind(&[Elem::F64, Elem::F64], 2 * L_F64),
];

/// Every kernel program, with the case kinds its `main` dispatches on (in order).
const KERNELS: &[Kernel] = &[
    Kernel {
        name: "sum",
        kinds: &[
            kind(&[Elem::F32], 4, Param::None),
            kind(&[Elem::F64], 2, Param::None),
            kind(&[Elem::I32], 4, Param::None),
            kind(&[Elem::U8], 16, Param::None),
        ],
    },
    Kernel {
        name: "dot",
        kinds: &[
            kind(&[Elem::F32, Elem::F32], 4, Param::None),
            kind(&[Elem::F64, Elem::F64], 2, Param::None),
            kind(&[Elem::I32, Elem::I32], 4, Param::None),
            kind(&[Elem::U8, Elem::U8], 16, Param::None),
        ],
    },
    Kernel {
        name: "minmax",
        kinds: &[
            kind(&[Elem::F32], 4, Param::None),
            kind(&[Elem::F64], 2, Param::None),
            kind(&[Elem::I32], 4, Param::None),
            kind(&[Elem::U8], 16, Param::None),
        ],
    },
    Kernel {
        name: "filter",
        kinds: &[
            kind(&[Elem::F32], 4, Param::F32),
            kind(&[Elem::I32], 4, Param::I32),
            kind(&[Elem::U8], 16, Param::Byte),
        ],
    },
    Kernel { name: "bytesearch", kinds: &[kind(&[Elem::U8], 16, Param::Byte)] },
    Kernel {
        name: "prefix",
        kinds: &[
            kind(&[Elem::F32], 4, Param::None),
            kind(&[Elem::F64], 2, Param::None),
            kind(&[Elem::I32], 4, Param::None),
        ],
    },
    Kernel { name: "histogram", kinds: &[kind(&[Elem::U8], 16, Param::None)] },
    Kernel { name: "matmul4", kinds: &[kind(&[Elem::F32], 4, Param::F32)] },
    Kernel { name: "utf8", kinds: &[kind(&[Elem::Utf8], 16, Param::None)] },
    Kernel { name: "mulhi", kinds: &[kind(&[Elem::I32, Elem::I32], 4, Param::None)] },
    Kernel { name: "divconst", kinds: &[kind(&[Elem::I32], 4, Param::None)] },
    Kernel {
        name: "combinators",
        kinds: &[
            kind(&[Elem::F32], 16, Param::F32),
            kind(&[Elem::F64], 8, Param::None),
            kind(&[Elem::I32], 4, Param::I32),
            kind(&[Elem::U8], 16, Param::Byte),
        ],
    },
    Kernel { name: "reduce", kinds: REDUCE_KINDS },
    Kernel {
        name: "stream_copy",
        kinds: &[
            kind(&[Elem::F32], 4, Param::None),
            kind(&[Elem::F64], 2, Param::None),
            kind(&[Elem::I32], 4, Param::None),
            kind(&[Elem::U8], 16, Param::None),
        ],
    },
    Kernel {
        name: "sum_v3",
        kinds: &[
            kind(&[Elem::F32], 8, Param::None),
            kind(&[Elem::F64], 4, Param::None),
            kind(&[Elem::I32], 8, Param::None),
            kind(&[Elem::U8], 32, Param::None),
        ],
    },
    Kernel {
        name: "dot_v3",
        kinds: &[
            kind(&[Elem::F32, Elem::F32], 8, Param::None),
            kind(&[Elem::F64, Elem::F64], 4, Param::None),
            kind(&[Elem::I32, Elem::I32], 8, Param::None),
            kind(&[Elem::U8, Elem::U8], 32, Param::None),
        ],
    },
    Kernel {
        name: "minmax_v3",
        kinds: &[
            kind(&[Elem::F32], 8, Param::None),
            kind(&[Elem::F64], 4, Param::None),
            kind(&[Elem::I32], 8, Param::None),
            kind(&[Elem::U8], 32, Param::None),
        ],
    },
    Kernel {
        name: "filter_v3",
        kinds: &[
            kind(&[Elem::F32], 8, Param::F32),
            kind(&[Elem::I32], 8, Param::I32),
            kind(&[Elem::U8], 32, Param::Byte),
        ],
    },
    Kernel { name: "bytesearch_v3", kinds: &[kind(&[Elem::U8], 32, Param::Byte)] },
    Kernel { name: "bytesearch_disp", kinds: &[kind(&[Elem::U8], 64, Param::Byte)] },
    Kernel {
        name: "sum_disp",
        kinds: &[kind(&[Elem::I32], 16, Param::None), kind(&[Elem::U8], 64, Param::None)],
    },
    Kernel { name: "reduce_disp", kinds: REDUCE_KINDS },
    Kernel {
        name: "prefix_v3",
        kinds: &[
            kind(&[Elem::F32], 8, Param::None),
            kind(&[Elem::F64], 4, Param::None),
            kind(&[Elem::I32], 8, Param::None),
        ],
    },
    Kernel { name: "histogram_v3", kinds: &[kind(&[Elem::U8], 32, Param::None)] },
    Kernel { name: "matmul4_v3", kinds: &[kind(&[Elem::F32], 8, Param::F32)] },
    Kernel { name: "utf8_v3", kinds: &[kind(&[Elem::Utf8], 32, Param::None)] },
    Kernel { name: "mulhi_v3", kinds: &[kind(&[Elem::I32, Elem::I32], 8, Param::None)] },
    Kernel { name: "divconst_v3", kinds: &[kind(&[Elem::I32], 8, Param::None)] },
];

/// Legs that are known to fail: (kernel, leg, why). They are reported, not failed; one that
/// passes is an error, so the entry gets removed.
const KNOWN: &[(&str, &str, &str)] = &[];

const BASE_TARGETS: [&str; 3] = ["x86_64-linux", "x86_64-linux-v3", "aarch64-linux"];
const V3_TARGETS: [&str; 1] = ["x86_64-linux-v3"];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn env_u64(name: &str) -> Option<u64> {
    let v = std::env::var(name).ok()?;
    let v = v.trim();
    let r = match v.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16),
        None => v.parse(),
    };
    Some(r.unwrap_or_else(|_| panic!("{name}={v}: not a number")))
}

fn plan() -> Plan {
    Plan {
        seed: env_u64("STAVE_DIFF_SEED").unwrap_or(0x5eed),
        full: std::env::var("STAVE_DIFF_FULL").is_ok_and(|v| v == "1"),
        random: env_u64("STAVE_DIFF_RANDOM").map_or(24, |n| n as usize),
    }
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// `Some(None)` native, `Some(Some(cmd))` through `cmd`, `None` not runnable here.
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

/// A dispatch arm of a `_disp` kernel: the dispatch fn `main.<f>` calls `main.<f>_v4`,
/// `main.<f>_v3`, `main.<f>_sve`, or for `_` a kernel of module `base`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    V4,
    V3,
    Sve,
    Base,
}

impl Arm {
    /// Whether `target` is this arm of the dispatch fn `main.<f>`.
    fn is(self, f: &str, target: &str) -> bool {
        match self {
            Arm::V4 => target == format!("main.{f}_v4"),
            Arm::V3 => target == format!("main.{f}_v3"),
            Arm::Sve => target == format!("main.{f}_sve"),
            Arm::Base => target.starts_with("base."),
        }
    }
}

/// The dispatch fns of a `_disp` kernel; its `main` calls each of them.
fn dispatch_fns(kernel: &str) -> &'static [&'static str] {
    match kernel {
        "bytesearch_disp" => &["find_first", "find_last", "count"],
        "sum_disp" => &["sum_i32", "sum_u8"],
        "reduce_disp" => &["sum_f32", "dot_f32", "sum_f64", "dot_f64"],
        _ => panic!("{kernel}: no dispatch fns listed"),
    }
}

/// One run of a leg's binary.
#[derive(Clone, Debug)]
struct Run {
    /// None: native.
    runner: Option<String>,
    /// `STAVE_CPU` (None: unset).
    cpu: Option<&'static str>,
    /// Some: with `STAVE_DISPATCH_TRACE=1`, and every call of the kernel's dispatch fns picks
    /// this arm.
    arm: Option<Arm>,
    /// Under Intel SDE, which may write lines of its own to stderr.
    sde: bool,
    /// Leg label suffix.
    label: String,
}

impl Run {
    fn plain(runner: Option<&str>) -> Run {
        Run { runner: runner.map(String::from), cpu: None, arm: None, sde: false, label: String::new() }
    }

    fn picks(runner: Option<&str>, cpu: Option<&'static str>, arm: Arm, label: &str) -> Run {
        let label = match cpu {
            Some(c) => format!(" {label} STAVE_CPU={c}"),
            None => format!(" {label}"),
        };
        Run { runner: runner.map(String::from), cpu, arm: Some(arm), sde: false, label }
    }
}

/// Every AVX-512 name of the `v4` group.
const NO_AVX512: &str = "-avx512f,-avx512bw,-avx512vl,-avx512dq,-avx512cd";

fn host_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The runs of a `_disp` kernel's binary for `target` (`how`: its leg, as [`runner`] gives it),
/// each with the arm it must pick:
///
/// - x86_64-linux: natively the widest arm the CPU has, and with `-avx2` the `_` arms; on the
///   v4 leg (natively or under SDE) the 512-bit arms, and without the five AVX-512 names, or
///   without just one of them, the 256-bit arms; under `qemu-x86_64 -cpu Nehalem` the `_` arms
///   (not a leg: skipped without qemu-x86_64).
/// - x86_64-linux-v3: the baseline covers the 256-bit arms but not the `v4` test: natively the
///   256-bit arms (512-bit on an AVX-512 CPU), on the v4 leg the 512-bit ones.
/// - aarch64-linux: the x86-64 arms are left out of the build, the `sve` test stays: under
///   `qemu-aarch64 -cpu max` the `sve` arms, with `-sve` and under `-cpu cortex-a76` the `_` arms.
fn gate_runs(kernel: &str, target: Triple, how: Option<Option<&'static str>>) -> Vec<Run> {
    let mut r = Vec::new();
    let v3 = Triple::parse("x86_64-linux-v3").unwrap();
    let v4 = Triple::parse("x86_64-linux-v4").unwrap();
    match target.name() {
        "x86_64-linux-v2" | "x86_64-linux-v3" => {
            let wide = target == v3;
            let host = if exec::can_run(v4) && exec::can_run(v3) {
                Arm::V4
            } else if host_avx2() {
                Arm::V3
            } else {
                Arm::Base
            };
            if how.is_some() {
                r.push(Run::picks(None, None, host, "native"));
                if !wide {
                    r.push(Run::picks(None, Some("-avx2"), Arm::Base, "native"));
                }
            }
            match legs::v4(exec::can_run(v4)) {
                Some(leg) => {
                    let (runner, label, sde) = match &leg {
                        legs::V4Run::Runner(c) => (Some(c.as_str()), "sde", true),
                        legs::V4Run::Native => (None, "v4", false),
                    };
                    if sde {
                        r.push(Run { sde, ..Run::picks(runner, None, Arm::V4, label) });
                    }
                    r.push(Run { sde, ..Run::picks(runner, Some(NO_AVX512), Arm::V3, label) });
                    if !wide {
                        r.push(Run { sde, ..Run::picks(runner, Some("-avx512cd"), Arm::V3, label) });
                    }
                }
                None => eprintln!(
                    "note: no AVX-512 CPU and no STAVE_SDE: the v4 runs of {kernel} ({target}) are skipped"
                ),
            }
            if !wide {
                if cfg!(target_os = "linux") && have("qemu-x86_64") {
                    r.push(Run::picks(Some("qemu-x86_64 -cpu Nehalem"), None, Arm::Base, "Nehalem"));
                } else {
                    eprintln!("note: no qemu-x86_64: the Nehalem runs of {kernel} are skipped");
                }
            }
        }
        "aarch64-linux" => match how {
            Some(None) => {
                let sve = exec::can_run(Triple::parse("aarch64-linux+sve").unwrap());
                r.push(Run::picks(None, None, if sve { Arm::Sve } else { Arm::Base }, "native"));
                r.push(Run::picks(None, Some("-sve"), Arm::Base, "native"));
            }
            Some(Some(q)) => {
                let (max, a76) = (format!("{q} -cpu max"), format!("{q} -cpu cortex-a76"));
                r.push(Run::picks(Some(&max), None, Arm::Sve, "qemu max"));
                r.push(Run::picks(Some(&max), Some("-sve"), Arm::Base, "qemu max"));
                r.push(Run::picks(Some(&a76), None, Arm::Base, "qemu cortex-a76"));
            }
            None => {}
        },
        t => panic!("{kernel}: no runs for {t}"),
    }
    r
}

/// Runs `bin` with exactly the `STAVE_CPU` and `STAVE_DISPATCH_TRACE` of `run`. ETXTBSY (another
/// thread's fork still holds a just-written binary open) is retried.
fn run_bin(bin: &Path, run: &Run, args: &[String], cwd: &Path) -> Ran {
    let spec = RunSpec { args, runner: run.runner.as_deref(), cwd: Some(cwd), inherit: false };
    let mut tries = 0;
    loop {
        let mut cmd = exec::command(bin, &spec).unwrap();
        cmd.env_remove("STAVE_CPU").env_remove("STAVE_DISPATCH_TRACE");
        if let Some(c) = run.cpu {
            cmd.env("STAVE_CPU", c);
        }
        if run.arm.is_some() {
            cmd.env("STAVE_DISPATCH_TRACE", "1");
        }
        match cmd.stdin(std::process::Stdio::null()).output() {
            Err(e) if e.raw_os_error() == Some(26) && tries < 50 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => panic!("{}: {e}", bin.display()),
            Ok(o) => return Ran { exit: exec::status_code(o.status), stdout: o.stdout, stderr: o.stderr },
        }
    }
}

/// The std dispatch fns (`std.slice.<f>`) a `_disp` kernel reaches whose arms it checks too.
fn std_dispatch_fns(kernel: &str) -> &'static [&'static str] {
    match kernel {
        "reduce_disp" => &[
            "sum_f32_cpu",
            "dot_f32_cpu",
            "sum_f64_cpu",
            "dot_f64_cpu",
            "sum2_f32_cpu",
            "dot2_f32_cpu",
            "sum2_f64_cpu",
            "dot2_f64_cpu",
        ],
        _ => &[],
    }
}

/// The arm of std dispatch fn `f` (`<g>_cpu`) for `want`: `<g>_v4`, `<g>_v3`, else `<g>` (std
/// has no `sve` arms).
fn std_arm(f: &str, want: Arm) -> String {
    let g = f.strip_suffix("_cpu").unwrap_or(f);
    match want {
        Arm::V4 => format!("std.slice.{g}_v4"),
        Arm::V3 => format!("std.slice.{g}_v3"),
        Arm::Sve | Arm::Base => format!("std.slice.{g}"),
    }
}

/// Checks the trace of a traced run: every `dispatch: main.<f> -> <arm>` line names `want`'s
/// arm of `f`, and each dispatch fn of the kernel has a line; so do the std fns of
/// [`std_dispatch_fns`] when `std_traced` (x86-64: an AArch64 build resolves them at compile
/// time and prints no line). Returns the run with its stderr less the trace (other std dispatch
/// fns too; under SDE every line).
fn picked(kernel: &str, mut r: Ran, want: Arm, sde: bool, std_traced: bool) -> Result<Ran, String> {
    let err = String::from_utf8_lossy(&r.stderr).into_owned();
    let fns = dispatch_fns(kernel);
    let mut seen = vec![false; fns.len()];
    let sfns = std_dispatch_fns(kernel);
    let mut sseen = vec![!std_traced; sfns.len()];
    let mut rest = String::new();
    for line in err.lines() {
        let Some(t) = line.strip_prefix("dispatch: ") else {
            if !sde {
                rest.push_str(line);
                rest.push('\n');
            }
            continue;
        };
        let Some((f, arm)) = t.split_once(" -> ") else { return Err(format!("trace line `{line}`")) };
        if let Some(i) = f.strip_prefix("std.slice.").and_then(|g| sfns.iter().position(|h| *h == g)) {
            if arm != std_arm(sfns[i], want) {
                return Err(format!("trace line `{line}`: want the {want:?} arm, {}", show(&r)));
            }
            sseen[i] = true;
            continue;
        }
        let Some(f) = f.strip_prefix("main.") else { continue };
        let Some(i) = fns.iter().position(|g| *g == f) else {
            return Err(format!("trace line `{line}`: not a dispatch fn of {kernel}"));
        };
        if !want.is(f, arm) {
            return Err(format!("trace line `{line}`: want the {want:?} arm, {}", show(&r)));
        }
        seen[i] = true;
    }
    if let Some(i) = seen.iter().position(|s| !s) {
        return Err(format!("no trace line for `main.{}` (want the {want:?} arm), {}", fns[i], show(&r)));
    }
    if let Some(i) = sseen.iter().position(|s| !s) {
        return Err(format!(
            "no trace line for `std.slice.{}` (want the {want:?} arm), {}",
            sfns[i],
            show(&r)
        ));
    }
    r.stderr = rest.into_bytes();
    Ok(r)
}

/// The package of one kernel: `stave.toml`, `src/main.stv` (the kernel file), `src/diff.stv`,
/// and for `<k>_v3` and `<k>_disp` the kernels and references of `<k>.stv` as `src/base.stv`.
fn package(dir: &Path, name: &str) {
    let src = dir.join("src");
    std::fs::create_dir_all(&src).unwrap();
    let here = repo().join("tests/differential");
    std::fs::write(dir.join("stave.toml"), "name = \"kernel\"\nkind = \"bin\"\n").unwrap();
    std::fs::copy(here.join(format!("kernels/{name}.stv")), src.join("main.stv")).unwrap();
    std::fs::copy(here.join("diff.stv"), src.join("diff.stv")).unwrap();
    if let Some(base) = name.strip_suffix("_v3").or_else(|| name.strip_suffix("_disp")) {
        std::fs::copy(here.join(format!("kernels/{base}.stv")), src.join("base.stv")).unwrap();
    }
}

/// A build, or why there is none (`Unsupported`: "not built").
fn build(dir: &Path, target: Triple, release: bool) -> Result<Output, String> {
    let mut o = Options::new(Goal::Build).threads(Some(2));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(dir, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{} ({target}): diagnostics {:#?}", dir.display(), out.diagnostics);
    match &out.failure {
        None => Ok(out),
        Some(Failure::Unsupported { message, .. }) => Err(format!("not built: {message}")),
        Some(f) => panic!("{} ({target}, release {release}): {f}", dir.display()),
    }
}

/// Equal tokens. NaN sign and payload are unspecified, so two NaN tokens of one float type
/// (`f`/`d`) are equal whatever their bits; every other token, `-0.0` included, compares exactly.
fn same(a: &str, b: &str) -> bool {
    a == b || (a.as_bytes().first() == b.as_bytes().first() && is_nan(a) && is_nan(b))
}

fn is_nan(t: &str) -> bool {
    match (t.as_bytes().first(), t.get(1..)) {
        (Some(b'f'), Some(n)) => n.parse::<u32>().is_ok_and(|b| f32::from_bits(b).is_nan()),
        (Some(b'd'), Some(n)) => n.parse::<u64>().is_ok_and(|b| f64::from_bits(b).is_nan()),
        _ => false,
    }
}

fn same_tokens(a: &str, b: &str) -> bool {
    let (x, y): (Vec<&str>, Vec<&str>) = (a.split_whitespace().collect(), b.split_whitespace().collect());
    x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| same(p, q))
}

/// Whether a case line's vector tokens match its reference tokens.
fn line_ok(line: &str) -> bool {
    line.split_once(" =").is_some_and(|(v, r)| !v.trim().is_empty() && same_tokens(v, r))
}

/// What one leg printed, or why it did not run.
enum Leg {
    Ran(Ran),
    NotBuilt(String),
    Skipped(String),
    /// Ran, but not as the leg must (the dispatch trace).
    Failed(String),
}

struct Report {
    kernel: &'static str,
    leg: String,
    cases: usize,
    /// Lines whose vector tokens differ from the reference tokens.
    mismatches: usize,
    /// Lines that differ from the interpreter's.
    vs_interp: usize,
    /// None: compared; Some: why not.
    note: Option<String>,
    detail: Vec<String>,
}

fn show(r: &Ran) -> String {
    let err = String::from_utf8_lossy(&r.stderr);
    format!("exit {}, stderr: {}", r.exit, err.lines().take(4).collect::<Vec<_>>().join(" | "))
}

/// The first elements of each input array of a case.
fn inputs(case: &generate::Case) -> String {
    const SHOW: usize = 48;
    let mut out = Vec::new();
    for (e, a) in case.elems.iter().zip(&case.arrays) {
        let w = e.size();
        let items: Vec<String> = a
            .chunks(w)
            .take(SHOW)
            .map(|b| match e {
                Elem::F32 => format!("{:?}", f32::from_le_bytes(b.try_into().unwrap())),
                Elem::F64 => format!("{:?}", f64::from_le_bytes(b.try_into().unwrap())),
                Elem::I32 => i32::from_le_bytes(b.try_into().unwrap()).to_string(),
                Elem::U8 | Elem::Utf8 => format!("{:02x}", b[0]),
            })
            .collect();
        let more = if case.len > SHOW { " .." } else { "" };
        out.push(format!("[{}{more}]", items.join(" ")));
    }
    out.join(" ")
}

/// Re-runs one case alone with element-by-element output.
fn detail(dir: &Path, case: &generate::Case, run: &dyn Fn(&Path, &[u8]) -> Option<Ran>) -> String {
    let one = dir.join("one");
    std::fs::create_dir_all(&one).unwrap();
    let bytes = generate::encode(std::slice::from_ref(case));
    std::fs::write(one.join("cases.bin"), &bytes).unwrap();
    let head = format!(
        "kind {} len {} mis {} param {:#x}\n      input: {}",
        case.kind,
        case.len,
        case.mis,
        case.param,
        inputs(case)
    );
    match run(&one, &bytes) {
        Some(r) => format!("{head}\n      full: {}", String::from_utf8_lossy(&r.stdout).trim_end()),
        None => head,
    }
}

/// Reference tokens computed here, for kernels whose reference is easy to get wrong.
fn oracle(kernel: &str, case: &generate::Case) -> Option<String> {
    let xs = case.arrays.first()?;
    match kernel {
        "utf8" | "utf8_v3" => {
            let valid = std::str::from_utf8(xs).is_ok();
            let points = xs.iter().filter(|&&b| b & 0xc0 != 0x80).count();
            Some(format!("u{} u{points}", u8::from(valid)))
        }
        "bytesearch" | "bytesearch_v3" | "bytesearch_disp" => {
            let b = case.param as u8;
            let first = xs.iter().position(|&x| x == b).unwrap_or(xs.len());
            let last = xs.iter().rposition(|&x| x == b).unwrap_or(xs.len());
            let count = xs.iter().filter(|&&x| x == b).count();
            Some(format!("u{first} u{last} u{count}"))
        }
        "sum_disp" if case.kind == 0 => {
            let s = xs.chunks(4).fold(0i32, |s, b| s.wrapping_add(i32::from_le_bytes(b.try_into().unwrap())));
            Some(format!("i{s}"))
        }
        "sum_disp" => Some(format!("u{}", xs.iter().fold(0u32, |s, &b| s.wrapping_add(u32::from(b))))),
        "reduce" | "reduce_disp" => {
            let t = reduce_token(case, STD_RULE);
            Some(format!("{t} {t} {t} {t}"))
        }
        _ => None,
    }
}

fn le_f32(b: &[u8]) -> Vec<f32> {
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn le_f64(b: &[u8]) -> Vec<f64> {
    b.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// The logical-lane rule over L lanes: lane l starts at `start` and adds terms l, l + L, .. in
/// order, then lane j adds lane j + w for w = L/2, .., 1. std uses `start` = -0.0.
fn lanes_f32(terms: &[f32], l: usize, start: f32) -> f32 {
    let mut lane = vec![start; l];
    for (i, t) in terms.iter().enumerate() {
        lane[i % l] += t;
    }
    let mut w = l / 2;
    while w > 0 {
        for j in 0..w {
            lane[j] += lane[j + w];
        }
        w /= 2;
    }
    lane[0]
}

fn lanes_f64(terms: &[f64], l: usize, start: f64) -> f64 {
    let mut lane = vec![start; l];
    for (i, t) in terms.iter().enumerate() {
        lane[i % l] += t;
    }
    let mut w = l / 2;
    while w > 0 {
        for j in 0..w {
            lane[j] += lane[j + w];
        }
        w /= 2;
    }
    lane[0]
}

/// The block tree in level passes: each pass adds neighbours 2i + 1 into 2i and carries a last
/// odd one up unchanged, until one sum is left.
fn tree<T: Copy + std::ops::Add<Output = T>>(mut v: Vec<T>) -> T {
    while v.len() > 1 {
        v = v.chunks(2).map(|p| if p.len() == 2 { p[0] + p[1] } else { p[0] }).collect();
    }
    v[0]
}

/// A variant of the rule: `l32` / `l64` lanes, lanes from `start`, blocks of `block` terms
/// (`usize::MAX`: one block, the one-level rule), block sums by the block tree or, without
/// `tree`, left to right.
#[derive(Clone, Copy)]
struct Rule {
    l32: usize,
    l64: usize,
    start: f64,
    block: usize,
    tree: bool,
}

/// std's rule.
const STD_RULE: Rule = Rule { l32: L_F32, l64: L_F64, start: -0.0, block: BLOCK, tree: true };

fn combine<T: Copy + std::ops::Add<Output = T>>(v: Vec<T>, r: Rule) -> T {
    if r.tree { tree(v) } else { v[1..].iter().fold(v[0], |a, &b| a + b) }
}

fn rule_f32(t: &[f32], r: Rule) -> f32 {
    let s = r.start as f32;
    if t.is_empty() {
        return lanes_f32(t, r.l32, s);
    }
    combine(t.chunks(r.block.min(t.len())).map(|b| lanes_f32(b, r.l32, s)).collect(), r)
}

fn rule_f64(t: &[f64], r: Rule) -> f64 {
    if t.is_empty() {
        return lanes_f64(t, r.l64, r.start);
    }
    combine(t.chunks(r.block.min(t.len())).map(|b| lanes_f64(b, r.l64, r.start)).collect(), r)
}

/// The token of a reduction case (kinds of [`REDUCE_KINDS`]) under rule `r`.
fn reduce_token(case: &generate::Case, r: Rule) -> String {
    let a = &case.arrays[0];
    match case.kind {
        0 => ftok(rule_f32(&le_f32(a), r).to_bits()),
        1 => dtok(rule_f64(&le_f64(a), r).to_bits()),
        2 => {
            let ps: Vec<f32> =
                le_f32(a).into_iter().zip(le_f32(&case.arrays[1])).map(|(x, y)| x * y).collect();
            ftok(rule_f32(&ps, r).to_bits())
        }
        _ => {
            let ps: Vec<f64> =
                le_f64(a).into_iter().zip(le_f64(&case.arrays[1])).map(|(x, y)| x * y).collect();
            dtok(rule_f64(&ps, r).to_bits())
        }
    }
}

fn check(
    kernel: &'static str,
    leg: String,
    got: Leg,
    want: Option<&str>,
    cases: &[generate::Case],
    dir: &Path,
    rerun: &dyn Fn(&Path, &[u8]) -> Option<Ran>,
) -> Report {
    let mut rep =
        Report { kernel, leg, cases: 0, mismatches: 0, vs_interp: 0, note: None, detail: Vec::new() };
    let r = match got {
        Leg::Ran(r) => r,
        Leg::NotBuilt(m) | Leg::Skipped(m) => {
            rep.note = Some(m);
            return rep;
        }
        Leg::Failed(m) => {
            rep.note = Some(format!("failed: {m}"));
            return rep;
        }
    };
    let out = String::from_utf8_lossy(&r.stdout);
    let lines: Vec<&str> = out.lines().collect();
    if r.exit != 0 || !r.stderr.is_empty() || lines.len() != cases.len() {
        rep.note = Some(format!("failed: {} lines for {} cases, {}", lines.len(), cases.len(), show(&r)));
        return rep;
    }
    rep.cases = lines.len();
    let want: Vec<&str> = want.map(|w| w.lines().collect()).unwrap_or_default();
    for (i, l) in lines.iter().enumerate() {
        let known = oracle(kernel, &cases[i]);
        let bad = !line_ok(l)
            || known.as_ref().is_some_and(|o| l.split_once(" =").is_none_or(|(_, r)| !same_tokens(r, o)));
        let differs = !want.is_empty() && !same_tokens(l, want[i]);
        rep.mismatches += usize::from(bad);
        rep.vs_interp += usize::from(differs);
        if (bad || differs) && rep.detail.len() < 3 {
            let mut d = format!("case {i}: {}\n      got: {l}", detail(dir, &cases[i], rerun));
            if differs {
                d.push_str(&format!("\n      interpreter: {}", want[i]));
            }
            if let Some(o) = &known {
                d.push_str(&format!("\n      expected reference: {o}"));
            }
            rep.detail.push(d);
        }
    }
    rep
}

fn files(bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
    vec![("cases.bin".to_string(), bytes.to_vec())]
}

fn run_kernel(k: &'static Kernel, root: &Path, plan: Plan) -> Vec<Report> {
    let dir = root.join(k.name);
    package(&dir, k.name);
    let cases = generate::cases(k.name, k.kinds, plan);
    let bytes = generate::encode(&cases);
    std::fs::write(dir.join("cases.bin"), &bytes).unwrap();
    let v3 = k.name.ends_with("_v3");
    let targets: &[&str] = if v3 { &V3_TARGETS } else { &BASE_TARGETS };
    let mut reports = Vec::new();

    // Interpreter: the checked IR of the first target's debug build.
    let t0 = Triple::parse(targets[0]).unwrap();
    let mut o = Options::new(Goal::Check).threads(Some(2));
    o.target = Some(t0);
    o.lower = true;
    let checked = pipeline::run(&dir, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!checked.has_errors(), "{}: diagnostics {:#?}", k.name, checked.diagnostics);
    let m =
        checked.module.as_ref().unwrap_or_else(|| panic!("{}: not lowered: {:?}", k.name, checked.failure));
    let main = stave_driver::build::main_path(&checked.program).unwrap();
    let interp_run = |files: &[(String, Vec<u8>)], args: &[&[u8]]| {
        interp::run(m, &main, t0, args, files, Limits::default())
            .unwrap_or_else(|e| panic!("{} interpreter: {e}", k.name))
            .ran
    };
    let want = interp_run(&files(&bytes), &[]);
    let ileg = format!("interpreter ({t0})");
    let rerun_i = |_: &Path, b: &[u8]| Some(interp_run(&files(b), &[b"full"]));
    reports.push(check(k.name, ileg, Leg::Ran(want.clone()), None, &cases, &dir, &rerun_i));
    let want_text = String::from_utf8_lossy(&want.stdout).into_owned();

    for t in targets {
        let target = Triple::parse(t).unwrap();
        let how = runner(target);
        let runs = if k.name.ends_with("_disp") {
            gate_runs(k.name, target, how)
        } else {
            how.map(|r| vec![Run::plain(r)]).unwrap_or_default()
        };
        let no_rerun = |_: &Path, _: &[u8]| -> Option<Ran> { None };
        for release in [false, true] {
            let mode = if release { "release" } else { "debug" };
            let out = match build(&dir, target, release) {
                Ok(out) => out,
                Err(m) => {
                    reports.push(check(
                        k.name,
                        format!("{t} {mode}"),
                        Leg::NotBuilt(m),
                        None,
                        &[],
                        &dir,
                        &no_rerun,
                    ));
                    continue;
                }
            };
            if runs.is_empty() {
                let got = Leg::Skipped("cannot run here".into());
                reports.push(check(k.name, format!("{t} {mode}"), got, None, &[], &dir, &no_rerun));
                continue;
            }
            // one binary for every run of the leg
            let bin = dir.join(format!("bin-{t}-{mode}"));
            out.image.as_ref().unwrap().write_file(&bin).unwrap();
            for run in &runs {
                let leg = format!("{t} {mode}{}", run.label);
                let got = run_bin(&bin, run, &[], &dir);
                let got = match run.arm {
                    None => Leg::Ran(got),
                    Some(arm) => match picked(k.name, got, arm, run.sde, t.starts_with("x86_64")) {
                        Ok(r) => Leg::Ran(r),
                        Err(e) => Leg::Failed(e),
                    },
                };
                let rerun = |cwd: &Path, _: &[u8]| {
                    let plain = Run { arm: None, ..run.clone() };
                    Some(run_bin(&bin, &plain, &["full".to_string()], cwd))
                };
                reports.push(check(k.name, leg, got, Some(&want_text), &cases, &dir, &rerun));
            }
        }
    }
    reports
}

#[test]
fn kernels_match_their_references() {
    let plan = plan();
    let only: Option<Vec<String>> = std::env::var("STAVE_DIFF_KERNELS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let chosen: Vec<&'static Kernel> =
        KERNELS.iter().filter(|k| only.as_ref().is_none_or(|o| o.iter().any(|n| n == k.name))).collect();
    let tmp = TempDir::new("diff").unwrap();
    let root = tmp.path();
    let t = Instant::now();
    let reports: Vec<Report> = std::thread::scope(|s| {
        let hs: Vec<_> = chosen.iter().map(|&k| s.spawn(move || run_kernel(k, root, plan))).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut failures = Vec::new();
    let mut passed_known = Vec::new();
    eprintln!("seed {:#x}, full {}, random {} per kind", plan.seed, plan.full, plan.random);
    for r in &reports {
        let known = KNOWN.iter().find(|(k, l, _)| *k == r.kernel && *l == r.leg);
        let bad = r.mismatches > 0
            || r.vs_interp > 0
            || r.note.as_deref().is_some_and(|n| !n.starts_with("cannot"));
        let status = match &r.note {
            Some(n) => n.clone(),
            None => format!(
                "{} cases, {} mismatches, {} differ from the interpreter",
                r.cases, r.mismatches, r.vs_interp
            ),
        };
        let status = match known {
            Some((_, _, why)) if bad => format!("known failure ({why}): {status}"),
            _ => status,
        };
        eprintln!("{:<12} {:<28} {status}", r.kernel, r.leg);
        for d in &r.detail {
            eprintln!("    {d}");
        }
        if bad && known.is_none() {
            failures.push(format!("{} [{}]: {status}", r.kernel, r.leg));
        }
        if !bad && r.note.is_none() && known.is_some() {
            passed_known.push(format!("{} [{}]", r.kernel, r.leg));
        }
    }
    eprintln!("{} legs in {:.1} s", reports.len(), t.elapsed().as_secs_f64());
    assert!(failures.is_empty(), "{} failing leg(s):\n{}", failures.len(), failures.join("\n"));
    assert!(passed_known.is_empty(), "known failures that pass now (remove from KNOWN): {passed_known:?}");
}

fn ftok(bits: u32) -> String {
    format!("f{bits}")
}

fn dtok(bits: u64) -> String {
    format!("d{bits}")
}

#[test]
fn nan_tokens_compare_as_a_class() {
    let q = f32::NAN.to_bits();
    for other in [q | 0x8000_0000, 0x7f80_0001, 0x7fc0_1234, 0xffff_ffff] {
        assert!(f32::from_bits(other).is_nan());
        assert!(same(&ftok(q), &ftok(other)), "f32 NaN {q:#x} vs {other:#x}");
    }
    let dq = f64::NAN.to_bits();
    for other in [dq | (1 << 63), 0x7ff0_0000_0000_0001, 0x7ff8_dead_beef_0000] {
        assert!(f64::from_bits(other).is_nan());
        assert!(same(&dtok(dq), &dtok(other)), "f64 NaN {dq:#x} vs {other:#x}");
    }
    let (a, b) = (ftok(0x7f80_0001), ftok(0xffc0_0000));
    assert!(same_tokens(&format!("u3 {a} h7"), &format!("u3 {b} h7")));
    assert!(line_ok(&format!(" {} {} = {} {}", ftok(q), dtok(dq), ftok(0xff80_0002), dtok(dq ^ 1))));
}

#[test]
fn other_tokens_compare_bit_exact() {
    let (pz, nz) = (ftok(0f32.to_bits()), ftok((-0f32).to_bits()));
    assert!(!same(&pz, &nz), "f32 -0.0 vs 0.0");
    assert!(!same(&dtok(0f64.to_bits()), &dtok((-0f64).to_bits())), "f64 -0.0 vs 0.0");
    assert!(!same(&ftok(f32::INFINITY.to_bits()), &ftok(f32::NAN.to_bits())), "inf vs NaN");
    assert!(!same(&ftok(1f32.to_bits()), &ftok(1f32.to_bits() + 1)), "one ulp apart");
    assert!(!same(&ftok(f32::NAN.to_bits()), &dtok(f64::NAN.to_bits())), "NaNs of two float types");
    let (u1, u2) = (format!("u{}", f32::NAN.to_bits()), format!("u{}", 0x7fc0_0001u32));
    assert!(!same(&u1, &u2), "an int token is never a NaN");
    assert!(!line_ok(&format!(" {pz} = {nz}")));
    assert!(!same_tokens(&pz, &format!("{pz} {pz}")));
}

/// The reduction cases catch the likely mistakes: for every kind, some default case gives other
/// bits with half or twice the lane count, with the sequential sum, with +0.0 lane starts, with
/// the one-level rule, with half or twice the block, and with block sums added left to right;
/// and some case is a NaN under one lane count but not under another.
#[test]
fn reduce_cases_tell_lane_counts_blocks_orders_and_zero_signs_apart() {
    let plan = Plan { seed: 0x5eed, full: false, random: 24 };
    let cases = generate::cases("reduce", REDUCE_KINDS, plan);
    let s = STD_RULE;
    let variants = [
        Rule { l32: L_F32 / 2, l64: L_F64 / 2, ..s },
        Rule { l32: 2 * L_F32, l64: 2 * L_F64, ..s },
        Rule { l32: 1, l64: 1, block: usize::MAX, ..s },
        Rule { start: 0.0, ..s },
        Rule { block: usize::MAX, ..s },
        Rule { block: BLOCK / 2, ..s },
        Rule { block: 2 * BLOCK, ..s },
        Rule { tree: false, ..s },
    ];
    for kind in 0..REDUCE_KINDS.len() as u32 {
        let mut n = [0usize; 9];
        for c in cases.iter().filter(|c| c.kind == kind) {
            let want = reduce_token(c, s);
            let got: Vec<String> = variants.iter().map(|&r| reduce_token(c, r)).collect();
            for (k, g) in got.iter().enumerate() {
                n[k] += usize::from(!same(&want, g));
            }
            n[8] += usize::from(is_nan(&want) != is_nan(&got[0]) || is_nan(&want) != is_nan(&got[1]));
        }
        assert!(
            n.iter().all(|&k| k > 0),
            "kind {kind}: L/2, 2L, sequential, +0.0, one-level, B/2, 2B, left fold, NaN-ness: {n:?}"
        );
    }
}
