//! Differential test of hand-written SIMD kernels (`kernels/*.stv`) against their scalar
//! references. Each kernel file and `diff.stv` become one package; the cases of `gen.rs` go to
//! its `cases.bin`. The program prints one line per case, the vector kernels' tokens, ` =`, then
//! the references' tokens; every line must match itself, and every leg must print what the
//! interpreter prints. Legs: the stave-ir interpreter, then native debug and release builds for
//! x86_64-linux, x86_64-linux-v3 and aarch64-linux (under qemu-aarch64 on an x86-64 host);
//! a `_v3` kernel builds for x86_64-linux-v3 only. A leg that cannot run here is skipped with a
//! note, or fails when `STAVE_REQUIRE_LEGS` lists it.
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
    Kind { arrays, lanes, param }
}

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
    Kernel {
        name: "combinators",
        kinds: &[
            kind(&[Elem::F32], 16, Param::F32),
            kind(&[Elem::F64], 8, Param::None),
            kind(&[Elem::I32], 4, Param::I32),
            kind(&[Elem::U8], 16, Param::Byte),
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

/// The package of one kernel: `stave.toml`, `src/main.stv` (the kernel file), `src/diff.stv`,
/// and for `<k>_v3` the references of `<k>.stv` as `src/base.stv`.
fn package(dir: &Path, name: &str) {
    let src = dir.join("src");
    std::fs::create_dir_all(&src).unwrap();
    let here = repo().join("tests/differential");
    std::fs::write(dir.join("stave.toml"), "name = \"kernel\"\nkind = \"bin\"\n").unwrap();
    std::fs::copy(here.join(format!("kernels/{name}.stv")), src.join("main.stv")).unwrap();
    std::fs::copy(here.join("diff.stv"), src.join("diff.stv")).unwrap();
    if let Some(base) = name.strip_suffix("_v3") {
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

/// Equal tokens; two NaNs (`f`/`d` tokens) are equal whatever their bits.
fn same(a: &str, b: &str) -> bool {
    a == b || (is_nan(a) && is_nan(b))
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
        "bytesearch" | "bytesearch_v3" => {
            let b = case.param as u8;
            let first = xs.iter().position(|&x| x == b).unwrap_or(xs.len());
            let last = xs.iter().rposition(|&x| x == b).unwrap_or(xs.len());
            let count = xs.iter().filter(|&&x| x == b).count();
            Some(format!("u{first} u{last} u{count}"))
        }
        _ => None,
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
        for release in [false, true] {
            let leg = format!("{t} {}", if release { "release" } else { "debug" });
            let got = match build(&dir, target, release) {
                Err(m) => Leg::NotBuilt(m),
                Ok(_) if how.is_none() => Leg::Skipped("cannot run here".into()),
                Ok(out) => {
                    let bin = dir.join(format!("bin-{t}-{release}"));
                    out.image.as_ref().unwrap().write_file(&bin).unwrap();
                    let spec = RunSpec { args: &[], runner: how.unwrap(), cwd: Some(&dir), inherit: false };
                    Leg::Ran(exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display())))
                }
            };
            let bin = dir.join(format!("bin-{t}-{release}"));
            let rerun = |cwd: &Path, _: &[u8]| {
                let args = ["full".to_string()];
                let spec = RunSpec { args: &args, runner: how.flatten(), cwd: Some(cwd), inherit: false };
                exec::run(&bin, &spec).ok()
            };
            reports.push(check(k.name, leg, got, Some(&want_text), &cases, &dir, &rerun));
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
