//! The borrow checker's sparse dataflow against the dense one, and its cost on very large fns.
//!
//! - `borrow_tests_agree`: every test of `tests/conformance/borrow` and `tests/borrowcheck`
//!   under both dataflows in one run (`stave_borrow::Dataflow::Both`: a body whose diagnostics
//!   or dump lines differ fails the run as an internal error);
//! - `everything_agrees` (`--ignored`; `tools/ci-step/run.sh borrow-diff`): every package and
//!   every `.stv` file of the repository (tests, conformance corpus, `bench`, `stage1`, `std`,
//!   `runtime`), a generated corpus, and the fuzz classes at many seeds, each checked once per
//!   dataflow: `--diagnostics json` must be byte-identical, and so must the verdict; a third
//!   run with both in one process compares the per-access dump lines too;
//! - `large_fns_stay_cheap`: fns of 10 000 statements in the shapes that cost the dense
//!   dataflow the most are checked within a time bound, and the dataflow's own counts (states
//!   applied and copied per statement) stay under fixed bounds, which no machine load changes.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use gen_corpus::classes::{self, Class};
use stave_borrow::Dataflow;
use stave_driver::pipeline::{self, Goal, Options, Output, Stage};
use stave_driver::shipped::Library;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tmp(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("borrow_dataflow").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn check(root: &Path, dataflow: Dataflow, dump: bool, threads: usize) -> Output {
    let mut o = Options::new(Goal::Check).threads(Some(threads));
    o.borrow_dataflow = dataflow;
    o.borrow_dump = dump;
    pipeline::run(root, &Library::in_dir(&repo()), &o).unwrap()
}

/// What a run says about a program: the diagnostics as `--diagnostics json` prints them, and
/// why it failed without one.
fn verdict(out: &Output) -> String {
    let json = stave_diag::emit_json(&out.diagnostics, out.files()).unwrap();
    format!("{json}\nfailure: {:?}\n", out.failure.as_ref().map(|f| format!("{f:?}")))
}

/// Runs `f` on every root, on `jobs` threads; the `Some` results, in root order.
fn each(roots: &[PathBuf], jobs: usize, f: impl Fn(&Path) -> Option<String> + Sync) -> Vec<String> {
    let next = AtomicUsize::new(0);
    let found: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(root) = roots.get(i) else { break };
                    if let Some(m) = f(root) {
                        found.lock().unwrap().push((i, m));
                    }
                }
            });
        }
    });
    let mut found = found.into_inner().unwrap();
    found.sort();
    found.into_iter().map(|x| x.1).collect()
}

/// The `.stv` files and the package directories (those with a `stave.toml`) under `dir`.
fn roots_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() {
            if name.starts_with('.') || name == "target" || name == "specs" {
                continue;
            }
            if p.join("stave.toml").is_file() {
                out.push(p.clone());
            }
            roots_under(&p, out);
        } else if name.ends_with(".stv") {
            out.push(p);
        }
    }
}

/// `None` when the two dataflows agree on `root`, else what differs.
fn differs(root: &Path) -> Option<String> {
    let dense = check(root, Dataflow::Dense, false, 1);
    let sparse = check(root, Dataflow::Sparse, false, 1);
    let (a, b) = (verdict(&dense), verdict(&sparse));
    if a != b {
        return Some(format!("{}:\n--- dense\n{a}--- sparse\n{b}", root.display()));
    }
    let both = check(root, Dataflow::Both, true, 1);
    if let Some(f) = &both.failure
        && format!("{f:?}").contains("dataflow disagree")
    {
        return Some(format!("{}: {f:?}", root.display()));
    }
    None
}

fn jobs() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).min(16)
}

#[test]
fn borrow_tests_agree() {
    let mut roots = Vec::new();
    roots_under(&repo().join("tests/conformance/borrow"), &mut roots);
    roots_under(&repo().join("tests/borrowcheck"), &mut roots);
    assert!(roots.len() >= 150, "{}", roots.len());
    let bad = each(&roots, jobs(), |root| {
        let out = check(root, Dataflow::Both, true, 1);
        let f = format!("{:?}", out.failure);
        f.contains("dataflow disagree").then(|| format!("{}: {f}", root.display()))
    });
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
#[ignore = "minutes: every program of the repository, a generated corpus, the fuzz classes (run with --ignored)"]
fn everything_agrees() {
    let mut roots = Vec::new();
    roots_under(&repo(), &mut roots);
    let in_repo = roots.len();
    assert!(in_repo >= 1300, "{in_repo}");
    // a generated corpus, as one package and file by file
    let dir = tmp("corpus");
    let mix = gen_corpus::Mix::default_mix().unwrap();
    gen_corpus::generate(&gen_corpus::Options {
        lines: 30_000,
        seed: 7,
        out: dir.clone(),
        mix,
        threads: 0,
        force: true,
    })
    .unwrap();
    roots.push(dir.clone());
    roots_under(&dir, &mut roots);
    let generated = roots.len() - in_repo;
    // the fuzz classes
    let dir = tmp("classes");
    let seeds: u64 = std::env::var("STAVE_DIFF_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(150);
    for class in Class::ALL {
        for seed in 0..seeds {
            let p = dir.join(format!("{}-{seed}.stv", class.name()));
            std::fs::write(&p, classes::generate(class, seed).text).unwrap();
            roots.push(p);
        }
    }
    let fuzzed = roots.len() - in_repo - generated;
    let bad = each(&roots, jobs(), differs);
    // what the programs exercise: borrow diagnostics, bodies solved with an entry state per block
    let seen = each(&roots, jobs(), |r| {
        let out = check(r, Dataflow::Sparse, false, 1);
        let code = out.diagnostics.iter().any(|d| d.code().starts_with("E05"));
        let bc = out.borrowck.as_ref()?;
        Some(format!("{} {} {}", u8::from(code), bc.checked, bc.cost.exact))
    });
    let sum = |k: usize| -> u64 {
        seen.iter().map(|s| s.split(' ').nth(k).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0)).sum()
    };
    eprintln!(
        "programs: {} ({in_repo} of the repository, {generated} generated, {fuzzed} of the fuzz classes); \
         {} with a borrow diagnostic; bodies checked: {}, {} of them with an entry state per block; \
         differences: {}",
        roots.len(),
        sum(0),
        sum(1),
        sum(2),
        bad.len()
    );
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

// ---------- large fns ----------

const HEAD: &str = "import std.io;
import std.vec;
import std.string;

fn id(x: &u32) -> &u32 from x { return x; }
fn pick(a: &u32, b: &u32) -> &u32 from a + b { return a; }
";
const BASE: &str = "    let mut a0: u32 = 1;\n    let mut a1: u32 = 2;\n    let mut a2: u32 = 3;\n    let mut a3: u32 = 4;\n    let mut s: u32 = 0;\n";
const TAIL: &str = "    io.print(f\"{s}\\n\");\n    return 0;\n}\n";

/// A program whose `main` has about `n` statements of one shape.
fn large(shape: &str, n: usize) -> String {
    let mut o = String::from(HEAD);
    let shared = BASE.replace("let mut", "let").replace("let s", "let mut s");
    match shape {
        // straight line: a borrow, one use, dead
        "line" => {
            o += "fn main() -> i32 {\n";
            o += BASE;
            for i in 0..n / 2 {
                let k = i % 4;
                if i % 2 == 0 {
                    o += &format!("    let r{i} = &a{k};\n    s = s +% *r{i};\n");
                } else {
                    o += &format!("    let r{i} = &mut a{k};\n    *r{i} = *r{i} +% s;\n");
                }
            }
        }
        // many loans, all held to the end
        "loans" => {
            o += "fn main() -> i32 {\n";
            o += &shared;
            for i in 0..n / 2 {
                o += &format!("    let r{i} = &a{};\n", i % 4);
            }
            for i in (0..n / 2).rev() {
                o += &format!("    s = s +% *r{i};\n");
            }
        }
        // many locals, each mutably borrowed, all borrows held
        "locals" => {
            o += "fn main() -> i32 {\n    let mut s: u32 = 0;\n";
            for i in 0..n / 3 {
                o += &format!("    let mut x{i}: u32 = {};\n", i % 1000);
            }
            for i in 0..n / 3 {
                o += &format!("    let r{i} = &mut x{i};\n");
            }
            for i in 0..n / 3 {
                o += &format!("    *r{i} = *r{i} +% s;\n");
            }
        }
        // temporaries: a borrow passed through three calls
        "calls" => {
            o += "fn main() -> i32 {\n";
            o += BASE;
            for i in 0..n {
                o += &format!("    s = s +% *id(id(id(&a{})));\n", i % 4);
            }
        }
        // one `match` with n / 2 arms, each borrowing
        "match" => {
            o += "fn main() -> i32 {\n";
            o += BASE;
            o += "    let k: u32 = 7;\n    match k {\n";
            for i in 0..n / 2 {
                o += &format!("        {i} => {{ let r = &mut a{}; *r = *r +% {i}; }}\n", i % 4);
            }
            o += "        _ => { s = 1; }\n    }\n";
        }
        // `if`, `while`, `match` in sequence, a loan live across each
        "joins" => {
            o += "fn main() -> i32 {\n";
            o += &shared;
            for i in 0..n / 4 {
                o += &format!("    let r{i} = &a{};\n", i % 4);
                o += &match i % 3 {
                    0 => format!("    if s > {i} {{ s = s +% *r{i}; }} else {{ s = s +% 1; }}\n"),
                    1 => format!("    while s < {i} {{ s = s +% *r{i}; }}\n"),
                    _ => format!("    match s {{ 0 => {{ s = s +% *r{i}; }} _ => {{ s = s +% 2; }} }}\n"),
                };
                o += &format!("    s = s +% *r{i};\n");
            }
        }
        // one nest of n / 4 `if`s, a loan live across each
        "nest" => {
            o += "fn main() -> i32 {\n";
            o += &shared;
            for i in 0..n / 4 {
                o += &format!("let r{i} = &a{};\nif s > {i} {{\n", i % 4);
            }
            o += "s = s +% 1;\n";
            for i in (0..n / 4).rev() {
                o += &format!("}}\ns = s +% *r{i};\n");
            }
        }
        _ => panic!("{shape}"),
    }
    o += TAIL;
    o
}

struct Measured {
    ms: f64,
    cost: stave_borrow::loans::Cost,
}

fn measure(shape: &str, n: usize) -> Measured {
    let path = tmp("large").join(format!("{shape}_{n}.stv"));
    std::fs::write(&path, large(shape, n)).unwrap();
    let out = check(&path, Dataflow::Sparse, false, 2);
    let errors: Vec<String> =
        out.diagnostics.iter().take(3).map(|d| format!("{} {}", d.code(), d.message())).collect();
    assert!(errors.is_empty() && out.failure.is_none(), "{shape} {n}: {errors:?} {:?}", out.failure);
    let ms = out.timings.iter().find(|t| t.stage == Stage::Borrowck).unwrap().ms;
    Measured { ms, cost: out.borrowck.as_ref().unwrap().cost }
}

/// Per shape: the most statements applied, and state entries copied, per statement of the
/// CFG that the fn gains from 5 000 to 10 000 source statements (0: no bound of its own). The
/// counts depend on the program alone; the bounds are twice what the dataflow needs (a
/// statement is applied once per round and once by the checks; a straight line copies no
/// state).
const SHAPES: [(&str, u64, u64); 7] = [
    ("line", 4, 1),
    ("loans", 4, 1),
    ("locals", 4, 1),
    ("calls", 4, 1),
    ("match", 6, 2),
    ("joins", 6, 6),
    // every loan is live across every inner `if`: the copies follow the live locals per block
    ("nest", 6, 0),
];

/// The most state entries copied per (local, block) pair with the local live at the block's
/// entry, plus one per statement: an entry state is copied a fixed number of times per round.
const MAX_PER_LIVE: f64 = 16.0;

/// The borrowck stage of one fn of 10 000 statements, in a debug build on a loaded machine.
const MAX_MS: f64 = 60_000.0;

#[test]
fn large_fns_stay_cheap() {
    let mut bad = Vec::new();
    for (shape, max_applied, max_copied) in SHAPES {
        let (half, full) = (measure(shape, 5_000), measure(shape, 10_000));
        let stmts = full.cost.stmts - half.cost.stmts;
        let applied = (full.cost.applied - half.cost.applied) as f64 / stmts as f64;
        let copied = (full.cost.copied - half.cost.copied) as f64 / stmts as f64;
        let live = full.cost.live + full.cost.stmts;
        let per_live = full.cost.copied as f64 / live as f64;
        eprintln!(
            "{shape}: borrowck {:.0} ms; +{stmts} statements: applied {applied:.2}, copied {copied:.2} per \
             statement; copied {} for {} live locals per block and {} statements ({per_live:.2})",
            full.ms, full.cost.copied, full.cost.live, full.cost.stmts
        );
        assert!(stmts >= 4_000, "{shape}: {stmts}");
        if full.ms > MAX_MS {
            bad.push(format!("{shape}: borrowck took {:.0} ms", full.ms));
        }
        if applied > max_applied as f64 {
            bad.push(format!("{shape}: {applied:.2} statements applied per statement"));
        }
        if max_copied > 0 && copied > max_copied as f64 {
            bad.push(format!("{shape}: {copied:.2} state entries copied per statement"));
        }
        if per_live > MAX_PER_LIVE {
            bad.push(format!("{shape}: {per_live:.2} state entries copied per live local and block"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
