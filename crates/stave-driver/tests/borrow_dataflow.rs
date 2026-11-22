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
//!   dataflow the most are checked within a time bound, and the dataflow's own counts stay under
//!   fixed bounds, which no machine load changes: per statement the statements applied, the
//!   state entries copied (a state is copied only into or out of an entry state), the loans
//!   counted in or out of the live loans, the atoms of the sets entered or joined, and in all
//!   the times the loans reached through the live ones were searched again.

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
fn cond(x: u32) -> bool { return x > 3; }
fn eat(s: string.String) -> usize { return s.len(); }
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
        // loans held to the end in a body where a store grows a view object: `stale` with a
        // read, before the store, through a borrow of a reference written while borrowed (one
        // E0501; the body needs an entry state per block)
        "grown" | "stale" => {
            o += "fn main() -> i32 {\n";
            o += BASE;
            o += "    let z: u32 = 9;\n";
            if shape == "stale" {
                o += "    let mut sx: &u32 = &a0;\n    let sp = &sx;\n    sx = &a1;\n    let sy: &u32 = *sp;\n";
            }
            o += "    let mut vr = vec.Vec::<&u32>.new();\n    vr.push(&z);\n";
            for i in 0..n / 2 {
                o += &format!("    let r{i} = &a{};\n", i % 4);
            }
            for i in (0..n / 2).rev() {
                o += &format!("    s = s +% *r{i};\n");
            }
            if shape == "stale" {
                o += "    s = s +% *sy;\n";
            }
            o += "    let k: usize = vr.len();\n";
        }
        // one E0501 per 500 statements: a write of a local whose loan is used right after
        "errors" => {
            o += "fn main() -> i32 {\n";
            o += BASE;
            o += "    let r = &a0;\n";
            for i in 0..n {
                if i % 500 == 499 {
                    o += &format!("    a0 = {};\n    s = s +% *r;\n", i % 100);
                } else {
                    o += "    s = s +% a1;\n";
                }
            }
            o += "    s = s +% *r;\n";
        }
        // owners moved and set again on one edge of a join each, all used three times after
        "owners" => {
            o += "fn main() -> i32 {\n";
            o += BASE;
            for i in 0..n / 6 {
                o += &format!("    let mut t{i} = string.String.new();\n");
            }
            for i in 0..n / 6 {
                o += &format!(
                    "    if cond(s) {{ let k{i}: usize = eat(t{i}); t{i} = string.String.new(); }}\n"
                );
            }
            for i in 0..n / 6 {
                o += &format!(
                    "    let n{i}: usize = t{i}.len();\n    let m{i}: usize = t{i}.len();\n    let p{i}: usize = t{i}.len();\n"
                );
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

/// Checks the program of `shape` at `n` statements; it must get `errors` E0501 and nothing else.
fn measure(shape: &str, n: usize, errors: usize) -> Measured {
    let path = tmp("large").join(format!("{shape}_{n}.stv"));
    std::fs::write(&path, large(shape, n)).unwrap();
    let out = check(&path, Dataflow::Sparse, false, 2);
    let codes: Vec<&str> = out.diagnostics.iter().map(|d| d.code()).collect();
    let first: Vec<String> =
        out.diagnostics.iter().take(3).map(|d| format!("{} {}", d.code(), d.message())).collect();
    assert!(
        codes.len() == errors && codes.iter().all(|c| *c == "E0501") && out.failure.is_none(),
        "{shape} {n}: {} diagnostics, {first:?} {:?}",
        codes.len(),
        out.failure
    );
    let ms = out.timings.iter().find(|t| t.stage == Stage::Borrowck).unwrap().ms;
    Measured { ms, cost: out.borrowck.as_ref().unwrap().cost }
}

/// One gated shape. The bounds are per statement of the CFG that the fn gains from 5 000 to
/// 10 000 source statements, and about twice what the dataflow needs; 0: no bound of its own.
struct Shape {
    name: &'static str,
    /// E0501 diagnostics per 500 source statements (`stale`: one in all).
    errors: usize,
    /// Statements applied: once per round and once by the checks; an E0501 replays its block
    /// to find the later use.
    applied: u64,
    /// State entries copied into or out of an entry state. A straight line copies none.
    copied: u64,
    /// Loans counted in or out of the live loans, or met by a search.
    counted: u64,
    /// Atoms of the sets entered into the set table or joined there.
    atoms: u64,
    /// Searches of the loans reached through the live ones, in the whole program.
    rebuilt: u64,
}

const SHAPES: [Shape; 11] = [
    Shape { name: "line", errors: 0, applied: 4, copied: 1, counted: 2, atoms: 2, rebuilt: 0 },
    Shape { name: "loans", errors: 0, applied: 4, copied: 1, counted: 2, atoms: 2, rebuilt: 0 },
    Shape { name: "locals", errors: 0, applied: 4, copied: 1, counted: 2, atoms: 2, rebuilt: 0 },
    Shape { name: "calls", errors: 0, applied: 4, copied: 1, counted: 3, atoms: 3, rebuilt: 0 },
    Shape { name: "match", errors: 0, applied: 6, copied: 2, counted: 2, atoms: 2, rebuilt: 0 },
    Shape { name: "joins", errors: 0, applied: 6, copied: 5, counted: 2, atoms: 2, rebuilt: 0 },
    // every loan is live across every inner `if`: the copies follow the live locals per block
    Shape { name: "nest", errors: 0, applied: 6, copied: 0, counted: 2, atoms: 2, rebuilt: 0 },
    // one more round after the store; the reached loans are searched where the borrow of the
    // view comes and goes, not per statement
    Shape { name: "grown", errors: 0, applied: 6, copied: 1, counted: 2, atoms: 3, rebuilt: 12 },
    // solved twice: without and with an entry state per block
    Shape { name: "stale", errors: 0, applied: 10, copied: 1, counted: 2, atoms: 4, rebuilt: 20 },
    // each E0501 replays its block to find the later use: 20 replays of 15 000 statements
    Shape { name: "errors", errors: 1, applied: 36, copied: 1, counted: 1, atoms: 1, rebuilt: 0 },
    Shape { name: "owners", errors: 0, applied: 6, copied: 1, counted: 1, atoms: 1, rebuilt: 0 },
];

/// The most state entries copied per (local, block) pair with the local live at the block's
/// entry, plus one per statement: an entry state is copied a fixed number of times per round.
const MAX_PER_LIVE: f64 = 16.0;

/// The borrowck stage of one fn of 10 000 statements, in a debug build on a loaded machine.
const MAX_MS: f64 = 60_000.0;

#[test]
fn large_fns_stay_cheap() {
    let mut bad = Vec::new();
    // what the shipped library's bodies add to every program
    let mut lib = None;
    for sh in SHAPES {
        let shape = sh.name;
        let errors = |n: usize| if shape == "stale" { 1 } else { sh.errors * (n / 500) };
        let (half, full) = (measure(shape, 5_000, errors(5_000)), measure(shape, 10_000, errors(10_000)));
        let lib = *lib.get_or_insert(full.cost);
        let stmts = full.cost.stmts - half.cost.stmts;
        let per = |a: u64, b: u64| (a - b) as f64 / stmts as f64;
        let applied = per(full.cost.applied, half.cost.applied);
        let copied = per(full.cost.copied, half.cost.copied);
        let counted = per(full.cost.counted, half.cost.counted);
        let atoms = per(full.cost.atoms, half.cost.atoms);
        let live = full.cost.live + full.cost.stmts;
        let per_live = full.cost.copied as f64 / live as f64;
        let (grown, exact) = (full.cost.grown - lib.grown, full.cost.exact - lib.exact);
        let rebuilt = full.cost.rebuilt - lib.rebuilt;
        eprintln!(
            "{shape}: borrowck {:.0} ms; +{stmts} statements, per statement: applied {applied:.2}, copied \
             {copied:.2}, counted {counted:.2}, atoms {atoms:.2}; searches {rebuilt}; copied {} for {} live \
             locals per block and {} statements ({per_live:.2}); grown {grown}, per-block {exact}",
            full.ms, full.cost.copied, full.cost.live, full.cost.stmts
        );
        assert!(stmts >= 4_000, "{shape}: {stmts}");
        if full.ms > MAX_MS {
            bad.push(format!("{shape}: borrowck took {:.0} ms", full.ms));
        }
        for (what, got, max) in [
            ("statements applied", applied, sh.applied),
            ("state entries copied", copied, sh.copied),
            ("loans counted", counted, sh.counted),
            ("set atoms", atoms, sh.atoms),
        ] {
            if max > 0 && got > max as f64 {
                bad.push(format!("{shape}: {got:.2} {what} per statement (bound {max})"));
            }
        }
        if rebuilt > sh.rebuilt {
            bad.push(format!("{shape}: {rebuilt} searches of the reached loans (bound {})", sh.rebuilt));
        }
        if per_live > MAX_PER_LIVE {
            bad.push(format!("{shape}: {per_live:.2} state entries copied per live local and block"));
        }
        // the store of `grown` and `stale` grows a view; only `stale` needs the per-block mode
        let want = (u64::from(shape == "grown" || shape == "stale"), u64::from(shape == "stale"));
        if (grown, exact) != want {
            bad.push(format!("{shape}: {grown} bodies grown, {exact} with an entry state per block"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
