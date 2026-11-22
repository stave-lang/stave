//! The borrow checker's dataflow on very large fns (`large_fns_stay_cheap`): fns of 10 000
//! statements in the shapes that cost a dataflow with a full state per block the most are
//! checked within a time bound, and the dataflow's own counts stay under fixed bounds, which no
//! machine load changes: per statement the statements applied, the state entries copied (a
//! state is copied only into or out of an entry state), the loans counted in or out of the live
//! loans, the atoms of the sets entered or joined, and in all the times the loans reached
//! through the live ones were searched again and the whole states made.

use std::path::{Path, PathBuf};

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

fn check(root: &Path, threads: usize) -> Output {
    let o = Options::new(Goal::Check).threads(Some(threads));
    pipeline::run(root, &Library::in_dir(&repo()), &o).unwrap()
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
        // n / 4 borrows of one view held to the end in a grown body, n / 20 joins after them:
        // every search meets all the borrows of the view, which hold one set
        "views" => {
            o += "view struct V { r: &u32 }\nfn main() -> i32 {\n";
            o += BASE;
            o += "    let z: u32 = 9;\n    let y: u32 = 8;\n";
            o += "    let mut vr = vec.Vec::<&u32>.new();\n    vr.push(&z);\n    let mut u = V.{ r: &y };\n";
            for i in 0..4 {
                o += &format!("    if cond(s) {{ u = V.{{ r: &a{i} }}; }}\n");
            }
            for i in 0..n / 4 {
                o += &format!("    let r{i} = &u;\n");
            }
            for i in 0..n / 20 {
                o += &format!("    if cond(s) {{ s = s +% {}; }}\n", i % 100);
            }
            for i in (0..n / 4).rev() {
                o += &format!("    s = s +% *r{i}.r;\n");
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
    let out = check(&path, 2);
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
    /// Whole states made at 10 000 statements beyond the ones made at 5 000: a body makes one,
    /// and its solve, checks and replays all walk that one.
    states: u64,
}

const SHAPES: [Shape; 12] = [
    Shape { name: "line", errors: 0, applied: 4, copied: 1, counted: 2, atoms: 2, rebuilt: 0, states: 0 },
    Shape { name: "loans", errors: 0, applied: 4, copied: 1, counted: 2, atoms: 2, rebuilt: 0, states: 0 },
    Shape { name: "locals", errors: 0, applied: 4, copied: 1, counted: 2, atoms: 2, rebuilt: 0, states: 0 },
    Shape { name: "calls", errors: 0, applied: 4, copied: 1, counted: 3, atoms: 3, rebuilt: 0, states: 0 },
    Shape { name: "match", errors: 0, applied: 6, copied: 2, counted: 2, atoms: 2, rebuilt: 0, states: 0 },
    Shape { name: "joins", errors: 0, applied: 6, copied: 5, counted: 2, atoms: 2, rebuilt: 0, states: 0 },
    // every loan is live across every inner `if`: the copies follow the live locals per block
    Shape { name: "nest", errors: 0, applied: 6, copied: 0, counted: 2, atoms: 2, rebuilt: 0, states: 0 },
    // one more round after the store; the reached loans are searched where the borrow of the
    // view comes and goes, not per statement
    Shape { name: "grown", errors: 0, applied: 6, copied: 1, counted: 2, atoms: 3, rebuilt: 12, states: 0 },
    // solved twice: without and with an entry state per block
    Shape { name: "stale", errors: 0, applied: 10, copied: 1, counted: 2, atoms: 4, rebuilt: 20, states: 0 },
    // a search per borrow of the view and per join; each reads the view's set once
    Shape {
        name: "views",
        errors: 0,
        applied: 6,
        copied: 0,
        counted: 12,
        atoms: 12,
        rebuilt: 11_000,
        states: 0,
    },
    // each E0501 replays its block to find the later use: 20 replays of 15 000 statements
    Shape { name: "errors", errors: 1, applied: 36, copied: 1, counted: 1, atoms: 1, rebuilt: 0, states: 0 },
    Shape { name: "owners", errors: 0, applied: 6, copied: 1, counted: 1, atoms: 1, rebuilt: 0, states: 0 },
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
        let states = full.cost.states.saturating_sub(half.cost.states);
        eprintln!(
            "{shape}: borrowck {:.0} ms; +{stmts} statements, per statement: applied {applied:.2}, copied \
             {copied:.2}, counted {counted:.2}, atoms {atoms:.2}; searches {rebuilt}; states {} ({states} \
             more than at half the size); copied {} for {} live locals per block and {} statements ({per_live:.2}); grown \
             {grown}, per-block {exact}",
            full.ms, full.cost.states, full.cost.copied, full.cost.live, full.cost.stmts
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
        if states > sh.states {
            bad.push(format!("{shape}: {states} whole states made (bound {})", sh.states));
        }
        if per_live > MAX_PER_LIVE {
            bad.push(format!("{shape}: {per_live:.2} state entries copied per live local and block"));
        }
        // the store of `grown`, `stale` and `views` grows a view; only `stale` needs the per-block mode
        let want = (u64::from(matches!(shape, "grown" | "stale" | "views")), u64::from(shape == "stale"));
        if (grown, exact) != want {
            bad.push(format!("{shape}: {grown} bodies grown, {exact} with an entry state per block"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
