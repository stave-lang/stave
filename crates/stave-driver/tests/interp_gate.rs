//! The interpreter's checking modes as a gate on accepted programs. Every positive run test of
//! the conformance suite (`tests/conformance/**`, the LANG.md blocks that run among them, in
//! `langmd/`) and every program of `tests/borrowcheck/` (accepted programs) runs in the
//! interpreter with borrow tracking on, which includes the memory checks
//! (`STAVE_INTERP_CHECK=borrow`, contracts/cli.md; contracts/borrow-model.md).
//!
//! - **No unsafe**: the program's own packages (every package but the built-in `std`, `runtime`
//!   and their dependencies) have no site in the unsafe inventory (`stave unsafe-report`,
//!   [`stave_sema::unsafety::inventory`]). A report on such a program fails: the language says
//!   it cannot break the model, so the checker has a hole or the tracker is wrong.
//! - **Unsafe**: the program runs clean, or ends with the report it declares
//!   (`//~ INTERP-REPORT <class>`, contracts/conformance-annotations.md; in `tests/borrowcheck/`
//!   the class of its `//~ BORROWCHECK <class> ..` line). Any other report fails, and so does a
//!   declared report that does not come or a declaration on a program without unsafe.
//! - **Not runnable**: the interpreter did not run the program as written: no CPU target, a
//!   pipeline failure, an interpreter error that is no report (step limit, an operation it does
//!   not support), or a run whose exit status, stdout, or stderr differs from the program's
//!   expectations (no `[run] stack` limit; thread ids in panic lines are not compared), or, in
//!   `check`, a program listed in [`SLOW`]. Counted and listed with the reason, never passed
//!   silently.
//! - **Open finding**: a report on a program without unsafe that is known and not fixed yet
//!   ([`OPEN`]). Printed with the counts; an entry whose program no longer gives that report
//!   fails the test, so the list only shrinks.
//!
//! The test prints the counts, the expected reports, the not-runnable programs and the open
//! findings, and the slowest runs.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use stave_driver::conformance::stage0::registry_codes;
use stave_driver::conformance::{TestKind, discover, load_spec};
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_sema::package::PackageKind;
use stave_target::{Arch, Triple};

/// Steps of one run: a program that needs more is not runnable here.
const MAX_STEPS: u64 = 300_000_000;
/// Programs run at once (each pipeline single-threaded).
const JOBS: usize = 4;
/// Worker stack: the parser and the interpreter recurse.
const STACK: usize = 64 << 20;
/// Programs that take most of the wall time, started first (scheduling only).
const FIRST: &[&str] = &["conformance/run/num_float_long.stv"];
/// Programs too slow for the gate of `check`, with the reason: listed as not runnable there, run by
/// [`with_the_slow_programs`] (the nightly `run.sh interp-gate-slow`).
const SLOW: &[(&str, &str)] = &[(
    "conformance/std/num_float_exp7.stv",
    "too slow for `check` under tracking: the 10^6-digit float parse takes 185 M steps",
)];

/// Reports on programs without unsafe that are known and not fixed yet: (program, class).
const OPEN: &[(&str, &str)] = &[
    // `&4u32` in a loop: the checker lets a borrow of a literal outlive the next evaluation of
    // its expression, the model ends that temporary there.
    ("conformance/borrow/pos_temp_loop.stv", "write-under-shared"),
];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// What a clean run must give.
struct Expect {
    exit: u8,
    stdout: Option<Vec<u8>>,
    stderr_contains: Vec<String>,
}

struct Case {
    name: String,
    path: PathBuf,
    package: bool,
    /// The target to lower for, or why there is none.
    target: Result<Triple, String>,
    release: bool,
    /// The declared report class.
    report: Option<String>,
    /// `None`: the run is not compared (a declared report ends it).
    expect: Option<Expect>,
}

enum Verdict {
    Clean,
    Expected(String),
    Open(String),
    NotRunnable(String),
    Fail(String),
}

fn x86() -> Triple {
    Triple::parse("x86_64-linux").unwrap()
}

/// The positive run tests of the conformance suite.
fn conformance_cases() -> Vec<Case> {
    let root = repo().join("tests/conformance");
    let codes = registry_codes();
    let mut v = Vec::new();
    for t in discover(&root).unwrap() {
        // a malformed test fails in its own suite
        let Ok(spec) = load_spec(&t, Some(&codes)) else { continue };
        if spec.kind() != TestKind::Positive || !spec.is_run_test() || spec.check_only || spec.parse_only {
            continue;
        }
        let target = if spec.targets.is_empty() {
            Ok(x86())
        } else {
            spec.targets.iter().copied().find(|t| t.arch() != Arch::Gpu).ok_or_else(|| {
                let ts: Vec<String> = spec.targets.iter().map(|t| t.to_string()).collect();
                format!("no CPU target ({})", ts.join(", "))
            })
        };
        let stdout =
            spec.stdout.as_ref().map(|p| std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())));
        let expect = spec.interp_report.is_none().then(|| Expect {
            exit: spec.exit.unwrap_or(0),
            stdout,
            stderr_contains: spec.stderr_contains.clone(),
        });
        v.push(Case {
            name: format!("conformance/{}", t.name),
            path: t.path.clone(),
            package: t.package,
            target,
            release: spec.args.iter().any(|a| a == "--release"),
            report: spec.interp_report.clone(),
            expect,
        });
    }
    v
}

/// The accepted programs of `tests/borrowcheck/` (first line `//~ BORROWCHECK ok` or
/// `//~ BORROWCHECK <class> <line> <cause line>`).
fn borrowcheck_cases() -> Vec<Case> {
    let dir = repo().join("tests/borrowcheck");
    let mut ps: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "stv"))
        .collect();
    ps.sort();
    ps.into_iter()
        .map(|p| {
            let src = std::fs::read_to_string(&p).unwrap();
            let head = src
                .lines()
                .next()
                .and_then(|l| l.strip_prefix("//~ BORROWCHECK "))
                .unwrap_or_else(|| panic!("{}: no `//~ BORROWCHECK ..` first line", p.display()));
            let report = (head.trim() != "ok").then(|| head.split_whitespace().next().unwrap().to_string());
            let expect =
                report.is_none().then_some(Expect { exit: 0, stdout: None, stderr_contains: Vec::new() });
            Case {
                name: format!("borrowcheck/{}", p.file_name().unwrap().to_string_lossy()),
                path: p,
                package: false,
                target: Ok(x86()),
                release: false,
                report,
                expect,
            }
        })
        .collect()
}

/// The data files a package program may read from its working directory.
fn data_files(c: &Case) -> Vec<(String, Vec<u8>)> {
    let mut v = Vec::new();
    if c.package {
        for e in std::fs::read_dir(&c.path).unwrap().flatten() {
            let f = e.path();
            if f.is_file() {
                v.push((f.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read(&f).unwrap()));
            }
        }
    }
    v
}

/// The unsafe sites of the program's own packages, as `file:line: kind`.
fn own_unsafe(out: &Output) -> Result<Vec<String>, String> {
    let tc = out.typeck.as_ref().ok_or("no type-checked program")?;
    let p = &out.program;
    let builtin = |s: &stave_sema::unsafety::UnsafeSite| {
        p.file(s.span.file)
            .and_then(|f| p.packages.get(f.package.0 as usize))
            .is_none_or(|k| k.kind == PackageKind::Builtin)
    };
    let mut v = Vec::new();
    for s in stave_sema::unsafety::inventory(p, tc).iter().filter(|s| !builtin(s)) {
        let line = out.files().locate(s.span).map_or(0, |l| l.line);
        v.push(format!("{}:{line}: {}", s.file, s.kind.name()));
    }
    Ok(v)
}

/// The class of an interpreter report (`interp: memory error [<class>]`,
/// `interp: borrow violation [<class>]`), or `None` for any other error.
fn report_class(e: &str) -> Option<&str> {
    let rest =
        e.strip_prefix("interp: memory error [").or_else(|| e.strip_prefix("interp: borrow violation ["))?;
    rest.split_once(']').map(|x| x.0)
}

/// `s` without the ` (thread <n>)` of panic lines. The interpreter has one thread: it runs a spawned
/// body on the thread that spawns it, so the ids in its panic lines are not the native ones.
fn without_thread_ids(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(k) = rest.find(" (thread ") {
        out.push_str(&rest[..k]);
        let after = &rest[k + " (thread ".len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after[digits..].starts_with(')') {
            rest = &after[digits + 1..];
        } else {
            out.push_str(" (thread ");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// How a clean run differs from what the program expects, if it does. `STDERR-CONTAINS` is
/// compared without thread ids ([`without_thread_ids`]).
fn differs(want: &Expect, r: &interp::Interpreted) -> Option<String> {
    let exit = r.ran.exit & 0xff;
    if exit != i32::from(want.exit) {
        return Some(format!("exit {exit}, want {}", want.exit));
    }
    if want.stdout.as_ref().is_some_and(|s| *s != r.ran.stdout) {
        return Some("stdout differs from the golden file".to_string());
    }
    let stderr = without_thread_ids(&String::from_utf8_lossy(&r.ran.stderr));
    want.stderr_contains
        .iter()
        .find(|s| !stderr.contains(&without_thread_ids(s)))
        .map(|s| format!("stderr lacks {s:?}"))
}

/// The verdict, the time of the interpreter run, and its steps.
fn judge(c: &Case, lib: &Library, slow: bool) -> (Verdict, Duration, u64) {
    let (v, took, steps) = judge_run(c, lib, slow);
    let v = match (v, OPEN.iter().find(|o| o.0 == c.name)) {
        (v @ (Verdict::Open(_) | Verdict::Fail(_)), _) => v,
        (Verdict::Clean, Some(_)) => Verdict::Fail("runs clean now: remove its entry from OPEN".to_string()),
        (_, Some(o)) => Verdict::Fail(format!(
            "listed in OPEN with [{}], which it no longer gives: remove the entry, or correct its class",
            o.1
        )),
        (v, None) => v,
    };
    (v, took, steps)
}

fn judge_run(c: &Case, lib: &Library, slow: bool) -> (Verdict, Duration, u64) {
    let none = Duration::ZERO;
    if let Some(s) = SLOW.iter().find(|s| !slow && s.0 == c.name) {
        return (Verdict::NotRunnable(s.1.to_string()), none, 0);
    }
    let target = match &c.target {
        Ok(t) => *t,
        Err(why) => return (Verdict::NotRunnable(why.clone()), none, 0),
    };
    let mut o = Options::new(Goal::Check).threads(Some(1));
    o.lower = true;
    o.borrow_marks = true;
    o.tracked = false;
    o.target = Some(target);
    o.release = c.release;
    let out = match pipeline::run(&c.path, lib, &o) {
        Ok(out) => out,
        Err(e) => return (Verdict::Fail(format!("pipeline: {e}")), none, 0),
    };
    if out.has_errors() {
        let codes: Vec<&str> = out.diagnostics.iter().map(|d| d.code()).collect();
        return (Verdict::Fail(format!("diagnostics {codes:?}")), none, 0);
    }
    if let Some(f) = &out.failure {
        return (Verdict::NotRunnable(format!("not lowered: {f}")), none, 0);
    }
    let Some(m) = out.module.as_ref() else { return (Verdict::NotRunnable("not lowered".into()), none, 0) };
    let own = match own_unsafe(&out) {
        Ok(v) => v,
        Err(e) => return (Verdict::Fail(e), none, 0),
    };
    if let Some(w) = &c.report
        && own.is_empty()
    {
        return (Verdict::Fail(format!("declares the report [{w}] but has no unsafe of its own")), none, 0);
    }
    let main = out.main.and_then(|f| m.func(f)).map_or_else(|| "main.main".to_string(), |f| f.path.clone());
    let lim = Limits { check: true, borrow: true, max_steps: MAX_STEPS, ..Limits::default() };
    let t = Instant::now();
    let got = interp::run(m, &main, target, &[], &data_files(c), lim);
    let took = t.elapsed();
    let steps = got.as_ref().map_or(0, |r| r.steps);
    let v = match (got, &c.report) {
        (Err(e), want) => match report_class(&e) {
            Some(class) if own.is_empty() => {
                if OPEN.iter().any(|o| o.0 == c.name && o.1 == class) {
                    Verdict::Open(class.to_string())
                } else {
                    Verdict::Fail(format!(
                        "a report on a program without unsafe (checker hole or tracker bug):\n{e}"
                    ))
                }
            }
            Some(class) if want.as_deref() == Some(class) => Verdict::Expected(class.to_string()),
            Some(_) => {
                Verdict::Fail(format!("unsafe at {}; declared report {want:?}, got:\n{e}", own.join(", ")))
            }
            None => Verdict::NotRunnable(e.lines().next().unwrap_or("").to_string()),
        },
        (Ok(r), Some(w)) => {
            Verdict::Fail(format!("declares the report [{w}], runs clean (exit {})", r.ran.exit))
        }
        (Ok(r), None) => match c.expect.as_ref().and_then(|x| differs(x, &r)) {
            None => Verdict::Clean,
            Some(d) => Verdict::NotRunnable(format!("the run differs from its expectations: {d}")),
        },
    };
    (v, took, steps)
}

#[test]
#[ignore = "minutes: run by `tools/ci-step/run.sh interp-gate`, a part of `check`"]
fn accepted_programs_run_clean_under_borrow_tracking() {
    gate(false);
}

/// The same gate with the programs of [`SLOW`] run too.
#[test]
#[ignore = "longer: run nightly by `tools/ci-step/run.sh interp-gate-slow`"]
fn with_the_slow_programs() {
    gate(true);
}

fn gate(slow: bool) {
    let mut cases = conformance_cases();
    cases.extend(borrowcheck_cases());
    let is_slow = |c: &Case| SLOW.iter().any(|s| s.0 == c.name);
    cases.sort_by_key(|c| !(FIRST.contains(&c.name.as_str()) || (slow && is_slow(c))));
    assert!(cases.len() > 150, "{} programs: discovery is broken", cases.len());
    let lib = Library::in_dir(&repo());
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<(Verdict, Duration, u64)>>> =
        Mutex::new((0..cases.len()).map(|_| None).collect());
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..JOBS {
            std::thread::Builder::new()
                .stack_size(STACK)
                .spawn_scoped(s, || {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(c) = cases.get(i) else { break };
                        let r = judge(c, &lib, slow);
                        results.lock().unwrap()[i] = Some(r);
                    }
                })
                .unwrap();
        }
    });
    let wall = t0.elapsed();
    let results: Vec<(Verdict, Duration, u64)> =
        results.into_inner().unwrap().into_iter().map(Option::unwrap).collect();
    let (mut clean, mut expected, mut open, mut skipped, mut failed) =
        (0, Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (c, (v, ..)) in cases.iter().zip(&results) {
        match v {
            Verdict::Clean => clean += 1,
            Verdict::Expected(class) => expected.push(format!("{}: [{class}]", c.name)),
            Verdict::Open(class) => open.push(format!("{}: [{class}]", c.name)),
            Verdict::NotRunnable(why) => skipped.push(format!("{}: {why}", c.name)),
            Verdict::Fail(why) => failed.push(format!("{}: {why}", c.name)),
        }
    }
    for o in OPEN.iter().map(|o| o.0).chain(SLOW.iter().map(|s| s.0)) {
        assert!(cases.iter().any(|c| c.name == o), "{o} is listed, but it is not a program here");
    }
    let mut slow: Vec<(Duration, u64, &str)> =
        cases.iter().zip(&results).map(|(c, r)| (r.1, r.2, c.name.as_str())).collect();
    slow.sort_by_key(|a| std::cmp::Reverse(a.0));
    println!(
        "{} programs under borrow tracking: {clean} ran clean, {} expected report, {} not runnable, {} open finding, {} failed; wall {} s",
        cases.len(),
        expected.len(),
        skipped.len(),
        open.len(),
        failed.len(),
        wall.as_secs()
    );
    println!("--- expected report\n{}", expected.join("\n"));
    println!("--- not runnable\n{}", skipped.join("\n"));
    println!("--- open finding (no unsafe, known report)\n{}", open.join("\n"));
    let slow: Vec<String> = slow
        .iter()
        .take(5)
        .map(|(d, steps, n)| format!("{n}: {} ms, {steps} steps", d.as_millis()))
        .collect();
    println!("--- slowest runs\n{}", slow.join("\n"));
    assert!(failed.is_empty(), "{} failed:\n{}", failed.len(), failed.join("\n"));
}
