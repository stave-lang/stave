//! Borrow tracking of the interpreter (contracts/borrow-model.md) on the programs of
//! `tests/borrowcheck/`. The first line of each says what the run must give:
//! `//~ BORROWCHECK <class> <line> <cause line>`: a violation of that class whose violating
//! event and whose cause are at those lines (cause line 0: the cause is in another file);
//! `//~ BORROWCHECK signature <line> <line>`: no violation, and a signature report (M24) for
//! the return at the first line, of a reference whose tag was made at the second;
//! `//~ BORROWCHECK ok`: no report and exit status 0.
//!
//! `tests/borrowcheck/holes/` holds programs the borrow checker rejects, lowered all the same
//! ([`Options::unchecked_borrows`]): what the dynamic model says of an execution the static
//! rules forbid. A checker that accepts one of them by mistake has a hole, and the report
//! here is what shows it. Their first line is as above, or
//! `//~ BORROWCHECK gap`: the tracking mode does not see the violating event, and the run gives
//! no report.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn programs_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "stv"))
        .collect();
    v.sort();
    v
}

/// `None`: no report; else (class, line, cause line).
fn expected(p: &Path) -> Option<(String, u32, u32)> {
    let src = std::fs::read_to_string(p).unwrap();
    let head = src
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("//~ BORROWCHECK "))
        .unwrap_or_else(|| panic!("{}: no `//~ BORROWCHECK ..` first line", p.display()));
    if head.trim() == "ok" {
        return None;
    }
    let mut w = head.split_whitespace();
    let class = w.next().unwrap().to_string();
    let mut line =
        || w.next().and_then(|n| n.parse().ok()).unwrap_or_else(|| panic!("{}: header", p.display()));
    Some((class, line(), line()))
}

/// Whether a report line is at `line` of `file`: exactly, or, for an instruction without a site
/// of its own (`near <site>`), at most 3 lines before it.
fn at_line(l: &str, file: &str, line: u32) -> bool {
    if l.contains(&format!("({file}:{line}:")) {
        return true;
    }
    let Some(rest) = l.split_once(&format!("(near {file}:")).map(|x| x.1) else { return false };
    let n: u32 = rest.split(':').next().and_then(|n| n.parse().ok()).unwrap_or(0);
    n <= line && line - n <= 3
}

#[test]
fn each_program_gives_its_report() {
    let mut failures = Vec::new();
    let target = Triple::parse("x86_64-linux").unwrap();
    let progs = programs_in(&repo().join("tests/borrowcheck"));
    assert!(!progs.is_empty());
    for p in &progs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let mut o = Options::new(Goal::Check).threads(Some(4));
        o.lower = true;
        o.borrow_marks = true;
        o.target = Some(target);
        let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
        assert!(!out.has_errors(), "{name}: diagnostics {:?}", out.diagnostics);
        let m = out.module.as_ref().unwrap_or_else(|| panic!("{name}: no module"));
        let lim = Limits { check: true, borrow: true, ..Limits::default() };
        let got = interp::run(m, "main.main", target, &[], &[], lim);
        match (expected(p), got) {
            (None, Ok(r)) if r.ran.exit == 0 => {}
            (None, Ok(r)) => failures.push(format!("{name}: exit {}", r.ran.exit)),
            (None, Err(e)) => failures.push(format!("{name}: want no report, got:\n{e}")),
            (Some(_), Ok(r)) => failures.push(format!("{name}: no report, exit {}", r.ran.exit)),
            (Some((class, line, cause)), Err(e)) => {
                if !is_report(&e, &name, &class, line, cause) {
                    failures
                        .push(format!("{name}: want [{class}] at line {line}, cause at {cause}, got:\n{e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A suite that takes the default options and limits runs under tracking exactly when
/// `STAVE_INTERP_CHECK=borrow` is set, and the checked IR is the same module either way.
#[test]
fn default_options_follow_the_environment() {
    let on = std::env::var_os("STAVE_INTERP_CHECK").is_some_and(|v| v == "borrow");
    let target = Triple::parse("x86_64-linux").unwrap();
    let p = repo().join("tests/borrowcheck/write_under_shared.stv");
    let mut o = Options::new(Goal::Check).threads(Some(4));
    o.lower = true;
    o.target = Some(target);
    assert_eq!(o.tracked, on);
    let out = pipeline::run(&p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{:?}", out.diagnostics);
    assert_eq!(out.tracked.is_some(), on);
    let marks =
        |m: &stave_ir::Module| m.funcs.iter().filter_map(|f| f.body.as_ref()).any(|b| !b.borrows.is_empty());
    assert!(!marks(out.module.as_ref().unwrap()));
    assert_eq!(marks(out.for_interp().unwrap()), on);
    let got = interp::run(out.for_interp().unwrap(), "main.main", target, &[], &[], Limits::default());
    match got {
        Err(e) if on => assert!(e.starts_with("interp: borrow violation [write-under-shared]"), "{e}"),
        Ok(r) if !on => assert_eq!(r.ran.exit, 1),
        r => panic!("tracking {on}: {r:?}"),
    }
}

/// `tests/borrowcheck/std/`: safe programs (no `unsafe` of their own) that drive the `unsafe`
/// sites of std and the runtime with edge inputs. Each runs in the memory checking mode and
/// under borrow tracking, on both CPU architectures; a report in either is a fault of std, of
/// the runtime, or of an interpreter mode. First line: `//~ BORROWCHECK ok` (exit 0) or
/// `//~ BORROWCHECK exit <n>`.
#[test]
fn std_programs_have_no_report() {
    let dir = repo().join("tests/borrowcheck/std");
    let progs = programs_in(&dir);
    assert!(!progs.is_empty());
    let files: Vec<(String, Vec<u8>)> = vec![
        ("input.txt".into(), b"hello\n".to_vec()),
        ("empty.txt".into(), Vec::new()),
        ("bad.txt".into(), vec![0x61, 0xC0, 0x80]),
        ("big.txt".into(), (0..70_000u32).map(|i| b'a' + (i % 26) as u8).collect()),
    ];
    let mut failures = Vec::new();
    for p in &progs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(p).unwrap();
        assert!(!src.contains("unsafe("), "{name}: a std driver program has no `unsafe` of its own");
        let head = src.lines().next().and_then(|l| l.strip_prefix("//~ BORROWCHECK ")).map(str::trim);
        let want = match head {
            Some("ok") => 0,
            Some(h) => h
                .strip_prefix("exit ")
                .and_then(|n| n.parse::<i32>().ok())
                .unwrap_or_else(|| panic!("{name}: header `{h}`")),
            None => panic!("{name}: no `//~ BORROWCHECK ..` first line"),
        };
        for t in ["x86_64-linux", "aarch64-linux"] {
            let target = Triple::parse(t).unwrap();
            let mut o = Options::new(Goal::Check).threads(Some(4));
            o.lower = true;
            o.borrow_marks = true;
            o.target = Some(target);
            let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
            if out.has_errors() || out.module.is_none() {
                failures.push(format!("{name} ({t}): diagnostics {:?}", out.diagnostics));
                continue;
            }
            let m = out.module.as_ref().unwrap();
            for borrow in [false, true] {
                let lim = Limits { check: true, borrow, max_steps: 400_000_000, ..Limits::default() };
                let mode = if borrow { "borrow" } else { "memory" };
                match interp::run(m, "main.main", target, &[&b"one"[..]], &files, lim) {
                    Ok(r) if r.ran.exit == want && (want != 101 || r.ran.stderr.starts_with(b"panic: ")) => {}
                    Ok(r) => failures.push(format!(
                        "{name} ({t}, {mode}): exit {}, want {want}; stderr:\n{}",
                        r.ran.exit,
                        String::from_utf8_lossy(&r.ran.stderr)
                    )),
                    Err(e) => failures.push(format!("{name} ({t}, {mode}):\n{e}")),
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A signature report on a closure body names the closure by the site of its literal and the
/// fn that holds it.
#[test]
fn signature_report_names_the_closure() {
    let target = Triple::parse("x86_64-linux").unwrap();
    let p = repo().join("tests/borrowcheck/closure_store_capture.stv");
    let mut o = Options::new(Goal::Check).threads(Some(2));
    o.lower = true;
    o.borrow_marks = true;
    o.target = Some(target);
    let out = pipeline::run(&p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{:?}", out.diagnostics);
    let m = out.module.as_ref().expect("checked IR");
    let lim = Limits { check: true, borrow: true, ..Limits::default() };
    let e = interp::run(m, "main.main", target, &[], &[], lim).unwrap_err();
    let first = e.lines().next().unwrap_or("");
    assert!(
        first.ends_with("param 1 of the closure at closure_store_capture.stv:11:15 in main.main took a reference the `from` clauses do not give"),
        "{e}"
    );
}

/// Borrow tracking walks the values it is given with lists, not by recursion: a view nested
/// 341 levels deep, loaded through raw pointers (each load gives its reference words new tags),
/// runs clean on a 256 KB native stack, as it does without tracking.
#[test]
fn deep_values_need_no_deep_stack_under_tracking() {
    let target = Triple::parse("x86_64-linux").unwrap();
    let p = repo().join("tests/borrowcheck/ok_deep_view_raw_load.stv");
    let mut o = Options::new(Goal::Check).threads(Some(2));
    o.lower = true;
    o.borrow_marks = true;
    o.target = Some(target);
    let out = pipeline::run(&p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{:?}", out.diagnostics);
    let m = out.module.expect("checked IR");
    for borrow in [false, true] {
        let m = m.clone();
        let ran = std::thread::Builder::new()
            .stack_size(256 << 10)
            .spawn(move || {
                let lim = Limits { check: true, borrow, ..Limits::default() };
                interp::run(&m, "main.main", target, &[], &[], lim).map(|r| r.ran.exit)
            })
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(ran, Ok(0), "tracking {borrow}");
    }
}

/// What the borrow checker and the tracking interpreter say of one program lowered without
/// regard to the checker.
struct Verdict {
    /// Error codes of the borrow checker.
    rejected: Vec<&'static str>,
    /// Why the program did not run (other errors, or not lowered).
    stopped: Option<String>,
    /// The interpreter's error, or the exit status.
    ran: Result<i32, String>,
}

fn verdict(p: &Path, target: Triple) -> Verdict {
    let mut o = Options::new(Goal::Check).threads(Some(4));
    o.lower = true;
    o.borrow_marks = true;
    o.unchecked_borrows = true;
    o.target = Some(target);
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    let errors = || out.diagnostics.iter().filter(|d| d.severity() == stave_diag::Severity::Error);
    let borrowck = |c: &str| pipeline::Stage::of_code(c) == Some(pipeline::Stage::Borrowck);
    let rejected: Vec<&'static str> = errors().map(|d| d.code()).filter(|c| borrowck(c)).collect();
    let other: Vec<&'static str> = errors().map(|d| d.code()).filter(|c| !borrowck(c)).collect();
    let Some(m) = out.module.as_ref().filter(|_| other.is_empty()) else {
        let why =
            format!("not lowered: errors {other:?}, failure {:?}", out.failure.as_ref().map(|f| f.stage()));
        return Verdict { rejected, stopped: Some(why), ran: Ok(0) };
    };
    // `STAVE_BORROWCHECK_PLAIN`: the same module with the memory checks only (to compare the
    // time and memory of a run with and without tracking)
    let borrow = std::env::var_os("STAVE_BORROWCHECK_PLAIN").is_none();
    // `STAVE_BORROWCHECK_STEPS`: the step limit (a directory with programs that never end)
    let max_steps =
        std::env::var("STAVE_BORROWCHECK_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000_000);
    let lim = Limits { check: true, borrow, max_steps, ..Limits::default() };
    let ran = interp::run(m, "main.main", target, &[], &[], lim).map(|r| r.ran.exit);
    Verdict { rejected, stopped: None, ran }
}

/// Whether the error of a run is the report (class, line, cause line) of program `name`.
fn is_report(e: &str, name: &str, class: &str, line: u32, cause: u32) -> bool {
    let find = |prefix: &str| e.lines().find(|l| l.starts_with(prefix)).unwrap_or("");
    if class == "signature" {
        return e.starts_with("interp: borrow signature [signature]")
            && at_line(find("  at "), name, line)
            && (cause == 0 || at_line(find("  tag made at "), name, cause));
    }
    e.starts_with(&format!("interp: borrow violation [{class}]"))
        && at_line(find("  at "), name, line)
        && (cause == 0 || at_line(find("  cause "), name, cause))
}

#[test]
fn each_rejected_program_gives_its_report() {
    let dir = repo().join("tests/borrowcheck/holes");
    let target = Triple::parse("x86_64-linux").unwrap();
    let progs = programs_in(&dir);
    assert!(!progs.is_empty());
    let mut failures = Vec::new();
    for p in &progs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let v = verdict(p, target);
        if v.rejected.is_empty() {
            failures.push(format!("{name}: the borrow checker accepts it"));
        }
        if let Some(why) = &v.stopped {
            failures.push(format!("{name}: {why}"));
            continue;
        }
        let gap = std::fs::read_to_string(p).unwrap().starts_with("//~ BORROWCHECK gap");
        match (if gap { None } else { expected(p) }, v.ran) {
            (None, Err(e)) if gap && e.starts_with("interp: borrow ") => {
                failures.push(format!("{name}: marked `gap`, now reported:\n{e}"))
            }
            (None, _) if gap => {}
            (None, r) => failures.push(format!("{name}: a rejected program marked `ok`: {r:?}")),
            (Some((class, line, cause)), Err(e)) if is_report(&e, &name, &class, line, cause) => {}
            (Some((class, line, cause)), r) => {
                failures.push(format!("{name}: want [{class}] at line {line}, cause at {cause}, got:\n{r:?}"))
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Prints what the checker and the tracking interpreter say of each program in the directory
/// `STAVE_BORROWCHECK_DIR`: the way to sort new candidates for `tests/borrowcheck/holes/`.
#[test]
#[ignore = "a tool: it reads a directory named in the environment and asserts nothing"]
fn verdicts_of_a_directory() {
    let Some(dir) = std::env::var_os("STAVE_BORROWCHECK_DIR") else { return };
    let target = Triple::parse("x86_64-linux").unwrap();
    for p in programs_in(Path::new(&dir)) {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let v = verdict(&p, target);
        let ran = match (&v.stopped, &v.ran) {
            (Some(why), _) => why.clone(),
            (None, Ok(c)) => format!("no report, exit {c}"),
            (None, Err(e)) => e.lines().take(3).collect::<Vec<_>>().join(" | "),
        };
        println!("{name}: checker {:?}; {ran}", v.rejected);
    }
}

/// The run programs of the conformance suite (all accepted, most without `unsafe` of their
/// own) under tracking: a report is a hole of the static rules, a violation in std or the
/// runtime, or a fault of the tracking mode.
#[test]
#[ignore = "three runs of every run program, one of them 100M steps: minutes in a debug build"]
fn run_programs_have_no_report() {
    let dir = repo().join("tests/conformance/run");
    let mut progs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            !name.starts_with('.')
                && !name.starts_with('_')
                && (p.join("stave.toml").is_file() || p.extension().is_some_and(|x| x == "stv"))
        })
        .collect();
    progs.sort();
    let target = Triple::parse("x86_64-linux").unwrap();
    let (mut reports, mut other, mut clean) = (Vec::new(), Vec::new(), 0);
    // wall time of the interpreter: the canonical module with the memory checks, the module
    // with the side table with the memory checks, and that module with tracking
    let (mut plain, mut marked, mut tracked) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    for p in &progs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let mut o = Options::new(Goal::Check).threads(Some(4));
        o.lower = true;
        o.tracked = true;
        o.target = Some(target);
        let out = match pipeline::run(p, &Library::in_dir(&repo()), &o) {
            Ok(out) if !out.has_errors() && out.module.is_some() && out.tracked.is_some() => out,
            _ => {
                other.push(format!("{name}: not lowered"));
                continue;
            }
        };
        let canonical = out.module.as_ref().unwrap();
        let m = out.tracked.as_ref().unwrap();
        let mut files = Vec::new();
        if p.is_dir() {
            for f in std::fs::read_dir(p).unwrap().flatten().map(|e| e.path()).filter(|f| f.is_file()) {
                files.push((
                    f.file_name().unwrap().to_string_lossy().into_owned(),
                    std::fs::read(&f).unwrap(),
                ));
            }
        }
        let checked = Limits { check: true, borrow: false, ..Limits::default() };
        let t = Instant::now();
        let base = interp::run(canonical, "main.main", target, &[], &files, checked);
        let d0 = t.elapsed();
        let t = Instant::now();
        let with_table = interp::run(m, "main.main", target, &[], &files, checked);
        let d1 = t.elapsed();
        let lim = Limits { check: true, borrow: true, ..Limits::default() };
        let t = Instant::now();
        let got = interp::run(m, "main.main", target, &[], &files, lim);
        let d2 = t.elapsed();
        plain += d0;
        marked += d1;
        tracked += d2;
        println!(
            "{name}: {} steps, {} with the side table; {} ms plain, {} ms with the side table, {} ms tracked",
            base.as_ref().map_or(0, |r| r.steps),
            with_table.as_ref().map_or(0, |r| r.steps),
            d0.as_millis(),
            d1.as_millis(),
            d2.as_millis()
        );
        match got {
            Ok(_) => clean += 1,
            Err(e) if e.starts_with("interp: borrow ") => reports.push(format!("{name}: {e}")),
            Err(e) => other.push(format!("{name}: {}", e.lines().next().unwrap_or(""))),
        }
    }
    println!(
        "{} programs: {clean} clean, {} report, {} not run; {} ms plain, {} ms with the side table, {} ms tracked",
        progs.len(),
        reports.len(),
        other.len(),
        plain.as_millis(),
        marked.as_millis(),
        tracked.as_millis()
    );
    assert!(
        reports.is_empty(),
        "{} of {} programs report ({clean} clean, {} not run):\n{}\n--- not run\n{}",
        reports.len(),
        progs.len(),
        other.len(),
        reports.join("\n"),
        other.join("\n")
    );
}
