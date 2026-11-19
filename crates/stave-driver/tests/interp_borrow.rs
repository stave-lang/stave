//! Borrow tracking of the interpreter (contracts/borrow-model.md) on the programs of
//! `tests/borrowcheck/`. The first line of each says what the run must give:
//! `//~ BORROWCHECK <class> <line> <cause line>`: a violation of that class whose violating
//! event and whose cause are at those lines (cause line 0: the cause is in another file);
//! `//~ BORROWCHECK ok`: no report and exit status 0.

use std::path::{Path, PathBuf};

use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn programs() -> Vec<PathBuf> {
    let dir = repo().join("tests/borrowcheck");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
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
    let progs = programs();
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
                let find = |prefix: &str| e.lines().find(|l| l.starts_with(prefix)).unwrap_or("");
                let ok = e.starts_with(&format!("interp: borrow violation [{class}]"))
                    && at_line(find("  at "), &name, line)
                    && (cause == 0 || at_line(find("  cause "), &name, cause));
                if !ok {
                    failures
                        .push(format!("{name}: want [{class}] at line {line}, cause at {cause}, got:\n{e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The run programs of the conformance suite (all accepted, most without `unsafe` of their
/// own) under tracking: a report is a hole of the static rules, a violation in std or the
/// runtime, or a fault of the tracking mode.
#[test]
#[ignore = "the tracking mode does not see every event yet: its reports are to be read, not gated on"]
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
    for p in &progs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let mut o = Options::new(Goal::Check).threads(Some(4));
        o.lower = true;
        o.borrow_marks = true;
        o.target = Some(target);
        let out = match pipeline::run(p, &Library::in_dir(&repo()), &o) {
            Ok(out) if !out.has_errors() && out.module.is_some() => out,
            _ => {
                other.push(format!("{name}: not lowered"));
                continue;
            }
        };
        let m = out.module.as_ref().unwrap();
        let mut files = Vec::new();
        if p.is_dir() {
            for f in std::fs::read_dir(p).unwrap().flatten().map(|e| e.path()).filter(|f| f.is_file()) {
                files.push((
                    f.file_name().unwrap().to_string_lossy().into_owned(),
                    std::fs::read(&f).unwrap(),
                ));
            }
        }
        let lim = Limits { check: true, borrow: true, ..Limits::default() };
        match interp::run(m, "main.main", target, &[], &files, lim) {
            Ok(_) => clean += 1,
            Err(e) if e.starts_with("interp: borrow violation") => reports.push(format!("{name}: {e}")),
            Err(e) => other.push(format!("{name}: {}", e.lines().next().unwrap_or(""))),
        }
    }
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
