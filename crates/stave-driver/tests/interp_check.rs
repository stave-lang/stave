//! Checking mode of the interpreter: each program of `tests/memcheck/` makes one memory error
//! through `unsafe` code. Run with checking on, it ends with the error its first line names,
//! `//~ MEMCHECK <class> <line>`: the report's class, and a frame of the access trace at that
//! line of the program.

use std::path::{Path, PathBuf};

use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_ir::memcheck::Class;
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn programs() -> Vec<PathBuf> {
    let dir = repo().join("tests/memcheck");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "stv"))
        .collect();
    v.sort();
    v
}

/// (class, line) of the `//~ MEMCHECK` header.
fn expected(p: &Path) -> (String, u32) {
    let src = std::fs::read_to_string(p).unwrap();
    let head = src.lines().next().and_then(|l| l.strip_prefix("//~ MEMCHECK ")).unwrap_or_else(|| {
        panic!("{}: no `//~ MEMCHECK <class> <line>` first line", p.display())
    });
    let (class, line) = head.split_once(' ').unwrap();
    (class.to_string(), line.trim().parse().unwrap())
}

/// Whether a trace line is at `line` of `file`: exactly, or, for an instruction without a site
/// of its own (`near <site>`), at most 3 lines before it within the same fn.
fn at_line(l: &str, file: &str, line: u32) -> bool {
    if l.contains(&format!("({file}:{line}:")) {
        return true;
    }
    let Some(rest) = l.split_once(&format!("(near {file}:")).map(|x| x.1) else { return false };
    let n: u32 = rest.split(':').next().and_then(|n| n.parse().ok()).unwrap_or(0);
    n <= line && line - n <= 3
}

#[test]
fn each_class_is_reported_at_its_site() {
    let progs = programs();
    let mut seen = std::collections::BTreeSet::new();
    let mut failures = Vec::new();
    let target = Triple::parse("x86_64-linux").unwrap();
    for p in &progs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let (class, line) = expected(p);
        let mut o = Options::new(Goal::Check).threads(Some(4));
        o.lower = true;
        o.target = Some(target);
        let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
        assert!(!out.has_errors(), "{name}: diagnostics {:?}", out.diagnostics);
        let m = out.module.as_ref().unwrap_or_else(|| panic!("{name}: no module"));
        let lim = Limits { check: true, ..Limits::default() };
        let err = match interp::run(m, "main.main", target, &[], &[], lim) {
            Ok(r) => {
                failures.push(format!("{name}: no report, exit {}", r.ran.exit));
                continue;
            }
            Err(e) => e,
        };
        let head = format!("interp: memory error [{class}]");
        let trace: Vec<&str> = err.lines().skip(1).take_while(|l| !l.contains("allocated at")).collect();
        if !err.starts_with(&head) || !trace.iter().any(|l| at_line(l, &name, line)) {
            failures.push(format!("{name}: want [{class}] at line {line}, got:\n{err}"));
        }
        seen.insert(class);
    }
    for c in Class::ALL {
        if !seen.contains(c.name()) {
            failures.push(format!("no program of class {}", c.name()));
        }
    }
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}
