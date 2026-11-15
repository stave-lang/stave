//! The run-test programs of `tests/conformance/run/` (package dirs and
//! single `.stv` files) load with the shipped std and runtime and type-check with no diagnostic
//! on both CPU architectures. They have run expectations only, so a type error would otherwise
//! surface only once a backend or interpreter runs them.

use std::path::{Path, PathBuf};

use stave_sema::{Config, Program};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Package dirs (with `stave.toml`) and single `.stv` files directly under `run/`, sorted.
fn programs() -> Vec<PathBuf> {
    let dir = repo().join("tests/conformance/run");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            !name.starts_with('.')
                && !name.starts_with('_')
                && (p.join("stave.toml").is_file() || p.extension().is_some_and(|x| x == "stv"))
        })
        .collect();
    v.sort();
    v
}

fn diagnostics(root: &Path, target: &str) -> Option<String> {
    let mut c = Config::new(root).with_shipped(repo().join("std"), repo().join("runtime"));
    c.target = stave_target::Triple::parse(target).ok();
    let p = Program::load(&c);
    let tc = stave_sema::check(&p);
    let d = tc.all_diagnostics(&p);
    if d.is_empty() {
        // the program's own fns were checked, none flagged
        let own: Vec<bool> = tc
            .bodies
            .iter()
            .filter(|b| {
                p.decl(b.decl)
                    .is_some_and(|x| p.module(x.module).is_some_and(|m| p.packages[m.package.idx()].is_root))
            })
            .map(|b| b.has_error)
            .collect();
        if own.is_empty() || own.iter().any(|&e| e) {
            return Some(format!("{} ({target}): bodies {own:?}", root.display()));
        }
        return None;
    }
    let text = stave_diag::emit_human(&d, &p.source_map).unwrap_or_else(|e| format!("{d:?} ({e:?})"));
    Some(format!("{} ({target}): {} diagnostic(s)\n{text}", root.display(), d.len()))
}

#[test]
fn run_programs_type_check_clean() {
    let progs = programs();
    for want in ["hello", "fib", "wordcount", "json_lite", "overflow_add.stv", "panic_stack.stv"] {
        assert!(
            progs.iter().any(|p| p.file_name().is_some_and(|n| n == want)),
            "missing tests/conformance/run/{want}"
        );
    }
    let mut failures = Vec::new();
    for p in &progs {
        for t in ["x86_64-linux", "aarch64-linux"] {
            failures.extend(diagnostics(p, t));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} checks failed:\n{}",
        failures.len(),
        progs.len() * 2,
        failures.join("\n")
    );
}
