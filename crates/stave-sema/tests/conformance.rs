//! Runs name resolution and type checking on `tests/conformance/{resolve,syntax,types}/` and
//! checks the `//~ ERROR` (`#~ ERROR` in `stave.toml`) annotations
//! (contracts/conformance-annotations.md): the multiset of (code, file, line) of the diagnostics
//! must equal the annotated one. Other directives (`CHECK-ONLY`, ...) are ignored here. A minimal
//! scanner is used instead of stave-driver's (no dependency on the driver crate). The resolve and
//! syntax suites use the test stand-in std (`tests/fixtures/std`); the types suite uses the
//! shipped `std/` and `runtime/`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_sema::{Config, Program};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_std() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/std")
}

type Key = (String, Option<(String, u32)>);

/// Expected (code, Some((file, line))) / (code, None) for NOLOC.
fn scan(text: &str, file: &str, marker: &str, out: &mut BTreeMap<Key, i32>) {
    for (ln, line) in text.lines().enumerate() {
        let Some(p) = line.find(marker) else { continue };
        let mut rest = &line[p + marker.len()..];
        let mut up = 0u32;
        while let Some(r) = rest.strip_prefix('^') {
            up += 1;
            rest = r;
        }
        let words: Vec<&str> = rest.split_whitespace().collect();
        let (noloc, words) = match words.first() {
            Some(&"NOLOC") => (true, &words[1..]),
            _ => (false, &words[..]),
        };
        if let [kind, code] = words
            && (*kind == "ERROR" || *kind == "WARNING")
        {
            let at = if noloc { None } else { Some((file.to_string(), (ln as u32 + 1).saturating_sub(up))) };
            *out.entry((code.to_string(), at)).or_default() += 1;
        }
    }
}

fn files_of(dir: &Path, base: &Path, out: &mut Vec<(String, PathBuf)>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            files_of(&p, base, out);
        } else {
            let rel = p.strip_prefix(base).unwrap().to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
}

fn expected(test: &Path) -> BTreeMap<Key, i32> {
    let mut out = BTreeMap::new();
    if test.is_dir() {
        let mut fs = Vec::new();
        files_of(test, test, &mut fs);
        for (rel, p) in fs {
            let text = String::from_utf8_lossy(&std::fs::read(&p).unwrap()).into_owned();
            if rel.ends_with(".stv") {
                scan(&text, &rel, "//~", &mut out);
            } else if rel.ends_with("stave.toml") {
                scan(&text, &rel, "#~", &mut out);
            }
        }
    } else {
        let text = String::from_utf8_lossy(&std::fs::read(test).unwrap()).into_owned();
        let name = test.file_name().unwrap().to_string_lossy().into_owned();
        scan(&text, &name, "//~", &mut out);
    }
    out
}

fn observed(p: &Program, diags: &[stave_diag::Diagnostic]) -> BTreeMap<Key, i32> {
    let mut out = BTreeMap::new();
    for d in diags {
        let at = d.primary().map(|s| {
            let loc = p.source_map.locate(s).unwrap();
            (loc.path.to_string(), loc.line)
        });
        *out.entry((d.code().to_string(), at)).or_default() += 1;
    }
    out
}

pub fn check(test: &Path) -> Result<(), String> {
    check_with(test, false)
}

/// Checks one test; `shipped`: the real std and runtime instead of the fixture std.
pub fn check_with(test: &Path, shipped: bool) -> Result<(), String> {
    let cfg = if shipped {
        Config::new(test).with_shipped(repo().join("std"), repo().join("runtime"))
    } else {
        Config::new(test).with_std(fixture_std())
    };
    let prog = Program::load(&cfg);
    let tc = stave_sema::check(&prog);
    let diags = tc.all_diagnostics(&prog);
    let exp = expected(test);
    let obs = observed(&prog, &diags);
    if exp == obs {
        return Ok(());
    }
    let mut msg = format!("{}:\n", test.display());
    for (k, n) in &exp {
        if obs.get(k) != Some(n) {
            msg += &format!("  expected {n}x {k:?}, got {}\n", obs.get(k).copied().unwrap_or(0));
        }
    }
    for (k, n) in &obs {
        if !exp.contains_key(k) {
            msg += &format!("  unexpected {n}x {k:?}\n");
        }
    }
    for d in &diags {
        let loc = d.primary().and_then(|s| prog.source_map.locate(s).ok());
        msg += &format!(
            "    {} {:?}: {}\n",
            d.code(),
            loc.map(|l| (l.path.to_string(), l.line, l.col)),
            d.message()
        );
    }
    Err(msg)
}

fn tests_in(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            !name.starts_with('.') && !name.starts_with('_')
        })
        .filter(|p| p.extension().is_some_and(|e| e == "stv") || p.join("stave.toml").is_file())
        .collect();
    v.sort();
    v
}

#[test]
fn resolve_suite() {
    let tests = tests_in(&repo().join("tests/conformance/resolve"));
    assert!(tests.len() >= 10, "suite has {} tests", tests.len());
    let failures: Vec<String> = tests.iter().filter_map(|t| check(t).err()).collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}

#[test]
fn selftest_manifest_fixture() {
    // The runner's own fixture: unknown key + non-path dependency, both E1002.
    check(&repo().join("tests/conformance/_selftest/pkg_manifest")).unwrap();
}

/// The syntax suite's semantically checked tests (not `PARSE-ONLY`) must resolve with exactly
/// their annotated (syntax) diagnostics: no resolution cascade from dropped items.
#[test]
fn syntax_suite_resolves() {
    let mut failures = Vec::new();
    for t in tests_in(&repo().join("tests/conformance/syntax")) {
        let parse_only = |p: &Path| std::fs::read_to_string(p).is_ok_and(|s| s.contains("//~ PARSE-ONLY"));
        if (t.is_file() && parse_only(&t)) || (t.is_dir() && parse_only(&t.join("src/main.stv"))) {
            continue;
        }
        if let Err(e) = check(&t) {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn types_suite() {
    let tests = tests_in(&repo().join("tests/conformance/types"));
    assert!(tests.len() >= 6, "suite has {} tests", tests.len());
    let failures: Vec<String> = tests.iter().filter_map(|t| check_with(t, true).err()).collect();
    assert!(failures.is_empty(), "{} failing:\n{}", failures.len(), failures.join("\n"));
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}
