//! `stave unsafe-report` (contracts/cli.md): the
//! `tests/conformance/borrow/unsafe_report/expected.json` golden (LANG.md's `read_raw` plus one
//! site of each kind), built against the pinned stand-in library of the dump goldens so the golden
//! does not move with the shipped std; the human form; errors; the shipped std's own sites.
//!
//! `STAVE_BLESS=1` rewrites expected.json from the current output (review the diff!).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn stave(args: &[&str], lib: Option<&Path>) -> (i32, String, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_stave"));
    c.args(args).env_remove("STAVE_THREADS").env_remove("STAVE_STD_DIR").env_remove("STAVE_SEMA_SHUFFLE");
    if let Some(l) = lib {
        c.env("STAVE_STD_DIR", l);
    }
    let o = c.output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

fn pinned() -> PathBuf {
    repo().join("tests/conformance/dumps/_lib")
}

fn case() -> PathBuf {
    repo().join("tests/conformance/borrow/unsafe_report")
}

#[test]
fn json_golden() {
    let dir = case();
    let (c, out, err) = stave(&["unsafe-report", dir.to_str().unwrap(), "--json"], Some(&pinned()));
    assert_eq!(c, 0, "{err}");
    let golden = dir.join("expected.json");
    if std::env::var_os("STAVE_BLESS").is_some() {
        std::fs::write(&golden, &out).unwrap();
    }
    let want = std::fs::read_to_string(&golden).unwrap();
    assert_eq!(out, want, "--- got\n{out}--- want\n{want}");
    // Structure: one object per line, canonical (file, line, col) order, `read_raw` present.
    let mut last: Option<(String, u64, u64)> = None;
    let mut kinds = std::collections::BTreeSet::new();
    for l in out.lines() {
        let v: serde_json::Value = serde_json::from_str(l).unwrap();
        let key = (
            v["file"].as_str().unwrap().to_string(),
            v["line"].as_u64().unwrap(),
            v["col"].as_u64().unwrap(),
        );
        assert!(last.as_ref().is_none_or(|p| *p <= key), "not in (file, offset) order: {l}");
        last = Some(key);
        assert!(v["obligation"].is_string() && v["fn"].is_string(), "{l}");
        kinds.insert(v["kind"].as_str().unwrap().to_string());
    }
    let kinds: Vec<&str> = kinds.iter().map(String::as_str).collect();
    assert_eq!(kinds, ["block", "expr", "extern-call", "fn", "impl", "trait"]);
    assert!(
        out.contains(
            "\"fn\":\"main.read_raw\",\"kind\":\"expr\",\"obligation\":\"p is valid for reads: caller passes &x as *const u8\""
        ),
        "{out}"
    );
}

#[test]
fn human_form_and_errors() {
    let dir = case();
    let (c, out, err) = stave(&["unsafe-report", dir.to_str().unwrap()], Some(&pinned()));
    assert_eq!(c, 0, "{err}");
    assert!(
        out.lines().any(|l| l.starts_with("src/main.stv:")
            && l.contains(
                ": expr in main.read_raw: \"p is valid for reads: caller passes &x as *const u8\""
            )),
        "{out}"
    );
    // A program with errors: diagnostics on stderr, nothing on stdout, exit 1.
    let bad = repo().join("tests/conformance/borrow/unsafe_e0803.stv");
    let (c, out, err) = stave(&["unsafe-report", bad.to_str().unwrap(), "--json"], Some(&pinned()));
    assert_eq!((c, out.as_str()), (1, ""), "{err}");
    assert!(err.contains("\"code\":\"E0803\""), "{err}");
    // Usage errors.
    assert_eq!(stave(&["unsafe-report", "--release"], None).0, 2);
}

/// With the shipped library, the program's sites include std's and runtime's (every package is
/// inventoried), and the report is stable across runs.
#[test]
fn shipped_std_sites() {
    let dir = case();
    let (c, a, err) = stave(&["unsafe-report", dir.to_str().unwrap(), "--json"], None);
    assert_eq!(c, 0, "{err}");
    let (_, b, _) = stave(&["unsafe-report", dir.to_str().unwrap(), "--json"], None);
    assert_eq!(a, b);
    assert!(a.lines().count() > 7, "std/runtime sites expected:\n{a}");
    assert!(a.contains("\"fn\":\"main.read_raw\""));
}
