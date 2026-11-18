//! `stave effects` (contracts/cli.md): the `tests/conformance/effects/report/expected.json`
//! golden (one fn per source of an effect), built against the pinned stand-in library of the
//! dump goldens so the golden does not move with the shipped std; the human form; deny-clause
//! errors after the report; usage errors.
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
    repo().join("tests/conformance/effects/report")
}

#[test]
fn json_golden() {
    let dir = case();
    let (c, out, err) = stave(&["effects", dir.to_str().unwrap(), "--json"], Some(&pinned()));
    assert_eq!(c, 0, "{err}");
    let golden = dir.join("expected.json");
    if std::env::var_os("STAVE_BLESS").is_some() {
        std::fs::write(&golden, &out).unwrap();
    }
    let want = std::fs::read_to_string(&golden).unwrap();
    assert_eq!(out, want, "--- got\n{out}--- want\n{want}");
    // Structure: one object per line, (file, line, col) order, every effect name known.
    let mut last: Option<(String, u64, u64)> = None;
    for l in out.lines() {
        let v: serde_json::Value = serde_json::from_str(l).unwrap();
        let key = (
            v["file"].as_str().unwrap().to_string(),
            v["line"].as_u64().unwrap(),
            v["col"].as_u64().unwrap(),
        );
        assert!(last.as_ref().is_none_or(|p| *p <= key), "not in (file, offset) order: {l}");
        last = Some(key);
        assert!(v["fn"].is_string(), "{l}");
        for e in v["effects"].as_array().unwrap() {
            assert!(stave_sema::effects::Effects::of_name(e.as_str().unwrap()).is_some(), "{l}");
        }
    }
}

#[test]
fn human_form_errors_and_usage() {
    let dir = case();
    let (c, out, err) = stave(&["effects", dir.to_str().unwrap()], Some(&pinned()));
    assert_eq!(c, 0, "{err}");
    assert!(
        out.lines().any(|l| l.starts_with("src/main.stv:") && l.ends_with(": main.add_wrap: none")),
        "{out}"
    );
    assert!(out.lines().any(|l| l.ends_with(": main.depth: trap recursive")), "{out}");
    assert!(out.lines().any(|l| l.ends_with(": main.main: trap recursive unsafe")), "{out}");
    // A deny clause violated: the report, then the diagnostics on stderr, exit 1.
    let bad = repo().join("tests/conformance/effects/trap_neg.stv");
    let (c, out, err) = stave(&["effects", bad.to_str().unwrap()], None);
    assert_eq!(c, 1, "{err}");
    assert!(out.lines().any(|l| l.ends_with(": main.h: trap")), "{out}");
    assert!(
        err.contains("E1401")
            && err.contains("`main.g` denies `trap`, but it reaches `main.h`, which may trap"),
        "{err}"
    );
    // Errors before the effect pass: nothing on stdout.
    let bad = repo().join("tests/conformance/effects/unknown_effect.stv");
    let (c, out, err) = stave(&["effects", bad.to_str().unwrap(), "--json"], None);
    assert_eq!((c, out.as_str()), (1, ""), "{err}");
    assert!(err.contains("\"code\":\"E1404\""), "{err}");
    // Usage errors.
    assert_eq!(stave(&["effects", "--release"], None).0, 2);
}
