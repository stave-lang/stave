//! `stave dup` (contracts/cli.md) on the planted package `tests/dup/report`, built against the
//! pinned stand-in library of the dump goldens so the goldens do not move with the shipped std:
//! the report golden (`expected.txt`), the decisions of the canonical form at the source level,
//! `--json`, the baseline golden (`baseline.txt`), and the gate (contracts/cli.md
//! `dup-gate`): it passes on its baseline, fails on a planted new duplicate naming the fn to call,
//! and passes when a class goes away.
//!
//! `STAVE_BLESS=1` rewrites expected.txt and baseline.txt from the current output (review the
//! diff!).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn case() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dup/report")
}

fn stave(args: &[&str], threads: Option<&str>) -> (i32, String, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_stave"));
    c.args(args)
        .env_remove("STAVE_SEMA_SHUFFLE")
        .env("STAVE_STD_DIR", repo().join("tests/conformance/dumps/_lib"));
    match threads {
        Some(t) => c.env("STAVE_THREADS", t),
        None => c.env_remove("STAVE_THREADS"),
    };
    let o = c.output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

fn golden(name: &str, got: &str) {
    let path = case().join("..").join(name);
    if std::env::var_os("STAVE_BLESS").is_some() {
        std::fs::write(&path, got).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap();
    assert_eq!(got, want, "--- got\n{got}--- want\n{want}");
}

/// A copy of the planted package in a fresh directory.
fn copy(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::copy(case().join("stave.toml"), dir.join("stave.toml")).unwrap();
    for f in ["main.stv", "util.stv", "geo.stv"] {
        std::fs::copy(case().join("src").join(f), dir.join("src").join(f)).unwrap();
    }
    dir
}

#[test]
fn report_golden_and_thread_count() {
    let dir = case();
    let (c, out, err) = stave(&["dup", dir.to_str().unwrap()], None);
    assert_eq!(c, 0, "{err}");
    golden("expected.txt", &out);
    for t in ["1", "7"] {
        let (c, again, err) = stave(&["dup", dir.to_str().unwrap()], Some(t));
        assert_eq!(c, 0, "{err}");
        assert_eq!(again, out, "STAVE_THREADS={t}");
    }
}

/// The source fn paths of every class of the JSON report, and of every const group.
fn classes(args: &[&str]) -> Vec<(String, Vec<String>)> {
    let dir = case();
    let mut all = vec!["dup", dir.to_str().unwrap(), "--json"];
    all.extend_from_slice(args);
    let (c, out, err) = stave(&all, None);
    assert_eq!(c, 0, "{err}");
    out.lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            let key = if v["kind"] == "class" { "fn" } else { "const" };
            let kind = format!("{}{}", v["kind"].as_str().unwrap(), v["form"].as_str().unwrap_or(""));
            let paths = v["members"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m[key].as_str().unwrap().to_string())
                .collect();
            (kind, paths)
        })
        .collect()
}

fn together(cs: &[(String, Vec<String>)], kind: &str, a: &str, b: &str) -> bool {
    cs.iter().any(|(k, ps)| k == kind && ps.iter().any(|p| p == a) && ps.iter().any(|p| p == b))
}

fn listed(cs: &[(String, Vec<String>)], a: &str) -> bool {
    cs.iter().any(|(_, ps)| ps.iter().any(|p| p == a))
}

#[test]
fn what_is_the_same_code() {
    let cs = classes(&["--min-size", "1"]);
    // Other names of the fn and its locals: the same code.
    assert!(together(&cs, "classexact", "main.checksum", "util.sum_mix"));
    // Another literal, another callee, independent statements in another order: not.
    assert!(!listed(&cs, "util.sum_mix7"));
    assert!(!listed(&cs, "util.scaled_a") && !listed(&cs, "util.scaled_b"));
    assert!(!listed(&cs, "util.order_a") && !listed(&cs, "util.order_b"));
    // Callees with the same code; equal closures; self calls.
    assert!(together(&cs, "classexact", "util.twice_a", "util.twice_b"));
    assert!(together(&cs, "classexact", "main.apply_a", "main.apply_b") && !listed(&cs, "main.apply_c"));
    assert!(together(&cs, "classexact", "main.fact_a", "main.fact_b"));
    // Two generic fns, equal at the args both are built with.
    assert!(together(&cs, "classexact", "main.last_or", "util.first_or"));
    // Equal at `u32`, different at `[u8; 3]`: never a class.
    assert!(!listed(&cs, "util.size_plus") && !listed(&cs, "util.align_plus"));
    // Methods of two types, and the trait impls that must exist per type: up to `Self`.
    assert!(together(&cs, "classself", "geo.Meters.grow", "geo.Feet.grow"));
    assert!(together(&cs, "classself", "geo.Meters.score", "geo.Feet.score"));
    assert!(together(&cs, "classself", "geo.Meters.get", "geo.Feet.get"));
    // `test fn`s are not compared.
    assert!(!listed(&cs, "util.t_a"));
    // The constants.
    assert!(together(&cs, "const", "util.NONE", "geo.NONE"));
    assert!(together(&cs, "alias", "geo.NONE", "main.NONE"));
    assert!(together(&cs, "const", "util.LIMIT", "main.LIMIT"));
    // The default size threshold leaves the getter out.
    let cs = classes(&[]);
    assert!(!listed(&cs, "geo.Meters.get") && listed(&cs, "geo.Meters.grow"));
}

#[test]
fn baseline_golden_and_gate() {
    let dir = case();
    let base = dir.join("../baseline.txt");
    let written = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dup-baseline.txt");
    let (c, out, err) =
        stave(&["dup", dir.to_str().unwrap(), "--write-baseline", written.to_str().unwrap()], None);
    assert_eq!((c, out.as_str()), (0, ""), "{err}");
    golden("baseline.txt", &std::fs::read_to_string(&written).unwrap());
    // The tree its baseline was written from passes.
    let (c, out, err) = stave(&["dup", dir.to_str().unwrap(), "--baseline", base.to_str().unwrap()], None);
    assert_eq!((c, out.as_str(), err.as_str()), (0, "", ""));

    // A planted new duplicate fails, naming the existing fn to call.
    let planted = copy("dup-planted");
    std::fs::write(
        planted.join("src/extra.stv"),
        "pub fn fold31(ws: &[u32]) -> u32 {\n    let mut r: u32 = 0;\n    for j in 0..ws.len() {\n        r = r *% 31u32 +% ws[j];\n    }\n    return r;\n}\n",
    )
    .unwrap();
    let (c, out, err) =
        stave(&["dup", planted.to_str().unwrap(), "--baseline", base.to_str().unwrap()], None);
    assert_eq!(c, 1, "{out}{err}");
    let want = "src/extra.stv:1: new duplicate (exact, size 13): `extra.fold31` is the same code as \
                `util.sum_mix` (src/util.stv:7): call it instead\ndup: 1 new duplicates not in the baseline\n";
    assert_eq!(out, want);

    // A class that goes away passes (the ratchet), with a note.
    let fewer = copy("dup-fewer");
    let main = std::fs::read_to_string(fewer.join("src/main.stv")).unwrap();
    let main = main.replace("h = h *% 31u32 +% xs[i];", "h = h *% 37u32 +% xs[i];");
    std::fs::write(fewer.join("src/main.stv"), main).unwrap();
    let (c, out, err) = stave(&["dup", fewer.to_str().unwrap(), "--baseline", base.to_str().unwrap()], None);
    assert_eq!((c, out.as_str()), (0, ""), "{err}");
    assert!(err.contains("1 baseline classes no longer occur"), "{err}");
}

#[test]
fn usage_errors() {
    let dir = case();
    let missing = dir.join("no-such-baseline.txt");
    let (c, _, err) = stave(&["dup", dir.to_str().unwrap(), "--baseline", missing.to_str().unwrap()], None);
    assert_eq!(c, 2, "{err}");
    let bad = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dup-bad-baseline.txt");
    std::fs::write(&bad, "exact only.one\n").unwrap();
    let (c, _, err) = stave(&["dup", dir.to_str().unwrap(), "--baseline", bad.to_str().unwrap()], None);
    assert_eq!(c, 2);
    assert!(err.contains("at least two fn paths"), "{err}");
    let (c, _, _) = stave(&["dup", "--json", "--write-baseline", "x"], None);
    assert_eq!(c, 2);
}
