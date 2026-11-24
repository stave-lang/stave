//! `--compiler auto|cpu|gpu` end to end (contracts/cli.md probe table): the summary
//! line names the compiler used, `--compiler gpu` without a stage1 image is E1302 (no location,
//! with a fix), a target stage1 does not support is E1303 first (probe step 1, even with a GPU),
//! `--timings` names `compiler: cpu (stage0)` (JSON: the `compiler` field), and the summary line
//! is never part of a dump file.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn hello() -> String {
    repo().join("tests/conformance/run/hello").display().to_string()
}

fn stave(args: &[&str]) -> (i32, String, String) {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("compiler_flag");
    std::fs::create_dir_all(&tmp).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_stave"))
        .args(args)
        .current_dir(&tmp)
        .env_remove("STAVE_THREADS")
        .env_remove("STAVE_STD_DIR")
        .env_remove("STAVE_SEMA_SHUFFLE")
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

#[test]
fn cpu_and_auto_summary_lines() {
    for cmd in ["check", "build"] {
        let (c, _, err) = stave(&[cmd, &hello(), "--compiler", "cpu"]);
        assert_eq!(c, 0, "{err}");
        assert!(err.ends_with("compiled on CPU (--compiler cpu)\n"), "{err}");
        let (c, _, err) = stave(&[cmd, &hello()]);
        assert_eq!(c, 0, "{err}");
        assert!(err.ends_with("compiled on CPU (no usable GPU: stage1 image missing)\n"), "{err}");
    }
    let (_, _, err) = stave(&["build", &hello(), "--target", "aarch64-linux"]);
    assert!(err.ends_with("compiled on CPU (target aarch64-linux not supported by stage1)\n"), "{err}");
}

#[test]
fn json_diagnostics_leave_no_summary_line() {
    for cmd in ["check", "build"] {
        let (c, _, err) = stave(&[cmd, &hello(), "--diagnostics", "json"]);
        assert_eq!(c, 0, "{err}");
        assert!(err.is_empty(), "{err}");
    }
}

#[test]
fn gpu_without_stage1_is_e1302() {
    for fmt in ["human", "json"] {
        let (c, out, err) = stave(&["build", &hello(), "--compiler", "gpu", "--diagnostics", fmt]);
        assert_eq!(c, 1, "{err}");
        assert!(out.is_empty());
        let first = err.lines().next().unwrap_or("");
        if fmt == "human" {
            assert!(first.starts_with("error[E1302]: `--compiler gpu`: no usable GPU ("), "{err}");
            assert!(err.lines().nth(1).is_some_and(|l| l.starts_with("  fix: ")), "{err}");
        } else {
            let d: serde_json::Value = serde_json::from_str(first).unwrap();
            assert_eq!(d["code"], "E1302");
            assert!(d.get("primary").is_none(), "no location: {d}");
            assert!(d["fix"]["description"].as_str().is_some_and(|s| !s.is_empty()), "{d}");
            assert_eq!(d["related"], serde_json::json!([]), "{d}");
        }
        assert!(!err.contains("compiled on"), "nothing was compiled: {err}");
    }
}

#[test]
fn gpu_with_aarch64_is_e1303_only() {
    let (c, _, err) = stave(&[
        "build",
        &hello(),
        "--compiler",
        "gpu",
        "--target",
        "aarch64-linux",
        "--diagnostics",
        "json",
    ]);
    assert_eq!(c, 1, "{err}");
    let codes: Vec<String> = err
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|v| v["code"].as_str().unwrap_or("").to_string())
        .collect();
    assert_eq!(codes, ["E1303"], "{err}");
}

#[test]
fn timings_name_the_compiler() {
    let (c, _, err) = stave(&["build", &hello(), "--timings"]);
    assert_eq!(c, 0, "{err}");
    assert!(err.contains("compiler: cpu (stage0)\n"), "{err}");
    for s in [
        "read", "lex", "items", "parse", "intern", "resolve", "typeck", "borrowck", "lower", "codegen",
        "link", "total",
    ] {
        assert!(err.contains(&format!("\n  {s} ")), "{s}: {err}");
    }
    let (c, _, err) = stave(&["build", &hello(), "--timings", "--diagnostics", "json"]);
    assert_eq!(c, 0, "{err}");
    let t: serde_json::Value = serde_json::from_str(err.lines().next().unwrap()).unwrap();
    assert_eq!((t["kind"].as_str(), t["compiler"].as_str()), (Some("timings"), Some("cpu (stage0)")));
    let stages: Vec<&str> =
        t["stages"].as_array().unwrap().iter().map(|s| s["stage"].as_str().unwrap()).collect();
    assert_eq!(
        stages,
        [
            "read", "lex", "items", "parse", "intern", "resolve", "typeck", "borrowck", "lower", "codegen",
            "link"
        ]
    );
}

#[test]
fn summary_line_is_never_in_a_dump() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("compiler_flag_dumps");
    let _ = std::fs::remove_dir_all(&dir);
    let d = dir.display().to_string();
    let (c, _, err) = stave(&[
        "build",
        &hello(),
        "--dump=tokens,items,ast,strings,names,sigs,kinds,types,ir,diags,stats",
        "--dump-dir",
        &d,
        "--timings",
    ]);
    assert_eq!(c, 0, "{err}");
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(!text.contains("compiled on") && !text.contains("compiler: cpu"), "{}", p.display());
        n += 1;
    }
    assert_eq!(n, 11);
}
