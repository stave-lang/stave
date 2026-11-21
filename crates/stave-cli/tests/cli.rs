//! End-to-end exit codes of the `stave` binary (contracts/cli.md).

use std::path::{Path, PathBuf};
use std::process::Command;

fn stave(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_stave")).args(args).output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

#[test]
fn usage_errors_exit_2() {
    assert_eq!(stave(&[]).0, 2);
    assert_eq!(stave(&["build", "--bogus"]).0, 2);
    assert_eq!(stave(&["build", "--target", "x86_64-linux-v5"]).0, 2);
    assert_eq!(stave(&["build", "--features", "avx3"]).0, 2);
    assert_eq!(stave(&["build", "--alloc-check"]).0, 2);
}

#[test]
fn help_and_version_exit_0() {
    let (c, out, _) = stave(&["--help"]);
    assert_eq!(c, 0);
    assert!(out.contains("exit status: 0 ok, 1 diagnostics, 2 usage error, 3 internal compiler error"));
    let (c, out, _) = stave(&["--version"]);
    assert_eq!(c, 0);
    assert!(out.starts_with("stave "));
}

#[test]
fn build_without_package_is_usage_error() {
    // The test's working directory (the crate) has no stave.toml.
    let (c, _, err) = stave(&["build"]);
    assert_eq!(c, 2);
    assert!(err.contains("not a package (no stave.toml)"), "{err}");
}

#[test]
fn compiler_gpu_without_stage1_is_e1302_exit_1() {
    let (c, _, err) = stave(&["build", "--compiler", "gpu"]);
    assert_eq!(c, 1);
    assert!(err.starts_with("error[E1302]: `--compiler gpu`: no usable GPU (stage1 image missing)"), "{err}");
}

#[test]
fn daemon_status() {
    assert_eq!(stave(&["daemon", "status"]), (0, "daemon: not running\n".to_string(), String::new()));
}

#[cfg(unix)]
#[test]
fn non_utf8_argument_is_usage_error() {
    use std::os::unix::ffi::OsStrExt;
    let bad = std::ffi::OsStr::from_bytes(b"\xff");
    let o = Command::new(env!("CARGO_BIN_EXE_stave")).arg("build").arg(bad).output().unwrap();
    assert_eq!(o.status.code(), Some(2));
}

// ---------------------------------------------------------------------------------------------
// The pipeline through the CLI: front end, resolution, typeck

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn input(case: &str) -> String {
    repo().join("tests/conformance/dumps").join(case).join("input").display().to_string()
}

/// The pinned stand-in library of the dump goldens (`STAVE_STD_DIR`).
fn golden_lib() -> String {
    repo().join("tests/conformance/dumps/_lib").display().to_string()
}

fn stave_env(args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_stave"));
    c.args(args).env_remove("STAVE_THREADS").env_remove("STAVE_STD_DIR").env_remove("STAVE_SEMA_SHUFFLE");
    for (k, v) in env {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

const SUMMARY_AUTO: &str = "compiled on CPU (no usable GPU: stage1 image missing)\n";

#[test]
fn check_valid_package() {
    let (c, out, err) = stave_env(&["check", &input("valid_small")], &[]);
    assert_eq!(c, 0, "{err}");
    assert!(out.is_empty());
    assert!(!err.contains("stave: note:"), "check runs through borrowck: {err}");
    assert!(err.ends_with(SUMMARY_AUTO), "{err}");
    let (c, _, err) = stave_env(&["check", &input("valid_small"), "--compiler", "cpu"], &[]);
    assert_eq!(c, 0);
    assert!(err.ends_with("compiled on CPU (--compiler cpu)\n"), "{err}");
}

#[test]
fn build_links_an_executable() {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cli_build");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let out = tmp.join("valid_small").display().to_string();
    let (c, _, err) = stave_env(&["build", &input("valid_small"), "-o", &out], &[]);
    assert_eq!(c, 0, "{err}");
    // the build summary: main's stack bound, then the summary line
    let (stack, rest) = err.split_once('\n').unwrap_or_default();
    assert!(stack.starts_with("stack: main: ") && stack.ends_with(" bytes"), "{err}");
    assert_eq!(rest, SUMMARY_AUTO);
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        let o = Command::new(&out).output().unwrap();
        assert_eq!((o.status.code(), o.stdout.as_slice()), (Some(0), &b"s = 3\n"[..]));
    }
    // With diagnostics: exit 1, no executable.
    let bad = tmp.join("dropped").display().to_string();
    let (c, _, err) = stave_env(&["build", &input("dropped_item"), "-o", &bad], &[]);
    assert_eq!(c, 1, "{err}");
    assert!(err.starts_with("src/main.stv:8:9: error[E0004]: "), "{err}");
    assert!(!Path::new(&bad).exists());
}

#[test]
fn diagnostics_human_and_json() {
    let (c, _, err) = stave_env(&["check", &input("multi_file")], &[]);
    assert_eq!(c, 1);
    assert!(err.starts_with("src/util/math.stv:6:20: error[E0004]: "), "{err}");
    let (c, _, err) = stave_env(&["check", &input("multi_file"), "--diagnostics", "json", "--timings"], &[]);
    assert_eq!(c, 1);
    let lines: Vec<&str> = err.lines().collect();
    assert_eq!(lines.len(), 3, "{err}");
    let d: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!((d["kind"].as_str(), d["code"].as_str()), (Some("diagnostic"), Some("E0004")));
    assert_eq!(d["primary"]["file"], "src/util/math.stv");
    let t: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(t["kind"], "timings");
    assert_eq!(t["compiler"], "cpu (stage0)");
    let stages: Vec<&str> =
        t["stages"].as_array().unwrap().iter().map(|s| s["stage"].as_str().unwrap()).collect();
    assert_eq!(stages, ["read", "lex", "items", "parse", "intern", "resolve", "typeck", "borrowck"]);
    assert!(t["stages"][0]["items"].as_u64().unwrap() > 18, "read counts lines: 8 + 3 + 7 + std");
    // With the stand-in library: exactly the package's 18 lines plus its 23 + 3 + 3.
    let (_, _, err) = stave_env(
        &["check", &input("multi_file"), "--diagnostics", "json", "--timings"],
        &[("STAVE_STD_DIR", &golden_lib())],
    );
    let t: serde_json::Value = serde_json::from_str(err.lines().nth(1).unwrap()).unwrap();
    assert_eq!(t["stages"][0]["items"], 18 + STD_LINES, "{err}");
    assert_eq!(t["stages"][0]["items"], t["stages"][1]["items"], "read and parse count the same lines");
    assert_eq!(format!("{}\n", lines[2]), SUMMARY_AUTO);
}

#[test]
fn timings_human() {
    let (c, _, err) = stave_env(&["check", &input("valid_small"), "--timings"], &[]);
    assert_eq!(c, 0);
    assert!(err.contains("compiler: cpu (stage0)\n  read "), "{err}");
    for s in ["parse", "intern", "resolve", "typeck", "total"] {
        assert!(err.contains(&format!("\n  {s} ")), "{s}: {err}");
    }
}

#[test]
fn check_reports_resolution_and_type_errors() {
    let f = repo().join("tests/conformance/resolve/unresolved.stv");
    let (c, _, err) = stave_env(&["check", &f.display().to_string(), "--diagnostics", "json"], &[]);
    assert_eq!(c, 1, "{err}");
    let codes: Vec<String> = err
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["kind"] == "diagnostic")
        .map(|v| v["code"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(!codes.is_empty() && codes.iter().all(|c| c == "E0101"), "{err}");
    let f = repo().join("tests/conformance/types/mismatch.stv");
    let (c, _, err) = stave_env(&["check", &f.display().to_string()], &[]);
    assert_eq!(c, 1, "{err}");
    assert!(err.contains(": error[E0201]: "), "{err}");
}

#[test]
fn single_file_argument() {
    let f = repo().join("tests/conformance/syntax/recover_two_items.stv");
    let (c, _, err) = stave_env(&["check", &f.display().to_string()], &[]);
    assert_eq!(c, 1, "{err}");
    assert!(err.starts_with("recover_two_items.stv:"), "{err}");
}

/// Lines of `tests/conformance/dumps/_lib` (std/core.stv, std/io.stv, std/string.stv, runtime/empty.stv).
const STD_LINES: u64 = 40;

const GOLDEN_DUMPS: &str = "--dump=tokens,items,ast,strings,names,types,diags,stats";
const GOLDEN_FILES: [&str; 7] =
    ["tokens.txt", "items.txt", "ast.txt", "strings.txt", "names.txt", "types.txt", "diags.txt"];

#[test]
fn dumps_match_goldens_under_stave_threads_and_shuffle() {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cli_dumps");
    let lib = golden_lib();
    for case in ["valid_small", "dropped_item", "unbalanced", "multi_file"] {
        for (threads, shuffle) in [("1", ""), ("2", ""), ("12", ""), ("12", "7"), ("3", "3735928559")] {
            let dir = tmp.join(format!("{case}_{threads}_{shuffle}"));
            let _ = std::fs::remove_dir_all(&dir);
            let d = dir.display().to_string();
            let (c, _, err) = stave_env(
                &["check", &input(case), GOLDEN_DUMPS, "--dump-dir", &d],
                &[("STAVE_THREADS", threads), ("STAVE_STD_DIR", &lib), ("STAVE_SEMA_SHUFFLE", shuffle)],
            );
            assert!(c == 0 || c == 1, "{case}: {err}");
            for f in GOLDEN_FILES {
                let got = std::fs::read(dir.join(f)).unwrap();
                let want = std::fs::read(repo().join("tests/conformance/dumps").join(case).join(f)).unwrap();
                assert!(
                    got == want,
                    "{case}/{f} with STAVE_THREADS={threads} STAVE_SEMA_SHUFFLE={shuffle} differs from the golden"
                );
                assert!(!String::from_utf8_lossy(&got).contains("compiled on"), "no summary line in dumps");
            }
            assert!(dir.join("stats.txt").is_file());
        }
    }
}

#[test]
fn bad_stave_threads_and_dump_stage_are_usage_errors() {
    let (c, _, err) = stave_env(&["check", &input("valid_small")], &[("STAVE_THREADS", "0")]);
    assert_eq!(c, 2);
    assert!(err.starts_with("stave: STAVE_THREADS must be a thread count"), "{err}");
    let (c, _, err) = stave_env(&["check", &input("valid_small"), "--dump=lex", "--dump-dir", "/tmp/x"], &[]);
    assert_eq!(c, 2);
    assert!(err.contains("unknown --dump stage"), "{err}");
    // A bad STAVE_STD_DIR is a usage error, never a silent fallback.
    let (c, _, err) = stave_env(&["check", &input("valid_small")], &[("STAVE_STD_DIR", "/nonexistent")]);
    assert_eq!(c, 2);
    assert!(err.starts_with("stave: STAVE_STD_DIR=/nonexistent: not a Stave library directory"), "{err}");
}

#[test]
fn test_command_runs_the_syntax_suite() {
    let suite = repo().join("tests/conformance/syntax").display().to_string();
    let (c, out, err) = stave_env(&["test", &suite], &[("STAVE_THREADS", "4")]);
    // Every test passes (the empty `lex_empty.stv` runs as a PARSE-ONLY positive).
    let failed: Vec<&str> = out.lines().filter_map(|l| l.strip_prefix("FAILED    ")).collect();
    assert!(failed.is_empty(), "{out}{err}");
    assert_eq!(c, 0, "{out}{err}");
    let last = out.lines().last().unwrap_or("");
    assert!(last.starts_with("conformance: "), "{out}");
    assert!(last.contains(" 0 failed, 0 malformed, 0 not run, "), "{last}");
    assert!(out.contains("ok        parse_fn.stv"), "{out}");
    assert!(out.contains("ok        lex_empty.stv"), "{out}");
    assert!(err.ends_with(SUMMARY_AUTO), "{err}");
    // --filter
    let (c, out, _) = stave_env(&["test", &suite, "--filter", "refine_"], &[]);
    assert_eq!(c, 0);
    assert!(out.lines().all(|l| l.contains("refine_") || l.starts_with("conformance: ")), "{out}");
}
