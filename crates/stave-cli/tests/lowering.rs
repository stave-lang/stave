//! `stave lowering`: the table of each CPU tier against its golden (`tests/lowering/<target>.txt`),
//! target aliases, `--op`, and errors.
//!
//! `STAVE_BLESS=1` rewrites the goldens from the current output (review the diff!).

use std::path::Path;
use std::process::Command;

fn stave(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_stave")).args(args).output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

fn table(target: &str) -> String {
    let (c, out, err) = stave(&["lowering", "--target", target]);
    assert_eq!(c, 0, "{target}: {err}");
    out
}

/// The table without its header lines.
fn body(text: &str) -> Vec<&str> {
    text.lines().filter(|l| !l.starts_with('#')).collect()
}

#[test]
fn tables_match_goldens() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/lowering");
    let bless = std::env::var_os("STAVE_BLESS").is_some();
    for target in
        ["x86_64-linux-v2", "x86_64-linux-v3", "x86_64-linux-v4", "aarch64-linux", "aarch64-linux+sve"]
    {
        let out = table(target);
        let golden = dir.join(format!("{target}.txt"));
        if bless {
            std::fs::write(&golden, &out).unwrap();
        }
        let want = std::fs::read_to_string(&golden).unwrap_or_else(|e| panic!("{}: {e}", golden.display()));
        assert_eq!(out, want, "{target} differs from {}", golden.display());
    }
}

#[test]
fn aliases_and_sve2_share_a_tier() {
    assert_eq!(table("x86_64-linux"), table("x86_64-linux-v2"));
    let sve = table("aarch64-linux+sve");
    let sve2 = table("aarch64-linux+sve2");
    assert!(sve2.starts_with("# stave lowering --target aarch64-linux+sve2: tier sve\n"), "{sve2}");
    assert_eq!(body(&sve), body(&sve2));
}

#[test]
fn op_filter() {
    let (c, out, err) = stave(&["lowering", "--target", "aarch64-linux", "--op", "gather"]);
    assert_eq!(c, 0, "{err}");
    let lines = body(&out);
    assert!(!lines.is_empty() && lines.iter().all(|l| l.starts_with("vec.gather ")), "{out}");
    assert_eq!(stave(&["lowering", "--target", "aarch64-linux", "--op", "vec.gather"]).1, out);
    let all = table("aarch64-linux");
    assert!(lines.iter().all(|l| all.lines().any(|a| a == *l)));
    let (_, v4, _) = stave(&["lowering", "--target", "x86_64-linux-v4", "--op", "compress"]);
    assert!(body(&v4).iter().any(|l| l.contains("| +avx512vbmi2: N 1 vpcompress* {z}")), "{v4}");
}

#[test]
fn errors() {
    let (c, _, err) = stave(&["lowering", "--target", "x86_64-linux-v3", "--op", "nosuch"]);
    assert_eq!(c, 2);
    assert!(err.contains("no lowering entry for op `nosuch` on x86_64-linux-v3"), "{err}");
    let (c, _, err) = stave(&["lowering", "--target", "gpu-sm86"]);
    assert_eq!(c, 2);
    assert!(err.starts_with("stave: `lowering --target gpu-sm86` is not implemented yet ("), "{err}");
    assert_eq!(stave(&["lowering"]).0, 2);
}
