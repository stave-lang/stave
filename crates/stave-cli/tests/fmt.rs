//! `stave fmt` (contracts/cli.md): files are rewritten in the canonical layout, a file with a
//! syntax error is left as it is and reported, and `--check` only lists.

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

const MESSY: &str = "// a comment\nfn f(x:i32)->i32{let a=x+1;return a;} // done\n";
const MESSY_FORMATTED: &str =
    "// a comment\nfn f(x: i32) -> i32 {\n    let a = x + 1;\n    return a;\n} // done\n";
const CLEAN: &str = "fn g() -> i32 {\n    return 1;\n}\n";
const BROKEN: &str = "fn h( {\n   let x = 1;\n";
const BROKEN_ITEM: &str = "fn ok() {}\nfn h() { let   = 1; }\n";

/// A fresh directory with one file of each kind.
fn tree(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::create_dir_all(dir.join(".hidden")).unwrap();
    std::fs::write(dir.join("messy.stv"), MESSY).unwrap();
    std::fs::write(dir.join("clean.stv"), CLEAN).unwrap();
    std::fs::write(dir.join("sub/broken.stv"), BROKEN).unwrap();
    std::fs::write(dir.join("sub/broken_item.stv"), BROKEN_ITEM).unwrap();
    std::fs::write(dir.join("sub/deep.stv"), MESSY).unwrap();
    std::fs::write(dir.join("notes.txt"), MESSY).unwrap();
    std::fs::write(dir.join(".hidden/skipped.stv"), MESSY).unwrap();
    dir
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn check_lists_and_writes_nothing() {
    let dir = tree("fmt_check");
    let d = dir.to_str().unwrap();
    let (c, out, err) = stave(&["fmt", "--check", d]);
    assert_eq!(c, 1, "{err}");
    assert_eq!(out, format!("{d}/messy.stv\n{d}/sub/deep.stv\n"));
    assert!(err.contains("error[E0016]") && err.contains("sub/broken.stv"), "{err}");
    assert!(err.contains("error[E0004]") && err.contains("sub/broken_item.stv"), "{err}");
    assert_eq!(read(&dir.join("messy.stv")), MESSY);
    assert_eq!(read(&dir.join("sub/deep.stv")), MESSY);
    // only files that would change: exit 1, nothing on stderr
    let (c, out, err) = stave(&["fmt", "--check", &format!("{d}/messy.stv"), &format!("{d}/clean.stv")]);
    assert_eq!((c, out, err), (1, format!("{d}/messy.stv\n"), String::new()));
    let (c, out, err) = stave(&["fmt", "--check", &format!("{d}/clean.stv")]);
    assert_eq!((c, out, err), (0, String::new(), String::new()));
}

#[test]
fn rewrites_files_and_leaves_syntax_errors_alone() {
    let dir = tree("fmt_write");
    let d = dir.to_str().unwrap();
    let (c, out, err) = stave(&["fmt", d]);
    assert_eq!(c, 1, "{err}");
    assert!(out.is_empty(), "{out}");
    assert!(err.contains("error[E0016]") && err.contains("error[E0004]"), "{err}");
    assert_eq!(read(&dir.join("messy.stv")), MESSY_FORMATTED);
    assert_eq!(read(&dir.join("sub/deep.stv")), MESSY_FORMATTED);
    assert_eq!(read(&dir.join("clean.stv")), CLEAN);
    assert_eq!(read(&dir.join("sub/broken.stv")), BROKEN);
    assert_eq!(read(&dir.join("sub/broken_item.stv")), BROKEN_ITEM);
    // not `.stv`, or in a hidden directory: not visited
    assert_eq!(read(&dir.join("notes.txt")), MESSY);
    assert_eq!(read(&dir.join(".hidden/skipped.stv")), MESSY);
    // a file named on the command line is formatted whatever its name; a second run changes nothing
    let (c, out, err) = stave(&["fmt", &format!("{d}/notes.txt"), &format!("{d}/messy.stv")]);
    assert_eq!((c, out, err), (0, String::new(), String::new()));
    assert_eq!(read(&dir.join("notes.txt")), MESSY_FORMATTED);
    assert_eq!(read(&dir.join("messy.stv")), MESSY_FORMATTED);
    let (c, out, _) = stave(&["fmt", "--check", &format!("{d}/messy.stv"), &format!("{d}/sub/deep.stv")]);
    assert_eq!((c, out), (0, String::new()));
}

#[test]
fn usage_errors() {
    let (c, _, err) = stave(&["fmt", "/nonexistent/stave-fmt"]);
    assert_eq!(c, 2);
    assert!(err.starts_with("stave: /nonexistent/stave-fmt: "), "{err}");
    assert_eq!(stave(&["fmt", "--json"]).0, 2);
    assert_eq!(stave(&["fmt", "--check=yes"]).0, 2);
}
