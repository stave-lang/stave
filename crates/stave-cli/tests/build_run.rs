//! `stave build`, `stave run`, `stave test` end to end (contracts/cli.md): static
//! executables named after the package (or `-o`), `stave run` exits with the program's status,
//! E1001 for a target the host cannot run, `test fn`s under `stave test PKG_DIR`, the `ir` dump.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run_prog(name: &str) -> String {
    repo().join("tests/conformance/run").join(name).display().to_string()
}

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("build_run").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn stave_in(dir: &Path, args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_stave"))
        .args(args)
        .current_dir(dir)
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

fn host_runs() -> bool {
    cfg!(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))
}

fn run_bin(p: &Path) -> (i32, String, String) {
    let o = Command::new(p).output().unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

fn is_static_exec(p: &Path) -> bool {
    let b = std::fs::read(p).unwrap();
    let phoff = u64::from_le_bytes(b[32..40].try_into().unwrap()) as usize;
    let phnum = u16::from_le_bytes([b[56], b[57]]) as usize;
    &b[..4] == b"\x7fELF"
        && u16::from_le_bytes([b[16], b[17]]) == 2
        && (0..phnum).all(|k| {
            let ty = u32::from_le_bytes(b[phoff + 56 * k..phoff + 56 * k + 4].try_into().unwrap());
            ty != 2 && ty != 3
        })
}

#[test]
fn build_hello_default_name_and_dash_o() {
    let d = tmp("hello");
    let (c, out, err) = stave_in(&d, &["build", &run_prog("hello")]);
    assert_eq!(c, 0, "{err}");
    assert!(out.is_empty());
    let bin = d.join("hello");
    assert!(is_static_exec(&bin), "{}", bin.display());
    if host_runs() {
        assert_eq!(run_bin(&bin), (0, "hello\n".to_string(), String::new()));
    }
    let (c, _, err) = stave_in(&d, &["build", &run_prog("hello"), "-o", "out/h2", "--release"]);
    assert_eq!(c, 2, "missing directory: {err}");
    std::fs::create_dir_all(d.join("out")).unwrap();
    let (c, _, err) = stave_in(&d, &["build", &run_prog("hello"), "-o", "out/h2", "--release"]);
    assert_eq!(c, 0, "{err}");
    assert!(is_static_exec(&d.join("out/h2")));
    if host_runs() {
        assert_eq!(run_bin(&d.join("out/h2")).1, "hello\n");
    }
    // A single file is named after its stem; AArch64 builds anywhere.
    let (c, _, err) = stave_in(&d, &["build", &run_prog("exit_code.stv"), "--target", "aarch64-linux"]);
    assert_eq!(c, 0, "{err}");
    assert_eq!(std::fs::read(d.join("exit_code")).unwrap()[18], 183, "EM_AARCH64");
}

#[test]
fn build_with_errors_writes_nothing() {
    let d = tmp("errors");
    let input = repo().join("tests/conformance/dumps/dropped_item/input").display().to_string();
    let (c, _, err) = stave_in(&d, &["build", &input, "-o", "x"]);
    assert_eq!(c, 1, "{err}");
    assert!(err.starts_with("src/main.stv:8:9: error[E0004]: "), "{err}");
    assert!(!d.join("x").exists());
}

#[test]
fn run_exits_with_the_program_status() {
    if !host_runs() {
        return;
    }
    let d = tmp("run");
    let (c, out, err) = stave_in(&d, &["run", &run_prog("hello"), "--", "a", "b"]);
    assert_eq!((c, out.as_str()), (0, "hello\n"), "{err}");
    let (c, _, err) = stave_in(&d, &["run", &run_prog("exit_code.stv")]);
    assert_eq!(c, 7, "{err}");
    let (c, _, err) = stave_in(&d, &["run", &run_prog("overflow_add.stv")]);
    assert_eq!(c, 101, "{err}");
    assert!(err.contains("panic: Overflow at overflow_add.stv:5:14 in main.add_one (thread 0)"), "{err}");
    // The binary was temporary: nothing left in the working directory.
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
}

#[test]
fn run_foreign_target_is_e1001() {
    let foreign = if cfg!(target_arch = "x86_64") { "aarch64-linux" } else { "x86_64-linux" };
    let d = tmp("e1001");
    let (c, out, err) = stave_in(&d, &["run", &run_prog("hello"), "--target", foreign]);
    assert_eq!(c, 1, "{err}");
    assert!(out.is_empty());
    assert!(err.starts_with("error[E1001]: target `"), "{err}");
    let (c, _, err) =
        stave_in(&d, &["run", &run_prog("hello"), "--target", foreign, "--diagnostics", "json"]);
    assert_eq!(c, 1);
    let v: serde_json::Value = serde_json::from_str(err.lines().next().unwrap()).unwrap();
    assert_eq!(v["code"], "E1001");
}

#[test]
fn ir_dump_written_without_errors_only() {
    let d = tmp("ir");
    let dd = d.join("dumps").display().to_string();
    let (c, _, err) = stave_in(&d, &["build", &run_prog("fib"), "--dump=ir", "--dump-dir", &dd]);
    assert_eq!(c, 0, "{err}");
    let ir = std::fs::read_to_string(d.join("dumps/ir.txt")).unwrap();
    assert!(
        ir.lines().any(|l| l.starts_with("fn @") && l.contains(" main.main ")),
        "{}",
        &ir[..ir.len().min(2000)]
    );
    // `check` lowers too when `ir` is asked for; the dump is deterministic.
    let dd2 = d.join("dumps2").display().to_string();
    let (c, _, err) = stave_in(&d, &["check", &run_prog("fib"), "--dump=ir", "--dump-dir", &dd2]);
    assert_eq!(c, 0, "{err}");
    assert_eq!(std::fs::read_to_string(d.join("dumps2/ir.txt")).unwrap(), ir);
    // With errors: no ir.txt (and an old one is removed).
    let input = repo().join("tests/conformance/dumps/dropped_item/input").display().to_string();
    let (c, _, _) = stave_in(&d, &["build", &input, "--dump=ir,diags", "--dump-dir", &dd]);
    assert_eq!(c, 1);
    assert!(!d.join("dumps/ir.txt").exists());
    assert!(d.join("dumps/diags.txt").is_file());
}

#[test]
fn test_fns_of_a_package() {
    let pkg = run_prog("test_fn");
    let d = tmp("testfn");
    let (c, out, err) = stave_in(&d, &["test", &pkg]);
    if !host_runs() {
        return;
    }
    assert_eq!(c, 0, "{out}{err}");
    assert!(out.contains("ok        test_fn\n"), "the package's own run test: {out}");
    for t in ["main.adds_up", "main.squares", "main.reads_input"] {
        assert!(out.contains(&format!("ok        test {t}\n")), "{t}: {out}");
    }
    assert!(out.ends_with("test fns: 3 passed, 0 failed, 0 not run\n"), "{out}");
    let (c, out, _) = stave_in(&d, &["test", &pkg, "--filter", "squares"]);
    assert_eq!(c, 0);
    assert!(out.ends_with("test fns: 1 passed, 0 failed, 0 not run\n"), "{out}");
    // A failing test fn fails `stave test` with its panic report.
    let bad = tmp("testfn_bad");
    std::fs::create_dir_all(bad.join("src")).unwrap();
    std::fs::write(bad.join("stave.toml"), "name = \"bad\"\nkind = \"bin\"\n").unwrap();
    std::fs::write(
        bad.join("src/main.stv"),
        "import std.debug;\n\nfn main() -> i32 {\n    return 0;\n}\n\ntest fn fails() {\n    debug.assert(1u32 + 1 == 3, \"one plus one\");\n}\n\ntest fn passes() {\n    debug.assert(true, \"fine\");\n}\n",
    )
    .unwrap();
    let (c, out, err) = stave_in(&bad, &["test", "."]);
    assert_eq!(c, 1, "{out}{err}");
    assert!(out.contains("FAILED    test main.fails (exit 101)\n"), "{out}");
    assert!(out.contains("    panic: Assert at src/main.stv:8:"), "{out}");
    assert!(out.contains("    one plus one\n"), "{out}");
    assert!(out.contains("ok        test main.passes\n"), "{out}");
    assert!(out.ends_with("test fns: 1 passed, 1 failed, 0 not run\n"), "{out}");
}

/// A package directory `dir` with manifest name `name` (TOML-escaped) and an empty `main`.
fn named_package(dir: &Path, name: &str) {
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let quoted = name.replace('\\', "\\\\").replace('"', "\\\"").replace('\0', "\\u0000");
    std::fs::write(dir.join("stave.toml"), format!("name = \"{quoted}\"\nkind = \"bin\"\n")).unwrap();
    std::fs::write(dir.join("src/main.stv"), "fn main() -> i32 {\n    return 0;\n}\n").unwrap();
}

/// The default output name comes from the manifest: anything but one plain path component is
/// refused, so a package cannot choose where `stave build` writes (or `stave run` executes) a file.
/// The manifest check (E1002 `bad-name`) rejects it in `stave check` too; the output-name check
/// behind it is defence in depth (unit test in the crate).
#[test]
fn unsafe_package_names_are_refused() {
    let d = tmp("names");
    let work = d.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let abs = d.join("abs-target");
    let abs_name = abs.display().to_string();
    for (k, name) in ["", ".", "..", "../escaped", abs_name.as_str(), "a/b", "a\\b", "a\0b", "-o", "--help"]
        .into_iter()
        .enumerate()
    {
        let pkg = d.join(format!("p{k}"));
        named_package(&pkg, name);
        let p = pkg.display().to_string();
        for cmd in ["check", "build", "run"] {
            let (c, _, err) = stave_in(&work, &[cmd, &p]);
            assert_eq!(c, 1, "{cmd} {name:?}: {err}");
            assert!(err.contains("E1002"), "{cmd} {name:?}: {err}");
        }
        // `-o` names the file explicitly.
        let (c, _, err) = stave_in(&work, &["build", &p, "-o", "explicit"]);
        if c == 0 {
            assert!(is_static_exec(&work.join("explicit")), "{name:?}");
            std::fs::remove_file(work.join("explicit")).unwrap();
        } else {
            assert_eq!(c, 1, "{name:?}: {err}");
        }
    }
    // Nothing escaped the working directory, nothing was left in it.
    assert!(!d.join("escaped").exists() && !abs.exists());
    assert_eq!(std::fs::read_dir(&work).unwrap().count(), 0);
}

#[test]
fn single_file_stem_is_checked() {
    let d = tmp("stem");
    let work = d.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let f = d.join("-x.stv");
    std::fs::write(&f, "fn main() -> i32 {\n    return 0;\n}\n").unwrap();
    let (c, _, err) = stave_in(&work, &["build", &f.display().to_string()]);
    assert_eq!(c, 2, "{err}");
    assert!(err.contains("file stem `-x` cannot be the output file name"), "{err}");
    assert_eq!(std::fs::read_dir(&work).unwrap().count(), 0);
    let ok = d.join("plain.stv");
    std::fs::write(&ok, "fn main() -> i32 {\n    return 0;\n}\n").unwrap();
    let (c, _, err) = stave_in(&work, &["build", &ok.display().to_string()]);
    assert_eq!(c, 0, "{err}");
    assert!(is_static_exec(&work.join("plain")));
}

/// A valid program with vector values is "not implemented yet" (exit 2, naming vector code
/// generation), never an internal error (exit 3); `stave check` accepts it.
#[test]
fn vector_program_is_not_implemented() {
    let d = tmp("simd");
    let f = d.join("simd.stv");
    std::fs::write(&f, "fn main() -> i32 {\n    let a = f32x4.splat(0.0);\n    return 0;\n}\n").unwrap();
    let p = f.display().to_string();
    let (c, _, err) = stave_in(&d, &["check", &p]);
    assert_eq!(c, 0, "{err}");
    for target in ["x86_64-linux", "aarch64-linux"] {
        let (c, _, err) = stave_in(&d, &["build", &p, "--target", target]);
        assert_eq!(c, 2, "{target}: {err}");
        assert!(err.contains("is not implemented yet"), "{target}: {err}");
        assert!(err.contains("vector code generation"), "{target}: {err}");
        assert!(!d.join("simd").exists());
    }
}

/// Syntax that parses but has no semantics yet (parameter `from`, contracts, or-patterns) is
/// "not implemented yet" for `stave check` and `stave build` (exit 2, naming the construct and
/// its place); an error in the same program is reported instead (exit 1).
#[test]
fn parsed_only_syntax_is_not_implemented() {
    let d = tmp("v07");
    let cases = [
        (
            "contract.stv",
            "fn at(xs: &[u32], i: usize) -> u32 requires(i < xs.len()) {\n    return xs[i];\n}\n\
             fn main() -> i32 {\n    return 0;\n}\n",
            "a `requires` / `ensures` contract at ",
            ":1:36",
        ),
        (
            "param_from.stv",
            "view struct S { r: &u32 }\nfn put(s: &mut S from x, x: &u32) {\n    s.r = x;\n}\n\
             fn main() -> i32 {\n    return 0;\n}\n",
            "a `from` clause on a parameter at ",
            ":2:18",
        ),
        (
            "or_pattern.stv",
            "fn main() -> i32 {\n    let x: u8 = 1;\n    return match x {\n        0 | 1 => 0,\n        \
             _ => 1,\n    };\n}\n",
            "an or-pattern (`A | B`) at ",
            ":4:9",
        ),
    ];
    for (name, src, what, at) in cases {
        let f = d.join(name);
        std::fs::write(&f, src).unwrap();
        let p = f.display().to_string();
        for cmd in ["check", "build"] {
            let (c, _, err) = stave_in(&d, &[cmd, &p]);
            assert_eq!(c, 2, "{name} {cmd}: {err}");
            assert!(err.contains("is not implemented yet"), "{name} {cmd}: {err}");
            assert!(err.contains(what) && err.contains(at), "{name} {cmd}: {err}");
        }
    }
    let f = d.join("error.stv");
    let src = "fn f(n: u32) -> u32 requires(n > 0) {\n    return n;\n}\n\
               fn main() -> i32 {\n    return nope;\n}\n";
    std::fs::write(&f, src).unwrap();
    let (c, _, err) = stave_in(&d, &["check", &f.display().to_string()]);
    assert_eq!(c, 1, "{err}");
    assert!(err.contains("E0101"), "{err}");
}
