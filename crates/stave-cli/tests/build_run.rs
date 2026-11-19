//! `stave build`, `stave run`, `stave test` end to end (contracts/cli.md): static
//! executables named after the package (or `-o`), `stave run` exits with the program's status,
//! E1001 for a target the host cannot run, `test fn`s under `stave test PKG_DIR`, the `ir` dump.
//! `stave run` and `stave test` run programs from memory: no file is written (checked on the
//! system calls when `strace` works here), and the results equal those of the built file.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

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

/// Runs the executable script `script` with `stave` first on `PATH`, as a shell would.
#[cfg(unix)]
fn run_script(script: &Path, args: &[&str]) -> (i32, String, String) {
    let bin_dir = Path::new(env!("CARGO_BIN_EXE_stave")).parent().unwrap().to_path_buf();
    let rest = std::env::var_os("PATH").unwrap_or_default();
    let path = std::env::join_paths(std::iter::once(bin_dir).chain(std::env::split_paths(&rest))).unwrap();
    let mut tries = 0;
    loop {
        let r = Command::new(script)
            .args(args)
            .current_dir(script.parent().unwrap())
            .env("PATH", &path)
            .env_remove("STAVE_THREADS")
            .env_remove("STAVE_STD_DIR")
            .env_remove("STAVE_SEMA_SHUFFLE")
            .output();
        match r {
            // ETXTBSY: another test thread forked a child while this one held the script open
            // for writing; the child closes its copy when it execs.
            Err(e) if e.raw_os_error() == Some(26) && tries < 100 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            r => {
                let o = r.unwrap_or_else(|e| panic!("{}: {e}", script.display()));
                return (
                    o.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&o.stdout).into(),
                    String::from_utf8_lossy(&o.stderr).into(),
                );
            }
        }
    }
}

/// An executable `.stv` file that starts with `#!/usr/bin/env -S stave run` runs directly: the
/// kernel starts `stave run SCRIPT ARGS...`. The shebang line is line 1 of the diagnostics.
#[cfg(unix)]
#[test]
fn shebang_script_runs_directly() {
    use std::os::unix::fs::PermissionsExt;
    if !host_runs() {
        return;
    }
    let d = tmp("shebang");
    let script = |name: &str, body: &str| {
        let p = d.join(name);
        std::fs::write(&p, format!("#!/usr/bin/env -S stave run\n{body}")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    };
    let greet = script(
        "greet.stv",
        "import std.env;\nimport std.io;\n\nfn main() -> i32 {\n    let n = env.count();\n    \
         io.print(f\"args {n}\\n\");\n    match env.arg(2) {\n        Option.Some(a) => {\n            \
         io.print(a.as_str());\n            io.print(\"\\n\");\n        }\n        Option.None => { }\n    }\n    \
         return 7;\n}\n",
    );
    // A script run prints no summary line: stderr is the program's.
    let (c, out, err) = run_script(&greet, &["one", "two"]);
    assert_eq!((c, out.as_str(), err.as_str()), (7, "args 3\ntwo\n", ""));
    let (c, out, err) = run_script(&greet, &[]);
    assert_eq!((c, out.as_str(), err.as_str()), (7, "args 1\n", ""));
    // The same file through the commands that take a root.
    let p = greet.display().to_string();
    let (c, _, err) = stave_in(&d, &["check", &p]);
    assert_eq!(c, 0, "{err}");
    assert!(err.contains("compiled on "), "{err}");
    let (c, out, err) = stave_in(&d, &["run", &p, "one", "--", "-x", "y"]);
    assert_eq!((c, out.as_str(), err.as_str()), (7, "args 4\n-x\n", ""));
    // Without a shebang line the run names the compiler.
    let plain = d.join("plain.stv");
    std::fs::write(&plain, "fn main() -> i32 {\n    return 0;\n}\n").unwrap();
    let (c, _, err) = stave_in(&d, &["run", &plain.display().to_string()]);
    assert_eq!(c, 0, "{err}");
    assert!(err.contains("compiled on "), "{err}");
    // Line 3 of the file is line 3 of the diagnostic.
    let bad = script("bad.stv", "fn main() -> i32 {\n    return nope;\n}\n");
    let (c, out, err) = run_script(&bad, &[]);
    assert_eq!((c, out.as_str()), (1, ""), "{err}");
    assert!(err.starts_with("bad.stv:3:12: error[E0101]: "), "{err}");
    assert!(!err.contains("compiled on "), "{err}");
}

/// `stave` in `dir` with `TMPDIR` set to `tmpdir`, optionally under `strace` with the flags
/// `trace`.
#[cfg(unix)]
fn stave_env(dir: &Path, tmpdir: &Path, trace: Option<&[&str]>, args: &[&str]) -> (i32, String, String) {
    let mut cmd = match trace {
        Some(flags) => {
            let mut c = Command::new("strace");
            c.args(flags).arg(env!("CARGO_BIN_EXE_stave"));
            c
        }
        None => Command::new(env!("CARGO_BIN_EXE_stave")),
    };
    let o = cmd
        .args(args)
        .current_dir(dir)
        .env("TMPDIR", tmpdir)
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

/// Whether `strace` can trace a process here (installed, and ptrace allowed).
#[cfg(unix)]
fn strace_works() -> bool {
    Command::new("strace").args(["-o", "/dev/null", "true"]).output().is_ok_and(|o| o.status.success())
}

#[cfg(unix)]
fn have_qemu_aarch64() -> bool {
    legs::gate(
        "aarch64-linux",
        Command::new("qemu-aarch64").arg("--version").output().is_ok_and(|o| o.status.success()),
    )
}

/// `stave run` and `stave test` hold every executable in memory. They work where nothing can
/// be written (no temporary directory, a read-only working directory), and under `strace` no
/// system call creates a file or directory: the image is a `memfd_create` file started as
/// `/proc/self/fd/N`, or handed to the runner as `/dev/fd/N`.
#[cfg(unix)]
#[test]
fn run_and_test_write_no_file() {
    use std::os::unix::fs::PermissionsExt;
    if !host_runs() {
        return;
    }
    let d = tmp("nofile");
    let cwd = d.join("readonly");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::set_permissions(&cwd, std::fs::Permissions::from_mode(0o555)).unwrap();
    let no_tmp = d.join("no-such-directory");
    let (hello, test_fn) = (run_prog("hello"), run_prog("test_fn"));
    let (hello, test_fn) = (hello.as_str(), test_fn.as_str());
    let mut commands: Vec<(Vec<&str>, &str)> = vec![
        (vec!["run", hello], "hello\n"),
        (vec!["run", hello, "--release", "--", "a"], "hello\n"),
        (vec!["test", test_fn], "test fns: 3 passed, 0 failed, 0 not run\n"),
    ];
    if have_qemu_aarch64() {
        commands.push((
            vec!["test", hello, "--target", "aarch64-linux", "--runner", "qemu-aarch64"],
            "1 passed, 0 failed, 0 malformed, 0 not run, 0 not run (stage not implemented)\n",
        ));
    }
    for (args, want) in &commands {
        let (c, out, err) = stave_env(&cwd, &no_tmp, None, args);
        assert_eq!(c, 0, "{args:?}: {out}{err}");
        assert!(out.contains(want), "{args:?}: {out}");
        assert!(!err.contains("temporary file"), "{args:?}: {err}");
    }
    assert_eq!(std::fs::read_dir(&cwd).unwrap().count(), 0);
    if !strace_works() {
        eprintln!("note: strace cannot trace here: system calls not checked");
        return;
    }
    let log = d.join("trace.log");
    let log_arg = log.display().to_string();
    let flags = ["-f", "-e", "trace=%file,memfd_create", "-o", log_arg.as_str()];
    let creating = [
        "O_CREAT",
        "O_TMPFILE",
        " creat(",
        " mkdir(",
        " mkdirat(",
        " mknod(",
        " mknodat(",
        " symlink(",
        " symlinkat(",
        " link(",
        " linkat(",
        " rename(",
        " renameat(",
        " renameat2(",
    ];
    for (args, want) in &commands {
        let (c, out, err) = stave_env(&cwd, &no_tmp, Some(&flags), args);
        assert_eq!(c, 0, "{args:?}: {out}{err}");
        assert!(out.contains(want), "{args:?}: {out}");
        let trace = std::fs::read_to_string(&log).unwrap();
        for l in trace.lines() {
            assert!(!creating.iter().any(|c| l.contains(c)), "{args:?} creates a file: {l}");
        }
        assert!(trace.contains(" memfd_create("), "{args:?}:\n{trace}");
        if args.contains(&"--runner") {
            assert!(trace.lines().any(|l| l.contains(" execve(") && l.contains("\"/dev/fd/")), "{trace}");
        } else {
            assert!(
                trace.lines().any(|l| l.contains(" execve(") && l.contains("\"/proc/self/fd/")),
                "{trace}"
            );
        }
    }
}

/// Without `memfd_create` in the kernel (ENOSYS, injected by `strace`) the program runs from a
/// temporary file, removed afterwards, and stderr says so. Any other error is reported, not
/// worked around.
#[cfg(unix)]
#[test]
fn run_falls_back_to_a_file_only_on_enosys() {
    if !host_runs() {
        return;
    }
    if !strace_works() {
        eprintln!("note: skipped (strace cannot trace here)");
        return;
    }
    let d = tmp("fallback");
    let tmpdir = d.join("tmp");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let call = "memfd_create";
    let (trace, inject) = (format!("trace={call}"), format!("inject={call}:error=ENOSYS"));
    let flags = ["-f", "-o", "/dev/null", "-e", trace.as_str(), "-e", inject.as_str()];
    let note = format!("stave: the kernel has no `{call}` (ENOSYS): programs run from temporary files");
    let (c, out, err) = stave_env(&d, &tmpdir, Some(&flags), &["run", &run_prog("hello"), "--", "a"]);
    assert_eq!((c, out.as_str()), (0, "hello\n"), "{call}: {err}");
    assert_eq!(err.matches(&note).count(), 1, "{call}: {err}");
    let (c, _, err) = stave_env(&d, &tmpdir, Some(&flags), &["run", &run_prog("exit_code.stv")]);
    assert_eq!(c, 7, "{call}: {err}");
    let (c, out, err) = stave_env(&d, &tmpdir, Some(&flags), &["test", &run_prog("test_fn")]);
    assert_eq!(c, 0, "{call}: {out}{err}");
    assert!(out.ends_with("test fns: 3 passed, 0 failed, 0 not run\n"), "{call}: {out}");
    assert_eq!(err.matches(&note).count(), 1, "{call}: {err}");
    assert_eq!(std::fs::read_dir(&tmpdir).unwrap().count(), 0, "{call}: temporary files left");
    let flags =
        ["-f", "-o", "/dev/null", "-e", "trace=memfd_create", "-e", "inject=memfd_create:error=EACCES"];
    let (c, out, err) = stave_env(&d, &tmpdir, Some(&flags), &["run", &run_prog("hello")]);
    assert_eq!((c, out.as_str()), (2, ""), "{err}");
    assert!(err.contains("stave: cannot load the program: "), "{err}");
    assert!(!err.contains("temporary files"), "{err}");
}

/// `stave run` gives the exit status, stdout, and stderr of the file `stave build` writes.
#[test]
fn run_equals_the_built_binary() {
    if !host_runs() {
        return;
    }
    let d = tmp("same");
    for (k, prog) in
        ["hello", "exit_code.stv", "overflow_add.stv", "env_args.stv", "fib"].into_iter().enumerate()
    {
        for release in [false, true] {
            let (bin, p) = (format!("bin{k}{release}"), run_prog(prog));
            let mut build = vec!["build", "-o", bin.as_str(), p.as_str()];
            let mut run = vec!["run", p.as_str()];
            if release {
                build.push("--release");
                run.push("--release");
            }
            let (c, _, err) = stave_in(&d, &build);
            assert_eq!(c, 0, "{prog}: {err}");
            let (code, out, err) = run_bin(&d.join(&bin));
            let (run_code, run_out, run_err) = stave_in(&d, &run);
            assert_eq!((run_code, &run_out), (code, &out), "{prog}: {run_err}");
            // after the compiler's summary line
            assert!(run_err.ends_with(&err), "{prog}: {run_err:?} does not end with {err:?}");
        }
    }
}

const OVERFLOWING_SUM: &str = "import std.io;

fn main() -> i32 {
    let a: [u8; 40] = [7; 40];
    let s = (&a[0..40]).sum();
    io.print(f\"{s}\\n\");
    return 0;
}
";

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dst = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_tree(&e.path(), &dst);
        } else {
            std::fs::copy(e.path(), dst).unwrap();
        }
    }
}

/// Whatever goes into a binary names the files of `std` and `runtime` package-relative, so the
/// bytes do not depend on where the library is installed.
#[test]
fn binaries_do_not_depend_on_the_library_location() {
    let d = tmp("library_location");
    let src = d.join("prog.stv");
    std::fs::write(&src, OVERFLOWING_SUM).unwrap();
    let libs: Vec<PathBuf> = ["lib", "a-much-longer-library-directory-name"]
        .iter()
        .map(|n| {
            let lib = d.join(n);
            for p in ["std", "runtime"] {
                copy_tree(&repo().join(p), &lib.join(p));
            }
            lib
        })
        .collect();
    for target in ["x86_64-linux", "aarch64-linux"] {
        for release in [false, true] {
            let mut bins = Vec::new();
            for (i, lib) in libs.iter().enumerate() {
                let out = d.join(format!("prog-{target}-{release}-{i}"));
                let mut args = vec!["build", "prog.stv", "--target", target, "-o", out.to_str().unwrap()];
                if release {
                    args.push("--release");
                }
                let o = Command::new(env!("CARGO_BIN_EXE_stave"))
                    .args(&args)
                    .current_dir(&d)
                    .env("STAVE_STD_DIR", lib)
                    .env_remove("STAVE_THREADS")
                    .env_remove("STAVE_SEMA_SHUFFLE")
                    .output()
                    .unwrap();
                assert!(
                    o.status.success(),
                    "{target} release={release}: {}",
                    String::from_utf8_lossy(&o.stderr)
                );
                let bytes = std::fs::read(&out).unwrap();
                let at = lib.display().to_string();
                assert!(
                    !bytes.windows(at.len()).any(|w| w == at.as_bytes()),
                    "{target} release={release}: the binary names the library directory {at}"
                );
                bins.push(bytes);
            }
            assert!(bins[0] == bins[1], "{target} release={release}: the binaries differ");
        }
    }
}

/// A trap inside `std` names its file package-relative.
#[test]
fn panic_in_std_names_the_package_relative_file() {
    if !host_runs() {
        return;
    }
    let d = tmp("std_panic_path");
    let src = d.join("prog.stv");
    std::fs::write(&src, OVERFLOWING_SUM).unwrap();
    for release in [false, true] {
        let mut args = vec!["build", "prog.stv", "-o", "prog"];
        if release {
            args.push("--release");
        }
        let (c, _, err) = stave_in(&d, &args);
        assert_eq!(c, 0, "{err}");
        let (code, _, err) = run_bin(&d.join("prog"));
        assert_eq!(code, 101, "{err}");
        assert!(err.contains("panic: Overflow at std/slice.stv:"), "release={release}: {err}");
        assert!(err.contains(" in std.slice.impl.sum "), "release={release}: {err}");
    }
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

fn contains(hay: &[u8], needle: &str) -> bool {
    hay.windows(needle.len()).any(|w| w == needle.as_bytes())
}

/// A debug build roots the program's own fns but only the `std` and `runtime` fns it reaches.
#[test]
fn debug_build_roots_own_fns_not_unused_std() {
    let d = tmp("roots");
    std::fs::create_dir_all(d.join("p/src")).unwrap();
    std::fs::write(d.join("p/stave.toml"), "name = \"p\"\nkind = \"bin\"\n").unwrap();
    std::fs::write(
        d.join("p/src/main.stv"),
        "import std.io;\n\nfn never_called() -> i32 {\n    return 7;\n}\n\nfn main() -> i32 {\n    io.print(\"hi\\n\");\n    return 0;\n}\n",
    )
    .unwrap();
    let pkg = d.join("p").display().to_string();
    let (c, _, err) = stave_in(&d, &["build", &pkg, "-o", "p.bin"]);
    assert_eq!(c, 0, "{err}");
    let bin = std::fs::read(d.join("p.bin")).unwrap();
    assert!(contains(&bin, "main.never_called"), "own fn is a root");
    assert!(contains(&bin, "std.io.print"), "reached std fn is kept");
    assert!(!contains(&bin, "std.num.parse_f64"), "unreached std fn is not lowered");
    assert!(!contains(&bin, "std.env.count"), "unreached std fn is not lowered");
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

/// The `test fn` lines of a `stave test` output.
fn test_lines(out: &str) -> Vec<String> {
    out.lines().filter(|l| l.contains(" test main.")).map(String::from).collect()
}

/// `test fn`s with parameters: inputs from the parameter types narrowed by `requires`, the same
/// on every run, build mode, and target; a failing input is named with its literals and a
/// command that runs it alone.
#[test]
fn test_fn_inputs() {
    let pkg = run_prog("test_fn_params");
    let d = tmp("testfn_params");
    let (c, out, err) = stave_in(&d, &["test", &pkg]);
    if !host_runs() {
        return;
    }
    assert_eq!(c, 0, "{out}{err}");
    let want = [
        "ok        test main.scalars: 1006 inputs run, 0 skipped",
        "ok        test main.slices: 438 inputs run, 0 skipped",
        "ok        test main.aggregates: 429 inputs run, 0 skipped",
        "ok        test main.ranges: 285 inputs run, 0 skipped",
        "ok        test main.filtered: 276 inputs run, 23 skipped",
        "ok        test main.sums: 302 inputs run, 0 skipped",
    ];
    assert_eq!(test_lines(&out), want, "{out}");
    assert!(out.ends_with("test fns: 6 passed, 0 failed, 0 not run\n"), "{out}");
    let (c, again, _) = stave_in(&d, &["test", &pkg]);
    assert_eq!((c, test_lines(&again)), (0, test_lines(&out)), "a second run: {again}");
    let (c, rel, _) = stave_in(&d, &["test", &pkg, "--release"]);
    assert_eq!((c, test_lines(&rel)), (0, test_lines(&out)), "--release: {rel}");
    if have_qemu_aarch64() {
        let (c, a64, err) =
            stave_in(&d, &["test", &pkg, "--target", "aarch64-linux", "--runner", "qemu-aarch64"]);
        assert_eq!((c, test_lines(&a64)), (0, test_lines(&out)), "aarch64: {a64}{err}");
    }

    // A seeded bug: the tail loop drops the last element when three remain.
    let bad = tmp("testfn_seeded");
    std::fs::create_dir_all(bad.join("src")).unwrap();
    std::fs::write(bad.join("stave.toml"), "name = \"seeded\"\nkind = \"bin\"\n").unwrap();
    std::fs::write(
        bad.join("src/main.stv"),
        "import std.debug;

fn main() -> i32 {
    return 0;
}

fn sum4(xs: &[u32]) -> u64 {
    let mut s: u64 = 0;
    let mut i: usize = 0;
    while i + 4 <= xs.len() {
        s = s + (xs[i] as u64) + (xs[i + 1] as u64) + (xs[i + 2] as u64) + (xs[i + 3] as u64);
        i = i + 4;
    }
    let stop = if xs.len() - i == 3 then xs.len() - 1 else xs.len();
    while i < stop {
        s = s + (xs[i] as u64);
        i = i + 1;
    }
    return s;
}

fn sum_ref(xs: &[u32]) -> u64 {
    let mut s: u64 = 0;
    for i in 0..xs.len() {
        s = s + (xs[i] as u64);
    }
    return s;
}

test fn sums_agree(xs: &[u32]) requires(xs.len() <= 9) {
    debug.assert(sum4(xs) == sum_ref(xs), \"sum4 agrees with the reference\");
}

test fn rare(a: u32) requires(a % 1000 == 7) { }

test fn none_left(a: u8) requires(a != a) { }

test fn empty_range(a: u8) requires(a > 3 && a < 2) { }
",
    )
    .unwrap();
    let (c, out, err) = stave_in(&bad, &["test", "."]);
    assert_eq!(c, 1, "{out}{err}");
    assert!(out.contains("FAILED    test main.sums_agree (exit 101) at input 3 (4 of "), "{out}");
    assert!(out.contains("    panic: Assert at src/main.stv:31:"), "{out}");
    assert!(out.contains("    input 3: xs = &[3u32, 4u32, 5u32]\n"), "{out}");
    let replay =
        out.lines().find_map(|l| l.strip_prefix("    replay: stave ")).unwrap_or_else(|| panic!("{out}"));
    assert!(replay.starts_with("test . --filter main.sums_agree --exact --input 3 --target "), "{replay}");
    let words: Vec<&str> = replay.split_whitespace().collect();
    let (c, again, _) = stave_in(&bad, &words);
    assert_eq!(c, 1, "{again}");
    assert!(
        again.contains("FAILED    test main.sums_agree (exit 101) at input 3 (1 of 1 started)\n"),
        "{again}"
    );
    assert!(again.contains("    input 3: xs = &[3u32, 4u32, 5u32]\n"), "{again}");
    assert!(out.contains("ok        test main.rare: 1 input run, "), "{out}");
    assert!(out.contains("    note: `requires` skipped "), "{out}");
    assert!(out.contains("FAILED    test main.none_left: `requires` skipped all "), "{out}");
    assert!(out.contains("FAILED    test main.empty_range: its `requires` ranges admit no input\n"), "{out}");
    assert!(out.ends_with("test fns: 1 passed, 3 failed, 0 not run\n"), "{out}");
    let (c, none, _) = stave_in(&bad, &["test", ".", "--filter", "sums_agree", "--input", "100000"]);
    assert_eq!(c, 1);
    assert!(none.contains("FAILED    test main.sums_agree: no input 100000: the test has "), "{none}");
}

/// `--input N` runs one input of exactly one `test fn`: the replay line names the failing test
/// with `--filter T --exact`, so a test whose name contains T is not run; a selection of several
/// tests, of none, or of a test without parameters is a usage error (exit 2).
#[test]
fn input_selects_one_test() {
    let pkg = run_prog("test_fn_replay");
    let d = tmp("testfn_replay");
    let (c, out, err) = stave_in(&d, &["test", &pkg]);
    if !host_runs() {
        return;
    }
    assert_eq!(c, 1, "{out}{err}");
    assert!(out.contains("FAILED    test main.sum_agrees (exit 101) at input "), "{out}");
    assert!(out.contains("ok        test main.sum_agrees_short: "), "{out}");
    assert!(out.ends_with("test fns: 2 passed, 1 failed, 0 not run\n"), "{out}");
    let replay =
        out.lines().find_map(|l| l.strip_prefix("    replay: stave ")).unwrap_or_else(|| panic!("{out}"));
    let words: Vec<&str> = replay.split_whitespace().collect();
    let at = words.iter().position(|w| *w == "--filter").unwrap_or_else(|| panic!("{replay}"));
    assert_eq!(words[at + 1..at + 4], ["main.sum_agrees", "--exact", "--input"], "{replay}");
    let input = words[at + 4];
    let (c, again, err) = stave_in(&d, &words);
    assert_eq!(c, 1, "{again}{err}");
    let lines = test_lines(&again);
    assert_eq!(lines.len(), 1, "only the failing test runs: {again}");
    assert!(
        lines[0].starts_with(&format!("FAILED    test main.sum_agrees (exit 101) at input {input} (1 of 1"))
    );
    assert!(again.ends_with("test fns: 0 passed, 1 failed, 0 not run\n"), "{again}");

    // `--filter main.sum_agrees` alone selects both tests.
    let (c, out, err) = stave_in(&d, &["test", &pkg, "--filter", "main.sum_agrees", "--input", input]);
    assert_eq!(c, 2, "{out}{err}");
    assert!(
        err.contains(&format!("stave: --input {input}: --filter main.sum_agrees selects 2 test fns (")),
        "{err}"
    );
    assert!(err.contains("main.sum_agrees_short") && err.contains("--filter NAME --exact"), "{err}");
    assert!(test_lines(&out).is_empty(), "{out}");
    let (c, _, err) = stave_in(&d, &["test", &pkg, "--input", "0"]);
    assert_eq!(c, 2);
    assert!(err.contains("stave: --input 0: the package selects 3 test fns ("), "{err}");
    let (c, _, err) = stave_in(&d, &["test", &pkg, "--filter", "main.sum", "--exact", "--input", "0"]);
    assert_eq!(c, 2);
    assert!(err.contains("stave: --input 0: --filter main.sum --exact selects no test fn"), "{err}");
    let (c, _, err) = stave_in(&d, &["test", &pkg, "--filter", "plain", "--input", "0"]);
    assert_eq!(c, 2);
    assert!(
        err.contains("stave: --input 0: test fn main.plain has no parameters, so it has no input 0"),
        "{err}"
    );
    let (c, _, err) = stave_in(&d, &["test", &d.display().to_string(), "--input", "0"]);
    assert_eq!(c, 2);
    assert!(err.contains("is not a package, so it has no test fn"), "{err}");
    // Without `--input`, `--exact` selects the one test and runs all its inputs.
    let (c, out, _) = stave_in(&d, &["test", &pkg, "--filter", "main.sum_agrees_short", "--exact"]);
    assert_eq!((c, test_lines(&out).len()), (0, 1), "{out}");
}

/// `test fn` parameters of struct type: the inputs of `run/test_fn_structs` (the same with
/// `--release` and on AArch64); a failing input of `run/test_fn_struct_fail` prints as struct
/// literals that compile where the test's module would write them, and its replay reruns it.
#[test]
fn test_fn_struct_inputs() {
    let pkg = run_prog("test_fn_structs");
    let d = tmp("testfn_structs");
    let (c, out, err) = stave_in(&d, &["test", &pkg]);
    if !host_runs() {
        return;
    }
    assert_eq!(c, 0, "{out}{err}");
    // `field_ranges`: x in -3..=3, y = 7, id in 0..=9, a bool: 140 values, 121 of them inputs.
    let want = [
        "ok        test main.containers: 621 inputs run, 0 skipped",
        "ok        test main.generic_and_imported: 446 inputs run, 0 skipped",
        "ok        test main.nested: 622 inputs run, 0 skipped",
        "ok        test main.drops: 96 inputs run, 0 skipped",
        "ok        test main.field_filter: 438 inputs run, 183 skipped",
        "ok        test main.field_ranges: 121 inputs run, 0 skipped",
    ];
    assert_eq!(test_lines(&out), want, "{out}");
    let (c, rel, _) = stave_in(&d, &["test", &pkg, "--release"]);
    assert_eq!((c, test_lines(&rel)), (0, test_lines(&out)), "--release: {rel}");
    if have_qemu_aarch64() {
        let (c, a64, err) =
            stave_in(&d, &["test", &pkg, "--target", "aarch64-linux", "--runner", "qemu-aarch64"]);
        assert_eq!((c, test_lines(&a64)), (0, test_lines(&out)), "aarch64: {a64}{err}");
    }

    let pkg = run_prog("test_fn_struct_fail");
    let (c, out, err) = stave_in(&d, &["test", &pkg]);
    assert_eq!(c, 1, "{out}{err}");
    // Diagonal input 18: `start` takes edge 18 of u32 (127), `end` edge 18 of 0..=99 (98).
    assert!(out.contains("FAILED    test main.span_len (exit 101) at input 18 (19 of "), "{out}");
    assert!(out.contains("    panic: Overflow at src/main.stv:16:18 in main.len"), "{out}");
    let shown = out.lines().find_map(|l| l.strip_prefix("    input ")).unwrap_or_else(|| panic!("{out}"));
    let (input, values) = shown.split_once(": ").unwrap();
    assert_eq!(
        values, "s = Span::<u32>.{ start: 127u32, end: 98u32 }, c = geo.Cell.{ row: 127u32, col: 127u32 }",
        "{out}"
    );
    assert!(out.contains("ok        test main.span_len_ordered: "), "{out}");
    let replay =
        out.lines().find_map(|l| l.strip_prefix("    replay: stave ")).unwrap_or_else(|| panic!("{out}"));
    let (c, again, _) = stave_in(&d, &replay.split_whitespace().collect::<Vec<_>>());
    assert_eq!(c, 1, "{again}");
    assert!(again.contains(&format!("    input {shown}\n")), "{again}");
    assert_eq!(test_lines(&again).len(), 1, "{again}");

    // The literals compile in the test's module.
    let paste = tmp("testfn_struct_paste");
    std::fs::create_dir_all(paste.join("src")).unwrap();
    std::fs::write(paste.join("stave.toml"), "name = \"paste\"\nkind = \"bin\"\n").unwrap();
    std::fs::copy(Path::new(&pkg).join("src/geo.stv"), paste.join("src/geo.stv")).unwrap();
    let (s, c) = values.split_once(", c = ").unwrap();
    let s = s.strip_prefix("s = ").unwrap();
    std::fs::write(
        paste.join("src/main.stv"),
        format!(
            "import geo;\n\nstruct Span<T> {{\n    start: T,\n    end: T,\n}}\n\nfn main() -> i32 {{\n    let s: Span<u32> = {s};\n    let c: geo.Cell = {c};\n    return 0;\n}}\n"
        ),
    )
    .unwrap();
    let (c, out, err) = stave_in(&paste, &["check", "."]);
    assert_eq!(c, 0, "input {input}: {out}{err}");
}

/// E0214 says why a struct is not generatable: the field that is not, or what is private.
#[test]
fn test_fn_struct_e0214_messages() {
    let d = tmp("testfn_struct_e0214");
    let types = repo().join("tests/conformance/types");
    let (c, _, err) = stave_in(&d, &["check", &types.join("test_fn_params.stv").display().to_string()]);
    assert_eq!(c, 1, "{err}");
    for m in [
        "the test runner cannot generate values of type `V`",
        "the test runner cannot generate values of type `E`",
        "the test runner cannot generate values of type `Outer`: it cannot generate field `h.e` of type `E`",
        "the test runner cannot generate values of type `WithVec`: it cannot generate field `v` of type `vec.Vec<u8>`",
        "the test runner cannot generate values of type `WithString`: it cannot generate field `name` of type `string.String`",
        "the test runner cannot generate values of type `Vec<u32>`\n",
        "values of `Big` are too large to generate (more than 4096 scalars)",
    ] {
        assert!(err.contains(m), "{m}: {err}");
    }
    let (c, _, err) = stave_in(&d, &["check", &types.join("test_fn_private").display().to_string()]);
    assert_eq!(c, 1, "{err}");
    for m in [
        "the test runner cannot generate values of type `Sealed`: `y` is private to module `geo`",
        "the test runner cannot generate values of type `Outer`: `sealed.y` is private to module `geo`",
        "the test runner cannot generate values of type `[Sealed; 2]`: `y` is private to module `geo`",
    ] {
        assert!(err.contains(m), "{m}: {err}");
    }
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

/// A valid program with vector values builds for every native target, and `stave check` accepts
/// it.
#[test]
fn vector_program_builds() {
    let d = tmp("simd");
    let f = d.join("simd.stv");
    std::fs::write(&f, "fn main() -> i32 {\n    let a = f32x4.splat(0.0);\n    return 0;\n}\n").unwrap();
    let p = f.display().to_string();
    let (c, _, err) = stave_in(&d, &["check", &p]);
    assert_eq!(c, 0, "{err}");
    for target in ["x86_64-linux", "x86_64-linux-v3", "aarch64-linux"] {
        let (c, _, err) = stave_in(&d, &["build", &p, "--target", target]);
        assert_eq!(c, 0, "{target}: {err}");
        assert!(is_static_exec(&d.join("simd")), "{target}");
        std::fs::remove_file(d.join("simd")).unwrap();
    }
}

/// Every class U op inside `emulate { }` gets an `emulated:` line before the summary line; a U op
/// outside one is E0700.
#[test]
fn emulated_ops_are_listed() {
    let d = tmp("emulate");
    let f = d.join("emu.stv");
    std::fs::write(
        &f,
        "fn f(a: f32x4) -> f32x4 {\n    let mut r = a;\n    emulate {\n        r = a.fma(a, a);\n    }\n    return r;\n}\n\
         fn main() -> i32 {\n    return 0;\n}\n",
    )
    .unwrap();
    let p = f.display().to_string();
    let (c, _, err) = stave_in(&d, &["check", &p, "--target", "x86_64-linux-v2"]);
    assert_eq!(c, 0, "{err}");
    let line = err.lines().find(|l| l.starts_with("emulated: ")).unwrap_or_else(|| panic!("{err}"));
    assert!(line.ends_with("emu.stv:4:15: vec.fma on f32x4 (x86_64-linux-v2)"), "{line}");
    let last = err.lines().last().unwrap();
    assert!(!last.starts_with("emulated: "), "{err}");
    let (c, _, err) = stave_in(&d, &["check", &p, "--target", "x86_64-linux-v3"]);
    assert_eq!(c, 0, "{err}");
    assert!(!err.contains("emulated: "), "{err}");
    std::fs::write(&f, "fn f(a: f32x4) -> f32x4 {\n    return a.fma(a, a);\n}\n").unwrap();
    let (c, _, err) = stave_in(&d, &["check", &p, "--target", "x86_64-linux-v2"]);
    assert_eq!(c, 1, "{err}");
    assert!(err.contains("E0700"), "{err}");
}

/// A `from` clause on a parameter is part of the signature: `stave check` accepts the store it
/// declares, reports the store without it (E0513), and reports the clause on a parameter that
/// is not a `&mut` view (E0514).
#[test]
fn param_from_is_checked_with_the_signature() {
    let d = tmp("v07");
    let f = d.join("param_from.stv");
    let src = "view struct S { r: &u32 }\nfn put(s: &mut S from x, x: &u32) {\n    s.r = x;\n}\n\
               fn main() -> i32 {\n    return 0;\n}\n";
    std::fs::write(&f, src).unwrap();
    let (c, _, err) = stave_in(&d, &["check", &f.display().to_string()]);
    assert_eq!(c, 0, "{err}");
    let f = d.join("no_clause.stv");
    std::fs::write(&f, src.replace(" from x", "")).unwrap();
    let (c, _, err) = stave_in(&d, &["check", &f.display().to_string()]);
    assert_eq!(c, 1, "{err}");
    assert!(err.contains("E0513") && err.contains("add `from x` to parameter `s`"), "{err}");
    let f = d.join("shared.stv");
    std::fs::write(&f, src.replace("&mut S from x", "&S from x").replace("    s.r = x;\n", "")).unwrap();
    let (c, _, err) = stave_in(&d, &["check", &f.display().to_string()]);
    assert_eq!(c, 1, "{err}");
    assert!(err.contains("E0514"), "{err}");
}
