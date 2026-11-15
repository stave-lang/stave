//! End to end on both CPU architectures:
//! every program of `tests/conformance/run/` is built by the driver pipeline (debug and release
//! layouts) into a static ELF executable, which runs natively (x86-64 host) or under
//! `qemu-aarch64`, and in the stave-ir reference interpreter from the same checked IR. stdout,
//! stderr, and the exit status must be identical, and identical again when the same executable
//! runs from memory instead of from its file. The executables have no `PT_INTERP` or
//! `PT_DYNAMIC`, and make only system calls of `runtime/syscalls.list`: checked on the
//! interpreter's syscall log (same IR, same runtime) and, when `strace` is installed, on the
//! native x86-64 runs under `strace -c`.
//!
//! A run that cannot happen here (no qemu-aarch64, not Linux) is skipped with a message; the
//! interpreter side always runs.

use std::path::{Path, PathBuf};
use std::process::Command;

use stave_driver::exec::{self, Executable, Ran, RunSpec, TempDir};
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Package dirs and single `.stv` files directly under `run/`, sorted.
fn programs() -> Vec<PathBuf> {
    let dir = repo().join("tests/conformance/run");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            !name.starts_with('.')
                && !name.starts_with('_')
                && (p.join("stave.toml").is_file() || p.extension().is_some_and(|x| x == "stv"))
        })
        .collect();
    v.sort();
    v
}

fn base_dir(p: &Path) -> PathBuf {
    if p.is_dir() { p.to_path_buf() } else { p.parent().unwrap().to_path_buf() }
}

/// The data files a package program may read from its working directory.
fn data_files(p: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v = Vec::new();
    if p.is_dir() {
        for e in std::fs::read_dir(p).unwrap().flatten() {
            let f = e.path();
            if f.is_file() {
                v.push((f.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read(&f).unwrap()));
            }
        }
    }
    v
}

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{}: diagnostics {:?}", p.display(), out.diagnostics);
    assert!(out.failure.is_none(), "{} ({target}): {}", p.display(), out.failure.as_ref().unwrap());
    assert!(out.image.is_some() && out.module.is_some() && out.main.is_some());
    out
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// The runner for `target` here: `Some(None)` native, `Some(Some(cmd))` through `cmd`, `None`
/// not runnable.
fn runner(target: Triple) -> Option<Option<&'static str>> {
    if exec::can_run(target) {
        return Some(None);
    }
    if target == Triple::parse("aarch64-linux").unwrap() && cfg!(target_os = "linux") && have("qemu-aarch64")
    {
        return Some(Some("qemu-aarch64"));
    }
    None
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// ET_EXEC, the target's machine, no PT_INTERP (3) or PT_DYNAMIC (2).
fn assert_static(bytes: &[u8], target: Triple, what: &str) {
    assert_eq!(&bytes[..4], b"\x7fELF", "{what}");
    assert_eq!(u16_at(bytes, 16), 2, "{what}: ET_EXEC");
    let machine = if target.arch() == stave_target::Arch::X86_64 { 62 } else { 183 };
    assert_eq!(u16_at(bytes, 18), machine, "{what}: e_machine");
    let (phoff, phnum) = (u64_at(bytes, 32) as usize, u16_at(bytes, 56) as usize);
    for k in 0..phnum {
        let ty = u32_at(bytes, phoff + 56 * k);
        assert!(ty != 2 && ty != 3, "{what}: program header {k} has type {ty}");
    }
}

fn show(r: &Ran) -> String {
    format!(
        "exit {}\n--- stdout\n{}--- stderr\n{}",
        r.exit,
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    )
}

#[test]
fn native_equals_interpreter_on_both_arches() {
    let progs = programs();
    assert!(progs.len() >= 19, "run programs: {progs:?}");
    let tmp = TempDir::new("ivn").unwrap();
    let mut failures = Vec::new();
    let mut native_runs = 0;
    for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        let how = runner(target);
        if how.is_none() {
            eprintln!(
                "note: {target} binaries cannot run here (no native CPU, no qemu-aarch64): interpreter only"
            );
        }
        for p in &progs {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let out = build(p, target, false);
            let (m, img) = (out.module.as_ref().unwrap(), out.image.as_ref().unwrap());
            assert_static(img.bytes(), target, &name);
            let main_path = m.func(out.main.unwrap()).unwrap().path.clone();
            let lim = Limits::default();
            let want = match interp::run(m, &main_path, target, &[], &data_files(p), lim) {
                Ok(r) => {
                    let allowed = listed_numbers(target);
                    for nr in &r.syscalls {
                        if !allowed.contains(nr) {
                            failures.push(format!(
                                "{name} [{target}]: syscall {nr} is not in runtime/syscalls.list"
                            ));
                        }
                    }
                    r.ran
                }
                Err(e) => {
                    failures.push(format!("{name} [{target}]: interpreter: {e}"));
                    continue;
                }
            };
            let Some(runner) = how else { continue };
            for release in [false, true] {
                let bin = tmp.path().join(format!("{name}-{target}-{release}"));
                let rel = release.then(|| build(p, target, true));
                let img = rel.as_ref().map_or(img, |r| r.image.as_ref().unwrap());
                assert_static(img.bytes(), target, &name);
                img.write_file(&bin).unwrap();
                let spec = RunSpec { args: &[], runner, cwd: Some(&base_dir(p)), inherit: false };
                let got = exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
                native_runs += 1;
                let mut held = Executable::new(&name, img.bytes()).unwrap();
                assert!(held.in_memory(), "{name}: {:?}", exec::fallback_note());
                let from_memory = held.run(&spec).unwrap_or_else(|e| panic!("{name} from memory: {e}"));
                if from_memory != got {
                    failures.push(format!(
                        "{name} [{target}{}]: the run from memory differs from the file's\n== memory\n{}== file\n{}",
                        if release { ", release" } else { "" },
                        show(&from_memory),
                        show(&got)
                    ));
                }
                // The interpreter's status is the full `exit_group` argument; the kernel keeps 8 bits.
                let want = Ran { exit: want.exit & 0xff, ..want.clone() };
                if got != want {
                    failures.push(format!(
                        "{name} [{target}{}]: native differs from the interpreter\n== native\n{}== interpreter\n{}",
                        if release { ", release" } else { "" },
                        show(&got),
                        show(&want)
                    ));
                }
            }
        }
    }
    eprintln!("{native_runs} native runs compared");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// The syscall numbers of runtime/<arch>/sys.stv (`pub const NR_*`; stave-obj's static_elf test
/// checks them against runtime/syscalls.list and the kernel ABI).
fn listed_numbers(t: Triple) -> Vec<u64> {
    let arch = if t.arch() == stave_target::Arch::X86_64 { "x86_64" } else { "aarch64" };
    let text = std::fs::read_to_string(repo().join(format!("runtime/{arch}/sys.stv"))).unwrap();
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("pub const NR_")?.split_once(": usize = "))
        .filter_map(|(_, v)| v.trim_end_matches(';').trim().parse().ok())
        .collect()
}

/// System calls of runtime/syscalls.list.
fn syscall_list() -> Vec<String> {
    std::fs::read_to_string(repo().join("runtime/syscalls.list"))
        .unwrap()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect()
}

/// `strace -c` of every x86-64 run program: only syscalls of the list (plus the tracer's own
/// `execve` of the program).
#[test]
fn native_binaries_use_only_listed_syscalls() {
    let x64 = Triple::parse("x86_64-linux").unwrap();
    if !exec::can_run(x64) || !have("strace") {
        eprintln!("note: skipped (needs an x86-64 Linux host with strace)");
        return;
    }
    let list = syscall_list();
    let tmp = TempDir::new("strace").unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for p in programs() {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let out = build(&p, x64, false);
        let bin = tmp.path().join(&name);
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        let log = tmp.path().join(format!("{name}.strace"));
        Command::new("strace")
            .arg("-c")
            .arg("-o")
            .arg(&log)
            .arg(&bin)
            .current_dir(base_dir(&p))
            .output()
            .unwrap();
        let text = std::fs::read_to_string(&log).unwrap();
        // Rows: `% time seconds usecs/call calls [errors] syscall`; the name is the last column.
        for l in text.lines() {
            let l = l.trim();
            if l.is_empty() || l.starts_with('%') || l.starts_with('-') || l.ends_with("total") {
                continue;
            }
            let Some(sc) = l.split_whitespace().last() else { continue };
            if sc == "execve" {
                continue;
            }
            assert!(
                list.iter().any(|x| x == sc),
                "{name}: syscall `{sc}` is not in runtime/syscalls.list\n{text}"
            );
            seen.insert(sc.to_string());
        }
    }
    eprintln!("syscalls used: {seen:?}");
    // `strace -c` never counts `exit_group`: it does not return, so only `write` is required.
    assert!(seen.contains("write"), "write never traced: {seen:?}");
}
