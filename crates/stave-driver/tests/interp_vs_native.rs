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
//! A run that cannot happen here (no qemu-aarch64, not Linux) is skipped with a message, or
//! fails when `STAVE_REQUIRE_LEGS` lists its leg; the interpreter side always runs.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::conformance::stage0::{Stage0, registry_codes};
use stave_driver::conformance::{Config, Outcome, run_suite};
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
                // the interpreter has no stack size: a program that overflows its `[run] stack`
                // runs to its end there
                && !(std::fs::read_to_string(p.join("stave.toml")).is_ok_and(|t| t.contains("[run]"))
                    && std::fs::read_to_string(p.join("src/main.stv"))
                        .is_ok_and(|t| t.contains("//~ RESOURCE-LIMIT")))
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
    assert!(out.image.is_some() && out.module.as_ref().is_some_and(|m| m.main.is_some()));
    out
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// The runner for `target` here: `Some(None)` native, `Some(Some(cmd))` through `cmd`, `None`
/// not runnable.
fn runner(target: Triple) -> Option<Option<&'static str>> {
    let how = if exec::can_run(target) {
        Some(None)
    } else if target == Triple::parse("aarch64-linux").unwrap()
        && cfg!(target_os = "linux")
        && have("qemu-aarch64")
    {
        Some(Some("qemu-aarch64"))
    } else {
        None
    };
    legs::gate(target.name(), how.is_some());
    how
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
            let main_path = m.func(m.main.unwrap()).unwrap().path.clone();
            let lim = Limits::default();
            let want =
                match interp::run(out.for_interp().unwrap(), &main_path, target, &[], &data_files(p), lim) {
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
        .map(|l| String::from(l.split_whitespace().next().unwrap_or(l)))
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

/// Targets whose native vector code is compared with the interpreter on the SIMD programs (as a
/// debug and as a release build). A target joins this list when its backend lowers every vector
/// op.
const SIMD_NATIVE: &[&str] = &["x86_64-linux-v2", "x86_64-linux-v3", "aarch64-linux"];

/// `tests/conformance/simd/run_*.stv`, then the std slice method programs `combinators_*.stv`.
fn simd_programs() -> Vec<PathBuf> {
    let mut v = simd_files("run_");
    v.extend(simd_files("combinators_"));
    v
}

/// `tests/conformance/simd/<prefix>*.stv`, sorted.
fn simd_files(prefix: &str) -> Vec<PathBuf> {
    let dir = repo().join("tests/conformance/simd");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            name.starts_with(prefix) && p.extension().is_some_and(|x| x == "stv")
        })
        .collect();
    v.sort();
    v
}

/// Every file of `tests/conformance/simd/` passes under `stave test`: the negatives report exactly
/// their `//~ ERROR` codes, the others no diagnostic.
#[test]
fn simd_suite_passes() {
    let mut c = Stage0::new(Some(4), Library::in_dir(&repo())).unwrap();
    let cfg = Config { known_codes: Some(registry_codes()), ..Config::default() };
    let r = run_suite(&repo().join("tests/conformance/simd"), &cfg, &mut c).unwrap();
    eprintln!("{}", r.render());
    assert!(r.results.len() >= 10, "suite too small: {}", r.results.len());
    let bad: Vec<String> = r
        .results
        .iter()
        .filter(|t| t.outcome != Outcome::Passed)
        .map(|t| format!("{}: {:?}", t.name, t.outcome))
        .collect();
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The checked IR of `p` for `target` (no code generation).
fn lower_only(p: &Path, target: Triple) -> Output {
    let mut o = Options::new(Goal::Check).threads(Some(4));
    o.lower = true;
    o.target = Some(target);
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{}: diagnostics {:?}", p.display(), out.diagnostics);
    assert!(out.failure.is_none(), "{} ({target}): {}", p.display(), out.failure.as_ref().unwrap());
    assert!(out.module.is_some(), "{}: no module", p.display());
    out
}

/// Every SIMD program runs in the interpreter on both arches: stdout equals `<name>.stdout`, the
/// exit status `<name>.exit` (default 0), and stderr contains `<name>.stderr` when present
/// (`STAVE_BLESS=1` rewrites the stdout files). On the targets of [`SIMD_NATIVE`] the native
/// debug and release builds must also match, and the x86_64-linux builds again under
/// `qemu-x86_64 -cpu Nehalem` (no AVX2) when it is installed.
#[test]
fn simd_programs_run_in_the_interpreter() {
    let progs = simd_programs();
    assert!(progs.len() >= 4, "SIMD run programs: {progs:?}");
    let bless = std::env::var_os("STAVE_BLESS").is_some();
    let tmp = TempDir::new("ivn-simd").unwrap();
    let mut failures = Vec::new();
    let v2 = Triple::parse("x86_64-linux").unwrap();
    let nehalem = cfg!(target_os = "linux") && have("qemu-x86_64");
    if !nehalem {
        eprintln!("note: no qemu-x86_64: the Nehalem runs are skipped");
    }
    for target in ["x86_64-linux", "x86_64-linux-v3", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        for p in &progs {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            let side = |ext: &str| p.with_extension(ext);
            let out = lower_only(p, target);
            let m = out.for_interp().unwrap();
            let r = match interp::run(m, "main.main", target, &[], &[], Limits::default()) {
                Ok(r) => r.ran,
                Err(e) => {
                    failures.push(format!("{name} [{target}]: interpreter: {e}"));
                    continue;
                }
            };
            if bless {
                std::fs::write(side("stdout"), &r.stdout).unwrap();
            }
            let want_out = std::fs::read(side("stdout")).unwrap_or_default();
            let want_exit: i32 =
                std::fs::read_to_string(side("exit")).map_or(0, |s| s.trim().parse().unwrap());
            let want_err = std::fs::read_to_string(side("stderr")).unwrap_or_default();
            let stderr = String::from_utf8_lossy(&r.stderr).into_owned();
            if r.stdout != want_out || r.exit & 0xff != want_exit || !stderr.contains(want_err.trim_end()) {
                failures.push(format!(
                    "{name} [{target}]: interpreter run differs from the expectation\n{}",
                    show(&r)
                ));
            }
            if !SIMD_NATIVE.contains(&target.to_string().as_str()) {
                continue;
            }
            let Some(runner) = runner(target) else { continue };
            // x86-64 without AVX2 as well: the std.slice loops a 256-bit CPU skips
            let mut runners = vec![(runner, "")];
            if target == v2 && nehalem {
                runners.push((Some("qemu-x86_64 -cpu Nehalem"), " Nehalem"));
            }
            let want = Ran { exit: r.exit & 0xff, ..r.clone() };
            for (release, mode) in [(false, "debug"), (true, "release")] {
                let built = build(p, target, release);
                let bin = tmp.path().join(format!("{name}-{target}-{mode}"));
                built.image.as_ref().unwrap().write_file(&bin).unwrap();
                for (runner, how) in &runners {
                    let spec =
                        RunSpec { args: &[], runner: *runner, cwd: Some(&base_dir(p)), inherit: false };
                    let got = exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
                    if got != want {
                        failures.push(format!(
                            "{name} [{target} {mode}{how}]: native differs from the interpreter\n{}",
                            show(&got)
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// A mask loaded from memory with a lane that is neither all zeros nor all ones is undefined
/// behaviour: the interpreter reports a fault at the load.
#[test]
fn malformed_mask_load_is_an_interpreter_fault() {
    let p = repo().join("tests/conformance/simd/ub_mask_bits.stv");
    for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        let out = lower_only(&p, target);
        let m = out.for_interp().unwrap();
        match interp::run(m, "main.main", target, &[], &[], Limits::default()) {
            Err(e) => assert!(e.contains("Fault"), "{target}: {e}"),
            Ok(r) => panic!("{target}: ran to the end\n{}", show(&r.ran)),
        }
    }
}

/// (lane bits, lanes) of vectors that `load_masked` reads lane by lane on some x86-64 level: 128
/// bits or less before AVX, and 8- and 16-bit lanes at every level below the v4 set.
const MASKED_LOAD_SHAPES: &[(u32, u32)] =
    &[(8, 2), (8, 16), (8, 32), (16, 8), (16, 16), (32, 2), (32, 4), (64, 2)];

/// A program that loads through a mask whose lane 0 is all ones, whose last lane is `bad`, and
/// whose other lanes are zero: first from a slice that holds every lane, then from a slice of one
/// element. Element 0 is 7, the others 99.
fn masked_load_source(bits: u32, n: u32, bad: u64) -> String {
    let (t, ones, last) = (format!("u{bits}"), u64::MAX >> (64 - bits), n - 1);
    format!(
        r#"import std.io;

fn show(v: {t}x{n}) {{
    let mut out: [{t}; {n}] = [0; {n}];
    v.store(&mut out, 0);
    for k in 0..out.len() {{
        let x = out[k];
        io.print(f" {{x}}");
    }}
    io.print("\n");
}}

fn main() -> i32 {{
    let mut xs: [{t}; {n}] = [99; {n}];
    xs[0] = 7;
    let mut raw: [{t}; {n}] = [0; {n}];
    raw[0] = {ones};
    raw[{last}] = {bad};
    let p = &raw as *const [{t}; {n}] as *const mask{bits}x{n};
    let m = unsafe("the last lane is neither all zeros nor all ones: undefined behaviour") *p;
    show({t}x{n}.load_masked(&xs, 0, m));
    show({t}x{n}.load_masked(&xs[0..1], 0, m));
    return 0;
}}
"#
    )
}

/// Which bit of an ill-formed mask lane a backend tests is open, but `load_masked` bounds-checks
/// and reads the same lanes, so it never reads past its slice. The interpreter faults at the mask
/// load. A native build takes the lane as active (the load of the short slice traps `Bounds`) or
/// as inactive (neither load reads it).
#[test]
fn malformed_mask_load_masked_reads_only_checked_lanes() {
    struct Case {
        name: String,
        path: PathBuf,
        vector_bits: u32,
        /// The accepted (exit status, stdout) pairs.
        accepted: Vec<(i32, String)>,
    }
    let tmp = TempDir::new("ivn-mask").unwrap();
    // `u32x4`, lane 1 = 1, a slice of one element: the exit status is the sum of the lanes read
    let probe = repo().join("tests/conformance/simd/ub_mask_load_masked.stv");
    let mut cases = vec![Case {
        name: "ub_mask_load_masked".into(),
        path: probe,
        vector_bits: 128,
        accepted: vec![(10, String::new()), (101, String::new())],
    }];
    for &(bits, n) in MASKED_LOAD_SHAPES {
        // the lowest bit alone, and the sign bit alone (the same pattern kind in 8-bit lanes)
        for bad in [1u64, 1 << (bits - 1)] {
            let name = format!("mask{bits}x{n}_{bad:#x}");
            let path = tmp.path().join(format!("{name}.stv"));
            std::fs::write(&path, masked_load_source(bits, n, bad)).unwrap();
            let line = |last: u32| format!(" 7{} {last}\n", " 0".repeat(n as usize - 2));
            let accepted = vec![(101, line(99)), (0, line(0).repeat(2))];
            cases.push(Case { name, path, vector_bits: bits * n, accepted });
        }
    }
    let mut failures = Vec::new();
    for target in ["x86_64-linux", "x86_64-linux-v3", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        let how = runner(target);
        let width = if target.to_string() == "x86_64-linux-v3" { 256 } else { 128 };
        for c in cases.iter().filter(|c| c.vector_bits <= width) {
            let name = &c.name;
            let out = lower_only(&c.path, target);
            match interp::run(out.for_interp().unwrap(), "main.main", target, &[], &[], Limits::default()) {
                Err(e) if e.contains("Fault") => {}
                Err(e) => failures.push(format!("{name} [{target}]: interpreter: {e}")),
                Ok(r) => failures.push(format!("{name} [{target}]: interpreter ran on\n{}", show(&r.ran))),
            }
            let Some(runner) = how else { continue };
            for (release, mode) in [(false, "debug"), (true, "release")] {
                let built = build(&c.path, target, release);
                let bin = tmp.path().join(format!("{name}-{target}-{mode}"));
                built.image.as_ref().unwrap().write_file(&bin).unwrap();
                let spec = RunSpec { args: &[], runner, cwd: Some(tmp.path()), inherit: false };
                let got = exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
                let trap = String::from_utf8_lossy(&got.stderr).starts_with("panic: Bounds ");
                let ok = c.accepted.iter().any(|(exit, stdout)| {
                    got.exit == *exit && got.stdout == stdout.as_bytes() && trap == (*exit == 101)
                });
                if !ok {
                    failures.push(format!("{name} [{target} {mode}]\n{}", show(&got)));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// The check-only SIMD positives lower to verified checked IR on both arches.
#[test]
fn simd_positives_lower() {
    let files = simd_files("pos_");
    assert!(!files.is_empty());
    for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        for p in &files {
            lower_only(p, target);
        }
    }
}
