//! The `stave` command (contracts/cli.md), as a library so tests can drive it in process.
//!
//! Exit status: 0 success, 1 user error (diagnostics, including E1302/E1303/E1304 from
//! `--compiler gpu`), 2 usage error, 3 internal compiler error (always a bug).
//!
//! Commands whose implementation has not landed print "stave: `<cmd>` is not implemented yet
//! (...)" and exit 2: the request cannot be served, like an unknown command, and exit 3 is
//! reserved for bugs. Exit 2 also keeps CI from mistaking a missing feature for a crash.
//!
//! Compiling commands run the stage0 pipeline with the shipped `std` and `runtime`
//! (located by `stave_driver::shipped`: `STAVE_STD_DIR`, `<exe dir>/../lib/stave`, the repository root)
//! and print the diagnostics in canonical order:
//!
//! - `check`: through type and borrow checking.
//! - `build [-o OUT]`: type and borrow checking, then lowering, code generation, and the link.
//!   The static executable goes to `OUT`, default `./<package name>` (a single `.stv` file:
//!   `./<file stem>`); a name that is not a plain file name ([`output_file_name`]) is refused
//!   (exit 2) unless `-o` is given. A program without `fn main` is E0108 (exit 1). A program stage0
//!   cannot compile yet (vectors, closure captures, ...) exits 2 with the reason; any other
//!   back-end failure on a valid program is exit 3. `kind = "lib"` packages are not built yet.
//! - `run [PKG_DIR [ARGS]] [-- ARGS]`: builds like `build`, keeps the executable in memory
//!   (`stave_driver::exec::Executable`: no file is written), then runs it with ARGS and the
//!   inherited stdio; the exit status is the program's (`128 + signal` if killed). A target the
//!   host cannot run is E1001 (exit 1), checked before building.
//! - `test [DIR]`: the conformance runner over DIR with the stage0 test compiler (builds and runs
//!   run tests, natively or through `--runner`; `--target` applies to tests without `TARGET`);
//!   when DIR is a package, also its `test fn`s (stave_driver::harness).
//! - `fix [--dry-run]`: `check` in rounds, applying the `exact` fixes between them ([`fix`]).
//! - `lowering --target T [--op OP]`: the lowering table of the target's tier ([`lowering`]).
//!
//! `--features` beyond the target's baseline and `--cache` are parsed and validated, then refused
//! as not implemented (exit 2): no flag is accepted and ignored.
//!
//! `STAVE_THREADS` sets the worker count; `STAVE_SEMA_SHUFFLE` is stave-sema's test knob (dumps must not
//! change); `STAVE_PROFILE` prints the step profile of a compile on stderr (`stave_driver::prof`).

pub mod args;
pub mod effects_report;
pub mod fix;
pub mod fmt;
pub mod lowering;
pub mod unsafe_report;

use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

use args::{Args, Cmd, DaemonCmd, DiagFormat, Parsed, USAGE};
use stave_driver::build::Failure;
use stave_driver::conformance::{self, Config, stage0};
use stave_driver::dump::{self, DumpError, DumpStage};
use stave_driver::exec::{self, Executable, RunSpec};
use stave_driver::harness::{self, HarnessError};
use stave_driver::package;
use stave_driver::pipeline::{self, Goal, Options, Output, StageTiming};
use stave_driver::prof;
use stave_driver::select::{self, ProbeInput, SelectError, Selection};
use stave_driver::shipped::{self, Library};
use stave_sema::PackageKind;
use stave_target::Triple;

pub const EXIT_OK: i32 = 0;
pub const EXIT_USER: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_ICE: i32 = 3;

/// This binary's build id (16 hex digits, resident-protocol.md): `STAVE_BUILD_ID` at build time.
/// Without one no stage1 image matches (probe step 2 fails).
const BUILD_ID: Option<&str> = option_env!("STAVE_BUILD_ID");

/// Largest accepted `STAVE_THREADS`.
const MAX_THREADS: usize = 4096;

/// Run `stave` with `argv` (without the program name). Never panics on bad input; output errors
/// (e.g. a closed pipe) are ignored.
pub fn run(argv: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    match args::parse(argv) {
        Err(e) => {
            let _ = writeln!(err, "stave: {e}\nrun `stave --help` for usage");
            EXIT_USAGE
        }
        Ok(Parsed::Help) => {
            let _ = out.write_all(USAGE.as_bytes());
            EXIT_OK
        }
        Ok(Parsed::Version) => {
            let _ = writeln!(out, "stave {} (stage0)", env!("CARGO_PKG_VERSION"));
            EXIT_OK
        }
        Ok(Parsed::Args(a)) => dispatch(&a, out, err),
    }
}

pub(crate) fn not_implemented(what: &str, reason: &str, err: &mut dyn Write) -> i32 {
    let _ = writeln!(err, "stave: `{what}` is not implemented yet ({reason})");
    EXIT_USAGE
}

fn dispatch(a: &Args, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    if let Some(command) = a.cmd.compile_command() {
        if a.target.features != a.target.triple.baseline() {
            return not_implemented(
                "--features",
                "stage0 builds use the target's baseline features; choose a level with --target",
                err,
            );
        }
        if a.cache.is_some() {
            return not_implemented("--cache", "stage0 has no incremental cache", err);
        }
        let exe = std::env::current_exe().ok();
        let input = ProbeInput {
            command,
            target: a.target.triple,
            // The manifest has no C-link keys (an unknown key is E1002): no package declares
            // C-link inputs.
            links_c: false,
            exe: exe.as_deref(),
            build_id: BUILD_ID,
        };
        return match select::select(a.compiler, &input) {
            Err(e) => emit_select_error(&e, a.diagnostics, err),
            // No stage1 image exists yet, so the probe never selects stage1.
            Ok(Selection::Stage1(_)) => not_implemented("--compiler gpu (stage1)", "needs stage1", err),
            Ok(selection) => match threads_from_env(std::env::var("STAVE_THREADS").ok().as_deref()) {
                Err(msg) => {
                    let _ = writeln!(err, "stave: {msg}\nrun `stave --help` for usage");
                    EXIT_USAGE
                }
                Ok(threads) if a.cmd == Cmd::Test => test(a, threads, &selection, out, err),
                Ok(threads) => compile(a, threads, &selection, err),
            },
        };
    }
    match (a.cmd, a.daemon) {
        // No stage1 daemon exists yet, so none is running.
        (Cmd::Daemon, Some(DaemonCmd::Status)) => {
            let _ = writeln!(out, "daemon: not running");
            EXIT_OK
        }
        // contracts/cli.md: `daemon stop` with no daemon running prints the status and succeeds.
        (Cmd::Daemon, Some(DaemonCmd::Stop)) => {
            let _ = writeln!(out, "daemon: not running");
            EXIT_OK
        }
        (Cmd::Daemon, _) => not_implemented("daemon evict", "needs the stage1 daemon", err),
        (Cmd::UnsafeReport, _) => unsafe_report::run(a, out, err),
        (Cmd::Effects, _) => effects_report::run(a, out, err),
        (Cmd::Fix, _) => fix::run(a, out, err),
        (Cmd::Fmt, _) => fmt::run(a, out, err),
        (Cmd::Lowering, _) => lowering::run(a, out, err),
        (cmd, _) => not_implemented(cmd.name(), "planned", err),
    }
}

/// `STAVE_THREADS`: unset or empty = the default count; otherwise 1..=4096.
pub fn threads_from_env(v: Option<&str>) -> Result<Option<usize>, String> {
    match v.map(str::trim) {
        None | Some("") => Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(n) if (1..=MAX_THREADS).contains(&n) => Ok(Some(n)),
            _ => Err(format!("STAVE_THREADS must be a thread count 1..={MAX_THREADS}, not `{s}`")),
        },
    }
}

fn pkg_path(a: &Args) -> PathBuf {
    a.pkg_dir.clone().unwrap_or_else(|| PathBuf::from("."))
}

/// The shipped `std` and `runtime`; not found is a usage error (the installation cannot serve
/// the request).
fn library(err: &mut dyn Write) -> Result<Library, i32> {
    shipped::locate().map_err(|e| {
        let _ = writeln!(err, "stave: {e}");
        EXIT_USAGE
    })
}

/// One location-less diagnostic in the `--diagnostics` format; exit 1.
fn emit_unlocated(d: &stave_diag::Diagnostic, format: DiagFormat, err: &mut dyn Write) -> i32 {
    let files = stave_diag::SourceFiles::default();
    let text = match format {
        DiagFormat::Human => stave_diag::emit_human(std::slice::from_ref(d), &files),
        DiagFormat::Json => stave_diag::emit_json(std::slice::from_ref(d), &files),
    };
    match text {
        Ok(t) => {
            let _ = err.write_all(t.as_bytes());
            EXIT_USER
        }
        Err(e) => internal_error(&e.to_string(), err),
    }
}

/// E1001 (no location) in the `--diagnostics` format: `target` cannot run on this host.
fn emit_e1001(target: Triple, format: DiagFormat, err: &mut dyn Write) -> i32 {
    match stave_diag::Diagnostic::new("E1001", None, &[("target", target.name())], None, &[]) {
        Ok(d) => emit_unlocated(&d, format, err),
        Err(e) => internal_error(&format!("E1001: {e}"), err),
    }
}

/// `name` as the default output file name. It must be one plain path component that is not a
/// flag: not empty, `.`, or `..`, no `/`, `\`, or NUL, no leading `-`. The name comes from the
/// manifest (or the source file name), so anything else would let the input choose where the
/// executable is written (`/usr/local/bin/ls`, `../x`).
pub fn output_file_name(name: &str) -> Result<&str, String> {
    let bad = name.is_empty()
        || name == "."
        || name == ".."
        || name.starts_with('-')
        || name.contains(['/', '\\', '\0']);
    if bad {
        Err(format!(
            "`{}` cannot be the output file name (one path component: no `/`, `\\`, or NUL, not \
             `.` or `..`, no leading `-`)",
            name.escape_debug()
        ))
    } else {
        Ok(name)
    }
}

/// The default executable name: the root package's name, or the stem of a single `.stv` file,
/// checked by [`output_file_name`].
fn default_out_name(output: &Output, root: &Path) -> Result<String, String> {
    let (what, name) = match output.program.packages.iter().find(|p| p.is_root) {
        Some(p) if p.kind != PackageKind::SingleFile => ("package name", p.name.clone()),
        _ => (
            "file stem",
            root.file_stem().map_or_else(|| "a.out".to_string(), |s| s.to_string_lossy().into_owned()),
        ),
    };
    output_file_name(&name).map(str::to_string).map_err(|e| format!("{what} {e}"))
}

/// Where `stave build` writes the executable: `-o OUT`, else the default name in the current
/// directory.
fn artifact_path(a: &Args, output: &Output, root: &Path) -> Result<PathBuf, String> {
    if let Some(o) = &a.out {
        return Ok(o.clone());
    }
    default_out_name(output, root)
        .map(PathBuf::from)
        .map_err(|e| format!("{e}; name the executable with -o OUT"))
}

/// Says on `err`, once per `said`, that programs run from temporary files
/// (`stave_driver::exec::fallback_note`).
fn note_fallback(said: &mut bool, err: &mut dyn Write) {
    if !*said && let Some(note) = exec::fallback_note() {
        let _ = writeln!(err, "stave: {note}");
        *said = true;
    }
}

/// Whether this is `stave run` of a file that starts with a shebang line. Such a run prints no
/// summary line: stderr holds only the diagnostics and what the program writes.
fn runs_shebang_script(a: &Args, root: &Path) -> bool {
    use std::io::Read;
    let mut head = [0u8; 2];
    a.cmd == Cmd::Run
        && std::fs::File::open(root).and_then(|mut f| f.read_exact(&mut head)).is_ok()
        && head == *b"#!"
}

/// `stave check`, `stave build`, `stave run`, and the `STAVE_PROFILE` step profile of the run.
fn compile(a: &Args, threads: Option<usize>, selection: &Selection, err: &mut dyn Write) -> i32 {
    prof::start();
    let code = compile_steps(a, threads, selection, err);
    // the pipeline's output and the worker pool are released by now
    prof::mark("release");
    let _ = err.write_all(prof::report().as_bytes());
    code
}

fn compile_steps(a: &Args, threads: Option<usize>, selection: &Selection, err: &mut dyn Write) -> i32 {
    let path = pkg_path(a);
    if let Err(e) = package::check_root(&path) {
        let _ = writeln!(err, "stave: {e}");
        return EXIT_USAGE;
    }
    let triple = a.target.triple;
    if a.watch {
        return not_implemented("build --watch", "planned", err);
    }
    if a.cmd == Cmd::Run && triple != Triple::GpuSm86 && !exec::can_run(triple) {
        return emit_e1001(triple, a.diagnostics, err);
    }
    let lib = match library(err) {
        Ok(l) => l,
        Err(code) => return code,
    };
    let goal = if a.cmd == Cmd::Check { Goal::Check } else { Goal::Build };
    let pool = match pipeline::thread_pool(threads) {
        Ok(p) => p,
        Err(e) => {
            let _ = writeln!(err, "stave: {e}");
            return EXIT_USAGE;
        }
    };
    let start = Instant::now();
    let mut opts = Options::new(goal).threads(threads);
    opts.target = Some(triple);
    opts.release = a.release;
    opts.lower = a.dump.contains(&"ir");
    opts.borrow_dump = a.dump.contains(&"borrow");
    prof::mark("setup");
    let output = pipeline::run_on(&pool, &path, &lib, &opts);

    let text = match a.diagnostics {
        DiagFormat::Human => stave_diag::emit_human(&output.diagnostics, output.files()),
        DiagFormat::Json => stave_diag::emit_json(&output.diagnostics, output.files()),
    };
    match text {
        Ok(t) => {
            let _ = err.write_all(t.as_bytes());
        }
        Err(e) => return internal_error(&e.to_string(), err),
    }
    prof::mark("emit-diagnostics");

    if let Some(dir) = &a.dump_dir {
        let stages: Vec<DumpStage> = a.dump.iter().filter_map(|s| DumpStage::parse(s)).collect();
        match pool.install(|| dump::write_dumps(&output, &stages, dir)) {
            Ok(()) => {}
            Err(DumpError::Locate(e)) => return internal_error(&e.to_string(), err),
            Err(DumpError::NotRun(DumpStage::Ir)) if output.failure.is_some() => {}
            Err(e) => {
                let _ = writeln!(err, "stave: {e}");
                return EXIT_USAGE;
            }
        }
    }

    let lib_pkg = output.program.packages.iter().any(|p| p.is_root && p.kind == PackageKind::Lib);
    let mut program: Option<Executable> = None;
    let code = if output.has_errors() {
        EXIT_USER
    } else if goal == Goal::Build && lib_pkg {
        let _ = writeln!(err, "stave: building a library (`kind = \"lib\"`) is not implemented yet");
        EXIT_USAGE
    } else if let Some(f) = &output.failure {
        match f {
            Failure::Internal { .. } => {
                let _ = internal_error(&f.to_string(), err);
                EXIT_ICE
            }
            _ => {
                let _ = writeln!(err, "stave: {f}");
                EXIT_USAGE
            }
        }
    } else if let Some(img) = &output.image {
        let done = if a.cmd == Cmd::Run {
            // The program is named like the file `build` would write, and refused the same way.
            default_out_name(&output, &path).and_then(|name| {
                let x = Executable::new(&name, img.bytes())
                    .map_err(|e| format!("cannot load the program: {e}"))?;
                program = Some(x);
                Ok(())
            })
        } else {
            artifact_path(a, &output, &path)
                .and_then(|p| img.write_file(&p).map_err(|e| format!("cannot write {}: {e}", p.display())))
        };
        match done {
            Ok(()) => EXIT_OK,
            Err(e) => {
                let _ = writeln!(err, "stave: {e}");
                EXIT_USAGE
            }
        }
    } else {
        EXIT_OK
    };
    prof::mark("write");
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    if a.timings {
        print_timings(&output, total_ms, a.diagnostics, err);
    }
    if !runs_shebang_script(a, &path) {
        for e in output.typeck.iter().flat_map(|t| t.emulated()) {
            let at = match output.files().locate(e.span) {
                Ok(l) => format!("{}:{}:{}", l.path, l.line, l.col),
                Err(_) => "?".to_string(),
            };
            let _ = writeln!(err, "emulated: {at}: {} on {} ({triple})", e.op, e.ty);
        }
        for l in output.stack_lines() {
            let _ = writeln!(err, "{l}");
        }
        let _ = writeln!(err, "{}", select::summary_line(selection, total_ms as u64));
    }
    release(output);
    let Some(mut program) = program else { return code };
    let mut said = false;
    note_fallback(&mut said, err);
    let _ = err.flush();
    let spec = RunSpec { args: &a.run_args, runner: None, cwd: None, inherit: true };
    let ran = program.run(&spec);
    note_fallback(&mut said, err);
    match ran {
        Ok(r) => r.exit,
        Err(e) => {
            let _ = writeln!(err, "stave: cannot run the program: {e}");
            EXIT_USAGE
        }
    }
}

/// Frees a finished run's output off the calling thread: the command does not wait for the
/// millions of small frees (the `stave` process exits meanwhile; an in-process caller gets its
/// memory back when the thread is done).
fn release(output: Output) {
    // a thread that cannot start drops the closure, and with it the output, right here
    let _ = std::thread::Builder::new().name("release".to_string()).spawn(move || drop(output));
}

/// The `compiler` field of `--timings` for a stage0 run.
const COMPILER_STAGE0: &str = "cpu (stage0)";

fn print_timings(output: &Output, total_ms: f64, format: DiagFormat, err: &mut dyn Write) {
    let _ = err.write_all(render_timings(&output.timings, total_ms, format).as_bytes());
}

/// `--timings` text: human lines, or one `{"kind":"timings",...}` JSON line
/// (diagnostics.schema.json `$defs/timings`).
pub fn render_timings(timings: &[StageTiming], total_ms: f64, format: DiagFormat) -> String {
    let mut s = String::new();
    match format {
        DiagFormat::Human => {
            s.push_str(&format!("compiler: {COMPILER_STAGE0}\n"));
            for t in timings {
                s.push_str(&format!(
                    "  {:<10} {:>10.3} ms {:>10} {}\n",
                    t.stage.name(),
                    t.ms,
                    t.items,
                    t.stage.unit()
                ));
            }
            s.push_str(&format!("  {:<10} {:>10.3} ms\n", "total", total_ms));
        }
        DiagFormat::Json => {
            s.push_str("{\"kind\":\"timings\",\"stages\":[");
            for (i, t) in timings.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                let ms = if t.ms.is_finite() { t.ms.max(0.0) } else { 0.0 };
                s.push_str(&format!(
                    "{{\"stage\":\"{}\",\"ms\":{ms:.3},\"items\":{}",
                    t.stage.name(),
                    t.items
                ));
                if let Some(r) = t.rounds {
                    s.push_str(&format!(",\"rounds\":{r}"));
                }
                if let Some(f) = t.fallbacks {
                    s.push_str(&format!(",\"fallbacks\":{f}"));
                }
                s.push('}');
            }
            s.push_str(&format!("],\"compiler\":{}}}\n", stave_diag::json_string(COMPILER_STAGE0)));
        }
    }
    s
}

/// `stave test [DIR]`: the conformance runner over DIR (default `.`) with the stage0 compiler.
fn test(
    a: &Args,
    threads: Option<usize>,
    selection: &Selection,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let root = pkg_path(a);
    let lib = match library(err) {
        Ok(l) => l,
        Err(code) => return code,
    };
    let start = Instant::now();
    let mut compiler = match stage0::Stage0::new(threads, lib.clone()) {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(err, "stave: {e}");
            return EXIT_USAGE;
        }
    };
    let target = a.explicit_target.then_some(a.target.triple);
    let cfg = Config {
        filter: a.filter.clone(),
        runner: a.runner.clone(),
        known_codes: Some(stage0::registry_codes()),
        target,
    };
    let report = match conformance::run_suite(&root, &cfg, &mut compiler) {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(err, "stave: {}: {e}", display(&root));
            return EXIT_USAGE;
        }
    };
    let _ = out.write_all(report.render().as_bytes());
    let mut ok = report.is_success();
    // A package: its `test fn`s too.
    if root.join("stave.toml").is_file() {
        let pool = match pipeline::thread_pool(threads) {
            Ok(p) => p,
            Err(e) => {
                let _ = writeln!(err, "stave: {e}");
                return EXIT_USAGE;
            }
        };
        let mut opts = Options::new(Goal::Check).threads(threads);
        opts.target = Some(a.target.triple);
        opts.release = a.release;
        match harness::run(&pool, &root, &lib, &opts, a.filter.as_deref(), a.runner.as_deref()) {
            Ok(r) => {
                let _ = out.write_all(r.render().as_bytes());
                ok &= r.failed() == 0;
            }
            // The conformance run reported the diagnostics already.
            Err(HarnessError::Diagnostics) => ok = false,
            Err(HarnessError::Failure(f)) => {
                let _ = writeln!(err, "stave: test fns: {f}");
                ok = false;
            }
            Err(HarnessError::Io(e)) => {
                let _ = writeln!(err, "stave: test fns: {e}");
                ok = false;
            }
        }
    }
    note_fallback(&mut false, err);
    let ms = start.elapsed().as_millis() as u64;
    let _ = writeln!(err, "{}", select::summary_line(selection, ms));
    if ok { EXIT_OK } else { EXIT_USER }
}

fn display(p: &Path) -> String {
    p.display().to_string()
}

fn internal_error(msg: &str, err: &mut dyn Write) -> i32 {
    let _ = writeln!(
        err,
        "stave: internal compiler error: {msg}\nnote: this is a bug in Stave (exit status 3), never in your program."
    );
    EXIT_ICE
}

/// E1302/E1303/E1304 in the `--diagnostics` format; exit 1. E1304 is located at a C-link input
/// of `stave.toml`, which no manifest declares, so it never reaches this point.
fn emit_select_error(e: &SelectError, format: DiagFormat, err: &mut dyn Write) -> i32 {
    match e.diagnostic(None) {
        Ok(d) => emit_unlocated(&d, format, err),
        Err(x) => internal_error(&format!("{}: {x}", e.code), err),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s
    } else {
        "unknown panic payload"
    }
}

/// Run `f`; any panic becomes exit 3 with a bug-report hint on stderr.
pub fn guarded(f: impl FnOnce() -> i32) -> i32 {
    panic::set_hook(Box::new(|info| {
        let loc = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
        let _ = writeln!(
            std::io::stderr(),
            "stave: internal compiler error: {}{loc}\n\
             note: this is a bug in Stave (exit status 3), never in your program. Please report it with \
             the command line, the `stave --version` output, and the input files.",
            panic_message(info.payload())
        );
    }));
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or(EXIT_ICE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stave(line: &str) -> (i32, String, String) {
        let argv: Vec<String> = line.split_whitespace().map(String::from).collect();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run(&argv, &mut out, &mut err);
        (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
    }

    #[test]
    fn exit_codes() {
        assert_eq!(stave("").0, EXIT_USAGE);
        let (c, out, _) = stave("--help");
        assert_eq!(c, EXIT_OK);
        assert!(out.starts_with("usage: stave"));
        let (c, _, err) = stave("build --bogus");
        assert_eq!(c, EXIT_USAGE);
        assert_eq!(err, "stave: unknown flag `--bogus`\nrun `stave --help` for usage\n");
        // No package here (the crate directory has no stave.toml): a usage error.
        let (c, _, err) = stave("build /nonexistent/stave-pkg");
        assert_eq!(c, EXIT_USAGE);
        assert!(err.starts_with("stave: /nonexistent/stave-pkg: "), "{err}");
        let (c, _, err) = stave("check .");
        assert_eq!(c, EXIT_USAGE);
        assert!(err.contains("not a package (no stave.toml)"), "{err}");
        let (c, _, err) = stave("test /nonexistent/stave-tests --compiler cpu");
        assert_eq!(c, EXIT_USAGE);
        assert!(err.starts_with("stave: /nonexistent/stave-tests: "), "{err}");
        assert_eq!(stave("explain E0201").0, EXIT_USAGE);
        assert_eq!(stave("daemon stop"), (EXIT_OK, "daemon: not running\n".to_string(), String::new()));
        assert_eq!(stave("daemon status"), (EXIT_OK, "daemon: not running\n".to_string(), String::new()));
    }

    #[test]
    fn compiler_gpu_reports_e1302() {
        let (c, out, err) = stave("build --compiler gpu");
        assert_eq!(c, EXIT_USER);
        assert!(out.is_empty());
        assert!(
            err.starts_with("error[E1302]: `--compiler gpu`: no usable GPU (stage1 image missing)\n  fix: "),
            "{err}"
        );
        let (c, _, err) = stave("check --compiler gpu --diagnostics json");
        assert_eq!(c, EXIT_USER);
        assert!(
            err.starts_with(r#"{"kind":"diagnostic","code":"E1302","severity":"error","message":"`--compiler gpu`: no usable GPU (stage1 image missing)","related":[],"fix":{"description":"use `--compiler auto`"#),
            "{err}"
        );
        assert!(err.ends_with("\"edits\":[]}}\n") && err.lines().count() == 1, "{err}");
        // Step 1 comes first for code-generating commands; `check` is not affected.
        let (c, _, err) = stave("build --compiler gpu --target aarch64-linux");
        assert_eq!(c, EXIT_USER);
        assert!(err.starts_with("error[E1303]: "), "{err}");
        assert!(stave("check --compiler gpu --target aarch64-linux").2.starts_with("error[E1302]: "));
        // auto falls back to stage0 (no package in the crate directory: usage error).
        assert_eq!(stave("build --compiler auto").0, EXIT_USAGE);
        // Commands that always run in stage0 ignore --compiler gpu.
        assert_eq!(stave("explain E0201 --compiler gpu").0, EXIT_USAGE);
    }

    #[test]
    fn stave_threads() {
        assert_eq!(threads_from_env(None), Ok(None));
        assert_eq!(threads_from_env(Some("")), Ok(None));
        assert_eq!(threads_from_env(Some("12")), Ok(Some(12)));
        assert!(threads_from_env(Some("0")).is_err());
        assert!(threads_from_env(Some("x")).is_err());
        assert!(threads_from_env(Some("5000")).is_err());
    }

    #[test]
    fn timings_text() {
        use stave_driver::pipeline::Stage;
        let t = [
            StageTiming { stage: Stage::Read, ms: 1.5, items: 10, rounds: None, fallbacks: None },
            StageTiming { stage: Stage::Items, ms: 0.25, items: 3, rounds: Some(2), fallbacks: Some(0) },
        ];
        assert_eq!(
            render_timings(&t, 2.0, DiagFormat::Json),
            "{\"kind\":\"timings\",\"stages\":[{\"stage\":\"read\",\"ms\":1.500,\"items\":10},\
             {\"stage\":\"items\",\"ms\":0.250,\"items\":3,\"rounds\":2,\"fallbacks\":0}],\
             \"compiler\":\"cpu (stage0)\"}\n"
        );
        let h = render_timings(&t, 2.0, DiagFormat::Human);
        assert!(
            h.starts_with("compiler: cpu (stage0)\n  read            1.500 ms         10 lines\n"),
            "{h}"
        );
        assert!(h.ends_with("  total           2.000 ms\n"), "{h}");
    }

    #[test]
    fn output_names() {
        for ok in ["app", "a.out", "my-app", "x_1", ".hidden", "é"] {
            assert_eq!(output_file_name(ok), Ok(ok));
        }
        for bad in ["", ".", "..", "../x", "/home/u/.local/bin/ls", "a/b", "a\\b", "a\0b", "-o", "--help"] {
            assert!(output_file_name(bad).is_err(), "{bad:?}");
        }
        assert!(output_file_name("a\0b").unwrap_err().contains("`a\\0b`"));
    }

    #[test]
    fn unimplemented_flags_are_refused() {
        // Refused before the package is looked at (the crate directory has none).
        let feature = if cfg!(target_arch = "aarch64") { "sve" } else { "avx2" };
        for cmd in ["build", "run", "check", "test"] {
            let (c, _, err) = stave(&format!("{cmd} --features {feature}"));
            assert_eq!(c, EXIT_USAGE, "{cmd}: {err}");
            assert!(err.starts_with("stave: `--features` is not implemented yet ("), "{cmd}: {err}");
            let (c, _, err) = stave(&format!("{cmd} --cache c"));
            assert_eq!(c, EXIT_USAGE, "{cmd}: {err}");
            assert!(err.starts_with("stave: `--cache` is not implemented yet ("), "{cmd}: {err}");
        }
    }

    #[test]
    fn panic_is_exit_3() {
        assert_eq!(guarded(|| panic!("boom")), EXIT_ICE);
        assert_eq!(guarded(|| 0), EXIT_OK);
        assert_eq!(guarded(|| 1), EXIT_USER);
    }
}
