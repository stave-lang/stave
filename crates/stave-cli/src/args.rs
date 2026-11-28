//! Hand-written argument parser for contracts/cli.md (no external parser crate).
//!
//! Form: `stave <command> [args...]`. Flags come after the command. A value flag takes its value
//! as `--flag=value` or as the next argument (`-o OUT` only in the second form). Every error is
//! a [`UsageError`] (exit 2).

use std::fmt;
use std::path::PathBuf;

use stave_driver::select::{CompileCommand, CompilerChoice};
use stave_target::{Target, Triple};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    Build,
    Run,
    Check,
    Test,
    Fix,
    UnsafeReport,
    Effects,
    Loops,
    Dup,
    Fmt,
    Lowering,
    Explain,
    Daemon,
    Ir,
}

impl Cmd {
    pub fn name(self) -> &'static str {
        match self {
            Cmd::Build => "build",
            Cmd::Run => "run",
            Cmd::Check => "check",
            Cmd::Test => "test",
            Cmd::Fix => "fix",
            Cmd::UnsafeReport => "unsafe-report",
            Cmd::Effects => "effects",
            Cmd::Loops => "loops",
            Cmd::Dup => "dup",
            Cmd::Fmt => "fmt",
            Cmd::Lowering => "lowering",
            Cmd::Explain => "explain",
            Cmd::Daemon => "daemon",
            Cmd::Ir => "ir",
        }
    }

    fn parse(s: &str) -> Option<Cmd> {
        [
            Cmd::Build,
            Cmd::Run,
            Cmd::Check,
            Cmd::Test,
            Cmd::Fix,
            Cmd::UnsafeReport,
            Cmd::Effects,
            Cmd::Loops,
            Cmd::Dup,
            Cmd::Fmt,
            Cmd::Lowering,
            Cmd::Explain,
            Cmd::Daemon,
            Cmd::Ir,
        ]
        .into_iter()
        .find(|c| c.name() == s)
    }

    /// The commands that compile a program (and print a summary line).
    pub fn compile_command(self) -> Option<CompileCommand> {
        match self {
            Cmd::Build => Some(CompileCommand::Build),
            Cmd::Run => Some(CompileCommand::Run),
            Cmd::Check => Some(CompileCommand::Check),
            Cmd::Test => Some(CompileCommand::Test),
            // Only `stave ir emit` compiles a program; the other verbs read IR text.
            Cmd::Ir => Some(CompileCommand::Build),
            _ => None,
        }
    }
}

/// The verbs of `stave ir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrVerb {
    Check,
    Fmt,
    Opt,
    Run,
    Build,
    Emit,
}

impl IrVerb {
    pub fn name(self) -> &'static str {
        match self {
            IrVerb::Check => "check",
            IrVerb::Fmt => "fmt",
            IrVerb::Opt => "opt",
            IrVerb::Run => "run",
            IrVerb::Build => "build",
            IrVerb::Emit => "emit",
        }
    }

    fn parse(s: &str) -> Option<IrVerb> {
        [IrVerb::Check, IrVerb::Fmt, IrVerb::Opt, IrVerb::Run, IrVerb::Build, IrVerb::Emit]
            .into_iter()
            .find(|v| v.name() == s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DaemonCmd {
    Status,
    Stop,
    Evict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DiagFormat {
    #[default]
    Human,
    Json,
}

/// `--dump` stages, in contract order (stage-gates.md).
pub const DUMP_STAGES: [&str; 14] = [
    "tokens", "items", "ast", "strings", "names", "sigs", "kinds", "types", "borrow", "lir", "ir", "opt",
    "diags", "stats",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Args {
    pub cmd: Cmd,
    /// `stave ir`: the verb.
    pub ir: Option<IrVerb>,
    /// `PKG_DIR` (default: the current directory); `stave ir`: the FILE, or emit's `PKG_DIR`.
    pub pkg_dir: Option<PathBuf>,
    pub release: bool,
    /// Target and features, validated (baseline ∪ `--features`; default: host baseline).
    pub target: Target,
    /// `--target` was given.
    pub explicit_target: bool,
    pub out: Option<PathBuf>,
    pub compiler: CompilerChoice,
    /// Distinct stages in the order given.
    pub dump: Vec<&'static str>,
    pub dump_dir: Option<PathBuf>,
    pub diagnostics: DiagFormat,
    pub timings: bool,
    /// `--cache DIR` (refused as not implemented by every command).
    pub cache: Option<PathBuf>,
    pub runner: Option<String>,
    pub filter: Option<String>,
    /// `test --exact`: `--filter` is a whole test name, not a substring.
    pub exact: bool,
    /// `test --input N`: only input N of the one selected `test fn`, which has parameters.
    pub input: Option<usize>,
    pub watch: bool,
    pub alloc_check: bool,
    /// `unsafe-report --json`, `effects --json`, `loops --json`, `dup --json`.
    pub json: bool,
    /// `dup --all`: also the classes inside the built-in packages. `loops --all`: the loops of
    /// every package.
    pub all: bool,
    /// `dup --min-size N`.
    pub min_size: Option<u32>,
    /// `dup --baseline FILE`: the gate.
    pub baseline: Option<PathBuf>,
    /// `dup --write-baseline FILE`.
    pub write_baseline: Option<PathBuf>,
    /// `fix --dry-run`.
    pub dry_run: bool,
    /// `fmt` paths: files, and directories standing for every `.stv` file below them.
    pub paths: Vec<PathBuf>,
    /// `fmt --check`.
    pub check: bool,
    /// `lowering --op OP`.
    pub op: Option<String>,
    /// `explain EXXXX`.
    pub code: Option<String>,
    pub daemon: Option<DaemonCmd>,
    /// `run PKG_DIR ARGS... -- ARGS...`: the program's arguments.
    pub run_args: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parsed {
    Help,
    Version,
    Args(Box<Args>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageError(pub String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, UsageError> {
    Err(UsageError(msg.into()))
}

pub const USAGE: &str = "\
usage: stave <command> [args]

commands:
  build [PKG_DIR] [--release] [--target T] [--features F,...] [-o OUT] [--watch]
                        (default OUT: ./<package name>)
  run [PKG_DIR [ARGS...]] [--release] [--target T] [-- ARGS...]
                        (host targets only, else E1001; ARGS go to the program,
                        those starting with `-` only after `--`)
  check [PKG_DIR]
  test [PKG_DIR] [--release] [--target T] [--filter S [--exact]] [--runner CMD] [--input N]
                        (--filter S: tests whose name contains S; --exact: named S;
                        --input N: only input N of the one selected test fn)
  fix [PKG_DIR] [--dry-run] [--target T] [--diagnostics human|json]
                        (apply the exact fixes, re-check, repeat; --dry-run prints a diff)
  unsafe-report [PKG_DIR] [--json]
  effects [PKG_DIR] [--json]
                        (the effects of each fn of the program's own packages)
  loops [PKG_DIR] [--json] [--all]
                        (the loops of the root package and their classes; --all: of every
                        package)
  dup [PKG_DIR] [--target T] [--all] [--min-size N] [--json]
      [--baseline FILE | --write-baseline FILE]
                        (classes of fns with the same code, and re-declared consts;
                        --baseline: exit 1 on a class the baseline does not hold)
  fmt [PKG_DIR | FILE...] [--check]
                        rewrite sources in the canonical layout (a directory: every
                        .stv file below it; --check: list the files that would change)
  lowering --target T [--op OP]
  explain EXXXX
  daemon status|stop|evict [PKG_DIR]
  ir check|fmt|opt|run|build FILE [-o OUT] [-- ARGS...]
  ir emit [PKG_DIR] [--release] [--target T] [-o FILE]
                        (IR text: contracts/checked-ir.md; `emit` prints a program's IR text with
                        its stack section, the other verbs load, check, reprint, optimize,
                        interpret, or build a file; -o: `build` and `emit` only)

flags of build, run, check, test:
  --target T            x86_64-linux[-v2|-v3|-v4], aarch64-linux[+sve|+sve2], gpu-sm86
                        (default: the host baseline, never auto-upgraded)
  --features F,...      add features to the target baseline (not implemented yet)
  --compiler auto|cpu|gpu
  --dump=STAGE,... --dump-dir DIR
                        stages: tokens items ast strings names sigs kinds types
                        borrow lir ir opt diags stats (opt: with --release only)
  --diagnostics human|json
  --timings             per-stage times on stderr
  --cache DIR           incremental cache (not implemented yet)
  --alloc-check         device allocator checks (gpu-sm86 only)

exit status: 0 ok, 1 diagnostics, 2 usage error, 3 internal compiler error (a bug)
";

/// Which flags a command accepts.
fn allowed(cmd: Cmd, flag: &str) -> bool {
    const COMPILE: [&str; 10] = [
        "--release",
        "--target",
        "--features",
        "--compiler",
        "--dump",
        "--dump-dir",
        "--diagnostics",
        "--timings",
        "--cache",
        "--alloc-check",
    ];
    match cmd {
        Cmd::Build => COMPILE.contains(&flag) || matches!(flag, "-o" | "--watch"),
        Cmd::Run | Cmd::Check => COMPILE.contains(&flag),
        Cmd::Test => {
            COMPILE.contains(&flag) || matches!(flag, "--filter" | "--exact" | "--runner" | "--input")
        }
        // Always stage0: the check of a round reads edits that are not written yet.
        Cmd::Fix => matches!(flag, "--dry-run" | "--target" | "--diagnostics"),
        // These always run in stage0 whatever --compiler says (cli-stage1.md).
        Cmd::UnsafeReport | Cmd::Effects => matches!(flag, "--json" | "--compiler"),
        Cmd::Loops => matches!(flag, "--json" | "--all"),
        Cmd::Dup => matches!(
            flag,
            "--json" | "--compiler" | "--target" | "--all" | "--min-size" | "--baseline" | "--write-baseline"
        ),
        Cmd::Fmt => flag == "--check",
        Cmd::Lowering => matches!(flag, "--target" | "--op" | "--compiler"),
        Cmd::Explain => flag == "--compiler",
        Cmd::Daemon => false,
        Cmd::Ir => matches!(flag, "--release" | "--target" | "-o"),
    }
}

fn takes_value(flag: &str) -> bool {
    matches!(
        flag,
        "--target"
            | "--features"
            | "-o"
            | "--compiler"
            | "--dump"
            | "--dump-dir"
            | "--diagnostics"
            | "--cache"
            | "--runner"
            | "--filter"
            | "--input"
            | "--op"
            | "--min-size"
            | "--baseline"
            | "--write-baseline"
    )
}

const KNOWN_FLAGS: [&str; 25] = [
    "--release",
    "--target",
    "--features",
    "-o",
    "--compiler",
    "--dump",
    "--dump-dir",
    "--diagnostics",
    "--timings",
    "--cache",
    "--runner",
    "--filter",
    "--exact",
    "--input",
    "--watch",
    "--alloc-check",
    "--json",
    "--dry-run",
    "--check",
    "--op",
    "--help",
    "--all",
    "--min-size",
    "--baseline",
    "--write-baseline",
];

fn set_once<T>(slot: &mut Option<T>, v: T, flag: &str) -> Result<(), UsageError> {
    if slot.is_some() {
        return err(format!("{flag} given twice"));
    }
    *slot = Some(v);
    Ok(())
}

fn set_flag(slot: &mut bool, flag: &str) -> Result<(), UsageError> {
    if *slot {
        return err(format!("{flag} given twice"));
    }
    *slot = true;
    Ok(())
}

/// Parse the arguments after the program name.
pub fn parse(argv: &[String]) -> Result<Parsed, UsageError> {
    let Some(first) = argv.first() else {
        return err("no command given");
    };
    match first.as_str() {
        "-h" | "--help" | "help" => return Ok(Parsed::Help),
        "-V" | "--version" => return Ok(Parsed::Version),
        _ => {}
    }
    let Some(cmd) = Cmd::parse(first) else {
        return if first.starts_with('-') {
            err(format!("expected a command before `{first}`"))
        } else {
            err(format!("unknown command `{first}`"))
        };
    };

    let mut positional: Vec<String> = Vec::new();
    let mut run_args = Vec::new();
    let mut target: Option<String> = None;
    let mut features: Vec<String> = Vec::new();
    let (mut out, mut compiler, mut dump, mut dump_dir, mut diagnostics, mut cache) =
        (None, None, None, None, None, None);
    let (mut runner, mut filter, mut op) = (None, None, None);
    let mut input: Option<usize> = None;
    let (mut release, mut timings, mut watch, mut alloc_check, mut json, mut dry_run, mut check) =
        (false, false, false, false, false, false, false);
    let mut exact = false;
    let mut all = false;
    let mut min_size: Option<u32> = None;
    let (mut baseline, mut write_baseline): (Option<PathBuf>, Option<PathBuf>) = (None, None);

    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].as_str();
        i += 1;
        if arg == "--" {
            if cmd != Cmd::Run && cmd != Cmd::Ir {
                return err(format!("`--` is only accepted by `stave run`, not `stave {}`", cmd.name()));
            }
            run_args = argv[i..].to_vec();
            break;
        }
        if arg == "-h" || arg == "--help" {
            return Ok(Parsed::Help);
        }
        if !arg.starts_with('-') || arg == "-" {
            positional.push(arg.to_string());
            continue;
        }
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v)),
            _ => (arg, None),
        };
        if !KNOWN_FLAGS.contains(&flag) {
            return err(format!("unknown flag `{flag}`"));
        }
        if !allowed(cmd, flag) {
            return err(format!("`stave {}` does not take {flag}", cmd.name()));
        }
        let value = if takes_value(flag) {
            let v = match inline {
                Some(v) => v.to_string(),
                None => match argv.get(i) {
                    Some(v) => {
                        i += 1;
                        v.clone()
                    }
                    None => return err(format!("{flag} needs a value")),
                },
            };
            if v.is_empty() {
                return err(format!("{flag} needs a non-empty value"));
            }
            v
        } else {
            if inline.is_some() {
                return err(format!("{flag} takes no value"));
            }
            String::new()
        };
        match flag {
            "--release" => set_flag(&mut release, flag)?,
            "--timings" => set_flag(&mut timings, flag)?,
            "--watch" => set_flag(&mut watch, flag)?,
            "--alloc-check" => set_flag(&mut alloc_check, flag)?,
            "--json" => set_flag(&mut json, flag)?,
            "--dry-run" => set_flag(&mut dry_run, flag)?,
            "--check" => set_flag(&mut check, flag)?,
            "--exact" => set_flag(&mut exact, flag)?,
            "--all" => set_flag(&mut all, flag)?,
            "--min-size" => {
                let Ok(n) = value.parse::<u32>() else {
                    return err(format!("--min-size takes a number, not `{value}`"));
                };
                set_once(&mut min_size, n, flag)?;
            }
            "--baseline" => set_once(&mut baseline, PathBuf::from(value), flag)?,
            "--write-baseline" => set_once(&mut write_baseline, PathBuf::from(value), flag)?,
            "--target" => set_once(&mut target, value, flag)?,
            // A list: repeated --features accumulate.
            "--features" => features.push(value),
            "-o" => set_once(&mut out, PathBuf::from(value), flag)?,
            "--compiler" => {
                let c = CompilerChoice::parse(&value).ok_or_else(|| {
                    UsageError(format!("--compiler takes auto, cpu, or gpu, not `{value}`"))
                })?;
                set_once(&mut compiler, c, flag)?;
            }
            "--dump" => set_once(&mut dump, value, flag)?,
            "--dump-dir" => set_once(&mut dump_dir, PathBuf::from(value), flag)?,
            "--diagnostics" => {
                let d = match value.as_str() {
                    "human" => DiagFormat::Human,
                    "json" => DiagFormat::Json,
                    _ => return err(format!("--diagnostics takes human or json, not `{value}`")),
                };
                set_once(&mut diagnostics, d, flag)?;
            }
            "--cache" => set_once(&mut cache, PathBuf::from(value), flag)?,
            "--runner" => set_once(&mut runner, value, flag)?,
            "--filter" => set_once(&mut filter, value, flag)?,
            "--input" => {
                let Ok(n) = value.parse::<usize>() else {
                    return err(format!("--input takes an input number, not `{value}`"));
                };
                set_once(&mut input, n, flag)?;
            }
            "--op" => set_once(&mut op, value, flag)?,
            _ => return err(format!("unknown flag `{flag}`")),
        }
    }

    // Positional arguments.
    let mut pkg_dir = None;
    let mut ir = None;
    let mut code = None;
    let mut daemon = None;
    let mut paths = Vec::new();
    let mut pos = positional.into_iter();
    match cmd {
        Cmd::Fmt => paths.extend(pos.by_ref().map(PathBuf::from)),
        Cmd::Explain => {
            let Some(c) = pos.next() else { return err("`stave explain` needs a diagnostic code EXXXX") };
            let digits = c.strip_prefix('E').unwrap_or("");
            let well_formed = digits.len() == 4 && digits.bytes().all(|b| b.is_ascii_digit());
            if !well_formed {
                return err(format!("`{c}` is not a diagnostic code EXXXX"));
            }
            code = Some(c);
        }
        Cmd::Daemon => {
            let Some(sub) = pos.next() else { return err("`stave daemon` needs status, stop, or evict") };
            let d = match sub.as_str() {
                "status" => DaemonCmd::Status,
                "stop" => DaemonCmd::Stop,
                "evict" => DaemonCmd::Evict,
                _ => return err(format!("`stave daemon` takes status, stop, or evict, not `{sub}`")),
            };
            if d == DaemonCmd::Evict {
                pkg_dir = pos.next().map(PathBuf::from);
            }
            daemon = Some(d);
        }
        Cmd::Lowering => {}
        Cmd::Ir => {
            let Some(v) = pos.next() else {
                return err("`stave ir` needs a verb: check, fmt, opt, run, build, or emit");
            };
            let Some(verb) = IrVerb::parse(&v) else {
                return err(format!("`stave ir` takes check, fmt, opt, run, build, or emit, not `{v}`"));
            };
            pkg_dir = pos.next().map(PathBuf::from);
            if pkg_dir.is_none() && verb != IrVerb::Emit {
                return err(format!("`stave ir {v}` needs a FILE"));
            }
            if !run_args.is_empty() && verb != IrVerb::Run {
                return err(format!("`--` is only accepted by `stave ir run`, not `stave ir {v}`"));
            }
            if verb != IrVerb::Emit && (release || target.is_some()) {
                return err(format!(
                    "`stave ir {v}` takes the target and mode from the file: no --release or --target"
                ));
            }
            if out.is_some() && !matches!(verb, IrVerb::Build | IrVerb::Emit) {
                return err(format!("`stave ir {v}` does not take -o"));
            }
            ir = Some(verb);
        }
        // `stave run SCRIPT a b` is what the kernel runs for `./SCRIPT a b` when SCRIPT starts
        // with `#!/usr/bin/env -S stave run`: the words after the root are the program's.
        Cmd::Run => {
            pkg_dir = pos.next().map(PathBuf::from);
            let after_dashes = std::mem::replace(&mut run_args, pos.by_ref().collect());
            run_args.extend(after_dashes);
        }
        _ => pkg_dir = pos.next().map(PathBuf::from),
    }
    if let Some(extra) = pos.next() {
        return err(format!("unexpected argument `{extra}`"));
    }

    // Cross-flag rules.
    let feature_refs: Vec<&str> = features.iter().map(String::as_str).collect();
    let tgt = Target::from_cli(target.as_deref(), &feature_refs).map_err(|e| UsageError(e.to_string()))?;
    if cmd == Cmd::Lowering && target.is_none() {
        return err("`stave lowering` needs --target T");
    }
    if alloc_check && tgt.triple != Triple::GpuSm86 {
        return err(format!("--alloc-check needs --target gpu-sm86 (target is {})", tgt.triple));
    }
    let dump = match dump {
        Some(list) => {
            let mut stages: Vec<&'static str> = Vec::new();
            for s in list.split(',') {
                let Some(stage) = DUMP_STAGES.iter().copied().find(|d| *d == s) else {
                    return err(format!("unknown --dump stage `{s}` (stages: {})", DUMP_STAGES.join(" ")));
                };
                if !stages.contains(&stage) {
                    stages.push(stage);
                }
            }
            stages
        }
        None => Vec::new(),
    };
    if cmd == Cmd::Test && !dump.is_empty() {
        return err("`stave test` does not take --dump");
    }
    if dump.contains(&"opt") && !release {
        return err("--dump=opt needs --release");
    }
    if !dump.is_empty() && dump_dir.is_none() {
        return err("--dump needs --dump-dir DIR");
    }
    if dump.is_empty() && dump_dir.is_some() {
        return err("--dump-dir needs --dump=STAGE,...");
    }
    if exact && filter.is_none() {
        return err("--exact needs --filter NAME");
    }
    if [json, baseline.is_some(), write_baseline.is_some()].iter().filter(|x| **x).count() > 1 {
        return err("--json, --baseline, and --write-baseline exclude each other");
    }

    Ok(Parsed::Args(Box::new(Args {
        cmd,
        ir,
        pkg_dir,
        release,
        target: tgt,
        explicit_target: target.is_some(),
        out,
        compiler: compiler.unwrap_or_default(),
        dump,
        dump_dir,
        diagnostics: diagnostics.unwrap_or_default(),
        timings,
        cache,
        runner,
        filter,
        exact,
        input,
        watch,
        alloc_check,
        json,
        all,
        min_size,
        baseline,
        write_baseline,
        dry_run,
        paths,
        check,
        op,
        code,
        daemon,
        run_args,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Parsed, UsageError> {
        let v: Vec<String> = s.split_whitespace().map(String::from).collect();
        parse(&v)
    }

    fn args(s: &str) -> Args {
        match p(s) {
            Ok(Parsed::Args(a)) => *a,
            other => panic!("{s}: {other:?}"),
        }
    }

    fn usage(s: &str) -> String {
        match p(s) {
            Err(UsageError(m)) => m,
            other => panic!("{s}: expected a usage error, got {other:?}"),
        }
    }

    #[test]
    fn ir_verbs() {
        let a = args("ir emit app --release --target aarch64-linux -o out.ir");
        assert_eq!((a.cmd, a.ir), (Cmd::Ir, Some(IrVerb::Emit)));
        assert_eq!(a.pkg_dir, Some(PathBuf::from("app")));
        assert!(a.release && a.out == Some(PathBuf::from("out.ir")));
        assert_eq!(args("ir emit").pkg_dir, None);
        for v in ["check", "fmt", "opt", "run", "build"] {
            let a = args(&format!("ir {v} f.ir"));
            assert_eq!(a.ir.map(IrVerb::name), Some(v));
            assert_eq!(a.pkg_dir, Some(PathBuf::from("f.ir")));
            assert!(usage(&format!("ir {v}")).contains("needs a FILE"));
            assert!(usage(&format!("ir {v} f.ir --release")).contains("from the file"));
            assert!(usage(&format!("ir {v} f.ir --target aarch64-linux")).contains("from the file"));
        }
        assert_eq!(args("ir run f.ir -- a -b").run_args, ["a", "-b"]);
        assert!(usage("ir check f.ir -- a").contains("only accepted by `stave ir run`"));
        assert_eq!(args("ir build f.ir -o x").out, Some(PathBuf::from("x")));
        assert!(usage("ir check f.ir -o x").contains("does not take -o"));
        assert!(usage("ir").contains("needs a verb"));
        assert!(usage("ir show f.ir").contains("not `show`"));
        assert!(usage("ir emit a b").contains("unexpected argument"));
        assert!(usage("ir emit --dump=ir").contains("does not take --dump"));
    }

    #[test]
    fn commands_and_defaults() {
        let a = args("build");
        assert_eq!(a.cmd, Cmd::Build);
        assert_eq!(a.pkg_dir, None);
        assert_eq!(a.compiler, CompilerChoice::Auto);
        assert_eq!(a.diagnostics, DiagFormat::Human);
        assert_eq!(a.target, Target::host().unwrap());
        assert!(!a.explicit_target);
        for c in ["build", "run", "check", "test", "unsafe-report", "effects"] {
            assert_eq!(args(c).cmd.name(), c);
        }
        assert_eq!(args("check pkg").pkg_dir, Some(PathBuf::from("pkg")));
        assert_eq!(p("--help"), Ok(Parsed::Help));
        assert_eq!(p("help"), Ok(Parsed::Help));
        assert_eq!(p("build --help"), Ok(Parsed::Help));
        assert_eq!(p("--version"), Ok(Parsed::Version));
    }

    #[test]
    fn build_flags() {
        let a = args(
            "build app --release --target x86_64-linux-v3 --features avx512f -o out/app --compiler=cpu --dump=tokens,ast,tokens --dump-dir d --diagnostics json --timings --cache c --watch",
        );
        assert_eq!(a.pkg_dir, Some(PathBuf::from("app")));
        assert!(a.release && a.timings && a.watch && a.explicit_target);
        assert_eq!(a.target.triple, Triple::parse("x86_64-linux-v3").unwrap());
        assert!(a.target.features.contains(stave_target::Feature::Avx512f));
        assert_eq!(a.out, Some(PathBuf::from("out/app")));
        assert_eq!(a.compiler, CompilerChoice::Cpu);
        assert_eq!(a.dump, ["tokens", "ast"]);
        assert_eq!(a.dump_dir, Some(PathBuf::from("d")));
        assert_eq!(a.diagnostics, DiagFormat::Json);
        assert_eq!(a.cache, Some(PathBuf::from("c")));
        let a = args("build --features avx2 --features=fma,bmi2 --target=x86_64-linux");
        assert_eq!(a.target.features.to_string(), "sse42,avx,avx2,fma,bmi2");
    }

    #[test]
    fn test_run_and_others() {
        let a = args("test t --filter lex --runner qemu-aarch64 --target aarch64-linux+sve --input 7");
        assert_eq!((a.filter.as_deref(), a.runner.as_deref()), (Some("lex"), Some("qemu-aarch64")));
        assert_eq!((a.input, a.exact), (Some(7), false));
        let a = args("test t --filter main.lex --exact --input 0");
        assert_eq!((a.filter.as_deref(), a.exact, a.input), (Some("main.lex"), true, Some(0)));
        let v: Vec<String> = ["run", "--release", "--", "-x", "--target", "y"].map(String::from).to_vec();
        let Ok(Parsed::Args(a)) = parse(&v) else { panic!() };
        assert_eq!(a.run_args, ["-x", "--target", "y"]);
        assert!(a.release);
        // Words after the root are program arguments too, before those after `--`.
        let a = args("run s.stv a --release b -- -c d");
        assert_eq!(a.pkg_dir, Some(PathBuf::from("s.stv")));
        assert_eq!(a.run_args, ["a", "b", "-c", "d"]);
        assert!(a.release);
        assert!(args("run s.stv").run_args.is_empty());
        assert!(args("unsafe-report --json").json);
        assert!(args("effects pkg --json").json);
        let a = args("loops pkg --json --all");
        assert_eq!((a.cmd, a.json, a.all), (Cmd::Loops, true, true));
        assert!(!args("loops").all && usage("loops --compiler cpu").contains("does not take --compiler"));
        let a = args("fix pkg --dry-run --diagnostics json");
        assert_eq!((a.cmd, a.dry_run, a.diagnostics), (Cmd::Fix, true, DiagFormat::Json));
        assert_eq!(a.pkg_dir, Some(PathBuf::from("pkg")));
        assert!(!args("fix").dry_run);
        let a = args("fmt std runtime/os.stv --check");
        assert_eq!(a.paths, [PathBuf::from("std"), PathBuf::from("runtime/os.stv")]);
        assert!(a.check && a.pkg_dir.is_none());
        assert!(args("fmt").paths.is_empty() && !args("fmt").check);
        let a = args("lowering --target aarch64-linux --op gather");
        assert_eq!(a.op.as_deref(), Some("gather"));
        assert_eq!(args("explain E0201").code.as_deref(), Some("E0201"));
        assert_eq!(args("daemon status").daemon, Some(DaemonCmd::Status));
        let a = args("daemon evict pkg");
        assert_eq!((a.daemon, a.pkg_dir), (Some(DaemonCmd::Evict), Some(PathBuf::from("pkg"))));
        assert_eq!(args("build --target gpu-sm86 --alloc-check").target.triple, Triple::GpuSm86);
        assert_eq!(args("run --target gpu-sm86").target.triple, Triple::GpuSm86);
        let a = args("dup stage1 --all --min-size 20 --baseline b.txt");
        assert_eq!((a.cmd, a.all, a.min_size), (Cmd::Dup, true, Some(20)));
        assert_eq!((a.pkg_dir, a.baseline), (Some(PathBuf::from("stage1")), Some(PathBuf::from("b.txt"))));
        assert_eq!(args("dup --write-baseline=b.txt").write_baseline, Some(PathBuf::from("b.txt")));
        assert!(args("dup --json").json && !args("dup").all);
        assert!(usage("dup --json --baseline b").contains("exclude each other"));
        assert!(usage("dup --min-size x").contains("takes a number"));
        assert!(usage("check --all").contains("does not take --all"));
    }

    #[test]
    fn usage_errors() {
        assert_eq!(usage(""), "no command given");
        assert!(usage("frobnicate").contains("unknown command"));
        assert!(usage("--release build").contains("expected a command"));
        assert!(usage("build --bogus").contains("unknown flag `--bogus`"));
        assert!(usage("build --target").contains("needs a value"));
        assert!(usage("build --target=").contains("non-empty"));
        assert!(usage("build --target x86_64-linux-v5").contains("unknown target"));
        assert!(usage("build --features avx3").contains("unknown feature `avx3`"));
        assert!(usage("build --target x86_64-linux --features neon").contains("not available on target"));
        assert!(usage("build --target x86_64-linux --target x86_64-linux").contains("given twice"));
        assert!(usage("build --release --release").contains("given twice"));
        assert!(usage("build --release=yes").contains("takes no value"));
        assert!(usage("build --compiler fast").contains("auto, cpu, or gpu"));
        assert!(usage("build --diagnostics xml").contains("human or json"));
        assert!(usage("build --dump=tokens").contains("--dump-dir"));
        assert!(usage("build --dump-dir d").contains("--dump=STAGE"));
        assert!(usage("build --dump=lex --dump-dir d").contains("unknown --dump stage"));
        assert_eq!(args("check --dump=tokens,borrow --dump-dir d").dump, ["tokens", "borrow"]);
        assert_eq!(args("build --dump=ir,types --dump-dir d").dump, ["ir", "types"]);
        assert_eq!(args("build --dump=lir,opt --release --dump-dir d").dump, ["lir", "opt"]);
        assert!(usage("build --dump=opt --dump-dir d").contains("--release"));
        assert!(usage("test --dump=tokens --dump-dir d").contains("does not take --dump"));
        assert!(usage("build --alloc-check").contains("gpu-sm86"));
        assert!(usage("build a b").contains("unexpected argument `b`"));
        assert!(usage("check a.stv b").contains("unexpected argument `b`"));
        assert!(usage("run a.stv -x").contains("unknown flag `-x`"));
        assert!(usage("build -- x").contains("only accepted by `stave run`"));
        assert!(usage("check --watch").contains("does not take --watch"));
        assert!(usage("check -o x").contains("does not take -o"));
        assert!(usage("run --filter x").contains("does not take --filter"));
        assert!(usage("test --input x").contains("--input takes an input number"));
        assert!(usage("build --input 1").contains("does not take --input"));
        assert!(usage("test --exact").contains("--exact needs --filter NAME"));
        assert!(usage("test --filter a --exact --exact").contains("--exact given twice"));
        assert!(usage("run --exact").contains("does not take --exact"));
        assert!(usage("check --dry-run").contains("does not take --dry-run"));
        assert!(usage("fix --release").contains("does not take --release"));
        assert!(usage("fix --compiler gpu").contains("does not take --compiler"));
        assert!(usage("fix --dry-run --dry-run").contains("given twice"));
        assert!(usage("check --check").contains("does not take --check"));
        assert!(usage("fmt --json").contains("does not take --json"));
        assert!(usage("fmt --check --check").contains("given twice"));
        assert!(usage("lowering").contains("needs --target"));
        assert!(usage("lowering --release --target x86_64-linux").contains("does not take"));
        assert!(usage("explain").contains("needs a diagnostic code"));
        assert!(usage("explain E201").contains("not a diagnostic code"));
        assert!(usage("explain e0201").contains("not a diagnostic code"));
        assert!(usage("explain E0201 E0202").contains("unexpected argument"));
        assert!(usage("daemon").contains("status, stop, or evict"));
        assert!(usage("daemon restart").contains("status, stop, or evict"));
        assert!(usage("daemon status pkg").contains("unexpected argument"));
        assert!(usage("daemon status --json").contains("does not take"));
        // `stave run` of a foreign target parses; running it is E1001 (stave_cli::run).
        let foreign = if cfg!(target_arch = "x86_64") { "aarch64-linux" } else { "x86_64-linux" };
        assert_eq!(args(&format!("run --target {foreign}")).cmd, Cmd::Run);
    }
}
