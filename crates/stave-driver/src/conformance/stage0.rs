//! The stage0 [`TestCompiler`]: builds each conformance test in process with the driver
//! pipeline and runs its binary (`stave test`).
//!
//! Every test is built with the shipped `std` and `runtime` ([`Library`]), for its `TARGET` (or
//! `stave test --target`), as a debug build. `ARGS` may hold `--release` (a release build) and
//! `--compiler cpu|auto` (stage0 is the CPU compiler, and no stage1 is ever selected); any other
//! flag is a harness error, never silently dropped. Outcomes:
//!
//! - `PARSE-ONLY`: the front end only; complete.
//! - `CHECK-ONLY`: through type checking and the borrow checker; complete.
//! - builds: type and borrow checking, then (no errors) lowering, code generation, and the link.
//!   A program the back end cannot compile yet ([`Failure::Unsupported`]) is "needs
//!   lower/codegen"; any other back-end failure is exit 3 with the message.
//!
//! Binaries go to a private temporary directory (removed when the compiler is dropped) and run
//! with the test's base directory (the package directory, or the directory of the file) as the
//! working directory, natively or through `--runner`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use super::{
    BuildCommand, BuildOutcome, BuildRequest, Code, Incomplete, ObservedDiag, RunOutcome, RunRequest,
    Severity, TestCompiler,
};
use crate::build::Failure;
use crate::exec::{self, RunSpec, TempDir};
use crate::package;
use crate::pipeline::{self, Goal, Options, PipelineError, Stage};
use crate::shipped::Library;
use stave_target::Triple;

/// Every registered code (contracts/diagnostic-codes.md ∪ the GPU tables, through the stave-diag
/// registry generated from diagnostics.toml): the `known_codes` of `stave test`.
pub fn registry_codes() -> BTreeSet<Code> {
    stave_diag::entries().iter().filter_map(|e| Code::parse(e.code)).collect()
}

/// Codes whose stage0 stage reports every instance (final once those stages ran).
pub fn final_codes() -> BTreeSet<Code> {
    stave_diag::entries()
        .iter()
        .filter(|e| Stage::of_code(e.code).is_some_and(Stage::reports_all_codes))
        .filter_map(|e| Code::parse(e.code))
        .collect()
}

pub struct Stage0 {
    pool: rayon::ThreadPool,
    threads: Option<usize>,
    lib: Library,
    final_codes: BTreeSet<Code>,
    /// Built binaries (created on the first build).
    out: Option<TempDir>,
    built: u64,
}

impl Stage0 {
    /// `threads`: worker count (`STAVE_THREADS`), `None` = default; `lib`: the built-in packages
    /// (normally [`crate::shipped::locate`]).
    pub fn new(threads: Option<usize>, lib: Library) -> Result<Stage0, PipelineError> {
        Ok(Stage0 {
            pool: pipeline::thread_pool(threads)?,
            threads,
            lib,
            final_codes: final_codes(),
            out: None,
            built: 0,
        })
    }

    fn artifact_path(&mut self, name: &str) -> Result<PathBuf, String> {
        if self.out.is_none() {
            self.out = Some(TempDir::new("test").map_err(|e| format!("temporary directory: {e}"))?);
        }
        let dir = self.out.as_ref().map(|d| d.path().to_path_buf()).unwrap_or_default();
        self.built += 1;
        let safe: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        Ok(dir.join(format!("{}-{safe}", self.built)))
    }
}

/// Whether the `ARGS` flags ask for a release build. Flags the stage0 test compiler does not
/// implement are an error.
fn release_from_args(args: &[String]) -> Result<bool, String> {
    let mut release = false;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        i += 1;
        let compiler = match a {
            "--release" => {
                release = true;
                continue;
            }
            "--compiler" => {
                i += 1;
                args.get(i - 1).map(String::as_str)
            }
            _ => a.strip_prefix("--compiler="),
        };
        match compiler {
            Some("cpu" | "auto") => {}
            _ => return Err(format!("ARGS `{}`: not supported by the stage0 test compiler", args.join(" "))),
        }
    }
    Ok(release)
}

fn severity(s: stave_diag::Severity) -> Severity {
    match s {
        stave_diag::Severity::Error => Severity::Error,
        stave_diag::Severity::Warning => Severity::Warning,
        stave_diag::Severity::Note => Severity::Note,
    }
}

impl TestCompiler for Stage0 {
    fn build(&mut self, req: &BuildRequest<'_>) -> Result<BuildOutcome, String> {
        package::check_root(&req.test.path).map_err(|e| e.to_string())?;
        let goal = match req.command {
            BuildCommand::Parse => Goal::Parse,
            BuildCommand::Check => Goal::Check,
            BuildCommand::Build => Goal::Build,
        };
        let mut opts = Options::new(goal).threads(self.threads);
        opts.target = req.target;
        opts.release = release_from_args(req.args)?;
        let out = pipeline::run_on(&self.pool, &req.test.path, &self.lib, &opts);
        let mut diagnostics = Vec::with_capacity(out.diagnostics.len());
        for d in &out.diagnostics {
            let primary = match d.primary() {
                Some(s) => {
                    let loc = out.files().locate(s).map_err(|e| format!("internal: {e}"))?;
                    Some((PathBuf::from(loc.path), loc.line))
                }
                None => None,
            };
            diagnostics.push(ObservedDiag {
                severity: severity(d.severity()),
                code: d.code().to_string(),
                primary,
            });
        }
        let mut b = BuildOutcome {
            exit: i32::from(out.has_errors()),
            diagnostics,
            artifact: None,
            incomplete: None,
            error: None,
        };
        match &out.failure {
            None => {}
            Some(f @ (Failure::Unsupported { .. } | Failure::Target(_) | Failure::NoHostTarget)) => {
                b.exit = 2;
                b.incomplete = Some(Incomplete {
                    needs: f.stage().name().to_string(),
                    final_codes: self.final_codes.clone(),
                });
                b.error = Some(f.to_string());
            }
            Some(f) => {
                b.exit = if matches!(f, Failure::NoMain) { 2 } else { 3 };
                b.error = Some(f.to_string());
            }
        }
        if let Some(img) = &out.image {
            let path = self.artifact_path(&req.test.name)?;
            img.write_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            b.artifact = Some(path);
        }
        Ok(b)
    }

    fn can_run(&self, target: Option<Triple>, runner: Option<&str>) -> bool {
        runner.is_some_and(|r| !r.trim().is_empty())
            || target.or_else(Triple::host).is_some_and(exec::can_run)
    }

    fn run(&mut self, req: &RunRequest<'_>) -> Result<RunOutcome, String> {
        let bin = req.artifact.ok_or("no binary was built")?;
        let base = req.test.base_dir();
        let spec = RunSpec { args: &[], runner: req.runner, cwd: Some(&base), inherit: false };
        let r = exec::run(bin, &spec).map_err(|e| format!("cannot run {}: {e}", bin.display()))?;
        Ok(RunOutcome { exit: r.exit, stdout: r.stdout, stderr: r.stderr })
    }
}

#[cfg(test)]
mod tests {
    use super::release_from_args;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn args_annotation() {
        assert_eq!(release_from_args(&[]), Ok(false));
        assert_eq!(release_from_args(&args("--release --compiler cpu")), Ok(true));
        assert_eq!(release_from_args(&args("--compiler=auto")), Ok(false));
        for bad in ["--features avx2", "--compiler gpu", "--compiler", "--release -o x", "--cache c"] {
            let e = release_from_args(&args(bad)).unwrap_err();
            assert!(e.contains("not supported by the stage0 test compiler"), "{bad}: {e}");
        }
    }
}
