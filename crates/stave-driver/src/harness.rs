//! `test fn` harness of `stave test PKG_DIR` (LANG.md §2: "`test fn name() { }`
//! runs under `stave test` and is excluded from builds").
//!
//! The package is lowered once in [`stave_sema::Mode::Test`] (every `test fn` is a root); the
//! harness entry table is the list of the root package's `test fn`s in canonical decl order
//! ([`crate::exec::test_fns`]). Each entry is linked as its own executable whose `__stave_main`
//! calls that test (one process per test, so a panic, exit 101, fails only that test), and run
//! with the package directory as the working directory. A test passes iff it exits 0.

use std::fmt::Write as _;
use std::path::Path;

use crate::build::{self, Failure};
use crate::exec::{self, RunSpec, TempDir};
use crate::pipeline::{self, Goal, Options};
use crate::shipped::Library;

/// One `test fn` run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestFnResult {
    /// Checked-IR path (`main.adds_up`).
    pub name: String,
    pub outcome: TestFnOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TestFnOutcome {
    Passed,
    /// Exit status and stderr of the test process.
    Failed {
        exit: i32,
        stderr: String,
    },
    /// Built but not runnable here (E1001).
    NotRun,
}

/// Everything the harness did.
#[derive(Debug, Default)]
pub struct TestFnReport {
    pub results: Vec<TestFnResult>,
}

impl TestFnReport {
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.outcome == TestFnOutcome::Passed).count()
    }

    pub fn failed(&self) -> usize {
        self.results.iter().filter(|r| matches!(r.outcome, TestFnOutcome::Failed { .. })).count()
    }

    pub fn not_run(&self) -> usize {
        self.results.iter().filter(|r| r.outcome == TestFnOutcome::NotRun).count()
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for r in &self.results {
            match &r.outcome {
                TestFnOutcome::Passed => {
                    let _ = writeln!(s, "ok        test {}", r.name);
                }
                TestFnOutcome::NotRun => {
                    let _ = writeln!(s, "not run   test {}: E1001: target not runnable on this host", r.name);
                }
                TestFnOutcome::Failed { exit, stderr } => {
                    let _ = writeln!(s, "FAILED    test {} (exit {exit})", r.name);
                    for l in stderr.lines() {
                        let _ = writeln!(s, "    {l}");
                    }
                }
            }
        }
        let _ = writeln!(
            s,
            "test fns: {} passed, {} failed, {} not run",
            self.passed(),
            self.failed(),
            self.not_run()
        );
        s
    }
}

/// Why the harness could not run.
#[derive(Debug)]
pub enum HarnessError {
    /// The package has error diagnostics (`stave test` reports them through the conformance run).
    Diagnostics,
    Failure(Failure),
    Io(String),
}

/// Builds and runs the `test fn`s of the package at `root` (filtered by `filter`, a substring of
/// the path). `opts.goal`/`opts.tests`/`opts.lower` are set here.
pub fn run(
    pool: &rayon::ThreadPool,
    root: &Path,
    lib: &Library,
    opts: &Options,
    filter: Option<&str>,
    runner: Option<&str>,
) -> Result<TestFnReport, HarnessError> {
    let mut o = opts.clone();
    o.goal = Goal::Check;
    o.tests = true;
    o.lower = true;
    let out = pipeline::run_on(pool, root, lib, &o);
    if out.has_errors() {
        return Err(HarnessError::Diagnostics);
    }
    if let Some(f) = out.failure {
        return Err(HarnessError::Failure(f));
    }
    let Some(module) = &out.module else { return Ok(TestFnReport::default()) };
    let triple = opts
        .target
        .or_else(stave_target::Triple::host)
        .ok_or(HarnessError::Failure(Failure::NoHostTarget))?;
    let runnable = runner.is_some_and(|r| !r.trim().is_empty()) || exec::can_run(triple);
    let tmp = TempDir::new("testfn").map_err(|e| HarnessError::Io(e.to_string()))?;
    let cwd = if root.is_dir() { root.to_path_buf() } else { root.parent().unwrap_or(root).to_path_buf() };
    let mut report = TestFnReport::default();
    for (k, (name, f)) in exec::test_fns(&out.program, module).into_iter().enumerate() {
        if filter.is_some_and(|s| !name.contains(s)) {
            continue;
        }
        let built = pool
            .install(|| build::codegen_link(module, f, triple, opts.release, opts.threads))
            .map_err(HarnessError::Failure)?;
        let bin = tmp.path().join(format!("test{k}"));
        built.image.write_file(&bin).map_err(|e| HarnessError::Io(format!("{}: {e}", bin.display())))?;
        let outcome = if !runnable {
            TestFnOutcome::NotRun
        } else {
            let spec = RunSpec { args: &[], runner, cwd: Some(&cwd), inherit: false };
            let r =
                exec::run(&bin, &spec).map_err(|e| HarnessError::Io(format!("{}: {e}", bin.display())))?;
            if r.exit == 0 {
                TestFnOutcome::Passed
            } else {
                TestFnOutcome::Failed {
                    exit: r.exit,
                    stderr: String::from_utf8_lossy(&r.stderr).into_owned(),
                }
            }
        };
        report.results.push(TestFnResult { name, outcome });
    }
    Ok(report)
}
