//! Conformance runner (contracts/conformance-annotations.md; `stave test`).
//!
//! The runner discovers tests ([`discover`]), reads their expectations ([`load_spec`]), asks a
//! [`TestCompiler`] to build (or check) and run them, and applies the pass rules:
//!
//! - **negative** (some `ERROR` annotation): exit 1, and the multiset of (severity, code, file,
//!   line) of located diagnostics plus (severity, code) of locationless ones equals the annotated
//!   multiset exactly; `note`s are ignored.
//! - **positive**: exit 0 and exactly the annotated `WARNING`s. With `STDOUT`, `STDERR-CONTAINS`,
//!   `EXIT`, or `expected.*` files it is a run test: the binary runs and every expectation holds.
//!   A target the host cannot run (without `--runner`) is E1001: "not run", never "passed".
//!
//! - **stage not implemented**: a build that stopped before a stage it needed (stage0: a
//!   construct its back end does not support yet) sets [`BuildOutcome::incomplete`]. Diagnostics
//!   whose code is final after the stages that ran ([`Incomplete::final_codes`]) are still
//!   compared exactly, so a mismatch in them fails. A negative test whose every annotated code is
//!   final then passes on that evidence. Every other test is [`Outcome::NeedsStage`]: neither
//!   passed nor failed, listed and counted separately ("not run: needs <stage>"), and it does not
//!   fail the suite.
//!
//! The compiler is a callback ([`TestCompiler`]); [`stage0::Stage0`] is the in-process one backed
//! by the driver pipeline. An out-of-process runner would parse `--diagnostics json` lines into
//! the same [`ObservedDiag`].

mod annot;
mod spec;
pub mod stage0;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use stave_target::Triple;

pub use annot::{Annot, AnnotAt, AnnotError, Code, Directive, Severity, Syntax, scan};
pub use spec::{
    ExpectedDiag, PackageFile, TestCase, TestKind, TestSpec, codes_from_registry, discover, load_spec,
    package_files, rel_key,
};

/// A diagnostic as the runner sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedDiag {
    pub severity: Severity,
    /// As reported (`EXXXX`).
    pub code: String,
    /// File and 1-based line where the primary span starts; `None` if the diagnostic has no
    /// primary span. The file may be absolute or relative to the test's base directory.
    pub primary: Option<(PathBuf, u32)>,
}

/// Conversion of a compiler diagnostic into the runner's view (wired to `stave_diag::Diagnostic`).
pub trait AsObserved {
    fn observed(&self) -> ObservedDiag;
}

impl AsObserved for ObservedDiag {
    fn observed(&self) -> ObservedDiag {
        self.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildCommand {
    /// `stave build --diagnostics json`.
    Build,
    /// `stave check --diagnostics json` (`CHECK-ONLY`).
    Check,
    /// Lex and parse only (`PARSE-ONLY`): no semantic stages, no `stave` command (stage0 runs
    /// `stave-syntax` in process). The exit status is 1 iff there is an error diagnostic.
    Parse,
}

/// One build of one test for one target.
#[derive(Clone, Copy, Debug)]
pub struct BuildRequest<'a> {
    pub test: &'a TestCase,
    pub command: BuildCommand,
    /// `None`: no `--target` (host baseline).
    pub target: Option<Triple>,
    /// `ARGS` flags.
    pub args: &'a [String],
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuildOutcome {
    /// The compiler's exit status (0 ok, 1 diagnostics, 2 usage, 3 internal error).
    pub exit: i32,
    pub diagnostics: Vec<ObservedDiag>,
    /// The built binary, if any.
    pub artifact: Option<PathBuf>,
    /// Set when the compiler stopped before a stage the command needs, because that stage is not
    /// implemented yet for this program.
    pub incomplete: Option<Incomplete>,
    /// The compiler's own error message when it failed without diagnostics (exit 3 or 2).
    pub error: Option<String>,
}

/// A build that stopped early because a stage is not implemented yet.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Incomplete {
    /// The first missing stage (a `timings` stage name, e.g. `resolve`).
    pub needs: String,
    /// Codes that only the stages that ran can report: for these the observed set is final.
    pub final_codes: BTreeSet<Code>,
}

#[derive(Clone, Copy, Debug)]
pub struct RunRequest<'a> {
    pub test: &'a TestCase,
    pub target: Option<Triple>,
    pub artifact: Option<&'a Path>,
    /// `--runner CMD` of `stave test`.
    pub runner: Option<&'a str>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunOutcome {
    pub exit: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The compile/run callback. `Err` = the harness could not do the step (the test fails).
pub trait TestCompiler {
    fn build(&mut self, req: &BuildRequest<'_>) -> Result<BuildOutcome, String>;
    /// Whether binaries for `target` (`None` = host baseline) can run here, given `runner`.
    fn can_run(&self, target: Option<Triple>, runner: Option<&str>) -> bool;
    fn run(&mut self, req: &RunRequest<'_>) -> Result<RunOutcome, String>;
}

#[derive(Clone, Debug, Default)]
pub struct Config {
    /// `--filter S`: only tests whose name contains `S` (`exact`: equals `S`).
    pub filter: Option<String>,
    /// `--exact`.
    pub exact: bool,
    /// `--runner CMD`.
    pub runner: Option<String>,
    /// Codes annotations may use (contracts/diagnostic-codes.md plus the GPU tables); `None`
    /// skips the check.
    pub known_codes: Option<BTreeSet<Code>>,
    /// `--target T`: the target of tests without a `TARGET` annotation (`None`: host baseline).
    pub target: Option<Triple>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Passed,
    Failed(Vec<String>),
    /// Bad annotations or `expected.*` files.
    Malformed(Vec<String>),
    /// E1001: the binary was built but the target cannot run on this host.
    NotRun(String),
    /// The compiler does not implement a stage this test needs yet (the stage name). Not
    /// passed, not failed; counted separately.
    NeedsStage(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestResult {
    pub name: String,
    /// The `TARGET` of this run, if the test names one.
    pub target: Option<Triple>,
    /// `None` when malformed.
    pub kind: Option<TestKind>,
    pub outcome: Outcome,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// One entry per (test, target).
    pub results: Vec<TestResult>,
    /// Well-formed negative tests (counted once, whatever their target count).
    pub negative: usize,
    /// Well-formed positive tests.
    pub positive: usize,
}

impl Report {
    fn count(&self, f: impl Fn(&Outcome) -> bool) -> usize {
        self.results.iter().filter(|r| f(&r.outcome)).count()
    }

    pub fn passed(&self) -> usize {
        self.count(|o| matches!(o, Outcome::Passed))
    }

    pub fn failed(&self) -> usize {
        self.count(|o| matches!(o, Outcome::Failed(_)))
    }

    pub fn malformed(&self) -> usize {
        self.count(|o| matches!(o, Outcome::Malformed(_)))
    }

    pub fn not_run(&self) -> usize {
        self.count(|o| matches!(o, Outcome::NotRun(_)))
    }

    /// Tests waiting for an unimplemented compiler stage.
    pub fn needs_stage(&self) -> usize {
        self.count(|o| matches!(o, Outcome::NeedsStage(_)))
    }

    /// No failures and no malformed tests. Not-run tests (E1001 or a missing stage) are reported
    /// but do not fail the suite.
    pub fn is_success(&self) -> bool {
        self.failed() == 0 && self.malformed() == 0
    }

    /// The last line of the report (gate counts).
    pub fn summary(&self) -> String {
        format!(
            "conformance: {} negative, {} positive; {} passed, {} failed, {} malformed, {} not run, {} not run \
             (stage not implemented)",
            self.negative,
            self.positive,
            self.passed(),
            self.failed(),
            self.malformed(),
            self.not_run(),
            self.needs_stage()
        )
    }

    /// One line per result (details indented below failures), then [`Report::summary`].
    pub fn render(&self) -> String {
        let mut s = String::new();
        for r in &self.results {
            let name = match r.target {
                Some(t) => format!("{} [{t}]", r.name),
                None => r.name.clone(),
            };
            let (tag, details): (&str, &[String]) = match &r.outcome {
                Outcome::Passed => ("ok", &[]),
                Outcome::Failed(d) => ("FAILED", d),
                Outcome::Malformed(d) => ("MALFORMED", d),
                Outcome::NotRun(why) => {
                    let _ = writeln!(s, "not run   {name}: {why}");
                    continue;
                }
                Outcome::NeedsStage(stage) => {
                    let _ = writeln!(s, "not run   {name}: needs {stage} (not implemented yet)");
                    continue;
                }
            };
            let _ = writeln!(s, "{tag:<9} {name}");
            for d in details {
                let _ = writeln!(s, "    {d}");
            }
        }
        s.push_str(&self.summary());
        s.push('\n');
        s
    }
}

/// Discover tests under `root` and run them.
pub fn run_suite(root: &Path, cfg: &Config, compiler: &mut dyn TestCompiler) -> std::io::Result<Report> {
    Ok(run_tests(&discover(root)?, cfg, compiler))
}

/// Run the given tests (after `cfg.filter`).
pub fn run_tests(cases: &[TestCase], cfg: &Config, compiler: &mut dyn TestCompiler) -> Report {
    let mut report = Report::default();
    for case in cases {
        if let Some(f) = &cfg.filter
            && !(if cfg.exact { case.name == *f } else { case.name.contains(f.as_str()) })
        {
            continue;
        }
        let spec = match load_spec(case, cfg.known_codes.as_ref()) {
            Ok(s) => s,
            Err(errs) => {
                report.results.push(TestResult {
                    name: case.name.clone(),
                    target: None,
                    kind: None,
                    outcome: Outcome::Malformed(errs),
                });
                continue;
            }
        };
        let kind = spec.kind();
        match kind {
            TestKind::Negative => report.negative += 1,
            TestKind::Positive => report.positive += 1,
        }
        let targets: Vec<Option<Triple>> = if spec.targets.is_empty() {
            vec![cfg.target]
        } else {
            spec.targets.iter().copied().map(Some).collect()
        };
        for target in targets {
            let outcome = run_one(case, &spec, target, cfg, compiler);
            report.results.push(TestResult { name: case.name.clone(), target, kind: Some(kind), outcome });
        }
    }
    report
}

type DiagKey = (Severity, String, Option<(String, u32)>);

fn fmt_key(k: &DiagKey) -> String {
    match &k.2 {
        Some((f, l)) => format!("{} {} at {f}:{l}", k.0, k.1),
        None => format!("{} {} (no location)", k.0, k.1),
    }
}

/// Multiset difference: (expected but not seen, seen but not expected), each sorted.
fn diff(mut expected: Vec<DiagKey>, mut observed: Vec<DiagKey>) -> (Vec<DiagKey>, Vec<DiagKey>) {
    expected.sort();
    observed.sort();
    let (mut missing, mut extra) = (Vec::new(), Vec::new());
    let (mut i, mut j) = (0, 0);
    while i < expected.len() || j < observed.len() {
        match (expected.get(i), observed.get(j)) {
            (Some(e), Some(o)) if e == o => {
                i += 1;
                j += 1;
            }
            (Some(e), Some(o)) if e < o => {
                missing.push(e.clone());
                i += 1;
            }
            (Some(_), Some(o)) | (None, Some(o)) => {
                extra.push(o.clone());
                j += 1;
            }
            (Some(e), None) => {
                missing.push(e.clone());
                i += 1;
            }
            (None, None) => break,
        }
    }
    (missing, extra)
}

fn run_one(
    case: &TestCase,
    spec: &TestSpec,
    target: Option<Triple>,
    cfg: &Config,
    compiler: &mut dyn TestCompiler,
) -> Outcome {
    let command = if spec.parse_only {
        BuildCommand::Parse
    } else if spec.check_only {
        BuildCommand::Check
    } else {
        BuildCommand::Build
    };
    let req = BuildRequest { test: case, command, target, args: &spec.args };
    let built = match compiler.build(&req) {
        Ok(b) => b,
        Err(e) => return Outcome::Failed(vec![format!("harness: {e}")]),
    };
    let base = case.base_dir();
    let observed: Vec<DiagKey> = built
        .diagnostics
        .iter()
        .filter(|d| d.severity != Severity::Note)
        .map(|d| (d.severity, d.code.clone(), d.primary.as_ref().map(|(f, l)| (rel_key(&base, f), *l))))
        .collect();
    let expected: Vec<DiagKey> =
        spec.expected.iter().map(|e| (e.severity, e.code.to_string(), e.loc.clone())).collect();

    let mut fails = Vec::new();
    if let Some(e) = &built.error {
        fails.push(format!("compiler: {e}"));
    }
    if let Some(inc) = &built.incomplete {
        let mut o = judge_incomplete(spec, inc, built.exit, expected, observed);
        if let (Outcome::Failed(d), Some(e)) = (&mut o, &built.error) {
            d.push(format!("compiler: {e}"));
        }
        return o;
    }

    let want_exit = match spec.kind() {
        TestKind::Negative => 1,
        TestKind::Positive => 0,
    };
    if built.exit != want_exit {
        let what = if built.exit == 3 { " (internal compiler error)" } else { "" };
        fails.push(format!("compiler exit {}{what}, expected {want_exit}", built.exit));
    }
    let (missing, extra) = diff(expected, observed);
    fails.extend(missing.iter().map(|k| format!("missing: {}", fmt_key(k))));
    fails.extend(extra.iter().map(|k| format!("unexpected: {}", fmt_key(k))));
    if !fails.is_empty() {
        return Outcome::Failed(fails);
    }
    if spec.kind() == TestKind::Negative || spec.check_only || spec.parse_only || !spec.is_run_test() {
        return Outcome::Passed;
    }

    let runner = cfg.runner.as_deref();
    if !compiler.can_run(target, runner) {
        let t = target.or_else(Triple::host).map(Triple::name).unwrap_or("host");
        return Outcome::NotRun(format!(
            "E1001: target {t} is not runnable on this host (binary built; pass --runner to run it)"
        ));
    }
    let ran =
        match compiler.run(&RunRequest { test: case, target, artifact: built.artifact.as_deref(), runner }) {
            Ok(r) => r,
            Err(e) => return Outcome::Failed(vec![format!("harness: {e}")]),
        };
    let want = i32::from(spec.exit.unwrap_or(0));
    if ran.exit != want {
        fails.push(format!("exit status {}, expected {want}", ran.exit));
    }
    if let Some(golden) = &spec.stdout {
        match std::fs::read(golden) {
            Ok(g) if g == ran.stdout => {}
            Ok(g) => fails.push(format!(
                "stdout differs from {} ({} bytes expected, {} bytes got{})",
                rel_key(&base, golden),
                g.len(),
                ran.stdout.len(),
                first_difference(&g, &ran.stdout)
            )),
            Err(e) => return Outcome::Malformed(vec![format!("{}: cannot read: {e}", golden.display())]),
        }
    }
    for text in &spec.stderr_contains {
        if !contains(&ran.stderr, text.as_bytes()) {
            fails.push(format!("stderr does not contain {text:?}"));
        }
    }
    if fails.is_empty() { Outcome::Passed } else { Outcome::Failed(fails) }
}

/// The verdict for a build that stopped before a stage it needed (module docs).
fn judge_incomplete(
    spec: &TestSpec,
    inc: &Incomplete,
    exit: i32,
    expected: Vec<DiagKey>,
    observed: Vec<DiagKey>,
) -> Outcome {
    if exit == 3 {
        return Outcome::Failed(vec!["compiler exit 3 (internal compiler error)".to_string()]);
    }
    let is_final = |code: &str| Code::parse(code).is_some_and(|c| inc.final_codes.contains(&c));
    let exp_final: Vec<DiagKey> = expected.into_iter().filter(|k| is_final(&k.1)).collect();
    let obs_final: Vec<DiagKey> = observed.into_iter().filter(|k| is_final(&k.1)).collect();
    let (missing, extra) = diff(exp_final, obs_final);
    let mut fails: Vec<String> = missing.iter().map(|k| format!("missing: {}", fmt_key(k))).collect();
    fails.extend(extra.iter().map(|k| format!("unexpected: {}", fmt_key(k))));
    if !fails.is_empty() {
        fails.push(format!("(stopped before `{}`, not implemented yet; these codes are final)", inc.needs));
        return Outcome::Failed(fails);
    }
    let all_final = spec.expected.iter().all(|e| inc.final_codes.contains(&e.code));
    if spec.kind() == TestKind::Negative && all_final {
        if exit != 1 {
            let what = if exit == 3 { " (internal compiler error)" } else { "" };
            return Outcome::Failed(vec![format!("compiler exit {exit}{what}, expected 1")]);
        }
        return Outcome::Passed;
    }
    Outcome::NeedsStage(inc.needs.clone())
}

fn first_difference(a: &[u8], b: &[u8]) -> String {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(i) => format!(", first difference at byte {i}"),
        None => String::new(),
    }
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    needle.is_empty() || hay.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(sev: Severity, code: &str, loc: Option<(&str, u32)>) -> DiagKey {
        (sev, code.to_string(), loc.map(|(f, l)| (f.to_string(), l)))
    }

    #[test]
    fn multiset_diff() {
        let e = Severity::Error;
        let a = k(e, "E0201", Some(("a.stv", 3)));
        let b = k(e, "E0202", None);
        assert_eq!(diff(vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]), (vec![], vec![]));
        // Multiplicity counts.
        assert_eq!(diff(vec![a.clone(), a.clone()], vec![a.clone()]), (vec![a.clone()], vec![]));
        assert_eq!(diff(vec![a.clone()], vec![a.clone(), a.clone()]), (vec![], vec![a.clone()]));
        assert_eq!(diff(vec![a.clone()], vec![b.clone()]), (vec![a.clone()], vec![b.clone()]));
        assert_eq!(diff(vec![], vec![]), (vec![], vec![]));
    }

    #[test]
    fn rel_keys() {
        assert_eq!(rel_key(Path::new("/t/pkg"), Path::new("/t/pkg/src/a.stv")), "src/a.stv");
        assert_eq!(rel_key(Path::new("/t"), Path::new("./x.stv")), "x.stv");
        assert_eq!(rel_key(Path::new("/t"), Path::new("x.stv")), "x.stv");
    }

    #[test]
    fn registry_codes() {
        let md = "| Code | Rule |\n|---|---|\n| E0001 | bad byte |\n  | E1205 | warning | x |\n| E12xx | range |\n";
        let codes: Vec<String> = codes_from_registry(md).iter().map(Code::to_string).collect();
        assert_eq!(codes, ["E0001", "E1205"]);
    }

    #[test]
    fn substring() {
        assert!(contains(b"panic: Overflow at", b"Overflow"));
        assert!(!contains(b"pan", b"panic"));
        assert!(contains(b"", b""));
    }
}
