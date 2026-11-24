//! Conformance runner self-tests.
//!
//! Fixtures: `tests/conformance/_selftest/`. They run against a fake compiler, so each test here
//! decides what the "compiler" reports and checks the runner's verdict. `stave test
//! tests/conformance` skips `_selftest/` (discovery skips names starting with `_`).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use stave_driver::conformance::{
    BuildCommand, BuildOutcome, BuildRequest, Code, Config, ExpectedDiag, ObservedDiag, Outcome, Report,
    RunOutcome, RunRequest, Severity, TestCase, TestCompiler, TestKind, discover, load_spec, run_suite,
    run_tests,
};
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn root() -> PathBuf {
    repo().join("tests/conformance/_selftest")
}

fn registry() -> BTreeSet<Code> {
    let mut codes = BTreeSet::new();
    for f in ["diagnostic-codes.md", "gpu-diagnostics.md", "gpu-program-diagnostics.md"] {
        let text = std::fs::read_to_string(repo().join("contracts").join(f)).unwrap();
        codes.extend(stave_driver::conformance::codes_from_registry(&text));
    }
    codes
}

fn case(name: &str) -> TestCase {
    discover(&root())
        .unwrap()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no fixture {name}"))
}

fn config() -> Config {
    Config { known_codes: Some(registry()), ..Config::default() }
}

fn diag(severity: Severity, code: &str, at: Option<(&str, u32)>) -> ObservedDiag {
    ObservedDiag { severity, code: code.to_string(), primary: at.map(|(f, l)| (PathBuf::from(f), l)) }
}

const E: Severity = Severity::Error;
const W: Severity = Severity::Warning;

/// Scripted compiler: every build returns `build`, every run returns `run`.
struct Fake {
    build: BuildOutcome,
    run: RunOutcome,
    runnable: bool,
    builds: Vec<(String, BuildCommand, Option<Triple>, Vec<String>)>,
    runs: Vec<(String, Option<Triple>, Option<String>)>,
}

impl Fake {
    fn new(exit: i32, diagnostics: Vec<ObservedDiag>) -> Fake {
        Fake {
            build: BuildOutcome {
                exit,
                diagnostics,
                artifact: Some(PathBuf::from("/tmp/a.out")),
                incomplete: None,
                error: None,
            },
            run: RunOutcome::default(),
            runnable: true,
            builds: Vec::new(),
            runs: Vec::new(),
        }
    }

    fn ran(mut self, exit: i32, stdout: &str, stderr: &str) -> Fake {
        self.run = RunOutcome { exit, stdout: stdout.into(), stderr: stderr.into() };
        self
    }
}

impl TestCompiler for Fake {
    fn build(&mut self, req: &BuildRequest<'_>) -> Result<BuildOutcome, String> {
        self.builds.push((req.test.name.clone(), req.command, req.target, req.args.to_vec()));
        Ok(self.build.clone())
    }

    fn can_run(&self, _target: Option<Triple>, _runner: Option<&str>) -> bool {
        self.runnable
    }

    fn run(&mut self, req: &RunRequest<'_>) -> Result<RunOutcome, String> {
        self.runs.push((req.test.name.clone(), req.target, req.runner.map(str::to_string)));
        Ok(self.run.clone())
    }
}

fn report(name: &str, fake: &mut Fake) -> Report {
    run_tests(&[case(name)], &config(), fake)
}

fn outcome(name: &str, fake: &mut Fake) -> Outcome {
    let r = report(name, fake);
    assert_eq!(r.results.len(), 1, "{r:?}");
    r.results[0].outcome.clone()
}

fn assert_failed(o: &Outcome, needle: &str) {
    match o {
        Outcome::Failed(why) => assert!(why.iter().any(|w| w.contains(needle)), "{needle:?} not in {why:?}"),
        other => panic!("expected failure containing {needle:?}, got {other:?}"),
    }
}

fn assert_malformed(o: &Outcome, needle: &str) {
    match o {
        Outcome::Malformed(why) => {
            assert!(why.iter().any(|w| w.contains(needle)), "{needle:?} not in {why:?}")
        }
        other => panic!("expected malformed containing {needle:?}, got {other:?}"),
    }
}

fn exp(severity: Severity, code: u16, loc: Option<(&str, u32)>) -> ExpectedDiag {
    ExpectedDiag { severity, code: Code(code), loc: loc.map(|(f, l)| (f.to_string(), l)) }
}

// ---------------------------------------------------------------------------------------------
// Discovery and parsing of every annotation

#[test]
fn discovery() {
    let names: Vec<(String, bool)> =
        discover(&root()).unwrap().into_iter().map(|c| (c.name, c.package)).collect();
    let want = [
        ("check_only.stv", false),
        ("malformed_registry.stv", false),
        ("malformed_unknown.stv", false),
        ("neg_basic.stv", false),
        ("neg_located.stv", false),
        ("parse_only.stv", false),
        ("pkg_bad_directive", true),
        ("pkg_manifest", true),
        ("pkg_nested", true),
        ("pkg_run", true),
        ("pos_plain.stv", false),
        ("run_directives.stv", false),
        ("run_golden.stv", false),
        ("run_target.stv", false),
    ];
    let want: Vec<(String, bool)> = want.iter().map(|(n, p)| (n.to_string(), *p)).collect();
    assert_eq!(names, want);
    // The real suite root skips `_selftest/`.
    let suite = discover(&repo().join("tests/conformance")).unwrap();
    assert!(suite.iter().all(|c| !c.name.starts_with('_')), "{suite:?}");
    // A single file or package as root is one test.
    assert_eq!(discover(&root().join("pos_plain.stv")).unwrap().len(), 1);
    assert!(discover(&root().join("pkg_run")).unwrap()[0].package);
}

#[test]
fn parses_located_shifted_and_noloc() {
    let s = load_spec(&case("neg_located.stv"), Some(&registry())).unwrap();
    let f = "neg_located.stv";
    assert_eq!(
        s.expected,
        [
            exp(E, 1302, None),
            exp(W, 1205, None),
            exp(E, 201, Some((f, 6))),
            exp(E, 15, Some((f, 7))), // `//~^` on line 8; the `//~` inside the string on line 7 is text
            exp(W, 1206, Some((f, 9))), // `//~^^` on line 11
        ]
    );
    assert_eq!(s.kind(), TestKind::Negative);
    assert!(!s.is_run_test());
}

#[test]
fn parses_run_directives() {
    let s = load_spec(&case("run_directives.stv"), Some(&registry())).unwrap();
    assert_eq!(s.stdout, Some(root().join("run_directives.stdout")));
    assert_eq!(s.stderr_contains, ["warn: \"x\"\n", "tab\there"]);
    assert_eq!(s.exit, Some(7));
    assert_eq!(
        s.targets,
        [Triple::parse("x86_64-linux-v3").unwrap(), Triple::parse("aarch64-linux+sve2").unwrap()]
    );
    assert_eq!(s.args, ["--release", "--compiler", "cpu"]);
    assert!(s.resource_limit);
    assert!(!s.check_only);
    assert_eq!(s.expected, [exp(W, 1206, Some(("run_directives.stv", 12)))]);
    assert_eq!(s.kind(), TestKind::Positive);
    assert!(s.is_run_test());
}

#[test]
fn parses_check_only() {
    let s = load_spec(&case("check_only.stv"), Some(&registry())).unwrap();
    assert!(s.check_only);
    assert!(!s.parse_only);
    assert_eq!(s.expected, [exp(W, 1205, Some(("check_only.stv", 2)))]);
}

#[test]
fn parse_only_lexes_and_parses_only() {
    let s = load_spec(&case("parse_only.stv"), Some(&registry())).unwrap();
    assert!(s.parse_only && !s.check_only);
    assert_eq!(s.kind(), TestKind::Positive);
    let mut fake = Fake::new(0, vec![]);
    assert_eq!(outcome("parse_only.stv", &mut fake), Outcome::Passed);
    assert_eq!(fake.builds[0].1, BuildCommand::Parse, "PARSE-ONLY runs lex + parse only");
    assert!(fake.runs.is_empty());
    let o = outcome("parse_only.stv", &mut Fake::new(1, vec![diag(E, "E0004", Some(("parse_only.stv", 3)))]));
    assert_failed(&o, "unexpected: error E0004 at parse_only.stv:3");
}

#[test]
fn parses_manifest_annotations() {
    let s = load_spec(&case("pkg_manifest"), Some(&registry())).unwrap();
    assert_eq!(s.expected, [exp(E, 1002, Some(("stave.toml", 4))), exp(E, 1002, Some(("stave.toml", 7)))]);
    assert_eq!(s.kind(), TestKind::Negative);
}

#[test]
fn parses_package_expected_files() {
    let s = load_spec(&case("pkg_run"), Some(&registry())).unwrap();
    assert_eq!(s.stdout, Some(root().join("pkg_run/expected.stdout")));
    assert_eq!(s.exit, Some(3));
    assert_eq!(s.expected, [exp(W, 1205, Some(("src/util.stv", 1)))]);
    assert!(s.is_run_test());
}

// ---------------------------------------------------------------------------------------------
// Malformed tests

#[test]
fn unknown_annotation_is_malformed() {
    let mut fake = Fake::new(1, vec![diag(E, "E0201", Some(("malformed_unknown.stv", 4)))]);
    let r = report("malformed_unknown.stv", &mut fake);
    assert_malformed(&r.results[0].outcome, "unknown annotation `EROR`");
    assert!(fake.builds.is_empty(), "a malformed test is never built");
    assert_eq!((r.negative, r.positive, r.malformed()), (0, 0, 1));
    assert!(!r.is_success());
}

#[test]
fn code_outside_registry_is_malformed() {
    let o = outcome("malformed_registry.stv", &mut Fake::new(1, vec![]));
    assert_malformed(&o, "E9999 is not in the diagnostic registry");
}

#[test]
fn directive_outside_root_module_is_malformed() {
    let o = outcome("pkg_bad_directive", &mut Fake::new(0, vec![]));
    assert_malformed(&o, "src/other.stv:1: EXIT is allowed only in the root module");
}

#[test]
fn nested_package_is_not_scanned_for_directives() {
    // The outer package's own `CHECK-ONLY` counts once; the nested package's marker and its
    // malformed annotation are the nested package's own, but its located diagnostic is expected
    // of this build.
    let s = load_spec(&case("pkg_nested"), Some(&registry())).unwrap();
    assert!(s.check_only);
    assert_eq!(s.expected, [exp(E, 201, Some(("inner/src/lib.stv", 3)))]);
    // Run by itself, the nested package is the root and reads its own annotations.
    let inner = TestCase {
        path: root().join("pkg_nested/inner"),
        name: "pkg_nested/inner".to_string(),
        package: true,
    };
    let why = load_spec(&inner, Some(&registry())).unwrap_err();
    assert!(why.iter().any(|w| w.contains("src/other.stv:1: malformed annotation")), "{why:?}");
    // One definition of a package's files: the nested package's are flagged.
    let files: Vec<(String, bool)> = stave_driver::conformance::package_files(&root().join("pkg_nested"))
        .unwrap()
        .into_iter()
        .map(|f| (f.key, f.nested))
        .collect();
    let want = [
        ("inner/src/lib.stv", true),
        ("inner/src/other.stv", true),
        ("inner/stave.toml", true),
        ("src/main.stv", false),
        ("stave.toml", false),
    ];
    assert_eq!(files, want.map(|(k, n)| (k.to_string(), n)));
}

// ---------------------------------------------------------------------------------------------
// Positive tests

#[test]
fn positive_passes_clean() {
    let mut fake = Fake::new(0, vec![]);
    assert_eq!(outcome("pos_plain.stv", &mut fake), Outcome::Passed);
    assert_eq!(fake.builds, [("pos_plain.stv".to_string(), BuildCommand::Build, None, vec![])]);
    assert!(fake.runs.is_empty(), "no run expectations: build only");
    // Notes are ignored.
    let o = outcome("pos_plain.stv", &mut Fake::new(0, vec![diag(Severity::Note, "E0201", None)]));
    assert_eq!(o, Outcome::Passed);
}

#[test]
fn positive_with_unannotated_warning_fails() {
    let o = outcome("pos_plain.stv", &mut Fake::new(0, vec![diag(W, "E1205", Some(("pos_plain.stv", 2)))]));
    assert_failed(&o, "unexpected: warning E1205 at pos_plain.stv:2");
}

#[test]
fn positive_with_any_error_fails() {
    let o = outcome("pos_plain.stv", &mut Fake::new(1, vec![diag(E, "E0201", Some(("pos_plain.stv", 3)))]));
    assert_failed(&o, "compiler exit 1, expected 0");
    assert_failed(&o, "unexpected: error E0201 at pos_plain.stv:3");
    // Even with exit 0 (a compiler bug), an error diagnostic fails a positive test.
    let o = outcome("pos_plain.stv", &mut Fake::new(0, vec![diag(E, "E1302", None)]));
    assert_failed(&o, "unexpected: error E1302 (no location)");
}

#[test]
fn positive_needs_exactly_the_annotated_warnings() {
    let at = |l| diag(W, "E1205", Some(("check_only.stv", l)));
    let mut fake = Fake::new(0, vec![at(2)]);
    assert_eq!(outcome("check_only.stv", &mut fake), Outcome::Passed);
    assert_eq!(fake.builds[0].1, BuildCommand::Check, "CHECK-ONLY runs `stave check`");
    assert!(fake.runs.is_empty());
    assert_failed(
        &outcome("check_only.stv", &mut Fake::new(0, vec![])),
        "missing: warning E1205 at check_only.stv:2",
    );
    assert_failed(
        &outcome("check_only.stv", &mut Fake::new(0, vec![at(3)])),
        "missing: warning E1205 at check_only.stv:2",
    );
    assert_failed(
        &outcome("check_only.stv", &mut Fake::new(0, vec![at(2), at(2)])),
        "unexpected: warning E1205",
    );
}

// ---------------------------------------------------------------------------------------------
// Negative tests

#[test]
fn negative_exact_multiset_passes() {
    let f = "neg_located.stv";
    let all = vec![
        diag(W, "E1206", Some((f, 9))),
        diag(E, "E0015", Some((f, 7))),
        diag(Severity::Note, "E0201", Some((f, 1))),
        diag(E, "E0201", Some((f, 6))),
        diag(W, "E1205", None),
        diag(E, "E1302", None),
    ];
    assert_eq!(outcome(f, &mut Fake::new(1, all.clone())), Outcome::Passed);
    // Absolute paths under the test's directory match too.
    let abs = root().join(f);
    let mut abs_all = all.clone();
    abs_all[1].primary = Some((abs, 7));
    assert_eq!(outcome(f, &mut Fake::new(1, abs_all)), Outcome::Passed);
}

#[test]
fn negative_with_extra_or_missing_diagnostic_fails() {
    let f = "neg_basic.stv";
    let ok = diag(E, "E0201", Some((f, 4)));
    let mut fake = Fake::new(1, vec![ok.clone()]);
    assert_eq!(outcome(f, &mut fake), Outcome::Passed);
    assert_eq!(fake.builds[0].1, BuildCommand::Build);
    // Extra.
    let o = outcome(f, &mut Fake::new(1, vec![ok.clone(), diag(E, "E0202", Some((f, 4)))]));
    assert_failed(&o, "unexpected: error E0202 at neg_basic.stv:4");
    // Same code twice is an extra too (multiset).
    assert_failed(&outcome(f, &mut Fake::new(1, vec![ok.clone(), ok.clone()])), "unexpected: error E0201");
    // Missing.
    assert_failed(&outcome(f, &mut Fake::new(1, vec![])), "missing: error E0201 at neg_basic.stv:4");
    // Wrong line, wrong severity, wrong file.
    let o = outcome(f, &mut Fake::new(1, vec![diag(E, "E0201", Some((f, 5)))]));
    assert_failed(&o, "missing: error E0201 at neg_basic.stv:4");
    assert_failed(&o, "unexpected: error E0201 at neg_basic.stv:5");
    assert_failed(&outcome(f, &mut Fake::new(1, vec![diag(W, "E0201", Some((f, 4)))])), "missing");
    assert_failed(&outcome(f, &mut Fake::new(1, vec![diag(E, "E0201", Some(("other.stv", 4)))])), "missing");
    // Located vs locationless.
    assert_failed(
        &outcome(f, &mut Fake::new(1, vec![diag(E, "E0201", None)])),
        "unexpected: error E0201 (no location)",
    );
    // Exit status must be 1.
    assert_failed(&outcome(f, &mut Fake::new(0, vec![ok.clone()])), "compiler exit 0, expected 1");
    assert_failed(&outcome(f, &mut Fake::new(3, vec![ok])), "internal compiler error");
}

#[test]
fn manifest_diagnostics_match_stave_toml_lines() {
    let at = |l| diag(E, "E1002", Some(("stave.toml", l)));
    assert_eq!(outcome("pkg_manifest", &mut Fake::new(1, vec![at(7), at(4)])), Outcome::Passed);
    assert_failed(
        &outcome("pkg_manifest", &mut Fake::new(1, vec![at(4), at(8)])),
        "missing: error E1002 at stave.toml:7",
    );
}

#[test]
fn negative_tests_ignore_runnability() {
    let mut fake = Fake::new(1, vec![diag(E, "E0201", Some(("neg_basic.stv", 4)))]);
    fake.runnable = false;
    assert_eq!(outcome("neg_basic.stv", &mut fake), Outcome::Passed);
    let mut fake = Fake::new(0, vec![diag(W, "E1205", Some(("check_only.stv", 2)))]);
    fake.runnable = false;
    assert_eq!(outcome("check_only.stv", &mut fake), Outcome::Passed);
}

// ---------------------------------------------------------------------------------------------
// Run tests

#[test]
fn golden_stdout_and_exit() {
    let f = "run_golden.stv";
    let mut fake = Fake::new(0, vec![]).ran(3, "fib 55\n", "");
    assert_eq!(outcome(f, &mut fake), Outcome::Passed);
    assert_eq!(fake.runs.len(), 1);
    assert_failed(
        &outcome(f, &mut Fake::new(0, vec![]).ran(3, "fib 56\n", "")),
        "stdout differs from run_golden.stdout",
    );
    assert_failed(&outcome(f, &mut Fake::new(0, vec![]).ran(3, "fib 55", "")), "stdout differs");
    assert_failed(&outcome(f, &mut Fake::new(0, vec![]).ran(0, "fib 55\n", "")), "exit status 0, expected 3");
    // A failed build is a failure, and nothing runs.
    let mut fake = Fake::new(1, vec![diag(E, "E0201", Some((f, 6)))]).ran(3, "fib 55\n", "");
    assert_failed(&outcome(f, &mut fake), "compiler exit 1");
    assert!(fake.runs.is_empty());
}

#[test]
fn package_golden_files() {
    let w = diag(W, "E1205", Some(("src/util.stv", 1)));
    assert_eq!(
        outcome("pkg_run", &mut Fake::new(0, vec![w.clone()]).ran(3, "answer 42\n", "")),
        Outcome::Passed
    );
    assert_failed(
        &outcome("pkg_run", &mut Fake::new(0, vec![w.clone()]).ran(0, "answer 42\n", "")),
        "exit status 0, expected 3",
    );
    assert_failed(
        &outcome("pkg_run", &mut Fake::new(0, vec![w]).ran(3, "answer 41\n", "")),
        "stdout differs from expected.stdout",
    );
}

#[test]
fn run_directives_per_target() {
    let w = diag(W, "E1206", Some(("run_directives.stv", 12)));
    let mut fake = Fake::new(0, vec![w.clone()]).ran(7, "hello\n", "warn: \"x\"\nand tab\there\n");
    let cfg = Config { runner: Some("qemu-aarch64 -cpu max".to_string()), ..config() };
    let r = run_tests(&[case("run_directives.stv")], &cfg, &mut fake);
    let v3 = Triple::parse("x86_64-linux-v3").unwrap();
    let sve2 = Triple::parse("aarch64-linux+sve2").unwrap();
    assert_eq!(
        r.results.iter().map(|x| (x.target, x.outcome.clone())).collect::<Vec<_>>(),
        [(Some(v3), Outcome::Passed), (Some(sve2), Outcome::Passed)]
    );
    assert_eq!((r.positive, r.negative, r.passed()), (1, 0, 2), "one test, two runs");
    let args: Vec<String> = ["--release", "--compiler", "cpu"].map(String::from).to_vec();
    assert_eq!(
        fake.builds,
        [
            ("run_directives.stv".to_string(), BuildCommand::Build, Some(v3), args.clone()),
            ("run_directives.stv".to_string(), BuildCommand::Build, Some(sve2), args),
        ]
    );
    assert_eq!(
        fake.runs[1],
        ("run_directives.stv".to_string(), Some(sve2), Some("qemu-aarch64 -cpu max".to_string()))
    );
    // Each STDERR-CONTAINS must hold.
    let mut fake = Fake::new(0, vec![w]).ran(7, "hello\n", "warn: \"x\"\n");
    let r = report("run_directives.stv", &mut fake);
    assert_failed(&r.results[0].outcome, "stderr does not contain \"tab\\there\"");
}

#[test]
fn non_runnable_target_is_not_run_never_passed() {
    let mut fake = Fake::new(0, vec![]).ran(0, "", "");
    fake.runnable = false;
    let r = report("run_target.stv", &mut fake);
    match &r.results[0].outcome {
        Outcome::NotRun(why) => {
            assert!(why.starts_with("E1001"), "{why}");
            assert!(why.contains("aarch64-linux+sve2"), "{why}");
        }
        other => panic!("expected not run, got {other:?}"),
    }
    assert_eq!(fake.builds.len(), 1, "the binary is still built");
    assert!(fake.runs.is_empty());
    assert_eq!((r.passed(), r.not_run(), r.failed()), (0, 1, 0));
    assert!(r.render().contains("not run   run_target.stv [aarch64-linux+sve2]: E1001"));
    assert!(
        r.summary()
            .ends_with("0 passed, 0 failed, 0 malformed, 1 not run, 0 not run (stage not implemented)")
    );
    // A build failure is still a failure, not "not run".
    let mut fake = Fake::new(1, vec![diag(E, "E0201", Some(("run_target.stv", 3)))]);
    fake.runnable = false;
    assert_failed(&outcome("run_target.stv", &mut fake), "compiler exit 1");
}

// ---------------------------------------------------------------------------------------------
// Whole suite

#[test]
fn suite_counts_and_filter() {
    let mut fake = Fake::new(0, vec![]);
    let r = run_suite(&root(), &config(), &mut fake).unwrap();
    // negative: neg_basic, neg_located, pkg_manifest, pkg_nested; malformed: malformed_registry,
    // malformed_unknown, pkg_bad_directive; positive: the other seven.
    assert_eq!((r.negative, r.positive, r.malformed()), (4, 7, 3));
    let summary = r.summary();
    assert!(summary.starts_with("conformance: 4 negative, 7 positive; "), "{summary}");
    assert!(r.render().ends_with(&format!("{summary}\n")));
    let filtered =
        run_suite(&root(), &Config { filter: Some("neg_".into()), ..config() }, &mut fake).unwrap();
    let names: Vec<&str> = filtered.results.iter().map(|x| x.name.as_str()).collect();
    assert_eq!(names, ["neg_basic.stv", "neg_located.stv"]);
}

// ---------------------------------------------------------------------------------------------
// Builds that stop before an unimplemented stage

fn incomplete(final_codes: &[u16]) -> Option<stave_driver::conformance::Incomplete> {
    Some(stave_driver::conformance::Incomplete {
        needs: "resolve".to_string(),
        final_codes: final_codes.iter().map(|&c| Code(c)).collect(),
    })
}

/// A one-file test case written under the cargo temp dir.
fn temp_case(name: &str, text: &str) -> TestCase {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("runner_incomplete");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    TestCase { path, name: name.to_string(), package: false }
}

#[test]
fn incomplete_build_semantic_negative_needs_stage() {
    let mut fake = Fake::new(0, vec![]);
    fake.build.incomplete = incomplete(&[4, 16]);
    let r = report("neg_basic.stv", &mut fake);
    assert_eq!(r.results[0].outcome, Outcome::NeedsStage("resolve".to_string()));
    assert_eq!((r.passed(), r.failed(), r.not_run(), r.needs_stage()), (0, 0, 0, 1));
    assert!(r.is_success());
    assert!(r.render().contains("not run   neg_basic.stv: needs resolve (not implemented yet)"));
    assert!(r.summary().ends_with("0 not run, 1 not run (stage not implemented)"), "{}", r.summary());
}

#[test]
fn incomplete_build_positive_needs_stage_unless_front_end_fails() {
    let mut fake = Fake::new(0, vec![]);
    fake.build.incomplete = incomplete(&[4]);
    assert_eq!(outcome("pos_plain.stv", &mut fake), Outcome::NeedsStage("resolve".to_string()));
    // A final-code diagnostic the test does not expect is a real failure.
    let mut fake = Fake::new(1, vec![diag(E, "E0004", Some(("pos_plain.stv", 2)))]);
    fake.build.incomplete = incomplete(&[4]);
    assert_failed(&outcome("pos_plain.stv", &mut fake), "unexpected: error E0004 at pos_plain.stv:2");
    // An internal compiler error fails even when the test cannot be judged.
    let mut fake = Fake::new(3, vec![]);
    fake.build.incomplete = incomplete(&[4]);
    assert_failed(&outcome("pos_plain.stv", &mut fake), "internal compiler error");
}

#[test]
fn incomplete_build_syntax_negative_passes_on_final_codes() {
    let case = temp_case("syn_neg.stv", "//~ CHECK-ONLY\nfn f() {\n    let = 1; //~ ERROR E0004\n}\n");
    let run =
        |fake: &mut Fake| run_tests(std::slice::from_ref(&case), &config(), fake).results[0].outcome.clone();
    let mut fake = Fake::new(1, vec![diag(E, "E0004", Some(("syn_neg.stv", 3)))]);
    fake.build.incomplete = incomplete(&[4]);
    assert_eq!(run(&mut fake), Outcome::Passed);
    // Wrong line: the front end decided it, so it fails.
    let mut fake = Fake::new(1, vec![diag(E, "E0004", Some(("syn_neg.stv", 2)))]);
    fake.build.incomplete = incomplete(&[4]);
    assert_failed(&run(&mut fake), "missing: error E0004 at syn_neg.stv:3");
    // Mixed: the syntax part matches, the semantic part waits for its stage.
    let mixed = temp_case(
        "syn_mixed.stv",
        "//~ CHECK-ONLY\nfn f() {\n    let = 1; //~ ERROR E0004\n}\nfn g() -> i64 { let a: i32 = 1; return a; } //~ ERROR E0201\n",
    );
    let mut fake = Fake::new(1, vec![diag(E, "E0004", Some(("syn_mixed.stv", 3)))]);
    fake.build.incomplete = incomplete(&[4]);
    let r = run_tests(&[mixed], &config(), &mut fake);
    assert_eq!(r.results[0].outcome, Outcome::NeedsStage("resolve".to_string()));
}

/// A compiler error message is reported with the failure.
#[test]
fn compiler_error_message_is_reported() {
    let mut fake = Fake::new(3, vec![]);
    fake.build.error = Some("codegen failed: boom".to_string());
    assert_failed(&outcome("pos_plain.stv", &mut fake), "compiler: codegen failed: boom");
}
