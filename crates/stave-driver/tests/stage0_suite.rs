//! `stave test` with the stage0 compiler (shipped std and runtime) over the syntax,
//! resolve, and types suites and the dump golden inputs: no test fails or is malformed;
//! `PARSE-ONLY` tests and every negative test pass, and so do the positives (the borrow checker
//! runs on every check and build). The `run`
//! suite builds, runs, and passes completely: natively on the x86-64 host, under
//! `qemu-aarch64` for AArch64 (when installed), and as E1001 "not run" without a runner. So do
//! the threaded programs of the borrow suite (`threads_*`) and the `std` suite, which the
//! interpreter comparison (`interp_vs_native`) cannot run.

use std::path::{Path, PathBuf};

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::conformance::stage0::{Stage0, registry_codes};
use stave_driver::conformance::{Config, Outcome, Report, run_suite};
use stave_driver::shipped::Library;
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn stage0(threads: usize) -> Stage0 {
    Stage0::new(Some(threads), Library::in_dir(&repo())).unwrap()
}

fn suite(name: &str) -> Report {
    let mut c = stage0(4);
    let r = run_suite(&repo().join("tests/conformance").join(name), &config(), &mut c).unwrap();
    eprintln!("{}", r.render());
    r
}

/// Every test passes: nothing fails, is malformed, or is "not run" (the borrow checker runs on
/// every check and build; the empty `lex_empty.stv` is a `PARSE-ONLY` positive).
fn all_decided(r: &Report) {
    let failed: Vec<&str> = r
        .results
        .iter()
        .filter(|t| matches!(t.outcome, Outcome::Failed(_)))
        .map(|t| t.name.as_str())
        .collect();
    assert!(failed.is_empty(), "failures: {failed:?}\n{}", r.render());
    assert_eq!((r.malformed(), r.not_run()), (0, 0));
    for t in &r.results {
        assert_eq!(t.outcome, Outcome::Passed, "{}", t.name);
    }
}

fn config() -> Config {
    Config { known_codes: Some(registry_codes()), ..Config::default() }
}

fn have(tool: &str) -> bool {
    std::process::Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

fn run_suite_for(target: Option<Triple>, runner: Option<&str>) -> Report {
    let mut c = stage0(8);
    let cfg = Config { target, runner: runner.map(String::from), ..config() };
    let r = run_suite(&repo().join("tests/conformance/run"), &cfg, &mut c).unwrap();
    eprintln!("{}", r.render());
    r
}

/// Every run test of the host architecture passes natively.
#[test]
fn run_suite_passes_natively() {
    let Some(host) = Triple::host().filter(|t| legs::gate(t.name(), stave_driver::exec::can_run(*t))) else {
        eprintln!("note: skipped (not a Linux host)");
        return;
    };
    let r = run_suite_for(Some(host), None);
    assert!(r.positive >= 19, "{}", r.summary());
    assert_eq!(r.passed(), r.results.len(), "{}", r.render());
}

/// AArch64: every run test passes under qemu; without a runner each is E1001 "not run", never
/// passed.
#[test]
fn run_suite_aarch64() {
    let a64 = Triple::parse("aarch64-linux").unwrap();
    if !stave_driver::exec::can_run(a64) {
        let r = run_suite_for(Some(a64), None);
        assert_eq!((r.passed(), r.failed(), r.not_run()), (0, 0, r.results.len()), "{}", r.render());
        assert!(r.is_success());
        assert!(r.render().contains("E1001: target aarch64-linux is not runnable"), "{}", r.render());
    }
    let qemu = cfg!(all(target_os = "linux", target_arch = "x86_64")) && have("qemu-aarch64");
    if legs::gate(a64.name(), qemu || stave_driver::exec::can_run(a64)) && qemu {
        let r = run_suite_for(Some(a64), Some("qemu-aarch64"));
        assert_eq!(r.passed(), r.results.len(), "{}", r.render());
    } else {
        eprintln!("note: qemu-aarch64 not installed: AArch64 run tests not executed");
    }
}

/// The programs the reference interpreter cannot run (it has no threads; the 10^6-digit float
/// parse of `std/` is too slow for it), so `interp_vs_native` leaves them out: every test of
/// `borrow/threads_*` (runs and their CHECK-ONLY negatives) and of `std/`, for `target` (with
/// `runner`). Returns (passed, total) and the rendered reports.
fn threads_and_std(target: Triple, runner: Option<&str>) -> (usize, usize, String) {
    let mut c = stage0(8);
    let cfg = Config { target: Some(target), runner: runner.map(String::from), ..config() };
    let threads = Config { filter: Some("threads_".to_string()), ..cfg.clone() };
    let conf = repo().join("tests/conformance");
    let a = run_suite(&conf.join("borrow"), &threads, &mut c).unwrap();
    let b = run_suite(&conf.join("std"), &cfg, &mut c).unwrap();
    assert!(a.results.len() >= 13 && a.positive >= 6, "threads tests: {}", a.render());
    assert!(!b.results.is_empty(), "std tests: {}", b.render());
    let passed = a.passed() + b.passed();
    let total = a.results.len() + b.results.len();
    (passed, total, format!("{}\n{}", a.render(), b.render()))
}

/// Threaded and `std/` programs pass natively on the host architecture.
#[test]
fn threads_and_std_natively() {
    let Some(host) = Triple::host().filter(|t| legs::gate(t.name(), stave_driver::exec::can_run(*t))) else {
        eprintln!("note: skipped (not a Linux host)");
        return;
    };
    let (passed, total, r) = threads_and_std(host, None);
    eprintln!("{r}");
    assert_eq!(passed, total, "{r}");
}

/// Threaded and `std/` programs pass on AArch64 under qemu.
#[test]
fn threads_and_std_aarch64() {
    let a64 = Triple::parse("aarch64-linux").unwrap();
    let runner = if stave_driver::exec::can_run(a64) {
        None
    } else if legs::gate(
        a64.name(),
        cfg!(all(target_os = "linux", target_arch = "x86_64")) && have("qemu-aarch64"),
    ) {
        Some("qemu-aarch64")
    } else {
        eprintln!("note: qemu-aarch64 not installed: AArch64 threaded tests not executed");
        return;
    };
    let (passed, total, r) = threads_and_std(a64, runner);
    eprintln!("{r}");
    assert_eq!(passed, total, "{r}");
}

#[test]
fn syntax_suite_passes() {
    let r = suite("syntax");
    all_decided(&r);
    assert!(r.negative + r.positive >= 50, "suite too small");
    let parse_only = ["parse_fn.stv", "parse_kernel.stv", "parse_types.stv"];
    for name in parse_only {
        let t = r.results.iter().find(|t| t.name == name).unwrap();
        assert_eq!(t.outcome, Outcome::Passed, "{name}");
    }
    assert!(r.passed() > r.negative, "PARSE-ONLY positives pass too");
}

/// Driver-level codes of builds (E0108 no `fn main`): every test passes.
#[test]
fn driver_suite_passes() {
    let r = suite("driver");
    all_decided(&r);
    assert!(r.negative >= 1, "suite too small");
}

#[test]
fn resolve_suite_passes() {
    let r = suite("resolve");
    all_decided(&r);
    assert!(r.negative >= 10, "suite too small");
}

#[test]
fn types_suite_passes() {
    let r = suite("types");
    all_decided(&r);
    assert!(r.negative >= 10, "suite too small");
}

/// Feature-typed fns and `dispatch` (E07xx signature, coercion, and arm rules; dispatch runs,
/// natively on the x86-64 host): every test passes.
#[test]
fn dispatch_suite_passes() {
    let r = suite("dispatch");
    all_decided(&r);
    assert!(r.negative >= 6, "suite too small");
}

/// Effects and deny clauses (E1401, E1403, E1404): one positive and one negative test per
/// effect; the `trap` run test passes in debug and release alike, natively on the x86-64 host.
#[test]
fn effects_suite_passes() {
    let r = suite("effects");
    all_decided(&r);
    assert!(r.negative >= 10 && r.positive >= 8, "suite too small");
}

#[test]
fn dump_inputs_as_conformance_tests() {
    let mut c = stage0(2);
    let r = run_suite(&repo().join("tests/conformance/dumps"), &config(), &mut c).unwrap();
    assert!(r.is_success(), "{}", r.render());
    // The `ir_<construct>` golden inputs are valid packages (CHECK-ONLY), except
    // `ir_error`, a negative; all pass.
    let (names, ir): (Vec<_>, Vec<_>) =
        r.results.iter().map(|t| (t.name.as_str(), &t.outcome)).partition(|(n, _)| !n.starts_with("ir_"));
    assert_eq!(
        names,
        [
            ("dropped_item/input", &Outcome::Passed),
            ("intern_collision/input", &Outcome::Passed),
            ("kinds_errors/input", &Outcome::Passed),
            ("kinds_layout/input", &Outcome::Passed),
            ("mir_forms/input", &Outcome::Passed),
            ("multi_file/input", &Outcome::Passed),
            ("sigs_decls/input", &Outcome::Passed),
            ("sigs_errors/input", &Outcome::Passed),
            ("unbalanced/input", &Outcome::Passed),
            ("valid_small/input", &Outcome::Passed),
        ]
    );
    assert!(ir.len() >= 15, "{ir:?}");
    for (n, o) in ir {
        assert_eq!(o, &Outcome::Passed, "{n}");
    }
}

/// View and origin rules: every test passes (signature-level codes from type
/// checking, loan and returned-origin codes from the borrow checker).
#[test]
fn views_suite() {
    let r = suite("views");
    let failed: Vec<&str> = r
        .results
        .iter()
        .filter(|t| matches!(t.outcome, Outcome::Failed(_)))
        .map(|t| t.name.as_str())
        .collect();
    assert!(failed.is_empty(), "{failed:?}\n{}", r.render());
    assert_eq!((r.malformed(), r.not_run()), (0, 0), "{}", r.render());
    for t in &r.results {
        assert_eq!(t.outcome, Outcome::Passed, "{}", t.name);
    }
    let passed = r.results.iter().filter(|t| t.outcome == Outcome::Passed).count();
    assert!(passed >= 15, "{}", r.render());
}
