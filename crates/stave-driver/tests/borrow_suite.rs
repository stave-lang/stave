//! The borrow checker through the driver:
//!
//! - `tests/conformance/borrow/{neg_,e0501_pick_,pos_}*.stv` pass under `stave test` (stage0),
//!   and so do the lock-rank tests `rank_*.stv`;
//! - related-span goldens: `--diagnostics json` of every negative there equals
//!   `tests/conformance/borrow/_json/<name>.json` (only the diagnostics located in the test file;
//!   `STAVE_BLESS=1` rewrites them; a missing one is written and fails the run once);
//! - no valid program of the other suites gets an E05xx (no false positives);
//! - diagnostics are identical with 1 and 8 threads;
//! - the 100k-line corpus has no borrow error, and the `borrowck` stage time is printed
//!   (`--ignored`: `corpus_100k_borrowck_time`).

use std::path::{Path, PathBuf};

use stave_driver::conformance::stage0::{Stage0, registry_codes};
use stave_driver::conformance::{Config, Outcome, Report, run_suite};
use stave_driver::pipeline::{self, Goal, Options, Output, Stage};
use stave_driver::shipped::Library;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lib() -> Library {
    Library::in_dir(&repo())
}

fn config() -> Config {
    Config { known_codes: Some(registry_codes()), ..Config::default() }
}

fn suite(name: &str) -> Report {
    let mut c = Stage0::new(Some(4), lib()).unwrap();
    let r = run_suite(&repo().join("tests/conformance").join(name), &config(), &mut c).unwrap();
    eprintln!("{}", r.render());
    r
}

/// The borrow checker's own tests of the suite (threads_ and unsafe_ files belong to other tasks).
fn ours(name: &str) -> bool {
    name.starts_with("neg_") || name.starts_with("e0501_pick_") || name.starts_with("pos_")
}

#[test]
fn borrow_suite_passes() {
    let mut mine = Vec::new();
    for prefix in ["neg_", "e0501_pick_", "pos_"] {
        let mut c = Stage0::new(Some(4), lib()).unwrap();
        let cfg = Config { filter: Some(prefix.to_string()), ..config() };
        let r = run_suite(&repo().join("tests/conformance/borrow"), &cfg, &mut c).unwrap();
        eprintln!("{}", r.render());
        mine.extend(r.results.into_iter().filter(|t| ours(&t.name)));
    }
    assert!(mine.len() >= 20, "suite too small: {}", mine.len());
    let bad: Vec<String> = mine
        .iter()
        .filter(|t| t.outcome != Outcome::Passed)
        .map(|t| format!("{}: {:?}", t.name, t.outcome))
        .collect();
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The lock-rank tests (`rank_*`: E1405, E1406, and what the rank check accepts) pass.
#[test]
fn rank_suite_passes() {
    let mut c = Stage0::new(Some(4), lib()).unwrap();
    let cfg = Config { filter: Some("rank_".to_string()), ..config() };
    let r = run_suite(&repo().join("tests/conformance/borrow"), &cfg, &mut c).unwrap();
    eprintln!("{}", r.render());
    let mine: Vec<_> = r.results.iter().filter(|t| t.name.starts_with("rank_")).collect();
    assert!(mine.len() >= 6, "rank suite too small: {}", mine.len());
    let bad: Vec<String> = mine
        .iter()
        .filter(|t| t.outcome != Outcome::Passed)
        .map(|t| format!("{}: {:?}", t.name, t.outcome))
        .collect();
    assert!(bad.is_empty(), "{bad:#?}");
}

fn check(root: &Path, threads: usize) -> Output {
    let opts = Options::new(Goal::Check).threads(Some(threads));
    pipeline::run(root, &lib(), &opts).unwrap()
}

/// `--diagnostics json` of the diagnostics located in `file` (the test file).
fn json_of(out: &Output, file: &str) -> String {
    let own: Vec<_> = out
        .diagnostics
        .iter()
        .filter(|d| d.primary().is_some_and(|s| out.files().locate(s).is_ok_and(|l| l.path.ends_with(file))))
        .cloned()
        .collect();
    stave_diag::emit_json(&own, out.files()).unwrap()
}

fn negatives() -> Vec<PathBuf> {
    let dir = repo().join("tests/conformance/borrow");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            (n.starts_with("neg_") || n.starts_with("e0501_pick_")) && n.ends_with(".stv")
        })
        .collect();
    v.sort();
    v
}

#[test]
fn related_span_goldens() {
    let bless = std::env::var_os("STAVE_BLESS").is_some();
    let gdir = repo().join("tests/conformance/borrow/_json");
    let mut diffs = Vec::new();
    for f in negatives() {
        let name = f.file_name().unwrap().to_str().unwrap().to_string();
        let out = check(&f, 2);
        let got = json_of(&out, &name);
        assert!(got.contains("\"code\":\"E05"), "{name}: no borrow diagnostic\n{got}");
        let path = gdir.join(name.replace(".stv", ".json"));
        if bless {
            std::fs::create_dir_all(&gdir).unwrap();
            std::fs::write(&path, &got).unwrap();
            continue;
        }
        let Ok(want) = std::fs::read_to_string(&path) else {
            // a new case: its golden is written (review it), and the test fails once
            std::fs::create_dir_all(&gdir).unwrap();
            std::fs::write(&path, &got).unwrap();
            diffs.push(format!("{name}: new golden written"));
            continue;
        };
        if want != got {
            diffs.push(format!("{name}:\n--- want\n{want}--- got\n{got}"));
        }
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
    assert_eq!(stave_borrow::diagnostic_build_failures(), 0);
}

#[test]
fn diagnostics_identical_across_threads() {
    for f in negatives() {
        let name = f.file_name().unwrap().to_str().unwrap().to_string();
        let a = json_of(&check(&f, 1), &name);
        let b = json_of(&check(&f, 8), &name);
        assert_eq!(a, b, "{name}");
    }
}

/// No E05xx in any valid program of the other suites.
#[test]
fn no_borrow_errors_in_valid_programs() {
    let mut bad = Vec::new();
    for s in ["run", "types", "resolve", "syntax", "dumps", "langmd"] {
        if !repo().join("tests/conformance").join(s).is_dir() {
            continue;
        }
        let r = suite(s);
        for t in &r.results {
            if let Outcome::Failed(msgs) = &t.outcome {
                for m in msgs {
                    if m.contains("error E05") {
                        bad.push(format!("{s}/{}: {m}", t.name));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

fn corpus(lines: u64, name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let mix = gen_corpus::Mix::default_mix().unwrap();
    gen_corpus::generate(&gen_corpus::Options {
        lines,
        seed: 1,
        out: dir.clone(),
        mix,
        threads: 0,
        force: true,
    })
    .unwrap();
    dir
}

fn borrow_errors(out: &Output) -> Vec<String> {
    out.diagnostics
        .iter()
        .filter(|d| d.code().starts_with("E05"))
        .map(|d| {
            let loc = d.primary().and_then(|s| out.files().locate(s).ok());
            // the source around the primary line
            let mut ctx = String::new();
            if let (Some(s), Some(l)) = (d.primary(), &loc)
                && let Some(f) = out.program.files.get(s.file.0 as usize)
            {
                let text = String::from_utf8_lossy(&f.bytes);
                let lo = (l.line as usize).saturating_sub(30);
                for (i, line) in text.lines().enumerate().skip(lo).take(34) {
                    ctx.push_str(&format!("{:5} {line}\n", i + 1));
                }
            }
            format!("{} {} {:?}\n{ctx}", d.code(), d.message(), loc.map(|l| (l.path, l.line)))
        })
        .collect()
}

#[test]
fn corpus_10k_has_no_borrow_errors() {
    let root = corpus(10_000, "borrow10k");
    let out = check(&root, 4);
    let e = borrow_errors(&out);
    assert!(e.is_empty(), "{}", e.join("\n"));
    let bc = out.borrowck.as_ref().unwrap();
    assert!(bc.checked > 100, "{}", bc.checked);
    eprintln!("unsupported bodies: {:?}", bc.unsupported);
}

#[test]
#[ignore = "100k-line corpus timing (run with --ignored)"]
fn corpus_100k_borrowck_time() {
    let root = corpus(100_000, "borrow100k");
    for threads in [1usize, 16] {
        let out = check(&root, threads);
        let e = borrow_errors(&out);
        assert!(e.is_empty(), "{e:#?}");
        let t = out.timings.iter().find(|t| t.stage == Stage::Borrowck).unwrap();
        let tc = out.timings.iter().find(|t| t.stage == Stage::Typeck).unwrap();
        let bc = out.borrowck.as_ref().unwrap();
        eprintln!(
            "threads {threads}: lines {} borrowck {:.1} ms ({} bodies, {} unsupported, max rounds {}; phases build/flow/check {:?}), typeck {:.1} ms",
            out.lines,
            t.ms,
            t.items,
            bc.unsupported.len(),
            bc.max_rounds,
            bc.phases,
            tc.ms
        );
    }
}
