//! Dump determinism: a 10k-line gen-corpus package, built with the
//! shipped std and runtime, dumped with `tokens,items,ast,strings,names,types,diags,ir` (`ir`:
//! the checked IR of a debug lowering, `Options::lower`) under 1, 2, and 12 worker
//! threads and under the shuffle knob (`STAVE_SEMA_SHUFFLE=<seed>`, here
//! `Options::shuffle_seed`: it permutes stave-sema's internal interning and decl creation order)
//! gives byte-identical dumps; 100 repeated runs are identical. The corpus uses the default mix,
//! whose `m2` tier (views, borrows, closures) is on. The `STAVE_THREADS` and
//! `STAVE_SEMA_SHUFFLE` environment paths are tested end to end in `crates/stave-cli/tests/cli.rs`.

use std::path::{Path, PathBuf};

use stave_driver::dump::{DumpStage, render_to_vec};
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;

const STAGES: [DumpStage; 9] = [
    DumpStage::Tokens,
    DumpStage::Items,
    DumpStage::Ast,
    DumpStage::Strings,
    DumpStage::Names,
    DumpStage::Types,
    DumpStage::Diags,
    DumpStage::Ir,
    DumpStage::Lir,
];

/// The generated package, made once for all tests of this file (they run in parallel: two
/// generations into one directory race).
fn corpus() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("det10k");
        // the `m2` tier (borrows, `from` returns, closures with `&`/`&mut` captures) is on: the
        // `ir` dump covers closure bodies and `fn.make` environments
        let mix = gen_corpus::Mix::default_mix().unwrap();
        assert!(mix.tiers.m2 && mix.stmt.closure > 0, "the default mix has the m2 tier");
        gen_corpus::generate(&gen_corpus::Options {
            lines: 10_000,
            seed: 1,
            out: dir.clone(),
            mix,
            threads: 0,
            force: true,
        })
        .unwrap();
        dir
    })
    .clone()
}

fn shipped() -> Library {
    Library::in_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn dumps(pool: &rayon::ThreadPool, root: &Path, shuffle: Option<u64>) -> Vec<Vec<u8>> {
    let mut opts = Options::new(Goal::Check);
    opts.shuffle_seed = shuffle;
    opts.lower = true;
    opts.keep_lowered = true;
    let out = pipeline::run_on(pool, root, &shipped(), &opts);
    pool.install(|| STAGES.iter().map(|&s| render_to_vec(s, &out).unwrap()).collect())
}

/// The `opt` dump (release passes) and the emit form of `stave ir emit` of a release lowering.
fn release_dumps(pool: &rayon::ThreadPool, root: &Path, shuffle: Option<u64>) -> Vec<Vec<u8>> {
    let mut opts = Options::new(Goal::Check);
    opts.shuffle_seed = shuffle;
    opts.lower = true;
    opts.release = true;
    let out = pipeline::run_on(pool, root, &shipped(), &opts);
    pool.install(|| {
        vec![
            render_to_vec(DumpStage::Opt, &out).unwrap(),
            render_to_vec(DumpStage::Ir, &out).unwrap(),
            stave_driver::dump::emit_text(&out).unwrap().into_bytes(),
        ]
    })
}

#[test]
fn dumps_identical_across_threads_shuffles_and_runs() {
    let root = corpus();
    let p1 = pipeline::thread_pool(Some(1)).unwrap();
    let base = dumps(&p1, &root, None);
    assert!(base[0].len() > 100_000, "tokens dump unexpectedly small");
    for (k, min) in [(3, 1_000), (4, 10_000), (5, 10_000)] {
        assert!(base[k].len() > min, "{} dump unexpectedly small", STAGES[k].name());
    }
    assert!(base[6].is_empty(), "the corpus is valid: {}", String::from_utf8_lossy(&base[6]));
    assert!(base[7].len() > 100_000, "ir dump unexpectedly small");
    let ir = String::from_utf8_lossy(&base[7]);
    assert!(ir.contains("fn.make"), "the m2 corpus lowers closures with environments");
    assert!(base[8].len() > 100_000, "lir dump unexpectedly small");
    assert!(base[8].starts_with(b"stave.ir 4 "), "lir header");
    let differs = |got: &[Vec<u8>], what: &str| {
        for (k, s) in STAGES.iter().enumerate() {
            assert!(got[k] == base[k], "{} differs {what}", s.name());
        }
    };
    for n in [2, 12] {
        let pool = pipeline::thread_pool(Some(n)).unwrap();
        differs(&dumps(&pool, &root, None), &format!("with {n} threads"));
    }
    let pool = pipeline::thread_pool(Some(12)).unwrap();
    for seed in [1u64, 7, 0xdead_beef] {
        differs(&dumps(&pool, &root, Some(seed)), &format!("with shuffle seed {seed}"));
    }
    for run in 0..100 {
        let got = dumps(&pool, &root, None);
        assert!(got == base, "run {run} differs");
    }
}

#[test]
fn opt_and_emit_dumps_identical_across_threads_and_shuffles() {
    let root = corpus();
    let p1 = pipeline::thread_pool(Some(1)).unwrap();
    let base = release_dumps(&p1, &root, None);
    assert!(base[0].len() > 50_000, "opt dump unexpectedly small");
    assert!(base[0].starts_with(b"stave.ir 4 "));
    assert!(String::from_utf8_lossy(&base[2]).contains("\nnode "), "the emit form has a stack section");
    for n in [2, 12] {
        let pool = pipeline::thread_pool(Some(n)).unwrap();
        assert!(release_dumps(&pool, &root, None) == base, "release dumps differ with {n} threads");
    }
    let pool = pipeline::thread_pool(Some(12)).unwrap();
    for seed in [1u64, 0xdead_beef] {
        assert!(release_dumps(&pool, &root, Some(seed)) == base, "release dumps differ with seed {seed}");
    }
}
