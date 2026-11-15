//! `cargo run --release -p stave-sema --example typeck_bench -- DIR [--target T] [--threads N]
//! [--iters K] [--fixture-std]`: loads DIR with the shipped std and runtime (or the test fixture
//! std), type-checks and lowers it K times on an N-thread pool, and prints the median wall time
//! of the loader stages, type checking, and lowering per 100k lines (budget: type + borrow
//! <= 120 ms per 100k lines on 12 threads), plus the first diagnostics.

use std::path::PathBuf;
use std::time::Duration;

use stave_sema::{Config, Program};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir: Option<PathBuf> = None;
    let mut threads = 12usize;
    let mut iters = 9usize;
    let mut target = None;
    let mut fixture = false;
    let mut show_err = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => threads = args.next().and_then(|s| s.parse().ok()).unwrap_or(threads),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            "--target" => target = args.next().and_then(|s| stave_target::Triple::parse(&s).ok()),
            "--fixture-std" => fixture = true,
            "--show-err-bodies" => show_err = true,
            _ => dir = Some(PathBuf::from(a)),
        }
    }
    let Some(dir) = dir else {
        eprintln!("usage: typeck_bench DIR [--target T] [--threads N] [--iters K] [--fixture-std]");
        std::process::exit(2);
    };
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut cfg = if fixture {
        Config::new(&dir).with_std(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/std"))
    } else {
        Config::new(&dir).with_shipped(repo.join("std"), repo.join("runtime"))
    };
    cfg.target = target;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool");
    let mut res: Vec<Duration> = Vec::new();
    let mut tck: Vec<Duration> = Vec::new();
    let mut low: Vec<Duration> = Vec::new();
    let mut stages: Vec<Vec<Duration>> = vec![Vec::new(); 6];
    let mut last = None;
    for _ in 0..iters.max(1) {
        let p = pool.install(|| Program::load(&cfg));
        let t0 = std::time::Instant::now();
        let tc = pool.install(|| stave_sema::check(&p));
        tck.push(t0.elapsed());
        res.push(p.timings.intern + p.timings.resolve);
        for (k, (_, d)) in p.timings.stages().iter().enumerate() {
            stages[k].push(*d);
        }
        if tc.all_diagnostics(&p).is_empty() {
            let t1 = std::time::Instant::now();
            let r = pool.install(|| stave_sema::lower(&p, &tc, stave_sema::Mode::Debug));
            low.push(t1.elapsed());
            if let Err(e) = r {
                eprintln!("lowering: {} errors, first: {:?}", e.len(), e.first());
            }
        }
        last = Some((p, tc));
    }
    let Some((p, tc)) = last else { return };
    let lines: usize = p.files.iter().map(|f| f.bytes.iter().filter(|&&b| b == b'\n').count()).sum();
    let med = |v: &mut Vec<Duration>| {
        v.sort();
        v[v.len() / 2]
    };
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let per100k = |d: Duration| ms(d) * 100_000.0 / lines.max(1) as f64;
    let (r, t) = (med(&mut res), med(&mut tck));
    println!(
        "{} files, {lines} lines, {} decls, {} bodies, {} types, {} instances; threads {threads}, {} runs",
        p.files.len(),
        p.decls.len(),
        tc.bodies.len(),
        tc.types.len(),
        tc.instances.len(),
        tck.len()
    );
    let names = p.timings.stages().map(|s| s.0);
    let front: Vec<String> =
        names.iter().zip(stages.iter_mut()).map(|(n, v)| format!("{n} {:.3}", per100k(med(v)))).collect();
    println!("loader ms/100k lines: {}", front.join(", "));
    // the fused front end (`stave_syntax::parse_file` per file) for comparison with lex+items+parse
    let mut fused: Vec<Duration> = Vec::new();
    for _ in 0..iters.max(1) {
        let t0 = std::time::Instant::now();
        let n: usize = pool.install(|| {
            use rayon::prelude::*;
            p.files.par_iter().map(|f| stave_syntax::parse_file(&f.bytes, f.id).tokens.len()).sum()
        });
        fused.push(t0.elapsed());
        std::hint::black_box(n);
    }
    let staged = |v: &[Vec<Duration>]| v[1..4].iter().map(|x| x.iter().sum::<Duration>()).sum::<Duration>();
    println!(
        "fused parse_file: {:.3} ms/100k lines (staged lex+items+parse mean {:.3})",
        per100k(med(&mut fused)),
        per100k(staged(&stages) / stages[1].len().max(1) as u32)
    );
    println!("resolve: {:8.3} ms ({:.3} ms/100k lines)", ms(r), per100k(r));
    println!(
        "typeck:  {:8.3} ms ({:.3} ms/100k lines)  [sigs {:.3} ms, bodies {:.3} ms, canon {:.3} ms, last run]",
        ms(t),
        per100k(t),
        ms(tc.timings.sigs),
        ms(tc.timings.bodies),
        ms(tc.timings.canon)
    );
    if !low.is_empty() {
        let l = med(&mut low);
        println!("lower:   {:8.3} ms ({:.3} ms/100k lines, Mode::Debug)", ms(l), per100k(l));
    }
    let diags = tc.all_diagnostics(&p);
    println!(
        "{} diagnostics ({} body errors)",
        diags.len(),
        tc.bodies.iter().filter(|b| b.has_error).count()
    );
    let mut by_code: std::collections::BTreeMap<&str, usize> = Default::default();
    for d in &diags {
        *by_code.entry(d.code()).or_default() += 1;
    }
    println!("by code: {by_code:?}");
    if show_err {
        for b in tc.bodies.iter().filter(|b| b.has_error) {
            println!("  error body: {}", p.decl_path(b.decl));
        }
    }
    let shown: Vec<_> = diags.iter().take(40).cloned().collect();
    if let Ok(s) = stave_diag::emit_human(&shown, &p.source_map) {
        print!("{s}");
    }
}
