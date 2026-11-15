//! `cargo run --release -p stave-sema --example resolve_bench -- DIR [--std DIR] [--threads N]
//! [--iters K]`: loads the package at DIR K times on an N-thread pool and prints the median wall
//! time per phase (budget: resolve + intern <= 25 ms per 100k lines on 12 threads),
//! plus the diagnostic count and the first diagnostics.

use std::path::PathBuf;
use std::time::Duration;

use stave_sema::{Config, Program};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir: Option<PathBuf> = None;
    let mut std_dir: Option<PathBuf> = None;
    let mut threads = 12usize;
    let mut iters = 15usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--std" => std_dir = args.next().map(PathBuf::from),
            "--threads" => threads = args.next().and_then(|s| s.parse().ok()).unwrap_or(threads),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            _ => dir = Some(PathBuf::from(a)),
        }
    }
    let Some(dir) = dir else {
        eprintln!("usage: resolve_bench DIR [--std DIR] [--threads N] [--iters K]");
        std::process::exit(2);
    };
    let std_dir =
        std_dir.unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/std"));
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool");
    let cfg = Config::new(&dir).with_std(&std_dir);
    let mut runs: Vec<(Duration, Duration, Duration, Duration)> = Vec::new();
    let mut last = None;
    for _ in 0..iters.max(1) {
        let p = pool.install(|| Program::load(&cfg));
        let t = p.timings;
        runs.push((t.parse, t.intern, t.resolve, t.intern + t.resolve));
        last = Some(p);
    }
    let Some(p) = last else { return };
    let lines: usize = p.files.iter().map(|f| f.bytes.iter().filter(|&&b| b == b'\n').count()).sum();
    let med = |f: fn(&(Duration, Duration, Duration, Duration)) -> Duration| {
        let mut v: Vec<Duration> = runs.iter().map(f).collect();
        v.sort();
        v[v.len() / 2]
    };
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let per100k = |d: Duration| ms(d) * 100_000.0 / lines.max(1) as f64;
    println!(
        "{} files, {lines} lines, {} items, {} decls, {} strings, {} uses; threads {threads}, {} runs",
        p.files.len(),
        p.items.len(),
        p.decls.len(),
        p.strings.len(),
        p.res.iter().map(|r| r.uses.len()).sum::<usize>(),
        runs.len()
    );
    println!(
        "parse (read+lex+parse): {:8.3} ms  ({:.3} ms/100k lines)",
        ms(med(|r| r.0)),
        per100k(med(|r| r.0))
    );
    println!(
        "intern:                 {:8.3} ms  ({:.3} ms/100k lines)",
        ms(med(|r| r.1)),
        per100k(med(|r| r.1))
    );
    println!(
        "resolve:                {:8.3} ms  ({:.3} ms/100k lines)",
        ms(med(|r| r.2)),
        per100k(med(|r| r.2))
    );
    println!(
        "intern+resolve:         {:8.3} ms  ({:.3} ms/100k lines)",
        ms(med(|r| r.3)),
        per100k(med(|r| r.3))
    );
    println!("diagnostics: {}", p.diagnostics.len());
    for d in p.diagnostics.iter().take(20) {
        let l = d.primary().and_then(|s| p.source_map.locate(s).ok());
        println!("  {} {:?} {}", d.code(), l.map(|l| (l.path.to_string(), l.line, l.col)), d.message());
    }
}
