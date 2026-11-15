//! Codegen speed (budget: ≤ 50 µs per function per thread). Prints the measurement;
//! asserts the budget only in optimized builds (`cargo test --release`).

use super::harness::Prog;
use super::randgen::Gen;
use crate::debug::{bench_ctx, compile_func};

#[test]
fn codegen_speed() {
    let n = if cfg!(debug_assertions) { 600 } else { 4000 };
    let mut p = Prog::new();
    let mut fs = Vec::with_capacity(n);
    for i in 0..n {
        let size = 20 + (i * 37) % 60;
        let (f, _, _) = Gen::new(&mut p, i as u64 + 1, size, true, false).function(&format!("b.f{i}"));
        fs.push(f);
    }
    stave_ir::verify_module(&p.m).unwrap_or_else(|e| panic!("{:?}", &e[..e.len().min(3)]));
    let insts: usize =
        fs.iter().map(|f| p.m.funcs[f.index()].body.as_ref().map_or(0, |b| b.insts.len())).sum();
    let cx = bench_ctx(&p.m);
    for &f in fs.iter().take(50) {
        compile_func(&cx, f).unwrap();
    }
    let mut best = f64::MAX;
    let mut bytes = 0usize;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        bytes = 0;
        for &f in &fs {
            bytes += compile_func(&cx, f).unwrap().code.len();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    let t = std::time::Instant::now();
    let mut nblocks = 0usize;
    for &f in &fs {
        if let Some(b) = &p.m.funcs[f.index()].body {
            nblocks += crate::debug::analysis::analyze(b).rpo.len();
        }
    }
    let analysis_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    eprintln!(
        "analysis pass alone: {analysis_us:.2} µs/fn ({:.1} reachable blocks/fn)",
        nblocks as f64 / n as f64
    );
    let us = best * 1e6 / n as f64;
    eprintln!(
        "codegen: {n} fns, {:.1} IR insts/fn, {:.0} bytes/fn: {us:.2} µs/fn, {:.1} ns/inst (1 thread{})",
        insts as f64 / n as f64,
        bytes as f64 / n as f64,
        best * 1e9 / insts as f64,
        if cfg!(debug_assertions) { ", unoptimized build" } else { "" }
    );
    if !cfg!(debug_assertions) {
        assert!(us <= 50.0, "{us:.2} µs per function exceeds the 50 µs budget");
    }
}
