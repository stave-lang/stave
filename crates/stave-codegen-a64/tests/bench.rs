//! Codegen throughput (budget: ≤ 50 µs per function per thread). Run in release:
//! `cargo test --release -p stave-codegen-a64 --test bench -- --ignored --nocapture`.

mod common;

use std::time::Instant;

use common::rgen::program_n;
use stave_codegen_a64::debug::{Codegen, FuncScratch, prepare};
use stave_ir::FuncId;
use stave_obj::{ImageBuilder, Machine, Options};

#[test]
#[ignore]
fn us_per_function() {
    let p = program_n(42, Some(3000));
    let m = &p.m;
    let mut b = ImageBuilder::new(Machine::Aarch64, Options::release(Machine::Aarch64));
    let syms = prepare(&mut b, m, None).unwrap();
    let t0 = Instant::now();
    let cg = Codegen::new(m, &syms);
    let setup = t0.elapsed();
    let funcs: Vec<FuncId> =
        (0..m.funcs.len()).filter(|&i| m.funcs[i].body.is_some()).map(|i| FuncId(i as u32)).collect();
    let insts: usize = funcs.iter().map(|f| m.funcs[f.index()].body.as_ref().unwrap().insts.len()).sum();
    let mut s = FuncScratch::default();
    let mut bytes = 0usize;
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for &f in &funcs {
            let code = cg.compile(f, &mut s).unwrap();
            bytes += code.code.len();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    let n = funcs.len() as f64;
    let us = best * 1e6 / n;
    println!(
        "a64 debug codegen: {} fns, {:.1} IR insts/fn, {:.1} µs/fn ({:.0} ns/IR inst), {:.0} bytes/fn, module setup {:?}",
        funcs.len(),
        insts as f64 / n,
        us,
        best * 1e9 / insts as f64,
        bytes as f64 / 5.0 / n,
        setup
    );
    assert!(us <= 50.0, "{us:.1} µs per function");
}
