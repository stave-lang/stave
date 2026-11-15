//! Interpreter throughput (the oracle must run ≥ 10M simple ops/s). Run in release:
//! `cargo test --release -p stave-ir --test bench -- --ignored --nocapture`.

mod common;

use std::time::Instant;

use common::*;
use stave_ir::interp::{Config, Interp, Outcome, Val};
use stave_ir::{FuncBuilder, Module, Op};

/// `fn f(n: u64) -> u64 { let mut i = 0; let mut s = 0; while i < n { s = s +% (i ^ 3); i = i + 1; } return s; }`
fn loop_module() -> (Module, stave_ir::FuncId) {
    let mut m = Module::new();
    let u64t = m.types.u64();
    let bt = m.types.bool();
    let f = m.declare(inst(&[0]), "bench.f", vec![u64t], Some(u64t));
    let mut b = FuncBuilder::new(&[u64t]);
    let n = b.param(0).unwrap();
    let (head, hp) = b.create_block_with(&[u64t, u64t]);
    let body = b.create_block();
    let (exit, ep) = b.create_block_with(&[u64t]);
    let (z, z2) = (b.int(0), b.int(0));
    b.br(head, &[z, z2]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, bt, hp[0], n);
    b.cbr(c, body, &[], exit, &[hp[1].into()]);
    b.switch_to(body);
    let three = b.int(3);
    let x = b.binary(Op::Xor, u64t, hp[0], three);
    let s = b.binary(Op::AddWrap, u64t, hp[1], x);
    let one = b.int(1);
    let i = b.binary(Op::Add, u64t, hp[0], one);
    b.br(head, &[i.into(), s.into()]);
    b.switch_to(exit);
    b.ret(Some(ep[0].into()));
    m.define(f, b.finish().unwrap());
    (m, f)
}

fn expected(n: u64) -> u64 {
    (0..n).fold(0u64, |s, i| s.wrapping_add(i ^ 3))
}

#[test]
fn loop_result() {
    let (m, f) = loop_module();
    assert_eq!(ret_val(run(&m, f, vec![Val::S(1000)])), Val::S(expected(1000)));
}

#[test]
#[ignore]
fn throughput() {
    let (m, f) = loop_module();
    let n = 5_000_000u64;
    let mut it = Interp::new(&m, host(), Config::default()).unwrap();
    let t = Instant::now();
    let o = it.run(f, vec![Val::S(n)]).unwrap();
    let dt = t.elapsed().as_secs_f64();
    assert_eq!(o, Outcome::Return(Some(Val::S(expected(n)))));
    let steps = it.steps();
    let rate = steps as f64 / dt;
    println!("interp: {steps} steps in {dt:.3} s = {:.1} M steps/s", rate / 1e6);
    assert!(rate >= 10e6, "interpreter too slow: {:.1} M steps/s", rate / 1e6);
}
