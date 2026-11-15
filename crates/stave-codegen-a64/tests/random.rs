//! Random programs, native (qemu) vs the interpreter: straight-line and branchy (if/else joins,
//! switches, bounded loops with block params) int and float code with casts, selects,
//! aggregates, stack slots, and calls between the generated functions. Trapping ops are allowed
//! (less often than wrapping/saturating ones): the output up to the trap, the trap kind and the
//! site must match too.

mod common;

use common::rgen::program;
use common::*;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn random_programs(seed in any::<u64>()) {
        let p = program(seed);
        check(&p, &format!("rnd-{seed:x}"));
    }
}

/// A fixed set of seeds (fast, deterministic regression set).
#[test]
fn random_fixed_seeds() {
    for seed in 0..40u64 {
        let p = program(seed * 7919);
        check(&p, &format!("seed-{seed}"));
    }
}

/// `STAVE_SEED=<n> cargo test --test random debug_seed -- --ignored --nocapture`: prints the IR.
#[test]
#[ignore]
fn debug_seed() {
    let seed: u64 = std::env::var("STAVE_SEED").unwrap().parse().unwrap();
    let p = program(seed);
    println!("{}", stave_ir::print_checked(&p.m));
    check(&p, "dbg");
}
