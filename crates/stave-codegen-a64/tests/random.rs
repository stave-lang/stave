//! Random programs, native (qemu) vs the interpreter: straight-line and branchy (if/else joins,
//! switches, bounded loops with block params) int and float code with casts, selects,
//! aggregates, stack slots, and calls between the generated functions. Trapping ops are allowed
//! (less often than wrapping/saturating ones): the output up to the trap, the trap kind and the
//! site must match too.

mod common;

use common::rgen::{program, program_stress};
use common::*;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn random_programs(seed in any::<u64>()) {
        let p = program(seed);
        check(&p, &format!("rnd-{seed:x}"));
    }

    /// The stress shape: big frames, many params, deep nesting, fn values.
    #[test]
    fn random_stress_programs(seed in any::<u64>()) {
        let p = program_stress(seed);
        check(&p, &format!("stress-{seed:x}"));
    }
}

/// A fixed set of stress-shape seeds.
#[test]
fn random_stress_fixed_seeds() {
    for seed in 0..60u64 {
        let p = program_stress(seed * 104_729 + 3);
        check(&p, &format!("stress-seed-{seed}"));
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
    println!("{}", stave_ir::print_checked(&p.m).unwrap_or_else(|e| format!("{e:?}")));
    check(&p, "dbg");
}

/// [`debug_seed`] for the stress shape: `STAVE_SEED=<n> cargo test --test random
/// debug_stress_seed -- --ignored --nocapture`.
#[test]
#[ignore]
fn debug_stress_seed() {
    let seed: u64 = std::env::var("STAVE_SEED").unwrap().parse().unwrap();
    let p = program_stress(seed);
    println!("{}", stave_ir::print_checked(&p.m).unwrap_or_else(|e| format!("{e:?}")));
    check(&p, "dbg");
}
