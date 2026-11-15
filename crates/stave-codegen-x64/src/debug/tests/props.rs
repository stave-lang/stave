//! Property test: random straight-line and branchy integer / float programs (`gen`) give the
//! same stdout and exit status natively as in the interpreter.

use proptest::prelude::*;

use super::harness::{Prog, check};
use super::randgen::Gen;

fn program(seed: u64, size: usize, traps: bool) -> Prog {
    let mut p = Prog::new();
    let (u64_, i32_) = (p.t.u64, p.t.i32);
    let (f, args, mut mb) = Gen::new(&mut p, seed, size, traps, true).function("t.rand");
    let r = mb.call(f, Some(u64_), &args).unwrap();
    p.print(&mut mb, r.into(), u64_);
    let z = mb.int(0);
    mb.ret(Some(z));
    let main = p.decl("t.main", &[], Some(i32_));
    p.def(main, mb);
    p.set_main(main);
    p
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, max_shrink_iters: 50, ..ProptestConfig::default() })]

    #[test]
    fn random_programs(seed in any::<u64>(), size in 1usize..70, traps in any::<bool>()) {
        let p = program(seed, size, traps);
        check(&p, &format!("rand-{seed:x}-{size}-{traps}"));
    }
}

/// A fixed set of larger programs (`STAVE_RAND_LARGE=n` runs more).
#[test]
fn random_programs_large() {
    let n: u64 = std::env::var("STAVE_RAND_LARGE").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    for seed in 0..n {
        let p = program(seed * 7919 + 1, 150 + (seed as usize * 13) % 200, seed % 3 == 0);
        let r = check(&p, &format!("rand-large-{seed}"));
        let insts: usize = p.m.funcs.iter().filter_map(|f| f.body.as_ref()).map(|b| b.insts.len()).sum();
        let blocks: usize = p.m.funcs.iter().filter_map(|f| f.body.as_ref()).map(|b| b.blocks.len()).sum();
        if std::env::var_os("STAVE_TEST_VERBOSE").is_some() {
            eprintln!(
                "seed {seed}: {insts} insts, {blocks} blocks, {} output lines, exit {}",
                r.stdout.iter().filter(|c| **c == b'\n').count(),
                r.code
            );
        }
    }
}
