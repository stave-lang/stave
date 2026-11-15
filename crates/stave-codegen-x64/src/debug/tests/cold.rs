//! Cold paths. A release build lays a function out as its hot blocks, then the blocks that
//! cannot return (a `panic.*`, a call of a fn that never returns), then the trap stubs: a loop
//! with a bounds check, a trapping add, an assert, and such a call has no panic code between its
//! head and its back edge. A debug build keeps the trap stubs of the checks out of line.

use stave_ir::{FuncBuilder, Op, PanicKind};

use super::harness::*;
use crate::debug::{Options, compile_module};

fn program() -> Prog {
    let mut p = Prog::new();
    let t = p.t;
    let die = p.decl("t.die", &[], None);
    let mut b = FuncBuilder::new(&[]);
    let (nr, code) = (b.int(231), b.int(77));
    b.intrinsic(p.sys1, Some(t.isize), &[nr, code]);
    b.unreachable();
    p.def(die, b);

    let f = p.decl("t.f", &[t.u64], Some(t.u64));
    let mut b = FuncBuilder::new(&[t.u64]);
    let n = b.param(0).unwrap();
    let (head, hp) = b.create_block_with(&[t.u64, t.u64]);
    let (body, fail, cont, assert_fail, rest, exit) = (
        b.create_block(),
        b.create_block(),
        b.create_block(),
        b.create_block(),
        b.create_block(),
        b.create_block(),
    );
    let (z, z2) = (b.int(0), b.int(0));
    b.br(head, &[z, z2]);
    b.switch_to(head);
    let (i, acc) = (hp[0], hp[1]);
    let c = b.cmp(Op::Lt, t.bool, i, n);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    let k = b.int(1000);
    let far = b.cmp(Op::Gt, t.bool, i, k);
    b.cbr(far, fail, &[], cont, &[]);
    b.switch_to(fail);
    b.call(die, None, &[]);
    b.unreachable();
    b.switch_to(cont);
    let k = b.int(12403);
    let ok = b.cmp(Op::Ne, t.bool, acc, k);
    b.cbr(ok, rest, &[], assert_fail, &[]);
    b.switch_to(assert_fail);
    b.panic(PanicKind::Assert);
    b.unreachable();
    b.switch_to(rest);
    let len = b.int(4096);
    b.bounds_check(i, len);
    let acc2 = b.binary(Op::Add, t.u64, acc, i);
    let one = b.int(1);
    let i2 = b.binary(Op::AddWrap, t.u64, i, one);
    b.br(head, &[i2.into(), acc2.into()]);
    b.switch_to(exit);
    b.ret(Some(acc.into()));
    p.def(f, b);

    p.main(|p, b| {
        for n in [0i128, 1, 100] {
            let c = b.int(n);
            let r = b.call(f, Some(t.u64), &[c]).unwrap();
            p.print(b, r.into(), t.u64);
        }
        // the assert fails when the sum is 12403, before iteration 158
        let c = b.int(500);
        let r = b.call(f, Some(t.u64), &[c]).unwrap();
        p.print(b, r.into(), t.u64);
    });
    p
}

/// Offsets of the calls of `runtime.panic.trap` and of `t.die` in `t.f`, and of its `leave; ret`.
fn layout(p: &Prog, opts: Options) -> (Vec<u32>, Vec<u32>, u32) {
    let img = compile_module(&p.m, &Options { main: p.main, ..opts }).unwrap();
    let code = img.func_code(img.lookup("t.f").unwrap()).unwrap();
    let (trap, die) = (img.lookup("runtime.panic.trap").unwrap(), img.lookup("t.die").unwrap());
    let at = |sym| code.relocs.iter().filter(|r| r.sym == sym).map(|r| r.offset).collect::<Vec<u32>>();
    let ret = code.code.windows(2).position(|w| w == [0xC9, 0xC3]).expect("leave; ret") as u32;
    (at(trap), at(die), ret)
}

#[test]
fn panic_code_follows_the_hot_code() {
    let p = program();
    let r = check(&p, "cold");
    assert_eq!(r.code, 100 + kind_index(PanicKind::Assert));
    assert_eq!(r.stdout, b"0000000000000000\n0000000000000000\n0000000000001356\n");

    let (traps, dies, ret) = layout(&p, release_options());
    assert_eq!((traps.len(), dies.len()), (3, 1), "bounds, overflow and assert stubs; one call of die");
    assert!(traps.iter().chain(&dies).all(|&o| o > ret), "panic code at {traps:?} {dies:?}, ret at {ret}");

    // debug: the stubs of the checks are out of line; blocks stay in reverse postorder
    let (traps, dies, ret) = layout(&p, Options::default());
    assert_eq!((traps.len(), dies.len()), (3, 1));
    assert!(traps.iter().all(|&o| o > ret), "trap stubs at {traps:?}, ret at {ret}");
}
