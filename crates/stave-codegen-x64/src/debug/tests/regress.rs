//! Regressions: `bool` loads, wrapping `addr.elem` offsets, the probed stack-argument area, casts
//! of fn values.

use stave_ir::{CastKind, FuncBuilder, Op};

use super::harness::*;
use crate::debug::{Error, Options, bench_ctx, compile_func, compile_module};

/// A `bool` byte read from memory is 1 iff nonzero (interpreter and AArch64 agree).
#[test]
fn bool_load_normalizes() {
    let mut p = Prog::new();
    let t = p.t;
    let pu8 = p.m.types.ref_(true, t.u8);
    let pbool = p.m.types.ref_(true, t.bool);
    p.main(|p, b| {
        let s = b.slot(pu8);
        let two = b.int(2);
        b.store(s, two);
        let q = b.cast(CastKind::Exact, pbool, s);
        let x = b.load(t.bool, q);
        p.print(b, x.into(), t.bool);
        let n = b.unary(Op::Not, t.bool, x);
        p.print(b, n.into(), t.bool);
        let y = b.binary(Op::Xor, t.bool, x, n);
        p.print(b, y.into(), t.bool);
    });
    let r = check(&p, "bool_load");
    assert_eq!(r.stdout, b"0000000000000001\n0000000000000000\n0000000000000001\n");
}

/// `addr.elem` with a constant index ≥ 2^63 wraps like the interpreter's `p + i * size`.
#[test]
fn elem_addr_constant_index_wraps() {
    let mut p = Prog::new();
    let t = p.t;
    let arr = p.m.types.array(t.u64, 4);
    let parr = p.m.types.ref_(true, arr);
    let pu64 = p.m.types.ref_(true, t.u64);
    let rp = p.m.types.ptr(true, t.u64);
    p.main(|p, b| {
        let s = b.slot(parr);
        let z = b.int(0);
        let e0 = b.op(Op::ElemAddr, pu64, &[s.into(), z]);
        let k = b.int(77);
        b.store(e0, k);
        let q = b.cast(CastKind::Exact, rp, s);
        // q[-1] (index 2^64 - 1), then one element up: back at q[0]
        let m1 = b.int(i128::from(u64::MAX));
        let before = b.op(Op::ElemAddr, rp, &[q.into(), m1]);
        let one = b.int(1);
        let back = b.op(Op::ElemAddr, rp, &[before.into(), one]);
        let x = b.load(t.u64, back);
        p.print(b, x.into(), t.u64);
        let (qa, ba) = (b.cast(CastKind::Exact, t.usize, q), b.cast(CastKind::Exact, t.usize, before));
        let d = b.binary(Op::SubWrap, t.usize, qa, ba);
        p.print(b, d.into(), t.usize);
    });
    let r = check(&p, "elem_wrap");
    assert_eq!(r.stdout, b"000000000000004d\n0000000000000008\n");
}

/// A by-value aggregate passed on the stack: the argument area is probed page by page.
#[test]
fn stack_argument_area_is_probed() {
    let mut p = Prog::new();
    let t = p.t;
    let n = 40_000u64;
    let arr = p.m.types.array(t.u8, n);
    let parr = p.m.types.ref_(true, arr);
    let pu8 = p.m.types.ref_(true, t.u8);
    let take = p.decl("t.take", &[arr], Some(t.u64));
    let mut b = FuncBuilder::new(&[arr]);
    let a = b.param(0).unwrap();
    let last = b.int(i128::from(n - 1));
    let x = b.binary(Op::Get, t.u8, a, last);
    let w = b.cast(CastKind::Exact, t.u64, x);
    b.ret(Some(w.into()));
    p.def(take, b);
    let main = p.main(|p, b| {
        let s = b.slot(parr);
        let last = b.int(i128::from(n - 1));
        let e = b.op(Op::ElemAddr, pu8, &[s.into(), last]);
        let seven = b.int(7);
        b.store(e, seven);
        let v = b.load(arr, s);
        let r = b.call(take, Some(t.u64), &[v.into()]).unwrap();
        p.print(b, r.into(), t.u64);
    });
    let r = check(&p, "stack_args_probe");
    assert_eq!(r.stdout, b"0000000000000007\n");
    // `or qword [rsp], 0`: one in the prologue's probe loop (the frame is > 8 pages), one in
    // the argument area's (also > 8 pages)
    let code = compile_func(&bench_ctx(&p.m), main).unwrap().code;
    let probe = [0x48, 0x83, 0x0c, 0x24, 0x00];
    let count = code.windows(probe.len()).filter(|w| *w == probe).count();
    assert!(count >= 2, "{count} probes");
}

/// `cast` of a fn value (code, environment) is malformed IR, never a silent truncation.
#[test]
fn cast_of_a_fn_value_is_malformed() {
    let mut p = Prog::new();
    let t = p.t;
    let fnty = p.m.types.fn_type(vec![], t.u64, stave_target::FeatureSet::EMPTY);
    let f = p.decl("t.c", &[fnty], Some(t.usize));
    let mut b = FuncBuilder::new(&[fnty]);
    let x = b.param(0).unwrap();
    let y = b.cast(CastKind::Wrap, t.usize, x);
    b.ret(Some(y.into()));
    p.def(f, b);
    p.main(|_, _| {});
    let e = compile_module(&p.m, &Options { verify: false, ..Options::default() }).unwrap_err();
    assert!(matches!(&e, Error::Malformed { func: Some(g), .. } if *g == f), "{e}");
    let e = compile_module(&p.m, &Options::default()).unwrap_err();
    assert!(matches!(&e, Error::Verify(_)), "{e}");
}
