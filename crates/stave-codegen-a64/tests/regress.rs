//! Regressions: scratch registers across far frame offsets, frame-size overflow, functions over
//! the conditional-branch range, user fns named like compiler symbols, the thread child's stack
//! alignment, references to fns without a body, zero-size statics.

mod common;

use common::*;
use stave_codegen_a64::debug::{BuildConfig, ErrorKind, build_image};
use stave_ir::{CastKind, Op, Operand};

/// fn values compared when their homes are beyond the scaled-offset range (≥ 32 KiB from sp):
/// loading the second operand must not clobber the first one's words.
#[test]
fn fn_value_compare_in_a_large_frame() {
    let mut p = Prog::new();
    let t = p.t;
    let fnty = p.m.types.fn_type(vec![t.i64], t.i64, stave_target::FeatureSet::EMPTY);
    let body = |b: &mut stave_ir::FuncBuilder, _: &Prog| {
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
    };
    let k = p.func("app.k", &[t.i64, t.usize], Some(t.i64), body);
    let k2 = p.func("app.k2", &[t.i64, t.usize], Some(t.i64), body);
    let buf = p.m.types.array(t.u8, 40000);
    let pbuf = p.m.types.ref_(true, buf);
    let pu8 = p.m.types.ref_(true, t.u8);
    p.set_main(|b, p| {
        let s = b.slot(pbuf);
        let z = b.int(0);
        let e = b.op(Op::ElemAddr, pu8, &[s.into(), z]);
        let one = b.int(1);
        b.store(e, one);
        let (e7, e8) = (b.int(7), b.int(8));
        let f = b.fn_make(fnty, k, e7);
        let g = b.fn_make(fnty, k, e7);
        let h = b.fn_make(fnty, k, e8);
        let h2 = b.fn_make(fnty, k2, e7);
        for (x, y) in [(f, g), (f, h), (f, h2), (h2, h2)] {
            for op in [Op::Eq, Op::Ne] {
                let c = b.cmp(op, t.bool, x, y);
                p.out(b, c, t.bool);
            }
        }
    });
    let r = check(&p, "fn_eq_far");
    let words: Vec<u8> = r.stdout.chunks(8).map(|c| c[0]).collect();
    assert_eq!(words, [1, 0, 0, 1, 0, 1, 1, 0]);
}

/// Slot sizes add up past `u32`: rejected like any frame of 16 MiB or more, never wrapped.
#[test]
fn frame_size_overflow_is_rejected() {
    let mut p = Prog::new();
    let t = p.t;
    let big = p.m.types.array(t.u8, 1 << 30);
    let pbig = p.m.types.ref_(true, big);
    let pu8 = p.m.types.ref_(true, t.u8);
    let f = p.func("app.huge", &[], None, |b, _| {
        for _ in 0..4 {
            let s = b.slot(pbig);
            let z = b.int(0);
            let e = b.op(Op::ElemAddr, pu8, &[s.into(), z]);
            let one = b.int(1);
            b.store(e, one);
        }
        b.ret(None);
    });
    p.set_main(|b, _| {
        b.call(f, None, &[]);
    });
    let e = build_image(&p.m, &BuildConfig::new(None)).unwrap_err();
    assert!(matches!(&e.kind, ErrorKind::Unsupported(s) if s.contains("16 MiB")), "{e}");
    assert_eq!(e.func, Some(f));
}

/// A function with more than 1 MiB of code: `b.cond` to its trap stub (and blocks) is out of
/// the ±1 MiB range, so the function is emitted with long branches.
#[test]
fn function_over_one_mib() {
    let mut p = Prog::new();
    let t = p.t;
    let big = p.func("app.big", &[t.u64], Some(t.u64), |b, _| {
        let (body, done) = (b.create_block(), b.create_block());
        let x0 = b.param(0).unwrap();
        let z = b.int(0);
        let c = b.cmp(Op::Eq, t.bool, x0, z);
        b.cbr(c, done, &[], body, &[]);
        b.switch_to(body);
        let mut x: Operand = x0.into();
        for _ in 0..200_000 {
            let one = b.int(1);
            x = b.binary(Op::Add, t.u64, x, one).into();
        }
        b.ret(Some(x));
        b.switch_to(done);
        let k = b.int(42);
        b.ret(Some(k));
    });
    p.set_main(|b, p| {
        for v in [5u64, 0, u64::MAX - 10] {
            let a = b.int(i128::from(v));
            let r = b.call(big, Some(t.u64), &[a]).unwrap();
            p.out(b, r, t.u64);
        }
    });
    let r = check(&p, "over_1mib");
    assert_eq!(r.code, 101, "the last call overflows");
    assert_eq!(r.stdout.len(), 16);
}

/// User functions whose paths are compiler symbol names get `path@instance` symbols.
#[test]
fn reserved_fn_names() {
    let mut p = Prog::new();
    let t = p.t;
    let mut fs = Vec::new();
    for (k, path) in ["__stave_fmod", "_start", "__stave_main", "__stave_strings"].into_iter().enumerate() {
        fs.push(p.func(path, &[], Some(t.u64), move |b, _| {
            let v = b.int(100 + k as i128);
            b.ret(Some(v));
        }));
    }
    p.set_main(|b, p| {
        for &f in &fs {
            let r = b.call(f, Some(t.u64), &[]).unwrap();
            p.out(b, r, t.u64);
        }
        // the real __stave_fmod still backs float `rem`
        let (x, y) = (b.f64(7.5), b.f64(2.0));
        let r = b.binary(Op::Rem, t.f64, x, y);
        p.out(b, r, t.f64);
    });
    check(&p, "reserved_names");
}

/// The thread child aligns its stack down to 16: the child's frame (a slot at `sp`) is aligned
/// even when the stack top passed to `clone` is not. Native only (the interpreter has one thread).
#[test]
fn thread_child_stack_is_aligned() {
    let mut p = Prog::new();
    let t = p.t;
    let us = t.usize;
    let sys6 = p.m.intrinsic("syscall6");
    let sys0 = p.m.intrinsic("syscall0");
    let clone = p.m.intrinsic("thread_clone");
    let load = p.m.intrinsic("atomic_load_u64");
    let xchg = p.m.intrinsic("atomic_xchg_u64");
    let pu64 = p.m.types.ref_(true, t.u64);
    p.func("runtime.thread.entry", &[us], Some(us), |b, _| {
        let arg = b.param(0).unwrap();
        let s = b.slot(pu64);
        let a = b.cast(CastKind::Exact, us, s);
        let m = b.int(15);
        let lo = b.binary(Op::And, us, a, m);
        let one = b.int(1);
        let v = b.binary(Op::AddWrap, us, lo, one);
        b.intrinsic(xchg, Some(t.u64), &[arg.into(), v.into()]);
        let z = b.int(0);
        b.ret(Some(z));
    });
    p.set_main(|b, p| {
        let ops: Vec<Operand> =
            [222i128, 0, 65536, 3, 0x22, i128::from(u64::MAX), 0].iter().map(|&k| b.int(k)).collect();
        let base = b.intrinsic(sys6, Some(t.isize), &ops).unwrap();
        let base = b.cast(CastKind::Wrap, us, base);
        let off = b.int(65536 - 8);
        let sp = b.binary(Op::AddWrap, us, base, off);
        let (nr, flags, z) = (b.int(220), b.int(0x50f00), b.int(0));
        b.intrinsic(clone, Some(t.isize), &[nr, flags, sp.into(), z, z, z, base.into()]);
        let (head, wait, done) = (b.create_block(), b.create_block(), b.create_block());
        b.br(head, &[]);
        b.switch_to(head);
        let v = b.intrinsic(load, Some(t.u64), &[base.into()]).unwrap();
        let z = b.int(0);
        let c = b.cmp(Op::Eq, t.bool, v, z);
        b.cbr(c, wait, &[], done, &[]);
        b.switch_to(wait);
        let y = b.int(124);
        b.intrinsic(sys0, Some(t.isize), &[y]);
        b.br(head, &[]);
        b.switch_to(done);
        p.out(b, v, t.u64);
    });
    if let Some(r) = native(&p.m, "thread_align", false) {
        assert_eq!(r.code, 0, "{r:?}");
        assert_eq!(r.stdout, 1u64.to_le_bytes(), "child frame address mod 16, plus 1");
    }
}

/// A call of a fn without a body (an `extern "C"` import) is rejected as unsupported before
/// anything is linked, at the calling fn, as on x86-64.
#[test]
fn reference_to_bodiless_fn_is_unsupported() {
    let mut p = Prog::new();
    let t = p.t;
    let ext = p.declare("app.puts", &[t.u64], Some(t.u64));
    p.set_main(|b, _| {
        let x = b.int(1);
        b.call(ext, Some(t.u64), &[x]);
    });
    let e = build_image(&p.m, &BuildConfig::new(Some(p.main))).unwrap_err();
    assert!(matches!(&e.kind, ErrorKind::Unsupported(s) if s.contains("`app.puts` without a body")), "{e}");
    assert_eq!(e.func, Some(p.main));
}
