//! Whole programs, native (qemu) vs the interpreter: strings and syscalls, recursion, loops with
//! block-param permutations, switch, aggregates, enums, memory ops, calls (many args, fat and
//! aggregate args/results, fn values), traps of every kind.

mod common;

use common::*;
use stave_ir::{AdtShape, CastKind, DeclId, FuncBuilder, GenericArg, Op, Operand, PanicKind, Site, TypeId};

#[test]
fn hello_and_exit_code() {
    let mut p = Prog::new();
    let s = p.m.add_string("hello, world\n");
    let sys1 = p.sys1;
    let isize_t = p.t.isize;
    p.set_main(|b, p| {
        b.call(p.out_str, None, &[b.str(s)]);
        let nr = b.int(94);
        let code = b.int(42);
        b.intrinsic(sys1, Some(isize_t), &[nr, code]);
    });
    let r = check(&p, "hello");
    assert_eq!(r.stdout, b"hello, world\n");
    assert_eq!(r.code, 42);
    if can_run() {
        let n = native(&p.m, "hello-native").unwrap();
        assert_eq!((n.stdout.as_slice(), n.code), (&b"hello, world\n"[..], 42));
    }
}

#[test]
fn fib_recursion() {
    let mut p = Prog::new();
    let u64t = p.t.u64;
    let fib = p.declare("app.fib", &[u64t], Some(u64t));
    p.define(fib, &[u64t], |b, p| {
        let n = b.param(0).unwrap();
        let (base, rec) = (b.create_block(), b.create_block());
        let two = b.int(2);
        let c = b.cmp(Op::Lt, p.t.bool, n, two);
        b.cbr(c, base, &[], rec, &[]);
        b.switch_to(base);
        b.ret(Some(n.into()));
        b.switch_to(rec);
        let one = b.int(1);
        let n1 = b.binary(Op::Sub, u64t, n, one);
        let two = b.int(2);
        let n2 = b.binary(Op::Sub, u64t, n, two);
        let a = b.call(fib, Some(u64t), &[n1.into()]).unwrap();
        let c2 = b.call(fib, Some(u64t), &[n2.into()]).unwrap();
        let s = b.binary(Op::Add, u64t, a, c2);
        b.ret(Some(s.into()));
    });
    p.set_main(|b, p| {
        for k in [0, 1, 2, 10, 20] {
            let n = b.int(k);
            let r = b.call(fib, Some(u64t), &[n]).unwrap();
            p.out(b, r, u64t);
        }
    });
    let r = check(&p, "fib");
    assert_eq!(r.stdout.len(), 5 * 8);
}

/// `while i < n { (a, b) = (b, a + b); i += 1 }` with a block-param swap, plus a rotation of three
/// params, which needs the parallel-copy scratch area.
#[test]
fn loop_param_permutations() {
    let mut p = Prog::new();
    let u64t = p.t.u64;
    let f = p.func("app.rot", &[u64t], Some(u64t), |b, p| {
        let n = b.param(0).unwrap();
        let (head, hp) = b.create_block_with(&[u64t, u64t, u64t, u64t, u64t]);
        let body = b.create_block();
        let (exit, ep) = b.create_block_with(&[u64t]);
        let (z, one, two, three) = (b.int(0), b.int(1), b.int(2), b.int(3));
        b.br(head, &[z, one, two, three, n.into()]);
        b.switch_to(head);
        let i = hp[0];
        let c = b.cmp(Op::Lt, p.t.bool, i, hp[4]);
        b.cbr(c, body, &[], exit, &[hp[1].into()]);
        b.switch_to(body);
        let one = b.int(1);
        let i1 = b.binary(Op::Add, u64t, i, one);
        let s = b.binary(Op::AddWrap, u64t, hp[1], hp[2]);
        let s = b.binary(Op::MulWrap, u64t, s, hp[3]);
        // rotate (x, y, z) -> (y, z, x + y*z)
        b.br(head, &[i1.into(), hp[2].into(), hp[3].into(), s.into(), hp[4].into()]);
        b.switch_to(exit);
        b.ret(Some(ep[0].into()));
    });
    p.set_main(|b, p| {
        for k in [0, 1, 2, 5, 17, 40] {
            let n = b.int(k);
            let r = b.call(f, Some(u64t), &[n]).unwrap();
            p.out(b, r, u64t);
        }
    });
    check(&p, "rot");
}

#[test]
fn switch_and_select() {
    let mut p = Prog::new();
    let (i32t, u8t) = (p.t.i32, p.t.u8);
    let f = p.func("app.sw", &[i32t], Some(i32t), |b, _| {
        let x = b.param(0).unwrap();
        let (b1, b2, b3) = (b.create_block(), b.create_block(), b.create_block());
        let (join, jp) = b.create_block_with(&[i32t]);
        let c1 = b.int(100);
        let c2 = b.int(-7);
        b.switch(x, &[(0, b1, &[]), (-7, b2, &[]), (100_000, b3, &[]), (1, join, &[c1])], join, &[c2]);
        b.switch_to(b1);
        let k = b.int(11);
        b.br(join, &[k]);
        b.switch_to(b2);
        let k = b.int(22);
        let y = b.binary(Op::Mul, i32t, x, k);
        b.br(join, &[y.into()]);
        b.switch_to(b3);
        let k = b.int(3);
        let y = b.binary(Op::Div, i32t, x, k);
        b.br(join, &[y.into()]);
        b.switch_to(join);
        b.ret(Some(jp[0].into()));
    });
    let g = p.func("app.sel", &[u8t, u8t], Some(u8t), |b, p| {
        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
        let c = b.cmp(Op::Gt, p.t.bool, x, y);
        let r = b.select(u8t, c, x, y);
        b.ret(Some(r.into()));
    });
    p.set_main(|b, p| {
        for k in [0, -7, 100_000, 1, 5, -1] {
            let n = b.int(k);
            let r = b.call(f, Some(i32t), &[n]).unwrap();
            p.out(b, r, i32t);
        }
        for (x, y) in [(1, 2), (200, 100), (255, 255)] {
            let (x, y) = (b.int(x), b.int(y));
            let r = b.call(g, Some(u8t), &[x, y]).unwrap();
            p.out(b, r, u8t);
        }
    });
    check(&p, "switch");
}

fn struct_ty(p: &mut Prog, d: u32, fields: Vec<TypeId>) -> TypeId {
    p.m.types.adt(DeclId(d), vec![], AdtShape::structure(fields))
}

/// Aggregates: make/get/set (constant and dynamic indices), structs by value through calls
/// (by-reference args, x8 results), fat pointers, enum make/tag/get, Option from cast.try.
#[test]
fn aggregates_and_enums() {
    let mut p = Prog::new();
    let t = p.t;
    let arr = p.m.types.array(t.i64, 5);
    let pair = p.m.types.tuple(vec![t.u8, t.f64]);
    let st = struct_ty(&mut p, 1, vec![t.u32, pair, arr, t.str_ref]);
    let opt = p.m.types.adt(
        DeclId(2),
        vec![GenericArg::Type(t.i64)],
        AdtShape::enumeration(vec![vec![], vec![t.i64]]),
    );
    let en = p.m.types.adt(
        DeclId(3),
        vec![],
        AdtShape::enumeration(vec![vec![t.u8], vec![], vec![t.f32, t.u16, st]]),
    );
    let s_hello = p.m.add_string("agg\n");
    // fn mk(a: u32, x: f64, k: i64) -> S
    let mk = p.func("app.mk", &[t.u32, t.f64, t.i64], Some(st), |b, _| {
        let (a, x, k) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
        let seven = b.int(7);
        let pr = b.op(Op::Make, pair, &[seven, x.into()]);
        let one = b.int(1);
        let k1 = b.binary(Op::Add, t.i64, k, one);
        let m3 = b.int(-3);
        let z = b.int(0);
        let ar = b.op(Op::Make, arr, &[k.into(), k1.into(), m3, z, k.into()]);
        let s = b.op(Op::Make, st, &[a.into(), pr.into(), ar.into(), b.str(s_hello)]);
        b.ret(Some(s.into()));
    });
    // fn sum(s: S, i: usize) -> i64 { s.2[i] + s.2[4] }   (dynamic get traps Bounds for i >= 5)
    let sum = p.func("app.sum", &[st, t.usize], Some(t.i64), |b, _| {
        let (s, i) = (b.param(0).unwrap(), b.param(1).unwrap());
        let two = b.int(2);
        let ar = b.binary(Op::Get, arr, s, two);
        b.set_site(Site(77));
        let e = b.binary(Op::Get, t.i64, ar, i);
        let four = b.int(4);
        let e4 = b.binary(Op::Get, t.i64, ar, four);
        let r = b.binary(Op::AddWrap, t.i64, e, e4);
        b.ret(Some(r.into()));
    });
    // fn upd(s: S, i: usize, v: i64) -> S { s.2[i] = v; s.0 = 9; s }
    let upd = p.func("app.upd", &[st, t.usize, t.i64], Some(st), |b, _| {
        let (s, i, v) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
        let two = b.int(2);
        let ar = b.binary(Op::Get, arr, s, two);
        let ar2 = b.op(Op::Set, arr, &[ar.into(), i.into(), v.into()]);
        let s2 = b.op(Op::Set, st, &[s.into(), two, ar2.into()]);
        let z = b.int(0);
        let nine = b.int(9);
        let s3 = b.op(Op::Set, st, &[s2.into(), z, nine]);
        b.ret(Some(s3.into()));
    });
    // fn classify(e: E) -> f64
    let cls = p.func("app.cls", &[en], Some(t.f64), |b, _| {
        let e = b.param(0).unwrap();
        let tag = b.unary(Op::EnumTag, t.u32, e);
        let (v0, v1, v2) = (b.create_block(), b.create_block(), b.create_block());
        b.switch(tag, &[(0, v0, &[]), (1, v1, &[])], v2, &[]);
        b.switch_to(v0);
        let (z, z2) = (b.int(0), b.int(0));
        let x = b.op(Op::EnumGet, t.u8, &[e.into(), z, z2]);
        let f = b.cast(CastKind::Exact, t.f64, x);
        b.ret(Some(f.into()));
        b.switch_to(v1);
        let m = b.f64(-1.5);
        b.ret(Some(m));
        b.switch_to(v2);
        let (two, z) = (b.int(2), b.int(0));
        let a = b.op(Op::EnumGet, t.f32, &[e.into(), two, z]);
        let (two, one) = (b.int(2), b.int(1));
        let c = b.op(Op::EnumGet, t.u16, &[e.into(), two, one]);
        let (two, k2) = (b.int(2), b.int(2));
        let s = b.op(Op::EnumGet, st, &[e.into(), two, k2]);
        let z = b.int(0);
        let s0 = b.binary(Op::Get, t.u32, s, z);
        let af = b.cast(CastKind::Exact, t.f64, a);
        let cf = b.cast(CastKind::Exact, t.f64, c);
        let sf = b.cast(CastKind::Exact, t.f64, s0);
        let r = b.binary(Op::Add, t.f64, af, cf);
        let r = b.binary(Op::Mul, t.f64, r, sf);
        b.ret(Some(r.into()));
    });
    p.set_main(|b, p| {
        let (a, x, k) = (b.int(5), b.f64(2.25), b.int(-40));
        let s = b.call(mk, Some(st), &[a, x, k]).unwrap();
        let three = b.int(3);
        let fat = b.binary(Op::Get, t.str_ref, s, three);
        b.call(p.out_str, None, &[fat.into()]);
        let one = b.int(1);
        let pr = b.binary(Op::Get, pair, s, one);
        let z = b.int(0);
        let pu = b.binary(Op::Get, t.u8, pr, z);
        p.out(b, pu, t.u8);
        let one = b.int(1);
        let pf = b.binary(Op::Get, t.f64, pr, one);
        p.out(b, pf, t.f64);
        for i in [0, 1, 4] {
            let i = b.int(i);
            let r = b.call(sum, Some(t.i64), &[s.into(), i]).unwrap();
            p.out(b, r, t.i64);
        }
        let (i, v) = (b.int(1), b.int(1000));
        let s2 = b.call(upd, Some(st), &[s.into(), i, v]).unwrap();
        let i = b.int(1);
        let r = b.call(sum, Some(t.i64), &[s2.into(), i]).unwrap();
        p.out(b, r, t.i64);
        let r = b.call(sum, Some(t.i64), &[s.into(), i]).unwrap();
        p.out(b, r, t.i64);
        let z = b.int(0);
        let s20 = b.binary(Op::Get, t.u32, s2, z);
        p.out(b, s20, t.u32);
        // enums
        let (vz, x) = (b.int(0), b.int(200));
        let e0 = b.op(Op::EnumMake, en, &[vz, x]);
        let one = b.int(1);
        let e1 = b.op(Op::EnumMake, en, &[one]);
        let (two, fa, c) = (b.int(2), b.f32(0.5), b.int(65535));
        let e2 = b.op(Op::EnumMake, en, &[two, fa, c, s2.into()]);
        for e in [e0, e1, e2] {
            let r = b.call(cls, Some(t.f64), &[e.into()]).unwrap();
            p.out(b, r, t.f64);
            let tag = b.unary(Op::EnumTag, t.u8, e);
            p.out(b, tag, t.u8);
        }
        // Option from cast.try, then a match on it
        for k in [5i128, -129, 127, 300] {
            let k = b.int(k);
            let kv = b.cast(CastKind::Exact, t.i64, k);
            let small = b.cast(CastKind::Try, opt, kv);
            let tag = b.unary(Op::EnumTag, t.u32, small);
            p.out(b, tag, t.u32);
            let _ = small;
        }
        // dynamic index out of bounds: trap Bounds at site 77
        let i = b.int(5);
        b.call(sum, Some(t.i64), &[s.into(), i]);
    });
    let r = check(&p, "agg");
    assert_eq!(r.code, 101);
    assert_eq!(r.stderr, trap_bytes(PanicKind::Bounds, 77));
}

/// Slots, loads, stores, field/elem/enum-field addresses, slices over a slot array.
#[test]
fn memory_ops() {
    let mut p = Prog::new();
    let t = p.t;
    let arr = p.m.types.array(t.u16, 6);
    let st = struct_ty(&mut p, 5, vec![t.u8, arr, t.i64]);
    let pst = p.m.types.ref_(true, st);
    let parr = p.m.types.ref_(true, arr);
    let pu16 = p.m.types.ref_(true, t.u16);
    let rpu16 = p.m.types.ptr(true, t.u16);
    let pi64 = p.m.types.ref_(true, t.i64);
    let sl = p.m.types.slice(t.u16);
    let psl = p.m.types.ref_(false, sl);
    let en = p.m.types.adt(DeclId(6), vec![], AdtShape::enumeration(vec![vec![t.u8], vec![t.u32, t.i64]]));
    let pen = p.m.types.ref_(true, en);
    let pu32 = p.m.types.ref_(true, t.u32);
    // fn total(s: &[u16]) -> u64: sum over the slice with a loop and check.bounds
    let total = p.func("app.total", &[psl], Some(t.u64), |b, p| {
        let s = b.param(0).unwrap();
        let n = b.unary(Op::SliceLen, t.usize, s);
        let (head, hp) = b.create_block_with(&[t.usize, t.u64]);
        let body = b.create_block();
        let (exit, ep) = b.create_block_with(&[t.u64]);
        let (z, z2) = (b.int(0), b.int(0));
        b.br(head, &[z, z2]);
        b.switch_to(head);
        let c = b.cmp(Op::Lt, p.t.bool, hp[0], n);
        b.cbr(c, body, &[], exit, &[hp[1].into()]);
        b.switch_to(body);
        b.bounds_check(hp[0], n);
        let e = b.binary(Op::ElemAddr, pu16, s, hp[0]);
        let x = b.load(t.u16, e);
        let xw = b.cast(CastKind::Exact, t.u64, x);
        let acc = b.binary(Op::Add, t.u64, hp[1], xw);
        let one = b.int(1);
        let i1 = b.binary(Op::Add, t.usize, hp[0], one);
        b.br(head, &[i1.into(), acc.into()]);
        b.switch_to(exit);
        b.ret(Some(ep[0].into()));
    });
    p.set_main(|b, p| {
        let sp = b.slot(pst);
        let ep = b.slot(pen);
        let (a, z) = (b.int(3), b.int(0));
        let zero_arr = b.op(Op::Make, arr, &[z, z, z, z, z, z]);
        let m = b.int(-5);
        let v = b.op(Op::Make, st, &[a, zero_arr.into(), m]);
        b.store(sp, v);
        let one = b.int(1);
        let pa = b.binary(Op::FieldAddr, parr, sp, one);
        for k in 0..6i128 {
            let kk = b.int(k);
            let e = b.binary(Op::ElemAddr, pu16, pa, kk);
            let val = b.int(1000 * k + 7);
            b.store(e, val);
        }
        // raw pointer arithmetic: elem 2 via a raw *mut u16 and a dynamic index
        let raw = b.cast(CastKind::Exact, rpu16, pa);
        let i0 = b.int(0);
        let base = b.binary(Op::ElemAddr, rpu16, raw, i0);
        let two = b.int(2);
        let two_u = b.cast(CastKind::Exact, t.usize, two);
        let e2 = b.binary(Op::ElemAddr, rpu16, base, two_u);
        let x2 = b.load(t.u16, e2);
        p.out(b, x2, t.u16);
        let six = b.int(6);
        let p0 = b.binary(Op::ElemAddr, pu16, pa, i0);
        let sl_v = b.binary(Op::SliceMake, psl, p0, six);
        let s = b.call(total, Some(t.u64), &[sl_v.into()]).unwrap();
        p.out(b, s, t.u64);
        let two = b.int(2);
        let pi = b.binary(Op::FieldAddr, pi64, sp, two);
        let x = b.load(t.i64, pi);
        p.out(b, x, t.i64);
        let whole = b.load(st, sp);
        let z = b.int(0);
        let f0 = b.binary(Op::Get, t.u8, whole, z);
        p.out(b, f0, t.u8);
        // enum through memory, write a field through addr.enum_field
        let (one, w, q) = (b.int(1), b.int(77), b.int(-9));
        let e = b.op(Op::EnumMake, en, &[one, w, q]);
        b.store(ep, e);
        let (one, z) = (b.int(1), b.int(0));
        let f = b.op(Op::EnumFieldAddr, pu32, &[ep.into(), one, z]);
        let nv = b.int(123456);
        b.store(f, nv);
        let e2 = b.load(en, ep);
        let (one, z) = (b.int(1), b.int(0));
        let g = b.op(Op::EnumGet, t.u32, &[e2.into(), one, z]);
        p.out(b, g, t.u32);
        let tag = b.unary(Op::EnumTag, t.u32, e2);
        p.out(b, tag, t.u32);
        // a too-long slice: check.bounds traps at site 5
        let seven = b.int(7);
        let bad_sl = b.binary(Op::SliceMake, psl, p0, seven);
        b.set_site(Site(5));
        let n = b.unary(Op::SliceLen, t.usize, bad_sl);
        let six = b.int(6);
        b.bounds_check(n, six);
    });
    let r = check(&p, "mem");
    assert_eq!(r.stderr, trap_bytes(PanicKind::Bounds, 5));
}

/// Calls with 10 ints, 10 floats, fat pointers and aggregates past the registers (stack args),
/// through direct calls and fn values.
#[test]
fn many_args_and_fn_values() {
    let mut p = Prog::new();
    let t = p.t;
    let pair = p.m.types.tuple(vec![t.i32, t.i32]);
    let mut params = vec![];
    for k in 0..10 {
        params.push(if k % 3 == 0 { t.i64 } else { t.u8 });
        params.push(if k % 2 == 0 { t.f64 } else { t.f32 });
    }
    params.push(t.str_ref);
    params.push(pair);
    params.push(t.str_ref);
    let np = params.len();
    let ps2 = params.clone();
    let big = p.func("app.big", &params, Some(t.f64), |b, _| {
        let mut acc = b.f64(0.0);
        for (k, &ty) in ps2.iter().enumerate() {
            let v = b.param(k).unwrap();
            let x = if ty == t.f64 {
                v
            } else if ty == t.f32 || ty == t.i64 || ty == t.u8 {
                if ty == t.i64 {
                    b.cast(CastKind::Round, t.f64, v)
                } else {
                    b.cast(CastKind::Exact, t.f64, v)
                }
            } else if ty == pair {
                let one = b.int(1);
                let y = b.binary(Op::Get, t.i32, v, one);
                b.cast(CastKind::Exact, t.f64, y)
            } else {
                let n = b.unary(Op::SliceLen, t.usize, v);
                b.cast(CastKind::Round, t.f64, n)
            };
            let kf = b.f64((k + 1) as f64);
            let x = b.binary(Op::Mul, t.f64, x, kf);
            let s = b.binary(Op::Add, t.f64, acc, x);
            acc = s.into();
        }
        b.ret(Some(acc));
    });
    let fnty = p.m.types.fn_type(params.clone(), t.f64, stave_target::FeatureSet::EMPTY);
    let fnref = p.m.types.intern(stave_ir::TypeKey::Ref { mutable: true, inner: fnty });
    let s1 = p.m.add_string("abc");
    let s2 = p.m.add_string("0123456789");
    let ps3 = params.clone();
    p.set_main(|b, p| {
        let mut args = vec![];
        for (k, &ty) in ps3.iter().enumerate() {
            let a: Operand = if ty == t.f64 {
                b.f64(k as f64 * 0.75 - 3.0)
            } else if ty == t.f32 {
                b.f32(k as f32 * 1.5)
            } else if ty == t.i64 {
                b.int(-(k as i128) * 1000)
            } else if ty == t.u8 {
                b.int(200 + k as i128)
            } else if ty == pair {
                let (x, y) = (b.int(1), b.int(-77));
                b.op(Op::Make, pair, &[x, y]).into()
            } else if k == np - 1 {
                b.str(s2)
            } else {
                b.str(s1)
            };
            args.push(a);
        }
        let r = b.call(big, Some(t.f64), &args).unwrap();
        p.out(b, r, t.f64);
        // a fn value through memory: (code, null environment)
        let sl = b.slot(fnref);
        b.store(sl, b.func(big));
        let fv = b.load(fnty, sl);
        let r = b.call_ind(fv, Some(t.f64), &args).unwrap();
        p.out(b, r, t.f64);
    });
    check(&p, "args");
}

/// One program per trap kind / trapping op.
#[test]
fn traps() {
    type Mk = fn(&mut FuncBuilder, &Prog);
    let cases: Vec<(PanicKind, Mk)> = vec![
        (PanicKind::Overflow, |b, p| {
            let (x, y) = (b.int(127), b.int(1));
            let x = b.cast(CastKind::Exact, p.t.i8, x);
            b.set_site(Site(1));
            let r = b.binary(Op::Add, p.t.i8, x, y);
            p.out(b, r, p.t.i8);
        }),
        (PanicKind::DivByZero, |b, p| {
            let x = b.int(10);
            let x = b.cast(CastKind::Exact, p.t.u32, x);
            let z = b.int(0);
            let z = b.cast(CastKind::Exact, p.t.u32, z);
            b.set_site(Site(2));
            let r = b.binary(Op::Rem, p.t.u32, x, z);
            p.out(b, r, p.t.u32);
        }),
        (PanicKind::Overflow, |b, p| {
            let x = b.int(i64::MIN as i128);
            let x = b.cast(CastKind::Exact, p.t.i64, x);
            let y = b.int(-1);
            b.set_site(Site(3));
            let r = b.binary(Op::Div, p.t.i64, x, y);
            p.out(b, r, p.t.i64);
        }),
        (PanicKind::Overflow, |b, p| {
            let x = b.int(1);
            let x = b.cast(CastKind::Exact, p.t.u16, x);
            let n = b.int(16);
            let n = b.cast(CastKind::Exact, p.t.u16, n);
            b.set_site(Site(4));
            let r = b.binary(Op::Shl, p.t.u16, x, n);
            p.out(b, r, p.t.u16);
        }),
        (PanicKind::Unwrap, |b, _| {
            b.set_site(Site(6));
            b.panic(PanicKind::Unwrap);
        }),
        (PanicKind::Assert, |b, _| {
            b.set_site(Site(7));
            b.panic(PanicKind::Assert);
        }),
        (PanicKind::Overflow, |b, p| {
            let x = b.int(i32::MIN as i128);
            let x = b.cast(CastKind::Exact, p.t.i32, x);
            b.set_site(Site(8));
            let r = b.unary(Op::Neg, p.t.i32, x);
            p.out(b, r, p.t.i32);
        }),
        (PanicKind::Overflow, |b, p| {
            let x = b.int(u64::MAX as i128);
            let x = b.cast(CastKind::Exact, p.t.u64, x);
            let y = b.int(2);
            b.set_site(Site(9));
            let r = b.binary(Op::Mul, p.t.u64, x, y);
            p.out(b, r, p.t.u64);
        }),
    ];
    for (k, (kind, mk)) in cases.into_iter().enumerate() {
        let mut p = Prog::new();
        p.set_main(|b, p| {
            p.mark(b, 1);
            mk(b, p);
            p.mark(b, 2);
        });
        let r = check(&p, &format!("trap{k}"));
        assert_eq!(r.code, 101, "case {k}");
        assert_eq!(&r.stderr[..4], &trap_bytes(kind, 0)[..4], "case {k}");
    }
}
