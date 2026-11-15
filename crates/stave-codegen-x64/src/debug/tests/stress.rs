//! Stress programs: large frames (stack probes) and block copies, aggregate block params,
//! register pressure across calls, f16 / bf16 conversions, zero-sized values, branch arguments.

use stave_ir::{CastKind, Const, FloatBits, FuncBuilder, Op, Operand, TypeKey};

use super::harness::*;

#[test]
fn big_frames_and_copies() {
    let mut p = Prog::new();
    let t = p.t;
    let huge = p.m.types.array(t.u64, 20_000);
    let phuge = p.m.types.ref_(true, huge);
    let pel = p.m.types.ref_(true, t.u64);
    let arr = p.m.types.array(t.u32, 100);
    // fn bump(a: [u32; 100], k: u32) -> [u32; 100]: memory arg, sret result
    let bump = p.decl("t.bump", &[arr, t.u32], Some(arr));
    {
        let mut b = FuncBuilder::new(&[arr, t.u32]);
        let (a, k) = (b.param(0).unwrap(), b.param(1).unwrap());
        let mut cur = a;
        for i in [0i128, 17, 99] {
            let ci = b.int(i);
            let e = b.op(Op::Get, t.u32, &[cur.into(), ci]);
            let e2 = b.binary(Op::AddWrap, t.u32, e, k);
            let ci = b.int(i);
            cur = b.op(Op::Set, arr, &[cur.into(), ci, e2.into()]);
        }
        b.ret(Some(cur.into()));
        p.def(bump, b);
    }
    let pair = p.m.types.tuple(vec![t.u64, t.f64]);
    p.main(|p, b| {
        let s = b.slot(phuge);
        for (i, v) in [(0i128, 11i128), (19_999, 22), (12_345, 33)] {
            let ci = b.int(i);
            let idx = p.opaque(b, ci, t.usize);
            let e = b.op(Op::ElemAddr, pel, &[s.into(), idx.into()]);
            let cv = b.int(v);
            b.store(e, cv);
        }
        for i in [12_345i128, 0, 19_999] {
            let ci = b.int(i);
            let e = b.op(Op::ElemAddr, pel, &[s.into(), ci]);
            let x = b.load(t.u64, e);
            p.print(b, x.into(), t.u64);
        }
        let es: Vec<Operand> = (0..100).map(|i| b.int(i * 3)).collect();
        let av = b.op(Op::Make, arr, &es);
        let c50 = b.int(50);
        let i50 = p.opaque(b, c50, t.u32);
        let k = b.int(7);
        let av2 = b.op(Op::Set, arr, &[av.into(), i50.into(), k]);
        let k = b.int(1000);
        let r = b.call(bump, Some(arr), &[av2.into(), k]).unwrap();
        for i in [0i128, 17, 50, 98, 99] {
            let ci = b.int(i);
            let idx = p.opaque(b, ci, t.usize);
            let e = b.op(Op::Get, t.u32, &[r.into(), idx.into()]);
            p.print(b, e.into(), t.u32);
            let e = b.op(Op::Get, t.u32, &[av.into(), idx.into()]);
            p.print(b, e.into(), t.u32);
        }
        // aggregate block params swapped every iteration
        let (x0, x1, y0, y1) = (b.int(1), b.f64(0.5), b.int(2), b.f64(-4.0));
        let x = b.op(Op::Make, pair, &[x0, x1]);
        let y = b.op(Op::Make, pair, &[y0, y1]);
        let (h, hv) = b.create_block_with(&[pair, pair, t.u32]);
        let (body, exit) = (b.create_block(), b.create_block());
        let n = b.int(5);
        b.br(h, &[x.into(), y.into(), n]);
        b.switch_to(h);
        let z = b.int(0);
        let more = b.cmp(Op::Gt, t.bool, hv[2], z);
        b.cbr(more, body, &[], exit, &[]);
        b.switch_to(body);
        let k0 = b.int(0);
        let a0 = b.op(Op::Get, t.u64, &[hv[0].into(), k0]);
        let three = b.int(3);
        let a0m = b.binary(Op::MulWrap, t.u64, a0, three);
        let k1 = b.int(1);
        let a1 = b.op(Op::Get, t.f64, &[hv[0].into(), k1]);
        let nx = b.op(Op::Make, pair, &[a0m.into(), a1.into()]);
        let one = b.int(1);
        let n2 = b.binary(Op::Sub, t.u32, hv[2], one);
        b.br(h, &[hv[1].into(), nx.into(), n2.into()]);
        b.switch_to(exit);
        for v in [hv[0], hv[1]] {
            for k in 0..2 {
                let kk = b.int(k);
                let ft = if k == 0 { t.u64 } else { t.f64 };
                let f = b.op(Op::Get, ft, &[v.into(), kk]);
                p.print(b, f.into(), ft);
            }
        }
    });
    check(&p, "big_frames_and_copies");
}

#[test]
fn register_pressure() {
    let mut p = Prog::new();
    let t = p.t;
    p.main(|p, b| {
        let ints: Vec<_> = (0..40)
            .map(|i| {
                let c = b.int(i * 1_000_003 + 7);
                p.opaque(b, c, t.u64)
            })
            .collect();
        let floats: Vec<_> = (0..24)
            .map(|i| {
                let c = b.f64(i as f64 * 1.5 - 3.0);
                p.opaque(b, c, t.f64)
            })
            .collect();
        // all live at once: a chain that uses each value twice, across a call
        let mut acc: Operand = b.int(0);
        for (k, v) in ints.iter().enumerate() {
            let m = b.binary(Op::MulWrap, t.u64, *v, ints[(k * 7) % ints.len()]);
            acc = b.binary(Op::AddWrap, t.u64, acc, m).into();
        }
        p.print(b, acc, t.u64);
        let mut facc: Operand = b.f64(0.0);
        for (k, v) in floats.iter().enumerate() {
            let m = b.binary(Op::Mul, t.f64, *v, floats[(k * 5 + 1) % floats.len()]);
            facc = b.binary(Op::Add, t.f64, facc, m).into();
        }
        p.print(b, facc, t.f64);
        let mut acc2: Operand = b.int(1);
        for v in ints.iter().rev() {
            acc2 = b.binary(Op::Xor, t.u64, acc2, *v).into();
            let one = b.int(1);
            acc2 = b.binary(Op::ShlWrap, t.u64, acc2, one).into();
        }
        p.print(b, acc2, t.u64);
        let mut f2: Operand = b.f64(1.0);
        for v in floats.iter().rev() {
            f2 = b.binary(Op::Sub, t.f64, f2, *v).into();
        }
        p.print(b, f2, t.f64);
    });
    check(&p, "register_pressure");
}

#[test]
fn half_floats() {
    let mut p = Prog::new();
    let t = p.t;
    let f16 = p.m.types.intern(TypeKey::Float(FloatBits::F16));
    let bf16 = p.m.types.intern(TypeKey::Float(FloatBits::B16));
    let oi16 = p.option(t.i16);
    p.main(|p, b| {
        let f16s: [u16; 13] = [
            0, 0x8000, 0x3c00, 0xc000, 0x7bff, 0x0001, 0x03ff, 0x7c00, 0xfc00, 0x7e00, 0x7d00, 0x3555, 0x5640,
        ];
        let bf16s: [u16; 9] = [0x3f80, 0x0001, 0x7f80, 0x7fc0, 0x7f81, 0xff80, 0x4049, 0x8000, 0x4780];
        for (ty, vals) in [(f16, f16s.to_vec()), (bf16, bf16s.to_vec())] {
            let mut prev = None;
            for bits in vals {
                let c = b.konst(if ty == f16 { Const::F16(bits) } else { Const::BF16(bits) });
                let x = p.opaque(b, c, ty);
                p.print(b, x.into(), ty);
                let a = b.cast(CastKind::Exact, t.f32, x);
                p.print(b, a.into(), t.f32);
                let a = b.cast(CastKind::Exact, t.f64, x);
                p.print(b, a.into(), t.f64);
                let s = b.cast(CastKind::Sat, t.i32, x);
                p.print(b, s.into(), t.i32);
                let o = b.cast(CastKind::Try, oi16, x);
                let tag = b.op(Op::EnumTag, t.u32, &[o.into()]);
                p.print(b, tag.into(), t.u32);
                let c2 = b.cast(CastKind::Exact, t.f32, c);
                p.print(b, c2.into(), t.f32);
                if let Some(y) = prev {
                    for op in [Op::Eq, Op::Ne, Op::Lt, Op::Ge] {
                        let r = b.cmp(op, t.bool, x, y);
                        p.print(b, r.into(), t.bool);
                    }
                }
                prev = Some(x);
            }
        }
    });
    check(&p, "half_floats");
}

#[test]
fn zero_sized_and_branch_args() {
    let mut p = Prog::new();
    let t = p.t;
    let unit = p.m.types.unit();
    let empty = p.m.types.array(t.u8, 0);
    let z = p.decl("t.z", &[unit, t.u32, empty, unit], Some(unit));
    {
        let mut b = FuncBuilder::new(&[unit, t.u32, empty, unit]);
        let x = b.param(1).unwrap();
        p.print(&mut b, x.into(), t.u32);
        let u = b.op(Op::Make, unit, &[]);
        b.ret(Some(u.into()));
        p.def(z, b);
    }
    p.main(|p, b| {
        let u = b.op(Op::Make, unit, &[]);
        let e = b.op(Op::Make, empty, &[]);
        let k = b.int(77);
        let r = b.call(z, Some(unit), &[u.into(), k, e.into(), u.into()]).unwrap();
        let one = b.int(1);
        let c = p.opaque(b, one, t.bool);
        let s = b.select(unit, c, r, u);
        let k = b.int(78);
        b.call(z, Some(unit), &[s.into(), k, e.into(), r.into()]);
        // cbr with arguments on both edges, then on one
        for v in [0i128, 1] {
            let cv = b.int(v);
            let c = p.opaque(b, cv, t.bool);
            let (j, jv) = b.create_block_with(&[t.i64, t.f32]);
            let (a1, a2, b1, b2) = (b.int(-5), b.f32(1.5), b.int(9), b.f32(-0.0));
            b.cbr(c, j, &[a1, a2], j, &[b1, b2]);
            b.switch_to(j);
            p.print(b, jv[0].into(), t.i64);
            p.print(b, jv[1].into(), t.f32);
            let (j2, jv2) = b.create_block_with(&[t.u8]);
            let mid = b.create_block();
            let a = b.int(200);
            b.cbr(c, j2, &[a], mid, &[]);
            b.switch_to(mid);
            let a = b.int(100);
            b.br(j2, &[a]);
            b.switch_to(j2);
            p.print(b, jv2[0].into(), t.u8);
        }
    });
    check(&p, "zero_sized_and_branch_args");
}
