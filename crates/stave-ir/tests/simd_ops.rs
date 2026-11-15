//! Lane semantics of the vector and mask ops in the interpreter: IEEE corner cases (NaN, ±0,
//! denormals, no contraction), the fixed reduction tree, integer wrap / saturate / trap rules,
//! memory forms with their bounds rules, masks, shapes; verifier typing, printing, bounds audit.

mod common;

use common::*;
use stave_ir::interp::{Config, Interp, Outcome, Val};
use stave_ir::{
    AdtShape, CastKind, DeclId, FloatBits, FuncBuilder, GenericArg, IntBits, Lanes, Module, Op, Operand,
    PanicKind, TypeId, TypeKey, VecOp, verify_module,
};

fn int(signed: bool, bits: u8) -> TypeKey {
    let bits = match bits {
        8 => IntBits::B8,
        16 => IntBits::B16,
        32 => IntBits::B32,
        _ => IntBits::B64,
    };
    TypeKey::Int { signed, bits }
}
const F32: TypeKey = TypeKey::Float(FloatBits::F32);
const F64: TypeKey = TypeKey::Float(FloatBits::F64);
const F16: TypeKey = TypeKey::Float(FloatBits::F16);

fn vk(m: &mut Module, lane: TypeKey, n: u8) -> TypeKey {
    let l = m.types.intern(lane);
    TypeKey::Vec { lane: l, n: Lanes::Fixed(n) }
}
fn mk(w: u8, n: u8) -> TypeKey {
    TypeKey::Mask { width: w, n: Lanes::Fixed(n) }
}
fn option(m: &mut Module, t: TypeId) -> TypeId {
    m.types.adt(DeclId(0), vec![GenericArg::Type(t)], AdtShape::enumeration(vec![vec![], vec![t]]))
}

fn lanes(v: &[u64]) -> Val {
    Val::agg(v.iter().map(|&x| Val::S(x)).collect())
}
fn f32s(v: &[f32]) -> Val {
    Val::agg(v.iter().map(|&x| Val::f32(x)).collect())
}
fn f64s(v: &[f64]) -> Val {
    Val::agg(v.iter().map(|&x| Val::f64(x)).collect())
}
fn out_lanes(v: &Val) -> Vec<u64> {
    v.fields().expect("vector").iter().map(|l| l.bits().expect("lane")).collect()
}
fn out_f32(v: &Val) -> Vec<f32> {
    out_lanes(v).into_iter().map(|b| f32::from_bits(b as u32)).collect()
}

/// One vector op over params `params` (type keys built with the module), result `res`.
fn one(
    build: impl FnOnce(&mut Module) -> (Vec<TypeId>, Option<TypeId>, Vec<Operand>),
    op: VecOp,
    args: Vec<Val>,
) -> Result<Val, PanicKind> {
    let mut m = Module::new();
    let (ps, rt, extra) = build(&mut m);
    let o = Op::Vec(op);
    let f = m.declare(inst(&[0]), "t.f", ps.clone(), rt);
    let mut b = FuncBuilder::new(&ps);
    let mut ops: Vec<Operand> = b.params().iter().map(|v| (*v).into()).collect();
    ops.extend(extra);
    let r = b.inst(o, rt, &ops);
    b.ret(r.map(Operand::from));
    m.define(f, b.finish().expect("builder"));
    if let Err(e) = verify_module(&m) {
        panic!("verify: {e:?}");
    }
    match run(&m, f, args) {
        Outcome::Return(Some(v)) => Ok(v),
        Outcome::Return(None) => Ok(Val::unit()),
        Outcome::Trap(t) => Err(t.kind),
        o => panic!("unexpected {o:?}"),
    }
}

/// `op` on `k` params of vector type `lane x n`, result of the same type.
fn same(lane: TypeKey, n: u8, k: usize, op: VecOp, args: Vec<Val>) -> Result<Val, PanicKind> {
    one(
        |m| {
            let v = vk(m, lane, n);
            let t = m.types.intern(v);
            (vec![t; k], Some(t), vec![])
        },
        op,
        args,
    )
}

fn f32_bin(op: VecOp, a: &[f32], b: &[f32]) -> Vec<u32> {
    let r = same(F32, a.len() as u8, 2, op, vec![f32s(a), f32s(b)]).unwrap();
    out_lanes(&r).into_iter().map(|x| x as u32).collect()
}

const QNAN: u32 = 0x7fc0_0000;
const SNAN: u32 = 0x7f80_0001;
const NEG_NAN_PAYLOAD: u32 = 0xffc1_2345;

fn is_qnan32(b: u32) -> bool {
    f32::from_bits(b).is_nan() && b & (1 << 22) != 0
}

#[test]
fn float_arithmetic_nan_results_are_quiet() {
    let s = f32::from_bits(SNAN);
    let q = f32::from_bits(QNAN);
    let inf = f32::INFINITY;
    let r = f32_bin(VecOp::Add, &[s, q, inf, 1.0], &[1.0, 2.0, -inf, 2.0]);
    assert!(is_qnan32(r[0]) && is_qnan32(r[1]) && is_qnan32(r[2]), "{r:x?}");
    assert_eq!(f32::from_bits(r[3]), 3.0);
    let r = f32_bin(VecOp::Mul, &[0.0, inf, s, -0.0], &[inf, 0.0, 0.0, 5.0]);
    assert!(r[..3].iter().all(|&x| is_qnan32(x)), "{r:x?}");
    assert_eq!(r[3], (-0.0f32).to_bits());
    let r = f32_bin(VecOp::Div, &[0.0, 1.0, -1.0, s], &[0.0, 0.0, 0.0, 1.0]);
    assert!(is_qnan32(r[0]) && is_qnan32(r[3]));
    assert_eq!((f32::from_bits(r[1]), f32::from_bits(r[2])), (inf, -inf));
    let sq = same(F32, 4, 1, VecOp::Sqrt, vec![f32s(&[-1.0, s, -0.0, 4.0])]).unwrap();
    let sq: Vec<u32> = out_lanes(&sq).into_iter().map(|x| x as u32).collect();
    assert!(is_qnan32(sq[0]) && is_qnan32(sq[1]));
    assert_eq!(sq[2], (-0.0f32).to_bits(), "sqrt(-0) = -0");
    assert_eq!(f32::from_bits(sq[3]), 2.0);
    for op in [VecOp::Floor, VecOp::Ceil, VecOp::Round, VecOp::Trunc] {
        let r = same(F32, 2, 1, op, vec![f32s(&[s, -0.5])]).unwrap();
        let r = out_lanes(&r);
        assert!(is_qnan32(r[0] as u32), "{op:?}");
        // floor(-0.5) = -1, the others -0: the sign is kept
        assert_eq!(r[1] >> 31, 1, "{op:?}");
    }
}

#[test]
fn neg_abs_copysign_are_bit_exact() {
    let n = f32::from_bits(NEG_NAN_PAYLOAD);
    let s = f32::from_bits(SNAN);
    let r = same(F32, 4, 1, VecOp::Neg, vec![f32s(&[n, s, 0.0, -1.5])]).unwrap();
    assert_eq!(out_lanes(&r), vec![0x7fc1_2345, 0xff80_0001, 0x8000_0000, 0x3fc0_0000]);
    let r = same(F32, 4, 1, VecOp::Abs, vec![f32s(&[n, s, -0.0, -1.5])]).unwrap();
    assert_eq!(out_lanes(&r), vec![0x7fc1_2345, 0x7f80_0001, 0, 0x3fc0_0000]);
    let r = f32_bin(VecOp::Copysign, &[s, 1.0, -0.0, 2.0], &[-1.0, n, 1.0, -0.0]);
    assert_eq!(r, vec![0xff80_0001, 0xbf80_0000, 0, 0xc000_0000]);
    // f64 and f16 lanes too
    let r = same(F64, 2, 1, VecOp::Neg, vec![f64s(&[0.0, f64::from_bits(0x7ff0_0000_0000_0001)])]).unwrap();
    assert_eq!(out_lanes(&r), vec![1 << 63, 0xfff0_0000_0000_0001]);
    let r = same(F16, 8, 1, VecOp::Abs, vec![lanes(&[0xfc01, 0x8000, 0xbc00, 0x3c00, 0, 1, 0x8001, 0xffff])])
        .unwrap();
    assert_eq!(out_lanes(&r), vec![0x7c01, 0, 0x3c00, 0x3c00, 0, 1, 1, 0x7fff]);
}

#[test]
fn min_max_forms() {
    let q = f32::from_bits(QNAN);
    let a = [-0.0, 0.0, q, 1.0];
    let b = [0.0, -0.0, 1.0, q];
    let min = f32_bin(VecOp::Min, &a, &b);
    assert_eq!(&min[..2], &[0x8000_0000, 0x8000_0000], "-0 < +0");
    assert!(is_qnan32(min[2]) && is_qnan32(min[3]), "NaN propagates");
    let max = f32_bin(VecOp::Max, &a, &b);
    assert_eq!(&max[..2], &[0, 0]);
    assert!(is_qnan32(max[2]) && is_qnan32(max[3]));
    let mn = f32_bin(VecOp::MinNum, &a, &b);
    assert_eq!(mn, vec![0x8000_0000, 0x8000_0000, 1f32.to_bits(), 1f32.to_bits()], "NaN is missing");
    let mx = f32_bin(VecOp::MaxNum, &[q, 2.0, q, -0.0], &[3.0, q, q, 0.0]);
    assert_eq!(&mx[..2], &[3f32.to_bits(), 2f32.to_bits()]);
    assert!(is_qnan32(mx[2]));
    assert_eq!(mx[3], 0);
    // x86: `a < b ? a : b`, so a NaN or equal pair yields b
    let x = f32_bin(VecOp::MinX86, &a, &b);
    assert_eq!(x[0], 0, "min_x86(-0, +0) = +0 (second operand)");
    assert_eq!(x[1], 0x8000_0000);
    assert_eq!(x[2], 1f32.to_bits());
    assert!(is_qnan32(x[3]));
    let x = f32_bin(VecOp::MaxX86, &[q, 1.0, 2.0, 0.0], &[1.0, q, 1.0, -0.0]);
    assert_eq!(x[0], 1f32.to_bits());
    assert!(is_qnan32(x[1]));
    assert_eq!(x[2], 2f32.to_bits());
    assert_eq!(x[3], 0x8000_0000);
}

#[test]
fn denormals_are_kept() {
    let d = f32::from_bits(1);
    let r = f32_bin(VecOp::Add, &[d, d, f32::MIN_POSITIVE, 0.0], &[d, -d, -d, d]);
    assert_eq!(r, vec![2, 0, f32::MIN_POSITIVE.to_bits() - 1, 1]);
    // 1.5 * 2^-149 rounds to even (2^-148), 0.5 * 2^-149 rounds to 0
    let r = f32_bin(VecOp::Mul, &[f32::from_bits(3), d, d, 1e-25], &[0.5, 0.5, 1.5, 1e-25]);
    assert_eq!(r, vec![2, 0, 2, 0]);
    let r =
        same(F64, 2, 2, VecOp::Sub, vec![f64s(&[f64::MIN_POSITIVE, 0.0]), f64s(&[f64::from_bits(1), 0.0])])
            .unwrap();
    assert_eq!(out_lanes(&r), vec![f64::MIN_POSITIVE.to_bits() - 1, 0]);
}

#[test]
fn no_contraction_but_fma_is_fused() {
    // x = 1 + 2^-12: x*x = 1 + 2^-11 + 2^-24 rounds (tie) to 1 + 2^-11
    let x = 1.0 + 2f32.powi(-12);
    let c = -(1.0 + 2f32.powi(-11));
    let prod = f32_bin(VecOp::Mul, &[x, x], &[x, x]);
    let p = f32::from_bits(prod[0]);
    let sum = f32_bin(VecOp::Add, &[p, p], &[c, c]);
    assert_eq!(f32::from_bits(sum[0]), 0.0, "mul then add rounds twice");
    let fma = same(F32, 2, 3, VecOp::Fma, vec![f32s(&[x, x]), f32s(&[x, x]), f32s(&[c, c])]).unwrap();
    assert_eq!(out_f32(&fma)[0], 2f32.powi(-24), "fma rounds once");
    let fma64 = same(
        F64,
        2,
        3,
        VecOp::Fma,
        vec![f64s(&[0.1, f64::INFINITY]), f64s(&[10.0, 0.0]), f64s(&[-1.0, 1.0])],
    )
    .unwrap();
    let l = out_lanes(&fma64);
    assert_eq!(f64::from_bits(l[0]), 0.1f64.mul_add(10.0, -1.0));
    assert!(f64::from_bits(l[1]).is_nan());
}

#[test]
fn round_is_ties_to_even() {
    let r = same(F32, 8, 1, VecOp::Round, vec![f32s(&[0.5, 1.5, 2.5, -2.5, -0.5, 3.7, -3.2, 1e30])]).unwrap();
    assert_eq!(out_f32(&r), vec![0.0, 2.0, 2.0, -2.0, -0.0, 4.0, -3.0, 1e30]);
    assert_eq!(out_lanes(&r)[4], 0x8000_0000);
    let f = same(F64, 4, 1, VecOp::Floor, vec![f64s(&[-0.5, 0.5, -1.0, 2.9])]).unwrap();
    assert_eq!(out_lanes(&f).into_iter().map(f64::from_bits).collect::<Vec<_>>(), vec![-1.0, 0.0, -1.0, 2.0]);
}

#[test]
fn reduction_tree_and_ordered_sum() {
    let v = f32s(&[1e8, 1.0, -1e8, 1.0]);
    let tree = one(
        |m| {
            let k = vk(m, F32, 4);
            let t = m.types.intern(k);
            let f = m.types.f32();
            (vec![t], Some(f), vec![])
        },
        VecOp::ReduceAdd,
        vec![v.clone()],
    )
    .unwrap();
    // lanes (0, 2) and (1, 3) first: (1e8 - 1e8) + (1 + 1)
    assert_eq!(tree.as_f32(), Some(2.0));
    let ordered = one(
        |m| {
            let k = vk(m, F32, 4);
            let t = m.types.intern(k);
            let f = m.types.f32();
            (vec![t, f], Some(f), vec![])
        },
        VecOp::ReduceAddOrdered,
        vec![v, Val::f32(0.0)],
    )
    .unwrap();
    // ((0 + 1e8) + 1) + -1e8 + 1 = 1 (the + 1 is lost)
    assert_eq!(ordered.as_f32(), Some(1.0));
    // 8 lanes: (0,4) (1,5) (2,6) (3,7), then (0,2) (1,3), then (0,1)
    let eight: Vec<f32> = vec![1e8, 1.0, 1.0, 1.0, -1e8, 1.0, 1.0, 1.0];
    let r = one(
        |m| {
            let k = vk(m, F32, 8);
            let t = m.types.intern(k);
            let f = m.types.f32();
            (vec![t], Some(f), vec![])
        },
        VecOp::ReduceAdd,
        vec![f32s(&eight)],
    )
    .unwrap();
    assert_eq!(r.as_f32(), Some(6.0));
}

fn reduce(lane: TypeKey, n: u8, op: VecOp, v: Vec<u64>) -> u64 {
    one(
        |m| {
            let k = vk(m, lane.clone(), n);
            let t = m.types.intern(k);
            let l = m.types.intern(lane.clone());
            (vec![t], Some(l), vec![])
        },
        op,
        vec![lanes(&v)],
    )
    .unwrap()
    .bits()
    .unwrap()
}

#[test]
fn int_reductions_wrap() {
    let i8t = int(true, 8);
    let v: Vec<u64> = [127i64, 1, -128, -1].iter().map(|&x| x as u64).collect();
    assert_eq!(reduce(i8t.clone(), 4, VecOp::ReduceAddWrap, v.clone()) as i64, -1);
    assert_eq!(reduce(i8t.clone(), 4, VecOp::ReduceMin, v.clone()) as i64, -128);
    assert_eq!(reduce(i8t.clone(), 4, VecOp::ReduceMax, v.clone()) as i64, 127);
    let u = int(false, 16);
    assert_eq!(reduce(u.clone(), 2, VecOp::ReduceAddWrap, vec![0xffff, 2]), 1);
    assert_eq!(
        reduce(u.clone(), 4, VecOp::ReduceAnd, vec![0xff0f, 0x0fff, 0xffff, 0xf0ff]),
        0x000f & 0x00ff & 0x0f0f
    );
    assert_eq!(reduce(u, 4, VecOp::ReduceOr, vec![1, 2, 4, 0x8000]), 0x8007);
}

/// Every lane type and every lane count from 2 to 64: splat, add, reduce, compare, to_bits.
#[test]
fn every_lane_count() {
    let lanes_t = [
        int(true, 8),
        int(false, 8),
        int(true, 16),
        int(false, 16),
        int(true, 32),
        int(false, 32),
        int(true, 64),
        int(false, 64),
        F32,
        F64,
    ];
    for lane in lanes_t {
        for n in [2u8, 4, 8, 16, 32, 64] {
            let mut m = Module::new();
            let l = m.types.intern(lane.clone());
            let k = vk(&mut m, lane.clone(), n);
            let v = m.types.intern(k);
            let bits = match &lane {
                TypeKey::Int { bits, .. } => bits.width(),
                TypeKey::Float(f) => f.width(),
                _ => 0,
            };
            let mt = m.types.intern(mk(bits, n));
            let ut = m.types.u64();
            let float = matches!(lane, TypeKey::Float(_));
            let (splat, add, red, lt, tb) = (
                Op::Vec(VecOp::Splat),
                Op::Vec(if float { VecOp::Add } else { VecOp::AddWrap }),
                Op::Vec(if float { VecOp::ReduceAdd } else { VecOp::ReduceAddWrap }),
                Op::Vec(VecOp::Lt),
                Op::Vec(VecOp::ToBits),
            );
            let tbt = match n {
                2..=8 => m.types.u8(),
                16 => m.types.u16(),
                32 => m.types.u32(),
                _ => ut,
            };
            let tup = m.types.tuple(vec![l, tbt]);
            let f = define(&mut m, &[0], "t.f", &[l], Some(tup), |b, _| {
                let x = b.param(0).unwrap();
                let s = b.op(splat, v, &[x.into()]);
                let s2 = b.op(add, v, &[s.into(), s.into()]);
                let r = b.op(red, l, &[s2.into()]);
                let c = b.op(lt, mt, &[s.into(), s2.into()]);
                let bits = b.op(tb, tbt, &[c.into()]);
                let t = b.op(Op::Make, tup, &[r.into(), bits.into()]);
                b.ret(Some(t.into()));
            });
            verify_module(&m).unwrap_or_else(|e| panic!("{lane:?} x{n}: {e:?}"));
            let arg = if float { if bits == 32 { Val::f32(1.0) } else { Val::f64(1.0) } } else { Val::S(1) };
            let out = ret_val(run(&m, f, vec![arg]));
            let fs = out.fields().unwrap();
            let want = 2 * n as u64;
            let got = match (float, bits) {
                (true, 32) => fs[0].as_f32().unwrap() as u64,
                (true, _) => fs[0].as_f64().unwrap() as u64,
                _ => fs[0].bits().unwrap() & if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 },
            };
            let wrapped = if !float && bits < 64 { want & ((1u64 << bits) - 1) } else { want };
            assert_eq!(got, wrapped, "{lane:?} x{n}");
            let all = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            assert_eq!(fs[1].bits(), Some(all), "{lane:?} x{n}: to_bits");
        }
    }
}

fn i32x4(m: &mut Module) -> TypeId {
    let k = vk(m, int(true, 32), 4);
    m.types.intern(k)
}

fn ival(v: &Val) -> Vec<i64> {
    out_lanes(v).into_iter().map(|x| x as i64).collect()
}

fn ilanes(v: &[i64]) -> Val {
    lanes(&v.iter().map(|&x| x as u64).collect::<Vec<_>>())
}

#[test]
fn int_wrap_saturate_trap() {
    let min = i32::MIN as i64;
    let max = i32::MAX as i64;
    let a = ilanes(&[max, min, -1, 7]);
    let b = ilanes(&[1, -1, min, -2]);
    let w = same(int(true, 32), 4, 2, VecOp::AddWrap, vec![a.clone(), b.clone()]).unwrap();
    assert_eq!(ival(&w), vec![min, max, max, 5]);
    let s = same(int(true, 32), 4, 2, VecOp::AddSat, vec![a.clone(), b.clone()]).unwrap();
    assert_eq!(ival(&s), vec![max, min, min, 5]);
    let s = same(int(true, 32), 4, 2, VecOp::SubSat, vec![a.clone(), b.clone()]).unwrap();
    assert_eq!(ival(&s), vec![max - 1, min + 1, max, 9]);
    let m = same(int(true, 32), 4, 2, VecOp::MulWrap, vec![a.clone(), b.clone()]).unwrap();
    assert_eq!(ival(&m), vec![max, min, min, -14]);
    let u = same(int(false, 8), 4, 2, VecOp::AddSat, vec![lanes(&[250, 1, 0, 255]), lanes(&[10, 2, 0, 255])])
        .unwrap();
    assert_eq!(out_lanes(&u), vec![255, 3, 0, 255]);
    let u = same(int(false, 8), 4, 2, VecOp::SubSat, vec![lanes(&[1, 5, 0, 255]), lanes(&[2, 2, 0, 255])])
        .unwrap();
    assert_eq!(out_lanes(&u), vec![0, 3, 0, 0]);
    let abs = same(int(true, 32), 4, 1, VecOp::AbsWrap, vec![ilanes(&[min, -5, 0, max])]).unwrap();
    assert_eq!(ival(&abs), vec![min, 5, 0, max]);
    // division: MIN / -1 traps Overflow, a zero divisor traps DivByZero
    let d = same(int(true, 32), 4, 2, VecOp::DivLanes, vec![ilanes(&[7, -7, min, 1]), ilanes(&[2, 2, 3, 1])])
        .unwrap();
    assert_eq!(ival(&d), vec![3, -3, min / 3, 1]);
    let d = same(int(true, 32), 4, 2, VecOp::DivLanes, vec![ilanes(&[1, min, 1, 1]), ilanes(&[1, -1, 1, 1])]);
    assert_eq!(d, Err(PanicKind::Overflow));
    let d = same(int(true, 32), 4, 2, VecOp::DivLanes, vec![ilanes(&[1, 1, 1, 1]), ilanes(&[1, 1, 0, 1])]);
    assert_eq!(d, Err(PanicKind::DivByZero));
    let mut m = Module::new();
    let t = i32x4(&mut m);
    let dc = Op::Vec(VecOp::DivConst);
    let f = define(&mut m, &[0], "t.f", &[t], Some(t), |b, _| {
        let x = b.param(0).unwrap();
        let k = b.int(-3);
        let r = b.op(dc, t, &[x.into(), k]);
        b.ret(Some(r.into()));
    });
    verify_module(&m).unwrap();
    assert_eq!(ival(&ret_val(run(&m, f, vec![ilanes(&[9, -9, 10, min])]))), vec![-3, 3, -3, min / -3]);
}

#[test]
fn checked_ops_yield_option() {
    let res = |op: VecOp, a: Vec<i64>, b: Vec<i64>| {
        one(
            |m| {
                let t = i32x4(m);
                let o = option(m, t);
                (vec![t, t], Some(o), vec![])
            },
            op,
            vec![ilanes(&a), ilanes(&b)],
        )
        .unwrap()
    };
    let ok = res(VecOp::CheckedAdd, vec![1, 2, 3, 4], vec![1, 1, 1, 1]);
    assert_eq!(ok.fields().unwrap()[0], Val::S(1));
    assert_eq!(ival(&ok.fields().unwrap()[1]), vec![2, 3, 4, 5]);
    let bad = res(VecOp::CheckedAdd, vec![i32::MAX as i64, 0, 0, 0], vec![1, 0, 0, 0]);
    assert_eq!(bad.fields().unwrap(), &[Val::S(0)]);
    let bad = res(VecOp::CheckedSub, vec![0, i32::MIN as i64, 0, 0], vec![0, 1, 0, 0]);
    assert_eq!(bad.fields().unwrap(), &[Val::S(0)]);
    let bad = res(VecOp::CheckedMul, vec![0, 0, 65536, 0], vec![0, 0, 65536, 0]);
    assert_eq!(bad.fields().unwrap(), &[Val::S(0)]);
}

#[test]
fn shifts_trap_or_mask() {
    let u32k = int(false, 32);
    let a = lanes(&[1, 2, 3, 0x8000_0000]);
    let ok = same(u32k.clone(), 4, 2, VecOp::Shl, vec![a.clone(), lanes(&[0, 1, 31, 1])]).unwrap();
    assert_eq!(out_lanes(&ok), vec![1, 4, 0x8000_0000, 0]);
    let trap = same(u32k.clone(), 4, 2, VecOp::Shl, vec![a.clone(), lanes(&[0, 1, 32, 1])]);
    assert_eq!(trap, Err(PanicKind::Overflow));
    let wrap = same(u32k.clone(), 4, 2, VecOp::ShlWrap, vec![a.clone(), lanes(&[32, 33, 63, 1])]).unwrap();
    assert_eq!(out_lanes(&wrap), vec![1, 4, 0x8000_0000, 0]);
}

#[test]
fn shift_by_scalar_constant() {
    let mut m = Module::new();
    let t = i32x4(&mut m);
    let shr = Op::Vec(VecOp::Shr);
    let f = define(&mut m, &[0], "t.f", &[t], Some(t), |b, _| {
        let x = b.param(0).unwrap();
        let k = b.int(2);
        let r = b.op(shr, t, &[x.into(), k]);
        b.ret(Some(r.into()));
    });
    verify_module(&m).unwrap();
    let r = ret_val(run(&m, f, vec![ilanes(&[-8, 8, i32::MIN as i64, -1])]));
    assert_eq!(ival(&r), vec![-2, 2, i32::MIN as i64 / 4, -1]);
}

#[test]
fn compares_and_masks() {
    let q = f32::from_bits(QNAN);
    let r = one(
        |m| {
            let k = vk(m, F32, 4);
            let t = m.types.intern(k);
            let mt = m.types.intern(mk(32, 4));
            (vec![t, t], Some(mt), vec![])
        },
        VecOp::Ne,
        vec![f32s(&[q, 1.0, 0.0, 2.0]), f32s(&[q, 1.0, -0.0, 3.0])],
    )
    .unwrap();
    assert_eq!(out_lanes(&r), vec![1, 0, 0, 1], "unordered: only ne holds; -0 == +0");
    let r = one(
        |m| {
            let k = vk(m, F32, 4);
            let t = m.types.intern(k);
            let mt = m.types.intern(mk(32, 4));
            (vec![t, t], Some(mt), vec![])
        },
        VecOp::Lt,
        vec![f32s(&[q, 1.0, -1.0, 2.0]), f32s(&[0.0, q, 0.0, 3.0])],
    )
    .unwrap();
    assert_eq!(out_lanes(&r), vec![0, 0, 1, 1]);
    // unsigned vs signed compare of the same bits
    let cmp = |signed: bool| {
        one(
            |m| {
                let k = vk(m, int(signed, 8), 2);
                let t = m.types.intern(k);
                let mt = m.types.intern(mk(8, 2));
                (vec![t, t], Some(mt), vec![])
            },
            VecOp::Lt,
            vec![lanes(&[if signed { u64::MAX } else { 255 }, 1]), lanes(&[1, 2])],
        )
        .unwrap()
    };
    assert_eq!(out_lanes(&cmp(true)), vec![1, 1]);
    assert_eq!(out_lanes(&cmp(false)), vec![0, 1]);
}

/// A function over one `mask32x8` param running mask op `op` (`extra` builds more operands).
fn mask_fn(op: VecOp, res: impl FnOnce(&mut Module) -> TypeId, args: Vec<Val>, extra: usize) -> Val {
    one(
        |m| {
            let mt = m.types.intern(mk(32, 8));
            let r = res(m);
            (vec![mt; extra], Some(r), vec![])
        },
        op,
        args,
    )
    .unwrap()
}

fn bools(v: &[u8]) -> Val {
    Val::agg(v.iter().map(|&x| Val::bool(x != 0)).collect())
}

#[test]
fn mask_ops() {
    let a = bools(&[1, 0, 1, 0, 1, 1, 0, 0]);
    let b = bools(&[1, 1, 0, 0, 0, 1, 1, 0]);
    let mt = |m: &mut Module| m.types.intern(mk(32, 8));
    assert_eq!(
        out_lanes(&mask_fn(VecOp::MaskAnd, mt, vec![a.clone(), b.clone()], 2)),
        vec![1, 0, 0, 0, 0, 1, 0, 0]
    );
    assert_eq!(
        out_lanes(&mask_fn(VecOp::MaskOr, mt, vec![a.clone(), b.clone()], 2)),
        vec![1, 1, 1, 0, 1, 1, 1, 0]
    );
    assert_eq!(
        out_lanes(&mask_fn(VecOp::MaskXor, mt, vec![a.clone(), b.clone()], 2)),
        vec![0, 1, 1, 0, 1, 0, 1, 0]
    );
    assert_eq!(out_lanes(&mask_fn(VecOp::MaskNot, mt, vec![a.clone()], 1)), vec![0, 1, 0, 1, 0, 0, 1, 1]);
    let bits = mask_fn(VecOp::ToBits, |m| m.types.u8(), vec![a.clone()], 1);
    assert_eq!(bits.bits(), Some(0b0011_0101));
    let v = mask_fn(
        VecOp::ToVec,
        |m| {
            let k = vk(m, int(true, 32), 8);
            m.types.intern(k)
        },
        vec![a.clone()],
        1,
    );
    assert_eq!(ival(&v), vec![-1, 0, -1, 0, -1, -1, 0, 0]);
    let v = mask_fn(
        VecOp::ToVec,
        |m| {
            let k = vk(m, int(false, 32), 8);
            m.types.intern(k)
        },
        vec![a.clone()],
        1,
    );
    assert_eq!(out_lanes(&v)[0], 0xffff_ffff);
    let w = mask_fn(VecOp::Rewiden, |m| m.types.intern(mk(8, 8)), vec![a.clone()], 1);
    assert_eq!(out_lanes(&w), out_lanes(&a));
    for (op, want) in [(VecOp::MaskAll, 1), (VecOp::MaskNone, 0)] {
        let r = mask_fn(op, mt, vec![], 0);
        assert_eq!(out_lanes(&r), vec![want; 8]);
    }
    for k in [0u64, 3, 8, 9, u64::MAX] {
        let r = one(
            |m| {
                let us = m.types.usize();
                let mt = m.types.intern(mk(32, 8));
                (vec![us], Some(mt), vec![])
            },
            VecOp::FirstN,
            vec![Val::S(k)],
        )
        .unwrap();
        let got = out_lanes(&r).iter().filter(|&&x| x == 1).count() as u64;
        assert_eq!(got, k.min(8), "first_n({k})");
    }
}

#[test]
fn select_is_bit_exact() {
    let r = one(
        |m| {
            let mt = m.types.intern(mk(32, 4));
            let k = vk(m, F32, 4);
            let t = m.types.intern(k);
            (vec![mt, t, t], Some(t), vec![])
        },
        VecOp::Select,
        vec![
            bools(&[1, 0, 1, 0]),
            lanes(&[SNAN as u64, 1, NEG_NAN_PAYLOAD as u64, 2]),
            lanes(&[5, QNAN as u64, 6, SNAN as u64]),
        ],
    )
    .unwrap();
    assert_eq!(out_lanes(&r), vec![SNAN as u64, QNAN as u64, NEG_NAN_PAYLOAD as u64, SNAN as u64]);
}

#[test]
fn shapes() {
    // permute
    let mut m = Module::new();
    let t = i32x4(&mut m);
    let p = Op::Vec(VecOp::Permute);
    let f = define(&mut m, &[0], "t.f", &[t], Some(t), |b, _| {
        let x = b.param(0).unwrap();
        let ks: Vec<Operand> = [3, 3, 0, 1].iter().map(|&k| b.int(k)).collect();
        let mut ops = vec![x.into()];
        ops.extend(ks);
        let r = b.op(p, t, &ops);
        b.ret(Some(r.into()));
    });
    verify_module(&m).unwrap();
    assert_eq!(ival(&ret_val(run(&m, f, vec![ilanes(&[10, 11, 12, 13])]))), vec![13, 13, 10, 11]);
    // lookup: out-of-range index → 0
    let r = one(
        |m| {
            let k = vk(m, int(false, 8), 4);
            let t = m.types.intern(k);
            (vec![t, t], Some(t), vec![])
        },
        VecOp::Lookup,
        vec![lanes(&[7, 8, 9, 10]), lanes(&[3, 4, 0, 255])],
    )
    .unwrap();
    assert_eq!(out_lanes(&r), vec![10, 0, 7, 0]);
    // compress / expand
    let ce = |op| {
        one(
            |m| {
                let t = i32x4(m);
                let mt = m.types.intern(mk(32, 4));
                (vec![t, mt], Some(t), vec![])
            },
            op,
            vec![ilanes(&[1, 2, 3, 4]), bools(&[0, 1, 0, 1])],
        )
        .unwrap()
    };
    assert_eq!(ival(&ce(VecOp::Compress)), vec![2, 4, 0, 0]);
    assert_eq!(ival(&ce(VecOp::Expand)), vec![0, 1, 0, 2]);
    // widen / narrow
    let w = |op| {
        one(
            |m| {
                let k = vk(m, int(true, 16), 4);
                let t = m.types.intern(k);
                let k2 = vk(m, int(true, 32), 2);
                let r = m.types.intern(k2);
                (vec![t], Some(r), vec![])
            },
            op,
            vec![ilanes(&[-1, 2, -32768, 32767])],
        )
        .unwrap()
    };
    assert_eq!(ival(&w(VecOp::WidenLo)), vec![-1, 2]);
    assert_eq!(ival(&w(VecOp::WidenHi)), vec![-32768, 32767]);
    let nar = |op| {
        one(
            |m| {
                let k = vk(m, int(true, 32), 2);
                let t = m.types.intern(k);
                let k2 = vk(m, int(true, 16), 4);
                let r = m.types.intern(k2);
                (vec![t, t], Some(r), vec![])
            },
            op,
            vec![ilanes(&[70000, -1]), ilanes(&[-70000, 5])],
        )
        .unwrap()
    };
    assert_eq!(ival(&nar(VecOp::NarrowSat)), vec![32767, -1, -32768, 5]);
    assert_eq!(ival(&nar(VecOp::NarrowWrap)), vec![4464, -1, -4464, 5]);
    // bitcast: f32 lanes to u32 lanes and u8 lanes
    let bc = one(
        |m| {
            let k = vk(m, F32, 2);
            let t = m.types.intern(k);
            let k2 = vk(m, int(false, 8), 8);
            let r = m.types.intern(k2);
            (vec![t], Some(r), vec![])
        },
        VecOp::Bitcast,
        vec![f32s(&[1.0, -0.0])],
    )
    .unwrap();
    assert_eq!(out_lanes(&bc), vec![0, 0, 0x80, 0x3f, 0, 0, 0, 0x80]);
    // split / join
    let sp = one(
        |m| {
            let t = i32x4(m);
            let k = vk(m, int(true, 32), 2);
            let h = m.types.intern(k);
            let tup = m.types.tuple(vec![h, h]);
            (vec![t], Some(tup), vec![])
        },
        VecOp::Split,
        vec![ilanes(&[1, 2, 3, 4])],
    )
    .unwrap();
    assert_eq!(ival(&sp.fields().unwrap()[1]), vec![3, 4]);
    let j = one(
        |m| {
            let t = i32x4(m);
            let k = vk(m, int(true, 32), 2);
            let h = m.types.intern(k);
            (vec![h, h], Some(t), vec![])
        },
        VecOp::Join,
        vec![ilanes(&[1, 2]), ilanes(&[3, 4])],
    )
    .unwrap();
    assert_eq!(ival(&j), vec![1, 2, 3, 4]);
    // scan_add
    let s = same(int(false, 8), 4, 1, VecOp::ScanAdd, vec![lanes(&[200, 50, 10, 1])]).unwrap();
    assert_eq!(out_lanes(&s), vec![200, 250, 4, 5]);
    // popcount / clz / not / andnot
    let pc = same(int(false, 16), 2, 1, VecOp::Popcount, vec![lanes(&[0xffff, 0x0101])]).unwrap();
    assert_eq!(out_lanes(&pc), vec![16, 2]);
    let cz = same(int(false, 16), 2, 1, VecOp::Clz, vec![lanes(&[0, 1])]).unwrap();
    assert_eq!(out_lanes(&cz), vec![16, 15]);
    let an =
        same(int(false, 8), 2, 2, VecOp::Andnot, vec![lanes(&[0xff, 0x0f]), lanes(&[0x0f, 0xff])]).unwrap();
    assert_eq!(out_lanes(&an), vec![0xf0, 0]);
}

#[test]
fn lane_casts() {
    let cast = |kind: CastKind, from: TypeKey, to: TypeKey, n: u8, v: Val, try_: bool| {
        one(
            |m| {
                let a = vk(m, from.clone(), n);
                let at = m.types.intern(a);
                let b = vk(m, to.clone(), n);
                let bt = m.types.intern(b);
                let r = if try_ { option(m, bt) } else { bt };
                (vec![at], Some(r), vec![])
            },
            VecOp::Cast(kind),
            vec![v],
        )
        .unwrap()
    };
    let q = f32::from_bits(QNAN);
    let r = cast(CastKind::Sat, F32, int(true, 32), 4, f32s(&[q, 1e10, -1e10, -1.9]), false);
    assert_eq!(ival(&r), vec![0, i32::MAX as i64, i32::MIN as i64, -1], "NaN → 0, saturate, truncate");
    let r = cast(CastKind::Exact, int(true, 32), F64, 2, ilanes(&[i32::MIN as i64, -1]), false);
    assert_eq!(
        out_lanes(&r).into_iter().map(f64::from_bits).collect::<Vec<_>>(),
        vec![i32::MIN as f64, -1.0]
    );
    let r = cast(CastKind::Round, int(true, 32), F32, 2, ilanes(&[16_777_217, -16_777_219]), false);
    assert_eq!(out_f32(&r), vec![16_777_216.0, -16_777_220.0], "nearest even");
    let r = cast(CastKind::Try, F32, int(true, 32), 2, f32s(&[1.0, 2.0]), true);
    assert_eq!(r.fields().unwrap()[0], Val::S(1));
    let r = cast(CastKind::Try, F32, int(true, 32), 2, f32s(&[1.0, 2.5]), true);
    assert_eq!(r.fields().unwrap(), &[Val::S(0)]);
    let r = cast(CastKind::Round, F32, F16, 2, f32s(&[1.0, 65520.0]), false);
    assert_eq!(out_lanes(&r), vec![0x3c00, 0x7c00], "rounds to inf past f16 max");
    let r = cast(CastKind::Wrap, int(true, 32), int(false, 8), 2, ilanes(&[257, -1]), false);
    assert_eq!(out_lanes(&r), vec![1, 255]);
}

/// Runs `f(slice, extra..)` over a `u32` slice of `data`; returns the outcome and the slice after.
fn run_slice(m: &Module, f: stave_ir::FuncId, data: &[u32], extra: Vec<Val>) -> (Outcome, Vec<u32>) {
    let mut it = Interp::new(m, host(), Config::default()).expect("verify");
    let a = it.mem.alloc((data.len() as u64 * 4).max(4), 4).unwrap();
    for (k, &x) in data.iter().enumerate() {
        it.mem.write_uint(a + 4 * k as u64, 4, x as u64).unwrap();
    }
    let mut args = vec![Val::Pair(a, data.len() as u64)];
    args.extend(extra);
    let o = it.run(f, args).expect("run");
    let after = (0..data.len()).map(|k| it.mem.read_uint(a + 4 * k as u64, 4).unwrap() as u32).collect();
    (o, after)
}

/// `f(s: &mut [u32], i: usize, m: mask32x4)` performing a vector memory op `op` at `i`, with a
/// `check.bounds i (sub.sat len 3)` first when `checked`. Loads return the vector; stores store
/// lanes `100..104`.
fn mem_fn(op: VecOp, checked: bool) -> (Module, stave_ir::FuncId) {
    let mut m = Module::new();
    let u = m.types.u32();
    let sl = m.types.slice(u);
    let rs = m.types.ref_(true, sl);
    let us = m.types.usize();
    let mt = m.types.intern(mk(32, 4));
    let k = vk(&mut m, int(false, 32), 4);
    let v = m.types.intern(k);
    let o = Op::Vec(op);
    let splat = Op::Vec(VecOp::Splat);
    let ret = if op.is_store() { None } else { Some(v) };
    let f = define(&mut m, &[0], "t.f", &[rs, us, mt], ret, |b, _| {
        let (s, i, mk_) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
        if checked {
            let n = b.op(Op::SliceLen, us, &[s.into()]);
            let three = b.int(3);
            let lim = b.op(Op::SubSat, us, &[n.into(), three]);
            b.bounds_check(i, lim);
        }
        let masked = matches!(op, VecOp::LoadMasked | VecOp::StoreMasked);
        if op.is_store() {
            let c = b.int(100);
            let x = b.op(splat, v, &[c]);
            let mut ops: Vec<Operand> = vec![s.into(), i.into(), x.into()];
            if masked {
                ops.push(mk_.into());
            }
            b.inst(o, None, &ops);
            b.ret(None);
        } else {
            let mut ops: Vec<Operand> = vec![s.into(), i.into()];
            if masked {
                ops.push(mk_.into());
            }
            let r = b.op(o, v, &ops);
            b.ret(Some(r.into()));
        }
    });
    (m, f)
}

fn trap_kind(o: &Outcome) -> Option<PanicKind> {
    match o {
        Outcome::Trap(t) => Some(t.kind),
        _ => None,
    }
}

#[test]
fn load_store_bounds() {
    let data: Vec<u32> = (0..6).collect();
    let all = bools(&[1, 1, 1, 1]);
    let (m, f) = mem_fn(VecOp::Load, true);
    stave_ir::bounds::audit(&m).expect("checked load is guarded");
    for i in 0..=6u64 {
        let (o, _) = run_slice(&m, f, &data, vec![Val::S(i), all.clone()]);
        if i + 4 <= 6 {
            let want: Vec<u64> = (i..i + 4).collect();
            assert_eq!(out_lanes(&ret_val(o)), want, "load at {i}");
        } else {
            assert_eq!(trap_kind(&o), Some(PanicKind::Bounds), "load at {i}");
        }
    }
    let (o, _) = run_slice(&m, f, &data, vec![Val::S(u64::MAX), all.clone()]);
    assert_eq!(trap_kind(&o), Some(PanicKind::Bounds), "huge index");
    let (m, f) = mem_fn(VecOp::Store, true);
    let (o, after) = run_slice(&m, f, &data, vec![Val::S(2), all.clone()]);
    assert_eq!(o, Outcome::Return(None));
    assert_eq!(after, vec![0, 1, 100, 100, 100, 100]);
    let (o, after) = run_slice(&m, f, &data, vec![Val::S(3), all.clone()]);
    assert_eq!(trap_kind(&o), Some(PanicKind::Bounds));
    assert_eq!(after, data, "nothing written on a trap");
    // without the check the audit reports the access
    let (m, _) = mem_fn(VecOp::Load, false);
    let bad = stave_ir::bounds::audit(&m).unwrap_err();
    assert_eq!(bad[0].kind, stave_ir::bounds::AccessKind::Vector);
}

#[test]
fn masked_forms_check_active_lanes_only() {
    let data: Vec<u32> = (0..6).collect();
    let (m, f) = mem_fn(VecOp::LoadMasked, false);
    stave_ir::bounds::audit(&m).expect("masked forms check themselves");
    // at 4: lanes 0, 1 in range; inactive lanes read 0 and are never touched
    let (o, _) = run_slice(&m, f, &data, vec![Val::S(4), bools(&[1, 1, 0, 0])]);
    assert_eq!(out_lanes(&ret_val(o)), vec![4, 5, 0, 0]);
    let (o, _) = run_slice(&m, f, &data, vec![Val::S(4), bools(&[1, 0, 1, 0])]);
    assert_eq!(trap_kind(&o), Some(PanicKind::Bounds));
    let (o, _) = run_slice(&m, f, &data, vec![Val::S(u64::MAX), bools(&[0, 0, 0, 0])]);
    assert_eq!(out_lanes(&ret_val(o)), vec![0, 0, 0, 0], "no active lane: no access");
    let (m, f) = mem_fn(VecOp::StoreMasked, false);
    let (o, after) = run_slice(&m, f, &data, vec![Val::S(3), bools(&[1, 0, 1, 0])]);
    assert_eq!(o, Outcome::Return(None));
    assert_eq!(after, vec![0, 1, 2, 100, 4, 100]);
    let (o, after) = run_slice(&m, f, &data, vec![Val::S(3), bools(&[1, 0, 1, 1])]);
    assert_eq!(trap_kind(&o), Some(PanicKind::Bounds));
    assert_eq!(after, data, "nothing written on a trap");
}

#[test]
fn aligned_forms_trap_on_misaligned_index() {
    let mut m = Module::new();
    let u = m.types.u32();
    let sl = m.types.intern(TypeKey::Slice { elem: u, align: Some(16) });
    let rs = m.types.ref_(false, sl);
    let us = m.types.usize();
    let k = vk(&mut m, int(false, 32), 4);
    let v = m.types.intern(k);
    let o = Op::Vec(VecOp::LoadAligned);
    let f = define(&mut m, &[0], "t.f", &[rs, us], Some(v), |b, _| {
        let (s, i) = (b.param(0).unwrap(), b.param(1).unwrap());
        let n = b.op(Op::SliceLen, us, &[s.into()]);
        let three = b.int(3);
        let lim = b.op(Op::SubSat, us, &[n.into(), three]);
        b.bounds_check(i, lim);
        let r = b.op(o, v, &[s.into(), i.into()]);
        b.ret(Some(r.into()));
    });
    verify_module(&m).unwrap();
    let data: Vec<u32> = (0..8).collect();
    let (o, _) = run_slice(&m, f, &data, vec![Val::S(4)]);
    assert_eq!(out_lanes(&ret_val(o)), vec![4, 5, 6, 7]);
    let (o, _) = run_slice(&m, f, &data, vec![Val::S(2)]);
    assert_eq!(trap_kind(&o), Some(PanicKind::Bounds));
    // an unaligned slice type is rejected
    let mut m2 = Module::new();
    let u = m2.types.u32();
    let sl = m2.types.slice(u);
    let rs = m2.types.ref_(false, sl);
    let us = m2.types.usize();
    let k = vk(&mut m2, int(false, 32), 4);
    let v = m2.types.intern(k);
    let o = Op::Vec(VecOp::LoadAligned);
    define(&mut m2, &[0], "t.f", &[rs, us], Some(v), |b, _| {
        let (s, i) = (b.param(0).unwrap(), b.param(1).unwrap());
        let r = b.op(o, v, &[s.into(), i.into()]);
        b.ret(Some(r.into()));
    });
    let e = verify_module(&m2).unwrap_err();
    assert!(e[0].msg.contains("aligned"), "{e:?}");
}

/// A gather / scatter of `u32x4` on `&mut [u32]` with index lanes `idx` (any width).
fn gather_scatter_with(op: VecOp, idx: TypeKey) -> (Module, stave_ir::FuncId) {
    let mut m = Module::new();
    let u = m.types.u32();
    let sl = m.types.slice(u);
    let rs = m.types.ref_(true, sl);
    let ik = vk(&mut m, idx, 4);
    let it = m.types.intern(ik);
    let k = vk(&mut m, int(false, 32), 4);
    let v = m.types.intern(k);
    let o = Op::Vec(op);
    let ret = if op == VecOp::Gather { Some(v) } else { None };
    let f = define(&mut m, &[0], "t.f", &[rs, it, v], ret, |b, _| {
        let (s, idx, x) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
        if op == VecOp::Gather {
            let r = b.op(o, v, &[s.into(), idx.into()]);
            b.ret(Some(r.into()));
        } else {
            b.inst(o, None, &[s.into(), idx.into(), x.into()]);
            b.ret(None);
        }
    });
    (m, f)
}

fn gather_scatter(op: VecOp) -> (Module, stave_ir::FuncId) {
    let (m, f) = gather_scatter_with(op, int(false, 16));
    verify_module(&m).unwrap();
    (m, f)
}

#[test]
fn gather_scatter_bounds_and_duplicates() {
    let data: Vec<u32> = (10..16).collect();
    let (m, f) = gather_scatter(VecOp::Gather);
    stave_ir::bounds::audit(&m).unwrap();
    let (o, _) = run_slice(&m, f, &data, vec![lanes(&[5, 0, 5, 2]), lanes(&[0; 4])]);
    assert_eq!(out_lanes(&ret_val(o)), vec![15, 10, 15, 12]);
    let (o, _) = run_slice(&m, f, &data, vec![lanes(&[0, 6, 0, 0]), lanes(&[0; 4])]);
    assert_eq!(trap_kind(&o), Some(PanicKind::Bounds));
    let (o, _) = run_slice(&m, f, &data, vec![lanes(&[0, 0xffff, 0, 0]), lanes(&[0; 4])]);
    assert_eq!(trap_kind(&o), Some(PanicKind::Bounds), "index lanes are zero-extended");
    let (m, f) = gather_scatter(VecOp::Scatter);
    let (o, after) = run_slice(&m, f, &data, vec![lanes(&[1, 3, 1, 1]), lanes(&[7, 8, 9, 20])]);
    assert_eq!(o, Outcome::Return(None));
    assert_eq!(after, vec![10, 20, 12, 8, 14, 15], "duplicate index: the highest lane wins");
    let (o, after) = run_slice(&m, f, &data, vec![lanes(&[0, 1, 2, 6]), lanes(&[1, 2, 3, 4])]);
    assert_eq!(trap_kind(&o), Some(PanicKind::Bounds));
    assert_eq!(after, data, "nothing written on a trap");
}

#[test]
fn gather_scatter_index_types() {
    for op in [VecOp::Gather, VecOp::Scatter] {
        for bits in [8, 16, 32, 64] {
            let (m, _) = gather_scatter_with(op, int(false, bits));
            verify_module(&m).unwrap_or_else(|e| panic!("{op:?} u{bits}: {e:?}"));
            let (m, _) = gather_scatter_with(op, int(true, bits));
            assert!(verify_module(&m).is_err(), "{op:?} i{bits} index accepted");
        }
    }
}

/// A chunk loop `for i = 0; i < len -| 3; i += 4 { load s i }` proves its loads.
#[test]
fn chunk_loop_proof() {
    let mut m = Module::new();
    let u = m.types.u32();
    let sl = m.types.slice(u);
    let rs = m.types.ref_(false, sl);
    let us = m.types.usize();
    let bt = m.types.bool();
    let k = vk(&mut m, int(false, 32), 4);
    let v = m.types.intern(k);
    let load = Op::Vec(VecOp::Load);
    let add = Op::Vec(VecOp::AddWrap);
    let splat = Op::Vec(VecOp::Splat);
    let f = define(&mut m, &[0], "t.f", &[rs], Some(v), |b, _| {
        let s = b.param(0).unwrap();
        let n = b.op(Op::SliceLen, us, &[s.into()]);
        let three = b.int(3);
        let lim = b.op(Op::SubSat, us, &[n.into(), three]);
        let z = b.int(0);
        let acc0 = b.op(splat, v, &[z]);
        let (head, hp) = b.create_block_with(&[us, v]);
        let (body, _) = b.create_block_with(&[]);
        let (exit, _) = b.create_block_with(&[]);
        b.br(head, &[z, acc0.into()]);
        b.switch_to(head);
        let c = b.cmp(Op::Lt, bt, hp[0], lim);
        b.cbr(c, body, &[], exit, &[]);
        b.switch_to(body);
        let x = b.op(load, v, &[s.into(), hp[0].into()]);
        let acc = b.op(add, v, &[hp[1].into(), x.into()]);
        let four = b.int(4);
        let i2 = b.op(Op::AddWrap, us, &[hp[0].into(), four]);
        b.br(head, &[i2.into(), acc.into()]);
        b.switch_to(exit);
        b.ret(Some(hp[1].into()));
    });
    verify_module(&m).unwrap();
    let acc = stave_ir::bounds::audit(&m).unwrap();
    assert_eq!(acc.len(), 1);
    assert_eq!(acc[0].guard, stave_ir::bounds::Guard::Proof(stave_ir::bounds::Proof::ChunkLoop));
    let data: Vec<u32> = (1..=10).collect();
    let (o, _) = run_slice(&m, f, &data, vec![]);
    // chunks [1..4] and [5..8]; 9, 10 are the tail
    assert_eq!(out_lanes(&ret_val(o)), vec![6, 8, 10, 12]);
}

/// Builds a case's (param types, result type).
type Sig = fn(&mut Module) -> (Vec<TypeId>, Option<TypeId>);

#[test]
fn verifier_rejects_mistyped_vector_ops() {
    let cases: Vec<(VecOp, Sig)> = vec![
        // float arithmetic on int lanes
        (VecOp::Add, |m| {
            let t = i32x4(m);
            (vec![t, t], Some(t))
        }),
        // wrapping on float lanes
        (VecOp::AddWrap, |m| {
            let k = vk(m, F32, 4);
            let t = m.types.intern(k);
            (vec![t, t], Some(t))
        }),
        // compare into a mask of the wrong width
        (VecOp::Lt, |m| {
            let t = i32x4(m);
            let mt = m.types.intern(mk(16, 4));
            (vec![t, t], Some(mt))
        }),
        // select with a mask of another lane count
        (VecOp::Select, |m| {
            let t = i32x4(m);
            let mt = m.types.intern(mk(32, 8));
            (vec![mt, t, t], Some(t))
        }),
        // reduction to the wrong type
        (VecOp::ReduceAddWrap, |m| {
            let t = i32x4(m);
            let u = m.types.u64();
            (vec![t], Some(u))
        }),
        // the plain add reduction takes float lanes only
        (VecOp::ReduceAdd, |m| {
            let t = i32x4(m);
            let i = m.types.i32();
            (vec![t], Some(i))
        }),
        // the wrapping add reduction takes int lanes only
        (VecOp::ReduceAddWrap, |m| {
            let k = vk(m, F32, 4);
            let t = m.types.intern(k);
            let f = m.types.f32();
            (vec![t], Some(f))
        }),
        // to_bits of 8 lanes is u8, not u16
        (VecOp::ToBits, |m| {
            let mt = m.types.intern(mk(32, 8));
            let u = m.types.u16();
            (vec![mt], Some(u))
        }),
        // a cast that changes the lane count
        (VecOp::Cast(CastKind::Exact), |m| {
            let t = i32x4(m);
            let k = vk(m, F64, 2);
            let r = m.types.intern(k);
            (vec![t], Some(r))
        }),
    ];
    for (op, build) in cases {
        let mut m = Module::new();
        let (ps, rt) = build(&mut m);
        let o = Op::Vec(op);
        define(&mut m, &[0], "t.f", &ps, rt, |b, _| {
            let ops: Vec<Operand> = b.params().iter().map(|v| (*v).into()).collect();
            let r = b.inst(o, rt, &ops);
            b.ret(r.map(Operand::from));
        });
        assert!(verify_module(&m).is_err(), "{op:?} accepted");
    }
}

#[test]
fn printed_as_ops() {
    let mut m = Module::new();
    let t = i32x4(&mut m);
    let mt = m.types.intern(mk(32, 4));
    let lt = Op::Vec(VecOp::Lt);
    let sel = Op::Vec(VecOp::Select);
    define(&mut m, &[0], "t.f", &[t, t], Some(t), |b, _| {
        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
        let c = b.op(lt, mt, &[x.into(), y.into()]);
        let r = b.op(sel, t, &[c.into(), x.into(), y.into()]);
        b.ret(Some(r.into()));
    });
    m.canonicalize_types().unwrap();
    let text = stave_ir::print_checked(&m);
    assert!(text.contains("= vec.lt t"), "{text}");
    assert!(text.contains("= mask.select t"), "{text}");
    assert!(!text.contains("intrinsic."), "{text}");
    let mut names: Vec<&str> = VecOp::ALL.iter().map(|v| v.name()).collect();
    assert!(names.iter().all(|n| n.starts_with("vec.") || n.starts_with("mask.")), "{names:?}");
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), VecOp::ALL.len());
}

#[test]
fn reserved_intrinsic_name_rejected() {
    let mut m = Module::new();
    let t = i32x4(&mut m);
    let id = m.intrinsic("vec.add.wrap");
    define(&mut m, &[0], "t.f", &[t, t], Some(t), |b, _| {
        let ops: Vec<Operand> = b.params().iter().map(|v| (*v).into()).collect();
        let r = b.intrinsic(id, Some(t), &ops);
        b.ret(r.map(Operand::from));
    });
    assert!(verify_module(&m).is_err());
}
