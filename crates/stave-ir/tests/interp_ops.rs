//! Semantics of every scalar op in the interpreter: overflow edges, MIN / -1, shifts ≥ width,
//! wrapping and saturating forms, IEEE floats, and the `as` family.

mod common;

use common::*;
use stave_ir::interp::{Outcome, Val};
use stave_ir::{
    AdtShape, CastKind, DeclId, FloatBits, FuncBuilder, IntBits, Module, Op, Operand, PanicKind, TypeKey,
};

fn int(signed: bool, bits: u8) -> TypeKey {
    let bits = match bits {
        8 => IntBits::B8,
        16 => IntBits::B16,
        32 => IntBits::B32,
        64 => IntBits::B64,
        _ => IntBits::Size,
    };
    TypeKey::Int { signed, bits }
}
fn i(bits: u8) -> TypeKey {
    int(true, bits)
}
fn u(bits: u8) -> TypeKey {
    int(false, bits)
}
const F32: TypeKey = TypeKey::Float(FloatBits::F32);
const F64: TypeKey = TypeKey::Float(FloatBits::F64);
const F16: TypeKey = TypeKey::Float(FloatBits::F16);
const BF16: TypeKey = TypeKey::Float(FloatBits::B16);

/// One instruction `op` over params of `params`, result `res`; returns the value or the trap kind.
fn one(params: &[TypeKey], res: TypeKey, op: Op, args: Vec<Val>) -> Result<Val, PanicKind> {
    let mut m = Module::new();
    let ps: Vec<_> = params.iter().map(|k| m.types.intern(k.clone())).collect();
    let rt = m.types.intern(res);
    let f = define(&mut m, &[0], "t.f", &ps, Some(rt), |b, _| {
        let ops: Vec<Operand> = b.params().iter().map(|v| (*v).into()).collect();
        let r = b.op(op, rt, &ops);
        b.ret(Some(r.into()));
    });
    match run(&m, f, args) {
        Outcome::Return(Some(v)) => Ok(v),
        Outcome::Trap(t) => Err(t.kind),
        o => panic!("unexpected {o:?}"),
    }
}

fn bits(r: Result<Val, PanicKind>) -> Result<u64, PanicKind> {
    r.map(|v| v.bits().expect("scalar"))
}

/// Binary op on one int type, values given as i128 (normalized by the harness).
fn ib(op: Op, k: TypeKey, a: i128, b: i128) -> Result<i128, PanicKind> {
    let (signed, w) = match &k {
        TypeKey::Int { signed, bits } => (*signed, bits.width()),
        _ => unreachable!(),
    };
    let t = stave_ir::arith::IntTy::new(w, signed);
    bits(one(&[k.clone(), k.clone()], k, op, vec![Val::S(t.wrap(a)), Val::S(t.wrap(b))])).map(|x| t.value(x))
}

const OV: Result<i128, PanicKind> = Err(PanicKind::Overflow);
const DZ: Result<i128, PanicKind> = Err(PanicKind::DivByZero);

#[test]
fn add_sub_mul_trap_on_overflow() {
    assert_eq!(ib(Op::Add, i(8), 127, 1), OV);
    assert_eq!(ib(Op::Add, i(8), -128, -1), OV);
    assert_eq!(ib(Op::Add, i(8), 100, 27), Ok(127));
    assert_eq!(ib(Op::Add, u(8), 255, 1), OV);
    assert_eq!(ib(Op::Add, u(64), u64::MAX as i128, 1), OV);
    assert_eq!(ib(Op::Add, i(64), i64::MAX as i128, 1), OV);
    assert_eq!(ib(Op::Add, u(0), 1, 2), Ok(3)); // usize
    assert_eq!(ib(Op::Sub, u(32), 0, 1), OV);
    assert_eq!(ib(Op::Sub, i(64), i64::MIN as i128, 1), OV);
    assert_eq!(ib(Op::Sub, i(16), -32768, -1), Ok(-32767));
    assert_eq!(ib(Op::Mul, i(32), 65536, 32768), OV);
    assert_eq!(ib(Op::Mul, i(32), -65536, 32768), Ok(i32::MIN as i128));
    assert_eq!(ib(Op::Mul, i(64), i64::MIN as i128, -1), OV);
    assert_eq!(ib(Op::Mul, u(64), u64::MAX as i128, 2), OV);
    assert_eq!(ib(Op::Mul, u(64), 1 << 32, 1 << 32), OV);
    assert_eq!(ib(Op::Mul, u(64), u64::MAX as i128, u64::MAX as i128), OV);
    assert_eq!(ib(Op::Mul, u(64), 1 << 32, (1 << 32) - 1), Ok((1i128 << 64) - (1 << 32)));
    assert_eq!(ib(Op::Mul, i(8), -1, -128), OV);
}

#[test]
fn div_rem_traps() {
    assert_eq!(ib(Op::Div, i(32), i32::MIN as i128, -1), OV);
    assert_eq!(ib(Op::Div, i(8), -128, -1), OV);
    assert_eq!(ib(Op::Div, i(64), i64::MIN as i128, -1), OV);
    assert_eq!(ib(Op::Rem, i(32), i32::MIN as i128, -1), OV);
    assert_eq!(ib(Op::Div, i(32), 5, 0), DZ);
    assert_eq!(ib(Op::Rem, i(32), 5, 0), DZ);
    assert_eq!(ib(Op::Div, u(8), 7, 0), DZ);
    assert_eq!(ib(Op::Div, i(32), -7, 2), Ok(-3));
    assert_eq!(ib(Op::Rem, i(32), -7, 2), Ok(-1));
    assert_eq!(ib(Op::Rem, i(32), 7, -2), Ok(1));
    assert_eq!(ib(Op::Div, i(32), i32::MIN as i128, 1), Ok(i32::MIN as i128));
    assert_eq!(ib(Op::Div, u(64), u64::MAX as i128, 3), Ok((u64::MAX / 3) as i128));
    assert_eq!(ib(Op::Rem, u(8), 255, 16), Ok(15));
}

#[test]
fn wrapping_ops() {
    assert_eq!(ib(Op::AddWrap, i(8), 127, 1), Ok(-128));
    assert_eq!(ib(Op::SubWrap, u(8), 0, 1), Ok(255));
    assert_eq!(ib(Op::MulWrap, i(64), i64::MIN as i128, -1), Ok(i64::MIN as i128));
    assert_eq!(ib(Op::MulWrap, u(16), 300, 300), Ok((90000 % 65536) as i128));
    assert_eq!(ib(Op::AddWrap, u(64), u64::MAX as i128, 2), Ok(1));
    assert_eq!(ib(Op::MulWrap, i(32), 65536, 32768), Ok(i32::MIN as i128));
}

#[test]
fn saturating_ops() {
    assert_eq!(ib(Op::AddSat, i(8), 100, 100), Ok(127));
    assert_eq!(ib(Op::AddSat, i(8), -100, -100), Ok(-128));
    assert_eq!(ib(Op::AddSat, u(8), 200, 100), Ok(255));
    assert_eq!(ib(Op::SubSat, u(8), 5, 10), Ok(0));
    assert_eq!(ib(Op::SubSat, i(32), i32::MIN as i128, 1), Ok(i32::MIN as i128));
    assert_eq!(ib(Op::SubSat, i(32), i32::MAX as i128, -1), Ok(i32::MAX as i128));
    assert_eq!(ib(Op::MulSat, i(32), i32::MIN as i128, -1), Ok(i32::MAX as i128));
    assert_eq!(ib(Op::MulSat, i(64), i64::MIN as i128, i64::MIN as i128), Ok(i64::MAX as i128));
    assert_eq!(ib(Op::MulSat, i(64), i64::MIN as i128, 2), Ok(i64::MIN as i128));
    assert_eq!(ib(Op::MulSat, i(64), i64::MAX as i128, -2), Ok(i64::MIN as i128));
    assert_eq!(ib(Op::MulSat, u(64), u64::MAX as i128, u64::MAX as i128), Ok(u64::MAX as i128));
    assert_eq!(ib(Op::MulSat, u(8), 16, 16), Ok(255));
    assert_eq!(ib(Op::AddSat, u(64), u64::MAX as i128, 1), Ok(u64::MAX as i128));
    assert_eq!(ib(Op::AddSat, i(16), 1, 2), Ok(3));
}

#[test]
fn bitwise_and_counts() {
    assert_eq!(ib(Op::And, i(8), -1, 0x0f), Ok(0x0f));
    assert_eq!(ib(Op::Or, u(8), 0xf0, 0x0f), Ok(0xff));
    assert_eq!(ib(Op::Xor, i(16), -1, 1), Ok(-2));
    let un = |op, k: TypeKey, a: u64| bits(one(std::slice::from_ref(&k), k.clone(), op, vec![Val::S(a)]));
    assert_eq!(un(Op::Not, i(8), 0), Ok(u64::MAX)); // -1i8 normalized
    assert_eq!(un(Op::Not, u(8), 0), Ok(255));
    assert_eq!(un(Op::Not, TypeKey::Bool, 1), Ok(0));
    assert_eq!(un(Op::Clz, u(16), 1), Ok(15));
    assert_eq!(un(Op::Clz, u(16), 0), Ok(16));
    assert_eq!(un(Op::Clz, i(8), (-1i64) as u64), Ok(0));
    assert_eq!(un(Op::Ctz, i(8), 0), Ok(8));
    assert_eq!(un(Op::Ctz, u(64), 1 << 40), Ok(40));
    assert_eq!(un(Op::Popcount, i(8), (-1i64) as u64), Ok(8));
    assert_eq!(un(Op::Popcount, u(32), 0xf0f0), Ok(8));
    let bb = |op, a: u64, b: u64| {
        bits(one(&[TypeKey::Bool, TypeKey::Bool], TypeKey::Bool, op, vec![Val::S(a), Val::S(b)]))
    };
    assert_eq!(bb(Op::And, 1, 0), Ok(0));
    assert_eq!(bb(Op::Or, 1, 0), Ok(1));
    assert_eq!(bb(Op::Xor, 1, 1), Ok(0));
}

#[test]
fn neg_traps_on_min() {
    let neg =
        |k: TypeKey, a: i64| bits(one(std::slice::from_ref(&k), k.clone(), Op::Neg, vec![Val::S(a as u64)]));
    assert_eq!(neg(i(32), i32::MIN as i64), Err(PanicKind::Overflow));
    assert_eq!(neg(i(8), -128), Err(PanicKind::Overflow));
    assert_eq!(neg(i(64), i64::MIN), Err(PanicKind::Overflow));
    assert_eq!(neg(i(8), 5), Ok((-5i64) as u64));
    assert_eq!(neg(i(8), -127), Ok(127));
    let fneg = |x: f64| bits(one(&[F64], F64, Op::Neg, vec![Val::f64(x)]));
    assert_eq!(fneg(0.0), Ok((-0.0f64).to_bits()));
    assert_eq!(fneg(f64::NAN), Ok(f64::NAN.to_bits() ^ (1 << 63)));
    assert_eq!(bits(one(&[F32], F32, Op::Neg, vec![Val::f32(1.5)])), Ok((-1.5f32).to_bits() as u64));
}

fn sh(op: Op, k: TypeKey, ak: TypeKey, a: i128, n: i128) -> Result<i128, PanicKind> {
    let (signed, w) = match &k {
        TypeKey::Int { signed, bits } => (*signed, bits.width()),
        _ => unreachable!(),
    };
    let t = stave_ir::arith::IntTy::new(w, signed);
    let at = match &ak {
        TypeKey::Int { signed, bits } => stave_ir::arith::IntTy::new(bits.width(), *signed),
        _ => unreachable!(),
    };
    bits(one(&[k.clone(), ak], k, op, vec![Val::S(t.wrap(a)), Val::S(at.wrap(n))])).map(|x| t.value(x))
}

#[test]
fn shifts() {
    assert_eq!(sh(Op::Shl, u(32), u(32), 1, 32), OV);
    assert_eq!(sh(Op::Shl, u(32), u(32), 1, 31), Ok(1 << 31));
    assert_eq!(sh(Op::Shl, u(32), u(8), 3, 255), OV);
    assert_eq!(sh(Op::Shl, i(8), u(32), 1, 7), Ok(-128)); // bits shifted out: no trap
    assert_eq!(sh(Op::Shl, i(8), u(32), 3, 7), Ok(-128));
    assert_eq!(sh(Op::Shl, u(64), u(64), 1, 64), OV);
    assert_eq!(sh(Op::Shl, u(64), u(64), 1, 63), Ok(1 << 63));
    assert_eq!(sh(Op::Shl, u(32), u(64), 1, 1 << 40), OV);
    assert_eq!(sh(Op::Shl, i(32), i(32), 1, -1), OV); // negative amount
    assert_eq!(sh(Op::Shr, i(8), u(8), -128, 7), Ok(-1)); // arithmetic
    assert_eq!(sh(Op::Shr, i(8), u(8), -128, 8), OV);
    assert_eq!(sh(Op::Shr, u(8), u(8), 128, 7), Ok(1)); // logical
    assert_eq!(sh(Op::Shr, i(64), u(32), -1, 63), Ok(-1));
    assert_eq!(sh(Op::Shr, u(64), u(32), u64::MAX as i128, 63), Ok(1));
    assert_eq!(sh(Op::Shr, i(32), u(32), -8, 1), Ok(-4));
    assert_eq!(sh(Op::ShlWrap, u(32), u(32), 1, 33), Ok(2));
    assert_eq!(sh(Op::ShlWrap, u(32), u(32), 1, 32), Ok(1));
    assert_eq!(sh(Op::ShlWrap, i(8), i(8), 1, -1), Ok(-128)); // -1 & 7 = 7
    assert_eq!(sh(Op::ShlWrap, u(64), u(64), 1, 64 + 5), Ok(32));
}

#[test]
fn shift_by_constant_amount() {
    let mut m = Module::new();
    let t = m.types.u16();
    let f = define(&mut m, &[0], "t.f", &[t], Some(t), |b, _| {
        let x = b.param(0).unwrap();
        let k = b.int(16);
        let r = b.binary(Op::Shl, t, x, k);
        b.ret(Some(r.into()));
    });
    assert_eq!(
        run(&m, f, vec![Val::S(1)]),
        Outcome::Trap(stave_ir::Trap { kind: PanicKind::Overflow, site: stave_ir::Site::NONE, func: f })
    );
}

#[test]
fn mul_wide() {
    let mut m = Module::new();
    let t = m.types.u64();
    let pair = m.types.tuple(vec![t, t]);
    let f = define(&mut m, &[0], "t.f", &[t, t], Some(pair), |b, _| {
        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
        let r = b.binary(Op::MulWide, pair, x, y);
        b.ret(Some(r.into()));
    });
    assert_eq!(
        ret_val(run(&m, f, vec![Val::S(u64::MAX), Val::S(u64::MAX)])),
        Val::agg(vec![Val::S(u64::MAX - 1), Val::S(1)])
    );
    let mut m = Module::new();
    let t = m.types.u32();
    let pair = m.types.tuple(vec![t, t]);
    let f = define(&mut m, &[0], "t.f", &[t, t], Some(pair), |b, _| {
        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
        let r = b.binary(Op::MulWide, pair, x, y);
        b.ret(Some(r.into()));
    });
    assert_eq!(
        ret_val(run(&m, f, vec![Val::S(0xffff_ffff), Val::S(2)])),
        Val::agg(vec![Val::S(1), Val::S(0xffff_fffe)])
    );
}

fn cmp(op: Op, k: TypeKey, a: Val, b: Val) -> bool {
    bits(one(&[k.clone(), k], TypeKey::Bool, op, vec![a, b])).unwrap() == 1
}

#[test]
fn compares() {
    let si = |x: i64| Val::S(x as u64);
    assert!(cmp(Op::Lt, i(8), si(-1), si(0)));
    assert!(!cmp(Op::Lt, u(8), Val::S(255), Val::S(0)));
    assert!(cmp(Op::Gt, u(8), Val::S(255), Val::S(0)));
    assert!(cmp(Op::Ge, i(64), si(i64::MIN), si(i64::MIN)));
    assert!(cmp(Op::Le, u(64), Val::S(1), Val::S(u64::MAX)));
    assert!(cmp(Op::Ne, i(32), si(1), si(2)));
    assert!(cmp(Op::Eq, TypeKey::Char, Val::S('a' as u64), Val::S(97)));
    assert!(cmp(Op::Lt, TypeKey::Char, Val::S('a' as u64), Val::S('é' as u64)));
    assert!(cmp(Op::Lt, TypeKey::Bool, Val::S(0), Val::S(1)));
    let nan = Val::f64(f64::NAN);
    assert!(!cmp(Op::Eq, F64.clone(), nan.clone(), nan.clone()));
    assert!(cmp(Op::Ne, F64.clone(), nan.clone(), nan.clone()));
    assert!(!cmp(Op::Lt, F64.clone(), nan.clone(), Val::f64(1.0)));
    assert!(!cmp(Op::Ge, F64.clone(), nan.clone(), Val::f64(1.0)));
    assert!(cmp(Op::Eq, F64.clone(), Val::f64(-0.0), Val::f64(0.0)));
    assert!(cmp(Op::Lt, F32.clone(), Val::f32(-1.0), Val::f32(f32::INFINITY)));
    assert!(!cmp(Op::Gt, F32.clone(), Val::f32(f32::NAN), Val::f32(0.0)));
}

#[test]
fn compare_with_constant() {
    let mut m = Module::new();
    let t = m.types.i8();
    let bt = m.types.bool();
    let f = define(&mut m, &[0], "t.f", &[t], Some(bt), |b, _| {
        let x = b.param(0).unwrap();
        let k = b.int(-1);
        let r = b.cmp(Op::Lt, bt, k, x);
        b.ret(Some(r.into()));
    });
    assert_eq!(ret_val(run(&m, f, vec![Val::S(0)])), Val::S(1));
    assert_eq!(ret_val(run(&m, f, vec![Val::S((-2i64) as u64)])), Val::S(0));
}

#[test]
fn select() {
    let mut m = Module::new();
    let t = m.types.i32();
    let bt = m.types.bool();
    let f = define(&mut m, &[0], "t.f", &[bt, t, t], Some(t), |b, _| {
        let p = b.params().to_vec();
        let r = b.select(t, p[0], p[1], p[2]);
        b.ret(Some(r.into()));
    });
    assert_eq!(ret_val(run(&m, f, vec![Val::S(1), Val::S(7), Val::S(9)])), Val::S(7));
    assert_eq!(ret_val(run(&m, f, vec![Val::S(0), Val::S(7), Val::S(9)])), Val::S(9));
}

#[test]
fn float_arithmetic_is_ieee() {
    let f = |op, a: f64, b: f64| {
        f64::from_bits(bits(one(&[F64, F64], F64, op, vec![Val::f64(a), Val::f64(b)])).unwrap())
    };
    assert_eq!(f(Op::Add, 0.1, 0.2).to_bits(), (0.1f64 + 0.2).to_bits());
    assert_eq!(f(Op::Div, 1.0, 0.0), f64::INFINITY);
    assert_eq!(f(Op::Div, -1.0, 0.0), f64::NEG_INFINITY);
    assert!(f(Op::Div, 0.0, 0.0).is_nan());
    assert_eq!(f(Op::Rem, 5.5, 2.0), 1.5);
    assert_eq!(f(Op::Rem, -5.5, 2.0), -1.5);
    assert_eq!(f(Op::Sub, 1e308, -1e308), f64::INFINITY);
    assert_eq!(f(Op::Mul, 1e-310, 1e-10).to_bits(), (1e-310f64 * 1e-10).to_bits()); // denormal, no FTZ
    assert_eq!(f(Op::Add, -0.0, 0.0).to_bits(), 0.0f64.to_bits());
    let g = |op, a: f32, b: f32| {
        f32::from_bits(bits(one(&[F32, F32], F32, op, vec![Val::f32(a), Val::f32(b)])).unwrap() as u32)
    };
    assert_eq!(g(Op::Div, 1.0, 3.0).to_bits(), (1.0f32 / 3.0).to_bits());
    assert_eq!(g(Op::Mul, 1e30, 1e30), f32::INFINITY);
    assert_eq!(g(Op::Add, 16777216.0, 1.0), 16777216.0); // RNE, ties to even
    assert_eq!(g(Op::Add, 16777216.0, 3.0), 16777220.0);
}

// ---- casts ----

/// One cast; `Err(None)` = rejected by the verifier, `Err(Some(kind))` = trap.
fn cast(kind: CastKind, from: TypeKey, to: TypeKey, x: Val) -> Result<Val, Option<PanicKind>> {
    let mut m = Module::new();
    let ft = m.types.intern(from);
    let tt = m.types.intern(to);
    let rt = if kind == CastKind::Try {
        m.types.adt(
            DeclId(1),
            vec![stave_ir::GenericArg::Type(tt)],
            AdtShape::enumeration(vec![vec![], vec![tt]]),
        )
    } else {
        tt
    };
    let f = m.declare(inst(&[0]), "t.f", vec![ft], Some(rt));
    let mut b = FuncBuilder::new(&[ft]);
    let x0 = b.param(0).unwrap();
    let r = b.cast(kind, rt, x0);
    b.ret(Some(r.into()));
    m.define(f, b.finish().unwrap());
    if stave_ir::verify_module(&m).is_err() {
        return Err(None);
    }
    match run(&m, f, vec![x]) {
        Outcome::Return(Some(v)) => Ok(v),
        Outcome::Trap(t) => Err(Some(t.kind)),
        o => panic!("{o:?}"),
    }
}

fn cs(kind: CastKind, from: TypeKey, to: TypeKey, x: Val) -> u64 {
    cast(kind, from, to, x).unwrap().bits().unwrap()
}

fn some(v: u64) -> Val {
    Val::agg(vec![Val::S(1), Val::S(v)])
}
fn none() -> Val {
    Val::agg(vec![Val::S(0)])
}

#[test]
fn cast_exact() {
    use CastKind::Exact as E;
    assert_eq!(cs(E, i(8), i(64), Val::S(u64::MAX)), u64::MAX);
    assert_eq!(cs(E, u(8), i(16), Val::S(255)), 255);
    assert_eq!(cs(E, u(32), i(64), Val::S(0xffff_ffff)), 0xffff_ffff);
    assert_eq!(cs(E, u(32), u(0), Val::S(7)), 7);
    assert_eq!(cs(E, TypeKey::Bool, u(8), Val::S(1)), 1);
    assert_eq!(cs(E, TypeKey::Char, u(32), Val::S(0x10ffff)), 0x10ffff);
    assert_eq!(cs(E, u(8), TypeKey::Char, Val::S(0xe9)), 0xe9);
    assert_eq!(cs(E, i(32), F64, Val::S(i32::MIN as i64 as u64)), (i32::MIN as f64).to_bits());
    assert_eq!(cs(E, u(16), F32, Val::S(65535)), (65535.0f32).to_bits() as u64);
    assert_eq!(cs(E, F32, F64, Val::f32(0.1)), (0.1f32 as f64).to_bits());
    assert_eq!(cs(E, F16, F32, Val::S(0x3c00)), 1.0f32.to_bits() as u64);
    assert_eq!(cs(E, BF16, F64, Val::S(0xc0a0)), (-5.0f64).to_bits());
    // lossy plain `as` is rejected
    for (a, b) in [
        (i(32), F32),
        (i(64), i(32)),
        (u(32), i(32)),
        (i(8), u(64)),
        (F64, F32),
        (F64, i(64)),
        (u(16), TypeKey::Char),
    ] {
        assert_eq!(cast(E, a.clone(), b.clone(), Val::S(0)), Err(None), "{a:?} as {b:?}");
    }
}

#[test]
fn cast_wrap() {
    use CastKind::Wrap as W;
    assert_eq!(cs(W, i(64), i(8), Val::S(300)), 44);
    assert_eq!(cs(W, i(64), i(8), Val::S((-129i64) as u64)), 127);
    assert_eq!(cs(W, u(64), i(32), Val::S(0xffff_ffff)), u64::MAX);
    assert_eq!(cs(W, i(8), u(64), Val::S(u64::MAX)), u64::MAX);
    assert_eq!(cs(W, i(8), u(16), Val::S(u64::MAX)), 0xffff);
    assert_eq!(cs(W, TypeKey::Char, u(8), Val::S(0x1e9)), 0xe9);
    assert_eq!(cs(W, TypeKey::Bool, i(8), Val::S(1)), 1);
    assert_eq!(cast(W, F64, i(32), Val::f64(1.0)), Err(None));
}

#[test]
fn cast_sat() {
    use CastKind::Sat as S;
    let si = |x: i64| x as u64;
    assert_eq!(cs(S, i(64), u(8), Val::S(si(-5))), 0);
    assert_eq!(cs(S, i(64), u(8), Val::S(300)), 255);
    assert_eq!(cs(S, u(64), i(64), Val::S(u64::MAX)), i64::MAX as u64);
    assert_eq!(cs(S, i(32), i(8), Val::S(si(-1000))), si(-128));
    assert_eq!(cs(S, F64, i(32), Val::f64(f64::NAN)), 0);
    assert_eq!(cs(S, F64, i(32), Val::f64(-f64::NAN)), 0);
    assert_eq!(cs(S, F64, i(32), Val::f64(f64::INFINITY)), i32::MAX as u64);
    assert_eq!(cs(S, F64, i(32), Val::f64(f64::NEG_INFINITY)), si(i32::MIN as i64));
    assert_eq!(cs(S, F64, i(32), Val::f64(1e10)), i32::MAX as u64);
    assert_eq!(cs(S, F64, i(32), Val::f64(-3.9)), si(-3));
    assert_eq!(cs(S, F64, i(32), Val::f64(3.9)), 3);
    assert_eq!(cs(S, F64, i(32), Val::f64(-0.0)), 0);
    assert_eq!(cs(S, F32, u(8), Val::f32(-1.0)), 0);
    assert_eq!(cs(S, F32, u(8), Val::f32(255.5)), 255);
    assert_eq!(cs(S, F32, u(8), Val::f32(256.0)), 255);
    assert_eq!(cs(S, F64, u(64), Val::f64(18446744073709551615.0)), u64::MAX);
    assert_eq!(cs(S, F64, u(64), Val::f64(18446744073709549568.0)), 18446744073709549568);
    assert_eq!(cs(S, F64, i(64), Val::f64(9223372036854775808.0)), i64::MAX as u64);
    assert_eq!(cs(S, F64, i(64), Val::f64(-9223372036854775808.0)), si(i64::MIN));
    assert_eq!(cs(S, F64, i(64), Val::f64(-9223372036854777856.0)), si(i64::MIN));
    assert_eq!(cs(S, F32, i(64), Val::f32(f32::MAX)), i64::MAX as u64);
}

#[test]
fn cast_round() {
    use CastKind::Round as N;
    let f64b = |x: f64| x.to_bits();
    let f32b = |x: f32| x.to_bits() as u64;
    assert_eq!(cs(N, i(64), F64, Val::S(9007199254740993)), f64b(9007199254740992.0)); // tie → even
    assert_eq!(cs(N, i(64), F64, Val::S(9007199254740995)), f64b(9007199254740996.0));
    assert_eq!(cs(N, i(64), F64, Val::S(i64::MIN as u64)), f64b(-9223372036854775808.0));
    assert_eq!(cs(N, u(64), F32, Val::S(u64::MAX)), f32b(18446744073709551616.0));
    assert_eq!(cs(N, i(32), F32, Val::S(16777217)), f32b(16777216.0));
    assert_eq!(cs(N, i(32), F32, Val::S(16777219)), f32b(16777220.0));
    assert_eq!(cs(N, F64, F32, Val::f64(0.1)), f32b(0.1));
    assert_eq!(cs(N, F64, F32, Val::f64(1.0 + 2f64.powi(-24))), f32b(1.0)); // tie → even
    assert_eq!(cs(N, F64, F32, Val::f64(1.0 + 3.0 * 2f64.powi(-24))), f32b(1.0 + 2f32.powi(-22)));
    assert_eq!(cs(N, F64, F32, Val::f64(1.0 + 2f64.powi(-24) + 2f64.powi(-50))), f32b(1.0 + 2f32.powi(-23)));
    assert_eq!(cs(N, F64, F32, Val::f64(1e39)), f32b(f32::INFINITY));
    assert_eq!(cs(N, F64, F32, Val::f64(-1e-50)), f32b(-0.0));
    assert!(f32::from_bits(cs(N, F64, F32, Val::f64(f64::NAN)) as u32).is_nan());
    // f16: max 65504; 65520 is the tie between 65504 and 65536 (overflow) → even → inf
    assert_eq!(cs(N, F32, F16, Val::f32(65520.0)), 0x7c00);
    assert_eq!(cs(N, F32, F16, Val::f32(65519.0)), 0x7bff);
    assert_eq!(cs(N, F32, F16, Val::f32(1.0 / 3.0)), 0x3555);
    assert_eq!(cs(N, F64, F16, Val::f64(2f64.powi(-24))), 0x0001); // min subnormal
    assert_eq!(cs(N, F64, F16, Val::f64(2f64.powi(-25))), 0x0000); // tie → even (0)
    assert_eq!(cs(N, F64, F16, Val::f64(1.5 * 2f64.powi(-24))), 0x0002); // tie → even (2)
    assert_eq!(cs(N, F64, F16, Val::f64(2f64.powi(-14) - 2f64.powi(-25))), 0x0400); // rounds up to min normal
    assert_eq!(cs(N, F64, F16, Val::f64(-0.0)), 0x8000);
    assert_eq!(cs(N, F32, F16, Val::f32(f32::NAN)), 0x7e00);
    assert_eq!(cs(N, F32, F16, Val::f32(f32::NEG_INFINITY)), 0xfc00);
    // bf16
    assert_eq!(cs(N, F32, BF16, Val::f32(1.0 + 2f32.powi(-8))), 0x3f80); // tie → even
    assert_eq!(cs(N, F32, BF16, Val::f32(1.0 + 3.0 * 2f32.powi(-8))), 0x3f82); // tie → even
    assert_eq!(cs(N, F32, BF16, Val::f32(f32::MAX)), 0x7f80); // overflow → inf
    // int → f16
    assert_eq!(cs(N, i(32), F16, Val::S(2049)), 0x6800); // 2048
    assert_eq!(cs(N, i(32), F16, Val::S(2051)), 0x6802); // 2052
    assert_eq!(cs(N, i(32), F16, Val::S(70000)), 0x7c00);
    assert_eq!(cs(N, i(32), F16, Val::S(u64::MAX)), 0xbc00); // -1
    assert_eq!(cs(N, u(64), BF16, Val::S(u64::MAX)), 0x5f80); // 2^64
    assert_eq!(cast(N, i(32), i(64), Val::S(0)), Err(None));
}

#[test]
fn cast_try() {
    use CastKind::Try as T;
    let c = |from, to, x| cast(T, from, to, x).unwrap();
    assert_eq!(c(i(64), u(8), Val::S(300)), none());
    assert_eq!(c(i(64), u(8), Val::S(200)), some(200));
    assert_eq!(c(i(64), u(8), Val::S(u64::MAX)), none());
    assert_eq!(c(u(64), i(64), Val::S(u64::MAX)), none());
    assert_eq!(c(i(8), i(64), Val::S(u64::MAX)), some(u64::MAX));
    assert_eq!(c(F64, i(32), Val::f64(2.5)), none());
    assert_eq!(c(F64, i(32), Val::f64(2.0)), some(2));
    assert_eq!(c(F64, i(32), Val::f64(-2147483648.0)), some((-2147483648i64) as u64));
    assert_eq!(c(F64, i(32), Val::f64(2147483648.0)), none());
    assert_eq!(c(F64, i(32), Val::f64(f64::NAN)), none());
    assert_eq!(c(F64, u(64), Val::f64(18446744073709549568.0)), some(18446744073709549568));
    assert_eq!(c(F64, u(64), Val::f64(18446744073709551616.0)), none());
    assert_eq!(c(u(32), TypeKey::Char, Val::S(0xd800)), none());
    assert_eq!(c(u(32), TypeKey::Char, Val::S(0x110000)), none());
    assert_eq!(c(u(32), TypeKey::Char, Val::S(0x41)), some(0x41));
    assert_eq!(c(i(64), F64, Val::S(9007199254740993)), none());
    assert_eq!(c(i(64), F64, Val::S(9007199254740992)), some(9007199254740992f64.to_bits()));
    assert_eq!(c(u(64), F64, Val::S(u64::MAX)), none());
    assert_eq!(c(F64, F32, Val::f64(0.1)), none());
    assert_eq!(c(F64, F32, Val::f64(0.5)), some(0.5f32.to_bits() as u64));
    assert!(matches!(c(F64, F32, Val::f64(f64::NAN)).fields().unwrap()[0], Val::S(1)));
    assert_eq!(c(TypeKey::Char, u(8), Val::S(0x100)), none());
}

#[test]
fn cast_of_constant_operand() {
    let mut m = Module::new();
    let t = m.types.u8();
    let f = define(&mut m, &[0], "t.f", &[], Some(t), |b, _| {
        let k = b.int(300);
        let r = b.cast(CastKind::Wrap, t, k);
        b.ret(Some(r.into()));
    });
    assert_eq!(ret_val(run(&m, f, vec![])), Val::S(44));
}
