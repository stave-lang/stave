//! Every scalar op of the stave-ir op table over edge values, native (qemu) vs the interpreter:
//! integer ops of every width and signedness (trapping, wrapping, saturating, bitwise, shifts,
//! counts, compares, `mul.wide`), with register and constant operands; IEEE f32/f64 ops and
//! compares; the whole `as` family between every pair of scalar types; trap kinds and sites.

mod common;

use common::*;
use stave_ir::arith::{self, CastTy, IntTy};
use stave_ir::{CastKind, FloatBits, FuncBuilder, FuncId, Op, Operand, PanicKind, Site, TypeId};

fn int_ty(p: &Prog, t: TypeId) -> IntTy {
    let tt = p.t;
    let (bits, signed) = if t == tt.u8 {
        (8, false)
    } else if t == tt.u16 {
        (16, false)
    } else if t == tt.u32 {
        (32, false)
    } else if t == tt.u64 || t == tt.usize {
        (64, false)
    } else if t == tt.i8 {
        (8, true)
    } else if t == tt.i16 {
        (16, true)
    } else if t == tt.i32 {
        (32, true)
    } else {
        (64, true)
    };
    IntTy::new(bits, signed)
}

fn edges(it: IntTy) -> Vec<i128> {
    let (min, max) = (it.min(), it.max());
    let mut v =
        vec![0, 1, 2, 3, 7, 100, max, max - 1, max / 2, max / 2 + 1, min, min + 1, min / 2, -1, -2, -100];
    v.push(0x5555_5555_5555_5555 & max);
    v.retain(|x| it.fits(*x));
    v.sort();
    v.dedup();
    v
}

fn cint(b: &mut FuncBuilder, v: i128) -> Operand {
    b.int(v)
}

const BIN: [Op; 14] = [
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Div,
    Op::Rem,
    Op::AddWrap,
    Op::SubWrap,
    Op::MulWrap,
    Op::AddSat,
    Op::SubSat,
    Op::MulSat,
    Op::And,
    Op::Or,
    Op::Xor,
];
const CMP: [Op; 6] = [Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge];
const SHIFT: [Op; 3] = [Op::Shl, Op::Shr, Op::ShlWrap];

/// `fn(a: ty, b: ty) -> rt { op a b }` or, with `rhs`, `fn(a: ty) -> rt { op a rhs }`, or with
/// `lhs`, `fn(b: ty) -> rt { op lhs b }`.
fn bin_fn(
    p: &mut Prog,
    op: Op,
    ty: TypeId,
    bty: TypeId,
    rt: TypeId,
    lhs: Option<i128>,
    rhs: Option<i128>,
) -> FuncId {
    let params: Vec<TypeId> = match (lhs, rhs) {
        (None, None) => vec![ty, bty],
        (None, Some(_)) => vec![ty],
        _ => vec![bty],
    };
    p.func(&format!("t.{op:?}"), &params, Some(rt), |b, _| {
        let (x, y): (Operand, Operand) = match (lhs, rhs) {
            (None, None) => (b.param(0).unwrap().into(), b.param(1).unwrap().into()),
            (None, Some(c)) => (b.param(0).unwrap().into(), cint(b, c)),
            (Some(c), _) => (cint(b, c), b.param(0).unwrap().into()),
        };
        b.set_site(Site(500));
        let r = b.op(op, rt, &[x, y]);
        b.ret(Some(r.into()));
    })
}

fn semantics(op: Op, it: IntTy, a: i128, c: i128) -> Result<(), PanicKind> {
    let (x, y) = (it.wrap(a), it.wrap(c));
    match op {
        Op::Shl | Op::Shr | Op::ShlWrap => arith::int_shift(op, it, x, c).map(|_| ()),
        o if CMP.contains(&o) => Ok(()),
        _ => arith::int_binary(op, it, x, y).map(|_| ()),
    }
}

/// Integer binary ops, compares, and shifts on every int type, register and constant forms;
/// only non-trapping inputs (traps end the program; see `int_traps`).
#[test]
fn int_binary_ops() {
    let mut p = Prog::new();
    let t = p.t;
    let mut drivers = Vec::new();
    for ty in t.ints() {
        let it = int_ty(&p, ty);
        let vals = edges(it);
        let mut calls: Vec<(FuncId, Vec<i128>, TypeId)> = Vec::new();
        let ops: Vec<Op> = BIN.iter().chain(CMP.iter()).chain(SHIFT.iter()).copied().collect();
        for op in ops {
            let rt = if CMP.contains(&op) { t.bool } else { ty };
            let f = bin_fn(&mut p, op, ty, ty, rt, None, None);
            let amounts: Vec<i128> = if SHIFT.contains(&op) {
                let w = i128::from(it.bits);
                [0, 1, 3, w - 1, w, w + 1, 127, -1].into_iter().filter(|x| it.fits(*x)).collect()
            } else {
                vals.clone()
            };
            for &a in &vals {
                for &c in &amounts {
                    if semantics(op, it, a, c).is_ok() {
                        calls.push((f, vec![a, c], rt));
                    }
                }
            }
            // constant right operand (immediate forms), constant left operand
            for &c in [1i128, 3, 4095, 4096, -1, -4095, it.max(), it.min()].iter().filter(|c| it.fits(**c)) {
                let c = if SHIFT.contains(&op) { c.rem_euclid(i128::from(it.bits)) } else { c };
                let f = bin_fn(&mut p, op, ty, ty, rt, None, Some(c));
                for &a in &vals {
                    if semantics(op, it, a, c).is_ok() {
                        calls.push((f, vec![a], rt));
                    }
                }
                if !SHIFT.contains(&op) {
                    let g = bin_fn(&mut p, op, ty, ty, rt, Some(c), None);
                    for &a in &vals {
                        if semantics(op, it, c, a).is_ok() {
                            calls.push((g, vec![a], rt));
                        }
                    }
                }
            }
        }
        // shift amounts of another int type
        let at = if it.bits == 64 { t.u8 } else { t.i64 };
        for op in SHIFT {
            let f = bin_fn(&mut p, op, ty, at, ty, None, None);
            for &a in &vals {
                for c in [0i128, 1, i128::from(it.bits) - 1, i128::from(it.bits), 200, -3] {
                    let ok = int_ty(&p, at).fits(c) && semantics(op, it, a, c).is_ok();
                    if ok {
                        calls.push((f, vec![a, c], ty));
                    }
                }
            }
        }
        let d = p.func("t.drive", &[], None, |b, p| {
            for (f, args, rt) in &calls {
                let ops: Vec<Operand> = args.iter().map(|&a| b.int(a)).collect();
                let r = b.call(*f, Some(*rt), &ops).unwrap();
                p.out(b, r, *rt);
            }
            b.ret(None);
        });
        drivers.push(d);
    }
    p.set_main(|b, _| {
        for d in &drivers {
            b.call(*d, None, &[]);
        }
    });
    let r = check(&p, "intops");
    assert!(r.stdout.len() > 8 * 10_000, "{}", r.stdout.len());
}

/// Unary int ops (neg, not, popcount, clz, ctz) and mul.wide.
#[test]
fn int_unary_ops_and_mul_wide() {
    let mut p = Prog::new();
    let t = p.t;
    let mut calls: Vec<(FuncId, Vec<i128>, TypeId)> = Vec::new();
    for ty in t.ints() {
        let it = int_ty(&p, ty);
        for op in [Op::Neg, Op::Not, Op::Popcount, Op::Clz, Op::Ctz] {
            if op == Op::Neg && !it.signed {
                continue;
            }
            let f = p.func("t.un", &[ty], Some(ty), |b, _| {
                let x = b.param(0).unwrap();
                let r = b.unary(op, ty, x);
                b.ret(Some(r.into()));
            });
            for a in edges(it) {
                if op == Op::Neg && a == it.min() {
                    continue;
                }
                calls.push((f, vec![a], ty));
            }
        }
        let b2 = p.m.types.bool();
        let f = p.func("t.not_bool", &[b2], Some(b2), |b, _| {
            let x = b.param(0).unwrap();
            let r = b.unary(Op::Not, b2, x);
            b.ret(Some(r.into()));
        });
        calls.push((f, vec![0], b2));
        calls.push((f, vec![1], b2));
    }
    for ty in [t.u32, t.u64] {
        let it = int_ty(&p, ty);
        let pair = p.m.types.tuple(vec![ty, ty]);
        let f = p.func("t.wide", &[ty, ty], Some(ty), |b, _| {
            let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
            let r = b.binary(Op::MulWide, pair, x, y);
            let z = b.int(0);
            let hi = b.binary(Op::Get, ty, r, z);
            let one = b.int(1);
            let lo = b.binary(Op::Get, ty, r, one);
            let x = b.binary(Op::Xor, ty, hi, lo);
            let k = b.int(3);
            let x = b.binary(Op::ShlWrap, ty, x, k);
            let s = b.binary(Op::AddWrap, ty, x, hi);
            b.ret(Some(s.into()));
        });
        for a in edges(it) {
            for c in edges(it) {
                calls.push((f, vec![a, c], ty));
            }
        }
    }
    p.set_main(|b, p| {
        for (f, args, rt) in &calls {
            let ops: Vec<Operand> = args.iter().map(|&a| b.int(a)).collect();
            let r = b.call(*f, Some(*rt), &ops).unwrap();
            p.out(b, r, *rt);
        }
    });
    check(&p, "unops");
}

/// The first trapping input of each trapping op and int type, one program each; also checks the
/// reported kind and site.
#[test]
fn int_traps() {
    let mut n = 0;
    let base = Prog::new();
    let t = base.t;
    for ty in t.ints() {
        let it = int_ty(&base, ty);
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Rem, Op::Shl, Op::Shr, Op::Neg] {
            if op == Op::Neg && !it.signed {
                continue;
            }
            let vals = edges(it);
            let mut found = Vec::new();
            for &a in &vals {
                if op == Op::Neg {
                    if let Err(k) = arith::int_neg(it, it.wrap(a)) {
                        found.push((a, 0, k));
                    }
                    continue;
                }
                for c in vals.iter().copied().chain([i128::from(it.bits)]) {
                    if !it.fits(c) {
                        continue;
                    }
                    if let Err(k) = semantics(op, it, a, c) {
                        found.push((a, c, k));
                    }
                }
            }
            // a few per op: the first, and one of each kind
            let mut pick: Vec<(i128, i128, PanicKind)> = Vec::new();
            for f in found {
                if !pick.iter().any(|x| x.2 == f.2) {
                    pick.push(f);
                }
            }
            for (a, c, kind) in pick {
                let mut p = Prog::new();
                let f = if op == Op::Neg {
                    p.func("t.neg", &[ty], Some(ty), |b, _| {
                        let x = b.param(0).unwrap();
                        b.set_site(Site(900 + n));
                        let r = b.unary(Op::Neg, ty, x);
                        b.ret(Some(r.into()));
                    })
                } else {
                    p.func("t.op", &[ty, ty], Some(ty), |b, _| {
                        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
                        b.set_site(Site(900 + n));
                        let r = b.binary(op, ty, x, y);
                        b.ret(Some(r.into()));
                    })
                };
                p.set_main(|b, p| {
                    p.mark(b, 7);
                    let mut args = vec![b.int(a)];
                    if op != Op::Neg {
                        args.push(b.int(c));
                    }
                    let r = b.call(f, Some(ty), &args).unwrap();
                    p.out(b, r, ty);
                });
                let r = check(&p, "trap");
                assert_eq!(r.stderr, trap_bytes(kind, 900 + n), "{op:?} {a} {c}");
                n += 1;
            }
        }
    }
    assert!(n > 60, "{n} trap programs");
}

fn f64_edges() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -2.5,
        1.0 / 3.0,
        3.0,
        7.25,
        1e300,
        -1e300,
        f64::MAX,
        f64::MIN_POSITIVE,
        5e-324,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        9007199254740993.0,
        123456.789,
    ]
}

fn f32_edges() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -2.5,
        1.0 / 3.0,
        3.0,
        7.25,
        3e38,
        -3e38,
        f32::MAX,
        f32::MIN_POSITIVE,
        1e-45,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        16777217.0,
        1234.5678,
    ]
}

/// IEEE ops and compares on f32/f64 (register and constant operands), including `rem` (fmod).
#[test]
fn float_ops() {
    let mut p = Prog::new();
    let t = p.t;
    let mut fns = Vec::new();
    for (ty, is64) in [(t.f64, true), (t.f32, false)] {
        for op in
            [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Rem, Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge]
        {
            let rt = if CMP.contains(&op) { t.bool } else { ty };
            let f = p.func("t.f", &[ty, ty], Some(rt), |b, _| {
                let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
                let r = b.binary(op, rt, x, y);
                b.ret(Some(r.into()));
            });
            let g = p.func("t.fc", &[ty], Some(rt), |b, _| {
                let x = b.param(0).unwrap();
                let c = if is64 { b.f64(1.5) } else { b.f32(1.5) };
                let r = b.binary(op, rt, x, c);
                b.ret(Some(r.into()));
            });
            fns.push((f, g, ty, rt, is64));
        }
        let f = p.func("t.neg", &[ty], Some(ty), |b, _| {
            let x = b.param(0).unwrap();
            let r = b.unary(Op::Neg, ty, x);
            b.ret(Some(r.into()));
        });
        fns.push((f, f, ty, ty, is64));
    }
    p.set_main(|b, p| {
        for &(f, g, ty, rt, is64) in &fns {
            let unary = f == g;
            let xs: Vec<Operand> = if is64 {
                f64_edges().into_iter().map(|x| b.f64(x)).collect()
            } else {
                f32_edges().into_iter().map(|x| b.f32(x)).collect()
            };
            for &x in &xs {
                if unary {
                    let r = b.call(f, Some(ty), &[x]).unwrap();
                    p.out(b, r, ty);
                    continue;
                }
                for &y in &xs {
                    let r = b.call(f, Some(rt), &[x, y]).unwrap();
                    p.out(b, r, rt);
                }
                let r = b.call(g, Some(rt), &[x]).unwrap();
                p.out(b, r, rt);
            }
        }
    });
    check(&p, "floats");
}

fn scalar_tys(p: &Prog) -> Vec<TypeId> {
    let t = p.t;
    let mut v = t.ints().to_vec();
    v.extend([t.f32, t.f64, t.bool, t.char]);
    v
}

fn cast_ty(p: &Prog, ty: TypeId) -> CastTy {
    let t = p.t;
    if ty == t.f32 {
        CastTy::Float(FloatBits::F32)
    } else if ty == t.f64 {
        CastTy::Float(FloatBits::F64)
    } else if ty == t.bool {
        CastTy::Bool
    } else if ty == t.char {
        CastTy::Char
    } else {
        CastTy::Int(int_ty(p, ty))
    }
}

#[derive(Clone, Copy)]
enum In {
    I(i128),
    F32(f32),
    F64(f64),
}

fn operand(b: &mut FuncBuilder, x: In) -> Operand {
    match x {
        In::I(v) => b.int(v),
        In::F32(v) => b.f32(v),
        In::F64(v) => b.f64(v),
    }
}

/// Edge inputs of a scalar type.
fn inputs(p: &Prog, ty: TypeId) -> Vec<In> {
    let t = p.t;
    if ty == t.f64 {
        let mut v = f64_edges();
        v.extend([
            2147483647.5,
            2147483648.0,
            -2147483649.0,
            4294967296.0,
            255.5,
            256.0,
            -129.0,
            -128.9,
            1e10,
            9.223_372_036_854_776e18,
            -9.223_372_036_854_776e18,
            1.8446744073709552e19,
            65535.99,
            -0.75,
        ]);
        v.into_iter().map(In::F64).collect()
    } else if ty == t.f32 {
        let mut v = f32_edges();
        v.extend([
            2147483648.0,
            -2147483904.0,
            4294967296.0,
            255.5,
            256.0,
            -129.0,
            9.223372e18,
            1.8446744e19,
            -0.75,
        ]);
        v.into_iter().map(In::F32).collect()
    } else if ty == t.bool {
        vec![In::I(0), In::I(1)]
    } else if ty == t.char {
        [0i128, 97, 255, 0xd7ff, 0xe000, 0x10ffff].into_iter().map(In::I).collect()
    } else {
        let it = int_ty(p, ty);
        let mut v = edges(it);
        v.extend([
            0xd800,
            0xdfff,
            0x10ffff,
            0x110000,
            255,
            256,
            65535,
            1 << 24,
            (1 << 24) + 1,
            (1 << 53) + 1,
        ]);
        v.retain(|x| it.fits(*x));
        v.sort();
        v.dedup();
        v.into_iter().map(In::I).collect()
    }
}

/// Whether the verifier accepts `kind` on the constant `x` (int constants are typed as a 64-bit
/// int by their sign; `as` needs the exact conversion).
fn const_cast_ok(kind: CastKind, x: In, from: CastTy, to: CastTy) -> bool {
    match x {
        In::I(v) => {
            let src = CastTy::Int(IntTy::new(64, v < 0));
            if kind == CastKind::Exact {
                arith::convert(CastKind::Try, src, to, v as u64).is_some()
            } else {
                arith::cast_allowed(kind, src, to)
            }
        }
        _ => arith::cast_allowed(kind, from, to),
    }
}

/// Every allowed `(kind, from, to)` cast over edge inputs, value and constant operands.
#[test]
fn casts() {
    let mut p = Prog::new();
    let tys = scalar_tys(&p);
    let mut fns: Vec<(FuncId, TypeId)> = Vec::new();
    let mut consts_fns: Vec<FuncId> = Vec::new();
    let mut n = 0u32;
    for &from in &tys {
        for &to in &tys {
            for kind in [CastKind::Exact, CastKind::Wrap, CastKind::Sat, CastKind::Round, CastKind::Try] {
                if !arith::cast_allowed(kind, cast_ty(&p, from), cast_ty(&p, to)) {
                    continue;
                }
                // The interpreter has no `bool as bool` (arith::convert yields None for a bool
                // target); the backend compiles it as a move.
                if to == p.t.bool {
                    continue;
                }
                let opt = if kind == CastKind::Try {
                    Some(p.m.types.adt(
                        stave_ir::DeclId(10_000 + n),
                        vec![],
                        stave_ir::AdtShape::enumeration(vec![vec![], vec![to]]),
                    ))
                } else {
                    None
                };
                n += 1;
                let f = p.func("t.cast", &[from], None, |b, p| {
                    let x = b.param(0).unwrap();
                    match opt {
                        None => {
                            let r = b.cast(kind, to, x);
                            p.out(b, r, to);
                        }
                        Some(o) => {
                            let r = b.cast(kind, o, x);
                            let tag = b.unary(Op::EnumTag, p.t.u32, r);
                            p.out(b, tag, p.t.u32);
                            let (some, join) = (b.create_block(), b.create_block());
                            b.switch(tag, &[(1, some, &[])], join, &[]);
                            b.switch_to(some);
                            let (one, z) = (b.int(1), b.int(0));
                            let v = b.op(Op::EnumGet, to, &[r.into(), one, z]);
                            p.out(b, v, to);
                            b.br(join, &[]);
                            b.switch_to(join);
                        }
                    }
                    b.ret(None);
                });
                fns.push((f, from));
                // constant-operand form (folded at compile time)
                let (fc, tc) = (cast_ty(&p, from), cast_ty(&p, to));
                let consts: Vec<In> =
                    inputs(&p, from).into_iter().filter(|&x| const_cast_ok(kind, x, fc, tc)).collect();
                let g = p.func("t.castc", &[], None, |b, p| {
                    for &x in &consts {
                        let x = operand(b, x);
                        if let Some(o) = opt {
                            let r = b.cast(kind, o, x);
                            let tag = b.unary(Op::EnumTag, p.t.u32, r);
                            p.out(b, tag, p.t.u32);
                        } else {
                            let r = b.cast(kind, to, x);
                            p.out(b, r, to);
                        }
                    }
                    b.ret(None);
                });
                consts_fns.push(g);
            }
        }
    }
    p.set_main(|b, p| {
        for &(f, from) in &fns {
            for x in inputs(p, from) {
                let x = operand(b, x);
                b.call(f, None, &[x]);
            }
        }
        for &g in &consts_fns {
            b.call(g, None, &[]);
        }
    });
    let r = check(&p, "casts");
    assert!(r.stdout.len() > 8 * 1000);
}
