//! Hand-written differential programs: every op of the table on edge values, traps of every
//! kind, casts, aggregates, enums, switch, calls (many args, aggregates by value, sret),
//! recursion, loops, memory ops, string literals, and intrinsics.

use stave_ir::arith::{self, CastTy, IntTy};
use stave_ir::{CastKind, FloatBits, FuncBuilder, Op, Operand, PanicKind, TypeId, TypeKey};

use super::harness::*;

fn ity(p: &Prog, t: TypeId) -> IntTy {
    match p.m.types.key(t) {
        Some(TypeKey::Int { signed, bits }) => IntTy::new(bits.width(), *signed),
        _ => panic!("not an int type"),
    }
}

fn edges(t: IntTy) -> Vec<i128> {
    let (lo, hi) = (t.min(), t.max());
    let mut v = vec![0, 1, 2, 3, 5, 7, -1, -2, -3, lo, lo + 1, hi, hi - 1, hi / 2, lo / 2, 100, -100];
    v.extend([
        i128::from(t.bits) - 1,
        i128::from(t.bits),
        0x5555_5555_5555_5555 & hi,
        0x1234_5678_9abc_def0 & hi,
    ]);
    v.extend([(hi as u128 / 3) as i128, 12345, -12345, 255, 256, 65535, 65536]);
    let mut v: Vec<i128> = v.into_iter().filter(|x| *x >= lo && *x <= hi).collect();
    v.sort();
    v.dedup();
    v
}

const BIN_NOTRAP: [Op; 10] = [
    Op::AddWrap,
    Op::SubWrap,
    Op::MulWrap,
    Op::AddSat,
    Op::SubSat,
    Op::MulSat,
    Op::And,
    Op::Or,
    Op::Xor,
    Op::ShlWrap,
];
const BIN_TRAP: [Op; 7] = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Rem, Op::Shl, Op::Shr];
const CMPS: [Op; 6] = [Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge];

/// The interpreter's verdict for one int op (None = traps).
fn oracle(op: Op, t: IntTy, a: i128, b: i128) -> Option<u64> {
    let (x, y) = (t.wrap(a), t.wrap(b));
    match op {
        Op::Shl | Op::Shr | Op::ShlWrap => arith::int_shift(op, t, x, b).ok(),
        _ => arith::int_binary(op, t, x, y).ok(),
    }
}

#[test]
fn hello() {
    let mut p = Prog::new();
    p.main(|p, b| {
        p.print_str(b, "hello, Stave\n");
        let c = b.int(7);
        b.ret(Some(c));
    });
    let r = check(&p, "hello");
    assert_eq!(r.stdout, b"hello, Stave\n");
    assert_eq!(r.code, 7);
}

/// Every int op on edge-value pairs, with both operands values, a constant right operand, and a
/// constant left operand.
#[test]
fn int_ops_edges() {
    let base = Prog::new();
    for ti in 0..10 {
        let mut p = Prog::new();
        let t = int_types(&base.t)[ti];
        let it = ity(&p, t);
        let vals = edges(it);
        let tb = p.t.bool;
        p.main(|p, b| {
            for &a in &vals {
                let ca = b.int(a);
                let x = p.opaque(b, ca, t);
                for &bv in &vals {
                    let cb = b.int(bv);
                    let y = p.opaque(b, cb, t);
                    for op in BIN_NOTRAP.iter().chain(&BIN_TRAP).copied() {
                        if oracle(op, it, a, bv).is_none() {
                            continue;
                        }
                        let r = b.binary(op, t, x, y);
                        p.print(b, r.into(), t);
                        let kb = b.int(bv);
                        let r = b.binary(op, t, x, kb);
                        p.print(b, r.into(), t);
                        if !matches!(op, Op::Shl | Op::Shr | Op::ShlWrap) {
                            let ka = b.int(a);
                            let r = b.binary(op, t, ka, y);
                            p.print(b, r.into(), t);
                        }
                    }
                    for op in CMPS {
                        let r = b.cmp(op, tb, x, y);
                        p.print(b, r.into(), tb);
                        let kb = b.int(bv);
                        let r = b.cmp(op, tb, x, kb);
                        p.print(b, r.into(), tb);
                        let ka = b.int(a);
                        let r = b.cmp(op, tb, ka, y);
                        p.print(b, r.into(), tb);
                    }
                }
                for op in [Op::Not, Op::Popcount, Op::Clz, Op::Ctz] {
                    let r = b.unary(op, t, x);
                    p.print(b, r.into(), t);
                }
                if it.signed && a != it.min() {
                    let r = b.unary(Op::Neg, t, x);
                    p.print(b, r.into(), t);
                }
                // shifts by an amount of another int type
                for (amt_t, amt) in [(p.t.u8, 3i128), (p.t.i64, 1), (p.t.u32, i128::from(it.bits) - 1)] {
                    let k = b.int(amt);
                    let n = p.opaque(b, k, amt_t);
                    for op in [Op::Shl, Op::Shr, Op::ShlWrap] {
                        let r = b.binary(op, t, x, n);
                        p.print(b, r.into(), t);
                    }
                }
            }
        });
        check(&p, &format!("int_ops_t{ti}"));
    }
}

/// One program per (type, trapping op): prints a marker, then traps with the right kind.
#[test]
fn int_traps() {
    let base = Prog::new();
    for ti in 0..10 {
        let t0 = int_types(&base.t)[ti];
        let it = ity(&base, t0);
        let vals = edges(it);
        for op in BIN_TRAP {
            let Some((a, bv)) = vals
                .iter()
                .flat_map(|&a| vals.iter().map(move |&b| (a, b)))
                .find(|&(a, b)| oracle(op, it, a, b).is_none())
            else {
                continue;
            };
            for konst in [false, true] {
                let mut p = Prog::new();
                let t = int_types(&p.t)[ti];
                p.main(|p, b| {
                    p.print_str(b, "before\n");
                    let ca = b.int(a);
                    let x = p.opaque(b, ca, t);
                    let y: Operand = if konst {
                        b.int(bv)
                    } else {
                        let cb = b.int(bv);
                        p.opaque(b, cb, t).into()
                    };
                    let r = b.binary(op, t, x, y);
                    p.print(b, r.into(), t);
                });
                let r = check(&p, &format!("trap_t{ti}_{op:?}_{konst}"));
                let want = match op {
                    Op::Div | Op::Rem if bv == 0 => PanicKind::DivByZero,
                    _ => PanicKind::Overflow,
                };
                assert_eq!(r.code, 100 + kind_index(want), "t{ti} {op:?} {a} {bv}");
                assert_eq!(r.stdout, b"before\n");
            }
        }
        if it.signed {
            let mut p = Prog::new();
            let t = int_types(&p.t)[ti];
            p.main(|p, b| {
                let ca = b.int(it.min());
                let x = p.opaque(b, ca, t);
                let r = b.unary(Op::Neg, t, x);
                p.print(b, r.into(), t);
            });
            assert_eq!(check(&p, &format!("trap_neg_t{ti}")).code, 101);
        }
    }
}

#[test]
fn mul_wide() {
    let mut p = Prog::new();
    let (u32_, u64_) = (p.t.u32, p.t.u64);
    let p32 = p.m.types.tuple(vec![u32_, u32_]);
    let p64 = p.m.types.tuple(vec![u64_, u64_]);
    p.main(|p, b| {
        for (t, pt, vals) in [
            (u32_, p32, vec![0u64, 1, 0xffff_ffff, 0x8000_0000, 12345, 0xdead_beef]),
            (u64_, p64, vec![0, 1, u64::MAX, 1 << 63, 0xdead_beef_cafe_f00d, 3]),
        ] {
            for &x in &vals {
                for &y in &vals {
                    let cx = b.int(i128::from(x));
                    let a = p.opaque(b, cx, t);
                    let cy = b.int(i128::from(y));
                    let c = p.opaque(b, cy, t);
                    let r = b.binary(Op::MulWide, pt, a, c);
                    for k in 0..2 {
                        let kk = b.int(k);
                        let h = b.op(Op::Get, t, &[r.into(), kk]);
                        p.print(b, h.into(), t);
                    }
                }
            }
        }
    });
    check(&p, "mul_wide");
}

fn f64_vals() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.5,
        2.5,
        3.7,
        -7.25,
        0.1,
        1e300,
        -1e300,
        1e-310,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        2f64.powi(63),
        -(2f64.powi(63)),
        2f64.powi(63) - 1024.0,
        2f64.powi(64) - 2048.0,
        1.8446744073709552e19,
        4294967296.5,
        -2147483648.5,
        127.9,
        -128.9,
        65535.99,
        1e20,
    ]
}

#[test]
fn float_ops() {
    let mut p = Prog::new();
    let (tf32, tf64, tb) = (p.t.f32, p.t.f64, p.t.bool);
    p.main(|p, b| {
        for (t, is32) in [(tf64, false), (tf32, true)] {
            let vals = f64_vals();
            let k = |b: &mut FuncBuilder, x: f64| if is32 { b.f32(x as f32) } else { b.f64(x) };
            for &x in &vals {
                let cx = k(b, x);
                let a = p.opaque(b, cx, t);
                let r = b.unary(Op::Neg, t, a);
                p.print(b, r.into(), t);
                for &y in &vals {
                    let cy = k(b, y);
                    let c = p.opaque(b, cy, t);
                    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Rem] {
                        let r = b.binary(op, t, a, c);
                        p.print(b, r.into(), t);
                    }
                    let ky = k(b, y);
                    let r = b.binary(Op::Sub, t, a, ky);
                    p.print(b, r.into(), t);
                    for op in CMPS {
                        let r = b.cmp(op, tb, a, c);
                        p.print(b, r.into(), tb);
                        let ky = k(b, y);
                        let r = b.cmp(op, tb, a, ky);
                        p.print(b, r.into(), tb);
                    }
                    let cnd = b.cmp(Op::Lt, tb, a, c);
                    let r = b.select(t, cnd, a, c);
                    p.print(b, r.into(), t);
                }
            }
        }
    });
    check(&p, "float_ops");
}

fn scalar_types(p: &Prog) -> Vec<TypeId> {
    let t = p.t;
    let mut v = int_types(&t).to_vec();
    v.extend([t.bool, t.char, t.f32, t.f64]);
    v
}

fn cty(p: &Prog, t: TypeId) -> CastTy {
    match p.m.types.key(t) {
        Some(TypeKey::Int { .. }) => CastTy::Int(ity(p, t)),
        Some(TypeKey::Bool) => CastTy::Bool,
        Some(TypeKey::Char) => CastTy::Char,
        Some(TypeKey::Float(f)) => CastTy::Float(*f),
        _ => CastTy::Ptr,
    }
}

/// Every allowed (kind, from, to) over scalar types on edge values, from values and constants.
#[test]
fn casts() {
    let base = Prog::new();
    let tys = scalar_types(&base);
    for (fi, &from) in tys.iter().enumerate() {
        let mut p = Prog::new();
        let opts: Vec<TypeId> = tys.iter().map(|&t| p.option(t)).collect();
        let tu32 = p.t.u32;
        let fc = cty(&p, from);
        p.main(|p, b| {
            let srcs: Vec<(Operand, Option<i128>)> = match fc {
                CastTy::Int(it) => edges(it).into_iter().map(|x| (b.int(x), Some(x))).collect(),
                CastTy::Bool => vec![(b.int(0), Some(0)), (b.int(1), Some(1))],
                CastTy::Char => [0i128, 65, 0x7f, 0xff, 0x100, 0xd7ff, 0xe000, 0x10ffff]
                    .map(|x| (b.int(x), Some(x)))
                    .to_vec(),
                CastTy::Float(FloatBits::F32) => {
                    f64_vals().into_iter().map(|x| (b.f32(x as f32), None)).collect()
                }
                _ => f64_vals().into_iter().map(|x| (b.f64(x), None)).collect(),
            };
            for &(c, cv) in &srcs {
                let x = p.opaque(b, c, from);
                for (ti, &to) in tys.iter().enumerate() {
                    let tc = cty(p, to);
                    for kind in
                        [CastKind::Exact, CastKind::Wrap, CastKind::Sat, CastKind::Round, CastKind::Try]
                    {
                        // (bool → bool has no defined conversion in the interpreter)
                        if !arith::cast_allowed(kind, fc, tc) || tc == CastTy::Bool {
                            continue;
                        }
                        for src in [x.into(), c] {
                            if let (Operand::Const(_), Some(v)) = (src, cv) {
                                // the verifier's rule for constant operands
                                let src64 = CastTy::Int(IntTy::new(64, v < 0));
                                let ok = if kind == CastKind::Exact {
                                    arith::convert(CastKind::Try, src64, tc, v as u64).is_some()
                                } else {
                                    arith::cast_allowed(kind, src64, tc)
                                };
                                if !ok {
                                    continue;
                                }
                            }
                            if kind == CastKind::Try {
                                let ot = opts[ti];
                                let r = b.cast(kind, ot, src);
                                let tag = b.op(Op::EnumTag, tu32, &[r.into()]);
                                p.print(b, tag.into(), tu32);
                                let some = b.create_block();
                                let join = b.create_block();
                                let one = b.int(1);
                                let tb = p.t.bool;
                                let is = b.cmp(Op::Eq, tb, tag, one);
                                b.cbr(is, some, &[], join, &[]);
                                b.switch_to(some);
                                let (v1, f0) = (b.int(1), b.int(0));
                                let pl = b.op(Op::EnumGet, to, &[r.into(), v1, f0]);
                                p.print(b, pl.into(), to);
                                b.br(join, &[]);
                                b.switch_to(join);
                            } else {
                                let r = b.cast(kind, to, src);
                                p.print(b, r.into(), to);
                            }
                        }
                    }
                }
            }
        });
        check(&p, &format!("casts_from_{fi}"));
    }
}

/// Prints every field of a record value.
fn print_fields(p: &mut Prog, b: &mut FuncBuilder, v: Operand, fields: &[TypeId]) {
    for (k, &ft) in fields.iter().enumerate() {
        let kk = b.int(k as i128);
        let f = b.op(Op::Get, ft, &[v, kk]);
        p.print(b, f.into(), ft);
    }
}

#[test]
fn aggregates_and_calls() {
    let mut p = Prog::new();
    let t = p.t;
    // struct S { u8, i64, f32, u16 }: 24 bytes → passed in memory
    let sf = vec![t.u8, t.i64, t.f32, t.u16];
    let s = p.structure(sf.clone());
    // small mixed tuples: (f64, i32) = SSE + INTEGER, (f32, f32, u64) = SSE + INTEGER
    let tf = vec![t.f64, t.i32];
    let tup = p.m.types.tuple(tf.clone());
    let tf2 = vec![t.f32, t.f32, t.u64];
    let tup2 = p.m.types.tuple(tf2.clone());
    let big_f = vec![t.u64; 6];
    let big = p.m.types.tuple(big_f.clone());
    let arr = p.m.types.array(t.u32, 5);

    // fn swap(x: (f64, i32)) -> (i32, f64) style: returns (f64 * 2, i32 + 1)
    let ftup = p.decl("t.tup", &[tup, tup2], Some(tup));
    {
        let mut b = FuncBuilder::new(&[tup, tup2]);
        let x = b.param(0).unwrap();
        let y = b.param(1).unwrap();
        let (k0, k1, k2) = (b.int(0), b.int(1), b.int(2));
        let a = b.op(Op::Get, t.f64, &[x.into(), k0]);
        let c = b.op(Op::Get, t.i32, &[x.into(), k1]);
        let y0 = b.op(Op::Get, t.f32, &[y.into(), k0]);
        let y2 = b.op(Op::Get, t.u64, &[y.into(), k2]);
        let y0d = b.cast(CastKind::Exact, t.f64, y0);
        let a2 = b.binary(Op::Add, t.f64, a, y0d);
        let y2i = b.cast(CastKind::Wrap, t.i32, y2);
        let c2 = b.binary(Op::AddWrap, t.i32, c, y2i);
        let r = b.op(Op::Make, tup, &[a2.into(), c2.into()]);
        b.ret(Some(r.into()));
        p.def(ftup, b);
    }
    // fn big(s: S, b: big, k: u64) -> big: sret, memory args
    let fbig = p.decl("t.big", &[s, big, t.u64], Some(big));
    {
        let mut b = FuncBuilder::new(&[s, big, t.u64]);
        let sv = b.param(0).unwrap();
        let bv = b.param(1).unwrap();
        let k = b.param(2).unwrap();
        let k1 = b.int(1);
        let sf1 = b.op(Op::Get, t.i64, &[sv.into(), k1]);
        let sf1u = b.cast(CastKind::Wrap, t.u64, sf1);
        let mut out = bv;
        for i in 0..6 {
            let ki = b.int(i);
            let e = b.op(Op::Get, t.u64, &[bv.into(), ki]);
            let e2 = b.binary(Op::MulWrap, t.u64, e, k);
            let e3 = b.binary(Op::AddWrap, t.u64, e2, sf1u);
            let ki2 = b.int(i);
            out = b.op(Op::Set, big, &[out.into(), ki2, e3.into()]);
        }
        b.ret(Some(out.into()));
        p.def(fbig, b);
    }
    // fn many(i1..i8: i64, f1..f10: f64, t: (f64,i32)) -> f64
    let mut ps = vec![t.i64; 8];
    ps.extend(vec![t.f64; 10]);
    ps.push(tup);
    let fmany = p.decl("t.many", &ps, Some(t.f64));
    {
        let mut b = FuncBuilder::new(&ps);
        let mut acc_i: Operand = b.int(0);
        for i in 0..8 {
            let x = b.param(i).unwrap();
            let k = b.int(i as i128 + 1);
            let m = b.binary(Op::MulWrap, t.i64, x, k);
            acc_i = b.binary(Op::AddWrap, t.i64, acc_i, m).into();
        }
        let mut acc_f: Operand = b.f64(0.0);
        for i in 8..18 {
            let x = b.param(i).unwrap();
            let k = b.f64(i as f64);
            let m = b.binary(Op::Mul, t.f64, x, k);
            acc_f = b.binary(Op::Add, t.f64, acc_f, m).into();
        }
        let tp = b.param(18).unwrap();
        let k0 = b.int(0);
        let t0 = b.op(Op::Get, t.f64, &[tp.into(), k0]);
        let ai = b.cast(CastKind::Round, t.f64, acc_i);
        let s1 = b.binary(Op::Add, t.f64, acc_f, ai);
        let s2 = b.binary(Op::Add, t.f64, s1, t0);
        b.ret(Some(s2.into()));
        p.def(fmany, b);
    }
    p.main(|p, b| {
        let (c1, c2, c3, c4) = (b.int(200), b.int(-5_000_000_000), b.f32(1.25), b.int(65000));
        let sv = b.op(Op::Make, s, &[c1, c2, c3, c4]);
        print_fields(p, b, sv.into(), &sf);
        let (a, c) = (b.f64(3.5), b.int(-7));
        let x = b.op(Op::Make, tup, &[a, c]);
        let (y0, y1, y2) = (b.f32(0.5), b.f32(9.0), b.int(41));
        let y = b.op(Op::Make, tup2, &[y0, y1, y2]);
        let rtup = b.call(ftup, Some(tup), &[x.into(), y.into()]).unwrap();
        print_fields(p, b, rtup.into(), &tf);
        let elems: Vec<Operand> = (0..6).map(|i| b.int(i * 1000 + 1)).collect();
        let bigv = b.op(Op::Make, big, &elems);
        let k = b.int(3);
        let r = b.call(fbig, Some(big), &[sv.into(), bigv.into(), k]).unwrap();
        print_fields(p, b, r.into(), &big_f);
        let mut args: Vec<Operand> = (0..8).map(|i| b.int(i * 7 - 20)).collect();
        args.extend((0..10).map(|i| b.f64(i as f64 * 0.75 - 2.0)));
        args.push(x.into());
        let r = b.call(fmany, Some(t.f64), &args).unwrap();
        p.print(b, r.into(), t.f64);
        // arrays: make, dynamic get, dynamic set, const get
        let es: Vec<Operand> = (0..5).map(|i| b.int(i * i + 10)).collect();
        let av = b.op(Op::Make, arr, &es);
        let mut cur = av;
        for i in 0..5 {
            let ci = b.int(4 - i);
            let idx = p.opaque(b, ci, t.usize);
            let e = b.op(Op::Get, t.u32, &[cur.into(), idx.into()]);
            p.print(b, e.into(), t.u32);
            let ci2 = b.int(i);
            let idx2 = p.opaque(b, ci2, t.u64);
            let nv = b.binary(Op::MulWrap, t.u32, e, e);
            cur = b.op(Op::Set, arr, &[cur.into(), idx2.into(), nv.into()]);
        }
        for i in 0..5 {
            let ci = b.int(i);
            let e = b.op(Op::Get, t.u32, &[cur.into(), ci]);
            p.print(b, e.into(), t.u32);
            let e = b.op(Op::Get, t.u32, &[av.into(), ci]);
            p.print(b, e.into(), t.u32);
        }
        // select of aggregates
        let one = b.int(1);
        let k = p.opaque(b, one, t.bool);
        let sel = b.select(tup, k, rtup, x);
        print_fields(p, b, sel.into(), &tf);
        let zero = b.int(0);
        let k = p.opaque(b, zero, t.bool);
        let sel = b.select(tup, k, rtup, x);
        print_fields(p, b, sel.into(), &tf);
    });
    check(&p, "aggregates_and_calls");
}

#[test]
fn enums_and_switch() {
    let mut p = Prog::new();
    let t = p.t;
    let pair = p.m.types.tuple(vec![t.u32, t.u32]);
    let e = p.enumeration(vec![vec![], vec![t.u8, t.u64], vec![t.f64], vec![pair]]);
    // fn describe(e: E) -> u64 { switch on tag }
    let fdesc = p.decl("t.describe", &[e], Some(t.u64));
    {
        let mut b = FuncBuilder::new(&[e]);
        let ev = b.param(0).unwrap();
        let tag = b.op(Op::EnumTag, t.u32, &[ev.into()]);
        let (ba, bb, bc, bd) = (b.create_block(), b.create_block(), b.create_block(), b.create_block());
        b.switch(tag, &[(0, ba, &[]), (1, bb, &[]), (2, bc, &[])], bd, &[]);
        b.switch_to(ba);
        let r = b.int(1000);
        b.ret(Some(r));
        b.switch_to(bb);
        let (v, f0, f1) = (b.int(1), b.int(0), b.int(1));
        let x = b.op(Op::EnumGet, t.u8, &[ev.into(), v, f0]);
        let y = b.op(Op::EnumGet, t.u64, &[ev.into(), v, f1]);
        let x64 = b.cast(CastKind::Exact, t.u64, x);
        let r = b.binary(Op::AddWrap, t.u64, x64, y);
        b.ret(Some(r.into()));
        b.switch_to(bc);
        let (v, f0) = (b.int(2), b.int(0));
        let x = b.op(Op::EnumGet, t.f64, &[ev.into(), v, f0]);
        let r = b.cast(CastKind::Sat, t.u64, x);
        b.ret(Some(r.into()));
        b.switch_to(bd);
        let (v, f0) = (b.int(3), b.int(0));
        let x = b.op(Op::EnumGet, pair, &[ev.into(), v, f0]);
        let (k0, k1) = (b.int(0), b.int(1));
        let a = b.op(Op::Get, t.u32, &[x.into(), k0]);
        let c = b.op(Op::Get, t.u32, &[x.into(), k1]);
        let r = b.binary(Op::MulWrap, t.u32, a, c);
        let r = b.cast(CastKind::Exact, t.u64, r);
        b.ret(Some(r.into()));
        p.def(fdesc, b);
    }
    // fn sw(x: i32) -> i32: many cases (binary search), negative values, args on edges
    let fsw = p.decl("t.sw", &[t.i32], Some(t.i32));
    {
        let mut b = FuncBuilder::new(&[t.i32]);
        let x = b.param(0).unwrap();
        let (join, jv) = b.create_block_with(&[t.i32]);
        let vals: Vec<i128> = vec![-100, -3, 0, 1, 2, 5, 17, 99, 1000, 2_000_000_000, -2_000_000_000];
        let blocks: Vec<_> = vals.iter().map(|_| b.create_block()).collect();
        let dflt = b.create_block();
        let consts: Vec<Operand> = vals.iter().map(|v| b.int(v / 2 + 1)).collect();
        let cases: Vec<(i128, stave_ir::Block, Vec<Operand>)> = vals
            .iter()
            .zip(&blocks)
            .enumerate()
            .map(|(k, (v, bl))| (*v, *bl, if k % 2 == 0 { vec![] } else { vec![consts[k]] }))
            .collect();
        // odd cases jump straight to the join with an argument; even ones go through a block
        let mut sw_cases: Vec<(i128, stave_ir::Block, &[Operand])> = Vec::new();
        for (v, bl, args) in &cases {
            if args.is_empty() {
                sw_cases.push((*v, *bl, &[]));
            } else {
                sw_cases.push((*v, join, args.as_slice()));
            }
        }
        let m1 = b.int(-1);
        b.switch(x, &sw_cases, dflt, &[]);
        for (k, bl) in blocks.iter().enumerate() {
            b.switch_to(*bl);
            let c = b.int(k as i128 * 10);
            b.br(join, &[c]);
        }
        b.switch_to(dflt);
        b.br(join, &[m1]);
        b.switch_to(join);
        b.ret(Some(jv[0].into()));
        p.def(fsw, b);
    }
    p.main(|p, b| {
        let a = b.int(0);
        let ea = b.op(Op::EnumMake, e, &[a]);
        let (v, x, y) = (b.int(1), b.int(250), b.int(1 << 40));
        let eb = b.op(Op::EnumMake, e, &[v, x, y]);
        let (v, x) = (b.int(2), b.f64(12345.75));
        let ec = b.op(Op::EnumMake, e, &[v, x]);
        let (k3, k4) = (b.int(3), b.int(4));
        let pv = b.op(Op::Make, pair, &[k3, k4]);
        let v = b.int(3);
        let ed = b.op(Op::EnumMake, e, &[v, pv.into()]);
        for ev in [ea, eb, ec, ed] {
            let tag = b.op(Op::EnumTag, t.u8, &[ev.into()]);
            p.print(b, tag.into(), t.u8);
            let r = b.call(fdesc, Some(t.u64), &[ev.into()]).unwrap();
            p.print(b, r.into(), t.u64);
        }
        for x in [-100i128, -3, 0, 1, 2, 3, 5, 17, 99, 1000, 2_000_000_000, -2_000_000_000, 42, -42] {
            let c = b.int(x);
            let r = b.call(fsw, Some(t.i32), &[c]).unwrap();
            p.print(b, r.into(), t.i32);
            let o = p.opaque(b, c, t.i32);
            let r = b.call(fsw, Some(t.i32), &[o.into()]).unwrap();
            p.print(b, r.into(), t.i32);
        }
        // switch on char and bool
        for (ty, vals) in [(t.char, vec![97i128, 0x10ffff, 0]), (t.bool, vec![0, 1])] {
            for x in vals {
                let c = b.int(x);
                let o = p.opaque(b, c, ty);
                let (b1, b2, j) = (b.create_block(), b.create_block(), b.create_block());
                let cv = if ty == t.bool { 1 } else { 97 };
                b.switch(o, &[(cv, b1, &[])], b2, &[]);
                b.switch_to(b1);
                p.print_str(b, "hit\n");
                b.br(j, &[]);
                b.switch_to(b2);
                p.print_str(b, "miss\n");
                b.br(j, &[]);
                b.switch_to(j);
            }
        }
    });
    check(&p, "enums_and_switch");
}

#[test]
fn control_flow() {
    let mut p = Prog::new();
    let t = p.t;
    // recursive fib
    let fib = p.decl("t.fib", &[t.u64], Some(t.u64));
    {
        let mut b = FuncBuilder::new(&[t.u64]);
        let n = b.param(0).unwrap();
        let two = b.int(2);
        let small = b.cmp(Op::Lt, t.bool, n, two);
        let (base, rec) = (b.create_block(), b.create_block());
        b.cbr(small, base, &[], rec, &[]);
        b.switch_to(base);
        b.ret(Some(n.into()));
        b.switch_to(rec);
        let one = b.int(1);
        let n1 = b.binary(Op::Sub, t.u64, n, one);
        let two = b.int(2);
        let n2 = b.binary(Op::Sub, t.u64, n, two);
        let a = b.call(fib, Some(t.u64), &[n1.into()]).unwrap();
        let c = b.call(fib, Some(t.u64), &[n2.into()]).unwrap();
        let r = b.binary(Op::Add, t.u64, a, c);
        b.ret(Some(r.into()));
        p.def(fib, b);
    }
    // iterative fib with swapping block params, plus a rotation of three params
    let itfib = p.decl("t.itfib", &[t.u64], Some(t.u64));
    {
        let mut b = FuncBuilder::new(&[t.u64]);
        let n = b.param(0).unwrap();
        let (head, hp) = b.create_block_with(&[t.u64, t.u64, t.u64, t.u64, t.u64, t.u64]);
        let (body, exit) = (b.create_block(), b.create_block());
        let (z, o, z2, c1, c2, c3) = (b.int(0), b.int(1), b.int(0), b.int(1), b.int(2), b.int(3));
        b.br(head, &[z, o, z2, c1, c2, c3]);
        b.switch_to(head);
        let (a, c, i, x, y, w) = (hp[0], hp[1], hp[2], hp[3], hp[4], hp[5]);
        let lt = b.cmp(Op::Lt, t.bool, i, n);
        b.cbr(lt, body, &[], exit, &[]);
        b.switch_to(body);
        let s = b.binary(Op::AddWrap, t.u64, a, c);
        let one = b.int(1);
        let i2 = b.binary(Op::Add, t.u64, i, one);
        b.br(head, &[c.into(), s.into(), i2.into(), w.into(), x.into(), y.into()]);
        b.switch_to(exit);
        let k = b.int(1000);
        let xs = b.binary(Op::MulWrap, t.u64, x, k);
        let r = b.binary(Op::AddWrap, t.u64, a, xs);
        b.ret(Some(r.into()));
        p.def(itfib, b);
    }
    // nested loops: sum over i < n, j < i of i * j (wrapping)
    let nested = p.decl("t.nested", &[t.u32], Some(t.u64));
    {
        let mut b = FuncBuilder::new(&[t.u32]);
        let n = b.param(0).unwrap();
        let (oh, ov) = b.create_block_with(&[t.u32, t.u64]);
        let (ob, oexit) = (b.create_block(), b.create_block());
        let (ih, iv) = b.create_block_with(&[t.u32, t.u64]);
        let (ib, iexit) = (b.create_block(), b.create_block());
        let (z, z2) = (b.int(0), b.int(0));
        b.br(oh, &[z, z2]);
        b.switch_to(oh);
        let lt = b.cmp(Op::Lt, t.bool, ov[0], n);
        b.cbr(lt, ob, &[], oexit, &[]);
        b.switch_to(ob);
        let z = b.int(0);
        b.br(ih, &[z, ov[1].into()]);
        b.switch_to(ih);
        let lt = b.cmp(Op::Lt, t.bool, iv[0], ov[0]);
        b.cbr(lt, ib, &[], iexit, &[]);
        b.switch_to(ib);
        let pr = b.binary(Op::MulWrap, t.u32, ov[0], iv[0]);
        let pr64 = b.cast(CastKind::Exact, t.u64, pr);
        let acc = b.binary(Op::AddWrap, t.u64, iv[1], pr64);
        let one = b.int(1);
        let j2 = b.binary(Op::Add, t.u32, iv[0], one);
        b.br(ih, &[j2.into(), acc.into()]);
        b.switch_to(iexit);
        let one = b.int(1);
        let i2 = b.binary(Op::Add, t.u32, ov[0], one);
        b.br(oh, &[i2.into(), iv[1].into()]);
        b.switch_to(oexit);
        b.ret(Some(ov[1].into()));
        p.def(nested, b);
    }
    p.main(|p, b| {
        for n in [0i128, 1, 2, 10, 20] {
            let c = b.int(n);
            let r = b.call(fib, Some(t.u64), &[c]).unwrap();
            p.print(b, r.into(), t.u64);
        }
        for n in [0i128, 1, 5, 93, 200] {
            let c = b.int(n);
            let r = b.call(itfib, Some(t.u64), &[c]).unwrap();
            p.print(b, r.into(), t.u64);
        }
        for n in [0i128, 3, 100, 700] {
            let c = b.int(n);
            let r = b.call(nested, Some(t.u64), &[c]).unwrap();
            p.print(b, r.into(), t.u64);
        }
        // fn values: direct operand, through a slot, compared
        let ft = p.m.types.fn_type(vec![t.u64], t.u64, stave_target::FeatureSet::EMPTY);
        let pft = p.m.types.ref_(true, ft);
        let sl = b.slot(pft);
        let fv = b.func(fib);
        b.store(sl, fv);
        let g = b.load(ft, sl);
        let c = b.int(15);
        let r = b.call_ind(g, Some(t.u64), &[c]).unwrap();
        p.print(b, r.into(), t.u64);
        let c = b.int(15);
        let fi = b.func(itfib);
        let g2 = p.opaque(b, fi, ft);
        let r = b.call_ind(g2, Some(t.u64), &[c]).unwrap();
        p.print(b, r.into(), t.u64);
        let fv2 = b.func(fib);
        let eq = b.cmp(Op::Eq, t.bool, g, fv2);
        p.print(b, eq.into(), t.bool);
        let fv3 = b.func(itfib);
        let ne = b.cmp(Op::Eq, t.bool, g, fv3);
        p.print(b, ne.into(), t.bool);
        // a const condition and a const switch
        let tr = b.int(1);
        let (x1, x2, j) = (b.create_block(), b.create_block(), b.create_block());
        b.cbr(tr, x1, &[], x2, &[]);
        b.switch_to(x1);
        p.print_str(b, "const-then\n");
        b.br(j, &[]);
        b.switch_to(x2);
        p.print_str(b, "const-else\n");
        b.br(j, &[]);
        b.switch_to(j);
        let r = b.int(3);
        b.ret(Some(r));
    });
    let r = check(&p, "control_flow");
    assert_eq!(r.code, 3);
}

#[test]
fn memory_ops() {
    let mut p = Prog::new();
    let t = p.t;
    let st = p.structure(vec![t.u8, t.f64, t.i16, t.char]);
    let e = p.enumeration(vec![vec![t.u16], vec![t.i64, t.f32]]);
    let arr = p.m.types.array(t.i32, 8);
    let sl = p.m.types.slice(t.i32);
    let slref = p.m.types.ref_(false, sl);
    // fn sum(s: &[i32]) -> i64 with bounds checks
    let fsum = p.decl("t.sum", &[slref], Some(t.i64));
    {
        let mut b = FuncBuilder::new(&[slref]);
        let s = b.param(0).unwrap();
        let n = b.op(Op::SliceLen, t.usize, &[s.into()]);
        let (h, hv) = b.create_block_with(&[t.usize, t.i64]);
        let (body, exit) = (b.create_block(), b.create_block());
        let (z, z2) = (b.int(0), b.int(0));
        b.br(h, &[z, z2]);
        b.switch_to(h);
        let lt = b.cmp(Op::Lt, t.bool, hv[0], n);
        b.cbr(lt, body, &[], exit, &[]);
        b.switch_to(body);
        b.bounds_check(hv[0], n);
        let pel = p.m.types.ref_(false, t.i32);
        let ep = b.op(Op::ElemAddr, pel, &[s.into(), hv[0].into()]);
        let x = b.load(t.i32, ep);
        let x64 = b.cast(CastKind::Exact, t.i64, x);
        let acc = b.binary(Op::Add, t.i64, hv[1], x64);
        let one = b.int(1);
        let i2 = b.binary(Op::Add, t.usize, hv[0], one);
        b.br(h, &[i2.into(), acc.into()]);
        b.switch_to(exit);
        b.ret(Some(hv[1].into()));
        p.def(fsum, b);
    }
    // fn show(s: &str): writes it
    let fshow = p.decl("t.show", &[t.str_ref], None);
    {
        let mut b = FuncBuilder::new(&[t.str_ref]);
        let s = b.param(0).unwrap();
        let ptr = b.op(Op::SlicePtr, t.pu8, &[s.into()]);
        let n = b.op(Op::SliceLen, t.usize, &[s.into()]);
        b.call(p.write, None, &[ptr.into(), n.into()]);
        b.ret(None);
        p.def(fshow, b);
    }
    p.main(|p, b| {
        // scalar slots of every type
        for (ty, c) in [
            (t.u8, b.int(200)),
            (t.i8, b.int(-100)),
            (t.u16, b.int(60000)),
            (t.i16, b.int(-30000)),
            (t.u32, b.int(4_000_000_000)),
            (t.i32, b.int(-2_000_000_000)),
            (t.u64, b.int(u64::MAX as i128)),
            (t.i64, b.int(i64::MIN as i128)),
            (t.bool, b.int(1)),
            (t.char, b.int(0x1f600)),
            (t.f32, b.f32(-3.25)),
            (t.f64, b.f64(1e-300)),
        ] {
            let pt = p.m.types.ref_(true, ty);
            let s = b.slot(pt);
            b.store(s, c);
            let x = b.load(ty, s);
            p.print(b, x.into(), ty);
        }
        // struct in a slot: field addresses, whole loads/stores
        let pst = p.m.types.ref_(true, st);
        let s = b.slot(pst);
        let (a, c, d, f) = (b.int(7), b.f64(2.5), b.int(-9), b.int(66));
        let v = b.op(Op::Make, st, &[a, c, d, f]);
        b.store(s, v);
        let pf = p.m.types.ref_(true, t.i16);
        let k2 = b.int(2);
        let fa = b.op(Op::FieldAddr, pf, &[s.into(), k2]);
        let nv = b.int(1234);
        b.store(fa, nv);
        let w = b.load(st, s);
        print_fields(p, b, w.into(), &[t.u8, t.f64, t.i16, t.char]);
        // through an opaque pointer
        let ps2 = p.m.types.ptr(true, st);
        let rp = b.cast(CastKind::Exact, ps2, s);
        let rp = p.opaque(b, rp.into(), ps2);
        let k1 = b.int(1);
        let pf64 = p.m.types.ptr(true, t.f64);
        let fa = b.op(Op::FieldAddr, pf64, &[rp.into(), k1]);
        let x = b.load(t.f64, fa);
        p.print(b, x.into(), t.f64);
        let w2 = b.load(st, rp);
        print_fields(p, b, w2.into(), &[t.u8, t.f64, t.i16, t.char]);
        // enum in a slot with addr.enum_field
        let pe = p.m.types.ref_(true, e);
        let es = b.slot(pe);
        let (v1, x1, x2) = (b.int(1), b.int(-77), b.f32(0.25));
        let ev = b.op(Op::EnumMake, e, &[v1, x1, x2]);
        b.store(es, ev);
        let pi64 = p.m.types.ref_(true, t.i64);
        let (v1, f0) = (b.int(1), b.int(0));
        let efa = b.op(Op::EnumFieldAddr, pi64, &[es.into(), v1, f0]);
        let x = b.load(t.i64, efa);
        p.print(b, x.into(), t.i64);
        let ev2 = b.load(e, es);
        let tag = b.op(Op::EnumTag, t.u32, &[ev2.into()]);
        p.print(b, tag.into(), t.u32);
        // array slot → slice → sum
        let parr = p.m.types.ref_(true, arr);
        let asl = b.slot(parr);
        let es: Vec<Operand> =
            (0..8).map(|i| b.int(i * 1_000_000_007 % 2_000_000_000 - 1_000_000_000)).collect();
        let av = b.op(Op::Make, arr, &es);
        b.store(asl, av);
        let pel = p.m.types.ref_(true, t.i32);
        let z = b.int(0);
        let e0 = b.op(Op::ElemAddr, pel, &[asl.into(), z]);
        let pel_c = p.m.types.ref_(false, t.i32);
        let e0c = b.cast(CastKind::Exact, pel_c, e0);
        for n in [0i128, 3, 8] {
            let nn = b.int(n);
            let sv = b.op(Op::SliceMake, slref, &[e0c.into(), nn]);
            let r = b.call(fsum, Some(t.i64), &[sv.into()]).unwrap();
            p.print(b, r.into(), t.i64);
        }
        // raw pointer arithmetic with a negative index
        let pr = p.m.types.ptr(true, t.i32);
        let k5 = b.int(5);
        let e5 = b.op(Op::ElemAddr, pel, &[asl.into(), k5]);
        let e5r = b.cast(CastKind::Exact, pr, e5);
        let m2 = b.int(-2);
        let mi = p.opaque(b, m2, t.i64);
        let e3 = b.op(Op::ElemAddr, pr, &[e5r.into(), mi.into()]);
        let x = b.load(t.i32, e3);
        p.print(b, x.into(), t.i32);
        // string literals as values
        let s1 = p.m.add_string(b"literal one\n".to_vec());
        let so = b.str(s1);
        b.call(fshow, None, &[so]);
        let pstr = p.m.types.ref_(true, t.str_ref);
        let ss = b.slot(pstr);
        b.store(ss, so);
        let sv = b.load(t.str_ref, ss);
        b.call(fshow, None, &[sv.into()]);
        let s2 = p.m.add_string(Vec::new());
        let so2 = b.str(s2);
        b.call(fshow, None, &[so2]);
        let n = b.op(Op::SliceLen, t.usize, &[so]);
        p.print(b, n.into(), t.usize);
    });
    check(&p, "memory_ops");
}

/// Bounds traps, every `panic.*` kind, and dynamic `get`/`set` out of range.
#[test]
fn panics_and_bounds() {
    for (k, kind) in PanicKind::ALL.iter().enumerate() {
        let mut p = Prog::new();
        p.main(|p, b| {
            p.print_str(b, "x\n");
            b.panic(*kind);
            b.unreachable();
        });
        assert_eq!(check(&p, &format!("panic_{k}")).code, 100 + k as i32);
    }
    for case in 0..4 {
        let mut p = Prog::new();
        let t = p.t;
        let arr = p.m.types.array(t.u16, 3);
        p.main(|p, b| {
            let es: Vec<Operand> = (0..3).map(|i| b.int(i)).collect();
            let av = b.op(Op::Make, arr, &es);
            let c = b.int(3);
            let i = p.opaque(b, c, t.usize);
            match case {
                0 => {
                    let e = b.op(Op::Get, t.u16, &[av.into(), i.into()]);
                    p.print(b, e.into(), t.u16);
                }
                1 => {
                    let z = b.int(0);
                    let r = b.op(Op::Set, arr, &[av.into(), i.into(), z]);
                    let k0 = b.int(0);
                    let e = b.op(Op::Get, t.u16, &[r.into(), k0]);
                    p.print(b, e.into(), t.u16);
                }
                2 => {
                    let n = b.int(3);
                    b.bounds_check(i, n);
                }
                _ => {
                    let k = b.int(2);
                    let n = p.opaque(b, k, t.usize);
                    let z = b.int(1);
                    b.bounds_check(z, n);
                    b.bounds_check(k, n);
                }
            }
        });
        assert_eq!(check(&p, &format!("bounds_{case}")).code, 100, "case {case}");
    }
}

/// Without `runtime.panic.trap` the backend's fallback reports `panic: <Kind>` and exits 101.
#[test]
fn trap_fallback() {
    let mut p = Prog::with_trap(false);
    let t = p.t;
    p.main(|p, b| {
        let c = b.int(i64::MAX as i128);
        let x = p.opaque(b, c, t.i64);
        let one = b.int(1);
        let r = b.binary(Op::Add, t.i64, x, one);
        p.print(b, r.into(), t.i64);
    });
    let want = interp(&p);
    assert_eq!(want.code, 101);
    if let Some(got) = native(&p, "fallback", &crate::debug::Options::default()) {
        assert_eq!(got.code, 101);
        assert_eq!(String::from_utf8_lossy(&got.stderr), "panic: Overflow\n");
    }
}
