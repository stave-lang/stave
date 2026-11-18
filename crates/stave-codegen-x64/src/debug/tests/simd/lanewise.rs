//! Lane-wise arithmetic, compares, shifts, division, reductions and scans, and their traps.

use stave_ir::{PanicKind, VecOp};

use super::values::*;
use super::{Arg, Case, Lane, Res, Shape, batch, full_shapes, sh, shapes, trap_case};

fn v(s: Shape, lanes: Vec<u64>) -> Arg {
    Arg::V(s, lanes)
}

fn pairs(s: Shape, a: &[u64], b: &[u64]) -> Vec<Vec<Arg>> {
    cross(a, b, s.n as usize).into_iter().map(|(x, y)| vec![v(s, x), v(s, y)]).collect()
}

fn unary(s: Shape, vals: &[u64]) -> Vec<Vec<Arg>> {
    spread(vals, s.n as usize).into_iter().map(|x| vec![v(s, x)]).collect()
}

/// `a * b + c` triples whose fused result differs from the unfused one, or is denormal, zero
/// with a sign, overflowing, or NaN.
fn fma_triples(l: Lane) -> Vec<(u64, u64, u64)> {
    let [dmin, _, nmin, max, inf] = extremes(l);
    let e = match l {
        Lane::F32 => 2f64.powi(-12),
        _ => 2f64.powi(-30),
    };
    let f = |x: f64| fl(l, x);
    vec![
        (f(1.0 + e), f(1.0 - e), f(-1.0)),
        (f(1.0 - e), f(1.0 + e), f(1.0)),
        (f(0.1), f(10.0), f(-1.0)),
        (f(1.0 / 3.0), f(3.0), f(-1.0)),
        (max, f(2.0), neg(l, max)),
        (max, max, neg(l, inf)),
        (nmin, f(0.5), f(0.0)),
        (dmin, f(0.5), f(0.0)),
        (dmin, f(1.5), f(0.0)),
        (nmin, f(-1.0), nmin),
        (f(-0.0), f(1.0), f(-0.0)),
        (f(0.0), f(-1.0), f(0.0)),
        (inf, f(0.0), f(1.0)),
        (inf, f(1.0), neg(l, inf)),
        (nans(l)[3], f(1.0), f(1.0)),
        (f(1.0), f(1.0), nans(l)[5]),
    ]
}

pub fn float_arith() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::FLOATS {
        let ps = float_pairset(l);
        let inexact = float_inexact(l);
        let un = float_unary(l);
        for s in shapes(l) {
            use VecOp::*;
            for op in [Add, Sub, Mul, Div, Copysign, Min, Max, MinNum, MaxNum, MinX86, MaxX86] {
                let mut t = pairs(s, &ps, &ps);
                t.extend(pairs(s, &inexact, &inexact));
                out.extend(batch(op, Res::V(s), op != Copysign, t));
            }
            out.extend(batch(Copysign, Res::V(s), false, pairs(s, &un, &[fl(l, 1.0), fl(l, -0.0)])));
            for op in [Neg, Abs, Sqrt, Floor, Ceil, Round, Trunc] {
                out.extend(batch(op, Res::V(s), !matches!(op, Neg | Abs), unary(s, &un)));
            }
            let small: Vec<u64> = {
                let [dmin, _, _, max, inf] = extremes(l);
                vec![fl(l, -0.0), fl(l, 1.0), fl(l, -1.0), fl(l, 0.1), max, dmin, inf, nans(l)[2]]
            };
            let mut triples: Vec<(u64, u64, u64)> = Vec::new();
            for &a in &small {
                for &b in &small {
                    for &c in &small {
                        triples.push((a, b, c));
                    }
                }
            }
            triples.extend(fma_triples(l));
            let n = s.n as usize;
            let t = triples
                .chunks(n)
                .map(|c| {
                    let c: Vec<(u64, u64, u64)> = (0..n).map(|j| c[j % c.len()]).collect();
                    vec![
                        v(s, c.iter().map(|x| x.0).collect()),
                        v(s, c.iter().map(|x| x.1).collect()),
                        v(s, c.iter().map(|x| x.2).collect()),
                    ]
                })
                .collect();
            out.extend(batch(Fma, Res::V(s), true, t));
        }
    }
    out
}

/// Vectors for checked ops: all-safe ones, and for each lane position one with a single
/// overflowing pair there.
fn checked_tuples(op: VecOp, s: Shape) -> Vec<Vec<Arg>> {
    let l = s.lane;
    let t = l.int().unwrap();
    let ps = int_pairset(l);
    let mut safe = Vec::new();
    let mut bad = Vec::new();
    for &a in &ps {
        for &b in &ps {
            match stave_ir::simd::int_binary(op, t, t.norm(a), t.norm(b)) {
                Ok(_) => safe.push((a, b)),
                Err(_) => bad.push((a, b)),
            }
        }
    }
    let n = s.n as usize;
    let mut out = Vec::new();
    for c in safe.chunks(n) {
        let c: Vec<(u64, u64)> = (0..n).map(|j| c[j % c.len()]).collect();
        out.push(c);
    }
    let picks: Vec<(u64, u64)> = bad.iter().step_by((bad.len() / 4).max(1)).copied().take(4).collect();
    for (k, &p) in picks.iter().enumerate() {
        for j in 0..n {
            let mut c: Vec<(u64, u64)> = (0..n).map(|i| safe[(i + j + k * 7) % safe.len()]).collect();
            c[j] = p;
            out.push(c);
        }
    }
    out.into_iter()
        .map(|c| vec![v(s, c.iter().map(|p| p.0).collect()), v(s, c.iter().map(|p| p.1).collect())])
        .collect()
}

pub fn int_arith() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::INTS {
        let ps = int_pairset(l);
        let ed = int_edges(l);
        for s in shapes(l) {
            use VecOp::*;
            for op in
                [AddWrap, SubWrap, MulWrap, AddSat, SubSat, MulSat, MulHi, Min, Max, And, Or, Xor, Andnot]
            {
                out.extend(batch(op, Res::V(s), false, pairs(s, &ps, &ps)));
            }
            for op in [AbsWrap, Popcount, Clz, Not] {
                if op == AbsWrap && !l.signed() {
                    continue;
                }
                out.extend(batch(op, Res::V(s), false, unary(s, &ed)));
            }
            for op in [CheckedAdd, CheckedSub, CheckedMul] {
                out.extend(batch(op, Res::Opt(s), false, checked_tuples(op, s)));
            }
        }
    }
    out
}

fn traps_div(t: stave_ir::arith::IntTy, a: u64, b: u64) -> bool {
    stave_ir::simd::int_binary(VecOp::DivLanes, t, t.norm(a), t.norm(b)).is_err()
}

/// `div_const` divisors: one of each form of the multiply-high plans (±1, ±2, ±3, 5, 6, ±7,
/// ±10, ±15, 25, 641, powers of two, MIN, MAX).
fn divisors(l: Lane) -> Vec<i128> {
    let t = l.int().unwrap();
    let b = l.bits();
    let mut v = vec![
        1,
        -1,
        2,
        -2,
        3,
        -3,
        5,
        6,
        7,
        -7,
        10,
        -10,
        15,
        -15,
        25,
        641,
        1 << (b - 2),
        -(1 << (b - 2)),
        t.min(),
        t.max(),
        t.max() - 1,
    ];
    v.retain(|&k| k != 0 && t.fits(k));
    v.dedup();
    v
}

pub fn int_div_shift() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::INTS {
        let t = l.int().unwrap();
        let ps = int_pairset(l);
        let ed = int_edges(l);
        let bits = l.bits() as i128;
        for s in shapes(l) {
            let n = s.n as usize;
            let ok: Vec<(u64, u64)> = ps
                .iter()
                .flat_map(|&a| ps.iter().map(move |&b| (a, b)))
                .filter(|&(a, b)| !traps_div(t, a, b))
                .collect();
            let tuples = ok
                .chunks(n)
                .map(|c| {
                    let c: Vec<(u64, u64)> = (0..n).map(|j| c[j % c.len()]).collect();
                    vec![v(s, c.iter().map(|p| p.0).collect()), v(s, c.iter().map(|p| p.1).collect())]
                })
                .collect();
            out.extend(batch(VecOp::DivLanes, Res::V(s), false, tuples));
            let mut tuples = Vec::new();
            for k in divisors(l) {
                let vals: Vec<u64> = ed.iter().copied().filter(|&a| !traps_div(t, a, l.raw(k))).collect();
                for x in spread(&vals, n) {
                    tuples.push(vec![v(s, x), Arg::K(k)]);
                }
            }
            out.extend(batch(VecOp::DivConst, Res::V(s), false, tuples));
            // shifts: in-range vector amounts, scalar amounts of several types
            let amts: Vec<u64> = [0, 1, bits / 2, bits - 1, 3].iter().map(|&k| l.raw(k)).collect();
            for op in [VecOp::Shl, VecOp::Shr, VecOp::ShlWrap] {
                let mut tuples = Vec::new();
                for (k, x) in spread(&ed, n).into_iter().enumerate() {
                    let a: Vec<u64> = (0..n).map(|j| amts[(j + k) % amts.len()]).collect();
                    tuples.push(vec![v(s, x), v(s, a)]);
                }
                for x in chunks(&ed, n).into_iter().take(3) {
                    for amt in [0, 1, bits - 1] {
                        for konst in [false, true] {
                            tuples.push(vec![v(s, x.clone()), Arg::S(l, l.raw(amt), konst)]);
                            tuples.push(vec![v(s, x.clone()), Arg::S(Lane::U8, amt as u64, konst)]);
                            tuples.push(vec![v(s, x.clone()), Arg::S(Lane::I64, amt as u64, konst)]);
                            tuples.push(vec![v(s, x.clone()), Arg::U(amt as u64, konst)]);
                        }
                    }
                }
                if op == VecOp::ShlWrap {
                    // amounts at and past the lane width, negative ones: masked by bits-1
                    let wild: Vec<u64> =
                        [bits, bits + 1, 2 * bits - 1, 2 * bits, 255, -1, -bits, t.max(), t.min()]
                            .iter()
                            .filter(|&&k| t.fits(k))
                            .map(|&k| l.raw(k))
                            .collect();
                    for (k, x) in chunks(&ed, n).into_iter().enumerate() {
                        let a: Vec<u64> = (0..n).map(|j| wild[(j + k) % wild.len()]).collect();
                        tuples.push(vec![v(s, x), v(s, a)]);
                    }
                    let x = chunks(&ed, n).remove(0);
                    for konst in [false, true] {
                        for amt in [bits, bits + 3, 255] {
                            if t.fits(amt) {
                                tuples.push(vec![v(s, x.clone()), Arg::S(l, l.raw(amt), konst)]);
                            }
                        }
                        tuples.push(vec![v(s, x.clone()), Arg::S(Lane::U8, 200, konst)]);
                        tuples.push(vec![v(s, x.clone()), Arg::S(Lane::I64, u64::MAX, konst)]);
                        tuples.push(vec![v(s, x.clone()), Arg::S(Lane::U64, 1 << 40, konst)]);
                        tuples.push(vec![v(s, x.clone()), Arg::U(u64::MAX, konst)]);
                        tuples.push(vec![v(s, x.clone()), Arg::U(64 + 5, konst)]);
                    }
                }
                out.extend(batch(op, Res::V(s), false, tuples));
            }
        }
    }
    out
}

pub fn compares() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::ALL {
        let ps = if l.is_float() { float_pairset(l) } else { int_pairset(l) };
        for s in shapes(l) {
            use VecOp::*;
            for op in [Eq, Ne, Lt, Le, Gt, Ge] {
                out.extend(batch(op, Res::M(l.bits(), s.n), false, pairs(s, &ps, &ps)));
            }
        }
    }
    out
}

/// Float vectors where the order of additions matters (big values cancelling around small ones),
/// plus all-`-0`, mixed zeros, opposite infinities, NaNs in single lanes.
fn float_reduce_inputs(l: Lane, n: usize) -> Vec<Vec<u64>> {
    let big = match l {
        Lane::F32 => 16777216.0,
        _ => 9007199254740992.0,
    };
    let pool: Vec<u64> = [big, -big, 1.0, -1.0, 0.5, 1e-3, 3.0, -0.0, 0.0, big * 1e3, -7.25]
        .iter()
        .map(|&x| fl(l, x))
        .collect();
    let mut out = Vec::new();
    let mut r = Rng::new(n as u64 * 31 + l.bits() as u64);
    for _ in 0..10 {
        out.push((0..n).map(|_| r.pick(&pool)).collect());
    }
    let alt: Vec<u64> = (0..n).map(|j| [fl(l, big), fl(l, 1.0), fl(l, -big), fl(l, 1.0)][j % 4]).collect();
    out.push(alt);
    out.push(vec![fl(l, -0.0); n]);
    out.push((0..n).map(|j| fl(l, if j % 2 == 0 { -0.0 } else { 0.0 })).collect());
    let [_, _, _, max, inf] = extremes(l);
    out.push(
        (0..n)
            .map(|j| {
                if j == 0 {
                    inf
                } else if j == n - 1 {
                    neg(l, inf)
                } else {
                    fl(l, 1.0)
                }
            })
            .collect(),
    );
    out.push(vec![max; n]);
    for pos in [0, n - 1, n / 2] {
        let mut x: Vec<u64> = (0..n).map(|j| fl(l, j as f64 - 1.5)).collect();
        x[pos] = nans(l)[3];
        out.push(x);
    }
    out.extend(spread(&float_unary(l), n));
    out
}

pub fn reductions() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::ALL {
        for s in shapes(l) {
            let n = s.n as usize;
            use VecOp::*;
            if l.is_float() {
                let ins = float_reduce_inputs(l, n);
                let one = |x: &Vec<u64>| vec![v(s, x.clone())];
                for op in [ReduceAdd, ReduceMin, ReduceMax] {
                    out.extend(batch(op, Res::S(l), true, ins.iter().map(one).collect()));
                }
                out.extend(batch(ScanAdd, Res::V(s), true, ins.iter().map(one).collect()));
                let inits: Vec<u64> = [0.0, -0.0, 1.0, 16777216.0, 1e300]
                    .iter()
                    .map(|&x| fl(l, x))
                    .chain([nans(l)[2], extremes(l)[4]])
                    .collect();
                let mut t = Vec::new();
                for (k, x) in ins.iter().enumerate() {
                    t.push(vec![v(s, x.clone()), Arg::S(l, inits[k % inits.len()], k % 2 == 0)]);
                }
                out.extend(batch(ReduceAddOrdered, Res::S(l), true, t));
            } else {
                let ed = int_edges(l);
                let mut ins = spread(&ed, n);
                ins.push(vec![l.ones(); n]);
                ins.push((0..n).map(|j| if j == n - 1 { 0 } else { l.ones() }).collect());
                let one = |x: &Vec<u64>| vec![v(s, x.clone())];
                for op in [ReduceAddWrap, ReduceMin, ReduceMax, ReduceAnd, ReduceOr] {
                    out.extend(batch(op, Res::S(l), false, ins.iter().map(one).collect()));
                }
                out.extend(batch(ScanAddWrap, Res::V(s), false, ins.iter().map(one).collect()));
            }
        }
    }
    out
}

/// Lane-wise traps, lowest lane first.
pub fn traps() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::INTS {
        let t = l.int().unwrap();
        let bits = l.bits() as i128;
        let mut ss = full_shapes(l);
        ss.push(sh(l, 2));
        for s in ss {
            let n = s.n as usize;
            let base = |x: i128| -> Vec<u64> { vec![l.raw(x); n] };
            let (dz, ov) = if n == 2 { (0, 1) } else { (1, 2) };
            // a zero divisor and MIN / -1 in two lanes: the lower one decides the kind
            for (zero, min) in [(dz, ov), (ov, dz)] {
                let mut a = base(100);
                let mut b = base(7);
                b[zero] = 0;
                let kind = if t.signed {
                    a[min] = l.raw(t.min());
                    b[min] = l.raw(-1);
                    if zero < min { PanicKind::DivByZero } else { PanicKind::Overflow }
                } else {
                    PanicKind::DivByZero
                };
                out.push(trap_case(VecOp::DivLanes, Res::V(s), vec![v(s, a), v(s, b)], kind));
            }
            let mut b = base(3);
            b[n - 1] = 0;
            out.push(trap_case(
                VecOp::DivLanes,
                Res::V(s),
                vec![v(s, base(9)), v(s, b)],
                PanicKind::DivByZero,
            ));
            out.push(trap_case(
                VecOp::DivConst,
                Res::V(s),
                vec![v(s, base(9)), Arg::K(0)],
                PanicKind::DivByZero,
            ));
            if t.signed {
                let mut a = base(-5);
                a[n - 1] = l.raw(t.min());
                out.push(trap_case(
                    VecOp::DivConst,
                    Res::V(s),
                    vec![v(s, a), Arg::K(-1)],
                    PanicKind::Overflow,
                ));
            }
            for op in [VecOp::Shl, VecOp::Shr] {
                let mut a = base(1);
                a[n - 1] = l.raw(bits);
                out.push(trap_case(op, Res::V(s), vec![v(s, base(3)), v(s, a)], PanicKind::Overflow));
                let mut a = base(bits - 1);
                a[0] = if t.signed { l.raw(-1) } else { l.raw(t.max()) };
                out.push(trap_case(op, Res::V(s), vec![v(s, base(3)), v(s, a)], PanicKind::Overflow));
                if s.bits() == 128 {
                    out.push(trap_case(
                        op,
                        Res::V(s),
                        vec![v(s, base(3)), Arg::U(bits as u64, false)],
                        PanicKind::Overflow,
                    ));
                    out.push(trap_case(
                        op,
                        Res::V(s),
                        vec![v(s, base(3)), Arg::S(Lane::I64, u64::MAX, false)],
                        PanicKind::Overflow,
                    ));
                }
            }
        }
    }
    out
}
