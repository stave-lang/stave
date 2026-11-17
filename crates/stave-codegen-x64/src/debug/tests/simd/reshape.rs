//! Casts, bitcasts, splat, split / join, widen / narrow, permute, lookup, compress / expand, and
//! the mask ops.

use stave_ir::arith;
use stave_ir::{CastKind, VecOp};

use super::values::*;
use super::{Arg, Case, Lane, Res, Shape, batch, sh, shapes};

fn v(s: Shape, lanes: Vec<u64>) -> Arg {
    Arg::V(s, lanes)
}

const KINDS: [CastKind; 5] = [CastKind::Exact, CastKind::Wrap, CastKind::Sat, CastKind::Round, CastKind::Try];

/// Source values for a cast from `from` to `to`: the source's edges plus values at and beyond
/// the target's range and precision.
fn cast_inputs(from: Lane, to: Lane) -> Vec<u64> {
    let mut targets: Vec<f64> = Vec::new();
    if let Some(t) = to.int() {
        let (lo, hi) = (t.min() as f64, t.max() as f64);
        targets.extend([
            lo,
            lo - 1.0,
            hi,
            hi + 1.0,
            hi * 2.0 + 2.0,
            lo * 2.0 - 2.0,
            hi - 0.5,
            lo + 0.5,
            -0.99,
            0.99,
        ]);
        targets.extend([lo - 0.5, hi + 0.5]);
        // inside the range: large values, and the middle of an unsigned range (the sign bit of
        // the lane)
        targets.extend([
            hi * 0.75,
            lo * 0.75,
            (hi + 1.0) / 2.0,
            (hi + 1.0) / 2.0 - 1.0,
            (hi + 1.0) / 2.0 + 256.0,
        ]);
    } else {
        targets.extend([16777216.0, 16777217.0, 16777219.0, 9007199254740993.0, 9223372036854775807.0]);
        targets.extend([
            -9223372036854775808.0,
            18446744073709551615.0,
            3.4028235677973366e38,
            3.4028235e38 * 1.0000001,
        ]);
        targets.extend([
            1.401298464324817e-45,
            7.006492321624085e-46,
            1.0000000596046448,
            1.0000000894069672,
        ]);
    }
    let mut vals = Vec::new();
    match from.int() {
        Some(t) => {
            vals.extend(int_edges(from));
            let mut extra: Vec<i128> = Vec::new();
            if let Some(tt) = to.int() {
                extra.extend([tt.min(), tt.min() - 1, tt.max(), tt.max() + 1, tt.max() * 2 + 1]);
            }
            extra.extend([16777217, 16777216, -16777217, (1 << 53) + 1, (1 << 62) + 1]);
            // 32-bit values whose f32 rounding is a tie or next to one, with and without the top bit
            extra.extend([0x8000_0080, 0x8000_0081, 0x8000_0180, 0xffff_ff7f, 0xffff_ff80, 0x7fff_ffbf]);
            extra.extend([0x7fff_ffc0, 0x0001_ffff, 0x0100_0003]);
            extra.extend([i64::MAX as i128, i64::MIN as i128, u64::MAX as i128, (1i128 << 63) + 1025]);
            extra.extend(targets.iter().filter(|x| x.is_finite() && x.abs() < 1e20).map(|x| *x as i128));
            for x in extra {
                if t.fits(x) && !vals.contains(&from.raw(x)) {
                    vals.push(from.raw(x));
                }
            }
        }
        None => {
            vals.extend(float_unary(from));
            for x in targets {
                let b = fl(from, x);
                if !vals.contains(&b) {
                    vals.push(b);
                }
                let nb = fl(from, -x);
                if !vals.contains(&nb) {
                    vals.push(nb);
                }
            }
        }
    }
    vals
}

/// The lane counts a cast between `a` and `b` runs at: 2, the widest 128-bit one, the 256-bit
/// one, the 512-bit one.
fn cast_lanes(a: Lane, b: Lane) -> Vec<u32> {
    let w = a.bits().max(b.bits());
    let mut v = vec![2, 128 / w, 256 / w, 512 / w];
    v.dedup();
    v
}

fn converts(kind: CastKind, from: Lane, to: Lane, x: u64) -> bool {
    let src = from.int().map(|t| t.norm(x)).unwrap_or(x);
    arith::convert(kind, from.cast_ty(), to.cast_ty(), src).is_some()
}

pub fn casts() -> Vec<Case> {
    let mut out = Vec::new();
    for from in Lane::ALL {
        for to in Lane::ALL {
            for kind in KINDS {
                if !arith::cast_allowed(kind, from.cast_ty(), to.cast_ty()) {
                    continue;
                }
                let vals = cast_inputs(from, to);
                let op = VecOp::Cast(kind);
                let nan = to.is_float();
                for n in cast_lanes(from, to) {
                    let (s, r) = (sh(from, n), sh(to, n));
                    let nn = n as usize;
                    let mut tuples: Vec<Vec<Arg>> = Vec::new();
                    if kind == CastKind::Try {
                        let good: Vec<u64> =
                            vals.iter().copied().filter(|&x| converts(kind, from, to, x)).collect();
                        let bad: Vec<u64> =
                            vals.iter().copied().filter(|&x| !converts(kind, from, to, x)).collect();
                        for c in chunks(&good, nn) {
                            tuples.push(vec![v(s, c)]);
                        }
                        for (k, &x) in bad.iter().enumerate() {
                            let mut c: Vec<u64> = (0..nn).map(|j| good[(j + k) % good.len()]).collect();
                            c[if k % 2 == 0 { nn - 1 } else { 0 }] = x;
                            tuples.push(vec![v(s, c)]);
                        }
                        out.extend(batch(op, Res::Opt(r), nan, tuples));
                    } else {
                        for c in spread(&vals, nn) {
                            tuples.push(vec![v(s, c)]);
                        }
                        out.extend(batch(op, Res::V(r), nan, tuples));
                    }
                }
            }
        }
    }
    out
}

/// Bit patterns for pure moves: every NaN kind, both zeros, distinct bytes.
fn patterns(l: Lane) -> Vec<u64> {
    let mut v: Vec<u64> = if l.is_float() {
        let mut v = nans(l);
        v.extend(extremes(l));
        v.extend([fl(l, -0.0), fl(l, 0.0), fl(l, 1.5), fl(l, -2.0)]);
        v
    } else {
        int_edges(l)
    };
    let mut r = Rng::new(l.bits() as u64 + 77);
    for _ in 0..8 {
        v.push(r.next() & l.ones());
    }
    v
}

fn distinct(s: Shape, seed: u64) -> Vec<u64> {
    let mut r = Rng::new(seed);
    let p = patterns(s.lane);
    (0..s.n as usize)
        .map(|j| if j < p.len() && seed == 0 { p[j] } else { r.next() & s.lane.ones() })
        .collect()
}

/// Permutation patterns from `n` lanes to `m`: identity / prefix, reverse, broadcasts, rotate,
/// interleave, even / odd, half swap (crosses 128-bit lanes on 256-bit vectors), pseudo-random.
fn perms(n: u32, m: u32) -> Vec<Vec<i128>> {
    let (n, m) = (n as i128, m as usize);
    let mut out: Vec<Vec<i128>> = vec![
        (0..m).map(|j| j as i128 % n).collect(),
        (0..m).map(|j| n - 1 - (j as i128 % n)).collect(),
        vec![0; m],
        vec![n - 1; m],
        (0..m).map(|j| (j as i128 + 1) % n).collect(),
        (0..m).map(|j| if j % 2 == 0 { (j as i128 / 2) % n } else { (n / 2 + j as i128 / 2) % n }).collect(),
        (0..m).map(|j| (2 * j as i128) % n + (j as i128 * 2 / n) % 2).collect(),
        (0..m).map(|j| (j as i128 + n / 2) % n).collect(),
    ];
    let mut r = Rng::new((n as u64) << 8 | m as u64);
    for _ in 0..3 {
        out.push((0..m).map(|_| r.below(n as u64) as i128).collect());
    }
    out.dedup();
    out
}

pub fn reshape() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::ALL {
        let pat = patterns(l);
        for s in shapes(l) {
            let n = s.n as usize;
            // splat: constants and opaque scalars, every pattern
            let tuples = pat.iter().enumerate().map(|(k, &x)| vec![Arg::S(l, x, k % 2 == 1)]).collect();
            out.extend(batch(VecOp::Splat, Res::V(s), false, tuples));
            // bitcast to every shape of the same width
            for to in Lane::ALL {
                if s.bits() % to.bits() != 0 || s.bits() / to.bits() < 2 {
                    continue;
                }
                let r = sh(to, s.bits() / to.bits());
                let tuples = chunks(&pat, n).into_iter().map(|c| vec![v(s, c)]).collect();
                out.extend(batch(VecOp::Bitcast, Res::V(r), false, tuples));
            }
            if n >= 4 {
                let h = sh(l, s.n / 2);
                let tuples = chunks(&pat, n).into_iter().map(|c| vec![v(s, c)]).collect();
                out.extend(batch(VecOp::Split, Res::Pair(h), false, tuples));
                let tuples = chunks(&pat, n / 2)
                    .chunks(2)
                    .filter(|c| c.len() == 2)
                    .map(|c| vec![v(h, c[0].clone()), v(h, c[1].clone())])
                    .collect::<Vec<_>>();
                out.extend(batch(VecOp::Join, Res::V(s), false, tuples));
                if let Some(w) = l.wide() {
                    let r = sh(w, s.n / 2);
                    let src = if l.is_float() { float_unary(l) } else { int_edges(l) };
                    for op in [VecOp::WidenLo, VecOp::WidenHi] {
                        let tuples = spread(&src, n).into_iter().map(|c| vec![v(s, c)]).collect();
                        out.extend(batch(op, Res::V(r), l.is_float(), tuples));
                    }
                }
            }
            // narrow: two vectors of the double-width lane into one of this lane
            if let (Some(w), false, true) = (l.wide(), l.is_float(), n >= 4) {
                let ws = sh(w, s.n / 2);
                let mut src = int_edges(w);
                let t = l.int().unwrap();
                for x in [t.min() - 1, t.min(), t.max(), t.max() + 1, -1, 2 * t.max() + 1] {
                    if w.int().unwrap().fits(x) {
                        src.push(w.raw(x));
                    }
                }
                for op in [VecOp::NarrowWrap, VecOp::NarrowSat] {
                    let tuples = spread(&src, n / 2)
                        .chunks(2)
                        .filter(|c| c.len() == 2)
                        .map(|c| vec![v(ws, c[0].clone()), v(ws, c[1].clone())])
                        .collect();
                    out.extend(batch(op, Res::V(s), false, tuples));
                }
            }
            // permute to n, n/2 and 2n lanes
            let src = distinct(s, 0);
            for m in [s.n, s.n / 2, s.n * 2] {
                if m < 2 || m * l.bits() > 256 {
                    continue;
                }
                let tuples = perms(s.n, m)
                    .into_iter()
                    .map(|p| {
                        let mut a = vec![v(s, src.clone())];
                        a.extend(p.into_iter().map(Arg::K));
                        a
                    })
                    .collect();
                out.extend(batch(VecOp::Permute, Res::V(sh(l, m)), false, tuples));
            }
            // lookup with indices of several int lanes
            let tbl = distinct(s, 5);
            let mut ils =
                vec![Lane::uint(l.bits()), Lane::sint(l.bits()), Lane::U8, Lane::I8, Lane::U32, Lane::U64];
            ils.sort();
            ils.dedup();
            for il in ils {
                let is = sh(il, s.n);
                if is.bits() > 256 {
                    continue;
                }
                let t = il.int().unwrap();
                let mut idx: Vec<i128> = (0..n as i128).rev().collect();
                idx.extend([
                    0,
                    n as i128,
                    n as i128 + 1,
                    15,
                    16,
                    17,
                    31,
                    32,
                    63,
                    127,
                    128,
                    255,
                    256,
                    -1,
                    -16,
                    t.min(),
                    t.max(),
                ]);
                let idx: Vec<u64> = idx.into_iter().filter(|&x| t.fits(x)).map(|x| il.raw(x)).collect();
                let tuples = spread(&idx, n).into_iter().map(|c| vec![v(s, tbl.clone()), v(is, c)]).collect();
                out.extend(batch(VecOp::Lookup, Res::V(s), false, tuples));
            }
        }
    }
    out
}

/// Every mask of up to 8 lanes; for more, the masks at the edges of the 8-lane groups and the
/// 128-bit halves (single lanes, prefixes, suffixes, whole groups) plus pseudo-random ones.
pub fn sweep_masks(n: u32) -> Vec<u64> {
    let all = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    if n <= 8 {
        return (0..=all).collect();
    }
    let mut v = mask_patterns(n);
    for k in [1, 7, 8, 9, n / 2 - 1, n / 2, n / 2 + 1, n - 9, n - 8, n - 1] {
        v.extend([1 << k, all >> (n - k), all << k & all, !(1 << k) & all]);
    }
    for g in (0..n).step_by(8) {
        v.push(0xff << g & all);
        v.push(!(0xff << g) & all);
    }
    let mut r = Rng::new(u64::from(n) * 77);
    v.extend((0..24).map(|_| r.next() & all));
    let mut out = Vec::new();
    for x in v {
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out
}

/// Lane bits that tell every lane apart and are never zero: the lane number in the low byte.
pub fn lane_ids(s: Shape, salt: u64) -> Vec<u64> {
    let mut r = Rng::new(salt.wrapping_add(5));
    (0..u64::from(s.n))
        .map(|j| {
            if s.lane.bits() == 8 {
                (salt % 3) * 0x40 + j + 1
            } else {
                (r.next() << 8 | (j + 1)) & s.lane.ones()
            }
        })
        .collect()
}

/// `compress` / `expand` of every shape under [`sweep_masks`].
pub fn compress() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::ALL {
        for s in shapes(l) {
            for op in [VecOp::Compress, VecOp::Expand] {
                let tuples = sweep_masks(s.n)
                    .into_iter()
                    .map(|m| vec![v(s, lane_ids(s, m)), Arg::M(l.bits(), s.n, m)])
                    .collect();
                out.extend(batch(op, Res::V(s), false, tuples));
            }
        }
    }
    out
}

/// Every mask shape: lane width 8 to 64, 2 to 64 lanes, at most 512 bits.
fn mask_shapes() -> Vec<(u32, u32)> {
    let mut v = Vec::new();
    for w in [8, 16, 32, 64] {
        for n in [2, 4, 8, 16, 32, 64] {
            if w * n <= 512 && w * n >= 16 {
                v.push((w, n));
            }
        }
    }
    v
}

pub fn masks() -> Vec<Case> {
    let mut out = Vec::new();
    for (w, n) in mask_shapes() {
        let pats = mask_patterns(n);
        let m = |x: u64| Arg::M(w, n, x);
        for op in [VecOp::MaskAnd, VecOp::MaskOr, VecOp::MaskXor] {
            let tuples =
                pats.iter().flat_map(|&a| pats.iter().step_by(2).map(move |&b| vec![m(a), m(b)])).collect();
            out.extend(batch(op, Res::M(w, n), false, tuples));
        }
        out.extend(batch(VecOp::MaskNot, Res::M(w, n), false, pats.iter().map(|&a| vec![m(a)]).collect()));
        // select between two masks and between two vectors of every lane type of this width
        let tuples = pats.iter().map(|&a| vec![m(a), m(pats[1]), m(pats[pats.len() - 1])]).collect();
        out.extend(batch(VecOp::Select, Res::M(w, n), false, tuples));
        for l in Lane::ALL.into_iter().filter(|l| l.bits() == w) {
            let s = sh(l, n);
            let (a, b) = (distinct(s, 1), distinct(s, 0));
            let tuples = pats.iter().map(|&x| vec![m(x), v(s, a.clone()), v(s, b.clone())]).collect();
            out.extend(batch(VecOp::Select, Res::V(s), false, tuples));
        }
        let bt = Lane::uint(stave_ir::simd::to_bits_width(n));
        out.extend(batch(VecOp::ToBits, Res::S(bt), false, pats.iter().map(|&a| vec![m(a)]).collect()));
        for l in [Lane::uint(w), Lane::sint(w)] {
            let s = sh(l, n);
            let tuples = pats.iter().map(|&a| vec![m(a)]).collect();
            out.extend(batch(VecOp::ToVec, Res::V(s), false, tuples));
        }
        let ks = [0u64, 1, n as u64 - 1, n as u64, n as u64 + 1, 1 << 32, (1 << 32) + 1, 1 << 63, u64::MAX];
        let tuples = ks.iter().flat_map(|&k| [vec![Arg::U(k, false)], vec![Arg::U(k, true)]]).collect();
        out.extend(batch(VecOp::FirstN, Res::M(w, n), false, tuples));
        for w2 in [8, 16, 32, 64] {
            if w2 * n > 512 {
                continue;
            }
            let tuples = pats.iter().map(|&a| vec![m(a)]).collect();
            out.extend(batch(VecOp::Rewiden, Res::M(w2, n), false, tuples));
        }
        out.extend(batch(VecOp::MaskAll, Res::M(w, n), false, vec![vec![]]));
        out.extend(batch(VecOp::MaskNone, Res::M(w, n), false, vec![vec![]]));
    }
    out
}
