//! Edge-biased lane values (raw lane bits) and their arrangement into vectors.

use super::Lane;

/// Bits of `x` in float lane `l` (`f32`: rounded).
pub fn fl(l: Lane, x: f64) -> u64 {
    match l {
        Lane::F32 => (x as f32).to_bits() as u64,
        _ => x.to_bits(),
    }
}

/// NaNs: quiet, quiet negative, quiet with payload, signaling, signaling with payload, negative
/// signaling with payload.
pub fn nans(l: Lane) -> Vec<u64> {
    match l {
        Lane::F32 => vec![0x7fc0_0000, 0xffc0_0000, 0x7fc1_2345, 0x7f80_0001, 0x7fa5_a5a5, 0xff81_2345],
        _ => vec![
            0x7ff8_0000_0000_0000,
            0xfff8_0000_0000_0000,
            0x7ff8_0000_1234_5678,
            0x7ff0_0000_0000_0001,
            0x7ff4_a5a5_a5a5_a5a5,
            0xfff0_0000_0001_2345,
        ],
    }
}

/// Smallest denormal, largest denormal, smallest normal, largest finite, +inf.
pub fn extremes(l: Lane) -> [u64; 5] {
    match l {
        Lane::F32 => [1, 0x007f_ffff, 0x0080_0000, 0x7f7f_ffff, 0x7f80_0000],
        _ => [1, 0x000f_ffff_ffff_ffff, 0x0010_0000_0000_0000, 0x7fef_ffff_ffff_ffff, 0x7ff0_0000_0000_0000],
    }
}

pub fn neg(l: Lane, x: u64) -> u64 {
    x ^ (1u64 << (l.bits() - 1))
}

/// Floats for cross products (16 values): ±0, ordinary, denormals, extremes, ±inf, two NaNs.
pub fn float_pairset(l: Lane) -> Vec<u64> {
    let [dmin, dmax, nmin, max, inf] = extremes(l);
    let nn = nans(l);
    vec![
        fl(l, 0.0),
        fl(l, -0.0),
        fl(l, 1.0),
        fl(l, -1.0),
        fl(l, 0.5),
        fl(l, 3.0),
        fl(l, -2.5),
        dmin,
        neg(l, dmax),
        nmin,
        max,
        neg(l, max),
        inf,
        neg(l, inf),
        nn[2],
        nn[5],
    ]
}

/// Inexact operands for `+ - * /` rounding.
pub fn float_inexact(l: Lane) -> Vec<u64> {
    [0.1, 1.0 / 3.0, 1e8, 1e-8, 7.0, -0.3, 16777217.0, 9007199254740993.0].iter().map(|&x| fl(l, x)).collect()
}

/// Floats for unary ops: every special plus values around rounding boundaries.
pub fn float_unary(l: Lane) -> Vec<u64> {
    let [dmin, dmax, nmin, max, inf] = extremes(l);
    let mut v: Vec<u64> = vec![0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 0.25, -0.75]
        .into_iter()
        .chain([0.1, -0.1, 2.0, 4.0, 3.0, 123.456, -98765.4321, 1e10, -1e10, 1e30, -1e30])
        .map(|x| fl(l, x))
        .collect();
    let half = fl(l, 0.5);
    // the neighbours of 0.5 and 1.0
    v.extend([half - 1, half + 1, fl(l, 1.0) + 1, fl(l, 1.0) - 1]);
    v.extend([neg(l, half - 1), neg(l, half + 1)]);
    // the last fractional values and the first integral ones
    let (frac, int) = match l {
        Lane::F32 => (8388607.5f64, 8388608f64),
        _ => (4503599627370495.5, 4503599627370496.0),
    };
    v.extend([fl(l, frac), fl(l, -frac), fl(l, int), fl(l, int + 1.0), fl(l, -int - 1.0), fl(l, int * 2.0)]);
    v.extend([
        dmin,
        neg(l, dmin),
        dmax,
        neg(l, dmax),
        nmin,
        neg(l, nmin),
        max,
        neg(l, max),
        inf,
        neg(l, inf),
    ]);
    v.extend(nans(l));
    v
}

/// Ints for cross products (about 16 values).
pub fn int_pairset(l: Lane) -> Vec<u64> {
    let t = l.int().unwrap();
    let (lo, hi) = (t.min(), t.max());
    let bits = l.bits() as i128;
    let mut v = vec![0, 1, 2, 3, -1, -2, lo, lo + 1, hi, hi - 1, bits - 1, bits, 100, -100];
    v.extend([0x5555_5555_5555_5555 & hi, 0x1234_5678_9abc_def0 & hi, hi / 2, hi / 2 + 1]);
    dedup(l, v)
}

/// Ints for unary ops and single operands.
pub fn int_edges(l: Lane) -> Vec<u64> {
    let t = l.int().unwrap();
    let (lo, hi) = (t.min(), t.max());
    let bits = l.bits() as i128;
    let mut v = vec![0, 1, 2, 3, 5, 7, -1, -2, -3, lo, lo + 1, hi, hi - 1, hi / 2, lo / 2, 100, -100];
    v.extend([bits - 1, bits, bits + 1, 0x5555_5555_5555_5555 & hi, 0x1234_5678_9abc_def0 & hi]);
    v.extend([
        (hi as u128 / 3) as i128,
        127,
        128,
        255,
        256,
        32767,
        32768,
        65535,
        65536,
        0x8000_0000,
        0xffff_ffff,
    ]);
    v.extend([1 << 40, -(1 << 40), 0x00ff_00ff_00ff_00ff & hi, 0x0f0f_0f0f_0f0f_0f0f & hi]);
    dedup(l, v)
}

fn dedup(l: Lane, v: Vec<i128>) -> Vec<u64> {
    let t = l.int().unwrap();
    let mut out: Vec<u64> = Vec::new();
    for x in v {
        if t.fits(x) {
            let r = l.raw(x);
            if !out.contains(&r) {
                out.push(r);
            }
        }
    }
    out
}

/// `vals` cut into vectors of `n` lanes, the last one padded from the start; then the same with
/// the values reversed, so each value lands in two lane positions.
pub fn spread(vals: &[u64], n: usize) -> Vec<Vec<u64>> {
    let mut out = chunks(vals, n);
    let rev: Vec<u64> = vals.iter().rev().copied().collect();
    out.extend(chunks(&rev, n));
    out
}

pub fn chunks(vals: &[u64], n: usize) -> Vec<Vec<u64>> {
    let k = vals.len().div_ceil(n).max(1);
    (0..k).map(|c| (0..n).map(|j| vals[(c * n + j) % vals.len()]).collect()).collect()
}

/// Every pair of `a × b`, laid out lane by lane into vector pairs.
pub fn cross(a: &[u64], b: &[u64], n: usize) -> Vec<(Vec<u64>, Vec<u64>)> {
    let pairs: Vec<(u64, u64)> = a.iter().flat_map(|&x| b.iter().map(move |&y| (x, y))).collect();
    pairs
        .chunks(n)
        .map(|c| {
            let c: Vec<(u64, u64)> = (0..n).map(|j| c[j % c.len()]).collect();
            (c.iter().map(|p| p.0).collect(), c.iter().map(|p| p.1).collect())
        })
        .collect()
}

/// A deterministic pseudo-random sequence.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[self.below(v.len() as u64) as usize]
    }
}

/// Mask patterns of `n` lanes: none, all, only first, only last, alternating both ways, low and
/// high half, two pseudo-random ones.
pub fn mask_patterns(n: u32) -> Vec<u64> {
    let all = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let alt = 0x5555_5555_5555_5555 & all;
    let low = all >> (n / 2);
    let mut v = vec![0, all, 1, 1 << (n - 1), alt, !alt & all, low, all & !low];
    let mut r = Rng::new(n as u64);
    v.push(r.next() & all);
    v.push(r.next() & all);
    let mut out = Vec::new();
    for x in v {
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out
}
