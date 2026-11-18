//! The division plans against truncating division: every divisor and dividend of 8-bit lanes
//! (also as values widened to 16-bit lanes, as x86 divides bytes), every dividend of 16-bit
//! lanes for a spread of divisors, and random dividends of 32- and 64-bit lanes.

use stave_ir::arith::IntTy;
use stave_ir::divmagic::{DivPlan, apply, div_plan};

/// `x / d` truncated (`d` as the lane bits), as normalized lane bits; `None` for `MIN / -1`.
fn quotient(t: IntTy, x: u64, d: i64) -> Option<u64> {
    let (x, d) = (t.value(x), t.value(t.norm(d as u64)));
    t.exact(x / d)
}

fn check(n: u32, prec: u32, signed: bool, d: i64, xs: impl Iterator<Item = u64>) {
    let t = IntTy::new(n as u8, signed);
    let plan = div_plan(d, n, prec, signed);
    for x in xs {
        let x = t.norm(x);
        if let Some(q) = quotient(t, x, d) {
            assert_eq!(apply(plan, x, n, signed), q, "{x:#x} / {d} on {n}-bit lanes, prec {prec}: {plan:?}");
        }
    }
}

/// Divisors of `n`-bit lanes: small ones, powers of two and their neighbours, the extremes.
fn divisors(n: u32, signed: bool) -> Vec<i64> {
    let t = IntTy::new(n as u8, signed);
    let mut v: Vec<i128> = (1..=70).collect();
    for k in 1..n {
        for p in [(1i128 << k) - 1, 1 << k, (1 << k) + 1] {
            v.push(p);
        }
    }
    v.extend([641, 1000, 3333, 6700417, 10_000_000, t.max(), t.max() - 1]);
    if signed {
        let neg: Vec<i128> = v.iter().map(|d| -d).collect();
        v.extend(neg);
        v.push(t.min());
    }
    v.retain(|&d| d != 0 && t.fits(d));
    v.sort();
    v.dedup();
    v.into_iter().map(|d| d as i64).collect()
}

#[test]
fn every_8_bit_division() {
    for signed in [false, true] {
        let t = IntTy::new(8, signed);
        for d in t.min()..=t.max() {
            if d == 0 {
                continue;
            }
            check(8, if signed { 7 } else { 8 }, signed, d as i64, 0..256);
            // bytes widened to words
            let wide = IntTy::new(16, signed);
            let xs = (0..256u64).map(move |x| wide.wrap(t.value(t.norm(x))));
            check(16, if signed { 7 } else { 8 }, signed, d as i64, xs);
        }
    }
}

/// Every 16-bit divisor against the dividends near both ends and a spread between.
#[test]
fn every_16_bit_divisor() {
    let edges = |lo: i64, hi: i64| {
        let mut v: Vec<i64> = (lo..lo + 300).chain(hi - 300..=hi).collect();
        v.extend((lo..=hi).step_by(97));
        v.into_iter().map(|x| x as u64)
    };
    for d in 1..=0xffff {
        check(16, 16, false, d, edges(0, 0xffff));
        check(16, 15, true, ((d as u16) as i16).into(), edges(-0x8000, 0x7fff));
    }
}

#[test]
fn every_16_bit_dividend() {
    for signed in [false, true] {
        for d in divisors(16, signed) {
            check(16, if signed { 15 } else { 16 }, signed, d, 0..65536);
        }
    }
}

#[test]
fn random_32_and_64_bit_dividends() {
    let mut s: u64 = 0x5eed;
    let mut next = move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    for (n, signed) in [(32, false), (32, true), (64, false), (64, true)] {
        let t = IntTy::new(n as u8, signed);
        let mut ds = divisors(n, signed);
        ds.extend(1..2000);
        ds.extend((0..200).map(|_| t.value(t.norm(next())) as i64).filter(|&d| d != 0));
        // the top bits of a random dividend vary too
        ds.extend((0..64).map(|k| t.value(t.norm(next() >> k)) as i64).filter(|&d| d != 0));
        let half = 1u64 << (n - 1);
        let edges = [0, 1, 2, half - 1, half, half + 1, t.norm(u64::MAX - 1), t.norm(u64::MAX)];
        for d in ds {
            let xs: Vec<u64> = edges.iter().copied().chain((0..2000).map(|k| next() >> (k % 64))).collect();
            check(n, if signed { n - 1 } else { n }, signed, d, xs.into_iter());
        }
    }
}

/// Each form occurs: the backends' sequences are chosen by it.
#[test]
fn every_form_occurs() {
    let forms = |n: u32, prec: u32, signed: bool| {
        let mut seen = [false; 8];
        let ds = divisors(if prec < n - 1 { 8 } else { n }, signed);
        for d in ds {
            let k = match div_plan(d, n, prec, signed) {
                DivPlan::Copy { neg } => usize::from(neg),
                DivPlan::Shr(_) | DivPlan::Ge(_) => 2,
                DivPlan::Mul { .. } => 3,
                DivPlan::MulAdd { .. } => 4,
                DivPlan::Pow { .. } => 5,
                DivPlan::SMul { add: false, .. } => 6,
                DivPlan::SMul { add: true, .. } => 7,
            };
            seen[k] = true;
        }
        seen
    };
    for n in [8, 16, 32] {
        let u = forms(n, n, false);
        assert!(u[0] && u[2] && u[3] && u[4], "unsigned {n}: {u:?}");
        let s = forms(n, n - 1, true);
        assert!(s[0] && s[1] && s[5] && s[6] && s[7], "signed {n}: {s:?}");
    }
    // bytes widened to words never need the extra multiplier bit
    let (u, s) = (forms(16, 8, false), forms(16, 7, true));
    assert!(!u[4] && !s[7], "{u:?} {s:?}");
}
