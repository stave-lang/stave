//! Division of int lanes by a constant through the high half of a product (Granlund and
//! Montgomery, "Division by invariant integers using multiplication"): the plan the backends
//! lower `vec.div_const` with. `t` below is the high half of `x * m` in lanes of `n` bits,
//! signed or unsigned as the lanes ([`crate::arith::mul_hi`]).

use crate::arith::{self, IntTy};

/// How lanes divide by a constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivPlan {
    /// `x`, or `-x` (a signed divisor of -1).
    Copy { neg: bool },
    /// Unsigned, a power of two: `x >> s`.
    Shr(u32),
    /// Unsigned 64-bit lanes, `d > 2^63` (its multiplier would need 129 bits): `x >= d`, 1 or 0.
    Ge(u64),
    /// Unsigned: `t >> s`.
    Mul { m: u64, s: u32 },
    /// Unsigned, a multiplier one bit too wide (`m` its low bits): `(t + ((x - t) >> 1)) >> s`.
    /// `t + ((x - t) >> 1)` is `(x + t) >> 1` without the carry out (`t <= x`).
    MulAdd { m: u64, s: u32 },
    /// Signed, `|d| = 2^l`: `(x + ((x >> (l - 1)) >>> (n - l))) >> l`, negated for `d < 0` (a
    /// negative `x` gains `|d| - 1`, so the shift rounds toward zero).
    Pow { l: u32, neg: bool },
    /// Signed: `((t [+ x]) >> s) - (x >> (n - 1))` (a negative `x` gains 1), negated for `d < 0`;
    /// with `add`, `m` is the multiplier less `2^n`, negative.
    SMul { m: i64, s: u32, add: bool, neg: bool },
}

/// `(m, sh_post)` of the multiplier for a divisor `d` (`1 <= d < 2^n`) of dividends of `prec`
/// bits, with `n`-bit high products (CHOOSE_MULTIPLIER); `m` may have `n + 1` bits. `n + l`
/// stays below 128 (`l` the bits of `d - 1`).
pub fn choose_multiplier(d: u64, n: u32, prec: u32) -> (u128, u32) {
    let l = if d <= 1 { 0 } else { 64 - (d - 1).leading_zeros() };
    let mut s = l;
    let mut lo = (1u128 << (n + l)) / u128::from(d);
    let mut hi = ((1u128 << (n + l)) + (1u128 << (n + l - prec))) / u128::from(d);
    while lo / 2 < hi / 2 && s > 0 {
        lo /= 2;
        hi /= 2;
        s -= 1;
    }
    (hi, s)
}

/// The plan for the divisor `d` (not 0; the lane value, sign-extended when `signed`) of dividends
/// of `prec` bits (the sign not counted) in lanes of `n` bits: `prec` is `n` (`n - 1` signed)
/// for lanes of their own width, less for narrower values widened to `n` bits.
pub fn div_plan(d: i64, n: u32, prec: u32, signed: bool) -> DivPlan {
    if !signed {
        let d = d as u64;
        if d == 1 {
            return DivPlan::Copy { neg: false };
        }
        if d.is_power_of_two() {
            return DivPlan::Shr(d.trailing_zeros());
        }
        if n == 64 && d > 1 << 63 {
            return DivPlan::Ge(d);
        }
        let (m, s) = choose_multiplier(d, n, prec);
        return if m < 1 << n {
            DivPlan::Mul { m: m as u64, s }
        } else {
            DivPlan::MulAdd { m: (m - (1 << n)) as u64, s: s - 1 }
        };
    }
    let (a, neg) = (d.unsigned_abs(), d < 0);
    if a == 1 {
        return DivPlan::Copy { neg };
    }
    if a.is_power_of_two() {
        return DivPlan::Pow { l: a.trailing_zeros(), neg };
    }
    let (m, s) = choose_multiplier(a, n, prec);
    if m < 1 << (n - 1) {
        DivPlan::SMul { m: m as i64, s, add: false, neg }
    } else {
        DivPlan::SMul { m: (m as i128 - (1i128 << n)) as i64, s, add: true, neg }
    }
}

/// The plan's quotient of the lane `x` (normalized bits of an `n`-bit lane), with lane
/// arithmetic as the backends do it: wrapping adds, shifts within the lane.
pub fn apply(plan: DivPlan, x: u64, n: u32, signed: bool) -> u64 {
    let t = IntTy::new(n as u8, signed);
    let u = IntTy::new(n as u8, false);
    let lane = |v: i128| t.wrap(v);
    let sar = |v: u64, k: u32| lane(t.value(v) >> k);
    let shr = |v: u64, k: u32| u.norm(v) >> k;
    let neg = |v: u64, on: bool| if on { lane(-t.value(v)) } else { v };
    match plan {
        DivPlan::Copy { neg: on } => neg(x, on),
        DivPlan::Shr(s) => shr(x, s),
        DivPlan::Ge(d) => u64::from(u.norm(x) >= d),
        DivPlan::Mul { m, s } => shr(arith::mul_hi(u, x, m), s),
        DivPlan::MulAdd { m, s } => {
            let h = arith::mul_hi(u, x, m);
            let half = shr(u.norm(x.wrapping_sub(h)), 1);
            shr(u.norm(h.wrapping_add(half)), s)
        }
        DivPlan::Pow { l, neg: on } => {
            let bias = if l > 1 { shr(sar(x, l - 1), n - l) } else { shr(x, n - 1) };
            neg(sar(lane(t.value(x) + t.value(t.norm(bias))), l), on)
        }
        DivPlan::SMul { m, s, add, neg: on } => {
            let mut h = arith::mul_hi(t, x, t.wrap(i128::from(m)));
            if add {
                h = lane(t.value(h) + t.value(x));
            }
            neg(lane(t.value(sar(h, s)) - t.value(sar(x, n - 1))), on)
        }
    }
}
