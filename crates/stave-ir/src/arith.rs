//! Scalar semantics shared by the interpreter (and usable by constant folding): integer ops with
//! the trapping / wrapping / saturating modes, IEEE float ops, and the `as`
//! family. Pure functions over raw bits; no panics.
//!
//! Integer values are carried as `u64` *normalized* to their type: truncated to the width, then
//! sign-extended (signed) or zero-extended (unsigned).

use crate::ir::{CastKind, Op, PanicKind};
use crate::types::FloatBits;

/// An integer type: width and signedness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntTy {
    pub bits: u8,
    pub signed: bool,
}

impl IntTy {
    pub const fn new(bits: u8, signed: bool) -> IntTy {
        IntTy { bits, signed }
    }

    fn mask(self) -> u64 {
        if self.bits >= 64 { u64::MAX } else { (1u64 << self.bits) - 1 }
    }

    /// Truncates `v` to the width, then extends by signedness.
    pub fn norm(self, v: u64) -> u64 {
        if self.bits >= 64 {
            return v;
        }
        let t = v & self.mask();
        if self.signed && (t >> (self.bits - 1)) & 1 == 1 { t | !self.mask() } else { t }
    }

    /// The mathematical value of normalized bits.
    pub fn value(self, v: u64) -> i128 {
        let v = self.norm(v);
        if self.signed { v as i64 as i128 } else { v as i128 }
    }

    pub fn min(self) -> i128 {
        if self.signed { -(1i128 << (self.bits - 1)) } else { 0 }
    }

    pub fn max(self) -> i128 {
        if self.signed { (1i128 << (self.bits - 1)) - 1 } else { (1i128 << self.bits) - 1 }
    }

    pub fn fits(self, r: i128) -> bool {
        r >= self.min() && r <= self.max()
    }

    /// `r` modulo 2^bits, normalized.
    pub fn wrap(self, r: i128) -> u64 {
        self.norm(r as u64)
    }

    pub fn clamp(self, r: i128) -> u64 {
        self.wrap(r.clamp(self.min(), self.max()))
    }

    /// Normalized bits of `r`, or `None` if it is out of range.
    pub fn exact(self, r: i128) -> Option<u64> {
        if self.fits(r) { Some(self.wrap(r)) } else { None }
    }
}

/// Integer binary op (`add sub mul div rem` and their wrap / sat forms, `and or xor`).
/// `Err` is the trap kind.
pub fn int_binary(op: Op, t: IntTy, a: u64, b: u64) -> Result<u64, PanicKind> {
    let (x, y) = (t.value(a), t.value(b));
    let trap = |r: Option<i128>| r.and_then(|r| t.exact(r)).ok_or(PanicKind::Overflow);
    let sat = |r: Option<i128>, neg: bool| match r {
        Some(r) => t.clamp(r),
        None => t.clamp(if neg { i128::MIN } else { i128::MAX }),
    };
    match op {
        Op::Add => trap(x.checked_add(y)),
        Op::Sub => trap(x.checked_sub(y)),
        Op::Mul => trap(x.checked_mul(y)),
        Op::AddWrap => Ok(t.norm(a.wrapping_add(b))),
        Op::SubWrap => Ok(t.norm(a.wrapping_sub(b))),
        Op::MulWrap => Ok(t.norm(a.wrapping_mul(b))),
        Op::AddSat => Ok(sat(x.checked_add(y), false)),
        Op::SubSat => Ok(sat(x.checked_sub(y), false)),
        // Overflow of the i128 product happens only for u64 × u64: the true product is positive.
        Op::MulSat => Ok(sat(x.checked_mul(y), (x < 0) != (y < 0))),
        Op::Div | Op::Rem => {
            if y == 0 {
                return Err(PanicKind::DivByZero);
            }
            if t.signed && x == t.min() && y == -1 {
                return Err(PanicKind::Overflow);
            }
            // i128 truncating division / remainder of in-range values cannot overflow.
            let r = if op == Op::Div { x / y } else { x % y };
            Ok(t.wrap(r))
        }
        Op::And => Ok(a & b),
        Op::Or => Ok(a | b),
        Op::Xor => Ok(a ^ b),
        _ => Ok(0),
    }
}

/// `shl shr shl.wrap`. `amount` is the shift amount's value (any int type; negative counts as
/// ≥ width).
pub fn int_shift(op: Op, t: IntTy, a: u64, amount: i128) -> Result<u64, PanicKind> {
    let bits = t.bits as i128;
    let n = match op {
        Op::ShlWrap => (amount & (bits - 1)) as u32,
        _ => {
            if !(0..bits).contains(&amount) {
                return Err(PanicKind::Overflow);
            }
            amount as u32
        }
    };
    Ok(match op {
        Op::Shr => {
            if t.signed {
                t.norm(((t.value(a) as i64) >> n) as u64)
            } else {
                t.norm((a & t.mask()) >> n)
            }
        }
        _ => t.norm(a << n),
    })
}

/// `neg` on a signed int.
pub fn int_neg(t: IntTy, a: u64) -> Result<u64, PanicKind> {
    t.exact(-t.value(a)).ok_or(PanicKind::Overflow)
}

/// `not` on an int (`~`).
pub fn int_not(t: IntTy, a: u64) -> u64 {
    t.norm(!a)
}

/// `popcount clz ctz` within the type's width.
pub fn int_count(op: Op, t: IntTy, a: u64) -> u64 {
    let v = a & t.mask();
    match op {
        Op::Popcount => v.count_ones() as u64,
        Op::Clz => (v.leading_zeros() - (64 - t.bits as u32)) as u64,
        _ => (v.trailing_zeros().min(t.bits as u32)) as u64,
    }
}

/// `eq ne lt le gt ge` on ints of one type.
pub fn int_cmp(op: Op, t: IntTy, a: u64, b: u64) -> bool {
    let (x, y) = (t.value(a), t.value(b));
    match op {
        Op::Eq => x == y,
        Op::Ne => x != y,
        Op::Lt => x < y,
        Op::Le => x <= y,
        Op::Gt => x > y,
        _ => x >= y,
    }
}

/// `mul.wide` on u32 / u64: `(hi, lo)`.
pub fn mul_wide(bits: u8, a: u64, b: u64) -> (u64, u64) {
    if bits == 32 {
        let p = (a & 0xffff_ffff) * (b & 0xffff_ffff);
        (p >> 32, p & 0xffff_ffff)
    } else {
        let p = (a as u128) * (b as u128);
        ((p >> 64) as u64, p as u64)
    }
}

/// IEEE binary op on f32 / f64 bits (round to nearest even, no FMA contraction).
pub fn float_binary(op: Op, f: FloatBits, a: u64, b: u64) -> u64 {
    match f {
        FloatBits::F32 => {
            let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
            let r = match op {
                Op::Add => x + y,
                Op::Sub => x - y,
                Op::Mul => x * y,
                Op::Div => x / y,
                _ => x % y,
            };
            r.to_bits() as u64
        }
        _ => {
            let (x, y) = (f64::from_bits(a), f64::from_bits(b));
            let r = match op {
                Op::Add => x + y,
                Op::Sub => x - y,
                Op::Mul => x * y,
                Op::Div => x / y,
                _ => x % y,
            };
            r.to_bits()
        }
    }
}

/// `neg` on a float: flips the sign bit (also of NaN).
pub fn float_neg(f: FloatBits, a: u64) -> u64 {
    match f {
        FloatBits::F64 => a ^ (1 << 63),
        FloatBits::F32 => a ^ (1 << 31),
        FloatBits::F16 | FloatBits::B16 => a ^ (1 << 15),
    }
}

/// IEEE comparison (unordered: `ne` true, the rest false).
pub fn float_cmp(op: Op, f: FloatBits, a: u64, b: u64) -> bool {
    let (x, y) = (to_f64(f, a), to_f64(f, b));
    match op {
        Op::Eq => x == y,
        Op::Ne => x != y,
        Op::Lt => x < y,
        Op::Le => x <= y,
        Op::Gt => x > y,
        _ => x >= y,
    }
}

/// Exact conversion of float bits to f64.
pub fn to_f64(f: FloatBits, a: u64) -> f64 {
    match f {
        FloatBits::F64 => f64::from_bits(a),
        FloatBits::F32 => f32::from_bits(a as u32) as f64,
        FloatBits::F16 => small_to_f64(a as u16, 5, 10),
        FloatBits::B16 => small_to_f64(a as u16, 8, 7),
    }
}

fn pow2(k: i32) -> f64 {
    // 2^k for -1022 <= k <= 1023, exact.
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// Decodes a 16-bit float (f16: 5/10, bf16: 8/7) exactly.
pub fn small_to_f64(bits: u16, exp_bits: u32, man_bits: u32) -> f64 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let emax = (1u32 << exp_bits) - 1;
    let bias = (emax >> 1) as i32;
    let e = ((bits as u32) >> man_bits) & emax;
    let m = (bits as u32) & ((1 << man_bits) - 1);
    if e == emax {
        return if m == 0 { sign * f64::INFINITY } else { f64::NAN.copysign(sign) };
    }
    let (mant, exp) = if e == 0 {
        (m, 1 - bias - man_bits as i32)
    } else {
        (m | (1 << man_bits), e as i32 - bias - man_bits as i32)
    };
    sign * (mant as f64) * pow2(exp)
}

/// Rounds the exact value `(-1)^neg · m · 2^e` to a binary format with `exp_bits` exponent and
/// `man_bits` fraction bits (≤ 52), round to nearest even, overflow to infinity. Returns the bits.
pub fn round_to_format(neg: bool, m: u64, e: i32, exp_bits: u32, man_bits: u32) -> u64 {
    let sign = (neg as u64) << (exp_bits + man_bits);
    if m == 0 {
        return sign;
    }
    let bias = (1i32 << (exp_bits - 1)) - 1;
    let emax_biased = (1u64 << exp_bits) - 1;
    let msb = 63 - m.leading_zeros() as i32;
    let ue = msb + e; // unbiased exponent of the leading bit
    let q = ue.max(1 - bias) - man_bits as i32; // exponent of the result's last bit
    let shift = q - e;
    let mut r: u64 = if shift <= 0 {
        m << ((-shift) as u32) // fits: ue - q <= man_bits
    } else if shift >= 65 {
        0 // m < 2^64 <= half the quantum
    } else {
        let (keep, rem, half) = if shift == 64 {
            (0, m, 1u64 << 63)
        } else {
            (m >> shift, m & ((1u64 << shift) - 1), 1u64 << (shift - 1))
        };
        if rem > half || (rem == half && keep & 1 == 1) { keep + 1 } else { keep }
    };
    let mut q = q;
    if r >> (man_bits + 1) != 0 {
        r >>= 1;
        q += 1;
    }
    if r < (1u64 << man_bits) {
        return sign | r; // subnormal or zero
    }
    let biased = (q + man_bits as i32 + bias) as i64;
    if biased >= emax_biased as i64 {
        return sign | (emax_biased << man_bits);
    }
    sign | ((biased as u64) << man_bits) | (r - (1u64 << man_bits))
}

/// Rounds an f64 to a 16-bit format (f16: 5/10, bf16: 8/7), RNE; NaN stays a quiet NaN.
pub fn f64_to_small(x: f64, exp_bits: u32, man_bits: u32) -> u16 {
    let bits = x.to_bits();
    let neg = bits >> 63 == 1;
    let sign = (neg as u64) << (exp_bits + man_bits);
    let inf = ((1u64 << exp_bits) - 1) << man_bits;
    if x.is_nan() {
        return (sign | inf | (1 << (man_bits - 1))) as u16;
    }
    if x.is_infinite() {
        return (sign | inf) as u16;
    }
    let e = ((bits >> 52) & 0x7ff) as i32;
    let f = bits & ((1u64 << 52) - 1);
    let (m, ex) = if e == 0 { (f, -1074) } else { (f | (1 << 52), e - 1075) };
    round_to_format(neg, m, ex, exp_bits, man_bits) as u16
}

/// A scalar type class for casts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastTy {
    Int(IntTy),
    Bool,
    Char,
    Float(FloatBits),
    /// Thin pointer or fn value (64-bit address).
    Ptr,
}

impl CastTy {
    fn int_like(self) -> Option<IntTy> {
        match self {
            CastTy::Int(t) => Some(t),
            CastTy::Bool => Some(IntTy::new(1, false)),
            CastTy::Char => Some(IntTy::new(32, false)),
            CastTy::Ptr => Some(IntTy::new(64, false)),
            CastTy::Float(_) => None,
        }
    }
}

/// Mantissa digits of a float format.
fn digits(f: FloatBits) -> u8 {
    match f {
        FloatBits::F16 => 11,
        FloatBits::B16 => 8,
        FloatBits::F32 => 24,
        FloatBits::F64 => 53,
    }
}

fn is_char(v: i128) -> bool {
    (0..=0x10ffff).contains(&v) && !(0xd800..=0xdfff).contains(&v)
}

/// Whether a cast of `kind` from `from` to `to` is defined (the verifier's rule). `Exact` is
/// allowed only when lossless for every value.
pub fn cast_allowed(kind: CastKind, from: CastTy, to: CastTy) -> bool {
    use CastTy as C;
    match kind {
        CastKind::Exact => match (from, to) {
            _ if from == to => true,
            (C::Int(a), C::Int(b)) => {
                (a.signed == b.signed && b.bits >= a.bits) || (!a.signed && b.signed && b.bits > a.bits)
            }
            (C::Bool, C::Int(_)) => true,
            (C::Char, C::Int(b)) => b.bits >= 32 && (!b.signed || b.bits > 32),
            (C::Int(a), C::Char) => !a.signed && a.bits <= 8,
            (C::Int(a), C::Float(f)) => a.bits - a.signed as u8 <= digits(f),
            (C::Float(a), C::Float(b)) => {
                matches!((a, b), (FloatBits::F16 | FloatBits::B16 | FloatBits::F32, FloatBits::F64))
                    || matches!((a, b), (FloatBits::F16 | FloatBits::B16, FloatBits::F32))
            }
            (C::Ptr, C::Ptr) => true,
            (C::Ptr, C::Int(b)) | (C::Int(b), C::Ptr) => b.bits == 64,
            _ => false,
        },
        CastKind::Wrap => matches!((from, to), (C::Int(_) | C::Bool | C::Char | C::Ptr, C::Int(_))),
        CastKind::Sat => matches!((from, to), (C::Int(_) | C::Char | C::Float(_), C::Int(_))),
        CastKind::Round => matches!((from, to), (C::Int(_) | C::Float(_), C::Float(_))),
        CastKind::Try => matches!(
            (from, to),
            (C::Int(_) | C::Char | C::Float(_), C::Int(_))
                | (C::Int(_), C::Char | C::Float(_))
                | (C::Float(_), C::Float(_))
        ),
    }
}

fn int_to_float(v: i128, f: FloatBits) -> u64 {
    // Sources are at most 64 bits wide, so v fits i64 or u64.
    match f {
        FloatBits::F64 => (if v < 0 { v as i64 as f64 } else { v as u64 as f64 }).to_bits(),
        FloatBits::F32 => (if v < 0 { v as i64 as f32 } else { v as u64 as f32 }).to_bits() as u64,
        FloatBits::F16 => round_to_format(v < 0, v.unsigned_abs() as u64, 0, 5, 10),
        FloatBits::B16 => round_to_format(v < 0, v.unsigned_abs() as u64, 0, 8, 7),
    }
}

fn f64_to_float(x: f64, f: FloatBits) -> u64 {
    match f {
        FloatBits::F64 => x.to_bits(),
        FloatBits::F32 => (x as f32).to_bits() as u64,
        FloatBits::F16 => f64_to_small(x, 5, 10) as u64,
        FloatBits::B16 => f64_to_small(x, 8, 7) as u64,
    }
}

/// Float → int, truncating toward zero and saturating; NaN → 0 (`as|`).
fn float_to_int_sat(x: f64, t: IntTy) -> u64 {
    if x.is_nan() {
        return 0;
    }
    let (lo, hi) = (t.min() as f64, t.max() as f64);
    if x <= lo {
        t.wrap(t.min())
    } else if x >= hi {
        t.wrap(t.max())
    } else {
        t.wrap(x.trunc() as i128)
    }
}

/// Converts `v` (bits of type `from`) to `to`. `Ok(None)` only for `Try` when the value does not
/// convert exactly; `Err(())` for a cast [`cast_allowed`] rejects.
#[allow(clippy::result_unit_err)]
pub fn cast(kind: CastKind, from: CastTy, to: CastTy, v: u64) -> Result<Option<u64>, ()> {
    if !cast_allowed(kind, from, to) {
        return Err(());
    }
    Ok(convert(kind, from, to, v))
}

/// [`cast`] without the legality check (any int-like source converts by its value). `None` for
/// a failed `Try` or a meaningless pair (e.g. to `bool`).
pub fn convert(kind: CastKind, from: CastTy, to: CastTy, v: u64) -> Option<u64> {
    if let CastTy::Float(ff) = from {
        let x = to_f64(ff, v);
        return match to {
            CastTy::Int(t) => match kind {
                CastKind::Try => {
                    if x.is_finite() && x.trunc() == x && x.abs() < 1e30 && t.fits(x as i128) {
                        Some(t.wrap(x as i128))
                    } else {
                        None
                    }
                }
                _ => Some(float_to_int_sat(x, t)),
            },
            CastTy::Float(tf) => {
                let r = f64_to_float(x, tf);
                if kind == CastKind::Try && !x.is_nan() && to_f64(tf, r) != x { None } else { Some(r) }
            }
            _ => None,
        };
    }
    let src = from.int_like()?;
    let val = src.value(v);
    match to {
        CastTy::Int(t) => match kind {
            CastKind::Sat => Some(t.clamp(val)),
            CastKind::Try => t.exact(val),
            _ => Some(t.wrap(val)),
        },
        CastTy::Ptr => Some(val as u64),
        CastTy::Char => {
            if is_char(val) {
                Some(val as u64)
            } else {
                None
            }
        }
        CastTy::Float(tf) => {
            let r = int_to_float(val, tf);
            if kind == CastKind::Try {
                let back = to_f64(tf, r);
                if back.is_finite() && back.abs() < 1e30 && back as i128 == val { Some(r) } else { None }
            } else {
                Some(r)
            }
        }
        CastTy::Bool => None,
    }
}
