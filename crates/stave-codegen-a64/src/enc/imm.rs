//! Immediate encodings: logical (bitmask) immediates and 8-bit floating-point immediates.

/// Encodes `imm` as a logical immediate for a `width`-bit (32 or 64) instruction: `(N, immr, imms)`,
/// or `None` (all-zero, all-one, and non-repeating-run patterns are not encodable).
pub fn encode_logical(imm: u64, width: u32) -> Option<(u32, u32, u32)> {
    let imm = match width {
        32 => {
            let v = imm & 0xffff_ffff;
            if v != imm && (imm >> 32) != 0 {
                return None;
            }
            v | (v << 32)
        }
        64 => imm,
        _ => return None,
    };
    if imm == 0 || imm == u64::MAX {
        return None;
    }
    let mut size = 64u32;
    while size > 2 {
        let half = size / 2;
        let mask = (1u64 << half) - 1;
        if imm & mask != (imm >> half) & mask {
            break;
        }
        size = half;
    }
    let mask = if size == 64 { u64::MAX } else { (1u64 << size) - 1 };
    let elt = imm & mask;
    let ones = elt.count_ones();
    let pattern = if ones == 64 { u64::MAX } else { (1u64 << ones) - 1 };
    let ror = |v: u64, r: u32| if r == 0 { v } else { ((v >> r) | (v << (size - r))) & mask };
    let r = (0..size).find(|&r| ror(pattern, r) == elt)?;
    let size_bits = match size {
        64 | 32 => 0,
        16 => 0b10_0000,
        8 => 0b11_0000,
        4 => 0b11_1000,
        _ => 0b11_1100,
    };
    let n = u32::from(size == 64);
    Some((n, r, size_bits | (ones - 1)))
}

/// `DecodeBitMasks` (wmask) from the Arm ARM: the value a logical immediate stands for.
pub fn decode_logical(n: u32, immr: u32, imms: u32, width: u32) -> Option<u64> {
    let combined = (n << 6) | (!imms & 0x3f);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 || (width == 32 && n == 1) {
        return None;
    }
    let size = 1u32 << len;
    let levels = size - 1;
    let s = imms & levels;
    let r = immr & levels;
    if s == levels {
        return None;
    }
    let mask = if size == 64 { u64::MAX } else { (1u64 << size) - 1 };
    let welem = (1u64 << (s + 1)) - 1;
    let rot = if r == 0 { welem } else { ((welem >> r) | (welem << (size - r))) & mask };
    let mut v = 0u64;
    let mut i = 0;
    while i < 64 {
        v |= rot << i;
        i += size;
    }
    Some(if width == 32 { v & 0xffff_ffff } else { v })
}

/// Encodes `v` as the 8-bit `FMOV` immediate (`±(16 + m)/16 × 2^e`, m in 0..16, e in -3..=4), or
/// `None` when not exactly representable.
pub fn encode_fp8(v: f64) -> Option<u32> {
    let bits = v.to_bits();
    let sign = (bits >> 63) as u32;
    let exp = ((bits >> 52) & 0x7ff) as u32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    if frac & ((1 << 48) - 1) != 0 {
        return None;
    }
    let b = (exp >> 9) & 1;
    let rep = (exp >> 2) & 0xff;
    if (exp >> 10) & 1 == b || rep != if b == 1 { 0xff } else { 0 } {
        return None;
    }
    let cd = exp & 3;
    Some((sign << 7) | (b << 6) | (cd << 4) | ((frac >> 48) as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known() {
        assert_eq!(encode_logical(0xff, 64), Some((1, 0, 7)));
        assert_eq!(encode_logical(0x5555_5555_5555_5555, 64), Some((0, 0, 0b11_1100)));
        assert_eq!(encode_logical(0xffff_fff0, 32), Some((0, 28, 27)));
        assert_eq!(encode_logical(0, 64), None);
        assert_eq!(encode_logical(u64::MAX, 64), None);
        assert_eq!(encode_logical(0xffff_ffff, 32), None);
        assert_eq!(encode_logical(0x1234, 64), None);
        assert_eq!(encode_fp8(1.0), Some(0x70));
        assert_eq!(encode_fp8(2.0), Some(0x00));
        assert_eq!(encode_fp8(-0.5), Some(0xe0));
        assert_eq!(encode_fp8(31.0), Some(0x3f));
        assert_eq!(encode_fp8(0.0), None);
        assert_eq!(encode_fp8(0.1), None);
        assert_eq!(encode_fp8(32.0), None);
    }
}
