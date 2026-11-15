//! Literals: parsing of int, float, char, and string literal text, suffixes,
//! and range checks (E0203).
//!
//! The order-free typing itself (rules 1–4, pass-through contexts) is done by the body checker
//! ([`crate::typeck`]): a literal without context from rules 1–3 is *pending* (its node joins a
//! literal group), and the group is fixed when an operand, an arm, or the enclosing context
//! supplies a type; a group still pending at a context-free root is E0202. The result depends only
//! on the expression tree, never on checking order.
//!
//! The lexer accepts any `idcont*` tail on a number (`5abc`, `0xZZ`); validity is checked here:
//! a malformed literal is E0203 case `malformed` (no code of its own; added as a case).

use crate::types::{IntBits, TypeId, TypeKind, ids};

/// A literal suffix. Its types are the well-known ids of a fresh table ([`crate::types::ids`]);
/// on a canonical table map them with [`crate::types::Interner::fixed_ty`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Suffix {
    None,
    /// An int type suffix (`5u8`).
    Int(TypeId),
    /// A float type suffix (`1.0f32`, also `5f32`).
    Float(TypeId),
}

fn int_suffix(s: &str) -> Option<TypeId> {
    Some(match s {
        "i8" => ids::I8,
        "i16" => ids::I16,
        "i32" => ids::I32,
        "i64" => ids::I64,
        "isize" => ids::ISIZE,
        "u8" => ids::U8,
        "u16" => ids::U16,
        "u32" => ids::U32,
        "u64" => ids::U64,
        "usize" => ids::USIZE,
        _ => return None,
    })
}

fn float_suffix(s: &str) -> Option<TypeId> {
    Some(match s {
        "f32" => ids::F32,
        "f64" => ids::F64,
        "f16" => ids::F16,
        "bf16" => ids::BF16,
        _ => return None,
    })
}

/// Splits `text` into digits and suffix at the first byte that is not a digit of `radix` (or
/// `_`). For radix 16 the suffix must start with `i` or `u`.
fn split_suffix(text: &str, radix: u32) -> (&str, &str) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let digit = c == b'_' || (c as char).is_digit(radix);
        if !digit {
            break;
        }
        i += 1;
    }
    text.split_at(i)
}

/// Parses an `IntLit` token: (magnitude, suffix). `None`: malformed (no digits, bad digits, or
/// an unknown suffix). A magnitude above `u128::MAX` saturates (always out of range).
pub fn parse_int(text: &str) -> Option<(u128, Suffix)> {
    let (radix, body) = if let Some(r) = text.strip_prefix("0x") {
        (16, r)
    } else if let Some(r) = text.strip_prefix("0b") {
        (2, r)
    } else if let Some(r) = text.strip_prefix("0o") {
        (8, r)
    } else {
        (10, text)
    };
    let (digits, suf) = split_suffix(body, radix);
    let mut v: u128 = 0;
    let mut any = false;
    for c in digits.bytes() {
        if c == b'_' {
            continue;
        }
        let d = (c as char).to_digit(radix)? as u128;
        any = true;
        v = v.saturating_mul(radix as u128).saturating_add(d);
    }
    if !any {
        return None;
    }
    let suffix = if suf.is_empty() {
        Suffix::None
    } else if let Some(t) = int_suffix(suf) {
        Suffix::Int(t)
    } else if radix == 10
        && let Some(t) = float_suffix(suf)
    {
        Suffix::Float(t)
    } else {
        return None;
    };
    Some((v, suffix))
}

/// [`parse_int`] for places that need a plain number (array `align`, const args).
pub fn parse_int_text(text: &str) -> Option<(u128, Suffix)> {
    parse_int(text)
}

/// Parses a `FloatLit` token: (value as f64, suffix). An f32/f16 suffix parses the decimal
/// directly at that precision (`value` then holds the exactly widened result).
pub fn parse_float(text: &str) -> Option<(f64, Suffix)> {
    let b = text.as_bytes();
    // number part: digits, '.', digits, exponent
    let mut i = 0;
    while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
            i += 1;
        }
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let save = i;
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let ds = i;
        while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
            i += 1;
        }
        if ds == i {
            i = save;
        }
    }
    let (num, suf) = text.split_at(i);
    let clean: String = num.chars().filter(|&c| c != '_').collect();
    let suffix = if suf.is_empty() { Suffix::None } else { Suffix::Float(float_suffix(suf)?) };
    let v = match suffix {
        Suffix::Float(t) if t == ids::F32 => clean.parse::<f32>().ok()? as f64,
        _ => clean.parse::<f64>().ok()?,
    };
    Some((v, suffix))
}

/// The value range of an int type key: `(min, max)` (any table).
pub fn int_range_kind(k: &TypeKind) -> Option<(i128, i128)> {
    let TypeKind::Int { signed, bits } = *k else { return None };
    Some(match (signed, bits) {
        (true, IntBits::B8) => (i8::MIN as i128, i8::MAX as i128),
        (true, IntBits::B16) => (i16::MIN as i128, i16::MAX as i128),
        (true, IntBits::B32) => (i32::MIN as i128, i32::MAX as i128),
        (true, IntBits::B64 | IntBits::Size) => (i64::MIN as i128, i64::MAX as i128),
        (false, IntBits::B8) => (0, u8::MAX as i128),
        (false, IntBits::B16) => (0, u16::MAX as i128),
        (false, IntBits::B32) => (0, u32::MAX as i128),
        (false, IntBits::B64 | IntBits::Size) => (0, u64::MAX as i128),
    })
}

/// The value range of an int type: `(min, max)`. `t` is a well-known id of a fresh table
/// ([`crate::types::ids`]); see [`int_range_kind`] for any table.
pub fn int_range(t: TypeId) -> Option<(i128, i128)> {
    Some(match t {
        ids::I8 => (i8::MIN as i128, i8::MAX as i128),
        ids::I16 => (i16::MIN as i128, i16::MAX as i128),
        ids::I32 => (i32::MIN as i128, i32::MAX as i128),
        ids::I64 | ids::ISIZE => (i64::MIN as i128, i64::MAX as i128),
        ids::U8 => (0, u8::MAX as i128),
        ids::U16 => (0, u16::MAX as i128),
        ids::U32 => (0, u32::MAX as i128),
        ids::U64 | ids::USIZE => (0, u64::MAX as i128),
        _ => return None,
    })
}

/// Whether the (possibly negated) magnitude fits int type `t`.
pub fn int_fits(mag: u128, neg: bool, t: TypeId) -> bool {
    let Some((lo, hi)) = int_range(t) else { return false };
    if mag > (i128::MAX as u128) {
        return false;
    }
    let v = if neg { -(mag as i128) } else { mag as i128 };
    v >= lo && v <= hi
}

/// Whether a float literal value is finite in float type `t`.
pub fn float_fits(v: f64, t: TypeId) -> bool {
    match t {
        ids::F32 => (v as f32).is_finite(),
        ids::F16 | ids::BF16 => v.is_finite() && v.abs() <= 65504.0,
        _ => v.is_finite(),
    }
}

/// Decodes the escapes of a quoted literal body (`\n \t \r \0 \\ \' \" \xHH \u{H..}`).
fn unescape(body: &str) -> Option<Vec<u8>> {
    let b = body.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'\\' {
            out.push(b[i]);
            i += 1;
            continue;
        }
        let c = *b.get(i + 1)?;
        i += 2;
        match c {
            b'n' => out.push(b'\n'),
            b't' => out.push(b'\t'),
            b'r' => out.push(b'\r'),
            b'0' => out.push(0),
            b'\\' => out.push(b'\\'),
            b'\'' => out.push(b'\''),
            b'"' => out.push(b'"'),
            b'{' => out.push(b'{'),
            b'}' => out.push(b'}'),
            b'x' => {
                let h = body.get(i..i + 2)?;
                out.push(u8::from_str_radix(h, 16).ok()?);
                i += 2;
            }
            b'u' => {
                if b.get(i) != Some(&b'{') {
                    return None;
                }
                let end = body[i..].find('}')? + i;
                let cp = u32::from_str_radix(&body[i + 1..end], 16).ok()?;
                let ch = char::from_u32(cp)?;
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                i = end + 1;
            }
            _ => return None,
        }
    }
    Some(out)
}

/// The bytes of a `StrLit` token (`None`: a bad escape).
pub fn parse_str(text: &str) -> Option<Vec<u8>> {
    let body = text.strip_prefix('"')?.strip_suffix('"')?;
    unescape(body)
}

/// The code point of a `CharLit` token (`None`: not exactly one char, or a bad escape).
pub fn parse_char(text: &str) -> Option<u32> {
    let body = text.strip_prefix('\'')?.strip_suffix('\'')?;
    let bytes = unescape(body)?;
    let s = std::str::from_utf8(&bytes).ok()?;
    let mut it = s.chars();
    let c = it.next()?;
    if it.next().is_some() {
        return None;
    }
    Some(c as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints() {
        assert_eq!(parse_int("1_000"), Some((1000, Suffix::None)));
        assert_eq!(parse_int("0xff"), Some((255, Suffix::None)));
        assert_eq!(parse_int("0x86ci32"), Some((0x86c, Suffix::Int(ids::I32))));
        assert_eq!(parse_int("0b1010u8"), Some((10, Suffix::Int(ids::U8))));
        assert_eq!(parse_int("0o17"), Some((15, Suffix::None)));
        assert_eq!(parse_int("5f32"), Some((5, Suffix::Float(ids::F32))));
        assert_eq!(parse_int("5abc"), None);
        assert_eq!(parse_int("0x"), None);
        assert_eq!(parse_int("0b2"), None);
        assert!(int_fits(128, true, ids::I8));
        assert!(!int_fits(128, false, ids::I8));
        assert!(!int_fits(1, true, ids::U8));
        assert!(int_fits(u64::MAX as u128, false, ids::U64));
    }

    #[test]
    fn floats_chars_strings() {
        assert_eq!(parse_float("1.5e-7"), Some((1.5e-7, Suffix::None)));
        assert_eq!(parse_float("1.0f32"), Some((1.0, Suffix::Float(ids::F32))));
        assert_eq!(parse_float("6_577.0625"), Some((6577.0625, Suffix::None)));
        assert_eq!(parse_float("1.0u8"), None);
        assert!(!float_fits(1e39, ids::F32));
        assert_eq!(parse_char("'a'"), Some(97));
        assert_eq!(parse_char("'\\n'"), Some(10));
        assert_eq!(parse_char("'ab'"), None);
        assert_eq!(parse_char("'é'"), Some(0xe9));
        assert_eq!(parse_str("\"a\\\"b\\n\""), Some(b"a\"b\n".to_vec()));
    }
}
