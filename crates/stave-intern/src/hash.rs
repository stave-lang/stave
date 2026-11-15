//! xxHash32 and XXH3-128 (xxHash v0.8 algorithms, default secret, seed 0 for XXH3).
//!
//! These are reimplementations of the xxHash algorithms by Yann Collet (BSD 2-Clause,
//! <https://github.com/Cyan4973/xxHash>); the primes and the default secret are the constants
//! defined there.
//!
//! - [`xxh32`]: canonical string ids rank by `(xxh32(bytes, 0), bytes)`
//!   (the `strings` dump of contracts/stage-gates.md prints it as 8 lowercase hex digits).
//! - [`xxh3_128`]: file content hash (`SourceFile.hash`), build id (resident-protocol.md).
//!
//! Both are pure functions of the bytes: identical on every host, in stage0 and stage1. Reads are
//! little-endian regardless of the host. No input panics.

const P32_1: u32 = 0x9E37_79B1;
const P32_2: u32 = 0x85EB_CA77;
const P32_3: u32 = 0xC2B2_AE3D;
const P32_4: u32 = 0x27D4_EB2F;
const P32_5: u32 = 0x1656_67B1;

const P64_1: u64 = 0x9E37_79B1_85EB_CA87;
const P64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P64_3: u64 = 0x1656_67B1_9E37_79F9;
const P64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const P64_5: u64 = 0x27D4_EB2F_1656_67C5;

const PRIME_MX1: u64 = 0x1656_6791_9E37_79F9;
const PRIME_MX2: u64 = 0x9FB2_1C65_1E98_DF25;

/// XXH3 default secret (`XXH3_kSecret`, 192 bytes).
const SECRET: [u8; 192] = [
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c, 0xde,
    0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f, 0xcb, 0x79,
    0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21, 0xb8, 0x08, 0x46,
    0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c, 0x3c, 0x28, 0x52, 0xbb,
    0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3, 0x71, 0x64, 0x48, 0x97, 0xa2,
    0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8, 0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c,
    0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d, 0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59,
    0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64, 0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3,
    0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb, 0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3,
    0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e, 0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8,
    0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce, 0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb,
    0xca, 0xbb, 0x4b, 0x40, 0x7e,
];

const STRIPE_LEN: usize = 64;
const SECRET_CONSUME_RATE: usize = 8;
const ACC_NB: usize = 8;
const MIDSIZE_MAX: usize = 240;
const MIDSIZE_STARTOFFSET: usize = 3;
const MIDSIZE_LASTOFFSET: usize = 17;
const SECRET_SIZE_MIN: usize = 136;
const SECRET_LASTACC_START: usize = 7;
const SECRET_MERGEACCS_START: usize = 11;

/// Little-endian u32 at `i`; 0 for bytes past the end (callers stay in bounds; this keeps the code
/// free of panicking paths).
#[inline]
fn r32(b: &[u8], i: usize) -> u32 {
    let mut a = [0u8; 4];
    if let Some(s) = i.checked_add(4).and_then(|end| b.get(i..end)) {
        a.copy_from_slice(s);
    }
    u32::from_le_bytes(a)
}

#[inline]
fn r64(b: &[u8], i: usize) -> u64 {
    let mut a = [0u8; 8];
    if let Some(s) = i.checked_add(8).and_then(|end| b.get(i..end)) {
        a.copy_from_slice(s);
    }
    u64::from_le_bytes(a)
}

#[inline]
fn byte(b: &[u8], i: usize) -> u8 {
    b.get(i).copied().unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// XXH32

#[inline]
fn round32(acc: u32, input: u32) -> u32 {
    acc.wrapping_add(input.wrapping_mul(P32_2)).rotate_left(13).wrapping_mul(P32_1)
}

/// xxHash32 of `data` with `seed`. Canonical string ids use seed 0.
pub fn xxh32(data: &[u8], seed: u32) -> u32 {
    let mut rest = data;
    let mut h = if data.len() >= 16 {
        let mut v = [
            seed.wrapping_add(P32_1).wrapping_add(P32_2),
            seed.wrapping_add(P32_2),
            seed,
            seed.wrapping_sub(P32_1),
        ];
        let (stripes, tail) = data.as_chunks::<16>();
        for s in stripes {
            for (lane, acc) in v.iter_mut().enumerate() {
                *acc = round32(*acc, r32(s, lane * 4));
            }
        }
        rest = tail;
        v[0].rotate_left(1)
            .wrapping_add(v[1].rotate_left(7))
            .wrapping_add(v[2].rotate_left(12))
            .wrapping_add(v[3].rotate_left(18))
    } else {
        seed.wrapping_add(P32_5)
    };
    // The reference adds the length modulo 2^32.
    h = h.wrapping_add(data.len() as u32);
    let (words, tail) = rest.as_chunks::<4>();
    for w in words {
        h = h.wrapping_add(u32::from_le_bytes(*w).wrapping_mul(P32_3)).rotate_left(17).wrapping_mul(P32_4);
    }
    for &b in tail {
        h = h.wrapping_add(u32::from(b).wrapping_mul(P32_5)).rotate_left(11).wrapping_mul(P32_1);
    }
    h ^= h >> 15;
    h = h.wrapping_mul(P32_2);
    h ^= h >> 13;
    h = h.wrapping_mul(P32_3);
    h ^ (h >> 16)
}

// ---------------------------------------------------------------------------------------------
// XXH3-128

#[inline]
fn mul128(a: u64, b: u64) -> (u64, u64) {
    let p = u128::from(a) * u128::from(b);
    (p as u64, (p >> 64) as u64)
}

#[inline]
fn mul128_fold64(a: u64, b: u64) -> u64 {
    let (lo, hi) = mul128(a, b);
    lo ^ hi
}

#[inline]
fn xxh64_avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(P64_2);
    h ^= h >> 29;
    h = h.wrapping_mul(P64_3);
    h ^ (h >> 32)
}

#[inline]
fn xxh3_avalanche(mut h: u64) -> u64 {
    h ^= h >> 37;
    h = h.wrapping_mul(PRIME_MX1);
    h ^ (h >> 32)
}

/// A 128-bit hash as the reference's `{low64, high64}` pair.
#[derive(Clone, Copy)]
struct H128 {
    lo: u64,
    hi: u64,
}

fn len_1to3(input: &[u8]) -> H128 {
    let len = input.len();
    let c1 = u32::from(byte(input, 0));
    let c2 = u32::from(byte(input, len >> 1));
    let c3 = u32::from(byte(input, len.wrapping_sub(1)));
    let combined_l = (c1 << 16) | (c2 << 24) | c3 | ((len as u32) << 8);
    let combined_h = combined_l.swap_bytes().rotate_left(13);
    let bitflip_l = u64::from(r32(&SECRET, 0) ^ r32(&SECRET, 4));
    let bitflip_h = u64::from(r32(&SECRET, 8) ^ r32(&SECRET, 12));
    H128 {
        lo: xxh64_avalanche(u64::from(combined_l) ^ bitflip_l),
        hi: xxh64_avalanche(u64::from(combined_h) ^ bitflip_h),
    }
}

fn len_4to8(input: &[u8]) -> H128 {
    let len = input.len();
    let input_lo = r32(input, 0);
    let input_hi = r32(input, len - 4);
    let input64 = u64::from(input_lo).wrapping_add(u64::from(input_hi) << 32);
    let bitflip = r64(&SECRET, 16) ^ r64(&SECRET, 24);
    let keyed = input64 ^ bitflip;
    let (mut lo, mut hi) = mul128(keyed, P64_1.wrapping_add((len as u64) << 2));
    hi = hi.wrapping_add(lo << 1);
    lo ^= hi >> 3;
    lo ^= lo >> 35;
    lo = lo.wrapping_mul(PRIME_MX2);
    lo ^= lo >> 28;
    H128 { lo, hi: xxh3_avalanche(hi) }
}

fn len_9to16(input: &[u8]) -> H128 {
    let len = input.len();
    let bitflip_l = r64(&SECRET, 32) ^ r64(&SECRET, 40);
    let bitflip_h = r64(&SECRET, 48) ^ r64(&SECRET, 56);
    let input_lo = r64(input, 0);
    let mut input_hi = r64(input, len - 8);
    let (mut m_lo, mut m_hi) = mul128(input_lo ^ input_hi ^ bitflip_l, P64_1);
    m_lo = m_lo.wrapping_add(((len - 1) as u64) << 54);
    input_hi ^= bitflip_h;
    m_hi = m_hi.wrapping_add(input_hi).wrapping_add(u64::from(input_hi as u32) * u64::from(P32_2 - 1));
    m_lo ^= m_hi.swap_bytes();
    let (h_lo, h_hi) = mul128(m_lo, P64_2);
    let h_hi = h_hi.wrapping_add(m_hi.wrapping_mul(P64_2));
    H128 { lo: xxh3_avalanche(h_lo), hi: xxh3_avalanche(h_hi) }
}

fn len_0to16(input: &[u8]) -> H128 {
    match input.len() {
        0 => H128 {
            lo: xxh64_avalanche(r64(&SECRET, 64) ^ r64(&SECRET, 72)),
            hi: xxh64_avalanche(r64(&SECRET, 80) ^ r64(&SECRET, 88)),
        },
        1..=3 => len_1to3(input),
        4..=8 => len_4to8(input),
        _ => len_9to16(input),
    }
}

#[inline]
fn mix16(input: &[u8], at: usize, secret_at: usize, seed: u64) -> u64 {
    mul128_fold64(
        r64(input, at) ^ r64(&SECRET, secret_at).wrapping_add(seed),
        r64(input, at + 8) ^ r64(&SECRET, secret_at + 8).wrapping_sub(seed),
    )
}

#[inline]
fn mix32(acc: H128, input: &[u8], at1: usize, at2: usize, secret_at: usize, seed: u64) -> H128 {
    let mut lo = acc.lo.wrapping_add(mix16(input, at1, secret_at, seed));
    lo ^= r64(input, at2).wrapping_add(r64(input, at2 + 8));
    let mut hi = acc.hi.wrapping_add(mix16(input, at2, secret_at + 16, seed));
    hi ^= r64(input, at1).wrapping_add(r64(input, at1 + 8));
    H128 { lo, hi }
}

fn finish_mid(acc: H128, len: usize) -> H128 {
    let lo = acc.lo.wrapping_add(acc.hi);
    let hi = acc
        .lo
        .wrapping_mul(P64_1)
        .wrapping_add(acc.hi.wrapping_mul(P64_4))
        .wrapping_add((len as u64).wrapping_mul(P64_2));
    H128 { lo: xxh3_avalanche(lo), hi: 0u64.wrapping_sub(xxh3_avalanche(hi)) }
}

fn len_17to128(input: &[u8]) -> H128 {
    let len = input.len();
    let mut acc = H128 { lo: (len as u64).wrapping_mul(P64_1), hi: 0 };
    if len > 32 {
        if len > 64 {
            if len > 96 {
                acc = mix32(acc, input, 48, len - 64, 96, 0);
            }
            acc = mix32(acc, input, 32, len - 48, 64, 0);
        }
        acc = mix32(acc, input, 16, len - 32, 32, 0);
    }
    acc = mix32(acc, input, 0, len - 16, 0, 0);
    finish_mid(acc, len)
}

fn len_129to240(input: &[u8]) -> H128 {
    let len = input.len();
    let mut acc = H128 { lo: (len as u64).wrapping_mul(P64_1), hi: 0 };
    let mut i = 32;
    while i < 160 {
        acc = mix32(acc, input, i - 32, i - 16, i - 32, 0);
        i += 32;
    }
    acc = H128 { lo: xxh3_avalanche(acc.lo), hi: xxh3_avalanche(acc.hi) };
    // `i <= len` repeats the last 32 bytes when len % 32 == 0, as the reference does.
    let mut i = 160;
    while i <= len {
        acc = mix32(acc, input, i - 32, i - 16, MIDSIZE_STARTOFFSET + i - 160, 0);
        i += 32;
    }
    acc = mix32(acc, input, len - 16, len - 32, SECRET_SIZE_MIN - MIDSIZE_LASTOFFSET - 16, 0);
    finish_mid(acc, len)
}

#[inline]
fn accumulate_512(acc: &mut [u64; ACC_NB], input: &[u8], at: usize, secret_at: usize) {
    for lane in 0..ACC_NB {
        let data_val = r64(input, at + lane * 8);
        let data_key = data_val ^ r64(&SECRET, secret_at + lane * 8);
        acc[lane ^ 1] = acc[lane ^ 1].wrapping_add(data_val);
        acc[lane] = acc[lane].wrapping_add(u64::from(data_key as u32) * (data_key >> 32));
    }
}

#[inline]
fn scramble(acc: &mut [u64; ACC_NB], secret_at: usize) {
    for (lane, a) in acc.iter_mut().enumerate() {
        let key = r64(&SECRET, secret_at + lane * 8);
        let mut v = *a;
        v ^= v >> 47;
        v ^= key;
        *a = v.wrapping_mul(u64::from(P32_1));
    }
}

fn merge_accs(acc: &[u64; ACC_NB], secret_at: usize, start: u64) -> u64 {
    let mut r = start;
    for i in 0..4 {
        r = r.wrapping_add(mul128_fold64(
            acc[2 * i] ^ r64(&SECRET, secret_at + 16 * i),
            acc[2 * i + 1] ^ r64(&SECRET, secret_at + 16 * i + 8),
        ));
    }
    xxh3_avalanche(r)
}

fn hash_long(input: &[u8]) -> H128 {
    let len = input.len();
    let mut acc: [u64; ACC_NB] =
        [u64::from(P32_3), P64_1, P64_2, P64_3, P64_4, u64::from(P32_2), P64_5, u64::from(P32_1)];
    let stripes_per_block = (SECRET.len() - STRIPE_LEN) / SECRET_CONSUME_RATE;
    let block_len = STRIPE_LEN * stripes_per_block;
    let blocks = (len - 1) / block_len;
    for n in 0..blocks {
        for s in 0..stripes_per_block {
            accumulate_512(&mut acc, input, n * block_len + s * STRIPE_LEN, s * SECRET_CONSUME_RATE);
        }
        scramble(&mut acc, SECRET.len() - STRIPE_LEN);
    }
    let stripes = ((len - 1) - block_len * blocks) / STRIPE_LEN;
    for s in 0..stripes {
        accumulate_512(&mut acc, input, blocks * block_len + s * STRIPE_LEN, s * SECRET_CONSUME_RATE);
    }
    accumulate_512(&mut acc, input, len - STRIPE_LEN, SECRET.len() - STRIPE_LEN - SECRET_LASTACC_START);
    H128 {
        lo: merge_accs(&acc, SECRET_MERGEACCS_START, (len as u64).wrapping_mul(P64_1)),
        hi: merge_accs(&acc, SECRET.len() - 64 - SECRET_MERGEACCS_START, !(len as u64).wrapping_mul(P64_2)),
    }
}

/// XXH3-128 of `data` (seed 0, default secret) as `high64 << 64 | low64`. `to_be_bytes()` gives the
/// canonical (big-endian) byte order of the reference's `XXH128_canonical_t`.
pub fn xxh3_128(data: &[u8]) -> u128 {
    let h = match data.len() {
        0..=16 => len_0to16(data),
        17..=128 => len_17to128(data),
        129..=MIDSIZE_MAX => len_129to240(data),
        _ => hash_long(data),
    };
    (u128::from(h.hi) << 64) | u128::from(h.lo)
}

/// XXH3-128 as `[low64, high64]`, the reference's `XXH128_hash_t` field order (the
/// `[u64; 2]` file hash).
pub fn xxh3_128_parts(data: &[u8]) -> [u64; 2] {
    let h = xxh3_128(data);
    [h as u64, (h >> 64) as u64]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The xxHash sanity buffer (`XSUM_fillTestBuffer`, cli/xsum_sanity_check.c, v0.8.2). Its
    /// `PRIME64` is 11400714785074694797 (0x9E3779B185EBCA8D), not `XXH_PRIME64_1`.
    fn sanity_buffer(len: usize) -> Vec<u8> {
        let mut state: u64 = 2_654_435_761;
        (0..len)
            .map(|_| {
                let b = (state >> 56) as u8;
                state = state.wrapping_mul(11_400_714_785_074_694_797);
                b
            })
            .collect()
    }

    #[test]
    fn xxh32_reference_vectors() {
        // XSUM_XXH32_testdata (xsum_sanity_check.c v0.8.2): (len, seed, hash) over the sanity buffer.
        let buf = sanity_buffer(222);
        let cases: [(usize, u32, u32); 8] = [
            (0, 0, 0x02CC_5D05),
            (0, 2_654_435_761, 0x36B7_8AE7),
            (1, 0, 0xCF65_B03E),
            (1, 2_654_435_761, 0xB454_5AA4),
            (14, 0, 0x1208_E7E2),
            (14, 2_654_435_761, 0x6AF1_D1FE),
            (222, 0, 0x5BD1_1DBD),
            (222, 2_654_435_761, 0x5880_3C5F),
        ];
        for (len, seed, want) in cases {
            assert_eq!(xxh32(&buf[..len], seed), want, "len {len} seed {seed:#x}");
        }
    }

    #[test]
    fn xxh32_string_vectors() {
        assert_eq!(xxh32(b"", 0), 0x02CC_5D05);
        assert_eq!(xxh32(b"a", 0), 0x550D_7456);
        assert_eq!(xxh32(b"abc", 0), 0x32D1_53FF);
        assert_eq!(xxh32(b"Nobody inspects the spammish repetition", 0), 0xE229_3B2F);
    }

    #[test]
    fn xxh3_128_reference_vectors() {
        // XSUM_XXH128_testdata, seed 0 rows (xsum_sanity_check.c v0.8.2): (len, low64, high64).
        // Lengths cover every path: 0, 1-3, 4-8, 9-16, 17-128, 129-240, long (one block, two
        // blocks, three blocks, stripe-aligned and overlapping last stripe).
        let buf = sanity_buffer(2367);
        let cases: [(usize, u64, u64); 13] = [
            (0, 0x6001_C324_468D_497F, 0x99AA_06D3_0147_98D8),
            (1, 0xC44B_DFF4_074E_ECDB, 0xA6CD_5E93_9200_0F6A),
            (6, 0x3E70_39BD_DA43_CFC6, 0x082A_FE0B_8162_D12A),
            (12, 0x061A_1927_13F6_9AD9, 0x6E3E_FD8F_C780_2B18),
            (24, 0x1E70_44D2_8B1B_901D, 0x0CE9_66E4_678D_3761),
            (48, 0xF942_219A_ED80_F67B, 0xA002_AC4E_5478_227E),
            (81, 0x5E8B_AFB9_F95F_B803, 0x4952_F581_81AB_0042),
            (222, 0xF1AE_BD59_7CEC_6B3A, 0x337E_0964_1B94_8717),
            (403, 0xCDEB_804D_65C6_DEA4, 0x1B6D_E21E_332D_D73D),
            (512, 0x617E_4959_9013_CB6B, 0x18D2_D110_DCC9_BCA1),
            (2048, 0xDD59_E2C3_A5F0_38E0, 0xF736_557F_D470_73A5),
            (2240, 0x6E73_A905_39CF_2948, 0xCCB1_34FB_FA7C_E49D),
            (2367, 0xCB37_AEB9_E5D3_61ED, 0xE89C_0F6F_F369_B427),
        ];
        for (len, lo, hi) in cases {
            assert_eq!(xxh3_128_parts(&buf[..len]), [lo, hi], "len {len}");
            assert_eq!(xxh3_128(&buf[..len]), (u128::from(hi) << 64) | u128::from(lo));
        }
    }

    #[test]
    fn xxh3_128_canonical_bytes_of_empty() {
        // `xxh128sum` of an empty file prints the canonical big-endian form.
        let want: [u8; 16] =
            [0x99, 0xaa, 0x06, 0xd3, 0x01, 0x47, 0x98, 0xd8, 0x60, 0x01, 0xc3, 0x24, 0x46, 0x8d, 0x49, 0x7f];
        assert_eq!(xxh3_128(b"").to_be_bytes(), want);
    }

    #[test]
    fn every_length_path_is_panic_free_and_input_sensitive() {
        let buf = sanity_buffer(4096);
        let mut prev = None;
        for len in 0..=buf.len() {
            let h = xxh3_128(&buf[..len]);
            assert_ne!(Some(h), prev, "len {len}");
            prev = Some(h);
            let _ = xxh32(&buf[..len], 0);
        }
    }
}
