//! Cases for the differential kernels: every length from 0 to three vectors and one element past
//! them, at several misalignments from a 64-byte boundary (every one with `STAVE_DIFF_FULL=1`),
//! then random lengths. Values are edge-biased: each case draws its elements from one palette
//! (few distinct values, edge values, moderate random values, or raw bits; the float reduction
//! kernels also all -0.0, cancelling sums, overflowing sums and NaNs).
//!
//! The encoding is the one `diff.stv` reads: little endian, u32 case count, then per case u32
//! kind, len, mis, param and the case's arrays, `len` elements each.

/// SplitMix64.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`; the modulo bias does not matter here).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

/// FNV-1a of a name: a per-kernel seed offset, so kernels do not share their cases.
pub fn name_seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// Element type of an input array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F32,
    F64,
    I32,
    U8,
    /// Bytes that are mostly valid UTF-8, with one defect in about half of the cases.
    Utf8,
}

impl Elem {
    pub fn size(self) -> usize {
        match self {
            Elem::F64 => 8,
            Elem::F32 | Elem::I32 => 4,
            Elem::U8 | Elem::Utf8 => 1,
        }
    }
}

/// How a case's `param` is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    None,
    /// A byte value, often one that occurs in the input.
    Byte,
    /// The bits of an f32 threshold (edge values included).
    F32,
    /// An i32 threshold.
    I32,
}

/// One case shape a kernel program understands: its arrays, the widest lane count its kernels
/// use for them, and the palettes its cases draw from.
#[derive(Clone, Copy, Debug)]
pub struct Kind {
    pub arrays: &'static [Elem],
    pub lanes: usize,
    pub param: Param,
    pub palettes: &'static [Palette],
}

#[derive(Clone, Debug)]
pub struct Case {
    pub kind: u32,
    pub len: usize,
    pub mis: usize,
    pub param: u32,
    pub elems: &'static [Elem],
    /// Little-endian element bytes of each array.
    pub arrays: Vec<Vec<u8>>,
}

/// How many cases to make.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub seed: u64,
    /// Every misalignment for every short length instead of two of them.
    pub full: bool,
    /// Random-length cases per kind.
    pub random: usize,
}

const F32_EDGES: [u32; 22] = [
    0x0000_0000, // 0.0
    0x8000_0000, // -0.0
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x3f00_0000, // 0.5
    0xc020_0000, // -2.5
    0x4040_0000, // 3.0
    0x3a83_126f, // 1e-3
    0x7f7f_ffff, // f32::MAX
    0xff7f_ffff, // -f32::MAX
    0x0080_0000, // smallest normal
    0x0000_0001, // smallest subnormal
    0x8000_0001, // -smallest subnormal
    0x007f_ffff, // largest subnormal
    0x7f80_0000, // inf
    0xff80_0000, // -inf
    0x7fc0_0000, // quiet NaN
    0xffc0_0001, // negative NaN with a payload
    0x7f80_0001, // signalling NaN
    0x4b80_0000, // 2^24
    0x7149_f2ca, // 1e30
    0x4b80_0001, // 2^24 + 2
];

const F64_EDGES: [u64; 20] = [
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x3ff0_0000_0000_0000,
    0xbff0_0000_0000_0000,
    0x3fe0_0000_0000_0000,
    0xc004_0000_0000_0000,
    0x4008_0000_0000_0000,
    0x3f50_6249_76f7_ced9, // 1e-3
    0x7fef_ffff_ffff_ffff,
    0xffef_ffff_ffff_ffff,
    0x0010_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x8000_0000_0000_0001,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0000,
    0xfff8_0000_0000_0001,
    0x7ff0_0000_0000_0001,
    0x4340_0000_0000_0000, // 2^53
    0x4340_0000_0000_0001,
];

const I32_EDGES: [i32; 12] =
    [0, 1, -1, 2, -2, i32::MIN, i32::MAX, i32::MIN + 1, i32::MAX - 1, 127, 128, -128];

const U8_EDGES: [u8; 9] = [0, 1, 0x7f, 0x80, 0xff, b'a', b'\n', 0xc0, 0x0f];

/// The value distribution of one case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Palette {
    Few,
    Edge,
    Moderate,
    Bits,
    /// Floats: the first array all -0.0, the others all 1.0 (sums and products of -0.0).
    NegZero,
    /// Floats: either sign, magnitudes 2^-20 to 2^21: sums cancel and round, and their bits
    /// depend on the order of the additions.
    Cancel,
    /// Floats: ±MAX or ±MAX/2 at every eighth element of the first array, ±1 elsewhere: whether
    /// a lane overflows, and whether +inf then meets -inf (NaN), depends on the lane count.
    Huge,
    /// Floats: moderate values and one NaN in eight (quiet, signalling, payloads, either sign).
    Nan,
}

pub const PALETTES: [Palette; 4] = [Palette::Few, Palette::Edge, Palette::Moderate, Palette::Bits];

/// The palettes of the float reduction kernels.
pub const REDUCE_PALETTES: [Palette; 8] = [
    Palette::Few,
    Palette::Edge,
    Palette::Moderate,
    Palette::Bits,
    Palette::NegZero,
    Palette::Cancel,
    Palette::Huge,
    Palette::Nan,
];

/// A finite f32 of moderate size, so sums round but rarely overflow.
fn moderate_f32(r: &mut Rng) -> f32 {
    let m = (r.below(1 << 24) as f32) / (1u32 << 23) as f32;
    let e = r.below(21) as i32 - 10;
    let s = if r.below(2) == 0 { 1.0 } else { -1.0 };
    s * m * 2f32.powi(e)
}

fn moderate_f64(r: &mut Rng) -> f64 {
    let m = (r.below(1 << 53) as f64) / (1u64 << 52) as f64;
    let e = r.below(41) as i32 - 20;
    let s = if r.below(2) == 0 { 1.0 } else { -1.0 };
    s * m * 2f64.powi(e)
}

/// `at`: (array, element) index.
fn f32_bits(r: &mut Rng, p: Palette, param: u32, at: (usize, usize)) -> u32 {
    match p {
        Palette::NegZero if at.0 == 0 => 0x8000_0000,
        Palette::NegZero => 0x3f80_0000,
        Palette::Cancel => {
            let m = (r.below(1 << 24) as f32) / (1u32 << 23) as f32;
            let s = if r.below(2) == 0 { 1.0 } else { -1.0 };
            (s * m * 2f32.powi(r.pick(&[-20, -1, 0, 1, 20]))).to_bits()
        }
        Palette::Huge if at.0 == 0 && at.1.is_multiple_of(8) => {
            r.pick(&[0x7f7f_ffff, 0xff7f_ffff, 0x7eff_ffff, 0xfeff_ffff])
        }
        Palette::Huge => r.pick(&[0x3f80_0000, 0xbf80_0000]),
        Palette::Nan if r.below(8) == 0 => r.pick(&[0x7fc0_0000, 0xffc0_0001, 0x7f80_0001, 0x7fc1_2345]),
        Palette::Nan => moderate_f32(r).to_bits(),
        Palette::Few => r.pick(&[0x3f80_0000, 0xbf80_0000, 0x3f00_0000, 0, 0x8000_0000, param]),
        Palette::Edge if r.below(2) == 0 => r.pick(&F32_EDGES),
        Palette::Edge | Palette::Moderate => {
            if r.below(64) == 0 {
                r.pick(&F32_EDGES)
            } else {
                moderate_f32(r).to_bits()
            }
        }
        Palette::Bits => r.next() as u32,
    }
}

fn f64_bits(r: &mut Rng, p: Palette, param: u64, at: (usize, usize)) -> u64 {
    match p {
        Palette::NegZero if at.0 == 0 => 0x8000_0000_0000_0000,
        Palette::NegZero => 0x3ff0_0000_0000_0000,
        Palette::Cancel => {
            let m = (r.below(1 << 53) as f64) / (1u64 << 52) as f64;
            let s = if r.below(2) == 0 { 1.0 } else { -1.0 };
            (s * m * 2f64.powi(r.pick(&[-20, -1, 0, 1, 20]))).to_bits()
        }
        Palette::Huge if at.0 == 0 && at.1.is_multiple_of(8) => r.pick(&[
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
            0x7fdf_ffff_ffff_ffff,
            0xffdf_ffff_ffff_ffff,
        ]),
        Palette::Huge => r.pick(&[0x3ff0_0000_0000_0000, 0xbff0_0000_0000_0000]),
        Palette::Nan if r.below(8) == 0 => r.pick(&[
            0x7ff8_0000_0000_0000,
            0xfff8_0000_0000_0001,
            0x7ff0_0000_0000_0001,
            0x7ff8_0012_3450_0000,
        ]),
        Palette::Nan => moderate_f64(r).to_bits(),
        Palette::Few => {
            r.pick(&[0x3ff0_0000_0000_0000, 0xbff0_0000_0000_0000, 0, 0x8000_0000_0000_0000, param])
        }
        Palette::Edge if r.below(2) == 0 => r.pick(&F64_EDGES),
        Palette::Edge | Palette::Moderate => {
            if r.below(64) == 0 {
                r.pick(&F64_EDGES)
            } else {
                moderate_f64(r).to_bits()
            }
        }
        Palette::Bits => r.next(),
    }
}

fn i32_value(r: &mut Rng, p: Palette, param: u32) -> i32 {
    match p {
        Palette::Few => r.pick(&[-2, -1, 0, 1, 2, param as i32]),
        Palette::Edge if r.below(2) == 0 => r.pick(&I32_EDGES),
        Palette::Bits => r.next() as i32,
        _ => r.below(2001) as i32 - 1000,
    }
}

fn u8_value(r: &mut Rng, p: Palette, param: u32) -> u8 {
    match p {
        Palette::Few => r.pick(&[param as u8, param as u8 ^ 1, 0]),
        Palette::Edge if r.below(2) == 0 => r.pick(&U8_EDGES),
        Palette::Bits => r.next() as u8,
        _ => {
            if r.below(16) == 0 {
                param as u8
            } else {
                b'a' + r.below(26) as u8
            }
        }
    }
}

/// UTF-8 of exactly `len` bytes: random scalar values of every encoded length, then (in about
/// half of the cases) one defect: a stray continuation byte, a cut sequence, an overlong form, a
/// surrogate, a value past U+10FFFF, or a byte that never occurs in UTF-8.
fn utf8(r: &mut Rng, len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    while v.len() < len {
        let room = len - v.len();
        let c = match r.below(4) {
            0 => char::from_u32(r.below(0x80) as u32),
            1 => char::from_u32(0x80 + r.below(0x780) as u32),
            2 => char::from_u32(0x800 + r.below(0xf800) as u32),
            _ => char::from_u32(0x10000 + r.below(0x10_0000) as u32),
        }
        .unwrap_or('\u{fffd}');
        let mut b = [0u8; 4];
        let s = c.encode_utf8(&mut b);
        if s.len() <= room {
            v.extend_from_slice(s.as_bytes());
        } else {
            v.push(b'a' + r.below(26) as u8);
        }
    }
    if len > 0 && r.below(2) == 0 {
        let at = r.below(len as u64) as usize;
        let bad: &[u8] = match r.below(7) {
            0 => &[0x80],
            1 => &[0xe2, 0x82],
            2 => &[0xc0, 0x80],
            3 => &[0xe0, 0x80, 0x80],
            4 => &[0xed, 0xa0, 0x80],
            5 => &[0xf4, 0x90, 0x80, 0x80],
            _ => &[0xff],
        };
        for (k, &b) in bad.iter().enumerate() {
            if at + k < len {
                v[at + k] = b;
            }
        }
    }
    v
}

fn param(r: &mut Rng, p: Param) -> u32 {
    match p {
        Param::None => 0,
        Param::Byte => {
            if r.below(2) == 0 {
                u32::from(r.pick(&U8_EDGES))
            } else {
                u32::from(b'a' + r.below(26) as u8)
            }
        }
        Param::F32 => {
            if r.below(4) == 0 {
                r.pick(&F32_EDGES)
            } else {
                moderate_f32(r).to_bits()
            }
        }
        Param::I32 => i32_value(r, Palette::Edge, 0) as u32,
    }
}

fn array(r: &mut Rng, e: Elem, p: Palette, len: usize, prm: u32, idx: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len * e.size());
    let prm64 = f64::from(f32::from_bits(prm)).to_bits();
    match e {
        Elem::Utf8 => return utf8(r, len),
        Elem::F32 => (0..len).for_each(|i| v.extend_from_slice(&f32_bits(r, p, prm, (idx, i)).to_le_bytes())),
        Elem::F64 => {
            (0..len).for_each(|i| v.extend_from_slice(&f64_bits(r, p, prm64, (idx, i)).to_le_bytes()))
        }
        Elem::I32 => (0..len).for_each(|_| v.extend_from_slice(&i32_value(r, p, prm).to_le_bytes())),
        Elem::U8 => (0..len).for_each(|_| v.push(u8_value(r, p, prm))),
    }
    v
}

fn case(r: &mut Rng, kind: u32, k: &'static Kind, len: usize, mis: usize) -> Case {
    let prm = param(r, k.param);
    let pal = r.pick(k.palettes);
    let arrays = k.arrays.iter().enumerate().map(|(i, &e)| array(r, e, pal, len, prm, i)).collect();
    Case { kind, len, mis, param: prm, elems: k.arrays, arrays }
}

/// The cases of one kernel program.
pub fn cases(name: &str, kinds: &'static [Kind], plan: Plan) -> Vec<Case> {
    let mut r = Rng::new(plan.seed ^ name_seed(name));
    let mut out = Vec::new();
    for (ki, k) in kinds.iter().enumerate() {
        let slots = 64 / k.arrays[0].size();
        for len in 0..=3 * k.lanes + 1 {
            let mut mis: Vec<usize> =
                if plan.full { (0..slots).collect() } else { vec![0, (len * 5 + 1) % slots] };
            mis.dedup();
            for m in mis {
                out.push(case(&mut r, ki as u32, k, len, m));
            }
        }
        for _ in 0..plan.random {
            let len = r.below((3 * k.lanes + 258) as u64) as usize;
            let mis = r.below(slots as u64) as usize;
            out.push(case(&mut r, ki as u32, k, len, mis));
        }
    }
    out
}

pub fn encode(cases: &[Case]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(cases.len() as u32).to_le_bytes());
    for c in cases {
        for x in [c.kind, c.len as u32, c.mis as u32, c.param] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for a in &c.arrays {
            b.extend_from_slice(a);
        }
    }
    b
}
