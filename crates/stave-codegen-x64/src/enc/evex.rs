//! AVX-512: EVEX-encoded vector instructions ([`Asm::evex`]) and the VEX-encoded instructions on
//! the mask registers `k0`..`k7` (`kmov*`, `kand*`, ...).
//!
//! An EVEX instruction is `62 P0 P1 P2 opcode ModRM [SIB] [disp] [imm8]`. Only the registers
//! 0..16 are encoded (`zmm16`..`zmm31` are never used), no embedded broadcast and no embedded
//! rounding. A one-byte displacement is the compressed `disp8 * N`, where `N` follows from the
//! instruction's memory operand ([`Tt`]) and the vector length.
//!
//! Operands are register numbers: which register file a field names (`zmm`, `k`, a general
//! register) follows from the instruction, as in the manual's operand tables. The callers keep
//! them apart by type.

use stave_target::Feature;

use super::inst::RmEnc;
use super::simd::{VLen, Vex};
use super::simd_ops::{Map, Pp};
use super::{Asm, EncError, Gpr, Mem, Xmm};

/// A mask register `k0`..`k7`. `k0` as a write mask means "no mask".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KReg(u8);

impl KReg {
    pub fn new(n: u8) -> Option<KReg> {
        (n < 8).then_some(KReg(n))
    }

    /// `k(n & 7)`: total, for register tables.
    pub const fn masked(n: u8) -> KReg {
        KReg(n & 7)
    }

    pub fn num(self) -> u8 {
        self.0
    }
}

/// The vector length of an EVEX instruction (`L'L`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EVL {
    L128 = 0,
    L256 = 1,
    L512 = 2,
}

impl EVL {
    pub fn bytes(self) -> u32 {
        16 << self as u32
    }

    /// The register length that holds a vector of `size` bytes.
    pub fn of_size(size: u32) -> EVL {
        match size {
            0..=16 => EVL::L128,
            17..=32 => EVL::L256,
            _ => EVL::L512,
        }
    }
}

/// The bytes of an instruction's memory operand, which scale its one-byte displacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Tt {
    /// The whole vector.
    Full,
    /// Half of the vector, a quarter, an eighth (the sources of the extensions).
    Half,
    Quarter,
    Eighth,
    /// This many bytes at every vector length (scalars, 128- and 256-bit tuples).
    Fixed(u8),
}

/// The CPUID feature of an EVEX instruction at 512 bits; the 128- and 256-bit forms add
/// `avx512vl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ef {
    F,
    Bw,
    Dq,
    Cd,
}

impl Ef {
    fn feature(self) -> Feature {
        match self {
            Ef::F => Feature::Avx512f,
            Ef::Bw => Feature::Avx512bw,
            Ef::Dq => Feature::Avx512dq,
            Ef::Cd => Feature::Avx512cd,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EDesc {
    pub pp: Pp,
    pub map: Map,
    pub w: bool,
    pub opc: u8,
    pub tt: Tt,
    pub f: Ef,
}

macro_rules! eops {
    ($(#[$m:meta])* $name:ident { $($v:ident = ($pp:ident, $map:ident, $w:expr, $opc:expr, $tt:expr, $f:ident)),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($v),*
        }

        impl $name {
            /// Every variant.
            pub const ALL: &'static [$name] = &[$($name::$v),*];

            pub(super) fn desc(self) -> EDesc {
                match self {
                    $($name::$v => EDesc {
                        pp: Pp::$pp,
                        map: Map::$map,
                        w: $w != 0,
                        opc: $opc,
                        tt: $tt,
                        f: Ef::$f,
                    }),*
                }
            }
        }
    };
}

use Tt::{Eighth, Fixed, Full, Half, Quarter};

eops! {
    /// The EVEX instructions, named after the mnemonic without its `v`. A trailing `St` is the
    /// store form of a move, `R` the form with a general-register source, `X` the shift by the
    /// count in an `xmm`.
    EOp {
        // moves
        Movdqu8 = (F2, M0F, 0, 0x6f, Full, Bw), Movdqu16 = (F2, M0F, 1, 0x6f, Full, Bw),
        Movdqu32 = (F3, M0F, 0, 0x6f, Full, F), Movdqu64 = (F3, M0F, 1, 0x6f, Full, F),
        Movdqa32 = (P66, M0F, 0, 0x6f, Full, F), Movdqa64 = (P66, M0F, 1, 0x6f, Full, F),
        Movdqu8St = (F2, M0F, 0, 0x7f, Full, Bw), Movdqu16St = (F2, M0F, 1, 0x7f, Full, Bw),
        Movdqu32St = (F3, M0F, 0, 0x7f, Full, F), Movdqu64St = (F3, M0F, 1, 0x7f, Full, F),
        Movdqa32St = (P66, M0F, 0, 0x7f, Full, F), Movdqa64St = (P66, M0F, 1, 0x7f, Full, F),
        Movaps = (N, M0F, 0, 0x28, Full, F), Movapd = (P66, M0F, 1, 0x28, Full, F),
        Movshdup = (F3, M0F, 0, 0x16, Full, F),
        // float arithmetic
        Addps = (N, M0F, 0, 0x58, Full, F), Addpd = (P66, M0F, 1, 0x58, Full, F),
        Mulps = (N, M0F, 0, 0x59, Full, F), Mulpd = (P66, M0F, 1, 0x59, Full, F),
        Subps = (N, M0F, 0, 0x5c, Full, F), Subpd = (P66, M0F, 1, 0x5c, Full, F),
        Minps = (N, M0F, 0, 0x5d, Full, F), Minpd = (P66, M0F, 1, 0x5d, Full, F),
        Divps = (N, M0F, 0, 0x5e, Full, F), Divpd = (P66, M0F, 1, 0x5e, Full, F),
        Maxps = (N, M0F, 0, 0x5f, Full, F), Maxpd = (P66, M0F, 1, 0x5f, Full, F),
        Sqrtps = (N, M0F, 0, 0x51, Full, F), Sqrtpd = (P66, M0F, 1, 0x51, Full, F),
        Andps = (N, M0F, 0, 0x54, Full, Dq), Andpd = (P66, M0F, 1, 0x54, Full, Dq),
        Andnps = (N, M0F, 0, 0x55, Full, Dq), Andnpd = (P66, M0F, 1, 0x55, Full, Dq),
        Orps = (N, M0F, 0, 0x56, Full, Dq), Orpd = (P66, M0F, 1, 0x56, Full, Dq),
        Xorps = (N, M0F, 0, 0x57, Full, Dq), Xorpd = (P66, M0F, 1, 0x57, Full, Dq),
        Rndscaleps = (P66, M3A, 0, 0x08, Full, F), Rndscalepd = (P66, M3A, 1, 0x09, Full, F),
        Fmadd213ps = (P66, M38, 0, 0xa8, Full, F), Fmadd213pd = (P66, M38, 1, 0xa8, Full, F),
        Cmpps = (N, M0F, 0, 0xc2, Full, F), Cmppd = (P66, M0F, 1, 0xc2, Full, F),
        Pternlogd = (P66, M3A, 0, 0x25, Full, F), Pternlogq = (P66, M3A, 1, 0x25, Full, F),
        // integer arithmetic
        Paddb = (P66, M0F, 0, 0xfc, Full, Bw), Paddw = (P66, M0F, 0, 0xfd, Full, Bw),
        Paddd = (P66, M0F, 0, 0xfe, Full, F), Paddq = (P66, M0F, 1, 0xd4, Full, F),
        Psubb = (P66, M0F, 0, 0xf8, Full, Bw), Psubw = (P66, M0F, 0, 0xf9, Full, Bw),
        Psubd = (P66, M0F, 0, 0xfa, Full, F), Psubq = (P66, M0F, 1, 0xfb, Full, F),
        Pmullw = (P66, M0F, 0, 0xd5, Full, Bw), Pmulld = (P66, M38, 0, 0x40, Full, F),
        Pmullq = (P66, M38, 1, 0x40, Full, Dq),
        Pabsb = (P66, M38, 0, 0x1c, Full, Bw), Pabsw = (P66, M38, 0, 0x1d, Full, Bw),
        Pabsd = (P66, M38, 0, 0x1e, Full, F), Pabsq = (P66, M38, 1, 0x1f, Full, F),
        Paddsb = (P66, M0F, 0, 0xec, Full, Bw), Paddsw = (P66, M0F, 0, 0xed, Full, Bw),
        Paddusb = (P66, M0F, 0, 0xdc, Full, Bw), Paddusw = (P66, M0F, 0, 0xdd, Full, Bw),
        Psubsb = (P66, M0F, 0, 0xe8, Full, Bw), Psubsw = (P66, M0F, 0, 0xe9, Full, Bw),
        Psubusb = (P66, M0F, 0, 0xd8, Full, Bw), Psubusw = (P66, M0F, 0, 0xd9, Full, Bw),
        Pminsb = (P66, M38, 0, 0x38, Full, Bw), Pminsw = (P66, M0F, 0, 0xea, Full, Bw),
        Pminsd = (P66, M38, 0, 0x39, Full, F), Pminsq = (P66, M38, 1, 0x39, Full, F),
        Pminub = (P66, M0F, 0, 0xda, Full, Bw), Pminuw = (P66, M38, 0, 0x3a, Full, Bw),
        Pminud = (P66, M38, 0, 0x3b, Full, F), Pminuq = (P66, M38, 1, 0x3b, Full, F),
        Pmaxsb = (P66, M38, 0, 0x3c, Full, Bw), Pmaxsw = (P66, M0F, 0, 0xee, Full, Bw),
        Pmaxsd = (P66, M38, 0, 0x3d, Full, F), Pmaxsq = (P66, M38, 1, 0x3d, Full, F),
        Pmaxub = (P66, M0F, 0, 0xde, Full, Bw), Pmaxuw = (P66, M38, 0, 0x3e, Full, Bw),
        Pmaxud = (P66, M38, 0, 0x3f, Full, F), Pmaxuq = (P66, M38, 1, 0x3f, Full, F),
        Psadbw = (P66, M0F, 0, 0xf6, Full, Bw), Pmaddwd = (P66, M0F, 0, 0xf5, Full, Bw),
        // logic
        Pandd = (P66, M0F, 0, 0xdb, Full, F), Pandq = (P66, M0F, 1, 0xdb, Full, F),
        Pandnd = (P66, M0F, 0, 0xdf, Full, F), Pandnq = (P66, M0F, 1, 0xdf, Full, F),
        Pord = (P66, M0F, 0, 0xeb, Full, F), Porq = (P66, M0F, 1, 0xeb, Full, F),
        Pxord = (P66, M0F, 0, 0xef, Full, F), Pxorq = (P66, M0F, 1, 0xef, Full, F),
        // shifts by the count in an `xmm`
        PsllwX = (P66, M0F, 0, 0xf1, Fixed(16), Bw), PslldX = (P66, M0F, 0, 0xf2, Fixed(16), F),
        PsllqX = (P66, M0F, 1, 0xf3, Fixed(16), F),
        PsrlwX = (P66, M0F, 0, 0xd1, Fixed(16), Bw), PsrldX = (P66, M0F, 0, 0xd2, Fixed(16), F),
        PsrlqX = (P66, M0F, 1, 0xd3, Fixed(16), F),
        PsrawX = (P66, M0F, 0, 0xe1, Fixed(16), Bw), PsradX = (P66, M0F, 0, 0xe2, Fixed(16), F),
        PsraqX = (P66, M0F, 1, 0xe2, Fixed(16), F),
        // shifts by per-lane counts
        Psllvw = (P66, M38, 1, 0x12, Full, Bw), Psllvd = (P66, M38, 0, 0x47, Full, F),
        Psllvq = (P66, M38, 1, 0x47, Full, F),
        Psrlvw = (P66, M38, 1, 0x10, Full, Bw), Psrlvd = (P66, M38, 0, 0x45, Full, F),
        Psrlvq = (P66, M38, 1, 0x45, Full, F),
        Psravw = (P66, M38, 1, 0x11, Full, Bw), Psravd = (P66, M38, 0, 0x46, Full, F),
        Psravq = (P66, M38, 1, 0x46, Full, F),
        // compares into a mask register
        Pcmpeqb = (P66, M0F, 0, 0x74, Full, Bw), Pcmpeqw = (P66, M0F, 0, 0x75, Full, Bw),
        Pcmpeqd = (P66, M0F, 0, 0x76, Full, F), Pcmpeqq = (P66, M38, 1, 0x29, Full, F),
        Pcmpgtb = (P66, M0F, 0, 0x64, Full, Bw), Pcmpgtw = (P66, M0F, 0, 0x65, Full, Bw),
        Pcmpgtd = (P66, M0F, 0, 0x66, Full, F), Pcmpgtq = (P66, M38, 1, 0x37, Full, F),
        Pcmpb = (P66, M3A, 0, 0x3f, Full, Bw), Pcmpw = (P66, M3A, 1, 0x3f, Full, Bw),
        Pcmpd = (P66, M3A, 0, 0x1f, Full, F), Pcmpq = (P66, M3A, 1, 0x1f, Full, F),
        Pcmpub = (P66, M3A, 0, 0x3e, Full, Bw), Pcmpuw = (P66, M3A, 1, 0x3e, Full, Bw),
        Pcmpud = (P66, M3A, 0, 0x1e, Full, F), Pcmpuq = (P66, M3A, 1, 0x1e, Full, F),
        Ptestmb = (P66, M38, 0, 0x26, Full, Bw), Ptestmw = (P66, M38, 1, 0x26, Full, Bw),
        Ptestmd = (P66, M38, 0, 0x27, Full, F), Ptestmq = (P66, M38, 1, 0x27, Full, F),
        // blends by a mask register
        Pblendmb = (P66, M38, 0, 0x66, Full, Bw), Pblendmw = (P66, M38, 1, 0x66, Full, Bw),
        Pblendmd = (P66, M38, 0, 0x64, Full, F), Pblendmq = (P66, M38, 1, 0x64, Full, F),
        // shuffles and permutes (`Shufi64x2` has no 128-bit form)
        Pshufb = (P66, M38, 0, 0x00, Full, Bw),
        Shufi64x2 = (P66, M3A, 1, 0x43, Full, F), Alignq = (P66, M3A, 1, 0x03, Full, F),
        Permi2w = (P66, M38, 1, 0x75, Full, Bw), Permi2d = (P66, M38, 0, 0x76, Full, F),
        Permi2q = (P66, M38, 1, 0x76, Full, F),
        // high halves and widening products, packs
        Pmulhw = (P66, M0F, 0, 0xe5, Full, Bw), Pmulhuw = (P66, M0F, 0, 0xe4, Full, Bw),
        Pmuldq = (P66, M38, 1, 0x28, Full, F), Pmuludq = (P66, M0F, 1, 0xf4, Full, F),
        Pmaddubsw = (P66, M38, 0, 0x04, Full, Bw), Packuswb = (P66, M0F, 0, 0x67, Full, Bw),
        Plzcntd = (P66, M38, 0, 0x44, Full, Cd), Plzcntq = (P66, M38, 1, 0x44, Full, Cd),
        // compress (`reg` the source, `rm` the destination) and expand, one element per lane
        Pcompressd = (P66, M38, 0, 0x8b, Fixed(4), F), Pcompressq = (P66, M38, 1, 0x8b, Fixed(8), F),
        Pexpandd = (P66, M38, 0, 0x89, Fixed(4), F), Pexpandq = (P66, M38, 1, 0x89, Fixed(8), F),
        // non-temporal stores
        MovntpsSt = (N, M0F, 0, 0x2b, Full, F), MovntpdSt = (P66, M0F, 1, 0x2b, Full, F),
        MovntdqSt = (P66, M0F, 0, 0xe7, Full, F),
        // mask register to and from a vector of all-ones lanes
        Pmovm2b = (F3, M38, 0, 0x28, Full, Bw), Pmovm2w = (F3, M38, 1, 0x28, Full, Bw),
        Pmovm2d = (F3, M38, 0, 0x38, Full, Dq), Pmovm2q = (F3, M38, 1, 0x38, Full, Dq),
        Pmovb2m = (F3, M38, 0, 0x29, Full, Bw), Pmovw2m = (F3, M38, 1, 0x29, Full, Bw),
        Pmovd2m = (F3, M38, 0, 0x39, Full, Dq), Pmovq2m = (F3, M38, 1, 0x39, Full, Dq),
        // broadcasts of the low lane of an `xmm`, of memory, of a general register
        Pbroadcastb = (P66, M38, 0, 0x78, Fixed(1), Bw), Pbroadcastw = (P66, M38, 0, 0x79, Fixed(2), Bw),
        Pbroadcastd = (P66, M38, 0, 0x58, Fixed(4), F), Pbroadcastq = (P66, M38, 1, 0x59, Fixed(8), F),
        PbroadcastbR = (P66, M38, 0, 0x7a, Full, Bw), PbroadcastwR = (P66, M38, 0, 0x7b, Full, Bw),
        PbroadcastdR = (P66, M38, 0, 0x7c, Full, F), PbroadcastqR = (P66, M38, 1, 0x7c, Full, F),
        Broadcastss = (P66, M38, 0, 0x18, Fixed(4), F), Broadcastsd = (P66, M38, 1, 0x19, Fixed(8), F),
        // extensions (the source is a narrower register)
        Pmovsxbw = (P66, M38, 0, 0x20, Half, Bw), Pmovsxbd = (P66, M38, 0, 0x21, Quarter, F),
        Pmovsxbq = (P66, M38, 0, 0x22, Eighth, F), Pmovsxwd = (P66, M38, 0, 0x23, Half, F),
        Pmovsxwq = (P66, M38, 0, 0x24, Quarter, F), Pmovsxdq = (P66, M38, 0, 0x25, Half, F),
        Pmovzxbw = (P66, M38, 0, 0x30, Half, Bw), Pmovzxbd = (P66, M38, 0, 0x31, Quarter, F),
        Pmovzxbq = (P66, M38, 0, 0x32, Eighth, F), Pmovzxwd = (P66, M38, 0, 0x33, Half, F),
        Pmovzxwq = (P66, M38, 0, 0x34, Quarter, F), Pmovzxdq = (P66, M38, 0, 0x35, Half, F),
        // down-conversions, truncating, signed- and unsigned-saturating: `reg` is the wide
        // source, `rm` the narrow destination or memory
        Pmovwb = (F3, M38, 0, 0x30, Half, Bw), Pmovswb = (F3, M38, 0, 0x20, Half, Bw),
        Pmovuswb = (F3, M38, 0, 0x10, Half, Bw),
        Pmovdb = (F3, M38, 0, 0x31, Quarter, F), Pmovsdb = (F3, M38, 0, 0x21, Quarter, F),
        Pmovusdb = (F3, M38, 0, 0x11, Quarter, F),
        Pmovdw = (F3, M38, 0, 0x33, Half, F), Pmovsdw = (F3, M38, 0, 0x23, Half, F),
        Pmovusdw = (F3, M38, 0, 0x13, Half, F),
        Pmovqb = (F3, M38, 0, 0x32, Eighth, F), Pmovsqb = (F3, M38, 0, 0x22, Eighth, F),
        Pmovusqb = (F3, M38, 0, 0x12, Eighth, F),
        Pmovqw = (F3, M38, 0, 0x34, Quarter, F), Pmovsqw = (F3, M38, 0, 0x24, Quarter, F),
        Pmovusqw = (F3, M38, 0, 0x14, Quarter, F),
        Pmovqd = (F3, M38, 0, 0x35, Half, F), Pmovsqd = (F3, M38, 0, 0x25, Half, F),
        Pmovusqd = (F3, M38, 0, 0x15, Half, F),
        // int <-> float conversions (`cvtt*`: truncating); the source of a `Half` one is half
        // the destination, the destination of `cvtpd2ps`, `cvt?qq2ps`, `cvttpd2?dq` half the
        // source
        Cvtdq2ps = (N, M0F, 0, 0x5b, Full, F), Cvtqq2ps = (N, M0F, 1, 0x5b, Full, Dq),
        Cvtudq2ps = (F2, M0F, 0, 0x7a, Full, F), Cvtuqq2ps = (F2, M0F, 1, 0x7a, Full, Dq),
        Cvtdq2pd = (F3, M0F, 0, 0xe6, Half, F), Cvtqq2pd = (F3, M0F, 1, 0xe6, Full, Dq),
        Cvtudq2pd = (F3, M0F, 0, 0x7a, Half, F), Cvtuqq2pd = (F3, M0F, 1, 0x7a, Full, Dq),
        Cvtps2pd = (N, M0F, 0, 0x5a, Half, F), Cvtpd2ps = (P66, M0F, 1, 0x5a, Full, F),
        Cvttps2dq = (F3, M0F, 0, 0x5b, Full, F), Cvttpd2dq = (P66, M0F, 1, 0xe6, Full, F),
        Cvttps2udq = (N, M0F, 0, 0x78, Full, F), Cvttpd2udq = (N, M0F, 1, 0x78, Full, F),
        Cvttps2qq = (P66, M0F, 0, 0x7a, Half, Dq), Cvttpd2qq = (P66, M0F, 1, 0x7a, Full, Dq),
        Cvttps2uqq = (P66, M0F, 0, 0x78, Half, Dq), Cvttpd2uqq = (P66, M0F, 1, 0x78, Full, Dq),
        // packs, two-source permutes (`vvvv` the indices, `reg` the first table and the
        // result), `vpermq` by an index vector (`vvvv`; 256 and 512 bits only), unpacks,
        // `vpshufd`
        Packsswb = (P66, M0F, 0, 0x63, Full, Bw), Packssdw = (P66, M0F, 0, 0x6b, Full, Bw),
        Permt2w = (P66, M38, 1, 0x7d, Full, Bw), Permt2d = (P66, M38, 0, 0x7e, Full, F),
        Permq = (P66, M38, 1, 0x36, Full, F),
        Punpcklwd = (P66, M0F, 0, 0x61, Full, Bw), Punpckldq = (P66, M0F, 0, 0x62, Full, F),
        Punpcklqdq = (P66, M0F, 1, 0x6c, Full, F),
        Pshufd = (P66, M0F, 0, 0x70, Full, F),
        // halves and quarters: `reg` is the wide register, `rm` the narrow one or memory
        Extracti64x4 = (P66, M3A, 1, 0x3b, Fixed(32), F), Extractf64x4 = (P66, M3A, 1, 0x1b, Fixed(32), F),
        Extracti32x4 = (P66, M3A, 0, 0x39, Fixed(16), F), Extractf32x4 = (P66, M3A, 0, 0x19, Fixed(16), F),
        Inserti64x4 = (P66, M3A, 1, 0x3a, Fixed(32), F), Insertf64x4 = (P66, M3A, 1, 0x1a, Fixed(32), F),
        Inserti32x4 = (P66, M3A, 0, 0x38, Fixed(16), F), Insertf32x4 = (P66, M3A, 0, 0x18, Fixed(16), F),
    }
}

/// Packed shifts by an immediate: `vop dst, src, imm8` with the destination in `vvvv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EShiftI {
    Psllw,
    Pslld,
    Psllq,
    Psrlw,
    Psrld,
    Psrlq,
    Psraw,
    Psrad,
    Psraq,
    /// Byte shifts within each 128-bit lane.
    Pslldq,
    Psrldq,
}

impl EShiftI {
    pub const ALL: &'static [EShiftI] = &[
        EShiftI::Psllw,
        EShiftI::Pslld,
        EShiftI::Psllq,
        EShiftI::Psrlw,
        EShiftI::Psrld,
        EShiftI::Psrlq,
        EShiftI::Psraw,
        EShiftI::Psrad,
        EShiftI::Psraq,
        EShiftI::Pslldq,
        EShiftI::Psrldq,
    ];

    /// `(opcode, /digit, W, feature)`.
    fn desc(self) -> (u8, u8, bool, Ef) {
        match self {
            EShiftI::Psllw => (0x71, 6, false, Ef::Bw),
            EShiftI::Pslld => (0x72, 6, false, Ef::F),
            EShiftI::Psllq => (0x73, 6, true, Ef::F),
            EShiftI::Psrlw => (0x71, 2, false, Ef::Bw),
            EShiftI::Psrld => (0x72, 2, false, Ef::F),
            EShiftI::Psrlq => (0x73, 2, true, Ef::F),
            EShiftI::Psraw => (0x71, 4, false, Ef::Bw),
            EShiftI::Psrad => (0x72, 4, false, Ef::F),
            EShiftI::Psraq => (0x72, 4, true, Ef::F),
            EShiftI::Pslldq => (0x73, 7, false, Ef::Bw),
            EShiftI::Psrldq => (0x73, 3, false, Ef::Bw),
        }
    }
}

/// The r/m operand of an EVEX instruction: a register number or memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ERm {
    R(u8),
    M(Mem),
}

impl From<Xmm> for ERm {
    fn from(x: Xmm) -> ERm {
        ERm::R(x.num())
    }
}

impl From<KReg> for ERm {
    fn from(k: KReg) -> ERm {
        ERm::R(k.num())
    }
}

impl From<Gpr> for ERm {
    fn from(g: Gpr) -> ERm {
        ERm::R(g.num())
    }
}

impl From<Mem> for ERm {
    fn from(m: Mem) -> ERm {
        ERm::M(m)
    }
}

/// One EVEX instruction: `op reg {k}{z}, vvvv, rm [, imm]`. `vvvv` is 0 for an instruction
/// without that operand; `k` 0 is no write mask.
#[derive(Clone, Copy, Debug)]
pub struct Ev {
    pub op: EOp,
    pub vl: EVL,
    pub reg: u8,
    pub vvvv: u8,
    pub rm: ERm,
    pub k: KReg,
    /// Zeroing instead of merging (with a write mask only).
    pub z: bool,
    pub imm: Option<u8>,
}

impl Ev {
    /// `op reg, vvvv, rm` without mask or immediate.
    pub fn new(op: EOp, vl: EVL, reg: u8, vvvv: u8, rm: impl Into<ERm>) -> Ev {
        Ev { op, vl, reg, vvvv, rm: rm.into(), k: KReg(0), z: false, imm: None }
    }

    pub fn mask(mut self, k: KReg, z: bool) -> Ev {
        self.k = k;
        self.z = z;
        self
    }

    pub fn imm(mut self, i: u8) -> Ev {
        self.imm = Some(i);
        self
    }
}

/// The operand width of a mask-register instruction (`kmovb` .. `kmovq`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KSize {
    B,
    W,
    D,
    Q,
}

impl KSize {
    pub const ALL: [KSize; 4] = [KSize::B, KSize::W, KSize::D, KSize::Q];

    /// The narrowest width that holds `n` mask bits.
    pub fn of_lanes(n: u32) -> KSize {
        match n {
            0..=8 => KSize::B,
            9..=16 => KSize::W,
            17..=32 => KSize::D,
            _ => KSize::Q,
        }
    }

    /// `(pp, W)` of the forms between mask registers and memory.
    fn kk(self) -> (Pp, bool) {
        match self {
            KSize::W => (Pp::N, false),
            KSize::B => (Pp::P66, false),
            KSize::Q => (Pp::N, true),
            KSize::D => (Pp::P66, true),
        }
    }

    /// `(pp, W)` of the moves to and from a general register.
    fn kg(self) -> (Pp, bool) {
        match self {
            KSize::W => (Pp::N, false),
            KSize::B => (Pp::P66, false),
            KSize::D => (Pp::F2, false),
            KSize::Q => (Pp::F2, true),
        }
    }

    /// The CPUID feature of the plain instructions of this width.
    fn feature(self) -> Feature {
        match self {
            KSize::W => Feature::Avx512f,
            KSize::B => Feature::Avx512dq,
            KSize::D | KSize::Q => Feature::Avx512bw,
        }
    }
}

/// `kop dst, a, b` on mask registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KBin {
    And = 0x41,
    Andn = 0x42,
    Or = 0x45,
    Xnor = 0x46,
    Xor = 0x47,
}

impl KBin {
    pub const ALL: [KBin; 5] = [KBin::And, KBin::Andn, KBin::Or, KBin::Xnor, KBin::Xor];
}

impl Asm {
    /// Encodes one EVEX instruction.
    pub fn evex(&mut self, e: Ev) -> Result<(), EncError> {
        let d = e.op.desc();
        if e.reg > 15 || e.vvvv > 15 {
            return Err(EncError::Invalid("EVEX register above 15"));
        }
        if e.z && e.k.0 == 0 {
            return Err(EncError::Invalid("zeroing without a write mask"));
        }
        let rm = match e.rm {
            ERm::R(r) if r > 15 => return Err(EncError::Invalid("EVEX register above 15")),
            ERm::R(r) => RmEnc::Reg(r),
            ERm::M(m) => RmEnc::Mem(m),
        };
        self.evex_raw(d.pp, d.map, d.w, d.opc, e.vl, e.reg, e.vvvv, rm, e.k, e.z, d.tt, e.imm)?;
        self.note(d.f.feature());
        if e.vl != EVL::L512 {
            self.note(Feature::Avx512vl);
        }
        Ok(())
    }

    /// `vop dst {k}{z}, src, imm8`: a packed shift by an immediate.
    #[allow(clippy::too_many_arguments)]
    pub fn evex_shift_i(
        &mut self,
        op: EShiftI,
        vl: EVL,
        dst: u8,
        src: impl Into<ERm>,
        imm: u8,
        k: KReg,
        z: bool,
    ) -> Result<(), EncError> {
        let (opc, digit, w, f) = op.desc();
        if dst > 15 {
            return Err(EncError::Invalid("EVEX register above 15"));
        }
        let rm = match src.into() {
            ERm::R(r) if r > 15 => return Err(EncError::Invalid("EVEX register above 15")),
            ERm::R(r) => RmEnc::Reg(r),
            ERm::M(m) => RmEnc::Mem(m),
        };
        self.evex_raw(Pp::P66, Map::M0F, w, opc, vl, digit, dst, rm, k, z, Tt::Full, Some(imm))?;
        self.note(f.feature());
        if vl != EVL::L512 {
            self.note(Feature::Avx512vl);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn evex_raw(
        &mut self,
        pp: Pp,
        map: Map,
        w: bool,
        opc: u8,
        vl: EVL,
        reg: u8,
        vvvv: u8,
        rm: RmEnc,
        k: KReg,
        z: bool,
        tt: Tt,
        imm: Option<u8>,
    ) -> Result<(), EncError> {
        let (x, b) = self.rm_xb(&rm)?;
        let r = (reg >> 3) & 1;
        // R, X, B, R' and V' are stored inverted; R' and V' select registers 16..32.
        let p0 = ((r ^ 1) << 7) | ((x ^ 1) << 6) | ((b ^ 1) << 5) | 0x10 | map as u8;
        let p1 = (u8::from(w) << 7) | ((!vvvv & 15) << 3) | 4 | pp as u8;
        let p2 = (u8::from(z) << 7) | ((vl as u8) << 5) | 8 | k.0;
        self.buf.extend_from_slice(&[0x62, p0, p1, p2, opc]);
        let n = match tt {
            Tt::Full => vl.bytes(),
            Tt::Half => vl.bytes() / 2,
            Tt::Quarter => vl.bytes() / 4,
            Tt::Eighth => vl.bytes() / 8,
            Tt::Fixed(n) => u32::from(n),
        };
        self.emit_modrm_scaled(reg, rm, u8::from(imm.is_some()), n as i32);
        if let Some(i) = imm {
            self.buf.push(i);
        }
        Ok(())
    }

    // ---- mask registers (VEX-encoded) ----

    #[allow(clippy::too_many_arguments)]
    fn kvex(
        &mut self,
        pp: Pp,
        map: Map,
        w: bool,
        l1: bool,
        opc: u8,
        reg: u8,
        vvvv: u8,
        rm: RmEnc,
        imm: Option<u8>,
    ) -> Result<(), EncError> {
        let l = if l1 { VLen::X256 } else { VLen::X128 };
        let trailing = u8::from(imm.is_some());
        self.vex(Vex { pp, map, w, l, opc, reg, vvvv, rm, trailing })?;
        if let Some(i) = imm {
            self.buf.push(i);
        }
        Ok(())
    }

    /// `kmov dst, src` between mask registers.
    pub fn kmov_kk(&mut self, s: KSize, dst: KReg, src: KReg) {
        let (pp, w) = s.kk();
        self.note(s.feature());
        let _ = self.kvex(pp, Map::M0F, w, false, 0x90, dst.0, 0, RmEnc::Reg(src.0), None);
    }

    /// `kmov dst, [m]`: the low bits of the width from memory, zeros above.
    pub fn kmov_km(&mut self, s: KSize, dst: KReg, m: Mem) -> Result<(), EncError> {
        let (pp, w) = s.kk();
        self.note(s.feature());
        self.kvex(pp, Map::M0F, w, false, 0x90, dst.0, 0, RmEnc::Mem(m), None)
    }

    /// `kmov [m], src`.
    pub fn kmov_mk(&mut self, s: KSize, m: Mem, src: KReg) -> Result<(), EncError> {
        let (pp, w) = s.kk();
        self.note(s.feature());
        self.kvex(pp, Map::M0F, w, false, 0x91, src.0, 0, RmEnc::Mem(m), None)
    }

    /// `kmov dst, src` from a general register (its low 32 bits unless `Q`).
    pub fn kmov_kg(&mut self, s: KSize, dst: KReg, src: Gpr) {
        let (pp, w) = s.kg();
        self.note(s.feature());
        let _ = self.kvex(pp, Map::M0F, w, false, 0x92, dst.0, 0, RmEnc::Reg(src.num()), None);
    }

    /// `kmov dst, src` into a general register, zero-extended.
    pub fn kmov_gk(&mut self, s: KSize, dst: Gpr, src: KReg) {
        let (pp, w) = s.kg();
        self.note(s.feature());
        let _ = self.kvex(pp, Map::M0F, w, false, 0x93, dst.num(), 0, RmEnc::Reg(src.0), None);
    }

    /// `kop dst, a, b` (`kandn`: `!a & b`).
    pub fn kbin(&mut self, op: KBin, s: KSize, dst: KReg, a: KReg, b: KReg) {
        let (pp, w) = s.kk();
        self.note(s.feature());
        let _ = self.kvex(pp, Map::M0F, w, true, op as u8, dst.0, a.0, RmEnc::Reg(b.0), None);
    }

    /// `knot dst, src`.
    pub fn knot(&mut self, s: KSize, dst: KReg, src: KReg) {
        let (pp, w) = s.kk();
        self.note(s.feature());
        let _ = self.kvex(pp, Map::M0F, w, false, 0x44, dst.0, 0, RmEnc::Reg(src.0), None);
    }

    /// `kortest a, b`: ZF = the OR is zero, CF = the OR is all ones (of the width).
    pub fn kortest(&mut self, s: KSize, a: KReg, b: KReg) {
        let (pp, w) = s.kk();
        self.note(s.feature());
        let _ = self.kvex(pp, Map::M0F, w, false, 0x98, a.0, 0, RmEnc::Reg(b.0), None);
    }

    /// `kshiftl dst, src, imm` / `kshiftr`: a shift within the width, zeros above it.
    pub fn kshift(&mut self, left: bool, s: KSize, dst: KReg, src: KReg, imm: u8) {
        let (w, opc) = match (left, s) {
            (true, KSize::B) => (false, 0x32),
            (true, KSize::W) => (true, 0x32),
            (true, KSize::D) => (false, 0x33),
            (true, KSize::Q) => (true, 0x33),
            (false, KSize::B) => (false, 0x30),
            (false, KSize::W) => (true, 0x30),
            (false, KSize::D) => (false, 0x31),
            (false, KSize::Q) => (true, 0x31),
        };
        self.note(s.feature());
        let _ = self.kvex(Pp::P66, Map::M3A, w, false, opc, dst.0, 0, RmEnc::Reg(src.0), Some(imm));
    }
}

/// Gathers and scatters: EVEX with a vector-indexed memory operand (VSIB), under a write mask
/// that the instruction clears lane by lane as it completes. Named as the mnemonic: data then
/// index lane size, `Pgatherqd` has quadword indices and doubleword data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EVsib {
    Pgatherdd,
    Pgatherdq,
    Pgatherqd,
    Pgatherqq,
    Pscatterdd,
    Pscatterdq,
    Pscatterqd,
    Pscatterqq,
}

impl EVsib {
    pub const ALL: [EVsib; 8] = [
        EVsib::Pgatherdd,
        EVsib::Pgatherdq,
        EVsib::Pgatherqd,
        EVsib::Pgatherqq,
        EVsib::Pscatterdd,
        EVsib::Pscatterdq,
        EVsib::Pscatterqd,
        EVsib::Pscatterqq,
    ];

    /// `(opcode, W)`.
    fn desc(self) -> (u8, bool) {
        match self {
            EVsib::Pgatherdd => (0x90, false),
            EVsib::Pgatherdq => (0x90, true),
            EVsib::Pgatherqd => (0x91, false),
            EVsib::Pgatherqq => (0x91, true),
            EVsib::Pscatterdd => (0xa0, false),
            EVsib::Pscatterdq => (0xa0, true),
            EVsib::Pscatterqd => (0xa1, false),
            EVsib::Pscatterqq => (0xa1, true),
        }
    }

    pub fn scatter(self) -> bool {
        matches!(self, EVsib::Pscatterdd | EVsib::Pscatterdq | EVsib::Pscatterqd | EVsib::Pscatterqq)
    }
}

impl Asm {
    /// `vpgather* data {k}, [base + index * scale + disp]` / `vpscatter* [..] {k}, data`. `vl`
    /// is the length of the wider of the data and the index register; `k` is not `k0`, and a
    /// gather's data register is not its index register.
    #[allow(clippy::too_many_arguments)]
    pub fn evex_vsib(
        &mut self,
        op: EVsib,
        vl: EVL,
        data: Xmm,
        base: Gpr,
        index: Xmm,
        scale: super::Scale,
        disp: i32,
        k: KReg,
    ) -> Result<(), EncError> {
        let (opc, w) = op.desc();
        if data.num() > 15 || index.num() > 15 {
            return Err(EncError::Invalid("EVEX register above 15"));
        }
        if k.0 == 0 {
            return Err(EncError::Invalid("a gather or scatter needs a mask register other than k0"));
        }
        if !op.scatter() && data.num() == index.num() {
            return Err(EncError::Invalid("a gather's data and index registers must differ"));
        }
        let rm = RmEnc::Vsib { base: Some(base), index: index.num(), scale, disp };
        let n = if w { 8 } else { 4 };
        self.evex_raw(Pp::P66, Map::M38, w, opc, vl, data.num(), 0, rm, k, false, Tt::Fixed(n), None)?;
        self.note(Feature::Avx512f);
        if vl != EVL::L512 {
            self.note(Feature::Avx512vl);
        }
        Ok(())
    }
}
