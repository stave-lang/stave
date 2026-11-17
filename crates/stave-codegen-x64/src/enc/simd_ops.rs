//! Opcode tables of the packed-vector instructions. One enum per operand shape; each variant is
//! named after the legacy mnemonic (`Addps` is `addps` and, VEX-encoded, `vaddps`), and VEX-only
//! instructions are named without the `v` (`Psllvd` is `vpsllvd`).

use stave_target::Feature;

/// Mandatory prefix (VEX `pp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pp {
    N = 0,
    P66 = 1,
    F3 = 2,
    F2 = 3,
}

impl Pp {
    pub(super) fn legacy(self) -> Option<u8> {
        match self {
            Pp::N => None,
            Pp::P66 => Some(0x66),
            Pp::F3 => Some(0xf3),
            Pp::F2 => Some(0xf2),
        }
    }
}

/// Opcode map (VEX `mmmmm`): `0F`, `0F 38`, `0F 3A`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Map {
    M0F = 1,
    M38 = 2,
    M3A = 3,
}

impl Map {
    /// The escape bytes of the legacy encoding, then the opcode.
    pub(super) fn legacy_opc(self, opc: u8) -> ([u8; 3], usize) {
        match self {
            Map::M0F => ([0x0f, opc, 0], 2),
            Map::M38 => ([0x0f, 0x38, opc], 3),
            Map::M3A => ([0x0f, 0x3a, opc], 3),
        }
    }
}

/// Operand-class flags of a table entry.
pub(super) mod flag {
    /// VEX only: no legacy SSE encoding.
    pub const V: u16 = 1;
    /// 128-bit only.
    pub const X: u16 = 2;
    /// 256-bit only.
    pub const Y: u16 = 4;
    /// The r/m register operand is an `xmm` whatever the vector width.
    pub const RX: u16 = 8;
    /// The r/m operand must be a register.
    pub const REG: u16 = 16;
    /// The r/m operand must be memory.
    pub const MEM: u16 = 32;
    /// The immediate is a compare predicate (below 8 legacy, below 32 VEX).
    pub const CMP: u16 = 64;
    /// VEX.W = 1 (the legacy form never needs REX.W).
    pub const W1: u16 = 128;
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Desc {
    pub pp: Pp,
    pub map: Map,
    pub opc: u8,
    pub flags: u16,
}

impl Desc {
    pub(super) fn has(self, f: u16) -> bool {
        self.flags & f != 0
    }
}

macro_rules! ops {
    ($(#[$m:meta])* $name:ident { $($v:ident = ($pp:ident, $map:ident, $opc:expr $(, $f:ident)*)),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($v),*
        }

        impl $name {
            /// Every variant.
            pub const ALL: &'static [$name] = &[$($name::$v),*];

            pub(super) fn desc(self) -> Desc {
                match self {
                    $($name::$v => Desc {
                        pp: Pp::$pp,
                        map: Map::$map,
                        opc: $opc,
                        flags: 0 $(| flag::$f)*,
                    }),*
                }
            }
        }
    };
}

ops! {
    /// `op dst, src` / `vop dst, a, b`: two register-or-memory sources.
    VBin {
        Addps = (N, M0F, 0x58), Addpd = (P66, M0F, 0x58),
        Subps = (N, M0F, 0x5c), Subpd = (P66, M0F, 0x5c),
        Mulps = (N, M0F, 0x59), Mulpd = (P66, M0F, 0x59),
        Divps = (N, M0F, 0x5e), Divpd = (P66, M0F, 0x5e),
        Minps = (N, M0F, 0x5d), Minpd = (P66, M0F, 0x5d),
        Maxps = (N, M0F, 0x5f), Maxpd = (P66, M0F, 0x5f),
        Andps = (N, M0F, 0x54), Andpd = (P66, M0F, 0x54),
        Andnps = (N, M0F, 0x55), Andnpd = (P66, M0F, 0x55),
        Orps = (N, M0F, 0x56), Orpd = (P66, M0F, 0x56),
        Xorps = (N, M0F, 0x57), Xorpd = (P66, M0F, 0x57),
        Unpcklps = (N, M0F, 0x14), Unpcklpd = (P66, M0F, 0x14),
        Unpckhps = (N, M0F, 0x15), Unpckhpd = (P66, M0F, 0x15),
        Haddps = (F2, M0F, 0x7c), Haddpd = (P66, M0F, 0x7c),
        Movhlps = (N, M0F, 0x12, X, REG), Movlhps = (N, M0F, 0x16, X, REG),
        Paddb = (P66, M0F, 0xfc), Paddw = (P66, M0F, 0xfd), Paddd = (P66, M0F, 0xfe), Paddq = (P66, M0F, 0xd4),
        Psubb = (P66, M0F, 0xf8), Psubw = (P66, M0F, 0xf9), Psubd = (P66, M0F, 0xfa), Psubq = (P66, M0F, 0xfb),
        Paddsb = (P66, M0F, 0xec), Paddsw = (P66, M0F, 0xed),
        Paddusb = (P66, M0F, 0xdc), Paddusw = (P66, M0F, 0xdd),
        Psubsb = (P66, M0F, 0xe8), Psubsw = (P66, M0F, 0xe9),
        Psubusb = (P66, M0F, 0xd8), Psubusw = (P66, M0F, 0xd9),
        Pmullw = (P66, M0F, 0xd5), Pmulld = (P66, M38, 0x40),
        Pmuludq = (P66, M0F, 0xf4), Pmuldq = (P66, M38, 0x28),
        Pminsb = (P66, M38, 0x38), Pminsw = (P66, M0F, 0xea), Pminsd = (P66, M38, 0x39),
        Pminub = (P66, M0F, 0xda), Pminuw = (P66, M38, 0x3a), Pminud = (P66, M38, 0x3b),
        Pmaxsb = (P66, M38, 0x3c), Pmaxsw = (P66, M0F, 0xee), Pmaxsd = (P66, M38, 0x3d),
        Pmaxub = (P66, M0F, 0xde), Pmaxuw = (P66, M38, 0x3e), Pmaxud = (P66, M38, 0x3f),
        Pand = (P66, M0F, 0xdb), Pandn = (P66, M0F, 0xdf), Por = (P66, M0F, 0xeb), Pxor = (P66, M0F, 0xef),
        Pcmpeqb = (P66, M0F, 0x74), Pcmpeqw = (P66, M0F, 0x75), Pcmpeqd = (P66, M0F, 0x76),
        Pcmpeqq = (P66, M38, 0x29),
        Pcmpgtb = (P66, M0F, 0x64), Pcmpgtw = (P66, M0F, 0x65), Pcmpgtd = (P66, M0F, 0x66),
        Pcmpgtq = (P66, M38, 0x37),
        Pshufb = (P66, M38, 0x00),
        Punpcklbw = (P66, M0F, 0x60), Punpcklwd = (P66, M0F, 0x61),
        Punpckldq = (P66, M0F, 0x62), Punpcklqdq = (P66, M0F, 0x6c),
        Punpckhbw = (P66, M0F, 0x68), Punpckhwd = (P66, M0F, 0x69),
        Punpckhdq = (P66, M0F, 0x6a), Punpckhqdq = (P66, M0F, 0x6d),
        Packsswb = (P66, M0F, 0x63), Packssdw = (P66, M0F, 0x6b),
        Packuswb = (P66, M0F, 0x67), Packusdw = (P66, M38, 0x2b),
        Psllw = (P66, M0F, 0xf1, RX), Pslld = (P66, M0F, 0xf2, RX), Psllq = (P66, M0F, 0xf3, RX),
        Psrlw = (P66, M0F, 0xd1, RX), Psrld = (P66, M0F, 0xd2, RX), Psrlq = (P66, M0F, 0xd3, RX),
        Psraw = (P66, M0F, 0xe1, RX), Psrad = (P66, M0F, 0xe2, RX),
        Psllvd = (P66, M38, 0x47, V), Psllvq = (P66, M38, 0x47, V, W1),
        Psrlvd = (P66, M38, 0x45, V), Psrlvq = (P66, M38, 0x45, V, W1),
        Psravd = (P66, M38, 0x46, V),
        Permd = (P66, M38, 0x36, V, Y), Permps = (P66, M38, 0x16, V, Y),
        Permilps = (P66, M38, 0x0c, V), Permilpd = (P66, M38, 0x0d, V),
        Maskmovps = (P66, M38, 0x2c, V, MEM), Maskmovpd = (P66, M38, 0x2d, V, MEM),
        Pmaskmovd = (P66, M38, 0x8c, V, MEM), Pmaskmovq = (P66, M38, 0x8c, V, MEM, W1),
        Fmadd132ps = (P66, M38, 0x98, V), Fmadd132pd = (P66, M38, 0x98, V, W1),
        Fmadd132ss = (P66, M38, 0x99, V, X), Fmadd132sd = (P66, M38, 0x99, V, X, W1),
        Fmadd213ps = (P66, M38, 0xa8, V), Fmadd213pd = (P66, M38, 0xa8, V, W1),
        Fmadd213ss = (P66, M38, 0xa9, V, X), Fmadd213sd = (P66, M38, 0xa9, V, X, W1),
        Fmadd231ps = (P66, M38, 0xb8, V), Fmadd231pd = (P66, M38, 0xb8, V, W1),
        Fmadd231ss = (P66, M38, 0xb9, V, X), Fmadd231sd = (P66, M38, 0xb9, V, X, W1),
        Fmsub132ps = (P66, M38, 0x9a, V), Fmsub132pd = (P66, M38, 0x9a, V, W1),
        Fmsub132ss = (P66, M38, 0x9b, V, X), Fmsub132sd = (P66, M38, 0x9b, V, X, W1),
        Fmsub213ps = (P66, M38, 0xaa, V), Fmsub213pd = (P66, M38, 0xaa, V, W1),
        Fmsub213ss = (P66, M38, 0xab, V, X), Fmsub213sd = (P66, M38, 0xab, V, X, W1),
        Fmsub231ps = (P66, M38, 0xba, V), Fmsub231pd = (P66, M38, 0xba, V, W1),
        Fmsub231ss = (P66, M38, 0xbb, V, X), Fmsub231sd = (P66, M38, 0xbb, V, X, W1),
        Fnmadd132ps = (P66, M38, 0x9c, V), Fnmadd132pd = (P66, M38, 0x9c, V, W1),
        Fnmadd132ss = (P66, M38, 0x9d, V, X), Fnmadd132sd = (P66, M38, 0x9d, V, X, W1),
        Fnmadd213ps = (P66, M38, 0xac, V), Fnmadd213pd = (P66, M38, 0xac, V, W1),
        Fnmadd213ss = (P66, M38, 0xad, V, X), Fnmadd213sd = (P66, M38, 0xad, V, X, W1),
        Fnmadd231ps = (P66, M38, 0xbc, V), Fnmadd231pd = (P66, M38, 0xbc, V, W1),
        Fnmadd231ss = (P66, M38, 0xbd, V, X), Fnmadd231sd = (P66, M38, 0xbd, V, X, W1),
        Fnmsub132ps = (P66, M38, 0x9e, V), Fnmsub132pd = (P66, M38, 0x9e, V, W1),
        Fnmsub132ss = (P66, M38, 0x9f, V, X), Fnmsub132sd = (P66, M38, 0x9f, V, X, W1),
        Fnmsub213ps = (P66, M38, 0xae, V), Fnmsub213pd = (P66, M38, 0xae, V, W1),
        Fnmsub213ss = (P66, M38, 0xaf, V, X), Fnmsub213sd = (P66, M38, 0xaf, V, X, W1),
        Fnmsub231ps = (P66, M38, 0xbe, V), Fnmsub231pd = (P66, M38, 0xbe, V, W1),
        Fnmsub231ss = (P66, M38, 0xbf, V, X), Fnmsub231sd = (P66, M38, 0xbf, V, X, W1),
    }
}

ops! {
    /// `op dst, src` / `vop dst, src`: one source.
    VUn {
        Sqrtps = (N, M0F, 0x51), Sqrtpd = (P66, M0F, 0x51),
        Movups = (N, M0F, 0x10), Movaps = (N, M0F, 0x28),
        Movdqu = (F3, M0F, 0x6f), Movdqa = (P66, M0F, 0x6f),
        Movq = (F3, M0F, 0x7e, X),
        Pabsb = (P66, M38, 0x1c), Pabsw = (P66, M38, 0x1d), Pabsd = (P66, M38, 0x1e),
        Ptest = (P66, M38, 0x17),
        Cvtdq2ps = (N, M0F, 0x5b), Cvtps2dq = (P66, M0F, 0x5b), Cvttps2dq = (F3, M0F, 0x5b),
        Cvtdq2pd = (F3, M0F, 0xe6, RX), Cvtps2pd = (N, M0F, 0x5a, RX),
        Pmovsxbw = (P66, M38, 0x20, RX), Pmovsxbd = (P66, M38, 0x21, RX), Pmovsxbq = (P66, M38, 0x22, RX),
        Pmovsxwd = (P66, M38, 0x23, RX), Pmovsxwq = (P66, M38, 0x24, RX), Pmovsxdq = (P66, M38, 0x25, RX),
        Pmovzxbw = (P66, M38, 0x30, RX), Pmovzxbd = (P66, M38, 0x31, RX), Pmovzxbq = (P66, M38, 0x32, RX),
        Pmovzxwd = (P66, M38, 0x33, RX), Pmovzxwq = (P66, M38, 0x34, RX), Pmovzxdq = (P66, M38, 0x35, RX),
        Broadcastss = (P66, M38, 0x18, V, RX), Broadcastsd = (P66, M38, 0x19, V, RX, Y),
        Pbroadcastb = (P66, M38, 0x78, V, RX), Pbroadcastw = (P66, M38, 0x79, V, RX),
        Pbroadcastd = (P66, M38, 0x58, V, RX), Pbroadcastq = (P66, M38, 0x59, V, RX),
        Cvtph2ps = (P66, M38, 0x13, V, RX),
    }
}

ops! {
    /// `op dst, src, imm8` / `vop dst, a, b, imm8`.
    VBinI {
        Cmpps = (N, M0F, 0xc2, CMP), Cmppd = (P66, M0F, 0xc2, CMP),
        Shufps = (N, M0F, 0xc6), Shufpd = (P66, M0F, 0xc6),
        Blendps = (P66, M3A, 0x0c), Blendpd = (P66, M3A, 0x0d), Pblendw = (P66, M3A, 0x0e),
        Palignr = (P66, M3A, 0x0f),
        Insertps = (P66, M3A, 0x21, X),
        Roundss = (P66, M3A, 0x0a, X), Roundsd = (P66, M3A, 0x0b, X),
        Pblendd = (P66, M3A, 0x02, V),
        Perm2f128 = (P66, M3A, 0x06, V, Y), Perm2i128 = (P66, M3A, 0x46, V, Y),
        Insertf128 = (P66, M3A, 0x18, V, Y, RX), Inserti128 = (P66, M3A, 0x38, V, Y, RX),
    }
}

ops! {
    /// `op dst, src, imm8` / `vop dst, src, imm8`.
    VUnI {
        Pshufd = (P66, M0F, 0x70), Pshuflw = (F2, M0F, 0x70), Pshufhw = (F3, M0F, 0x70),
        Roundps = (P66, M3A, 0x08), Roundpd = (P66, M3A, 0x09),
        Permq = (P66, M3A, 0x00, V, Y, W1), Permpd = (P66, M3A, 0x01, V, Y, W1),
        Permilps = (P66, M3A, 0x04, V), Permilpd = (P66, M3A, 0x05, V),
    }
}

ops! {
    /// Stores of a vector register to memory.
    VStore {
        Movups = (N, M0F, 0x11), Movaps = (N, M0F, 0x29),
        Movdqu = (F3, M0F, 0x7f), Movdqa = (P66, M0F, 0x7f),
        Movq = (P66, M0F, 0xd6, X),
        Movntps = (N, M0F, 0x2b), Movntpd = (P66, M0F, 0x2b), Movntdq = (P66, M0F, 0xe7),
    }
}

ops! {
    /// Conversions that narrow a vector to an `xmm` result (the source may be 256 bits wide).
    VCvtN {
        Cvtpd2dq = (F2, M0F, 0xe6), Cvttpd2dq = (P66, M0F, 0xe6), Cvtpd2ps = (P66, M0F, 0x5a),
    }
}

ops! {
    /// Sign-bit gathers into a general register.
    VMovMsk {
        Movmskps = (N, M0F, 0x50), Movmskpd = (P66, M0F, 0x50), Pmovmskb = (P66, M0F, 0xd7),
    }
}

ops! {
    /// Variable blends; the legacy forms take the mask in `xmm0`.
    VBlendV {
        Blendvps = (P66, M38, 0x14), Blendvpd = (P66, M38, 0x15), Pblendvb = (P66, M38, 0x10),
    }
}

ops! {
    /// Stores of a register to a vector or `xmm` destination with an immediate.
    VExtract {
        Extractf128 = (P66, M3A, 0x19, V, Y), Extracti128 = (P66, M3A, 0x39, V, Y),
        Cvtps2ph = (P66, M3A, 0x1d, V),
    }
}

ops! {
    /// Masked stores (`mask` in `vvvv`).
    VMaskStore {
        Maskmovps = (P66, M38, 0x2e, V), Maskmovpd = (P66, M38, 0x2f, V),
        Pmaskmovd = (P66, M38, 0x8e, V), Pmaskmovq = (P66, M38, 0x8e, V, W1),
    }
}

ops! {
    /// Gathers through a vector index (VSIB). `d`/`q` name the index element width, `ps/pd/d/q`
    /// the data.
    VGather {
        Gatherdps = (P66, M38, 0x92, V), Gatherdpd = (P66, M38, 0x92, V, W1),
        Gatherqps = (P66, M38, 0x93, V), Gatherqpd = (P66, M38, 0x93, V, W1),
        Pgatherdd = (P66, M38, 0x90, V), Pgatherdq = (P66, M38, 0x90, V, W1),
        Pgatherqd = (P66, M38, 0x91, V), Pgatherqq = (P66, M38, 0x91, V, W1),
    }
}

/// The CPUID feature of the VEX form of an integer instruction that has a legacy SSE form: AVX
/// encodes it at 128 bits, AVX2 added the 256-bit form.
fn int_feature(wide: bool) -> Feature {
    if wide { Feature::Avx2 } else { Feature::Avx }
}

impl VBin {
    /// The CPUID feature of the VEX form (`wide`: at 256 bits).
    pub(super) fn feature(self, wide: bool) -> Feature {
        use VBin::*;
        match self {
            Addps | Addpd | Subps | Subpd | Mulps | Mulpd | Divps | Divpd | Minps | Minpd | Maxps | Maxpd
            | Andps | Andpd | Andnps | Andnpd | Orps | Orpd | Xorps | Xorpd | Unpcklps | Unpcklpd
            | Unpckhps | Unpckhpd | Haddps | Haddpd | Movhlps | Movlhps | Permilps | Permilpd | Maskmovps
            | Maskmovpd => Feature::Avx,
            Psllvd | Psllvq | Psrlvd | Psrlvq | Psravd | Permd | Permps | Pmaskmovd | Pmaskmovq => {
                Feature::Avx2
            }
            // the other VEX-only entries are the fused multiply-adds
            _ if self.desc().has(flag::V) => Feature::Fma,
            _ => int_feature(wide),
        }
    }
}

impl VUn {
    /// The CPUID feature of the VEX form (`wide`: at 256 bits; `mem`: with a memory source).
    pub(super) fn feature(self, wide: bool, mem: bool) -> Feature {
        use VUn::*;
        match self {
            Sqrtps | Sqrtpd | Movups | Movaps | Movdqu | Movdqa | Movq | Ptest | Cvtdq2ps | Cvtps2dq
            | Cvttps2dq | Cvtdq2pd | Cvtps2pd => Feature::Avx,
            // AVX broadcasts from memory only
            Broadcastss | Broadcastsd if mem => Feature::Avx,
            Broadcastss | Broadcastsd | Pbroadcastb | Pbroadcastw | Pbroadcastd | Pbroadcastq => {
                Feature::Avx2
            }
            Cvtph2ps => Feature::F16c,
            Pabsb | Pabsw | Pabsd | Pmovsxbw | Pmovsxbd | Pmovsxbq | Pmovsxwd | Pmovsxwq | Pmovsxdq
            | Pmovzxbw | Pmovzxbd | Pmovzxbq | Pmovzxwd | Pmovzxwq | Pmovzxdq => int_feature(wide),
        }
    }
}

impl VBinI {
    /// The CPUID feature of the VEX form (`wide`: at 256 bits).
    pub(super) fn feature(self, wide: bool) -> Feature {
        use VBinI::*;
        match self {
            Cmpps | Cmppd | Shufps | Shufpd | Blendps | Blendpd | Insertps | Roundss | Roundsd
            | Perm2f128 | Insertf128 => Feature::Avx,
            Pblendd | Perm2i128 | Inserti128 => Feature::Avx2,
            Pblendw | Palignr => int_feature(wide),
        }
    }
}

impl VUnI {
    /// The CPUID feature of the VEX form (`wide`: at 256 bits).
    pub(super) fn feature(self, wide: bool) -> Feature {
        use VUnI::*;
        match self {
            Roundps | Roundpd | Permilps | Permilpd => Feature::Avx,
            Permq | Permpd => Feature::Avx2,
            Pshufd | Pshuflw | Pshufhw => int_feature(wide),
        }
    }
}

impl VMovMsk {
    /// The CPUID feature of the VEX form (`wide`: of a `ymm` source).
    pub(super) fn feature(self, wide: bool) -> Feature {
        match self {
            VMovMsk::Movmskps | VMovMsk::Movmskpd => Feature::Avx,
            VMovMsk::Pmovmskb => int_feature(wide),
        }
    }
}

impl VBlendV {
    /// The CPUID feature of the VEX form (`wide`: at 256 bits).
    pub(super) fn feature(self, wide: bool) -> Feature {
        match self {
            VBlendV::Blendvps | VBlendV::Blendvpd => Feature::Avx,
            VBlendV::Pblendvb => int_feature(wide),
        }
    }
}

impl VExtract {
    pub(super) fn feature(self) -> Feature {
        match self {
            VExtract::Extractf128 => Feature::Avx,
            VExtract::Extracti128 => Feature::Avx2,
            VExtract::Cvtps2ph => Feature::F16c,
        }
    }
}

impl VMaskStore {
    pub(super) fn feature(self) -> Feature {
        match self {
            VMaskStore::Maskmovps | VMaskStore::Maskmovpd => Feature::Avx,
            VMaskStore::Pmaskmovd | VMaskStore::Pmaskmovq => Feature::Avx2,
        }
    }
}

/// Packed shifts by an immediate: the opcode group and the `/digit` of the ModRM `reg` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VShiftI {
    Psllw,
    Pslld,
    Psllq,
    Psrlw,
    Psrld,
    Psrlq,
    Psraw,
    Psrad,
    /// Byte shift of each 128-bit lane.
    Pslldq,
    Psrldq,
}

impl VShiftI {
    pub const ALL: &'static [VShiftI] = &[
        VShiftI::Psllw,
        VShiftI::Pslld,
        VShiftI::Psllq,
        VShiftI::Psrlw,
        VShiftI::Psrld,
        VShiftI::Psrlq,
        VShiftI::Psraw,
        VShiftI::Psrad,
        VShiftI::Pslldq,
        VShiftI::Psrldq,
    ];

    /// The CPUID feature of the VEX form (`wide`: at 256 bits).
    pub(super) fn feature(self, wide: bool) -> Feature {
        int_feature(wide)
    }

    /// `(opcode, /digit)`.
    pub(super) fn group(self) -> (u8, u8) {
        match self {
            VShiftI::Psrlw => (0x71, 2),
            VShiftI::Psraw => (0x71, 4),
            VShiftI::Psllw => (0x71, 6),
            VShiftI::Psrld => (0x72, 2),
            VShiftI::Psrad => (0x72, 4),
            VShiftI::Pslld => (0x72, 6),
            VShiftI::Psrlq => (0x73, 2),
            VShiftI::Psrldq => (0x73, 3),
            VShiftI::Psllq => (0x73, 6),
            VShiftI::Pslldq => (0x73, 7),
        }
    }
}

/// Extracts a lane into a general register or memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PExtr {
    Pextrb,
    Pextrw,
    Pextrd,
    Pextrq,
    Extractps,
}

impl PExtr {
    pub const ALL: &'static [PExtr] =
        &[PExtr::Pextrb, PExtr::Pextrw, PExtr::Pextrd, PExtr::Pextrq, PExtr::Extractps];

    /// `(opcode, W)` in map `0F 3A`.
    pub(super) fn code(self) -> (u8, bool) {
        match self {
            PExtr::Pextrb => (0x14, false),
            PExtr::Pextrw => (0x15, false),
            PExtr::Pextrd => (0x16, false),
            PExtr::Pextrq => (0x16, true),
            PExtr::Extractps => (0x17, false),
        }
    }
}

/// Inserts a lane from a general register or memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PInsr {
    Pinsrb,
    Pinsrw,
    Pinsrd,
    Pinsrq,
}

impl PInsr {
    pub const ALL: &'static [PInsr] = &[PInsr::Pinsrb, PInsr::Pinsrw, PInsr::Pinsrd, PInsr::Pinsrq];

    /// `(map, opcode, W)`.
    pub(super) fn code(self) -> (Map, u8, bool) {
        match self {
            PInsr::Pinsrb => (Map::M3A, 0x20, false),
            PInsr::Pinsrw => (Map::M0F, 0xc4, false),
            PInsr::Pinsrd => (Map::M3A, 0x22, false),
            PInsr::Pinsrq => (Map::M3A, 0x22, true),
        }
    }
}

/// Software prefetch hints (`/digit` of `0F 18`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefetch {
    Nta = 0,
    T0 = 1,
    T1 = 2,
    T2 = 3,
}

/// The AVX compare predicates `0..8` (the only ones the legacy `cmpps/cmppd` encode).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpPred {
    Eq = 0,
    Lt = 1,
    Le = 2,
    Unord = 3,
    Neq = 4,
    Nlt = 5,
    Nle = 6,
    Ord = 7,
}
