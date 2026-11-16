//! Packed-vector instructions in two encodings.
//!
//! The `sse_*` methods emit the legacy SSE forms (128-bit, `xmm0`..`xmm15`, destructive
//! two-operand: the destination is also the first source) and never a VEX prefix. The `vex_*`
//! methods emit VEX forms at 128 or 256 bits; the width follows the register type
//! ([`Xmm`] or [`Ymm`]). Operations are table-driven: an enum per operand shape ([`VBin`],
//! [`VUn`], [`VBinI`], [`VUnI`], ...) names the instruction and carries its opcode, and one method
//! per shape encodes it. Combinations the hardware rejects (a 256-bit `movhlps`, a memory
//! `vmaskmov` source, a VEX-only op through `sse_*`) are [`EncError`]s.

use super::inst::{Enc, RmEnc};
use super::simd_ops::{Desc, Map, Pp, flag};
use super::{
    Asm, EncError, Gpr, Mem, PExtr, PInsr, Prefetch, Rm, Scale, Size, VBin, VBinI, VBlendV, VCvtN, VExtract,
    VGather, VMaskStore, VMovMsk, VShiftI, VStore, VUn, VUnI, XRm, Xmm, Ymm,
};

/// Vector length of a VEX instruction (`VEX.L`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VLen {
    X128,
    X256,
}

/// A vector register of either width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VReg {
    X(Xmm),
    Y(Ymm),
}

impl VReg {
    pub fn num(self) -> u8 {
        match self {
            VReg::X(x) => x.0,
            VReg::Y(y) => y.0,
        }
    }

    pub fn len(self) -> VLen {
        match self {
            VReg::X(_) => VLen::X128,
            VReg::Y(_) => VLen::X256,
        }
    }
}

impl From<Xmm> for VReg {
    fn from(x: Xmm) -> VReg {
        VReg::X(x)
    }
}

impl From<Ymm> for VReg {
    fn from(y: Ymm) -> VReg {
        VReg::Y(y)
    }
}

/// A vector register or memory operand of a VEX instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VRm {
    R(VReg),
    M(Mem),
}

impl From<Xmm> for VRm {
    fn from(x: Xmm) -> VRm {
        VRm::R(VReg::X(x))
    }
}

impl From<Ymm> for VRm {
    fn from(y: Ymm) -> VRm {
        VRm::R(VReg::Y(y))
    }
}

impl From<VReg> for VRm {
    fn from(r: VReg) -> VRm {
        VRm::R(r)
    }
}

impl From<Mem> for VRm {
    fn from(m: Mem) -> VRm {
        VRm::M(m)
    }
}

/// A register type that fixes the VEX vector length: [`Xmm`] (128) or [`Ymm`] (256).
pub trait VecReg: Copy + Into<VReg> {
    const LEN: VLen;

    fn num(self) -> u8 {
        self.into().num()
    }
}

impl VecReg for Xmm {
    const LEN: VLen = VLen::X128;
}

impl VecReg for Ymm {
    const LEN: VLen = VLen::X256;
}

/// One VEX encoding: `[C5 | C4] opcode ModRM [SIB] [disp]` (an immediate is written by the
/// caller; `trailing` is its length, for RIP-relative displacements).
pub(super) struct Vex {
    pub(super) pp: Pp,
    pub(super) map: Map,
    pub(super) w: bool,
    pub(super) l: VLen,
    pub(super) opc: u8,
    pub(super) reg: u8,
    /// `vvvv`: the extra source register number (0 when unused).
    pub(super) vvvv: u8,
    pub(super) rm: RmEnc,
    pub(super) trailing: u8,
}

fn check_len(d: Desc, l: VLen) -> Result<(), EncError> {
    if d.has(flag::X) && l != VLen::X128 {
        return Err(EncError::Invalid("instruction exists only at 128 bits"));
    }
    if d.has(flag::Y) && l != VLen::X256 {
        return Err(EncError::Invalid("instruction exists only at 256 bits"));
    }
    Ok(())
}

fn check_imm(d: Desc, imm: u8, limit: u8) -> Result<(), EncError> {
    if d.has(flag::CMP) && imm >= limit {
        return Err(EncError::ImmRange { imm: i64::from(imm), bits: limit.trailing_zeros() });
    }
    Ok(())
}

/// The r/m operand of a VEX instruction at length `l`, checked against the entry's flags.
fn vrm(d: Desc, l: VLen, rm: VRm) -> Result<RmEnc, EncError> {
    check_len(d, l)?;
    match rm {
        VRm::M(_) if d.has(flag::REG) => Err(EncError::Invalid("operand must be a register")),
        VRm::M(m) => Ok(RmEnc::Mem(m)),
        VRm::R(_) if d.has(flag::MEM) => Err(EncError::Invalid("operand must be memory")),
        VRm::R(r) => {
            let want = if d.has(flag::RX) { VLen::X128 } else { l };
            if r.len() != want {
                return Err(EncError::Invalid("register width does not match the instruction"));
            }
            Ok(RmEnc::Reg(r.num()))
        }
    }
}

impl Asm {
    /// Writes a VEX prefix, the opcode, and the ModRM operand, using the 2-byte prefix when it can
    /// express the instruction.
    pub(super) fn vex(&mut self, v: Vex) -> Result<(), EncError> {
        let (x, b) = self.rm_xb(&v.rm)?;
        let r = (v.reg >> 3) & 1;
        let l = u8::from(v.l == VLen::X256);
        let vv = !v.vvvv & 15;
        let pp = v.pp as u8;
        if v.map == Map::M0F && !v.w && x == 0 && b == 0 {
            self.buf.extend_from_slice(&[0xc5, ((r ^ 1) << 7) | (vv << 3) | (l << 2) | pp]);
        } else {
            self.buf.extend_from_slice(&[
                0xc4,
                ((r ^ 1) << 7) | ((x ^ 1) << 6) | ((b ^ 1) << 5) | v.map as u8,
                (u8::from(v.w) << 7) | (vv << 3) | (l << 2) | pp,
            ]);
        }
        self.buf.push(v.opc);
        self.emit_modrm(v.reg, v.rm, v.trailing);
        Ok(())
    }

    /// A table instruction in VEX form.
    fn vex_d(
        &mut self,
        d: Desc,
        l: VLen,
        reg: u8,
        vvvv: u8,
        rm: RmEnc,
        imm: Option<u8>,
    ) -> Result<(), EncError> {
        let trailing = u8::from(imm.is_some());
        self.vex(Vex { pp: d.pp, map: d.map, w: d.has(flag::W1), l, opc: d.opc, reg, vvvv, rm, trailing })?;
        if let Some(i) = imm {
            self.buf.push(i);
        }
        Ok(())
    }

    /// A table instruction in legacy SSE form.
    fn legacy_d(&mut self, d: Desc, reg: u8, rm: RmEnc, imm: Option<u8>) -> Result<(), EncError> {
        if d.has(flag::V) {
            return Err(EncError::Invalid("instruction has no legacy SSE encoding"));
        }
        let (buf, n) = d.map.legacy_opc(d.opc);
        let trailing = u8::from(imm.is_some());
        self.enc(Enc {
            pfx: d.pp.legacy(),
            w: d.has(flag::W1),
            rex8: false,
            opc: &buf[..n],
            reg,
            rm,
            trailing,
        })?;
        if let Some(i) = imm {
            self.buf.push(i);
        }
        Ok(())
    }

    fn legacy_rm(d: Desc, src: XRm) -> Result<RmEnc, EncError> {
        if d.has(flag::REG) && matches!(src, XRm::M(_)) {
            return Err(EncError::Invalid("operand must be a register"));
        }
        Ok(src.into())
    }

    // ---- legacy SSE (128-bit) ----

    /// `op dst, src`: packed arithmetic, logic, compares, shuffles, shifts by an `xmm` count.
    pub fn sse_bin(&mut self, op: VBin, dst: Xmm, src: impl Into<XRm>) -> Result<(), EncError> {
        let d = op.desc();
        let rm = Self::legacy_rm(d, src.into())?;
        self.legacy_d(d, dst.0, rm, None)
    }

    /// `op dst, src`: square roots, moves, absolute values, conversions, extensions, `ptest`.
    pub fn sse_un(&mut self, op: VUn, dst: Xmm, src: impl Into<XRm>) -> Result<(), EncError> {
        let d = op.desc();
        let rm = Self::legacy_rm(d, src.into())?;
        self.legacy_d(d, dst.0, rm, None)
    }

    /// `op dst, src, imm8` (`cmpps`, `shufps`, `blendps`, `palignr`, `insertps`, ...). Compares
    /// take a predicate below 8 (see [`CmpPred`]).
    pub fn sse_bin_i(&mut self, op: VBinI, dst: Xmm, src: impl Into<XRm>, imm: u8) -> Result<(), EncError> {
        let d = op.desc();
        check_imm(d, imm, 8)?;
        let rm = Self::legacy_rm(d, src.into())?;
        self.legacy_d(d, dst.0, rm, Some(imm))
    }

    /// `op dst, src, imm8` (`pshufd`, `roundps`, ...).
    pub fn sse_un_i(&mut self, op: VUnI, dst: Xmm, src: impl Into<XRm>, imm: u8) -> Result<(), EncError> {
        let d = op.desc();
        let rm = Self::legacy_rm(d, src.into())?;
        self.legacy_d(d, dst.0, rm, Some(imm))
    }

    /// `op [m], src`: unaligned/aligned stores and non-temporal stores.
    pub fn sse_store(&mut self, op: VStore, m: Mem, src: Xmm) -> Result<(), EncError> {
        self.legacy_d(op.desc(), src.0, RmEnc::Mem(m), None)
    }

    /// `cvtpd2dq`, `cvttpd2dq`, `cvtpd2ps`.
    pub fn sse_cvt_n(&mut self, op: VCvtN, dst: Xmm, src: impl Into<XRm>) -> Result<(), EncError> {
        self.legacy_d(op.desc(), dst.0, src.into().into(), None)
    }

    /// `movmskps/movmskpd/pmovmskb dst32, src`.
    pub fn sse_movmsk(&mut self, op: VMovMsk, dst: Gpr, src: Xmm) {
        let _ = self.legacy_d(op.desc(), dst.0, RmEnc::Reg(src.0), None);
    }

    /// `blendvps/blendvpd/pblendvb dst, src` with the mask implicitly in `xmm0`.
    pub fn sse_blendv(&mut self, op: VBlendV, dst: Xmm, src: impl Into<XRm>) -> Result<(), EncError> {
        self.legacy_d(op.desc(), dst.0, src.into().into(), None)
    }

    /// `psllw/pslld/psllq/psrlw/psrld/psrlq/psraw/psrad/pslldq/psrldq dst, imm8`.
    pub fn sse_shift_i(&mut self, op: VShiftI, dst: Xmm, imm: u8) {
        let (opc, digit) = op.group();
        let e = Enc {
            pfx: Some(0x66),
            w: false,
            rex8: false,
            opc: &[0x0f, opc],
            reg: digit,
            rm: RmEnc::Reg(dst.0),
            trailing: 1,
        };
        if self.enc(e).is_ok() {
            self.buf.push(imm);
        }
    }

    fn check_dq(s: Size) -> Result<(), EncError> {
        if s < Size::D {
            return Err(EncError::Invalid("movd/movq needs a 32- or 64-bit operand"));
        }
        Ok(())
    }

    /// `movd/movq xmm, r/m32|r/m64` (zero-extends into the register).
    pub fn sse_movd_x(&mut self, s: Size, dst: Xmm, src: impl Into<Rm>) -> Result<(), EncError> {
        Self::check_dq(s)?;
        self.enc(Enc {
            pfx: Some(0x66),
            w: s == Size::Q,
            rex8: false,
            opc: &[0x0f, 0x6e],
            reg: dst.0,
            rm: src.into().into(),
            trailing: 0,
        })
    }

    /// `movd/movq r/m32|r/m64, xmm` (low lane).
    pub fn sse_movd_r(&mut self, s: Size, dst: impl Into<Rm>, src: Xmm) -> Result<(), EncError> {
        Self::check_dq(s)?;
        self.enc(Enc {
            pfx: Some(0x66),
            w: s == Size::Q,
            rex8: false,
            opc: &[0x0f, 0x7e],
            reg: src.0,
            rm: dst.into().into(),
            trailing: 0,
        })
    }

    /// `pextrb/pextrw/pextrd/pextrq/extractps dst, src, imm8`.
    pub fn sse_pextr(&mut self, op: PExtr, dst: impl Into<Rm>, src: Xmm, imm: u8) -> Result<(), EncError> {
        let (opc, w) = op.code();
        self.enc(Enc {
            pfx: Some(0x66),
            w,
            rex8: false,
            opc: &[0x0f, 0x3a, opc],
            reg: src.0,
            rm: dst.into().into(),
            trailing: 1,
        })?;
        self.buf.push(imm);
        Ok(())
    }

    /// `pinsrb/pinsrw/pinsrd/pinsrq dst, src, imm8`.
    pub fn sse_pinsr(&mut self, op: PInsr, dst: Xmm, src: impl Into<Rm>, imm: u8) -> Result<(), EncError> {
        let (map, opc, w) = op.code();
        let (buf, n) = map.legacy_opc(opc);
        self.enc(Enc {
            pfx: Some(0x66),
            w,
            rex8: false,
            opc: &buf[..n],
            reg: dst.0,
            rm: src.into().into(),
            trailing: 1,
        })?;
        self.buf.push(imm);
        Ok(())
    }

    /// `prefetcht0/t1/t2/nta [m]`.
    pub fn prefetch(&mut self, hint: Prefetch, m: Mem) -> Result<(), EncError> {
        self.enc(Enc {
            pfx: None,
            w: false,
            rex8: false,
            opc: &[0x0f, 0x18],
            reg: hint as u8,
            rm: RmEnc::Mem(m),
            trailing: 0,
        })
    }

    /// `movnti [m], src`: a non-temporal store of a 32- (`D`) or 64-bit (`Q`) register.
    pub fn movnti(&mut self, s: Size, m: Mem, src: Gpr) -> Result<(), EncError> {
        if !matches!(s, Size::D | Size::Q) {
            return Err(EncError::Invalid("movnti stores 32 or 64 bits"));
        }
        self.enc(Enc {
            pfx: None,
            w: s == Size::Q,
            rex8: false,
            opc: &[0x0f, 0xc3],
            reg: src.0,
            rm: RmEnc::Mem(m),
            trailing: 0,
        })
    }

    /// `sfence`: orders non-temporal stores.
    pub fn sfence(&mut self) {
        self.buf.extend_from_slice(&[0x0f, 0xae, 0xf8]);
    }

    /// `popcnt dst, src` (`W`, `D`, `Q`).
    pub fn popcnt(&mut self, s: Size, dst: Gpr, src: impl Into<Rm>) -> Result<(), EncError> {
        if s == Size::B {
            return Err(EncError::Invalid("popcnt has no 8-bit form"));
        }
        let rm: RmEnc = src.into().into();
        self.rm_xb(&rm)?;
        if s == Size::W {
            self.buf.push(0x66);
        }
        self.enc(Enc {
            pfx: Some(0xf3),
            w: s == Size::Q,
            rex8: false,
            opc: &[0x0f, 0xb8],
            reg: dst.0,
            rm,
            trailing: 0,
        })
    }

    // ---- VEX (128 and 256 bit) ----

    /// `vop dst, a, b`: three-operand form of [`Asm::sse_bin`], plus the AVX2/FMA-only
    /// instructions (variable shifts, permutes, masked loads, fused multiply-add).
    pub fn vex_bin<R: VecReg>(&mut self, op: VBin, dst: R, a: R, b: impl Into<VRm>) -> Result<(), EncError> {
        let d = op.desc();
        let rm = vrm(d, R::LEN, b.into())?;
        self.vex_d(d, R::LEN, dst.num(), a.num(), rm, None)
    }

    /// `vop dst, src`: one-source form (moves, `vsqrtps`, conversions, broadcasts, extensions).
    /// Entries whose source is always an `xmm` (broadcasts, widening) take an [`Xmm`] source with
    /// a [`Ymm`] destination.
    pub fn vex_un<R: VecReg>(&mut self, op: VUn, dst: R, src: impl Into<VRm>) -> Result<(), EncError> {
        let d = op.desc();
        let rm = vrm(d, R::LEN, src.into())?;
        self.vex_d(d, R::LEN, dst.num(), 0, rm, None)
    }

    /// `vop dst, a, b, imm8`. Compares take a predicate below 32.
    pub fn vex_bin_i<R: VecReg>(
        &mut self,
        op: VBinI,
        dst: R,
        a: R,
        b: impl Into<VRm>,
        imm: u8,
    ) -> Result<(), EncError> {
        let d = op.desc();
        check_imm(d, imm, 32)?;
        let rm = vrm(d, R::LEN, b.into())?;
        self.vex_d(d, R::LEN, dst.num(), a.num(), rm, Some(imm))
    }

    /// `vop dst, src, imm8` (`vpshufd`, `vroundps`, `vpermq`, `vpermilps`, ...).
    pub fn vex_un_i<R: VecReg>(
        &mut self,
        op: VUnI,
        dst: R,
        src: impl Into<VRm>,
        imm: u8,
    ) -> Result<(), EncError> {
        let d = op.desc();
        let rm = vrm(d, R::LEN, src.into())?;
        self.vex_d(d, R::LEN, dst.num(), 0, rm, Some(imm))
    }

    /// `vop [m], src`: stores and non-temporal stores.
    pub fn vex_store<R: VecReg>(&mut self, op: VStore, m: Mem, src: R) -> Result<(), EncError> {
        let d = op.desc();
        check_len(d, R::LEN)?;
        self.vex_d(d, R::LEN, src.num(), 0, RmEnc::Mem(m), None)
    }

    /// `vcvtpd2dq/vcvttpd2dq/vcvtpd2ps dst, src` with a 128- or 256-bit source (`len`; a register
    /// source must have that width).
    pub fn vex_cvt_n(&mut self, op: VCvtN, dst: Xmm, src: impl Into<VRm>, len: VLen) -> Result<(), EncError> {
        let d = op.desc();
        let rm = vrm(d, len, src.into())?;
        self.vex_d(d, len, dst.0, 0, rm, None)
    }

    /// `vmovmskps/vmovmskpd/vpmovmskb dst32, src`.
    pub fn vex_movmsk<R: VecReg>(&mut self, op: VMovMsk, dst: Gpr, src: R) {
        let _ = self.vex_d(op.desc(), R::LEN, dst.0, 0, RmEnc::Reg(src.num()), None);
    }

    /// `vblendvps/vblendvpd/vpblendvb dst, a, b, mask` (the mask register is the `is4` byte).
    pub fn vex_blendv<R: VecReg>(
        &mut self,
        op: VBlendV,
        dst: R,
        a: R,
        b: impl Into<VRm>,
        mask: R,
    ) -> Result<(), EncError> {
        let opc = match op {
            VBlendV::Blendvps => 0x4a,
            VBlendV::Blendvpd => 0x4b,
            VBlendV::Pblendvb => 0x4c,
        };
        let rm = match b.into() {
            VRm::M(m) => RmEnc::Mem(m),
            VRm::R(r) if r.len() == R::LEN => RmEnc::Reg(r.num()),
            VRm::R(_) => return Err(EncError::Invalid("register width does not match the instruction")),
        };
        self.vex(Vex {
            pp: Pp::P66,
            map: Map::M3A,
            w: false,
            l: R::LEN,
            opc,
            reg: dst.num(),
            vvvv: a.num(),
            rm,
            trailing: 1,
        })?;
        self.buf.push(mask.num() << 4);
        Ok(())
    }

    /// `vpsllw/.../vpsrldq dst, src, imm8` (the source is a register: the group has no memory form).
    pub fn vex_shift_i<R: VecReg>(&mut self, op: VShiftI, dst: R, src: R, imm: u8) -> Result<(), EncError> {
        let (opc, digit) = op.group();
        self.vex(Vex {
            pp: Pp::P66,
            map: Map::M0F,
            w: false,
            l: R::LEN,
            opc,
            reg: digit,
            vvvv: dst.num(),
            rm: RmEnc::Reg(src.num()),
            trailing: 1,
        })?;
        self.buf.push(imm);
        Ok(())
    }

    /// `vmovd/vmovq xmm, r/m32|r/m64`.
    pub fn vex_movd_x(&mut self, s: Size, dst: Xmm, src: impl Into<Rm>) -> Result<(), EncError> {
        Self::check_dq(s)?;
        self.vex(Vex {
            pp: Pp::P66,
            map: Map::M0F,
            w: s == Size::Q,
            l: VLen::X128,
            opc: 0x6e,
            reg: dst.0,
            vvvv: 0,
            rm: src.into().into(),
            trailing: 0,
        })
    }

    /// `vmovd/vmovq r/m32|r/m64, xmm`.
    pub fn vex_movd_r(&mut self, s: Size, dst: impl Into<Rm>, src: Xmm) -> Result<(), EncError> {
        Self::check_dq(s)?;
        self.vex(Vex {
            pp: Pp::P66,
            map: Map::M0F,
            w: s == Size::Q,
            l: VLen::X128,
            opc: 0x7e,
            reg: src.0,
            vvvv: 0,
            rm: dst.into().into(),
            trailing: 0,
        })
    }

    /// `vextractf128/vextracti128 dst, src, imm8` (`src` a `ymm`) and `vcvtps2ph dst, src, imm8`
    /// (`src` either width); the destination is an `xmm` or memory.
    pub fn vex_extract<R: VecReg>(
        &mut self,
        op: VExtract,
        dst: impl Into<XRm>,
        src: R,
        imm: u8,
    ) -> Result<(), EncError> {
        let d = op.desc();
        check_len(d, R::LEN)?;
        self.vex_d(d, R::LEN, src.num(), 0, dst.into().into(), Some(imm))
    }

    /// `vmaskmovps/vmaskmovpd/vpmaskmovd/vpmaskmovq [m], mask, src`.
    pub fn vex_maskstore<R: VecReg>(
        &mut self,
        op: VMaskStore,
        m: Mem,
        mask: R,
        src: R,
    ) -> Result<(), EncError> {
        self.vex_d(op.desc(), R::LEN, src.num(), mask.num(), RmEnc::Mem(m), None)
    }

    /// `vpextrb/vpextrw/vpextrd/vpextrq/vextractps dst, src, imm8`.
    pub fn vex_pextr(&mut self, op: PExtr, dst: impl Into<Rm>, src: Xmm, imm: u8) -> Result<(), EncError> {
        let (opc, w) = op.code();
        self.vex(Vex {
            pp: Pp::P66,
            map: Map::M3A,
            w,
            l: VLen::X128,
            opc,
            reg: src.0,
            vvvv: 0,
            rm: dst.into().into(),
            trailing: 1,
        })?;
        self.buf.push(imm);
        Ok(())
    }

    /// `vpinsrb/vpinsrw/vpinsrd/vpinsrq dst, a, src, imm8`.
    pub fn vex_pinsr(
        &mut self,
        op: PInsr,
        dst: Xmm,
        a: Xmm,
        src: impl Into<Rm>,
        imm: u8,
    ) -> Result<(), EncError> {
        let (map, opc, w) = op.code();
        self.vex(Vex {
            pp: Pp::P66,
            map,
            w,
            l: VLen::X128,
            opc,
            reg: dst.0,
            vvvv: a.0,
            rm: src.into().into(),
            trailing: 1,
        })?;
        self.buf.push(imm);
        Ok(())
    }

    /// `vgather* dst, [base + index * scale + disp], mask`. `dst`, `index`, and `mask` must be
    /// three different registers (the hardware faults otherwise); widths follow the instruction:
    /// `d`-indexed `ps`/`d` and `q`-indexed `pd`/`q` forms need equal widths, `dpd`/`dq` take an
    /// `xmm` index, and `qps`/`qd` an `xmm` destination and mask with either index width.
    pub fn vex_gather(
        &mut self,
        op: VGather,
        dst: impl Into<VReg>,
        base: Option<Gpr>,
        index: impl Into<VReg>,
        scale: Scale,
        disp: i32,
        mask: impl Into<VReg>,
    ) -> Result<(), EncError> {
        let (dst, index, mask) = (dst.into(), index.into(), mask.into());
        if dst.num() == index.num() || dst.num() == mask.num() || index.num() == mask.num() {
            return Err(EncError::Invalid("gather needs distinct destination, index, and mask registers"));
        }
        let (dl, il, ml) = (dst.len(), index.len(), mask.len());
        let l = match op {
            VGather::Gatherdps | VGather::Pgatherdd | VGather::Gatherqpd | VGather::Pgatherqq => {
                if dl != il || dl != ml {
                    return Err(EncError::Invalid("gather registers must have the same width"));
                }
                dl
            }
            VGather::Gatherdpd | VGather::Pgatherdq => {
                if il != VLen::X128 || dl != ml {
                    return Err(EncError::Invalid(
                        "gather needs an xmm index and equal destination/mask widths",
                    ));
                }
                dl
            }
            VGather::Gatherqps | VGather::Pgatherqd => {
                if dl != VLen::X128 || ml != VLen::X128 {
                    return Err(EncError::Invalid("gather needs an xmm destination and mask"));
                }
                il
            }
        };
        let d = op.desc();
        let rm = RmEnc::Vsib { base, index: index.num(), scale, disp };
        self.vex_d(d, l, dst.num(), mask.num(), rm, None)
    }

    /// `vzeroupper`: clears the upper halves of all `ymm` registers.
    pub fn vzeroupper(&mut self) {
        self.buf.extend_from_slice(&[0xc5, 0xf8, 0x77]);
    }

    /// `vzeroall`.
    pub fn vzeroall(&mut self) {
        self.buf.extend_from_slice(&[0xc5, 0xfc, 0x77]);
    }

    // ---- BMI2 (VEX-encoded general-register instructions) ----

    fn vex_gpr(&mut self, pp: Pp, s: Size, reg: Gpr, vvvv: Gpr, rm: Rm) -> Result<(), EncError> {
        if s < Size::D {
            return Err(EncError::Invalid("BMI2 instructions need a 32- or 64-bit operand"));
        }
        self.vex(Vex {
            pp,
            map: Map::M38,
            w: s == Size::Q,
            l: VLen::X128,
            opc: 0xf5,
            reg: reg.0,
            vvvv: vvvv.0,
            rm: rm.into(),
            trailing: 0,
        })
    }

    /// `pext dst, src, mask`: gathers the `src` bits selected by `mask` into the low bits.
    pub fn pext(&mut self, s: Size, dst: Gpr, src: Gpr, mask: impl Into<Rm>) -> Result<(), EncError> {
        self.vex_gpr(Pp::F3, s, dst, src, mask.into())
    }

    /// `pdep dst, src, mask`: scatters the low bits of `src` to the `mask` positions.
    pub fn pdep(&mut self, s: Size, dst: Gpr, src: Gpr, mask: impl Into<Rm>) -> Result<(), EncError> {
        self.vex_gpr(Pp::F2, s, dst, src, mask.into())
    }

    /// `bzhi dst, src, index`: zeroes the bits of `src` from bit `index` up.
    pub fn bzhi(&mut self, s: Size, dst: Gpr, src: impl Into<Rm>, index: Gpr) -> Result<(), EncError> {
        self.vex_gpr(Pp::N, s, dst, index, src.into())
    }
}
