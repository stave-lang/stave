//! Vector and mask ops (`vec.*`, `mask.*`; semantics in stave-ir `simd`).
//!
//! An op takes its operands in registers ([`Cg::vget`]: one it may overwrite, [`Cg::vread`]: one
//! it only reads), computes register to register, and defines the result ([`Cg::vput`]); these
//! three and [`Cg::vmem`] are the only places that know where a vector value is. Ops without an
//! instruction at the level run lane by lane on the home bytes.
//!
//! In a debug build a vector value lives in its frame home (`VCls::V`): every operand is loaded
//! and every result stored. In a release build it lives in a register like a float: in the
//! block-local cache, or in the register it owns across blocks, and its home is written only
//! when a register is needed for something else, before a call, or for an op that works on
//! the home bytes ([`Cg::vmem`]).
//!
//! Encoding follows the level: legacy SSE (128-bit, destructive) at `V2`, VEX at `V3` (the same
//! destructive shape, `ymm` for 32-byte values). `xmm14` / `xmm15` are scratch here as in the
//! scalar code. A legacy SSE instruction never takes a vector memory operand (it would have to
//! be 16-byte aligned): constants and homes are loaded with unaligned moves.
//!
//! A mask is a vector of its lane width with all-ones (set) or all-zero lanes. A value of fewer
//! than 16 bytes is loaded and stored at its exact size, so nothing outside it is read or
//! written, and its register has zeros above it: after a load, and in a release build after
//! [`Cg::vput`] too, since the next op reads that register.
//!
//! `ymm` upper halves. Debug: every op ends with them clean ([`Cg::vclean`]); no 256-bit value
//! stays in a register between instructions, and the scalar code is legacy SSE. Release: a
//! function with 32-byte values keeps them in `ymm` registers and cleans before a call and
//! before a return that carries no 32-byte value; its scalar float code is VEX-encoded
//! (`Asm::set_vex_scalars`), so it never waits for an upper half. A function without 32-byte
//! values cleans after an op that used a `ymm` register inside.

mod arith;
pub(super) mod compress;
mod memory;
mod shape;

use stave_ir::arith::IntTy;
use stave_ir::simd::{self as vs, LaneTy, VecOp};
use stave_ir::{FloatBits, Operand, TypeId, Value};

use super::abi::VCls;
use super::func::*;
use super::{Level, Res, bad, unsup};
use crate::enc::*;

/// The shape of a vector or mask value.
#[derive(Clone, Copy, Debug)]
pub(super) struct Sh {
    /// The lane class; a mask has unsigned int lanes of its width.
    pub lane: LaneTy,
    /// Lane count.
    pub n: u32,
    /// Bytes per lane.
    pub w: u32,
    /// Bytes of the value.
    pub size: u32,
}

impl Sh {
    /// Held in a `ymm` register.
    pub fn wide(self) -> bool {
        self.size > 16
    }

    pub fn bits(self) -> u32 {
        self.w * 8
    }

    /// The int type of the lanes (float lanes: `None`).
    pub fn int(self) -> Option<IntTy> {
        match self.lane {
            LaneTy::Int(t) => Some(t),
            LaneTy::Float(_) => None,
        }
    }

    pub fn float(self) -> Option<Prec> {
        match self.lane {
            LaneTy::Float(FloatBits::F32) => Some(Prec::F32),
            LaneTy::Float(FloatBits::F64) => Some(Prec::F64),
            _ => None,
        }
    }

    /// The bit of the lane's sign.
    pub fn sign(self) -> u64 {
        1u64 << (self.bits() - 1)
    }

    /// All bits of a lane.
    pub fn ones(self) -> u64 {
        if self.w == 8 { u64::MAX } else { (1u64 << self.bits()) - 1 }
    }
}

/// A vector operand an op only reads: its register, or the memory of the `vec.load` that was
/// left to the op.
#[derive(Clone, Copy)]
pub(super) enum VSrc {
    R(Xmm),
    M(Mem),
}

/// The encoding of one vector instruction.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Form {
    Sse,
    X,
    Y,
}

pub(super) fn y(x: Xmm) -> Ymm {
    Ymm::masked(x.num())
}

/// Picks by lane bytes (1, 2, 4, 8).
pub(super) fn by_w<T: Copy>(w: u32, t: [T; 4]) -> T {
    t[(w.trailing_zeros() as usize).min(3)]
}

pub(super) fn scale_of(w: u32) -> Scale {
    by_w(w, [Scale::S1, Scale::S2, Scale::S4, Scale::S8])
}

pub(super) fn size_of(w: u32) -> Size {
    by_w(w, [Size::B, Size::W, Size::D, Size::Q])
}

/// The extension of `fw`-byte lanes to `tw`-byte ones (`pmovsx*` / `pmovzx*`).
pub(super) fn pmovx(fw: u32, tw: u32, signed: bool) -> VUn {
    match (fw, tw, signed) {
        (1, 2, true) => VUn::Pmovsxbw,
        (1, 4, true) => VUn::Pmovsxbd,
        (1, _, true) => VUn::Pmovsxbq,
        (2, 4, true) => VUn::Pmovsxwd,
        (2, _, true) => VUn::Pmovsxwq,
        (_, _, true) => VUn::Pmovsxdq,
        (1, 2, false) => VUn::Pmovzxbw,
        (1, 4, false) => VUn::Pmovzxbd,
        (1, _, false) => VUn::Pmovzxbq,
        (2, 4, false) => VUn::Pmovzxwd,
        (2, _, false) => VUn::Pmovzxwq,
        (_, _, false) => VUn::Pmovzxdq,
    }
}

/// The memory operand of lane `k` (`w` bytes per lane) of the vector at `m`.
pub(super) fn lane(m: Mem, k: u32, w: u32) -> Res<Mem> {
    disp(m, i64::from(k) * i64::from(w))
}

pub(super) const PADD: [VBin; 4] = [VBin::Paddb, VBin::Paddw, VBin::Paddd, VBin::Paddq];
pub(super) const PSUB: [VBin; 4] = [VBin::Psubb, VBin::Psubw, VBin::Psubd, VBin::Psubq];
pub(super) const PCMPEQ: [VBin; 4] = [VBin::Pcmpeqb, VBin::Pcmpeqw, VBin::Pcmpeqd, VBin::Pcmpeqq];
pub(super) const PCMPGT: [VBin; 4] = [VBin::Pcmpgtb, VBin::Pcmpgtw, VBin::Pcmpgtd, VBin::Pcmpgtq];

impl Cg<'_, '_> {
    // ---------------------------------------------------------------- shapes

    pub(super) fn vshape(&self, t: TypeId) -> Res<Sh> {
        let VCls::V { size } = self.vcls(t)? else { return Err(bad("vector op on a non-vector type")) };
        let types = &self.cx.m.types;
        if let Some((_, lane, n)) = vs::vec_shape(types, t) {
            return Ok(Sh { lane, n, w: lane.bits() / 8, size });
        }
        let (w, n) = vs::mask_shape(types, t).ok_or_else(|| bad("vector op on a non-vector type"))?;
        Ok(Sh { lane: LaneTy::Int(IntTy::new(w as u8, false)), n, w: w / 8, size })
    }

    pub(super) fn voshape(&self, o: Operand) -> Res<Sh> {
        self.vshape(self.oty(o).ok_or_else(|| bad("vector operand must be a value"))?)
    }

    // ---------------------------------------------------------------- encodings

    pub(super) fn vform(&mut self, wide: bool) -> Res<Form> {
        match (self.cx.level, wide) {
            (Level::V2, false) => Ok(Form::Sse),
            (Level::V2, true) => Err(bad("internal: a 256-bit vector instruction at the SSE level")),
            (Level::V3, false) => Ok(Form::X),
            (Level::V3, true) => {
                self.ydirty = true;
                self.yused = true;
                Ok(Form::Y)
            }
        }
    }

    pub(super) fn vex(&self) -> bool {
        self.cx.level == Level::V3
    }

    /// Clears the `ymm` upper halves if a 256-bit instruction dirtied them.
    pub(crate) fn vclean(&mut self) {
        if self.ydirty {
            self.a.vzeroupper();
            self.ydirty = false;
        }
    }

    /// `d = d op s`.
    pub(super) fn vb(&mut self, op: VBin, wide: bool, d: Xmm, s: Xmm) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_bin(op, d, s)?,
            Form::X => self.a.vex_bin(op, d, d, s)?,
            Form::Y => self.a.vex_bin(op, y(d), y(d), y(s))?,
        }
        Ok(())
    }

    /// `d = d op s`, the second operand a register or unaligned memory (VEX only).
    pub(super) fn vb_rm(&mut self, op: VBin, wide: bool, d: Xmm, s: VSrc) -> Res<()> {
        let m = match s {
            VSrc::R(s) => return self.vb(op, wide, d, s),
            VSrc::M(m) => m,
        };
        match self.vform(wide)? {
            Form::Sse => return Err(bad("internal: an unaligned memory operand at the SSE level")),
            Form::X => self.a.vex_bin(op, d, d, m)?,
            Form::Y => self.a.vex_bin(op, y(d), y(d), m)?,
        }
        Ok(())
    }

    /// `d = d op s` where the second operand is an `xmm` at every width (shift counts).
    pub(super) fn vb_x(&mut self, op: VBin, wide: bool, d: Xmm, s: Xmm) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_bin(op, d, s)?,
            Form::X => self.a.vex_bin(op, d, d, s)?,
            Form::Y => self.a.vex_bin(op, y(d), y(d), s)?,
        }
        Ok(())
    }

    /// `d = a op s` (VEX only).
    pub(super) fn vb3(&mut self, op: VBin, wide: bool, d: Xmm, a: Xmm, s: Xmm) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => return Err(bad("internal: a VEX-only instruction at the SSE level")),
            Form::X => self.a.vex_bin(op, d, a, s)?,
            Form::Y => self.a.vex_bin(op, y(d), y(a), y(s))?,
        }
        Ok(())
    }

    /// `d = op s`.
    pub(super) fn vu(&mut self, op: VUn, wide: bool, d: Xmm, s: Xmm) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_un(op, d, s)?,
            Form::X => self.a.vex_un(op, d, s)?,
            Form::Y => self.a.vex_un(op, y(d), y(s))?,
        }
        Ok(())
    }

    /// `d = op s` where the source is an `xmm` at every width (extensions, broadcasts).
    pub(super) fn vu_x(&mut self, op: VUn, wide: bool, d: Xmm, s: Xmm) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_un(op, d, s)?,
            Form::X => self.a.vex_un(op, d, s)?,
            Form::Y => self.a.vex_un(op, y(d), s)?,
        }
        Ok(())
    }

    /// `d = d op s, imm`.
    pub(super) fn vbi(&mut self, op: VBinI, wide: bool, d: Xmm, s: Xmm, imm: u8) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_bin_i(op, d, s, imm)?,
            Form::X => self.a.vex_bin_i(op, d, d, s, imm)?,
            Form::Y => self.a.vex_bin_i(op, y(d), y(d), y(s), imm)?,
        }
        Ok(())
    }

    /// `d = op s, imm`.
    pub(super) fn vui(&mut self, op: VUnI, wide: bool, d: Xmm, s: Xmm, imm: u8) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_un_i(op, d, s, imm)?,
            Form::X => self.a.vex_un_i(op, d, s, imm)?,
            Form::Y => self.a.vex_un_i(op, y(d), y(s), imm)?,
        }
        Ok(())
    }

    /// `d = op s` for the conversions that give an `xmm` whatever the source width.
    pub(super) fn vcvt_n(&mut self, op: VCvtN, wide: bool, d: Xmm, s: Xmm) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_cvt_n(op, d, s)?,
            Form::X => self.a.vex_cvt_n(op, d, s, VLen::X128)?,
            Form::Y => self.a.vex_cvt_n(op, d, y(s), VLen::X256)?,
        }
        Ok(())
    }

    /// `d = d shift imm` (per lane, or `pslldq` / `psrldq` bytes per 128 bits).
    pub(super) fn vsh(&mut self, op: VShiftI, wide: bool, d: Xmm, imm: u8) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_shift_i(op, d, imm),
            Form::X => self.a.vex_shift_i(op, d, d, imm)?,
            Form::Y => self.a.vex_shift_i(op, y(d), y(d), imm)?,
        }
        Ok(())
    }

    /// `d = s` (whole register).
    pub(super) fn vmov(&mut self, wide: bool, d: Xmm, s: Xmm) -> Res<()> {
        if d == s { Ok(()) } else { self.vu(VUn::Movdqa, wide, d, s) }
    }

    /// `g = sign bits of s`.
    pub(super) fn vmsk(&mut self, op: VMovMsk, wide: bool, g: Gpr, s: Xmm) -> Res<()> {
        match self.vform(wide)? {
            Form::Sse => self.a.sse_movmsk(op, g, s),
            Form::X => self.a.vex_movmsk(op, g, s),
            Form::Y => self.a.vex_movmsk(op, g, y(s)),
        }
        Ok(())
    }

    /// `d = 0` (the whole `ymm`).
    pub(super) fn vzero(&mut self, d: Xmm) -> Res<()> {
        self.vb(VBin::Pxor, false, d, d)
    }

    /// `d = all ones`.
    pub(super) fn vones(&mut self, wide: bool, d: Xmm) -> Res<()> {
        self.vb(VBin::Pcmpeqd, wide, d, d)
    }

    /// `x = g` (`movd` / `movq`, zero above).
    pub(super) fn vmovd_xg(&mut self, s: Size, x: Xmm, g: Gpr) -> Res<()> {
        if self.vex() {
            self.a.vex_movd_x(s, x, g)?
        } else {
            self.a.sse_movd_x(s, x, g)?
        }
        Ok(())
    }

    /// `g = low lane of x` (`movd` / `movq`).
    pub(super) fn vmovd_gx(&mut self, s: Size, g: Gpr, x: Xmm) -> Res<()> {
        if self.vex() {
            self.a.vex_movd_r(s, g, x)?
        } else {
            self.a.sse_movd_r(s, g, x)?
        }
        Ok(())
    }

    /// Flags of `ptest a, b`: ZF = `a & b` is zero.
    pub(super) fn vptest(&mut self, wide: bool, a: Xmm, b: Xmm) -> Res<()> {
        self.vu(VUn::Ptest, wide, a, b)
    }

    /// The upper 128 bits of `ymm s` into `xmm d` (the upper half of `ymm d` becomes zero).
    pub(super) fn vhigh(&mut self, d: Xmm, s: Xmm) -> Res<()> {
        self.vform(true)?;
        Ok(self.a.vex_extract(VExtract::Extracti128, d, y(s), 1)?)
    }

    // ---------------------------------------------------------------- values

    /// Loads the `size` bytes of a vector at `m` into `x`; the bytes above are zero. Uses `S0`
    /// for a 2-byte value.
    pub(crate) fn vload(&mut self, x: Xmm, m: Mem, size: u32) -> Res<()> {
        let f = self.vform(size > 16)?;
        let sse = f == Form::Sse;
        match size {
            32 => self.a.vex_un(VUn::Movdqu, y(x), m)?,
            16 if sse => self.a.sse_un(VUn::Movdqu, x, m)?,
            16 => self.a.vex_un(VUn::Movdqu, x, m)?,
            8 if sse => self.a.sse_un(VUn::Movq, x, m)?,
            8 => self.a.vex_un(VUn::Movq, x, m)?,
            4 if sse => self.a.sse_movd_x(Size::D, x, m)?,
            4 => self.a.vex_movd_x(Size::D, x, m)?,
            2 => {
                self.a.movzx(Size::D, S0, Size::W, m)?;
                self.vmovd_xg(Size::D, x, S0)?;
            }
            _ => return Err(bad("internal: vector size")),
        }
        Ok(())
    }

    /// Stores the low `size` bytes of `x` at `m`.
    pub(crate) fn vstore(&mut self, m: Mem, x: Xmm, size: u32) -> Res<()> {
        let f = self.vform(size > 16)?;
        let sse = f == Form::Sse;
        match size {
            32 => self.a.vex_store(VStore::Movdqu, m, y(x))?,
            16 if sse => self.a.sse_store(VStore::Movdqu, m, x)?,
            16 => self.a.vex_store(VStore::Movdqu, m, x)?,
            8 if sse => self.a.sse_store(VStore::Movq, m, x)?,
            8 => self.a.vex_store(VStore::Movq, m, x)?,
            4 if sse => self.a.sse_movd_r(Size::D, m, x)?,
            4 => self.a.vex_movd_r(Size::D, m, x)?,
            2 if sse => self.a.sse_pextr(PExtr::Pextrw, m, x, 0)?,
            2 => self.a.vex_pextr(PExtr::Pextrw, m, x, 0)?,
            _ => return Err(bad("internal: vector size")),
        }
        Ok(())
    }

    /// The home of a vector operand, holding its bytes (written here from the value's register
    /// when needed: call it where the code runs on every path through the op).
    pub(super) fn vmem(&mut self, o: Operand) -> Res<Mem> {
        match self.msrc(o)? {
            MSrc::Mem(m) => Ok(m),
            _ => Err(bad("vector operand must be a value")),
        }
    }

    /// A vector operand in a register the op may overwrite.
    pub(super) fn vget(&mut self, o: Operand, size: u32) -> Res<Xmm> {
        if self.rel {
            return self.vec_take(o);
        }
        let m = self.vmem(o)?;
        let x = self.alloc_x()?;
        self.vload(x, m, size)?;
        Ok(x)
    }

    /// A vector operand in a register the op must not write. An op reads its other operands
    /// this way before it takes the one it overwrites ([`Self::vget`]), so the register it
    /// computes into can be the one a block param is about to leave.
    pub(super) fn vread(&mut self, o: Operand, size: u32) -> Res<Xmm> {
        if self.rel { self.vec_read(o) } else { self.vget(o, size) }
    }

    /// [`Self::vread`], or the memory of the load [`Self::fold_vload`] left to this op.
    pub(super) fn vsrc(&mut self, o: Operand, sh: Sh) -> Res<VSrc> {
        if let Operand::Value(v) = o
            && let Some(&j) = self.folded.get(v.index())
            && j != super::analysis::NONE
        {
            let b = self.b;
            let &[s, i] = b.args(j) else { return Err(bad("internal: folded vector load")) };
            return Ok(VSrc::M(self.vslice(s, i, sh.w)?));
        }
        Ok(VSrc::R(self.vread(o, sh.size)?))
    }

    /// Defines the vector result `v` from register `x`, which the op allocated or took.
    pub(super) fn vput(&mut self, v: Value, x: Xmm, size: u32) -> Res<()> {
        if self.rel {
            return self.vec_bind(v, x, size);
        }
        let d = self.def_mem(v)?;
        self.vstore(d, x, size)
    }

    /// Zeroes the bytes of `x` above a value of `size` bytes.
    pub(crate) fn vnorm(&mut self, x: Xmm, size: u32) -> Res<()> {
        match size {
            16.. => Ok(()),
            8 => self.vu(VUn::Movq, false, x, x),
            _ => {
                let up = (16 - size) as u8;
                self.vsh(VShiftI::Pslldq, false, x, up)?;
                self.vsh(VShiftI::Psrldq, false, x, up)
            }
        }
    }

    // ---------------------------------------------------------------- constants

    /// The address of a constant in the function's pool.
    fn vpool_mem(&mut self, bytes: Vec<u8>) -> Mem {
        if let Some((l, _)) = self.vpool.iter().find(|(_, b)| *b == bytes) {
            return Mem::rip_label(*l, 0);
        }
        let l = self.a.new_label();
        self.vpool.push((l, bytes));
        Mem::rip_label(l, 0)
    }

    /// `d = bytes`: 16 bytes (repeated in the upper half of a `ymm`) or 32.
    pub(super) fn vconst(&mut self, d: Xmm, wide: bool, bytes: &[u8]) -> Res<()> {
        let mut v = bytes.to_vec();
        if wide && v.len() == 16 {
            v.extend_from_slice(bytes);
        }
        let size = if wide { 32 } else { 16 };
        if v.len() != size {
            return Err(bad("internal: vector constant size"));
        }
        let m = self.vpool_mem(v);
        self.vload(d, m, size as u32)
    }

    /// `d = value` in every lane of `w` bytes.
    pub(super) fn vsplat(&mut self, d: Xmm, wide: bool, w: u32, value: u64) -> Res<()> {
        let lane = value.to_le_bytes();
        let mut b = Vec::with_capacity(16);
        for _ in 0..16 / w {
            b.extend_from_slice(&lane[..w as usize]);
        }
        self.vconst(d, wide, &b)
    }

    /// Copies lane 0 (`w` bytes) of `x` into every lane.
    pub(super) fn vbroadcast(&mut self, w: u32, wide: bool, x: Xmm) -> Res<()> {
        if self.vex() {
            let op = by_w(w, [VUn::Pbroadcastb, VUn::Pbroadcastw, VUn::Pbroadcastd, VUn::Pbroadcastq]);
            return self.vu_x(op, wide, x, x);
        }
        match w {
            1 => {
                self.vzero(XS0)?;
                self.vb(VBin::Pshufb, false, x, XS0)
            }
            2 => {
                self.vui(VUnI::Pshuflw, false, x, x, 0)?;
                self.vb(VBin::Punpcklqdq, false, x, x)
            }
            4 => self.vui(VUnI::Pshufd, false, x, x, 0),
            _ => self.vb(VBin::Punpcklqdq, false, x, x),
        }
    }

    /// `d = m ? s : d` for a full-lane mask `m`; `t` is clobbered.
    pub(super) fn vblend(&mut self, wide: bool, d: Xmm, s: Xmm, m: Xmm, t: Xmm) -> Res<()> {
        self.vmov(wide, t, s)?;
        self.vb(VBin::Pxor, wide, t, d)?;
        self.vb(VBin::Pand, wide, t, m)?;
        self.vb(VBin::Pxor, wide, d, t)
    }

    /// `d = d > s` per lane of `w` bytes as a full-lane mask. Unsigned: clobbers `XS1`.
    pub(super) fn vcmp_gt(&mut self, w: u32, signed: bool, wide: bool, d: Xmm, s: Xmm) -> Res<()> {
        let gt = by_w(w, PCMPGT);
        if signed {
            return self.vb(gt, wide, d, s);
        }
        self.vsplat(XS1, wide, w, 1u64 << (8 * w - 1))?;
        self.vb(VBin::Pxor, wide, d, XS1)?;
        self.vb(VBin::Pxor, wide, XS1, s)?;
        self.vb(gt, wide, d, XS1)
    }

    // ---------------------------------------------------------------- dispatch

    pub(crate) fn vec_op(
        &mut self,
        op: VecOp,
        args: &[Operand],
        ty: Option<TypeId>,
        res: Option<Value>,
    ) -> Res<()> {
        if let Some(v) = res
            && !self.an.used.get(v.index()).copied().unwrap_or(true)
            && op.trap_kinds().is_empty()
            && !op.is_memory()
        {
            return Ok(());
        }
        let r = self.vec_emit(op, args, ty, res);
        if !(self.rel && self.ywide) {
            self.vclean();
        }
        r
    }

    fn vec_emit(&mut self, op: VecOp, args: &[Operand], ty: Option<TypeId>, res: Option<Value>) -> Res<()> {
        use VecOp::*;
        let arg =
            |k: usize| args.get(k).copied().ok_or_else(|| bad(format!("{}: missing operand {k}", op.name())));
        let rv = res.ok_or_else(|| bad(format!("{}: missing result", op.name())));
        let rt = ty.ok_or_else(|| bad(format!("{}: missing result type", op.name())));
        match op {
            Splat => self.v_splat(arg(0)?, rt?, rv?),
            Add | Sub | Mul | Div | MinX86 | MaxX86 => self.v_float_bin(op, arg(0)?, arg(1)?, rt?, rv?),
            Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc => self.v_float_un(op, arg(0)?, rt?, rv?),
            Copysign => self.v_copysign(arg(0)?, arg(1)?, rt?, rv?),
            Fma => self.v_fma(arg(0)?, arg(1)?, arg(2)?, rt?, rv?),
            Min | Max | MinNum | MaxNum => self.v_minmax(op, arg(0)?, arg(1)?, rt?, rv?),
            AddWrap | SubWrap | MulWrap | And | Or | Xor | Andnot | MaskAnd | MaskOr | MaskXor => {
                self.v_int_bin(op, arg(0)?, arg(1)?, rt?, rv?)
            }
            AddSat | SubSat => self.v_sat(op, arg(0)?, arg(1)?, rt?, rv?),
            MulSat => self.v_mul_sat(arg(0)?, arg(1)?, rt?, rv?),
            CheckedAdd | CheckedSub | CheckedMul => self.v_checked(op, arg(0)?, arg(1)?, rt?, rv?),
            DivLanes | DivConst => self.v_div(op, arg(0)?, arg(1)?, rt?, rv?),
            AbsWrap | Popcount | Clz | Not | MaskNot => self.v_int_un(op, arg(0)?, rt?, rv?),
            Shl | Shr | ShlWrap => self.v_shift(op, arg(0)?, arg(1)?, rt?, rv?),
            Eq | Ne | Lt | Le | Gt | Ge => self.v_compare(op, arg(0)?, arg(1)?, rv?),
            ReduceAdd | ReduceAddWrap | ReduceMin | ReduceMax | ReduceAnd | ReduceOr => {
                self.v_reduce(op, arg(0)?, rv?)
            }
            ReduceAddOrdered => self.v_reduce_ordered(arg(0)?, arg(1)?, rv?),
            ScanAdd | ScanAddWrap => self.v_scan(arg(0)?, rt?, rv?),
            Cast(kind) => self.v_cast(kind, arg(0)?, rt?, rv?),
            Bitcast | ToVec => self.v_copy(arg(0)?, rt?, rv?),
            Split => self.v_split(arg(0)?, rt?, rv?),
            Join => self.v_join(arg(0)?, arg(1)?, rt?, rv?),
            WidenLo | WidenHi => self.v_widen(op == WidenHi, arg(0)?, rt?, rv?),
            NarrowWrap | NarrowSat => self.v_narrow(op == NarrowSat, arg(0)?, arg(1)?, rt?, rv?),
            Permute => self.v_permute(args, rt?, rv?),
            Lookup => self.v_lookup(arg(0)?, arg(1)?, rt?, rv?),
            Compress | Expand => self.v_compress(op == Expand, arg(0)?, arg(1)?, rt?, rv?),
            Load | LoadAligned => self.v_load(op == LoadAligned, arg(0)?, arg(1)?, rt?, rv?),
            Store | StoreAligned => self.v_store(op == StoreAligned, arg(0)?, arg(1)?, arg(2)?),
            LoadMasked => self.v_load_masked(arg(0)?, arg(1)?, arg(2)?, rt?, rv?),
            StoreMasked => self.v_store_masked(arg(0)?, arg(1)?, arg(2)?, arg(3)?),
            Gather => self.v_gather(arg(0)?, arg(1)?, rt?, rv?),
            Scatter => self.v_scatter(arg(0)?, arg(1)?, arg(2)?),
            Select => self.v_select(arg(0)?, arg(1)?, arg(2)?, rt?, rv?),
            ToBits => self.v_to_bits(arg(0)?, rv?),
            FirstN => self.v_first_n(arg(0)?, rt?, rv?),
            Rewiden => self.v_rewiden(arg(0)?, rt?, rv?),
            MaskAll | MaskNone => {
                let sh = self.vshape(rt?)?;
                let x = self.alloc_x()?;
                if op == MaskAll {
                    self.vones(sh.wide(), x)?;
                } else {
                    self.vzero(x)?;
                }
                self.vput(rv?, x, sh.size)
            }
        }
    }

    /// The float precision of a shape's lanes.
    pub(super) fn vprec(&self, sh: Sh) -> Res<Prec> {
        sh.float().ok_or_else(|| unsup("float vector op on other lanes"))
    }

    /// The int type of a shape's lanes.
    pub(super) fn vint(&self, sh: Sh) -> Res<IntTy> {
        sh.int().ok_or_else(|| bad("int vector op on float lanes"))
    }
}
