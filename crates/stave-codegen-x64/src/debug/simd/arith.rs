//! Lane arithmetic: float and int ops, shifts, compares, reductions, scans.

use stave_ir::arith::IntTy;
use stave_ir::simd::{LaneTy, VecOp};
use stave_ir::{Operand, PanicKind, TypeId, Value};

use super::{PADD, PCMPEQ, PCMPGT, PSUB, Sh, by_w, lane, size_of, y};
use crate::debug::func::*;
use crate::debug::{Res, bad};
use crate::enc::*;

/// Bit counts of the nibbles 0..16.
const POPCOUNT_LUT: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
/// Leading zeros of the nibbles 0..16 within 4 bits.
const CLZ_LUT: [u8; 16] = [4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

fn ps_pd(p: Prec, ps: VBin, pd: VBin) -> VBin {
    if p == Prec::F32 { ps } else { pd }
}

impl Cg<'_, '_> {
    pub(super) fn v_splat(&mut self, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let x = self.alloc_x()?;
        match sh.float() {
            Some(p) => {
                let s = self.read_x(a, p)?;
                self.vmov(false, x, s)?;
            }
            None => {
                let g = self.read_g(a)?;
                self.vmovd_xg(if sh.w == 8 { Size::Q } else { Size::D }, x, g)?;
            }
        }
        self.vbroadcast(sh.w, sh.wide(), x)?;
        self.vput(rv, x, sh.size)
    }

    pub(super) fn v_float_bin(
        &mut self,
        op: VecOp,
        a: Operand,
        b: Operand,
        rt: TypeId,
        rv: Value,
    ) -> Res<()> {
        let sh = self.vshape(rt)?;
        let p = self.vprec(sh)?;
        let o = match op {
            VecOp::Add => ps_pd(p, VBin::Addps, VBin::Addpd),
            VecOp::Sub => ps_pd(p, VBin::Subps, VBin::Subpd),
            VecOp::Mul => ps_pd(p, VBin::Mulps, VBin::Mulpd),
            VecOp::Div => ps_pd(p, VBin::Divps, VBin::Divpd),
            VecOp::MinX86 => ps_pd(p, VBin::Minps, VBin::Minpd),
            _ => ps_pd(p, VBin::Maxps, VBin::Maxpd),
        };
        let s = self.vsrc(b, sh)?;
        let x = self.vget(a, sh.size)?;
        self.vb_rm(o, sh.wide(), x, s)?;
        self.vput(rv, x, sh.size)
    }

    pub(super) fn v_float_un(&mut self, op: VecOp, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let p = self.vprec(sh)?;
        let wide = sh.wide();
        let x = self.vget(a, sh.size)?;
        let round = if p == Prec::F32 { VUnI::Roundps } else { VUnI::Roundpd };
        match op {
            VecOp::Neg => {
                self.vsplat(XS0, wide, sh.w, sh.sign())?;
                self.vb(VBin::Pxor, wide, x, XS0)?;
            }
            VecOp::Abs => {
                self.vsplat(XS0, wide, sh.w, sh.ones() ^ sh.sign())?;
                self.vb(VBin::Pand, wide, x, XS0)?;
            }
            VecOp::Sqrt => self.vu(if p == Prec::F32 { VUn::Sqrtps } else { VUn::Sqrtpd }, wide, x, x)?,
            // imm8: bit 3 = no precision exception; bits 1:0 = nearest even, down, up, toward zero
            VecOp::Round => self.vui(round, wide, x, x, 8)?,
            VecOp::Floor => self.vui(round, wide, x, x, 9)?,
            VecOp::Ceil => self.vui(round, wide, x, x, 10)?,
            _ => self.vui(round, wide, x, x, 11)?,
        }
        self.vput(rv, x, sh.size)
    }

    pub(super) fn v_copysign(&mut self, a: Operand, s: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let wide = sh.wide();
        let x = self.vget(a, sh.size)?;
        let sx = self.vget(s, sh.size)?;
        self.vsplat(XS0, wide, sh.w, sh.sign())?;
        self.vb(VBin::Pand, wide, sx, XS0)?;
        self.vb(VBin::Pandn, wide, XS0, x)?;
        self.vb(VBin::Por, wide, XS0, sx)?;
        self.vput(rv, XS0, sh.size)
    }

    pub(super) fn v_fma(&mut self, a: Operand, b: Operand, c: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let p = self.vprec(sh)?;
        if self.fx.fma {
            let (bx, cx) = (self.vread(b, sh.size)?, self.vread(c, sh.size)?);
            let x = self.vget(a, sh.size)?;
            // x = bx * x + cx
            self.vb3(ps_pd(p, VBin::Fmadd213ps, VBin::Fmadd213pd), sh.wide(), x, bx, cx)?;
            return self.vput(rv, x, sh.size);
        }
        // No FMA instruction: one call of the software routine per lane.
        let syms = self.cx.fma_sym.ok_or_else(|| bad("internal: no software fma in the image"))?;
        let sym = if p == Prec::F32 { syms[0] } else { syms[1] };
        let (ma, mb, mc) = (self.vmem(a)?, self.vmem(b)?, self.vmem(c)?);
        self.spill_caller_saved()?;
        // the routine is baseline code
        self.vclean();
        let d = self.def_mem(rv)?;
        for k in 0..sh.n {
            for (n, m) in [(0, ma), (1, mb), (2, mc)] {
                self.a.movs(p, Xmm::masked(n), lane(m, k, sh.w)?)?;
            }
            self.a.call_sym(sym);
            self.a.movs_store(p, lane(d, k, sh.w)?, Xmm::masked(0))?;
        }
        Ok(())
    }

    /// `a = minimum(a, b)` / `maximum(a, b)`: a NaN in either gives a quiet NaN, -0 is below +0.
    /// `minps` returns its second operand for an unordered or equal pair, so the two operand
    /// orders differ only for (NaN, x) and (±0, ∓0): OR (min) or AND (max) of the two picks the
    /// zero by sign, and unordered lanes are set to all ones. Clobbers `XS0`, `XS1`.
    fn vfminmax(&mut self, wide: bool, p: Prec, min: bool, a: Xmm, b: Xmm) -> Res<()> {
        let m = if min { ps_pd(p, VBin::Minps, VBin::Minpd) } else { ps_pd(p, VBin::Maxps, VBin::Maxpd) };
        let cmp = if p == Prec::F32 { VBinI::Cmpps } else { VBinI::Cmppd };
        self.vmov(wide, XS0, b)?;
        self.vb(m, wide, XS0, a)?;
        self.vmov(wide, XS1, a)?;
        self.vbi(cmp, wide, XS1, b, CmpPred::Unord as u8)?;
        self.vb(m, wide, a, b)?;
        self.vb(if min { VBin::Por } else { VBin::Pand }, wide, a, XS0)?;
        self.vb(VBin::Por, wide, a, XS1)
    }

    /// `a = min(a, b)` / `max(a, b)` of int lanes; on equal lanes either (same bits). Clobbers
    /// `XS0`, `XS1` for 64-bit lanes.
    pub(super) fn viminmax(
        &mut self,
        w: u32,
        signed: bool,
        wide: bool,
        min: bool,
        a: Xmm,
        b: Xmm,
    ) -> Res<()> {
        if w < 8 {
            let k = w.trailing_zeros() as usize;
            let op = match (min, signed) {
                (true, true) => [VBin::Pminsb, VBin::Pminsw, VBin::Pminsd][k],
                (true, false) => [VBin::Pminub, VBin::Pminuw, VBin::Pminud][k],
                (false, true) => [VBin::Pmaxsb, VBin::Pmaxsw, VBin::Pmaxsd][k],
                (false, false) => [VBin::Pmaxub, VBin::Pmaxuw, VBin::Pmaxud][k],
            };
            return self.vb(op, wide, a, b);
        }
        // XS0 = a > b; a ^= (a ^ b) where b is to be taken
        self.vmov(wide, XS0, a)?;
        self.vcmp_gt(w, signed, wide, XS0, b)?;
        self.vmov(wide, XS1, b)?;
        self.vb(VBin::Pxor, wide, XS1, a)?;
        if min {
            self.vb(VBin::Pand, wide, XS0, XS1)?;
        } else {
            self.vb(VBin::Pandn, wide, XS0, XS1)?;
        }
        self.vb(VBin::Pxor, wide, a, XS0)
    }

    pub(super) fn v_minmax(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let wide = sh.wide();
        let min = matches!(op, VecOp::Min | VecOp::MinNum);
        let s = self.vread(b, sh.size)?;
        let x = self.vget(a, sh.size)?;
        let t = match sh.lane {
            LaneTy::Int(t) => {
                self.viminmax(sh.w, t.signed, wide, min, x, s)?;
                return self.vput(rv, x, sh.size);
            }
            LaneTy::Float(_) => self.vprec(sh)?,
        };
        if matches!(op, VecOp::Min | VecOp::Max) {
            self.vfminmax(wide, t, min, x, s)?;
            return self.vput(rv, x, sh.size);
        }
        // min_num / max_num: a NaN lane is missing; both NaN give a quiet NaN.
        let m = if min { ps_pd(t, VBin::Minps, VBin::Minpd) } else { ps_pd(t, VBin::Maxps, VBin::Maxpd) };
        let cmp = if t == Prec::F32 { VBinI::Cmpps } else { VBinI::Cmppd };
        let (r, na) = (self.alloc_x()?, self.alloc_x()?);
        self.vmov(wide, XS0, s)?;
        self.vb(m, wide, XS0, x)?;
        self.vmov(wide, r, x)?;
        self.vb(m, wide, r, s)?;
        self.vb(if min { VBin::Por } else { VBin::Pand }, wide, r, XS0)?;
        self.vmov(wide, na, x)?;
        self.vbi(cmp, wide, na, x, CmpPred::Unord as u8)?;
        self.vmov(wide, XS1, s)?;
        self.vbi(cmp, wide, XS1, s, CmpPred::Unord as u8)?;
        self.vblend(wide, r, x, XS1, XS0)?;
        self.vblend(wide, r, s, na, XS0)?;
        self.vb(VBin::Pand, wide, na, XS1)?;
        self.vb(VBin::Por, wide, r, na)?;
        self.vput(rv, r, sh.size)
    }

    /// `a = a * b` (wrapping). Clobbers `XS0`, `XS1`.
    fn vmul_wrap(&mut self, sh: Sh, a: Xmm, b: Xmm) -> Res<()> {
        let wide = sh.wide();
        match sh.w {
            1 => {
                // odd bytes as words, shifted back up; even bytes are the low bytes of the
                // word products
                let u = self.alloc_x()?;
                self.vmov(wide, XS0, a)?;
                self.vsh(VShiftI::Psrlw, wide, XS0, 8)?;
                self.vmov(wide, u, b)?;
                self.vsh(VShiftI::Psrlw, wide, u, 8)?;
                self.vb(VBin::Pmullw, wide, XS0, u)?;
                self.vsh(VShiftI::Psllw, wide, XS0, 8)?;
                self.vb(VBin::Pmullw, wide, a, b)?;
                self.vsplat(XS1, wide, 2, 0x00ff)?;
                self.vb(VBin::Pand, wide, a, XS1)?;
                self.vb(VBin::Por, wide, a, XS0)
            }
            2 => self.vb(VBin::Pmullw, wide, a, b),
            4 => self.vb(VBin::Pmulld, wide, a, b),
            _ => {
                // lo(a) lo(b) + ((hi(a) lo(b) + hi(b) lo(a)) << 32)
                self.vmov(wide, XS0, a)?;
                self.vsh(VShiftI::Psrlq, wide, XS0, 32)?;
                self.vb(VBin::Pmuludq, wide, XS0, b)?;
                self.vmov(wide, XS1, b)?;
                self.vsh(VShiftI::Psrlq, wide, XS1, 32)?;
                self.vb(VBin::Pmuludq, wide, XS1, a)?;
                self.vb(VBin::Paddq, wide, XS0, XS1)?;
                self.vsh(VShiftI::Psllq, wide, XS0, 32)?;
                self.vb(VBin::Pmuludq, wide, a, b)?;
                self.vb(VBin::Paddq, wide, a, XS0)
            }
        }
    }

    pub(super) fn v_int_bin(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let wide = sh.wide();
        if op == VecOp::Andnot {
            // a & ~b, computed into `b`'s register
            let x = self.vread(a, sh.size)?;
            let s = self.vget(b, sh.size)?;
            self.vb(VBin::Pandn, wide, s, x)?;
            return self.vput(rv, s, sh.size);
        }
        if op == VecOp::MulWrap {
            let s = self.vread(b, sh.size)?;
            let x = self.vget(a, sh.size)?;
            self.vmul_wrap(sh, x, s)?;
            return self.vput(rv, x, sh.size);
        }
        let s = self.vsrc(b, sh)?;
        let x = self.vget(a, sh.size)?;
        let o = match op {
            VecOp::AddWrap => by_w(sh.w, PADD),
            VecOp::SubWrap => by_w(sh.w, PSUB),
            VecOp::And | VecOp::MaskAnd => VBin::Pand,
            VecOp::Or | VecOp::MaskOr => VBin::Por,
            _ => VBin::Pxor,
        };
        self.vb_rm(o, wide, x, s)?;
        self.vput(rv, x, sh.size)
    }

    /// The wrapped sum or difference of signed lanes and a vector whose lane sign bits are set
    /// where it overflowed. Clobbers `XS0`.
    fn vsigned_ovf(&mut self, sh: Sh, add: bool, a: Xmm, b: Xmm) -> Res<(Xmm, Xmm)> {
        let wide = sh.wide();
        let (s, o) = (self.alloc_x()?, self.alloc_x()?);
        self.vmov(wide, s, a)?;
        self.vb(by_w(sh.w, if add { PADD } else { PSUB }), wide, s, b)?;
        self.vmov(wide, o, a)?;
        if add {
            // operands of one sign, the sum of the other
            self.vb(VBin::Pxor, wide, o, s)?;
            self.vmov(wide, XS0, b)?;
        } else {
            // operands of different signs, the difference of another sign than `a`
            self.vb(VBin::Pxor, wide, o, b)?;
            self.vmov(wide, XS0, a)?;
        }
        self.vb(VBin::Pxor, wide, XS0, s)?;
        self.vb(VBin::Pand, wide, o, XS0)?;
        Ok((s, o))
    }

    pub(super) fn v_sat(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let wide = sh.wide();
        let add = op == VecOp::AddSat;
        let x = self.vget(a, sh.size)?;
        let s = self.vget(b, sh.size)?;
        if sh.w <= 2 {
            let k = (sh.w - 1) as usize;
            let o = match (add, t.signed) {
                (true, true) => [VBin::Paddsb, VBin::Paddsw][k],
                (true, false) => [VBin::Paddusb, VBin::Paddusw][k],
                (false, true) => [VBin::Psubsb, VBin::Psubsw][k],
                (false, false) => [VBin::Psubusb, VBin::Psubusw][k],
            };
            self.vb(o, wide, x, s)?;
            return self.vput(rv, x, sh.size);
        }
        if !t.signed {
            let r = self.alloc_x()?;
            self.vmov(wide, r, x)?;
            if add {
                // a + b wrapped is below a: all ones
                self.vb(by_w(sh.w, PADD), wide, r, s)?;
                self.vcmp_gt(sh.w, false, wide, x, r)?;
                self.vb(VBin::Por, wide, r, x)?;
                return self.vput(rv, r, sh.size);
            }
            // b above a: zero
            self.vb(by_w(sh.w, PSUB), wide, r, s)?;
            self.vcmp_gt(sh.w, false, wide, s, x)?;
            self.vb(VBin::Pandn, wide, s, r)?;
            return self.vput(rv, s, sh.size);
        }
        let (r, o) = self.vsigned_ovf(sh, add, x, s)?;
        let gt = by_w(sh.w, PCMPGT);
        self.vzero(XS0)?;
        self.vb(gt, wide, XS0, o)?;
        // the bound on `a`'s side: MAX, or MIN = !MAX for a negative `a`
        self.vzero(o)?;
        self.vb(gt, wide, o, x)?;
        self.vsplat(XS1, wide, sh.w, sh.sign() - 1)?;
        self.vb(VBin::Pxor, wide, o, XS1)?;
        self.vblend(wide, r, o, XS0, XS1)?;
        self.vput(rv, r, sh.size)
    }

    /// Lane-by-lane multiply of the vectors at `ma`, `mb` into `d`: saturating, or jumping to
    /// `fail` on the first overflow. `rax` and `rdx` must be reserved.
    fn vmul_lanes(&mut self, sh: Sh, t: IntTy, ma: Mem, mb: Mem, d: Mem, fail: Option<Label>) -> Res<()> {
        let wb = sh.w as u8;
        for k in 0..sh.n {
            let (la, lb) = (lane(ma, k, sh.w)?, lane(mb, k, sh.w)?);
            self.sized_load(RAX, la, wb, t.signed)?;
            self.sized_load(RDX, lb, wb, t.signed)?;
            if t.bits < 64 {
                // exact in 64 bits
                self.a.imul_rr(Size::Q, RAX, RDX)?;
                match fail {
                    None => self.clamp(RAX, t, t.signed, true, t.signed)?,
                    Some(l) => {
                        self.norm_into(S0, RAX, t)?;
                        self.a.alu_rr(AluOp::Cmp, Size::Q, S0, RAX);
                        self.a.jcc(Cond::Ne, l)?;
                    }
                }
            } else if t.signed {
                self.a.imul_rr(Size::Q, RAX, RDX)?;
                match fail {
                    Some(l) => self.a.jcc(Cond::O, l)?,
                    None => {
                        let ok = self.a.new_label();
                        self.a.jcc(Cond::No, ok)?;
                        // MAX, or MIN = !MAX when the signs differ
                        self.a.mov_rm(Size::Q, RAX, la)?;
                        self.a.alu_rm(AluOp::Xor, Size::Q, RAX, lb)?;
                        self.a.shift_ri(ShiftOp::Sar, Size::Q, RAX, 63)?;
                        self.a.mov_ri(Size::Q, S0, i64::MAX)?;
                        self.a.alu_rr(AluOp::Xor, Size::Q, RAX, S0);
                        self.a.bind(ok)?;
                    }
                }
            } else {
                self.a.unary(UnaryOp::Mul, Size::Q, RDX)?;
                match fail {
                    Some(l) => self.a.jcc(Cond::C, l)?,
                    None => {
                        let ok = self.a.new_label();
                        self.a.jcc(Cond::NC, ok)?;
                        self.a.mov_ri(Size::Q, RAX, -1)?;
                        self.a.bind(ok)?;
                    }
                }
            }
            self.a.mov_mr(size_of(sh.w), lane(d, k, sh.w)?, RAX)?;
        }
        Ok(())
    }

    pub(super) fn v_mul_sat(&mut self, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let (ma, mb) = (self.vmem(a)?, self.vmem(b)?);
        self.reserve_g(RAX)?;
        self.reserve_g(RDX)?;
        let d = self.def_mem(rv)?;
        self.vmul_lanes(sh, t, ma, mb, d, None)
    }

    pub(super) fn v_checked(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let (none, some, pty) = self
            .cx
            .m
            .types
            .option_shape(rt)
            .ok_or_else(|| bad("checked vector op result must be an Option"))?;
        let sh = self.vshape(pty)?;
        let t = self.vint(sh)?;
        let wide = sh.wide();
        let (lnone, lend) = (self.a.new_label(), self.a.new_label());
        // Every register is taken before the first branch.
        if op == VecOp::CheckedMul {
            let (ma, mb) = (self.vmem(a)?, self.vmem(b)?);
            self.reserve_g(RAX)?;
            self.reserve_g(RDX)?;
            let (d, off, _) = self.option_home(rt, some, rv)?;
            self.vmul_lanes(sh, t, ma, mb, disp(d, off)?, Some(lnone))?;
            self.store_imm(d, 4, i64::from(some))?;
            self.a.jmp(lend)?;
            self.a.bind(lnone)?;
            self.store_imm(d, 4, i64::from(none))?;
            return Ok(self.a.bind(lend)?);
        }
        let add = op == VecOp::CheckedAdd;
        let (d, off, _) = self.option_home(rt, some, rv)?;
        let x = self.vget(a, sh.size)?;
        let s = self.vget(b, sh.size)?;
        // ZF = no lane overflowed
        let r = if t.signed {
            let (r, o) = self.vsigned_ovf(sh, add, x, s)?;
            self.vsplat(XS1, wide, sh.w, sh.sign())?;
            self.vptest(wide, o, XS1)?;
            r
        } else {
            let r = self.alloc_x()?;
            self.vmov(wide, r, x)?;
            if add {
                self.vb(by_w(sh.w, PADD), wide, r, s)?;
                self.vcmp_gt(sh.w, false, wide, x, r)?;
                self.vptest(wide, x, x)?;
            } else {
                self.vb(by_w(sh.w, PSUB), wide, r, s)?;
                self.vcmp_gt(sh.w, false, wide, s, x)?;
                self.vptest(wide, s, s)?;
            }
            r
        };
        self.a.jcc(Cond::Ne, lnone)?;
        self.vstore(disp(d, off)?, r, sh.size)?;
        self.store_imm(d, 4, i64::from(some))?;
        self.a.jmp(lend)?;
        self.a.bind(lnone)?;
        self.store_imm(d, 4, i64::from(none))?;
        Ok(self.a.bind(lend)?)
    }

    /// Truncating division lane by lane, traps in lane order: a zero divisor `DivByZero`, then
    /// `MIN / -1` `Overflow`.
    pub(super) fn v_div(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let wb = sh.w as u8;
        let ma = self.vmem(a)?;
        let konst = if op == VecOp::DivConst {
            let k = self.int_const(b).ok_or_else(|| bad("div_const divisor must be a constant"))?;
            Some(t.wrap(k) as i64)
        } else {
            None
        };
        if konst == Some(0) {
            self.def_mem(rv)?;
            return self.trap_now(PanicKind::DivByZero);
        }
        let mb = if konst.is_none() { Some(self.vmem(b)?) } else { None };
        self.reserve_g(RAX)?;
        self.reserve_g(RDX)?;
        let c = self.alloc_g()?;
        let d = self.def_mem(rv)?;
        if let Some(k) = konst {
            self.a.mov_ri(Size::Q, c, k)?;
        }
        let minus_one = t.signed && konst.is_none_or(|k| k == -1);
        for k in 0..sh.n {
            if let Some(mb) = mb {
                self.sized_load(c, lane(mb, k, sh.w)?, wb, t.signed)?;
                self.a.test_rr(Size::Q, c, c);
                self.jtrap(Cond::E, PanicKind::DivByZero)?;
            }
            self.sized_load(RAX, lane(ma, k, sh.w)?, wb, t.signed)?;
            if t.signed {
                if minus_one {
                    let ok = self.a.new_label();
                    self.a.alu_ri(AluOp::Cmp, Size::Q, c, -1)?;
                    self.a.jcc(Cond::Ne, ok)?;
                    self.a.mov_ri(Size::Q, S0, t.min() as i64)?;
                    self.a.alu_rr(AluOp::Cmp, Size::Q, RAX, S0);
                    self.jtrap(Cond::E, PanicKind::Overflow)?;
                    self.a.bind(ok)?;
                }
                self.a.cqo();
                self.a.unary(UnaryOp::Idiv, Size::Q, c)?;
            } else {
                self.a.alu_rr(AluOp::Xor, Size::D, RDX, RDX);
                self.a.unary(UnaryOp::Div, Size::Q, c)?;
            }
            self.a.mov_mr(size_of(sh.w), lane(d, k, sh.w)?, RAX)?;
        }
        Ok(())
    }

    /// Per-byte results of a nibble table: `(table[low nibble], table[high nibble], high
    /// nibble)` in (`lo`, `lut`, `XS0`). Clobbers `a`, `XS1`.
    fn vnibbles(&mut self, wide: bool, a: Xmm, table: &[u8; 16]) -> Res<(Xmm, Xmm)> {
        let (lut, lo) = (self.alloc_x()?, self.alloc_x()?);
        self.vconst(lut, wide, table)?;
        self.vsplat(XS1, wide, 1, 0x0f)?;
        self.vmov(wide, XS0, a)?;
        self.vsh(VShiftI::Psrlw, wide, XS0, 4)?;
        self.vb(VBin::Pand, wide, XS0, XS1)?;
        self.vb(VBin::Pand, wide, a, XS1)?;
        self.vmov(wide, lo, lut)?;
        self.vb(VBin::Pshufb, wide, lo, a)?;
        self.vb(VBin::Pshufb, wide, lut, XS0)?;
        Ok((lo, lut))
    }

    pub(super) fn v_int_un(&mut self, op: VecOp, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let wide = sh.wide();
        let x = self.vget(a, sh.size)?;
        let psrl = |w: u32| by_w(w, [VShiftI::Psrlw, VShiftI::Psrlw, VShiftI::Psrld, VShiftI::Psrlq]);
        let r = match op {
            VecOp::Not | VecOp::MaskNot => {
                self.vones(wide, XS0)?;
                self.vb(VBin::Pxor, wide, x, XS0)?;
                x
            }
            VecOp::AbsWrap => {
                if t.signed && sh.w < 8 {
                    self.vu(by_w(sh.w, [VUn::Pabsb, VUn::Pabsw, VUn::Pabsd, VUn::Pabsd]), wide, x, x)?;
                } else if t.signed {
                    // (a ^ s) - s with s = all ones for a negative lane
                    self.vzero(XS0)?;
                    self.vb(VBin::Pcmpgtq, wide, XS0, x)?;
                    self.vb(VBin::Pxor, wide, x, XS0)?;
                    self.vb(VBin::Psubq, wide, x, XS0)?;
                } else {
                    return Err(bad(format!("{} on these lanes", op.name())));
                }
                x
            }
            VecOp::Popcount => {
                let (c, hi) = self.vnibbles(wide, x, &POPCOUNT_LUT)?;
                self.vb(VBin::Paddb, wide, c, hi)?;
                // sums of adjacent halves, up to the lane width
                for v in [1, 2, 4] {
                    if sh.w > v {
                        self.vmov(wide, XS0, c)?;
                        self.vsh(psrl(2 * v), wide, XS0, (8 * v) as u8)?;
                        self.vb(by_w(2 * v, PADD), wide, c, XS0)?;
                        self.vsplat(XS1, wide, 2 * v, (1u64 << (8 * v)) - 1)?;
                        self.vb(VBin::Pand, wide, c, XS1)?;
                    }
                }
                c
            }
            _ => {
                // per byte: clz(high nibble), plus clz(low nibble) when the high one is zero
                let (lo, c) = self.vnibbles(wide, x, &CLZ_LUT)?;
                self.vzero(XS1)?;
                self.vb(VBin::Pcmpeqb, wide, XS0, XS1)?;
                self.vb(VBin::Pand, wide, lo, XS0)?;
                self.vb(VBin::Paddb, wide, c, lo)?;
                // per doubled lane: clz(high half), plus clz(low half) when the high half is
                // zero (its count is its width)
                for v in [1, 2, 4] {
                    if sh.w > v {
                        let w2 = 2 * v;
                        self.vmov(wide, XS0, c)?;
                        self.vsh(psrl(w2), wide, XS0, (8 * v) as u8)?;
                        self.vsplat(XS1, wide, w2, (1u64 << (8 * v)) - 1)?;
                        self.vb(VBin::Pand, wide, c, XS1)?;
                        self.vsplat(XS1, wide, w2, u64::from(8 * v))?;
                        self.vb(by_w(w2, PCMPEQ), wide, XS1, XS0)?;
                        self.vb(VBin::Pand, wide, c, XS1)?;
                        self.vb(by_w(w2, PADD), wide, c, XS0)?;
                    }
                }
                c
            }
        };
        self.vput(rv, r, sh.size)
    }

    /// `a = a shift c` with one count (below the lane width) in the low quadword of `c`.
    /// Clobbers `XS0`.
    fn vshift_by(&mut self, op: VecOp, sh: Sh, signed: bool, a: Xmm, c: Xmm) -> Res<()> {
        let wide = sh.wide();
        let right = op == VecOp::Shr;
        let arith = right && signed;
        let k = (sh.w.trailing_zeros() as usize).saturating_sub(1);
        let shl = [VBin::Psllw, VBin::Pslld, VBin::Psllq][k];
        let shr = [VBin::Psrlw, VBin::Psrld, VBin::Psrlq][k];
        match sh.w {
            1 => {
                // even and odd bytes as words, each shifted in its own word
                let m = self.alloc_x()?;
                self.vsplat(m, wide, 2, 0x00ff)?;
                self.vmov(wide, XS0, a)?;
                if arith {
                    self.vsh(VShiftI::Psraw, wide, XS0, 8)?;
                    self.vb_x(VBin::Psraw, wide, XS0, c)?;
                    self.vsh(VShiftI::Psllw, wide, a, 8)?;
                    self.vsh(VShiftI::Psraw, wide, a, 8)?;
                    self.vb_x(VBin::Psraw, wide, a, c)?;
                    self.vb(VBin::Pand, wide, a, m)?;
                } else if right {
                    self.vsh(VShiftI::Psrlw, wide, XS0, 8)?;
                    self.vb_x(VBin::Psrlw, wide, XS0, c)?;
                    self.vb(VBin::Pand, wide, a, m)?;
                    self.vb_x(VBin::Psrlw, wide, a, c)?;
                } else {
                    self.vsh(VShiftI::Psrlw, wide, XS0, 8)?;
                    self.vb_x(VBin::Psllw, wide, XS0, c)?;
                    self.vb(VBin::Pand, wide, a, m)?;
                    self.vb_x(VBin::Psllw, wide, a, c)?;
                    self.vb(VBin::Pand, wide, a, m)?;
                }
                self.vsh(VShiftI::Psllw, wide, XS0, 8)?;
                self.vb(VBin::Por, wide, a, XS0)
            }
            8 if arith => {
                // !(!a >> c) for a negative lane
                self.vzero(XS0)?;
                self.vb(VBin::Pcmpgtq, wide, XS0, a)?;
                self.vb(VBin::Pxor, wide, a, XS0)?;
                self.vb_x(VBin::Psrlq, wide, a, c)?;
                self.vb(VBin::Pxor, wide, a, XS0)
            }
            _ if arith => self.vb_x(if sh.w == 2 { VBin::Psraw } else { VBin::Psrad }, wide, a, c),
            _ => self.vb_x(if right { shr } else { shl }, wide, a, c),
        }
    }

    pub(super) fn v_shift(&mut self, op: VecOp, a: Operand, amt: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let wide = sh.wide();
        let bits = i64::from(sh.bits());
        let trap = op != VecOp::ShlWrap;
        if self.oty(amt) != Some(rt) {
            // one amount for every lane
            let x = self.vget(a, sh.size)?;
            if let Some(k) = self.int_const(amt) {
                if trap && !(0..i128::from(bits)).contains(&k) {
                    self.vput(rv, x, sh.size)?;
                    return self.trap_now(PanicKind::Overflow);
                }
                self.a.mov_ri(Size::D, S0, (k & i128::from(bits - 1)) as i64)?;
                self.vmovd_xg(Size::D, XS1, S0)?;
            } else {
                let g = self.read_g(amt)?;
                if trap {
                    self.a.alu_ri(AluOp::Cmp, Size::Q, g, bits)?;
                    self.jtrap(Cond::Ae, PanicKind::Overflow)?;
                    self.vmovd_xg(Size::Q, XS1, g)?;
                } else {
                    self.a.mov_rr(Size::D, S0, g);
                    self.a.alu_ri(AluOp::And, Size::D, S0, bits - 1)?;
                    self.vmovd_xg(Size::D, XS1, S0)?;
                }
            }
            self.vshift_by(op, sh, t.signed, x, XS1)?;
            return self.vput(rv, x, sh.size);
        }
        if self.fx.avx2 && sh.w >= 4 {
            let x = self.vget(a, sh.size)?;
            let s = self.vget(amt, sh.size)?;
            self.vsplat(XS0, wide, sh.w, (bits - 1) as u64)?;
            if trap {
                let over = self.alloc_x()?;
                self.vmov(wide, over, s)?;
                self.vcmp_gt(sh.w, false, wide, over, XS0)?;
                self.vptest(wide, over, over)?;
                self.jtrap(Cond::Ne, PanicKind::Overflow)?;
            } else {
                self.vb(VBin::Pand, wide, s, XS0)?;
            }
            let q = sh.w == 8;
            match op {
                VecOp::Shr if t.signed && q => {
                    self.vzero(XS0)?;
                    self.vb(VBin::Pcmpgtq, wide, XS0, x)?;
                    self.vb(VBin::Pxor, wide, x, XS0)?;
                    self.vb3(VBin::Psrlvq, wide, x, x, s)?;
                    self.vb(VBin::Pxor, wide, x, XS0)?;
                }
                VecOp::Shr if t.signed => self.vb3(VBin::Psravd, wide, x, x, s)?,
                VecOp::Shr => self.vb3(if q { VBin::Psrlvq } else { VBin::Psrlvd }, wide, x, x, s)?,
                _ => self.vb3(if q { VBin::Psllvq } else { VBin::Psllvd }, wide, x, x, s)?,
            }
            return self.vput(rv, x, sh.size);
        }
        // lane by lane with `cl`
        let (ma, mb) = (self.vmem(a)?, self.vmem(amt)?);
        self.reserve_g(RCX)?;
        let r = self.alloc_g()?;
        let d = self.def_mem(rv)?;
        let wb = sh.w as u8;
        let sop = match op {
            VecOp::Shr if t.signed => ShiftOp::Sar,
            VecOp::Shr => ShiftOp::Shr,
            _ => ShiftOp::Shl,
        };
        for k in 0..sh.n {
            // the amount's raw bits: a negative amount is far above the width
            self.sized_load(RCX, lane(mb, k, sh.w)?, wb, false)?;
            if trap {
                self.a.alu_ri(AluOp::Cmp, Size::Q, RCX, bits)?;
                self.jtrap(Cond::Ae, PanicKind::Overflow)?;
            } else {
                self.a.alu_ri(AluOp::And, Size::D, RCX, bits - 1)?;
            }
            self.sized_load(r, lane(ma, k, sh.w)?, wb, t.signed)?;
            self.a.shift_cl(sop, Size::Q, r)?;
            self.a.mov_mr(size_of(sh.w), lane(d, k, sh.w)?, r)?;
        }
        Ok(())
    }

    pub(super) fn v_compare(&mut self, op: VecOp, a: Operand, b: Operand, rv: Value) -> Res<()> {
        let sh = self.voshape(a)?;
        let wide = sh.wide();
        // The compare instructions test "less" on floats and "greater" on ints, into their
        // first operand: the other orders swap the operands.
        let swap = match sh.lane {
            LaneTy::Float(_) => matches!(op, VecOp::Gt | VecOp::Ge),
            LaneTy::Int(_) => matches!(op, VecOp::Lt | VecOp::Ge),
        };
        let (l, r) = if swap { (b, a) } else { (a, b) };
        let s = self.vread(r, sh.size)?;
        let x = self.vget(l, sh.size)?;
        match sh.lane {
            LaneTy::Float(_) => {
                let p = self.vprec(sh)?;
                let cmp = if p == Prec::F32 { VBinI::Cmpps } else { VBinI::Cmppd };
                // an unordered pair is only `ne`
                let pred = match op {
                    VecOp::Eq => CmpPred::Eq,
                    VecOp::Ne => CmpPred::Neq,
                    VecOp::Lt | VecOp::Gt => CmpPred::Lt,
                    _ => CmpPred::Le,
                };
                self.vbi(cmp, wide, x, s, pred as u8)?;
            }
            LaneTy::Int(t) => {
                if matches!(op, VecOp::Eq | VecOp::Ne) {
                    self.vb(by_w(sh.w, PCMPEQ), wide, x, s)?;
                } else {
                    self.vcmp_gt(sh.w, t.signed, wide, x, s)?;
                }
                if matches!(op, VecOp::Ne | VecOp::Le | VecOp::Ge) {
                    self.vones(wide, XS0)?;
                    self.vb(VBin::Pxor, wide, x, XS0)?;
                }
            }
        }
        self.vput(rv, x, sh.size)
    }

    /// The fixed tree: lane i with lane i + n/2 until one lane is left.
    pub(super) fn v_reduce(&mut self, op: VecOp, a: Operand, rv: Value) -> Res<()> {
        let sh = self.voshape(a)?;
        let x = self.vget(a, sh.size)?;
        let t = self.alloc_x()?;
        let mut wide = sh.wide();
        let mut n = sh.n;
        while n > 1 {
            let h = n / 2;
            // t = the upper half of the remaining lanes
            if wide {
                self.vhigh(t, x)?;
                wide = false;
            } else {
                self.vmov(false, t, x)?;
                self.vsh(VShiftI::Psrldq, false, t, (h * sh.w) as u8)?;
            }
            match (op, sh.lane) {
                (VecOp::ReduceAdd, LaneTy::Float(_)) => {
                    let p = self.vprec(sh)?;
                    self.vb(ps_pd(p, VBin::Addps, VBin::Addpd), false, x, t)?;
                }
                (VecOp::ReduceAddWrap, LaneTy::Int(_)) => self.vb(by_w(sh.w, PADD), false, x, t)?,
                (VecOp::ReduceAnd, LaneTy::Int(_)) => self.vb(VBin::Pand, false, x, t)?,
                (VecOp::ReduceOr, LaneTy::Int(_)) => self.vb(VBin::Por, false, x, t)?,
                (VecOp::ReduceMin | VecOp::ReduceMax, LaneTy::Float(_)) => {
                    let p = self.vprec(sh)?;
                    self.vfminmax(false, p, op == VecOp::ReduceMin, x, t)?;
                }
                (VecOp::ReduceMin | VecOp::ReduceMax, LaneTy::Int(it)) => {
                    self.viminmax(sh.w, it.signed, false, op == VecOp::ReduceMin, x, t)?;
                }
                _ => return Err(bad(format!("{} on these lanes", op.name()))),
            }
            n = h;
        }
        match sh.lane {
            LaneTy::Float(_) => self.bind(rv, R::X(x)),
            LaneTy::Int(it) => {
                let g = self.alloc_g()?;
                self.vmovd_gx(if sh.w == 8 { Size::Q } else { Size::D }, g, x)?;
                self.norm(g, it)?;
                self.bind_g(rv, g)
            }
        }
    }

    /// `((init + v[0]) + v[1]) + …`.
    pub(super) fn v_reduce_ordered(&mut self, v: Operand, init: Operand, rv: Value) -> Res<()> {
        let sh = self.voshape(v)?;
        let p = self.vprec(sh)?;
        let m = self.vmem(v)?;
        let acc = self.new_x(init, p)?;
        for k in 0..sh.n {
            self.a.sse(SseOp::Add, p, acc, lane(m, k, sh.w)?)?;
        }
        self.bind(rv, R::X(acc))
    }

    /// Inclusive prefix sum from lane 0. Float lanes add in lane order, one at a time. Int
    /// lanes wrap, so any order gives the same sums: every lane gains the lane 1, then 2, 4, …
    /// below it (byte shifts within each 128-bit half), and the upper half of a 256-bit value
    /// then gains the total of the lower one.
    pub(super) fn v_scan(&mut self, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        if let Some(p) = sh.float() {
            let m = self.vmem(a)?;
            let acc = self.alloc_x()?;
            let d = self.def_mem(rv)?;
            self.a.movs(p, acc, m)?;
            self.a.movs_store(p, d, acc)?;
            for k in 1..sh.n {
                self.a.sse(SseOp::Add, p, acc, lane(m, k, sh.w)?)?;
                self.a.movs_store(p, lane(d, k, sh.w)?, acc)?;
            }
            return Ok(());
        }
        let wide = sh.wide();
        let add = by_w(sh.w, PADD);
        let x = self.vget(a, sh.size)?;
        let t = self.alloc_x()?;
        let mut step = sh.w;
        while step < sh.size.min(16) {
            self.vmov(wide, t, x)?;
            self.vsh(VShiftI::Pslldq, wide, t, step as u8)?;
            self.vb(add, wide, x, t)?;
            step *= 2;
        }
        if wide {
            // XS0 = the last lane of the lower half in every lane of the upper half
            self.vmov(false, t, x)?;
            self.vsh(VShiftI::Psrldq, false, t, (16 - sh.w) as u8)?;
            self.vbroadcast(sh.w, false, t)?;
            self.vzero(XS0)?;
            self.vform(true)?;
            self.a.vex_bin_i(VBinI::Inserti128, y(XS0), y(XS0), t, 1)?;
            self.vb(add, true, x, XS0)?;
        }
        self.vput(rv, x, sh.size)
    }
}
