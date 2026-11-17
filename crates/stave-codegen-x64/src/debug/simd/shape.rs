//! Casts, lane moves (split, join, widen, narrow, permute, lookup, compress, expand), and mask ops.

use stave_ir::arith::IntTy;
use stave_ir::layout::Shape;
use stave_ir::simd::LaneTy;
use stave_ir::{CastKind, Operand, TypeId, Value};

use super::{Form, PCMPGT, Sh, by_w, lane, pmovx, scale_of, size_of, y};
use crate::debug::func::*;
use crate::debug::{Res, bad};
use crate::enc::*;

impl Cg<'_, '_> {
    /// Keeps the low `tw` bytes of each `fw`-byte lane of `x`, packed low. Clobbers `XS0`,
    /// `XS1`.
    fn vtrunc(&mut self, x: Xmm, fw: u32, tw: u32, wide: bool) -> Res<()> {
        let mut ctl = [0x80u8; 16];
        for j in 0..16 / fw {
            for b in 0..tw {
                ctl[(j * tw + b) as usize] = (j * fw + b) as u8;
            }
        }
        self.vconst(XS0, wide, &ctl)?;
        self.vb(VBin::Pshufb, wide, x, XS0)?;
        if wide {
            // each half has its lanes packed low: those of the upper half go right above
            self.vhigh(XS1, x)?;
            let op = match 16 * tw / fw {
                8 => VBin::Punpcklqdq,
                4 => VBin::Punpckldq,
                _ => VBin::Punpcklwd,
            };
            self.vb(op, false, x, XS1)?;
        }
        Ok(())
    }

    /// Clamps the lanes of `x` (of shape `sh`, lane type `s`) to the range of `t`. `c` is a
    /// free register; clobbers `XS0`, `XS1`.
    fn vclamp(&mut self, x: Xmm, c: Xmm, sh: Sh, s: IntTy, t: IntTy) -> Res<()> {
        let wide = sh.wide();
        if t.max() < s.max() {
            self.vsplat(c, wide, sh.w, t.max() as u64)?;
            self.viminmax(sh.w, s.signed, wide, true, x, c)?;
        }
        if t.min() > s.min() {
            self.vsplat(c, wide, sh.w, t.min() as u64)?;
            self.viminmax(sh.w, s.signed, wide, false, x, c)?;
        }
        Ok(())
    }

    /// `x` = `x` with its NaN lanes replaced by +0. Clobbers `XS0`.
    fn vnan_zero(&mut self, p: Prec, wide: bool, x: Xmm) -> Res<()> {
        self.vmov(wide, XS0, x)?;
        let cmp = if p == Prec::F32 { VBinI::Cmpps } else { VBinI::Cmppd };
        self.vbi(cmp, wide, XS0, x, CmpPred::Ord as u8)?;
        self.vb(VBin::Pand, wide, x, XS0)
    }

    /// Whether a lane cast has a packed form: every one but `cast.try` (its result depends on
    /// the first failing lane) and the conversions between floats and 64-bit ints.
    fn cast_packed(kind: CastKind, from: Sh, to: Sh) -> bool {
        kind != CastKind::Try
            && match (from.lane, to.lane) {
                (LaneTy::Int(_), LaneTy::Float(_)) => from.w < 8,
                (LaneTy::Float(_), LaneTy::Int(_)) => to.w < 8,
                _ => true,
            }
    }

    /// The casts of [`Self::cast_packed`].
    fn v_cast_packed(&mut self, kind: CastKind, a: Operand, from: Sh, to: Sh, rv: Value) -> Res<()> {
        let x = self.vget(a, from.size)?;
        match (from.lane, to.lane) {
            (LaneTy::Int(s), LaneTy::Int(t)) => {
                if kind == CastKind::Sat {
                    let c = self.alloc_x()?;
                    self.vclamp(x, c, from, s, t)?;
                }
                if to.w > from.w {
                    self.vu_x(pmovx(from.w, to.w, s.signed), to.wide(), x, x)?;
                } else if to.w < from.w {
                    self.vtrunc(x, from.w, to.w, from.wide())?;
                }
                self.vput(rv, x, to.size)
            }
            (LaneTy::Int(s), LaneTy::Float(_)) => {
                let p = self.vprec(to)?;
                // 32-bit lanes: in an `xmm` for `cvtdq2pd`, at the result's width for `cvtdq2ps`
                let dw = p == Prec::F32 && to.wide();
                if from.w < 4 {
                    self.vu_x(pmovx(from.w, 4, s.signed), dw, x, x)?;
                }
                let unsigned = from.w == 4 && !s.signed;
                match (p, unsigned) {
                    (Prec::F32, false) => self.vu(VUn::Cvtdq2ps, dw, x, x)?,
                    (Prec::F32, true) => {
                        // high half * 2^16 + low half: both terms are exact, the sum rounds once
                        self.vmov(dw, XS1, x)?;
                        self.vsh(VShiftI::Psrld, dw, XS1, 16)?;
                        self.vu(VUn::Cvtdq2ps, dw, XS1, XS1)?;
                        self.vsplat(XS0, dw, 4, u64::from(65536f32.to_bits()))?;
                        self.vb(VBin::Mulps, dw, XS1, XS0)?;
                        self.vsplat(XS0, dw, 4, 0xffff)?;
                        self.vb(VBin::Pand, dw, x, XS0)?;
                        self.vu(VUn::Cvtdq2ps, dw, x, x)?;
                        self.vb(VBin::Addps, dw, x, XS1)?;
                    }
                    (Prec::F64, false) => self.vu_x(VUn::Cvtdq2pd, to.wide(), x, x)?,
                    (Prec::F64, true) => {
                        // the lane less 2^31 as a signed one, then 2^31 added back: both exact
                        self.vsplat(XS0, false, 4, 1 << 31)?;
                        self.vb(VBin::Pxor, false, x, XS0)?;
                        self.vu_x(VUn::Cvtdq2pd, to.wide(), x, x)?;
                        self.vsplat(XS0, to.wide(), 8, 2147483648f64.to_bits())?;
                        self.vb(VBin::Addpd, to.wide(), x, XS0)?;
                    }
                }
                self.vput(rv, x, to.size)
            }
            (LaneTy::Float(_), LaneTy::Int(t)) => {
                let p = self.vprec(from)?;
                let wide = from.wide();
                let r = self.alloc_x()?;
                let fbits = |v: i128| match p {
                    Prec::F32 => u64::from((v as f32).to_bits()),
                    Prec::F64 => (v as f64).to_bits(),
                };
                let (min, max) = match p {
                    Prec::F32 => (VBin::Minps, VBin::Maxps),
                    Prec::F64 => (VBin::Minpd, VBin::Maxpd),
                };
                match (p, to.w, t.signed) {
                    (Prec::F32, 4, false) => {
                        // `maxps` gives its second operand for a NaN: NaN and negative lanes are 0
                        self.vzero(XS0)?;
                        self.vb(max, wide, x, XS0)?;
                        // r = the lanes below 2^31; x = the lanes from 2^31 on less 2^31. Out of
                        // range `cvttps2dq` gives the sign bit alone, which marks the lanes of
                        // `r` that take `x` and the lanes of `x` (2^32 and above) that saturate.
                        self.vu(VUn::Cvttps2dq, wide, r, x)?;
                        self.vsplat(XS0, wide, 4, u64::from(2147483648f32.to_bits()))?;
                        self.vb(VBin::Subps, wide, x, XS0)?;
                        self.vu(VUn::Cvttps2dq, wide, x, x)?;
                        self.vmov(wide, XS0, x)?;
                        self.vsh(VShiftI::Psrad, wide, XS0, 31)?;
                        self.vb(VBin::Por, wide, x, XS0)?;
                        self.vmov(wide, XS0, r)?;
                        self.vsh(VShiftI::Psrad, wide, XS0, 31)?;
                        self.vb(VBin::Pand, wide, x, XS0)?;
                        self.vb(VBin::Por, wide, r, x)?;
                    }
                    (Prec::F32, 4, true) => {
                        self.vnan_zero(p, wide, x)?;
                        // out of range `cvttps2dq` gives MIN: the lanes from 2^31 on flip to MAX
                        self.vu(VUn::Cvttps2dq, wide, r, x)?;
                        self.vsplat(XS0, wide, 4, u64::from(2147483648f32.to_bits()))?;
                        self.vbi(VBinI::Cmpps, wide, XS0, x, CmpPred::Le as u8)?;
                        self.vb(VBin::Pxor, wide, r, XS0)?;
                    }
                    _ => {
                        // the bounds of the target are exact in the source format
                        self.vnan_zero(p, wide, x)?;
                        self.vsplat(XS0, wide, from.w, fbits(t.min()))?;
                        self.vb(max, wide, x, XS0)?;
                        self.vsplat(XS0, wide, from.w, fbits(t.max()))?;
                        self.vb(min, wide, x, XS0)?;
                        if p == Prec::F32 {
                            self.vu(VUn::Cvttps2dq, wide, r, x)?;
                            if to.w < 4 {
                                self.vtrunc(r, 4, to.w, wide)?;
                            }
                        } else {
                            // `cvttpd2dq` gives signed lanes: an unsigned 32-bit one is
                            // truncated first (the subtraction is then exact) and moved by 2^31
                            let unsigned = to.w == 4 && !t.signed;
                            if unsigned {
                                self.vui(VUnI::Roundpd, wide, x, x, 11)?;
                                self.vsplat(XS0, wide, 8, (-2147483648f64).to_bits())?;
                                self.vb(VBin::Addpd, wide, x, XS0)?;
                            }
                            self.vcvt_n(VCvtN::Cvttpd2dq, wide, r, x)?;
                            if unsigned {
                                self.vsplat(XS0, false, 4, 1 << 31)?;
                                self.vb(VBin::Pxor, false, r, XS0)?;
                            }
                            if to.w < 4 {
                                self.vtrunc(r, 4, to.w, false)?;
                            }
                        }
                    }
                }
                self.vput(rv, r, to.size)
            }
            (LaneTy::Float(_), LaneTy::Float(_)) => {
                match (self.vprec(from)?, self.vprec(to)?) {
                    (Prec::F32, Prec::F64) => self.vu_x(VUn::Cvtps2pd, to.wide(), x, x)?,
                    (Prec::F64, Prec::F32) => self.vcvt_n(VCvtN::Cvtpd2ps, from.wide(), x, x)?,
                    _ => {}
                }
                self.vput(rv, x, to.size)
            }
        }
    }

    /// The scalar cast of each lane: packed where [`Self::cast_packed`], else the register-level
    /// sequences of `cast` lane by lane. `cast.try` gives `None` at the first lane without a
    /// value.
    pub(super) fn v_cast(&mut self, kind: CastKind, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let from = self.voshape(a)?;
        if kind != CastKind::Try {
            let to = self.vshape(rt)?;
            if Self::cast_packed(kind, from, to) {
                return self.v_cast_packed(kind, a, from, to, rv);
            }
        }
        let opt = if kind == CastKind::Try {
            Some(
                self.cx
                    .m
                    .types
                    .option_shape(rt)
                    .ok_or_else(|| bad("vec.cast.try result must be an Option"))?,
            )
        } else {
            None
        };
        let to = self.vshape(opt.map_or(rt, |o| o.2))?;
        let src = self.vmem(a)?;
        // Every register is taken before the first branch.
        let (r, t2) = (self.alloc_g()?, self.alloc_g()?);
        let x = self.alloc_x()?;
        let (dst, tag) = match opt {
            Some((none, some, _)) => {
                let (d, off, _) = self.option_home(rt, some, rv)?;
                (disp(d, off)?, Some((d, none, some)))
            }
            None => (self.def_mem(rv)?, None),
        };
        let fail = self.a.new_label();
        for k in 0..from.n {
            let (sm, dm) = (lane(src, k, from.w)?, lane(dst, k, to.w)?);
            match (from.lane, to.lane) {
                (LaneTy::Int(s), LaneTy::Int(t)) => {
                    self.sized_load(r, sm, from.w as u8, s.signed)?;
                    match kind {
                        CastKind::Sat => self.sat_clamp(r, s, t)?,
                        CastKind::Try => self.range_jump(r, s, t, fail)?,
                        _ => {}
                    }
                    self.a.mov_mr(size_of(to.w), dm, r)?;
                }
                (LaneTy::Float(_), LaneTy::Int(t)) => {
                    // the lane as f64 in XS0
                    if self.vprec(from)? == Prec::F64 {
                        self.a.movs(Prec::F64, XS0, sm)?;
                    } else {
                        self.a.cvts2s(Prec::F32, XS0, sm)?;
                    }
                    if kind == CastKind::Try {
                        let some = self.a.new_label();
                        self.float_try_int(r, t, fail, some)?;
                        self.a.bind(some)?;
                    } else {
                        self.f64_to_int_sat(r, t)?;
                    }
                    self.a.mov_mr(size_of(to.w), dm, r)?;
                }
                (LaneTy::Int(s), LaneTy::Float(_)) => {
                    let p = self.vprec(to)?;
                    self.sized_load(r, sm, from.w as u8, s.signed)?;
                    if kind == CastKind::Try {
                        // exact iff the magnitude's set bits span at most the mantissa
                        let exact = self.a.new_label();
                        self.a.mov_rr(Size::Q, S0, r);
                        if s.signed {
                            self.a.mov_rr(Size::Q, S1, S0);
                            self.a.shift_ri(ShiftOp::Sar, Size::Q, S1, 63)?;
                            self.a.alu_rr(AluOp::Xor, Size::Q, S0, S1);
                            self.a.alu_rr(AluOp::Sub, Size::Q, S0, S1);
                        }
                        self.a.test_rr(Size::Q, S0, S0);
                        self.a.jcc(Cond::E, exact)?;
                        self.a.bsr(Size::Q, S1, S0)?;
                        self.a.bsf(Size::Q, t2, S0)?;
                        self.a.alu_rr(AluOp::Sub, Size::Q, S1, t2);
                        self.a.alu_ri(AluOp::Cmp, Size::Q, S1, if p == Prec::F32 { 24 } else { 53 })?;
                        self.a.jcc(Cond::Ae, fail)?;
                        self.a.bind(exact)?;
                    }
                    self.cvt_int(p, x, r, s)?;
                    self.a.movs_store(p, dm, x)?;
                }
                (LaneTy::Float(_), LaneTy::Float(_)) => {
                    let (fp, tp) = (self.vprec(from)?, self.vprec(to)?);
                    if fp == tp {
                        self.a.movs(fp, x, sm)?;
                    } else if kind == CastKind::Try && fp == Prec::F64 {
                        // exact iff the rounded value widens back to the lane
                        self.a.cvts2s(Prec::F64, x, sm)?;
                        self.a.cvts2s(Prec::F32, XS0, x)?;
                        self.a.ucomis(Prec::F64, XS0, sm)?;
                        self.a.jcc(Cond::Ne, fail)?;
                    } else {
                        self.a.cvts2s(fp, x, sm)?;
                    }
                    self.a.movs_store(tp, dm, x)?;
                }
            }
        }
        match tag {
            Some((d, none, some)) => {
                let end = self.a.new_label();
                self.store_imm(d, 4, i64::from(some))?;
                self.a.jmp(end)?;
                self.a.bind(fail)?;
                self.store_imm(d, 4, i64::from(none))?;
                Ok(self.a.bind(end)?)
            }
            None => Ok(self.a.bind(fail)?),
        }
    }

    /// Same bytes under another vector type (`bitcast`, `mask.to_vec`).
    pub(super) fn v_copy(&mut self, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let x = self.vget(a, sh.size)?;
        self.vput(rv, x, sh.size)
    }

    pub(super) fn v_split(&mut self, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.voshape(a)?;
        let offs: Vec<i64> = match &self.layout(rt)?.shape {
            Shape::Fields(fs) if fs.len() == 2 => fs.iter().map(|f| f.0 as i64).collect(),
            _ => return Err(bad("split result must be a pair")),
        };
        let h = sh.size / 2;
        let m = self.vmem(a)?;
        let d = self.def_mem(rv)?;
        self.copy_mem(disp(d, offs[0])?, MSrc::Mem(m), h)?;
        self.copy_mem(disp(d, offs[1])?, MSrc::Mem(disp(m, i64::from(h))?), h)
    }

    pub(super) fn v_join(&mut self, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let h = sh.size / 2;
        let (ma, mb) = (self.vmem(a)?, self.vmem(b)?);
        let d = self.def_mem(rv)?;
        self.copy_mem(d, MSrc::Mem(ma), h)?;
        self.copy_mem(disp(d, i64::from(h))?, MSrc::Mem(mb), h)
    }

    /// The low or high half of the lanes, each converted exactly to the double-width lane.
    pub(super) fn v_widen(&mut self, hi: bool, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let from = self.voshape(a)?;
        let to = self.vshape(rt)?;
        let x = self.vget(a, from.size)?;
        if hi {
            if from.wide() {
                self.vhigh(x, x)?;
            } else {
                self.vsh(VShiftI::Psrldq, false, x, (from.size / 2) as u8)?;
            }
        }
        let op = match from.lane {
            LaneTy::Float(_) => VUn::Cvtps2pd,
            LaneTy::Int(t) => pmovx(from.w, to.w, t.signed),
        };
        self.vu_x(op, to.wide(), x, x)?;
        self.vput(rv, x, to.size)
    }

    /// The lanes of `a` then `b`, truncated or saturated to the half-width lane. `pack*`
    /// saturates signed lanes to the signed (`packss*`) or the unsigned (`packus*`) range, so
    /// unsigned saturation clamps first and truncation masks first; 64-bit lanes have no pack:
    /// they are clamped, then `shufps` picks their low halves.
    pub(super) fn v_narrow(&mut self, sat: bool, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let to = self.vshape(rt)?;
        let from = self.voshape(a)?;
        let (t, s) = (self.vint(to)?, self.vint(from)?);
        let wide = from.wide();
        let x = self.vget(a, from.size)?;
        let mut xb = self.vget(b, from.size)?;
        let c = self.alloc_x()?;
        if from.size < 16 {
            // both operands fit one register, the lanes of `b` right above those of `a`
            let op = match from.size {
                8 => VBin::Punpcklqdq,
                4 => VBin::Punpckldq,
                _ => VBin::Punpcklwd,
            };
            self.vb(op, false, x, xb)?;
            xb = x;
        }
        let srcs = if xb == x { vec![x] } else { vec![x, xb] };
        if sat && from.w == 8 {
            for &r in &srcs {
                self.vclamp(r, c, from, s, t)?;
            }
        } else if !(sat && s.signed) && from.w < 8 {
            let op = match (sat, from.w) {
                (false, _) => VBin::Pand,
                (true, 2) => VBin::Pminuw,
                (true, _) => VBin::Pminud,
            };
            self.vsplat(c, wide, from.w, to.ones())?;
            for &r in &srcs {
                self.vb(op, wide, r, c)?;
            }
        }
        match from.w {
            2 => self.vb(if sat && s.signed { VBin::Packsswb } else { VBin::Packuswb }, wide, x, xb)?,
            4 => self.vb(if sat && s.signed { VBin::Packssdw } else { VBin::Packusdw }, wide, x, xb)?,
            _ => self.vbi(VBinI::Shufps, wide, x, xb, 0x88)?,
        }
        if wide {
            // each 128-bit half holds its part of `a` then of `b`
            self.vui(VUnI::Permq, true, x, x, 0xd8)?;
        }
        self.vput(rv, x, to.size)
    }

    /// Lane j of the result = lane `k_j` of the operand: a byte shuffle (`pshufb`) within 128
    /// bits. On 256 bits `vpermd` moves 32- and 64-bit lanes; narrower lanes take a byte shuffle
    /// in each half of the operand and of the operand with its halves swapped.
    pub(super) fn v_permute(&mut self, args: &[Operand], rt: TypeId, rv: Value) -> Res<()> {
        let to = self.vshape(rt)?;
        let a = args.first().copied().ok_or_else(|| bad("permute operand"))?;
        let from = self.voshape(a)?;
        let w = to.w as usize;
        let mut ks = Vec::with_capacity(to.n as usize);
        for &k in &args[1..] {
            let k = self
                .int_const(k)
                .and_then(|k| u32::try_from(k).ok())
                .filter(|k| *k < from.n)
                .ok_or_else(|| bad("permute index"))?;
            ks.push(k as usize);
        }
        if ks.len() != to.n as usize || from.w != to.w {
            return Err(bad("permute shape"));
        }
        // the source byte of each result byte
        let mut src = [None; 32];
        for (j, &k) in ks.iter().enumerate() {
            for b in 0..w {
                src[j * w + b] = Some((k * w + b) as u8);
            }
        }
        if (from.wide() || to.wide()) && !self.fx.avx2 {
            // no shuffle moves lanes between the halves of a `ymm` without AVX2: lane by lane
            let m = self.vmem(a)?;
            let d = self.def_mem(rv)?;
            for (j, &k) in ks.iter().enumerate() {
                self.sized_load(S0, lane(m, k as u32, to.w)?, to.w as u8, false)?;
                self.a.mov_mr(size_of(to.w), lane(d, j as u32, to.w)?, S0)?;
            }
            return Ok(());
        }
        let x = self.vget(a, from.size)?;
        if !from.wide() && !to.wide() {
            let ctl: Vec<u8> = src[..16].iter().map(|s| s.unwrap_or(0x80)).collect();
            self.vconst(XS0, false, &ctl)?;
            self.vb(VBin::Pshufb, false, x, XS0)?;
            return self.vput(rv, x, to.size);
        }
        if w >= 4 {
            let mut idx = [0u8; 32];
            for (d, i) in idx.chunks_mut(4).enumerate() {
                i[0] = src[d * 4].map_or(0, |b| b / 4);
            }
            self.vconst(XS0, true, &idx)?;
            self.vb3(VBin::Permd, true, x, XS0, x)?;
            return self.vput(rv, x, to.size);
        }
        if !from.wide() {
            // the operand in both halves: every byte shuffle is within a half
            self.vui(VUnI::Permq, true, x, x, 0x44)?;
            let ctl: Vec<u8> = src.iter().map(|s| s.unwrap_or(0x80)).collect();
            self.vconst(XS0, true, &ctl)?;
            self.vb(VBin::Pshufb, true, x, XS0)?;
            return self.vput(rv, x, to.size);
        }
        // `pshufb` shuffles within each half: bytes that stay in their half come from the
        // operand, the others from the operand with its halves swapped
        let ctl = |swapped: bool| -> Vec<u8> {
            src.iter()
                .enumerate()
                .map(|(i, s)| match s {
                    Some(b) if ((*b as usize / 16) != i / 16) == swapped => b % 16,
                    _ => 0x80,
                })
                .collect()
        };
        let (same, other) = (ctl(false), ctl(true));
        if other.iter().all(|&b| b == 0x80) {
            self.vconst(XS0, true, &same)?;
            self.vb(VBin::Pshufb, true, x, XS0)?;
            return self.vput(rv, x, to.size);
        }
        let sw = self.alloc_x()?;
        self.vui(VUnI::Permq, true, sw, x, 0x4e)?;
        self.vconst(XS0, true, &same)?;
        self.vb(VBin::Pshufb, true, x, XS0)?;
        self.vconst(XS0, true, &other)?;
        self.vb(VBin::Pshufb, true, sw, XS0)?;
        self.vb(VBin::Por, true, x, sw)?;
        self.vput(rv, x, to.size)
    }

    /// Lane j = `tbl[idx[j]]`, zero for an index outside the table (a negative one too).
    pub(super) fn v_lookup(&mut self, tbl: Operand, idx: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let ish = self.voshape(idx)?;
        let it = self.vint(ish)?;
        if sh.w == 1 && ish.w == 1 {
            return self.vlookup_bytes(tbl, idx, sh, rv);
        }
        let (mt, mi) = (self.vmem(tbl)?, self.vmem(idx)?);
        let Mem::Sib { base: Some(tb), index: None, disp: td } = mt else {
            return Err(bad("internal: vector home address"));
        };
        let r = self.alloc_g()?;
        let d = self.def_mem(rv)?;
        for j in 0..sh.n {
            let skip = self.a.new_label();
            self.sized_load(r, lane(mi, j, ish.w)?, ish.w as u8, it.signed)?;
            self.a.alu_rr(AluOp::Xor, Size::D, S0, S0);
            self.a.alu_ri(AluOp::Cmp, Size::Q, r, i64::from(sh.n))?;
            self.a.jcc(Cond::Ae, skip)?;
            self.sized_load(S0, Mem::index(tb, r, scale_of(sh.w), td), sh.w as u8, false)?;
            self.a.bind(skip)?;
            self.a.mov_mr(size_of(sh.w), lane(d, j, sh.w)?, S0)?;
        }
        Ok(())
    }

    /// A lookup of byte lanes through byte indices with `pshufb`, which gives zero for an index
    /// byte with its top bit set and else takes the low four bits: adding 0x70 with unsigned
    /// saturation sets the top bit of every index from 16 on and leaves the low four bits of
    /// the others. A table of fewer than 16 lanes has zeros above it in its register; one of 32
    /// lanes is looked up half by half, the upper half with bit 4 of the indices flipped.
    fn vlookup_bytes(&mut self, tbl: Operand, idx: Operand, sh: Sh, rv: Value) -> Res<()> {
        let wide = sh.wide();
        let t = self.vget(tbl, sh.size)?;
        let i = self.vget(idx, sh.size)?;
        self.vsplat(XS0, wide, 1, 0x70)?;
        if !wide {
            self.vb(VBin::Paddusb, false, i, XS0)?;
            self.vb(VBin::Pshufb, false, t, i)?;
            return self.vput(rv, t, sh.size);
        }
        let hi = self.alloc_x()?;
        self.vui(VUnI::Permq, true, hi, t, 0xee)?;
        self.vui(VUnI::Permq, true, t, t, 0x44)?;
        self.vsplat(XS1, true, 1, 0x10)?;
        self.vb(VBin::Pxor, true, XS1, i)?;
        self.vb(VBin::Paddusb, true, XS1, XS0)?;
        self.vb(VBin::Paddusb, true, i, XS0)?;
        self.vb(VBin::Pshufb, true, t, i)?;
        self.vb(VBin::Pshufb, true, hi, XS1)?;
        self.vb(VBin::Por, true, t, hi)?;
        self.vput(rv, t, sh.size)
    }

    /// `compress`: the active lanes packed low; `expand`: the low lanes spread to the active
    /// positions; other lanes zero. Table-driven where the shape has tables, else lane by lane
    /// with a running count of active lanes.
    pub(super) fn v_compress(
        &mut self,
        expand: bool,
        a: Operand,
        m: Operand,
        rt: TypeId,
        rv: Value,
    ) -> Res<()> {
        if self.v_compress_packed(expand, a, m, rt, rv)? {
            return Ok(());
        }
        let sh = self.vshape(rt)?;
        let (ma, mm) = (self.vmem(a)?, self.vmem(m)?);
        let k = self.alloc_g()?;
        let d = self.def_mem(rv)?;
        let (
            Mem::Sib { base: Some(ab), index: None, disp: ad },
            Mem::Sib { base: Some(db), index: None, disp: dd },
        ) = (ma, d)
        else {
            return Err(bad("internal: vector home address"));
        };
        let (sc, sz, wb) = (scale_of(sh.w), size_of(sh.w), sh.w as u8);
        self.zero_mem(d, sh.size)?;
        self.a.alu_rr(AluOp::Xor, Size::D, k, k);
        for j in 0..sh.n {
            let skip = self.a.new_label();
            self.a.alu_mi(AluOp::Cmp, Size::B, lane(mm, j, sh.w)?, 0)?;
            self.a.jcc(Cond::E, skip)?;
            if expand {
                self.sized_load(S0, Mem::index(ab, k, sc, ad), wb, false)?;
                self.a.mov_mr(sz, lane(d, j, sh.w)?, S0)?;
            } else {
                self.sized_load(S0, lane(ma, j, sh.w)?, wb, false)?;
                self.a.mov_mr(sz, Mem::index(db, k, sc, dd), S0)?;
            }
            self.a.alu_ri(AluOp::Add, Size::D, k, 1)?;
            self.a.bind(skip)?;
        }
        Ok(())
    }

    /// Lanes of `a` where the mask is set, else of `b`: a byte blend (every byte of a mask lane
    /// has the lane's value).
    pub(super) fn v_select(&mut self, m: Operand, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        if !self.vex() {
            // `pblendvb` reads its mask from `xmm0`
            let x0 = Xmm::masked(0);
            self.reserve(R::X(x0))?;
            if self.rel {
                self.vec_copy(m, x0)?;
            } else {
                let mm = self.vmem(m)?;
                self.vload(x0, mm, sh.size)?;
            }
            let x = self.vread(a, sh.size)?;
            let s = self.vget(b, sh.size)?;
            self.a.sse_blendv(VBlendV::Pblendvb, s, x)?;
            return self.vput(rv, s, sh.size);
        }
        let mx = self.vread(m, sh.size)?;
        let x = self.vread(a, sh.size)?;
        let s = self.vget(b, sh.size)?;
        match self.vform(sh.wide())? {
            Form::Y => {
                // the 256-bit byte blend is AVX2; AVX blends by the sign of a 32- or 64-bit
                // lane, which is the mask lane's value
                let op = match (self.fx.avx2, sh.w) {
                    (false, 4) => VBlendV::Blendvps,
                    (false, 8) => VBlendV::Blendvpd,
                    _ => VBlendV::Pblendvb,
                };
                self.a.vex_blendv(op, y(s), y(s), y(x), y(mx))?
            }
            _ => self.a.vex_blendv(VBlendV::Pblendvb, s, s, x, mx)?,
        }
        self.vput(rv, s, sh.size)
    }

    /// `g` = bit j set iff lane j of the mask in `x` is set. Clobbers `XS0`, `S0`.
    pub(super) fn vmask_bits(&mut self, g: Gpr, x: Xmm, sh: Sh) -> Res<()> {
        let wide = sh.wide();
        // the sign bits of the bytes of a `ymm` are an AVX2 instruction
        let ybytes = wide && self.fx.avx2;
        match sh.w {
            1 if wide && !ybytes => {
                self.vmsk(VMovMsk::Pmovmskb, false, g, x)?;
                self.vhigh(XS0, x)?;
                self.vmsk(VMovMsk::Pmovmskb, false, S0, XS0)?;
                self.a.shift_ri(ShiftOp::Shl, Size::D, S0, 16)?;
                self.a.alu_rr(AluOp::Or, Size::D, g, S0);
            }
            1 => self.vmsk(VMovMsk::Pmovmskb, wide, g, x)?,
            4 => self.vmsk(VMovMsk::Movmskps, wide, g, x)?,
            8 => self.vmsk(VMovMsk::Movmskpd, wide, g, x)?,
            _ if self.fx.bmi2 && (ybytes || !wide) => {
                // two equal bits per lane: keep one
                self.vmsk(VMovMsk::Pmovmskb, wide, g, x)?;
                self.a.mov_ri(Size::D, S0, 0xaaaa_aaaa)?;
                self.a.pext(Size::D, g, g, S0)?;
            }
            _ if wide => {
                // the lanes of both halves as the bytes of one `xmm`
                self.vhigh(XS0, x)?;
                self.vb3(VBin::Packsswb, false, XS0, x, XS0)?;
                self.vmsk(VMovMsk::Pmovmskb, false, g, XS0)?;
            }
            _ => {
                self.vmov(false, XS0, x)?;
                self.vb(VBin::Packsswb, false, XS0, XS0)?;
                self.vmsk(VMovMsk::Pmovmskb, false, g, XS0)?;
            }
        }
        // a full register gives one bit per lane, except two per lane from `packsswb` of one
        // `xmm` with itself
        let exact = (sh.size == 16 || sh.size == 32) && (sh.w != 2 || self.fx.bmi2 || wide);
        if sh.n < 32 && !exact {
            self.a.alu_ri(AluOp::And, Size::D, g, (1i64 << sh.n) - 1)?;
        }
        Ok(())
    }

    pub(super) fn v_to_bits(&mut self, m: Operand, rv: Value) -> Res<()> {
        let sh = self.voshape(m)?;
        let x = self.vread(m, sh.size)?;
        let g = self.alloc_g()?;
        self.vmask_bits(g, x, sh)?;
        self.bind_g(rv, g)
    }

    /// Lanes `j < k` set: `min(k, n)` in every lane against the lane numbers.
    pub(super) fn v_first_n(&mut self, k: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let wide = sh.wide();
        let g = self.read_g(k)?;
        let (x, iota) = (self.alloc_x()?, self.alloc_x()?);
        self.a.mov_ri(Size::D, S0, i64::from(sh.n))?;
        self.a.alu_rr(AluOp::Cmp, Size::Q, g, S0);
        self.a.cmov(Cond::B, Size::Q, S0, g)?;
        self.vmovd_xg(Size::D, x, S0)?;
        let lanes = if wide { 32 } else { 16 } / sh.w;
        let mut bytes = Vec::with_capacity(32);
        for j in 0..u64::from(lanes) {
            bytes.extend_from_slice(&j.to_le_bytes()[..sh.w as usize]);
        }
        if wide && !self.fx.avx2 {
            // no 256-bit integer compare without AVX2: each half against its lane numbers
            let gt = by_w(sh.w, PCMPGT);
            self.vbroadcast(sh.w, false, x)?;
            self.vmov(false, XS1, x)?;
            self.vconst(iota, false, &bytes[..16])?;
            self.vb(gt, false, x, iota)?;
            self.vconst(iota, false, &bytes[16..])?;
            self.vb(gt, false, XS1, iota)?;
            self.vset_high(x, XS1)?;
            return self.vput(rv, x, sh.size);
        }
        self.vbroadcast(sh.w, wide, x)?;
        self.vconst(iota, wide, &bytes)?;
        self.vb(by_w(sh.w, PCMPGT), wide, x, iota)?;
        self.vput(rv, x, sh.size)
    }

    /// The same lanes at another lane width: sign extension and truncation keep a lane all ones
    /// or zero.
    pub(super) fn v_rewiden(&mut self, m: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let from = self.voshape(m)?;
        let to = self.vshape(rt)?;
        if to.w == from.w {
            return self.v_copy(m, rt, rv);
        }
        let x = self.vget(m, from.size)?;
        if to.w > from.w {
            self.vu_x(pmovx(from.w, to.w, true), to.wide(), x, x)?;
        } else {
            self.vtrunc(x, from.w, to.w, from.wide())?;
        }
        self.vput(rv, x, to.size)
    }
}
