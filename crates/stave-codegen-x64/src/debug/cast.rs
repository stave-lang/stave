//! The `as` family: `cast` (lossless), `cast.wrap` (`as%`), `cast.sat` (`as|`, NaN → 0),
//! `cast.round` (`as~`, round to nearest even: SSE2 conversions in the default MXCSR mode), and
//! `cast.try` (`as?`, an `Option`). Semantics are those of `stave_ir::arith::convert`; constant
//! operands fold through it.

use stave_ir::arith::{self, CastTy, IntTy};
use stave_ir::layout::Shape;
use stave_ir::{CastKind, Const, FloatBits, Operand, TypeId, TypeKey, Value};

use super::abi::VCls;
use super::func::*;
use super::{Res, bad, unsup};
use crate::enc::*;

const F64_2P63: i64 = 0x43e0_0000_0000_0000;
const F64_2P64: i64 = 0x43f0_0000_0000_0000;
const F64_M2P63: i64 = 0xc3e0_0000_0000_0000_u64 as i64;
const F64_M1: i64 = 0xbff0_0000_0000_0000_u64 as i64;

fn prec(f: FloatBits) -> Res<Prec> {
    match f {
        FloatBits::F32 => Ok(Prec::F32),
        FloatBits::F64 => Ok(Prec::F64),
        _ => Err(unsup("f16 / bf16 as a cast target (needs software rounding)")),
    }
}

/// The integer view of an int-like source.
fn int_like(s: Sk) -> Option<IntTy> {
    match s {
        Sk::Int(t) => Some(t),
        Sk::Bool => Some(IntTy::new(8, false)),
        Sk::Char => Some(IntTy::new(32, false)),
        Sk::Ptr => Some(IntTy::new(64, false)),
        Sk::Float(_) => None,
    }
}

fn cast_ty(s: Sk) -> CastTy {
    match s {
        Sk::Int(t) => CastTy::Int(t),
        Sk::Bool => CastTy::Bool,
        Sk::Char => CastTy::Char,
        Sk::Float(f) => CastTy::Float(f),
        Sk::Ptr => CastTy::Ptr,
    }
}

impl Cg<'_, '_> {
    /// (none variant, some variant, payload type) of an `Option`-shaped enum.
    fn option_shape(&self, t: TypeId) -> Res<(u32, u32, TypeId)> {
        self.cx.m.types.option_shape(t).ok_or_else(|| bad("cast.try result must be an Option-shaped enum"))
    }

    pub(crate) fn cast(&mut self, kind: CastKind, a: Operand, rt: TypeId, v: Value) -> Res<()> {
        let opt = if kind == CastKind::Try { Some(self.option_shape(rt)?) } else { None };
        let to_t = opt.map_or(rt, |o| o.2);
        // fn values are (code, environment) pairs: only a fn item `@f` casts (as a pointer)
        let is_fn =
            |t: Option<TypeId>| t.is_some_and(|t| matches!(self.key(t), Some(TypeKey::FnType { .. })));
        if is_fn(Some(to_t)) || is_fn(self.oty(a)) {
            return Err(bad("cast of a fn value or to a fn type"));
        }
        let to = self.sk(to_t).ok_or_else(|| bad("cast to a non-scalar type"))?;
        if matches!(a, Operand::Const(_)) {
            return self.cast_const(kind, a, rt, to, opt, v);
        }
        let from = self.osk(a).ok_or_else(|| bad("cast of a non-scalar operand"))?;
        match opt {
            None => self.cast_plain(kind, from, to, a, v),
            Some(o) => self.cast_try(from, to, a, rt, o, v),
        }
    }

    fn cast_const(
        &mut self,
        kind: CastKind,
        a: Operand,
        rt: TypeId,
        to: Sk,
        opt: Option<(u32, u32, TypeId)>,
        v: Value,
    ) -> Res<()> {
        let Operand::Const(c) = a else { return Err(bad("constant cast")) };
        let (from, bits) = match self.b.konst(c).ok_or_else(|| bad("constant out of range"))? {
            Const::Int(x) => (CastTy::Int(IntTy::new(64, x < 0)), x as u64),
            Const::F32(b) => (CastTy::Float(FloatBits::F32), u64::from(b)),
            Const::F64(b) => (CastTy::Float(FloatBits::F64), b),
            Const::F16(b) => (CastTy::Float(FloatBits::F16), u64::from(b)),
            Const::BF16(b) => (CastTy::Float(FloatBits::B16), u64::from(b)),
        };
        let r = arith::convert(kind, from, cast_ty(to), bits);
        match opt {
            None => {
                let bits = r.ok_or_else(|| bad("constant cast has no value"))? as i64;
                match self.vcls(rt)? {
                    VCls::G { .. } => {
                        let g = self.alloc_g()?;
                        self.a.mov_ri(Size::Q, g, bits)?;
                        self.bind_g(v, g)
                    }
                    VCls::X(_) => {
                        let x = self.alloc_x()?;
                        self.a.mov_ri(Size::Q, S0, bits)?;
                        self.a.mov_xg(Size::Q, x, S0)?;
                        self.bind(v, R::X(x))
                    }
                    VCls::M { .. } => Err(bad("cast to an aggregate")),
                }
            }
            Some((none, some, pty)) => {
                let (d, off, _) = self.option_home(rt, some, v)?;
                match r {
                    Some(bits) => {
                        self.store_imm(d, 4, i64::from(some))?;
                        let bytes = match self.vcls(pty)? {
                            VCls::G { bytes, .. } => bytes,
                            VCls::X(Prec::F32) => 4,
                            VCls::X(Prec::F64) => 8,
                            VCls::M { .. } => return Err(bad("cast.try to an aggregate")),
                        };
                        self.store_imm(disp(d, off)?, bytes, bits as i64)
                    }
                    None => self.store_imm(d, 4, i64::from(none)),
                }
            }
        }
    }

    /// Allocates and zeroes the `Option` result; returns its home, the payload offset, and size.
    fn option_home(&mut self, rt: TypeId, some: u32, v: Value) -> Res<(Mem, i64, u32)> {
        let l = self.layout(rt)?;
        let Shape::Enum(vs) = &l.shape else { return Err(bad("cast.try result is not an enum")) };
        let off =
            vs.get(some as usize).and_then(|f| f.first()).map(|f| f.0 as i64).ok_or_else(|| bad("Option"))?;
        let size = u32::try_from(l.size).map_err(|_| bad("Option size"))?;
        let d = self.def_mem(v)?;
        self.zero_mem(d, size)?;
        Ok((d, off, size))
    }

    fn cast_plain(&mut self, kind: CastKind, from: Sk, to: Sk, a: Operand, v: Value) -> Res<()> {
        match (from, to) {
            (Sk::Float(ff), Sk::Float(tf)) => self.float_to_float(ff, tf, a, v),
            (Sk::Float(ff), Sk::Int(t)) => {
                self.f64_work(a, ff)?;
                let r = self.alloc_g()?;
                self.f64_to_int_sat(r, t)?;
                self.bind_g(v, r)
            }
            (_, Sk::Float(tf)) => {
                let s = int_like(from).ok_or_else(|| bad("cast source"))?;
                let p = prec(tf)?;
                let r = self.read_g(a)?;
                let x = self.alloc_x()?;
                self.cvt_int(p, x, r, s)?;
                self.bind(v, R::X(x))
            }
            (_, Sk::Int(t)) => {
                let s = int_like(from).ok_or_else(|| bad("cast source"))?;
                let r = self.take_g(a)?;
                match kind {
                    CastKind::Wrap => self.norm(r, t)?,
                    CastKind::Sat => self.sat_clamp(r, s, t)?,
                    _ => {}
                }
                self.bind_g(v, r)
            }
            (_, Sk::Char | Sk::Ptr) => {
                let r = self.take_g(a)?;
                self.bind_g(v, r)
            }
            _ => Err(unsup(format!("cast {from:?} -> {to:?}"))),
        }
    }

    /// Clamps the normalized value `r` of source type `s` to target `t`.
    fn sat_clamp(&mut self, r: Gpr, s: IntTy, t: IntTy) -> Res<()> {
        if s.signed {
            let hi = t.max() < s.max();
            let lo = t.min() > s.min();
            self.clamp(r, t, lo, hi, true)
        } else {
            self.clamp(r, t, false, t.max() < s.max(), false)
        }
    }

    /// Jumps to `fail` unless the normalized value `r` of type `s` is in `t`'s range.
    fn range_jump(&mut self, r: Gpr, s: IntTy, t: IntTy, fail: Label) -> Res<()> {
        let cmp = |cg: &mut Self, x: i128| -> Res<()> {
            let x = x as u64 as i64;
            if let Ok(i) = i32::try_from(x) {
                cg.a.alu_ri(AluOp::Cmp, Size::Q, r, i64::from(i))?;
            } else {
                cg.a.mov_ri(Size::Q, S0, x)?;
                cg.a.alu_rr(AluOp::Cmp, Size::Q, r, S0);
            }
            Ok(())
        };
        if s.signed {
            if t.max() < s.max() {
                cmp(self, t.max())?;
                self.a.jcc(Cond::G, fail)?;
            }
            if t.min() > s.min() {
                cmp(self, t.min())?;
                self.a.jcc(Cond::L, fail)?;
            }
        } else if t.max() < s.max() {
            cmp(self, t.max())?;
            self.a.jcc(Cond::A, fail)?;
        }
        Ok(())
    }

    /// `x = (float) r` for a normalized integer of type `s` (u64 via halving with the sticky bit).
    fn cvt_int(&mut self, p: Prec, x: Xmm, r: Gpr, s: IntTy) -> Res<()> {
        if s.signed || s.bits < 64 {
            self.a.cvtsi2s(p, x, Size::Q, r)?;
            return Ok(());
        }
        let (big, done) = (self.a.new_label(), self.a.new_label());
        self.a.test_rr(Size::Q, r, r);
        self.a.jcc(Cond::S, big)?;
        self.a.cvtsi2s(p, x, Size::Q, r)?;
        self.a.jmp(done)?;
        self.a.bind(big)?;
        self.a.mov_rr(Size::Q, S0, r);
        self.a.shift_ri(ShiftOp::Shr, Size::Q, S0, 1)?;
        self.a.mov_rr(Size::D, S1, r);
        self.a.alu_ri(AluOp::And, Size::D, S1, 1)?;
        self.a.alu_rr(AluOp::Or, Size::Q, S0, S1);
        self.a.cvtsi2s(p, x, Size::Q, S0)?;
        self.a.sse(SseOp::Add, p, x, x)?;
        self.a.bind(done)?;
        Ok(())
    }

    /// The value of float operand `a` (format `f`) as f64 in `XS0`.
    fn f64_work(&mut self, a: Operand, f: FloatBits) -> Res<()> {
        match f {
            FloatBits::F64 => self.copy_to_x(a, Prec::F64, XS0),
            FloatBits::F32 => {
                let x = self.read_x(a, Prec::F32)?;
                Ok(self.a.cvts2s(Prec::F32, XS0, x)?)
            }
            FloatBits::F16 | FloatBits::B16 => {
                let g = self.read_g(a)?;
                self.half_to_f32(g, f == FloatBits::B16, XS0)?;
                Ok(self.a.cvts2s(Prec::F32, XS0, XS0)?)
            }
        }
    }

    fn f64_const(&mut self, x: Xmm, bits: i64) -> Res<()> {
        self.a.mov_ri(Size::Q, S0, bits)?;
        Ok(self.a.mov_xg(Size::Q, x, S0)?)
    }

    /// `r = t as| XS0` (saturating, NaN → 0, truncating). Clobbers `XS0`, `XS1`.
    fn f64_to_int_sat(&mut self, r: Gpr, t: IntTy) -> Res<()> {
        let (lmax, lzero, done) = (self.a.new_label(), self.a.new_label(), self.a.new_label());
        if !t.signed && t.bits == 64 {
            let lhigh = self.a.new_label();
            self.f64_const(XS1, F64_2P64)?;
            self.a.ucomis(Prec::F64, XS0, XS1)?;
            self.a.jcc(Cond::Ae, lmax)?;
            self.f64_const(XS1, F64_2P63)?;
            self.a.ucomis(Prec::F64, XS0, XS1)?;
            self.a.jcc(Cond::Ae, lhigh)?;
            self.f64_const(XS1, F64_M1)?;
            self.a.ucomis(Prec::F64, XS0, XS1)?;
            self.a.jcc(Cond::Be, lzero)?;
            self.a.cvtts2si(Prec::F64, Size::Q, r, XS0)?;
            self.a.jmp(done)?;
            self.a.bind(lhigh)?;
            self.a.sse(SseOp::Sub, Prec::F64, XS0, XS1)?;
            self.a.cvtts2si(Prec::F64, Size::Q, r, XS0)?;
            self.a.mov_ri(Size::Q, S0, i64::MIN)?;
            self.a.alu_rr(AluOp::Xor, Size::Q, r, S0);
            self.a.jmp(done)?;
            self.a.bind(lmax)?;
            self.a.mov_ri(Size::Q, r, -1)?;
            self.a.jmp(done)?;
            self.a.bind(lzero)?;
            self.a.alu_rr(AluOp::Xor, Size::D, r, r);
            self.a.bind(done)?;
            return Ok(());
        }
        self.f64_const(XS1, F64_2P63)?;
        self.a.ucomis(Prec::F64, XS0, XS1)?;
        self.a.jcc(Cond::Ae, lmax)?;
        self.a.ucomis(Prec::F64, XS0, XS0)?;
        self.a.jcc(Cond::P, lzero)?;
        self.a.cvtts2si(Prec::F64, Size::Q, r, XS0)?;
        self.a.jmp(done)?;
        self.a.bind(lmax)?;
        self.a.mov_ri(Size::Q, r, i64::MAX)?;
        self.a.jmp(done)?;
        self.a.bind(lzero)?;
        self.a.alu_rr(AluOp::Xor, Size::D, r, r);
        self.a.bind(done)?;
        if t.bits < 64 {
            self.clamp(r, t, true, true, true)?;
        }
        Ok(())
    }

    fn float_to_float(&mut self, ff: FloatBits, tf: FloatBits, a: Operand, v: Value) -> Res<()> {
        let tp = prec(tf)?;
        match ff {
            FloatBits::F16 | FloatBits::B16 => {
                let g = self.read_g(a)?;
                let x = self.alloc_x()?;
                self.half_to_f32(g, ff == FloatBits::B16, x)?;
                if tp == Prec::F64 {
                    self.a.cvts2s(Prec::F32, x, x)?;
                }
                self.bind(v, R::X(x))
            }
            _ => {
                let fp = prec(ff)?;
                if fp == tp {
                    let x = self.take_x(a, fp)?;
                    return self.bind(v, R::X(x));
                }
                let ax = self.read_x(a, fp)?;
                let x = self.alloc_x()?;
                self.a.cvts2s(fp, x, ax)?;
                self.bind(v, R::X(x))
            }
        }
    }

    fn cast_try(
        &mut self,
        from: Sk,
        to: Sk,
        a: Operand,
        rt: TypeId,
        o: (u32, u32, TypeId),
        v: Value,
    ) -> Res<()> {
        let (none, some, pty) = o;
        let pcls = self.vcls(pty)?;
        let (lnone, lsome, lend) = (self.a.new_label(), self.a.new_label(), self.a.new_label());
        // Every register is taken before the first branch (the cache must agree on all paths).
        enum Payload {
            G(Gpr, u8),
            X(Xmm, Prec),
        }
        let payload = match (from, to) {
            (Sk::Float(ff), Sk::Int(t)) => {
                self.f64_work(a, ff)?;
                let r = self.alloc_g()?;
                let (d, off, _) = self.option_home(rt, some, v)?;
                self.float_try_int(r, t, lnone, lsome)?;
                (Payload::G(r, gbytes(pcls)), d, off)
            }
            (Sk::Float(ff), Sk::Float(tf)) => {
                let tp = prec(tf)?;
                if ff == FloatBits::F64 && tp == Prec::F32 {
                    let ax = self.read_x(a, Prec::F64)?;
                    let x = self.alloc_x()?;
                    let (d, off, _) = self.option_home(rt, some, v)?;
                    self.a.cvts2s(Prec::F64, x, ax)?;
                    self.a.cvts2s(Prec::F32, XS0, x)?;
                    self.a.ucomis(Prec::F64, XS0, ax)?;
                    self.a.jcc(Cond::Ne, lnone)?;
                    (Payload::X(x, tp), d, off)
                } else {
                    // widening or identity: always exact
                    let x = self.alloc_x()?;
                    match ff {
                        FloatBits::F16 | FloatBits::B16 => {
                            let g = self.read_g(a)?;
                            self.half_to_f32(g, ff == FloatBits::B16, x)?;
                            if tp == Prec::F64 {
                                self.a.cvts2s(Prec::F32, x, x)?;
                            }
                        }
                        _ => {
                            let fp = prec(ff)?;
                            let ax = self.read_x(a, fp)?;
                            if fp == tp {
                                self.a.movaps(x, ax);
                            } else {
                                self.a.cvts2s(fp, x, ax)?;
                            }
                        }
                    }
                    let (d, off, _) = self.option_home(rt, some, v)?;
                    (Payload::X(x, tp), d, off)
                }
            }
            (_, Sk::Float(tf)) => {
                let s = int_like(from).ok_or_else(|| bad("cast source"))?;
                let p = prec(tf)?;
                let digits = if p == Prec::F32 { 24 } else { 53 };
                let r = self.read_g(a)?;
                let t2 = self.alloc_g()?;
                let x = self.alloc_x()?;
                let (d, off, _) = self.option_home(rt, some, v)?;
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
                self.a.alu_ri(AluOp::Cmp, Size::Q, S1, digits)?;
                self.a.jcc(Cond::Ae, lnone)?;
                self.a.bind(exact)?;
                self.cvt_int(p, x, r, s)?;
                (Payload::X(x, p), d, off)
            }
            (_, Sk::Int(t)) => {
                let s = int_like(from).ok_or_else(|| bad("cast source"))?;
                let r = self.new_g(a)?;
                let (d, off, _) = self.option_home(rt, some, v)?;
                self.range_jump(r, s, t, lnone)?;
                (Payload::G(r, gbytes(pcls)), d, off)
            }
            (_, Sk::Char) => {
                let r = self.new_g(a)?;
                let (d, off, _) = self.option_home(rt, some, v)?;
                self.a.alu_ri(AluOp::Cmp, Size::Q, r, 0x10_ffff)?;
                self.a.jcc(Cond::A, lnone)?;
                self.a.lea(Size::Q, S0, Mem::base(r, -0xd800))?;
                self.a.alu_ri(AluOp::Cmp, Size::Q, S0, 0x7ff)?;
                self.a.jcc(Cond::Be, lnone)?;
                (Payload::G(r, 4), d, off)
            }
            _ => return Err(unsup(format!("cast.try {from:?} -> {to:?}"))),
        };
        let (pl, d, off) = payload;
        self.a.bind(lsome)?;
        self.store_imm(d, 4, i64::from(some))?;
        let pm = disp(d, off)?;
        match pl {
            Payload::G(r, bytes) => self.a.mov_mr(size_of_bytes(bytes), pm, r)?,
            Payload::X(x, p) => self.a.movs_store(p, pm, x)?,
        }
        self.a.jmp(lend)?;
        self.a.bind(lnone)?;
        self.store_imm(d, 4, i64::from(none))?;
        self.a.bind(lend)?;
        Ok(())
    }

    /// `r = XS0 as i64` when integral and in `t`'s range, else jump to `lnone`; may jump to
    /// `lsome` (u64 values ≥ 2^63).
    fn float_try_int(&mut self, r: Gpr, t: IntTy, lnone: Label, lsome: Label) -> Res<()> {
        let lhigh = self.a.new_label();
        self.f64_const(XS1, F64_M2P63)?;
        self.a.ucomis(Prec::F64, XS0, XS1)?;
        self.a.jcc(Cond::B, lnone)?;
        self.f64_const(XS1, F64_2P63)?;
        self.a.ucomis(Prec::F64, XS0, XS1)?;
        self.a.jcc(Cond::Ae, lhigh)?;
        self.a.cvtts2si(Prec::F64, Size::Q, r, XS0)?;
        self.a.cvtsi2s(Prec::F64, XS1, Size::Q, r)?;
        self.a.ucomis(Prec::F64, XS1, XS0)?;
        self.a.jcc(Cond::Ne, lnone)?;
        self.range_jump(r, IntTy::new(64, true), t, lnone)?;
        self.a.jmp(lsome)?;
        self.a.bind(lhigh)?;
        if !t.signed && t.bits == 64 {
            self.f64_const(XS1, F64_2P64)?;
            self.a.ucomis(Prec::F64, XS0, XS1)?;
            self.a.jcc(Cond::Ae, lnone)?;
            self.f64_const(XS1, F64_2P63)?;
            self.a.sse(SseOp::Sub, Prec::F64, XS0, XS1)?;
            self.a.cvtts2si(Prec::F64, Size::Q, r, XS0)?;
            self.a.mov_ri(Size::Q, S0, i64::MIN)?;
            self.a.alu_rr(AluOp::Xor, Size::Q, r, S0);
            self.a.jmp(lsome)?;
        } else {
            self.a.jmp(lnone)?;
        }
        Ok(())
    }
}

fn gbytes(c: VCls) -> u8 {
    match c {
        VCls::G { bytes, .. } => bytes,
        _ => 8,
    }
}
