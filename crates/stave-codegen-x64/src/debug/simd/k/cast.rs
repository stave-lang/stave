//! K mode: lane casts, widen and narrow, in the native AVX-512 forms: the `vpmov*`
//! down-conversions (truncating, signed- and unsigned-saturating), the `vcvt*` conversions
//! between floats and 32- and 64-bit ints of either signedness, and for `cast.try` the lanes that
//! do not convert as compares into a mask register, tested with `kortest`.
//!
//! The out-of-range result of a truncating float to int conversion is the integer indefinite:
//! MIN for a signed one, MAX for an unsigned one (NaN included).

use stave_ir::arith::IntTy;
use stave_ir::simd::LaneTy;
use stave_ir::{CastKind, Operand, TypeId, Value};

use super::{KPCMP, KPCMPEQ, KPCMPU, KPMAXS, KPMINU, NE, NLE, fp};
use crate::debug::func::*;
use crate::debug::simd::{Sh, by_w};
use crate::debug::{Res, bad};
use crate::enc::*;

/// Predicates of `vcmpp*`: not equal or unordered; ordered and not equal; at least (false for
/// NaN); ordered.
const NEQ_UQ: u8 = 0x04;
const NEQ_OQ: u8 = 0x0c;
const GE: u8 = 0x0d;
const ORD: u8 = 0x07;

/// How a down-conversion treats a lane out of the narrow range.
#[derive(Clone, Copy)]
enum Down {
    Wrap = 0,
    Signed = 1,
    Unsigned = 2,
}

/// The down-conversion of `fw`-byte lanes to `tw`-byte ones.
fn down_op(fw: u32, tw: u32, d: Down) -> EOp {
    use EOp::*;
    let ops = match (fw, tw) {
        (2, _) => [Pmovwb, Pmovswb, Pmovuswb],
        (4, 1) => [Pmovdb, Pmovsdb, Pmovusdb],
        (4, _) => [Pmovdw, Pmovsdw, Pmovusdw],
        (_, 1) => [Pmovqb, Pmovsqb, Pmovusqb],
        (_, 2) => [Pmovqw, Pmovsqw, Pmovusqw],
        _ => [Pmovqd, Pmovsqd, Pmovusqd],
    };
    ops[d as usize]
}

/// The extension of `fw`-byte lanes to `tw`-byte ones.
fn ext_op(fw: u32, tw: u32, signed: bool) -> EOp {
    use EOp::*;
    match (fw, tw, signed) {
        (1, 2, true) => Pmovsxbw,
        (1, 4, true) => Pmovsxbd,
        (1, _, true) => Pmovsxbq,
        (2, 4, true) => Pmovsxwd,
        (2, _, true) => Pmovsxwq,
        (_, _, true) => Pmovsxdq,
        (1, 2, false) => Pmovzxbw,
        (1, 4, false) => Pmovzxbd,
        (1, _, false) => Pmovzxbq,
        (2, 4, false) => Pmovzxwd,
        (2, _, false) => Pmovzxwq,
        (_, _, false) => Pmovzxdq,
    }
}

/// The conversion of `w`-byte int lanes to floats of `p`, rounding to nearest even.
fn cvt_op(p: Prec, w: u32, signed: bool) -> EOp {
    use EOp::*;
    match (p, w, signed) {
        (Prec::F32, 4, true) => Cvtdq2ps,
        (Prec::F32, 4, false) => Cvtudq2ps,
        (Prec::F32, _, true) => Cvtqq2ps,
        (Prec::F32, _, false) => Cvtuqq2ps,
        (Prec::F64, 4, true) => Cvtdq2pd,
        (Prec::F64, 4, false) => Cvtudq2pd,
        (Prec::F64, _, true) => Cvtqq2pd,
        (Prec::F64, _, false) => Cvtuqq2pd,
    }
}

/// The truncating conversion of floats of `p` to `w`-byte int lanes.
fn cvtt_op(p: Prec, w: u32, signed: bool) -> EOp {
    use EOp::*;
    match (p, w, signed) {
        (Prec::F32, 4, true) => Cvttps2dq,
        (Prec::F32, 4, false) => Cvttps2udq,
        (Prec::F32, _, true) => Cvttps2qq,
        (Prec::F32, _, false) => Cvttps2uqq,
        (Prec::F64, 4, true) => Cvttpd2dq,
        (Prec::F64, 4, false) => Cvttpd2udq,
        (Prec::F64, _, true) => Cvttpd2qq,
        (Prec::F64, _, false) => Cvttpd2uqq,
    }
}

/// The bits of `v` in the float format `p` (rounded).
fn fbits(p: Prec, v: i128) -> u64 {
    match p {
        Prec::F32 => u64::from((v as f32).to_bits()),
        Prec::F64 => (v as f64).to_bits(),
    }
}

/// `n` lanes of `w` bytes, each `f(lane)`, little-endian.
fn lanes(n: u32, w: u32, f: impl Fn(u32) -> u64) -> Vec<u8> {
    (0..n).flat_map(|j| f(j).to_le_bytes()[..w as usize].to_vec()).collect()
}

impl Cg<'_, '_> {
    /// `d` = the `n` lanes of `fw` bytes in `s` extended to `tw` bytes (`signed`: by sign).
    fn kext(&mut self, n: u32, fw: u32, tw: u32, signed: bool, d: Xmm, s: Xmm) -> Res<()> {
        self.ku(ext_op(fw, tw, signed), n * tw, d, s)
    }

    /// `d` = the `n` lanes of `fw` bytes in `s` down-converted by `op` (zeros above).
    fn kdown(&mut self, op: EOp, n: u32, fw: u32, d: Xmm, s: Xmm) -> Res<()> {
        self.ev(Ev::new(op, EVL::of_size(n * fw), s.num(), 0, d))
    }

    /// `d` = the constant `bytes` (a whole register of `size` bytes' length).
    fn kconst(&mut self, d: Xmm, size: u32, bytes: Vec<u8>) -> Res<()> {
        let m = self.vpool_mem(bytes);
        self.ku(EOp::Movdqu64, size, d, m)
    }

    /// `d` = the int lanes in `x` (shape `from`) as the floats of `to`, rounded to nearest even.
    fn kint_to_float(&mut self, d: Xmm, x: Xmm, from: Sh, to: Sh, s: IntTy) -> Res<()> {
        let p = self.vprec(to)?;
        let (src, w) = if from.w < 4 {
            // 8- and 16-bit lanes are exact as 32-bit ones, and signed after either extension
            self.kext(from.n, from.w, 4, s.signed, d, x)?;
            (d, 4)
        } else {
            (x, from.w)
        };
        let signed = s.signed || from.w < 4;
        self.ku(cvt_op(p, w, signed), (from.n * w).max(to.size), d, src)
    }

    /// `d` = the float lanes in `x` (shape `from`) in the float format of `to`.
    fn kfloat_to_float(&mut self, d: Xmm, x: Xmm, from: Sh, to: Sh) -> Res<()> {
        match (self.vprec(from)?, self.vprec(to)?) {
            (Prec::F32, Prec::F64) => self.ku(EOp::Cvtps2pd, to.size, d, x),
            (Prec::F64, Prec::F32) => self.ku(EOp::Cvtpd2ps, from.size, d, x),
            _ if d == x => Ok(()),
            _ => self.ku(EOp::Movdqa64, to.size, d, x),
        }
    }

    pub(in crate::debug::simd) fn k_cast(
        &mut self,
        kind: CastKind,
        a: Operand,
        rt: TypeId,
        rv: Value,
    ) -> Res<()> {
        let from = self.voshape(a)?;
        if kind == CastKind::Try {
            return self.k_cast_try(a, from, rt, rv);
        }
        let to = self.vshape(rt)?;
        match (from.lane, to.lane) {
            (LaneTy::Int(s), LaneTy::Int(t)) => self.k_cast_int(kind == CastKind::Sat, a, from, to, s, t, rv),
            (LaneTy::Int(s), LaneTy::Float(_)) => {
                let x = self.vget(a, from.size)?;
                self.kint_to_float(x, x, from, to, s)?;
                self.vput(rv, x, to.size)
            }
            (LaneTy::Float(_), LaneTy::Float(_)) => {
                let x = self.vget(a, from.size)?;
                self.kfloat_to_float(x, x, from, to)?;
                self.vput(rv, x, to.size)
            }
            (LaneTy::Float(_), LaneTy::Int(t)) if kind == CastKind::Sat => {
                self.k_float_sat(a, from, to, t, rv)
            }
            _ => Err(bad(format!("{kind:?} cast from float to int lanes"))),
        }
    }

    /// Int lanes to int lanes: extended by the source's sign, or down-converted; a saturating
    /// cast clamps first where the down-conversion does not saturate to the target's range.
    #[allow(clippy::too_many_arguments)]
    fn k_cast_int(
        &mut self,
        sat: bool,
        a: Operand,
        from: Sh,
        to: Sh,
        s: IntTy,
        t: IntTy,
        rv: Value,
    ) -> Res<()> {
        let x = self.vget(a, from.size)?;
        let (n, fw) = (from.n, from.w);
        if sat && s.signed && !t.signed {
            // negative lanes to 0
            self.vzero(XS0)?;
            self.kb(by_w(fw, KPMAXS), from.size, x, x, XS0)?;
        } else if sat && !s.signed && t.signed && t.bits <= s.bits {
            self.ksplat(XS0, from.size, fw, t.max() as u64)?;
            self.kb(by_w(fw, KPMINU), from.size, x, x, XS0)?;
        }
        if to.w > fw {
            self.kext(n, fw, to.w, s.signed, x, x)?;
        } else if to.w < fw {
            let d = match (sat, s.signed, t.signed) {
                (true, true, true) => Down::Signed,
                // a signed source is clamped to 0 and up
                (true, _, false) => Down::Unsigned,
                // clamped to the target's MAX, or wrapping
                _ => Down::Wrap,
            };
            self.kdown(down_op(fw, to.w, d), n, fw, x, x)?;
        }
        self.vput(rv, x, to.size)
    }

    /// Float lanes to int lanes, saturating: NaN gives 0, the lanes beyond the target's range
    /// its MIN or MAX, the others truncate.
    fn k_float_sat(&mut self, a: Operand, from: Sh, to: Sh, t: IntTy, rv: Value) -> Res<()> {
        let p = self.vprec(from)?;
        let (fs, ts, n) = (from.size, to.size, from.n);
        let size = fs.max(ts);
        let cmp = fp(p, EOp::Cmpps, EOp::Cmppd);
        if !t.signed && to.w >= 4 {
            // `vmaxp` gives its second operand for a NaN: NaN and negative lanes are +0; the
            // unsigned conversion gives MAX above the range
            let x = self.vget(a, fs)?;
            self.vzero(XS0)?;
            self.kb(fp(p, EOp::Maxps, EOp::Maxpd), fs, x, x, XS0)?;
            self.ku(cvtt_op(p, to.w, false), size, x, x)?;
            return self.vput(rv, x, ts);
        }
        if to.w >= 4 && !(p == Prec::F64 && to.w == 4) {
            // MAX is not exact in the source format: the lanes from 2^(bits-1) up take it over
            // the conversion's MIN, and NaN lanes are zeroed
            let x = self.vread(a, fs)?;
            let r = self.alloc_x()?;
            let (kmax, knum) = (self.alloc_k()?, self.alloc_k()?);
            let mov = if to.w == 4 { EOp::Movdqa32 } else { EOp::Movdqa64 };
            let vl = EVL::of_size(ts);
            self.ku(cvtt_op(p, to.w, true), size, r, x)?;
            self.ksplat(XS0, fs, from.w, fbits(p, 1i128 << (t.bits - 1)))?;
            self.kcmp(cmp, fs, kmax, x, XS0, Some(GE))?;
            self.ksplat(XS1, ts, to.w, t.max() as u64)?;
            self.ev(Ev::new(mov, vl, r.num(), 0, XS1).mask(kmax, false))?;
            self.kcmp(cmp, fs, knum, x, x, Some(ORD))?;
            self.ev(Ev::new(mov, vl, r.num(), 0, r).mask(knum, true))?;
            return self.vput(rv, r, ts);
        }
        // the target's bounds are exact in the source format: clamp, convert, zero NaN lanes
        let x = self.vget(a, fs)?;
        let k = self.alloc_k()?;
        self.kcmp(cmp, fs, k, x, x, Some(ORD))?;
        self.ksplat(XS0, fs, from.w, fbits(p, t.min()))?;
        self.kb(fp(p, EOp::Maxps, EOp::Maxpd), fs, x, x, XS0)?;
        self.ksplat(XS0, fs, from.w, fbits(p, t.max()))?;
        self.kb(fp(p, EOp::Minps, EOp::Minpd), fs, x, x, XS0)?;
        let cvt = fp(p, EOp::Cvttps2dq, EOp::Cvttpd2dq);
        self.ev(Ev::new(cvt, EVL::of_size(fs), x.num(), 0, x).mask(k, true))?;
        if to.w < 4 {
            self.kdown(down_op(4, to.w, Down::Wrap), n, 4, x, x)?;
        }
        self.vput(rv, x, ts)
    }

    /// `Some` of the converted lanes iff every lane converts exactly, as the scalar `as?`.
    /// The payload is computed in every case; a check sets ZF iff every lane converts.
    fn k_cast_try(&mut self, a: Operand, from: Sh, rt: TypeId, rv: Value) -> Res<()> {
        let (none, some, pty) =
            self.cx.m.types.option_shape(rt).ok_or_else(|| bad("vec.cast.try result must be an Option"))?;
        let to = self.vshape(pty)?;
        let (fs, ts, n, fw, tw) = (from.size, to.size, from.n, from.w, to.w);
        let size = fs.max(ts);
        let ks = KSize::of_lanes(n);
        let x = self.vread(a, fs)?;
        let r = self.alloc_x()?;
        // a second mask of lanes that fail, for the checks of two compares
        let two = match (from.int(), to.int()) {
            (Some(s), None) => !s.signed && fw >= 4 && !(fw == 4 && to.float() == Some(Prec::F64)),
            (None, Some(t)) => !t.signed && tw >= 4 && !(tw == 4 && from.float() == Some(Prec::F64)),
            _ => false,
        };
        let k2 = if two { self.alloc_k()? } else { KS0 };
        let (d, off, _) = self.option_home(rt, some, rv)?;
        // the register of the payload, and whether a check ran
        let (pl, checked) = match (from.lane, to.lane) {
            (LaneTy::Int(s), LaneTy::Int(t)) => {
                if tw == fw {
                    // the same lanes: they fit unless the sign changes and the top bit is set
                    if s.signed != t.signed {
                        self.kmask_from_vec(KS0, x, n, fw)?;
                        self.a.kortest(ks, KS0, KS0);
                    }
                    (x, s.signed != t.signed)
                } else if tw > fw {
                    // only a negative lane to an unsigned one does not fit
                    let neg = s.signed && !t.signed;
                    if neg {
                        self.kmask_from_vec(KS0, x, n, fw)?;
                        self.a.kortest(ks, KS0, KS0);
                    }
                    self.kext(n, fw, tw, s.signed, r, x)?;
                    (r, neg)
                } else if s.signed && t.signed {
                    // a lane fits iff the truncation extends back to it
                    self.kdown(down_op(fw, tw, Down::Wrap), n, fw, r, x)?;
                    self.kext(n, tw, fw, true, XS0, r)?;
                    self.kcmp(by_w(fw, KPCMP), fs, KS0, XS0, x, Some(NE))?;
                    self.a.kortest(ks, KS0, KS0);
                    (r, true)
                } else {
                    // as unsigned, a lane fits iff it is at most the target's MAX (a negative
                    // signed lane is above it)
                    self.ksplat(XS0, fs, fw, t.max() as u64)?;
                    self.kcmp(by_w(fw, KPCMPU), fs, KS0, x, XS0, Some(NLE))?;
                    self.a.kortest(ks, KS0, KS0);
                    self.kdown(down_op(fw, tw, Down::Wrap), n, fw, r, x)?;
                    (r, true)
                }
            }
            (LaneTy::Int(s), LaneTy::Float(_)) => {
                let p = self.vprec(to)?;
                self.kint_to_float(r, x, from, to, s)?;
                if fw < 4 || (fw == 4 && p == Prec::F64) {
                    (r, false)
                } else {
                    // exact iff the truncating conversion back gives the lane; that one gives
                    // MAX for an unsigned lane rounded up to 2^bits, so MAX itself fails apart
                    self.ku(cvtt_op(p, fw, s.signed), size, XS0, r)?;
                    self.kcmp(by_w(fw, KPCMP), fs, KS0, XS0, x, Some(NE))?;
                    if two {
                        self.ksplat(XS1, fs, fw, u64::MAX)?;
                        self.kcmp(by_w(fw, KPCMPEQ), fs, k2, x, XS1, None)?;
                    }
                    self.a.kortest(ks, KS0, k2);
                    (r, true)
                }
            }
            (LaneTy::Float(_), LaneTy::Float(_)) => match (self.vprec(from)?, self.vprec(to)?) {
                (Prec::F64, Prec::F32) => {
                    // exact iff the rounded lane widens back to it; a NaN converts
                    self.ku(EOp::Cvtpd2ps, fs, r, x)?;
                    self.ku(EOp::Cvtps2pd, fs, XS0, r)?;
                    self.kcmp(EOp::Cmppd, fs, KS0, XS0, x, Some(NEQ_OQ))?;
                    self.a.kortest(ks, KS0, KS0);
                    (r, true)
                }
                (Prec::F32, Prec::F64) => {
                    self.kfloat_to_float(r, x, from, to)?;
                    (r, false)
                }
                _ => (x, false),
            },
            (LaneTy::Float(_), LaneTy::Int(t)) => {
                let p = self.vprec(from)?;
                let cmp = fp(p, EOp::Cmpps, EOp::Cmppd);
                if tw >= 4 {
                    // exact iff the conversion back gives the lane: NaN, a fraction, and a lane
                    // out of range (converted to MIN, or MAX) do not. MAX converted back
                    // rounds up to 2^bits but for f64 to u32: that lane fails apart.
                    self.ku(cvtt_op(p, tw, t.signed), size, r, x)?;
                    self.ku(cvt_op(p, tw, t.signed), size, XS0, r)?;
                    self.kcmp(cmp, fs, KS0, XS0, x, Some(NEQ_UQ))?;
                    if two {
                        self.ksplat(XS1, fs, fw, fbits(p, 1i128 << t.bits))?;
                        self.kcmp(cmp, fs, k2, x, XS1, Some(GE))?;
                    }
                } else {
                    // clamped to the target's range (NaN to MIN), converted, and exact iff
                    // the conversion back gives the lane
                    self.ksplat(XS0, fs, fw, fbits(p, t.min()))?;
                    self.kb(fp(p, EOp::Maxps, EOp::Maxpd), fs, r, x, XS0)?;
                    self.ksplat(XS0, fs, fw, fbits(p, t.max()))?;
                    self.kb(fp(p, EOp::Minps, EOp::Minpd), fs, r, r, XS0)?;
                    self.ku(cvtt_op(p, 4, true), fs, r, r)?;
                    self.ku(cvt_op(p, 4, true), fs, XS0, r)?;
                    self.kcmp(cmp, fs, KS0, XS0, x, Some(NEQ_UQ))?;
                    self.kdown(down_op(4, tw, Down::Wrap), n, 4, r, r)?;
                }
                self.a.kortest(ks, KS0, k2);
                (r, true)
            }
        };
        if !checked {
            self.store_imm(d, 4, i64::from(some))?;
        } else if (none, some) == (0, 1) {
            self.a.setcc(Cond::E, S0);
            self.a.movzx(Size::D, S0, Size::B, S0)?;
            self.a.mov_mr(Size::D, d, S0)?;
        } else {
            self.a.mov_ri(Size::D, S0, i64::from(none))?;
            self.a.mov_ri(Size::D, S1, i64::from(some))?;
            self.a.cmov(Cond::E, Size::D, S0, S1)?;
            self.a.mov_mr(Size::D, d, S0)?;
        }
        self.vstore(disp(d, off)?, pl, ts)
    }

    /// The low or high half of the lanes, each converted exactly to the double-width lane.
    pub(in crate::debug::simd) fn k_widen(&mut self, hi: bool, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let from = self.voshape(a)?;
        let to = self.vshape(rt)?;
        let x = self.vget(a, from.size)?;
        let float = from.float().is_some();
        if hi {
            match from.size {
                64 => {
                    let op = if float { EOp::Extractf64x4 } else { EOp::Extracti64x4 };
                    self.ev(Ev::new(op, EVL::L512, x.num(), 0, x).imm(1))?;
                }
                32 => {
                    let op = if float { EOp::Extractf32x4 } else { EOp::Extracti32x4 };
                    self.ev(Ev::new(op, EVL::L256, x.num(), 0, x).imm(1))?;
                }
                // the upper quadword; below 16 bytes the value has zeros above it
                16 => self.kui(EOp::Pshufd, 16, x, x, 0xee)?,
                s => self.ksh(EShiftI::Psrlq, 16, x, (s * 4) as u8)?,
            }
        }
        let op = match from.lane {
            LaneTy::Float(_) => {
                self.vprec(from)?;
                EOp::Cvtps2pd
            }
            LaneTy::Int(t) => ext_op(from.w, to.w, t.signed),
        };
        self.ku(op, to.size, x, x)?;
        self.vput(rv, x, to.size)
    }

    /// The lanes of `a` then `b`, truncated or saturated to the half-width lane. 32- and
    /// 64-bit lanes truncate by a two-source permute of their low halves; signed 16- and 32-bit
    /// lanes saturate by a pack, whose 128-bit blocks a `vpermq` puts in order; the others are
    /// down-converted one operand at a time and joined.
    pub(in crate::debug::simd) fn k_narrow(
        &mut self,
        sat: bool,
        a: Operand,
        b: Operand,
        rt: TypeId,
        rv: Value,
    ) -> Res<()> {
        let to = self.vshape(rt)?;
        let from = self.voshape(a)?;
        let s = self.vint(from)?;
        let (fs, fw, tw) = (from.size, from.w, to.w);
        // the register length the operands are held at
        let reg = EVL::of_size(fs).bytes();
        let xb = self.vread(b, fs)?;
        let x = self.vget(a, fs)?;
        match (sat, s.signed, fw) {
            (false, _, 4 | 8) => {
                // lane j of the result: the low half of lane j of `a`, then of `b` (the second
                // table), then a zero half above `a`'s value
                let per = reg / tw;
                let idx = lanes(per, tw, |j| {
                    u64::from(if j < from.n {
                        2 * j
                    } else if j < 2 * from.n {
                        per + 2 * (j - from.n)
                    } else {
                        fs / tw
                    })
                });
                self.kconst(XS0, reg, idx)?;
                self.kb(if fw == 4 { EOp::Permt2w } else { EOp::Permt2d }, fs, x, XS0, xb)?;
            }
            (true, true, 2 | 4) if fs >= 16 => {
                // each 128-bit block: 8 bytes of `a`'s block, 8 of `b`'s
                self.kb(if fw == 2 { EOp::Packsswb } else { EOp::Packssdw }, fs, x, x, xb)?;
                if fs > 16 {
                    let q = fs / 16;
                    let idx = lanes(2 * q, 8, |j| u64::from(if j < q { 2 * j } else { 2 * (j - q) + 1 }));
                    self.kconst(XS0, fs, idx)?;
                    self.kb(EOp::Permq, fs, x, XS0, x)?;
                }
            }
            _ => {
                let d = match (sat, s.signed) {
                    (false, _) => Down::Wrap,
                    (true, true) => Down::Signed,
                    (true, false) => Down::Unsigned,
                };
                let op = down_op(fw, tw, d);
                self.kdown(op, from.n, fw, x, x)?;
                self.kdown(op, from.n, fw, XS0, xb)?;
                // `b`'s half right above `a`'s
                match fs {
                    64 => self.ev(Ev::new(EOp::Inserti64x4, EVL::L512, x.num(), x.num(), XS0).imm(1))?,
                    32 => self.ev(Ev::new(EOp::Inserti32x4, EVL::L256, x.num(), x.num(), XS0).imm(1))?,
                    16 => self.kb(EOp::Punpcklqdq, 16, x, x, XS0)?,
                    8 => self.kb(EOp::Punpckldq, 16, x, x, XS0)?,
                    _ => self.kb(EOp::Punpcklwd, 16, x, x, XS0)?,
                }
            }
        }
        self.vput(rv, x, to.size)
    }
}
