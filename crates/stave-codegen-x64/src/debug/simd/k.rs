//! K mode: the vector and mask ops of a function compiled for the AVX-512 set (`Feats::v4`).
//!
//! Every op is the v4 recipe of the lowering table (`stave-target/tables/x86.toml`): EVEX
//! instructions at the length of the value (`xmm` up to 16 bytes, `ymm` for 32, `zmm` for 64),
//! with VEX instructions on `xmm` where a recipe names them (the tails of the reductions).
//!
//! A mask value is one bit per lane in a mask register (`VCls::K`): bit `j` is lane `j`, and the
//! bits from the lane count up are unspecified, in the register and in its 8-byte home. What
//! reads them as an integer narrows first ([`Cg::kbits`]); a write mask of a value of fewer than
//! 16 bytes is narrowed too, so no lane above the value is touched. Vector values are held as
//! in every other function (`simd`), a 64-byte one in a `zmm`.
//!
//! Operands are read ([`Cg::vread`], [`Cg::read_k`]) before a temporary is asked for, so the
//! register an op computes into may be the one a block param is about to leave.

use stave_ir::layout::Shape;
use stave_ir::simd::{LaneTy, VecOp};
use stave_ir::{Operand, PanicKind, TypeId, Value};

use super::{Sh, VSrc, by_w, y};
use crate::debug::abi::VCls;
use crate::debug::func::*;
use crate::debug::{Res, bad, unsup};
use crate::enc::*;

fn fp(p: Prec, ps: EOp, pd: EOp) -> EOp {
    if p == Prec::F32 { ps } else { pd }
}

fn erm(s: VSrc) -> ERm {
    match s {
        VSrc::R(x) => ERm::R(x.num()),
        VSrc::M(m) => ERm::M(m),
    }
}

const KPADD: [EOp; 4] = [EOp::Paddb, EOp::Paddw, EOp::Paddd, EOp::Paddq];
const KPSUB: [EOp; 4] = [EOp::Psubb, EOp::Psubw, EOp::Psubd, EOp::Psubq];
const KPMINS: [EOp; 4] = [EOp::Pminsb, EOp::Pminsw, EOp::Pminsd, EOp::Pminsq];
const KPMINU: [EOp; 4] = [EOp::Pminub, EOp::Pminuw, EOp::Pminud, EOp::Pminuq];
const KPMAXS: [EOp; 4] = [EOp::Pmaxsb, EOp::Pmaxsw, EOp::Pmaxsd, EOp::Pmaxsq];
const KPMAXU: [EOp; 4] = [EOp::Pmaxub, EOp::Pmaxuw, EOp::Pmaxud, EOp::Pmaxuq];
const KPCMPEQ: [EOp; 4] = [EOp::Pcmpeqb, EOp::Pcmpeqw, EOp::Pcmpeqd, EOp::Pcmpeqq];
const KPCMPGT: [EOp; 4] = [EOp::Pcmpgtb, EOp::Pcmpgtw, EOp::Pcmpgtd, EOp::Pcmpgtq];
const KPCMP: [EOp; 4] = [EOp::Pcmpb, EOp::Pcmpw, EOp::Pcmpd, EOp::Pcmpq];
const KPCMPU: [EOp; 4] = [EOp::Pcmpub, EOp::Pcmpuw, EOp::Pcmpud, EOp::Pcmpuq];
const KPBLENDM: [EOp; 4] = [EOp::Pblendmb, EOp::Pblendmw, EOp::Pblendmd, EOp::Pblendmq];
const KMOVDQU: [EOp; 4] = [EOp::Movdqu8, EOp::Movdqu16, EOp::Movdqu32, EOp::Movdqu64];
const KMOVDQU_ST: [EOp; 4] = [EOp::Movdqu8St, EOp::Movdqu16St, EOp::Movdqu32St, EOp::Movdqu64St];
const KPBROADCAST: [EOp; 4] = [EOp::Pbroadcastb, EOp::Pbroadcastw, EOp::Pbroadcastd, EOp::Pbroadcastq];

/// Predicates of `vpcmp*` / `vpcmpu*`.
const LT: u8 = 1;
const LE: u8 = 2;
const NE: u8 = 4;
const NLT: u8 = 5;
const NLE: u8 = 6;
/// Predicate of `vcmpp*`: unordered.
const UNORD: u8 = 3;

impl Cg<'_, '_> {
    // ---------------------------------------------------------------- encodings

    /// Emits one EVEX instruction; one on `ymm` / `zmm` registers leaves upper parts dirty.
    pub(crate) fn ev(&mut self, e: Ev) -> Res<()> {
        if e.vl != EVL::L128 {
            self.ydirty = true;
            self.yused = true;
        }
        Ok(self.a.evex(e)?)
    }

    /// `d = a op b` on vectors of `size` bytes.
    fn kb(&mut self, op: EOp, size: u32, d: Xmm, a: Xmm, b: impl Into<ERm>) -> Res<()> {
        self.ev(Ev::new(op, EVL::of_size(size), d.num(), a.num(), b))
    }

    /// `d = a op b, imm`.
    fn kbi(&mut self, op: EOp, size: u32, d: Xmm, a: Xmm, b: impl Into<ERm>, imm: u8) -> Res<()> {
        self.ev(Ev::new(op, EVL::of_size(size), d.num(), a.num(), b).imm(imm))
    }

    /// `d = op s`.
    fn ku(&mut self, op: EOp, size: u32, d: Xmm, s: impl Into<ERm>) -> Res<()> {
        self.ev(Ev::new(op, EVL::of_size(size), d.num(), 0, s))
    }

    /// `d = op s, imm`.
    fn kui(&mut self, op: EOp, size: u32, d: Xmm, s: impl Into<ERm>, imm: u8) -> Res<()> {
        self.ev(Ev::new(op, EVL::of_size(size), d.num(), 0, s).imm(imm))
    }

    /// `k = a cmp b` (`imm`: the predicate of the instructions that take one).
    fn kcmp(&mut self, op: EOp, size: u32, k: KReg, a: Xmm, b: Xmm, imm: Option<u8>) -> Res<()> {
        let mut e = Ev::new(op, EVL::of_size(size), k.num(), a.num(), b);
        e.imm = imm;
        self.ev(e)
    }

    /// `d = d shift imm` per lane.
    fn ksh(&mut self, op: EShiftI, size: u32, d: Xmm, imm: u8) -> Res<()> {
        let vl = EVL::of_size(size);
        if vl != EVL::L128 {
            self.ydirty = true;
            self.yused = true;
        }
        Ok(self.a.evex_shift_i(op, vl, d.num(), d, imm, KS0, false)?)
    }

    /// `d = value` in every lane of `w` bytes, from the constant pool.
    fn ksplat(&mut self, d: Xmm, size: u32, w: u32, value: u64) -> Res<()> {
        let m = self.vpool_mem(value.to_le_bytes()[..w as usize].to_vec());
        self.ku(by_w(w, KPBROADCAST), size, d, m)
    }

    // ---------------------------------------------------------------- masks

    /// The vector of the lanes of mask `k` (`n` lanes of `w` bytes) in `x`: all ones where set.
    pub(crate) fn kmask_to_vec(&mut self, x: Xmm, k: KReg, n: u32, w: u32) -> Res<()> {
        let op = by_w(w, [EOp::Pmovm2b, EOp::Pmovm2w, EOp::Pmovm2d, EOp::Pmovm2q]);
        self.ev(Ev::new(op, EVL::of_size(n * w), x.num(), 0, k))
    }

    /// The mask of the lanes of the vector in `x` into `k`: a lane is set by its sign bit.
    pub(crate) fn kmask_from_vec(&mut self, k: KReg, x: Xmm, n: u32, w: u32) -> Res<()> {
        let op = by_w(w, [EOp::Pmovb2m, EOp::Pmovw2m, EOp::Pmovd2m, EOp::Pmovq2m]);
        self.ev(Ev::new(op, EVL::of_size(n * w), k.num(), 0, x))
    }

    /// Stores mask `k` at `dst` in its memory format, the vector of its lanes. Clobbers `XS0`.
    pub(crate) fn kmask_store(&mut self, dst: Mem, k: KReg, n: u32, w: u32) -> Res<()> {
        self.kmask_to_vec(XS0, k, n, w)?;
        self.vstore(dst, XS0, n * w)?;
        if !self.rel {
            self.vclean();
        }
        Ok(())
    }

    /// Loads the mask stored at `src` as the vector of its lanes into `k`. Clobbers `XS0`.
    pub(crate) fn kmask_load(&mut self, k: KReg, src: Mem, n: u32, w: u32) -> Res<()> {
        self.vload(XS0, src, n * w)?;
        self.kmask_from_vec(k, XS0, n, w)?;
        if !self.rel {
            self.vclean();
        }
        Ok(())
    }

    /// The bits of mask `k` of `n` lanes in `g`, zero above the lanes.
    pub(crate) fn kbits(&mut self, g: Gpr, k: KReg, n: u32) -> Res<()> {
        self.a.kmov_gk(KSize::of_lanes(n), g, k);
        if !matches!(n, 8 | 16 | 32 | 64) {
            self.a.alu_ri(AluOp::And, Size::D, g, (1i64 << n) - 1)?;
        }
        Ok(())
    }

    /// The write mask for the lanes of a value of shape `sh` under mask `k`: `k` itself for a
    /// value that fills its register, else a copy without the bits above the lanes (`bits`: a
    /// general register holding them already, from [`Self::kbits`]).
    fn kwrite_mask(&mut self, k: KReg, sh: Sh, bits: Gpr) -> Res<KReg> {
        if sh.size >= 16 {
            return Ok(k);
        }
        let t = self.alloc_k()?;
        self.a.kmov_kg(KSize::D, t, bits);
        Ok(t)
    }

    fn k_mask_bin(&mut self, op: VecOp, a: Operand, b: Operand, rv: Value) -> Res<()> {
        let (ka, kb) = (self.read_k(a)?, self.read_k(b)?);
        let r = self.alloc_k()?;
        let o = match op {
            VecOp::MaskAnd => KBin::And,
            VecOp::MaskOr => KBin::Or,
            _ => KBin::Xor,
        };
        self.a.kbin(o, KSize::Q, r, ka, kb);
        self.bind(rv, R::K(r))
    }

    fn k_mask_not(&mut self, a: Operand, rv: Value) -> Res<()> {
        let ka = self.read_k(a)?;
        let r = self.alloc_k()?;
        self.a.knot(KSize::Q, r, ka);
        self.bind(rv, R::K(r))
    }

    fn k_mask_const(&mut self, all: bool, rv: Value) -> Res<()> {
        let r = self.alloc_k()?;
        self.a.kbin(if all { KBin::Xnor } else { KBin::Xor }, KSize::Q, r, r, r);
        self.bind(rv, R::K(r))
    }

    fn k_to_bits(&mut self, m: Operand, rv: Value) -> Res<()> {
        let sh = self.voshape(m)?;
        let k = self.read_k(m)?;
        let g = self.alloc_g()?;
        self.kbits(g, k, sh.n)?;
        self.bind_g(rv, g)
    }

    fn k_to_vec(&mut self, m: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let k = self.read_k(m)?;
        let x = self.alloc_x()?;
        self.kmask_to_vec(x, k, sh.n, sh.w)?;
        self.vput(rv, x, sh.size)
    }

    /// The same lanes at another lane width: the same bits.
    fn k_rewiden(&mut self, m: Operand, rv: Value) -> Res<()> {
        let k = self.read_k(m)?;
        let r = self.alloc_k()?;
        self.a.kmov_kk(KSize::Q, r, k);
        self.bind(rv, R::K(r))
    }

    /// Lanes `j < k` set: the low `min(k, n)` bits.
    fn k_first_n(&mut self, k: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        if !self.fx.bmi2 {
            self.reserve_g(RCX)?;
        }
        let g = self.read_g(k)?;
        let t = self.alloc_g()?;
        let r = self.alloc_k()?;
        self.a.mov_rr(Size::Q, t, g);
        self.a.mov_ri(Size::D, S0, i64::from(sh.n))?;
        self.a.alu_rr(AluOp::Cmp, Size::Q, t, S0);
        self.a.cmov(Cond::A, Size::Q, t, S0)?;
        self.a.mov_ri(Size::Q, S0, -1)?;
        if self.fx.bmi2 {
            self.a.bzhi(Size::Q, t, S0, t)?;
        } else {
            // (1 << t) - 1, and all ones for a count of 64, which the shift takes as 0
            self.a.mov_rr(Size::Q, RCX, t);
            self.a.mov_ri(Size::D, S1, 1)?;
            self.a.shift_cl(ShiftOp::Shl, Size::Q, S1)?;
            self.a.alu_ri(AluOp::Sub, Size::Q, S1, 1)?;
            self.a.alu_ri(AluOp::Cmp, Size::Q, t, 64)?;
            self.a.cmov(Cond::B, Size::Q, S0, S1)?;
            self.a.mov_rr(Size::Q, t, S0);
        }
        self.a.kmov_kg(KSize::Q, r, t);
        self.bind(rv, R::K(r))
    }

    /// Lanes of `a` where the mask is set, else of `b`.
    fn k_select(&mut self, m: Operand, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let km = self.read_k(m)?;
        if matches!(self.vcls(rt)?, VCls::K { .. }) {
            let (ka, kb) = (self.read_k(a)?, self.read_k(b)?);
            let r = self.alloc_k()?;
            self.a.kbin(KBin::And, KSize::Q, KS0, km, ka);
            self.a.kbin(KBin::Andn, KSize::Q, r, km, kb);
            self.a.kbin(KBin::Or, KSize::Q, r, r, KS0);
            return self.bind(rv, R::K(r));
        }
        let (xa, xb) = (self.vread(a, sh.size)?, self.vread(b, sh.size)?);
        let r = self.alloc_x()?;
        let e = Ev::new(by_w(sh.w, KPBLENDM), EVL::of_size(sh.size), r.num(), xb.num(), xa);
        self.ev(e.mask(km, false))?;
        self.vput(rv, r, sh.size)
    }

    // ---------------------------------------------------------------- lanes

    fn k_splat(&mut self, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        match sh.float() {
            Some(p) => {
                let s = self.read_x(a, p)?;
                let x = self.alloc_x()?;
                // `vbroadcastsd` has no `xmm` form
                let op = match (p, sh.size > 16) {
                    (Prec::F32, _) => EOp::Broadcastss,
                    (Prec::F64, true) => EOp::Broadcastsd,
                    (Prec::F64, false) => EOp::Pbroadcastq,
                };
                self.ku(op, sh.size, x, s)?;
                self.vput(rv, x, sh.size)
            }
            None => {
                let g = self.read_g(a)?;
                let x = self.alloc_x()?;
                let op =
                    by_w(sh.w, [EOp::PbroadcastbR, EOp::PbroadcastwR, EOp::PbroadcastdR, EOp::PbroadcastqR]);
                self.ku(op, sh.size, x, g)?;
                self.vput(rv, x, sh.size)
            }
        }
    }

    fn k_float_bin(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let p = self.vprec(sh)?;
        let o = match op {
            VecOp::Add => fp(p, EOp::Addps, EOp::Addpd),
            VecOp::Sub => fp(p, EOp::Subps, EOp::Subpd),
            VecOp::Mul => fp(p, EOp::Mulps, EOp::Mulpd),
            VecOp::Div => fp(p, EOp::Divps, EOp::Divpd),
            VecOp::MinX86 => fp(p, EOp::Minps, EOp::Minpd),
            _ => fp(p, EOp::Maxps, EOp::Maxpd),
        };
        let s = self.vsrc(b, sh)?;
        let x = self.vget(a, sh.size)?;
        self.kb(o, sh.size, x, x, erm(s))?;
        self.vput(rv, x, sh.size)
    }

    fn k_float_un(&mut self, op: VecOp, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let p = self.vprec(sh)?;
        let sz = sh.size;
        let x = self.vget(a, sz)?;
        let round = fp(p, EOp::Rndscaleps, EOp::Rndscalepd);
        match op {
            VecOp::Neg => {
                self.ksplat(XS0, sz, sh.w, sh.sign())?;
                self.kb(fp(p, EOp::Xorps, EOp::Xorpd), sz, x, x, XS0)?;
            }
            VecOp::Abs => {
                self.ksplat(XS0, sz, sh.w, sh.ones() ^ sh.sign())?;
                self.kb(fp(p, EOp::Andps, EOp::Andpd), sz, x, x, XS0)?;
            }
            VecOp::Sqrt => self.ku(fp(p, EOp::Sqrtps, EOp::Sqrtpd), sz, x, x)?,
            // imm8: bit 3 = no precision exception; bits 1:0 = nearest even, down, up, toward zero
            VecOp::Round => self.kui(round, sz, x, x, 8)?,
            VecOp::Floor => self.kui(round, sz, x, x, 9)?,
            VecOp::Ceil => self.kui(round, sz, x, x, 10)?,
            _ => self.kui(round, sz, x, x, 11)?,
        }
        self.vput(rv, x, sz)
    }

    fn k_copysign(&mut self, a: Operand, s: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let sx = self.vread(s, sh.size)?;
        let x = self.vget(a, sh.size)?;
        self.ksplat(XS0, sh.size, sh.w, sh.sign())?;
        // per bit: the sign mask picks `s`, else `a`
        self.kbi(EOp::Pternlogq, sh.size, x, XS0, sx, 0xb8)?;
        self.vput(rv, x, sh.size)
    }

    fn k_fma(&mut self, a: Operand, b: Operand, c: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let p = self.vprec(sh)?;
        let (bx, cx) = (self.vread(b, sh.size)?, self.vread(c, sh.size)?);
        let x = self.vget(a, sh.size)?;
        // x = bx * x + cx
        self.kb(fp(p, EOp::Fmadd213ps, EOp::Fmadd213pd), sh.size, x, bx, cx)?;
        self.vput(rv, x, sh.size)
    }

    /// `XS0 = minimum(a, b)` / `maximum(a, b)` on the low `size` bytes: a NaN in either gives a
    /// quiet NaN, -0 is below +0. `vminp` returns its second operand for an unordered or equal
    /// pair, so the two operand orders differ only for (NaN, x) and (±0, ∓0): OR (min) or AND
    /// (max) of the two picks the zero by sign, and an unordered lane takes the sum, a quiet
    /// NaN. Clobbers `XS1` and `k`.
    fn kfminmax(&mut self, size: u32, p: Prec, min: bool, a: Xmm, b: Xmm, k: KReg) -> Res<()> {
        let m = if min { fp(p, EOp::Minps, EOp::Minpd) } else { fp(p, EOp::Maxps, EOp::Maxpd) };
        let join = match (min, p) {
            (true, Prec::F32) => EOp::Orps,
            (true, Prec::F64) => EOp::Orpd,
            (false, Prec::F32) => EOp::Andps,
            (false, Prec::F64) => EOp::Andpd,
        };
        self.kb(m, size, XS0, b, a)?;
        self.kb(m, size, XS1, a, b)?;
        self.kb(join, size, XS0, XS0, XS1)?;
        self.kcmp(fp(p, EOp::Cmpps, EOp::Cmppd), size, k, a, b, Some(UNORD))?;
        let add = Ev::new(fp(p, EOp::Addps, EOp::Addpd), EVL::of_size(size), XS0.num(), a.num(), b);
        self.ev(add.mask(k, false))
    }

    fn kiminmax(sh: Sh, signed: bool, min: bool) -> EOp {
        let t = match (min, signed) {
            (true, true) => KPMINS,
            (true, false) => KPMINU,
            (false, true) => KPMAXS,
            (false, false) => KPMAXU,
        };
        by_w(sh.w, t)
    }

    fn k_minmax(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let sz = sh.size;
        let min = matches!(op, VecOp::Min | VecOp::MinNum);
        let p = match sh.lane {
            LaneTy::Int(t) => {
                let s = self.vread(b, sz)?;
                let x = self.vget(a, sz)?;
                self.kb(Self::kiminmax(sh, t.signed, min), sz, x, x, s)?;
                return self.vput(rv, x, sz);
            }
            LaneTy::Float(_) => self.vprec(sh)?,
        };
        let (xa, xb) = (self.vread(a, sz)?, self.vread(b, sz)?);
        if matches!(op, VecOp::Min | VecOp::Max) {
            let k = self.alloc_k()?;
            self.kfminmax(sz, p, min, xa, xb, k)?;
            return self.vput(rv, XS0, sz);
        }
        // min_num / max_num: a NaN lane is missing, so each operand takes the other's lane
        // where it is NaN; two NaNs stay, and adding -0 (which changes no other value) quiets
        // them.
        let r = self.alloc_x()?;
        let (ka, kb) = (self.alloc_k()?, self.alloc_k()?);
        let m = if min { fp(p, EOp::Minps, EOp::Minpd) } else { fp(p, EOp::Maxps, EOp::Maxpd) };
        let cmp = fp(p, EOp::Cmpps, EOp::Cmppd);
        let mov = fp(p, EOp::Movaps, EOp::Movapd);
        let vl = EVL::of_size(sz);
        self.kcmp(cmp, sz, ka, xa, xa, Some(UNORD))?;
        self.kcmp(cmp, sz, kb, xb, xb, Some(UNORD))?;
        self.ku(mov, sz, XS0, xa)?;
        self.ev(Ev::new(mov, vl, XS0.num(), 0, xb).mask(ka, false))?;
        self.ku(mov, sz, XS1, xb)?;
        self.ev(Ev::new(mov, vl, XS1.num(), 0, xa).mask(kb, false))?;
        self.kb(m, sz, r, XS0, XS1)?;
        self.kb(m, sz, XS0, XS1, XS0)?;
        let join = match (min, p) {
            (true, Prec::F32) => EOp::Orps,
            (true, Prec::F64) => EOp::Orpd,
            (false, Prec::F32) => EOp::Andps,
            (false, Prec::F64) => EOp::Andpd,
        };
        self.kb(join, sz, r, r, XS0)?;
        self.ksplat(XS1, sz, sh.w, sh.sign())?;
        self.kb(fp(p, EOp::Addps, EOp::Addpd), sz, r, r, XS1)?;
        self.vput(rv, r, sz)
    }

    fn k_int_bin(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let sz = sh.size;
        if op == VecOp::Andnot {
            // a & !b, computed into `b`'s register
            let x = self.vread(a, sz)?;
            let s = self.vget(b, sz)?;
            self.kb(EOp::Pandnq, sz, s, s, x)?;
            return self.vput(rv, s, sz);
        }
        if op == VecOp::MulWrap {
            let s = self.vread(b, sz)?;
            let x = self.vget(a, sz)?;
            match sh.w {
                1 => {
                    // even bytes: the low bytes of the word products. Odd bytes: the product of
                    // the high bytes, which `b`'s high byte in place moves to the high byte.
                    let lo = self.alloc_x()?;
                    self.ksplat(XS0, sz, 2, 0xff00)?;
                    self.kb(EOp::Pmullw, sz, lo, x, s)?;
                    self.ksh(EShiftI::Psrlw, sz, x, 8)?;
                    self.kb(EOp::Pandq, sz, XS1, s, XS0)?;
                    self.kb(EOp::Pmullw, sz, x, x, XS1)?;
                    // per bit: x | (lo & !mask)
                    self.kbi(EOp::Pternlogq, sz, x, lo, XS0, 0xf4)?;
                }
                2 => self.kb(EOp::Pmullw, sz, x, x, s)?,
                4 => self.kb(EOp::Pmulld, sz, x, x, s)?,
                _ => self.kb(EOp::Pmullq, sz, x, x, s)?,
            }
            return self.vput(rv, x, sz);
        }
        let s = self.vsrc(b, sh)?;
        let x = self.vget(a, sz)?;
        let o = match op {
            VecOp::AddWrap => by_w(sh.w, KPADD),
            VecOp::SubWrap => by_w(sh.w, KPSUB),
            VecOp::And => EOp::Pandq,
            VecOp::Or => EOp::Porq,
            _ => EOp::Pxorq,
        };
        self.kb(o, sz, x, x, erm(s))?;
        self.vput(rv, x, sz)
    }

    fn k_int_un(&mut self, op: VecOp, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let x = self.vget(a, sh.size)?;
        match op {
            // per bit: !x
            VecOp::Not => self.kbi(EOp::Pternlogq, sh.size, x, x, x, 0x0f)?,
            _ if t.signed => {
                let o = by_w(sh.w, [EOp::Pabsb, EOp::Pabsw, EOp::Pabsd, EOp::Pabsq]);
                self.ku(o, sh.size, x, x)?;
            }
            _ => return Err(bad(format!("{} on these lanes", op.name()))),
        }
        self.vput(rv, x, sh.size)
    }

    /// The wrapped sum or difference of signed lanes in `r` and, in `XS0`, a vector whose lane
    /// sign bits are set where it overflowed (`tern`: the bitwise `vpternlog` form).
    fn ksigned_ovf(&mut self, sh: Sh, add: bool, tern: EOp, r: Xmm, a: Xmm, b: Xmm) -> Res<()> {
        let sz = sh.size;
        self.kb(by_w(sh.w, if add { KPADD } else { KPSUB }), sz, r, a, b)?;
        self.ku(EOp::Movdqa64, sz, XS0, a)?;
        // per bit, of (a, b, r): add: (a ^ r) & (b ^ r); sub: (a ^ b) & (a ^ r)
        self.kbi(tern, sz, XS0, b, r, if add { 0x42 } else { 0x18 })
    }

    fn k_sat(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let sz = sh.size;
        let add = op == VecOp::AddSat;
        if sh.w <= 2 {
            let k = (sh.w - 1) as usize;
            let o = match (add, t.signed) {
                (true, true) => [EOp::Paddsb, EOp::Paddsw][k],
                (true, false) => [EOp::Paddusb, EOp::Paddusw][k],
                (false, true) => [EOp::Psubsb, EOp::Psubsw][k],
                (false, false) => [EOp::Psubusb, EOp::Psubusw][k],
            };
            let s = self.vread(b, sz)?;
            let x = self.vget(a, sz)?;
            self.kb(o, sz, x, x, s)?;
            return self.vput(rv, x, sz);
        }
        let q = sh.w == 8;
        if !t.signed {
            let s = self.vread(b, sz)?;
            let x = self.vget(a, sz)?;
            if add {
                // min(a, !b) + b
                self.kbi(if q { EOp::Pternlogq } else { EOp::Pternlogd }, sz, XS0, s, s, 0x55)?;
                self.kb(by_w(sh.w, KPMINU), sz, x, x, XS0)?;
                self.kb(by_w(sh.w, KPADD), sz, x, x, s)?;
            } else {
                // max(a, b) - b
                self.kb(by_w(sh.w, KPMAXU), sz, x, x, s)?;
                self.kb(by_w(sh.w, KPSUB), sz, x, x, s)?;
            }
            return self.vput(rv, x, sz);
        }
        let (xa, xb) = (self.vread(a, sz)?, self.vread(b, sz)?);
        let r = self.alloc_x()?;
        let k = self.alloc_k()?;
        self.ksigned_ovf(sh, add, if q { EOp::Pternlogq } else { EOp::Pternlogd }, r, xa, xb)?;
        self.kmask_from_vec(k, XS0, sh.n, sh.w)?;
        // the bound on `a`'s side: MAX, or MIN = !MAX for a negative `a`
        self.ku(EOp::Movdqa64, sz, XS0, xa)?;
        self.ksh(if q { EShiftI::Psraq } else { EShiftI::Psrad }, sz, XS0, (sh.bits() - 1) as u8)?;
        self.ksplat(XS1, sz, sh.w, sh.sign() - 1)?;
        self.kb(if q { EOp::Pxorq } else { EOp::Pxord }, sz, XS0, XS0, XS1)?;
        let mov = if q { EOp::Movdqa64 } else { EOp::Movdqa32 };
        self.ev(Ev::new(mov, EVL::of_size(sz), r.num(), 0, XS0).mask(k, false))?;
        self.vput(rv, r, sz)
    }

    /// `Some(a op b)` iff no lane overflows: the tag from the flags without a branch.
    fn k_checked(&mut self, op: VecOp, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let (none, some, pty) = self
            .cx
            .m
            .types
            .option_shape(rt)
            .ok_or_else(|| bad("checked vector op result must be an Option"))?;
        let sh = self.vshape(pty)?;
        let t = self.vint(sh)?;
        let sz = sh.size;
        let add = op == VecOp::CheckedAdd;
        let (xa, xb) = (self.vread(a, sz)?, self.vread(b, sz)?);
        let r = self.alloc_x()?;
        let k = self.alloc_k()?;
        let (d, off, _) = self.option_home(rt, some, rv)?;
        // ZF = no lane overflowed
        if t.signed {
            self.ksigned_ovf(sh, add, EOp::Pternlogd, r, xa, xb)?;
            self.ksplat(XS1, sz, sh.w, sh.sign())?;
            self.kcmp(EOp::Ptestmq, sz, k, XS0, XS1, None)?;
            self.a.kortest(KSize::B, k, k);
        } else {
            self.kb(by_w(sh.w, if add { KPADD } else { KPSUB }), sz, r, xa, xb)?;
            // a wrapped sum is below `a`; a difference wraps when `a` is below `b`
            let (l, h) = if add { (r, xa) } else { (xa, xb) };
            // the zero lanes above a value of fewer than 16 bytes compare false
            self.kcmp(by_w(sh.w, KPCMPU), sz, k, l, h, Some(LT))?;
            self.a.kortest(KSize::Q, k, k);
        }
        if (none, some) == (0, 1) {
            self.a.setcc(Cond::E, S0);
            self.a.movzx(Size::D, S0, Size::B, S0)?;
        } else {
            self.a.mov_ri(Size::D, S0, i64::from(none))?;
            self.a.mov_ri(Size::D, S1, i64::from(some))?;
            self.a.cmov(Cond::E, Size::D, S0, S1)?;
        }
        self.a.mov_mr(Size::D, d, S0)?;
        self.vstore(disp(d, off)?, r, sz)
    }

    /// The count of a per-vector shift: `Some(k)` for a constant (masked by the lane width),
    /// else in `XS1` after the range check (`trap`) or masked.
    fn kshift_count(&mut self, amt: Operand, bits: i64, trap: bool) -> Res<Option<u8>> {
        if let Some(k) = self.int_const(amt) {
            return Ok(Some((k & i128::from(bits - 1)) as u8));
        }
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
        Ok(None)
    }

    /// `d = value shifted right by the count` in every byte, for the masks of the byte shifts:
    /// the count is `k`, or in `XS1`. `left`: shifted left, within the byte.
    fn kbyte_mask(&mut self, d: Xmm, size: u32, value: u8, left: bool, k: Option<u8>) -> Res<()> {
        if let Some(k) = k {
            let v = if left { value << k } else { value >> k };
            return self.ksplat(d, size, 1, u64::from(v));
        }
        // the low byte of the shifted word is the shifted byte
        self.ksplat(d, 16, 2, u64::from(value))?;
        self.kb(if left { EOp::PsllwX } else { EOp::PsrlwX }, 16, d, d, XS1)?;
        self.ku(EOp::Pbroadcastb, size, d, d)
    }

    fn k_shift(&mut self, op: VecOp, a: Operand, amt: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let sz = sh.size;
        let bits = i64::from(sh.bits());
        let trap = op != VecOp::ShlWrap;
        let right = op == VecOp::Shr;
        let arith = right && t.signed;
        if self.oty(amt) != Some(rt) {
            // one amount for every lane
            let x = self.vget(a, sz)?;
            if trap && self.int_const(amt).is_some_and(|k| !(0..i128::from(bits)).contains(&k)) {
                self.vput(rv, x, sz)?;
                return self.trap_now(PanicKind::Overflow);
            }
            let k = self.kshift_count(amt, bits, trap)?;
            // bytes shift as words
            let i = (sh.w.trailing_zeros() as usize).saturating_sub(1);
            let (oi, ox) = match (right, arith && sh.w > 1) {
                (false, _) => (
                    [EShiftI::Psllw, EShiftI::Pslld, EShiftI::Psllq][i],
                    [EOp::PsllwX, EOp::PslldX, EOp::PsllqX][i],
                ),
                (true, false) => (
                    [EShiftI::Psrlw, EShiftI::Psrld, EShiftI::Psrlq][i],
                    [EOp::PsrlwX, EOp::PsrldX, EOp::PsrlqX][i],
                ),
                (true, true) => (
                    [EShiftI::Psraw, EShiftI::Psrad, EShiftI::Psraq][i],
                    [EOp::PsrawX, EOp::PsradX, EOp::PsraqX][i],
                ),
            };
            if sh.w == 1 {
                // the bits that stay in their byte
                self.kbyte_mask(XS0, sz, 0xff, !right, k)?;
            }
            match k {
                Some(k) => self.ksh(oi, sz, x, k)?,
                None => self.kb(ox, sz, x, x, XS1)?,
            }
            if sh.w == 1 && arith {
                // (x & mask) ^ s - s extends the shifted sign bit `s`
                let s = self.alloc_x()?;
                self.kbyte_mask(s, sz, 0x80, false, k)?;
                self.kbi(EOp::Pternlogq, sz, x, XS0, s, 0x6a)?;
                self.kb(EOp::Psubb, sz, x, x, s)?;
            } else if sh.w == 1 {
                self.kb(EOp::Pandq, sz, x, x, XS0)?;
            }
            return self.vput(rv, x, sz);
        }
        if sh.w == 1 {
            return Err(unsup(format!(
                "{} of 8-bit lanes by a vector in a fn compiled for the AVX-512 set",
                op.name()
            )));
        }
        let s = self.vread(amt, sz)?;
        let x = self.vget(a, sz)?;
        self.ksplat(XS0, sz, sh.w, (bits - 1) as u64)?;
        let s = if trap {
            // the zero lanes above a value of fewer than 16 bytes are in range
            let k = self.alloc_k()?;
            self.kcmp(by_w(sh.w, KPCMPU), sz, k, s, XS0, Some(NLE))?;
            self.a.kortest(KSize::Q, k, k);
            self.jtrap(Cond::Ne, PanicKind::Overflow)?;
            s
        } else {
            self.kb(if sh.w == 4 { EOp::Pandd } else { EOp::Pandq }, sz, XS0, s, XS0)?;
            XS0
        };
        let i = (sh.w.trailing_zeros() as usize).saturating_sub(1);
        let o = match (right, arith) {
            (false, _) => [EOp::Psllvw, EOp::Psllvd, EOp::Psllvq][i],
            (true, false) => [EOp::Psrlvw, EOp::Psrlvd, EOp::Psrlvq][i],
            (true, true) => [EOp::Psravw, EOp::Psravd, EOp::Psravq][i],
        };
        self.kb(o, sz, x, x, s)?;
        self.vput(rv, x, sz)
    }

    fn k_compare(&mut self, op: VecOp, a: Operand, b: Operand, rv: Value) -> Res<()> {
        let sh = self.voshape(a)?;
        let sz = sh.size;
        let (xa, xb) = (self.vread(a, sz)?, self.vread(b, sz)?);
        let k = self.alloc_k()?;
        match sh.lane {
            LaneTy::Float(_) => {
                let p = self.vprec(sh)?;
                // an unordered pair is only `ne`
                let pred = match op {
                    VecOp::Eq => 0x00,
                    VecOp::Ne => 0x04,
                    VecOp::Lt => 0x01,
                    VecOp::Le => 0x02,
                    VecOp::Gt => 0x0e,
                    _ => 0x0d,
                };
                self.kcmp(fp(p, EOp::Cmpps, EOp::Cmppd), sz, k, xa, xb, Some(pred))?;
            }
            LaneTy::Int(t) => match (op, t.signed) {
                (VecOp::Eq, _) => self.kcmp(by_w(sh.w, KPCMPEQ), sz, k, xa, xb, None)?,
                (VecOp::Ne, _) => self.kcmp(by_w(sh.w, KPCMP), sz, k, xa, xb, Some(NE))?,
                (VecOp::Gt, true) => self.kcmp(by_w(sh.w, KPCMPGT), sz, k, xa, xb, None)?,
                (VecOp::Lt, true) => self.kcmp(by_w(sh.w, KPCMPGT), sz, k, xb, xa, None)?,
                (VecOp::Le, true) => self.kcmp(by_w(sh.w, KPCMP), sz, k, xa, xb, Some(LE))?,
                (_, true) => self.kcmp(by_w(sh.w, KPCMP), sz, k, xa, xb, Some(NLT))?,
                (VecOp::Lt, false) => self.kcmp(by_w(sh.w, KPCMPU), sz, k, xa, xb, Some(LT))?,
                (VecOp::Gt, false) => self.kcmp(by_w(sh.w, KPCMPU), sz, k, xa, xb, Some(NLE))?,
                (VecOp::Le, false) => self.kcmp(by_w(sh.w, KPCMPU), sz, k, xa, xb, Some(LE))?,
                (_, false) => self.kcmp(by_w(sh.w, KPCMPU), sz, k, xa, xb, Some(NLT))?,
            },
        }
        self.bind(rv, R::K(k))
    }

    // ---------------------------------------------------------------- reductions

    /// One step of the tree at `size` bytes: `x = x op t`. Clobbers `XS0`, `XS1`, `k`.
    fn kfold(&mut self, op: VecOp, sh: Sh, size: u32, x: Xmm, t: Xmm, k: KReg) -> Res<()> {
        let min = op == VecOp::ReduceMin;
        match (op, sh.lane) {
            (VecOp::ReduceAdd, LaneTy::Float(_)) => {
                let p = self.vprec(sh)?;
                self.kb(fp(p, EOp::Addps, EOp::Addpd), size, x, x, t)
            }
            (VecOp::ReduceMin | VecOp::ReduceMax, LaneTy::Float(_)) => {
                let p = self.vprec(sh)?;
                self.kfminmax(size, p, min, x, t, k)?;
                self.ku(EOp::Movdqa64, size, x, XS0)
            }
            (VecOp::ReduceAddWrap, LaneTy::Int(_)) => self.kb(by_w(sh.w, KPADD), size, x, x, t),
            // on an `xmm` the recipes name the VEX forms
            (VecOp::ReduceAnd, LaneTy::Int(_)) if size <= 16 => self.vb(VBin::Pand, false, x, t),
            (VecOp::ReduceOr, LaneTy::Int(_)) if size <= 16 => self.vb(VBin::Por, false, x, t),
            (VecOp::ReduceAnd, LaneTy::Int(_)) => self.kb(EOp::Pandq, size, x, x, t),
            (VecOp::ReduceOr, LaneTy::Int(_)) => self.kb(EOp::Porq, size, x, x, t),
            (VecOp::ReduceMin | VecOp::ReduceMax, LaneTy::Int(it)) => {
                self.kb(Self::kiminmax(sh, it.signed, min), size, x, x, t)
            }
            _ => Err(bad(format!("{} on these lanes", op.name()))),
        }
    }

    /// The fixed tree: lane i with lane i + n/2 until one lane is left.
    fn k_reduce(&mut self, op: VecOp, a: Operand, rv: Value) -> Res<()> {
        let sh = self.voshape(a)?;
        let float = sh.float();
        let x = self.vget(a, sh.size)?;
        let t = self.alloc_x()?;
        let k = if float.is_some() && op != VecOp::ReduceAdd { self.alloc_k()? } else { KS0 };
        let mut size = sh.size;
        if size == 64 {
            let ext = if float.is_some() { EOp::Extractf64x4 } else { EOp::Extracti64x4 };
            self.ev(Ev::new(ext, EVL::L512, x.num(), 0, t).imm(1))?;
            size = 32;
            self.kfold(op, sh, size, x, t, k)?;
        }
        if size == 32 {
            let ext = if float.is_some() { VExtract::Extractf128 } else { VExtract::Extracti128 };
            self.ydirty = true;
            self.yused = true;
            self.a.vex_extract(ext, t, y(x), 1)?;
            size = 16;
            self.kfold(op, sh, size, x, t, k)?;
        }
        let mut n = size / sh.w;
        match (float, sh.w) {
            (None, 1) if op == VecOp::ReduceAddWrap => {
                // the sums of the two halves' bytes as quadwords
                self.vzero(XS0)?;
                self.kb(EOp::Psadbw, 16, x, x, XS0)?;
                self.vui(VUnI::Pshufd, false, t, x, 0xee)?;
                self.vb(VBin::Paddq, false, x, t)?;
                n = 1;
            }
            (None, 2) if op == VecOp::ReduceAddWrap => {
                // the sums of adjacent words as doublewords (their low words are the sums)
                self.ksplat(XS0, 16, 2, 1)?;
                self.kb(EOp::Pmaddwd, 16, x, x, XS0)?;
                for imm in [0xee, 0x55] {
                    self.vui(VUnI::Pshufd, false, t, x, imm)?;
                    self.vb(VBin::Paddd, false, x, t)?;
                }
                n = 1;
            }
            _ => {}
        }
        while n > 1 {
            let h = n / 2;
            // t = the upper half of the remaining lanes
            match (float, h * sh.w) {
                (Some(Prec::F32), 8) => self.a.vex_bin(VBin::Movhlps, t, t, x)?,
                (Some(Prec::F32), _) => self.ku(EOp::Movshdup, 16, t, x)?,
                (Some(Prec::F64), _) => self.a.vex_bin(VBin::Unpckhpd, t, x, x)?,
                (None, 8) => self.vui(VUnI::Pshufd, false, t, x, 0xee)?,
                (None, 4) => self.vui(VUnI::Pshufd, false, t, x, 0x55)?,
                (None, 2) => self.vui(VUnI::Pshuflw, false, t, x, 0x55)?,
                (None, _) => {
                    self.vmov(false, t, x)?;
                    self.vsh(VShiftI::Psrlw, false, t, 8)?;
                }
            }
            match (op, float) {
                // the last sum is a scalar one
                (VecOp::ReduceAdd, Some(p)) if h == 1 => self.a.sse(SseOp::Add, p, x, t)?,
                _ => self.kfold(op, sh, 16, x, t, k)?,
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

    /// `((init + v[0]) + v[1]) + …`: lane by lane out of each 128-bit quarter.
    fn k_reduce_ordered(&mut self, v: Operand, init: Operand, rv: Value) -> Res<()> {
        let sh = self.voshape(v)?;
        let p = self.vprec(sh)?;
        let x = self.vread(v, sh.size)?;
        let acc = self.new_x(init, p)?;
        let per = 16 / sh.w;
        for q in 0..sh.size.div_ceil(16) {
            let src = if q == 0 {
                x
            } else {
                let vl = EVL::of_size(sh.size);
                self.ev(Ev::new(EOp::Extractf32x4, vl, x.num(), 0, XS0).imm(q as u8))?;
                XS0
            };
            for j in 0..per.min(sh.n - q * per) {
                let lane = match (p, j) {
                    (_, 0) => src,
                    (Prec::F32, 1) => {
                        self.ku(EOp::Movshdup, 16, XS1, src)?;
                        XS1
                    }
                    (Prec::F32, 2) => {
                        self.a.vex_bin(VBin::Movhlps, XS1, XS1, src)?;
                        XS1
                    }
                    (Prec::F32, _) => {
                        self.a.vex_bin_i(VBinI::Shufps, XS1, src, src, 0xff)?;
                        XS1
                    }
                    (Prec::F64, _) => {
                        self.a.vex_bin(VBin::Unpckhpd, XS1, src, src)?;
                        XS1
                    }
                };
                self.a.sse(SseOp::Add, p, acc, lane)?;
            }
        }
        self.bind(rv, R::X(acc))
    }

    // ---------------------------------------------------------------- halves

    fn k_split(&mut self, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.voshape(a)?;
        if sh.size < 32 {
            return self.v_split(a, rt, rv);
        }
        let offs: Vec<i64> = match &self.layout(rt)?.shape {
            Shape::Fields(fs) if fs.len() == 2 => fs.iter().map(|f| f.0 as i64).collect(),
            _ => return Err(bad("split result must be a pair")),
        };
        let h = sh.size / 2;
        let x = self.vread(a, sh.size)?;
        let d = self.def_mem(rv)?;
        self.vstore(disp(d, offs[0])?, x, h)?;
        let ext = if sh.size == 64 { EOp::Extracti64x4 } else { EOp::Extracti32x4 };
        self.ev(Ev::new(ext, EVL::of_size(sh.size), x.num(), 0, disp(d, offs[1])?).imm(1))
    }

    fn k_join(&mut self, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        if sh.size < 32 {
            return self.v_join(a, b, rt, rv);
        }
        let xb = self.vread(b, sh.size / 2)?;
        let x = self.vget(a, sh.size / 2)?;
        let ins = if sh.size == 64 { EOp::Inserti64x4 } else { EOp::Inserti32x4 };
        self.ev(Ev::new(ins, EVL::of_size(sh.size), x.num(), x.num(), xb).imm(1))?;
        self.vput(rv, x, sh.size)
    }

    // ---------------------------------------------------------------- memory

    fn k_load(&mut self, aligned: bool, s: Operand, i: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        if aligned {
            self.valigned(i, sh.n)?;
        }
        let m = self.vslice(s, i, sh.w)?;
        let x = self.alloc_x()?;
        if sh.size < 16 {
            self.vload(x, m, sh.size)?;
        } else {
            // an aligned access: the slice is aligned to the vector and the index to its lanes
            self.ku(if aligned { EOp::Movdqa64 } else { EOp::Movdqu64 }, sh.size, x, m)?;
        }
        self.vput(rv, x, sh.size)
    }

    fn k_store(&mut self, aligned: bool, s: Operand, i: Operand, v: Operand) -> Res<()> {
        let sh = self.voshape(v)?;
        if aligned {
            self.valigned(i, sh.n)?;
        }
        let m = self.vslice(s, i, sh.w)?;
        let x = self.vread(v, sh.size)?;
        if sh.size < 16 {
            return self.vstore(m, x, sh.size);
        }
        self.ku(if aligned { EOp::Movdqa64St } else { EOp::Movdqu64St }, sh.size, x, m)
    }

    /// Active lanes only: inactive lanes read as zero and their memory is not touched.
    fn k_load_masked(&mut self, s: Operand, i: Operand, m: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let km = self.read_k(m)?;
        let (base, len) = self.vbase(s)?;
        let ig = self.read_g(i)?;
        let bits = self.alloc_g()?;
        let x = self.alloc_x()?;
        self.kbits(bits, km, sh.n)?;
        self.kmasked_check(bits, ig, len)?;
        let k = self.kwrite_mask(km, sh, bits)?;
        let mem = Mem::index(base, ig, super::scale_of(sh.w), 0);
        let e = Ev::new(by_w(sh.w, KMOVDQU), EVL::of_size(sh.size), x.num(), 0, mem);
        self.ev(e.mask(k, true))?;
        self.vput(rv, x, sh.size)
    }

    /// Active lanes only: the memory of inactive lanes is not touched.
    fn k_store_masked(&mut self, s: Operand, i: Operand, v: Operand, m: Operand) -> Res<()> {
        let sh = self.voshape(v)?;
        let km = self.read_k(m)?;
        let x = self.vread(v, sh.size)?;
        let (base, len) = self.vbase(s)?;
        let ig = self.read_g(i)?;
        let bits = self.alloc_g()?;
        self.kbits(bits, km, sh.n)?;
        self.kmasked_check(bits, ig, len)?;
        let k = self.kwrite_mask(km, sh, bits)?;
        let mem = Mem::index(base, ig, super::scale_of(sh.w), 0);
        let e = Ev::new(by_w(sh.w, KMOVDQU_ST), EVL::of_size(sh.size), x.num(), 0, mem);
        self.ev(e.mask(k, false))
    }

    // ---------------------------------------------------------------- dispatch

    pub(super) fn kvec_emit(
        &mut self,
        op: VecOp,
        args: &[Operand],
        ty: Option<TypeId>,
        res: Option<Value>,
    ) -> Res<()> {
        use VecOp::*;
        let arg =
            |k: usize| args.get(k).copied().ok_or_else(|| bad(format!("{}: missing operand {k}", op.name())));
        let rv = res.ok_or_else(|| bad(format!("{}: missing result", op.name())));
        let rt = ty.ok_or_else(|| bad(format!("{}: missing result type", op.name())));
        match op {
            Splat => self.k_splat(arg(0)?, rt?, rv?),
            Add | Sub | Mul | Div | MinX86 | MaxX86 => self.k_float_bin(op, arg(0)?, arg(1)?, rt?, rv?),
            Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc => self.k_float_un(op, arg(0)?, rt?, rv?),
            Copysign => self.k_copysign(arg(0)?, arg(1)?, rt?, rv?),
            Fma => self.k_fma(arg(0)?, arg(1)?, arg(2)?, rt?, rv?),
            Min | Max | MinNum | MaxNum => self.k_minmax(op, arg(0)?, arg(1)?, rt?, rv?),
            AddWrap | SubWrap | MulWrap | And | Or | Xor | Andnot => {
                self.k_int_bin(op, arg(0)?, arg(1)?, rt?, rv?)
            }
            MaskAnd | MaskOr | MaskXor => self.k_mask_bin(op, arg(0)?, arg(1)?, rv?),
            MaskNot => self.k_mask_not(arg(0)?, rv?),
            AddSat | SubSat => self.k_sat(op, arg(0)?, arg(1)?, rt?, rv?),
            CheckedAdd | CheckedSub => self.k_checked(op, arg(0)?, arg(1)?, rt?, rv?),
            AbsWrap | Not => self.k_int_un(op, arg(0)?, rt?, rv?),
            Shl | Shr | ShlWrap => self.k_shift(op, arg(0)?, arg(1)?, rt?, rv?),
            Eq | Ne | Lt | Le | Gt | Ge => self.k_compare(op, arg(0)?, arg(1)?, rv?),
            ReduceAdd | ReduceAddWrap | ReduceMin | ReduceMax | ReduceAnd | ReduceOr => {
                self.k_reduce(op, arg(0)?, rv?)
            }
            ReduceAddOrdered => self.k_reduce_ordered(arg(0)?, arg(1)?, rv?),
            Bitcast => self.v_copy(arg(0)?, rt?, rv?),
            Split => self.k_split(arg(0)?, rt?, rv?),
            Join => self.k_join(arg(0)?, arg(1)?, rt?, rv?),
            Load | LoadAligned => self.k_load(op == LoadAligned, arg(0)?, arg(1)?, rt?, rv?),
            Store | StoreAligned => self.k_store(op == StoreAligned, arg(0)?, arg(1)?, arg(2)?),
            LoadMasked => self.k_load_masked(arg(0)?, arg(1)?, arg(2)?, rt?, rv?),
            StoreMasked => self.k_store_masked(arg(0)?, arg(1)?, arg(2)?, arg(3)?),
            Select => self.k_select(arg(0)?, arg(1)?, arg(2)?, rt?, rv?),
            ToBits => self.k_to_bits(arg(0)?, rv?),
            ToVec => self.k_to_vec(arg(0)?, rt?, rv?),
            FirstN => self.k_first_n(arg(0)?, rt?, rv?),
            Rewiden => self.k_rewiden(arg(0)?, rv?),
            MaskAll | MaskNone => self.k_mask_const(op == MaskAll, rv?),
            // no vector form at any tier: lane by lane on the home bytes
            DivLanes => self.v_div(op, arg(0)?, arg(1)?, rt?, rv?),
            MulSat | CheckedMul | DivConst | Popcount | Clz | ScanAdd | ScanAddWrap | Cast(_) | WidenLo
            | WidenHi | NarrowWrap | NarrowSat | Permute | Lookup | Compress | Expand | StoreStream
            | Gather | Scatter => Err(unsup(format!(
                "{} in a fn compiled for the AVX-512 set (`avx512f`, `avx512bw`, `avx512vl`, `avx512dq`, `avx512cd`)",
                op.name()
            ))),
        }
    }
}
