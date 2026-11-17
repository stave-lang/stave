//! K mode, continued: integer scans, saturating and checked multiplies, division by a constant,
//! bit counts, permutes, table lookups, compress / expand, gathers / scatters, streaming stores,
//! and 8-bit lanes shifted by a vector.
//!
//! Only the five features of the v4 set: no VBMI, VBMI2, BITALG, VPOPCNTDQ, IFMA or GFNI. Byte
//! and word lanes take sequences of word, doubleword and byte-shuffle instructions; what no
//! sequence does well runs lane by lane on the home bytes as on v3: float scans (their sums
//! are sequential), 64-bit `mul.sat`, `checked_mul` and `div_const` (no 64-bit high product),
//! gathers and scatters of 8- and 16-bit lanes, and lookups whose index lanes are not as wide
//! as the table's.

use stave_ir::simd::VecOp;
use stave_ir::{Operand, PanicKind, TypeId, Value};

use super::super::memory::VLen;
use super::super::{lane, scale_of};
use super::*;

/// Per nibble value: its set bits, and its leading zeros within four bits.
const POPCOUNT_LUT: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
const CLZ_LUT: [u8; 16] = [4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

/// How lanes divide by a constant: Granlund and Montgomery, "Division by invariant integers
/// using multiplication", with the high half of a product per lane. `t` below is that
/// high half of `x * m`, signed or unsigned as the lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DivPlan {
    /// `x`, or `-x` (a signed divisor of -1).
    Copy { neg: bool },
    /// Unsigned, a power of two: `x >> s`.
    Shr(u32),
    /// Unsigned: `t >> s`.
    Mul { m: u64, s: u32 },
    /// Unsigned, a multiplier one bit too wide (`m` its low bits): `(t + ((x - t) >> 1)) >> s`.
    MulAdd { m: u64, s: u32 },
    /// Signed, `|d| = 2^l`: `(x + ((x >> (l - 1)) >>> (N - l))) >> l`, negated for `d < 0`.
    Pow { l: u32, neg: bool },
    /// Signed: `((t [+ x]) >> s) - (x >> (N - 1))`, negated for `d < 0`; with `add`, `m` is
    /// the multiplier less 2^N, negative.
    SMul { m: i64, s: u32, add: bool, neg: bool },
}

/// `(m, sh_post)` of the multiplier for a divisor `d` (1 <= d < 2^n) of dividends of `prec`
/// bits, computed with `n`-bit high products (CHOOSE_MULTIPLIER): `m` may have n + 1 bits.
fn choose_multiplier(d: u64, n: u32, prec: u32) -> (u128, u32) {
    let l = if d <= 1 { 0 } else { 64 - (d - 1).leading_zeros() };
    let mut s = l;
    let mut lo = (1u128 << (n + l)) / u128::from(d);
    let mut hi = ((1u128 << (n + l)) + (1u128 << (n + l - prec))) / u128::from(d);
    while lo / 2 < hi / 2 && s > 0 {
        lo /= 2;
        hi /= 2;
        s -= 1;
    }
    (hi, s)
}

/// The plan for the divisor `d` (not 0; lane value, sign-extended when `signed`) of dividends
/// of `prec` bits (the sign not counted), on lanes of `n` bits.
fn div_plan(d: i64, n: u32, prec: u32, signed: bool) -> DivPlan {
    if !signed {
        let d = d as u64;
        if d == 1 {
            return DivPlan::Copy { neg: false };
        }
        if d.is_power_of_two() {
            return DivPlan::Shr(d.trailing_zeros());
        }
        let (m, s) = choose_multiplier(d, n, prec);
        return if m < 1 << n {
            DivPlan::Mul { m: m as u64, s }
        } else {
            DivPlan::MulAdd { m: (m - (1 << n)) as u64, s: s - 1 }
        };
    }
    let (a, neg) = (d.unsigned_abs(), d < 0);
    if a == 1 {
        return DivPlan::Copy { neg };
    }
    if a.is_power_of_two() {
        return DivPlan::Pow { l: a.trailing_zeros(), neg };
    }
    let (m, s) = choose_multiplier(a, n, prec);
    if m < 1 << (n - 1) {
        DivPlan::SMul { m: m as i64, s, add: false, neg }
    } else {
        DivPlan::SMul { m: (m as i128 - (1i128 << n)) as i64, s, add: true, neg }
    }
}

impl Cg<'_, '_> {
    // ---------------------------------------------------------------- helpers

    /// `d = s shift imm` per lane (or per 128-bit lane for the byte shifts).
    fn kshv(&mut self, op: EShiftI, size: u32, d: Xmm, s: Xmm, imm: u8) -> Res<()> {
        let vl = EVL::of_size(size);
        if vl != EVL::L128 {
            self.ydirty = true;
            self.yused = true;
        }
        Ok(self.a.evex_shift_i(op, vl, d.num(), s, imm, KS0, false)?)
    }

    /// A constant as long as the register of a `size`-byte value: `pattern` repeated.
    fn kpool(&mut self, size: u32, pattern: &[u8]) -> Mem {
        let n = EVL::of_size(size).bytes() as usize;
        self.vpool_mem(pattern.iter().copied().cycle().take(n).collect())
    }

    /// [`Self::kpool`] of `value` in every lane of `w` bytes.
    fn ksplat_mem(&mut self, size: u32, w: u32, value: u64) -> Mem {
        self.kpool(size, &value.to_le_bytes()[..w as usize])
    }

    /// `d` = [`Self::kpool`].
    fn kload(&mut self, d: Xmm, size: u32, pattern: &[u8]) -> Res<()> {
        let m = self.kpool(size, pattern);
        self.ku(EOp::Movdqu64, size, d, m)
    }

    /// `k` = the low `n` bits set, for an instruction on `lanes` lanes.
    fn kfirst(&mut self, k: KReg, n: u32, lanes: u32) -> Res<()> {
        if n == lanes {
            self.a.kbin(KBin::Xnor, KSize::of_lanes(n), k, k, k);
        } else {
            self.a.mov_ri(Size::D, S0, (1i64 << n) - 1)?;
            self.a.kmov_kg(KSize::W, k, S0);
        }
        Ok(())
    }

    /// `d = 0 - s`.
    fn kneg(&mut self, size: u32, w: u32, d: Xmm, s: Xmm) -> Res<()> {
        self.vzero(XS0)?;
        self.kb(by_w(w, KPSUB), size, d, XS0, s)
    }

    /// Doubleword indices of `vpermt2d` that interleave lanes `j` of two tables: `odd` picks
    /// the odd doublewords (the high halves of the quadword products), else the even ones.
    fn kinterleave_idx(&mut self, d: Xmm, size: u32, odd: bool) -> Res<()> {
        let lanes = EVL::of_size(size).bytes() / 4;
        let mut idx = Vec::with_capacity(lanes as usize * 4);
        for j in 0..lanes {
            let v = (j & !1) + u32::from(odd) + if j & 1 == 1 { lanes } else { 0 };
            idx.extend_from_slice(&v.to_le_bytes());
        }
        self.kload(d, size, &idx)
    }

    /// `d` = the high half of `x * m` per lane of `w` (2 or 4) bytes. Clobbers `XS0`, `XS1`.
    fn kmulhi(&mut self, size: u32, w: u32, signed: bool, d: Xmm, x: Xmm, m: u64) -> Res<()> {
        let mm = self.ksplat_mem(size, w, m);
        if w == 2 {
            return self.kb(if signed { EOp::Pmulhw } else { EOp::Pmulhuw }, size, d, x, mm);
        }
        // the even lanes' and the odd lanes' 64-bit products, then their high doublewords
        let mul = if signed { EOp::Pmuldq } else { EOp::Pmuludq };
        self.kshv(EShiftI::Psrlq, size, XS1, x, 32)?;
        self.kb(mul, size, d, x, mm)?;
        self.kb(mul, size, XS1, XS1, mm)?;
        self.kinterleave_idx(XS0, size, true)?;
        self.ev(Ev::new(EOp::Permt2d, EVL::of_size(size), d.num(), XS0.num(), XS1))
    }

    // ---------------------------------------------------------------- dispatch

    pub(super) fn kops_emit(
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
            ScanAdd | ScanAddWrap => self.k_scan(arg(0)?, rt?, rv?),
            StoreStream => self.k_store_stream(arg(0)?, arg(1)?, arg(2)?),
            Gather => self.k_gather(arg(0)?, arg(1)?, rt?, rv?),
            Scatter => self.k_scatter(arg(0)?, arg(1)?, arg(2)?),
            DivConst => self.k_div_const(arg(0)?, arg(1)?, rt?, rv?),
            MulSat => self.k_mul_sat(arg(0)?, arg(1)?, rt?, rv?),
            CheckedMul => self.k_checked_mul(arg(0)?, arg(1)?, rt?, rv?),
            Popcount | Clz => self.k_count(op, arg(0)?, rt?, rv?),
            Permute => self.k_permute(args, rt?, rv?),
            Lookup => self.k_lookup(arg(0)?, arg(1)?, rt?, rv?),
            Compress | Expand => self.k_compress(op == Expand, arg(0)?, arg(1)?, rt?, rv?),
            _ => Err(bad(format!("internal: {} is not a K-mode op of this module", op.name()))),
        }
    }

    // ---------------------------------------------------------------- scans

    /// Inclusive prefix sums. Int lanes wrap, so any order gives the same sums: every lane gains
    /// the lane 1, 2, 4, … below it within its 128-bit block, then every block the totals of
    /// the blocks below it.
    fn k_scan(&mut self, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        if sh.float().is_some() {
            return self.v_scan(a, rt, rv);
        }
        let sz = sh.size;
        let add = by_w(sh.w, KPADD);
        let x = self.vget(a, sz)?;
        let t = self.alloc_x()?;
        let mut step = sh.w;
        while step < sz.min(16) {
            self.kshv(EShiftI::Pslldq, sz, t, x, step as u8)?;
            self.kb(add, sz, x, x, t)?;
            step *= 2;
        }
        if sz > 16 {
            // t = the last lane of each block in every lane of the block
            match sh.w {
                8 => self.kui(EOp::Pshufd, sz, t, x, 0xee)?,
                4 => self.kui(EOp::Pshufd, sz, t, x, 0xff)?,
                w => {
                    let ctl: Vec<u8> = (0..16).map(|i| (16 - w + i % w) as u8).collect();
                    let m = self.kpool(sz, &ctl);
                    self.kb(EOp::Pshufb, sz, t, x, m)?;
                }
            }
            // t moved up by `b` blocks, zeros below
            self.vzero(XS0)?;
            for b in 1..sz / 16 {
                self.kbi(EOp::Alignq, sz, XS1, t, XS0, (sz / 8 - 2 * b) as u8)?;
                self.kb(add, sz, x, x, XS1)?;
            }
        }
        self.vput(rv, x, sz)
    }

    // ---------------------------------------------------------------- memory

    /// The alignment trap, then a non-temporal store; a vector under 16 bytes has none and is
    /// stored plainly.
    fn k_store_stream(&mut self, s: Operand, i: Operand, v: Operand) -> Res<()> {
        let sh = self.voshape(v)?;
        if sh.size < 16 {
            return self.v_store_stream(s, i, v);
        }
        self.valigned(i, sh.n)?;
        let m = self.vslice(s, i, sh.w)?;
        let x = self.vread(v, sh.size)?;
        let op = match sh.float() {
            Some(Prec::F32) => EOp::MovntpsSt,
            Some(Prec::F64) => EOp::MovntpdSt,
            None => EOp::MovntdqSt,
        };
        self.ku(op, sh.size, x, m)
    }

    fn kvsib(&mut self, op: EVsib, size: u32, data: Xmm, base: Gpr, index: Xmm, w: u32, k: KReg) -> Res<()> {
        let vl = EVL::of_size(size);
        if vl != EVL::L128 {
            self.ydirty = true;
            self.yused = true;
        }
        Ok(self.a.evex_vsib(op, vl, data, base, index, scale_of(w), 0, k)?)
    }

    /// The index lanes (`ish`) of a gather or scatter of `sh` in `xi`, zero-extended to the
    /// width the instruction takes, then the trap `Bounds` unless every one is below `len`.
    /// Returns that width and, for sixteen 32-bit lanes through 64-bit indices, the register of
    /// the upper eight indices (`xi` keeps the lower eight).
    fn kindex(&mut self, sh: Sh, ish: Sh, xi: Xmm, len: VLen, k: KReg) -> Res<(u32, Option<Xmm>)> {
        // `vp*d` sign-extend 32-bit indices: they serve only where no index can reach 2^31
        let short = sh.w == 4 && (ish.w < 4 || (ish.w == 4 && matches!(len, VLen::K(n) if n <= 1 << 31)));
        let iw = if short { 4 } else { 8 };
        let n = sh.n;
        let hi = if n * iw > 64 {
            let h = self.alloc_x()?;
            self.ev(Ev::new(EOp::Extracti64x4, EVL::L512, xi.num(), 0, h).imm(1))?;
            self.ku(EOp::Pmovzxdq, 64, h, h)?;
            self.ku(EOp::Pmovzxdq, 64, xi, xi)?;
            Some(h)
        } else {
            if ish.w < iw {
                let op = match (ish.w, iw) {
                    (1, 4) => EOp::Pmovzxbd,
                    (2, 4) => EOp::Pmovzxwd,
                    (1, _) => EOp::Pmovzxbq,
                    (2, _) => EOp::Pmovzxwq,
                    _ => EOp::Pmovzxdq,
                };
                self.ku(op, n * iw, xi, xi)?;
            }
            None
        };
        // the length in every lane, saturated to the lane width
        let isz = (n * iw).min(64);
        match len {
            VLen::K(l) => self.ksplat(XS0, isz, iw, if iw == 4 { l.min(0xffff_ffff) } else { l })?,
            VLen::R(g) if iw == 4 => {
                self.a.mov_ri(Size::Q, S0, 0xffff_ffff)?;
                self.a.alu_rr(AluOp::Cmp, Size::Q, g, S0);
                self.a.cmov(Cond::B, Size::Q, S0, g)?;
                self.ku(EOp::PbroadcastdR, isz, XS0, S0)?;
            }
            VLen::R(g) => self.ku(EOp::PbroadcastqR, isz, XS0, g)?,
        }
        // the zero lanes above a value of fewer than 16 bytes pass unless `len` is 0, where
        // every index fails anyway
        let cmp = if iw == 4 { EOp::Pcmpud } else { EOp::Pcmpuq };
        self.kcmp(cmp, isz, k, xi, XS0, Some(NLT))?;
        if let Some(h) = hi {
            self.kcmp(cmp, isz, KS0, h, XS0, Some(NLT))?;
            self.a.kbin(KBin::Or, KSize::B, k, k, KS0);
        }
        self.a.kortest(KSize::Q, k, k);
        self.jtrap(Cond::Ne, PanicKind::Bounds)?;
        Ok((iw, hi))
    }

    /// `vpgather*` after the bounds check of every index; byte and word lanes have no gather.
    fn k_gather(&mut self, s: Operand, idx: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        if sh.w < 4 {
            return self.v_gather(s, idx, rt, rv);
        }
        let ish = self.voshape(idx)?;
        let (base, len) = self.vbase(s)?;
        let xi = self.vget(idx, ish.size)?;
        let x = self.alloc_x()?;
        let k = self.alloc_k()?;
        let (iw, hi) = self.kindex(sh, ish, xi, len, k)?;
        if let Some(h) = hi {
            let x2 = self.alloc_x()?;
            for (d, i) in [(x, xi), (x2, h)] {
                self.kfirst(k, 8, 8)?;
                self.kvsib(EVsib::Pgatherqd, 64, d, base, i, 4, k)?;
            }
            self.ev(Ev::new(EOp::Inserti64x4, EVL::L512, x.num(), x.num(), x2).imm(1))?;
            return self.vput(rv, x, sh.size);
        }
        let vsz = sh.n * sh.w.max(iw);
        self.kfirst(k, sh.n, EVL::of_size(vsz).bytes() / sh.w.max(iw))?;
        let op = match (sh.w, iw) {
            (4, 4) => EVsib::Pgatherdd,
            (4, _) => EVsib::Pgatherqd,
            _ => EVsib::Pgatherqq,
        };
        self.kvsib(op, vsz, x, base, xi, sh.w, k)?;
        self.vput(rv, x, sh.size)
    }

    /// `vpscatter*` after the bounds check of every index. The instruction writes lanes from the
    /// lowest up where indices repeat, so the highest lane wins; two halves go low then high.
    fn k_scatter(&mut self, s: Operand, idx: Operand, v: Operand) -> Res<()> {
        let sh = self.voshape(v)?;
        if sh.w < 4 {
            return self.v_scatter(s, idx, v);
        }
        let ish = self.voshape(idx)?;
        let x = self.vread(v, sh.size)?;
        let (base, len) = self.vbase(s)?;
        let xi = self.vget(idx, ish.size)?;
        let k = self.alloc_k()?;
        let (iw, hi) = self.kindex(sh, ish, xi, len, k)?;
        if let Some(h) = hi {
            let x2 = self.alloc_x()?;
            self.ev(Ev::new(EOp::Extracti64x4, EVL::L512, x.num(), 0, x2).imm(1))?;
            for (d, i) in [(x, xi), (x2, h)] {
                self.kfirst(k, 8, 8)?;
                self.kvsib(EVsib::Pscatterqd, 64, d, base, i, 4, k)?;
            }
            return Ok(());
        }
        let vsz = sh.n * sh.w.max(iw);
        self.kfirst(k, sh.n, EVL::of_size(vsz).bytes() / sh.w.max(iw))?;
        let op = match (sh.w, iw) {
            (4, 4) => EVsib::Pscatterdd,
            (4, _) => EVsib::Pscatterqd,
            _ => EVsib::Pscatterqq,
        };
        self.kvsib(op, vsz, x, base, xi, sh.w, k)
    }

    // ---------------------------------------------------------------- division

    /// `x = x / d` per lane of `w` (2 or 4) bytes by `plan`. Clobbers `XS0`, `XS1`.
    fn kdiv_lanes(&mut self, size: u32, w: u32, plan: DivPlan, x: Xmm) -> Res<()> {
        let bits = (w * 8) as u8;
        let (srl, sra) =
            if w == 2 { (EShiftI::Psrlw, EShiftI::Psraw) } else { (EShiftI::Psrld, EShiftI::Psrad) };
        let (add, sub) = (by_w(w, KPADD), by_w(w, KPSUB));
        match plan {
            DivPlan::Copy { neg } => {
                if neg {
                    self.kneg(size, w, x, x)?;
                }
            }
            DivPlan::Shr(s) => self.ksh(srl, size, x, s as u8)?,
            DivPlan::Mul { m, s } => {
                self.kmulhi(size, w, false, x, x, m)?;
                if s > 0 {
                    self.ksh(srl, size, x, s as u8)?;
                }
            }
            DivPlan::MulAdd { m, s } => {
                let t = self.alloc_x()?;
                self.kmulhi(size, w, false, t, x, m)?;
                self.kb(sub, size, x, x, t)?;
                self.ksh(srl, size, x, 1)?;
                self.kb(add, size, x, x, t)?;
                if s > 0 {
                    self.ksh(srl, size, x, s as u8)?;
                }
            }
            DivPlan::Pow { l, neg } => {
                // a negative x gains d - 1, so the shift rounds toward zero
                let t = self.alloc_x()?;
                if l > 1 {
                    self.kshv(sra, size, t, x, (l - 1) as u8)?;
                    self.ksh(srl, size, t, bits - l as u8)?;
                } else {
                    self.kshv(srl, size, t, x, bits - 1)?;
                }
                self.kb(add, size, x, x, t)?;
                self.ksh(sra, size, x, l as u8)?;
                if neg {
                    self.kneg(size, w, x, x)?;
                }
            }
            DivPlan::SMul { m, s, add: plus_x, neg } => {
                let t = self.alloc_x()?;
                self.kmulhi(size, w, true, t, x, m as u64)?;
                if plus_x {
                    self.kb(add, size, t, t, x)?;
                }
                if s > 0 {
                    self.ksh(sra, size, t, s as u8)?;
                }
                // a negative x gains 1: the quotient rounds toward zero
                self.ksh(srl, size, x, bits - 1)?;
                self.kb(add, size, x, x, t)?;
                if neg {
                    self.kneg(size, w, x, x)?;
                }
            }
        }
        Ok(())
    }

    /// Truncating division by a constant: high products of word and doubleword lanes; byte
    /// lanes as their even and odd bytes extended to words. A divisor of 0 traps `DivByZero`
    /// at once, -1 traps `Overflow` where a lane is MIN. 64-bit lanes have no high product:
    /// lane by lane.
    fn k_div_const(&mut self, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let k = self.int_const(b).ok_or_else(|| bad("div_const divisor must be a constant"))?;
        let d = t.wrap(k) as i64;
        if d == 0 || sh.w == 8 {
            return self.v_div(VecOp::DivConst, a, b, rt, rv);
        }
        let sz = sh.size;
        let x = self.vget(a, sz)?;
        if t.signed && d == -1 {
            let k = self.alloc_k()?;
            self.ksplat(XS0, sz, sh.w, sh.sign())?;
            self.kcmp(by_w(sh.w, KPCMPEQ), sz, k, x, XS0, None)?;
            self.a.kortest(KSize::Q, k, k);
            self.jtrap(Cond::Ne, PanicKind::Overflow)?;
        }
        if sh.w > 1 {
            let n = sh.bits();
            let plan = div_plan(d, n, if t.signed { n - 1 } else { n }, t.signed);
            self.kdiv_lanes(sz, sh.w, plan, x)?;
            return self.vput(rv, x, sz);
        }
        // the even bytes in x, the odd ones in o, as 16-bit lanes of 8-bit values
        let o = self.alloc_x()?;
        let lo = self.ksplat_mem(sz, 2, 0x00ff);
        if t.signed {
            self.kshv(EShiftI::Psraw, sz, o, x, 8)?;
            self.ksh(EShiftI::Psllw, sz, x, 8)?;
            self.ksh(EShiftI::Psraw, sz, x, 8)?;
        } else {
            self.kshv(EShiftI::Psrlw, sz, o, x, 8)?;
            self.kb(EOp::Pandq, sz, x, x, lo)?;
        }
        let plan = div_plan(d, 16, if t.signed { 7 } else { 8 }, t.signed);
        self.kdiv_lanes(sz, 2, plan, x)?;
        self.kdiv_lanes(sz, 2, plan, o)?;
        self.ksh(EShiftI::Psllw, sz, o, 8)?;
        // per bit: the low byte of x's words, the high byte of o's
        let lo = self.ksplat_mem(sz, 2, 0x00ff);
        self.kbi(EOp::Pternlogq, sz, x, o, lo, 0xe4)?;
        self.vput(rv, x, sz)
    }

    // ---------------------------------------------------------------- multiplies

    /// The exact products of the even bytes (`e`) and of the odd bytes (`o`) as words: the
    /// high halves of the products of the bytes moved to the high byte of their word.
    fn kbyte_products(&mut self, sz: u32, signed: bool, xa: Xmm, xb: Xmm, e: Xmm, o: Xmm) -> Res<()> {
        let mulh = if signed { EOp::Pmulhw } else { EOp::Pmulhuw };
        let hi = self.ksplat_mem(sz, 2, 0xff00);
        self.kshv(EShiftI::Psllw, sz, XS0, xa, 8)?;
        self.kshv(EShiftI::Psllw, sz, XS1, xb, 8)?;
        self.kb(mulh, sz, e, XS0, XS1)?;
        self.kb(EOp::Pandq, sz, XS0, xa, hi)?;
        self.kb(EOp::Pandq, sz, XS1, xb, hi)?;
        self.kb(mulh, sz, o, XS0, XS1)
    }

    /// The 64-bit products of the even (`e`) and of the odd (`o`) 32-bit lanes. Clobbers `XS0`,
    /// `XS1`.
    fn kdword_products(&mut self, sz: u32, signed: bool, xa: Xmm, xb: Xmm, e: Xmm, o: Xmm) -> Res<()> {
        let mul = if signed { EOp::Pmuldq } else { EOp::Pmuludq };
        self.kb(mul, sz, e, xa, xb)?;
        self.kshv(EShiftI::Psrlq, sz, XS0, xa, 32)?;
        self.kshv(EShiftI::Psrlq, sz, XS1, xb, 32)?;
        self.kb(mul, sz, o, XS0, XS1)
    }

    /// Products clamped to the lane type's range.
    fn k_mul_sat(&mut self, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        if sh.w == 8 {
            return self.v_mul_sat(a, b, rt, rv);
        }
        let sz = sh.size;
        let vl = EVL::of_size(sz);
        let (xa, xb) = (self.vread(a, sz)?, self.vread(b, sz)?);
        let r = self.alloc_x()?;
        match (sh.w, t.signed) {
            (1, signed) => {
                // packed with saturation into each 128-bit block's low and high halves, then
                // interleaved
                let o = self.alloc_x()?;
                self.kbyte_products(sz, signed, xa, xb, r, o)?;
                if signed {
                    self.kb(EOp::Packsswb, sz, r, r, o)?;
                } else {
                    // `vpackuswb` reads signed words
                    let max = self.ksplat_mem(sz, 2, 0xff);
                    self.kb(EOp::Pminuw, sz, r, r, max)?;
                    self.kb(EOp::Pminuw, sz, o, o, max)?;
                    self.kb(EOp::Packuswb, sz, r, r, o)?;
                }
                let ctl: Vec<u8> = (0..16).map(|i| (i / 2 + 8 * (i % 2)) as u8).collect();
                let m = self.kpool(sz, &ctl);
                self.kb(EOp::Pshufb, sz, r, r, m)?;
            }
            (2, true) => {
                let hi = self.alloc_x()?;
                let k = self.alloc_k()?;
                self.kb(EOp::Pmullw, sz, r, xa, xb)?;
                self.kb(EOp::Pmulhw, sz, hi, xa, xb)?;
                // overflow where the high half is not the low half's sign
                self.kshv(EShiftI::Psraw, sz, XS0, r, 15)?;
                self.kcmp(EOp::Pcmpw, sz, k, hi, XS0, Some(NE))?;
                // MAX, or MIN = !MAX for a negative product
                self.kshv(EShiftI::Psraw, sz, XS0, hi, 15)?;
                let max = self.ksplat_mem(sz, 2, 0x7fff);
                self.kb(EOp::Pxorq, sz, XS0, XS0, max)?;
                self.ev(Ev::new(EOp::Movdqu16, vl, r.num(), 0, XS0).mask(k, false))?;
            }
            (2, false) => {
                let k = self.alloc_k()?;
                self.kb(EOp::Pmullw, sz, r, xa, xb)?;
                self.kb(EOp::Pmulhuw, sz, XS1, xa, xb)?;
                self.kcmp(EOp::Ptestmw, sz, k, XS1, XS1, None)?;
                self.ksplat(XS0, sz, 2, 0xffff)?;
                self.ev(Ev::new(EOp::Movdqu16, vl, r.num(), 0, XS0).mask(k, false))?;
            }
            (_, signed) => {
                // the 64-bit products clamped, then their low halves interleaved
                let o = self.alloc_x()?;
                self.kdword_products(sz, signed, xa, xb, r, o)?;
                if signed {
                    let (max, min) =
                        (self.ksplat_mem(sz, 8, 0x7fff_ffff), self.ksplat_mem(sz, 8, !0x7fff_ffff));
                    for x in [r, o] {
                        self.kb(EOp::Pminsq, sz, x, x, max)?;
                        self.kb(EOp::Pmaxsq, sz, x, x, min)?;
                    }
                } else {
                    let max = self.ksplat_mem(sz, 8, 0xffff_ffff);
                    for x in [r, o] {
                        self.kb(EOp::Pminuq, sz, x, x, max)?;
                    }
                }
                self.kinterleave_idx(XS0, sz, false)?;
                self.ev(Ev::new(EOp::Permt2d, vl, r.num(), XS0.num(), o))?;
            }
        }
        self.vput(rv, r, sz)
    }

    /// `Some(a * b)` iff no lane overflows: the tag from the flags without a branch.
    fn k_checked_mul(&mut self, a: Operand, b: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let (none, some, pty) = self
            .cx
            .m
            .types
            .option_shape(rt)
            .ok_or_else(|| bad("checked vector op result must be an Option"))?;
        let sh = self.vshape(pty)?;
        let t = self.vint(sh)?;
        if sh.w == 8 {
            return self.v_checked(VecOp::CheckedMul, a, b, rt, rv);
        }
        let sz = sh.size;
        let (xa, xb) = (self.vread(a, sz)?, self.vread(b, sz)?);
        let r = self.alloc_x()?;
        let h = self.alloc_x()?;
        let k = self.alloc_k()?;
        let (d, off, _) = self.option_home(rt, some, rv)?;
        // k = the lanes that overflow
        match (sh.w, t.signed) {
            (1, signed) => {
                self.kbyte_products(sz, signed, xa, xb, r, h)?;
                let hi = self.ksplat_mem(sz, 2, 0xff00);
                if signed {
                    // a word product fits a byte iff adding 0x80 leaves the high byte zero
                    let bias = self.ksplat_mem(sz, 2, 0x80);
                    self.kb(EOp::Paddw, sz, XS0, r, bias)?;
                    self.kb(EOp::Paddw, sz, XS1, h, bias)?;
                    self.kb(EOp::Porq, sz, XS0, XS0, XS1)?;
                } else {
                    self.kb(EOp::Porq, sz, XS0, r, h)?;
                }
                self.ev(Ev::new(EOp::Ptestmw, EVL::of_size(sz), k.num(), XS0.num(), hi))?;
                // the wrapped products: the low byte of r's words, the high byte of h's
                self.ksh(EShiftI::Psllw, sz, h, 8)?;
                let lo = self.ksplat_mem(sz, 2, 0x00ff);
                self.kbi(EOp::Pternlogq, sz, r, h, lo, 0xe4)?;
            }
            (2, signed) => {
                self.kb(EOp::Pmullw, sz, r, xa, xb)?;
                if signed {
                    self.kb(EOp::Pmulhw, sz, h, xa, xb)?;
                    self.kshv(EShiftI::Psraw, sz, XS0, r, 15)?;
                    self.kcmp(EOp::Pcmpw, sz, k, h, XS0, Some(NE))?;
                } else {
                    self.kb(EOp::Pmulhuw, sz, h, xa, xb)?;
                    self.kcmp(EOp::Ptestmw, sz, k, h, h, None)?;
                }
            }
            (_, signed) => {
                // the high halves of the 64-bit products against the low halves' signs
                let o = self.alloc_x()?;
                self.kb(EOp::Pmulld, sz, r, xa, xb)?;
                self.kdword_products(sz, signed, xa, xb, h, o)?;
                self.kinterleave_idx(XS0, sz, true)?;
                self.ev(Ev::new(EOp::Permt2d, EVL::of_size(sz), h.num(), XS0.num(), o))?;
                if signed {
                    self.kshv(EShiftI::Psrad, sz, XS0, r, 31)?;
                    self.kcmp(EOp::Pcmpd, sz, k, h, XS0, Some(NE))?;
                } else {
                    self.kcmp(EOp::Ptestmd, sz, k, h, h, None)?;
                }
            }
        }
        // ZF = no lane overflowed
        self.a.kortest(KSize::Q, k, k);
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

    // ---------------------------------------------------------------- bit counts

    /// Byte lanes count through a 16-entry table per nibble (`vpshufb` in each 128-bit block);
    /// wider lanes add their bytes' counts (`popcount`) or take `vplzcnt*` (`clz`).
    fn k_count(&mut self, op: VecOp, a: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let sz = sh.size;
        let x = self.vget(a, sz)?;
        let clz = op == VecOp::Clz;
        if clz && sh.w >= 4 {
            self.ku(if sh.w == 4 { EOp::Plzcntd } else { EOp::Plzcntq }, sz, x, x)?;
            return self.vput(rv, x, sz);
        }
        if clz && sh.w == 2 {
            // per doubleword, with bit 15 set below the word counted, so a zero word counts 16:
            // the low word moved up, and the doubleword itself for the high word
            let t = self.alloc_x()?;
            let b15 = self.ksplat_mem(sz, 4, 0x8000);
            self.kshv(EShiftI::Pslld, sz, t, x, 16)?;
            self.kb(EOp::Pord, sz, t, t, b15)?;
            self.ku(EOp::Plzcntd, sz, t, t)?;
            self.kb(EOp::Pord, sz, x, x, b15)?;
            self.ku(EOp::Plzcntd, sz, x, x)?;
            self.ksh(EShiftI::Pslld, sz, x, 16)?;
            self.kb(EOp::Pord, sz, x, x, t)?;
            return self.vput(rv, x, sz);
        }
        let lut = self.alloc_x()?;
        let hi = self.alloc_x()?;
        let table = if clz { &CLZ_LUT } else { &POPCOUNT_LUT };
        self.kload(lut, sz, table)?;
        let nib = self.ksplat_mem(sz, 1, 0x0f);
        self.kshv(EShiftI::Psrlw, sz, hi, x, 4)?;
        self.kb(EOp::Pandq, sz, hi, hi, nib)?;
        self.kb(EOp::Pandq, sz, x, x, nib)?;
        if clz {
            // clz(high nibble), plus clz(low nibble) where the high nibble is zero
            let k = self.alloc_k()?;
            self.vzero(XS0)?;
            self.kcmp(EOp::Pcmpeqb, sz, k, hi, XS0, None)?;
            self.kb(EOp::Pshufb, sz, x, lut, x)?;
            self.kb(EOp::Pshufb, sz, hi, lut, hi)?;
            self.ev(Ev::new(EOp::Paddb, EVL::of_size(sz), hi.num(), hi.num(), x).mask(k, false))?;
            return self.vput(rv, hi, sz);
        }
        self.kb(EOp::Pshufb, sz, x, lut, x)?;
        self.kb(EOp::Pshufb, sz, hi, lut, hi)?;
        self.kb(EOp::Paddb, sz, x, x, hi)?;
        match sh.w {
            1 => {}
            2 | 4 => {
                // the sums of adjacent bytes as words, of adjacent words as doublewords
                let ones = self.ksplat_mem(sz, 1, 1);
                self.kb(EOp::Pmaddubsw, sz, x, x, ones)?;
                if sh.w == 4 {
                    let ones = self.ksplat_mem(sz, 2, 1);
                    self.kb(EOp::Pmaddwd, sz, x, x, ones)?;
                }
            }
            _ => {
                self.vzero(XS0)?;
                self.kb(EOp::Psadbw, sz, x, x, XS0)?;
            }
        }
        self.vput(rv, x, sz)
    }

    // ---------------------------------------------------------------- permutes and lookups

    /// Lane j of the result = lane `k_j` of the operand (constant indices; the result may have
    /// another lane count). 16-, 32- and 64-bit lanes: one `vpermi2*` with the indices as a
    /// constant. Bytes: one `vpshufb` where every byte stays in its 128-bit block, else the
    /// word holding each result byte through `vpermi2w`, for even and odd result bytes, then
    /// the byte within it.
    fn k_permute(&mut self, args: &[Operand], rt: TypeId, rv: Value) -> Res<()> {
        let to = self.vshape(rt)?;
        let a = args.first().copied().ok_or_else(|| bad("permute operand"))?;
        let from = self.voshape(a)?;
        let mut ks = Vec::with_capacity(to.n as usize);
        for &k in &args[1..] {
            let k = self
                .int_const(k)
                .and_then(|k| u32::try_from(k).ok())
                .filter(|k| *k < from.n)
                .ok_or_else(|| bad("permute index"))?;
            ks.push(k);
        }
        if ks.len() != to.n as usize || from.w != to.w {
            return Err(bad("permute shape"));
        }
        let w = to.w;
        let sz = to.size.max(from.size);
        let vb = EVL::of_size(sz).bytes();
        let x = self.vread(a, from.size)?;
        let r = self.alloc_x()?;
        // lane `j` of an index vector of `lw`-byte lanes
        let lanes = |lw: u32, f: &dyn Fn(u32) -> u32| -> Vec<u8> {
            (0..vb / lw).flat_map(|j| u64::from(f(j)).to_le_bytes()[..lw as usize].to_vec()).collect()
        };
        let src = |j: u32| ks.get(j as usize).copied().unwrap_or(0);
        if w >= 2 {
            self.kload(r, sz, &lanes(w, &src))?;
            let op = by_w(w, [EOp::Permi2w, EOp::Permi2w, EOp::Permi2d, EOp::Permi2q]);
            self.kb(op, sz, r, x, x)?;
            return self.vput(rv, r, to.size);
        }
        let n = to.n;
        if (0..n).all(|j| src(j) / 16 == j / 16) {
            let ctl: Vec<u8> = (0..vb).map(|j| if j < n { (src(j) % 16) as u8 } else { 0x80 }).collect();
            let m = self.kpool(sz, &ctl);
            self.kb(EOp::Pshufb, sz, r, x, m)?;
            return self.vput(rv, r, to.size);
        }
        // word i of `r` = the word holding result byte 2i, of `t` the one holding byte 2i + 1
        self.kload(r, sz, &lanes(2, &|i| src(2 * i) / 2))?;
        self.kb(EOp::Permi2w, sz, r, x, x)?;
        if (0..n / 2).all(|i| src(2 * i) % 2 == 0 && src(2 * i + 1) == src(2 * i) + 1) {
            return self.vput(rv, r, to.size);
        }
        let t = self.alloc_x()?;
        self.kload(t, sz, &lanes(2, &|i| src(2 * i + 1) / 2))?;
        self.kb(EOp::Permi2w, sz, t, x, x)?;
        let pick = |odd: u32| -> Vec<u8> {
            (0..vb)
                .map(|j| if j % 2 == odd && j < n { ((j & !1) % 16 + src(j) % 2) as u8 } else { 0x80 })
                .collect()
        };
        let (me, mo) = (self.kpool(sz, &pick(0)), self.kpool(sz, &pick(1)));
        self.kb(EOp::Pshufb, sz, r, r, me)?;
        self.kb(EOp::Pshufb, sz, t, t, mo)?;
        self.kb(EOp::Porq, sz, r, r, t)?;
        self.vput(rv, r, to.size)
    }

    /// Lane j = `tbl[idx[j]]`, zero for an index outside the table (a negative one too).
    /// Index lanes as wide as the table's: an unsigned compare against the lane count and a
    /// zero-masked `vpermi2*`; bytes through `vpshufb` per 128-bit block of the table. Other
    /// index widths: lane by lane.
    fn k_lookup(&mut self, tbl: Operand, idx: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let ish = self.voshape(idx)?;
        if ish.w != sh.w {
            return self.v_lookup(tbl, idx, rt, rv);
        }
        let sz = sh.size;
        let vl = EVL::of_size(sz);
        let t = self.vread(tbl, sz)?;
        let i = self.vget(idx, sz)?;
        if sh.w >= 2 {
            let k = self.alloc_k()?;
            self.ksplat(XS0, sz, sh.w, u64::from(sh.n))?;
            self.kcmp(by_w(sh.w, KPCMPU), sz, k, i, XS0, Some(LT))?;
            let op = by_w(sh.w, [EOp::Permi2w, EOp::Permi2w, EOp::Permi2d, EOp::Permi2q]);
            self.ev(Ev::new(op, vl, i.num(), t.num(), t).mask(k, true))?;
            return self.vput(rv, i, sz);
        }
        // `vpshufb` gives zero for an index byte with its top bit set and else takes the low
        // four bits: adding 0x80 - 16 * blocks with unsigned saturation sets the top bit of
        // every index outside the table and leaves the other bits of the others. A table of
        // fewer than 16 lanes has zeros above it in its register.
        let blocks = sz.div_ceil(16);
        let bias = self.ksplat_mem(sz, 1, u64::from(0x80 - 16 * blocks));
        self.kb(EOp::Paddusb, sz, i, i, bias)?;
        let r = self.alloc_x()?;
        if blocks == 1 {
            self.kb(EOp::Pshufb, sz, r, t, i)?;
            return self.vput(rv, r, sz);
        }
        // index bit 4 picks the odd block of a pair, bit 5 the upper pair
        let (k4, k5) = (self.alloc_k()?, self.alloc_k()?);
        self.kshv(EShiftI::Psllw, sz, XS0, i, 3)?;
        self.kmask_from_vec(k4, XS0, sh.n, 1)?;
        if blocks == 4 {
            self.kshv(EShiftI::Psllw, sz, XS0, i, 2)?;
            self.kmask_from_vec(k5, XS0, sh.n, 1)?;
        }
        let (s0, s1) = (self.alloc_x()?, self.alloc_x()?);
        for pair in 0..blocks / 2 {
            for (s, b) in [(s0, 2 * pair), (s1, 2 * pair + 1)] {
                // block b of the table in every block
                let imm = if blocks == 2 { [0x00, 0x03][b as usize] } else { (b * 0x55) as u8 };
                self.kbi(EOp::Shufi64x2, sz, s, t, t, imm)?;
                self.kb(EOp::Pshufb, sz, s, s, i)?;
            }
            let d = if pair == 0 { r } else { s0 };
            self.ev(Ev::new(EOp::Pblendmb, vl, d.num(), s0.num(), s1).mask(k4, false))?;
        }
        if blocks == 4 {
            self.ev(Ev::new(EOp::Pblendmb, vl, r.num(), r.num(), s0).mask(k5, false))?;
        }
        self.vput(rv, r, sz)
    }

    // ---------------------------------------------------------------- compress / expand

    /// `compress`: the active lanes packed low; `expand`: the low lanes spread to the active
    /// positions; other lanes zero. 32- and 64-bit lanes: `vpcompress*` / `vpexpand*` in
    /// registers. Byte and word lanes as doublewords, 16 at a time: up to 16 lanes in
    /// registers; more through the home of the result, each chunk at the count of active lanes
    /// before it (a compressed chunk stored there ends at or before its own end, and the next
    /// one overwrites its zeros).
    fn k_compress(&mut self, expand: bool, a: Operand, m: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let sz = sh.size;
        let k = self.read_k(m)?;
        if sh.w >= 4 {
            let x = self.vread(a, sz)?;
            let r = self.alloc_x()?;
            let vl = EVL::of_size(sz);
            let e = match (expand, sh.w) {
                (true, 4) => Ev::new(EOp::Pexpandd, vl, r.num(), 0, x),
                (true, _) => Ev::new(EOp::Pexpandq, vl, r.num(), 0, x),
                (false, 4) => Ev::new(EOp::Pcompressd, vl, x.num(), 0, r),
                (false, _) => Ev::new(EOp::Pcompressq, vl, x.num(), 0, r),
            };
            self.ev(e.mask(k, true))?;
            return self.vput(rv, r, sz);
        }
        let (zx, tr) = if sh.w == 1 { (EOp::Pmovzxbd, EOp::Pmovdb) } else { (EOp::Pmovzxwd, EOp::Pmovdw) };
        // XS0 = XS1 compressed or expanded as doublewords
        let op = |vl: EVL| {
            if expand {
                Ev::new(EOp::Pexpandd, vl, XS0.num(), 0, XS1)
            } else {
                Ev::new(EOp::Pcompressd, vl, XS1.num(), 0, XS0)
            }
        };
        if sh.n <= 16 {
            let x = self.vread(a, sz)?;
            let r = self.alloc_x()?;
            let dsz = sh.n * 4;
            self.ku(zx, dsz, XS1, x)?;
            self.ev(op(EVL::of_size(dsz)).mask(k, true))?;
            self.ev(Ev::new(tr, EVL::of_size(dsz), XS0.num(), 0, r))?;
            return self.vput(rv, r, sz);
        }
        let ma = self.vmem(a)?;
        let bits = self.alloc_g()?;
        let kc = self.alloc_k()?;
        self.kbits(bits, k, sh.n)?;
        let d = self.def_mem(rv)?;
        let (
            Mem::Sib { base: Some(ab), index: None, disp: ad },
            Mem::Sib { base: Some(db), index: None, disp: dd },
        ) = (ma, d)
        else {
            return Err(bad("internal: vector home address"));
        };
        if !expand {
            self.zero_mem(d, sz)?;
        }
        let sc = scale_of(sh.w);
        for j in 0..sh.n / 16 {
            let kj = if j == 0 {
                k
            } else {
                self.a.kshift(false, KSize::of_lanes(sh.n), kc, k, (16 * j) as u8);
                kc
            };
            // S0 = the active lanes below the chunk
            if j > 0 {
                self.a.mov_rr(Size::Q, S0, bits);
                self.a.shift_ri(ShiftOp::Shl, Size::Q, S0, (64 - 16 * j) as u8)?;
                self.a.popcnt(Size::Q, S0, S0)?;
            }
            let at = |base: Gpr, disp: i32| {
                if j == 0 { Mem::base(base, disp) } else { Mem::index(base, S0, sc, disp) }
            };
            let (src, dst) = if expand {
                (at(ab, ad), lane(d, 16 * j, sh.w)?)
            } else {
                (lane(ma, 16 * j, sh.w)?, at(db, dd))
            };
            self.ku(zx, 64, XS1, src)?;
            self.ev(op(EVL::L512).mask(kj, true))?;
            self.ev(Ev::new(tr, EVL::L512, XS0.num(), 0, dst))?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------- shifts

    /// 8-bit lanes shifted by a vector of amounts: the even and the odd bytes each shifted in
    /// their word (`vpsllvw` / `vpsrlvw` / `vpsravw`), then merged. Amounts of 8 or more trap
    /// (`<<`, `>>`) or are masked (`<<%`).
    pub(super) fn k_shift_bytes(
        &mut self,
        op: VecOp,
        a: Operand,
        amt: Operand,
        rt: TypeId,
        rv: Value,
    ) -> Res<()> {
        let sh = self.vshape(rt)?;
        let t = self.vint(sh)?;
        let sz = sh.size;
        let right = op == VecOp::Shr;
        let s = self.vread(amt, sz)?;
        let x = self.vget(a, sz)?;
        let (e, o, c) = (self.alloc_x()?, self.alloc_x()?, self.alloc_x()?);
        let s = if op == VecOp::ShlWrap {
            let seven = self.ksplat_mem(sz, 1, 7);
            self.kb(EOp::Pandq, sz, XS1, s, seven)?;
            XS1
        } else {
            // the zero lanes above a value of fewer than 16 bytes are in range
            let k = self.alloc_k()?;
            self.ksplat(XS0, sz, 1, 7)?;
            self.kcmp(EOp::Pcmpub, sz, k, s, XS0, Some(NLE))?;
            self.a.kortest(KSize::Q, k, k);
            self.jtrap(Cond::Ne, PanicKind::Overflow)?;
            s
        };
        // c = the even bytes; XS0 = their amounts, o = the odd bytes' amounts
        self.ksplat(c, sz, 2, 0x00ff)?;
        self.kb(EOp::Pandq, sz, XS0, s, c)?;
        self.kshv(EShiftI::Psrlw, sz, o, s, 8)?;
        match (right, t.signed) {
            (false, _) => {
                self.kb(EOp::Psllvw, sz, e, x, XS0)?;
                // x & !c: the odd bytes in place
                self.kb(EOp::Pandnq, sz, x, c, x)?;
                self.kb(EOp::Psllvw, sz, x, x, o)?;
            }
            (true, false) => {
                self.kb(EOp::Pandq, sz, e, x, c)?;
                self.kb(EOp::Psrlvw, sz, e, e, XS0)?;
                self.kb(EOp::Psrlvw, sz, x, x, o)?;
            }
            (true, true) => {
                self.kshv(EShiftI::Psllw, sz, e, x, 8)?;
                self.ksh(EShiftI::Psraw, sz, e, 8)?;
                self.kb(EOp::Psravw, sz, e, e, XS0)?;
                self.kb(EOp::Psravw, sz, x, x, o)?;
            }
        }
        // per bit: e's low bytes, x's high bytes
        self.kbi(EOp::Pternlogq, sz, x, e, c, 0xd8)?;
        self.vput(rv, x, sz)
    }
}

#[cfg(test)]
mod tests {
    use super::{DivPlan, div_plan};

    /// The plan's lane operations on one `n`-bit lane, `x` normalized (sign-extended when
    /// `signed`): what [`super::Cg::kdiv_lanes`] emits.
    fn eval(plan: DivPlan, n: u32, signed: bool, x: i64) -> i64 {
        let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        let u = |v: i64| (v as u64) & mask;
        let s = |v: i64| ((u(v) << (64 - n)) as i64) >> (64 - n);
        let norm = |v: i64| if signed { s(v) } else { u(v) as i64 };
        let srl = |v: i64, k: u32| (u(v) >> k) as i64;
        let sra = |v: i64, k: u32| s(v) >> k;
        let mulhu = |v: i64, m: u64| ((u128::from(u(v)) * u128::from(m & mask)) >> n) as i64;
        let mulhs = |v: i64, m: i64| ((i128::from(s(v)) * i128::from(s(m))) >> n) as i64;
        let r = match plan {
            DivPlan::Copy { neg } => {
                if neg {
                    -x
                } else {
                    x
                }
            }
            DivPlan::Shr(k) => srl(x, k),
            DivPlan::Mul { m, s } => srl(mulhu(x, m), s),
            DivPlan::MulAdd { m, s } => {
                let t = mulhu(x, m);
                let q = u(t + srl(u(x) as i64 - t, 1)) as i64;
                srl(q, s)
            }
            DivPlan::Pow { l, neg } => {
                let t = if l > 1 { srl(sra(x, l - 1), n - l) } else { srl(x, n - 1) };
                let q = sra(x + t, l);
                if neg { -q } else { q }
            }
            DivPlan::SMul { m, s: sh, add, neg } => {
                let mut t = mulhs(x, m);
                if add {
                    t = s(t + x);
                }
                let q = s(sra(t, sh) + srl(x, n - 1));
                if neg { -q } else { q }
            }
        };
        norm(r)
    }

    fn check(n: u32, prec: u32, signed: bool, d: i64, xs: impl Iterator<Item = i64>) {
        let plan = div_plan(d, n, prec, signed);
        for x in xs {
            let want = if signed { x.wrapping_div(d) } else { ((x as u64) / (d as u64)) as i64 };
            let want = if signed {
                ((want as u64) << (64 - n)) as i64 >> (64 - n)
            } else {
                want & ((1i64 << n) - 1)
            };
            assert_eq!(eval(plan, n, signed, x), want, "{x} / {d} on {n}-bit lanes ({plan:?})");
        }
    }

    /// Every byte divisor and dividend, as the 16-bit lanes the byte lanes divide in.
    #[test]
    fn div_plans_bytes() {
        for d in 1..256 {
            check(16, 8, false, d, 0..256);
        }
        for d in -128..128 {
            if d != 0 {
                check(16, 7, true, d, (-128..128).filter(|&x| !(x == -128 && d == -1)));
            }
        }
    }

    /// Every 16-bit divisor against edge and spread dividends; every dividend for some.
    #[test]
    fn div_plans_words() {
        let edges = |lo: i64, hi: i64| {
            let mut v: Vec<i64> = (lo..lo + 300).chain(hi - 300..=hi).collect();
            v.extend((lo..=hi).step_by(97));
            v
        };
        let (ux, sx) = (edges(0, 0xffff), edges(-0x8000, 0x7fff));
        for d in 1..=0xffff {
            check(16, 16, false, d, ux.iter().copied());
        }
        for d in -0x8000..=0x7fff {
            if d != 0 {
                check(16, 15, true, d, sx.iter().copied().filter(|&x| !(x == -0x8000 && d == -1)));
            }
        }
        for d in [3, 5, 6, 7, 10, 11, 100, 641, 0x7fff, 0xfffe] {
            check(16, 16, false, d, 0..=0xffff);
            let d = ((d as u16) as i16) as i64;
            check(16, 15, true, d, (-0x8000..=0x7fff).filter(|&x| !(x == -0x8000 && d == -1)));
        }
    }

    /// 32-bit divisors: small ones, powers of two and their neighbours, extremes, a pseudo
    /// random spread; dividends likewise.
    #[test]
    fn div_plans_dwords() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut ds: Vec<u64> = (1..2000).collect();
        for k in 1..32 {
            ds.extend([(1u64 << k) - 1, 1 << k, (1 << k) + 1]);
        }
        ds.extend([0xffff_fffe, 0xffff_ffff, 0x8000_0001, 0x7fff_ffff]);
        ds.extend((0..3000).map(|_| next() & 0xffff_ffff));
        let mut xs: Vec<u64> = (0..500).chain((0..500).map(|i| 0xffff_ffff - i)).collect();
        xs.extend((0..500).map(|i| 0x8000_0000 - 250 + i));
        xs.extend((0..2000).map(|_| next() & 0xffff_ffff));
        for &d in &ds {
            if d == 0 {
                continue;
            }
            check(32, 32, false, d as i64, xs.iter().map(|&x| x as i64));
            let sd = (d as u32) as i32 as i64;
            check(
                32,
                31,
                true,
                sd,
                xs.iter()
                    .map(|&x| (x as u32) as i32 as i64)
                    .filter(|&x| !(x == i64::from(i32::MIN) && sd == -1)),
            );
        }
    }
}
