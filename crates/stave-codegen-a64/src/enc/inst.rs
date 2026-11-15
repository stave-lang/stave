//! A64 instruction forms (Arm ARM C4 encoding index). Every method validates first and emits one
//! word (or a short sequence) only when all operands are valid.

use stave_obj::{RelocKind, SymbolId};

use super::imm::{encode_fp8, encode_logical};
use super::{Asm, Cond, EncError, FixKind, Fp, Label, Reg, VReg, Width};

/// Add/subtract flavour (`op:S` bits 30:29).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddSub {
    Add = 0,
    /// Add, setting flags.
    Adds = 1,
    Sub = 2,
    /// Subtract, setting flags.
    Subs = 3,
}

impl AddSub {
    fn sets_flags(self) -> bool {
        matches!(self, AddSub::Adds | AddSub::Subs)
    }
}

/// Logical operations (`opc` bits 30:29, `N` inverts the second operand).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Logic {
    And,
    Orr,
    Eor,
    /// And, setting flags.
    Ands,
    Bic,
    Orn,
    Eon,
    Bics,
}

impl Logic {
    fn bits(self) -> (u32, u32) {
        match self {
            Logic::And => (0, 0),
            Logic::Orr => (1, 0),
            Logic::Eor => (2, 0),
            Logic::Ands => (3, 0),
            Logic::Bic => (0, 1),
            Logic::Orn => (1, 1),
            Logic::Eon => (2, 1),
            Logic::Bics => (3, 1),
        }
    }
}

/// Shift of a shifted-register operand, or the operation of a variable/immediate shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shift {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
    Ror = 3,
}

/// Alias kept for readability at call sites of [`Asm::shift_reg`].
pub type ShiftR = Shift;

/// Source width of a sign/zero extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ext {
    B,
    H,
    W,
}

/// Access size of an integer load/store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MemSize {
    B = 0,
    H = 1,
    W = 2,
    X = 3,
}

/// Access size of an FP/SIMD load/store (`s`, `d`, `q` registers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VecSize {
    S,
    D,
    Q,
}

impl VecSize {
    fn scale(self) -> u32 {
        match self {
            VecSize::S => 2,
            VecSize::D => 3,
            VecSize::Q => 4,
        }
    }
}

/// Extension of a register-offset index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IdxExt {
    /// 64-bit index (`lsl`).
    Lsl,
    /// Zero-extended 32-bit index.
    Uxtw,
    /// Sign-extended 32-bit index.
    Sxtw,
    /// 64-bit index (`sxtx`).
    Sxtx,
}

impl IdxExt {
    fn option(self) -> u32 {
        match self {
            IdxExt::Uxtw => 0b010,
            IdxExt::Lsl => 0b011,
            IdxExt::Sxtw => 0b110,
            IdxExt::Sxtx => 0b111,
        }
    }
}

/// A load/store address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Addr {
    /// `[base, #off]`: scaled unsigned 12-bit offset when aligned, else unscaled signed 9-bit.
    Off(Reg, i64),
    /// `[base, #off]!` (signed 9-bit, or scaled 7-bit for pairs).
    Pre(Reg, i64),
    /// `[base], #off`.
    Post(Reg, i64),
    /// `[base, index{, ext #scale}]`; `scaled` shifts the index by the access size.
    Idx { base: Reg, index: Reg, ext: IdxExt, scaled: bool },
}

/// FP two-source operations (`opcode` bits 15:12).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FpBin {
    Mul = 0,
    Div = 1,
    Add = 2,
    Sub = 3,
    Max = 4,
    Min = 5,
    MaxNm = 6,
    MinNm = 7,
    Nmul = 8,
}

/// FP one-source operations (`opcode` bits 20:15).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FpUn {
    Abs = 1,
    Neg = 2,
    Sqrt = 3,
    RintN = 8,
    RintP = 9,
    RintM = 10,
    RintZ = 11,
    RintA = 12,
    RintX = 14,
    RintI = 15,
}

/// FP fused multiply-add (`o1:o0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FpFma {
    Madd,
    Msub,
    Nmadd,
    Nmsub,
}

/// FP → integer conversion rounding and signedness (`fcvt{n,a,p,m,z}{s,u}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FcvtRound {
    Ns,
    Nu,
    As,
    Au,
    Ps,
    Pu,
    Ms,
    Mu,
    Zs,
    Zu,
}

impl FcvtRound {
    fn bits(self) -> (u32, u32) {
        match self {
            FcvtRound::Ns => (0, 0),
            FcvtRound::Nu => (0, 1),
            FcvtRound::As => (0, 4),
            FcvtRound::Au => (0, 5),
            FcvtRound::Ps => (1, 0),
            FcvtRound::Pu => (1, 1),
            FcvtRound::Ms => (2, 0),
            FcvtRound::Mu => (2, 1),
            FcvtRound::Zs => (3, 0),
            FcvtRound::Zu => (3, 1),
        }
    }
}

/// Register number where operand 31 means the zero register.
fn zr(r: Reg) -> Result<u32, EncError> {
    if r.is_sp() {
        return Err(EncError::Invalid("sp is not allowed here (register 31 is xzr)"));
    }
    Ok(u32::from(r.num()))
}

/// Register number where operand 31 means the stack pointer.
fn sp(r: Reg) -> Result<u32, EncError> {
    if r.is_zr() {
        return Err(EncError::Invalid("xzr is not allowed here (register 31 is sp)"));
    }
    Ok(u32::from(r.num()))
}

/// A plain general register (branch targets).
fn gpr(r: Reg) -> Result<u32, EncError> {
    if r.is_sp() || r.is_zr() {
        return Err(EncError::Invalid("needs x0..x30"));
    }
    Ok(u32::from(r.num()))
}

fn v(r: VReg) -> u32 {
    u32::from(r.num())
}

fn range(imm: i64, lo: i64, hi: i64, bits: u32) -> Result<(), EncError> {
    if (lo..=hi).contains(&imm) { Ok(()) } else { Err(EncError::ImmRange { imm, bits }) }
}

/// Encoding class of a single-register load/store.
struct Ls {
    size: u32,
    v: u32,
    opc: u32,
    scale: u32,
}

impl Asm {
    // ---- moves ----

    fn wide(&mut self, opc: u32, w: Width, rd: Reg, imm16: u64, shift: u32) -> Result<(), EncError> {
        let rd = zr(rd)?;
        range(imm16 as i64, 0, 0xffff, 16)?;
        if !shift.is_multiple_of(16) || shift >= w.bits() {
            return Err(EncError::Invalid("move-wide shift must be 0, 16 (W) or 0..=48 step 16 (X)"));
        }
        self.word(w.sf() | (opc << 29) | 0x1280_0000 | ((shift / 16) << 21) | ((imm16 as u32) << 5) | rd);
        Ok(())
    }

    /// `movz rd, #imm16, lsl #shift`.
    pub fn movz(&mut self, w: Width, rd: Reg, imm16: u64, shift: u32) -> Result<(), EncError> {
        self.wide(2, w, rd, imm16, shift)
    }

    /// `movn rd, #imm16, lsl #shift` (`rd = !(imm16 << shift)`).
    pub fn movn(&mut self, w: Width, rd: Reg, imm16: u64, shift: u32) -> Result<(), EncError> {
        self.wide(0, w, rd, imm16, shift)
    }

    /// `movk rd, #imm16, lsl #shift` (insert, keeping other bits).
    pub fn movk(&mut self, w: Width, rd: Reg, imm16: u64, shift: u32) -> Result<(), EncError> {
        self.wide(3, w, rd, imm16, shift)
    }

    /// Materializes any constant: one `movz`/`movn`, or `orr rd, zr, #bitmask`, or
    /// `movz`/`movn` + `movk`s. `W`: the value must fit in 32 bits.
    pub fn mov_imm(&mut self, w: Width, rd: Reg, val: u64) -> Result<(), EncError> {
        zr(rd)?;
        if w == Width::W && val > 0xffff_ffff {
            return Err(EncError::ImmRange { imm: val as i64, bits: 32 });
        }
        let n = (w.bits() / 16) as usize;
        let halves: Vec<u64> = (0..n).map(|i| (val >> (16 * i)) & 0xffff).collect();
        let zeros = halves.iter().filter(|&&h| h == 0).count();
        let ones = halves.iter().filter(|&&h| h == 0xffff).count();
        if zeros < n - 1 && ones < n - 1 && encode_logical(val, w.bits()).is_some() {
            return self.logic_imm(Logic::Orr, w, rd, Reg::ZR, val);
        }
        let (skip, first_op) = if ones > zeros { (0xffff, 0) } else { (0, 2) };
        let first = halves.iter().position(|&h| h != skip).unwrap_or(0);
        let h0 = if first_op == 0 { !halves[first] & 0xffff } else { halves[first] };
        self.wide(first_op, w, rd, h0, 16 * first as u32)?;
        for (i, &h) in halves.iter().enumerate() {
            if i != first && h != skip {
                self.wide(3, w, rd, h, 16 * i as u32)?;
            }
        }
        Ok(())
    }

    /// `mov rd, rn` (`orr rd, zr, rn`, or `add rd, rn, #0` when either is `sp`).
    pub fn mov(&mut self, w: Width, rd: Reg, rn: Reg) -> Result<(), EncError> {
        if rd.is_sp() || rn.is_sp() {
            self.addsub_imm(AddSub::Add, w, rd, rn, 0)
        } else {
            self.logic_reg(Logic::Orr, w, rd, Reg::ZR, rn, Shift::Lsl, 0)
        }
    }

    // ---- add/sub ----

    /// `add/adds/sub/subs rd, rn, #imm` with `imm` < 4096, or a multiple of 4096 below 2^24
    /// (`lsl #12`). `rn` may be `sp`; `rd` may be `sp` unless flags are set (then 31 is `zr`).
    pub fn addsub_imm(&mut self, op: AddSub, w: Width, rd: Reg, rn: Reg, imm: u64) -> Result<(), EncError> {
        let rd = if op.sets_flags() { zr(rd)? } else { sp(rd)? };
        let rn = sp(rn)?;
        let (imm12, sh) = if imm < 4096 {
            (imm as u32, 0)
        } else if imm & 0xfff == 0 && imm < (1 << 24) {
            ((imm >> 12) as u32, 1)
        } else {
            return Err(EncError::ImmRange { imm: imm as i64, bits: 12 });
        };
        self.word(w.sf() | ((op as u32) << 29) | 0x1100_0000 | (sh << 22) | (imm12 << 10) | (rn << 5) | rd);
        Ok(())
    }

    /// `add/adds/sub/subs rd, rn, rm{, shift #amount}`. With `sp` as `rd` or `rn` this uses the
    /// extended-register form (`uxtx`/`uxtw`, `lsl` 0..=4 only).
    #[allow(clippy::too_many_arguments)]
    pub fn addsub_reg(
        &mut self,
        op: AddSub,
        w: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: Shift,
        amount: u32,
    ) -> Result<(), EncError> {
        let m = zr(rm)?;
        let opb = (op as u32) << 29;
        if rd.is_sp() || rn.is_sp() {
            if shift != Shift::Lsl || amount > 4 {
                return Err(EncError::Invalid("with sp only lsl #0..=4 is encodable"));
            }
            let d = if op.sets_flags() { zr(rd)? } else { sp(rd)? };
            let n = sp(rn)?;
            let option = if w == Width::X { 0b011 } else { 0b010 };
            self.word(
                w.sf() | opb | 0x0b20_0000 | (m << 16) | (option << 13) | (amount << 10) | (n << 5) | d,
            );
            return Ok(());
        }
        if shift == Shift::Ror || amount >= w.bits() {
            return Err(EncError::Invalid("add/sub shift must be lsl/lsr/asr below the width"));
        }
        let (d, n) = (zr(rd)?, zr(rn)?);
        self.word(
            w.sf() | opb | 0x0b00_0000 | ((shift as u32) << 22) | (m << 16) | (amount << 10) | (n << 5) | d,
        );
        Ok(())
    }

    /// `add rd, rn, rm`.
    pub fn add(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.addsub_reg(AddSub::Add, w, rd, rn, rm, Shift::Lsl, 0)
    }

    /// `sub rd, rn, rm`.
    pub fn sub(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.addsub_reg(AddSub::Sub, w, rd, rn, rm, Shift::Lsl, 0)
    }

    /// `cmp rn, rm` (`subs zr, rn, rm`).
    pub fn cmp(&mut self, w: Width, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.addsub_reg(AddSub::Subs, w, Reg::ZR, rn, rm, Shift::Lsl, 0)
    }

    /// `cmp rn, #imm`.
    pub fn cmp_imm(&mut self, w: Width, rn: Reg, imm: u64) -> Result<(), EncError> {
        self.addsub_imm(AddSub::Subs, w, Reg::ZR, rn, imm)
    }

    /// `cmn rn, #imm` (`adds zr, rn, #imm`).
    pub fn cmn_imm(&mut self, w: Width, rn: Reg, imm: u64) -> Result<(), EncError> {
        self.addsub_imm(AddSub::Adds, w, Reg::ZR, rn, imm)
    }

    /// `neg rd, rm` (`sub rd, zr, rm`).
    pub fn neg(&mut self, w: Width, rd: Reg, rm: Reg) -> Result<(), EncError> {
        self.addsub_reg(AddSub::Sub, w, rd, Reg::ZR, rm, Shift::Lsl, 0)
    }

    // ---- logical ----

    /// `and/orr/eor/ands/bic/orn/eon/bics rd, rn, rm{, shift #amount}`.
    #[allow(clippy::too_many_arguments)]
    pub fn logic_reg(
        &mut self,
        op: Logic,
        w: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: Shift,
        amount: u32,
    ) -> Result<(), EncError> {
        let (d, n, m) = (zr(rd)?, zr(rn)?, zr(rm)?);
        if amount >= w.bits() {
            return Err(EncError::ImmRange { imm: i64::from(amount), bits: w.bits().trailing_zeros() });
        }
        let (opc, inv) = op.bits();
        self.word(
            w.sf()
                | (opc << 29)
                | 0x0a00_0000
                | ((shift as u32) << 22)
                | (inv << 21)
                | (m << 16)
                | (amount << 10)
                | (n << 5)
                | d,
        );
        Ok(())
    }

    /// `and/orr/eor/ands rd, rn, #bitmask`. `rd` may be `sp` except for `ands`.
    pub fn logic_imm(&mut self, op: Logic, w: Width, rd: Reg, rn: Reg, imm: u64) -> Result<(), EncError> {
        let (opc, inv) = op.bits();
        if inv != 0 {
            return Err(EncError::Invalid("bic/orn/eon/bics have no immediate form"));
        }
        let d = if op == Logic::Ands { zr(rd)? } else { sp(rd)? };
        let n = zr(rn)?;
        let (nb, immr, imms) =
            encode_logical(imm, w.bits()).ok_or(EncError::ImmRange { imm: imm as i64, bits: w.bits() })?;
        self.word(
            w.sf() | (opc << 29) | 0x1200_0000 | (nb << 22) | (immr << 16) | (imms << 10) | (n << 5) | d,
        );
        Ok(())
    }

    /// `mvn rd, rm` (`orn rd, zr, rm`).
    pub fn mvn(&mut self, w: Width, rd: Reg, rm: Reg) -> Result<(), EncError> {
        self.logic_reg(Logic::Orn, w, rd, Reg::ZR, rm, Shift::Lsl, 0)
    }

    /// `tst rn, rm` (`ands zr, rn, rm`).
    pub fn tst(&mut self, w: Width, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.logic_reg(Logic::Ands, w, Reg::ZR, rn, rm, Shift::Lsl, 0)
    }

    /// `tst rn, #bitmask`.
    pub fn tst_imm(&mut self, w: Width, rn: Reg, imm: u64) -> Result<(), EncError> {
        self.logic_imm(Logic::Ands, w, Reg::ZR, rn, imm)
    }

    // ---- shifts and bitfields ----

    /// `lslv/lsrv/asrv/rorv rd, rn, rm` (count taken modulo the width).
    pub fn shift_reg(&mut self, op: Shift, w: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        let (d, n, m) = (zr(rd)?, zr(rn)?, zr(rm)?);
        self.word(w.sf() | 0x1ac0_2000 | (m << 16) | ((op as u32) << 10) | (n << 5) | d);
        Ok(())
    }

    fn bitfield(
        &mut self,
        opc: u32,
        w: Width,
        rd: Reg,
        rn: Reg,
        immr: u32,
        imms: u32,
    ) -> Result<(), EncError> {
        let (d, n) = (zr(rd)?, zr(rn)?);
        if immr >= w.bits() || imms >= w.bits() {
            return Err(EncError::ImmRange {
                imm: i64::from(immr.max(imms)),
                bits: w.bits().trailing_zeros(),
            });
        }
        let nb = u32::from(w == Width::X) << 22;
        self.word(w.sf() | (opc << 29) | 0x1300_0000 | nb | (immr << 16) | (imms << 10) | (n << 5) | d);
        Ok(())
    }

    /// `sbfm rd, rn, #immr, #imms`.
    pub fn sbfm(&mut self, w: Width, rd: Reg, rn: Reg, immr: u32, imms: u32) -> Result<(), EncError> {
        self.bitfield(0, w, rd, rn, immr, imms)
    }

    /// `ubfm rd, rn, #immr, #imms`.
    pub fn ubfm(&mut self, w: Width, rd: Reg, rn: Reg, immr: u32, imms: u32) -> Result<(), EncError> {
        self.bitfield(2, w, rd, rn, immr, imms)
    }

    /// `lsl/lsr/asr/ror rd, rn, #amount` (`ubfm`/`sbfm`/`extr` aliases).
    pub fn shift_imm(&mut self, op: Shift, w: Width, rd: Reg, rn: Reg, amount: u32) -> Result<(), EncError> {
        let bits = w.bits();
        if amount >= bits {
            return Err(EncError::ImmRange { imm: i64::from(amount), bits: bits.trailing_zeros() });
        }
        match op {
            Shift::Lsl => self.ubfm(w, rd, rn, (bits - amount) % bits, bits - 1 - amount),
            Shift::Lsr => self.ubfm(w, rd, rn, amount, bits - 1),
            Shift::Asr => self.sbfm(w, rd, rn, amount, bits - 1),
            Shift::Ror => {
                let (d, n) = (zr(rd)?, zr(rn)?);
                let nb = if w == Width::X { 0x0040_0000 } else { 0 };
                self.word(w.sf() | 0x1380_0000 | nb | (n << 16) | (amount << 10) | (n << 5) | d);
                Ok(())
            }
        }
    }

    /// `sxtb/sxth/sxtw rd, rn` (`sxtw` needs `X`).
    pub fn sxt(&mut self, w: Width, rd: Reg, rn: Reg, from: Ext) -> Result<(), EncError> {
        let top = match from {
            Ext::B => 7,
            Ext::H => 15,
            Ext::W if w == Width::X => 31,
            Ext::W => return Err(EncError::Invalid("sxtw needs a 64-bit destination")),
        };
        self.sbfm(w, rd, rn, 0, top)
    }

    /// `uxtb/uxth rd, rn` (32-bit form; the write zero-extends to 64 bits). From `W`, use
    /// `mov wd, wn`.
    pub fn uxt(&mut self, rd: Reg, rn: Reg, from: Ext) -> Result<(), EncError> {
        match from {
            Ext::B => self.ubfm(Width::W, rd, rn, 0, 7),
            Ext::H => self.ubfm(Width::W, rd, rn, 0, 15),
            Ext::W => self.mov(Width::W, rd, rn),
        }
    }

    // ---- multiply / divide ----

    fn dp3(&mut self, base: u32, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> Result<(), EncError> {
        let (d, n, m, a) = (zr(rd)?, zr(rn)?, zr(rm)?, zr(ra)?);
        self.word(base | (m << 16) | (a << 10) | (n << 5) | d);
        Ok(())
    }

    /// `madd rd, rn, rm, ra` (`ra + rn * rm`).
    pub fn madd(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> Result<(), EncError> {
        self.dp3(w.sf() | 0x1b00_0000, rd, rn, rm, ra)
    }

    /// `msub rd, rn, rm, ra` (`ra - rn * rm`; remainder after division).
    pub fn msub(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> Result<(), EncError> {
        self.dp3(w.sf() | 0x1b00_8000, rd, rn, rm, ra)
    }

    /// `mul rd, rn, rm`.
    pub fn mul(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.madd(w, rd, rn, rm, Reg::ZR)
    }

    /// `smulh xd, xn, xm` (high 64 bits of the signed 128-bit product).
    pub fn smulh(&mut self, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.dp3(0x9b40_0000, rd, rn, rm, Reg::ZR)
    }

    /// `umulh xd, xn, xm`.
    pub fn umulh(&mut self, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.dp3(0x9bc0_0000, rd, rn, rm, Reg::ZR)
    }

    /// `smull xd, wn, wm` (64-bit product of 32-bit signed operands).
    pub fn smull(&mut self, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.dp3(0x9b20_0000, rd, rn, rm, Reg::ZR)
    }

    /// `umull xd, wn, wm`.
    pub fn umull(&mut self, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.dp3(0x9ba0_0000, rd, rn, rm, Reg::ZR)
    }

    fn dp2(&mut self, base: u32, w: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        let (d, n, m) = (zr(rd)?, zr(rn)?, zr(rm)?);
        self.word(w.sf() | base | (m << 16) | (n << 5) | d);
        Ok(())
    }

    /// `udiv rd, rn, rm` (division by zero yields 0; the backend traps before).
    pub fn udiv(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.dp2(0x1ac0_0800, w, rd, rn, rm)
    }

    /// `sdiv rd, rn, rm`.
    pub fn sdiv(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<(), EncError> {
        self.dp2(0x1ac0_0c00, w, rd, rn, rm)
    }

    // ---- conditional ----

    fn condsel(&mut self, base: u32, w: Width, rd: Reg, rn: Reg, rm: Reg, c: Cond) -> Result<(), EncError> {
        let (d, n, m) = (zr(rd)?, zr(rn)?, zr(rm)?);
        self.word(w.sf() | base | (m << 16) | (c.code() << 12) | (n << 5) | d);
        Ok(())
    }

    /// `csel rd, rn, rm, c`.
    pub fn csel(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg, c: Cond) -> Result<(), EncError> {
        self.condsel(0x1a80_0000, w, rd, rn, rm, c)
    }

    /// `csinc rd, rn, rm, c`.
    pub fn csinc(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg, c: Cond) -> Result<(), EncError> {
        self.condsel(0x1a80_0400, w, rd, rn, rm, c)
    }

    /// `csinv rd, rn, rm, c`.
    pub fn csinv(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg, c: Cond) -> Result<(), EncError> {
        self.condsel(0x5a80_0000, w, rd, rn, rm, c)
    }

    /// `csneg rd, rn, rm, c`.
    pub fn csneg(&mut self, w: Width, rd: Reg, rn: Reg, rm: Reg, c: Cond) -> Result<(), EncError> {
        self.condsel(0x5a80_0400, w, rd, rn, rm, c)
    }

    /// `cset rd, c` (1 if `c`, else 0). `al`/`nv` are rejected.
    pub fn cset(&mut self, w: Width, rd: Reg, c: Cond) -> Result<(), EncError> {
        if matches!(c, Cond::Al | Cond::Nv) {
            return Err(EncError::Invalid("cset needs a condition other than al/nv"));
        }
        self.csinc(w, rd, Reg::ZR, Reg::ZR, c.negate())
    }

    /// `csetm rd, c` (all ones if `c`, else 0).
    pub fn csetm(&mut self, w: Width, rd: Reg, c: Cond) -> Result<(), EncError> {
        if matches!(c, Cond::Al | Cond::Nv) {
            return Err(EncError::Invalid("csetm needs a condition other than al/nv"));
        }
        self.csinv(w, rd, Reg::ZR, Reg::ZR, c.negate())
    }

    fn ccmp_any(
        &mut self,
        base: u32,
        w: Width,
        rn: Reg,
        op2: u32,
        nzcv: u32,
        c: Cond,
    ) -> Result<(), EncError> {
        let n = zr(rn)?;
        range(i64::from(nzcv), 0, 15, 4)?;
        self.word(w.sf() | base | (op2 << 16) | (c.code() << 12) | (n << 5) | nzcv);
        Ok(())
    }

    /// `ccmp rn, rm, #nzcv, c` (flags = `c` ? compare(rn, rm) : nzcv).
    pub fn ccmp(&mut self, w: Width, rn: Reg, rm: Reg, nzcv: u32, c: Cond) -> Result<(), EncError> {
        let m = zr(rm)?;
        self.ccmp_any(0x7a40_0000, w, rn, m, nzcv, c)
    }

    /// `ccmp rn, #imm5, #nzcv, c`.
    pub fn ccmp_imm(&mut self, w: Width, rn: Reg, imm5: u32, nzcv: u32, c: Cond) -> Result<(), EncError> {
        range(i64::from(imm5), 0, 31, 5)?;
        self.ccmp_any(0x7a40_0800, w, rn, imm5, nzcv, c)
    }

    /// `ccmn rn, rm, #nzcv, c`.
    pub fn ccmn(&mut self, w: Width, rn: Reg, rm: Reg, nzcv: u32, c: Cond) -> Result<(), EncError> {
        let m = zr(rm)?;
        self.ccmp_any(0x3a40_0000, w, rn, m, nzcv, c)
    }

    // ---- branches ----

    /// `b label` (±128 MiB).
    pub fn b(&mut self, l: Label) -> Result<(), EncError> {
        self.word_fix(0x1400_0000, l, FixKind::B26)
    }

    /// `bl label`.
    pub fn bl(&mut self, l: Label) -> Result<(), EncError> {
        self.word_fix(0x9400_0000, l, FixKind::B26)
    }

    /// A conditional branch: `w` with its offset field empty, `inv` the same branch with the
    /// opposite condition. In long-branch mode it is `inv` over the next word, then `b label`.
    fn cond_branch(&mut self, w: u32, inv: u32, l: Label, kind: FixKind) -> Result<(), EncError> {
        if !self.long_branches {
            return self.word_fix(w, l, kind);
        }
        self.check_label(l)?;
        // imm19 and imm14 both start at bit 5: an offset of two words skips the `b`
        self.word(inv | (2 << 5));
        self.b(l)
    }

    /// `b.c label` (±1 MiB; any distance in long-branch mode).
    pub fn b_cond(&mut self, c: Cond, l: Label) -> Result<(), EncError> {
        if self.long_branches && matches!(c, Cond::Al | Cond::Nv) {
            return self.b(l);
        }
        self.cond_branch(0x5400_0000 | c.code(), 0x5400_0000 | c.negate().code(), l, FixKind::B19)
    }

    /// `cbz rt, label`.
    pub fn cbz(&mut self, w: Width, rt: Reg, l: Label) -> Result<(), EncError> {
        let t = zr(rt)?;
        self.cond_branch(w.sf() | 0x3400_0000 | t, w.sf() | 0x3500_0000 | t, l, FixKind::B19)
    }

    /// `cbnz rt, label`.
    pub fn cbnz(&mut self, w: Width, rt: Reg, l: Label) -> Result<(), EncError> {
        let t = zr(rt)?;
        self.cond_branch(w.sf() | 0x3500_0000 | t, w.sf() | 0x3400_0000 | t, l, FixKind::B19)
    }

    fn tb(&mut self, base: u32, rt: Reg, bit: u32, l: Label) -> Result<(), EncError> {
        let t = zr(rt)?;
        range(i64::from(bit), 0, 63, 6)?;
        let w = ((bit >> 5) << 31) | ((bit & 31) << 19) | t;
        // tbz (0x36..) and tbnz (0x37..) differ in bit 24
        self.cond_branch(w | base, w | (base ^ (1 << 24)), l, FixKind::B14)
    }

    /// `tbz rt, #bit, label` (±32 KiB).
    pub fn tbz(&mut self, rt: Reg, bit: u32, l: Label) -> Result<(), EncError> {
        self.tb(0x3600_0000, rt, bit, l)
    }

    /// `tbnz rt, #bit, label`.
    pub fn tbnz(&mut self, rt: Reg, bit: u32, l: Label) -> Result<(), EncError> {
        self.tb(0x3700_0000, rt, bit, l)
    }

    /// `bl sym` ([`RelocKind::A64Branch26`]).
    pub fn bl_sym(&mut self, sym: SymbolId) {
        self.word_reloc(0x9400_0000, RelocKind::A64Branch26, sym, 0);
    }

    /// `b sym` (tail call).
    pub fn b_sym(&mut self, sym: SymbolId) {
        self.word_reloc(0x1400_0000, RelocKind::A64Branch26, sym, 0);
    }

    /// `br rn`.
    pub fn br(&mut self, rn: Reg) -> Result<(), EncError> {
        let n = gpr(rn)?;
        self.word(0xd61f_0000 | (n << 5));
        Ok(())
    }

    /// `blr rn`.
    pub fn blr(&mut self, rn: Reg) -> Result<(), EncError> {
        let n = gpr(rn)?;
        self.word(0xd63f_0000 | (n << 5));
        Ok(())
    }

    /// `ret` (through `x30`).
    pub fn ret(&mut self) {
        self.word(0xd65f_03c0);
    }

    /// `ret rn`.
    pub fn ret_reg(&mut self, rn: Reg) -> Result<(), EncError> {
        let n = gpr(rn)?;
        self.word(0xd65f_0000 | (n << 5));
        Ok(())
    }

    // ---- addresses ----

    /// `adr rd, label` (±1 MiB).
    pub fn adr(&mut self, rd: Reg, l: Label) -> Result<(), EncError> {
        let d = zr(rd)?;
        self.word_fix(0x1000_0000 | d, l, FixKind::Adr21)
    }

    /// `adrp rd, sym` (4 KiB page of `sym + addend`).
    pub fn adrp_sym(&mut self, rd: Reg, sym: SymbolId, addend: i64) -> Result<(), EncError> {
        let d = zr(rd)?;
        self.word_reloc(0x9000_0000 | d, RelocKind::A64AdrPage21, sym, addend);
        Ok(())
    }

    /// `add xd, xn, :lo12:sym`.
    pub fn add_lo12_sym(&mut self, rd: Reg, rn: Reg, sym: SymbolId, addend: i64) -> Result<(), EncError> {
        let (d, n) = (sp(rd)?, sp(rn)?);
        self.word_reloc(0x9100_0000 | (n << 5) | d, RelocKind::A64AddLo12, sym, addend);
        Ok(())
    }

    /// `adrp rd, sym; add rd, rd, :lo12:sym` (address of a symbol, ±4 GiB).
    pub fn addr_of_sym(&mut self, rd: Reg, sym: SymbolId, addend: i64) -> Result<(), EncError> {
        if rd.is_sp() || rd.is_zr() {
            return Err(EncError::Invalid("needs x0..x30"));
        }
        self.adrp_sym(rd, sym, addend)?;
        self.add_lo12_sym(rd, rd, sym, addend)
    }

    /// `ldr rt, [rn, :lo12:sym]` (after `adrp rn, sym`); `sym + addend` must be aligned to `sz`.
    pub fn ldr_lo12_sym(
        &mut self,
        sz: MemSize,
        rt: Reg,
        rn: Reg,
        sym: SymbolId,
        addend: i64,
    ) -> Result<(), EncError> {
        let (t, n) = (zr(rt)?, sp(rn)?);
        let s = sz as u32;
        let w = (s << 30) | 0x3940_0000 | (n << 5) | t;
        self.word_reloc(w, RelocKind::A64LdstLo12 { shift: s as u8 }, sym, addend);
        Ok(())
    }

    // ---- loads and stores ----

    fn ls(&mut self, c: Ls, rt: u32, wb_check: Option<Reg>, addr: Addr) -> Result<(), EncError> {
        let base = (c.size << 30) | (7 << 27) | (c.v << 26) | (c.opc << 22) | rt;
        let wb = |rn: Reg| -> Result<(), EncError> {
            if let Some(t) = wb_check
                && !rn.is_sp()
                && t.num() == rn.num()
                && !t.is_zr()
            {
                return Err(EncError::Invalid("writeback with the base as transfer register"));
            }
            Ok(())
        };
        let word = match addr {
            Addr::Off(rn, off) => {
                let n = sp(rn)?;
                let unit = 1i64 << c.scale;
                if off >= 0 && off % unit == 0 && off / unit < 4096 {
                    base | (1 << 24) | (((off / unit) as u32) << 10) | (n << 5)
                } else if (-256..=255).contains(&off) {
                    base | (((off as u32) & 0x1ff) << 12) | (n << 5)
                } else {
                    return Err(EncError::ImmRange { imm: off, bits: 12 });
                }
            }
            Addr::Pre(rn, off) | Addr::Post(rn, off) => {
                let n = sp(rn)?;
                range(off, -256, 255, 9)?;
                wb(rn)?;
                let mode = if matches!(addr, Addr::Pre(..)) { 3 } else { 1 };
                base | (((off as u32) & 0x1ff) << 12) | (mode << 10) | (n << 5)
            }
            Addr::Idx { base: rn, index, ext, scaled } => {
                let n = sp(rn)?;
                let m = zr(index)?;
                base | (1 << 21)
                    | (m << 16)
                    | (ext.option() << 13)
                    | (u32::from(scaled) << 12)
                    | (2 << 10)
                    | (n << 5)
            }
        };
        self.word(word);
        Ok(())
    }

    /// `ldrb/ldrh/ldr wt/ldr xt` (zero-extending).
    pub fn ldr(&mut self, sz: MemSize, rt: Reg, addr: Addr) -> Result<(), EncError> {
        let t = zr(rt)?;
        let s = sz as u32;
        self.ls(Ls { size: s, v: 0, opc: 1, scale: s }, t, Some(rt), addr)
    }

    /// `ldrsb/ldrsh/ldrsw` into a `w` (B, H) or `x` (B, H, W) register.
    pub fn ldrs(&mut self, sz: MemSize, w: Width, rt: Reg, addr: Addr) -> Result<(), EncError> {
        let t = zr(rt)?;
        if sz == MemSize::X || (sz == MemSize::W && w == Width::W) {
            return Err(EncError::Invalid("sign-extending load must widen"));
        }
        let s = sz as u32;
        let opc = if w == Width::X { 2 } else { 3 };
        self.ls(Ls { size: s, v: 0, opc, scale: s }, t, Some(rt), addr)
    }

    /// `strb/strh/str wt/str xt`.
    pub fn str(&mut self, sz: MemSize, rt: Reg, addr: Addr) -> Result<(), EncError> {
        let t = zr(rt)?;
        let s = sz as u32;
        self.ls(Ls { size: s, v: 0, opc: 0, scale: s }, t, Some(rt), addr)
    }

    fn fp_ls(vs: VecSize, load: bool) -> Ls {
        let l = u32::from(load);
        match vs {
            VecSize::S => Ls { size: 2, v: 1, opc: l, scale: 2 },
            VecSize::D => Ls { size: 3, v: 1, opc: l, scale: 3 },
            VecSize::Q => Ls { size: 0, v: 1, opc: 2 | l, scale: 4 },
        }
    }

    /// `ldr st/dt/qt`.
    pub fn ldr_fp(&mut self, vs: VecSize, vt: VReg, addr: Addr) -> Result<(), EncError> {
        self.ls(Asm::fp_ls(vs, true), v(vt), None, addr)
    }

    /// `str st/dt/qt`.
    pub fn str_fp(&mut self, vs: VecSize, vt: VReg, addr: Addr) -> Result<(), EncError> {
        self.ls(Asm::fp_ls(vs, false), v(vt), None, addr)
    }

    /// `ldr wt/xt, label` (PC-relative literal, ±1 MiB).
    pub fn ldr_lit(&mut self, w: Width, rt: Reg, l: Label) -> Result<(), EncError> {
        let t = zr(rt)?;
        let opc = if w == Width::X { 0x5800_0000 } else { 0x1800_0000 };
        self.word_fix(opc | t, l, FixKind::B19)
    }

    /// `ldr st/dt/qt, label`.
    pub fn ldr_fp_lit(&mut self, vs: VecSize, vt: VReg, l: Label) -> Result<(), EncError> {
        let opc = match vs {
            VecSize::S => 0x1c00_0000,
            VecSize::D => 0x5c00_0000,
            VecSize::Q => 0x9c00_0000,
        };
        self.word_fix(opc | v(vt), l, FixKind::B19)
    }

    #[allow(clippy::too_many_arguments)]
    fn pair(
        &mut self,
        opc: u32,
        vbit: u32,
        scale: u32,
        load: bool,
        t1: u32,
        t2: u32,
        gpr_regs: Option<(Reg, Reg)>,
        addr: Addr,
    ) -> Result<(), EncError> {
        let (rn, off, mode) = match addr {
            Addr::Off(rn, off) => (rn, off, 2),
            Addr::Pre(rn, off) => (rn, off, 3),
            Addr::Post(rn, off) => (rn, off, 1),
            Addr::Idx { .. } => return Err(EncError::Invalid("ldp/stp take no register offset")),
        };
        let n = sp(rn)?;
        let unit = 1i64 << scale;
        if off % unit != 0 {
            return Err(EncError::ImmRange { imm: off, bits: 7 });
        }
        range(off / unit, -64, 63, 7)?;
        if load && t1 == t2 {
            return Err(EncError::Invalid("ldp with the same register twice"));
        }
        if mode != 2
            && let Some((a, b)) = gpr_regs
            && !rn.is_sp()
            && ((a.num() == rn.num() && !a.is_zr()) || (b.num() == rn.num() && !b.is_zr()))
        {
            return Err(EncError::Invalid("writeback with the base as transfer register"));
        }
        let imm7 = ((off / unit) as u32) & 0x7f;
        self.word(
            (opc << 30)
                | (5 << 27)
                | (vbit << 26)
                | (mode << 23)
                | (u32::from(load) << 22)
                | (imm7 << 15)
                | (t2 << 10)
                | (n << 5)
                | t1,
        );
        Ok(())
    }

    /// `ldp rt1, rt2, addr` (offset a multiple of the size, scaled 7-bit signed).
    pub fn ldp(&mut self, w: Width, rt1: Reg, rt2: Reg, addr: Addr) -> Result<(), EncError> {
        let (a, b) = (zr(rt1)?, zr(rt2)?);
        let (opc, scale) = if w == Width::X { (2, 3) } else { (0, 2) };
        self.pair(opc, 0, scale, true, a, b, Some((rt1, rt2)), addr)
    }

    /// `stp rt1, rt2, addr`.
    pub fn stp(&mut self, w: Width, rt1: Reg, rt2: Reg, addr: Addr) -> Result<(), EncError> {
        let (a, b) = (zr(rt1)?, zr(rt2)?);
        let (opc, scale) = if w == Width::X { (2, 3) } else { (0, 2) };
        self.pair(opc, 0, scale, false, a, b, Some((rt1, rt2)), addr)
    }

    fn fp_pair_class(vs: VecSize) -> u32 {
        match vs {
            VecSize::S => 0,
            VecSize::D => 1,
            VecSize::Q => 2,
        }
    }

    /// `ldp st1/dt1/qt1, ..., addr`.
    pub fn ldp_fp(&mut self, vs: VecSize, vt1: VReg, vt2: VReg, addr: Addr) -> Result<(), EncError> {
        self.pair(Asm::fp_pair_class(vs), 1, vs.scale(), true, v(vt1), v(vt2), None, addr)
    }

    /// `stp st1/dt1/qt1, ..., addr`.
    pub fn stp_fp(&mut self, vs: VecSize, vt1: VReg, vt2: VReg, addr: Addr) -> Result<(), EncError> {
        self.pair(Asm::fp_pair_class(vs), 1, vs.scale(), false, v(vt1), v(vt2), None, addr)
    }

    // ---- system ----

    /// `svc #imm16` (`svc #0` is the Linux system call).
    pub fn svc(&mut self, imm16: u16) {
        self.word(0xd400_0001 | (u32::from(imm16) << 5));
    }

    /// `brk #imm16`.
    pub fn brk(&mut self, imm16: u16) {
        self.word(0xd420_0000 | (u32::from(imm16) << 5));
    }

    /// `udf #imm16` (permanently undefined).
    pub fn udf(&mut self, imm16: u16) {
        self.word(u32::from(imm16));
    }

    pub fn nop(&mut self) {
        self.word(0xd503_201f);
    }

    // ---- scalar floating point ----

    /// `fmov vd, vn`.
    pub fn fmov(&mut self, fp: Fp, vd: VReg, vn: VReg) {
        self.word(0x1e20_4000 | fp.ftype() | (v(vn) << 5) | v(vd));
    }

    /// `fmov wd, sn` / `fmov xd, dn` (bit copy).
    pub fn fmov_to_gpr(&mut self, fp: Fp, rd: Reg, vn: VReg) -> Result<(), EncError> {
        let d = zr(rd)?;
        let base = if fp == Fp::S { 0x1e26_0000 } else { 0x9e66_0000 };
        self.word(base | (v(vn) << 5) | d);
        Ok(())
    }

    /// `fmov sd, wn` / `fmov dd, xn` (bit copy).
    pub fn fmov_from_gpr(&mut self, fp: Fp, vd: VReg, rn: Reg) -> Result<(), EncError> {
        let n = zr(rn)?;
        let base = if fp == Fp::S { 0x1e27_0000 } else { 0x9e67_0000 };
        self.word(base | (n << 5) | v(vd));
        Ok(())
    }

    /// `fmov vd, #imm` for `±(16 + m)/16 × 2^e` (m in 0..16, e in -3..=4).
    pub fn fmov_imm(&mut self, fp: Fp, vd: VReg, val: f64) -> Result<(), EncError> {
        let imm8 = encode_fp8(val).ok_or(EncError::ImmRange { imm: val as i64, bits: 8 })?;
        self.word(0x1e20_1000 | fp.ftype() | (imm8 << 13) | v(vd));
        Ok(())
    }

    /// `fadd/fsub/fmul/fdiv/fmax/fmin/fmaxnm/fminnm/fnmul vd, vn, vm`.
    pub fn fp_bin(&mut self, op: FpBin, fp: Fp, vd: VReg, vn: VReg, vm: VReg) {
        self.word(0x1e20_0800 | fp.ftype() | (v(vm) << 16) | ((op as u32) << 12) | (v(vn) << 5) | v(vd));
    }

    /// `fabs/fneg/fsqrt/frint* vd, vn`.
    pub fn fp_un(&mut self, op: FpUn, fp: Fp, vd: VReg, vn: VReg) {
        self.word(0x1e20_4000 | fp.ftype() | ((op as u32) << 15) | (v(vn) << 5) | v(vd));
    }

    /// `fcvt vd, vn` between single and double precision.
    pub fn fcvt(&mut self, to: Fp, vd: VReg, from: Fp, vn: VReg) -> Result<(), EncError> {
        if to == from {
            return Err(EncError::Invalid("fcvt needs different precisions"));
        }
        let opc = if to == Fp::S { 4 } else { 5 };
        self.word(0x1e20_4000 | from.ftype() | (opc << 15) | (v(vn) << 5) | v(vd));
        Ok(())
    }

    /// `fcmp vn, vm` (unordered sets C and V).
    pub fn fcmp(&mut self, fp: Fp, vn: VReg, vm: VReg) {
        self.word(0x1e20_2000 | fp.ftype() | (v(vm) << 16) | (v(vn) << 5));
    }

    /// `fcmp vn, #0.0`.
    pub fn fcmp_zero(&mut self, fp: Fp, vn: VReg) {
        self.word(0x1e20_2008 | fp.ftype() | (v(vn) << 5));
    }

    /// `fcmpe vn, vm` (signalling).
    pub fn fcmpe(&mut self, fp: Fp, vn: VReg, vm: VReg) {
        self.word(0x1e20_2010 | fp.ftype() | (v(vm) << 16) | (v(vn) << 5));
    }

    /// `fcsel vd, vn, vm, c`.
    pub fn fcsel(&mut self, fp: Fp, vd: VReg, vn: VReg, vm: VReg, c: Cond) {
        self.word(0x1e20_0c00 | fp.ftype() | (v(vm) << 16) | (c.code() << 12) | (v(vn) << 5) | v(vd));
    }

    /// `fmadd/fmsub/fnmadd/fnmsub vd, vn, vm, va`.
    pub fn fma(&mut self, op: FpFma, fp: Fp, vd: VReg, vn: VReg, vm: VReg, va: VReg) {
        let (o1, o0) = match op {
            FpFma::Madd => (0, 0),
            FpFma::Msub => (0, 1),
            FpFma::Nmadd => (1, 0),
            FpFma::Nmsub => (1, 1),
        };
        self.word(
            0x1f00_0000
                | fp.ftype()
                | (o1 << 21)
                | (v(vm) << 16)
                | (o0 << 15)
                | (v(va) << 10)
                | (v(vn) << 5)
                | v(vd),
        );
    }

    /// `scvtf vd, wn/xn` (signed integer → float).
    pub fn scvtf(&mut self, fp: Fp, vd: VReg, w: Width, rn: Reg) -> Result<(), EncError> {
        let n = zr(rn)?;
        self.word(w.sf() | 0x1e22_0000 | fp.ftype() | (n << 5) | v(vd));
        Ok(())
    }

    /// `ucvtf vd, wn/xn` (unsigned integer → float).
    pub fn ucvtf(&mut self, fp: Fp, vd: VReg, w: Width, rn: Reg) -> Result<(), EncError> {
        let n = zr(rn)?;
        self.word(w.sf() | 0x1e23_0000 | fp.ftype() | (n << 5) | v(vd));
        Ok(())
    }

    /// `fcvt{n,a,p,m,z}{s,u} wd/xd, vn` (float → integer, saturating).
    pub fn fcvt_int(&mut self, rnd: FcvtRound, w: Width, rd: Reg, fp: Fp, vn: VReg) -> Result<(), EncError> {
        let d = zr(rd)?;
        let (rmode, opcode) = rnd.bits();
        self.word(w.sf() | 0x1e20_0000 | fp.ftype() | (rmode << 19) | (opcode << 16) | (v(vn) << 5) | d);
        Ok(())
    }

    /// `fcvtzs wd/xd, vn` (truncating, the `as` conversion).
    pub fn fcvtzs(&mut self, w: Width, rd: Reg, fp: Fp, vn: VReg) -> Result<(), EncError> {
        self.fcvt_int(FcvtRound::Zs, w, rd, fp, vn)
    }

    /// `fcvtzu wd/xd, vn`.
    pub fn fcvtzu(&mut self, w: Width, rd: Reg, fp: Fp, vn: VReg) -> Result<(), EncError> {
        self.fcvt_int(FcvtRound::Zu, w, rd, fp, vn)
    }
}

// ---- bit counting (additions for the debug backend) ----

impl Asm {
    fn dp1(&mut self, opcode: u32, w: Width, rd: Reg, rn: Reg) -> Result<(), EncError> {
        let (d, n) = (zr(rd)?, zr(rn)?);
        self.word(w.sf() | 0x5ac0_0000 | (opcode << 10) | (n << 5) | d);
        Ok(())
    }

    /// `rbit rd, rn` (reverse the bit order).
    pub fn rbit(&mut self, w: Width, rd: Reg, rn: Reg) -> Result<(), EncError> {
        self.dp1(0, w, rd, rn)
    }

    /// `clz rd, rn` (count leading zeros; the width for zero).
    pub fn clz(&mut self, w: Width, rd: Reg, rn: Reg) -> Result<(), EncError> {
        self.dp1(4, w, rd, rn)
    }

    /// `cnt vd.8b, vn.8b` (population count per byte).
    pub fn cnt_8b(&mut self, vd: VReg, vn: VReg) {
        self.word(0x0e20_5800 | (v(vn) << 5) | v(vd));
    }

    /// `addv bd, vn.8b` (sum of the eight bytes).
    pub fn addv_8b(&mut self, vd: VReg, vn: VReg) {
        self.word(0x0e31_b800 | (v(vn) << 5) | v(vd));
    }
}
