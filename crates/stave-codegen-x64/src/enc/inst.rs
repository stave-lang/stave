//! Instruction forms. Layout of each encoding: `[66|F2|F3] [REX] opcode ModRM [SIB] [disp] [imm]`.

use stave_obj::{Reloc, RelocKind, SymbolId};
use stave_target::Feature;

use super::simd::Vex;
use super::simd_ops::{Map, Pp};
use super::{Asm, Cond, EncError, Gpr, Label, Mem, Rm, Scale, Size, Target, VLen, Xmm};

/// Two-operand integer ALU operations (the `/digit` of opcodes 80/81/83).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AluOp {
    Add = 0,
    Or = 1,
    Adc = 2,
    Sbb = 3,
    And = 4,
    Sub = 5,
    Xor = 6,
    Cmp = 7,
}

/// One-operand group F6/F7: `not neg mul imul div idiv` (`mul`/`imul`/`div`/`idiv` use
/// `rdx:rax`; zero `rdx` or sign-extend with [`Asm::cqo`]/[`Asm::cdq`] before dividing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Not = 2,
    Neg = 3,
    Mul = 4,
    Imul = 5,
    Div = 6,
    Idiv = 7,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShiftOp {
    Rol = 0,
    Ror = 1,
    Shl = 4,
    Shr = 5,
    Sar = 7,
}

/// Scalar float precision: `ss` (f32) or `sd` (f64) forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prec {
    F32,
    F64,
}

impl Prec {
    fn scalar_prefix(self) -> u8 {
        match self {
            Prec::F32 => 0xf3,
            Prec::F64 => 0xf2,
        }
    }

    /// Packed-form prefix (`ps` none, `pd` 66).
    fn packed_prefix(self) -> Option<u8> {
        match self {
            Prec::F32 => None,
            Prec::F64 => Some(0x66),
        }
    }
}

/// Scalar SSE arithmetic (`addsd` etc.; opcode byte after `0F`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SseOp {
    Sqrt = 0x51,
    Add = 0x58,
    Mul = 0x59,
    Sub = 0x5c,
    Min = 0x5d,
    Div = 0x5e,
    Max = 0x5f,
}

/// Packed bitwise ops on whole registers (`andps/andpd` …), for sign masks and zeroing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PLogic {
    And = 0x54,
    Andn = 0x55,
    Or = 0x56,
    Xor = 0x57,
}

/// An SSE register or memory operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum XRm {
    X(Xmm),
    M(Mem),
}

impl From<Xmm> for XRm {
    fn from(x: Xmm) -> XRm {
        XRm::X(x)
    }
}

impl From<Mem> for XRm {
    fn from(m: Mem) -> XRm {
        XRm::M(m)
    }
}

#[derive(Clone, Copy)]
pub(super) enum RmEnc {
    Reg(u8),
    Mem(Mem),
    /// Vector-indexed memory (gathers): the index is a vector register number.
    Vsib {
        base: Option<Gpr>,
        index: u8,
        scale: Scale,
        disp: i32,
    },
}

impl From<Rm> for RmEnc {
    fn from(r: Rm) -> RmEnc {
        match r {
            Rm::R(g) => RmEnc::Reg(g.0),
            Rm::M(m) => RmEnc::Mem(m),
        }
    }
}

impl From<XRm> for RmEnc {
    fn from(r: XRm) -> RmEnc {
        match r {
            XRm::X(x) => RmEnc::Reg(x.0),
            XRm::M(m) => RmEnc::Mem(m),
        }
    }
}

/// One encoding: prefix, REX.W, forced REX (byte registers `spl..dil`), opcode, ModRM.reg (a
/// register number or `/digit`), ModRM.rm operand, and the immediate bytes that follow.
pub(super) struct Enc<'a> {
    pub(super) pfx: Option<u8>,
    pub(super) w: bool,
    pub(super) rex8: bool,
    pub(super) opc: &'a [u8],
    pub(super) reg: u8,
    pub(super) rm: RmEnc,
    pub(super) trailing: u8,
}

fn fits_i8(v: i64) -> bool {
    (-128..=127).contains(&v)
}

/// The immediate fits `s` as either a signed or an unsigned value (`Q`: signed 32-bit, sign
/// extended by the hardware).
fn imm_fits(imm: i64, s: Size) -> bool {
    match s {
        Size::B => (-0x80..=0xff).contains(&imm),
        Size::W => (-0x8000..=0xffff).contains(&imm),
        Size::D => (-0x8000_0000..=0xffff_ffff).contains(&imm),
        Size::Q => (-0x8000_0000..=0x7fff_ffff).contains(&imm),
    }
}

fn imm_bytes(s: Size) -> u8 {
    match s {
        Size::B => 1,
        Size::W => 2,
        Size::D | Size::Q => 4,
    }
}

fn byte_reg(s: Size, r: u8) -> bool {
    s == Size::B && (4..8).contains(&r)
}

fn byte_rm(s: Size, rm: &RmEnc) -> bool {
    match rm {
        RmEnc::Reg(r) => byte_reg(s, *r),
        RmEnc::Mem(_) | RmEnc::Vsib { .. } => false,
    }
}

fn size_pfx(s: Size) -> Option<u8> {
    (s == Size::W).then_some(0x66)
}

impl Asm {
    /// The REX/VEX `X` and `B` bits an r/m operand needs; validates the operand.
    pub(super) fn rm_xb(&self, rm: &RmEnc) -> Result<(u8, u8), EncError> {
        Ok(match rm {
            RmEnc::Reg(r) => (0, (r >> 3) & 1),
            RmEnc::Mem(Mem::Sib { base, index, .. }) => {
                if let Some((i, _)) = index
                    && i.0 == 4
                {
                    return Err(EncError::Invalid("rsp cannot be an index register"));
                }
                (index.map_or(0, |(i, _)| (i.0 >> 3) & 1), base.map_or(0, |r| (r.0 >> 3) & 1))
            }
            RmEnc::Mem(Mem::Rip { target, .. }) => {
                if let Target::Label(l) = target {
                    self.check_label(*l)?;
                }
                (0, 0)
            }
            RmEnc::Vsib { base, index, .. } => ((index >> 3) & 1, base.map_or(0, |r| (r.0 >> 3) & 1)),
        })
    }

    pub(super) fn enc(&mut self, e: Enc<'_>) -> Result<(), EncError> {
        let (x, b) = self.rm_xb(&e.rm)?;
        if let Some(p) = e.pfx {
            self.buf.push(p);
        }
        let rex = 0x40 | (u8::from(e.w) << 3) | (((e.reg >> 3) & 1) << 2) | (x << 1) | b;
        if rex != 0x40 || e.rex8 {
            self.buf.push(rex);
        }
        self.buf.extend_from_slice(e.opc);
        self.emit_modrm(e.reg, e.rm, e.trailing);
        Ok(())
    }

    /// ModRM, SIB, and displacement for `reg` and `rm`; `trailing` is the number of immediate
    /// bytes that follow (a RIP-relative displacement counts from the end of the instruction).
    pub(super) fn emit_modrm(&mut self, reg: u8, rm: RmEnc, trailing: u8) {
        let reg3 = (reg & 7) << 3;
        match rm {
            RmEnc::Reg(r) => self.buf.push(0xc0 | reg3 | (r & 7)),
            RmEnc::Mem(Mem::Sib { base, index, disp }) => {
                self.emit_sib(reg3, base.map(|r| r.0), index.map(|(i, s)| (i.0, s as u8)), disp)
            }
            RmEnc::Vsib { base, index, scale, disp } => {
                self.emit_sib(reg3, base.map(|r| r.0), Some((index, scale as u8)), disp)
            }
            RmEnc::Mem(Mem::Rip { target, disp }) => {
                self.buf.push(reg3 | 5);
                let bias = i64::from(disp) - i64::from(trailing);
                match target {
                    Target::Label(l) => self.rel32_label(l, bias),
                    Target::Sym(s) => self.rel32_sym(s, bias - 4),
                }
            }
        }
    }

    fn emit_sib(&mut self, reg3: u8, base: Option<u8>, index: Option<(u8, u8)>, disp: i32) {
        let d = i64::from(disp);
        match base {
            None => {
                let sib = index.map_or(0x25, |(i, s)| (s << 6) | ((i & 7) << 3) | 5);
                self.buf.extend_from_slice(&[reg3 | 4, sib]);
                self.buf.extend_from_slice(&disp.to_le_bytes());
            }
            Some(bs) => {
                let b3 = bs & 7;
                let md = if d == 0 && b3 != 5 {
                    0
                } else if fits_i8(d) {
                    1
                } else {
                    2
                };
                if index.is_some() || b3 == 4 {
                    let sib = match index {
                        Some((i, s)) => (s << 6) | ((i & 7) << 3) | b3,
                        None => 0x20 | b3,
                    };
                    self.buf.extend_from_slice(&[(md << 6) | reg3 | 4, sib]);
                } else {
                    self.buf.push((md << 6) | reg3 | b3);
                }
                match md {
                    1 => self.buf.push(disp as i8 as u8),
                    2 => self.buf.extend_from_slice(&disp.to_le_bytes()),
                    _ => {}
                }
            }
        }
    }

    /// Register-only encodings cannot fail.
    fn enc_r(&mut self, e: Enc<'_>) {
        let _ = self.enc(e);
    }

    pub(super) fn imm(&mut self, v: i64, n: u8) {
        self.buf.extend_from_slice(&v.to_le_bytes()[..n as usize]);
    }

    /// A sized integer instruction: `opc8` for bytes, `opc` otherwise; 66 for words; REX.W for Q.
    #[allow(clippy::too_many_arguments)]
    fn sized(
        &mut self,
        s: Size,
        opc8: &[u8],
        opc: &[u8],
        reg: u8,
        reg_is_gpr: bool,
        rm: RmEnc,
        trailing: u8,
    ) -> Result<(), EncError> {
        let rex8 = byte_rm(s, &rm) || (reg_is_gpr && byte_reg(s, reg));
        let opc = if s == Size::B { opc8 } else { opc };
        self.enc(Enc { pfx: size_pfx(s), w: s == Size::Q, rex8, opc, reg, rm, trailing })
    }

    // ---- moves ----

    /// `mov dst, src`.
    pub fn mov_rr(&mut self, s: Size, dst: Gpr, src: Gpr) {
        let _ = self.sized(s, &[0x88], &[0x89], src.0, true, RmEnc::Reg(dst.0), 0);
    }

    /// `mov dst, [m]` (load).
    pub fn mov_rm(&mut self, s: Size, dst: Gpr, m: Mem) -> Result<(), EncError> {
        self.sized(s, &[0x8a], &[0x8b], dst.0, true, RmEnc::Mem(m), 0)
    }

    /// `mov [m], src` (store).
    pub fn mov_mr(&mut self, s: Size, m: Mem, src: Gpr) -> Result<(), EncError> {
        self.sized(s, &[0x88], &[0x89], src.0, true, RmEnc::Mem(m), 0)
    }

    /// `mov dst, imm`. `Q`: the shortest of `mov r32, imm32` (zero-extending, for 0..2^32),
    /// `mov r64, simm32`, and `movabs r64, imm64`. `B`/`W`/`D`: the value must fit as signed or
    /// unsigned.
    pub fn mov_ri(&mut self, s: Size, dst: Gpr, imm: i64) -> Result<(), EncError> {
        let r = dst.0;
        let rex_b = (r >> 3) & 1;
        match s {
            Size::Q if (0..=0xffff_ffff).contains(&imm) => self.mov_ri(Size::D, dst, imm),
            Size::Q if imm_fits(imm, Size::Q) => {
                self.enc_r(Enc {
                    pfx: None,
                    w: true,
                    rex8: false,
                    opc: &[0xc7],
                    reg: 0,
                    rm: RmEnc::Reg(r),
                    trailing: 4,
                });
                self.imm(imm, 4);
                Ok(())
            }
            Size::Q => {
                self.buf.extend_from_slice(&[0x48 | rex_b, 0xb8 + (r & 7)]);
                self.imm(imm, 8);
                Ok(())
            }
            _ => {
                if !imm_fits(imm, s) {
                    return Err(EncError::ImmRange { imm, bits: s.bits() });
                }
                if let Some(p) = size_pfx(s) {
                    self.buf.push(p);
                }
                if rex_b != 0 || byte_reg(s, r) {
                    self.buf.push(0x40 | rex_b);
                }
                let base = if s == Size::B { 0xb0 } else { 0xb8 };
                self.buf.push(base + (r & 7));
                self.imm(imm, s.bytes());
                Ok(())
            }
        }
    }

    /// `mov [m], imm` (`Q`: sign-extended 32-bit immediate).
    pub fn mov_mi(&mut self, s: Size, m: Mem, imm: i64) -> Result<(), EncError> {
        if !imm_fits(imm, s) {
            return Err(EncError::ImmRange { imm, bits: s.bits().min(32) });
        }
        let n = imm_bytes(s);
        self.sized(s, &[0xc6], &[0xc7], 0, false, RmEnc::Mem(m), n)?;
        self.imm(imm, n);
        Ok(())
    }

    /// `movabs dst, sym + addend` (absolute address, [`RelocKind::Abs64`]).
    pub fn mov_r_sym(&mut self, dst: Gpr, sym: SymbolId, addend: i64) {
        self.buf.extend_from_slice(&[0x48 | ((dst.0 >> 3) & 1), 0xb8 + (dst.0 & 7)]);
        let offset = self.offset();
        self.relocs.push(Reloc { offset, kind: RelocKind::Abs64, sym, addend });
        self.buf.extend_from_slice(&[0; 8]);
    }

    /// `lea dst, [m]` (`D` or `Q`).
    pub fn lea(&mut self, s: Size, dst: Gpr, m: Mem) -> Result<(), EncError> {
        if s < Size::D {
            return Err(EncError::Invalid("lea needs a 32- or 64-bit destination"));
        }
        self.sized(s, &[], &[0x8d], dst.0, true, RmEnc::Mem(m), 0)
    }

    /// `movzx dst, src` from a narrower `from` size. `from = D` with `dst = Q` is `mov r32, r/m32`
    /// (the hardware zero-extends).
    pub fn movzx(&mut self, dst_s: Size, dst: Gpr, from: Size, src: impl Into<Rm>) -> Result<(), EncError> {
        let rm = RmEnc::from(src.into());
        if from >= dst_s || dst_s == Size::B {
            return Err(EncError::Invalid("movzx needs a narrower source"));
        }
        let rex8 = byte_rm(from, &rm);
        let (w, opc): (bool, &[u8]) = match from {
            Size::B => (dst_s == Size::Q, &[0x0f, 0xb6]),
            Size::W => (dst_s == Size::Q, &[0x0f, 0xb7]),
            _ => (false, &[0x8b]),
        };
        let pfx = size_pfx(dst_s);
        self.enc(Enc { pfx, w, rex8, opc, reg: dst.0, rm, trailing: 0 })
    }

    /// `movsx`/`movsxd dst, src` from a narrower `from` size.
    pub fn movsx(&mut self, dst_s: Size, dst: Gpr, from: Size, src: impl Into<Rm>) -> Result<(), EncError> {
        let rm = RmEnc::from(src.into());
        if from >= dst_s || dst_s == Size::B {
            return Err(EncError::Invalid("movsx needs a narrower source"));
        }
        let rex8 = byte_rm(from, &rm);
        let opc: &[u8] = match from {
            Size::B => &[0x0f, 0xbe],
            Size::W => &[0x0f, 0xbf],
            _ => &[0x63],
        };
        let pfx = size_pfx(dst_s);
        self.enc(Enc { pfx, w: dst_s == Size::Q, rex8, opc, reg: dst.0, rm, trailing: 0 })
    }

    /// `cmovcc dst, src` (`W`, `D`, `Q`).
    pub fn cmov(&mut self, c: Cond, s: Size, dst: Gpr, src: impl Into<Rm>) -> Result<(), EncError> {
        if s == Size::B {
            return Err(EncError::Invalid("cmov has no 8-bit form"));
        }
        self.sized(s, &[], &[0x0f, 0x40 + c.code()], dst.0, true, src.into().into(), 0)
    }

    // ---- integer ALU ----

    /// `op dst, src`.
    pub fn alu_rr(&mut self, op: AluOp, s: Size, dst: Gpr, src: Gpr) {
        let o = (op as u8) << 3;
        let _ = self.sized(s, &[o], &[o + 1], src.0, true, RmEnc::Reg(dst.0), 0);
    }

    /// `op dst, [m]`.
    pub fn alu_rm(&mut self, op: AluOp, s: Size, dst: Gpr, m: Mem) -> Result<(), EncError> {
        let o = (op as u8) << 3;
        self.sized(s, &[o + 2], &[o + 3], dst.0, true, RmEnc::Mem(m), 0)
    }

    /// `op [m], src`.
    pub fn alu_mr(&mut self, op: AluOp, s: Size, m: Mem, src: Gpr) -> Result<(), EncError> {
        let o = (op as u8) << 3;
        self.sized(s, &[o], &[o + 1], src.0, true, RmEnc::Mem(m), 0)
    }

    fn alu_i(&mut self, op: AluOp, s: Size, rm: RmEnc, imm: i64) -> Result<(), EncError> {
        if !imm_fits(imm, s) {
            return Err(EncError::ImmRange { imm, bits: s.bits().min(32) });
        }
        let d = op as u8;
        // the value as the hardware sign-extends it from the operand size
        let sext = match s {
            Size::B => imm as i8 as i64,
            Size::W => imm as i16 as i64,
            Size::D => imm as i32 as i64,
            Size::Q => imm,
        };
        if s == Size::B {
            self.sized(s, &[0x80], &[], d, false, rm, 1)?;
            self.imm(imm, 1);
        } else if fits_i8(sext) {
            self.sized(s, &[], &[0x83], d, false, rm, 1)?;
            self.imm(sext, 1);
        } else {
            let n = imm_bytes(s);
            self.sized(s, &[], &[0x81], d, false, rm, n)?;
            self.imm(imm, n);
        }
        Ok(())
    }

    /// `op dst, imm` (imm8 form when the value fits).
    pub fn alu_ri(&mut self, op: AluOp, s: Size, dst: Gpr, imm: i64) -> Result<(), EncError> {
        self.alu_i(op, s, RmEnc::Reg(dst.0), imm)
    }

    /// `op [m], imm`.
    pub fn alu_mi(&mut self, op: AluOp, s: Size, m: Mem, imm: i64) -> Result<(), EncError> {
        self.alu_i(op, s, RmEnc::Mem(m), imm)
    }

    /// `test a, b`.
    pub fn test_rr(&mut self, s: Size, a: Gpr, b: Gpr) {
        let _ = self.sized(s, &[0x84], &[0x85], b.0, true, RmEnc::Reg(a.0), 0);
    }

    /// `test [m], r`.
    pub fn test_mr(&mut self, s: Size, m: Mem, r: Gpr) -> Result<(), EncError> {
        self.sized(s, &[0x84], &[0x85], r.0, true, RmEnc::Mem(m), 0)
    }

    /// `test dst, imm`.
    pub fn test_ri(&mut self, s: Size, dst: impl Into<Rm>, imm: i64) -> Result<(), EncError> {
        if !imm_fits(imm, s) {
            return Err(EncError::ImmRange { imm, bits: s.bits().min(32) });
        }
        let n = imm_bytes(s);
        self.sized(s, &[0xf6], &[0xf7], 0, false, dst.into().into(), n)?;
        self.imm(imm, n);
        Ok(())
    }

    /// `imul dst, src` (truncating two-operand form; `W`, `D`, `Q`).
    pub fn imul_rr(&mut self, s: Size, dst: Gpr, src: impl Into<Rm>) -> Result<(), EncError> {
        if s == Size::B {
            return Err(EncError::Invalid("two-operand imul has no 8-bit form"));
        }
        self.sized(s, &[], &[0x0f, 0xaf], dst.0, true, src.into().into(), 0)
    }

    /// `imul dst, src, imm`.
    pub fn imul_rri(&mut self, s: Size, dst: Gpr, src: impl Into<Rm>, imm: i64) -> Result<(), EncError> {
        if s == Size::B {
            return Err(EncError::Invalid("imul has no 8-bit immediate form"));
        }
        if !imm_fits(imm, s) {
            return Err(EncError::ImmRange { imm, bits: s.bits().min(32) });
        }
        let rm = src.into().into();
        if fits_i8(imm) {
            self.sized(s, &[], &[0x6b], dst.0, true, rm, 1)?;
            self.imm(imm, 1);
        } else {
            let n = imm_bytes(s);
            self.sized(s, &[], &[0x69], dst.0, true, rm, n)?;
            self.imm(imm, n);
        }
        Ok(())
    }

    /// `not/neg/mul/imul/div/idiv src` (group F6/F7).
    pub fn unary(&mut self, op: UnaryOp, s: Size, src: impl Into<Rm>) -> Result<(), EncError> {
        self.sized(s, &[0xf6], &[0xf7], op as u8, false, src.into().into(), 0)
    }

    /// `op dst, imm` (count masked by the hardware; must be below the operand width).
    pub fn shift_ri(&mut self, op: ShiftOp, s: Size, dst: impl Into<Rm>, imm: u8) -> Result<(), EncError> {
        if u32::from(imm) >= s.bits() {
            return Err(EncError::ImmRange { imm: i64::from(imm), bits: s.bits().trailing_zeros() });
        }
        self.sized(s, &[0xc0], &[0xc1], op as u8, false, dst.into().into(), 1)?;
        self.buf.push(imm);
        Ok(())
    }

    /// `op dst, cl`.
    pub fn shift_cl(&mut self, op: ShiftOp, s: Size, dst: impl Into<Rm>) -> Result<(), EncError> {
        self.sized(s, &[0xd2], &[0xd3], op as u8, false, dst.into().into(), 0)
    }

    /// `setcc dst8`.
    pub fn setcc(&mut self, c: Cond, dst: Gpr) {
        self.enc_r(Enc {
            pfx: None,
            w: false,
            rex8: byte_reg(Size::B, dst.0),
            opc: &[0x0f, 0x90 + c.code()],
            reg: 0,
            rm: RmEnc::Reg(dst.0),
            trailing: 0,
        });
    }

    /// `cqo` (sign-extend `rax` into `rdx:rax`).
    pub fn cqo(&mut self) {
        self.buf.extend_from_slice(&[0x48, 0x99]);
    }

    /// `cdq` (sign-extend `eax` into `edx:eax`).
    pub fn cdq(&mut self) {
        self.buf.push(0x99);
    }

    // ---- control flow ----

    /// `jcc rel32` to a label.
    pub fn jcc(&mut self, c: Cond, l: Label) -> Result<(), EncError> {
        self.check_label(l)?;
        self.buf.extend_from_slice(&[0x0f, 0x80 + c.code()]);
        self.rel32_label(l, 0);
        Ok(())
    }

    /// `jmp rel32` to a label.
    pub fn jmp(&mut self, l: Label) -> Result<(), EncError> {
        self.check_label(l)?;
        self.buf.push(0xe9);
        self.rel32_label(l, 0);
        Ok(())
    }

    /// `jmp sym` (tail call).
    pub fn jmp_sym(&mut self, sym: SymbolId) {
        self.buf.push(0xe9);
        self.rel32_sym(sym, -4);
    }

    /// `jmp r/m64` (indirect).
    pub fn jmp_rm(&mut self, t: impl Into<Rm>) -> Result<(), EncError> {
        self.enc(Enc {
            pfx: None,
            w: false,
            rex8: false,
            opc: &[0xff],
            reg: 4,
            rm: t.into().into(),
            trailing: 0,
        })
    }

    /// `call rel32` to a label.
    pub fn call(&mut self, l: Label) -> Result<(), EncError> {
        self.check_label(l)?;
        self.buf.push(0xe8);
        self.rel32_label(l, 0);
        Ok(())
    }

    /// `call sym`.
    pub fn call_sym(&mut self, sym: SymbolId) {
        self.buf.push(0xe8);
        self.rel32_sym(sym, -4);
    }

    /// `call r/m64` (indirect).
    pub fn call_rm(&mut self, t: impl Into<Rm>) -> Result<(), EncError> {
        self.enc(Enc {
            pfx: None,
            w: false,
            rex8: false,
            opc: &[0xff],
            reg: 2,
            rm: t.into().into(),
            trailing: 0,
        })
    }

    pub fn push(&mut self, r: Gpr) {
        if r.0 >= 8 {
            self.buf.push(0x41);
        }
        self.buf.push(0x50 + (r.0 & 7));
    }

    pub fn pop(&mut self, r: Gpr) {
        if r.0 >= 8 {
            self.buf.push(0x41);
        }
        self.buf.push(0x58 + (r.0 & 7));
    }

    /// `push imm` (sign-extended to 64 bits).
    pub fn push_i(&mut self, imm: i32) {
        if fits_i8(i64::from(imm)) {
            self.buf.extend_from_slice(&[0x6a, imm as i8 as u8]);
        } else {
            self.buf.push(0x68);
            self.buf.extend_from_slice(&imm.to_le_bytes());
        }
    }

    pub fn ret(&mut self) {
        self.buf.push(0xc3);
    }

    pub fn leave(&mut self) {
        self.buf.push(0xc9);
    }

    pub fn syscall(&mut self) {
        self.buf.extend_from_slice(&[0x0f, 0x05]);
    }

    pub fn ud2(&mut self) {
        self.buf.extend_from_slice(&[0x0f, 0x0b]);
    }

    pub fn int3(&mut self) {
        self.buf.push(0xcc);
    }

    pub fn nop(&mut self) {
        self.buf.push(0x90);
    }

    pub fn cpuid(&mut self) {
        self.buf.extend_from_slice(&[0x0f, 0xa2]);
    }

    pub fn xgetbv(&mut self) {
        self.buf.extend_from_slice(&[0x0f, 0x01, 0xd0]);
    }

    // ---- SSE2 scalar ----

    /// One scalar SSE instruction `[pfx] 0F opc`, or its VEX form ([`Asm::set_vex_scalars`]).
    /// Where the legacy form keeps or merges into its destination the VEX form has one more
    /// source operand: the destination again.
    fn sse_enc(&mut self, pfx: Option<u8>, w: bool, opc: u8, reg: u8, rm: RmEnc) -> Result<(), EncError> {
        if !self.vex_scalars {
            return self.enc(Enc { pfx, w, rex8: false, opc: &[0x0f, opc], reg, rm, trailing: 0 });
        }
        let pp = match pfx {
            None => Pp::N,
            Some(0x66) => Pp::P66,
            Some(0xf3) => Pp::F3,
            _ => Pp::F2,
        };
        let two_operand = match opc {
            // `movss` / `movsd`: a load replaces the register, a register move merges
            0x10 => matches!(rm, RmEnc::Mem(_)),
            // stores, `movaps`, conversions to an integer, compares, `movd` / `movq`
            0x11 | 0x28 | 0x2c | 0x2d | 0x2e | 0x2f | 0x6e | 0x7e => true,
            _ => false,
        };
        let vvvv = if two_operand { 0 } else { reg };
        self.note(Feature::Avx);
        self.vex(Vex { pp, map: Map::M0F, w, l: VLen::X128, opc, reg, vvvv, rm, trailing: 0 })
    }

    /// `addsd/subsd/mulsd/divsd/minsd/maxsd/sqrtsd dst, src` (or the `ss` forms).
    pub fn sse(&mut self, op: SseOp, p: Prec, dst: Xmm, src: impl Into<XRm>) -> Result<(), EncError> {
        self.sse_enc(Some(p.scalar_prefix()), false, op as u8, dst.0, src.into().into())
    }

    /// `movsd/movss dst, src` (register move or load).
    pub fn movs(&mut self, p: Prec, dst: Xmm, src: impl Into<XRm>) -> Result<(), EncError> {
        self.sse_enc(Some(p.scalar_prefix()), false, 0x10, dst.0, src.into().into())
    }

    /// `movsd/movss [m], src` (store).
    pub fn movs_store(&mut self, p: Prec, m: Mem, src: Xmm) -> Result<(), EncError> {
        self.sse_enc(Some(p.scalar_prefix()), false, 0x11, src.0, RmEnc::Mem(m))
    }

    /// `ucomisd/ucomiss a, b` (unordered compare: ZF PF CF; PF=1 for NaN).
    pub fn ucomis(&mut self, p: Prec, a: Xmm, b: impl Into<XRm>) -> Result<(), EncError> {
        self.sse_enc(p.packed_prefix(), false, 0x2e, a.0, b.into().into())
    }

    /// `comisd/comiss a, b` (signalling compare).
    pub fn comis(&mut self, p: Prec, a: Xmm, b: impl Into<XRm>) -> Result<(), EncError> {
        self.sse_enc(p.packed_prefix(), false, 0x2f, a.0, b.into().into())
    }

    /// `cvtsi2sd/cvtsi2ss dst, src` from a 32- or 64-bit signed integer.
    pub fn cvtsi2s(&mut self, p: Prec, dst: Xmm, s: Size, src: impl Into<Rm>) -> Result<(), EncError> {
        if s < Size::D {
            return Err(EncError::Invalid("cvtsi2s needs a 32- or 64-bit source"));
        }
        self.sse_enc(Some(p.scalar_prefix()), s == Size::Q, 0x2a, dst.0, src.into().into())
    }

    /// `cvttsd2si/cvttss2si dst, src` (truncating) into a 32- or 64-bit register.
    pub fn cvtts2si(&mut self, p: Prec, s: Size, dst: Gpr, src: impl Into<XRm>) -> Result<(), EncError> {
        if s < Size::D {
            return Err(EncError::Invalid("cvtts2si needs a 32- or 64-bit destination"));
        }
        self.sse_enc(Some(p.scalar_prefix()), s == Size::Q, 0x2c, dst.0, src.into().into())
    }

    /// `cvtsd2si/cvtss2si dst, src` (current rounding mode).
    pub fn cvts2si(&mut self, p: Prec, s: Size, dst: Gpr, src: impl Into<XRm>) -> Result<(), EncError> {
        if s < Size::D {
            return Err(EncError::Invalid("cvts2si needs a 32- or 64-bit destination"));
        }
        self.sse_enc(Some(p.scalar_prefix()), s == Size::Q, 0x2d, dst.0, src.into().into())
    }

    /// `cvtsd2ss` (`from = F64`) or `cvtss2sd` (`from = F32`).
    pub fn cvts2s(&mut self, from: Prec, dst: Xmm, src: impl Into<XRm>) -> Result<(), EncError> {
        self.sse_enc(Some(from.scalar_prefix()), false, 0x5a, dst.0, src.into().into())
    }

    /// `movd/movq xmm, r32/r64` (bit copy).
    pub fn mov_xg(&mut self, s: Size, dst: Xmm, src: Gpr) -> Result<(), EncError> {
        if s < Size::D {
            return Err(EncError::Invalid("movd/movq needs a 32- or 64-bit register"));
        }
        self.sse_enc(Some(0x66), s == Size::Q, 0x6e, dst.0, RmEnc::Reg(src.0))
    }

    /// `movd/movq r32/r64, xmm` (bit copy).
    pub fn mov_gx(&mut self, s: Size, dst: Gpr, src: Xmm) -> Result<(), EncError> {
        if s < Size::D {
            return Err(EncError::Invalid("movd/movq needs a 32- or 64-bit register"));
        }
        self.sse_enc(Some(0x66), s == Size::Q, 0x7e, src.0, RmEnc::Reg(dst.0))
    }

    /// `andps/andnps/orps/xorps` (`F32`) or the `pd` forms (`F64`).
    pub fn logic_p(&mut self, op: PLogic, p: Prec, dst: Xmm, src: impl Into<XRm>) -> Result<(), EncError> {
        self.sse_enc(p.packed_prefix(), false, op as u8, dst.0, src.into().into())
    }

    // ---- bit scans (baseline x86-64) ----

    /// `bsr dst, src` (`W`, `D`, `Q`): index of the highest set bit; ZF=1 (and `dst` undefined)
    /// when `src` is 0.
    pub fn bsr(&mut self, s: Size, dst: Gpr, src: impl Into<Rm>) -> Result<(), EncError> {
        if s == Size::B {
            return Err(EncError::Invalid("bsr has no 8-bit form"));
        }
        self.sized(s, &[], &[0x0f, 0xbd], dst.0, true, src.into().into(), 0)
    }

    /// `bsf dst, src` (`W`, `D`, `Q`): index of the lowest set bit; ZF=1 (and `dst` undefined)
    /// when `src` is 0.
    pub fn bsf(&mut self, s: Size, dst: Gpr, src: impl Into<Rm>) -> Result<(), EncError> {
        if s == Size::B {
            return Err(EncError::Invalid("bsf has no 8-bit form"));
        }
        self.sized(s, &[], &[0x0f, 0xbc], dst.0, true, src.into().into(), 0)
    }

    // ---- x87 (only for the exact float remainder, `fprem`) ----

    fn x87_m(&mut self, p: Prec, digit: u8, m: Mem) -> Result<(), EncError> {
        let opc: &[u8] = match p {
            Prec::F32 => &[0xd9],
            Prec::F64 => &[0xdd],
        };
        self.enc(Enc { pfx: None, w: false, rex8: false, opc, reg: digit, rm: RmEnc::Mem(m), trailing: 0 })
    }

    /// `fld dword [m]` / `fld qword [m]` (push).
    pub fn fld(&mut self, p: Prec, m: Mem) -> Result<(), EncError> {
        self.x87_m(p, 0, m)
    }

    /// `fstp dword [m]` / `fstp qword [m]` (store `st(0)` and pop).
    pub fn fstp(&mut self, p: Prec, m: Mem) -> Result<(), EncError> {
        self.x87_m(p, 3, m)
    }

    /// `fprem`: `st(0) = st(0) rem st(1)` (truncated, exact); C2 (status bit 10) set while partial.
    pub fn fprem(&mut self) {
        self.buf.extend_from_slice(&[0xd9, 0xf8]);
    }

    /// `fnstsw ax`.
    pub fn fnstsw_ax(&mut self) {
        self.buf.extend_from_slice(&[0xdf, 0xe0]);
    }

    /// `fstp st(0)` (pop and discard).
    pub fn fstp_st0(&mut self) {
        self.buf.extend_from_slice(&[0xdd, 0xd8]);
    }

    /// `movaps dst, src` (full-register move).
    pub fn movaps(&mut self, dst: Xmm, src: Xmm) {
        let _ = self.sse_enc(None, false, 0x28, dst.0, RmEnc::Reg(src.0));
    }
}
