//! x86-64 instruction encoder: general-purpose integer instructions, the SSE2 scalar float subset
//! a template/greedy debug backend needs, and packed-vector instructions in legacy SSE, VEX and
//! EVEX encodings (see [`Asm::sse_bin`], [`Asm::vex_bin`], [`Asm::evex`]).
//!
//! [`Asm`] is a byte buffer with labels and fixups. Branches to labels are always `rel32` (so code
//! size does not depend on label distance and patching is predictable). References to other
//! symbols become [`stave_obj::Reloc`]s. Every operand combination the hardware rejects is an
//! [`EncError`]; the encoder never panics.
//!
//! Operand order is Intel: destination first. Method suffixes name the operand kinds:
//! `r` register, `m` memory, `i` immediate (e.g. [`Asm::alu_mi`]: memory ← memory op imm).

mod evex;
mod inst;
mod simd;
mod simd_ops;

use std::fmt;

use stave_obj::{FileId, FuncCode, LineRow, Reloc, RelocKind, SymbolId};
use stave_target::{Feature, FeatureSet};

pub use evex::{EOp, ERm, EShiftI, EVL, EVsib, Ev, KBin, KReg, KSize};
pub use inst::{AluOp, PLogic, Prec, ShiftOp, SseOp, UnaryOp, XRm};
pub use simd::{VLen, VReg, VRm, VecReg};
pub use simd_ops::{
    CmpPred, PExtr, PInsr, Prefetch, VBin, VBinI, VBlendV, VCvtN, VExtract, VGather, VMaskStore, VMovMsk,
    VShiftI, VStore, VUn, VUnI,
};

/// A general-purpose register; `Gpr(n)` for n in 0..16 (`rax`..`r15`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Gpr(u8);

impl Gpr {
    pub const RAX: Gpr = Gpr(0);
    pub const RCX: Gpr = Gpr(1);
    pub const RDX: Gpr = Gpr(2);
    pub const RBX: Gpr = Gpr(3);
    pub const RSP: Gpr = Gpr(4);
    pub const RBP: Gpr = Gpr(5);
    pub const RSI: Gpr = Gpr(6);
    pub const RDI: Gpr = Gpr(7);
    pub const R8: Gpr = Gpr(8);
    pub const R9: Gpr = Gpr(9);
    pub const R10: Gpr = Gpr(10);
    pub const R11: Gpr = Gpr(11);
    pub const R12: Gpr = Gpr(12);
    pub const R13: Gpr = Gpr(13);
    pub const R14: Gpr = Gpr(14);
    pub const R15: Gpr = Gpr(15);

    pub fn new(n: u8) -> Option<Gpr> {
        (n < 16).then_some(Gpr(n))
    }

    pub fn num(self) -> u8 {
        self.0
    }

    /// The register's name at an operand size (`rax`, `eax`, `ax`, `al`; `r8`..`r8b`).
    pub fn name(self, s: Size) -> &'static str {
        const Q: [&str; 16] = [
            "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
            "r14", "r15",
        ];
        const D: [&str; 16] = [
            "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d",
            "r13d", "r14d", "r15d",
        ];
        const W: [&str; 16] = [
            "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w",
            "r14w", "r15w",
        ];
        const B: [&str; 16] = [
            "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b",
            "r14b", "r15b",
        ];
        let t = match s {
            Size::B => &B,
            Size::W => &W,
            Size::D => &D,
            Size::Q => &Q,
        };
        t[self.0 as usize & 15]
    }
}

/// An SSE register `xmm0`..`xmm15`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Xmm(u8);

impl Xmm {
    pub fn new(n: u8) -> Option<Xmm> {
        (n < 16).then_some(Xmm(n))
    }

    /// `xmm(n & 15)`: total, for register tables.
    pub const fn masked(n: u8) -> Xmm {
        Xmm(n & 15)
    }

    pub fn num(self) -> u8 {
        self.0
    }
}

/// An AVX register `ymm0`..`ymm15` (the 256-bit view of `xmm0`..`xmm15`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ymm(u8);

impl Ymm {
    pub fn new(n: u8) -> Option<Ymm> {
        (n < 16).then_some(Ymm(n))
    }

    /// `ymm(n & 15)`: total, for register tables.
    pub const fn masked(n: u8) -> Ymm {
        Ymm(n & 15)
    }

    pub fn num(self) -> u8 {
        self.0
    }
}

/// Operand size of an integer instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Size {
    /// 8 bits.
    B,
    /// 16 bits.
    W,
    /// 32 bits (writes zero the upper half of the 64-bit register).
    D,
    /// 64 bits.
    Q,
}

impl Size {
    pub fn bits(self) -> u32 {
        match self {
            Size::B => 8,
            Size::W => 16,
            Size::D => 32,
            Size::Q => 64,
        }
    }

    pub fn bytes(self) -> u8 {
        (self.bits() / 8) as u8
    }
}

/// Condition codes (the low nibble of `jcc`/`setcc`/`cmovcc`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    /// Overflow (OF=1): signed overflow traps.
    O = 0,
    No = 1,
    /// Below / carry (CF=1): unsigned overflow traps.
    B = 2,
    Ae = 3,
    E = 4,
    Ne = 5,
    Be = 6,
    A = 7,
    S = 8,
    Ns = 9,
    P = 10,
    Np = 11,
    L = 12,
    Ge = 13,
    Le = 14,
    G = 15,
}

impl Cond {
    /// Carry set (alias of [`Cond::B`]).
    pub const C: Cond = Cond::B;
    /// Carry clear (alias of [`Cond::Ae`]).
    pub const NC: Cond = Cond::Ae;
    pub const Z: Cond = Cond::E;
    pub const NZ: Cond = Cond::Ne;

    pub fn code(self) -> u8 {
        self as u8
    }

    /// The opposite condition.
    pub fn negate(self) -> Cond {
        const ALL: [Cond; 16] = [
            Cond::O,
            Cond::No,
            Cond::B,
            Cond::Ae,
            Cond::E,
            Cond::Ne,
            Cond::Be,
            Cond::A,
            Cond::S,
            Cond::Ns,
            Cond::P,
            Cond::Np,
            Cond::L,
            Cond::Ge,
            Cond::Le,
            Cond::G,
        ];
        ALL[(self as usize) ^ 1]
    }
}

/// Index scale of a memory operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scale {
    S1 = 0,
    S2 = 1,
    S4 = 2,
    S8 = 3,
}

/// A position in the code buffer, bound once with [`Asm::bind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(u32);

/// Target of a RIP-relative operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Label(Label),
    Sym(SymbolId),
}

/// A memory operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mem {
    /// `[base + index * scale + disp]`; either register may be absent (`[disp32]` is absolute).
    /// `rsp` cannot be an index.
    Sib { base: Option<Gpr>, index: Option<(Gpr, Scale)>, disp: i32 },
    /// `[rip + target + disp]`.
    Rip { target: Target, disp: i32 },
}

impl Mem {
    pub fn base(base: Gpr, disp: i32) -> Mem {
        Mem::Sib { base: Some(base), index: None, disp }
    }

    pub fn index(base: Gpr, index: Gpr, scale: Scale, disp: i32) -> Mem {
        Mem::Sib { base: Some(base), index: Some((index, scale)), disp }
    }

    pub fn rip_label(l: Label, disp: i32) -> Mem {
        Mem::Rip { target: Target::Label(l), disp }
    }

    pub fn rip_sym(s: SymbolId, disp: i32) -> Mem {
        Mem::Rip { target: Target::Sym(s), disp }
    }
}

/// A register or memory operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rm {
    R(Gpr),
    M(Mem),
}

impl From<Gpr> for Rm {
    fn from(r: Gpr) -> Rm {
        Rm::R(r)
    }
}

impl From<Mem> for Rm {
    fn from(m: Mem) -> Rm {
        Rm::M(m)
    }
}

/// Encoding errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncError {
    /// The operand combination does not exist (message names it).
    Invalid(&'static str),
    /// An immediate does not fit the instruction's field.
    ImmRange { imm: i64, bits: u32 },
    /// A label was bound twice.
    LabelRebound(Label),
    /// A label is not from this buffer.
    UnknownLabel(Label),
    /// A label referenced by a fixup was never bound.
    LabelUnbound(Label),
    /// The function exceeds 2 GiB.
    TooLarge,
}

impl fmt::Display for EncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncError::Invalid(w) => write!(f, "invalid operands: {w}"),
            EncError::ImmRange { imm, bits } => write!(f, "immediate {imm} does not fit {bits} bits"),
            EncError::LabelRebound(l) => write!(f, "label {} bound twice", l.0),
            EncError::UnknownLabel(l) => write!(f, "unknown label {}", l.0),
            EncError::LabelUnbound(l) => write!(f, "label {} never bound", l.0),
            EncError::TooLarge => f.write_str("function too large"),
        }
    }
}

impl std::error::Error for EncError {}

#[derive(Clone, Copy, Debug)]
struct Fixup {
    /// Offset of the rel32 field.
    at: u32,
    label: Label,
    /// Value = label - (at + 4) + bias.
    bias: i64,
}

/// The code buffer of one function.
#[derive(Clone, Debug, Default)]
pub struct Asm {
    buf: Vec<u8>,
    labels: Vec<Option<u32>>,
    fixups: Vec<Fixup>,
    relocs: Vec<Reloc>,
    lines: Vec<LineRow>,
    /// The scalar SSE functions (`sse`, `movs`, `movaps`, ...) emit their VEX forms.
    vex_scalars: bool,
    /// The CPUID features of the instructions encoded so far beyond SSE4.2.
    used: FeatureSet,
}

const MAX_CODE: usize = i32::MAX as usize;

impl Asm {
    pub fn new() -> Asm {
        Asm::default()
    }

    /// An empty buffer with room for `bytes` of code and `labels` labels.
    pub fn with_capacity(bytes: usize, labels: usize) -> Asm {
        Asm { buf: Vec::with_capacity(bytes), labels: Vec::with_capacity(labels), ..Asm::default() }
    }

    /// Makes the scalar SSE functions (`sse`, `movs`, `movaps`, ...) emit their VEX forms with
    /// the destination as the first source: the same result in the low 128 bits, zeros above
    /// them, and no dependence on the `ymm` upper halves.
    pub fn set_vex_scalars(&mut self, on: bool) {
        self.vex_scalars = on;
    }

    /// The CPUID features beyond SSE4.2 that the instructions encoded so far need (closed under
    /// implication): every VEX, FMA, F16C, and BMI2 instruction records its own.
    pub fn features(&self) -> FeatureSet {
        self.used
    }

    pub(super) fn note(&mut self, f: Feature) {
        self.used.insert(f);
    }

    /// Current offset.
    pub fn offset(&self) -> u32 {
        self.buf.len().min(u32::MAX as usize) as u32
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn relocs(&self) -> &[Reloc] {
        &self.relocs
    }

    pub fn new_label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() as u32 - 1)
    }

    pub fn bind(&mut self, l: Label) -> Result<(), EncError> {
        let off = self.offset();
        match self.labels.get_mut(l.0 as usize) {
            None => Err(EncError::UnknownLabel(l)),
            Some(Some(_)) => Err(EncError::LabelRebound(l)),
            Some(s) => {
                *s = Some(off);
                Ok(())
            }
        }
    }

    /// The bound offset of a label.
    pub fn label_offset(&self, l: Label) -> Option<u32> {
        self.labels.get(l.0 as usize).copied().flatten()
    }

    /// Starts a line-table row at the current offset.
    pub fn line(&mut self, file: FileId, line: u32, column: u32) {
        let offset = self.offset();
        if let Some(last) = self.lines.last_mut()
            && last.offset == offset
        {
            *last = LineRow { offset, file, line, column };
            return;
        }
        self.lines.push(LineRow { offset, file, line, column });
    }

    /// Appends raw bytes.
    pub fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    /// Pads with `nop`s (0x90) to a multiple of `align` (a power of two ≤ 4096).
    pub fn align(&mut self, align: u32) -> Result<(), EncError> {
        if !align.is_power_of_two() || align > 4096 {
            return Err(EncError::Invalid("alignment must be a power of two ≤ 4096"));
        }
        while !self.buf.len().is_multiple_of(align as usize) {
            self.buf.push(0x90);
        }
        Ok(())
    }

    fn check_label(&self, l: Label) -> Result<(), EncError> {
        if l.0 as usize >= self.labels.len() {
            return Err(EncError::UnknownLabel(l));
        }
        Ok(())
    }

    fn rel32_label(&mut self, l: Label, bias: i64) {
        let at = self.offset();
        self.fixups.push(Fixup { at, label: l, bias });
        self.buf.extend_from_slice(&[0; 4]);
    }

    fn rel32_sym(&mut self, s: SymbolId, addend: i64) {
        let offset = self.offset();
        self.relocs.push(Reloc { offset, kind: RelocKind::X86Pc32, sym: s, addend });
        self.buf.extend_from_slice(&[0; 4]);
    }

    /// Resolves label fixups and returns the function's code, relocations, and line rows.
    pub fn finish(mut self) -> Result<FuncCode, EncError> {
        if self.buf.len() > MAX_CODE {
            return Err(EncError::TooLarge);
        }
        for f in &self.fixups {
            let target = self.labels[f.label.0 as usize].ok_or(EncError::LabelUnbound(f.label))?;
            let v = i64::from(target) - (i64::from(f.at) + 4) + f.bias;
            let v = i32::try_from(v).map_err(|_| EncError::TooLarge)?;
            let at = f.at as usize;
            self.buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(FuncCode { code: self.buf, relocs: self.relocs, lines: self.lines })
    }
}

#[cfg(test)]
mod tests;
