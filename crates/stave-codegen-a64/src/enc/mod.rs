//! AArch64 (A64) instruction encoder: the integer and scalar floating-point subset a
//! template/greedy debug backend needs.
//!
//! [`Asm`] is a buffer of 32-bit instruction words with labels and fixups. References to other
//! symbols become [`stave_obj::Reloc`]s (`BL`/`B` → `A64Branch26`, `ADRP` → `A64AdrPage21`,
//! `ADD`/`LDR` `:lo12:` → `A64AddLo12`/`A64LdstLo12`). Register 31 is `sp` or `xzr` depending on
//! the operand; [`Reg::SP`] and [`Reg::ZR`] are distinct values and every encoder rejects the one
//! the operand cannot name. Invalid operands are an [`EncError`]; the encoder never panics, and a
//! failed instruction emits nothing.

mod imm;
mod inst;

use std::fmt;

use stave_obj::{FileId, FuncCode, LineRow, Reloc, RelocKind, SymbolId};

pub use imm::{decode_logical, encode_fp8, encode_logical};
pub use inst::{
    AddSub, Addr, Ext, FcvtRound, FpBin, FpFma, FpUn, IdxExt, Logic, MemSize, Shift, ShiftR, VecSize,
};

/// A general register `x0`..`x30`, or [`Reg::SP`], or [`Reg::ZR`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reg(u8);

impl Reg {
    /// Frame pointer `x29`.
    pub const FP: Reg = Reg(29);
    /// Link register `x30`.
    pub const LR: Reg = Reg(30);
    /// The zero register (`xzr`/`wzr`).
    pub const ZR: Reg = Reg(31);
    /// The stack pointer.
    pub const SP: Reg = Reg(32);

    /// `x<n>` for n in 0..=30.
    pub fn x(n: u8) -> Option<Reg> {
        (n <= 30).then_some(Reg(n))
    }

    /// `x<n>` with `n` clamped to 0..=30 (infallible, for constant register numbers).
    pub const fn xn(n: u8) -> Reg {
        Reg(if n > 30 { 30 } else { n })
    }

    pub fn is_sp(self) -> bool {
        self.0 == 32
    }

    pub fn is_zr(self) -> bool {
        self.0 == 31
    }

    /// Hardware number (31 for both `sp` and `zr`).
    pub fn num(self) -> u8 {
        self.0.min(31)
    }

    pub fn name(self, w: Width) -> String {
        match (self.0, w) {
            (31, Width::W) => "wzr".into(),
            (31, Width::X) => "xzr".into(),
            (32, Width::W) => "wsp".into(),
            (32, Width::X) => "sp".into(),
            (n, Width::W) => format!("w{n}"),
            (n, Width::X) => format!("x{n}"),
        }
    }
}

/// A SIMD/FP register `v0`..`v31` (used as `s<n>`, `d<n>`, `q<n>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(u8);

impl VReg {
    pub fn new(n: u8) -> Option<VReg> {
        (n < 32).then_some(VReg(n))
    }

    /// `v<n & 31>` (infallible, for constant register numbers).
    pub const fn vn(n: u8) -> VReg {
        VReg(n & 31)
    }

    pub fn num(self) -> u8 {
        self.0
    }
}

/// Operand width of an integer instruction (`sf`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Width {
    W,
    X,
}

impl Width {
    pub fn bits(self) -> u32 {
        match self {
            Width::W => 32,
            Width::X => 64,
        }
    }

    fn sf(self) -> u32 {
        match self {
            Width::W => 0,
            Width::X => 1 << 31,
        }
    }
}

/// Scalar float type (`ftype`): single or double precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp {
    S,
    D,
}

impl Fp {
    fn ftype(self) -> u32 {
        match self {
            Fp::S => 0,
            Fp::D => 1 << 22,
        }
    }
}

/// Condition codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    Eq = 0,
    Ne = 1,
    /// Carry set / unsigned higher or same (`cs`).
    Hs = 2,
    /// Carry clear / unsigned lower (`cc`).
    Lo = 3,
    Mi = 4,
    Pl = 5,
    /// Signed overflow.
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
    Al = 14,
    Nv = 15,
}

impl Cond {
    pub const ALL: [Cond; 16] = [
        Cond::Eq,
        Cond::Ne,
        Cond::Hs,
        Cond::Lo,
        Cond::Mi,
        Cond::Pl,
        Cond::Vs,
        Cond::Vc,
        Cond::Hi,
        Cond::Ls,
        Cond::Ge,
        Cond::Lt,
        Cond::Gt,
        Cond::Le,
        Cond::Al,
        Cond::Nv,
    ];

    pub fn code(self) -> u32 {
        self as u32
    }

    /// The opposite condition (`al`/`nv` have none and map to each other).
    pub fn negate(self) -> Cond {
        Cond::ALL[(self as usize) ^ 1]
    }
}

/// A position in the code buffer, bound once with [`Asm::bind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(u32);

/// Encoding errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncError {
    /// The operand combination does not exist (message names it).
    Invalid(&'static str),
    /// An immediate does not fit (or is not encodable in) the instruction's field.
    ImmRange {
        imm: i64,
        bits: u32,
    },
    /// A label is out of the branch's range (found when the label is resolved).
    BranchRange(Label),
    LabelRebound(Label),
    UnknownLabel(Label),
    LabelUnbound(Label),
    TooLarge,
}

impl fmt::Display for EncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncError::Invalid(w) => write!(f, "invalid operands: {w}"),
            EncError::ImmRange { imm, bits } => write!(f, "immediate {imm} not encodable in {bits} bits"),
            EncError::BranchRange(l) => write!(f, "label {} out of branch range", l.0),
            EncError::LabelRebound(l) => write!(f, "label {} bound twice", l.0),
            EncError::UnknownLabel(l) => write!(f, "unknown label {}", l.0),
            EncError::LabelUnbound(l) => write!(f, "label {} never bound", l.0),
            EncError::TooLarge => f.write_str("function too large"),
        }
    }
}

impl std::error::Error for EncError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixKind {
    /// `B`/`BL` imm26.
    B26,
    /// `B.cond`, `CBZ`, `CBNZ`, `LDR (literal)` imm19 at bits 23:5.
    B19,
    /// `TBZ`/`TBNZ` imm14 at bits 18:5.
    B14,
    /// `ADR` byte offset (immlo 30:29, immhi 23:5).
    Adr21,
}

#[derive(Clone, Copy, Debug)]
struct Fixup {
    at: u32,
    label: Label,
    kind: FixKind,
}

/// The code buffer of one function.
#[derive(Clone, Debug, Default)]
pub struct Asm {
    buf: Vec<u8>,
    labels: Vec<Option<u32>>,
    fixups: Vec<Fixup>,
    relocs: Vec<Reloc>,
    lines: Vec<LineRow>,
    /// Conditional branches to labels are emitted as an inverted branch over a `b` (±128 MiB).
    long_branches: bool,
}

impl Asm {
    pub fn new() -> Asm {
        Asm::default()
    }

    /// Long-branch mode: `b.cond`, `cbz`, `cbnz`, `tbz`, `tbnz` to a label reach ±128 MiB (two
    /// words each). For functions whose short branches fail with [`EncError::BranchRange`].
    pub fn set_long_branches(&mut self, on: bool) {
        self.long_branches = on;
    }

    /// Current offset in bytes.
    pub fn offset(&self) -> u32 {
        self.buf.len().min(u32::MAX as usize) as u32
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn relocs(&self) -> &[Reloc] {
        &self.relocs
    }

    /// Emits a raw instruction word.
    pub fn word(&mut self, w: u32) {
        self.buf.extend_from_slice(&w.to_le_bytes());
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

    fn check_label(&self, l: Label) -> Result<(), EncError> {
        if l.0 as usize >= self.labels.len() {
            return Err(EncError::UnknownLabel(l));
        }
        Ok(())
    }

    fn word_fix(&mut self, w: u32, l: Label, kind: FixKind) -> Result<(), EncError> {
        self.check_label(l)?;
        let at = self.offset();
        self.fixups.push(Fixup { at, label: l, kind });
        self.word(w);
        Ok(())
    }

    fn word_reloc(&mut self, w: u32, kind: RelocKind, sym: SymbolId, addend: i64) {
        let offset = self.offset();
        self.relocs.push(Reloc { offset, kind, sym, addend });
        self.word(w);
    }

    /// Resolves label fixups and returns the function's code, relocations, and line rows.
    pub fn finish(mut self) -> Result<FuncCode, EncError> {
        if self.buf.len() > (1 << 30) {
            return Err(EncError::TooLarge);
        }
        for f in &self.fixups {
            let target = self.labels[f.label.0 as usize].ok_or(EncError::LabelUnbound(f.label))?;
            let d = i64::from(target) - i64::from(f.at);
            let at = f.at as usize;
            let mut w =
                u32::from_le_bytes([self.buf[at], self.buf[at + 1], self.buf[at + 2], self.buf[at + 3]]);
            let in_range = |bits: u32| (-(1i64 << (bits - 1))..(1i64 << (bits - 1))).contains(&(d >> 2));
            match f.kind {
                FixKind::B26 if in_range(26) => w |= ((d >> 2) as u32) & 0x03ff_ffff,
                FixKind::B19 if in_range(19) => w |= (((d >> 2) as u32) & 0x7ffff) << 5,
                FixKind::B14 if in_range(14) => w |= (((d >> 2) as u32) & 0x3fff) << 5,
                FixKind::Adr21 if (-(1i64 << 20)..(1i64 << 20)).contains(&d) => {
                    let v = d as u32;
                    w |= ((v & 3) << 29) | (((v >> 2) & 0x7ffff) << 5);
                }
                _ => return Err(EncError::BranchRange(f.label)),
            }
            self.buf[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
        Ok(FuncCode { code: self.buf, relocs: self.relocs, lines: self.lines })
    }
}

#[cfg(test)]
mod tests;
