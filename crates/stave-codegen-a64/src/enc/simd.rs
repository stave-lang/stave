//! Advanced SIMD (NEON) instruction forms on 64- and 128-bit vectors.
//!
//! Operations are grouped by encoding class: an op enum selects the instruction, an [`Arr`]
//! selects the arrangement (`8b`, `16b`, `4h`, `8h`, `2s`, `4s`, `2d`). A combination the
//! architecture does not have is an [`EncError`] and emits nothing.
//!
//! Widening and narrowing forms take the arrangement of the narrow side: the `2` variants
//! (`sxtl2`, `xtn2`, `smull2`, ...) are the 128-bit narrow arrangements `16b`, `8h`, `4s`.

use super::imm::encode_fp8;
use super::inst::{FcvtRound, Ls, VecSize, gpr, range, sp, v, zr};
use super::{Addr, Asm, EncError, Fp, Reg, VReg, Width};

/// Lane size of a vector operand (`log2` of the byte size).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lane {
    B = 0,
    H = 1,
    S = 2,
    D = 3,
}

impl Lane {
    pub fn bits(self) -> u32 {
        8 << (self as u32)
    }

    /// The 128-bit arrangement with this lane size.
    pub fn arr(self) -> Arr {
        match self {
            Lane::B => Arr::B16,
            Lane::H => Arr::H8,
            Lane::S => Arr::S4,
            Lane::D => Arr::D2,
        }
    }

    fn mask(self) -> u8 {
        1 << (self as u32)
    }

    /// The `imm5` of an element index (`dup`, `ins`, `umov`, `smov`).
    fn imm5(self, idx: u32) -> Result<u32, EncError> {
        range(i64::from(idx), 0, i64::from(15 >> (self as u32)), 4)?;
        Ok((idx << (self as u32 + 1)) | (1 << (self as u32)))
    }
}

/// A vector arrangement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arr {
    B8,
    B16,
    H4,
    H8,
    S2,
    S4,
    D2,
}

const B: u8 = 1;
const H: u8 = 2;
const S: u8 = 4;
const D: u8 = 8;
const BHS: u8 = B | H | S;
const BHSD: u8 = BHS | D;

impl Arr {
    pub const ALL: [Arr; 7] = [Arr::B8, Arr::B16, Arr::H4, Arr::H8, Arr::S2, Arr::S4, Arr::D2];

    pub fn lane(self) -> Lane {
        match self {
            Arr::B8 | Arr::B16 => Lane::B,
            Arr::H4 | Arr::H8 => Lane::H,
            Arr::S2 | Arr::S4 => Lane::S,
            Arr::D2 => Lane::D,
        }
    }

    /// 128-bit arrangement (the `Q` bit).
    pub fn is_q(self) -> bool {
        matches!(self, Arr::B16 | Arr::H8 | Arr::S4 | Arr::D2)
    }

    pub fn lanes(self) -> u32 {
        (if self.is_q() { 128 } else { 64 }) / self.lane().bits()
    }

    /// The arrangement with lanes twice as wide and the same lane count.
    pub fn widened(self) -> Option<Arr> {
        match self {
            Arr::B8 | Arr::B16 => Some(Arr::H8),
            Arr::H4 | Arr::H8 => Some(Arr::S4),
            Arr::S2 | Arr::S4 => Some(Arr::D2),
            Arr::D2 => None,
        }
    }

    fn q(self) -> u32 {
        u32::from(self.is_q()) << 30
    }

    fn size(self) -> u32 {
        self.lane() as u32
    }

    fn has(self, mask: u8) -> bool {
        self.lane().mask() & mask != 0
    }

    fn require(self, mask: u8, msg: &'static str) -> Result<(), EncError> {
        if self.has(mask) { Ok(()) } else { Err(EncError::Invalid(msg)) }
    }

    /// `sz` (bit 22) of a floating-point arrangement: `2s`/`4s` are single, `2d` is double.
    fn fsz(self) -> Result<u32, EncError> {
        match self {
            Arr::S2 | Arr::S4 => Ok(0),
            Arr::D2 => Ok(1 << 22),
            _ => Err(EncError::Invalid("floating-point op needs 2s, 4s or 2d")),
        }
    }
}

/// Signedness of a widening or narrowing operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sign {
    Signed,
    Unsigned,
}

impl Sign {
    fn u(self) -> u32 {
        u32::from(self == Sign::Unsigned)
    }
}

/// Integer three-register operations (`U`, `opcode` of the three-same class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VBin {
    Add,
    Sub,
    /// No `2d` form.
    Mul,
    SqAdd,
    UqAdd,
    SqSub,
    UqSub,
    Smax,
    Umax,
    Smin,
    Umin,
    /// Pairwise add (also `2d`).
    Addp,
    Smaxp,
    Umaxp,
    Sminp,
    Uminp,
    Cmeq,
    Cmgt,
    Cmge,
    Cmhi,
    Cmhs,
    Cmtst,
    /// Shift left by the signed per-lane amount in `vm` (right when negative).
    Sshl,
    Ushl,
}

impl VBin {
    fn enc(self) -> (u32, u32, u8) {
        match self {
            VBin::Add => (0, 0x10, BHSD),
            VBin::Sub => (1, 0x10, BHSD),
            VBin::Mul => (0, 0x13, BHS),
            VBin::SqAdd => (0, 0x01, BHSD),
            VBin::UqAdd => (1, 0x01, BHSD),
            VBin::SqSub => (0, 0x05, BHSD),
            VBin::UqSub => (1, 0x05, BHSD),
            VBin::Smax => (0, 0x0c, BHS),
            VBin::Umax => (1, 0x0c, BHS),
            VBin::Smin => (0, 0x0d, BHS),
            VBin::Umin => (1, 0x0d, BHS),
            VBin::Addp => (0, 0x17, BHSD),
            VBin::Smaxp => (0, 0x14, BHS),
            VBin::Umaxp => (1, 0x14, BHS),
            VBin::Sminp => (0, 0x15, BHS),
            VBin::Uminp => (1, 0x15, BHS),
            VBin::Cmeq => (1, 0x11, BHSD),
            VBin::Cmgt => (0, 0x06, BHSD),
            VBin::Cmge => (0, 0x07, BHSD),
            VBin::Cmhi => (1, 0x06, BHSD),
            VBin::Cmhs => (1, 0x07, BHSD),
            VBin::Cmtst => (0, 0x11, BHSD),
            VBin::Sshl => (0, 0x08, BHSD),
            VBin::Ushl => (1, 0x08, BHSD),
        }
    }
}

/// Bitwise three-register operations on `8b`/`16b` (`U`, `size` of the three-same class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VLogic {
    And,
    /// `vn & ~vm`.
    Bic,
    Orr,
    /// `vn | ~vm`.
    Orn,
    Eor,
    /// Bitwise select: `vd = (vd & vn) | (~vd & vm)` (the mask is the old `vd`).
    Bsl,
    /// Bitwise insert if true: `vd = (vd & ~vm) | (vn & vm)`.
    Bit,
    /// Bitwise insert if false: `vd = (vd & vm) | (vn & ~vm)`.
    Bif,
}

impl VLogic {
    fn enc(self) -> (u32, u32) {
        match self {
            VLogic::And => (0, 0),
            VLogic::Bic => (0, 1),
            VLogic::Orr => (0, 2),
            VLogic::Orn => (0, 3),
            VLogic::Eor => (1, 0),
            VLogic::Bsl => (1, 1),
            VLogic::Bit => (1, 2),
            VLogic::Bif => (1, 3),
        }
    }
}

/// Floating-point three-register operations on `2s`, `4s`, `2d` (`U`, `a`, `opcode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VFpBin {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MaxNm,
    MinNm,
    /// `vd += vn * vm`, fused.
    Fmla,
    /// `vd -= vn * vm`, fused.
    Fmls,
    /// Pairwise add (`faddp`).
    Addp,
    Maxp,
    Minp,
    MaxNmp,
    MinNmp,
    Cmeq,
    Cmge,
    Cmgt,
    /// Absolute compare `|vn| >= |vm|`.
    Acge,
    Acgt,
    /// `|vn - vm|`.
    Abd,
}

impl VFpBin {
    fn enc(self) -> (u32, u32, u32) {
        match self {
            VFpBin::Add => (0, 0, 0x1a),
            VFpBin::Sub => (0, 1, 0x1a),
            VFpBin::Mul => (1, 0, 0x1b),
            VFpBin::Div => (1, 0, 0x1f),
            VFpBin::Max => (0, 0, 0x1e),
            VFpBin::Min => (0, 1, 0x1e),
            VFpBin::MaxNm => (0, 0, 0x18),
            VFpBin::MinNm => (0, 1, 0x18),
            VFpBin::Fmla => (0, 0, 0x19),
            VFpBin::Fmls => (0, 1, 0x19),
            VFpBin::Addp => (1, 0, 0x1a),
            VFpBin::Maxp => (1, 0, 0x1e),
            VFpBin::Minp => (1, 1, 0x1e),
            VFpBin::MaxNmp => (1, 0, 0x18),
            VFpBin::MinNmp => (1, 1, 0x18),
            VFpBin::Cmeq => (0, 0, 0x1c),
            VFpBin::Cmge => (1, 0, 0x1c),
            VFpBin::Cmgt => (1, 1, 0x1c),
            VFpBin::Acge => (1, 0, 0x1d),
            VFpBin::Acgt => (1, 1, 0x1d),
            VFpBin::Abd => (1, 1, 0x1a),
        }
    }
}

/// Integer two-register operations (`U`, `opcode`, lane sizes of the two-register-misc class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VIntUn {
    Abs,
    Neg,
    /// Count leading sign bits (`b`, `h`, `s` lanes).
    Cls,
    /// Count leading zeros (`b`, `h`, `s` lanes).
    Clz,
    /// Population count per byte (`8b`, `16b`).
    Cnt,
    /// Bitwise not (`mvn`; `8b`, `16b`).
    Not,
}

impl VIntUn {
    fn enc(self) -> (u32, u32, u8) {
        match self {
            VIntUn::Abs => (0, 0x0b, BHSD),
            VIntUn::Neg => (1, 0x0b, BHSD),
            VIntUn::Cls => (0, 0x04, BHS),
            VIntUn::Clz => (1, 0x04, BHS),
            VIntUn::Cnt => (0, 0x05, B),
            VIntUn::Not => (1, 0x05, B),
        }
    }
}

/// Floating-point two-register operations on `2s`, `4s`, `2d` (`U`, `a`, `opcode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VFpUn {
    Abs,
    Neg,
    Sqrt,
    /// Round to nearest, ties to even.
    RintN,
    /// Round toward minus infinity.
    RintM,
    /// Round toward plus infinity.
    RintP,
    /// Round toward zero.
    RintZ,
    /// Round to nearest, ties away from zero.
    RintA,
    /// Round using the current mode, signalling inexact.
    RintX,
    /// Round using the current mode.
    RintI,
}

impl VFpUn {
    fn enc(self) -> (u32, u32, u32) {
        match self {
            VFpUn::Abs => (0, 1, 0x0f),
            VFpUn::Neg => (1, 1, 0x0f),
            VFpUn::Sqrt => (1, 1, 0x1f),
            VFpUn::RintN => (0, 0, 0x18),
            VFpUn::RintM => (0, 0, 0x19),
            VFpUn::RintP => (0, 1, 0x18),
            VFpUn::RintZ => (0, 1, 0x19),
            VFpUn::RintA => (1, 0, 0x18),
            VFpUn::RintX => (1, 0, 0x19),
            VFpUn::RintI => (1, 1, 0x19),
        }
    }
}

/// Comparison of every lane with zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpZ {
    Eq,
    Gt,
    Ge,
    Lt,
    Le,
}

/// Reversal of the elements inside each 16-, 32- or 64-bit container.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rev {
    R16,
    R32,
    R64,
}

/// Shift by immediate (`U`, `opcode` of the shift-by-immediate class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VShift {
    /// Left, `0..lane bits`.
    Shl,
    /// Arithmetic right, `1..=lane bits`.
    Sshr,
    Ushr,
    /// Shift left and insert, `0..lane bits`.
    Sli,
    /// Shift right and insert, `1..=lane bits`.
    Sri,
}

/// Narrowing moves (`xtn`, `xtn2` by the arrangement).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Narrow {
    /// Truncate.
    Xtn,
    /// Signed to signed, saturating.
    Sqxtn,
    /// Unsigned to unsigned, saturating.
    Uqxtn,
    /// Signed to unsigned, saturating.
    Sqxtun,
}

/// Reductions across the lanes of one vector into a scalar register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Across {
    /// Sum, same width (`8b`, `16b`, `4h`, `8h`, `4s`).
    Addv,
    /// Signed sum into a lane twice as wide (`d` from `4s`).
    Saddlv,
    Uaddlv,
    Smaxv,
    Umaxv,
    Sminv,
    Uminv,
    /// Floating point, `4s` only.
    Fmaxv,
    Fminv,
    FmaxNmv,
    FminNmv,
}

/// Lane permutations of two vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Perm {
    Uzp1,
    Trn1,
    Zip1,
    Uzp2,
    Trn2,
    Zip2,
}

impl Perm {
    fn opcode(self) -> u32 {
        match self {
            Perm::Uzp1 => 1,
            Perm::Trn1 => 2,
            Perm::Zip1 => 3,
            Perm::Uzp2 => 5,
            Perm::Trn2 => 6,
            Perm::Zip2 => 7,
        }
    }
}

/// Table lookup flavour: `tbl` writes zero for an out-of-range index, `tbx` keeps the old lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tbl {
    Tbl,
    Tbx,
}

/// Shift applied to the immediate of `movi`/`mvni`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImmShift {
    /// `lsl #n`: 0 for bytes; 0 or 8 for halfwords; 0, 8, 16, 24 for words.
    Lsl(u8),
    /// `msl #n` (shift in ones): 8 or 16, word lanes.
    Msl(u8),
}

/// Address of `ld1`, `st1` and `ld1r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VecAddr {
    /// `[base]`.
    Base(Reg),
    /// `[base], #bytes`, where bytes is the size of the transfer (registers times 8 or 16, or
    /// the lane size for `ld1r`).
    PostImm(Reg),
    /// `[base], index`.
    PostReg { base: Reg, index: Reg },
}

/// Prefetch operation kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefKind {
    /// Load.
    Pld = 0,
    /// Instruction.
    Pli = 1,
    /// Store.
    Pst = 2,
}

/// Prefetch target cache level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefLevel {
    L1 = 0,
    L2 = 1,
    L3 = 2,
}

/// A `prfm` operation: `pldl1keep` is `Prefetch { kind: Pld, level: L1, stream: false }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Prefetch {
    pub kind: PrefKind,
    pub level: PrefLevel,
    /// `strm` (streaming, non-temporal) instead of `keep`.
    pub stream: bool,
}

impl Prefetch {
    fn code(self) -> u32 {
        ((self.kind as u32) << 3) | ((self.level as u32) << 1) | u32::from(self.stream)
    }
}

/// Splits an 8-bit modified immediate into its `abc` (18:16) and `defgh` (9:5) fields.
fn imm8_fields(imm8: u32) -> u32 {
    ((imm8 >> 5) << 16) | ((imm8 & 31) << 5)
}

fn rr(vd: VReg, vn: VReg) -> u32 {
    (v(vn) << 5) | v(vd)
}

fn rrr(vd: VReg, vn: VReg, vm: VReg) -> u32 {
    (v(vm) << 16) | rr(vd, vn)
}

impl Asm {
    fn emit(&mut self, w: u32) -> Result<(), EncError> {
        self.word(w);
        Ok(())
    }

    // ---- three registers ----

    /// Integer `add/sub/mul/sqadd/.../cmeq/cmgt/.../sshl/ushl vd.T, vn.T, vm.T`.
    pub fn vbin(&mut self, op: VBin, arr: Arr, vd: VReg, vn: VReg, vm: VReg) -> Result<(), EncError> {
        let (u, opcode, mask) = op.enc();
        arr.require(mask, "integer vector op does not exist for this arrangement")?;
        self.emit(0x0e20_0400 | arr.q() | (u << 29) | (arr.size() << 22) | (opcode << 11) | rrr(vd, vn, vm))
    }

    /// `and/bic/orr/orn/eor/bsl/bit/bif vd.T, vn.T, vm.T` (`8b` or `16b`).
    pub fn vlogic(&mut self, op: VLogic, arr: Arr, vd: VReg, vn: VReg, vm: VReg) -> Result<(), EncError> {
        arr.require(B, "bitwise vector op needs 8b or 16b")?;
        let (u, size) = op.enc();
        self.emit(0x0e20_1c00 | arr.q() | (u << 29) | (size << 22) | rrr(vd, vn, vm))
    }

    /// `mov vd.T, vn.T` (`orr vd, vn, vn`; `8b` or `16b`).
    pub fn vmov(&mut self, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        self.vlogic(VLogic::Orr, arr, vd, vn, vn)
    }

    /// Floating-point `fadd/fsub/fmul/fdiv/fmax/fmin/fmaxnm/fminnm/fmla/fmls/faddp/fcmeq/... `.
    pub fn vfp_bin(&mut self, op: VFpBin, arr: Arr, vd: VReg, vn: VReg, vm: VReg) -> Result<(), EncError> {
        let (u, a, opcode) = op.enc();
        let sz = arr.fsz()?;
        self.emit(0x0e20_0400 | arr.q() | (u << 29) | (a << 23) | sz | (opcode << 11) | rrr(vd, vn, vm))
    }

    /// `ext vd.T, vn.T, vm.T, #index` (`8b` or `16b`; `index` in bytes).
    pub fn vext(&mut self, arr: Arr, vd: VReg, vn: VReg, vm: VReg, index: u32) -> Result<(), EncError> {
        arr.require(B, "ext needs 8b or 16b")?;
        range(i64::from(index), 0, i64::from(arr.lanes() - 1), 4)?;
        self.emit(0x2e00_0000 | arr.q() | (index << 11) | rrr(vd, vn, vm))
    }

    /// `zip1/zip2/uzp1/uzp2/trn1/trn2 vd.T, vn.T, vm.T`.
    pub fn vperm(&mut self, op: Perm, arr: Arr, vd: VReg, vn: VReg, vm: VReg) -> Result<(), EncError> {
        self.emit(0x0e00_0800 | arr.q() | (arr.size() << 22) | (op.opcode() << 12) | rrr(vd, vn, vm))
    }

    /// `tbl/tbx vd.T, {vn.16b, ...}, vm.T` with `n` (1..=4) consecutive table registers from
    /// `first` (numbers wrap from `v31` to `v0`); `8b` or `16b`.
    pub fn vtbl(
        &mut self,
        op: Tbl,
        arr: Arr,
        vd: VReg,
        first: VReg,
        n: u8,
        vm: VReg,
    ) -> Result<(), EncError> {
        arr.require(B, "tbl/tbx need 8b or 16b")?;
        range(i64::from(n), 1, 4, 2)?;
        let x = u32::from(op == Tbl::Tbx);
        self.emit(0x0e00_0000 | arr.q() | ((u32::from(n) - 1) << 13) | (x << 12) | rrr(vd, first, vm))
    }

    // ---- two registers ----

    /// `abs/neg/cls/clz/cnt/mvn vd.T, vn.T`.
    pub fn vint_un(&mut self, op: VIntUn, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let (u, opcode, mask) = op.enc();
        arr.require(mask, "integer vector op does not exist for this arrangement")?;
        self.emit(0x0e20_0800 | arr.q() | (u << 29) | (arr.size() << 22) | (opcode << 12) | rr(vd, vn))
    }

    /// `rev16/rev32/rev64 vd.T, vn.T`.
    pub fn vrev(&mut self, op: Rev, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let (u, opcode, mask) = match op {
            Rev::R64 => (0, 0, BHS),
            Rev::R32 => (1, 0, B | H),
            Rev::R16 => (0, 1, B),
        };
        arr.require(mask, "rev container is not wider than the lanes")?;
        self.emit(0x0e20_0800 | arr.q() | (u << 29) | (arr.size() << 22) | (opcode << 12) | rr(vd, vn))
    }

    /// `fabs/fneg/fsqrt/frint* vd.T, vn.T`.
    pub fn vfp_un(&mut self, op: VFpUn, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let (u, a, opcode) = op.enc();
        self.fp_misc(u, a, opcode, arr, vd, vn)
    }

    fn fp_misc(&mut self, u: u32, a: u32, opcode: u32, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let sz = arr.fsz()?;
        self.emit(0x0e20_0800 | arr.q() | (u << 29) | (a << 23) | sz | (opcode << 12) | rr(vd, vn))
    }

    /// `cmeq/cmgt/cmge/cmlt/cmle vd.T, vn.T, #0`.
    pub fn vcmp_zero(&mut self, op: CmpZ, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let (u, opcode) = match op {
            CmpZ::Gt => (0, 0x08),
            CmpZ::Eq => (0, 0x09),
            CmpZ::Lt => (0, 0x0a),
            CmpZ::Ge => (1, 0x08),
            CmpZ::Le => (1, 0x09),
        };
        self.emit(0x0e20_0800 | arr.q() | (u << 29) | (arr.size() << 22) | (opcode << 12) | rr(vd, vn))
    }

    /// `fcmeq/fcmgt/fcmge/fcmlt/fcmle vd.T, vn.T, #0.0`.
    pub fn vfcmp_zero(&mut self, op: CmpZ, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let (u, opcode) = match op {
            CmpZ::Gt => (0, 0x0c),
            CmpZ::Eq => (0, 0x0d),
            CmpZ::Lt => (0, 0x0e),
            CmpZ::Ge => (1, 0x0c),
            CmpZ::Le => (1, 0x0d),
        };
        self.fp_misc(u, 1, opcode, arr, vd, vn)
    }

    /// `scvtf vd.T, vn.T` (signed integer lanes to float).
    pub fn scvtf_v(&mut self, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        self.fp_misc(0, 0, 0x1d, arr, vd, vn)
    }

    /// `ucvtf vd.T, vn.T`.
    pub fn ucvtf_v(&mut self, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        self.fp_misc(1, 0, 0x1d, arr, vd, vn)
    }

    /// `fcvt{n,a,p,m,z}{s,u} vd.T, vn.T` (float lanes to integer lanes, saturating).
    pub fn fcvt_v(&mut self, rnd: FcvtRound, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let (u, a, opcode) = match rnd {
            FcvtRound::Ns => (0, 0, 0x1a),
            FcvtRound::Nu => (1, 0, 0x1a),
            FcvtRound::Ms => (0, 0, 0x1b),
            FcvtRound::Mu => (1, 0, 0x1b),
            FcvtRound::As => (0, 0, 0x1c),
            FcvtRound::Au => (1, 0, 0x1c),
            FcvtRound::Ps => (0, 1, 0x1a),
            FcvtRound::Pu => (1, 1, 0x1a),
            FcvtRound::Zs => (0, 1, 0x1b),
            FcvtRound::Zu => (1, 1, 0x1b),
        };
        self.fp_misc(u, a, opcode, arr, vd, vn)
    }

    /// `fcvtl/fcvtl2 vd, vn.T`: widens `4h`/`8h` to `4s` or `2s`/`4s` to `2d` (`src` is the
    /// narrow arrangement; the `2` form is the 128-bit one).
    pub fn fcvtl(&mut self, src: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let sz = fcvt_narrow_sz(src)?;
        self.emit(0x0e21_7800 | src.q() | sz | rr(vd, vn))
    }

    /// `fcvtn/fcvtn2 vd.T, vn`: narrows `4s` to `4h`/`8h` or `2d` to `2s`/`4s` (`dst` is the
    /// narrow arrangement; the `2` form writes the upper half and is the 128-bit one).
    pub fn fcvtn(&mut self, dst: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let sz = fcvt_narrow_sz(dst)?;
        self.emit(0x0e21_6800 | dst.q() | sz | rr(vd, vn))
    }

    // ---- widening and narrowing ----

    /// `sxtl/uxtl vd.Tw, vn.T` and the `2` forms (`sshll/ushll #0`); `src` is the narrow
    /// arrangement.
    pub fn vwiden(&mut self, sign: Sign, src: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        self.vshll(sign, src, vd, vn, 0)
    }

    /// `sxtl/sxtl2`.
    pub fn sxtl(&mut self, src: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        self.vwiden(Sign::Signed, src, vd, vn)
    }

    /// `uxtl/uxtl2`.
    pub fn uxtl(&mut self, src: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        self.vwiden(Sign::Unsigned, src, vd, vn)
    }

    /// `sshll/ushll` (and `2`) `vd.Tw, vn.T, #shift` with `shift` in `0..lane bits`.
    pub fn vshll(&mut self, sign: Sign, src: Arr, vd: VReg, vn: VReg, shift: u32) -> Result<(), EncError> {
        src.require(BHS, "widening needs 8-, 16- or 32-bit source lanes")?;
        let esize = src.lane().bits();
        range(i64::from(shift), 0, i64::from(esize - 1), 3)?;
        self.emit(0x0f00_a400 | src.q() | (sign.u() << 29) | ((esize + shift) << 16) | rr(vd, vn))
    }

    /// `xtn/sqxtn/uqxtn/sqxtun vd.T, vn.Tw` and the `2` forms; `dst` is the narrow arrangement.
    pub fn vnarrow(&mut self, op: Narrow, dst: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        dst.require(BHS, "narrowing needs 8-, 16- or 32-bit destination lanes")?;
        let (u, opcode) = match op {
            Narrow::Xtn => (0, 0x12),
            Narrow::Sqxtn => (0, 0x14),
            Narrow::Uqxtn => (1, 0x14),
            Narrow::Sqxtun => (1, 0x12),
        };
        self.emit(0x0e20_0800 | dst.q() | (u << 29) | (dst.size() << 22) | (opcode << 12) | rr(vd, vn))
    }

    /// `shrn/rshrn vd.T, vn.Tw, #shift` (and `2`); `shift` in `1..=lane bits` of `dst`.
    pub fn vshrn(&mut self, round: bool, dst: Arr, vd: VReg, vn: VReg, shift: u32) -> Result<(), EncError> {
        dst.require(BHS, "narrowing needs 8-, 16- or 32-bit destination lanes")?;
        let esize = dst.lane().bits();
        range(i64::from(shift), 1, i64::from(esize), 4)?;
        let opcode = if round { 0x11 } else { 0x10 };
        self.emit(0x0f00_0400 | dst.q() | ((2 * esize - shift) << 16) | (opcode << 11) | rr(vd, vn))
    }

    /// `addhn/addhn2 vd.T, vn.Tw, vm.Tw` (high half of the sum); `dst` is the narrow arrangement.
    pub fn addhn(&mut self, dst: Arr, vd: VReg, vn: VReg, vm: VReg) -> Result<(), EncError> {
        dst.require(BHS, "narrowing needs 8-, 16- or 32-bit destination lanes")?;
        self.emit(0x0e20_4000 | dst.q() | (dst.size() << 22) | rrr(vd, vn, vm))
    }

    /// `smull/umull vd.Tw, vn.T, vm.T` and the `2` forms; `src` is the narrow arrangement.
    pub fn vmull(&mut self, sign: Sign, src: Arr, vd: VReg, vn: VReg, vm: VReg) -> Result<(), EncError> {
        src.require(BHS, "widening multiply needs 8-, 16- or 32-bit source lanes")?;
        self.emit(0x0e20_c000 | src.q() | (sign.u() << 29) | (src.size() << 22) | rrr(vd, vn, vm))
    }

    // ---- reductions ----

    /// `addv/saddlv/uaddlv/smaxv/umaxv/sminv/uminv/fmaxv/fminv/fmaxnmv/fminnmv <scalar>, vn.T`.
    pub fn vacross(&mut self, op: Across, arr: Arr, vd: VReg, vn: VReg) -> Result<(), EncError> {
        let (u, o1, opcode) = match op {
            Across::Addv => (0, 0, 0x1b),
            Across::Saddlv => (0, 0, 0x03),
            Across::Uaddlv => (1, 0, 0x03),
            Across::Smaxv => (0, 0, 0x0a),
            Across::Umaxv => (1, 0, 0x0a),
            Across::Sminv => (0, 0, 0x1a),
            Across::Uminv => (1, 0, 0x1a),
            Across::FmaxNmv => (1, 0, 0x0c),
            Across::FminNmv => (1, 1, 0x0c),
            Across::Fmaxv => (1, 0, 0x0f),
            Across::Fminv => (1, 1, 0x0f),
        };
        let fp = matches!(op, Across::Fmaxv | Across::Fminv | Across::FmaxNmv | Across::FminNmv);
        let size = if fp {
            if arr != Arr::S4 {
                return Err(EncError::Invalid("floating-point reduction needs 4s"));
            }
            o1 << 1
        } else {
            if arr == Arr::S2 || arr == Arr::D2 {
                return Err(EncError::Invalid("reduction needs 8b, 16b, 4h, 8h or 4s"));
            }
            arr.size()
        };
        self.emit(0x0e30_0800 | arr.q() | (u << 29) | (size << 22) | (opcode << 12) | rr(vd, vn))
    }

    /// `addp dd, vn.2d`.
    pub fn addp_scalar(&mut self, vd: VReg, vn: VReg) -> Result<(), EncError> {
        self.emit(0x5ef1_b800 | rr(vd, vn))
    }

    /// `faddp sd, vn.2s` / `faddp dd, vn.2d`.
    pub fn faddp_scalar(&mut self, fp: Fp, vd: VReg, vn: VReg) -> Result<(), EncError> {
        self.emit(0x7e30_d800 | fp.ftype() | rr(vd, vn))
    }

    // ---- shifts ----

    /// `shl/sshr/ushr/sli/sri vd.T, vn.T, #imm`.
    pub fn vshift(&mut self, op: VShift, arr: Arr, vd: VReg, vn: VReg, imm: u32) -> Result<(), EncError> {
        let esize = arr.lane().bits();
        let (u, opcode, left) = match op {
            VShift::Shl => (0, 0x0a, true),
            VShift::Sshr => (0, 0x00, false),
            VShift::Ushr => (1, 0x00, false),
            VShift::Sli => (1, 0x0a, true),
            VShift::Sri => (1, 0x08, false),
        };
        let field = if left {
            range(i64::from(imm), 0, i64::from(esize - 1), 6)?;
            esize + imm
        } else {
            range(i64::from(imm), 1, i64::from(esize), 7)?;
            2 * esize - imm
        };
        self.emit(0x0f00_0400 | arr.q() | (u << 29) | (field << 16) | (opcode << 11) | rr(vd, vn))
    }

    // ---- element moves ----

    /// `dup vd.T, wn/xn` (`xn` for `2d`).
    pub fn dup_gpr(&mut self, arr: Arr, vd: VReg, rn: Reg) -> Result<(), EncError> {
        let n = zr(rn)?;
        self.emit(0x0e00_0c00 | arr.q() | (1 << (16 + arr.size())) | (n << 5) | v(vd))
    }

    /// `dup vd.T, vn.<lane>[idx]`; the lane size comes from `arr`.
    pub fn dup_elem(&mut self, arr: Arr, vd: VReg, vn: VReg, idx: u32) -> Result<(), EncError> {
        let imm5 = arr.lane().imm5(idx)?;
        self.emit(0x0e00_0400 | arr.q() | (imm5 << 16) | rr(vd, vn))
    }

    /// `dup bd/hd/sd/dd, vn.<lane>[idx]` (`mov` of a lane to a scalar register).
    pub fn dup_scalar(&mut self, lane: Lane, vd: VReg, vn: VReg, idx: u32) -> Result<(), EncError> {
        let imm5 = lane.imm5(idx)?;
        self.emit(0x5e00_0400 | (imm5 << 16) | rr(vd, vn))
    }

    /// `ins vd.<lane>[idx], wn/xn` (`mov`; `xn` for `d`).
    pub fn ins_gpr(&mut self, lane: Lane, vd: VReg, idx: u32, rn: Reg) -> Result<(), EncError> {
        let n = zr(rn)?;
        let imm5 = lane.imm5(idx)?;
        self.emit(0x4e00_1c00 | (imm5 << 16) | (n << 5) | v(vd))
    }

    /// `ins vd.<lane>[dst_idx], vn.<lane>[src_idx]` (`mov`).
    pub fn ins_elem(
        &mut self,
        lane: Lane,
        vd: VReg,
        dst_idx: u32,
        vn: VReg,
        src_idx: u32,
    ) -> Result<(), EncError> {
        let imm5 = lane.imm5(dst_idx)?;
        range(i64::from(src_idx), 0, i64::from(15 >> (lane as u32)), 4)?;
        self.emit(0x6e00_0400 | (imm5 << 16) | ((src_idx << (lane as u32)) << 11) | rr(vd, vn))
    }

    /// `umov wd, vn.<lane>[idx]` (`b`, `h`, `s`) or `umov xd, vn.d[idx]`; both print as `mov`
    /// for `s` and `d`.
    pub fn umov(&mut self, lane: Lane, rd: Reg, vn: VReg, idx: u32) -> Result<(), EncError> {
        let d = zr(rd)?;
        let imm5 = lane.imm5(idx)?;
        let q = if lane == Lane::D { 1 << 30 } else { 0 };
        self.emit(0x0e00_3c00 | q | (imm5 << 16) | (v(vn) << 5) | d)
    }

    /// `smov wd/xd, vn.<lane>[idx]` (`b`, `h` into either width; `s` into `xd`).
    pub fn smov(&mut self, lane: Lane, w: Width, rd: Reg, vn: VReg, idx: u32) -> Result<(), EncError> {
        let d = zr(rd)?;
        if lane == Lane::D || (lane == Lane::S && w == Width::W) {
            return Err(EncError::Invalid("smov must widen"));
        }
        let imm5 = lane.imm5(idx)?;
        let q = if w == Width::X { 1 << 30 } else { 0 };
        self.emit(0x0e00_2c00 | q | (imm5 << 16) | (v(vn) << 5) | d)
    }

    // ---- immediates ----

    fn modified_imm(
        &mut self,
        invert: bool,
        arr: Arr,
        vd: VReg,
        imm8: u8,
        shift: ImmShift,
    ) -> Result<(), EncError> {
        let cmode = match (arr.lane(), shift) {
            (Lane::B, ImmShift::Lsl(0)) if !invert => 0xe,
            (Lane::H, ImmShift::Lsl(0)) => 0x8,
            (Lane::H, ImmShift::Lsl(8)) => 0xa,
            (Lane::S, ImmShift::Lsl(s @ (0 | 8 | 16 | 24))) => u32::from(s) / 4,
            (Lane::S, ImmShift::Msl(8)) => 0xc,
            (Lane::S, ImmShift::Msl(16)) => 0xd,
            _ => return Err(EncError::Invalid("no movi/mvni form for this arrangement and shift")),
        };
        self.emit(
            0x0f00_0400
                | arr.q()
                | (u32::from(invert) << 29)
                | (cmode << 12)
                | imm8_fields(u32::from(imm8))
                | v(vd),
        )
    }

    /// `movi vd.T, #imm8{, shift}`: `8b`/`16b` (no shift), `4h`/`8h`, `2s`/`4s`.
    pub fn movi(&mut self, arr: Arr, vd: VReg, imm8: u8, shift: ImmShift) -> Result<(), EncError> {
        self.modified_imm(false, arr, vd, imm8, shift)
    }

    /// `mvni vd.T, #imm8{, shift}`: `4h`/`8h`, `2s`/`4s`.
    pub fn mvni(&mut self, arr: Arr, vd: VReg, imm8: u8, shift: ImmShift) -> Result<(), EncError> {
        self.modified_imm(true, arr, vd, imm8, shift)
    }

    /// `movi vd.2d, #imm` where every byte of `imm` is `0x00` or `0xff`.
    pub fn movi_2d(&mut self, vd: VReg, imm: u64) -> Result<(), EncError> {
        let mut imm8 = 0u32;
        for i in 0..8 {
            match (imm >> (8 * i)) & 0xff {
                0 => {}
                0xff => imm8 |= 1 << i,
                _ => return Err(EncError::Invalid("movi 2d bytes must be 0x00 or 0xff")),
            }
        }
        self.emit(0x6f00_e400 | imm8_fields(imm8) | v(vd))
    }

    /// `fmov vd.T, #imm` (`2s`, `4s`, `2d`) for the values `fmov_imm` accepts.
    pub fn fmov_vec(&mut self, arr: Arr, vd: VReg, val: f64) -> Result<(), EncError> {
        arr.fsz()?;
        let imm8 = encode_fp8(val).ok_or(EncError::ImmRange { imm: val as i64, bits: 8 })?;
        let op = u32::from(arr == Arr::D2) << 29;
        self.emit(0x0f00_f400 | arr.q() | op | imm8_fields(imm8) | v(vd))
    }

    // ---- loads and stores ----

    /// Adds the base register and post-index fields (`Rm` of 31 is the immediate form).
    fn vec_addr(base_word: u32, addr: VecAddr) -> Result<u32, EncError> {
        match addr {
            VecAddr::Base(rn) => Ok(base_word | (sp(rn)? << 5)),
            VecAddr::PostImm(rn) => Ok(base_word | (1 << 23) | (0x1f << 16) | (sp(rn)? << 5)),
            VecAddr::PostReg { base, index } => {
                Ok(base_word | (1 << 23) | (gpr(index)? << 16) | (sp(base)? << 5))
            }
        }
    }

    fn ld1_st1(&mut self, load: bool, arr: Arr, first: VReg, n: u8, addr: VecAddr) -> Result<(), EncError> {
        let opcode = match n {
            1 => 0x7,
            2 => 0xa,
            3 => 0x6,
            4 => 0x2,
            _ => return Err(EncError::ImmRange { imm: i64::from(n), bits: 2 }),
        };
        let w =
            0x0c00_0000 | arr.q() | (u32::from(load) << 22) | (opcode << 12) | (arr.size() << 10) | v(first);
        let w = Asm::vec_addr(w, addr)?;
        self.emit(w)
    }

    /// `ld1 {vt.T, ...}, addr` with `n` (1..=4) consecutive registers from `first`.
    pub fn ld1(&mut self, arr: Arr, first: VReg, n: u8, addr: VecAddr) -> Result<(), EncError> {
        self.ld1_st1(true, arr, first, n, addr)
    }

    /// `st1 {vt.T, ...}, addr`.
    pub fn st1(&mut self, arr: Arr, first: VReg, n: u8, addr: VecAddr) -> Result<(), EncError> {
        self.ld1_st1(false, arr, first, n, addr)
    }

    /// `ld1r {vt.T}, addr` (load one element and replicate it).
    pub fn ld1r(&mut self, arr: Arr, vt: VReg, addr: VecAddr) -> Result<(), EncError> {
        let w = 0x0d40_c000 | arr.q() | (arr.size() << 10) | v(vt);
        let w = Asm::vec_addr(w, addr)?;
        self.emit(w)
    }

    /// `ldnp st1/dt1/qt1, ..., [base, #off]` (non-temporal pair).
    pub fn ldnp_fp(
        &mut self,
        vs: VecSize,
        vt1: VReg,
        vt2: VReg,
        base: Reg,
        off: i64,
    ) -> Result<(), EncError> {
        self.nt_pair(true, vs, vt1, vt2, base, off)
    }

    /// `stnp st1/dt1/qt1, ..., [base, #off]`.
    pub fn stnp_fp(
        &mut self,
        vs: VecSize,
        vt1: VReg,
        vt2: VReg,
        base: Reg,
        off: i64,
    ) -> Result<(), EncError> {
        self.nt_pair(false, vs, vt1, vt2, base, off)
    }

    fn nt_pair(
        &mut self,
        load: bool,
        vs: VecSize,
        vt1: VReg,
        vt2: VReg,
        base: Reg,
        off: i64,
    ) -> Result<(), EncError> {
        let w = Asm::pair_word(
            Asm::fp_pair_class(vs),
            1,
            vs.scale(),
            load,
            v(vt1),
            v(vt2),
            None,
            Addr::Off(base, off),
        )?;
        // Bits 24:23 are `10` for the offset form and `00` for the no-allocate pair.
        self.emit(w & !(1 << 24))
    }

    /// `prfm op, addr`: `[base, #off]` (scaled by 8, or unscaled `prfum`) or `[base, index{, ext}]`.
    pub fn prfm(&mut self, op: Prefetch, addr: Addr) -> Result<(), EncError> {
        if matches!(addr, Addr::Pre(..) | Addr::Post(..)) {
            return Err(EncError::Invalid("prfm has no writeback form"));
        }
        self.ls(Ls { size: 3, v: 0, opc: 2, scale: 3 }, op.code(), None, addr)
    }
}

/// `sz` (bit 22) of `fcvtl`/`fcvtn`: half↔single is 0, single↔double is 1.
fn fcvt_narrow_sz(narrow: Arr) -> Result<u32, EncError> {
    match narrow {
        Arr::H4 | Arr::H8 => Ok(0),
        Arr::S2 | Arr::S4 => Ok(1 << 22),
        _ => Err(EncError::Invalid("fcvtl/fcvtn narrow side is 4h, 8h, 2s or 4s")),
    }
}
