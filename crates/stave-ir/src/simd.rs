//! Vector and mask ops (contracts/simd-ops.md "Operation table") and their lane semantics.
//!
//! One [`VecOp`] per portable operation, printed `vec.<name>` (vectors) or `mask.<name>` (masks).
//! In a body a vector op is the instruction op [`crate::Op::Vec`]; the checked-IR printer prints
//! its name.
//! Operands and result types (V = `vec L n`, M = `mask W n` with W = bits of L, `s` = a reference to
//! a slice or array of L, `i` a `usize` index):
//!
//! | Name | Operands | Result | Semantics |
//! |---|---|---|---|
//! | `vec.splat` | `x: L` | V | every lane `x` |
//! | `vec.add sub mul div` | `a b` | V (float) | IEEE per lane |
//! | `vec.neg abs copysign` | `a` / `a` / `a s` | V (float) | bit-exact sign flip / clear / copy |
//! | `vec.sqrt floor ceil round trunc` | `a` | V (float) | per lane; `round` = ties to even |
//! | `vec.fma` | `a b c` | V (float) | `a * b + c` with one rounding |
//! | `vec.min max` | `a b` | V | float: IEEE minimum / maximum (NaN propagates, -0 < +0); int: by value |
//! | `vec.min_num max_num` | `a b` | V (float) | IEEE minimumNumber / maximumNumber (a NaN lane is missing) |
//! | `vec.min_x86 max_x86` | `a b` | V (float) | `a < b ? a : b` / `a > b ? a : b` (else `b`) |
//! | `vec.add.wrap sub.wrap mul.wrap` | `a b` | V (int) | wrapping |
//! | `vec.add.sat sub.sat mul.sat` | `a b` | V (int) | saturating |
//! | `vec.checked_add checked_sub checked_mul` | `a b` | `Option<V>` | `Some` iff no lane overflows |
//! | `vec.div_lanes` † / `vec.div_const` † | `a b` / `a K` | V (int) | truncating division per lane; a zero divisor traps `DivByZero`, `MIN / -1` traps `Overflow` (lowest lane first) |
//! | `vec.abs_wrap popcount clz` | `a` | V (int; `abs_wrap` signed) | `|MIN|` = MIN; bit counts within the lane |
//! | `vec.and or xor andnot`, `vec.not` | `a b`, `a` | V (int) | bitwise; `andnot` = `a & ~b` |
//! | `vec.shl shr` † / `vec.shl.wrap` | `a amount` | V (int) | amount: V or an int scalar; an amount ≥ bits traps `Overflow` / is masked by bits-1 |
//! | `vec.eq ne lt le gt ge` | `a b` | M | ints by signedness, floats IEEE (unordered: only `ne`) |
//! | `vec.reduce_add reduce_add_wrap reduce_min reduce_max reduce_and reduce_or` | `v` | L | fixed tree: lane i with lane i+N/2 until one lane remains; `reduce_add` float lanes, `reduce_add_wrap` int lanes (wrapping) |
//! | `vec.reduce_add_ordered` | `v init` | L (float) | `((init + v[0]) + v[1]) + …` |
//! | `vec.scan_add scan_add_wrap` | `v` | V | inclusive prefix sum, lane by lane from lane 0; `scan_add` float lanes, `scan_add_wrap` int lanes (wrapping) |
//! | `vec.cast cast.wrap cast.sat cast.round cast.try` | `v` | V' (`Option<V'>` for `try`) | the scalar cast per lane; `try` is `Some` iff every lane converts |
//! | `vec.bitcast` | `v` | V' | same bytes (little-endian lanes), equal total width |
//! | `vec.split` / `vec.join` | `v` / `a b` | `(H, H)` / V | halves: lanes `0..n/2` and `n/2..n` |
//! | `vec.widen_lo widen_hi` | `v` | `vec L2 n/2` | low / high half converted exactly to the double-width lane |
//! | `vec.narrow_wrap narrow_sat` | `a b` (`vec L2 n`) | `vec L 2n` | lanes of `a` then `b`, truncated / saturated to L |
//! | `vec.permute` | `v k0 … k(m-1)` | `vec L m` | lane j = `v[kj]` (constants < n) |
//! | `vec.lookup` | `tbl idx` | V | lane j = `tbl[idx[j]]`, 0 if `idx[j]` ≥ n |
//! | `vec.compress` / `vec.expand` | `v m` | V | active lanes packed low / low lanes spread to active positions; other lanes 0 |
//! | `vec.load` / `vec.store` | `s i` / `s i v` | V / — | lanes `s[i..i+n]`; no check of their own (a dominating `check.bounds i lim`, `lim = sub.sat len n-1`, or a chunk-loop proof guards them) |
//! | `vec.load_aligned` † / `vec.store_aligned` † | `s i` / `s i v` | V / — | as `load` / `store`; `i % n != 0` traps `Bounds` |
//! | `vec.load_masked` † / `vec.store_masked` † | `s i m` / `s i v m` | V / — | active lanes only (inactive lanes read 0, are never written); an active lane at `i + k ≥ len` traps `Bounds` |
//! | `vec.gather` † / `vec.scatter` † | `s idx` / `s idx v` | V / — | lane j at `s[idx[j]]`, `idx` unsigned lanes of any width (zero-extended); any index ≥ len traps `Bounds` before any lane is written; duplicate indices: the highest lane wins |
//! | `mask.and or xor`, `mask.not` | `a b`, `a` | M | lane-wise logic |
//! | `mask.select` | `m a b` | V or M | lanes of `a` where `m` is set, else `b` (both evaluated) |
//! | `mask.to_bits` | `m` | `u8 u16 u32 u64` | bit j = lane j, smallest type holding n bits |
//! | `mask.to_vec` | `m` | `vec iW n` / `vec uW n` | all ones where set, else 0 |
//! | `mask.first_n` | `k: usize` | M | lanes `j < k` set |
//! | `mask.rewiden` | `m` | `mask W2 n` | same lanes |
//! | `mask.all` / `mask.none` | — | M | every lane set / clear |
//!
//! Floats are IEEE 754 binary16/32/64 and bfloat16, round to nearest even, no flush to zero, no
//! contraction. A NaN produced by an arithmetic lane op is quiet (sign and payload unspecified);
//! `neg abs copysign select bitcast`, loads and stores are bit-exact.

use crate::arith::{self, CastTy, IntTy};
use crate::ir::{CastKind, Op, PanicKind};
use crate::types::{FloatBits, IntBits, Lanes, TypeId, TypeKey, TypeTable};

/// A vector or mask op. See the module doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VecOp {
    Splat,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Copysign,
    Sqrt,
    Floor,
    Ceil,
    Round,
    Trunc,
    Fma,
    Min,
    Max,
    MinNum,
    MaxNum,
    MinX86,
    MaxX86,
    AddWrap,
    SubWrap,
    MulWrap,
    AddSat,
    SubSat,
    MulSat,
    CheckedAdd,
    CheckedSub,
    CheckedMul,
    DivLanes,
    DivConst,
    AbsWrap,
    Popcount,
    Clz,
    And,
    Or,
    Xor,
    Andnot,
    Not,
    Shl,
    Shr,
    ShlWrap,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    ReduceAdd,
    ReduceAddWrap,
    ReduceMin,
    ReduceMax,
    ReduceAnd,
    ReduceOr,
    ReduceAddOrdered,
    ScanAdd,
    ScanAddWrap,
    Cast(CastKind),
    Bitcast,
    Split,
    Join,
    WidenLo,
    WidenHi,
    NarrowWrap,
    NarrowSat,
    Permute,
    Lookup,
    Compress,
    Expand,
    Load,
    Store,
    LoadAligned,
    StoreAligned,
    LoadMasked,
    StoreMasked,
    Gather,
    Scatter,
    MaskAnd,
    MaskOr,
    MaskXor,
    MaskNot,
    Select,
    ToBits,
    ToVec,
    FirstN,
    Rewiden,
    MaskAll,
    MaskNone,
}

impl VecOp {
    /// Every op, in declaration order (tests and tables iterate it).
    pub const ALL: [VecOp; 92] = {
        use VecOp::*;
        [
            Splat,
            Add,
            Sub,
            Mul,
            Div,
            Neg,
            Abs,
            Copysign,
            Sqrt,
            Floor,
            Ceil,
            Round,
            Trunc,
            Fma,
            Min,
            Max,
            MinNum,
            MaxNum,
            MinX86,
            MaxX86,
            AddWrap,
            SubWrap,
            MulWrap,
            AddSat,
            SubSat,
            MulSat,
            CheckedAdd,
            CheckedSub,
            CheckedMul,
            DivLanes,
            DivConst,
            AbsWrap,
            Popcount,
            Clz,
            And,
            Or,
            Xor,
            Andnot,
            Not,
            Shl,
            Shr,
            ShlWrap,
            Eq,
            Ne,
            Lt,
            Le,
            Gt,
            Ge,
            ReduceAdd,
            ReduceAddWrap,
            ReduceMin,
            ReduceMax,
            ReduceAnd,
            ReduceOr,
            ReduceAddOrdered,
            ScanAdd,
            ScanAddWrap,
            Cast(CastKind::Exact),
            Cast(CastKind::Wrap),
            Cast(CastKind::Sat),
            Cast(CastKind::Round),
            Cast(CastKind::Try),
            Bitcast,
            Split,
            Join,
            WidenLo,
            WidenHi,
            NarrowWrap,
            NarrowSat,
            Permute,
            Lookup,
            Compress,
            Expand,
            Load,
            Store,
            LoadAligned,
            StoreAligned,
            LoadMasked,
            StoreMasked,
            Gather,
            Scatter,
            MaskAnd,
            MaskOr,
            MaskXor,
            MaskNot,
            Select,
            ToBits,
            ToVec,
            FirstN,
            Rewiden,
            MaskAll,
            MaskNone,
        ]
    };

    /// The printed op name.
    pub fn name(self) -> &'static str {
        use VecOp::*;
        match self {
            Splat => "vec.splat",
            Add => "vec.add",
            Sub => "vec.sub",
            Mul => "vec.mul",
            Div => "vec.div",
            Neg => "vec.neg",
            Abs => "vec.abs",
            Copysign => "vec.copysign",
            Sqrt => "vec.sqrt",
            Floor => "vec.floor",
            Ceil => "vec.ceil",
            Round => "vec.round",
            Trunc => "vec.trunc",
            Fma => "vec.fma",
            Min => "vec.min",
            Max => "vec.max",
            MinNum => "vec.min_num",
            MaxNum => "vec.max_num",
            MinX86 => "vec.min_x86",
            MaxX86 => "vec.max_x86",
            AddWrap => "vec.add.wrap",
            SubWrap => "vec.sub.wrap",
            MulWrap => "vec.mul.wrap",
            AddSat => "vec.add.sat",
            SubSat => "vec.sub.sat",
            MulSat => "vec.mul.sat",
            CheckedAdd => "vec.checked_add",
            CheckedSub => "vec.checked_sub",
            CheckedMul => "vec.checked_mul",
            DivLanes => "vec.div_lanes",
            DivConst => "vec.div_const",
            AbsWrap => "vec.abs_wrap",
            Popcount => "vec.popcount",
            Clz => "vec.clz",
            And => "vec.and",
            Or => "vec.or",
            Xor => "vec.xor",
            Andnot => "vec.andnot",
            Not => "vec.not",
            Shl => "vec.shl",
            Shr => "vec.shr",
            ShlWrap => "vec.shl.wrap",
            Eq => "vec.eq",
            Ne => "vec.ne",
            Lt => "vec.lt",
            Le => "vec.le",
            Gt => "vec.gt",
            Ge => "vec.ge",
            ReduceAdd => "vec.reduce_add",
            ReduceAddWrap => "vec.reduce_add_wrap",
            ReduceMin => "vec.reduce_min",
            ReduceMax => "vec.reduce_max",
            ReduceAnd => "vec.reduce_and",
            ReduceOr => "vec.reduce_or",
            ReduceAddOrdered => "vec.reduce_add_ordered",
            ScanAdd => "vec.scan_add",
            ScanAddWrap => "vec.scan_add_wrap",
            Cast(CastKind::Exact) => "vec.cast",
            Cast(CastKind::Wrap) => "vec.cast.wrap",
            Cast(CastKind::Sat) => "vec.cast.sat",
            Cast(CastKind::Round) => "vec.cast.round",
            Cast(CastKind::Try) => "vec.cast.try",
            Bitcast => "vec.bitcast",
            Split => "vec.split",
            Join => "vec.join",
            WidenLo => "vec.widen_lo",
            WidenHi => "vec.widen_hi",
            NarrowWrap => "vec.narrow_wrap",
            NarrowSat => "vec.narrow_sat",
            Permute => "vec.permute",
            Lookup => "vec.lookup",
            Compress => "vec.compress",
            Expand => "vec.expand",
            Load => "vec.load",
            Store => "vec.store",
            LoadAligned => "vec.load_aligned",
            StoreAligned => "vec.store_aligned",
            LoadMasked => "vec.load_masked",
            StoreMasked => "vec.store_masked",
            Gather => "vec.gather",
            Scatter => "vec.scatter",
            MaskAnd => "mask.and",
            MaskOr => "mask.or",
            MaskXor => "mask.xor",
            MaskNot => "mask.not",
            Select => "mask.select",
            ToBits => "mask.to_bits",
            ToVec => "mask.to_vec",
            FirstN => "mask.first_n",
            Rewiden => "mask.rewiden",
            MaskAll => "mask.all",
            MaskNone => "mask.none",
        }
    }

    /// The trap kinds this op can raise.
    pub fn trap_kinds(self) -> &'static [PanicKind] {
        use VecOp::*;
        match self {
            Shl | Shr => &[PanicKind::Overflow],
            DivLanes | DivConst => &[PanicKind::DivByZero, PanicKind::Overflow],
            LoadAligned | StoreAligned | LoadMasked | StoreMasked | Gather | Scatter => &[PanicKind::Bounds],
            _ => &[],
        }
    }

    /// Reads or writes memory.
    pub fn is_memory(self) -> bool {
        use VecOp::*;
        matches!(
            self,
            Load | Store | LoadAligned | StoreAligned | LoadMasked | StoreMasked | Gather | Scatter
        )
    }

    /// A store (no result).
    pub fn is_store(self) -> bool {
        matches!(self, VecOp::Store | VecOp::StoreAligned | VecOp::StoreMasked | VecOp::Scatter)
    }

    /// The scalar op of a lane-wise int op, for [`arith::int_binary`] / [`arith::int_shift`].
    pub fn scalar(self) -> Option<Op> {
        use VecOp::*;
        Some(match self {
            Add => Op::Add,
            Sub => Op::Sub,
            Mul => Op::Mul,
            Div | DivLanes | DivConst => Op::Div,
            AddWrap => Op::AddWrap,
            SubWrap => Op::SubWrap,
            MulWrap => Op::MulWrap,
            AddSat => Op::AddSat,
            SubSat => Op::SubSat,
            MulSat => Op::MulSat,
            CheckedAdd => Op::Add,
            CheckedSub => Op::Sub,
            CheckedMul => Op::Mul,
            And => Op::And,
            Or => Op::Or,
            Xor => Op::Xor,
            Shl => Op::Shl,
            Shr => Op::Shr,
            ShlWrap => Op::ShlWrap,
            Eq => Op::Eq,
            Ne => Op::Ne,
            Lt => Op::Lt,
            Le => Op::Le,
            Gt => Op::Gt,
            Ge => Op::Ge,
            Popcount => Op::Popcount,
            Clz => Op::Clz,
            _ => return None,
        })
    }
}

/// The class of a lane type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneTy {
    Int(IntTy),
    Float(FloatBits),
}

impl LaneTy {
    pub fn bits(self) -> u32 {
        match self {
            LaneTy::Int(t) => t.bits as u32,
            LaneTy::Float(f) => f.width() as u32,
        }
    }

    pub fn cast_ty(self) -> CastTy {
        match self {
            LaneTy::Int(t) => CastTy::Int(t),
            LaneTy::Float(f) => CastTy::Float(f),
        }
    }

    /// Normalizes raw lane bits read from memory.
    pub fn norm(self, raw: u64) -> u64 {
        match self {
            LaneTy::Int(t) => t.norm(raw),
            LaneTy::Float(f) => raw & width_mask(f.width() as u32),
        }
    }
}

fn width_mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

fn sign_bit(f: FloatBits) -> u64 {
    1u64 << (f.width() - 1)
}

fn is_nan(f: FloatBits, a: u64) -> bool {
    arith::to_f64(f, a).is_nan()
}

/// The quiet bit of a NaN of format `f`.
fn quiet_bit(f: FloatBits) -> u64 {
    match f {
        FloatBits::F64 => 1 << 51,
        FloatBits::F32 => 1 << 22,
        FloatBits::F16 => 1 << 9,
        FloatBits::B16 => 1 << 6,
    }
}

/// `a`, quieted if it is a NaN (arithmetic results never carry a signaling NaN).
pub fn quiet(f: FloatBits, a: u64) -> u64 {
    if is_nan(f, a) { a | quiet_bit(f) } else { a }
}

/// Rounds an exact-or-f64 result to format `f`. Only used where the f64 result is either exact or
/// a correctly rounded `+ - * / sqrt` of narrower operands (double rounding through f64 is
/// innocuous for binary16, bfloat16 and binary32 there).
fn from_f64(f: FloatBits, x: f64) -> u64 {
    match f {
        FloatBits::F64 => x.to_bits(),
        FloatBits::F32 => (x as f32).to_bits() as u64,
        FloatBits::F16 => arith::f64_to_small(x, 5, 10) as u64,
        FloatBits::B16 => arith::f64_to_small(x, 8, 7) as u64,
    }
}

/// Lane-wise float unary op. `None`: not a float unary op.
pub fn float_unary(op: VecOp, f: FloatBits, a: u64) -> Option<u64> {
    Some(match op {
        VecOp::Neg => a ^ sign_bit(f),
        VecOp::Abs => a & !sign_bit(f),
        VecOp::Sqrt | VecOp::Floor | VecOp::Ceil | VecOp::Round | VecOp::Trunc => {
            let r = match f {
                FloatBits::F32 => {
                    let x = f32::from_bits(a as u32);
                    let y = match op {
                        VecOp::Sqrt => x.sqrt(),
                        VecOp::Floor => x.floor(),
                        VecOp::Ceil => x.ceil(),
                        VecOp::Round => x.round_ties_even(),
                        _ => x.trunc(),
                    };
                    y.to_bits() as u64
                }
                _ => {
                    let x = arith::to_f64(f, a);
                    let y = match op {
                        VecOp::Sqrt => x.sqrt(),
                        VecOp::Floor => x.floor(),
                        VecOp::Ceil => x.ceil(),
                        VecOp::Round => x.round_ties_even(),
                        _ => x.trunc(),
                    };
                    from_f64(f, y)
                }
            };
            quiet(f, r)
        }
        _ => return None,
    })
}

/// Lane-wise float binary op. `None`: not a float binary op.
pub fn float_binary(op: VecOp, f: FloatBits, a: u64, b: u64) -> Option<u64> {
    let (x, y) = (arith::to_f64(f, a), arith::to_f64(f, b));
    let neg = |v: u64| v & sign_bit(f) != 0;
    Some(match op {
        VecOp::Add | VecOp::Sub | VecOp::Mul | VecOp::Div => {
            let r = match f {
                FloatBits::F32 | FloatBits::F64 => arith::float_binary(op.scalar()?, f, a, b),
                _ => from_f64(
                    f,
                    match op {
                        VecOp::Add => x + y,
                        VecOp::Sub => x - y,
                        VecOp::Mul => x * y,
                        _ => x / y,
                    },
                ),
            };
            quiet(f, r)
        }
        VecOp::Copysign => (a & !sign_bit(f)) | (b & sign_bit(f)),
        VecOp::Min | VecOp::Max => {
            if x.is_nan() || y.is_nan() {
                return Some(quiet(f, if x.is_nan() { a } else { b }));
            }
            let min = op == VecOp::Min;
            if x < y {
                if min { a } else { b }
            } else if y < x {
                if min { b } else { a }
            } else if min == neg(a) {
                // equal: -0 is below +0
                a
            } else {
                b
            }
        }
        VecOp::MinNum | VecOp::MaxNum => {
            let m = if op == VecOp::MinNum { VecOp::Min } else { VecOp::Max };
            match (x.is_nan(), y.is_nan()) {
                (true, true) => quiet(f, a),
                (true, false) => b,
                (false, true) => a,
                _ => return float_binary(m, f, a, b),
            }
        }
        VecOp::MinX86 => quiet(f, if x < y { a } else { b }),
        VecOp::MaxX86 => quiet(f, if x > y { a } else { b }),
        _ => return None,
    })
}

/// `a * b + c` with one rounding. `None` for binary16 / bfloat16 lanes (not modelled).
pub fn float_fma(f: FloatBits, a: u64, b: u64, c: u64) -> Option<u64> {
    let r = match f {
        FloatBits::F32 => f32::from_bits(a as u32)
            .mul_add(f32::from_bits(b as u32), f32::from_bits(c as u32))
            .to_bits() as u64,
        FloatBits::F64 => f64::from_bits(a).mul_add(f64::from_bits(b), f64::from_bits(c)).to_bits(),
        _ => return None,
    };
    Some(quiet(f, r))
}

/// Lane-wise int op of two operands (no shifts, no traps except division). `Err` is the trap kind.
pub fn int_binary(op: VecOp, t: IntTy, a: u64, b: u64) -> Result<u64, PanicKind> {
    match op {
        VecOp::Min | VecOp::Max => {
            let (x, y) = (t.value(a), t.value(b));
            Ok(if (x <= y) == (op == VecOp::Min) { a } else { b })
        }
        VecOp::Andnot => Ok(t.norm(a & !b)),
        _ => match op.scalar() {
            Some(s) => arith::int_binary(s, t, a, b),
            None => Ok(0),
        },
    }
}

/// Lane-wise int unary op.
pub fn int_unary(op: VecOp, t: IntTy, a: u64) -> u64 {
    match op {
        VecOp::Not => arith::int_not(t, a),
        VecOp::AbsWrap => {
            let v = t.value(a);
            t.wrap(v.abs())
        }
        VecOp::Popcount | VecOp::Clz => arith::int_count(op.scalar().unwrap_or(Op::Popcount), t, a),
        _ => a,
    }
}

/// Lane compare.
pub fn compare(op: VecOp, lane: LaneTy, a: u64, b: u64) -> bool {
    let s = op.scalar().unwrap_or(Op::Eq);
    match lane {
        LaneTy::Int(t) => arith::int_cmp(s, t, a, b),
        LaneTy::Float(f) => arith::float_cmp(s, f, a, b),
    }
}

/// The add of reductions and scans: wrapping for ints, IEEE for floats. `None` for an
/// unsupported lane.
pub fn lane_add(lane: LaneTy, a: u64, b: u64) -> Option<u64> {
    match lane {
        LaneTy::Int(t) => Some(t.norm(a.wrapping_add(b))),
        LaneTy::Float(f) => float_binary(VecOp::Add, f, a, b),
    }
}

/// The fixed reduction tree: lane i combines with lane i + n/2 until one lane remains.
pub fn reduce_tree(lanes: &mut [u64], mut f: impl FnMut(u64, u64) -> Option<u64>) -> Option<u64> {
    let mut n = lanes.len();
    while n > 1 {
        let h = n / 2;
        for i in 0..h {
            lanes[i] = f(lanes[i], lanes[i + h])?;
        }
        n = h;
    }
    lanes.first().copied()
}

/// The reduction op of a lane: `reduce_add reduce_add_wrap reduce_min reduce_max reduce_and
/// reduce_or`. `None` when the op does not take the lane class.
pub fn reduce_step(op: VecOp, lane: LaneTy, a: u64, b: u64) -> Option<u64> {
    match (op, lane) {
        (VecOp::ReduceAdd, LaneTy::Float(_)) | (VecOp::ReduceAddWrap, LaneTy::Int(_)) => lane_add(lane, a, b),
        (VecOp::ReduceMin | VecOp::ReduceMax, LaneTy::Float(f)) => {
            float_binary(if op == VecOp::ReduceMin { VecOp::Min } else { VecOp::Max }, f, a, b)
        }
        (VecOp::ReduceMin | VecOp::ReduceMax, LaneTy::Int(t)) => {
            int_binary(if op == VecOp::ReduceMin { VecOp::Min } else { VecOp::Max }, t, a, b).ok()
        }
        (VecOp::ReduceAnd, LaneTy::Int(_)) => Some(a & b),
        (VecOp::ReduceOr, LaneTy::Int(_)) => Some(a | b),
        _ => None,
    }
}

/// The double-width lane of `lane` (`widen_lo` / `widen_hi`), if any.
pub fn wide_lane(lane: LaneTy) -> Option<LaneTy> {
    Some(match lane {
        LaneTy::Int(t) if t.bits < 64 => LaneTy::Int(IntTy::new(t.bits * 2, t.signed)),
        LaneTy::Float(FloatBits::F16 | FloatBits::B16) => LaneTy::Float(FloatBits::F32),
        LaneTy::Float(FloatBits::F32) => LaneTy::Float(FloatBits::F64),
        _ => return None,
    })
}

/// The mask lane count's `to_bits` width: the smallest of 8, 16, 32, 64 holding `n` bits.
pub fn to_bits_width(n: u32) -> u32 {
    match n {
        0..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

/// Splits lanes into little-endian bytes.
pub fn lanes_to_bytes(lanes: &[u64], bytes: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(lanes.len() * bytes as usize);
    for &l in lanes {
        out.extend_from_slice(&l.to_le_bytes()[..bytes as usize]);
    }
    out
}

/// Reassembles lanes of `lane` from little-endian bytes.
pub fn bytes_to_lanes(b: &[u8], lane: LaneTy) -> Vec<u64> {
    let w = (lane.bits() / 8) as usize;
    b.chunks(w.max(1))
        .map(|c| {
            let mut x = [0u8; 8];
            x[..c.len()].copy_from_slice(c);
            lane.norm(u64::from_le_bytes(x))
        })
        .collect()
}

/// A lane-count shape check used by the verifier and the interpreter: a supported fixed count.
pub fn valid_lanes(n: u32) -> bool {
    (2..=64).contains(&n) && n.is_power_of_two()
}

/// The lane class of a lane type (`isize`/`usize`, `bool`, `char` are not lane types).
pub fn lane_ty(types: &TypeTable, lane: TypeId) -> Option<LaneTy> {
    match types.key(lane)? {
        TypeKey::Int { signed, bits } if *bits != IntBits::Size => {
            Some(LaneTy::Int(IntTy::new(bits.width(), *signed)))
        }
        TypeKey::Float(f) => Some(LaneTy::Float(*f)),
        _ => None,
    }
}

/// A fixed-size vector type: (lane type, lane class, lane count).
pub fn vec_shape(types: &TypeTable, t: TypeId) -> Option<(TypeId, LaneTy, u32)> {
    match types.key(t)? {
        TypeKey::Vec { lane, n: Lanes::Fixed(n) } if valid_lanes(*n as u32) => {
            Some((*lane, lane_ty(types, *lane)?, *n as u32))
        }
        _ => None,
    }
}

/// A fixed-size mask type: (lane width in bits, lane count).
pub fn mask_shape(types: &TypeTable, t: TypeId) -> Option<(u32, u32)> {
    match types.key(t)? {
        TypeKey::Mask { width, n: Lanes::Fixed(n) }
            if valid_lanes(*n as u32) && matches!(width, 8 | 16 | 32 | 64) =>
        {
            Some((*width as u32, *n as u32))
        }
        _ => None,
    }
}

/// The memory operand of a vector load / store: a reference or raw pointer to a slice or an array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemOperand {
    pub elem: TypeId,
    /// The array length (`None`: a slice, length in the fat pointer).
    pub len: Option<u64>,
    pub mutable: bool,
    /// The slice's declared alignment (`&[T align A]`).
    pub align: Option<u32>,
}

pub fn mem_operand(types: &TypeTable, t: TypeId) -> Option<MemOperand> {
    let (mutable, inner) = match types.key(t)? {
        TypeKey::Ref { mutable, inner } | TypeKey::RawPtr { mutable, inner } => (*mutable, *inner),
        _ => return None,
    };
    match types.key(inner)? {
        TypeKey::Slice { elem, align } => Some(MemOperand { elem: *elem, len: None, mutable, align: *align }),
        TypeKey::Array { elem, len } => {
            Some(MemOperand { elem: *elem, len: Some(*len), mutable, align: None })
        }
        _ => None,
    }
}

/// The element pointer operand (`*T` / `*mut T`, `T` not a slice or array) that release passes
/// give a `vec.load` / `vec.store` in place of its slice: the slice's `slice.ptr`. Lane `i` is at
/// `p + i * size`, unchecked (the slice access's `check.bounds` stays a separate instruction).
/// Returns the element type and whether the pointer is mutable.
pub fn elem_ptr_operand(types: &TypeTable, t: TypeId) -> Option<(TypeId, bool)> {
    match types.key(t)? {
        TypeKey::RawPtr { mutable, inner }
            if !matches!(types.key(*inner)?, TypeKey::Slice { .. } | TypeKey::Array { .. }) =>
        {
            Some((*inner, *mutable))
        }
        _ => None,
    }
}

/// The first function (by id) whose body holds a vector or mask value or op: what a backend
/// without vector code generation cannot compile.
pub fn first_vector_func(m: &crate::ir::Module) -> Option<crate::ir::FuncId> {
    let simd: Vec<bool> =
        m.types.keys().iter().map(|k| matches!(k, TypeKey::Vec { .. } | TypeKey::Mask { .. })).collect();
    let is = |t: TypeId| simd.get(t.index()).copied().unwrap_or(false);
    m.funcs
        .iter()
        .position(|f| {
            f.params.iter().chain(f.ret.iter()).any(|&t| is(t))
                || f.body.as_ref().is_some_and(|b| {
                    b.values.iter().any(|v| is(v.ty))
                        || b.insts.op.iter().any(|op| matches!(op, crate::ir::Op::Vec(_)))
                })
        })
        .map(|i| crate::ir::FuncId(i as u32))
}
