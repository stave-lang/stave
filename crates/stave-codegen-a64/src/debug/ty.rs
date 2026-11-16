//! Machine classification of IR types: register scalars, fat pointers, vectors and masks, memory
//! aggregates.

use stave_ir::TypeId;
use stave_ir::layout::{Layout, Layouts, Shape};
use stave_ir::simd::{self, LaneTy};
use stave_ir::types::{FloatBits, IntBits, Lanes, TypeKey, TypeTable};

use crate::enc::{Arr, Fp, Lane, MemSize, VecSize};

/// A scalar kept in one register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sc {
    Int {
        bits: u8,
        signed: bool,
    },
    Bool,
    Char,
    /// Thin pointer or fn value.
    Ptr,
    F32,
    F64,
}

impl Sc {
    pub(crate) const USIZE: Sc = Sc::Int { bits: 64, signed: false };
    pub(crate) const U32: Sc = Sc::Int { bits: 32, signed: false };

    pub(crate) fn is_fp(self) -> bool {
        matches!(self, Sc::F32 | Sc::F64)
    }

    pub(crate) fn fp(self) -> Fp {
        if self == Sc::F32 { Fp::S } else { Fp::D }
    }

    pub(crate) fn vsize(self) -> VecSize {
        if self == Sc::F32 { VecSize::S } else { VecSize::D }
    }

    /// Width in bits of the integer view (`bool` 1, `char` 32, pointers 64).
    pub(crate) fn bits(self) -> u8 {
        match self {
            Sc::Int { bits, .. } => bits,
            Sc::Bool => 1,
            Sc::Char | Sc::F32 => 32,
            Sc::Ptr | Sc::F64 => 64,
        }
    }

    pub(crate) fn signed(self) -> bool {
        matches!(self, Sc::Int { signed: true, .. })
    }

    /// Memory access size.
    pub(crate) fn msize(self) -> MemSize {
        match self {
            Sc::Bool => MemSize::B,
            Sc::Int { bits: 8, .. } => MemSize::B,
            Sc::Int { bits: 16, .. } => MemSize::H,
            Sc::Int { bits: 32, .. } | Sc::Char | Sc::F32 => MemSize::W,
            _ => MemSize::X,
        }
    }

    /// The `stave_ir::arith` cast class.
    pub(crate) fn cast_ty(self) -> stave_ir::arith::CastTy {
        use stave_ir::arith::{CastTy, IntTy};
        match self {
            Sc::Int { bits, signed } => CastTy::Int(IntTy::new(bits, signed)),
            Sc::Bool => CastTy::Bool,
            Sc::Char => CastTy::Char,
            Sc::Ptr => CastTy::Ptr,
            Sc::F32 => CastTy::Float(FloatBits::F32),
            Sc::F64 => CastTy::Float(FloatBits::F64),
        }
    }

    /// (min, max) of the integer view.
    pub(crate) fn range(self) -> (i128, i128) {
        let b = self.bits() as u32;
        if self.signed() { (-(1i128 << (b - 1)), (1i128 << (b - 1)) - 1) } else { (0, (1i128 << b) - 1) }
    }
}

/// What the lanes of a vector hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaneKind {
    Int {
        signed: bool,
    },
    Float,
    /// Mask lanes: all ones (set) or zero.
    Mask,
}

/// A vector or mask of at most 128 bits: `size` bytes (2, 4, 8, 16) of little-endian lanes in
/// lane order, in memory and in a V register alike. Bytes and register bits past `size` are
/// unspecified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VTy {
    pub size: u8,
    pub lane: Lane,
    pub kind: LaneKind,
}

impl VTy {
    pub(crate) fn lane_bytes(self) -> u32 {
        1 << (self.lane as u32)
    }

    pub(crate) fn lanes(self) -> u32 {
        u32::from(self.size) / self.lane_bytes()
    }

    /// The arrangement holding the vector: the 128-bit one for 16 bytes, else the 64-bit one.
    pub(crate) fn arr(self) -> Arr {
        if self.size == 16 { self.lane.arr() } else { half_arr(self.lane) }
    }

    pub(crate) fn signed(self) -> bool {
        matches!(self.kind, LaneKind::Int { signed: true })
    }

    pub(crate) fn is_float(self) -> bool {
        self.kind == LaneKind::Float
    }

    /// The scalar class of one lane (a mask lane counts as an unsigned int).
    pub(crate) fn sc(self) -> Sc {
        match (self.kind, self.lane) {
            (LaneKind::Float, Lane::S) => Sc::F32,
            (LaneKind::Float, _) => Sc::F64,
            _ => Sc::Int { bits: self.lane.bits() as u8, signed: self.signed() },
        }
    }

    /// The access size of the register holding the whole vector in its frame home.
    pub(crate) fn home_size(self) -> VecSize {
        if self.size == 16 { VecSize::Q } else { VecSize::D }
    }
}

/// The 64-bit arrangement of a lane size (`d` lanes have none: the 128-bit one).
pub(crate) fn half_arr(lane: Lane) -> Arr {
    match lane {
        Lane::B => Arr::B8,
        Lane::H => Arr::H4,
        Lane::S => Arr::S2,
        Lane::D => Arr::D2,
    }
}

/// The lane size of `bytes` (1, 2, 4, 8).
pub(crate) fn lane_of(bytes: u32) -> Lane {
    match bytes {
        1 => Lane::B,
        2 => Lane::H,
        4 => Lane::S,
        _ => Lane::D,
    }
}

/// How a value of a type is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ty {
    Sc(Sc),
    /// Fat pointer: 16 bytes in memory (address, length).
    Fat,
    /// Vector or mask, in its frame home (loaded into a V register per use).
    Vec(VTy),
    /// Aggregate in memory.
    Mem {
        size: u32,
        align: u32,
    },
    /// Not supported by this backend (reason).
    Bad(&'static str),
}

impl Ty {
    pub(crate) fn size(self) -> u32 {
        match self {
            Ty::Sc(s) => 1 << (s.msize() as u32),
            Ty::Fat => 16,
            Ty::Vec(v) => u32::from(v.size),
            Ty::Mem { size, .. } => size,
            Ty::Bad(_) => 0,
        }
    }
}

pub(crate) fn classify(types: &TypeTable, layouts: &Layouts, t: TypeId) -> Ty {
    match types.key(t) {
        Some(TypeKey::Int { signed, bits }) => Ty::Sc(Sc::Int {
            bits: match bits {
                IntBits::Size => 64,
                b => b.width(),
            },
            signed: *signed,
        }),
        Some(TypeKey::Float(FloatBits::F32)) => Ty::Sc(Sc::F32),
        Some(TypeKey::Float(FloatBits::F64)) => Ty::Sc(Sc::F64),
        Some(TypeKey::Float(_)) => Ty::Bad("f16/bf16 values"),
        Some(TypeKey::Bool) => Ty::Sc(Sc::Bool),
        Some(TypeKey::Char) => Ty::Sc(Sc::Char),
        Some(TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. }) => {
            if stave_ir::layout::is_unsized_pointee(types, *inner) {
                Ty::Fat
            } else {
                Ty::Sc(Sc::Ptr)
            }
        }
        // (code, environment), like a fat pointer (contracts/checked-ir.md "Closures")
        Some(TypeKey::FnType { .. }) => Ty::Fat,
        Some(TypeKey::Array { .. } | TypeKey::Tuple(_) | TypeKey::Adt { .. }) => {
            match layouts.get_layout(t) {
                Ok(l) if l.size <= u32::MAX as u64 / 2 && l.align <= 16 => {
                    Ty::Mem { size: l.size as u32, align: l.align.max(1) as u32 }
                }
                Ok(_) => Ty::Bad("aggregate too large or over-aligned"),
                Err(_) => Ty::Bad("aggregate without a layout"),
            }
        }
        Some(TypeKey::Str | TypeKey::Slice { .. }) => Ty::Bad("unsized value"),
        Some(TypeKey::Vec { .. } | TypeKey::Mask { .. }) => vector(types, t),
        Some(TypeKey::Param(_) | TypeKey::ConstParam(_)) => Ty::Bad("generic parameter type"),
        None => Ty::Bad("type id out of range"),
    }
}

/// The class of a vector or mask type.
fn vector(types: &TypeTable, t: TypeId) -> Ty {
    let (bits, n, kind) = match types.key(t) {
        Some(TypeKey::Vec { lane, n: Lanes::Fixed(n) }) => match simd::lane_ty(types, *lane) {
            Some(LaneTy::Int(it)) => (u32::from(it.bits), u32::from(*n), LaneKind::Int { signed: it.signed }),
            Some(LaneTy::Float(FloatBits::F32)) => (32, u32::from(*n), LaneKind::Float),
            Some(LaneTy::Float(FloatBits::F64)) => (64, u32::from(*n), LaneKind::Float),
            Some(LaneTy::Float(_)) => return Ty::Bad("f16/bf16 vector lanes"),
            None => return Ty::Bad("vector lane type"),
        },
        Some(TypeKey::Mask { width, n: Lanes::Fixed(n) }) => {
            (u32::from(*width), u32::from(*n), LaneKind::Mask)
        }
        _ => return Ty::Bad("sizeless vectors"),
    };
    if !simd::valid_lanes(n) || !matches!(bits, 8 | 16 | 32 | 64) {
        return Ty::Bad("vector shape");
    }
    if bits * n > 128 {
        return Ty::Bad("vectors wider than 128 bits");
    }
    Ty::Vec(VTy { size: (bits * n / 8) as u8, lane: lane_of(bits / 8), kind })
}

/// Pointee of a ref / raw pointer type.
pub(crate) fn pointee(types: &TypeTable, t: TypeId) -> Option<TypeId> {
    match types.key(t) {
        Some(TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. }) => Some(*inner),
        _ => None,
    }
}

/// `(none, some, payload)` of an `Option`-shaped enum.
pub(crate) fn option_shape(types: &TypeTable, t: TypeId) -> Option<(u32, u32, TypeId)> {
    types.option_shape(t)
}

/// Field `k` of a tuple/struct layout: (offset, type).
pub(crate) fn field(l: &Layout, k: i128) -> Option<(u64, TypeId)> {
    match &l.shape {
        Shape::Fields(fs) => usize::try_from(k).ok().and_then(|k| fs.get(k)).copied(),
        _ => None,
    }
}

/// Field `f` of variant `v` of an enum layout.
pub(crate) fn variant_field(l: &Layout, v: i128, f: i128) -> Option<(u64, TypeId)> {
    match &l.shape {
        Shape::Enum(vs) => usize::try_from(v)
            .ok()
            .and_then(|v| vs.get(v))
            .and_then(|fs| usize::try_from(f).ok().and_then(|f| fs.get(f)))
            .copied(),
        _ => None,
    }
}
