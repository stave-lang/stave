//! Machine classification of IR types: register scalars, fat pointers, memory aggregates.

use stave_ir::TypeId;
use stave_ir::layout::{Layout, Layouts, Shape};
use stave_ir::types::{FloatBits, IntBits, TypeKey, TypeTable};

use crate::enc::{Fp, MemSize, VecSize};

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

/// How a value of a type is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ty {
    Sc(Sc),
    /// Fat pointer: 16 bytes in memory (address, length).
    Fat,
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
        Some(TypeKey::Vec { .. } | TypeKey::Mask { .. }) => Ty::Bad("vector values"),
        Some(TypeKey::Param(_) | TypeKey::ConstParam(_)) => Ty::Bad("generic parameter type"),
        None => Ty::Bad("type id out of range"),
    }
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
