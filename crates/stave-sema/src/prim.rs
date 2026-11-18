//! Built-in names (LANG.md §2, §4; prelude): the type names
//! (scalars, `str`, `never`, fixed and sizeless vectors and masks, the generic spellings `vec` /
//! `mask`), the built-in generic fns `size_of` / `align_of`, and `fence_stream`. They are not
//! declarations and not keywords: a use resolves to [`crate::Res::Prim`] or
//! [`crate::Res::Builtin`] and has no `names` dump line. Every module sees them, after its own
//! scope and the prelude.

/// A scalar type name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scalar {
    I8,
    I16,
    I32,
    I64,
    Isize,
    U8,
    U16,
    U32,
    U64,
    Usize,
    /// Lane-only.
    F16,
    /// Lane-only.
    Bf16,
    F32,
    F64,
    Bool,
    Char,
    Str,
}

impl Scalar {
    pub const ALL: [Scalar; 17] = [
        Scalar::I8,
        Scalar::I16,
        Scalar::I32,
        Scalar::I64,
        Scalar::Isize,
        Scalar::U8,
        Scalar::U16,
        Scalar::U32,
        Scalar::U64,
        Scalar::Usize,
        Scalar::F16,
        Scalar::Bf16,
        Scalar::F32,
        Scalar::F64,
        Scalar::Bool,
        Scalar::Char,
        Scalar::Str,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Scalar::I8 => "i8",
            Scalar::I16 => "i16",
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::Isize => "isize",
            Scalar::U8 => "u8",
            Scalar::U16 => "u16",
            Scalar::U32 => "u32",
            Scalar::U64 => "u64",
            Scalar::Usize => "usize",
            Scalar::F16 => "f16",
            Scalar::Bf16 => "bf16",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::Bool => "bool",
            Scalar::Char => "char",
            Scalar::Str => "str",
        }
    }

    /// May be a vector lane type (ints except `isize`/`usize`, and floats).
    pub fn is_lane(self) -> bool {
        !matches!(self, Scalar::Isize | Scalar::Usize | Scalar::Bool | Scalar::Char | Scalar::Str)
    }

    fn from_name(s: &str) -> Option<Scalar> {
        Scalar::ALL.into_iter().find(|t| t.name() == s)
    }
}

/// Vector or mask lane count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lanes {
    Fixed(u8),
    /// Sizeless (`f32xn`, SVE).
    Native,
}

/// A built-in type name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Prim {
    Scalar(Scalar),
    /// `TxN` / `Txn`.
    Vector {
        lane: Scalar,
        n: Lanes,
    },
    /// `maskWxN` / `maskWxn`.
    Mask {
        width: u8,
        n: Lanes,
    },
    /// `vec<T, N>` (generic spelling).
    VecGeneric,
    /// `mask<W, N>` (generic spelling).
    MaskGeneric,
    /// `never`: the return type of a fn that never returns (`-> never`).
    Never,
}

/// A built-in fn: the generic layout queries (const-evaluable), the stream fence, and the row
/// operations of `std.soa`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Builtin {
    /// `size_of::<T>() -> usize`.
    SizeOf,
    /// `align_of::<T>() -> usize`.
    AlignOf,
    /// `fence_stream()`: orders earlier streaming stores (an unsafe operation).
    FenceStream,
    /// `old(e)`: `e` evaluated at fn entry. A contextual name inside a contract clause only
    /// (resolution), never in the global table, so not in [`Builtin::ALL`].
    Old,
    /// `soa_row_size::<T>() -> usize`: the sum of the sizes of T's fields, one row of a
    /// `std.soa.Soa<T>`. This one and the two below resolve in module `std.soa` only.
    SoaRowSize,
    /// `soa_read::<T>(base: *const u8, cap: usize, i: usize) -> T` (unsafe): row `i` of the
    /// columns at `base` of capacity `cap` (contracts/intrinsics.md, `std.soa.Soa`).
    SoaRead,
    /// `soa_write::<T>(base: *mut u8, cap: usize, i: usize, x: T)` (unsafe): `x` as row `i`.
    SoaWrite,
}

impl Builtin {
    pub const ALL: [Builtin; 6] = [
        Builtin::SizeOf,
        Builtin::AlignOf,
        Builtin::FenceStream,
        Builtin::SoaRowSize,
        Builtin::SoaRead,
        Builtin::SoaWrite,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Builtin::SizeOf => "size_of",
            Builtin::AlignOf => "align_of",
            Builtin::FenceStream => "fence_stream",
            Builtin::Old => "old",
            Builtin::SoaRowSize => "soa_row_size",
            Builtin::SoaRead => "soa_read",
            Builtin::SoaWrite => "soa_write",
        }
    }

    /// The row built-ins of `std.soa`, resolved in that module only.
    pub fn soa_only(self) -> bool {
        matches!(self, Builtin::SoaRowSize | Builtin::SoaRead | Builtin::SoaWrite)
    }

    pub fn from_name(s: &str) -> Option<Builtin> {
        Builtin::ALL.into_iter().find(|b| b.name() == s)
    }
}

fn lanes(s: &str) -> Option<Lanes> {
    if s == "n" {
        return Some(Lanes::Native);
    }
    match s {
        "2" => Some(Lanes::Fixed(2)),
        "4" => Some(Lanes::Fixed(4)),
        "8" => Some(Lanes::Fixed(8)),
        "16" => Some(Lanes::Fixed(16)),
        "32" => Some(Lanes::Fixed(32)),
        "64" => Some(Lanes::Fixed(64)),
        _ => None,
    }
}

impl Prim {
    /// The built-in type spelled `s`, if any. Vector widths are not checked here (E0702 is a
    /// type rule).
    pub fn from_name(s: &str) -> Option<Prim> {
        if let Some(t) = Scalar::from_name(s) {
            return Some(Prim::Scalar(t));
        }
        match s {
            "vec" => return Some(Prim::VecGeneric),
            "mask" => return Some(Prim::MaskGeneric),
            "never" => return Some(Prim::Never),
            _ => {}
        }
        let (head, n) = s.rsplit_once('x')?;
        let n = lanes(n)?;
        if let Some(w) = head.strip_prefix("mask") {
            let width = match w {
                "8" => 8,
                "16" => 16,
                "32" => 32,
                "64" => 64,
                _ => return None,
            };
            return Some(Prim::Mask { width, n });
        }
        let lane = Scalar::from_name(head)?;
        if !lane.is_lane() {
            return None;
        }
        Some(Prim::Vector { lane, n })
    }
}

impl Prim {
    /// The spelling (`i32`, `f32x8`, `mask32xn`, `vec`).
    pub fn text(self) -> String {
        let lanes = |n: Lanes| match n {
            Lanes::Fixed(k) => k.to_string(),
            Lanes::Native => "n".to_string(),
        };
        match self {
            Prim::Scalar(s) => s.name().to_string(),
            Prim::Vector { lane, n } => format!("{}x{}", lane.name(), lanes(n)),
            Prim::Mask { width, n } => format!("mask{width}x{}", lanes(n)),
            Prim::VecGeneric => "vec".to_string(),
            Prim::MaskGeneric => "mask".to_string(),
            Prim::Never => "never".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(Prim::from_name("i32"), Some(Prim::Scalar(Scalar::I32)));
        assert_eq!(Prim::from_name("f32x8"), Some(Prim::Vector { lane: Scalar::F32, n: Lanes::Fixed(8) }));
        assert_eq!(Prim::from_name("f32xn"), Some(Prim::Vector { lane: Scalar::F32, n: Lanes::Native }));
        assert_eq!(Prim::from_name("mask32x8"), Some(Prim::Mask { width: 32, n: Lanes::Fixed(8) }));
        assert_eq!(Prim::from_name("bf16x4"), Some(Prim::Vector { lane: Scalar::Bf16, n: Lanes::Fixed(4) }));
        assert_eq!(Prim::from_name("vec"), Some(Prim::VecGeneric));
        assert_eq!(Prim::from_name("never"), Some(Prim::Never));
        assert_eq!(Builtin::from_name("align_of"), Some(Builtin::AlignOf));
        assert_eq!(Builtin::from_name("size_of").map(Builtin::name), Some("size_of"));
        for good in ["u8", "i64x8", "f32xn", "mask8x64", "vec", "mask", "never"] {
            assert_eq!(Prim::from_name(good).map(Prim::text).as_deref(), Some(good));
        }
        for bad in ["i128", "u8x3", "boolx4", "usizex2", "mask12x4", "x4", "f32x", "mask32", "I32", "neverx4"]
        {
            assert_eq!(Prim::from_name(bad), None, "{bad}");
        }
    }
}
