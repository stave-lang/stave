//! Memory layout of IR types for the interpreter's simulated memory (and a default for backends
//! until a layout contract exists).
//!
//! Rules: ints, floats, `bool` (1 byte), `char` (4 bytes) are naturally aligned. A ref or raw
//! pointer is 8 bytes, or 16 (`ptr, len`) when the pointee is a slice or `str` (fat). Fn values are
//! 16 bytes (`code, environment`, shape `Fat`; contracts/checked-ir.md "Closures"). Tuples and structs: fields in declaration order, each at the next offset aligned to
//! it, size rounded up to the max alignment (C rules). Arrays: `len` elements at stride = elem
//! size. Enums: a `u32` tag (the variant index) at offset 0, then each variant's fields laid out
//! like a struct starting at offset 4; size = the largest variant, rounded up. Vectors: lanes ×
//! lane size, aligned to their size.

use crate::types::{AdtKind, IntBits, Lanes, MAX_TYPE_DEPTH, TypeId, TypeKey, TypeTable};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// `str`, `[T]`, or a sizeless vector by value.
    Unsized,
    /// An ADT type without a shape.
    NoShape,
    /// A type that contains itself without indirection.
    Infinite,
    /// Size overflows `u64`.
    Overflow,
    /// A generic parameter (not monomorphized).
    NotMonomorphic,
    /// Nesting deeper than [`MAX_TYPE_DEPTH`].
    TooDeep,
    /// The type id is out of range.
    BadType,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            LayoutError::Unsized => "unsized type has no layout",
            LayoutError::NoShape => "ADT type has no shape",
            LayoutError::Infinite => "type contains itself without indirection",
            LayoutError::Overflow => "size overflows u64",
            LayoutError::NotMonomorphic => "generic parameter type",
            LayoutError::TooDeep => "type nesting too deep",
            LayoutError::BadType => "type id out of range",
        };
        f.write_str(s)
    }
}

/// How a value of the type is stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shape {
    /// One scalar of `size` bytes (int, float, bool, char, thin pointer, fn value).
    Scalar,
    /// A fat pointer: address at 0, length at 8.
    Fat,
    /// Tuple or struct: (offset, type) per field.
    Fields(Vec<(u64, TypeId)>),
    /// Array: `len` elements at `stride`.
    Array { elem: TypeId, stride: u64, len: u64 },
    /// Enum: `u32` tag at 0; (offset, type) per field per variant.
    Enum(Vec<Vec<(u64, TypeId)>>),
    /// Vector or mask: raw bytes.
    Bytes,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub size: u64,
    pub align: u64,
    pub shape: Shape,
}

/// Layouts of every type of a table.
#[derive(Clone, Debug)]
pub struct Layouts {
    v: Vec<Result<Layout, LayoutError>>,
}

#[derive(Clone)]
enum St {
    New,
    Busy,
    Done(Result<Layout, LayoutError>),
}

fn align_up(x: u64, a: u64) -> Option<u64> {
    let a = a.max(1);
    x.checked_add(a - 1).map(|y| y / a * a)
}

/// Laid-out fields, end offset, alignment.
type Record = (Vec<(u64, TypeId)>, u64, u64);

fn record(
    fields: impl Iterator<Item = Result<(Layout, TypeId), LayoutError>>,
    start: u64,
) -> Result<Record, LayoutError> {
    let mut off = start;
    let mut align = 1u64;
    let mut out = Vec::new();
    for f in fields {
        let (l, t) = f?;
        off = align_up(off, l.align).ok_or(LayoutError::Overflow)?;
        out.push((off, t));
        off = off.checked_add(l.size).ok_or(LayoutError::Overflow)?;
        align = align.max(l.align);
    }
    Ok((out, off, align))
}

impl Layouts {
    pub fn compute(types: &TypeTable) -> Layouts {
        let mut st = vec![St::New; types.len()];
        for i in 0..types.len() {
            let _ = Self::get(types, &mut st, TypeId(i as u32), 0);
        }
        Layouts {
            v: st
                .into_iter()
                .map(|s| match s {
                    St::Done(r) => r,
                    _ => Err(LayoutError::Infinite),
                })
                .collect(),
        }
    }

    pub fn get_layout(&self, t: TypeId) -> Result<&Layout, LayoutError> {
        match self.v.get(t.index()) {
            Some(Ok(l)) => Ok(l),
            Some(Err(e)) => Err(*e),
            None => Err(LayoutError::BadType),
        }
    }

    pub fn size_of(&self, t: TypeId) -> Option<u64> {
        self.get_layout(t).ok().map(|l| l.size)
    }

    fn get(types: &TypeTable, st: &mut [St], t: TypeId, depth: usize) -> Result<Layout, LayoutError> {
        match st.get(t.index()) {
            None => return Err(LayoutError::BadType),
            Some(St::Done(r)) => return r.clone(),
            Some(St::Busy) => return Err(LayoutError::Infinite),
            Some(St::New) => {}
        }
        if depth > MAX_TYPE_DEPTH {
            return Err(LayoutError::TooDeep);
        }
        st[t.index()] = St::Busy;
        let r = Self::calc(types, st, t, depth + 1);
        st[t.index()] = St::Done(r.clone());
        r
    }

    fn calc(types: &TypeTable, st: &mut [St], t: TypeId, depth: usize) -> Result<Layout, LayoutError> {
        let key = types.key(t).ok_or(LayoutError::BadType)?;
        let scalar = |n: u64| Ok(Layout { size: n, align: n, shape: Shape::Scalar });
        match key {
            TypeKey::Int { bits, .. } => scalar(match bits {
                IntBits::Size => 8,
                b => b.width() as u64 / 8,
            }),
            TypeKey::Float(b) => scalar(b.width() as u64 / 8),
            TypeKey::Bool => scalar(1),
            TypeKey::Char => scalar(4),
            TypeKey::Str | TypeKey::Slice { .. } => Err(LayoutError::Unsized),
            TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. } => {
                if is_unsized_pointee(types, *inner) {
                    Ok(Layout { size: 16, align: 8, shape: Shape::Fat })
                } else {
                    scalar(8)
                }
            }
            TypeKey::FnType { .. } => Ok(Layout { size: 16, align: 8, shape: Shape::Fat }),
            TypeKey::Array { elem, len } => {
                let e = Self::get(types, st, *elem, depth)?;
                let size = e.size.checked_mul(*len).ok_or(LayoutError::Overflow)?;
                Ok(Layout {
                    size,
                    align: e.align,
                    shape: Shape::Array { elem: *elem, stride: e.size, len: *len },
                })
            }
            TypeKey::Tuple(ts) => {
                let ts = ts.clone();
                let (fields, end, align) =
                    record(ts.iter().map(|&f| Self::get(types, st, f, depth).map(|l| (l, f))), 0)?;
                let size = align_up(end, align).ok_or(LayoutError::Overflow)?;
                Ok(Layout { size, align, shape: Shape::Fields(fields) })
            }
            TypeKey::Adt { .. } => {
                let shape = types.shape(t).ok_or(LayoutError::NoShape)?.clone();
                match shape.kind {
                    AdtKind::Struct => {
                        let fs = shape.variants.first().cloned().unwrap_or_default();
                        let (fields, end, align) =
                            record(fs.iter().map(|&f| Self::get(types, st, f, depth).map(|l| (l, f))), 0)?;
                        let size = align_up(end, align).ok_or(LayoutError::Overflow)?;
                        Ok(Layout { size, align, shape: Shape::Fields(fields) })
                    }
                    AdtKind::Enum => {
                        let mut align = 4u64;
                        let mut end = 4u64;
                        let mut vars = Vec::with_capacity(shape.variants.len());
                        for v in &shape.variants {
                            let (fields, e, a) =
                                record(v.iter().map(|&f| Self::get(types, st, f, depth).map(|l| (l, f))), 4)?;
                            align = align.max(a);
                            end = end.max(e);
                            vars.push(fields);
                        }
                        let size = align_up(end, align).ok_or(LayoutError::Overflow)?;
                        Ok(Layout { size, align, shape: Shape::Enum(vars) })
                    }
                }
            }
            TypeKey::Vec { lane, n } => {
                let Lanes::Fixed(n) = n else { return Err(LayoutError::Unsized) };
                let l = Self::get(types, st, *lane, depth)?;
                let size = l.size * *n as u64;
                Ok(Layout { size, align: size.max(1), shape: Shape::Bytes })
            }
            TypeKey::Mask { width, n } => {
                let Lanes::Fixed(n) = n else { return Err(LayoutError::Unsized) };
                let size = (*width as u64 / 8).max(1) * *n as u64;
                Ok(Layout { size, align: size.max(1), shape: Shape::Bytes })
            }
            TypeKey::Param(_) | TypeKey::ConstParam(_) => Err(LayoutError::NotMonomorphic),
        }
    }
}

/// True if a pointer to `t` is fat (`t` is a slice or `str`).
pub fn is_unsized_pointee(types: &TypeTable, t: TypeId) -> bool {
    matches!(types.key(t), Some(TypeKey::Str | TypeKey::Slice { .. }))
}
