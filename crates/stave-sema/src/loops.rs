//! The kinds of `for` sources (contracts/lang-details.md §4 "`for` sources"): one closed
//! classification of the source's type, read by the type checker (which reports every kind that
//! is not a source), the borrow checker, and the lowering. A type this does not recognise is no
//! source.
//!
//! Whether the source is an array or `[T]` *place* (E0219) is a matter of syntax, checked by the
//! type checker on top of this.

use crate::types::{Interner, TypeId, TypeKind};

/// What a `for` loop does with a source of a given type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForSource {
    /// `a..b`, `a..=b`: the values.
    Range { elem: TypeId },
    /// An array value (a place is E0219): the loop's temporary, its elements by value.
    Array { elem: TypeId },
    /// `[T]`: only a place (a sub-slice, a column), so always E0219.
    SlicePlace,
    /// `&[T]`, `&mut [T]`, `&[T; N]`, `&mut [T; N]`: element `i` of `*s` borrowed per iteration.
    Seq { mutable: bool, elem: TypeId },
    /// An iterator value (a struct, enum, or generic type: `next(&mut self)` must exist): moved
    /// into the loop, which owns it.
    Iter,
    /// `&mut It`, `It` an iterator type: advanced in place through the reference.
    IterMut { inner: TypeId },
    /// `&It`: no source (E0219 `shared-iter`).
    SharedIter,
    /// A reference to a reference (`&mut &mut It`, `& &[T]`, `&mut &[T; N]`, ..): no source
    /// (E0219 `ref-ref`).
    RefRef,
    /// Any other type (a scalar, vector, mask, `str`, tuple, raw pointer, fn value, `&` of one):
    /// no source (E0219 `not-source`).
    NotSource,
    /// The error type (already reported).
    Error,
}

impl ForSource {
    /// Whether a `for` loop may iterate a source of this kind.
    pub fn is_source(self) -> bool {
        matches!(
            self,
            ForSource::Range { .. }
                | ForSource::Array { .. }
                | ForSource::Seq { .. }
                | ForSource::Iter
                | ForSource::IterMut { .. }
        )
    }
}

/// The kind of a `for` source of type `t` in `tys`.
pub fn for_source<I: Interner + ?Sized>(tys: &I, t: TypeId) -> ForSource {
    if t.is_none() || tys.is_error_ty(t) {
        return ForSource::Error;
    }
    match tys.kind(t) {
        TypeKind::Range { elem, .. } => ForSource::Range { elem: *elem },
        TypeKind::Array { elem, .. } => ForSource::Array { elem: *elem },
        TypeKind::Slice { .. } => ForSource::SlicePlace,
        TypeKind::Adt { .. } | TypeKind::Param(_) => ForSource::Iter,
        TypeKind::Ref { mutable, inner } => match tys.kind(*inner) {
            TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } => {
                ForSource::Seq { mutable: *mutable, elem: *elem }
            }
            TypeKind::Ref { .. } => ForSource::RefRef,
            TypeKind::Adt { .. } | TypeKind::Param(_) if *mutable => ForSource::IterMut { inner: *inner },
            TypeKind::Adt { .. } | TypeKind::Param(_) => ForSource::SharedIter,
            TypeKind::Error => ForSource::Error,
            _ => ForSource::NotSource,
        },
        TypeKind::Error => ForSource::Error,
        _ => ForSource::NotSource,
    }
}
