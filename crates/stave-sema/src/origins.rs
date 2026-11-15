//! Origins: what a call result borrows from, per signature, and the `from` /
//! `from copy` rules of the origin table (LANG.md §3) (E0520, E0521). Checked when a fn signature is
//! built ([`check_sig`], from sigs), so these codes belong to the type-check stage: they depend on
//! the signature only, never on a body.
//!
//! | parameter `x` ([`ParamClass`]) | `from x` | `from copy x` |
//! |---|---|---|
//! | `&T` / `&mut T`, T not a view ([`ParamClass::Ref`]) | [`Flow::Arg`] | E0521 |
//! | view `V` by value ([`ParamClass::View`]) | [`Flow::Arg`] | E0521 |
//! | `&V` / `&mut V`, V a view ([`ParamClass::RefView`]) | [`Flow::Arg`] | [`Flow::Pointee`] |
//! | owned non-view by value ([`ParamClass::Owned`]) | E0520 | E0521 |
//!
//! A missing `from` clause declares the **empty** origin set: the result holds no loan (a string literal, a fn item, a view built only from such
//! values). Returning a borrow of a parameter that `from` does not name is E0510, checked in the
//! body by the borrow checker. `Self` in a trait signature counts as a possible view in a
//! parameter (the impls are checked with their concrete `Self`). C-ABI and intrinsic signatures
//! are not checked here (E0804 covers views in C signatures). A `from` clause on a non-view return
//! type is accepted (no code).
//!
//! # API for the borrow checker
//!
//! - [`of_sig`]`(prog, tys, sig)` → [`SigOrigins`] of a direct call (fn item, method, trait method
//!   through a bound): `origins` lists, per named param (sorted by param, deduplicated), how the
//!   result's origin set is formed from that argument:
//!   - [`Flow::Arg`]: the result's origin set ⊇ the argument *value's* origin set. For a `&T`
//!     argument that is the loan on `*x` (the caller's loan stays live while the result lives);
//!     for a view by value, the view's origins; for `&V`, the loan on the view object (a reborrow:
//!     the view object, and through it V's own origins, stay borrowed while the result lives).
//!   - [`Flow::Pointee`] (`from copy x`, `&V` / `&mut V` only): the result's origin set ⊇ the
//!     origin set of the view `*x`, not the loan on `x` itself, so the result may outlive the
//!     borrow of the view object (iterator `next`).
//!   - All other argument loans end at the call.
//!   - No `from` clause: `origins` is empty and the result holds no loan; the body's returns
//!     must have empty origin sets (else E0510).
//!   - `error`: the `from` clause has an E0520/E0521 entry. Its body should not be borrow-checked
//!     (no cascade); callers may use `origins` (the valid entries) as given.
//! - [`of_fn_type`]`(prog, tys, params, ret, generics)` → a call through a fn value (`fn(..) -> R`
//!   carries no `from`): every ref/view param as [`Flow::Arg`] (never `Pointee`), if `ret` is a
//!   view; empty otherwise.
//! - [`SigOrigins::weakened`]: a fn item coerced to a fn value ([`crate::Coercion::FnItem`])
//!   has its `from copy` weakened to `from` (`Pointee` → `Arg`). Its `from` set is always a
//!   subset of all its ref/view params, so a call through the value ([`of_fn_type`]) covers it.
//! - [`classify`]: the [`ParamClass`] of a param type.

use stave_diag::{Diagnostic, Span};

use crate::Program;
use crate::diag::mk;
use crate::kinds;
use crate::sigs::{Abi, FnSig, GenericParam, NameCx};
use crate::types::{self, Interner, TypeId, TypeKind};

/// A parameter's row of the origin table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamClass {
    /// `&T` / `&mut T`, T not a view.
    Ref,
    /// `&V` / `&mut V`, V a view.
    RefView,
    /// A view by value (fn values, `Option<&T>`, view ADTs, `T: view` params).
    View,
    /// An owned non-view value.
    Owned,
}

/// How a call result's origins come from one argument (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Flow {
    /// `from x`: the argument value's origins.
    Arg,
    /// `from copy x`: the origins of the view the argument points to.
    Pointee,
}

/// One resolved `from` entry: parameter index (methods: the receiver is 0) and flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResultOrigin {
    pub param: u8,
    pub flow: Flow,
}

/// The origins of a call's result (module docs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SigOrigins {
    /// The return type is or contains a ref or view.
    pub ret_view: bool,
    /// Valid entries, sorted, deduplicated.
    pub origins: Vec<ResultOrigin>,
    /// The signature's `from` clause has an invalid entry (E0520, E0521).
    pub error: bool,
}

impl SigOrigins {
    /// The origins through a fn value made from this fn item: `from copy` becomes `from`.
    pub fn weakened(&self) -> SigOrigins {
        let mut origins: Vec<ResultOrigin> =
            self.origins.iter().map(|o| ResultOrigin { param: o.param, flow: Flow::Arg }).collect();
        origins.dedup();
        SigOrigins { ret_view: self.ret_view, origins, error: self.error }
    }
}

/// `None` for a type with an error (no verdict, no cascade).
pub fn classify<I: Interner + ?Sized>(
    prog: &Program,
    tys: &I,
    t: TypeId,
    generics: &[GenericParam],
) -> Option<ParamClass> {
    if tys.has_error(t) {
        return None;
    }
    let maybe_view =
        |x: TypeId| matches!(tys.kind(x), TypeKind::SelfTy) || kinds::is_view_in(prog, tys, x, generics);
    Some(match tys.kind(t) {
        TypeKind::Ref { inner, .. } => {
            if maybe_view(*inner) {
                ParamClass::RefView
            } else {
                ParamClass::Ref
            }
        }
        _ if maybe_view(t) => ParamClass::View,
        _ => ParamClass::Owned,
    })
}

/// Whether a return type is or contains a ref or view (`Self` in a trait counts as non-view).
fn ret_is_view<I: Interner + ?Sized>(
    prog: &Program,
    tys: &I,
    ret: TypeId,
    generics: &[GenericParam],
) -> bool {
    !tys.has_error(ret) && kinds::is_view_in(prog, tys, ret, generics)
}

/// Whether `copy` on a param of class `c` is valid (`None`: unknown, valid).
fn valid(c: Option<ParamClass>, copy: bool) -> bool {
    match c {
        None => true,
        Some(c) if copy => c == ParamClass::RefView,
        Some(c) => c != ParamClass::Owned,
    }
}

/// The origins of a direct call of `sig` (module docs).
pub fn of_sig<I: Interner + ?Sized>(prog: &Program, tys: &I, sig: &FnSig) -> SigOrigins {
    let ret_view = ret_is_view(prog, tys, sig.ret, &sig.generics);
    let mut origins = Vec::new();
    let mut error = false;
    for o in &sig.origins {
        let c = sig.params.get(o.param as usize).and_then(|&t| classify(prog, tys, t, &sig.generics));
        if !valid(c, o.copy) {
            error = true;
            continue;
        }
        origins.push(ResultOrigin { param: o.param, flow: if o.copy { Flow::Pointee } else { Flow::Arg } });
    }
    origins.sort();
    origins.dedup();
    SigOrigins { ret_view, origins, error }
}

/// The origins of a call through a fn value of type `fn(params) -> ret` (module docs).
pub fn of_fn_type<I: Interner + ?Sized>(
    prog: &Program,
    tys: &I,
    params: &[TypeId],
    ret: TypeId,
    generics: &[GenericParam],
) -> SigOrigins {
    let ret_view = ret_is_view(prog, tys, ret, generics);
    let mut origins = Vec::new();
    if ret_view {
        for (i, &p) in params.iter().enumerate() {
            if matches!(
                classify(prog, tys, p, generics),
                Some(ParamClass::Ref | ParamClass::RefView | ParamClass::View)
            ) {
                origins.push(ResultOrigin { param: i.min(255) as u8, flow: Flow::Arg });
            }
        }
    }
    SigOrigins { ret_view, origins, error: false }
}

/// The `from` checks of a Stave-ABI signature: E0520, E0521 per entry of `sig.origins`
/// (`at[i]`: the span and the written name of entry i).
pub(crate) fn check_sig<I: Interner + ?Sized>(
    prog: &Program,
    tys: &I,
    sig: &FnSig,
    at: &[(Span, String)],
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    if sig.abi != Abi::Stave {
        return out;
    }
    let names = NameCx { prog, generics: &sig.generics };
    for (o, (sp, name)) in sig.origins.iter().zip(at) {
        let Some(&t) = sig.params.get(o.param as usize) else { continue };
        let c = classify(prog, tys, t, &sig.generics);
        if valid(c, o.copy) {
            continue;
        }
        let tn = types::display(tys, &names, t);
        let code = if o.copy { "E0521" } else { "E0520" };
        out.extend(mk(code, None, &[("param", name), ("type", &tn)], *sp, &[]));
    }
    out
}
