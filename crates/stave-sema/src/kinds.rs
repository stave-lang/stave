//! Kinds ([`Kind`]): `copy view send sync sized` per
//! type, and the view placement rules E0410–E0414. Every view test of the checker goes through
//! [`is_view`]; the diagnostics are built here only ([`e0410`], [`view_arg`], [`e0413`]).
//!
//! **View** is contagious: a type is a view iff it is a reference, a fn type (every fn value is a
//! view), a `view struct` / `view enum`, or a tuple, array, ADT instance, or `Option`
//! with a view component (an ADT instance through its type arguments: `Option<&T>`). Raw pointers
//! are never views (their pointee is not inspected). A generic parameter is a view only through
//! the caller's `param_view` (a `T: view` param follows the view rules at its definition
//! site); `Self` in a trait is not.
//!
//! **Placement rules** (where they are checked):
//! - E0410 (sigs, ADT build): a field whose type is a view in a non-view ADT. A field of type
//!   `T` with `T: view` is allowed (`Option<T: view>`): the instance becomes a view through its
//!   argument.
//! - E0411 / E0412 / E0414 ([`view_arg`], at a type argument in a type, sigs `view_args`, and at
//!   the generic args of a call, expr `check_bounds`): a view argument for a param without
//!   `view` is E0412 when the generic is `std.mem.forget`, E0411 when it is an ADT of the std
//!   package or an impl of one (its heap containers: `Vec`, `Soa*`), else E0414.
//! - E0413 (sigs, const build): a `const` or `static mut` of view type, except `&str` from a
//!   string literal or another such const.
//!
//! **Copy / Send / Sync** come from [`World`] (built-in Copy; `impl Copy`; structural or
//! `unsafe impl` Send/Sync). `std.cell.UnsafeCell<T>` (fixed path) is Send iff T is and never
//! Sync, and the structural rule carries that to every type containing it (`Cell`, `RefCell`, a
//! user struct with such a field): it is Sync only through an explicit `unsafe(..) impl Sync`
//! (`std.sync.Mutex`). A fn type is Copy (its captures are by reference or `copy`), never
//! Send or Sync (its captures are not in its type: [`crate::threads`] checks them where a closure crosses
//! threads, E0802/E0806). `str` and slices are the only unsized types.

use stave_diag::{Diagnostic, Span};

use crate::Program;
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::DeclId;
use crate::resolve::TypeHead;
use crate::sigs::{GenericParam, TraitRef};
use crate::traits::World;
use crate::types::{GArg, Interner, TypeId, TypeKind};

/// The kind of a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kind {
    pub copy: bool,
    pub view: bool,
    pub send: bool,
    pub sync: bool,
    pub sized: bool,
}

/// Whether ADT `d` is declared `view`.
pub fn adt_is_view(prog: &Program, d: DeclId) -> bool {
    prog.decl(d).is_some_and(|x| x.flags.has(DeclFlags::VIEW))
}

/// Whether `t` is (or contains) a reference or view (module docs). `param_view(i)`: whether
/// generic param `i` counts as a view.
pub fn is_view<I: Interner + ?Sized>(
    tys: &I,
    t: TypeId,
    adt_view: &dyn Fn(DeclId) -> bool,
    param_view: &dyn Fn(u32) -> bool,
) -> bool {
    let mut stack = vec![t];
    let mut kids = Vec::new();
    let mut steps = 0u32;
    while let Some(x) = stack.pop() {
        steps += 1;
        if steps > 1_000_000 {
            return false;
        }
        match tys.kind(x) {
            TypeKind::Ref { .. } | TypeKind::Fn { .. } => return true,
            TypeKind::Ptr { .. } | TypeKind::SelfTy => {}
            TypeKind::Param(i) => {
                if param_view(*i) {
                    return true;
                }
            }
            TypeKind::Adt { def, args } => {
                if adt_view(*def) {
                    return true;
                }
                for a in args.iter() {
                    if let GArg::Type(t) = a {
                        stack.push(*t);
                    }
                }
            }
            k => {
                kids.clear();
                k.children(&mut kids);
                stack.extend_from_slice(&kids);
            }
        }
    }
    false
}

/// [`is_view`] where generic params count as non-view (a concrete type, or a definition-site
/// field: E0410).
pub fn is_view_concrete<I: Interner + ?Sized>(prog: &Program, tys: &I, t: TypeId) -> bool {
    is_view(tys, t, &|d| adt_is_view(prog, d), &|_| false)
}

/// [`is_view`] in the scope of `generics`: a `T: view` param counts as a view.
pub fn is_view_in<I: Interner + ?Sized>(
    prog: &Program,
    tys: &I,
    t: TypeId,
    generics: &[GenericParam],
) -> bool {
    is_view(tys, t, &|d| adt_is_view(prog, d), &|i| generics.get(i as usize).is_some_and(|g| g.view))
}

/// The kind of `t` in the scope of `env`.
pub fn kind_of<I: Interner + ?Sized>(w: World<'_>, cx: &mut I, t: TypeId, env: &[GenericParam]) -> Kind {
    let view = is_view_in(w.prog, cx, t, env);
    let sized = !matches!(cx.kind(t), TypeKind::Str | TypeKind::Slice { .. });
    let copy = w.is_copy(cx, t, env);
    let fn_ty = matches!(cx.kind(t), TypeKind::Fn { .. });
    let lang = &w.sigs.lang;
    let mut marker = |tr: DeclId| {
        !fn_ty && tr.is_some() && w.implements(cx, t, &TraitRef { trait_decl: tr, args: Vec::new() }, env)
    };
    let send = marker(lang.send);
    let sync = marker(lang.sync);
    Kind { copy, view, send, sync, sized }
}

/// Whether `d` is `std.mem.forget` (E0412).
pub fn is_forget(prog: &Program, d: DeclId) -> bool {
    let Some(x) = prog.decl(d) else { return false };
    if x.kind != DeclKind::Fn || prog.strings.get(x.name) != "forget" {
        return false;
    }
    prog.module(x.module).is_some_and(|m| {
        m.path.len() == 2 && prog.strings.get(m.path[0]) == "std" && prog.strings.get(m.path[1]) == "mem"
    })
}

/// Whether decl `d` belongs to the built-in std package.
fn in_std(prog: &Program, d: DeclId) -> bool {
    prog.decl(d)
        .and_then(|x| prog.module(x.module))
        .is_some_and(|m| prog.packages.get(m.package.idx()).is_some_and(|p| p.name == "std"))
}

/// E0410: field `field` of non-view ADT `adt` has view type `ty` (display text).
pub(crate) fn e0410(field: &str, ty: &str, adt: &str, at: Span) -> Option<Diagnostic> {
    mk("E0410", None, &[("field", field), ("type", ty), ("adt", adt)], at, &[])
}

/// The diagnostic for view type `ty` (display text) given for generic param `g` of `generic`
/// (which has no `view` bound): E0412 for `std.mem.forget`, E0411 for an ADT of the std package
/// (or an impl of one: `vec.Vec.new()`), else E0414 naming the ADT of an impl, or `generic`.
pub(crate) fn view_arg(
    prog: &Program,
    generic: DeclId,
    g: &GenericParam,
    ty: &str,
    at: Span,
) -> Option<Diagnostic> {
    if is_forget(prog, generic) {
        return mk("E0412", None, &[("type", ty)], at, &[]);
    }
    let named = match prog.decl(generic).map(|d| d.kind) {
        Some(DeclKind::Impl) => match prog.impl_info(generic).and_then(|i| i.head) {
            Some(TypeHead::Adt(a)) => a,
            _ => generic,
        },
        _ => generic,
    };
    let is_adt = prog.decl(named).is_some_and(|d| matches!(d.kind, DeclKind::Struct | DeclKind::Enum));
    if is_adt && in_std(prog, named) {
        let cn = prog.decl(named).map_or("_".to_string(), |x| prog.strings.get(x.name).to_string());
        return mk("E0411", None, &[("type", ty), ("container", &cn)], at, &[]);
    }
    let pn = prog.strings.get(g.name).to_string();
    let gp = prog.decl_path(named);
    mk("E0414", None, &[("param", &pn), ("type", ty), ("generic", &gp)], at, &[("param", g.name_span)])
}

/// E0413: const (or `static mut`) `name` has view type `ty` (display text).
pub(crate) fn e0413(is_static: bool, name: &str, ty: &str, at: Span) -> Option<Diagnostic> {
    mk("E0413", if is_static { Some("static") } else { None }, &[("name", name), ("type", ty)], at, &[])
}
