//! Rules of `extern "C"` signatures and symbols, checked once the signatures are built.
//!
//! - **E0804**: a param or return type of a C import or export that is not C-safe. C-safe:
//!   `i8..i64 u8..u64 isize usize f32 f64 bool`, a raw pointer whose pointee is not `[T]` or `str`
//!   (those pointers are two words), and as a return also `()` and `never`. Structs by value
//!   are not C-safe: the AArch64 Stave convention passes every aggregate by reference.
//! - **E0812**: the symbol of a C import or export is its fn name, unmangled, one namespace for
//!   the whole program. Two exports of one name, two declarations of one symbol with different
//!   signatures (type ids are interned, so equal ids are equal types after alias expansion), or a
//!   symbol the compiler defines itself.
//! - **E0813** ([`c_variadic_ok`], checked by the type checker at each call): the C default
//!   argument promotions are written by the user, so only promoted types pass through `...`.

use rustc_hash::FxHashMap;
use stave_diag::Diagnostic;

use crate::Program;
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::DeclId;
use crate::sigs::{Abi, FnSig, NameCx, Sigs};
use crate::types::{self, FloatBits, IntBits, Interner, TypeId, TypeKind};

/// The E0804 verdict for `t` in a C signature: `None` if C-safe, else `Some(case)`.
fn c_unsafe_case<I: Interner + ?Sized>(
    prog: &Program,
    cx: &I,
    t: TypeId,
    ret: bool,
) -> Option<Option<&'static str>> {
    match cx.kind(t) {
        TypeKind::Error | TypeKind::Int { .. } | TypeKind::Bool => None,
        TypeKind::Float(FloatBits::F32 | FloatBits::F64) => None,
        TypeKind::Ptr { inner, .. } => match cx.kind(*inner) {
            TypeKind::Slice { .. } | TypeKind::Str => Some(None),
            _ => None,
        },
        TypeKind::Never if ret => None,
        TypeKind::Tuple(ts) if ts.is_empty() && ret => None,
        TypeKind::Vector { .. } | TypeKind::Mask { .. } | TypeKind::Ref { .. } | TypeKind::Fn { .. } => {
            Some(Some("vector-view"))
        }
        _ if crate::kinds::is_view_concrete(prog, cx, t) => Some(Some("vector-view")),
        TypeKind::Adt { def, .. } if prog.decl(*def).is_some_and(|d| d.kind == DeclKind::Struct) => {
            Some(Some("struct"))
        }
        _ => Some(None),
    }
}

/// Whether a variadic argument of type `t` can be passed to C as written: the promoted int
/// types, `f64`, or a thin raw pointer.
pub(crate) fn c_variadic_ok<I: Interner + ?Sized>(cx: &I, t: TypeId) -> bool {
    match cx.kind(t) {
        TypeKind::Error => true,
        TypeKind::Int { bits, .. } => matches!(bits, IntBits::B32 | IntBits::B64 | IntBits::Size),
        TypeKind::Float(f) => *f == FloatBits::F64,
        TypeKind::Ptr { inner, .. } => !matches!(cx.kind(*inner), TypeKind::Slice { .. } | TypeKind::Str),
        _ => false,
    }
}

/// The C import or export `d` with ABI `"C"`, and its signature.
fn c_fn<'s>(prog: &Program, sigs: &'s Sigs, d: DeclId) -> Option<&'s FnSig> {
    let decl = prog.decl(d)?;
    if decl.kind != DeclKind::Fn || !decl.flags.has(DeclFlags::EXTERN_IMPORT | DeclFlags::EXTERN_EXPORT) {
        return None;
    }
    sigs.fn_sig(d).filter(|s| s.abi == Abi::C)
}

/// E0804 and E0812 over every C import and export of the program.
pub(crate) fn check<I: Interner + ?Sized>(prog: &Program, sigs: &Sigs, cx: &I) -> Vec<Diagnostic> {
    let names = NameCx { prog, generics: &[] };
    let mut out = Vec::new();
    // per symbol: the first declaration, the first export
    let mut first: FxHashMap<&str, (DeclId, Option<DeclId>)> = FxHashMap::default();
    for k in 0..prog.decls.len() {
        let d = DeclId(k as u32);
        let Some(sig) = c_fn(prog, sigs, d) else { continue };
        let Some(decl) = prog.decl(d) else { continue };
        let typed = sig.params.iter().zip(&sig.param_spans).map(|(&t, &sp)| (t, sp, false));
        for (t, sp, ret) in typed.chain(sig.ret_span.map(|sp| (sig.ret, sp, true))) {
            if let Some(case) = c_unsafe_case(prog, cx, t, ret) {
                let tn = types::display(cx, &names, t);
                out.extend(mk("E0804", case, &[("type", &tn)], sp, &[]));
            }
        }
        let name = prog.strings.get(decl.name);
        if name == stave_ir::link::START || name.starts_with(stave_ir::link::RESERVED_PREFIX) {
            out.extend(mk("E0812", Some("reserved"), &[("name", name)], decl.name_span, &[]));
            continue;
        }
        let export = decl.flags.has(DeclFlags::EXTERN_EXPORT);
        let Some(&(f, fx)) = first.get(name) else {
            first.insert(name, (d, export.then_some(d)));
            continue;
        };
        let span_of = |x: DeclId| prog.decl(x).map(|y| y.name_span);
        if export && let Some(e) = fx {
            out.extend(
                span_of(e)
                    .and_then(|s| mk("E0812", None, &[("name", name)], decl.name_span, &[("first", s)])),
            );
            continue;
        }
        if export {
            first.insert(name, (f, Some(d)));
        }
        let Some(fs) = sigs.fn_sig(f) else { continue };
        if fs.params != sig.params || fs.ret != sig.ret || fs.variadic != sig.variadic {
            let roles: Vec<(&str, stave_diag::Span)> = span_of(f).map(|s| ("first", s)).into_iter().collect();
            out.extend(mk("E0812", Some("signature"), &[("name", name)], decl.name_span, &roles));
        }
    }
    out
}
