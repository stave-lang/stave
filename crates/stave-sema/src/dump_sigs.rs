//! The `sigs` dump (contracts/stage-gates.md): the type table of the types the declarations
//! name, then one line per decl with its signature facts. Built from [`Typeck::sigs`] and the
//! decl table only, never from a body, so it is what declaration collection and signature
//! building alone decide.
//!
//! The table holds every type a decl line names, closed over key components, numbered by rank
//! among themselves in the canonical type-key order: the canonical table is that order, so the
//! rank of a subset is its order there. The error type has no line and ranks last; a decl field
//! of the error type prints `-`.

use std::fmt::Write;

use crate::Program;
use crate::decls::DeclKind;
use crate::effects::Effects;
use crate::ids::DeclId;
use crate::sigs::{Abi, DeclSig, GenericParam, SelfKind, TraitRef, VariantShape};
use crate::typeck::Typeck;
use crate::types::{self, GArg, Interner, TypeId};

/// [`crate::DeclFlags`] names, by bit.
const FLAG_NAMES: [&str; 14] = [
    "pub",
    "unsafe",
    "view",
    "test",
    "kernel",
    "dispatch",
    "extern-import",
    "extern-export",
    "trait-impl",
    "tuple",
    "record",
    "has-self",
    "mut",
    "rec",
];

/// The `sigs` dump: `t<T> key` per table type, by T, then one decl line per decl, by D.
pub fn dump_sigs(prog: &Program, tc: &Typeck) -> String {
    let members = Members::new(prog, tc);
    let table = TypeRank::new(prog, tc, &members);
    let mut out = String::new();
    for &t in &table.order {
        let _ = writeln!(out, "t{} {}", table.at(t), types::type_key_text_as(&tc.types, t, &|c| table.at(c)));
    }
    let mut cx = Cx { tc, num: Some(&table.num), roots: Vec::new() };
    for d in 0..prog.decls.len() {
        decl_line(prog, tc, &members, &mut cx, DeclId(d as u32), &mut out);
    }
    out
}

/// A dump's type table: the types its lines name, closed over key components, numbered by rank
/// among themselves in the canonical type-key order. The `sigs` and `types` dumps both number
/// this way; every dump that names `t<T>` of `sigs` reads the `sigs` numbering.
pub(crate) struct TypeRank {
    /// The table's types by local T.
    pub order: Vec<TypeId>,
    /// Global id -> local T; the error type and types outside the table map to `next`.
    num: Vec<u32>,
    /// The number of table lines (the error type's number).
    next: u32,
}

impl TypeRank {
    /// The `sigs` table.
    pub(crate) fn of(prog: &Program, tc: &Typeck) -> TypeRank {
        TypeRank::new(prog, tc, &Members::new(prog, tc))
    }

    fn new(prog: &Program, tc: &Typeck, members: &Members) -> TypeRank {
        // The decl lines are rendered by the same code as the dump, to collect the types they
        // name, so the table holds exactly what the lines name.
        let mut cx = Cx { tc, num: None, roots: Vec::new() };
        let mut scratch = String::new();
        for d in 0..prog.decls.len() {
            scratch.clear();
            decl_line(prog, tc, members, &mut cx, DeclId(d as u32), &mut scratch);
        }
        TypeRank::from_roots(tc, std::mem::take(&mut cx.roots))
    }

    /// The table of `roots` and the types their keys reference.
    pub(crate) fn from_roots(tc: &Typeck, roots: Vec<TypeId>) -> TypeRank {
        let err = tc.error_type;
        let mut used = vec![false; tc.types.len()];
        let mut stack = roots;
        let mut kids = Vec::new();
        while let Some(t) = stack.pop() {
            let Some(u) = used.get_mut(t.idx()) else { continue };
            if *u {
                continue;
            }
            *u = true;
            kids.clear();
            tc.types.kind(t).children(&mut kids);
            stack.extend_from_slice(&kids);
        }
        let mut num = vec![u32::MAX; tc.types.len()];
        let mut order = Vec::new();
        for (i, &u) in used.iter().enumerate() {
            if u && TypeId(i as u32) != err {
                num[i] = order.len() as u32;
                order.push(TypeId(i as u32));
            }
        }
        let next = order.len() as u32;
        if let Some(e) = num.get_mut(err.idx()) {
            *e = next;
        }
        TypeRank { order, num, next }
    }

    /// The local T of global type `t` (the error type: the number of table lines).
    pub(crate) fn at(&self, t: TypeId) -> u32 {
        self.num.get(t.idx()).copied().unwrap_or(self.next)
    }
}

/// Field types and tuple payloads, by member decl (they live in their ADT's signature).
struct Members {
    field_ty: Vec<TypeId>,
    payload: Vec<Option<Vec<TypeId>>>,
}

impl Members {
    fn new(prog: &Program, tc: &Typeck) -> Members {
        let n = prog.decls.len();
        let mut m = Members { field_ty: vec![TypeId::NONE; n], payload: vec![None; n] };
        for s in &tc.sigs.decls {
            let DeclSig::Adt(a) = s else { continue };
            for v in &a.variants {
                if v.shape == VariantShape::Tuple
                    && let Some(p) = m.payload.get_mut(v.decl.idx())
                {
                    *p = Some(v.fields.iter().map(|f| f.ty).collect());
                }
                for f in &v.fields {
                    if let Some(x) = m.field_ty.get_mut(f.decl.idx()) {
                        *x = f.ty;
                    }
                }
            }
        }
        m
    }
}

struct Cx<'a> {
    tc: &'a Typeck,
    /// Global id → table number; `None` while collecting.
    num: Option<&'a [u32]>,
    roots: Vec<TypeId>,
}

impl Cx<'_> {
    fn ty(&mut self, out: &mut String, t: TypeId) {
        if t.is_none() || t == self.tc.error_type {
            out.push('-');
            return;
        }
        match self.num {
            None => self.roots.push(t),
            Some(num) => {
                let _ = write!(out, "t{}", num.get(t.idx()).copied().unwrap_or(u32::MAX));
            }
        }
    }

    fn tys(&mut self, out: &mut String, ts: &[TypeId]) {
        out.push('[');
        for (k, &t) in ts.iter().enumerate() {
            if k > 0 {
                out.push(' ');
            }
            self.ty(out, t);
        }
        out.push(']');
    }

    fn trait_ref(&mut self, out: &mut String, r: &TraitRef) {
        let _ = write!(out, "d{}", r.trait_decl.0);
        for a in &r.args {
            out.push(' ');
            match a {
                GArg::Type(t) => self.ty(out, *t),
                GArg::Const(c) => out.push_str(&types::const_arg_text(*c)),
            }
        }
    }

    fn generics(&mut self, out: &mut String, gs: &[GenericParam]) {
        out.push_str("gen[");
        for (k, g) in gs.iter().enumerate() {
            if k > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{} {}", g.name.0, g.node);
            if g.is_const {
                out.push_str(" const ");
                self.ty(out, g.const_ty);
            }
            if g.view {
                out.push_str(" view");
            }
            for b in &g.bounds {
                out.push_str(" bound ");
                self.trait_ref(out, b);
            }
        }
        out.push(']');
    }
}

fn kind_name(k: DeclKind) -> &'static str {
    match k {
        DeclKind::Module => "module",
        DeclKind::Fn => "fn",
        DeclKind::Struct => "struct",
        DeclKind::Enum => "enum",
        DeclKind::Trait => "trait",
        DeclKind::Const => "const",
        DeclKind::Static => "static",
        DeclKind::TypeAlias => "alias",
        DeclKind::Impl => "impl",
        DeclKind::Field => "field",
        DeclKind::Variant => "variant",
        DeclKind::Method => "method",
        DeclKind::TraitMethod => "trait-method",
        DeclKind::Error => "error",
    }
}

fn decl_line(prog: &Program, tc: &Typeck, m: &Members, cx: &mut Cx<'_>, d: DeclId, out: &mut String) {
    let Some(decl) = prog.decl(d) else { return };
    let _ = write!(out, "d{} {} ", d.0, kind_name(decl.kind));
    match prog.module(decl.module) {
        Some(md) if md.decl.is_some() => {
            let _ = write!(out, "d{} ", md.decl.0);
        }
        _ => out.push_str("- "),
    }
    match decl.parent.get() {
        Some(p) => {
            let _ = write!(out, "d{}", p.0);
        }
        None => out.push('-'),
    }
    let opt = |out: &mut String, v: u32| {
        out.push(' ');
        if v == u32::MAX {
            out.push('-');
        } else {
            let _ = write!(out, "{v}");
        }
    };
    opt(out, decl.name.0);
    opt(out, decl.item.0);
    opt(out, decl.node);
    let _ = write!(out, " {} ", decl.index);
    let mut any = false;
    for (bit, name) in FLAG_NAMES.iter().enumerate() {
        if decl.flags.0 & (1 << bit) != 0 {
            if any {
                out.push(',');
            }
            out.push_str(name);
            any = true;
        }
    }
    if !any {
        out.push('-');
    }
    match tc.sigs.get(d) {
        DeclSig::Fn(f) => {
            out.push(' ');
            cx.generics(out, f.generics.get(f.n_parent as usize..).unwrap_or(&[]));
            out.push_str(" params");
            cx.tys(out, &f.params);
            out.push_str(" ret ");
            cx.ty(out, f.ret);
            out.push_str(match f.self_kind {
                None => " self -",
                Some(SelfKind::Value) => " self value",
                Some(SelfKind::Ref) => " self ref",
                Some(SelfKind::RefMut) => " self ref-mut",
            });
            out.push_str(match f.abi {
                Abi::Stave => " abi stave",
                Abi::C => " abi c",
                Abi::Intrinsic => " abi intrinsic",
            });
            if f.is_unsafe {
                out.push_str(" unsafe");
            }
            if f.variadic {
                out.push_str(" variadic");
            }
            out.push_str(" needs[");
            out.push_str(&types::feature_names(f.features).join(" "));
            out.push_str("] deny[");
            out.push_str(&Effects(f.deny).names().join(" "));
            out.push_str("] from[");
            for (k, o) in f.origins.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}{}", if o.copy { "copy " } else { "" }, o.param);
            }
            out.push_str("] pfrom[");
            for (k, o) in f.param_origins.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}<{}{}", o.target, if o.copy { "copy " } else { "" }, o.source);
            }
            out.push(']');
        }
        DeclSig::Adt(a) => {
            out.push(' ');
            cx.generics(out, &a.generics);
        }
        DeclSig::Impl(i) => {
            out.push(' ');
            cx.generics(out, &i.generics);
            out.push_str(" self ");
            cx.ty(out, i.self_ty);
            out.push_str(" trait ");
            match &i.trait_ref {
                Some(r) => cx.trait_ref(out, r),
                None => out.push('-'),
            }
        }
        DeclSig::Trait(t) => {
            out.push(' ');
            cx.generics(out, &t.generics);
        }
        DeclSig::Const(c) | DeclSig::Static(c) => {
            out.push_str(" ty ");
            cx.ty(out, c.ty);
        }
        DeclSig::Alias(a) => {
            out.push(' ');
            cx.generics(out, &a.generics);
            out.push_str(" target ");
            cx.ty(out, a.target);
        }
        DeclSig::None => match decl.kind {
            DeclKind::Module => {
                if let Some(md) = prog.module(decl.module) {
                    let _ = write!(out, " file {} path[", md.file.0);
                    for (k, s) in md.path.iter().enumerate() {
                        if k > 0 {
                            out.push(' ');
                        }
                        let _ = write!(out, "{}", s.0);
                    }
                    out.push(']');
                }
            }
            DeclKind::Field => {
                if let Some(&t) = m.field_ty.get(d.idx())
                    && !t.is_none()
                {
                    out.push_str(" ty ");
                    cx.ty(out, t);
                }
            }
            DeclKind::Variant => {
                if let Some(Some(p)) = m.payload.get(d.idx()) {
                    out.push_str(" payload");
                    cx.tys(out, p);
                }
            }
            _ => {}
        },
    }
    out.push('\n');
}
