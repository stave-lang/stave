//! The `kinds` dump (contracts/stage-gates.md): per type of the `sigs` table its kind, depth,
//! and layout, then the impl table, then which impls apply to each type. Read from the signature
//! table, the canonical type table, and the trait and layout rules, never from a body.
//!
//! Every answer is taken in the empty environment: a generic param has no bound and is not
//! `view`, so a type with params gets what holds for every instance (a layout that does not
//! depend on them, `&T` Copy) and nothing a bound would add.

use std::fmt::Write;

use rustc_hash::FxHashMap;

use crate::Program;
use crate::dump_sigs::SigTable;
use crate::ids::DeclId;
use crate::resolve::TypeHead;
use crate::traits::{self, NoLayout, Unlaid, WithSigs, World};
use crate::typeck::Typeck;
use crate::types::{Interner, Overlay, TypeId, TypeKind};

/// The kind flags, in print order.
const KIND_NAMES: [&str; 10] =
    ["copy", "view", "send", "sync", "sized", "flat", "frozen", "drop-impl", "drop", "lock"];

/// The `kinds` dump: one `t<T>` line per `sigs` table type, by T; one `impl` line per impl, by
/// (trait, head, item); one `apply` line per (type, trait) with a candidate impl, by (T, trait).
pub fn dump_kinds(prog: &Program, tc: &Typeck) -> String {
    let table = SigTable::of(prog, tc);
    let sigs = &tc.sigs;
    let w = World { prog, sigs };
    let mut cx = Overlay::new(&tc.types);
    let mut out = String::new();

    let depths = crate::typedepth::depths(sigs, &tc.types, &table.order);
    let mut drop_memo = FxHashMap::default();
    for (&t, depth) in table.order.iter().zip(depths) {
        let k = crate::kinds::kind_of(w, &mut cx, t, &[]);
        let bits = [
            k.copy,
            k.view,
            k.send,
            k.sync,
            k.sized,
            k.flat,
            k.frozen,
            crate::mono::drop_impl(&w, &mut cx, t),
            crate::mono::needs_drop(&w, &mut cx, t, &[], &mut drop_memo),
            !crate::ranks::lock_parts(&mut cx, prog, sigs, &[], t).is_empty(),
        ];
        let _ = write!(out, "t{} ", table.at(t));
        let mut any = false;
        for (on, name) in bits.iter().zip(KIND_NAMES) {
            if *on {
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
        match depth {
            None => out.push_str(" depth -"),
            Some(d) if d > crate::typedepth::MAX_TYPE_DEPTH => out.push_str(" depth over"),
            Some(d) => {
                let _ = write!(out, " depth {d}");
            }
        }
        layout(&mut cx, tc, t, &mut out);
        out.push('\n');
    }

    // The impl table: inherent impls first, then by trait D; within one, by head, then item.
    let mut impls: Vec<(ImplKey, DeclId)> = Vec::new();
    for (k, s) in sigs.decls.iter().enumerate() {
        let crate::sigs::DeclSig::Impl(i) = s else { continue };
        let d = DeclId(k as u32);
        let tr = i.trait_ref.as_ref().map_or(0, |r| r.trait_decl.0.wrapping_add(1));
        let item = prog.decl(d).map_or(crate::ids::ItemId::NONE, |x| x.item);
        impls.push(((tr, head_key(i.head), item), d));
    }
    impls.sort();
    for (_, d) in &impls {
        let Some(i) = sigs.impl_def(*d) else { continue };
        let _ = write!(out, "impl d{} trait ", d.0);
        match &i.trait_ref {
            Some(r) => {
                let _ = write!(out, "d{}", r.trait_decl.0);
            }
            None => out.push('-'),
        }
        out.push_str(" head ");
        match i.head {
            None => out.push('-'),
            Some(TypeHead::Adt(a)) => {
                let _ = write!(out, "d{}", a.0);
            }
            Some(TypeHead::Prim(p)) => out.push_str(&p.text()),
        }
        let _ = write!(out, " gen {} methods[", i.generics.len());
        for (k, m) in i.methods.iter().enumerate() {
            if k > 0 {
                out.push(' ');
            }
            let _ = write!(out, "d{}", m.0);
        }
        out.push_str("]\n");
    }

    // Which impls apply: per type and trait, the candidates by head in the trait's impl order
    // (the order impl selection tries them), those that apply listed.
    let mut traits: Vec<DeclId> = sigs.impls_by_trait.keys().copied().collect();
    traits.sort();
    for &t in &table.order {
        let head = w.head(&cx, t);
        for &tr in &traits {
            let Some(list) = sigs.impls_by_trait.get(&tr) else { continue };
            let by_head = &sigs.trait_impls_by_head;
            let none = by_head.get(&(tr, None)).map_or(&[][..], |v| &v[..]);
            let own = match head {
                Some(h) => by_head.get(&(tr, Some(h))).map_or(&[][..], |v| &v[..]),
                None => &[][..],
            };
            let mut positions: Vec<u32> = own.iter().chain(none).copied().collect();
            if positions.is_empty() {
                continue;
            }
            positions.sort_unstable();
            let _ = write!(out, "apply t{} d{} [", table.at(t), tr.0);
            let mut first = true;
            for i in positions.into_iter().filter_map(|k| list.get(k as usize).copied()) {
                let Some(imp) = sigs.impl_def(i) else { continue };
                if w.impl_applies(&mut cx, imp, t, &[]).is_some() {
                    if !first {
                        out.push(' ');
                    }
                    let _ = write!(out, "d{}", i.0);
                    first = false;
                }
            }
            out.push_str("]\n");
        }
    }
    out
}

/// Sort key of an impl head: none, then ADTs by D, then built-in types by spelling.
type HeadKey = (u8, u32, String);

/// Sort key of an impl line: the trait (0: inherent, else D + 1), the head, the item.
type ImplKey = (u32, HeadKey, crate::ids::ItemId);

/// The [`HeadKey`] of `h`.
fn head_key(h: Option<TypeHead>) -> HeadKey {
    match h {
        None => (0, 0, String::new()),
        Some(TypeHead::Adt(d)) => (1, d.0, String::new()),
        Some(TypeHead::Prim(p)) => (2, 0, p.text()),
    }
}

/// ` size S align A` and the field offsets of a struct, tuple, or enum; or why there is no
/// layout.
fn layout(cx: &mut Overlay<'_>, tc: &Typeck, t: TypeId, out: &mut String) {
    let sigs = &tc.sigs;
    match traits::type_bytes(&mut WithSigs { cx, sigs }, t) {
        Ok((size, align)) => {
            let _ = write!(out, " size {size} align {align}");
            let is_enum = match cx.kind(t) {
                TypeKind::Adt { def, .. } => sigs.adt(*def).map(|a| a.is_enum),
                TypeKind::Tuple(_) => Some(false),
                _ => None,
            };
            let Some(is_enum) = is_enum else { return };
            let Some(offs) = traits::field_offsets(cx, sigs, t) else { return };
            let list = |out: &mut String, v: &[u64]| {
                out.push('[');
                for (k, o) in v.iter().enumerate() {
                    if k > 0 {
                        out.push(' ');
                    }
                    let _ = write!(out, "{o}");
                }
                out.push(']');
            };
            if is_enum {
                out.push_str(" tag 4 variants[");
                for (k, v) in offs.iter().enumerate() {
                    if k > 0 {
                        out.push(' ');
                    }
                    list(out, v);
                }
                out.push(']');
            } else {
                out.push_str(" fields");
                list(out, offs.first().map_or(&[][..], |v| &v[..]));
            }
        }
        Err(NoLayout::Overflow) => out.push_str(" size overflow"),
        Err(NoLayout::Unknown) => out.push_str(match traits::unlaid(&mut WithSigs { cx, sigs }, t) {
            Unlaid::Param => " size param",
            Unlaid::Unsized => " size unsized",
            Unlaid::Other => " size none",
        }),
    }
}
