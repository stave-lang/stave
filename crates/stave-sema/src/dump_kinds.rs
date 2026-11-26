//! The `kinds` dump (contracts/stage-gates.md): per type of the `sigs` table its kind, depth,
//! and layout, then the impl table, then which impls apply to each type. Read from the signature
//! table, the canonical type table, and the trait and layout rules, never from a body.
//!
//! Every answer is taken in the empty environment: a generic param has no bound and is not
//! `view`, so a type with params gets what holds for every instance (a layout that does not
//! depend on them, `&T` Copy) and nothing a bound would add.

use std::fmt::Write;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::Program;
use crate::dump_sigs::TypeRank;
use crate::ids::DeclId;
use crate::resolve::TypeHead;
use crate::sigs::TraitRef;
use crate::traits::{self, Answers, NoLayout, Unlaid, WithSigs, World};
use crate::typeck::Typeck;
use crate::types::{Interner, Overlay, TypeId, TypeKind};

/// The kind flags, in print order.
const KIND_NAMES: [&str; 10] =
    ["copy", "view", "send", "sync", "sized", "flat", "frozen", "drop-impl", "drop", "lock"];

/// The `kinds` dump: one `t<T>` line per `sigs` table type, by T; one `impl` line per impl, by
/// (trait, head, item); one `apply` line per (type, trait) with a candidate impl, by (T, trait).
///
/// Runs on the big-stack workers ([`crate::pool`]), as the checker's own trait queries do: a
/// query on a type of a cycle (through pointers, or by value, E0423) recurses through every
/// type of the cycle not answered yet, which [`Warm::parts_first`] cannot order away.
pub fn dump_kinds(prog: &Program, tc: &Typeck) -> String {
    crate::pool::on_big_stacks(|| dump_kinds_here(prog, tc))
}

fn dump_kinds_here(prog: &Program, tc: &Typeck) -> String {
    let table = TypeRank::of(prog, tc);
    let sigs = &tc.sigs;
    let w = World { prog, sigs };
    let mut cx = Overlay::new(&tc.types);
    let mut out = String::new();

    let depths = crate::typedepth::depths(sigs, &tc.types, &table.order);
    let mut drop_memo = FxHashMap::default();
    let mut kept = Answers::default();
    let mut warm = Warm { traits: warm_traits(&w), done: FxHashSet::default() };
    for &t in &table.order {
        warm.parts_first(&w, &mut cx, t, &mut kept);
    }
    for (&t, depth) in table.order.iter().zip(depths) {
        let k = crate::kinds::kind_of(w, &mut cx, t, &[], &mut kept);
        let bits = [
            k.copy,
            k.view,
            k.send,
            k.sync,
            k.sized,
            k.flat,
            k.frozen,
            crate::mono::drop_impl_kept(&w, &mut cx, t, &mut kept),
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
                if w.impl_applies_kept(&mut cx, imp, t, &[], &mut kept).is_some() {
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

/// The traits a query of this dump can reach on a part of a type: Copy and the derived and
/// marker traits ([`crate::kinds::kind_of`]), and every trait an impl param is bounded by
/// without trait arguments (impl selection asks those of the impl's arguments).
fn warm_traits(w: &World<'_>) -> Vec<DeclId> {
    let lang = &w.sigs.lang;
    let mut out: Vec<DeclId> =
        [lang.copy, lang.send, lang.sync, lang.flat, lang.frozen].into_iter().collect();
    for s in &w.sigs.decls {
        let crate::sigs::DeclSig::Impl(i) = s else { continue };
        let bounds = i.generics.iter().flat_map(|g| g.bounds.iter());
        out.extend(bounds.filter(|b| b.args.is_empty()).map(|b| b.trait_decl));
    }
    out.retain(|d| d.is_some());
    out.sort_unstable();
    out.dedup();
    out
}

/// Asks the trait queries of every part of a type before the type's own, deepest first, so
/// each query finds the answers one step down kept ([`Answers`]) and descends no further than
/// a cycle of parts: the dump's answers for a type nested past the depth limit, or holding a
/// chain of references or type arguments as long, take no stack in proportion to it.
struct Warm {
    traits: Vec<DeclId>,
    /// Types expanded: their parts answered, or pending below them on the stack.
    done: FxHashSet<TypeId>,
}

impl Warm {
    /// The queries of `root` and everything it reaches through components (type arguments,
    /// elements, pointees, fn params and result) and fields after substitution, in post-order
    /// by an explicit stack. A type met again while its parts are pending is a cycle: its
    /// queries run when its own turn comes.
    fn parts_first(&mut self, w: &World<'_>, cx: &mut Overlay<'_>, root: TypeId, kept: &mut Answers) {
        let mut stack = vec![(root, false)];
        let mut kids = Vec::new();
        while let Some((u, expanded)) = stack.pop() {
            if expanded {
                for &tr in &self.traits {
                    w.implements_kept(cx, u, &TraitRef { trait_decl: tr, args: Vec::new() }, &[], kept);
                }
                continue;
            }
            // a type queued twice is expanded at its first pop, below the holder that popped it
            if u.is_none() || !self.done.insert(u) {
                continue;
            }
            stack.push((u, true));
            kids.clear();
            let k = cx.kind(u).clone();
            k.children(&mut kids);
            if let TypeKind::Adt { def, args } = k
                && let Some(adt) = w.sigs.adt(def)
            {
                for f in adt.variants.iter().flat_map(|v| v.fields.iter()) {
                    kids.push(cx.subst(f.ty, &args, None));
                }
            }
            stack.extend(kids.iter().rev().filter(|c| !self.done.contains(c)).map(|&c| (c, false)));
        }
    }
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
