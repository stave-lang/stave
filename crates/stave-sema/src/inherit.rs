//! The features a generic instance inherits from its type arguments (LANG.md §4 "Features",
//! contracts/lang-details.md "Features of generic instances"), decided once per instance on the
//! call graph after type checking. Lowering compiles the instance and its closures with them;
//! the call rule below checks them.
//!
//! **The rule.** An instance needs the features its generic declares, plus those whose registers
//! hold each fixed vector or mask its substituted signature and body hold by value
//! ([`held_wide`]), plus what each generic instance it calls, drops (through drop glue), or makes
//! into a fn value inherits. A type the body holds by value: the type of a param, the return, a
//! local, or an expression, except an expression that is only a place the body does not read
//! whole (the operand of `&` / `&mut`, the head of a postfix chain, the target of a plain `=`, a
//! `match` scrutinee, a `for` source). Only a type mentioning a type parameter counts: a type the
//! body spells is gated where it is spelled. So `first<T>(xs: &[T]) -> Option<&T>` at `f32x8`
//! inherits nothing (it moves no `T`), and `id<T>(x: T) -> T` at `f32x8` inherits `avx2`.
//!
//! **Checks.** A non-generic fn (its closures included) that calls, drops, or makes a fn value of
//! an instance whose inherited features neither its `needs` nor the target baseline has is
//! E0704 case `inherited` at the first such site, naming the type argument; where no feature of
//! the target's arch holds the vector, E0702 there. A generic instance never fails: it inherits.

use rustc_hash::FxHashMap;
use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind as Syn, Term};
use stave_target::{FeatureSet, Triple};

use crate::Program;
use crate::callgraph::{CallGraph, EdgeKind, NodeKind};
use crate::itemcx::{Brackets, ItemCx};
use crate::mono::InstKey;
use crate::sigs::{NameCx, Sigs};
use crate::typeck::{TypedBody, is_expr_kind};
use crate::types::{GArg, Interner, TypeId, TypeKind, TypeTable};

/// The widest fixed vector or mask that a value of type `t` holds by value (not behind a
/// reference, pointer, slice, or fn type), with its width in bits.
pub(crate) fn held_wide<I: Interner + ?Sized>(cx: &mut I, sigs: &Sigs, t: TypeId) -> Option<(TypeId, u32)> {
    let mut seen: rustc_hash::FxHashSet<TypeId> = rustc_hash::FxHashSet::default();
    let mut stack = vec![t];
    let mut best: Option<(TypeId, u32)> = None;
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        let bits = match cx.kind(t).clone() {
            TypeKind::Vector { lane, n: crate::prim::Lanes::Fixed(k) } => {
                crate::types::lane_bits(cx.kind(lane)) * k as u32
            }
            TypeKind::Mask { width, n: crate::prim::Lanes::Fixed(k) } => width as u32 * k as u32,
            TypeKind::Tuple(ts) => {
                stack.extend(ts.iter().rev());
                continue;
            }
            TypeKind::Array { elem, .. } => {
                stack.push(elem);
                continue;
            }
            TypeKind::Adt { .. } => {
                let vs = crate::mono::adt_fields(sigs, cx, t).unwrap_or_default();
                stack.extend(vs.into_iter().flatten().rev());
                continue;
            }
            _ => continue,
        };
        if best.is_none_or(|b| bits > b.1) {
            best = Some((t, bits));
        }
    }
    best
}

/// What holding values needs on the target: features above the baseline, and the widest vector
/// no feature of the target's arch holds (with its width).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Need {
    pub feats: u64,
    pub wide: Option<(TypeId, u32)>,
}

impl Need {
    /// What holding a `bits`-wide vector `v` needs on `target`.
    pub(crate) fn of(target: Triple, v: TypeId, bits: u32) -> Need {
        let base = target.baseline();
        if bits <= crate::gates::feature_vector_bits(base) {
            return Need::default();
        }
        let fs = crate::gates::width_features(target, bits);
        if fs.is_empty() {
            return Need { feats: 0, wide: Some((v, bits)) };
        }
        Need { feats: FeatureSet::from_features(fs).bits() & !base.bits(), wide: None }
    }

    /// Adds `o`; whether that changed `self`.
    fn join(&mut self, o: Need) -> bool {
        let old = *self;
        self.feats |= o.feats;
        if self.wide.is_none() {
            self.wide = o.wide;
        }
        *self != old
    }

    fn is_empty(&self) -> bool {
        self.feats == 0 && self.wide.is_none()
    }
}

/// The parent of each node of the body `lo..=root` (`u32::MAX`: none).
fn parents(icx: &ItemCx<'_>, lo: u32, root: u32) -> Vec<u32> {
    let mut v = vec![u32::MAX; (root + 1 - lo) as usize];
    for n in lo..=root {
        for k in icx.kids_vec(n) {
            if let Some(slot) = k.checked_sub(lo).and_then(|i| v.get_mut(i as usize)) {
                *slot = n;
            }
        }
    }
    v
}

/// Whether expression `n` (parent `p`) is only a place the body does not read whole there.
fn place_only(icx: &ItemCx<'_>, n: u32, p: u32) -> bool {
    match icx.kind(p) {
        Some(Syn::Unary) => icx.is(icx.a(p), Term::Amp) || icx.is(icx.a(p), Term::AmpAmp),
        Some(Syn::Postfix) => icx.first_child(p) == Some(n),
        Some(Syn::ExprStmt) => {
            let kids = icx.kids_vec(p);
            kids.len() >= 3 && kids[0] == n && icx.is(icx.a(kids[1]), Term::Eq)
        }
        Some(Syn::MatchExpr) => icx.first_child(p) == Some(n),
        Some(Syn::Stmt) => icx.is(icx.a(p), Term::KwMatch) && icx.first_child(p) == Some(n),
        Some(Syn::ForStmt) => true,
        _ => false,
    }
}

/// The types mentioning a type parameter that body `b` (of a fn with params `params` and return
/// `ret`) holds by value, each once (module docs).
fn body_values<I: Interner + ?Sized>(
    icx: &ItemCx<'_>,
    cx: &I,
    b: &TypedBody,
    params: &[TypeId],
    ret: TypeId,
) -> Vec<TypeId> {
    let mut out: Vec<TypeId> = Vec::new();
    let add = |t: TypeId, out: &mut Vec<TypeId>| {
        if t.is_none() || !cx.has_params(t) || out.contains(&t) {
            return;
        }
        if matches!(cx.kind(t), TypeKind::Ref { .. } | TypeKind::Ptr { .. } | TypeKind::Fn { .. }) {
            return;
        }
        out.push(t);
    };
    for &t in params.iter().chain(std::iter::once(&ret)) {
        add(t, &mut out);
    }
    for &(_, t) in &b.locals {
        add(t, &mut out);
    }
    let lo = b.lo;
    let up = parents(icx, lo, b.node);
    for (i, &t) in b.node_types.iter().enumerate() {
        let n = lo + i as u32;
        if t.is_none() || !cx.has_params(t) || !icx.kind(n).is_some_and(is_expr_kind) {
            continue;
        }
        let p = up.get(i).copied().unwrap_or(u32::MAX);
        if p != u32::MAX && place_only(icx, n, p) {
            continue;
        }
        add(t, &mut out);
    }
    out
}

/// The inherited features of the generic instances (lowering's feature sets) and the
/// diagnostics of the call rule on them (module docs).
pub(crate) struct Inherited {
    /// Per generic instance whose inherited features are not all declared: those features
    /// beyond the declared ones and the target baseline (graph type ids).
    pub features: FxHashMap<InstKey, u64>,
    pub diagnostics: Vec<Diagnostic>,
}

pub(crate) fn check(
    prog: &Program,
    sigs: &Sigs,
    table: &TypeTable,
    bodies: &[TypedBody],
    g: &CallGraph,
    brackets: &[Brackets],
) -> Inherited {
    let target = prog.target;
    let base = target.baseline().bits();
    let mut cx = g.types(table);
    let n = g.nodes.len();
    let declared = |d: crate::ids::DeclId| sigs.fn_sig(d).map_or(0, |s| s.features);
    // the node whose features a node's code has: a closure's enclosing instance
    let owner: Vec<u32> = (0..n as u32).map(|v| g.fn_of(v).unwrap_or(v)).collect();
    // nodes that inherit: generic instances with a body, and drop glue
    let inherits: Vec<bool> = g
        .nodes
        .iter()
        .map(|x| match &x.kind {
            NodeKind::Fn { inst, body, .. } => *body && !inst.args.is_empty(),
            NodeKind::Glue { .. } => true,
            _ => false,
        })
        .collect();
    // per body (by decl): the param-mentioning types it holds by value
    let mut values: FxHashMap<crate::ids::DeclId, Vec<TypeId>> = FxHashMap::default();
    let mut own: Vec<Need> = vec![Need::default(); n];
    let mut held_memo: FxHashMap<TypeId, Need> = FxHashMap::default();
    for v in 0..n {
        let NodeKind::Fn { inst, body: true, .. } = &g.nodes[v].kind else { continue };
        if inst.args.is_empty() {
            continue;
        }
        let vals = values.entry(inst.decl).or_insert_with(|| {
            let Some(sig) = sigs.fn_sig(inst.decl) else { return Vec::new() };
            let Some(b) = bodies.binary_search_by_key(&inst.decl, |b| b.decl).ok().map(|i| &bodies[i]) else {
                return Vec::new();
            };
            let Some(icx) = brackets.get(b.item.idx()).and_then(|br| ItemCx::new(prog, b.item, br)) else {
                return Vec::new();
            };
            body_values(&icx, &cx, b, &sig.params, sig.ret)
        });
        let vals = vals.clone();
        let mut need = Need::default();
        for t in vals {
            let s = cx.subst(t, &inst.args, None);
            let h = match held_memo.get(&s) {
                Some(h) => *h,
                None => {
                    let h =
                        held_wide(&mut cx, sigs, s).map_or(Need::default(), |(v, b)| Need::of(target, v, b));
                    held_memo.insert(s, h);
                    h
                }
            };
            need.join(h);
        }
        own[v] = need;
    }
    // dependencies: (owner, dependency, site) for calls, drops, spawns, and fn values made
    let mut deps: Vec<(u32, u32, Span)> = Vec::new();
    for v in 0..n as u32 {
        for e in g.succ(v) {
            if matches!(e.kind, EdgeKind::Call | EdgeKind::Drop | EdgeKind::Spawn) {
                deps.push((owner[v as usize], e.to, e.site));
            }
        }
    }
    for &(f, t, s) in &g.made {
        deps.push((owner[f as usize], t, s));
    }
    let mut rdeps: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(o, m, _) in &deps {
        if inherits[o as usize] && m != o {
            rdeps[m as usize].push(o);
        }
    }
    let mut out_deps: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(o, m, _) in &deps {
        if inherits[o as usize] {
            out_deps[o as usize].push(m);
        }
    }
    // what a node passes on to a caller: inherited features only (declared ones are checked
    // where the call is written)
    let passes = |f: &[Need], m: usize| -> Need {
        if !inherits[m] {
            return Need::default();
        }
        let mut x = f[m];
        if let NodeKind::Fn { inst, .. } = &g.nodes[m].kind {
            x.feats &= !declared(inst.decl);
        }
        x.feats &= !base;
        x
    };
    let mut feats: Vec<Need> = own.clone();
    let mut queue: Vec<u32> = (0..n as u32).filter(|&v| inherits[v as usize]).collect();
    let mut queued: Vec<bool> = inherits.clone();
    while let Some(v) = queue.pop() {
        queued[v as usize] = false;
        let mut x = own[v as usize];
        for &m in &out_deps[v as usize] {
            x.join(passes(&feats, m as usize));
        }
        if feats[v as usize].join(x) {
            for &r in &rdeps[v as usize] {
                if !queued[r as usize] {
                    queued[r as usize] = true;
                    queue.push(r);
                }
            }
        }
    }
    let mut features: FxHashMap<InstKey, u64> = FxHashMap::default();
    for v in 0..n {
        if let NodeKind::Fn { inst, .. } = &g.nodes[v].kind
            && inherits[v]
        {
            let x = passes(&feats, v);
            if x.feats != 0 {
                *features.entry(inst.clone()).or_default() |= x.feats;
            }
        }
    }
    // the call rule at the non-generic callers
    let mut diagnostics = Vec::new();
    let mut done: rustc_hash::FxHashSet<(u32, u32)> = rustc_hash::FxHashSet::default();
    for &(o, m, site) in &deps {
        let NodeKind::Fn { inst: caller, .. } = &g.nodes[o as usize].kind else { continue };
        if !caller.args.is_empty() || !done.insert((o, m)) {
            continue;
        }
        let x = passes(&feats, m as usize);
        if x.is_empty() {
            continue;
        }
        let have = declared(caller.decl) | base;
        let missing = x.feats & !have;
        if missing == 0 && x.wide.is_none() {
            continue;
        }
        let Some(callee) = blame(g, &feats, &passes, m) else { continue };
        let names = NameCx { prog, generics: &[] };
        let show = |cx: &crate::types::Overlay<'_>, t: TypeId| crate::types::display(cx, &names, t);
        if let Some((v, bits)) = x.wide {
            let (ty, b, mx) =
                (show(&cx, v), bits.to_string(), crate::gates::max_vector_bits(target).to_string());
            diagnostics.extend(crate::diag::mk(
                "E0702",
                None,
                &[("type", &ty), ("bits", &b), ("max", &mx)],
                site,
                &[],
            ));
            continue;
        }
        let width = crate::gates::feature_vector_bits(FeatureSet::from_bits(have).unwrap_or_default());
        let arg = callee
            .args
            .iter()
            .filter_map(|a| match a {
                GArg::Type(t) => Some(*t),
                GArg::Const(_) => None,
            })
            .find(|&t| crate::gates::wide_part(&cx, t, width).is_some())
            .or_else(|| {
                callee.args.iter().find_map(|a| match a {
                    GArg::Type(t) => Some(*t),
                    GArg::Const(_) => None,
                })
            });
        let arg = arg.map_or_else(String::new, |t| show(&cx, t));
        let missing_names = crate::types::feature_names(missing);
        if missing_names.is_empty() {
            continue;
        }
        let f = missing_names.join(", ");
        let path = prog.decl_path(callee.decl);
        let roles: Vec<(&str, Span)> =
            prog.decl(callee.decl).map(|d| ("callee", d.name_span)).into_iter().collect();
        diagnostics.extend(crate::diag::mk(
            "E0704",
            Some("inherited"),
            &[("callee", &path), ("feature", &f), ("arg", &arg)],
            site,
            &roles,
        ));
    }
    Inherited { features, diagnostics }
}

/// The instance a caller reaches through node `m` that brings the features: `m` itself, or
/// through drop glue the first generic instance passing features on.
fn blame(g: &CallGraph, feats: &[Need], passes: &dyn Fn(&[Need], usize) -> Need, m: u32) -> Option<InstKey> {
    let mut seen = vec![false; g.nodes.len()];
    let mut stack = vec![m];
    while let Some(v) = stack.pop() {
        if std::mem::replace(&mut seen[v as usize], true) {
            continue;
        }
        match &g.nodes[v as usize].kind {
            NodeKind::Fn { inst, .. } => {
                if !passes(feats, v as usize).is_empty() {
                    return Some(inst.clone());
                }
            }
            NodeKind::Glue { .. } => {
                stack.extend(g.succ(v).iter().filter(|e| e.kind == EdgeKind::Drop).map(|e| e.to));
            }
            _ => {}
        }
    }
    None
}
