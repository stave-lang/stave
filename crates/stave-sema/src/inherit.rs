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
//! `match` scrutinee, a `for` source). Only a type mentioning a type parameter counts: the rest
//! is the generic body's, gated as in any body ([`gate_bodies`]). So `first<T>(xs: &[T]) -> Option<&T>` at `f32x8`
//! inherits nothing (it moves no `T`), and `id<T>(x: T) -> T` at `f32x8` inherits `avx2`.
//!
//! **Checks.** A non-generic fn (its closures included) that calls, drops, or makes a fn value of
//! an instance whose inherited features neither its `needs` nor the target baseline has is
//! E0704 case `inherited` at the first such site, naming the type argument; where no feature of
//! the target's arch holds the vector, E0702 there. A generic instance never fails: it inherits.
//! Every fn body (its closures included) holds by value only vectors the registers of its own
//! `needs` and the target baseline hold, written or inferred, by the same measure (E0702 at the
//! first other one, [`gate_bodies`]).

use rustc_hash::FxHashMap;
use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind as Syn, Term};
use stave_target::{FeatureSet, Triple};

use crate::Program;
use crate::callgraph::{CallGraph, EdgeKind, NodeKind};
use crate::itemcx::{Brackets, ItemCx};
use crate::mono::InstKey;
use crate::sigs::{NameCx, Sigs};
use crate::traits::MethodRef;
use crate::typeck::{Target, TypedBody, is_expr_kind};
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

/// Whether node `p` is a parenthesized expression `(e)` (a grouping, not a tuple).
fn is_group(icx: &ItemCx<'_>, p: u32) -> bool {
    if icx.kind(p) != Some(Syn::Primary) || !icx.is(icx.a(p), Term::LParen) {
        return false;
    }
    icx.kids_vec(p).len() == 1 && !icx.is(icx.close(icx.a(p)) - 1, Term::Comma)
}

/// How a body uses the value of an expression where it is written.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    /// Read whole.
    Read,
    /// Only a place there: the operand of `&` / `&mut`, the head of a postfix chain, the target
    /// of a plain `=`, a `for` source. Lowering takes its address or projects from it.
    Place,
    /// A `match` scrutinee: matching loads each enum value a pattern tests ([`match_loads`]).
    Match,
}

/// The use of expression `n`, classified at its nearest ancestor that is not a grouping `( )`
/// (`(*r).1` projects like `r.1`). `up`: the parents of the body's nodes from `lo`.
fn use_of(icx: &ItemCx<'_>, up: &[u32], lo: u32, mut n: u32) -> Use {
    loop {
        let Some(&p) = n.checked_sub(lo).and_then(|i| up.get(i as usize)) else { return Use::Read };
        if p == u32::MAX {
            return Use::Read;
        }
        if is_group(icx, p) {
            n = p;
            continue;
        }
        let place = match icx.kind(p) {
            Some(Syn::Unary) => icx.is(icx.a(p), Term::Amp) || icx.is(icx.a(p), Term::AmpAmp),
            Some(Syn::Postfix) => icx.first_child(p) == Some(n),
            Some(Syn::ExprStmt) => {
                let kids = icx.kids_vec(p);
                kids.len() >= 3 && kids[0] == n && icx.is(icx.a(kids[1]), Term::Eq)
            }
            Some(Syn::MatchExpr) if icx.first_child(p) == Some(n) => return Use::Match,
            Some(Syn::Stmt) if icx.is(icx.a(p), Term::KwMatch) && icx.first_child(p) == Some(n) => {
                return Use::Match;
            }
            Some(Syn::ForStmt) => true,
            _ => false,
        };
        return if place { Use::Place } else { Use::Read };
    }
}

/// The enum values a `match` on a scrutinee of type `t` may load: each enum held by value in
/// `t`, through references too (a pattern tests through them), tuples, arrays, struct fields,
/// and enum payloads.
fn match_loads<I: Interner + ?Sized>(cx: &mut I, sigs: &Sigs, t: TypeId) -> Vec<TypeId> {
    let mut out = Vec::new();
    let mut seen: rustc_hash::FxHashSet<TypeId> = rustc_hash::FxHashSet::default();
    let mut stack = vec![t];
    while let Some(t) = stack.pop() {
        if t.is_none() || !seen.insert(t) {
            continue;
        }
        match cx.kind(t).clone() {
            TypeKind::Ref { inner, .. } => stack.push(inner),
            TypeKind::Tuple(ts) => stack.extend(ts.iter().copied()),
            TypeKind::Array { elem, .. } => stack.push(elem),
            TypeKind::Adt { def, .. } => {
                if sigs.adt(def).is_some_and(|a| a.is_enum) {
                    out.push(t);
                }
                let vs = crate::mono::adt_fields(sigs, cx, t).unwrap_or_default();
                stack.extend(vs.into_iter().flatten());
            }
            _ => {}
        }
    }
    out
}

/// The values body `b` holds by value whose type passes `keep` (module docs), with where they
/// are: the one measure of the instance rule and of the body gate.
/// - Its local bindings, closure params included.
/// - Each expression, except where it is only a place ([`Use::Place`]); a `match` scrutinee
///   counts the enums the match loads.
/// - The values a postfix chain makes or reads between its steps, which have no node of their
///   own: a built-in's vector or mask receiver and its result, a literal's value, the operand
///   of `?`, and the result of every call that is not to a statically known fn, method, or
///   operator (a fn or closure value, an intrinsic). A statically known callee makes a value of
///   its own signature, which its features cover and the caller must have: declared ones by
///   E0704, inherited ones by the call rule here.
fn held_values<I: Interner + ?Sized>(
    icx: &ItemCx<'_>,
    cx: &mut I,
    sigs: &Sigs,
    b: &TypedBody,
    keep: &mut dyn FnMut(&mut I, TypeId) -> bool,
) -> Vec<(Span, TypeId)> {
    let mut out: Vec<(Span, TypeId)> = Vec::new();
    for &(n, t) in &b.locals {
        if !t.is_none() && keep(cx, t) {
            out.push((icx.node_span(n), t));
        }
    }
    let mut up: Option<Vec<u32>> = None;
    for (i, &t) in b.node_types.iter().enumerate() {
        let n = b.lo + i as u32;
        if t.is_none() || !icx.kind(n).is_some_and(is_expr_kind) {
            continue;
        }
        let up = up.get_or_insert_with(|| parents(icx, b.lo, b.node));
        match use_of(icx, up, b.lo, n) {
            Use::Place => {}
            Use::Read => {
                if keep(cx, t) {
                    out.push((icx.node_span(n), t));
                }
            }
            Use::Match => {
                for e in match_loads(cx, sigs, t) {
                    if keep(cx, e) {
                        out.push((icx.node_span(n), e));
                    }
                }
            }
        }
    }
    for (tok, tg) in &b.targets {
        let made: [TypeId; 2] = match tg {
            Target::Builtin { recv, ret, .. } => {
                let r = matches!(cx.kind(*recv), TypeKind::Vector { .. } | TypeKind::Mask { .. });
                [if r { *recv } else { TypeId::NONE }, *ret]
            }
            Target::Struct { ty, .. } | Target::Variant { ty, .. } => [*ty, TypeId::NONE],
            Target::FnValue { ty } => match cx.kind(*ty) {
                TypeKind::Fn { ret, .. } => [*ret, TypeId::NONE],
                _ => [*ty, TypeId::NONE],
            },
            // an intrinsic's own generic params are not the body's
            Target::Intrinsic { decl } => {
                let r = sigs.fn_sig(*decl).map_or(TypeId::NONE, |s| s.ret);
                [if r.is_none() || cx.has_params(r) { TypeId::NONE } else { r }, TypeId::NONE]
            }
            Target::Fn { .. } | Target::TraitMethod { .. } | Target::Op { .. } => [TypeId::NONE; 2],
            Target::Method { method, .. } => match method {
                MethodRef::Impl { .. } | MethodRef::Bound { .. } => [TypeId::NONE; 2],
                // built-in methods are recorded as `Target::Builtin`, with their types
                MethodRef::Builtin(_) => [TypeId::NONE; 2],
            },
            Target::Field { .. }
            | Target::TupleField { .. }
            | Target::Column { .. }
            | Target::Index { .. }
            | Target::Slice { .. }
            | Target::Global { .. }
            | Target::SoaRow { .. }
            | Target::SizeOf { .. }
            | Target::FenceStream
            | Target::Old => [TypeId::NONE; 2],
        };
        for t in made {
            if !t.is_none() && keep(cx, t) {
                out.push((icx.tok_span(*tok), t));
            }
        }
    }
    for &(tok, t) in &b.tries {
        if !t.is_none() && keep(cx, t) {
            out.push((icx.tok_span(tok), t));
        }
    }
    out
}

/// The types mentioning a type parameter that body `b` (of a fn with params `params` and return
/// `ret`) holds by value, each once (module docs).
fn body_values<I: Interner + ?Sized>(
    icx: &ItemCx<'_>,
    cx: &mut I,
    sigs: &Sigs,
    b: &TypedBody,
    params: &[TypeId],
    ret: TypeId,
) -> Vec<TypeId> {
    fn keep<I: Interner + ?Sized>(cx: &mut I, t: TypeId) -> bool {
        !t.is_none()
            && cx.has_params(t)
            && !matches!(cx.kind(t), TypeKind::Ref { .. } | TypeKind::Ptr { .. } | TypeKind::Fn { .. })
    }
    let sig: Vec<TypeId> =
        params.iter().copied().chain(std::iter::once(ret)).filter(|&t| keep(cx, t)).collect();
    let held = held_values(icx, cx, sigs, b, &mut keep::<I>);
    let mut out: Vec<TypeId> = Vec::new();
    for t in sig.into_iter().chain(held.into_iter().map(|x| x.1)) {
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// Whether a value of type `t` holds by value a vector or mask wider than `width` bits.
fn wide_held(
    cx: &mut crate::types::Overlay<'_>,
    memo: &mut FxHashMap<TypeId, Option<(TypeId, u32)>>,
    sigs: &Sigs,
    t: TypeId,
    width: u32,
) -> bool {
    let h = match memo.get(&t) {
        Some(h) => *h,
        None => {
            let h = held_wide(cx, sigs, t);
            memo.insert(t, h);
            h
        }
    };
    h.is_some_and(|w| w.1 > width)
}

/// The body gate's result: its diagnostics, and per fn it rejected what holding its own values
/// needs beyond its `needs` and the target baseline (the call rule leaves out a use that misses
/// no more than that: one cause, one diagnostic).
pub(crate) struct Gated {
    pub diagnostics: Vec<Diagnostic>,
    pub needs: FxHashMap<crate::ids::DeclId, Need>,
}

/// The width gate of fn bodies (LANG.md §4 "Types": 256 bits need `avx2`, 512 the AVX-512 part
/// of `v4`): E0702 for a fn body, its closures included, that holds by value a fixed vector or
/// mask wider than the registers of its `needs` and the target baseline ([`held_values`]), at
/// the first such value in source order, whether its type is written or inferred. The part of a
/// type that depends on a type parameter is not gated here: each instance inherits the features
/// it needs. A body with a type error, and a fn whose signature is too wide, already have their
/// diagnostic.
pub(crate) fn gate_bodies(
    prog: &Program,
    sigs: &Sigs,
    table: &TypeTable,
    bodies: &[TypedBody],
    brackets: &[Brackets],
) -> Gated {
    let target = prog.target;
    let mut cx = crate::types::Overlay::new(table);
    let mut memo: FxHashMap<TypeId, Option<(TypeId, u32)>> = FxHashMap::default();
    let mut out = Gated { diagnostics: Vec::new(), needs: FxHashMap::default() };
    for b in bodies {
        if b.has_error {
            continue;
        }
        let Some(sig) = sigs.fn_sig(b.decl) else { continue };
        let have = target.baseline().union(FeatureSet::from_bits(sig.features).unwrap_or_default());
        let width = crate::gates::feature_vector_bits(have);
        let sig_types = sig.params.iter().copied().chain(std::iter::once(sig.ret));
        if sig_types.filter(|t| !t.is_none()).any(|t| crate::gates::wide_part(&cx, t, width).is_some()) {
            continue;
        }
        let Some(icx) = brackets.get(b.item.idx()).and_then(|br| ItemCx::new(prog, b.item, br)) else {
            continue;
        };
        let mut keep =
            |cx: &mut crate::types::Overlay<'_>, t: TypeId| wide_held(cx, &mut memo, sigs, t, width);
        let found = held_values(&icx, &mut cx, sigs, b, &mut keep);
        let Some(&(span, t)) = found.iter().min_by_key(|x| x.0.offset) else { continue };
        let mut need = Need::default();
        for &(_, t) in &found {
            if let Some((v, bits)) = memo.get(&t).copied().flatten() {
                need.join(Need::of(target, v, bits));
            }
        }
        need.feats &= !have.bits();
        out.needs.insert(b.decl, need);
        let Some((v, bits)) = memo.get(&t).copied().flatten() else { continue };
        let ty = crate::types::display(&cx, &NameCx { prog, generics: &[] }, v);
        out.diagnostics.extend(crate::gates::too_wide(&ty, bits, width, span));
    }
    out
}

/// The inherited features of the generic instances (lowering's feature sets) and the
/// diagnostics of the call rule on them (module docs).
pub(crate) struct Inherited {
    /// Per generic instance whose inherited features are not all declared: those features
    /// beyond the declared ones and the target baseline (graph type ids).
    pub features: FxHashMap<InstKey, u64>,
    /// Per type whose drop glue loads a vector wider than the target baseline holds: the
    /// features that hold it (graph type ids). The glue is compiled with them.
    pub glue: FxHashMap<TypeId, u64>,
    pub diagnostics: Vec<Diagnostic>,
}

pub(crate) fn check(
    prog: &Program,
    sigs: &Sigs,
    table: &TypeTable,
    bodies: &[TypedBody],
    g: &CallGraph,
    brackets: &[Brackets],
    gated: &FxHashMap<crate::ids::DeclId, Need>,
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
    let mut held_memo: FxHashMap<TypeId, (Need, u32)> = FxHashMap::default();
    let mut glue: FxHashMap<TypeId, u64> = FxHashMap::default();
    for v in 0..n {
        // drop glue reads the enums it drops whole (their tag), payload included: it holds its
        // type by value, and every dropper must have what that needs
        if let NodeKind::Glue { ty } = &g.nodes[v].kind {
            let h = held_wide(&mut cx, sigs, *ty).map_or(Need::default(), |(w, b)| Need::of(target, w, b));
            if h.feats != 0 {
                glue.insert(*ty, h.feats);
            }
            own[v] = h;
            continue;
        }
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
            body_values(&icx, &mut cx, sigs, b, &sig.params, sig.ret)
        });
        let vals = vals.clone();
        let mut need = Need::default();
        for t in vals {
            let s = cx.subst(t, &inst.args, None);
            let mut held = |cx: &mut crate::types::Overlay<'_>, t: TypeId| match held_memo.get(&t) {
                Some(h) => *h,
                None => {
                    let h = held_wide(cx, sigs, t)
                        .map_or((Need::default(), 0), |(v, b)| (Need::of(target, v, b), b));
                    held_memo.insert(t, h);
                    h
                }
            };
            // a vector the generic body holds whatever the type arguments is the body's, gated
            // against the generic's own `needs`: only a wider one through a type argument counts
            let (h, bits) = held(&mut cx, s);
            if bits > held(&mut cx, t).1 {
                need.join(h);
            }
        }
        own[v] = need;
    }
    // dependencies: (owner, dependency, site) for calls, drops, spawns, and fn values made
    // (and whether it is a drop)
    let mut deps: Vec<(u32, u32, Span, bool)> = Vec::new();
    for v in 0..n as u32 {
        for e in g.succ(v) {
            if matches!(e.kind, EdgeKind::Call | EdgeKind::Drop | EdgeKind::Spawn) {
                deps.push((owner[v as usize], e.to, e.site, e.kind == EdgeKind::Drop));
            }
        }
    }
    for &(f, t, s) in &g.made {
        deps.push((owner[f as usize], t, s, false));
    }
    let mut rdeps: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(o, m, _, _) in &deps {
        if inherits[o as usize] && m != o {
            rdeps[m as usize].push(o);
        }
    }
    let mut out_deps: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(o, m, _, _) in &deps {
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
    // A drop the caller reaches is reported only where nothing else is for the same caller, type
    // argument, and missing features: the value dropped is one the caller made or received by
    // a use reported already (`Vec.new` and the `Vec` drop it brings are one cause).
    let at = |d: &Diagnostic| d.primary().map_or((u32::MAX, u32::MAX), |s| (s.file.0, s.offset));
    let mut uses: Vec<((u32, String, u64), Diagnostic)> = Vec::new();
    let mut drops: FxHashMap<(u32, String, u64), Diagnostic> = FxHashMap::default();
    let mut report = |key: (u32, String, u64), drop: bool, d: Option<Diagnostic>| {
        let Some(d) = d else { return };
        if !drop {
            uses.push((key, d));
            return;
        }
        match drops.get(&key) {
            Some(old) if at(old) <= at(&d) => {}
            _ => {
                drops.insert(key, d);
            }
        }
    };
    let mut done: rustc_hash::FxHashSet<(u32, u32)> = rustc_hash::FxHashSet::default();
    for &(o, m, site, drop) in &deps {
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
        // the caller's body already holds such values itself: its E0702 is the one report
        if let Some(own) = gated.get(&caller.decl)
            && missing & !own.feats == 0
            && (x.wide.is_none() || own.wide.is_some())
        {
            continue;
        }
        let names = NameCx { prog, generics: &[] };
        let show = |cx: &crate::types::Overlay<'_>, t: TypeId| crate::types::display(cx, &names, t);
        let width = crate::gates::feature_vector_bits(FeatureSet::from_bits(have).unwrap_or_default());
        let (path, arg, roles) = match blame(g, &own, &feats, &passes, m) {
            Some(Source::Inst(callee)) => {
                let types = || {
                    callee.args.iter().filter_map(|a| match a {
                        GArg::Type(t) => Some(*t),
                        GArg::Const(_) => None,
                    })
                };
                let arg = types()
                    .find(|&t| crate::gates::wide_part(&cx, t, width).is_some())
                    .or_else(|| types().next());
                let roles: Vec<(&str, Span)> =
                    prog.decl(callee.decl).map(|d| ("callee", d.name_span)).into_iter().collect();
                (prog.decl_path(callee.decl), arg.map_or_else(String::new, |t| show(&cx, t)), roles)
            }
            Some(Source::Glue(t)) => ("drop".to_string(), show(&cx, t), Vec::new()),
            None => ("drop".to_string(), String::new(), Vec::new()),
        };
        if let Some((v, bits)) = x.wide {
            let (ty, b, mx) =
                (show(&cx, v), bits.to_string(), crate::gates::max_vector_bits(target).to_string());
            report(
                (o, arg.clone(), 0),
                drop,
                crate::diag::mk(
                    "E0702",
                    Some("inherited"),
                    &[("type", &ty), ("bits", &b), ("max", &mx), ("callee", &path), ("arg", &arg)],
                    site,
                    &roles,
                ),
            );
            continue;
        }
        let f = crate::types::feature_names(missing).join(", ");
        report(
            (o, arg.clone(), missing),
            drop,
            crate::diag::mk(
                "E0704",
                Some("inherited"),
                &[("callee", &path), ("feature", &f), ("arg", &arg)],
                site,
                &roles,
            ),
        );
    }
    let used: rustc_hash::FxHashSet<&(u32, String, u64)> = uses.iter().map(|x| &x.0).collect();
    let mut diagnostics: Vec<Diagnostic> =
        drops.iter().filter(|(k, _)| !used.contains(k)).map(|(_, d)| d.clone()).collect();
    diagnostics.extend(uses.iter().map(|x| x.1.clone()));
    diagnostics.sort_by_key(at);
    Inherited { features, glue, diagnostics }
}

/// What brings a caller the features it lacks.
enum Source {
    Inst(InstKey),
    /// The drop glue of this type, for the values it loads itself.
    Glue(TypeId),
}

/// What a caller reaches through node `m` that brings the features: `m` itself, or through
/// drop glue a glue that loads such a value or the first generic instance passing features on.
fn blame(
    g: &CallGraph,
    own: &[Need],
    feats: &[Need],
    passes: &dyn Fn(&[Need], usize) -> Need,
    m: u32,
) -> Option<Source> {
    let mut seen = vec![false; g.nodes.len()];
    let mut stack = vec![m];
    while let Some(v) = stack.pop() {
        if std::mem::replace(&mut seen[v as usize], true) {
            continue;
        }
        match &g.nodes[v as usize].kind {
            NodeKind::Fn { inst, .. } => {
                if !passes(feats, v as usize).is_empty() {
                    return Some(Source::Inst(inst.clone()));
                }
            }
            NodeKind::Glue { ty } => {
                if !own[v as usize].is_empty() {
                    return Some(Source::Glue(*ty));
                }
                stack.extend(g.succ(v).iter().filter(|e| e.kind == EdgeKind::Drop).map(|e| e.to));
            }
            _ => {}
        }
    }
    None
}
