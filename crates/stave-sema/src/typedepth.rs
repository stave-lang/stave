//! Type nesting depth (E0005 case `type`).
//!
//! **Depth.** A type without components (a scalar, `str`, a mask, a generic parameter) is 1
//! level deep; a reference, raw pointer, array, slice, tuple, fn type, vector, or range is one
//! level above its deepest component, and a struct or enum one level above its deepest type
//! argument and its deepest field (after substitution). A field or element reached through a
//! reference, pointer, slice, or fn type adds no levels to its holder: the pointee is a type of
//! its own and is measured by itself.
//!
//! **What is measured.** Every type the program writes or builds: fn signatures, fields, alias
//! targets, impl self types, every value type of every checked body (generic bodies with their
//! parameters as leaves), the type args of every instance and the value types of its body after
//! substitution, the types of drop glue, and, closed over components and fields, every type any
//! of these reaches. A type deeper than [`MAX_TYPE_DEPTH`] whose parts and components are not is
//! reported once, at the first place (in that order) that reaches it; the types holding it are
//! not reported again.
//!
//! So no type that lowering, layout, or the IR sees nests deeper than the limit, and every
//! recursion over type structure in the checker (trait queries, `size_of`, drop analysis,
//! substitution) bounds itself by the same limit and answers the rejecting side past it.

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::{Diagnostic, Span};

use crate::Program;
use crate::callgraph::{CallGraph, NodeKind};
use crate::diag::mk;
use crate::ids::DeclId;
use crate::itemcx::{Brackets, ItemCx};
use crate::sigs::{DeclSig, Sigs};
use crate::typeck::TypedBody;
use crate::types::{ConstArg, GArg, Interner, Overlay, TypeId, TypeKind, TypeTable};

/// The deepest type nesting a program may use (module docs), shared by every recursion over
/// type structure in the checker, lowering, layout, and the IR.
pub const MAX_TYPE_DEPTH: u32 = stave_ir::types::MAX_TYPE_DEPTH as u32;

/// Most distinct types one check measures; past it the type being checked is rejected.
const TYPE_BUDGET: usize = 1 << 21;

struct Walk<'a, 'g> {
    cx: Overlay<'g>,
    sigs: &'a Sigs,
    /// Height of the type term (components only, no fields), and whether a generic parameter
    /// occurs in it.
    term: FxHashMap<TypeId, (u32, bool)>,
    /// Depth (module docs).
    depth: FxHashMap<TypeId, u32>,
    /// Types whose depth is being computed (a type holding itself through a pointer).
    active: FxHashSet<TypeId>,
    /// Types measured, their components queued.
    seen: FxHashSet<TypeId>,
    queue: Vec<TypeId>,
    /// Types too deep whose parts and components are not.
    causes: FxHashSet<TypeId>,
    over_budget: bool,
}

impl<'a, 'g> Walk<'a, 'g> {
    fn new(cx: Overlay<'g>, sigs: &'a Sigs) -> Self {
        Walk {
            cx,
            sigs,
            term: FxHashMap::default(),
            depth: FxHashMap::default(),
            active: FxHashSet::default(),
            seen: FxHashSet::default(),
            queue: Vec::new(),
            causes: FxHashSet::default(),
            over_budget: false,
        }
    }

    /// Term height and parameter occurrence, by an explicit post-order walk (term graphs are
    /// acyclic: components are interned before the types holding them).
    fn term(&mut self, t: TypeId) -> (u32, bool) {
        if let Some(&r) = self.term.get(&t) {
            return r;
        }
        let mut stack = vec![(t, false)];
        let mut kids = Vec::new();
        while let Some((u, expanded)) = stack.pop() {
            if self.term.contains_key(&u) {
                continue;
            }
            kids.clear();
            let k = self.cx.kind(u);
            k.children(&mut kids);
            let param = matches!(k, TypeKind::Param(_) | TypeKind::ConstParam(_) | TypeKind::SelfTy)
                || matches!(k, TypeKind::Adt { args, .. }
                    if args.iter().any(|a| matches!(a, GArg::Const(ConstArg::Param(_)))))
                || matches!(k, TypeKind::Array { len: ConstArg::Param(_), .. });
            if !expanded {
                stack.push((u, true));
                stack.extend(kids.iter().filter(|c| !self.term.contains_key(c)).map(|&c| (c, false)));
                continue;
            }
            let (mut h, mut p) = (0u32, param);
            for c in &kids {
                let (ch, cp) = self.term.get(c).copied().unwrap_or((u32::MAX, false));
                h = h.max(ch);
                p |= cp;
            }
            self.term.insert(u, (h.saturating_add(1), p));
        }
        self.term.get(&t).copied().unwrap_or((u32::MAX, false))
    }

    /// The parts of `t` held by value: array, tuple, range, and vector elements, ADT fields.
    fn parts(&mut self, t: TypeId) -> Vec<TypeId> {
        match self.cx.kind(t).clone() {
            TypeKind::Array { elem, .. } | TypeKind::Range { elem, .. } => vec![elem],
            TypeKind::Vector { lane, .. } => vec![lane],
            TypeKind::Tuple(ts) => ts.to_vec(),
            TypeKind::Adt { def, args } => {
                let Some(adt) = self.sigs.adt(def) else { return Vec::new() };
                let fields: Vec<TypeId> =
                    adt.variants.iter().flat_map(|v| v.fields.iter().map(|f| f.ty)).collect();
                fields.into_iter().map(|f| self.cx.subst(f, &args, None)).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Depth of `t` (module docs). Components that are not parts are queued to be measured by
    /// themselves. A chain of parts ends: a type holding itself by value is cut (it has no
    /// layout, diagnosed elsewhere), and an endless family of distinct parts grows its type
    /// arguments, so its term height passes the limit.
    fn depth(&mut self, t: TypeId) -> u32 {
        if let Some(&d) = self.depth.get(&t) {
            return d;
        }
        let (k, _) = self.term(t);
        if k > MAX_TYPE_DEPTH {
            self.depth.insert(t, k);
            let c = self.term_cause(t);
            self.causes.insert(c);
            return k;
        }
        if !self.active.insert(t) {
            return k;
        }
        let mut d = k;
        let mut part_too_deep = false;
        for p in self.parts(t) {
            let pd = self.depth(p);
            part_too_deep |= pd > MAX_TYPE_DEPTH;
            d = d.max(pd.saturating_add(1));
        }
        self.active.remove(&t);
        self.depth.insert(t, d);
        if d > MAX_TYPE_DEPTH {
            if !part_too_deep {
                self.causes.insert(t);
            }
        } else {
            let mut kids = Vec::new();
            self.cx.kind(t).children(&mut kids);
            self.queue.extend(kids.into_iter().filter(|c| !self.seen.contains(c)));
        }
        d
    }

    /// The component of `t` (or `t`) whose term height is just past the limit, along the first
    /// component that is past it.
    fn term_cause(&mut self, mut t: TypeId) -> TypeId {
        let mut kids = Vec::new();
        loop {
            kids.clear();
            self.cx.kind(t).children(&mut kids);
            match kids.iter().copied().find(|&c| self.term(c).0 > MAX_TYPE_DEPTH) {
                Some(c) => t = c,
                None => return t,
            }
        }
    }

    /// Measures `t` and everything it reaches. True if it reaches a type too deep whose parts
    /// and components are not, and no earlier check found that type.
    fn check(&mut self, t: TypeId) -> bool {
        if self.over_budget {
            return false;
        }
        let before = self.causes.len();
        self.queue.push(t);
        while let Some(u) = self.queue.pop() {
            if !self.seen.insert(u) {
                continue;
            }
            if self.seen.len() > TYPE_BUDGET {
                self.over_budget = true;
                self.queue.clear();
                return true;
            }
            self.depth(u);
        }
        self.causes.len() > before
    }

    fn has_params(&mut self, t: TypeId) -> bool {
        self.term(t).1
    }
}

/// E0005 case `type` diagnostics (module docs). `bodies` sorted by decl.
pub(crate) fn check(
    prog: &Program,
    sigs: &Sigs,
    table: &TypeTable,
    bodies: &[TypedBody],
    graph: &CallGraph,
    brackets: &[Brackets],
) -> Vec<Diagnostic> {
    let mut w = Walk::new(graph.types(table), sigs);
    let mut out = Vec::new();
    let mut report = |at: Span| out.extend(mk("E0005", Some("type"), &[], at, &[]));

    // Declarations, in source order; one report per declaration.
    let mut roots: Vec<(Span, u32, TypeId)> = Vec::new();
    for (k, d) in prog.decls.iter().enumerate() {
        if !prog.item(d.item).is_some_and(|it| it.ok) {
            continue;
        }
        let k = k as u32;
        match sigs.get(DeclId(k)) {
            DeclSig::Fn(f) => {
                roots.extend(f.params.iter().zip(&f.param_spans).map(|(&t, &s)| (s, k, t)));
                roots.push((f.ret_span.unwrap_or(d.name_span), k, f.ret));
            }
            DeclSig::Adt(a) => {
                // the type itself, with its parameters as arguments: used or not
                let args: Box<[GArg]> = (0..a.generics.len() as u32)
                    .zip(&a.generics)
                    .map(|(i, g)| {
                        if g.is_const {
                            GArg::Const(ConstArg::Param(i))
                        } else {
                            GArg::Type(w.cx.mk_param(i))
                        }
                    })
                    .collect();
                let own = w.cx.intern(TypeKind::Adt { def: DeclId(k), args });
                roots.push((d.name_span, k, own));
                roots.extend(a.variants.iter().flat_map(|v| v.fields.iter().map(|f| (f.span, k, f.ty))));
            }
            DeclSig::Alias(a) => roots.push((d.name_span, k, a.target)),
            DeclSig::Impl(i) => roots.push((d.name_span, k, i.self_ty)),
            DeclSig::Const(c) | DeclSig::Static(c) => roots.push((d.name_span, k, c.ty)),
            _ => {}
        }
    }
    roots.sort();
    let mut reported: FxHashSet<u32> = FxHashSet::default();
    for (at, k, t) in roots {
        if !t.is_none() && !reported.contains(&k) && w.check(t) {
            reported.insert(k);
            report(at);
        }
    }

    // Bodies, generic ones with their parameters as leaves; one report per body.
    let body_types = |b: &TypedBody| -> Vec<(TypeId, u32)> {
        let mut v: Vec<(TypeId, u32)> =
            b.node_types.iter().enumerate().map(|(i, &t)| (t, b.lo + i as u32)).collect();
        v.extend(b.locals.iter().map(|&(n, t)| (t, n)));
        for s in &b.sites {
            v.extend(s.inst.args.iter().filter_map(|a| match a {
                GArg::Type(t) => Some((*t, s.node)),
                GArg::Const(_) => None,
            }));
        }
        v.retain(|(t, _)| !t.is_none());
        v
    };
    // A body with other errors is measured too: its types are what it wrote.
    for b in bodies {
        let Some(icx) = brackets.get(b.item.idx()).and_then(|br| ItemCx::new(prog, b.item, br)) else {
            continue;
        };
        for (t, n) in body_types(b) {
            if w.check(t) {
                report(icx.node_span(n));
                break;
            }
        }
    }

    // Instances: their type args and their bodies' types after substitution; drop glue.
    let body_of = |d: DeclId| {
        bodies.binary_search_by_key(&d, |b| b.decl).ok().map(|i| &bodies[i]).filter(|b| !b.has_error)
    };
    let mut generic_types: FxHashMap<DeclId, Vec<TypeId>> = FxHashMap::default();
    for (n, node) in graph.nodes.iter().enumerate() {
        let at = graph
            .in_start
            .get(n)
            .and_then(|&s| graph.in_edges.get(s as usize))
            .filter(|_| graph.in_start.get(n + 1).is_some_and(|&e| e > graph.in_start[n]))
            .and_then(|&e| graph.out.get(e as usize))
            .map_or(node.span, |e| e.site);
        match &node.kind {
            NodeKind::Fn { inst, .. } if !inst.args.is_empty() => {
                let mut roots: Vec<TypeId> = inst
                    .args
                    .iter()
                    .filter_map(|a| match a {
                        GArg::Type(t) => Some(*t),
                        GArg::Const(_) => None,
                    })
                    .collect();
                if let Some(b) = body_of(inst.decl) {
                    let ts = match generic_types.get(&inst.decl) {
                        Some(ts) => ts.clone(),
                        None => {
                            let mut ts: Vec<TypeId> = body_types(b).into_iter().map(|(t, _)| t).collect();
                            ts.sort();
                            ts.dedup();
                            ts.retain(|&t| w.has_params(t));
                            generic_types.insert(inst.decl, ts.clone());
                            ts
                        }
                    };
                    for t in ts {
                        let s = w.cx.subst(t, &inst.args, None);
                        roots.push(s);
                    }
                }
                for t in roots {
                    if w.check(t) {
                        report(at);
                        break;
                    }
                }
            }
            NodeKind::Glue { ty } if w.check(*ty) => report(at),
            _ => {}
        }
    }
    out
}
