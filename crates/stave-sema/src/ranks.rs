//! The lock-rank check: a thread acquires locks in increasing rank order only (E1405), and waits
//! for its child threads only while every lock it holds ranks below what they acquire (E1406).
//!
//! **Ranks.** `std.sync.Mutex<T, R>` and `Once<T, R>` carry their rank `R` (a `u32` const param)
//! in the type. The acquisitions ([`crate::sigs::LockItems`]): an instance of `Mutex.lock`,
//! `Once.get_or_init` (waiting for the value counts as acquiring it), and `lock_pair` (two
//! mutexes of one rank, locked in address order) acquires rank `R`. `try_lock` never waits and
//! acquires nothing; its guard still counts as held.
//!
//! **What a call acquires.** Per call-graph node, the lowest rank anything it can run acquires
//! ([`lowest`]): the minimum of its own rank (an acquisition) and of what its out-edges reach
//! (calls, fn-value calls to the values that reach them, spawns, drops), one value per strongly
//! connected component, callee levels first. A `scope` end, `thread.join`, and `par_*` wait for threads
//! that run what the graph reaches through `Spawn` edges and through the thread fns' fn-value
//! calls, so the same minimum is what those children acquire.
//!
//! **What is held.** The borrow checker reports, per body, the locals that may hold a
//! `MutexGuard` (maybe-initialized and not moved out or dropped) at each call and at each `scope`
//! join ([`BodyLocks`]), with the guard ranks as written: a value, a const generic param of the
//! fn, or a `T: view` param an instance may make a guard. Each fn instance node substitutes its
//! args. A call the CFG shows by no token (the calls an f-string or a `&String` coercion implies)
//! gets every lock the body may hold anywhere. `Once.get_or_init` holds its own rank `R` while
//! it runs `f`.
//!
//! **Rule.** At every call and join of a node, the highest held rank must be below the lowest rank
//! the callee (or the joined children) acquires; equal ranks are a violation too. Instances of a
//! generic fn are checked per instance, an uninstantiated one is not checked. A violation inside a
//! built-in package (`with_lock` calling `f` under its guard, `get_or_init` running `f`) is
//! reported at the user call that leads there.
//!
//! **Drops.** A drop (scope end, temporary, partial drop, the old value an assignment
//! overwrites) runs the dropped type's drop glue, a node of the graph: it is checked like a call,
//! without the local it drops. Locals drop in reverse declaration order, so a value declared
//! before a guard drops after the guard is released. A value moved into a callee and dropped there
//! is that callee's drop: its own check covers the guards it holds, and every caller holding a
//! guard is checked at the call, whose lowest rank includes the callee's drop edges.

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::{Diagnostic, Span};

use crate::Program;
use crate::callgraph::{CallGraph, EdgeKind, NodeKind};
use crate::diag::mk;
use crate::ids::DeclId;
use crate::mono::InstKey;
use crate::sigs::{GenericParam, LockItems, NameCx, Sigs};
use crate::typeck::Typeck;
use crate::types::{ConstArg, GArg, Interner, TypeId, TypeKind};

/// No rank: acquires nothing.
const NONE: u64 = u64::MAX;

/// A part of a value that may hold a lock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    /// A `MutexGuard` of rank `rank` (a value, or a const param of the fn), printed `text`; `ty`
    /// is its type when typeck's table has it (else `NONE`), to print an instance's.
    Guard { rank: ConstArg, text: String, ty: TypeId },
    /// A value of the fn's `T: view` param of this flat index: an instance may make it a guard.
    Param(u32),
}

/// A local that may hold a lock: its parts and its binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Held {
    pub parts: Vec<Part>,
    pub span: Span,
}

/// Where locks are held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum At {
    /// A call, by the token the call graph keys it by ([`CallGraph::calls`]).
    Call(u32),
    /// The join of the threads of the `scope` statement spanning this.
    Join(Span),
    /// A drop of a value of this type (the old value of an assignment too): its drop glue runs.
    /// A local type indexes [`BodyLocks::types`].
    Drop(TypeId),
}

/// The locals that may hold a lock at one call, join, or drop; `span` is that point.
#[derive(Clone, Debug)]
pub struct HeldAt {
    pub at: At,
    pub span: Span,
    pub held: Vec<Held>,
}

/// The locks a fn body (its closures and `spawn` bodies included) may hold: per call, join, and
/// drop with any; `any`, every one it may hold anywhere; `calls`, the token of every call its CFG
/// has; `types`, the local types of the borrow checker's type overlay for the body.
#[derive(Clone, Debug)]
pub struct BodyLocks {
    pub decl: DeclId,
    pub sites: Vec<HeldAt>,
    pub any: Vec<Held>,
    pub calls: Vec<u32>,
    pub types: Vec<TypeKind>,
}

/// Type `t` of an overlay whose local types are `local`, interned into `cx` (a global id is the
/// same in both). Post-order with an explicit stack.
fn import<I: Interner + ?Sized>(cx: &mut I, local: &[TypeKind], t: TypeId) -> Option<TypeId> {
    let mut done: FxHashMap<TypeId, TypeId> = FxHashMap::default();
    let mut stack = vec![(t, false)];
    while let Some((u, ready)) = stack.pop() {
        if !u.is_local() || done.contains_key(&u) {
            continue;
        }
        let k = local.get(u.idx())?;
        if ready {
            let mapped = k.map(|c| if c.is_local() { done.get(&c).copied().unwrap_or(c) } else { c });
            let id = cx.intern(mapped);
            done.insert(u, id);
        } else {
            stack.push((u, true));
            let mut kids = Vec::new();
            k.children(&mut kids);
            stack.extend(kids.into_iter().filter(|c| c.is_local()).map(|c| (c, false)));
        }
    }
    if t.is_local() { done.get(&t).copied() } else { Some(t) }
}

/// The parts of a value of type `t` that may hold a lock: each `MutexGuard` reached through
/// fields, enum payloads, tuple and array elements, and the pointee of an ADT's raw pointer field
/// (not through references, other pointers, slices, or fn values), and each `T: view` param of
/// `generics` reached so. Iterative, each type once.
pub fn lock_parts<I: Interner + ?Sized>(
    cx: &mut I,
    prog: &Program,
    sigs: &Sigs,
    generics: &[GenericParam],
    t: TypeId,
) -> Vec<Part> {
    let guard = sigs.lang.locks.guard;
    let mut out = Vec::new();
    if guard.is_none() || t.is_none() {
        return out;
    }
    let names = NameCx { prog, generics };
    let mut seen: FxHashSet<TypeId> = FxHashSet::default();
    let mut work = vec![t];
    while let Some(t) = work.pop() {
        if t.is_none() || !seen.insert(t) {
            continue;
        }
        match cx.kind(t).clone() {
            TypeKind::Adt { def, args } if def == guard => {
                if let Some(&GArg::Const(rank)) = args.get(1) {
                    let text = crate::types::display(cx, &names, t);
                    out.push(Part::Guard { rank, text, ty: if t.is_local() { TypeId::NONE } else { t } });
                }
            }
            TypeKind::Adt { .. } => {
                if let Some(vs) = crate::mono::adt_fields(sigs, cx, t) {
                    for f in vs.into_iter().flatten() {
                        // an ADT owns what a raw pointer field points to (`vec.Vec`'s buffer)
                        if let TypeKind::Ptr { inner, .. } = cx.kind(f) {
                            work.push(*inner);
                        }
                        work.push(f);
                    }
                }
            }
            TypeKind::Tuple(ts) => work.extend(ts.iter().copied()),
            TypeKind::Array { elem, .. } => work.push(elem),
            TypeKind::Param(i) if generics.get(i as usize).is_some_and(|g| g.view) => {
                out.push(Part::Param(i))
            }
            _ => {}
        }
    }
    out
}

/// The rank an instance acquires as one of the [`LockItems`] acquisitions: its one const arg.
fn own_rank(sigs: &Sigs, l: &LockItems, inst: &InstKey) -> Option<u64> {
    if inst.decl.is_none() || ![l.lock, l.get_or_init, l.lock_pair].contains(&inst.decl) {
        return None;
    }
    let sig = sigs.fn_sig(inst.decl)?;
    let k = sig.generics.iter().position(|g| g.is_const)?;
    match inst.args.get(k)? {
        GArg::Const(ConstArg::Value(v)) => Some(*v),
        _ => None,
    }
}

/// The lowest rank each node may acquire ([`NONE`]: none), by levels of the condensed graph;
/// edges marked in `skip` are not taken.
fn lowest(g: &CallGraph, own: &[u64], skip: &[bool]) -> Vec<u64> {
    let mut low = vec![NONE; g.nodes.len()];
    for l in 0..g.level_count() {
        for s in g.level(l) {
            let members = g.scc(s);
            let mut m = NONE;
            for &v in members {
                m = m.min(own[v as usize]);
                let base = g.out_start[v as usize] as usize;
                for (k, e) in g.succ(v).iter().enumerate() {
                    if g.scc_of[e.to as usize] != s && !skip[base + k] {
                        m = m.min(low[e.to as usize]);
                    }
                }
            }
            for &v in members {
                low[v as usize] = m;
            }
        }
    }
    low
}

/// One held lock after substitution: rank, type text, where it is held.
struct Lock {
    rank: u64,
    text: String,
    span: Span,
}

struct Cx<'a> {
    prog: &'a Program,
    tc: &'a Typeck,
    g: &'a CallGraph,
    own: Vec<u64>,
    low: Vec<u64>,
    skip: Vec<bool>,
    threads: [Option<DeclId>; 3],
    out: Vec<Diagnostic>,
    seen: FxHashSet<(u32, u32, u64, u64)>,
}

impl Cx<'_> {
    fn decl_of(&self, n: u32) -> Option<DeclId> {
        match &self.g.nodes.get(self.g.fn_of(n)? as usize)?.kind {
            NodeKind::Fn { inst, .. } => Some(inst.decl),
            _ => None,
        }
    }

    fn builtin(&self, d: DeclId) -> bool {
        self.prog
            .decl(d)
            .and_then(|x| self.prog.module(x.module))
            .and_then(|m| self.prog.packages.get(m.package.idx()))
            .is_some_and(|p| p.kind == crate::package::PackageKind::Builtin)
    }

    /// The type a source node acquires, printed: `Mutex<T, R>` or `Once<T, R>`.
    fn acquired(&self, n: u32) -> String {
        let l = &self.tc.sigs.lang.locks;
        let NodeKind::Fn { inst, .. } = &self.g.nodes[n as usize].kind else { return String::new() };
        let def = if inst.decl == l.get_or_init { l.once } else { l.mutex };
        let ty = inst.args.iter().find_map(|a| match a {
            GArg::Type(t) => Some(*t),
            _ => None,
        });
        let rank = inst.args.iter().find(|a| matches!(a, GArg::Const(_))).copied();
        let mut cx = self.g.types(&self.tc.types);
        let (Some(ty), Some(rank)) = (ty, rank) else { return String::new() };
        let t = cx.intern(TypeKind::Adt { def, args: Box::new([GArg::Type(ty), rank]) });
        crate::types::display(&cx, &NameCx { prog: self.prog, generics: &[] }, t)
    }

    /// A shortest chain from `from` to a node acquiring rank `rank` or lower on its own: the
    /// edge indices and that node.
    fn chain(&self, from: u32, rank: u64) -> Option<(Vec<u32>, u32)> {
        let g = self.g;
        let mut parent: FxHashMap<u32, Option<u32>> = FxHashMap::default();
        parent.insert(from, None);
        let mut queue = std::collections::VecDeque::from([from]);
        while let Some(m) = queue.pop_front() {
            if self.own[m as usize] <= rank {
                let mut edges = Vec::new();
                let mut cur = m;
                while let Some(&Some(ei)) = parent.get(&cur) {
                    edges.push(ei);
                    cur = g.caller(ei);
                }
                edges.reverse();
                return Some((edges, m));
            }
            let base = g.out_start[m as usize];
            for (k, e) in g.succ(m).iter().enumerate() {
                if self.low[e.to as usize] > rank || self.skip[(base + k as u32) as usize] {
                    continue;
                }
                if let std::collections::hash_map::Entry::Vacant(v) = parent.entry(e.to) {
                    v.insert(Some(base + k as u32));
                    queue.push_back(e.to);
                }
            }
        }
        None
    }

    /// Where a violation in node `p` at `span` is reported: there for a user fn; for a fn of a
    /// built-in package, at the first user call leading to `p` (with that fn's node).
    fn user_site(&self, p: u32, span: Span) -> (u32, Span, Option<Span>) {
        if self.decl_of(p).is_none_or(|d| !self.builtin(d)) {
            return (p, span, None);
        }
        let g = self.g;
        let mut seen: FxHashSet<u32> = FxHashSet::from_iter([p]);
        let mut queue = std::collections::VecDeque::from([p]);
        while let Some(m) = queue.pop_front() {
            for (c, e) in g.pred(m) {
                if e.kind == EdgeKind::Member || !seen.insert(c) {
                    continue;
                }
                match self.decl_of(c) {
                    Some(d) if !self.builtin(d) => return (g.fn_of(c).unwrap_or(c), e.site, Some(span)),
                    _ => queue.push_back(c),
                }
            }
        }
        (p, span, None)
    }

    /// Reports the acquisition by `targets` at `span` of node `p` while `held` is held (E1405);
    /// `join`: the targets are threads joined there (E1406).
    fn report(&mut self, p: u32, span: Span, held: &Lock, targets: &[u32], join: bool, drop: bool) {
        let Some(&t) = targets.iter().min_by_key(|&&t| self.low[t as usize]) else { return };
        let Some((edges, end)) = self.chain(t, held.rank) else { return };
        let rank = self.own[end as usize];
        let (at_node, primary, inner) = self.user_site(p, span);
        if !self.seen.insert((primary.file.0, primary.offset, rank, held.rank)) {
            return;
        }
        let fn_path = self.g.source_names(self.prog, self.tc, &[at_node]).remove(0);
        let lock = self.acquired(end);
        let (rank_s, held_s) = (rank.to_string(), held.rank.to_string());
        let holes = [
            ("fn", fn_path.as_str()),
            ("lock", lock.as_str()),
            ("rank", rank_s.as_str()),
            ("held", held.text.as_str()),
            ("held_rank", held_s.as_str()),
        ];
        // labels at user code only: a chain through std shows its user calls
        let mut roles: Vec<(&str, Span)> = Vec::new();
        if held.span != primary && inner.is_none() {
            roles.push(("held", held.span));
        }
        for &ei in &edges {
            let e = &self.g.out[ei as usize];
            if self.decl_of(self.g.caller(ei)).is_none_or(|d| self.builtin(d)) {
                continue;
            }
            roles.push((if e.kind == EdgeKind::Drop { "drop" } else { "call" }, e.site));
        }
        let code = if join { "E1406" } else { "E1405" };
        let case = (drop && !join).then_some("drop");
        self.out.extend(mk(code, case, &holes, primary, &roles));
    }

    fn resolve(&self, held: &[Held], args: &[GArg], extra: Option<&Lock>) -> Vec<Lock> {
        let mut out = Vec::new();
        let mut cx = None;
        for h in held {
            for p in &h.parts {
                match p {
                    Part::Guard { rank, text, ty } => {
                        let r = match rank {
                            ConstArg::Value(v) => Some(*v),
                            ConstArg::Param(i) => match args.get(*i as usize) {
                                Some(GArg::Const(ConstArg::Value(v))) => Some(*v),
                                _ => None,
                            },
                            ConstArg::Error => None,
                        };
                        let Some(rank) = r else { continue };
                        // an instance prints its own args, not the generic body's params
                        let text = if args.is_empty() || ty.is_none() {
                            text.clone()
                        } else {
                            let cx = cx.get_or_insert_with(|| self.g.types(&self.tc.types));
                            let t = cx.subst(*ty, args, None);
                            crate::types::display(&*cx, &NameCx { prog: self.prog, generics: &[] }, t)
                        };
                        out.push(Lock { rank, text, span: h.span });
                    }
                    Part::Param(i) => {
                        let Some(GArg::Type(t)) = args.get(*i as usize) else { continue };
                        let cx = cx.get_or_insert_with(|| self.g.types(&self.tc.types));
                        for q in lock_parts(cx, self.prog, &self.tc.sigs, &[], *t) {
                            if let Part::Guard { rank: ConstArg::Value(rank), text, .. } = q {
                                out.push(Lock { rank, text, span: h.span });
                            }
                        }
                    }
                }
            }
        }
        if let Some(e) = extra {
            out.push(Lock { rank: e.rank, text: e.text.clone(), span: e.span });
        }
        out
    }

    fn is_thread_call(&self, targets: &[u32]) -> bool {
        targets.iter().any(|&t| {
            matches!(&self.g.nodes[t as usize].kind, NodeKind::Fn { inst, .. }
                if self.threads.contains(&Some(inst.decl)))
        })
    }

    /// The span of token `tok` of fn node `p`'s file.
    fn tok_span(&self, p: u32, tok: u32) -> Span {
        let file = self.g.nodes[p as usize].span.file;
        match self.prog.file(file) {
            Some(f) => crate::scan::tok_span(file, &f.bytes, &f.parsed.tokens, tok as usize),
            None => self.g.nodes[p as usize].span,
        }
    }

    /// The `Once` type a `get_or_init` node waits for, printed.
    fn once_text(&self, p: u32) -> String {
        self.acquired(p)
    }
}

/// The types a fn body (its closures included) may drop, from the borrow checker: the values
/// that may be initialized where a drop runs. `None`: not known (keep every drop edge).
#[derive(Clone, Debug)]
pub struct BodyDrops {
    pub decl: DeclId,
    pub types: Option<Vec<TypeId>>,
}

/// Marks the drop edges of fn and closure nodes to the glue of types their body never drops
/// (the graph's drop edges cover every type a body makes or owns, a returned one too). A body's
/// edges are narrowed only when every type it drops is the glue target of one of its own drop
/// edges; a dropped type without glue, or one reached only inside another glue (a part dropped
/// after a partial move), keeps them all.
fn unused_drops(
    g: &CallGraph,
    cx: &mut crate::types::Overlay<'_>,
    glue: &FxHashMap<TypeId, u32>,
    drops: &[BodyDrops],
) -> Vec<bool> {
    let mut skip = vec![false; g.out.len()];
    let by_decl: FxHashMap<DeclId, &Option<Vec<TypeId>>> = drops.iter().map(|d| (d.decl, &d.types)).collect();
    // per fn node: the drop edges of it and its closures, (edge index, target)
    let mut edges: FxHashMap<u32, Vec<(usize, u32)>> = FxHashMap::default();
    for v in 0..g.nodes.len() as u32 {
        let Some(p) = g.fn_of(v) else { continue };
        let base = g.out_start[v as usize] as usize;
        for (k, e) in g.succ(v).iter().enumerate() {
            if e.kind == EdgeKind::Drop {
                edges.entry(p).or_default().push((base + k, e.to));
            }
        }
    }
    for (p, list) in edges {
        let NodeKind::Fn { inst, .. } = &g.nodes[p as usize].kind else { continue };
        let Some(Some(types)) = by_decl.get(&inst.decl) else { continue };
        let targets: FxHashSet<u32> = list.iter().map(|x| x.1).collect();
        let mut used: FxHashSet<u32> = FxHashSet::default();
        let all_mapped = types.iter().all(|&t| match glue.get(&cx.subst(t, &inst.args, None)) {
            Some(&n) if targets.contains(&n) => {
                used.insert(n);
                true
            }
            _ => false,
        });
        if !all_mapped {
            continue;
        }
        for (ei, to) in list {
            if !used.contains(&to) {
                skip[ei] = true;
            }
        }
    }
    skip
}

/// The E1405 / E1406 diagnostics of a program without errors (`bodies`: the borrow checker's
/// held locks; `drops`: what each body may drop).
pub fn check(prog: &Program, tc: &Typeck, bodies: &[BodyLocks], drops: &[BodyDrops]) -> Vec<Diagnostic> {
    let g = &tc.call_graph;
    let sigs = &tc.sigs;
    let l = sigs.lang.locks;
    let own: Vec<u64> = g
        .nodes
        .iter()
        .map(|n| match &n.kind {
            NodeKind::Fn { inst, .. } => own_rank(sigs, &l, inst).unwrap_or(NONE),
            _ => NONE,
        })
        .collect();
    if own.iter().all(|&r| r == NONE) {
        return Vec::new();
    }
    // the dropped types of the bodies are interned here, over the graph's own types
    let mut tycx = g.types(&tc.types);
    let glue: FxHashMap<TypeId, u32> = g
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| match n.kind {
            NodeKind::Glue { ty } => Some((ty, i as u32)),
            _ => None,
        })
        .collect();
    let skip = unused_drops(g, &mut tycx, &glue, drops);
    let low = lowest(g, &own, &skip);
    let threads = [stave_ir::Known::ThreadJoin, stave_ir::Known::ParFor, stave_ir::Known::ParForMut]
        .map(|k| prog.known.decl(k));
    let mut cx = Cx { prog, tc, g, own, low, skip, threads, out: Vec::new(), seen: FxHashSet::default() };
    let by_decl: FxHashMap<DeclId, &BodyLocks> = bodies.iter().map(|b| (b.decl, b)).collect();
    // the calls of each fn node to check (its closures' calls under it): (token, callees)
    let mut calls: FxHashMap<u32, Vec<(u32, Vec<u32>)>> = FxHashMap::default();
    for &(from, tok, to) in &g.calls {
        let Some(p) = g.fn_of(from) else { continue };
        let wanted = cx.decl_of(p).is_some_and(|d| by_decl.contains_key(&d) || d == l.get_or_init);
        if !wanted {
            continue;
        }
        let list = calls.entry(p).or_default();
        match list.iter_mut().find(|(t, _)| *t == tok) {
            Some((_, ts)) => ts.push(to),
            None => list.push((tok, vec![to])),
        }
    }
    // fn nodes with a body's held locks but no call still have joins
    for (i, n) in g.nodes.iter().enumerate() {
        if let NodeKind::Fn { inst, body: true, .. } = &n.kind
            && by_decl.contains_key(&inst.decl)
        {
            calls.entry(i as u32).or_default();
        }
    }
    let mut nodes: Vec<u32> = calls.keys().copied().collect();
    nodes.sort_unstable();
    for p in nodes {
        let NodeKind::Fn { inst, .. } = &g.nodes[p as usize].kind else { continue };
        let (decl, args) = (inst.decl, inst.args.clone());
        let mut list = calls.remove(&p).unwrap_or_default();
        list.sort_unstable();
        let body = by_decl.get(&decl).copied();
        let site_of: FxHashMap<u32, &HeldAt> = body
            .map(|b| {
                b.sites
                    .iter()
                    .filter_map(|s| match s.at {
                        At::Call(t) => Some((t, s)),
                        At::Join(_) | At::Drop(_) => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let known: FxHashSet<u32> = body.map(|b| b.calls.iter().copied().collect()).unwrap_or_default();
        let any = body.map(|b| cx.resolve(&b.any, &args, None)).unwrap_or_default();
        // `get_or_init` runs `f` holding the Once's rank
        let once = (decl == l.get_or_init).then(|| Lock {
            rank: cx.own[p as usize],
            text: cx.once_text(p),
            span: g.nodes[p as usize].span,
        });
        for (tok, targets) in &list {
            let low = targets.iter().map(|&t| cx.low[t as usize]).min().unwrap_or(NONE);
            if low == NONE {
                continue;
            }
            let (held, span) = match (site_of.get(tok), &once) {
                (_, Some(o)) => (cx.resolve(&[], &args, Some(o)), cx.tok_span(p, *tok)),
                (Some(s), None) => (cx.resolve(&s.held, &args, None), s.span),
                (None, None) if known.contains(tok) => continue,
                (None, None) => {
                    let v = any
                        .iter()
                        .map(|h| Lock { rank: h.rank, text: h.text.clone(), span: h.span })
                        .collect();
                    (v, cx.tok_span(p, *tok))
                }
            };
            let Some(top) = held.into_iter().max_by_key(|h| h.rank) else { continue };
            if low > top.rank {
                continue;
            }
            let join = cx.is_thread_call(targets);
            cx.report(p, span, &top, targets, join, false);
        }
        // the joins of the node's `scope` statements: the spawn bodies inside each statement
        let Some(b) = body else { continue };
        for s in &b.sites {
            let At::Join(scope) = s.at else { continue };
            let held = cx.resolve(&s.held, &args, None);
            let Some(top) = held.into_iter().max_by_key(|h| h.rank) else { continue };
            let kids: Vec<u32> = (0..g.nodes.len() as u32)
                .filter(|&c| match g.nodes[c as usize].kind {
                    NodeKind::Closure { parent, spawn: true, .. } => {
                        let sp = g.nodes[c as usize].span;
                        parent == p
                            && sp.file == scope.file
                            && sp.offset >= scope.offset
                            && sp.offset < scope.offset + scope.len
                    }
                    _ => false,
                })
                .collect();
            let low = kids.iter().map(|&c| cx.low[c as usize]).min().unwrap_or(NONE);
            if low <= top.rank {
                cx.report(p, s.span, &top, &kids, true, false);
            }
        }
        // the drops: the dropped type's glue, else (a type the graph has no glue for) every
        // drop the node and its closures make
        for s in &b.sites {
            let At::Drop(t) = s.at else { continue };
            let held = cx.resolve(&s.held, &args, None);
            let Some(top) = held.into_iter().max_by_key(|h| h.rank) else { continue };
            let ty = import(&mut tycx, &b.types, t).map(|t| tycx.subst(t, &args, None));
            let targets: Vec<u32> = match ty.and_then(|t| glue.get(&t)) {
                Some(&n) => vec![n],
                None => (0..g.nodes.len() as u32)
                    .filter(|&c| c == p || matches!(g.nodes[c as usize].kind, NodeKind::Closure { parent, .. } if parent == p))
                    .flat_map(|c| g.succ(c).iter().filter(|e| e.kind == EdgeKind::Drop).map(|e| e.to))
                    .collect(),
            };
            let low = targets.iter().map(|&n| cx.low[n as usize]).min().unwrap_or(NONE);
            if low <= top.rank {
                cx.report(p, s.span, &top, &targets, false, true);
            }
        }
    }
    cx.out
}
