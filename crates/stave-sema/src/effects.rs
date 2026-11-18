//! The effect pass: one effect set per node of the call graph ([`crate::callgraph`]), the deny
//! clause `without(..)` checked against it (E1401), and fn values made where a fn type with a
//! deny clause is wanted (E1403 case `made`).
//!
//! **Effects** ([`Effects`]): `alloc trap recursive lock unsafe io block`. A node's own effects
//! ([`EffectSets::local`]) come from:
//!
//! | Source | Effects |
//! |---|---|
//! | the runtime fns of [`BOUNDARY`] (allocator, panic, lock word, futex, thread start and join) | their row; nothing they call counts |
//! | an `extern "C"` import | all except those its `without` lists (a claim of the block's obligation) |
//! | intrinsic `syscall0`..`syscall6`, `thread_clone` | `io` |
//! | intrinsic `call_on_stack`, `cpu_init` (they run code no edge shows) | all |
//! | a trapping instruction of the checked IR after the removal of proven bounds and contract checks ([`NodeIr`]; a kept `requires` / `ensures` check is its failure call) | `trap` |
//! | `panic.out_of_memory` in that IR | `alloc` |
//! | membership in a call cycle (an SCC that is a cycle) | `recursive`, `trap` (stack overflow) |
//! | `unsafe("..")` block or expression, outside the built-in packages | `unsafe` |
//! | a `scope { }` (it waits for its threads at the end) | `block` |
//! | a `spawn { }` (it maps a stack and starts a thread) | `alloc io trap` |
//!
//! A node's set ([`EffectSets::set`]) is its own effects and the sets of everything it calls or
//! drops (a drop edge to a glue node, which has no effects of its own, then to the `Drop` impls),
//! except through a boundary fn. An edge carries the deny clause of a fn-value call's type
//! ([`crate::callgraph::Edge::deny`]): those effects do not flow through it, because every value of
//! that type is checked to lack them (the coercion rule of typeck plus E1403 here).
//!
//! **Shape.** One parallel map over nodes for the local bits, then one parallel step per level of
//! the condensed graph (callees first): an SCC's set is the union of its members' own effects and
//! the sets of the SCCs they call. Members of one SCC share the set. No per-fn solver.
//!
//! **Checks.** A fn instance whose decl has a deny clause, and every fn value made where a fn type
//! denies effects its own type does not ([`crate::typeck::TypedBody::denied`]), are checked on
//! their sets. The diagnostic names the shortest call chain (breadth-first over edges whose
//! target has the effect) to the node that has it on its own, and that node's op.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind as Syn, Term};

use crate::Program;
use crate::callgraph::{CallGraph, NodeKind};
use crate::closures::ClosureKind;
use crate::decls::DeclFlags;
use crate::diag::mk;
use crate::ids::{DeclId, ItemId};
use crate::itemcx::{Brackets, ItemCx};
use crate::mono::InstKey;
use crate::sigs::Abi;
use crate::typeck::{Target, Typeck, TypedBody};
use crate::types::GArg;

/// A set of effects, one bit each in [`Effects::NAMES`] order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Effects(pub u8);

impl Effects {
    pub const NONE: Effects = Effects(0);
    pub const ALLOC: Effects = Effects(1);
    pub const TRAP: Effects = Effects(1 << 1);
    pub const RECURSIVE: Effects = Effects(1 << 2);
    pub const LOCK: Effects = Effects(1 << 3);
    pub const UNSAFE: Effects = Effects(1 << 4);
    pub const IO: Effects = Effects(1 << 5);
    pub const BLOCK: Effects = Effects(1 << 6);
    pub const ALL: Effects = Effects(0x7f);
    /// The effect names, by bit.
    pub const NAMES: [&'static str; 7] = ["alloc", "trap", "recursive", "lock", "unsafe", "io", "block"];

    pub fn of_name(s: &str) -> Option<Effects> {
        Self::NAMES.iter().position(|&n| n == s).map(|i| Effects(1 << i))
    }

    /// The names of the effects in the set, in [`Effects::NAMES`] order.
    pub fn names(self) -> Vec<&'static str> {
        (0..7).filter(|i| self.0 & (1 << i) != 0).map(|i| Self::NAMES[i]).collect()
    }

    pub fn contains(self, o: Effects) -> bool {
        self.0 & o.0 == o.0
    }

    pub fn meets(self, o: Effects) -> bool {
        self.0 & o.0 != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The single effects of the set, in [`Effects::NAMES`] order.
    pub fn each(self) -> impl Iterator<Item = Effects> {
        (0..7).map(|i| Effects(1 << i)).filter(move |e| self.meets(*e))
    }
}

impl std::ops::BitOr for Effects {
    type Output = Effects;
    fn bitor(self, o: Effects) -> Effects {
        Effects(self.0 | o.0)
    }
}

impl std::ops::BitOrAssign for Effects {
    fn bitor_assign(&mut self, o: Effects) {
        self.0 |= o.0;
    }
}

/// What gives a node an effect on its own (the diagnostics' `op`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    Alloc,
    Free,
    OutOfMemory,
    Trap,
    Cycle,
    Lock,
    Unsafe,
    Syscall,
    Join,
    Spawn,
    Extern,
    Unseen,
}

impl OpKind {
    /// The text of the `op` hole of E1401 / E1403.
    pub fn text(self) -> &'static str {
        match self {
            OpKind::Alloc => "allocates",
            OpKind::Free => "frees memory",
            OpKind::OutOfMemory => "may run out of memory",
            OpKind::Trap => "may trap",
            OpKind::Cycle => "is recursive",
            OpKind::Lock => "takes a lock",
            OpKind::Unsafe => "has an `unsafe` block",
            OpKind::Syscall => "makes a system call",
            OpKind::Join => "waits for other threads",
            OpKind::Spawn => "starts a thread",
            OpKind::Extern => "is an `extern` fn, which has every effect its `without` does not list",
            OpKind::Unseen => "runs code the call graph does not show",
        }
    }
}

/// An op that gives a node an effect: what it is and where (the node's name for a whole-fn
/// source).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub kind: OpKind,
    pub span: Span,
}

/// The runtime fns whose effects are fixed: (module, fn, effects, op). Their bodies implement
/// the effect (the allocator's lock word and system calls are its allocation; a panic's write and
/// exit are its trap), so what they call does not count.
pub const BOUNDARY: &[(&str, &str, Effects, OpKind)] = &[
    ("runtime.alloc", "alloc", Effects(Effects::ALLOC.0 | Effects::TRAP.0), OpKind::Alloc),
    ("runtime.alloc", "try_alloc", Effects::ALLOC, OpKind::Alloc),
    ("runtime.alloc", "free", Effects::ALLOC, OpKind::Free),
    ("runtime.alloc", "realloc", Effects(Effects::ALLOC.0 | Effects::TRAP.0), OpKind::Alloc),
    ("runtime.alloc", "try_realloc", Effects::ALLOC, OpKind::Alloc),
    ("runtime.panic", "trap", Effects::TRAP, OpKind::Trap),
    ("runtime.panic", "trap_msg", Effects::TRAP, OpKind::Trap),
    ("runtime.panic", "abort", Effects::TRAP, OpKind::Trap),
    ("runtime.panic", "requires_failed", Effects::TRAP, OpKind::Trap),
    ("runtime.panic", "ensures_failed", Effects::TRAP, OpKind::Trap),
    ("runtime.atomic", "lock", Effects::LOCK, OpKind::Lock),
    ("runtime.atomic", "try_lock", Effects::NONE, OpKind::Lock),
    ("runtime.atomic", "unlock", Effects::NONE, OpKind::Lock),
    ("runtime.os", "futex", Effects::LOCK, OpKind::Lock),
    ("runtime.thread", "start", Effects(Effects::ALLOC.0 | Effects::IO.0 | Effects::TRAP.0), OpKind::Spawn),
    (
        "runtime.thread",
        "scope_spawn",
        Effects(Effects::ALLOC.0 | Effects::IO.0 | Effects::TRAP.0),
        OpKind::Spawn,
    ),
    ("runtime.thread", "join", Effects::BLOCK, OpKind::Join),
    ("runtime.thread", "scope_join", Effects::BLOCK, OpKind::Join),
];

/// The effects of a `spawn { }` statement for the fn holding it (`runtime.thread.scope_spawn`).
const SPAWN: Effects = Effects(Effects::ALLOC.0 | Effects::IO.0 | Effects::TRAP.0);

/// What the checked IR says about one node (its function after the removal of proven bounds
/// checks): the first trapping instruction and the first `panic.out_of_memory`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeIr {
    pub trap: Option<Span>,
    pub out_of_memory: Option<Span>,
}

/// A node checked against effects it must not have.
#[derive(Clone, Debug)]
pub struct Check {
    pub node: u32,
    pub deny: Effects,
    pub why: Why,
}

#[derive(Clone, Debug)]
pub enum Why {
    /// The deny clause of the node's fn.
    Clause,
    /// A fn value made at `site` where a fn type denying `deny` is wanted.
    Made { site: Span },
}

/// The checks of a program and whether they need the checked IR (a `trap` among the denied
/// effects).
#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub checks: Vec<Check>,
}

impl Plan {
    /// Whether some check denies `trap`: then the IR of what it reaches is needed.
    pub fn needs_ir(&self) -> bool {
        self.checks.iter().any(|c| c.deny.meets(Effects::TRAP))
    }

    /// The nodes whose checked IR the `trap` bit needs: everything reachable from a check that
    /// denies `trap` (all nodes with `all`), fns and closures only.
    pub fn ir_nodes(&self, g: &CallGraph, all: bool) -> Vec<u32> {
        let n = g.nodes.len();
        let mut seen = vec![all; n];
        if !all {
            let mut stack: Vec<u32> =
                self.checks.iter().filter(|c| c.deny.meets(Effects::TRAP)).map(|c| c.node).collect();
            for &s in &stack {
                seen[s as usize] = true;
            }
            while let Some(m) = stack.pop() {
                for e in g.succ(m) {
                    if !seen[e.to as usize] {
                        seen[e.to as usize] = true;
                        stack.push(e.to);
                    }
                }
            }
        }
        (0..n as u32)
            .filter(|&i| {
                seen[i as usize]
                    && !matches!(g.nodes[i as usize].kind, NodeKind::Table { .. } | NodeKind::Glue { .. })
            })
            .collect()
    }
}

/// The fn nodes whose bodies have a contract check (a checked `requires` or an `ensures`): their
/// checked IR decides which checks stay, so their `trap` and the contract diagnostics need it.
pub fn contract_nodes(tc: &Typeck) -> Vec<u32> {
    let g = &tc.call_graph;
    let checked = |d: DeclId| {
        let Ok(bi) = tc.bodies.binary_search_by_key(&d, |b| b.decl) else { return false };
        let b = &tc.bodies[bi];
        !b.has_error
            && b.contracts
                .as_deref()
                .is_some_and(|c| !c.checked_requires().is_empty() || !c.ensures.is_empty())
    };
    (0..g.nodes.len() as u32)
        .filter(|&i| matches!(&g.nodes[i as usize].kind, NodeKind::Fn { inst, body: true, .. } if checked(inst.decl)))
        .collect()
}

/// Each node's own effects and its effect set ([`solve`]).
#[derive(Clone, Debug, Default)]
pub struct EffectSets {
    pub local: Vec<Effects>,
    pub set: Vec<Effects>,
    /// Boundary nodes: their set is their own effects.
    pub boundary: Vec<bool>,
    /// The nodes the program uses ([`used_nodes`]); a fn's effects are the union over its used
    /// nodes.
    pub used: Vec<bool>,
    /// The first op of each own effect: (node, effect bit index, op), sorted.
    ops: Vec<(u32, u8, Op)>,
    /// Whether `trap` includes the checked IR's trapping instructions (else only call cycles
    /// and the runtime's panic fns).
    pub trap_from_ir: bool,
}

impl EffectSets {
    /// The op that gives node `n` effect `e` (a single effect) on its own.
    pub fn op(&self, n: u32, e: Effects) -> Option<Op> {
        let bit = e.0.trailing_zeros() as u8;
        self.ops.binary_search_by_key(&(n, bit), |x| (x.0, x.1)).ok().map(|i| self.ops[i].2)
    }

    /// The shortest call chain from `from` to a node that has effect `e` on its own: the edge
    /// indices (into [`CallGraph::out`]) in call order, and that node. Edges whose deny clause
    /// names `e` are not taken unless no other chain exists.
    pub fn chain(&self, g: &CallGraph, from: u32, e: Effects) -> Option<(Vec<u32>, u32)> {
        if !self.set.get(from as usize)?.meets(e) {
            return None;
        }
        for strict in [true, false] {
            let mut parent: FxHashMap<u32, Option<u32>> = FxHashMap::default();
            parent.insert(from, None);
            let mut queue = std::collections::VecDeque::from([from]);
            while let Some(m) = queue.pop_front() {
                if self.local[m as usize].meets(e) {
                    let mut edges = Vec::new();
                    let mut cur = m;
                    while let Some(&Some(ei)) = parent.get(&cur) {
                        edges.push(ei);
                        cur = g.caller(ei);
                    }
                    edges.reverse();
                    return Some((edges, m));
                }
                if self.boundary[m as usize] {
                    continue;
                }
                let base = g.out_start[m as usize];
                for (k, edge) in g.succ(m).iter().enumerate() {
                    if (strict && Effects(edge.deny).meets(e)) || !self.set[edge.to as usize].meets(e) {
                        continue;
                    }
                    if let std::collections::hash_map::Entry::Vacant(v) = parent.entry(edge.to) {
                        v.insert(Some(base + k as u32));
                        queue.push_back(edge.to);
                    }
                }
            }
        }
        None
    }
}

/// The package of decl `d` is a built-in one (`std`, `runtime`, the runtime's arch packages).
fn builtin(prog: &Program, d: DeclId) -> bool {
    prog.decl(d)
        .and_then(|x| prog.module(x.module))
        .and_then(|m| prog.packages.get(m.package.idx()))
        .is_some_and(|p| p.kind == crate::package::PackageKind::Builtin)
}

fn item_cx<'p>(prog: &'p Program, item: ItemId, br: &'p mut Option<Brackets>) -> Option<ItemCx<'p>> {
    let it = prog.item(item)?;
    let f = prog.file(it.file)?;
    let info = f.parsed.items.get(it.index as usize)?;
    if !it.ok {
        return None;
    }
    *br = Some(Brackets::new(&f.parsed.tokens, info.start, info.end));
    ItemCx::new(prog, item, br.as_ref()?)
}

/// Own effects of a body from its source: per owner (0 the fn, `k + 1` closure `k`), (effect,
/// op).
type BodyOps = Vec<(u32, Effects, Op)>;

/// The sources in body `b` itself: `unsafe` blocks and expressions, `scope`, `spawn`, and calls
/// of intrinsics (not call-graph edges). Nothing is parsed again unless the item has one of
/// those tokens or an intrinsic call.
fn body_ops(prog: &Program, b: &TypedBody) -> BodyOps {
    let mut out = BodyOps::new();
    let Some(it) = prog.item(b.item) else { return out };
    let Some(f) = prog.file(it.file) else { return out };
    let Some(info) = f.parsed.items.get(it.index as usize) else { return out };
    let toks = f.parsed.tokens.get(info.start as usize..info.end as usize).unwrap_or(&[]);
    let wanted = |k: u8| k == Term::KwUnsafe as u8 || k == Term::KwScope as u8 || k == Term::KwSpawn as u8;
    let intrinsics: Vec<(u32, Effects, OpKind)> = b
        .targets
        .iter()
        .filter_map(|(tok, t)| match t {
            Target::Intrinsic { decl } => {
                intrinsic(prog, *decl).filter(|x| !x.0.is_empty()).map(|(e, k)| (*tok, e, k))
            }
            _ => None,
        })
        .collect();
    if intrinsics.is_empty() && !toks.iter().any(|t| wanted(t.kind)) {
        return out;
    }
    let mut br = None;
    let Some(icx) = item_cx(prog, b.item, &mut br) else { return out };
    let Some(block) = icx.child(b.node, Syn::Block) else { return out };
    let open = icx.a(block);
    let close = icx.close(open);
    let closures: Vec<(u32, u32, ClosureKind)> = crate::closures::closure_nodes(&icx, b.node)
        .into_iter()
        .map(|(node, kind)| {
            let lo = icx.a(node);
            let hi = icx.child(node, Syn::Block).map_or_else(|| icx.end_tok(node), |c| icx.close(icx.a(c)));
            (lo, hi, kind)
        })
        .collect();
    // the innermost closure holding token `t` (`k + 1`), else 0
    let owner = |t: u32| -> u32 {
        closures
            .iter()
            .enumerate()
            .filter(|(_, c)| c.0 <= t && t <= c.1)
            .max_by_key(|(_, c)| c.0)
            .map_or(0, |(k, _)| k as u32 + 1)
    };
    let user = !builtin(prog, b.decl);
    for t in open..=close {
        if icx.is(t, Term::KwUnsafe) && icx.is(t + 1, Term::LParen) && user {
            out.push((owner(t), Effects::UNSAFE, Op { kind: OpKind::Unsafe, span: icx.tok_span(t) }));
        } else if icx.is(t, Term::KwScope) {
            out.push((owner(t), Effects::BLOCK, Op { kind: OpKind::Join, span: icx.tok_span(t) }));
        }
    }
    for &(lo, _, kind) in &closures {
        if kind == ClosureKind::Spawn {
            out.push((
                owner(lo.saturating_sub(1)),
                SPAWN,
                Op { kind: OpKind::Spawn, span: icx.tok_span(lo) },
            ));
        }
    }
    for (tok, e, kind) in intrinsics {
        // keyed by the call's `(`: the callee's name is the token before it
        out.push((owner(tok), e, Op { kind, span: icx.tok_span(tok.saturating_sub(1)) }));
    }
    out
}

/// The effects of a call of intrinsic `d`: a system call is `io`; `call_on_stack` and `cpu_init`
/// run code no call-graph edge shows, so they have every effect.
fn intrinsic(prog: &Program, d: DeclId) -> Option<(Effects, OpKind)> {
    let name = prog.strings.get(prog.decl(d)?.name);
    Some(match name {
        "thread_clone" => (Effects::IO, OpKind::Syscall),
        n if n.starts_with("syscall") => (Effects::IO, OpKind::Syscall),
        "call_on_stack" | "cpu_init" => (Effects::ALL, OpKind::Unseen),
        _ => (Effects::NONE, OpKind::Unseen),
    })
}

/// The fixed effects of fn decl `d` without a body of its own to look into: a boundary fn, an
/// intrinsic, or an `extern "C"` import.
fn fixed(
    prog: &Program,
    tc: &Typeck,
    boundary: &FxHashMap<DeclId, (Effects, OpKind)>,
    d: DeclId,
) -> Option<(Effects, OpKind)> {
    if let Some(&b) = boundary.get(&d) {
        return Some(b);
    }
    let decl = prog.decl(d)?;
    if !decl.flags.has(DeclFlags::EXTERN_IMPORT) {
        return None;
    }
    let sig = tc.sigs.fn_sig(d)?;
    if sig.abi != Abi::Intrinsic {
        // the `without` list is a claim covered by the block's obligation, never checked
        return Some((Effects(Effects::ALL.0 & !sig.deny), OpKind::Extern));
    }
    intrinsic(prog, d)
}

/// Each node's own effects and effect set. `ir`: per node, what its checked IR says (`None`: the
/// IR was not built; `trap` then comes only from call cycles and the runtime's panic fns).
pub fn solve(prog: &Program, tc: &Typeck, ir: Option<&[NodeIr]>) -> EffectSets {
    let g = &tc.call_graph;
    let n = g.nodes.len();
    let boundary: FxHashMap<DeclId, (Effects, OpKind)> = BOUNDARY
        .iter()
        .filter_map(|&(m, f, e, k)| crate::mono::item_at(prog, m, f).map(|d| (d, (e, k))))
        .collect();
    // source ops per body, once per decl (instances share them)
    let body_ops: Vec<BodyOps> =
        tc.bodies.par_iter().map(|b| if b.has_error { Vec::new() } else { body_ops(prog, b) }).collect();
    let body_of = |d: DeclId| tc.bodies.binary_search_by_key(&d, |b| b.decl).ok();
    let cyclic: Vec<bool> = (0..g.scc_count() as u32).into_par_iter().map(|s| g.is_cycle(s)).collect();
    // per node: own effects, whether it is a boundary fn, the op of each own effect
    type Own = (Effects, bool, Vec<(u8, Op)>);
    let per_node: Vec<Own> = (0..n as u32)
        .into_par_iter()
        .map(|i| {
            let node = &g.nodes[i as usize];
            let mut eff = Effects::NONE;
            let mut ops: Vec<(u8, Op)> = Vec::new();
            let mut add = |e: Effects, op: Op, eff: &mut Effects| {
                for one in e.each() {
                    if !eff.meets(one) {
                        ops.push((one.0.trailing_zeros() as u8, op));
                    }
                }
                *eff |= e;
            };
            let (decl, owner) = match &node.kind {
                NodeKind::Fn { inst, .. } => (inst.decl, 0),
                NodeKind::Closure { parent, index, .. } => match &g.nodes[*parent as usize].kind {
                    NodeKind::Fn { inst, .. } => (inst.decl, index + 1),
                    _ => return (eff, false, ops),
                },
                NodeKind::Table { .. } | NodeKind::Glue { .. } => return (eff, false, ops),
            };
            if owner == 0
                && let Some((e, kind)) = fixed(prog, tc, &boundary, decl)
            {
                add(e, Op { kind, span: node.span }, &mut eff);
                return (eff, true, ops);
            }
            if let Some(bi) = body_of(decl) {
                for &(o, e, op) in &body_ops[bi] {
                    if o == owner {
                        add(e, op, &mut eff);
                    }
                }
            }
            if let Some(x) = ir.and_then(|r| r.get(i as usize)) {
                if let Some(s) = x.trap {
                    add(Effects::TRAP, Op { kind: OpKind::Trap, span: s }, &mut eff);
                }
                if let Some(s) = x.out_of_memory {
                    add(Effects::ALLOC, Op { kind: OpKind::OutOfMemory, span: s }, &mut eff);
                }
            }
            if cyclic[g.scc_of[i as usize] as usize] {
                add(
                    Effects::RECURSIVE | Effects::TRAP,
                    Op { kind: OpKind::Cycle, span: node.span },
                    &mut eff,
                );
            }
            (eff, false, ops)
        })
        .collect();
    let mut local = Vec::with_capacity(n);
    let mut boundary_node = Vec::with_capacity(n);
    let mut ops: Vec<(u32, u8, Op)> = Vec::new();
    for (i, (e, b, o)) in per_node.into_iter().enumerate() {
        local.push(e);
        boundary_node.push(b);
        ops.extend(o.into_iter().map(|(bit, op)| (i as u32, bit, op)));
    }
    ops.sort_by_key(|x| (x.0, x.1));
    // levels, callees first: an SCC's set from its members and the SCCs below
    let mut scc_set = vec![Effects::NONE; g.scc_count()];
    for l in 0..g.level_count() {
        let r = g.level(l);
        let vals: Vec<Effects> = r
            .clone()
            .into_par_iter()
            .map(|s| {
                let mut acc = Effects::NONE;
                for &m in g.scc(s) {
                    acc |= local[m as usize];
                    if boundary_node[m as usize] {
                        continue;
                    }
                    for e in g.succ(m) {
                        let ts = g.scc_of[e.to as usize];
                        if ts == s {
                            continue;
                        }
                        let t = if boundary_node[e.to as usize] {
                            local[e.to as usize]
                        } else {
                            scc_set[ts as usize]
                        };
                        acc |= Effects(t.0 & !e.deny);
                    }
                }
                acc
            })
            .collect();
        for (s, v) in r.zip(vals) {
            scc_set[s as usize] = v;
        }
    }
    let set: Vec<Effects> =
        (0..n).map(|i| if boundary_node[i] { local[i] } else { scc_set[g.scc_of[i] as usize] }).collect();
    EffectSets { local, set, boundary: boundary_node, used: used_nodes(g), ops, trap_from_ir: ir.is_some() }
}

/// Is `n` a fn node specialized on no fn value (the node calls without known values share)?
fn plain(g: &CallGraph, n: u32) -> bool {
    matches!(&g.nodes[n as usize].kind, NodeKind::Fn { spec, .. } if spec.iter().all(|s| s.is_empty()))
}

/// The nodes the program uses. The call graph gives a fn one node per set of known fn values
/// it is called with, plus a plain node every fn starts with; a plain node that no call reaches
/// stands for nothing that runs. Used: what an edge reaches from a used node, the closures of a
/// used node, the roots, and, for a fn none of whose nodes is otherwise used, its plain node
/// (first for fns no call reaches at all, then for those reached only from unused nodes, in node
/// order). Every fn and closure therefore has a used node unless its enclosing fn's are all
/// unused.
pub fn used_nodes(g: &CallGraph) -> Vec<bool> {
    let n = g.nodes.len();
    let mut used = vec![false; n];
    let mut closures: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut group_of: FxHashMap<&InstKey, usize> = FxHashMap::default();
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for (i, nd) in g.nodes.iter().enumerate() {
        match &nd.kind {
            NodeKind::Closure { parent, .. } => closures[*parent as usize].push(i as u32),
            NodeKind::Fn { inst, .. } => {
                let k = *group_of.entry(inst).or_insert_with(|| {
                    groups.push(Vec::new());
                    groups.len() - 1
                });
                groups[k].push(i as u32);
            }
            _ => {}
        }
    }
    let mut stack: Vec<u32> = g.roots.iter().map(|r| r.node).collect();
    let visit = |stack: &mut Vec<u32>, used: &mut Vec<bool>| {
        while let Some(m) = stack.pop() {
            if std::mem::replace(&mut used[m as usize], true) {
                continue;
            }
            stack.extend(g.succ(m).iter().map(|e| e.to).filter(|&t| !used[t as usize]));
            stack.extend(closures[m as usize].iter().copied().filter(|&c| !used[c as usize]));
        }
    };
    visit(&mut stack, &mut used);
    let entry = |nodes: &[u32]| -> u32 { nodes.iter().copied().find(|&m| plain(g, m)).unwrap_or(nodes[0]) };
    for nodes in &groups {
        let called =
            nodes.iter().any(|&m| g.pred(m).any(|(c, _)| !g.fn_of(c).is_some_and(|f| nodes.contains(&f))));
        if !called && !nodes.iter().any(|&m| used[m as usize]) {
            stack.push(entry(nodes));
        }
    }
    visit(&mut stack, &mut used);
    for nodes in &groups {
        if !nodes.iter().any(|&m| used[m as usize]) {
            stack.push(entry(nodes));
            visit(&mut stack, &mut used);
        }
    }
    used
}

/// The program's fns and closures in source terms: per group of nodes of one instance (or of one
/// closure of it), the used nodes and the union of their effect sets, in first-node order.
/// Groups with no used node (closures of fns that never run) are left out.
pub fn by_source(g: &CallGraph, sets: &EffectSets) -> Vec<(Vec<u32>, Effects)> {
    let mut group_of: FxHashMap<(&InstKey, u32), usize> = FxHashMap::default();
    let mut out: Vec<(Vec<u32>, Effects)> = Vec::new();
    for (i, nd) in g.nodes.iter().enumerate() {
        let key = match &nd.kind {
            NodeKind::Fn { inst, .. } => (inst, u32::MAX),
            NodeKind::Closure { parent, index, .. } => match &g.nodes[*parent as usize].kind {
                NodeKind::Fn { inst, .. } => (inst, *index),
                _ => continue,
            },
            _ => continue,
        };
        if !sets.used[i] {
            continue;
        }
        let k = *group_of.entry(key).or_insert_with(|| {
            out.push((Vec::new(), Effects::NONE));
            out.len() - 1
        });
        out[k].0.push(i as u32);
        out[k].1 |= sets.set[i];
    }
    out
}

/// The checks of the program: every fn instance whose decl has a deny clause, and every fn value
/// made where a fn type denies more than the value's own type ([`TypedBody::denied`]), per
/// instance of the body making it.
pub fn plan(prog: &Program, tc: &Typeck) -> Plan {
    let g = &tc.call_graph;
    let mut checks = Vec::new();
    let mut made: Vec<usize> = Vec::new();
    for (i, b) in tc.bodies.iter().enumerate() {
        if !b.denied.is_empty() && !b.has_error {
            made.push(i);
        }
    }
    let mut inst_of: FxHashMap<DeclId, Vec<u32>> = FxHashMap::default();
    let mut key_of: FxHashMap<&InstKey, Vec<u32>> = FxHashMap::default();
    for (i, nd) in g.nodes.iter().enumerate() {
        if let NodeKind::Fn { inst, .. } = &nd.kind {
            if let Some(s) = tc.sigs.fn_sig(inst.decl)
                && s.deny != 0
            {
                checks.push(Check { node: i as u32, deny: Effects(s.deny), why: Why::Clause });
            }
            if !made.is_empty() {
                inst_of.entry(inst.decl).or_default().push(i as u32);
                key_of.entry(inst).or_default().push(i as u32);
            }
        }
    }
    if made.is_empty() {
        return Plan { checks };
    }
    let mut first_closure = vec![u32::MAX; g.nodes.len()];
    for (i, nd) in g.nodes.iter().enumerate() {
        if let NodeKind::Closure { parent, index: 0, .. } = nd.kind {
            first_closure[parent as usize] = i as u32;
        }
    }
    let mut cx = g.types(&tc.types);
    for bi in made {
        let b = &tc.bodies[bi];
        let Some(insts) = inst_of.get(&b.decl) else { continue };
        let mut br = None;
        let Some(icx) = item_cx(prog, b.item, &mut br) else { continue };
        let closures: Vec<u32> =
            crate::closures::closure_nodes(&icx, b.node).into_iter().map(|(node, _)| icx.a(node)).collect();
        for &(tok, deny, coerced) in &b.denied {
            let site = icx.node_span(coerced);
            let deny = Effects(deny);
            for &inst in insts {
                let NodeKind::Fn { inst: key, .. } = &g.nodes[inst as usize].kind else { continue };
                let mut nodes: Vec<u32> = Vec::new();
                if let Some(k) = closures.iter().position(|&t| t == tok) {
                    let c = first_closure[inst as usize];
                    if c != u32::MAX {
                        nodes.push(c + k as u32);
                    }
                } else if let Some(Target::Fn { decl, args }) = b.target(tok) {
                    let args: Vec<GArg> = if key.args.is_empty() {
                        args.clone()
                    } else {
                        args.iter().map(|&a| crate::mono::subst_garg(&mut cx, a, &key.args)).collect()
                    };
                    let is_dispatch = prog.decl(*decl).is_some_and(|d| d.flags.has(DeclFlags::DISPATCH));
                    let decls = if is_dispatch {
                        let mut dbr = None;
                        match prog.decl(*decl).and_then(|d| item_cx(prog, d.item, &mut dbr)) {
                            Some(dcx) => crate::dispatch::arm_targets(prog, &tc.sigs, &dcx, *decl)
                                .into_iter()
                                .map(|a| InstKey { decl: a, args: Vec::new() })
                                .collect(),
                            None => Vec::new(),
                        }
                    } else {
                        vec![InstKey { decl: *decl, args }]
                    };
                    nodes.extend(decls.iter().filter_map(|k| key_of.get(k)).flatten().copied());
                }
                for m in nodes {
                    checks.push(Check { node: m, deny, why: Why::Made { site } });
                }
            }
        }
    }
    Plan { checks }
}

/// E1401 and E1403 (case `made`) for every check whose node has a denied effect, one per
/// effect, in check order (callers put them into canonical order).
pub fn check(prog: &Program, tc: &Typeck, plan: &Plan, sets: &EffectSets) -> Vec<Diagnostic> {
    let g = &tc.call_graph;
    let mut out = Vec::new();
    // (source fn, effect, where a made value is coerced): one diagnostic each, whichever of the
    // fn's specialized nodes has the effect
    type Key = (String, u8, Option<(u32, u32)>);
    let mut seen: FxHashSet<Key> = FxHashSet::default();
    for c in &plan.checks {
        if !sets.used[c.node as usize] {
            continue;
        }
        let have = Effects(sets.set[c.node as usize].0 & c.deny.0);
        for e in have.each() {
            let Some((edges, end)) = sets.chain(g, c.node, e) else { continue };
            // the names `stave effects` prints: checked-IR paths, type args of an instance,
            // `<fn>.<k>` for a closure, never the fn values a node is specialized on
            let names = g.source_names(prog, tc, &[c.node, end]);
            let (fn_path, at) = (&names[0], &names[1]);
            let site = match &c.why {
                Why::Clause => None,
                Why::Made { site } => Some((site.file.0, site.offset)),
            };
            if !seen.insert((fn_path.clone(), e.0, site)) {
                continue;
            }
            let op = sets.op(end, e);
            let effect = e.names()[0];
            let op_text = op.map_or(OpKind::Trap.text(), |o| o.kind.text());
            let mut roles: Vec<(&str, Span)> = Vec::new();
            let mut primary;
            match &c.why {
                Why::Clause => {
                    let NodeKind::Fn { inst, .. } = &g.nodes[c.node as usize].kind else { continue };
                    let clause = tc
                        .sigs
                        .fn_sig(inst.decl)
                        .and_then(|s| s.deny_spans.iter().find(|x| x.0 & e.0 != 0).map(|x| x.1))
                        .unwrap_or(g.nodes[c.node as usize].span);
                    primary = clause;
                    if !inst.args.is_empty() {
                        // an instance: at the call that instantiates it (the first caller's)
                        if let Some((_, edge)) =
                            g.pred(c.node).find(|(_, x)| x.kind != crate::callgraph::EdgeKind::Member)
                        {
                            primary = edge.site;
                            roles.push(("deny", clause));
                        }
                    }
                }
                Why::Made { site } => primary = *site,
            }
            for &ei in &edges {
                let edge = &g.out[ei as usize];
                let role = if edge.kind == crate::callgraph::EdgeKind::Drop { "drop" } else { "call" };
                roles.push((role, edge.site));
            }
            let whole_fn = op.is_none_or(|o| o.span == g.nodes[end as usize].span);
            if let Some(o) = op
                && !whole_fn
            {
                roles.push(("op", o.span));
            }
            let d = match &c.why {
                Why::Clause => {
                    let holes =
                        [("fn", fn_path.as_str()), ("effect", effect), ("at", at.as_str()), ("op", op_text)];
                    mk("E1401", if edges.is_empty() { Some("self") } else { None }, &holes, primary, &roles)
                }
                Why::Made { .. } => {
                    let holes =
                        [("fn", fn_path.as_str()), ("effect", effect), ("at", at.as_str()), ("op", op_text)];
                    mk("E1403", Some("made"), &holes, primary, &roles)
                }
            };
            out.extend(d);
        }
    }
    out
}
