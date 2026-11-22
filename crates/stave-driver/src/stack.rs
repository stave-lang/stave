//! The computed stack bound of a build: the most stack each thread of the program can use, from
//! the frame sizes the code generator reports ([`stave_obj::StackUse`]) and the call graph
//! ([`stave_sema::callgraph`]), written to the image as `__stave_stack_table`
//! (contracts/intrinsics.md), which the runtime reads when it maps a stack, and the reserve of
//! each function on a cycle, written as `__stave_reserve_table`, which its entry probes.
//!
//! **Graph** ([`graph`], before code generation: it needs no frame size). The nodes are the
//! call-graph nodes plus one node per function of the module that no call-graph node stands for
//! (drop glue, `dispatch` bodies). A function stands for every node of its instance
//! ([`stave_sema::lower_mapped`]), so a node weighs the frame of its function (0: not in this
//! build). The edges are the call graph's, without [`EdgeKind::Spawn`] (the body runs on another
//! stack), plus each direct call of the module (and the calls of the `cpu_init` and `copy_bytes`
//! intrinsics) that the call graph has no edge for at the caller's node: the runtime helpers
//! lowering inserts and calls an optimized body makes directly. A call through a fn value takes
//! the call graph's targets; a node of a function that has one but no such edge, or any such call
//! when a function made into a fn value has no node, is unresolved, and so is one that calls C
//! code (`ccall`, or an `extern "C"` import). A node is a *member* when its strongly connected
//! component is a cycle (more than one node, or a self-edge): a call cycle (named by its
//! `rec fn`), a drop cycle of a recursive type, or one the added edges close. A function with a
//! member node *probes*: it gets a slot of the reserve table, and its entry, after the frame,
//! touches every page down to its reserve below the frame (the back ends), so that a stack too
//! short for what runs below it faults in that entry, a member, and never further down.
//!
//! **Depth** of a node: its weight plus the largest depth among its successors and its *leaf*:
//! if its code has a trap stub, the depth `T` of a trap, twice the depth of `runtime.panic.trap`
//! (a trap raised while one is reported calls `runtime.panic.begin` first, which exits, so it
//! adds at most one more such path; that needs every function `begin` reaches to be free of trap
//! stubs, which the pass checks, else every trapping stack is unbounded), and, for a node of a
//! probing function, that function's reserve, which its entry touches. Computed sinks first by
//! out-degree peeling, on flat arrays without recursion. A node that is unresolved, on a cycle,
//! or reaches either has no bound. A `rec fn` on no cycle of this graph is bounded like any fn.
//!
//! **Reserve** of a member: the largest depth among its successors outside its component, and its
//! own trap leaf, where every member counts as the 16 bytes its caller's call puts down before
//! the member's entry probes (what is below a member is that member's own reserve), and what an
//! unresolved node calls counts as nothing (it is not known). A probing function's reserve is the
//! largest of its members'. A function with a member node and a node off every cycle below it
//! feeds its own reserve: the computation repeats until no reserve grows, at most
//! [`RESERVE_ROUNDS`] times.
//!
//! **Stacks.** Main's stack holds the `__stave_main` stub and `main`. A thread runs the
//! `__stave_thread_start` stub, `runtime.thread.entry`, and the fn value it was started with; each
//! function a call-graph [`EdgeKind::Spawn`] edge reaches (`spawn` bodies, the closures `std.thread`
//! starts, every fn value `runtime.thread.start` can be given) gets a row keyed by its code
//! address. The fn-value call of `runtime.thread.run_queued` is left out: it runs a queued value
//! at a join only in an image without a stack table (the IR interpreter; a native image fails to
//! create the thread instead, so a spawn body never runs on the joiner's bounded stack). A stack
//! with a bound gets it plus [`MARGIN`]. A stack that reaches a cycle gets `[run] stack` plus
//! its *acyclic bound* plus [`MARGIN`]: the depth over the graph with each component collapsed
//! to one node (the largest weight and leaf of its members, the successors of all of them), so
//! every path that runs while no member is live fits, and the first entry of a member finds its
//! reserve. A stack that reaches an unresolved node gets `[run] stack`. The runtime rounds up to
//! the page size. No error comes from the bound: frame sizes differ between debug and release.

use stave_ir::{FuncId, Known, Module, Op, Operand, Term};
use stave_obj::{DataObj, StackUse, SymbolId};
use stave_sema::callgraph::{CallGraph, EdgeKind, NodeKind};

/// `[run] stack` when the manifest has none (contracts/gpu-program.md; `runtime.config.RUN_STACK`).
pub const DEFAULT_RUN_STACK: u64 = 8 << 20;

/// Bytes added to every computed bound in the table. Signal handlers run on the alternate signal
/// stack and system calls use no user stack; the margin covers what no frame size counts (none
/// is known).
pub const MARGIN: u64 = 4096;

/// The `__stave_main` stub (return address and saved frame pointer on x86-64, `x29, x30` on
/// AArch64).
const MAIN_STUB: u64 = 16;

/// The `__stave_thread_start` stub and the thread's start-up code before `runtime.thread.entry`.
const THREAD_STUB: u64 = 16;

/// What the bound reads besides the module and the frame sizes.
pub struct Input<'a> {
    pub graph: &'a CallGraph,
    /// Per function of the module: the call-graph nodes it stands for.
    pub nodes_of_func: &'a [Vec<u32>],
    /// `[run] stack` of the root package.
    pub run_stack: Option<u64>,
}

/// A node of the bound's graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum At {
    /// A call-graph node.
    Graph(u32),
    /// A function no call-graph node stands for.
    Func(FuncId),
}

/// Why a stack has no bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// It reaches a call cycle through this `rec fn`.
    Rec(At),
    /// It reaches a call cycle through this node.
    Cycle(At),
    /// It reaches a call through a fn value whose targets are not known, in this node.
    FnValue(At),
    /// It reaches a call of C code (`ccall` intrinsics), whose stack use is not known, in this
    /// node.
    Foreign(At),
    /// The call graph is incomplete (an instantiation chain too deep, too many instances).
    Incomplete,
}

/// Bytes of stack, or why there is no bound.
pub type Depth = Result<u64, Why>;

/// The stack of threads started with one function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Thread {
    pub func: FuncId,
    /// The call-graph nodes of `func` that a [`EdgeKind::Spawn`] edge reaches.
    pub starts: Vec<u32>,
    pub depth: Depth,
    /// Without a bound for a cycle: the acyclic bound (each component collapsed), if known.
    pub acyclic: Option<u64>,
}

/// The bound of every stack of a build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    /// `[run] stack` of the manifest, if it sets one.
    pub run_stack: Option<u64>,
    pub main: Depth,
    /// Without a bound for main's stack: its acyclic bound, if known.
    pub main_acyclic: Option<u64>,
    /// By function.
    pub threads: Vec<Thread>,
    /// By reserve-table slot ([`Graph::slots`]): the bytes the function's entry probes below its
    /// frame.
    pub reserve: Vec<u64>,
    /// By reserve-table slot: the function.
    pub probes: Vec<FuncId>,
}

impl Report {
    /// `[run] stack` in effect.
    pub fn run_stack(&self) -> u64 {
        self.run_stack.unwrap_or(DEFAULT_RUN_STACK)
    }

    /// The stack bytes a table row holds: the bound plus [`MARGIN`]; without one, `[run] stack`
    /// plus the acyclic bound plus [`MARGIN`]; without that either, 0 (the default applies).
    pub fn bytes(&self, d: Depth, acyclic: Option<u64>) -> u64 {
        match (d, acyclic) {
            (Ok(b), _) => b.saturating_add(MARGIN),
            (Err(_), Some(a)) => self.run_stack().saturating_add(a).saturating_add(MARGIN),
            (Err(_), None) => 0,
        }
    }

    /// `__stave_stack_table`, with `syms` the symbol of each function.
    pub fn table(&self, syms: &[Option<SymbolId>]) -> DataObj {
        let rows: Vec<(SymbolId, u64)> = self
            .threads
            .iter()
            .map(|t| (t, self.bytes(t.depth, t.acyclic)))
            .filter(|&(_, n)| n != 0)
            .filter_map(|(t, n)| syms.get(t.func.index()).copied().flatten().map(|s| (s, n)))
            .collect();
        let main = self.bytes(self.main, self.main_acyclic);
        stave_obj::stub::stack_table(self.run_stack.unwrap_or(0), main, &rows)
    }

    /// `__stave_reserve_table`.
    pub fn reserve_table(&self) -> DataObj {
        stave_obj::stub::reserve_table(&self.reserve)
    }
}

/// The larger of two depths; no bound wins.
fn max_d(a: Depth, b: Depth) -> Depth {
    match (a, b) {
        (Err(w), _) | (_, Err(w)) => Err(w),
        (Ok(x), Ok(y)) => Ok(x.max(y)),
    }
}

fn add_d(w: u64, d: Depth) -> Depth {
    d.map(|x| x.saturating_add(w))
}

/// Longest paths over a graph in CSR form (`start[x]..start[x + 1]` index `to`): the depth of `x`
/// is `weight[x]` plus the largest of `leaf[x]` and its successors' depths; a `poison` node has
/// that error. Sinks first, by out-degree peeling (iterative). `None`: the node reaches a cycle.
pub(crate) fn longest(
    start: &[u32],
    to: &[u32],
    weight: &[u64],
    leaf: &[Depth],
    poison: &[Option<Why>],
) -> Vec<Option<Depth>> {
    let n = weight.len();
    let mut out_left: Vec<u32> = (0..n).map(|x| start[x + 1] - start[x]).collect();
    // in-edges (CSR of the transpose), duplicates kept so the counts match
    let mut in_start = vec![0u32; n + 1];
    for &t in to {
        in_start[t as usize + 1] += 1;
    }
    for x in 0..n {
        in_start[x + 1] += in_start[x];
    }
    let mut fill = in_start.clone();
    let mut from = vec![0u32; to.len()];
    for x in 0..n {
        for &t in &to[start[x] as usize..start[x + 1] as usize] {
            from[fill[t as usize] as usize] = x as u32;
            fill[t as usize] += 1;
        }
    }
    let mut done: Vec<Option<Depth>> = vec![None; n];
    let mut ready: Vec<u32> = (0..n as u32).filter(|&x| out_left[x as usize] == 0).collect();
    while let Some(x) = ready.pop() {
        let xi = x as usize;
        let d = match poison[xi] {
            Some(w) => Err(w),
            None => {
                let mut best = leaf[xi];
                for &s in &to[start[xi] as usize..start[xi + 1] as usize] {
                    // every successor is done: x became ready when its last one was
                    best = max_d(best, done[s as usize].unwrap_or(Ok(0)));
                }
                add_d(weight[xi], best)
            }
        };
        done[xi] = Some(d);
        for &p in &from[in_start[xi] as usize..in_start[xi + 1] as usize] {
            out_left[p as usize] -= 1;
            if out_left[p as usize] == 0 {
                ready.push(p);
            }
        }
    }
    done
}

/// The nodes of a cycle that `x` (not done by [`longest`]) reaches: every such node has a
/// successor that is not done either, so the walk along the first such successor repeats a node,
/// and from there it goes around one cycle.
fn on_cycle(start: &[u32], to: &[u32], done: &[Option<Depth>], x: u32) -> Vec<u32> {
    let next = |c: u32| {
        let succ = &to[start[c as usize] as usize..start[c as usize + 1] as usize];
        succ.iter().copied().find(|&s| done[s as usize].is_none())
    };
    let mut seen = vec![false; done.len()];
    let mut cur = x;
    while !seen[cur as usize] {
        seen[cur as usize] = true;
        match next(cur) {
            Some(s) => cur = s,
            None => return vec![cur],
        }
    }
    let mut cycle = vec![cur];
    let mut c = cur;
    while let Some(s) = next(c)
        && s != cur
    {
        cycle.push(s);
        c = s;
    }
    cycle
}

/// A node with a trap stub that `from` reaches (themselves included), if any.
fn trapping_node(start: &[u32], to: &[u32], traps: &[bool], from: &[u32]) -> Option<u32> {
    let mut seen = vec![false; traps.len()];
    let mut work: Vec<u32> = from.to_vec();
    while let Some(x) = work.pop() {
        if std::mem::replace(&mut seen[x as usize], true) {
            continue;
        }
        if traps[x as usize] {
            return Some(x);
        }
        work.extend_from_slice(&to[start[x as usize] as usize..start[x as usize + 1] as usize]);
    }
    None
}

/// What one function's body calls.
#[derive(Default)]
struct Calls {
    direct: Vec<FuncId>,
    indirect: bool,
    foreign: bool,
}

/// The fns the backends call in place of an intrinsic: `cpu_init` calls `runtime.cpu.init`,
/// `copy_bytes` calls `std.slice.copy_raw`.
struct IntrinsicCalls {
    cpu_init: Option<FuncId>,
    copy_raw: Option<FuncId>,
}

/// The direct callees and fn-value calls of `f`, and the functions it makes into fn values
/// (added to `made`).
fn calls_of(m: &Module, f: FuncId, called: &IntrinsicCalls, made: &mut [bool]) -> Calls {
    let mut c = Calls::default();
    let Some(b) = m.func(f).and_then(|x| x.body.as_ref()) else { return c };
    let value = |o: &Operand, made: &mut [bool]| {
        if let Operand::Func(g) = o
            && let Some(x) = made.get_mut(g.index())
        {
            *x = true;
        }
    };
    for bd in &b.blocks {
        for i in bd.insts.indices() {
            let args = b.args(i);
            let op = b.insts.op.get(i as usize).copied();
            let callee = match (op, args.first()) {
                (Some(Op::Call | Op::CallInd), Some(Operand::Func(g))) => Some(*g),
                _ => None,
            };
            match (op, callee) {
                (Some(Op::Call | Op::CallInd), Some(g)) => {
                    // an import runs C code on this stack: no frame size is known for it
                    if m.func(g).is_some_and(|x| x.body.is_none() && x.symbol.is_some()) {
                        c.foreign = true;
                    }
                    c.direct.push(g);
                }
                (Some(Op::CallInd), None) => c.indirect = true,
                (Some(Op::Intrinsic(id)), _) => match m.intrinsic_name(id) {
                    Some("cpu_init") => c.direct.extend(called.cpu_init),
                    Some("copy_bytes") => c.direct.extend(called.copy_raw),
                    Some(name) if name.starts_with("ccall") => c.foreign = true,
                    _ => {}
                },
                _ => {}
            }
            let skip = usize::from(callee.is_some());
            for o in &args[skip..] {
                value(o, made);
            }
        }
        let mut targets = Vec::new();
        match &bd.term {
            Term::Br(t) => targets.push(*t),
            Term::Cbr { cond, then, els } => {
                value(cond, made);
                targets.extend([*then, *els]);
            }
            Term::Switch { scrut, cases, default } => {
                value(scrut, made);
                targets.push(*default);
                targets.extend(b.switch_cases(*cases).iter().map(|c| c.target));
            }
            Term::Ret(Some(o)) => value(o, made),
            Term::Ret(None) | Term::Unreachable => {}
        }
        for t in &targets {
            for o in b.target_args(t) {
                value(o, made);
            }
        }
    }
    c.direct.sort_unstable();
    c.direct.dedup();
    c
}

/// What a caller's call puts on the stack before a member's entry probes: the return address
/// (x86-64), rounded to the stack alignment.
const ENTRY: u64 = 16;

/// Most passes of the reserve computation (see the module docs).
pub const RESERVE_ROUNDS: usize = 8;

/// No reserve-table slot.
pub const NO_SLOT: u32 = u32::MAX;

/// The bound's graph of a module (see the module docs), with its components.
pub struct Graph {
    /// Call-graph nodes; the nodes from `n` on are functions with no call-graph node.
    n: usize,
    total: usize,
    /// Per function: its nodes.
    ids: Vec<Vec<u32>>,
    /// Per node: the functions that stand for it.
    funcs_of: Vec<Vec<FuncId>>,
    pseudo_func: Vec<FuncId>,
    start: Vec<u32>,
    to: Vec<u32>,
    /// Per edge: through a fn value.
    ind: Vec<bool>,
    poison: Vec<Option<Why>>,
    /// Per node: its component, numbered sinks first (an edge between two components goes to
    /// the smaller number).
    comp: Vec<u32>,
    /// Per component: whether it is a cycle.
    cycle: Vec<bool>,
    /// Per function: its reserve-table slot ([`NO_SLOT`]: it does not probe).
    slots: Vec<u32>,
    nslots: usize,
    incomplete: bool,
}

impl Graph {
    /// Per function: the reserve-table slot of a function that probes at entry, else
    /// [`NO_SLOT`]. Slots are numbered in function order from 0.
    pub fn slots(&self) -> &[u32] {
        &self.slots
    }

    /// Whether call-graph node `x` is a member (its component in this graph is a cycle).
    pub fn member(&self, x: u32) -> bool {
        (x as usize) < self.n && self.cycle[self.comp[x as usize] as usize]
    }

    fn at(&self, x: u32) -> At {
        if (x as usize) < self.n { At::Graph(x) } else { At::Func(self.pseudo_func[x as usize - self.n]) }
    }

    fn succ(&self, x: u32) -> &[u32] {
        &self.to[self.start[x as usize] as usize..self.start[x as usize + 1] as usize]
    }
}

/// Strongly connected components of a graph in CSR form (iterative Tarjan): a component per
/// node, numbered in the order they complete, so an edge between two components goes to the
/// smaller number.
fn components(start: &[u32], to: &[u32]) -> Vec<u32> {
    const NONE: u32 = u32::MAX;
    let n = start.len() - 1;
    let mut index = vec![NONE; n];
    let mut low = vec![0u32; n];
    let mut on = vec![false; n];
    let mut comp = vec![NONE; n];
    let mut stack: Vec<u32> = Vec::new();
    // (node, its next out-edge)
    let mut path: Vec<(u32, u32)> = Vec::new();
    let (mut next, mut count) = (0u32, 0u32);
    for r in 0..n as u32 {
        if index[r as usize] != NONE {
            continue;
        }
        index[r as usize] = next;
        low[r as usize] = next;
        next += 1;
        stack.push(r);
        on[r as usize] = true;
        path.push((r, start[r as usize]));
        while let Some(&(v, e)) = path.last() {
            let vi = v as usize;
            if e < start[vi + 1] {
                if let Some(top) = path.last_mut() {
                    top.1 += 1;
                }
                let w = to[e as usize];
                let wi = w as usize;
                if index[wi] == NONE {
                    index[wi] = next;
                    low[wi] = next;
                    next += 1;
                    stack.push(w);
                    on[wi] = true;
                    path.push((w, start[wi]));
                } else if on[wi] {
                    low[vi] = low[vi].min(index[wi]);
                }
                continue;
            }
            path.pop();
            if let Some(&(p, _)) = path.last() {
                low[p as usize] = low[p as usize].min(low[vi]);
            }
            if low[vi] == index[vi] {
                while let Some(w) = stack.pop() {
                    on[w as usize] = false;
                    comp[w as usize] = count;
                    if w == v {
                        break;
                    }
                }
                count += 1;
            }
        }
    }
    comp
}

/// The bound's graph of module `m` (the module code generation compiles: release builds after
/// the optimizer).
pub fn graph(m: &Module, input: &Input<'_>) -> Graph {
    let g = input.graph;
    let n = g.nodes.len();
    let nf = m.funcs.len();
    let has_body = |f: usize| m.funcs[f].body.is_some();
    // the nodes of each function: its call-graph nodes, else one node of its own
    let mut ids: Vec<Vec<u32>> = Vec::with_capacity(nf);
    let mut funcs_of: Vec<Vec<FuncId>> = vec![Vec::new(); n];
    for f in 0..nf {
        let mut v: Vec<u32> = input
            .nodes_of_func
            .get(f)
            .map_or(&[][..], |v| &v[..])
            .iter()
            .copied()
            .filter(|&x| (x as usize) < n)
            .collect();
        if v.is_empty() && has_body(f) {
            v.push(funcs_of.len() as u32);
            funcs_of.push(Vec::new());
        }
        for &x in &v {
            funcs_of[x as usize].push(FuncId(f as u32));
        }
        ids.push(v);
    }
    let total = funcs_of.len();
    let pseudo_func: Vec<FuncId> = funcs_of[n..].iter().map(|fs| fs[0]).collect();
    let at = |x: u32| if (x as usize) < n { At::Graph(x) } else { At::Func(pseudo_func[x as usize - n]) };
    // A `rec fn` is bounded unless it is on a cycle of this graph: the marker only names the cycle.
    let mut poison: Vec<Option<Why>> = vec![None; total];

    // edges: (from, to, through a fn value)
    let mut edges: Vec<(u32, u32, bool)> = Vec::new();
    for x in 0..n as u32 {
        for e in g.succ(x) {
            if e.kind != EdgeKind::Spawn {
                edges.push((x, e.to, e.kind == EdgeKind::Indirect));
            }
        }
    }
    let queued = m.known_body(Known::RunQueued);
    let called =
        IntrinsicCalls { cpu_init: m.known_body(Known::CpuInit), copy_raw: m.known_body(Known::CopyRaw) };
    let mut made = vec![false; nf];
    let calls: Vec<Calls> = (0..nf).map(|f| calls_of(m, FuncId(f as u32), &called, &mut made)).collect();
    let unmapped_values = (0..nf).any(|f| made[f] && ids[f].iter().all(|&x| x as usize >= n));
    for (f, c) in calls.iter().enumerate() {
        for &x in &ids[f] {
            let graph = (x as usize) < n;
            let succ = if graph { g.succ(x) } else { &[] };
            for &callee in &c.direct {
                let ys = ids.get(callee.index()).map_or(&[][..], |v| &v[..]);
                // out-edges are sorted by target, one per target
                let covered = ys.iter().any(|&y| {
                    succ.binary_search_by_key(&y, |e| e.to).is_ok_and(|k| succ[k].kind != EdgeKind::Spawn)
                });
                if !covered {
                    edges.extend(ys.iter().map(|&y| (x, y, false)));
                }
            }
            if c.foreign && poison[x as usize].is_none() {
                poison[x as usize] = Some(Why::Foreign(at(x)));
            }
            if c.indirect
                && Some(FuncId(f as u32)) != queued
                && poison[x as usize].is_none()
                && (!graph || unmapped_values || !succ.iter().any(|e| e.kind == EdgeKind::Indirect))
            {
                poison[x as usize] = Some(Why::FnValue(at(x)));
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let mut start = vec![0u32; total + 1];
    for &(a, _, _) in &edges {
        start[a as usize + 1] += 1;
    }
    for x in 0..total {
        start[x + 1] += start[x];
    }
    let to: Vec<u32> = edges.iter().map(|e| e.1).collect();
    let ind: Vec<bool> = edges.iter().map(|e| e.2).collect();
    let comp = components(&start, &to);
    let ncomp = comp.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let mut size = vec![0u32; ncomp];
    for &c in &comp {
        size[c as usize] += 1;
    }
    let mut cycle: Vec<bool> = size.iter().map(|&k| k > 1).collect();
    for &(a, b, _) in &edges {
        if a == b {
            cycle[comp[a as usize] as usize] = true;
        }
    }
    let mut slots = vec![NO_SLOT; nf];
    let mut nslots = 0usize;
    for f in 0..nf {
        if has_body(f) && ids[f].iter().any(|&x| cycle[comp[x as usize] as usize]) {
            slots[f] = nslots as u32;
            nslots += 1;
        }
    }
    Graph {
        n,
        total,
        ids,
        funcs_of,
        pseudo_func,
        start,
        to,
        ind,
        poison,
        comp,
        cycle,
        slots,
        nslots,
        incomplete: g.truncated,
    }
}

/// The bound of every stack of graph `bg` of module `m`, with `uses` the backend's report per
/// function.
pub fn compute(bg: &Graph, m: &Module, input: &Input<'_>, uses: &[StackUse], main: FuncId) -> Report {
    let g = input.graph;
    let Graph { n, total, ref ids, ref start, ref to, ref ind, ref poison, ref comp, ref cycle, .. } = *bg;
    let frame = |f: FuncId| uses.get(f.index()).copied().unwrap_or_default();
    let mut weight = vec![0u64; total];
    let mut traps = vec![false; total];
    for (f, v) in ids.iter().enumerate() {
        let u = frame(FuncId(f as u32));
        for &x in v {
            weight[x as usize] = weight[x as usize].max(u64::from(u.frame));
            traps[x as usize] |= u.traps;
        }
    }
    let at = |x: u32| bg.at(x);
    let member = |x: u32| cycle[comp[x as usize] as usize];

    let finish = |done: &[Option<Depth>], x: u32| -> Depth {
        match done[x as usize] {
            Some(d) => d,
            None => {
                // a cycle through a `rec fn` is reported as that fn
                let cycle = on_cycle(start, to, done, x);
                match cycle.iter().find(|&&c| (c as usize) < n && g.nodes[c as usize].rec()) {
                    Some(&c) => Err(Why::Rec(at(c))),
                    None => Err(Why::Cycle(at(cycle[0]))),
                }
            }
        }
    };
    let over = |done: &[Option<Depth>], xs: &[u32]| -> Depth {
        xs.iter().try_fold(0, |acc, &x| finish(done, x).map(|d| acc.max(d)))
    };

    // the depth of a trap, from the depths without trap stubs: the report's own path, plus one
    // more trap raised anywhere on it, which calls `runtime.panic.begin` first and exits there.
    // That holds only if nothing `begin` reaches traps (else a trap there starts over).
    let zero = vec![Ok(0); total];
    let first = longest(start, to, &weight, &zero, poison);
    let trap = match (m.known_body(Known::Trap), m.known_body(Known::PanicBegin)) {
        (Some(t), Some(b)) => match trapping_node(start, to, &traps, &ids[b.index()]) {
            Some(x) => Err(Why::Cycle(at(x))),
            None => over(&first, &ids[t.index()]).map(|d| d.saturating_mul(2)),
        },
        (Some(t), None) => Err(Why::Cycle(at(ids[t.index()].first().copied().unwrap_or(0)))),
        // `__stave_panic_fallback`: called, no frame
        (None, _) => Ok(16),
    };
    let leaf: Vec<Depth> = traps.iter().map(|&t| if t { trap } else { Ok(0) }).collect();

    // nodes sinks first: by component, which numbers every callee's component below its caller's
    let mut order: Vec<u32> = (0..total as u32).collect();
    order.sort_unstable_by_key(|&x| comp[x as usize]);
    let slots_of: Vec<Vec<usize>> = (0..total)
        .map(|x| {
            bg.funcs_of[x]
                .iter()
                .map(|f| bg.slots[f.index()])
                .filter(|&s| s != NO_SLOT)
                .map(|s| s as usize)
                .collect()
        })
        .collect();
    // what is known: an unresolved node counts its own frame, a trap leaf without a bound nothing
    let known = |d: Depth| d.unwrap_or(0);
    let mut reserve = vec![0u64; bg.nslots];
    for _ in 0..RESERVE_ROUNDS {
        // the depth of a node below a member's frame, where a member counts as `ENTRY`
        let mut acyc = vec![0u64; total];
        let mut next = vec![0u64; bg.nslots];
        for &x in &order {
            let xi = x as usize;
            if member(x) {
                acyc[xi] = ENTRY;
                let mut r = known(leaf[xi]);
                for &y in bg.succ(x) {
                    if comp[y as usize] != comp[xi] {
                        r = r.max(acyc[y as usize]);
                    }
                }
                for &s in &slots_of[xi] {
                    next[s] = next[s].max(r);
                }
            } else {
                let mut d = known(leaf[xi]);
                for &s in &slots_of[xi] {
                    d = d.max(reserve[s]);
                }
                for &y in bg.succ(x) {
                    d = d.max(acyc[y as usize]);
                }
                acyc[xi] = weight[xi].saturating_add(d);
            }
        }
        if next == reserve {
            break;
        }
        reserve = next;
    }
    // a node of a probing function touches its reserve below its frame
    let leaf: Vec<Depth> =
        (0..total).map(|x| slots_of[x].iter().fold(leaf[x], |d, &s| max_d(d, Ok(reserve[s])))).collect();
    let done = longest(start, to, &weight, &leaf, poison);

    // the acyclic bound of each component: its largest member weight, plus the largest leaf of
    // its members and acyclic bound of a callee component (all smaller, so complete when read)
    let ncomp = cycle.len();
    let mut cw = vec![0u64; ncomp];
    for x in 0..total {
        let c = comp[x] as usize;
        cw[c] = cw[c].max(weight[x]);
    }
    let mut below: Vec<Depth> = vec![Ok(0); ncomp];
    for &x in &order {
        let (xi, c) = (x as usize, comp[x as usize] as usize);
        let mut d = max_d(below[c], leaf[xi]);
        if let Some(w) = poison[xi] {
            d = Err(w);
        }
        for &y in bg.succ(x) {
            let cy = comp[y as usize] as usize;
            if cy != c {
                d = max_d(d, add_d(cw[cy], below[cy]));
            }
        }
        below[c] = d;
    }
    let flat = |x: u32| -> Depth {
        let c = comp[x as usize] as usize;
        add_d(cw[c], below[c])
    };
    let over_flat = |xs: &[u32]| -> Depth { xs.iter().try_fold(0, |acc, &x| flat(x).map(|d| acc.max(d))) };

    let incomplete = bg.incomplete;
    let main_d =
        if incomplete { Err(Why::Incomplete) } else { add_d(MAIN_STUB, over(&done, &ids[main.index()])) };
    let main_acyclic = match main_d {
        Err(Why::Rec(_) | Why::Cycle(_)) => add_d(MAIN_STUB, over_flat(&ids[main.index()])).ok(),
        _ => None,
    };

    // what can run: reachable from main over the calls and the spawn edges (a call-graph node
    // exists for every non-generic fn, called or not)
    let mut live = vec![false; total];
    let mut work: Vec<u32> = ids[main.index()].clone();
    while let Some(x) = work.pop() {
        if std::mem::replace(&mut live[x as usize], true) {
            continue;
        }
        work.extend_from_slice(bg.succ(x));
        if (x as usize) < n {
            work.extend(g.succ(x).iter().filter(|e| e.kind == EdgeKind::Spawn).map(|e| e.to));
        }
    }

    // threads: the functions live spawn edges reach (a node of `runtime.thread.start` has one to
    // each fn value it can be given)
    let mut rows: Vec<(FuncId, u32)> = Vec::new();
    for x in (0..n as u32).filter(|&x| live[x as usize]) {
        for e in g.succ(x) {
            if e.kind == EdgeKind::Spawn {
                rows.extend(bg.funcs_of[e.to as usize].iter().map(|&f| (f, e.to)));
            }
        }
    }
    rows.sort_unstable();
    rows.dedup();
    let mut threads: Vec<Thread> = Vec::new();
    for (f, s) in rows {
        match threads.last_mut() {
            Some(t) if t.func == f => t.starts.push(s),
            _ => threads.push(Thread { func: f, starts: vec![s], depth: Ok(0), acyclic: None }),
        }
    }
    let entry = m.known_body(Known::ThreadEntry);
    // a thread's stack, from `body` (its start nodes) and `node` (the depth of any other node)
    let thread_depth = |body: Depth, node: &dyn Fn(u32) -> Depth| -> Depth {
        match entry {
            _ if incomplete => Err(Why::Incomplete),
            None => body,
            Some(e) => {
                // entry with its fn-value call replaced by a call of this body
                let mut d: Depth = Ok(0);
                for &x in &ids[e.index()] {
                    let xi = x as usize;
                    let own = match poison[xi] {
                        Some(Why::FnValue(_)) | None => {
                            let mut best = max_d(leaf[xi], body);
                            for k in start[xi]..start[xi + 1] {
                                if !ind[k as usize] {
                                    best = max_d(best, node(to[k as usize]));
                                }
                            }
                            add_d(weight[xi], best)
                        }
                        Some(w) => Err(w),
                    };
                    d = max_d(d, own);
                }
                add_d(THREAD_STUB, d)
            }
        }
    };
    for t in &mut threads {
        // the nodes a thread can start with, not every node of the function: a node no start
        // reaches (an unspecialized one whose fn-value slots stay empty) never runs on a thread
        t.depth = thread_depth(over(&done, &t.starts), &|x| finish(&done, x));
        if let Err(Why::Rec(_) | Why::Cycle(_)) = t.depth {
            t.acyclic = thread_depth(over_flat(&t.starts), &flat).ok();
        }
    }
    let mut probes = vec![FuncId(0); bg.nslots];
    for (f, &k) in bg.slots.iter().enumerate() {
        if k != NO_SLOT {
            probes[k as usize] = FuncId(f as u32);
        }
    }
    Report { run_stack: input.run_stack, main: main_d, main_acyclic, threads, reserve, probes }
}

/// The build summary's stack lines (contracts/cli.md): main first, then one per thread start
/// function. `locate` gives `<file>:<line>:<col>` of a span.
pub fn summary(
    r: &Report,
    prog: &stave_sema::Program,
    tc: &stave_sema::Typeck,
    m: &Module,
    locate: impl Fn(stave_diag::Span) -> String,
) -> Vec<String> {
    let g = &tc.call_graph;
    let name = |a: At| match a {
        At::Graph(x) => g.source_names(prog, tc, &[x]).pop().unwrap_or_default(),
        At::Func(f) => m.func(f).map_or_else(String::new, |f| f.path.clone()),
    };
    let line = |what: String, d: Depth, acyclic: Option<u64>| match d {
        Ok(b) => format!("stack: {what}: {b} bytes"),
        Err(w) => {
            let why = match w {
                Why::Rec(a) => format!("reaches rec fn `{}`", name(a)),
                Why::Cycle(a) => format!("reaches a call cycle through `{}`", name(a)),
                Why::FnValue(a) => format!("calls a fn value with unknown targets in `{}`", name(a)),
                Why::Foreign(a) => format!("calls C code in `{}`", name(a)),
                Why::Incomplete => "the call graph is incomplete".to_string(),
            };
            let run = r.run_stack();
            match acyclic {
                Some(a) => format!("stack: {what}: unbounded ({why}), [run] stack {run} + {a} bytes"),
                None => format!("stack: {what}: unbounded ({why}), [run] stack {run} bytes"),
            }
        }
    };
    let mut out = vec![line("main".to_string(), r.main, r.main_acyclic)];
    for t in &r.threads {
        let what = match t.starts.iter().find(|&&x| is_spawn(g, x)) {
            Some(&x) => format!("spawn at {}", locate(g.nodes[x as usize].span)),
            None => match t.starts.first() {
                Some(&x) => format!("thread `{}`", name(At::Graph(x))),
                None => format!("thread `{}`", name(At::Func(t.func))),
            },
        };
        out.push(line(what, t.depth, t.acyclic));
    }
    out
}

/// Whether call-graph node `x` is a `spawn` body.
pub fn is_spawn(g: &CallGraph, x: u32) -> bool {
    matches!(g.nodes.get(x as usize).map(|nd| &nd.kind), Some(NodeKind::Closure { spawn: true, .. }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_direct_call_of_an_import_is_foreign() {
        use stave_ir::{FuncBuilder, InstanceId};
        let mut m = Module::new();
        let import = m.declare(InstanceId(vec![0]), "c.puts", Vec::new(), None);
        m.func_mut(import).unwrap().symbol = Some("puts".to_string());
        let stave_fn = m.declare(InstanceId(vec![1]), "main.leaf", Vec::new(), None);
        let mut b = FuncBuilder::new(&[]);
        b.ret(None);
        m.define(stave_fn, b.finish().unwrap());
        let mut made = vec![false; 4];
        for (callee, foreign) in [(import, true), (stave_fn, false)] {
            let f = m.declare(InstanceId(vec![2 + callee.index() as u32]), "main.f", Vec::new(), None);
            let mut b = FuncBuilder::new(&[]);
            b.call(callee, None, &[]);
            b.ret(None);
            m.define(f, b.finish().unwrap());
            let none = IntrinsicCalls { cpu_init: None, copy_raw: None };
            let c = calls_of(&m, f, &none, &mut made);
            assert_eq!(c.foreign, foreign);
            assert_eq!(c.direct, vec![callee]);
        }
    }

    fn csr(n: usize, es: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
        let mut es = es.to_vec();
        es.sort_unstable();
        let mut start = vec![0u32; n + 1];
        for &(a, _) in &es {
            start[a as usize + 1] += 1;
        }
        for x in 0..n {
            start[x + 1] += start[x];
        }
        (start, es.iter().map(|e| e.1).collect())
    }

    fn run(
        n: usize,
        es: &[(u32, u32)],
        w: &[u64],
        leaf: &[Depth],
        poison: &[Option<Why>],
    ) -> Vec<Option<Depth>> {
        let (start, to) = csr(n, es);
        longest(&start, &to, w, leaf, poison)
    }

    /// A diamond takes its heavier side; a sink is its own weight.
    #[test]
    fn diamond_takes_the_longer_path() {
        let d = run(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], &[10, 100, 1, 7], &[Ok(0); 4], &[None; 4]);
        assert_eq!(d, vec![Some(Ok(117)), Some(Ok(107)), Some(Ok(8)), Some(Ok(7))]);
    }

    /// A leaf term counts like one more successor.
    #[test]
    fn leaf_term_is_a_successor() {
        let d = run(2, &[(0, 1)], &[5, 5], &[Ok(50), Ok(0)], &[None; 2]);
        assert_eq!(d[0], Some(Ok(55)));
        let d = run(2, &[(0, 1)], &[5, 5], &[Ok(1), Ok(0)], &[None; 2]);
        assert_eq!(d[0], Some(Ok(10)));
    }

    /// A poisoned node makes every node that reaches it unbounded, and nothing else.
    #[test]
    fn poison_reaches_callers_only() {
        let w = Why::Rec(At::Graph(2));
        let d = run(4, &[(0, 1), (1, 2), (0, 3)], &[1, 1, 1, 1], &[Ok(0); 4], &[None, None, Some(w), None]);
        assert_eq!(d, vec![Some(Err(w)), Some(Err(w)), Some(Err(w)), Some(Ok(1))]);
    }

    /// Nodes on or above a cycle are not done; a node below it is.
    #[test]
    fn cycle_leaves_callers_open() {
        let es = [(0, 1), (1, 2), (2, 1), (2, 3)];
        let (start, to) = csr(4, &es);
        let d = longest(&start, &to, &[1; 4], &[Ok(0); 4], &[None; 4]);
        assert_eq!(d, vec![None, None, None, Some(Ok(1))]);
        let mut c = on_cycle(&start, &to, &d, 0);
        c.sort_unstable();
        assert_eq!(c, vec![1, 2]);
    }

    /// A cycle is one component; every edge between components goes to a smaller number.
    #[test]
    fn components_number_callees_first() {
        let es = [(0, 1), (1, 2), (2, 1), (2, 3), (4, 4), (0, 4)];
        let (start, to) = csr(5, &es);
        let c = components(&start, &to);
        assert_eq!(c[1], c[2]);
        assert_eq!(c.iter().collect::<std::collections::BTreeSet<_>>().len(), 4);
        for &(a, b) in &es {
            assert!(c[a as usize] >= c[b as usize], "{a} -> {b}: {c:?}");
        }
        // a long chain needs no recursion
        let n = 200_000usize;
        let es: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).chain([(n as u32 - 1, 0)]).collect();
        let (start, to) = csr(n, &es);
        assert!(components(&start, &to).iter().all(|&k| k == 0));
    }

    /// The trap stubs `runtime.panic.begin`'s callees may not have: found anywhere below it.
    #[test]
    fn trapping_node_below() {
        let (start, to) = csr(4, &[(0, 1), (1, 2), (3, 0)]);
        assert_eq!(trapping_node(&start, &to, &[false, false, true, false], &[0]), Some(2));
        assert_eq!(trapping_node(&start, &to, &[false, false, false, true], &[0]), None);
    }

    /// Duplicate edges are counted consistently.
    #[test]
    fn duplicate_edges() {
        let d = run(2, &[(0, 1), (0, 1)], &[3, 4], &[Ok(0); 2], &[None; 2]);
        assert_eq!(d[0], Some(Ok(7)));
    }

    /// A long chain needs no recursion (the depth is linear in the chain).
    #[test]
    fn long_chain() {
        let n = 200_000usize;
        let es: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
        let d = run(n, &es, &vec![2; n], &vec![Ok(0); n], &vec![None; n]);
        assert_eq!(d[0], Some(Ok(2 * n as u64)));
    }

    #[test]
    fn table_bytes() {
        let r = Report {
            run_stack: Some(1 << 16),
            main: Ok(0),
            main_acyclic: None,
            threads: Vec::new(),
            reserve: Vec::new(),
            probes: Vec::new(),
        };
        assert_eq!(r.bytes(Ok(100), None), 100 + MARGIN);
        assert_eq!(r.bytes(Err(Why::Incomplete), None), 0);
        assert_eq!(r.bytes(Err(Why::Rec(At::Graph(0))), Some(100)), (1 << 16) + 100 + MARGIN);
    }
}
