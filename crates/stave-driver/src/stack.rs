//! The computed stack bound of a build: the most stack each thread of the program can use, from
//! the frame sizes the code generator reports ([`stave_obj::StackUse`]) and the call graph
//! ([`stave_sema::callgraph`]), written to the image as `__stave_stack_table`
//! (contracts/intrinsics.md), which the runtime reads when it maps a stack.
//!
//! **Graph.** The nodes are the call-graph nodes plus one node per function of the module that no
//! call-graph node stands for (drop glue, `dispatch` bodies). A function stands for every node of
//! its instance ([`stave_sema::lower_mapped`]), so a node weighs the frame of its function (0: not
//! in this build). The edges are the call graph's, without [`EdgeKind::Spawn`] (the body runs on
//! another stack), plus each direct call of the module (and the calls of the `cpu_init` and `copy_bytes` intrinsics)
//! that the call graph has no edge for at the caller's node: the runtime helpers lowering inserts
//! and calls an optimized body makes directly. A call through a fn value takes the call graph's
//! targets; a node of a function that has one but no such edge, or any such call when a function
//! made into a fn value has no node, is unresolved, and so is one that calls C code (`ccall`, or
//! an `extern "C"` import).
//!
//! **Depth** of a node: its weight plus the largest depth among its successors, and, if its code
//! has a trap stub, the depth `T` of a trap: twice the depth of `runtime.panic.trap`. A trap
//! raised while one is reported calls `runtime.panic.begin` first, which exits, so it adds at
//! most one more such path; that needs every function `begin` reaches to be free of trap stubs,
//! which the pass checks (else every trapping stack is unbounded).
//! Computed sinks first by out-degree peeling, on flat arrays without recursion. A node that is
//! unresolved, on a cycle, or reaches either has no bound: a call cycle (named by its `rec fn`),
//! a drop cycle of a recursive type, or one the added edges close. A `rec fn` on no cycle of
//! this graph is bounded like any fn.
//!
//! **Stacks.** Main's stack holds the `__stave_main` stub and `main`. A thread runs the
//! `__stave_thread_start` stub, `runtime.thread.entry`, and the fn value it was started with; each
//! function a call-graph [`EdgeKind::Spawn`] edge reaches (`spawn` bodies, the closures `std.thread`
//! starts, every fn value `runtime.thread.start` can be given) gets a row keyed by its code
//! address. The fn-value call of `runtime.thread.run_queued` is left out: it runs a queued value
//! at a join only in an image without a stack table (the IR interpreter; a native image fails to
//! create the thread instead, so a spawn body never runs on the joiner's bounded stack). The table holds the bound plus [`MARGIN`] for bounded stacks and 0 (the `[run] stack`
//! default) for the others; the runtime rounds up to the page size. No error comes from the bound: frame sizes differ
//! between debug and release.

use stave_ir::{FuncId, Module, Op, Operand, Term};
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
}

/// The bound of every stack of a build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    /// `[run] stack` of the manifest, if it sets one.
    pub run_stack: Option<u64>,
    pub main: Depth,
    /// By function.
    pub threads: Vec<Thread>,
}

impl Report {
    /// `[run] stack` in effect.
    pub fn run_stack(&self) -> u64 {
        self.run_stack.unwrap_or(DEFAULT_RUN_STACK)
    }

    /// The stack bytes a table row holds: the bound plus [`MARGIN`], or 0 (the default applies).
    pub fn bytes(d: Depth) -> u64 {
        d.map_or(0, |b| b.saturating_add(MARGIN))
    }

    /// `__stave_stack_table`, with `syms` the symbol of each function.
    pub fn table(&self, syms: &[Option<SymbolId>]) -> DataObj {
        let rows: Vec<(SymbolId, u64)> = self
            .threads
            .iter()
            .filter(|t| t.depth.is_ok())
            .filter_map(|t| syms.get(t.func.index()).copied().flatten().map(|s| (s, Report::bytes(t.depth))))
            .collect();
        stave_obj::stub::stack_table(self.run_stack.unwrap_or(0), Report::bytes(self.main), &rows)
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

fn find_body(m: &Module, path: &str) -> Option<FuncId> {
    m.funcs.iter().position(|f| f.path == path && f.body.is_some()).map(|i| FuncId(i as u32))
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

/// The bound of every stack of module `m` (the module code generation compiled: release builds
/// after the optimizer), with `uses` the backend's report per function.
pub fn compute(m: &Module, input: &Input<'_>, uses: &[StackUse], main: FuncId) -> Report {
    let g = input.graph;
    let n = g.nodes.len();
    let nf = m.funcs.len();
    let has_body = |f: usize| m.funcs[f].body.is_some();
    // the nodes of each function: its call-graph nodes, else one node of its own
    let mut ids: Vec<Vec<u32>> = Vec::with_capacity(nf);
    let mut funcs_of: Vec<Vec<FuncId>> = vec![Vec::new(); n];
    let mut total = n;
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
            v.push(total as u32);
            total += 1;
        }
        for &x in &v {
            if let Some(fs) = funcs_of.get_mut(x as usize) {
                fs.push(FuncId(f as u32));
            }
        }
        ids.push(v);
    }
    let mut pseudo_func = vec![FuncId(0); total - n];
    for (f, v) in ids.iter().enumerate() {
        for &x in v {
            if x as usize >= n {
                pseudo_func[x as usize - n] = FuncId(f as u32);
            }
        }
    }
    let at = |x: u32| if (x as usize) < n { At::Graph(x) } else { At::Func(pseudo_func[x as usize - n]) };
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
    let queued = find_body(m, "runtime.thread.run_queued");
    let called = IntrinsicCalls {
        cpu_init: find_body(m, "runtime.cpu.init"),
        copy_raw: find_body(m, "std.slice.copy_raw"),
    };
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

    let finish = |done: &[Option<Depth>], x: u32| -> Depth {
        match done[x as usize] {
            Some(d) => d,
            None => {
                // a cycle through a `rec fn` is reported as that fn
                let cycle = on_cycle(&start, &to, done, x);
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
    let first = longest(&start, &to, &weight, &zero, &poison);
    let trap = match (find_body(m, "runtime.panic.trap"), find_body(m, "runtime.panic.begin")) {
        (Some(t), Some(b)) => match trapping_node(&start, &to, &traps, &ids[b.index()]) {
            Some(x) => Err(Why::Cycle(at(x))),
            None => over(&first, &ids[t.index()]).map(|d| d.saturating_mul(2)),
        },
        (Some(t), None) => Err(Why::Cycle(at(ids[t.index()].first().copied().unwrap_or(0)))),
        // `__stave_panic_fallback`: called, no frame
        (None, _) => Ok(16),
    };
    let leaf: Vec<Depth> = traps.iter().map(|&t| if t { trap } else { Ok(0) }).collect();
    let done = longest(&start, &to, &weight, &leaf, &poison);

    let incomplete = g.truncated;
    let main_d =
        if incomplete { Err(Why::Incomplete) } else { add_d(MAIN_STUB, over(&done, &ids[main.index()])) };

    // what can run: reachable from main over the calls and the spawn edges (a call-graph node
    // exists for every non-generic fn, called or not)
    let mut live = vec![false; total];
    let mut work: Vec<u32> = ids[main.index()].clone();
    while let Some(x) = work.pop() {
        if std::mem::replace(&mut live[x as usize], true) {
            continue;
        }
        work.extend_from_slice(&to[start[x as usize] as usize..start[x as usize + 1] as usize]);
        if (x as usize) < n {
            work.extend(g.succ(x).iter().filter(|e| e.kind == EdgeKind::Spawn).map(|e| e.to));
        }
    }

    // threads: the functions live spawn edges reach (a node of `runtime.thread.start` has one to
    // each fn value it can be given)
    let mut rows: Vec<(FuncId, u32)> = Vec::new();
    let add_row = |x: u32, rows: &mut Vec<(FuncId, u32)>| {
        for &f in &funcs_of[x as usize] {
            rows.push((f, x));
        }
    };
    for x in (0..n as u32).filter(|&x| live[x as usize]) {
        for e in g.succ(x) {
            if e.kind == EdgeKind::Spawn {
                add_row(e.to, &mut rows);
            }
        }
    }
    rows.sort_unstable();
    rows.dedup();
    let mut threads: Vec<Thread> = Vec::new();
    for (f, s) in rows {
        match threads.last_mut() {
            Some(t) if t.func == f => t.starts.push(s),
            _ => threads.push(Thread { func: f, starts: vec![s], depth: Ok(0) }),
        }
    }
    let entry = find_body(m, "runtime.thread.entry");
    for t in &mut threads {
        // the nodes a thread can start with, not every node of the function: a node no start
        // reaches (an unspecialized one whose fn-value slots stay empty) never runs on a thread
        let body = over(&done, &t.starts);
        t.depth = match entry {
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
                                    best = max_d(best, finish(&done, to[k as usize]));
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
        };
    }
    Report { run_stack: input.run_stack, main: main_d, threads }
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
    let line = |what: String, d: Depth| match d {
        Ok(b) => format!("stack: {what}: {b} bytes"),
        Err(w) => {
            let why = match w {
                Why::Rec(a) => format!("reaches rec fn `{}`", name(a)),
                Why::Cycle(a) => format!("reaches a call cycle through `{}`", name(a)),
                Why::FnValue(a) => format!("calls a fn value with unknown targets in `{}`", name(a)),
                Why::Foreign(a) => format!("calls C code in `{}`", name(a)),
                Why::Incomplete => "the call graph is incomplete".to_string(),
            };
            format!("stack: {what}: unbounded ({why}), [run] stack {} bytes", r.run_stack())
        }
    };
    let mut out = vec![line("main".to_string(), r.main)];
    for t in &r.threads {
        let what = match t.starts.iter().find(|&&x| is_spawn(g, x)) {
            Some(&x) => format!("spawn at {}", locate(g.nodes[x as usize].span)),
            None => match t.starts.first() {
                Some(&x) => format!("thread `{}`", name(At::Graph(x))),
                None => format!("thread `{}`", name(At::Func(t.func))),
            },
        };
        out.push(line(what, t.depth));
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
        assert_eq!(Report::bytes(Ok(100)), 100 + MARGIN);
        assert_eq!(Report::bytes(Err(Why::Incomplete)), 0);
    }
}
