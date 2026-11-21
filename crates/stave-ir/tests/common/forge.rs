//! Wrong witnesses, made from right ones or from nothing: what a faulty check-removal pass
//! could hand the checker. The checker must refuse every one of them.

use stave_ir::bounds::prove_func;
use stave_ir::ir::{Body, Span, ValueDef};
use stave_ir::verify::Dominators;
use stave_ir::witness::check::{check_func, check_module};
use stave_ir::witness::{Edge, FnWitness, NO_BLOCK, Node, Removed, Source};
use stave_ir::{FuncId, Module, Op, Operand, Site, Value};

/// The instruction that defines `v`.
pub fn inst_of(b: &Body, v: Value) -> u32 {
    match b.values[v.index()].def {
        ValueDef::Inst(i) => i,
        ValueDef::Param { .. } => panic!("{v:?} is a block parameter"),
    }
}

/// The instructions of `b` with opcode `op`, in order.
pub fn insts_with(b: &Body, op: Op) -> Vec<u32> {
    (0..b.insts.len() as u32).filter(|&i| b.insts.op[i as usize] == op).collect()
}

pub fn body(m: &Module, f: FuncId) -> &Body {
    m.func(f).and_then(|f| f.body.as_ref()).expect("a body")
}

/// The dominance table of `b` as the pass writes it.
pub fn dom_table(b: &Body) -> Vec<u32> {
    let dom = Dominators::compute(b);
    (0..b.blocks.len()).map(|bk| dom.idom(bk).map_or(NO_BLOCK, |d| d as u32)).collect()
}

/// A witness of `f` with the true dominance table and one row per `(check, chain)`.
pub fn witness(m: &Module, f: FuncId, rows: &[(u32, Vec<Edge>)]) -> FnWitness {
    let mut w = FnWitness {
        func: f,
        dom: dom_table(body(m, f)),
        removed: Vec::new(),
        refuted: Vec::new(),
        edges: Vec::new(),
        summary: Default::default(),
        calls: Default::default(),
    };
    for (inst, chain) in rows {
        let edges = Span { start: w.edges.len() as u32, len: chain.len() as u32 };
        w.edges.extend(chain);
        w.removed.push(Removed { inst: *inst, site: Site::NONE, edges, also: Span::default() });
    }
    w
}

/// The node and offset an unsigned operand denotes (the witness format's reading).
pub fn node(b: &Body, o: Operand) -> (Node, i128) {
    match o {
        Operand::Value(v) => (Node::Val(v), 0),
        _ => (Node::Zero, b.int_const(o).expect("an int constant")),
    }
}

fn refused(m: &Module, w: &FnWitness, what: &str) {
    if let Ok(c) = check_func(m, w) {
        panic!("accepted {what}: checks {:?} of fn {:?}\n{w:#?}", c.insts(), w.func);
    }
}

fn shifted(src: Source) -> Source {
    match src {
        Source::Branch { block } => Source::Branch { block: block + 1 },
        Source::Assert { call, arg } => Source::Assert { call: call + 1, arg },
        Source::Check { inst } => Source::Check { inst: inst + 1 },
        Source::Arith { inst } => Source::Arith { inst: inst + 1 },
        Source::SliceLen { inst } => Source::SliceLen { inst: inst + 1 },
        Source::SliceMake { inst } => Source::SliceMake { inst: inst + 1 },
        Source::SubSat { inst } => Source::SubSat { inst: inst + 1 },
        Source::Range { value } => Source::Range { value: Value(value.0 + 1) },
        Source::Bits { inst } => Source::Bits { inst: inst + 1 },
        Source::Cast { inst } => Source::Cast { inst: inst + 1 },
        Source::Requires { clause } => Source::Requires { clause: clause + 1 },
        Source::Ensures { call, clause } => Source::Ensures { call: call + 1, clause },
        Source::Arg { call, clause } => Source::Arg { call: call + 1, clause },
    }
}

/// The witnesses `w` reads besides its own: those of the functions it calls that state a
/// summary, and what those read in turn.
fn read_by(m: &Module, all: &[FnWitness], w: &FnWitness) -> Vec<FnWitness> {
    let stated = |f: FuncId| {
        all.iter().find(|x| {
            let s = &x.summary;
            x.func == f && !(s.pre.is_empty() && s.entry.is_empty() && s.post.is_empty())
        })
    };
    let mut out: Vec<FnWitness> = Vec::new();
    let (mut work, mut seen) = (vec![w.func], vec![w.func]);
    while let Some(f) = work.pop() {
        let Some(b) = m.func(f).and_then(|f| f.body.as_ref()) else { continue };
        for i in 0..b.insts.len() as u32 {
            if b.insts.op[i as usize] == Op::Call
                && let Some(&Operand::Func(g)) = b.args(i).first()
                && !seen.contains(&g)
            {
                seen.push(g);
                if let Some(x) = stated(g) {
                    out.push(x.clone());
                    work.push(g);
                }
            }
        }
    }
    out
}

/// Every witness the pass writes for `m` is accepted, and every one-step corruption of it is
/// refused: an edge made stronger or weaker, turned around, moved to the next source, or dropped;
/// a row moved to the next instruction; a dominance table that names a block which does not
/// dominate. Returns the number of corrupted witnesses tried.
pub fn mutants_are_refused(m: &Module) -> usize {
    let mut tried = 0;
    let all: Vec<FnWitness> =
        (0..m.funcs.len() as u32).filter_map(|f| prove_func(m, FuncId(f))).map(|p| p.witness).collect();
    // A witness is checked with those it reads: the summaries of the functions it calls.
    let mut mutant = |w: &FnWitness, read: &[FnWitness], what: &str, change: &dyn Fn(&mut FnWitness)| {
        let mut ws = read.to_vec();
        ws.push(w.clone());
        if let Some(x) = ws.last_mut() {
            change(x);
        }
        if check_module(m, &ws).is_ok() {
            panic!("accepted {what}: fn {:?}\n{:#?}", w.func, ws.last());
        }
        tried += 1;
    };
    for f in 0..m.funcs.len() as u32 {
        let Some(p) = prove_func(m, FuncId(f)) else { continue };
        let w = &p.witness;
        assert_eq!(w.removed.len(), p.proofs.len());
        let read = read_by(m, &all, w);
        let read = &read[..];
        let mut own = read.to_vec();
        own.push(w.clone());
        if let Err(r) = check_module(m, &own) {
            panic!("the pass's own witness is refused: {r:?}\n{w:#?}");
        }
        // Every proof of the witness: (row kind, row, which span of it, the span).
        let mut spans: Vec<(u8, usize, u8, Span)> = Vec::new();
        for (ri, r) in w.removed.iter().enumerate() {
            spans.push((0, ri, 0, r.edges));
            if r.also.len > 0 {
                spans.push((0, ri, 1, r.also));
            }
        }
        for (ri, r) in w.refuted.iter().enumerate() {
            spans.push((1, ri, 0, r.edges));
        }
        for &(kind, ri, which, s) in &spans {
            for k in s.indices() {
                let k = k as usize;
                mutant(w, read, "a stronger edge", &|x| x.edges[k].w -= 1);
                mutant(w, read, "a weaker edge", &|x| x.edges[k].w += 1);
                mutant(w, read, "an edge at the next source", &|x| x.edges[k].src = shifted(x.edges[k].src));
                mutant(w, read, "an edge counted no times", &|x| x.edges[k].mul = 0);
                if w.edges[k].from != w.edges[k].to {
                    mutant(w, read, "an edge counted twice", &|x| x.edges[k].mul = 2);
                    mutant(w, read, "an edge turned around", &|x| {
                        let e = x.edges[k];
                        x.edges[k] = Edge { from: e.to, to: e.from, ..e };
                    });
                    mutant(w, read, "a chain without one edge", &|x| {
                        // Other proofs keep their spans: the dropped edge is overwritten by
                        // shifting this proof's later edges down and shortening it.
                        let end = (s.start + s.len) as usize;
                        x.edges.copy_within(k + 1..end, k);
                        match (kind, which) {
                            (0, 0) => x.removed[ri].edges.len -= 1,
                            (0, _) => x.removed[ri].also.len -= 1,
                            _ => x.refuted[ri].edges.len -= 1,
                        }
                    });
                }
            }
        }
        for ri in 0..w.removed.len() {
            mutant(w, read, "a row at the next instruction", &|x| x.removed[ri].inst += 1);
        }
        for ri in 0..w.refuted.len() {
            mutant(w, read, "a refuted row at the next instruction", &|x| x.refuted[ri].inst += 1);
            mutant(w, read, "a refuted check also removed", &|x| {
                let r = x.refuted[ri];
                let at = x.removed.partition_point(|y| y.inst < r.inst);
                x.removed.insert(
                    at,
                    Removed { inst: r.inst, site: r.site, edges: r.edges, also: Span::default() },
                );
            });
        }
        let b = body(m, FuncId(f));
        let dom = Dominators::compute(b);
        mutant(w, read, "a short dominance table", &|x| {
            x.dom.pop();
        });
        for bk in 1..b.blocks.len() {
            if !dom.reachable(bk) {
                continue;
            }
            mutant(w, read, "a block that dominates itself only", &|x| x.dom[bk] = bk as u32);
            mutant(w, read, "a reachable block without a dominator", &|x| x.dom[bk] = NO_BLOCK);
            for other in 0..b.blocks.len() {
                if dom.reachable(other) && !dom.dominates(other, bk) {
                    mutant(w, read, "a parent that does not dominate", &|x| x.dom[bk] = other as u32);
                }
            }
        }
    }
    tried
}

/// The reachable `check.bounds` of `f` the pass keeps.
pub fn kept_checks(m: &Module, f: FuncId) -> Vec<u32> {
    let b = body(m, f);
    let dom = Dominators::compute(b);
    let proved = prove_func(m, f);
    let removed: Vec<u32> = proved.iter().flat_map(|p| p.witness.removed.iter().map(|r| r.inst)).collect();
    b.blocks
        .iter()
        .enumerate()
        .filter(|(bk, _)| dom.reachable(*bk))
        .flat_map(|(_, bd)| bd.insts.indices())
        .filter(|&i| b.insts.op[i as usize] == Op::BoundsCheck && !removed.contains(&i))
        .collect()
}

/// No made-up witness gets a kept check of `f` removed: the empty chain, one edge from the
/// index straight to the length read at every block and every instruction as every kind of
/// source, and the chain of every check the pass did prove. Returns the number tried.
pub fn forgeries_are_refused(m: &Module, f: FuncId) -> usize {
    let b = body(m, f);
    let proved = prove_func(m, f);
    let mut tried = 0;
    for q in kept_checks(m, f) {
        let &[idx, n] = b.args(q) else { panic!("check.bounds takes two operands") };
        let ((from, ki), (to, kn)) = (node(b, idx), node(b, n));
        let mut forge = |chain: Vec<Edge>, what: &str| {
            refused(m, &witness(m, f, &[(q, chain)]), what);
            tried += 1;
        };
        forge(Vec::new(), "an empty chain");
        // Exactly the weight that would prove the check.
        let edge = |src| vec![Edge { from, to, w: kn - ki - 1, src, mul: 1 }];
        for bk in 0..b.blocks.len() as u32 {
            forge(edge(Source::Branch { block: bk }), "a branch that proves a kept check");
        }
        for inst in 0..b.insts.len() as u32 {
            for src in [
                Source::Check { inst },
                Source::Arith { inst },
                Source::SliceLen { inst },
                Source::SliceMake { inst },
                Source::SubSat { inst },
                Source::Bits { inst },
                Source::Cast { inst },
                Source::Assert { call: inst, arg: 0 },
                Source::Assert { call: inst, arg: 1 },
            ] {
                forge(edge(src), "an instruction that proves a kept check");
            }
        }
        for v in 0..b.values.len() as u32 {
            forge(edge(Source::Range { value: Value(v) }), "a type range that proves a kept check");
        }
        for r in proved.iter().flat_map(|p| p.witness.removed.iter().map(move |r| (p, r))) {
            forge(r.1.edges.of(&r.0.witness.edges).to_vec(), "another check's chain");
        }
    }
    tried
}

/// A goal `a <= b + w` as `(a, b, w)`.
pub type Goal = (Node, Node, i128);

/// The contract checks of `f`: per failure call (alone in a block that only the false edge of a
/// `cbr` enters), the condition's compare as its first goal, `None` for `!=` or a compare that is
/// no int compare.
pub fn contract_checks(m: &Module, f: FuncId) -> Vec<(u32, Option<Goal>)> {
    let b = body(m, f);
    let dom = Dominators::compute(b);
    let mut out = Vec::new();
    for (fb, bd) in b.blocks.iter().enumerate() {
        let q = bd.insts.start;
        if fb == 0 || !dom.reachable(fb) || bd.insts.len != 1 || b.insts.op[q as usize] != Op::Call {
            continue;
        }
        let Some(&Operand::Func(callee)) = b.args(q).first() else { continue };
        if !stave_ir::known::is_contract_failed(&m.funcs[callee.index()]) {
            continue;
        }
        let branch = b.blocks.iter().find_map(|p| match p.term {
            stave_ir::Term::Cbr { cond, els, .. } if els.block.index() == fb => Some(cond),
            _ => None,
        });
        let Some(mut cond) = branch else { continue };
        let mut holds = true;
        let goal = loop {
            let Operand::Value(v) = cond else { break None };
            let ValueDef::Inst(d) = b.values[v.index()].def else { break None };
            match (b.insts.op[d as usize], b.args(d)) {
                (Op::Not, &[x]) => {
                    cond = x;
                    holds = !holds;
                }
                (op, &[x, y]) => {
                    let int = |o: Operand| match o {
                        Operand::Value(v) => matches!(
                            b.value_ty(v).and_then(|t| m.types.key(t)),
                            Some(stave_ir::TypeKey::Int { .. })
                        ),
                        _ => b.int_const(o).is_some(),
                    };
                    if !int(x) || !int(y) {
                        break None;
                    }
                    let ((nx, kx), (ny, ky)) = (node(b, x), node(b, y));
                    let op = match (op, holds) {
                        (op, true) => op,
                        (Op::Lt, false) => Op::Ge,
                        (Op::Le, false) => Op::Gt,
                        (Op::Gt, false) => Op::Le,
                        (Op::Ge, false) => Op::Lt,
                        (Op::Eq, false) => Op::Ne,
                        (_, false) => Op::Eq,
                    };
                    break match op {
                        Op::Lt => Some((nx, ny, ky - kx - 1)),
                        Op::Le | Op::Eq => Some((nx, ny, ky - kx)),
                        Op::Gt => Some((ny, nx, kx - ky - 1)),
                        Op::Ge => Some((ny, nx, kx - ky)),
                        _ => None,
                    };
                }
                _ => break None,
            }
        };
        out.push((q, goal));
    }
    out.sort_unstable_by_key(|c| c.0);
    out
}

/// No made-up witness removes or refutes a contract check of `f` the pass leaves alone: for each
/// such check with an inequality goal, the empty proof, and one edge for the goal and one for
/// its negation, each with exactly the weight needed, read at every block, instruction, and value
/// as every kind of source, as a removal and as a refutation. Returns the number tried.
pub fn contract_forgeries_are_refused(m: &Module, f: FuncId) -> usize {
    let b = body(m, f);
    let proved = prove_func(m, f);
    let decided: Vec<u32> = proved
        .iter()
        .flat_map(|p| {
            p.witness.removed.iter().map(|r| r.inst).chain(p.witness.refuted.iter().map(|r| r.inst))
        })
        .collect();
    let mut tried = 0;
    for (q, goal) in contract_checks(m, f) {
        let Some((a, bn, w)) = goal else { continue };
        if decided.contains(&q) {
            continue;
        }
        let mut forge = |e: Vec<Edge>, what: &str| {
            let mut x = witness(m, f, &[(q, e.clone())]);
            refused(m, &x, what);
            x.removed.clear();
            let edges = Span { start: 0, len: e.len() as u32 };
            x.refuted.push(stave_ir::witness::Refuted { inst: q, site: Site::NONE, edges, half: 0 });
            refused(m, &x, what);
            tried += 2;
        };
        forge(Vec::new(), "an empty proof of a kept contract check");
        let mut srcs: Vec<Source> =
            (0..b.blocks.len() as u32).map(|block| Source::Branch { block }).collect();
        for inst in 0..b.insts.len() as u32 {
            srcs.extend([
                Source::Check { inst },
                Source::Arith { inst },
                Source::SliceLen { inst },
                Source::SliceMake { inst },
                Source::SubSat { inst },
                Source::Bits { inst },
                Source::Cast { inst },
                Source::Assert { call: inst, arg: 0 },
                Source::Assert { call: inst, arg: 1 },
            ]);
        }
        srcs.extend((0..b.values.len() as u32).map(|v| Source::Range { value: Value(v) }));
        for src in srcs {
            forge(
                vec![Edge { from: a, to: bn, w, src, mul: 1 }],
                "one edge that proves a kept contract check",
            );
            forge(vec![Edge { from: bn, to: a, w: -w - 1, src, mul: 1 }], "one edge that refutes it");
        }
    }
    tried
}
