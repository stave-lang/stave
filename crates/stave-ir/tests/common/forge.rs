//! Wrong witnesses, made from right ones or from nothing: what a faulty check-removal pass
//! could hand the checker. The checker must refuse every one of them.

use stave_ir::bounds::prove_func;
use stave_ir::ir::{Body, Span, ValueDef};
use stave_ir::verify::Dominators;
use stave_ir::witness::check::check_func;
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
    let mut w = FnWitness { func: f, dom: dom_table(body(m, f)), removed: Vec::new(), edges: Vec::new() };
    for (inst, chain) in rows {
        let edges = Span { start: w.edges.len() as u32, len: chain.len() as u32 };
        w.edges.extend(chain);
        w.removed.push(Removed { inst: *inst, site: Site::NONE, edges });
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
    }
}

/// Every witness the pass writes for `m` is accepted, and every one-step corruption of it is
/// refused: an edge made stronger or weaker, turned around, moved to the next source, or dropped;
/// a row moved to the next instruction; a dominance table that names a block which does not
/// dominate. Returns the number of corrupted witnesses tried.
pub fn mutants_are_refused(m: &Module) -> usize {
    let mut tried = 0;
    let mut mutant = |w: &FnWitness, what: &str, change: &dyn Fn(&mut FnWitness)| {
        let mut x = w.clone();
        change(&mut x);
        refused(m, &x, what);
        tried += 1;
    };
    for f in 0..m.funcs.len() as u32 {
        let Some(p) = prove_func(m, FuncId(f)) else { continue };
        let w = &p.witness;
        assert_eq!(w.removed.len(), p.proofs.len());
        if let Err(r) = check_func(m, w) {
            panic!("the pass's own witness is refused: {r}\n{w:#?}");
        }
        for (ri, r) in w.removed.iter().enumerate() {
            for k in r.edges.indices() {
                let k = k as usize;
                mutant(w, "a stronger edge", &|x| x.edges[k].w -= 1);
                mutant(w, "a weaker edge", &|x| x.edges[k].w += 1);
                mutant(w, "an edge at the next source", &|x| x.edges[k].src = shifted(x.edges[k].src));
                if w.edges[k].from != w.edges[k].to {
                    mutant(w, "an edge turned around", &|x| {
                        let e = x.edges[k];
                        x.edges[k] = Edge { from: e.to, to: e.from, ..e };
                    });
                    mutant(w, "a chain without one edge", &|x| {
                        // Rows after `ri` keep their spans: the dropped edge is overwritten by
                        // shifting this row's later edges down and shortening the row.
                        let end = (r.edges.start + r.edges.len) as usize;
                        x.edges.copy_within(k + 1..end, k);
                        x.removed[ri].edges.len -= 1;
                    });
                }
            }
            mutant(w, "a row at the next instruction", &|x| x.removed[ri].inst += 1);
        }
        let b = body(m, FuncId(f));
        let dom = Dominators::compute(b);
        mutant(w, "a short dominance table", &|x| {
            x.dom.pop();
        });
        for bk in 1..b.blocks.len() {
            if !dom.reachable(bk) {
                continue;
            }
            mutant(w, "a block that dominates itself only", &|x| x.dom[bk] = bk as u32);
            mutant(w, "a reachable block without a dominator", &|x| x.dom[bk] = NO_BLOCK);
            for other in 0..b.blocks.len() {
                if dom.reachable(other) && !dom.dominates(other, bk) {
                    mutant(w, "a parent that does not dominate", &|x| x.dom[bk] = other as u32);
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
        let edge = |src| vec![Edge { from, to, w: kn - ki - 1, src }];
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
