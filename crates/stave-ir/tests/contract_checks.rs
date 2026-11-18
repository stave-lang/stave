//! Contract checks in the check-removal pass and the witness checker, on hand-made functions:
//! what the pass proves, refutes, or leaves alone, what the checker accepts, how removal changes
//! the body, and one forged witness per way of being wrong.

mod common;

use common::forge::{body, dom_table, inst_of, insts_with};
use common::*;
use stave_ir::bounds::{Clause, Proof, prove_func};
use stave_ir::interp::Val;
use stave_ir::ir::Span;
use stave_ir::witness::check::{Reject, check_func, remove};
use stave_ir::witness::{Edge, FnWitness, Node, REQUIRES_FAILED, Refuted, Removed, Source};
use stave_ir::{
    Block, FuncBuilder, FuncId, Module, Op, Operand, Outcome, PanicKind, Site, TypeId, Value, verify_module,
};

/// Branches to a call of `fail` unless `c`, as lowering checks one operand of a clause; the
/// builder is left in the block past the check, which is returned.
fn require(b: &mut FuncBuilder, fail: FuncId, c: Value) -> Block {
    let (bad, ok) = (b.create_block(), b.create_block());
    b.cbr(c, ok, &[], bad, &[]);
    b.switch_to(bad);
    b.call(fail, None, &[]);
    b.unreachable();
    b.switch_to(ok);
    ok
}

/// `app.f(x: T, y: T) -> T` whose body `build` makes (given `x`, `y`, the bool type, and the
/// failure fn) before it returns `x`, in a module with a `runtime.panic.requires_failed` that
/// panics `Contract`.
fn contract_fn<R>(
    ty: impl FnOnce(&mut Module) -> TypeId,
    build: impl FnOnce(&mut FuncBuilder, Value, Value, TypeId, FuncId) -> R,
) -> (Module, FuncId, R) {
    let mut m = Module::new();
    let t = ty(&mut m);
    let bt = m.types.bool();
    let fail = define(&mut m, &[9], REQUIRES_FAILED, &[], None, |b, _| {
        b.inst(Op::Panic(PanicKind::Contract), None, &[]);
        b.unreachable();
    });
    let mut out = None;
    let f = define(&mut m, &[1], "app.f", &[t, t], Some(t), |b, _| {
        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
        out = Some(build(b, x, y, bt, fail));
        b.ret(Some(x.into()));
    });
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    (m, f, out.unwrap())
}

/// The failure calls of `f`, in order.
fn calls(m: &Module, f: FuncId) -> Vec<u32> {
    insts_with(body(m, f), Op::Call)
}

fn edge(from: Node, to: Node, w: i128, src: Source) -> Edge {
    Edge { from, to, w, src, mul: 1 }
}

/// A witness of `f` with the true dominance table, `removed` rows `(check, edges, also)` and
/// `refuted` rows `(check, edges, half)`.
fn witness(
    m: &Module,
    f: FuncId,
    removed: &[(u32, Vec<Edge>, Vec<Edge>)],
    refuted: &[(u32, Vec<Edge>, u8)],
) -> FnWitness {
    let mut w = FnWitness {
        func: f,
        dom: dom_table(body(m, f)),
        removed: Vec::new(),
        refuted: Vec::new(),
        edges: Vec::new(),
    };
    let span = |w: &mut FnWitness, es: &[Edge]| {
        let s = Span { start: w.edges.len() as u32, len: es.len() as u32 };
        w.edges.extend_from_slice(es);
        s
    };
    for (inst, edges, also) in removed {
        let (edges, also) = (span(&mut w, edges), span(&mut w, also));
        w.removed.push(Removed { inst: *inst, site: Site::NONE, edges, also });
    }
    for (inst, edges, half) in refuted {
        let edges = span(&mut w, edges);
        w.refuted.push(Refuted { inst: *inst, site: Site::NONE, edges, half: *half });
    }
    w
}

fn why(m: &Module, w: &FnWitness) -> Reject {
    match check_func(m, w) {
        Ok(c) => panic!("accepted: {:?} / {:?}\n{w:#?}", c.insts(), c.refuted()),
        Err(r) => r.why,
    }
}

fn x_of(m: &Module, f: FuncId) -> Node {
    Node::Val(body(m, f).params(Block(0))[0])
}

/// `requires(x < 10) requires(x < 20) requires(x > 30) requires(x > 2)`: the first is left to
/// run, the second follows from it, the third contradicts it, and the fourth holds where the
/// third did (a point no execution reaches: proved wins). Removal turns the proved checks'
/// branches into jumps, drops their calls and blocks, and changes no run.
#[test]
fn proved_refuted_and_kept() {
    let (m, f, oks) = contract_fn(
        |m| m.types.u32(),
        |b, x, _, bt, fail| {
            let mut oks = Vec::new();
            for (op, c) in [(Op::Lt, 10), (Op::Lt, 20), (Op::Gt, 30), (Op::Gt, 2)] {
                let k = b.int(c);
                let v = b.cmp(op, bt, x, k);
                oks.push(require(b, fail, v));
            }
            oks
        },
    );
    let q = calls(&m, f);
    let p = prove_func(&m, f).expect("proofs");
    let w = &p.witness;
    assert_eq!(w.removed.iter().map(|r| r.inst).collect::<Vec<_>>(), [q[1], q[3]]);
    assert_eq!(p.proofs, [Proof::Clause(Clause::Requires); 2]);
    assert_eq!((w.refuted.len(), w.refuted[0].inst, w.refuted[0].half), (1, q[2], 0));
    assert_eq!(p.refuted, [Clause::Requires]);
    let x = x_of(&m, f);
    // x < 20 from x <= 9
    assert_eq!(w.removed[0].edges.of(&w.edges), [edge(x, Node::Zero, 9, Source::Branch { block: oks[0].0 })]);
    let checked = check_func(&m, w).unwrap_or_else(|r| panic!("{r}"));
    assert_eq!((checked.insts(), checked.refuted()), (&[q[1], q[3]][..], &[q[2]][..]));

    let mut o = m.clone();
    let blocks = body(&m, f).blocks.len();
    remove(&mut o, &[checked]);
    verify_module(&o).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(calls(&o, f).len(), 2);
    assert_eq!(body(&o, f).blocks.len(), blocks - 2);
    for x in [0u64, 5, 9, 10, 15, 25, 31, 1 << 31] {
        let args = vec![Val::S(x), Val::S(0)];
        let (want, got) = (run(&m, f, args.clone()), run(&o, f, args));
        assert_eq!(got, want, "x = {x}");
        assert!(matches!(got, Outcome::Trap(t) if t.kind == PanicKind::Contract), "x = {x}");
    }
    // Nothing left to remove: the pass is idempotent on what it removed.
    let again = prove_func(&o, f).unwrap();
    assert!(again.witness.removed.is_empty() && again.witness.refuted.len() == 1);
}

/// Forged witnesses for the checks of [`proved_refuted_and_kept`]: the failing side's own
/// negated test, the check's own true side, a goal a check does not have, a refutation of a
/// check that is also removed, a row that names no contract check, and edges counted wrong.
#[test]
fn forged_contract_witnesses_are_refused() {
    let (m, f, oks) = contract_fn(
        |m| m.types.u32(),
        |b, x, _, bt, fail| {
            let mut oks = Vec::new();
            for (op, c) in [(Op::Lt, 10), (Op::Lt, 20), (Op::Gt, 30)] {
                let k = b.int(c);
                let v = b.cmp(op, bt, x, k);
                oks.push(require(b, fail, v));
            }
            oks
        },
    );
    let q = calls(&m, f);
    let bd = body(&m, f);
    let x = x_of(&m, f);
    let fail0 = bd.blocks.iter().position(|b| b.insts.start == q[0] && b.insts.len == 1).unwrap() as u32;
    let lt10 = |src| vec![edge(x, Node::Zero, 9, src)];
    // the check itself: its failing side states `x >= 10`, its true side `x < 10`; neither holds
    // where the check branches
    let w = witness(&m, f, &[(q[0], lt10(Source::Branch { block: fail0 }), vec![])], &[]);
    assert_eq!(why(&m, &w), Reject::Region);
    let w = witness(&m, f, &[(q[0], lt10(Source::Branch { block: oks[0].0 }), vec![])], &[]);
    assert_eq!(why(&m, &w), Reject::Region);
    let w =
        witness(&m, f, &[], &[(q[0], vec![edge(Node::Zero, x, -10, Source::Branch { block: fail0 })], 0)]);
    assert_eq!(why(&m, &w), Reject::Region);
    // the right proof of the second check, then wrong in one way each
    let good = lt10(Source::Branch { block: oks[0].0 });
    assert!(check_func(&m, &witness(&m, f, &[(q[1], good.clone(), vec![])], &[])).is_ok());
    let twice = vec![Edge { mul: 2, ..good[0] }];
    assert_eq!(why(&m, &witness(&m, f, &[(q[1], twice, vec![])], &[])), Reject::Chain);
    let never = vec![Edge { mul: 0, ..good[0] }];
    assert_eq!(why(&m, &witness(&m, f, &[(q[1], never, vec![])], &[])), Reject::Edge);
    assert_eq!(why(&m, &witness(&m, f, &[(q[1], good.clone(), good.clone())], &[])), Reject::Goal);
    assert_eq!(why(&m, &witness(&m, f, &[(q[1], vec![], vec![])], &[])), Reject::Chain);
    // the refutation of the third check, then wrong in one way each
    let neg = vec![edge(x, Node::Zero, 9, Source::Branch { block: oks[0].0 })];
    assert!(check_func(&m, &witness(&m, f, &[], &[(q[2], neg.clone(), 0)])).is_ok());
    assert_eq!(why(&m, &witness(&m, f, &[], &[(q[2], neg.clone(), 1)])), Reject::Goal);
    assert_eq!(
        why(&m, &witness(&m, f, &[(q[1], good.clone(), vec![])], &[(q[1], neg.clone(), 0)])),
        Reject::NotACheck
    );
    let cmp = inst_of(
        bd,
        match bd.blocks[0].term {
            stave_ir::Term::Cbr { cond: Operand::Value(c), .. } => c,
            _ => unreachable!(),
        },
    );
    assert_eq!(why(&m, &witness(&m, f, &[], &[(cmp, neg.clone(), 0)])), Reject::NotACheck);
    assert_eq!(why(&m, &witness(&m, f, &[(cmp, neg, vec![])], &[])), Reject::NotACheck);
}

/// `requires(x <= 5) requires(x >= 5) requires(x == 5)`: an `==` has two goals, each needs its
/// proof. `requires(x != 3)` has none: no witness removes it.
#[test]
fn equal_has_two_goals_and_not_equal_none() {
    let (m, f, oks) = contract_fn(
        |m| m.types.u32(),
        |b, x, _, bt, fail| {
            let mut oks = Vec::new();
            for (op, c) in [(Op::Le, 5), (Op::Ge, 5), (Op::Eq, 5), (Op::Ne, 3)] {
                let k = b.int(c);
                let v = b.cmp(op, bt, x, k);
                oks.push(require(b, fail, v));
            }
            oks
        },
    );
    let q = calls(&m, f);
    let x = x_of(&m, f);
    let p = prove_func(&m, f).unwrap();
    // `x == 5` proved by both halves; `x != 3` holds for `x = 5`, but the compare is no
    // inequality
    assert_eq!(p.witness.removed.iter().map(|r| r.inst).collect::<Vec<_>>(), [q[2]]);
    check_func(&m, &p.witness).unwrap_or_else(|r| panic!("{r}"));
    let le = vec![edge(x, Node::Zero, 5, Source::Branch { block: oks[0].0 })];
    let ge = vec![edge(Node::Zero, x, -5, Source::Branch { block: oks[1].0 })];
    assert!(check_func(&m, &witness(&m, f, &[(q[2], le.clone(), ge.clone())], &[])).is_ok());
    assert_eq!(why(&m, &witness(&m, f, &[(q[2], le.clone(), vec![])], &[])), Reject::Chain);
    assert_eq!(why(&m, &witness(&m, f, &[(q[2], ge.clone(), le.clone())], &[])), Reject::Chain);
    assert_eq!(why(&m, &witness(&m, f, &[(q[3], le, vec![])], &[])), Reject::Goal);
}

/// Signed values are facts under their own reading: `x < -5` gives `x < 0` and refutes
/// `x > -3`; the same compare on `u32` is about other numbers. Edges with multipliers add up like
/// a chain: `x <= y` twice and `y <= x` once is `x <= y`.
#[test]
fn signed_facts_and_multipliers() {
    let (m, f, oks) = contract_fn(
        |m| m.types.i32(),
        |b, x, y, bt, fail| {
            let mut oks = Vec::new();
            for (op, c) in [(Op::Lt, -5), (Op::Lt, 0)] {
                let k = b.int(c);
                let v = b.cmp(op, bt, x, k);
                oks.push(require(b, fail, v));
            }
            let e = b.cmp(Op::Eq, bt, x, y);
            oks.push(require(b, fail, e));
            let le = b.cmp(Op::Le, bt, x, y);
            oks.push(require(b, fail, le));
            let k = b.int(-3);
            let gt = b.cmp(Op::Gt, bt, x, k);
            require(b, fail, gt);
            oks
        },
    );
    let q = calls(&m, f);
    let bd = body(&m, f);
    let (x, y) = (x_of(&m, f), Node::Val(bd.params(Block(0))[1]));
    let p = prove_func(&m, f).unwrap();
    check_func(&m, &p.witness).unwrap_or_else(|r| panic!("{r}"));
    assert_eq!(p.witness.removed.iter().map(|r| r.inst).collect::<Vec<_>>(), [q[1], q[3]]);
    assert_eq!(p.witness.refuted.iter().map(|r| r.inst).collect::<Vec<_>>(), [q[4]]);
    let lt = edge(x, Node::Zero, -6, Source::Branch { block: oks[0].0 });
    assert!(check_func(&m, &witness(&m, f, &[(q[1], vec![lt], vec![])], &[])).is_ok());
    // read as unsigned, `x < -5` would be a large bound: no such fact
    let unsigned = edge(x, Node::Zero, (1 << 32) - 6, Source::Branch { block: oks[0].0 });
    assert_eq!(why(&m, &witness(&m, f, &[(q[1], vec![unsigned], vec![])], &[])), Reject::Edge);
    // x <= y + 0 from `x == y`: (x - y) * 2 + (y - x) = x - y
    let eq = Source::Branch { block: oks[2].0 };
    let combo = vec![Edge { mul: 2, ..edge(x, y, 0, eq) }, edge(y, x, 0, eq)];
    assert!(check_func(&m, &witness(&m, f, &[(q[3], combo.clone(), vec![])], &[])).is_ok());
    let weak = vec![Edge { mul: 2, ..edge(x, y, 0, eq) }, Edge { mul: 2, ..edge(y, x, 0, eq) }];
    assert_eq!(why(&m, &witness(&m, f, &[(q[3], weak, vec![])], &[])), Reject::Chain);
    let overflow = vec![Edge { w: i128::MAX / 2, mul: 4, ..combo[0] }];
    assert_eq!(why(&m, &witness(&m, f, &[(q[3], overflow, vec![])], &[])), Reject::Edge);
}
