//! The witness checker (`stave_ir::witness::check`) on hand-made functions: what it accepts,
//! and one forged witness per way of being wrong, each a proof a faulty check-removal pass
//! could write. Every class of refusal ([`Reject`]) is built at least once.

mod common;

use common::forge::{body, inst_of, insts_with, witness};
use common::*;
use stave_ir::bounds::{self, Proof, prove_func};
use stave_ir::interp::Val;
use stave_ir::witness::check::{Reject, check_func, check_module, remove};
use stave_ir::witness::{ASSERT_BLOCKS, Edge, FnWitness, NO_BLOCK, Node, Source};
use stave_ir::{
    Block, CastKind, FuncBuilder, FuncId, Module, Op, Operand, Outcome, PanicKind, TypeId, Value,
    verify_module,
};

struct Ty {
    u32t: TypeId,
    us: TypeId,
    i64t: TypeId,
    bt: TypeId,
    psl: TypeId,
    pe: TypeId,
}

/// The parameters of `app.f(a: &[u32], b: &[u32], i: usize, j: usize) -> u32`.
struct Params {
    a: Value,
    b: Value,
    i: Value,
    j: Value,
}

/// A module with `app.f` built by `build` (which ends every block it opens), plus what `build`
/// returns: the values and blocks the test names in its witnesses.
fn fixture<R>(build: impl FnOnce(&mut FuncBuilder, &mut Module, &Ty, &Params) -> R) -> (Module, FuncId, R) {
    let mut m = Module::new();
    let (u32t, us, i64t, bt) = (m.types.u32(), m.types.usize(), m.types.i64(), m.types.bool());
    let sl = m.types.slice(u32t);
    let t = Ty { u32t, us, i64t, bt, psl: m.types.ref_(false, sl), pe: m.types.ref_(false, u32t) };
    let mut out = None;
    let f = define(&mut m, &[1], "app.f", &[t.psl, t.psl, us, us], Some(u32t), |b, m| {
        let v = |k| b.param(k).unwrap();
        let p = Params { a: v(0), b: v(1), i: v(2), j: v(3) };
        out = Some(build(b, m, &t, &p));
    });
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    (m, f, out.unwrap())
}

/// `s[idx]` as the front end lowers it: `check.bounds idx n`, `addr.elem s idx`, `load`.
fn read(b: &mut FuncBuilder, t: &Ty, s: Value, n: impl Into<Operand>, idx: impl Into<Operand>) -> Value {
    let idx = idx.into();
    b.bounds_check(idx, n);
    let p = b.op(Op::ElemAddr, t.pe, &[s.into(), idx]);
    b.load(t.u32t, p)
}

fn ret(b: &mut FuncBuilder, v: Value) {
    b.ret(Some(v.into()));
}

fn ret0(b: &mut FuncBuilder) {
    let z = b.int(0);
    b.ret(Some(z));
}

/// `cbr c then() els()` into two new blocks; the builder is left in `then`.
fn branch(b: &mut FuncBuilder, c: Value) -> (Block, Block) {
    let (then, els) = (b.create_block(), b.create_block());
    b.cbr(c, then, &[], els, &[]);
    b.switch_to(then);
    (then, els)
}

fn val(v: Value) -> Node {
    Node::Val(v)
}

fn edge(from: Node, to: Node, w: i128, src: Source) -> Edge {
    Edge { from, to, w, src }
}

/// The only `check.bounds` of `f`.
fn the_check(m: &Module, f: FuncId) -> u32 {
    let checks = insts_with(body(m, f), Op::BoundsCheck);
    assert_eq!(checks.len(), 1, "{checks:?}");
    checks[0]
}

fn why(m: &Module, w: &FnWitness) -> Reject {
    match check_func(m, w) {
        Ok(c) => panic!("accepted: {:?}\n{w:#?}", c.insts()),
        Err(r) => {
            assert!(!r.to_string().is_empty());
            r.why
        }
    }
}

/// The checker's verdict on the one-row witness `chain` for the only check of `f`.
fn verdict(m: &Module, f: FuncId, chain: Vec<Edge>) -> Result<(), Reject> {
    check_func(m, &witness(m, f, &[(the_check(m, f), chain)])).map(|_| ()).map_err(|r| r.why)
}

/// What the pass itself proves for the only check of `f`, with the chain it wrote.
fn proved(m: &Module, f: FuncId) -> Option<(Proof, Vec<Edge>)> {
    let p = prove_func(m, f)?;
    assert_eq!(p.witness.removed.len(), 1);
    check_func(m, &p.witness).unwrap_or_else(|r| panic!("the pass's witness is refused: {r}"));
    Some((p.proofs[0], p.witness.removed[0].edges.of(&p.witness.edges).to_vec()))
}

/// `if i < a.len() { return a[i]; } return 0;`: the witness is the branch, one edge.
#[test]
fn a_dominating_test_is_a_one_edge_chain() {
    let (m, f, (n, then)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, p.i, n);
        let (then, els) = branch(b, c);
        let x = read(b, t, p.a, n, p.i);
        ret(b, x);
        b.switch_to(els);
        ret0(b);
        (n, then)
    });
    let i = body(&m, f).params(Block(0))[2];
    let want = vec![edge(val(i), val(n), -1, Source::Branch { block: then.0 })];
    let (proof, chain) = proved(&m, f).expect("a proof");
    assert!(matches!(proof, Proof::LoopBound { .. }), "{proof:?}");
    assert_eq!(chain, want);
    assert_eq!(verdict(&m, f, want), Ok(()));
    // The pass's table names a dominator per block: the entry for both branch targets.
    let w = prove_func(&m, f).unwrap().witness;
    assert_eq!(w.dom, [0, 0, 0]);

    // Removal deletes that check and nothing else; the program runs as before.
    let checked = check_module(&m, std::slice::from_ref(&w)).unwrap();
    assert_eq!((checked[0].func(), checked[0].insts()), (f, &[the_check(&m, f)][..]));
    let before = body(&m, f).insts.len();
    let mut o = m.clone();
    remove(&mut o, &checked);
    verify_module(&o).unwrap();
    assert_eq!(body(&o, f).insts.len(), before - 1);
    assert!(insts_with(body(&o, f), Op::BoundsCheck).is_empty());
    bounds::audit(&o).unwrap();
}

/// The same test, but the access sits on the false side, after the join, or behind `<=`: the
/// edge `i <= n - 1` is not what the branch states there.
#[test]
fn a_test_on_the_wrong_side_or_too_weak_is_refused() {
    // false side
    let (m, f, (n, els)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, p.i, n);
        let (_, els) = branch(b, c);
        ret0(b);
        b.switch_to(els);
        let x = read(b, t, p.a, n, p.i);
        ret(b, x);
        (n, els)
    });
    let i = body(&m, f).params(Block(0))[2];
    assert!(proved(&m, f).is_none());
    assert_eq!(
        verdict(&m, f, vec![edge(val(i), val(n), -1, Source::Branch { block: els.0 })]),
        Err(Reject::Edge)
    );
    // what that side does state proves nothing about the index
    assert_eq!(
        verdict(&m, f, vec![edge(val(n), val(i), 0, Source::Branch { block: els.0 })]),
        Err(Reject::Chain)
    );

    // `<=`
    let (m, f, (n, then)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Le, t.bt, p.i, n);
        let (then, els) = branch(b, c);
        let x = read(b, t, p.a, n, p.i);
        ret(b, x);
        b.switch_to(els);
        ret0(b);
        (n, then)
    });
    assert!(proved(&m, f).is_none());
    let src = Source::Branch { block: then.0 };
    assert_eq!(verdict(&m, f, vec![edge(val(i), val(n), -1, src)]), Err(Reject::Edge));
    assert_eq!(verdict(&m, f, vec![edge(val(i), val(n), 0, src)]), Err(Reject::Weak));
}

/// A branch proves nothing after the join of its two sides, and a block with two ways in is no
/// source at all; nor is the entry, nor a loop head (entered from before the loop and from its
/// end).
#[test]
fn a_test_that_does_not_dominate_is_refused() {
    let (m, f, (n, then, join)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, p.i, n);
        let (then, els) = branch(b, c);
        let join = b.create_block();
        b.br(join, &[]);
        b.switch_to(els);
        b.br(join, &[]);
        b.switch_to(join);
        let x = read(b, t, p.a, n, p.i);
        ret(b, x);
        (n, then, join)
    });
    let i = body(&m, f).params(Block(0))[2];
    assert!(proved(&m, f).is_none());
    let lt = |block: u32| vec![edge(val(i), val(n), -1, Source::Branch { block })];
    assert_eq!(verdict(&m, f, lt(then.0)), Err(Reject::Region));
    assert_eq!(verdict(&m, f, lt(join.0)), Err(Reject::Source));
    assert_eq!(verdict(&m, f, lt(0)), Err(Reject::Source));
    assert_eq!(verdict(&m, f, lt(99)), Err(Reject::Source));
    // A dominance table that puts the join under the true side would make the branch reach it:
    // the edge from the false side into the join gives the table away.
    let mut w = witness(&m, f, &[(the_check(&m, f), lt(then.0))]);
    assert_eq!(w.dom[join.index()], 0);
    w.dom[join.index()] = then.0;
    assert_eq!(why(&m, &w), Reject::Cfg);

    // `while i < n { a[i]; i += 1 }`, the access citing the head instead of the body
    let (m, f, (n, head, i)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let (head, hv) = b.create_block_with(&[t.us]);
        b.br(head, &[p.i.into()]);
        b.switch_to(head);
        let c = b.cmp(Op::Lt, t.bt, hv[0], n);
        let (_, exit) = branch(b, c);
        let _ = read(b, t, p.a, n, hv[0]);
        let one = b.int(1);
        let next = b.binary(Op::AddWrap, t.us, hv[0], one);
        b.br(head, &[next.into()]);
        b.switch_to(exit);
        ret0(b);
        (n, head, hv[0])
    });
    assert!(matches!(proved(&m, f), Some((Proof::LoopBound { .. }, _))));
    let at_head = vec![edge(val(i), val(n), -1, Source::Branch { block: head.0 })];
    assert_eq!(verdict(&m, f, at_head), Err(Reject::Source));
}

/// `not` flips the side a test holds on; the checker follows it.
#[test]
fn a_negated_test_holds_on_the_false_side() {
    let (m, f, (n, els)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, p.i, n);
        let nc = b.unary(Op::Not, t.bt, c);
        let (_, els) = branch(b, nc);
        ret0(b);
        b.switch_to(els);
        let x = read(b, t, p.a, n, p.i);
        ret(b, x);
        (n, els)
    });
    let i = body(&m, f).params(Block(0))[2];
    let (proof, chain) = proved(&m, f).expect("a proof");
    assert!(matches!(proof, Proof::Offset { k: 0, .. }), "{proof:?}");
    assert_eq!(chain.iter().map(|e| e.w).sum::<i128>(), -1);
    assert_eq!(verdict(&m, f, vec![edge(val(i), val(n), -1, Source::Branch { block: els.0 })]), Ok(()));
}

/// A signed compare says nothing about unsigned order: `-1 < n` as signed, far above `n` as an
/// index.
#[test]
fn a_signed_test_is_refused() {
    let (m, f, (n, then)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let si = b.cast(CastKind::Wrap, t.i64t, p.i);
        let sn = b.cast(CastKind::Wrap, t.i64t, n);
        let c = b.cmp(Op::Lt, t.bt, si, sn);
        let (then, els) = branch(b, c);
        let x = read(b, t, p.a, n, p.i);
        ret(b, x);
        b.switch_to(els);
        ret0(b);
        (n, then)
    });
    let i = body(&m, f).params(Block(0))[2];
    assert!(proved(&m, f).is_none());
    let forged = vec![edge(val(i), val(n), -1, Source::Branch { block: then.0 })];
    assert_eq!(verdict(&m, f, forged), Err(Reject::Operand));
}

/// `if i + 1 < n { a[i] }`: with a trapping `add` the sum is exact and the access is proven;
/// with `add.wrap` it may have wrapped to 0, and the same chain is refused.
#[test]
fn only_trapping_arithmetic_is_exact() {
    for wrapping in [false, true] {
        let (m, f, (n, i1, then)) = fixture(|b, _, t, p| {
            let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
            let one = b.int(1);
            let i1 = b.binary(if wrapping { Op::AddWrap } else { Op::Add }, t.us, p.i, one);
            let c = b.cmp(Op::Lt, t.bt, i1, n);
            let (then, els) = branch(b, c);
            let x = read(b, t, p.a, n, p.i);
            ret(b, x);
            b.switch_to(els);
            ret0(b);
            (n, i1, then)
        });
        let i = body(&m, f).params(Block(0))[2];
        let add = Source::Arith { inst: inst_of(body(&m, f), i1) };
        let chain =
            vec![edge(val(i), val(i1), -1, add), edge(val(i1), val(n), -1, Source::Branch { block: then.0 })];
        if wrapping {
            assert!(proved(&m, f).is_none());
            assert_eq!(verdict(&m, f, chain), Err(Reject::Source));
        } else {
            assert!(matches!(proved(&m, f), Some((Proof::Offset { k: 0, .. }, _))));
            assert_eq!(verdict(&m, f, chain), Ok(()));
            // the sum is `i + 1`, not `i + 2` and not `i`
            for w in [-2, 0, 1] {
                let wrong = vec![edge(val(i), val(i1), w, add), chain_tail(i1, n, then)];
                assert_eq!(verdict(&m, f, wrong), Err(Reject::Edge), "weight {w}");
            }
        }
    }
}

fn chain_tail(i1: Value, n: Value, then: Block) -> Edge {
    edge(val(i1), val(n), -1, Source::Branch { block: then.0 })
}

/// `if i < n { a[i + 1] }`: every edge is true, and together they prove `i + 1 <= n`, one short.
#[test]
fn a_chain_one_short_is_refused() {
    let (m, f, (n, i1, then)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, p.i, n);
        let (then, els) = branch(b, c);
        let one = b.int(1);
        let i1 = b.binary(Op::Add, t.us, p.i, one);
        let x = read(b, t, p.a, n, i1);
        ret(b, x);
        b.switch_to(els);
        ret0(b);
        (n, i1, then)
    });
    let i = body(&m, f).params(Block(0))[2];
    assert!(proved(&m, f).is_none());
    let add = Source::Arith { inst: inst_of(body(&m, f), i1) };
    let lt = edge(val(i), val(n), -1, Source::Branch { block: then.0 });
    assert_eq!(verdict(&m, f, vec![edge(val(i1), val(i), 1, add), lt]), Err(Reject::Weak));
    // leaving the `+ 1` out breaks the chain: it must start at the index
    assert_eq!(verdict(&m, f, vec![lt]), Err(Reject::Chain));
    assert_eq!(verdict(&m, f, Vec::new()), Err(Reject::Chain));
}

/// `debug.assert`'s shapes. A callee that returns on both paths asserts nothing, an assertion
/// after the access comes too late, and a large callee is not examined.
#[test]
fn only_an_asserting_call_before_the_check_counts() {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Callee {
        /// `if !c { panic } return`
        NotThenPanic,
        /// `if c { return } panic`
        ElsePanic,
        /// `if !c { return } return`
        Returns,
        /// `NotThenPanic` behind a chain of blocks, more than a callee may have.
        Long,
    }
    let callee = |m: &mut Module, bt: TypeId, kind: Callee| {
        define(m, &[9], "app.check", &[bt], None, |b, _| {
            let c = b.param(0).unwrap();
            if kind == Callee::Long {
                for _ in 0..ASSERT_BLOCKS {
                    let next = b.create_block();
                    b.br(next, &[]);
                    b.switch_to(next);
                }
            }
            let (ok, fail) = if kind == Callee::ElsePanic {
                branch(b, c)
            } else {
                let nc = b.unary(Op::Not, bt, c);
                let (fail, ok) = branch(b, nc);
                (ok, fail)
            };
            b.switch_to(fail);
            if kind == Callee::Returns {
                b.ret(None);
            } else {
                b.panic(PanicKind::Assert);
                b.unreachable();
            }
            b.switch_to(ok);
            b.ret(None);
        })
    };
    for (kind, late) in [
        (Callee::NotThenPanic, false),
        (Callee::ElsePanic, false),
        (Callee::Returns, false),
        (Callee::Long, false),
        (Callee::NotThenPanic, true),
    ] {
        let (m, f, n) = fixture(|b, m, t, p| {
            let check = callee(m, t.bt, kind);
            let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
            let c = b.cmp(Op::Lt, t.bt, p.i, n);
            if !late {
                b.call(check, None, &[c.into()]);
            }
            let x = read(b, t, p.a, n, p.i);
            if late {
                b.call(check, None, &[c.into()]);
            }
            ret(b, x);
            n
        });
        let i = body(&m, f).params(Block(0))[2];
        let call = insts_with(body(&m, f), Op::Call)[0];
        let forged = |arg| vec![edge(val(i), val(n), -1, Source::Assert { call, arg })];
        let want = match (kind, late) {
            (_, true) => Err(Reject::Region),
            (Callee::NotThenPanic | Callee::ElsePanic, _) => Ok(()),
            _ => Err(Reject::Source),
        };
        assert_eq!(verdict(&m, f, forged(0)), want, "{kind:?}, late: {late}");
        assert_eq!(proved(&m, f).is_some(), want.is_ok(), "{kind:?}, late: {late}");
        if want.is_ok() {
            // the callee has one parameter, and the call is the call
            assert_eq!(verdict(&m, f, forged(1)), Err(Reject::Source));
            let not_a_call = Source::Assert { call: call + 1, arg: 0 };
            assert_eq!(verdict(&m, f, vec![edge(val(i), val(n), -1, not_a_call)]), Err(Reject::Source));
        }
    }
}

/// An earlier identical check proves a later one, never the other way round, never itself, and
/// not from the other side of a branch.
#[test]
fn only_an_earlier_check_counts() {
    let (m, f, n) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let x = read(b, t, p.a, n, p.i);
        let y = read(b, t, p.a, n, p.i);
        let s = b.binary(Op::AddWrap, t.u32t, x, y);
        ret(b, s);
        n
    });
    let i = body(&m, f).params(Block(0))[2];
    let checks = insts_with(body(&m, f), Op::BoundsCheck);
    let cite = |inst| vec![edge(val(i), val(n), -1, Source::Check { inst })];
    let p = prove_func(&m, f).expect("the second check is dominated");
    assert_eq!(p.proofs, [Proof::Dominated { first: checks[0] }]);
    assert_eq!(p.witness.removed[0].inst, checks[1]);
    let row = |q: u32, cited: u32| {
        check_func(&m, &witness(&m, f, &[(q, cite(cited))])).map(|_| ()).map_err(|r| r.why)
    };
    assert_eq!(row(checks[1], checks[0]), Ok(()));
    assert_eq!(row(checks[0], checks[1]), Err(Reject::Region));
    assert_eq!(row(checks[0], checks[0]), Err(Reject::Region));
    assert_eq!(row(checks[1], checks[1]), Err(Reject::Region));
    // Two rows citing each other: the first one is refused, so nothing is removed.
    let both = witness(&m, f, &[(checks[0], cite(checks[1])), (checks[1], cite(checks[0]))]);
    assert_eq!(why(&m, &both), Reject::Region);

    // a check on the true side says nothing on the false side
    let (m, f, n) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, p.j, n);
        let (_, els) = branch(b, c);
        let x = read(b, t, p.a, n, p.i);
        ret(b, x);
        b.switch_to(els);
        let y = read(b, t, p.a, n, p.i);
        ret(b, y);
        n
    });
    let checks = insts_with(body(&m, f), Op::BoundsCheck);
    assert!(prove_func(&m, f).is_none());
    let forged =
        witness(&m, f, &[(checks[1], vec![edge(val(i), val(n), -1, Source::Check { inst: checks[0] })])]);
    assert_eq!(why(&m, &forged), Reject::Region);
}

/// The lowering reads a slice's length again at every access. Two reads of one slice are the
/// same length (the chain goes through the slice's length node); reads of two slices are not.
#[test]
fn lengths_are_per_slice_value() {
    for other in [false, true] {
        let (m, f, (n1, n2, then)) = fixture(|b, _, t, p| {
            let n1 = b.op(Op::SliceLen, t.us, &[p.a.into()]);
            let c = b.cmp(Op::Lt, t.bt, p.i, n1);
            let (then, els) = branch(b, c);
            let s = if other { p.b } else { p.a };
            let n2 = b.op(Op::SliceLen, t.us, &[s.into()]);
            let x = read(b, t, s, n2, p.i);
            ret(b, x);
            b.switch_to(els);
            ret0(b);
            (n1, n2, then)
        });
        let bd = body(&m, f);
        let (a, i) = (bd.params(Block(0))[0], bd.params(Block(0))[2]);
        let (r1, r2) =
            (Source::SliceLen { inst: inst_of(bd, n1) }, Source::SliceLen { inst: inst_of(bd, n2) });
        let chain = vec![
            edge(val(i), val(n1), -1, Source::Branch { block: then.0 }),
            edge(val(n1), Node::Len(a), 0, r1),
            edge(Node::Len(a), val(n2), 0, r2),
        ];
        if other {
            assert!(proved(&m, f).is_none());
            // the second read is of `b`: it does not state anything about `Len(a)`
            assert_eq!(verdict(&m, f, chain.clone()), Err(Reject::Edge));
            assert_eq!(verdict(&m, f, chain[..1].to_vec()), Err(Reject::Chain));
        } else {
            let (proof, got) = proved(&m, f).expect("a proof");
            assert!(matches!(proof, Proof::LoopBound { .. }), "{proof:?}");
            assert_eq!(got, chain);
            // a read is an equality, not a strict bound
            let mut tighter = chain.clone();
            tighter[1].w = -1;
            assert_eq!(verdict(&m, f, tighter), Err(Reject::Edge));
        }
    }
}

/// `w = &a[0..a.len() - 1]` after the test `i < a.len()`: the new slice has its own length, one
/// less; claiming it has `a`'s is refused. Its true length does give a proof for `i + 1 < n`.
#[test]
fn a_made_slice_has_the_length_it_was_made_with() {
    for tested_plus_one in [false, true] {
        let (m, f, (n, e, w, nw, i1, then)) = fixture(|b, _, t, p| {
            let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
            let one = b.int(1);
            let i1 = b.binary(Op::Add, t.us, p.i, one);
            let c = b.cmp(Op::Lt, t.bt, if tested_plus_one { i1 } else { p.i }, n);
            let (then, els) = branch(b, c);
            let one = b.int(1);
            let e = b.binary(Op::Sub, t.us, n, one);
            let z = b.int(0);
            let base = b.op(Op::ElemAddr, t.pe, &[p.a.into(), z]);
            let w = b.op(Op::SliceMake, t.psl, &[base.into(), e.into()]);
            let nw = b.op(Op::SliceLen, t.us, &[w.into()]);
            let x = read(b, t, w, nw, p.i);
            ret(b, x);
            b.switch_to(els);
            ret0(b);
            (n, e, w, nw, i1, then)
        });
        let bd = body(&m, f);
        let i = bd.params(Block(0))[2];
        let made = Source::SliceMake { inst: inst_of(bd, w) };
        let read_len = Source::SliceLen { inst: inst_of(bd, nw) };
        let test = Source::Branch { block: then.0 };
        if tested_plus_one {
            let (proof, _) = proved(&m, f).expect("i + 1 < n proves i < n - 1");
            assert!(matches!(proof, Proof::Offset { k: 0, .. }), "{proof:?}");
            let chain = vec![
                edge(val(i), val(i1), -1, Source::Arith { inst: inst_of(bd, i1) }),
                edge(val(i1), val(n), -1, test),
                edge(val(n), val(e), 1, Source::Arith { inst: inst_of(bd, e) }),
                edge(val(e), Node::Len(w), 0, made),
                edge(Node::Len(w), val(nw), 0, read_len),
            ];
            assert_eq!(verdict(&m, f, chain), Ok(()));
        } else {
            assert!(proved(&m, f).is_none());
            let forged = vec![
                edge(val(i), val(n), -1, test),
                edge(val(n), Node::Len(w), 0, made),
                edge(Node::Len(w), val(nw), 0, read_len),
            ];
            assert_eq!(verdict(&m, f, forged), Err(Reject::Edge));
        }
    }
}

/// A vector access of 4 lanes is checked against `lim = n -| 3`. `i < lim` and `i + 4 <= n`
/// prove it; `i <= lim` does not (`0 <= 0` for every short slice), and `lim` is at least
/// `n - 3`, not at most.
#[test]
fn saturating_limits() {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Test {
        LtLim,
        LeLim,
        PlusFourLe,
    }
    for test in [Test::LtLim, Test::LeLim, Test::PlusFourLe] {
        let (m, f, (n, lim, lim2, then)) = fixture(|b, _, t, p| {
            let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
            let three = b.int(3);
            let lim = b.binary(Op::SubSat, t.us, n, three);
            let c = match test {
                Test::LtLim => b.cmp(Op::Lt, t.bt, p.i, lim),
                Test::LeLim => b.cmp(Op::Le, t.bt, p.i, lim),
                Test::PlusFourLe => {
                    let four = b.int(4);
                    let end = b.binary(Op::Add, t.us, p.i, four);
                    b.cmp(Op::Le, t.bt, end, n)
                }
            };
            let (then, els) = branch(b, c);
            let three = b.int(3);
            let lim2 = b.binary(Op::SubSat, t.us, n, three);
            b.bounds_check(p.i, lim2);
            ret0(b);
            b.switch_to(els);
            ret0(b);
            (n, lim, lim2, then)
        });
        let bd = body(&m, f);
        let i = bd.params(Block(0))[2];
        let tested = Source::Branch { block: then.0 };
        let sat2 = Source::SubSat { inst: inst_of(bd, lim2) };
        // i + 3 < n, then n <= lim2 + 3
        let through_n = vec![edge(val(i), val(n), -4, tested), edge(val(n), val(lim2), 3, sat2)];
        match test {
            Test::LtLim => {
                assert!(proved(&m, f).is_some());
                assert_eq!(verdict(&m, f, through_n), Ok(()));
                // the limit bounds `n` from above only
                let backwards = vec![
                    edge(val(i), val(lim), -1, tested),
                    edge(val(lim), val(n), -3, Source::SubSat { inst: inst_of(bd, lim) }),
                    edge(val(n), val(lim2), 3, sat2),
                ];
                assert_eq!(verdict(&m, f, backwards), Err(Reject::Edge));
            }
            Test::LeLim => {
                assert!(proved(&m, f).is_none());
                assert_eq!(verdict(&m, f, through_n), Err(Reject::Edge));
            }
            Test::PlusFourLe => {
                assert!(proved(&m, f).is_some());
                // `i + 4 <= n` is not read through the limit: the branch does not state this edge
                assert_eq!(verdict(&m, f, through_n), Err(Reject::Edge));
            }
        }
    }
}

/// Rows must name reachable `check.bounds` instructions, in order, each once; chains must lie
/// inside the edge table.
#[test]
fn malformed_tables_are_refused() {
    let (m, f, (n, then, dead)) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, p.i, n);
        let (then, els) = branch(b, c);
        let x = read(b, t, p.a, n, p.i);
        let _ = read(b, t, p.a, n, p.i);
        ret(b, x);
        b.switch_to(els);
        ret0(b);
        // no way in
        let dead = b.create_block();
        b.switch_to(dead);
        let y = read(b, t, p.a, n, p.i);
        ret(b, y);
        (n, then, dead)
    });
    let bd = body(&m, f);
    let i = bd.params(Block(0))[2];
    let checks = insts_with(bd, Op::BoundsCheck);
    let chain = || vec![edge(val(i), val(n), -1, Source::Branch { block: then.0 })];
    let rows = |r: &[u32]| witness(&m, f, &r.iter().map(|&q| (q, chain())).collect::<Vec<_>>());
    assert!(check_func(&m, &rows(&checks[..2])).is_ok());
    assert_eq!(why(&m, &rows(&[checks[1], checks[0]])), Reject::NotACheck);
    assert_eq!(why(&m, &rows(&[checks[0], checks[0]])), Reject::NotACheck);
    assert_eq!(why(&m, &rows(&[checks[0] + 1])), Reject::NotACheck);
    assert_eq!(why(&m, &rows(&[bd.insts.len() as u32])), Reject::NotACheck);
    // the check in the block nothing reaches (the pass leaves it alone too)
    assert_eq!(bd.blocks[dead.index()].insts.indices().next(), Some(checks[2]));
    assert_eq!(why(&m, &rows(&[checks[2]])), Reject::NotACheck);
    assert!(prove_func(&m, f).unwrap().witness.removed.iter().all(|r| r.inst != checks[2]));

    let mut w = rows(&checks[..1]);
    w.removed[0].edges.len = 2;
    assert_eq!(why(&m, &w), Reject::Chain);
    let mut w = rows(&checks[..1]);
    w.func = FuncId(99);
    assert_eq!(why(&m, &w), Reject::Cfg);

    // dominance tables
    let good = rows(&checks[..1]);
    let with_dom = |change: &dyn Fn(&mut Vec<u32>)| {
        let mut w = good.clone();
        change(&mut w.dom);
        w
    };
    assert_eq!(why(&m, &with_dom(&|d| d.clear())), Reject::Cfg);
    assert_eq!(why(&m, &with_dom(&|d| d[0] = 1)), Reject::Cfg);
    assert_eq!(why(&m, &with_dom(&|d| d[then.index()] = NO_BLOCK)), Reject::Cfg);
    // two blocks naming each other hang off nothing
    let els = then.index() + 1;
    assert_eq!(why(&m, &with_dom(&|d| (d[then.index()], d[els]) = (els as u32, then.0))), Reject::Cfg);
    // the entry of a block nothing reaches is not read
    assert!(check_func(&m, &with_dom(&|d| d[dead.index()] = 7)).is_ok());
}

/// After removal the program still runs the same, and a kept check still traps.
#[test]
fn removal_keeps_results_and_traps() {
    let (mut m, f, _) = fixture(|b, _, t, p| {
        let n = b.op(Op::SliceLen, t.us, &[p.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, p.i, n);
        let (_, els) = branch(b, c);
        let x = read(b, t, p.a, n, p.i);
        let y = read(b, t, p.a, n, p.j);
        let s = b.binary(Op::AddWrap, t.u32t, x, y);
        ret(b, s);
        b.switch_to(els);
        ret0(b);
    });
    let t_us = m.types.usize();
    let u32t = m.types.u32();
    let arr = m.types.array(u32t, 4);
    let parr = m.types.ref_(true, arr);
    let (pm, pe) = (m.types.ref_(true, u32t), m.types.ref_(false, u32t));
    let sl = m.types.slice(u32t);
    let psl = m.types.ref_(false, sl);
    let main = define(&mut m, &[2], "app.main", &[t_us, t_us], Some(u32t), |b, _| {
        let (i, j) = (b.param(0).unwrap(), b.param(1).unwrap());
        let s = b.slot(parr);
        for k in 0..4 {
            let at = b.int(k);
            let p = b.op(Op::ElemAddr, pm, &[s.into(), at]);
            let v = b.int(10 * (k + 1));
            b.store(p, v);
        }
        let z = b.int(0);
        let base = b.op(Op::ElemAddr, pe, &[s.into(), z]);
        let four = b.int(4);
        let a = b.op(Op::SliceMake, psl, &[base.into(), four]);
        let r = b.call(f, Some(u32t), &[a.into(), a.into(), i.into(), j.into()]).unwrap();
        b.ret(Some(r.into()));
    });
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    let mut o = m.clone();
    let removed = bounds::elide(&mut o).unwrap();
    assert_eq!(removed.iter().filter(|e| e.func == f).count(), 1);
    assert_eq!(insts_with(body(&o, f), Op::BoundsCheck).len(), 1);
    verify_module(&o).unwrap();
    bounds::audit(&o).unwrap();
    for (i, j) in [(0u64, 0u64), (3, 1), (2, 3), (4, 0), (9, 9), (1, 4), (0, u64::MAX)] {
        let args = vec![Val::S(i), Val::S(j)];
        let (want, got) = (run(&m, main, args.clone()), run(&o, main, args));
        assert_eq!(got, want, "i = {i}, j = {j}");
        let traps = matches!(got, Outcome::Trap(t) if t.kind == PanicKind::Bounds);
        assert_eq!(traps, i < 4 && j >= 4, "i = {i}, j = {j}");
    }
    // Nothing is left to prove: the pass is idempotent.
    assert!(bounds::elide(&mut o).unwrap().iter().all(|e| e.func != f));
}

/// The checker reads the IR and the witness tables only: nothing of the pass that writes the
/// witnesses, nor of the verifier whose dominator tree that pass uses.
#[test]
fn the_checker_shares_no_code_with_the_pass() {
    let src = include_str!("../src/witness/check.rs");
    let code: Vec<&str> = src.lines().filter(|l| !l.trim_start().starts_with("//")).collect();
    let uses: Vec<&str> = code.iter().copied().filter(|l| l.starts_with("use ")).collect();
    assert_eq!(uses.len(), 3, "{uses:?}");
    assert!(uses[0].starts_with("use super::{") && uses[1].starts_with("use crate::ir::{"), "{uses:?}");
    assert_eq!(uses[2], "use crate::types::TypeKey;");
    for banned in ["bounds::", "verify", "facts", "Dominators", "interp", "crate::simd", "reachable()"] {
        assert!(code.iter().all(|l| !l.contains(banned)), "the checker uses `{banned}`");
    }
    // ... and the data module it reads holds no logic to share.
    let data = include_str!("../src/witness.rs");
    assert!(!data.contains("fn "), "witness.rs is tables and constants only");
}
