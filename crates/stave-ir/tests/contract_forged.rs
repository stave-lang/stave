//! More forged summaries and call rows, one per rule of the checker: rows that claim to read a
//! check, facts about calls at the wrong place or from the wrong row, `Requires` edges outside
//! the entry, entries and `safe` rows that do not fit the function, and weights at the edge of
//! `i128`.

mod common;

use common::forge::{body, insts_with};
use common::*;
use stave_ir::bounds::prove_func;
use stave_ir::ir::Span;
use stave_ir::witness::check::{Reject, check_module};
use stave_ir::witness::{
    CallRefuted, Edge, FnWitness, NO_BLOCK, Node, Pre, REQUIRES_FAILED, Removed, Row, Source, Term,
};
use stave_ir::{Block, FuncBuilder, FuncId, Module, Op, PanicKind, Site, Value, verify_module};

fn require(b: &mut FuncBuilder, fail: FuncId, c: Value) -> Block {
    let (bad, ok) = (b.create_block(), b.create_block());
    b.cbr(c, ok, &[], bad, &[]);
    b.switch_to(bad);
    b.call(fail, None, &[]);
    b.unreachable();
    b.switch_to(ok);
    ok
}

fn failure(m: &mut Module) -> FuncId {
    define(m, &[9], REQUIRES_FAILED, &[], None, |b, _| {
        b.inst(Op::Panic(PanicKind::Contract), None, &[]);
        b.unreachable();
    })
}

fn whys(m: &Module, ws: &[FnWitness]) -> Vec<Reject> {
    match check_module(m, ws) {
        Ok(_) => Vec::new(),
        Err(rs) => rs.iter().map(|r| r.why).collect(),
    }
}

fn witnesses(m: &Module, fs: &[FuncId]) -> Vec<FnWitness> {
    let ws: Vec<FnWitness> = fs.iter().map(|&f| prove_func(m, f).expect("a witness").witness).collect();
    assert!(whys(m, &ws).is_empty(), "the pass's own witnesses: {:?}", whys(m, &ws));
    ws
}

fn edge(from: Node, to: Node, w: i128, src: Source) -> Edge {
    Edge { from, to, w, src, mul: 1 }
}

/// `app.lim(x: u32) -> u32` with one `requires` check per `(op, constant)`, each on `x + add`
/// when `add` is not 0, returning `x`.
fn lim(m: &mut Module, fail: FuncId, add: i128, checks: &[(Op, i128)]) -> FuncId {
    let (t, bt) = (m.types.u32(), m.types.bool());
    define(m, &[1], "app.lim", &[t], Some(t), |b, _| {
        let x = b.param(0).unwrap();
        for &(op, k) in checks {
            let lhs = if add == 0 {
                x
            } else {
                let c = b.int(add);
                b.binary(Op::Add, t, x, c)
            };
            let k = b.int(k);
            let c = b.cmp(op, bt, lhs, k);
            require(b, fail, c);
        }
        b.ret(Some(x.into()));
    })
}

/// `app.user(y: u32) -> u32`: `check.bounds y 20`, `lim(y)`, `check.bounds y 10`, returns `y`.
fn user(m: &mut Module, lim: FuncId) -> FuncId {
    let t = m.types.u32();
    define(m, &[2], "app.user", &[t], Some(t), |b, _| {
        let y = b.param(0).unwrap();
        let (twenty, ten) = (b.int(20), b.int(10));
        b.bounds_check(y, twenty);
        b.call(lim, Some(t), &[y.into()]);
        b.bounds_check(y, ten);
        b.ret(Some(y.into()));
    })
}

/// After `lim(y)` returned, its `requires(x < 10)` is a fact about `y`: the check against 10 goes,
/// the one before the call stays. The fact is refused before the call, from a row that reads no
/// check, and as a row of the callee's `post`.
#[test]
fn a_requires_is_a_fact_after_the_call_only() {
    let mut m = Module::new();
    let fail = failure(&mut m);
    let lim = lim(&mut m, fail, 0, &[(Op::Lt, 10)]);
    let user = user(&mut m, lim);
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    let good = witnesses(&m, &[lim, user]);
    let checks = insts_with(body(&m, user), Op::BoundsCheck);
    let call = insts_with(body(&m, user), Op::Call)[0];
    assert_eq!(good[1].removed.iter().map(|r| r.inst).collect::<Vec<_>>(), [checks[1]]);
    let proof = good[1].removed[0].edges.of(&good[1].edges).to_vec();
    assert_eq!(proof.iter().map(|e| e.src).collect::<Vec<_>>(), [Source::Arg { call, clause: 0 }]);

    // the same edge for the check before the call
    let mut ws = good.clone();
    ws[1].removed.insert(0, Removed { inst: checks[0], ..good[1].removed[0] });
    assert_eq!(whys(&m, &ws), [Reject::Region]);

    // the row no longer reads a check: it is only what a caller has to prove
    let mut ws = good.clone();
    ws[0].summary.pre[0].entry = NO_BLOCK;
    assert_eq!(whys(&m, &ws), [Reject::Source]);

    // the callee states no such `post` row
    let mut ws = good.clone();
    for e in &mut ws[1].edges {
        e.src = Source::Ensures { call, clause: 0 };
    }
    assert_eq!(whys(&m, &ws), [Reject::Source]);

    // a refutation needs a row that reads a check
    let mut ws = good.clone();
    ws[0].summary.pre[0].entry = NO_BLOCK;
    ws[1].removed.clear();
    ws[1].calls.refuted.push(CallRefuted { inst: call, site: Site::NONE, row: 0, edges: Span::default() });
    assert_eq!(whys(&m, &ws), [Reject::Call]);
}

/// A row may claim to read a check only when its proof is the row and equalities: not for a
/// check the type range proves, and not when a return comes before the check.
#[test]
fn a_row_that_does_not_read_its_check_is_refused() {
    // `requires(x >= 0)` on a `u32`: the entry proof by the type range alone
    let mut m = Module::new();
    let fail = failure(&mut m);
    let f = lim(&mut m, fail, 0, &[(Op::Ge, 0)]);
    let good = witnesses(&m, &[f]);
    let x = Node::Val(body(&m, f).params(Block(0))[0]);
    let mut ws = good.clone();
    let at = ws[0].edges.len() as u32;
    ws[0].edges.push(edge(Node::Zero, x, 0, Source::Range { value: body(&m, f).params(Block(0))[0] }));
    ws[0].summary.entry[0].edges = Span { start: at, len: 1 };
    assert_eq!(whys(&m, &ws), [Reject::Summary]);

    // a `ret` on a path that does not pass the check
    let mut m = Module::new();
    let fail = failure(&mut m);
    let (t, bt) = (m.types.u32(), m.types.bool());
    let f = define(&mut m, &[1], "app.early", &[t], Some(t), |b, _| {
        let x = b.param(0).unwrap();
        let (hundred, zero) = (b.int(100), b.int(0));
        let big = b.cmp(Op::Gt, bt, x, hundred);
        let (out, rest) = (b.create_block(), b.create_block());
        b.cbr(big, out, &[], rest, &[]);
        b.switch_to(out);
        b.ret(Some(x.into()));
        b.switch_to(rest);
        let c = b.cmp(Op::Gt, bt, x, zero);
        require(b, fail, c);
        b.ret(Some(x.into()));
    });
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    let good = witnesses(&m, &[f]);
    assert_eq!(good[0].summary.pre.iter().map(|p| p.entry).collect::<Vec<_>>(), [NO_BLOCK]);
    let mut ws = good.clone();
    ws[0].summary.pre[0].entry = 0;
    assert_eq!(whys(&m, &ws), [Reject::Summary]);
}

/// The entry names every reachable `requires` check, in order, and nothing else; a `Requires`
/// edge is read nowhere but in its proofs.
#[test]
fn the_entry_is_every_requires_check_and_only_its_proofs_assume_the_rows() {
    let mut m = Module::new();
    let fail = failure(&mut m);
    let f = lim(&mut m, fail, 0, &[(Op::Gt, 0), (Op::Lt, 50)]);
    let t = m.types.u32();
    let caller = define(&mut m, &[2], "app.caller", &[t], Some(t), |b, _| {
        let five = b.int(5);
        let a = b.call(f, Some(t), &[five]).unwrap();
        b.ret(Some(a.into()));
    });
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    let good = witnesses(&m, &[f, caller]);
    assert_eq!((good[0].summary.entry.len(), good[1].calls.proved.len()), (2, 1));

    let mut ws = good.clone();
    ws[0].summary.entry.pop();
    assert!(whys(&m, &ws).contains(&Reject::Summary));
    let mut ws = good.clone();
    ws[0].summary.entry.swap(0, 1);
    assert!(!whys(&m, &ws).is_empty());

    // the call's proof of the first row replaced by that row, assumed
    let mut ws = good.clone();
    let at = ws[1].edges.len() as u32;
    ws[1].edges.push(edge(Node::Zero, Node::Zero, 4, Source::Requires { clause: 0 }));
    let first = ws[1].calls.proved[0].rows.start as usize;
    ws[1].calls.goals[first] = Span { start: at, len: 1 };
    assert_eq!(whys(&m, &ws), [Reject::Source]);

    // a row and a call goal at the edge of `i128`
    let mut ws = good.clone();
    ws[0].summary.pre[0].row.w = i128::MAX;
    let rejected = whys(&m, &ws);
    assert!(!rejected.is_empty() && rejected.contains(&Reject::Overflow), "{rejected:?}");
}

/// `requires(x + 1 > 5)`: the sum is shown in range from one more row, `x <= max - 1`. The
/// `safe` row is refused without its upper proof, for an instruction that is no such sum, and
/// the extra row is no row a call may skip.
#[test]
fn a_safe_row_needs_both_bounds_of_a_trapping_sum() {
    let mut m = Module::new();
    let fail = failure(&mut m);
    let f = lim(&mut m, fail, 1, &[(Op::Gt, 5)]);
    let t = m.types.u32();
    let caller = define(&mut m, &[2], "app.caller", &[t], Some(t), |b, _| {
        let y = b.param(0).unwrap();
        let a = b.call(f, Some(t), &[y.into()]).unwrap();
        b.ret(Some(a.into()));
    });
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    // nothing bounds `y` in the caller: the pass has nothing to say about it
    assert!(prove_func(&m, caller).is_none());
    let good = vec![witnesses(&m, &[f]).remove(0), common::forge::witness(&m, caller, &[])];
    assert!(whys(&m, &good).is_empty());
    let s = &good[0].summary;
    let max = i128::from(u32::MAX);
    assert_eq!(
        s.pre.iter().map(|p| p.row).collect::<Vec<_>>(),
        [
            Row { a: Term::Zero, b: Term::Param(0), w: -5 },
            Row { a: Term::Param(0), b: Term::Zero, w: max - 1 }
        ]
    );
    assert_eq!(s.pre[1], Pre { row: s.pre[1].row, entry: NO_BLOCK, half: 0 });
    assert_eq!(s.safe.iter().map(|x| x.inst).collect::<Vec<_>>(), insts_with(body(&m, f), Op::Add));

    let mut ws = good.clone();
    ws[0].summary.safe[0].hi.len = 0;
    assert_eq!(whys(&m, &ws), [Reject::Chain]);
    let mut ws = good.clone();
    ws[0].summary.safe[0].inst = insts_with(body(&m, f), Op::Gt)[0];
    assert!(!whys(&m, &ws).is_empty());
    // the caller claims the call proved with empty proofs
    let mut ws = good.clone();
    let call = insts_with(body(&m, caller), Op::Call)[0];
    ws[1].calls.goals = vec![Span::default(); 2];
    ws[1].calls.proved.push(stave_ir::witness::CallProved {
        inst: call,
        site: Site::NONE,
        rows: Span { start: 0, len: 2 },
    });
    assert_eq!(whys(&m, &ws), [Reject::Chain]);
}

/// A constant argument is read as a value of the parameter's type: one outside it proves nothing
/// (the verifier refuses such a call; the checker does not lean on that).
#[test]
fn a_constant_argument_outside_the_parameter_type_proves_nothing() {
    let mut m = Module::new();
    let fail = failure(&mut m);
    let f = lim(&mut m, fail, 0, &[(Op::Gt, 0)]);
    let t = m.types.u32();
    let caller = define(&mut m, &[2], "app.caller", &[t], Some(t), |b, _| {
        let huge = b.int(1 << 40);
        let a = b.call(f, Some(t), &[huge]).unwrap();
        b.ret(Some(a.into()));
    });
    let ws: Vec<FnWitness> =
        [f, caller].iter().map(|&g| prove_func(&m, g).expect("a witness").witness).collect();
    assert_eq!(ws[1].calls.proved.len(), 1, "the pass reads the constant as it is");
    assert_eq!(whys(&m, &ws), [Reject::Operand]);
}
