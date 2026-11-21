//! Summaries, call rows, and unchecked entries on hand-made functions: what the pass states and
//! proves is accepted by the checker and changes no run, and one forged summary or call row per
//! way of being wrong is refused.

mod common;

use common::forge::{body, insts_with};
use common::*;
use stave_ir::bounds::{claims, prove_func, prove_func_in};
use stave_ir::interp::Val;
use stave_ir::ir::Span;
use stave_ir::witness::check::{Checked, Reject, check_func, check_module, remove, remove_whole};
use stave_ir::witness::{CallProved, CallRefuted, FnWitness, PostAt, Removed, Row, Source, Term};
use stave_ir::{
    Block, FuncBuilder, FuncId, GenericArg, Module, Op, Operand, Outcome, PanicKind, Site, Value,
    verify_module,
};

/// Branches to a call of `fail` unless `c`; the builder is left in the block past the check,
/// which is returned.
fn require(b: &mut FuncBuilder, fail: FuncId, c: Value) -> Block {
    let (bad, ok) = (b.create_block(), b.create_block());
    b.cbr(c, ok, &[], bad, &[]);
    b.switch_to(bad);
    b.call(fail, None, &[]);
    b.unreachable();
    b.switch_to(ok);
    ok
}

/// A contract failure fn at `path` that panics `Contract`.
fn failure(m: &mut Module, id: u32, k: stave_ir::Known) -> FuncId {
    define_known(m, &[id], k, &[], None, |b, _| {
        b.inst(Op::Panic(PanicKind::Contract), None, &[]);
        b.unreachable();
    })
}

struct Calls {
    m: Module,
    pos: FuncId,
    caller: FuncId,
    /// The block of `pos` past its check, which returns.
    ok: Block,
}

/// What `app.caller` adds to `pos(5)`.
#[derive(Clone, Copy, PartialEq)]
enum Second {
    /// `pos(y)`: a call nothing proves.
    Unknown,
    /// `y`.
    Nothing,
    /// `pos(y)` through the fn value `@pos`.
    Value,
}

/// `app.pos(x: u32) -> u32 requires(x > 0)` returning `x`, and `app.caller(y: u32) -> u32`
/// returning `pos(5) +% pos(y)`.
fn calls_fixture() -> Calls {
    fixture(Second::Unknown)
}

fn fixture(second: Second) -> Calls {
    let mut m = Module::new();
    let t = m.types.u32();
    let bt = m.types.bool();
    let fail = failure(&mut m, 9, stave_ir::Known::RequiresFailed);
    let mut ok = Block(0);
    let pos = define(&mut m, &[1], "app.pos", &[t], Some(t), |b, _| {
        let x = b.param(0).unwrap();
        let zero = b.int(0);
        let c = b.cmp(Op::Gt, bt, x, zero);
        ok = require(b, fail, c);
        b.ret(Some(x.into()));
    });
    let caller = define(&mut m, &[2], "app.caller", &[t], Some(t), |b, _| {
        let y = b.param(0).unwrap();
        let five = b.int(5);
        let a = b.call(pos, Some(t), &[five]).unwrap();
        let c = match second {
            Second::Unknown => b.call(pos, Some(t), &[y.into()]).unwrap(),
            Second::Nothing => y,
            Second::Value => b.call_ind(Operand::Func(pos), Some(t), &[y.into()]).unwrap(),
        };
        let s = b.binary(Op::AddWrap, t, a, c);
        b.ret(Some(s.into()));
    });
    if second != Second::Value {
        verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    }
    Calls { m, pos, caller, ok }
}

/// The witnesses the pass writes for `pos` and `caller`, and their rows as one accepted set.
fn accepted(m: &Module, pos: FuncId, caller: FuncId) -> Vec<Checked> {
    let ws: Vec<FnWitness> =
        [pos, caller].iter().map(|&f| prove_func(m, f).expect("a witness").witness).collect();
    check_module(m, &ws).unwrap_or_else(|r| panic!("{r:?}"))
}

/// The reasons the checker refuses `ws` for, in order (none: accepted).
fn whys(m: &Module, ws: &[FnWitness]) -> Vec<Reject> {
    match check_module(m, ws) {
        Ok(_) => Vec::new(),
        Err(rs) => rs.iter().map(|r| r.why).collect(),
    }
}

/// The call of `pos` with a constant proves `0 < 5` and goes to the unchecked entry, a new
/// function without the failure call; the call with `y` keeps the checked one. No run changes.
#[test]
fn a_proved_call_enters_past_the_checks() {
    let Calls { m, pos, caller, .. } = calls_fixture();
    let stated = claims(&m);
    let ws: Vec<FnWitness> = [pos, caller]
        .iter()
        .map(|&f| prove_func_in(&m, f, Some(&stated)).expect("a witness").witness)
        .collect();
    let s = &ws[0].summary;
    assert_eq!(
        s.pre.iter().map(|p| p.row).collect::<Vec<_>>(),
        [Row { a: Term::Zero, b: Term::Param(0), w: -1 }]
    );
    assert_eq!((s.entry.len(), s.pre[0].entry, s.safe.len(), s.post.len()), (1, 0, 0, 0));
    let calls = insts_with(body(&m, caller), Op::Call);
    assert_eq!(ws[1].calls.proved.iter().map(|c| c.inst).collect::<Vec<_>>(), [calls[0]]);
    assert!(ws[1].calls.refuted.is_empty());
    let checked = check_module(&m, &ws).unwrap_or_else(|r| panic!("{r:?}"));

    let mut o = m.clone();
    let removal = remove(&mut o, &checked);
    verify_module(&o).unwrap_or_else(|e| panic!("{e:?}"));
    let entry = FuncId(m.funcs.len() as u32);
    assert_eq!(removal.unchecked, [(pos, entry)]);
    assert!(removal.proved.is_empty(), "one of the two calls is proved");
    let callee = |i: u32| body(&o, caller).args(i)[0];
    assert_eq!((callee(calls[0]), callee(calls[1])), (Operand::Func(entry), Operand::Func(pos)));
    assert!(insts_with(body(&o, entry), Op::Call).is_empty());
    assert_eq!(insts_with(body(&o, pos), Op::Call).len(), 1);
    assert_eq!(o.funcs[entry.index()].instance.to_string(), "1.u");
    for y in [0u64, 1, 7] {
        let (want, got) = (run(&m, caller, vec![Val::S(y)]), run(&o, caller, vec![Val::S(y)]));
        assert_eq!(got, want, "y = {y}");
        assert_eq!(matches!(got, Outcome::Trap(t) if t.kind == PanicKind::Contract), y == 0);
    }
}

/// Forged summaries and call rows for [`calls_fixture`].
#[test]
fn forged_summaries_and_calls_are_refused() {
    let Calls { m, pos, caller, ok } = calls_fixture();
    let good: Vec<FnWitness> =
        [pos, caller].iter().map(|&f| prove_func(&m, f).expect("a witness").witness).collect();
    assert!(whys(&m, &good).is_empty());
    let calls = insts_with(body(&m, caller), Op::Call);

    // a call reads the summary its callee's witness states: without it, nothing is proved
    assert_eq!(check_func(&m, &good[1]).unwrap_err().why, Reject::Call);

    // a row weaker than the check (`x >= 0`), with its proof weakened to match
    let mut ws = good.clone();
    ws[0].summary.pre[0].row.w = 0;
    for e in &mut ws[0].edges {
        if matches!(e.src, Source::Requires { .. }) {
            e.w = 0;
        }
    }
    assert_eq!(whys(&m, &ws), [Reject::Weak]);

    // the same row, the proof still citing the stronger one
    let mut ws = good.clone();
    ws[0].summary.pre[0].row.w = 0;
    assert_eq!(whys(&m, &ws), [Reject::Edge]);

    // rows without the proof of the check: no entry, and the call has nothing to enter
    let mut ws = good.clone();
    ws[0].summary.entry.clear();
    assert_eq!(whys(&m, &ws), [Reject::Summary, Reject::Call]);

    // the entry's proof used to remove the check itself: a `Requires` edge holds only where the
    // callers proved the rows
    let mut ws = good.clone();
    let e = ws[0].summary.entry[0];
    ws[0].removed.push(Removed { inst: e.inst, site: Site::NONE, edges: e.edges, also: Span::default() });
    assert_eq!(whys(&m, &ws), [Reject::Source]);

    // a `post` row nothing proves: `result <= 5`
    let mut ws = good.clone();
    ws[0].summary.post.push(Row { a: Term::Result, b: Term::Zero, w: 5 });
    assert_eq!(whys(&m, &ws), [Reject::Summary]);
    ws[0].summary.post_at.push(PostAt { row: 0, block: ok.0, edges: Span::default() });
    assert_eq!(whys(&m, &ws), [Reject::Chain]);

    // the call with `y` claimed proved by the proof of the call with 5
    let mut ws = good.clone();
    let p = ws[1].calls.proved[0];
    ws[1].calls.proved.push(CallProved { inst: calls[1], ..p });
    assert_eq!(whys(&m, &ws), [Reject::Chain]);

    // a call row that names no call, and one with a proof too few
    let mut ws = good.clone();
    ws[1].calls.proved[0].inst = calls[1] + 1;
    assert_eq!(whys(&m, &ws), [Reject::Call]);
    let mut ws = good.clone();
    ws[1].calls.proved[0].rows.len = 0;
    assert_eq!(whys(&m, &ws), [Reject::Call]);

    // refutations: of the call with `y` (nothing says `y <= 0`), and of the proved call
    let refuted = |inst: u32| CallRefuted { inst, site: Site::NONE, row: 0, edges: Span::default() };
    let mut ws = good.clone();
    ws[1].calls.refuted.push(refuted(calls[1]));
    assert_eq!(whys(&m, &ws), [Reject::Chain]);
    let mut ws = good.clone();
    ws[1].calls.refuted.push(refuted(calls[0]));
    assert_eq!(whys(&m, &ws), [Reject::Call]);
}

/// One witness per function, and one accepted set of rows per removal: a second witness for a
/// function is refused by the check; two rows for one function, or rows of two checks, are
/// refused by the removal, which then changes nothing.
#[test]
fn rows_that_are_not_one_accepted_set_are_refused() {
    let Calls { m, pos, caller, .. } = calls_fixture();
    let good: Vec<FnWitness> =
        [pos, caller].iter().map(|&f| prove_func(&m, f).expect("a witness").witness).collect();
    let twice = vec![good[0].clone(), good[1].clone(), good[1].clone()];
    assert_eq!(whys(&m, &twice), [Reject::Repeated]);

    let alone = check_func(&m, &good[0]).unwrap_or_else(|r| panic!("{r}"));
    let set = check_module(&m, &good).unwrap_or_else(|r| panic!("{r:?}"));
    let unchanged = |o: &Module| {
        o.funcs.len() == m.funcs.len()
            && insts_with(body(o, pos), Op::Call) == insts_with(body(&m, pos), Op::Call)
            && body(o, caller).args(0) == body(&m, caller).args(0)
    };
    let mut o = m.clone();
    assert_eq!(remove(&mut o, &[alone.clone(), alone.clone()]).refused, Some(pos));
    assert!(unchanged(&o));
    assert_eq!(remove(&mut o, &[set[1].clone(), alone]).refused, Some(pos));
    assert!(unchanged(&o));
    assert_eq!(remove(&mut o, &set).refused, None);
    assert!(!unchanged(&o));
}

/// A function loses its checked entry only in a whole program, only as an instance of a generic
/// function without a link symbol, and only when every operand naming it is the callee of an
/// accepted call. Else the unchecked entry is a new function, without the symbol.
#[test]
fn only_a_whole_program_replaces_a_function_by_its_unchecked_entry() {
    let generic = |m: &mut Module, f: FuncId| {
        let t = m.types.u32();
        m.funcs[f.index()].type_args = vec![GenericArg::Type(t)];
    };
    let failure_calls = |m: &Module, f: FuncId| insts_with(body(m, f), Op::Call).len();

    // one proved call, nothing else names the instance
    let Calls { mut m, pos, caller, .. } = fixture(Second::Nothing);
    generic(&mut m, pos);
    let rows = accepted(&m, pos, caller);
    let mut whole = m.clone();
    let removal = remove_whole(&mut whole, &rows);
    assert_eq!((removal.unchecked, removal.proved), (vec![(pos, pos)], vec![(caller, pos)]));
    assert_eq!((whole.funcs.len(), failure_calls(&whole, pos)), (m.funcs.len(), 0));
    // the same rows on a module that may be a part of the program
    let mut part = m.clone();
    let entry = FuncId(m.funcs.len() as u32);
    let removal = remove(&mut part, &rows);
    assert_eq!((removal.unchecked, removal.proved), (vec![(pos, entry)], vec![(caller, pos)]));
    assert_eq!((failure_calls(&part, pos), failure_calls(&part, entry)), (1, 0));

    // an exported function keeps its checked entry and its symbol; the new function has none
    let Calls { mut m, pos, caller, .. } = fixture(Second::Nothing);
    generic(&mut m, pos);
    m.funcs[pos.index()].symbol = Some("pos".to_string());
    let rows = accepted(&m, pos, caller);
    let mut o = m.clone();
    let removal = remove_whole(&mut o, &rows);
    let entry = FuncId(m.funcs.len() as u32);
    assert_eq!(removal.unchecked, [(pos, entry)]);
    let symbol = |f: FuncId| o.funcs[f.index()].symbol.clone();
    assert_eq!((symbol(pos), symbol(entry)), (Some("pos".to_string()), None));
    let names = stave_ir::link::func_symbols(&o, &[pos, entry]);
    assert_ne!(names[pos.index()], names[entry.index()]);

    // a fn value of the instance in the same caller: the value enters the checked entry
    let Calls { mut m, pos, caller, .. } = fixture(Second::Value);
    generic(&mut m, pos);
    let rows = accepted(&m, pos, caller);
    let mut o = m.clone();
    let removal = remove_whole(&mut o, &rows);
    let entry = FuncId(m.funcs.len() as u32);
    assert_eq!(removal.unchecked, [(pos, entry)]);
    assert!(removal.proved.is_empty());
    assert_eq!(failure_calls(&o, pos), 1);
}

/// `app.small(x: u32) -> u32 ensures(result < 10)` (`x` when below 10, else 9) states
/// `result <= 9`; `app.user(y)` checks `small(y)` against 10 and loses that check to the row.
/// A stronger row is refused in `small`, and so is the proof in `user` that cites it.
#[test]
fn an_ensures_is_a_fact_after_the_call() {
    let mut m = Module::new();
    let t = m.types.u32();
    let bt = m.types.bool();
    let fail = failure(&mut m, 9, stave_ir::Known::EnsuresFailed);
    let small = define(&mut m, &[1], "app.small", &[t], Some(t), |b, _| {
        let x = b.param(0).unwrap();
        let (ten, nine) = (b.int(10), b.int(9));
        let c = b.cmp(Op::Lt, bt, x, ten);
        let (then, els) = (b.create_block(), b.create_block());
        b.cbr(c, then, &[], els, &[]);
        b.switch_to(then);
        let e = b.cmp(Op::Lt, bt, x, ten);
        require(b, fail, e);
        b.ret(Some(x.into()));
        b.switch_to(els);
        let e = b.cmp(Op::Lt, bt, nine, ten);
        require(b, fail, e);
        b.ret(Some(nine));
    });
    let user = define(&mut m, &[2], "app.user", &[t], Some(t), |b, _| {
        let y = b.param(0).unwrap();
        let ten = b.int(10);
        let s = b.call(small, Some(t), &[y.into()]).unwrap();
        b.bounds_check(s, ten);
        b.ret(Some(s.into()));
    });
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    let good: Vec<FnWitness> =
        [small, user].iter().map(|&f| prove_func(&m, f).expect("a witness").witness).collect();
    assert_eq!(good[0].summary.post, [Row { a: Term::Result, b: Term::Zero, w: 9 }]);
    assert_eq!(good[0].summary.post_at.len(), 2);
    let call = insts_with(body(&m, user), Op::Call)[0];
    let proof = good[1].removed[0].edges.of(&good[1].edges);
    assert_eq!(proof.iter().map(|e| e.src).collect::<Vec<_>>(), [Source::Ensures { call, clause: 0 }]);
    let checked = check_module(&m, &good).unwrap_or_else(|r| panic!("{r:?}"));
    // alone, the user's witness reads no summary
    assert_eq!(check_func(&m, &good[1]).unwrap_err().why, Reject::Source);

    let mut o = m.clone();
    remove(&mut o, &checked);
    verify_module(&o).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(insts_with(body(&o, user), Op::BoundsCheck).is_empty());
    for y in [3u64, 9, 10, 500] {
        let (want, got) = (run(&m, user, vec![Val::S(y)]), run(&o, user, vec![Val::S(y)]));
        assert_eq!(got, want, "y = {y}");
        assert!(matches!(got, Outcome::Return(Some(_))), "y = {y}");
    }

    let mut ws = good.clone();
    ws[0].summary.post[0].w = 8;
    assert_eq!(whys(&m, &ws), [Reject::Weak, Reject::Edge]);
}
