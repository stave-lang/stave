//! Bounds-check audit and removal (`stave_ir::bounds`): every element access
//! keeps its check unless a proof is recorded; removal takes only proven checks, records each,
//! keeps the audit passing, and never changes a result or a trap (checked in the interpreter).
//! Every proof is written out as a witness the independent checker accepts (`elide` runs it),
//! and the checker refuses every corruption of those witnesses and every made-up witness for a
//! check the look-alike functions must keep (`common::forge`).

mod common;

use common::*;
use stave_ir::bounds::{AccessKind, BOUNDS_STEP_LIMIT, Elided, Guard, Proof, audit, elide};
use stave_ir::interp::Val;
use stave_ir::{
    CastKind, FuncBuilder, FuncId, Lanes, Module, Op, Operand, Outcome, PanicKind, TypeId, TypeKey, Value,
    VecOp, verify_module,
};

struct T {
    u32t: TypeId,
    us: TypeId,
    bt: TypeId,
    arr: TypeId,
    parr: TypeId,
    pe: TypeId,
    psl: TypeId,
    pce: TypeId,
}

fn types(m: &mut Module) -> T {
    let u32t = m.types.u32();
    let us = m.types.usize();
    let bt = m.types.bool();
    let arr = m.types.array(u32t, 4);
    let parr = m.types.ref_(true, arr);
    let pe = m.types.ref_(true, u32t);
    let sl = m.types.slice(u32t);
    let psl = m.types.ref_(false, sl);
    let pce = m.types.ref_(false, u32t);
    T { u32t, us, bt, arr, parr, pe, psl, pce }
}

/// `sum(n)`: an array `[1, 2, 3, 4]` in a slot, `for i in 0..n { acc += a[i] }` with the
/// front end's `check.bounds i 4`; plus `a[2]` with its `check.bounds 2 4`, checked twice.
fn sum_module(loop_bound: bool) -> (Module, FuncId) {
    let mut m = Module::new();
    let t = types(&mut m);
    let f = define(&mut m, &[0], "app.sum", &[t.us], Some(t.u32t), |b, _| {
        let n = b.param(0).unwrap();
        let s = b.slot(t.parr);
        let (k1, k2, k3, k4) = (b.int(1), b.int(2), b.int(3), b.int(4));
        let a = b.op(Op::Make, t.arr, &[k1, k2, k3, k4]);
        b.store(s, a);
        let (head, hv) = b.create_block_with(&[t.us, t.u32t]);
        let body = b.create_block();
        let (exit, ev) = b.create_block_with(&[t.u32t]);
        let z = b.int(0);
        b.br(head, &[z, z]);
        b.switch_to(head);
        // `lt i 4` (the array length) proves the body's index; `lt i n` does not.
        let bound = if loop_bound { k4 } else { n.into() };
        let c = b.cmp(Op::Lt, t.bt, hv[0], bound);
        b.cbr(c, body, &[], exit, &[hv[1].into()]);
        b.switch_to(body);
        b.bounds_check(hv[0], k4);
        let p = b.op(Op::ElemAddr, t.pe, &[s.into(), hv[0].into()]);
        let x = b.load(t.u32t, p);
        let acc = b.binary(Op::AddWrap, t.u32t, hv[1], x);
        let one = b.int(1);
        let i2 = b.binary(Op::AddWrap, t.us, hv[0], one);
        b.br(head, &[i2.into(), acc.into()]);
        b.switch_to(exit);
        b.bounds_check(k2, k4);
        let q = b.op(Op::ElemAddr, t.pe, &[s.into(), k2]);
        b.bounds_check(k2, k4);
        let y = b.load(t.u32t, q);
        let r = b.binary(Op::AddWrap, t.u32t, ev[0], y);
        b.ret(Some(r.into()));
    });
    verify_module(&m).unwrap();
    (m, f)
}

fn checks(m: &Module, f: FuncId) -> usize {
    m.func(f).unwrap().body.as_ref().unwrap().insts.op.iter().filter(|o| **o == Op::BoundsCheck).count()
}

#[test]
fn loop_and_const_checks_are_elided_with_proofs() {
    let (mut m, f) = sum_module(true);
    let acc = audit(&m).unwrap();
    assert_eq!(acc.len(), 2);
    assert!(acc.iter().all(|a| a.kind == AccessKind::Elem));
    let before = run(&m, f, vec![Val::S(4)]);
    assert_eq!(before, Outcome::Return(Some(Val::S(13))));
    assert_eq!(checks(&m, f), 3);
    let e = elide(&mut m).unwrap();
    let proofs: Vec<Proof> = e.iter().map(|x| x.proof).collect();
    assert_eq!(proofs.len(), 3, "{e:?}");
    assert!(matches!(proofs[0], Proof::LoopBound { .. }), "{proofs:?}");
    assert_eq!(proofs[1], Proof::ConstInRange { index: 2, len: 4 });
    assert_eq!(proofs[2], Proof::ConstInRange { index: 2, len: 4 });
    assert!(e.iter().all(|x| x.path == "@0 app.sum"));
    assert_eq!(checks(&m, f), 0);
    verify_module(&m).unwrap();
    // Still audited: the proofs stand on their own.
    let acc = audit(&m).unwrap();
    assert!(matches!(acc[0].guard, Guard::Proof(Proof::LoopBound { .. })));
    assert_eq!(acc[1].guard, Guard::Proof(Proof::ConstInRange { index: 2, len: 4 }));
    assert_eq!(run(&m, f, vec![Val::S(4)]), before);
}

/// The loop compares `i < n` (n a parameter, not the length): no proof, the check stays and still
/// traps on `n = 5`.
#[test]
fn unproven_check_is_kept_and_traps() {
    let (mut m, f) = sum_module(false);
    audit(&m).unwrap();
    let e = elide(&mut m).unwrap();
    assert!(
        e.iter().all(|x| matches!(x.proof, Proof::ConstInRange { .. } | Proof::Dominated { .. })),
        "{e:?}"
    );
    assert_eq!(checks(&m, f), 1, "the loop's check is kept");
    audit(&m).unwrap();
    assert_eq!(run(&m, f, vec![Val::S(4)]), Outcome::Return(Some(Val::S(13))));
    match run(&m, f, vec![Val::S(5)]) {
        Outcome::Trap(t) => assert_eq!(t.kind, PanicKind::Bounds),
        o => panic!("expected a Bounds trap, got {o:?}"),
    }
}

/// A slice made from an array (`addr.elem p 0` + `slice.make`: SliceBase), indexed with
/// `check.bounds i (slice.len s)`: the check is a guard and is never elided (dynamic index).
#[test]
fn slice_index_keeps_its_check() {
    let mut m = Module::new();
    let t = types(&mut m);
    let f = define(&mut m, &[0], "app.at", &[t.us], Some(t.u32t), |b, _| {
        let i = b.param(0).unwrap();
        let s = b.slot(t.parr);
        let (k0, k4, k9) = (b.int(0), b.int(4), b.int(9));
        let a = b.op(Op::Make, t.arr, &[k9, k9, k9, k9]);
        b.store(s, a);
        let base = b.op(Op::ElemAddr, t.pce, &[s.into(), k0]);
        let sl = b.op(Op::SliceMake, t.psl, &[base.into(), k4]);
        let n = b.op(Op::SliceLen, t.us, &[sl.into()]);
        b.bounds_check(i, n);
        let p = b.op(Op::ElemAddr, t.pce, &[sl.into(), i.into()]);
        let x = b.load(t.u32t, p);
        b.ret(Some(x.into()));
    });
    verify_module(&m).unwrap();
    let acc = audit(&m).unwrap();
    assert_eq!(acc.len(), 2);
    assert_eq!(acc[0].guard, Guard::Proof(Proof::ConstInRange { index: 0, len: 4 }));
    assert!(matches!(acc[1].guard, Guard::Check(_)));
    assert!(elide(&mut m).unwrap().is_empty());
    assert_eq!(checks(&m, f), 1);
    assert_eq!(run(&m, f, vec![Val::S(3)]), Outcome::Return(Some(Val::S(9))));
    assert!(matches!(run(&m, f, vec![Val::S(4)]), Outcome::Trap(t) if t.kind == PanicKind::Bounds));
}

/// No check, a check with the wrong length, or a check that does not dominate the access: the
/// access is unguarded (a compiler bug the driver reports).
#[test]
fn unguarded_accesses_are_reported() {
    for variant in 0..3 {
        let mut m = Module::new();
        let t = types(&mut m);
        define(&mut m, &[7], "app.bad", &[t.us, t.bt], Some(t.u32t), |b, _| {
            let i = b.param(0).unwrap();
            let c = b.param(1).unwrap();
            let s = b.slot(t.parr);
            let (k4, k8) = (b.int(4), b.int(8));
            let a = b.op(Op::Make, t.arr, &[k4, k4, k4, k4]);
            b.store(s, a);
            match variant {
                0 => {}
                1 => b.bounds_check(i, k8),
                _ => {
                    let (yes, join) = (b.create_block(), b.create_block());
                    b.cbr(c, yes, &[], join, &[]);
                    b.switch_to(yes);
                    b.bounds_check(i, k4);
                    b.br(join, &[]);
                    b.switch_to(join);
                }
            }
            let p = b.op(Op::ElemAddr, t.pe, &[s.into(), i.into()]);
            let x = b.load(t.u32t, p);
            b.ret(Some(x.into()));
        });
        verify_module(&m).unwrap();
        let bad = audit(&m).unwrap_err();
        assert_eq!(bad.len(), 1, "variant {variant}");
        assert_eq!(bad[0].kind, AccessKind::Elem);
        assert!(bad[0].to_string().starts_with("@7 app.bad: inst "), "{}", bad[0]);
        // Elision never invents a proof for it.
        let before = checks(&m, FuncId(0));
        elide(&mut m).unwrap();
        assert_eq!(checks(&m, FuncId(0)), before);
    }
}

/// `get`/`set` on an array value check a dynamic index themselves; a constant one must be in
/// range.
#[test]
fn aggregate_get_set() {
    let mut m = Module::new();
    let t = types(&mut m);
    define(&mut m, &[0], "app.agg", &[t.us], Some(t.u32t), |b, _| {
        let i = b.param(0).unwrap();
        let (k1, k3) = (b.int(1), b.int(3));
        let a = b.op(Op::Make, t.arr, &[k1, k1, k1, k1]);
        let a2 = b.op(Op::Set, t.arr, &[a.into(), i.into(), k3]);
        let x = b.op(Op::Get, t.u32t, &[a2.into(), k3]);
        b.ret(Some(x.into()));
    });
    verify_module(&m).unwrap();
    let acc = audit(&m).unwrap();
    assert_eq!(acc.len(), 2);
    assert_eq!((acc[0].kind, acc[0].guard), (AccessKind::Set, Guard::SelfChecked));
    assert_eq!(
        (acc[1].kind, acc[1].guard),
        (AccessKind::Get, Guard::Proof(Proof::ConstInRange { index: 3, len: 4 }))
    );
}

// Difference-constraint proofs: `LenEq` and `Offset`.

struct S {
    u8t: TypeId,
    u32t: TypeId,
    us: TypeId,
    i64t: TypeId,
    bt: TypeId,
    parr8: TypeId,
    pe: TypeId,
    psl: TypeId,
    pce: TypeId,
    v4: TypeId,
}

fn stypes(m: &mut Module) -> S {
    let u8t = m.types.u8();
    let u32t = m.types.u32();
    let us = m.types.usize();
    let i64t = m.types.i64();
    let bt = m.types.bool();
    let arr8 = m.types.array(u32t, 8);
    let parr8 = m.types.ref_(true, arr8);
    let pe = m.types.ref_(true, u32t);
    let sl = m.types.slice(u32t);
    let psl = m.types.ref_(false, sl);
    let pce = m.types.ref_(false, u32t);
    let v4 = m.types.intern(TypeKey::Vec { lane: u32t, n: Lanes::Fixed(4) });
    S { u8t, u32t, us, i64t, bt, parr8, pe, psl, pce, v4 }
}

/// The front end's slice read `s[idx]` (`n` = `slice.len s`): `check.bounds idx n`,
/// `addr.elem s idx`, `load`.
fn read(b: &mut FuncBuilder, t: &S, s: Value, n: Value, idx: impl Into<Operand>) -> Value {
    let idx = idx.into();
    b.bounds_check(idx, n);
    let p = b.op(Op::ElemAddr, t.pce, &[s.into(), idx]);
    b.load(t.u32t, p)
}

/// `for i in lo..hi { acc = step(i, acc) }` from `acc = 0` (the front end's loop: head test
/// `lt i hi`, increment `add.wrap i 1`); returns the final `acc`, the builder in the exit block.
fn for_range(
    b: &mut FuncBuilder,
    t: &S,
    lo: Operand,
    hi: Operand,
    step: impl FnOnce(&mut FuncBuilder, Value, Value) -> Value,
) -> Value {
    let (head, hv) = b.create_block_with(&[t.us, t.u32t]);
    let body = b.create_block();
    let (exit, ev) = b.create_block_with(&[t.u32t]);
    let z = b.int(0);
    b.br(head, &[lo, z]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, t.bt, hv[0], hi);
    b.cbr(c, body, &[], exit, &[hv[1].into()]);
    b.switch_to(body);
    let acc = step(b, hv[0], hv[1]);
    let one = b.int(1);
    let i2 = b.binary(Op::AddWrap, t.us, hv[0], one);
    b.br(head, &[i2.into(), acc.into()]);
    b.switch_to(exit);
    ev[0]
}

/// `app.check(c: bool)`: `std.debug.assert`'s shape (`if !c { panic }`, returns only when `c`);
/// with `asserting = false`, `app.note`, a look-alike that returns on both paths.
fn check_fn(m: &mut Module, t: &S, asserting: bool) -> FuncId {
    let (id, path) = if asserting { (10, "app.check") } else { (11, "app.note") };
    define(m, &[id], path, &[t.bt], None, |b, _| {
        let c = b.param(0).unwrap();
        let nc = b.unary(Op::Not, t.bt, c);
        let (fail, ok) = (b.create_block(), b.create_block());
        b.cbr(nc, fail, &[], ok, &[]);
        b.switch_to(fail);
        if asserting {
            b.panic(PanicKind::Assert);
            b.unreachable();
        } else {
            b.ret(None);
        }
        b.switch_to(ok);
        b.ret(None);
    })
}

/// What a [`two_slices`] body builder gets: the callees and `app.f`'s slices and their lengths.
struct Two {
    check: FuncId,
    note: FuncId,
    a: Value,
    b: Value,
    na: Value,
    nb: Value,
}

type BodyFn = fn(&mut FuncBuilder, &S, &Two) -> Value;

/// `app.f(a: &[u32], b: &[u32]) -> u32` with the body `f` builds (from the entry block, which
/// already holds `na = slice.len a`, `nb = slice.len b`; `f` returns the value to `ret`), and
/// `app.main(na, nb)`, which calls it on the prefixes of lengths `na`, `nb` of `[1, 2, .., 8]`.
fn two_slices(f: BodyFn) -> (Module, FuncId, FuncId) {
    let mut m = Module::new();
    let t = stypes(&mut m);
    let check = check_fn(&mut m, &t, true);
    let note = check_fn(&mut m, &t, false);
    let fid = define(&mut m, &[1], "app.f", &[t.psl, t.psl], Some(t.u32t), |b, _| {
        let (a, bb) = (b.param(0).unwrap(), b.param(1).unwrap());
        let na = b.op(Op::SliceLen, t.us, &[a.into()]);
        let nb = b.op(Op::SliceLen, t.us, &[bb.into()]);
        let r = f(b, &t, &Two { check, note, a, b: bb, na, nb });
        b.ret(Some(r.into()));
    });
    let main = define(&mut m, &[2], "app.main", &[t.us, t.us], Some(t.u32t), |b, _| {
        let (na, nb) = (b.param(0).unwrap(), b.param(1).unwrap());
        let s = b.slot(t.parr8);
        for j in 0..8 {
            let k = b.int(j);
            let p = b.op(Op::ElemAddr, t.pe, &[s.into(), k]);
            let v = b.int(j + 1);
            b.store(p, v);
        }
        let z = b.int(0);
        let base = b.op(Op::ElemAddr, t.pce, &[s.into(), z]);
        let sa = b.op(Op::SliceMake, t.psl, &[base.into(), na.into()]);
        let sb = b.op(Op::SliceMake, t.psl, &[base.into(), nb.into()]);
        let r = b.call(fid, Some(t.u32t), &[sa.into(), sb.into()]).unwrap();
        b.ret(Some(r.into()));
    });
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    audit(&m).unwrap();
    (m, fid, main)
}

/// Elides; the result verifies, audits, and `main` returns or traps the same on every `(na, nb)`
/// up to 8; no corrupted witness passes the checker. Returns the elided module and the proofs
/// recorded for `f`.
fn elide_same(m: &Module, f: FuncId, main: FuncId) -> (Module, Vec<Proof>) {
    let corrupted = forge::mutants_are_refused(m);
    let mut o = m.clone();
    let e = elide(&mut o).unwrap();
    assert!(e.is_empty() || corrupted > 0);
    verify_module(&o).unwrap_or_else(|e| panic!("{e:?}"));
    audit(&o).unwrap_or_else(|u| panic!("unguarded after elision: {u:?}"));
    for na in 0..=8u64 {
        for nb in 0..=8u64 {
            let args = vec![Val::S(na), Val::S(nb)];
            assert_eq!(run(&o, main, args.clone()), run(m, main, args), "na = {na}, nb = {nb}");
        }
    }
    (o, e.iter().filter(|x| x.func == f).map(|x| x.proof).collect())
}

fn trap(m: &Module, main: FuncId, na: u64, nb: u64) -> Option<PanicKind> {
    match run(m, main, vec![Val::S(na), Val::S(nb)]) {
        Outcome::Trap(t) => Some(t.kind),
        _ => None,
    }
}

/// `acc + a[i] * b[i]`.
fn dot_step(b: &mut FuncBuilder, t: &S, x: &Two, i: Value, acc: Value) -> Value {
    let p = read(b, t, x.a, x.na, i);
    let q = read(b, t, x.b, x.nb, i);
    let pq = b.binary(Op::MulWrap, t.u32t, p, q);
    b.binary(Op::AddWrap, t.u32t, acc, pq)
}

/// `for i in 0..a.len() { acc += a[i] * b[i] }`.
fn dot_loop(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let z = b.int(0);
    for_range(b, t, z, x.na.into(), |b, i, acc| dot_step(b, t, x, i, acc))
}

/// `if c { } else { return 0; }` then the loop `then` builds.
fn guarded(b: &mut FuncBuilder, t: &S, x: &Two, c: Value, then: BodyFn) -> Value {
    let ((out, ov), go) = (b.create_block_with(&[t.u32t]), b.create_block());
    let z = b.int(0);
    b.cbr(c, go, &[], out, &[z]);
    b.switch_to(go);
    let r = then(b, t, x);
    b.br(out, &[r.into()]);
    b.switch_to(out);
    ov[0]
}

/// `dot(a, b)` with `debug.assert(a.len() == b.len(), ..)`: no check is left, the loop bound
/// proves `a[i]` and the equality `b[i]`.
#[test]
fn dot_with_assert_has_no_check() {
    let (m, f, main) = two_slices(|b, t, x| {
        let eq = b.cmp(Op::Eq, t.bt, x.na, x.nb);
        b.call(x.check, None, &[eq.into()]);
        dot_loop(b, t, x)
    });
    assert_eq!(checks(&m, f), 2);
    let (o, proofs) = elide_same(&m, f, main);
    assert_eq!(checks(&o, f), 0);
    assert!(matches!(proofs[..], [Proof::LoopBound { .. }, Proof::LenEq { .. }]), "{proofs:?}");
    assert_eq!(trap(&o, main, 3, 2), Some(PanicKind::Assert));
    assert_eq!(run(&o, main, vec![Val::S(3), Val::S(3)]), Outcome::Return(Some(Val::S(14))));
}

/// `if a.len() != b.len() { return 0; }` before the loop proves `b[i]` as well.
#[test]
fn dot_with_early_return_has_no_check() {
    let (m, f, main) = two_slices(|b, t, x| {
        let ne = b.cmp(Op::Ne, t.bt, x.na, x.nb);
        let ((out, ov), go) = (b.create_block_with(&[t.u32t]), b.create_block());
        let z = b.int(0);
        b.cbr(ne, out, &[z], go, &[]);
        b.switch_to(go);
        let r = dot_loop(b, t, x);
        b.br(out, &[r.into()]);
        b.switch_to(out);
        ov[0]
    });
    let (o, proofs) = elide_same(&m, f, main);
    assert_eq!(checks(&o, f), 0);
    assert!(matches!(proofs[..], [Proof::LoopBound { .. }, Proof::LenEq { .. }]), "{proofs:?}");
}

/// `for i in 1..a.len() { acc += a[i] - a[i - 1] }`: `a[i - 1]` is `Offset { k: -1 }`.
#[test]
fn diff_previous_element_has_no_check() {
    let (m, f, main) = two_slices(|b, t, x| {
        let one = b.int(1);
        for_range(b, t, one, x.na.into(), |b, i, acc| {
            let p = read(b, t, x.a, x.na, i);
            let one = b.int(1);
            let im1 = b.binary(Op::Sub, t.us, i, one);
            let q = read(b, t, x.a, x.na, im1);
            let d = b.binary(Op::SubWrap, t.u32t, p, q);
            b.binary(Op::AddWrap, t.u32t, acc, d)
        })
    });
    let (o, proofs) = elide_same(&m, f, main);
    assert_eq!(checks(&o, f), 0);
    assert!(
        matches!(proofs[..], [Proof::LoopBound { .. }, Proof::Offset { base: Some(_), k: -1 }]),
        "{proofs:?}"
    );
}

/// `for i in 0..a.len() - 1 { acc += a[i + 1] }`: `Offset { k: 1 }`; an empty slice traps
/// `Overflow` at `a.len() - 1`, before and after.
#[test]
fn next_element_below_len_minus_one_has_no_check() {
    let (m, f, main) = two_slices(|b, t, x| {
        let one = b.int(1);
        let hi = b.binary(Op::Sub, t.us, x.na, one);
        let z = b.int(0);
        for_range(b, t, z, hi.into(), |b, i, acc| {
            let one = b.int(1);
            let ip1 = b.binary(Op::Add, t.us, i, one);
            let q = read(b, t, x.a, x.na, ip1);
            b.binary(Op::AddWrap, t.u32t, acc, q)
        })
    });
    let (o, proofs) = elide_same(&m, f, main);
    assert_eq!(checks(&o, f), 0);
    assert!(matches!(proofs[..], [Proof::Offset { base: Some(_), k: 1 }]), "{proofs:?}");
    assert_eq!(trap(&o, main, 0, 0), Some(PanicKind::Overflow));
}

/// `while i + 1 < a.len() { acc += a[i + 1] + a[i]; i += 1; }`, each `slice.len` and `i + 1`
/// computed again at its use (the checked IR before the optimizer unifies the lengths): both
/// checks go.
#[test]
fn while_offset_bound_with_separate_reads() {
    let (m, f, main) = two_slices(|b, t, x| {
        let (head, hv) = b.create_block_with(&[t.us, t.u32t]);
        let body = b.create_block();
        let (exit, ev) = b.create_block_with(&[t.u32t]);
        let z = b.int(0);
        b.br(head, &[z, z]);
        b.switch_to(head);
        let one = b.int(1);
        let ip1 = b.binary(Op::Add, t.us, hv[0], one);
        let n0 = b.op(Op::SliceLen, t.us, &[x.a.into()]);
        let c = b.cmp(Op::Lt, t.bt, ip1, n0);
        b.cbr(c, body, &[], exit, &[hv[1].into()]);
        b.switch_to(body);
        let ip1b = b.binary(Op::Add, t.us, hv[0], one);
        let n1 = b.op(Op::SliceLen, t.us, &[x.a.into()]);
        let p = read(b, t, x.a, n1, ip1b);
        let n2 = b.op(Op::SliceLen, t.us, &[x.a.into()]);
        let q = read(b, t, x.a, n2, hv[0]);
        let s = b.binary(Op::AddWrap, t.u32t, p, q);
        let acc = b.binary(Op::AddWrap, t.u32t, hv[1], s);
        b.br(head, &[ip1b.into(), acc.into()]);
        b.switch_to(exit);
        ev[0]
    });
    let (o, proofs) = elide_same(&m, f, main);
    assert_eq!(checks(&o, f), 0);
    assert!(matches!(proofs[..], [Proof::Offset { k: 1, .. }, Proof::Offset { k: 0, .. }]), "{proofs:?}");
}

/// `if b.len() <= a.len()`: `b` is no longer than `a`, so `b[i]` over `a`'s range can fail.
fn le_only(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let le = b.cmp(Op::Le, t.bt, x.nb, x.na);
    guarded(b, t, x, le, dot_loop)
}

/// The equality is asserted on one path only (`if a.len() < 5`).
fn eq_on_one_path(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let five = b.int(5);
    let c = b.cmp(Op::Lt, t.bt, x.na, five);
    let (yes, join) = (b.create_block(), b.create_block());
    b.cbr(c, yes, &[], join, &[]);
    b.switch_to(yes);
    let eq = b.cmp(Op::Eq, t.bt, x.na, x.nb);
    b.call(x.check, None, &[eq.into()]);
    b.br(join, &[]);
    b.switch_to(join);
    dot_loop(b, t, x)
}

/// A callee that returns whether or not its argument holds asserts nothing.
fn non_asserting_callee(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let eq = b.cmp(Op::Eq, t.bt, x.na, x.nb);
    b.call(x.note, None, &[eq.into()]);
    dot_loop(b, t, x)
}

/// `a[i + b.len()]`: the offset is a value, not a constant.
fn non_constant_offset(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let z = b.int(0);
    for_range(b, t, z, x.na.into(), |b, i, acc| {
        let ik = b.binary(Op::Add, t.us, i, x.nb);
        let q = read(b, t, x.a, x.na, ik);
        b.binary(Op::AddWrap, t.u32t, acc, q)
    })
}

/// `a[i] + a[i + 1]` under `i < a.len()` (not `i + 1 < a.len()`): the last `a[i + 1]` is out of
/// range; `a[i]` is proven.
fn plus_one_under_lt_len(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let z = b.int(0);
    for_range(b, t, z, x.na.into(), |b, i, acc| {
        let p = read(b, t, x.a, x.na, i);
        let one = b.int(1);
        let ip1 = b.binary(Op::Add, t.us, i, one);
        let q = read(b, t, x.a, x.na, ip1);
        let s = b.binary(Op::AddWrap, t.u32t, p, q);
        b.binary(Op::AddWrap, t.u32t, acc, s)
    })
}

/// `a[i -% 1]`: wrapping arithmetic is opaque (`i = 0` wraps to `usize::MAX`).
fn wrapping_offset(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let z = b.int(0);
    for_range(b, t, z, x.na.into(), |b, i, acc| {
        let one = b.int(1);
        let im1 = b.binary(Op::SubWrap, t.us, i, one);
        let q = read(b, t, x.a, x.na, im1);
        b.binary(Op::AddWrap, t.u32t, acc, q)
    })
}

/// `b` re-sliced to its first `b.len() - 1` elements after the equality: the new slice is shorter.
fn resliced_after_eq(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let eq = b.cmp(Op::Eq, t.bt, x.na, x.nb);
    b.call(x.check, None, &[eq.into()]);
    let one = b.int(1);
    let n = b.binary(Op::Sub, t.us, x.nb, one);
    let z = b.int(0);
    let base = b.op(Op::ElemAddr, t.pce, &[x.b.into(), z]);
    let c = b.op(Op::SliceMake, t.psl, &[base.into(), n.into()]);
    let nc = b.op(Op::SliceLen, t.us, &[c.into()]);
    for_range(b, t, z, x.na.into(), |b, i, acc| {
        let q = read(b, t, c, nc, i);
        b.binary(Op::AddWrap, t.u32t, acc, q)
    })
}

/// `a[(i as i64 + 1) as usize]`: signed arithmetic and casts are opaque.
fn signed_index(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let z = b.int(0);
    for_range(b, t, z, x.na.into(), |b, i, acc| {
        let j = b.cast(CastKind::Wrap, t.i64t, i);
        let one = b.int(1);
        let j1 = b.binary(Op::Add, t.i64t, j, one);
        let k = b.cast(CastKind::Wrap, t.us, j1);
        let q = read(b, t, x.a, x.na, k);
        b.binary(Op::AddWrap, t.u32t, acc, q)
    })
}

/// `for i in 0..8 { acc += a[op(i, y)] }` with the index `op` builds from `i`.
fn over_eight(
    b: &mut FuncBuilder,
    t: &S,
    x: &Two,
    index: fn(&mut FuncBuilder, &S, &Two, Value) -> Value,
) -> Value {
    let (z, eight) = (b.int(0), b.int(8));
    for_range(b, t, z, eight, |b, i, acc| {
        let k = index(b, t, x, i);
        let q = read(b, t, x.a, x.na, k);
        b.binary(Op::AddWrap, t.u32t, acc, q)
    })
}

/// `a[i & (a.len() - 1)]`: at most `a.len() - 1` (the trapping `- 1` rules out an empty `a`).
fn masked_by_len(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    over_eight(b, t, x, |b, t, x, i| {
        let one = b.int(1);
        let m = b.binary(Op::Sub, t.us, x.na, one);
        b.binary(Op::And, t.us, i, m)
    })
}

/// `a[i % a.len()]`: below the divisor (a zero divisor traps).
fn rem_by_len(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    over_eight(b, t, x, |b, t, x, i| b.binary(Op::Rem, t.us, i, x.na))
}

/// `for i in 0..a.len() { a[i >> 1] }`: a logical shift does not grow the value.
fn shr_under_len(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let z = b.int(0);
    for_range(b, t, z, x.na.into(), |b, i, acc| {
        let one = b.int(1);
        let k = b.binary(Op::Shr, t.us, i, one);
        let q = read(b, t, x.a, x.na, k);
        b.binary(Op::AddWrap, t.u32t, acc, q)
    })
}

/// `if a.len() == 8 { .. a[i & 7] .. }`: the mask's constant against a length equal to 8.
fn masked_const(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let eight = b.int(8);
    let eq = b.cmp(Op::Eq, t.bt, x.na, eight);
    guarded(b, t, x, eq, |b, t, x| {
        over_eight(b, t, x, |b, t, _, i| {
            let seven = b.int(7);
            b.binary(Op::And, t.us, i, seven)
        })
    })
}

/// `if a.len() >= 5 { .. a[i % 5] .. }`.
fn rem_const(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let five = b.int(5);
    let ge = b.cmp(Op::Ge, t.bt, x.na, five);
    guarded(b, t, x, ge, |b, t, x| {
        over_eight(b, t, x, |b, t, _, i| {
            let five = b.int(5);
            b.binary(Op::Rem, t.us, i, five)
        })
    })
}

/// `if a.len() >= 256 { .. a[(i as% u8) as usize] .. }`: a `u8` is at most 255, and the widening
/// cast keeps its value.
fn byte_index(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let k256 = b.int(256);
    let ge = b.cmp(Op::Ge, t.bt, x.na, k256);
    guarded(b, t, x, ge, |b, t, x| {
        over_eight(b, t, x, |b, t, _, i| {
            let byte = b.cast(CastKind::Wrap, t.u8t, i);
            b.cast(CastKind::Exact, t.us, byte)
        })
    })
}

/// Unsigned `and`, `rem`, `shr`, and the range of a `u8` bound the index: no check is left, and
/// every program runs and traps as before.
#[test]
fn bit_ops_and_type_ranges_have_no_check() {
    let cases: [(&str, BodyFn); 6] = [
        ("i & (len - 1)", masked_by_len),
        ("i % len", rem_by_len),
        ("i >> 1", shr_under_len),
        ("i & 7 under len == 8", masked_const),
        ("i % 5 under len >= 5", rem_const),
        ("u8 index under len >= 256", byte_index),
    ];
    for (name, body) in cases {
        let (m, f, main) = two_slices(body);
        let (o, proofs) = elide_same(&m, f, main);
        assert_eq!(checks(&o, f), 0, "{name}");
        assert_eq!(proofs.len(), 1, "{name}: {proofs:?}");
    }
}

/// `a[i & a.len()]`: at most the length, not below it.
fn masked_by_own_len(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    over_eight(b, t, x, |b, t, x, i| b.binary(Op::And, t.us, i, x.na))
}

/// `a[i & b.len()]`: the other slice's length.
fn masked_by_other_len(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    over_eight(b, t, x, |b, t, x, i| b.binary(Op::And, t.us, i, x.nb))
}

/// `a[i % a.len() + 1]`: one past the remainder's bound.
fn rem_plus_one(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    over_eight(b, t, x, |b, t, x, i| {
        let r = b.binary(Op::Rem, t.us, i, x.na);
        let one = b.int(1);
        b.binary(Op::Add, t.us, r, one)
    })
}

/// `for i in 0..a.len() { a[i << 1] }`: a left shift grows the value.
fn shl_under_len(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let z = b.int(0);
    for_range(b, t, z, x.na.into(), |b, i, acc| {
        let one = b.int(1);
        let k = b.binary(Op::Shl, t.us, i, one);
        let q = read(b, t, x.a, x.na, k);
        b.binary(Op::AddWrap, t.u32t, acc, q)
    })
}

/// `a[(i as% u8) as usize]`: a byte is below 256, not below the length.
fn byte_unguarded(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    over_eight(b, t, x, |b, t, _, i| {
        let byte = b.cast(CastKind::Wrap, t.u8t, i);
        b.cast(CastKind::Exact, t.us, byte)
    })
}

/// Look-alikes the proofs must not fire on: each keeps the check that can fail, and still traps
/// `Bounds` where the unoptimized code does.
#[test]
fn look_alikes_keep_their_checks() {
    let cases: [(&str, BodyFn, usize, (u64, u64)); 13] = [
        ("i & len", masked_by_own_len, 1, (4, 0)),
        ("i & other len", masked_by_other_len, 1, (3, 4)),
        ("i % len + 1", rem_plus_one, 1, (3, 0)),
        ("i << 1", shl_under_len, 1, (3, 0)),
        ("u8 index", byte_unguarded, 1, (3, 0)),
        ("le only", le_only, 1, (4, 2)),
        ("eq on one path", eq_on_one_path, 1, (6, 2)),
        ("non-asserting callee", non_asserting_callee, 1, (3, 1)),
        ("non-constant offset", non_constant_offset, 1, (3, 1)),
        ("i + 1 under i < len", plus_one_under_lt_len, 1, (3, 0)),
        ("wrapping offset", wrapping_offset, 1, (3, 0)),
        ("re-sliced after eq", resliced_after_eq, 1, (3, 3)),
        ("signed index", signed_index, 1, (3, 0)),
    ];
    for (name, body, kept, (na, nb)) in cases {
        let (m, f, main) = two_slices(body);
        let (o, _) = elide_same(&m, f, main);
        assert_eq!(checks(&o, f), kept, "{name}");
        assert_eq!(trap(&o, main, na, nb), Some(PanicKind::Bounds), "{name}");
        assert!(forge::forgeries_are_refused(&m, f) > 0, "{name}");
    }
}

/// Guards `x0 < x1 < .. < xk <= a.len()` around `a[x0]`: proven when the chain is short, given up
/// (check kept) when it is longer than `BOUNDS_STEP_LIMIT` edges.
#[test]
fn search_stops_at_the_step_limit() {
    for (len, kept) in [(3usize, 0usize), (BOUNDS_STEP_LIMIT as usize + 8, 1)] {
        let mut m = Module::new();
        let t = stypes(&mut m);
        let mut params = vec![t.psl];
        params.extend(std::iter::repeat_n(t.us, len + 1));
        let f = define(&mut m, &[1], "app.chain", &params, Some(t.u32t), |b, _| {
            let a = b.param(0).unwrap();
            let xs: Vec<Value> = (0..=len).map(|k| b.param(k + 1).unwrap()).collect();
            let na = b.op(Op::SliceLen, t.us, &[a.into()]);
            let (out, ov) = b.create_block_with(&[t.u32t]);
            let z = b.int(0);
            for w in xs.windows(2) {
                let c = b.cmp(Op::Lt, t.bt, w[0], w[1]);
                let go = b.create_block();
                b.cbr(c, go, &[], out, &[z]);
                b.switch_to(go);
            }
            let c = b.cmp(Op::Le, t.bt, xs[len], na);
            let go = b.create_block();
            b.cbr(c, go, &[], out, &[z]);
            b.switch_to(go);
            let v = read(b, &t, a, na, xs[0]);
            b.br(out, &[v.into()]);
            b.switch_to(out);
            b.ret(Some(ov[0].into()));
        });
        verify_module(&m).unwrap();
        let e = elide(&mut m).unwrap();
        assert_eq!(checks(&m, f), kept, "chain of {len}");
        if kept == 0 {
            assert!(matches!(e[..], [Elided { proof: Proof::Offset { k: 0, .. }, .. }]), "{e:?}");
        }
        audit(&m).unwrap();
    }
}

// Vector accesses under `while` tests: `u32x4.load(s, i)` is `check.bounds i (s.len() -| 3)`.

/// The front end's `u32x4.load(s, idx)` (`n` = `slice.len s`), reduced to its lane sum.
fn vload(b: &mut FuncBuilder, t: &S, s: Value, n: Value, idx: impl Into<Operand>) -> Value {
    let idx = idx.into();
    let three = b.int(3);
    let lim = b.binary(Op::SubSat, t.us, n, three);
    b.bounds_check(idx, lim);
    let v = b.op(Op::Vec(VecOp::Load), t.v4, &[s.into(), idx]);
    b.op(Op::Vec(VecOp::ReduceAddWrap), t.u32t, &[v.into()])
}

/// `let mut i = from; while test(i) { acc += step(i, acc); i = i + 4; }` (a trapping `add`),
/// from `acc = 0`; returns the final `acc`, the builder in the exit block.
fn while_step4(
    b: &mut FuncBuilder,
    t: &S,
    from: i128,
    test: impl FnOnce(&mut FuncBuilder, Value) -> Value,
    step: impl FnOnce(&mut FuncBuilder, Value, Value) -> Value,
) -> Value {
    let (head, hv) = b.create_block_with(&[t.us, t.u32t]);
    let body = b.create_block();
    let (exit, ev) = b.create_block_with(&[t.u32t]);
    let (i0, z) = (b.int(from), b.int(0));
    b.br(head, &[i0, z]);
    b.switch_to(head);
    let c = test(b, hv[0]);
    b.cbr(c, body, &[], exit, &[hv[1].into()]);
    b.switch_to(body);
    let acc = step(b, hv[0], hv[1]);
    let four = b.int(4);
    let i2 = b.binary(Op::Add, t.us, hv[0], four);
    b.br(head, &[i2.into(), acc.into()]);
    b.switch_to(exit);
    ev[0]
}

/// `i + k <= n` with `add` (`wrap`: `add.wrap`).
fn plus_le(b: &mut FuncBuilder, t: &S, i: Value, k: i128, n: Value, wrap: bool) -> Value {
    let k = b.int(k);
    let s = b.binary(if wrap { Op::AddWrap } else { Op::Add }, t.us, i, k);
    b.cmp(Op::Le, t.bt, s, n)
}

fn add_u32(b: &mut FuncBuilder, t: &S, x: Value, y: Value) -> Value {
    b.binary(Op::AddWrap, t.u32t, x, y)
}

/// `while i + 4 <= a.len() { acc += sum(u32x4.load(a, i)) }`.
fn vec_while_le(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    while_step4(
        b,
        t,
        0,
        |b, i| plus_le(b, t, i, 4, x.na, false),
        |b, i, acc| {
            let v = vload(b, t, x.a, x.na, i);
            add_u32(b, t, acc, v)
        },
    )
}

/// The same loop testing `i < a.len() -| 3`, written out.
fn vec_while_lt_sat(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let test = |b: &mut FuncBuilder, i: Value| {
        let three = b.int(3);
        let lim = b.binary(Op::SubSat, t.us, x.na, three);
        b.cmp(Op::Lt, t.bt, i, lim)
    };
    while_step4(b, t, 0, test, |b, i, acc| {
        let v = vload(b, t, x.a, x.na, i);
        add_u32(b, t, acc, v)
    })
}

/// `while i + 8 <= a.len() { acc += sum(u32x4.load(a, i + 4)) }`: the second half of a wider step.
fn vec_while_upper_half(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    while_step4(
        b,
        t,
        0,
        |b, i| plus_le(b, t, i, 8, x.na, false),
        |b, i, acc| {
            let four = b.int(4);
            let j = b.binary(Op::Add, t.us, i, four);
            let v = vload(b, t, x.a, x.na, j);
            add_u32(b, t, acc, v)
        },
    )
}

/// The scalar shape `while i + 4 <= a.len() { acc += a[i] + a[i + 3] }`.
fn scalar_while_le(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    while_step4(
        b,
        t,
        0,
        |b, i| plus_le(b, t, i, 4, x.na, false),
        |b, i, acc| {
            let p = read(b, t, x.a, x.na, i);
            let three = b.int(3);
            let j = b.binary(Op::Add, t.us, i, three);
            let q = read(b, t, x.a, x.na, j);
            let s = add_u32(b, t, p, q);
            add_u32(b, t, acc, s)
        },
    )
}

/// `while i + 4 <= a.len()`, the vector load and the scalar reads of every lane all proven, and
/// the result, the traps, and the audit unchanged.
#[test]
fn vector_and_scalar_while_le_have_no_check() {
    let cases: [(&str, BodyFn, usize, i128); 4] = [
        ("i + 4 <= len", vec_while_le, 1, 3),
        ("i < len -| 3", vec_while_lt_sat, 1, 3),
        ("i + 8 <= len, load at i + 4", vec_while_upper_half, 1, 7),
        ("scalar i + 4 <= len", scalar_while_le, 2, 3),
    ];
    for (name, body, n, last) in cases {
        let (m, f, main) = two_slices(body);
        assert_eq!(checks(&m, f), n, "{name}");
        let (o, proofs) = elide_same(&m, f, main);
        assert_eq!(checks(&o, f), 0, "{name}: {proofs:?}");
        assert!(
            matches!(proofs.last(), Some(Proof::Offset { base: Some(_), k }) if *k == last),
            "{name}: {proofs:?}"
        );
    }
    let (m, f, main) = two_slices(vec_while_le);
    let (o, _) = elide_same(&m, f, main);
    // 1 + 2 + .. + 8: two whole vectors of the eight elements
    assert_eq!(run(&o, main, vec![Val::S(8), Val::S(0)]), Outcome::Return(Some(Val::S(36))));
    assert_eq!(run(&o, main, vec![Val::S(7), Val::S(0)]), Outcome::Return(Some(Val::S(10))));
}

/// `while i +% 4 <= a.len()` from `i = usize::MAX - 1`: the wrapped sum passes the test.
fn vec_wrapping_test(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let from = i128::from(u64::MAX) - 1;
    while_step4(
        b,
        t,
        from,
        |b, i| plus_le(b, t, i, 4, x.na, true),
        |b, i, acc| {
            let v = vload(b, t, x.a, x.na, i);
            add_u32(b, t, acc, v)
        },
    )
}

/// `while i <= a.len() -| 3`: `0 <= n -| 3` holds for every `n`.
fn vec_le_sat(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    let test = |b: &mut FuncBuilder, i: Value| {
        let three = b.int(3);
        let lim = b.binary(Op::SubSat, t.us, x.na, three);
        b.cmp(Op::Le, t.bt, i, lim)
    };
    while_step4(b, t, 0, test, |b, i, acc| {
        let v = vload(b, t, x.a, x.na, i);
        add_u32(b, t, acc, v)
    })
}

/// `while i + 3 <= a.len()` around a load of 4 lanes.
fn vec_test_shorter_than_lanes(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    while_step4(
        b,
        t,
        0,
        |b, i| plus_le(b, t, i, 3, x.na, false),
        |b, i, acc| {
            let v = vload(b, t, x.a, x.na, i);
            add_u32(b, t, acc, v)
        },
    )
}

/// `while i + 4 <= b.len()` around a load from `a`.
fn vec_other_slice(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    while_step4(
        b,
        t,
        0,
        |b, i| plus_le(b, t, i, 4, x.nb, false),
        |b, i, acc| {
            let v = vload(b, t, x.a, x.na, i);
            add_u32(b, t, acc, v)
        },
    )
}

/// `while i + 4 <= a.len()` around a load from `&a[0..a.len() - 1]`, made in the body.
fn vec_shorter_slice_in_body(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    while_step4(
        b,
        t,
        0,
        |b, i| plus_le(b, t, i, 4, x.na, false),
        |b, i, acc| {
            let (zero, one) = (b.int(0), b.int(1));
            let base = b.op(Op::ElemAddr, t.pce, &[x.a.into(), zero]);
            let n1 = b.binary(Op::Sub, t.us, x.na, one);
            let c = b.op(Op::SliceMake, t.psl, &[base.into(), n1.into()]);
            let nc = b.op(Op::SliceLen, t.us, &[c.into()]);
            let v = vload(b, t, c, nc, i);
            add_u32(b, t, acc, v)
        },
    )
}

/// `while i + 4 <= a.len()` around a load at `i + 1`.
fn vec_index_moved(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    while_step4(
        b,
        t,
        0,
        |b, i| plus_le(b, t, i, 4, x.na, false),
        |b, i, acc| {
            let one = b.int(1);
            let j = b.binary(Op::Add, t.us, i, one);
            let v = vload(b, t, x.a, x.na, j);
            add_u32(b, t, acc, v)
        },
    )
}

/// `while i + 4 <= a.len() { acc += a[i + 4] }`: `<=` proves `i + 3`, not `i + 4`.
fn scalar_one_past(b: &mut FuncBuilder, t: &S, x: &Two) -> Value {
    while_step4(
        b,
        t,
        0,
        |b, i| plus_le(b, t, i, 4, x.na, false),
        |b, i, acc| {
            let four = b.int(4);
            let j = b.binary(Op::Add, t.us, i, four);
            let q = read(b, t, x.a, x.na, j);
            add_u32(b, t, acc, q)
        },
    )
}

/// Look-alikes of the `while i + N <= len` proofs: each keeps its check and traps `Bounds` where
/// the unoptimized code does.
#[test]
fn while_le_look_alikes_keep_their_checks() {
    let cases: [(&str, BodyFn, (u64, u64)); 7] = [
        ("wrapping add in the test", vec_wrapping_test, (8, 0)),
        ("<= against len -| 3", vec_le_sat, (2, 0)),
        ("test shorter than the lanes", vec_test_shorter_than_lanes, (3, 0)),
        ("another slice's length", vec_other_slice, (2, 4)),
        ("shorter slice made in the body", vec_shorter_slice_in_body, (4, 0)),
        ("index moved after the test", vec_index_moved, (4, 0)),
        ("scalar one past the test", scalar_one_past, (4, 0)),
    ];
    for (name, body, (na, nb)) in cases {
        let (m, f, main) = two_slices(body);
        let (o, proofs) = elide_same(&m, f, main);
        assert_eq!(checks(&o, f), 1, "{name}: {proofs:?}");
        assert_eq!(trap(&o, main, na, nb), Some(PanicKind::Bounds), "{name}");
        assert!(forge::forgeries_are_refused(&m, f) > 0, "{name}");
    }
}
