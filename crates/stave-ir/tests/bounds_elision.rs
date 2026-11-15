//! Bounds-check audit and elision (`stave_ir::bounds`): every element access
//! keeps its check unless a proof is recorded; elision removes only proven checks, records each,
//! keeps the audit passing, and never changes a result or a trap (checked in the interpreter).

mod common;

use common::*;
use stave_ir::bounds::{AccessKind, Guard, Proof, audit, elide};
use stave_ir::interp::Val;
use stave_ir::{FuncId, Module, Op, Outcome, PanicKind, TypeId, verify_module};

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
    let e = elide(&mut m);
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
    let e = elide(&mut m);
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
    assert!(elide(&mut m).is_empty());
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
        elide(&mut m);
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
