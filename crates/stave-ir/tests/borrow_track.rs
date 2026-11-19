//! Borrow tracking of the interpreter on hand-built bodies: a marked instruction result is a new
//! reference, its tag travels with the word (registers, memory, casts, arithmetic), and a load or
//! store through it is checked against contracts/borrow-model.md.

mod common;

use common::*;
use stave_ir::borrowtrack::{Cause, Class, Kind, Use};
use stave_ir::interp::{self, BorrowReport, Config, Outcome, Val};
use stave_ir::{BorrowMark, CastKind, FuncBuilder, FuncId, Module, Op, TypeId, Value};

struct Tys {
    u32t: TypeId,
    u64t: TypeId,
    shared: TypeId,
    excl: TypeId,
}

/// A fn `() -> u32` built by `body`, which returns the marks of its borrows.
fn program(body: impl FnOnce(&mut FuncBuilder, &mut Module, &Tys) -> Vec<BorrowMark>) -> (Module, FuncId) {
    let mut m = Module::new();
    let u32t = m.types.u32();
    let u64t = m.types.u64();
    let shared = m.types.ref_(false, u32t);
    let excl = m.types.ref_(true, u32t);
    let t = Tys { u32t, u64t, shared, excl };
    let f = m.declare(inst(&[0]), "t.f", vec![], Some(u32t));
    let mut b = FuncBuilder::new(&[]);
    let marks = body(&mut b, &mut m, &t);
    m.define(f, b.finish().expect("builder"));
    m.funcs[f.0 as usize].borrows = marks;
    (m, f)
}

fn run_with(m: &Module, f: FuncId, borrow: bool) -> Outcome {
    let cfg = Config { check: true, borrow, ..Config::default() };
    match interp::run(m, f, vec![], host(), cfg).0 {
        Ok(o) => o,
        Err(e) => panic!("interpreter error: {e:?}"),
    }
}

fn violation(m: &Module, f: FuncId) -> Box<BorrowReport> {
    match run_with(m, f, true) {
        Outcome::BorrowError(r) => r,
        o => panic!("expected a borrow violation, got {o:?}"),
    }
}

fn mark(value: Value, mutable: bool) -> BorrowMark {
    BorrowMark { value, mutable, size: 4 }
}

fn put(b: &mut FuncBuilder, p: Value, v: i128) {
    let c = b.int(v);
    b.store(p, c);
}

/// `let mut x = 1; let r = &x; x = 2; return *r;`
fn stale(b: &mut FuncBuilder, t: &Tys) -> Vec<BorrowMark> {
    let x = b.slot(t.excl);
    put(b, x, 1);
    let r = b.cast(CastKind::Exact, t.shared, x);
    put(b, x, 2);
    let v = b.load(t.u32t, r);
    b.ret(Some(v.into()));
    vec![mark(r, false)]
}

#[test]
fn write_under_a_shared_borrow_used_later() {
    let (m, f) = program(|b, _, t| stale(b, t));
    let r = violation(&m, f);
    assert_eq!(
        (r.report.class, r.report.event, r.report.kind),
        (Class::WriteUnderShared, Use::Read, Kind::Shared)
    );
    assert_eq!((r.report.cause, r.report.lo, r.report.hi), (Cause::Write, 0, 4));
    assert_eq!(r.at.first().map(|a| a.inst), Some(4));
    assert_eq!(r.cause_at.map(|a| a.inst), Some(3));
    assert_eq!(r.made_at.map(|a| a.inst), Some(2));
    assert!(r.render(&m).starts_with("borrow violation [write-under-shared] M1"));
}

#[test]
fn marks_are_inert_without_the_mode() {
    let (m, f) = program(|b, _, t| stale(b, t));
    assert_eq!(run_with(&m, f, false), Outcome::Return(Some(Val::S(2))));
    assert_eq!(run(&m, f, vec![]), Outcome::Return(Some(Val::S(2))));
}

#[test]
fn two_exclusive_borrows_both_used() {
    let (m, f) = program(|b, _, t| {
        let x = b.slot(t.excl);
        let p = b.cast(CastKind::Exact, t.excl, x);
        let q = b.cast(CastKind::Exact, t.excl, x);
        put(b, p, 1);
        put(b, q, 2);
        let v = b.load(t.u32t, x);
        b.ret(Some(v.into()));
        vec![mark(p, true), mark(q, true)]
    });
    let r = violation(&m, f);
    assert_eq!((r.report.class, r.report.event, r.report.kind), (Class::TwoExclusive, Use::Write, Kind::Mut));
    assert_eq!(r.at.first().map(|a| a.inst), Some(4));
    assert_eq!(r.cause_at.map(|a| a.inst), Some(3));
    assert_eq!(r.made_at.map(|a| a.inst), Some(2));
}

#[test]
fn a_borrow_alone_is_no_access() {
    let (m, f) = program(|b, _, t| {
        let x = b.slot(t.excl);
        put(b, x, 1);
        let r = b.cast(CastKind::Exact, t.shared, x);
        let p = b.cast(CastKind::Exact, t.excl, x);
        let a = b.load(t.u32t, r);
        let c = b.load(t.u32t, p);
        let s = b.binary(Op::Add, t.u32t, a, c);
        put(b, x, 5);
        b.ret(Some(s.into()));
        vec![mark(r, false), mark(p, true)]
    });
    assert_eq!(run_with(&m, f, true), Outcome::Return(Some(Val::S(2))));
}

#[test]
fn a_reborrow_dies_with_a_write_through_its_parent() {
    let (m, f) = program(|b, _, t| {
        let x = b.slot(t.excl);
        let p = b.cast(CastKind::Exact, t.excl, x);
        let c = b.cast(CastKind::Exact, t.excl, p);
        put(b, c, 1);
        put(b, p, 2);
        put(b, c, 3);
        let v = b.load(t.u32t, p);
        b.ret(Some(v.into()));
        vec![mark(p, true), mark(c, true)]
    });
    let r = violation(&m, f);
    assert_eq!((r.report.class, r.report.event), (Class::TwoExclusive, Use::Write));
    assert_eq!(
        (r.at.first().map(|a| a.inst), r.cause_at.map(|a| a.inst), r.made_at.map(|a| a.inst)),
        (Some(5), Some(4), Some(2))
    );
}

#[test]
fn disjoint_elements_never_meet() {
    let (m, f) = program(|b, m, t| {
        let arr = m.types.array(t.u32t, 2);
        let parr = m.types.ref_(true, arr);
        let x = b.slot(parr);
        let zero = b.int(0);
        let one = b.int(1);
        let e0 = b.binary(Op::ElemAddr, t.excl, x, zero);
        let e1 = b.binary(Op::ElemAddr, t.excl, x, one);
        put(b, e0, 1);
        put(b, e1, 2);
        let v = b.load(t.u32t, e0);
        b.ret(Some(v.into()));
        vec![mark(e0, true), mark(e1, true)]
    });
    assert_eq!(run_with(&m, f, true), Outcome::Return(Some(Val::S(1))));
}

#[test]
fn a_stored_reference_keeps_its_tag() {
    let (m, f) = program(|b, m, t| {
        let pp = m.types.ref_(true, t.shared);
        let x = b.slot(t.excl);
        let cell = b.slot(pp);
        put(b, x, 1);
        let r = b.cast(CastKind::Exact, t.shared, x);
        b.store(cell, r);
        put(b, x, 2);
        let r2 = b.load(t.shared, cell);
        let v = b.load(t.u32t, r2);
        b.ret(Some(v.into()));
        vec![mark(r, false)]
    });
    let r = violation(&m, f);
    assert_eq!(
        (r.report.class, r.report.event, r.report.cause),
        (Class::WriteUnderShared, Use::Read, Cause::Write)
    );
    assert_eq!(
        (r.at.first().map(|a| a.inst), r.cause_at.map(|a| a.inst), r.made_at.map(|a| a.inst)),
        (Some(7), Some(5), Some(3))
    );
}

#[test]
fn an_integer_round_trip_keeps_the_tag() {
    let (m, f) = program(|b, _, t| {
        let x = b.slot(t.excl);
        put(b, x, 1);
        let r = b.cast(CastKind::Exact, t.shared, x);
        let n = b.cast(CastKind::Exact, t.u64t, r);
        let zero = b.int(0);
        let n = b.binary(Op::AddWrap, t.u64t, n, zero);
        let back = b.cast(CastKind::Exact, t.shared, n);
        put(b, x, 2);
        let v = b.load(t.u32t, back);
        b.ret(Some(v.into()));
        vec![mark(r, false)]
    });
    let r = violation(&m, f);
    assert_eq!((r.report.class, r.made_at.map(|a| a.inst)), (Class::WriteUnderShared, Some(2)));
}
