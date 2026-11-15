//! Verifier: every class of error is reported (and nothing panics).

mod common;

use common::*;
use stave_ir::ir::{BlockData, Insts, Span, Term, ValueData, ValueDef};
use stave_ir::{Body, BuildError, FuncBuilder, Module, Op, Operand, TypeKey, Value, verify_module};

fn errs(m: &Module) -> Vec<String> {
    match verify_module(m) {
        Ok(()) => vec![],
        Err(e) => e.iter().map(|e| e.to_string()).collect(),
    }
}

fn has(m: &Module, needle: &str) {
    let e = errs(m);
    assert!(e.iter().any(|s| s.contains(needle)), "no error containing {needle:?} in {e:#?}");
}

#[test]
fn use_not_dominated() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let bt = m.types.bool();
    define(&mut m, &[0], "t.f", &[i32t, bt], Some(i32t), |b, _| {
        let (x, c) = (b.param(0).unwrap(), b.param(1).unwrap());
        let (t, e, j) = (b.create_block(), b.create_block(), b.create_block());
        b.cbr(c, t, &[], e, &[]);
        b.switch_to(t);
        let y = b.binary(Op::Add, i32t, x, x);
        b.br(j, &[]);
        b.switch_to(e);
        b.br(j, &[]);
        b.switch_to(j);
        b.ret(Some(y.into())); // y defined only on the then path
    });
    has(&m, "not dominated");
}

#[test]
fn use_before_def_in_block() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let f = m.declare(inst(&[0]), "t.f", vec![i32t], Some(i32t));
    let mut b = FuncBuilder::new(&[i32t]);
    let x = b.param(0).unwrap();
    let y = b.binary(Op::Add, i32t, x, x);
    let z = b.binary(Op::Add, i32t, y, x);
    b.ret(Some(z.into()));
    let mut body = b.finish().unwrap();
    // swap the operands of the first add to use the second's result
    let s = body.insts.args[0];
    body.operands[s.start as usize] = Operand::Value(z);
    m.define(f, body);
    has(&m, "not dominated");
}

#[test]
fn block_param_arity_and_types() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let i64t = m.types.i64();
    define(&mut m, &[0], "t.f", &[i32t], None, |b, _| {
        let x = b.param(0).unwrap();
        let (j, _) = b.create_block_with(&[i64t]);
        b.br(j, &[x.into()]);
        b.switch_to(j);
        b.ret(None);
    });
    has(&m, "branch to b1: argument 0");
    let mut m = Module::new();
    let i32t = m.types.i32();
    define(&mut m, &[0], "t.f", &[i32t], None, |b, _| {
        let (j, _) = b.create_block_with(&[i32t, i32t]);
        let k = b.int(1);
        b.br(j, &[k]);
        b.switch_to(j);
        b.ret(None);
    });
    has(&m, "1 arguments, expected 2");
}

#[test]
fn branch_to_entry() {
    let mut m = Module::new();
    define(&mut m, &[0], "t.f", &[], None, |b, _| {
        let e = b.entry();
        b.br(e, &[]);
    });
    has(&m, "entry");
}

#[test]
fn slot_outside_entry() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let p = m.types.ref_(true, i32t);
    define(&mut m, &[0], "t.f", &[], None, |b, _| {
        let n = b.create_block();
        b.br(n, &[]);
        b.switch_to(n);
        b.inst(Op::Slot, Some(p), &[]);
        b.ret(None);
    });
    has(&m, "slot outside b0");
    let mut m = Module::new();
    let i32t = m.types.i32();
    define(&mut m, &[0], "t.f", &[], None, |b, _| {
        b.slot(i32t); // not a pointer type
        b.ret(None);
    });
    has(&m, "slot result must be a pointer");
}

#[test]
fn constant_ranges() {
    let mut m = Module::new();
    let i8t = m.types.i8();
    let bt = m.types.bool();
    let u8t = m.types.u8();
    let f32t = m.types.f32();
    define(&mut m, &[0], "t.f", &[], Some(i8t), |b, _| {
        let (k, one) = (b.int(128), b.int(1));
        let _ = b.binary(Op::Add, i8t, k, one);
        let (t, f) = (b.int(2), b.int(0));
        let _ = b.select(i8t, t, f, f);
        let (neg, big) = (b.int(-1), b.int(0));
        let _ = b.binary(Op::Add, u8t, neg, big);
        let fl = b.f64(1.0);
        let fz = b.f32(0.0);
        let _ = b.binary(Op::Add, f32t, fl, fz);
        let _ = bt;
        let r = b.int(-128);
        b.ret(Some(r));
    });
    has(&m, "constant 128 out of range");
    has(&m, "constant -1 out of range");
    has(&m, "argument 0: constant Some(Int(2)) cannot have type");
    has(&m, "F64");
    assert_eq!(errs(&m).len(), 4, "{:#?}", errs(&m));
}

#[test]
fn terminator_errors() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let i64t = m.types.i64();
    let _bt = m.types.bool();
    define(&mut m, &[0], "t.a", &[i32t], Some(i64t), |b, _| {
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
    });
    has(&m, "ret: %0 has type");
    let mut m = Module::new();
    let i32t = m.types.i32();
    define(&mut m, &[0], "t.b", &[], Some(i32t), |b, _| b.ret(None));
    has(&m, "ret without a value");
    let mut m = Module::new();
    let i32t = m.types.i32();
    let _bt = m.types.bool();
    define(&mut m, &[0], "t.c", &[i32t], None, |b, _| {
        let x = b.param(0).unwrap();
        let (t, e) = (b.create_block(), b.create_block());
        b.cbr(x, t, &[], e, &[]);
        for blk in [t, e] {
            b.switch_to(blk);
            b.ret(None);
        }
    });
    has(&m, "cbr condition");
    let mut m = Module::new();
    let u8t = m.types.u8();
    define(&mut m, &[0], "t.d", &[u8t], None, |b, _| {
        let x = b.param(0).unwrap();
        let (t, e) = (b.create_block(), b.create_block());
        b.switch(x, &[(1, t, &[]), (1, t, &[]), (256, t, &[])], e, &[]);
        for blk in [t, e] {
            b.switch_to(blk);
            b.ret(None);
        }
    });
    has(&m, "duplicate switch case 1");
}

#[test]
fn call_errors() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let u8t = m.types.u8();
    let g = m.declare(inst(&[1]), "t.g", vec![i32t, i32t], Some(i32t));
    define(&mut m, &[0], "t.f", &[], None, |b, _| {
        let k = b.int(1);
        b.call(g, Some(i32t), &[k]);
        let (k1, k2) = (b.int(1), b.int(2));
        b.call(g, Some(u8t), &[k1, k2]);
        let k3 = b.int(3);
        b.call_ind(k3, None, &[]);
        b.ret(None);
    });
    has(&m, "1 arguments, expected 2");
    has(&m, "call result type differs");
    has(&m, "call.ind callee must be a value");
}

#[test]
fn intrinsic_signatures() {
    let mut m = Module::new();
    let usize_t = m.types.usize();
    let isize_t = m.types.isize();
    let i32t = m.types.i32();
    let s2 = m.intrinsic("syscall2");
    let bad = m.intrinsic("");
    define(&mut m, &[0], "t.f", &[i32t], None, |b, _| {
        let x = b.param(0).unwrap();
        let nr = b.int(1);
        b.intrinsic(s2, Some(isize_t), &[nr, x.into(), x.into()]); // i32 arg
        let nr = b.int(1);
        b.intrinsic(s2, Some(usize_t), &[nr, nr, nr]); // wrong result + arity
        b.intrinsic(bad, None, &[]);
        b.ret(None);
    });
    has(&m, "intrinsic.syscall2: argument 1");
    has(&m, "intrinsic.syscall2: wrong result type");
    has(&m, "empty intrinsic name");
}

#[test]
fn op_typing_errors() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let u32t = m.types.u32();
    let bt = m.types.bool();
    let f32t = m.types.f32();
    let p = m.types.ref_(false, i32t);
    define(&mut m, &[0], "t.f", &[i32t, u32t, f32t, p], None, |b, _| {
        let ps = b.params().to_vec();
        b.binary(Op::Add, i32t, ps[0], ps[1]); // mixed types
        b.unary(Op::Neg, u32t, ps[1]); // neg on unsigned
        b.binary(Op::AddWrap, f32t, ps[2], ps[2]); // wrap on float
        b.cmp(Op::Lt, i32t, ps[0], ps[0]); // non-bool compare result
        b.load(u32t, ps[3]); // wrong pointee
        b.store(ps[0], ps[0]); // store through a non-pointer
        b.cast(stave_ir::CastKind::Exact, i32t, ps[1]); // u32 as i32 is lossy
        b.op(Op::Get, i32t, &[ps[0].into(), ps[0].into()]); // get of a scalar
        let _ = bt;
        b.ret(None);
    });
    for n in [
        "%1 has type",
        "neg on",
        "int op on",
        "compare result must be bool",
        "load of",
        "store through a non-pointer",
        "Exact cast",
        "field of",
    ] {
        has(&m, n);
    }
}

#[test]
fn aggregate_typing_errors() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let pair = m.types.tuple(vec![i32t, i32t]);
    let arr = m.types.array(i32t, 2);
    let e =
        m.types.adt(stave_ir::DeclId(1), vec![], stave_ir::AdtShape::enumeration(vec![vec![], vec![i32t]]));
    define(&mut m, &[0], "t.f", &[pair, arr, e], None, |b, _| {
        let ps = b.params().to_vec();
        let k = b.int(1);
        b.op(Op::Make, pair, &[k]); // arity
        let k5 = b.int(5);
        b.op(Op::Get, i32t, &[ps[0].into(), k5]); // field out of range
        let k2 = b.int(2);
        b.op(Op::Get, i32t, &[ps[1].into(), k2]); // array const index out of range
        let (v, f) = (b.int(0), b.int(0));
        b.op(Op::EnumGet, i32t, &[ps[2].into(), v, f]); // variant 0 has no fields
        let v9 = b.int(9);
        b.op(Op::EnumMake, e, &[v9]);
        b.ret(None);
    });
    has(&m, "1 arguments, expected 2");
    has(&m, "field index out of range");
    has(&m, "array index constant out of range");
    has(&m, "variant index out of range");
}

#[test]
fn builder_misuse_is_an_error_not_a_panic() {
    let mut b = FuncBuilder::new(&[]);
    b.ret(None);
    b.ret(None);
    assert_eq!(b.finish().unwrap_err(), BuildError::Terminated(stave_ir::Block(0)));
    let mut b = FuncBuilder::new(&[]);
    let _ = b.create_block();
    b.ret(None);
    assert_eq!(b.finish().unwrap_err(), BuildError::Unterminated(stave_ir::Block(1)));
    let mut b = FuncBuilder::new(&[]);
    b.switch_to(stave_ir::Block(7));
    b.ret(None);
    assert_eq!(b.finish().unwrap_err(), BuildError::UnknownBlock(stave_ir::Block(7)));
}

#[test]
fn malformed_bodies_do_not_panic() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let f = m.declare(inst(&[0]), "t.f", vec![i32t], Some(i32t));
    // empty body
    m.define(f, Body::default());
    has(&m, "no blocks");
    // spans out of range, dangling values
    let body = Body {
        blocks: vec![BlockData {
            params: Span { start: 0, len: 3 },
            insts: Span { start: 0, len: 1 },
            term: Term::Ret(Some(Operand::Value(Value(99)))),
        }],
        block_params: vec![Value(0)],
        insts: Insts {
            op: vec![Op::Add],
            ty: vec![Some(stave_ir::TypeId(77))],
            result: vec![Some(Value(5))],
            args: vec![Span { start: 10, len: 2 }],
            site: vec![stave_ir::Site::NONE],
        },
        values: vec![ValueData { ty: i32t, def: ValueDef::Inst(3) }],
        ..Body::default()
    };
    m.define(f, body);
    assert!(!errs(&m).is_empty());
    // lengths differ
    let body = Body {
        blocks: vec![BlockData {
            params: Span::default(),
            insts: Span { start: 0, len: 1 },
            term: Term::Unreachable,
        }],
        insts: Insts { op: vec![Op::Add], ..Insts::default() },
        ..Body::default()
    };
    m.define(f, body);
    has(&m, "differ in length");
    // printer and interpreter also survive it
    let _ = stave_ir::print_checked(&m);
    let (r, _) = stave_ir::interp::run(&m, f, vec![], host(), stave_ir::Config::default());
    assert!(r.is_err());
}

#[test]
fn bad_type_table() {
    let mut m = Module::new();
    m.types.intern(TypeKey::Ref { mutable: false, inner: stave_ir::TypeId(5) });
    has(&m, "type table");
    // a cycle via direct interning of a self-referential key
    let mut m = Module::new();
    m.types.intern(TypeKey::Array { elem: stave_ir::TypeId(1), len: 1 });
    m.types.intern(TypeKey::Tuple(vec![stave_ir::TypeId(0)]));
    has(&m, "Cycle");
    // too deep
    let mut m = Module::new();
    let mut t = m.types.i32();
    for _ in 0..5000 {
        t = m.types.ref_(false, t);
    }
    has(&m, "TooDeep");
    assert!(m.canonicalize_types().is_err());
}

#[test]
fn infinite_struct_is_a_layout_error() {
    let mut m = Module::new();
    let s = m.types.intern(TypeKey::Adt { def: stave_ir::DeclId(1), args: vec![] });
    m.types.set_shape(s, stave_ir::AdtShape::structure(vec![s]));
    let ps = m.types.ref_(true, s);
    let f = define(&mut m, &[0], "t.f", &[], None, |b, _| {
        b.slot(ps);
        b.ret(None);
    });
    assert!(verify_module(&m).is_ok());
    let (r, _) = stave_ir::interp::run(&m, f, vec![], host(), stave_ir::Config::default());
    assert!(
        matches!(r, Err(stave_ir::InterpError::Layout { err: stave_ir::layout::LayoutError::Infinite, .. })),
        "{r:?}"
    );
}

#[test]
fn canonical_type_order() {
    let mut m = Module::new();
    let t = &mut m.types;
    // interned in scrambled order
    let b = t.bool();
    let s64 = t.i64();
    let u8t = t.u8();
    let r = t.ref_(true, s64);
    let usz = t.usize();
    let s8 = t.i8();
    let unit = t.unit();
    let tup = t.tuple(vec![u8t, b]);
    let tup2 = t.tuple(vec![u8t]);
    let f64t = t.f64();
    let f32t = t.f32();
    let rc = t.ref_(false, s64);
    let _ = (r, usz, s8, unit, tup, tup2, f64t, f32t, rc);
    m.canonicalize_types().unwrap();
    let texts: Vec<String> =
        (0..m.types.len()).map(|i| m.types.key_text(stave_ir::TypeId(i as u32))).collect();
    assert_eq!(
        texts,
        [
            "int u 8",
            "int u size",
            "int s 8",
            "int s 64",
            "float 32",
            "float 64",
            "bool",
            "ref t3",
            "ref mut t3",
            "tuple",
            "tuple t0",
            "tuple t0 t6",
        ]
    );
    // interning after canonicalization still dedups
    assert_eq!(m.types.bool(), stave_ir::TypeId(6));
}

/// A type's text with component ids expanded (independent of the numbering).
fn deep_text(types: &stave_ir::TypeTable, t: stave_ir::TypeId) -> String {
    match types.key(t) {
        Some(TypeKey::Tuple(ts)) => {
            let parts: Vec<String> = ts.iter().map(|&x| deep_text(types, x)).collect();
            format!("({})", parts.join(", "))
        }
        Some(TypeKey::Array { elem, len }) => format!("[{}; {len}]", deep_text(types, *elem)),
        _ => types.key_text(t),
    }
}

/// Static types are remapped exactly once, with or without functions in the module.
#[test]
fn canonicalize_remaps_statics_once() {
    for nfuncs in [0usize, 1, 3] {
        let mut m = Module::new();
        // interned so that canonical renumbering moves both types
        let b = m.types.bool();
        let tup = m.types.tuple(vec![b]);
        let u8t = m.types.u8();
        let arr = m.types.array(u8t, 3);
        m.statics.push(stave_ir::Static { path: "t.A".into(), ty: tup, init: vec![0] });
        m.statics.push(stave_ir::Static { path: "t.B".into(), ty: arr, init: vec![1, 2, 3] });
        for k in 0..nfuncs {
            define(&mut m, &[k as u32], &format!("t.f{k}"), &[], None, |b, _| b.ret(None));
        }
        let want: Vec<String> = m.statics.iter().map(|s| deep_text(&m.types, s.ty)).collect();
        m.canonicalize_types().unwrap();
        let got: Vec<String> = m.statics.iter().map(|s| deep_text(&m.types, s.ty)).collect();
        assert_eq!(got, want, "{nfuncs} fns");
    }
}

/// `cast` accepts a fn item (its code address) but no fn value and no fn-type target.
#[test]
fn cast_of_fn_values_is_rejected() {
    let mut m = Module::new();
    let us = m.types.usize();
    let u64t = m.types.u64();
    let fnt = m.types.fn_type(vec![], u64t, stave_target::FeatureSet::EMPTY);
    let g = define(&mut m, &[0], "t.g", &[], Some(u64t), |b, _| {
        let k = b.int(5);
        b.ret(Some(k));
    });
    // `@g as usize` verifies and runs: the interpreter's code address of `@g` is its index
    let f = define(&mut m, &[1], "t.f", &[], Some(us), |b, _| {
        let fg = b.func(g);
        let x = b.cast(stave_ir::CastKind::Exact, us, fg);
        b.ret(Some(x.into()));
    });
    assert_eq!(errs(&m), Vec::<String>::new());
    let v = ret_val(run(&m, f, vec![]));
    assert!(matches!(v, stave_ir::Val::S(x) if x == u64::from(g.0)), "{v:?}");
    // a fn value operand
    let mut m2 = m.clone();
    define(&mut m2, &[2], "t.h", &[fnt], Some(us), |b, _| {
        let p = b.param(0).unwrap();
        let x = b.cast(stave_ir::CastKind::Wrap, us, p);
        b.ret(Some(x.into()));
    });
    has(&m2, "cast of a fn value");
    // a fn-type target
    let mut m3 = m.clone();
    define(&mut m3, &[3], "t.k", &[us], Some(fnt), |b, _| {
        let p = b.param(0).unwrap();
        let x = b.cast(stave_ir::CastKind::Exact, fnt, p);
        b.ret(Some(x.into()));
    });
    has(&m3, "cast to");
}
