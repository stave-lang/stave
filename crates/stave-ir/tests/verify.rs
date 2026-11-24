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
fn variadic_import_calls() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let u8t = m.types.u8();
    let f32t = m.types.f32();
    let p = m.types.ptr(false, u8t);
    let g = m.declare(inst(&[1]), "t.printf", vec![p], Some(i32t));
    m.func_mut(g).unwrap().symbol = Some("printf".to_string());
    m.func_mut(g).unwrap().variadic = true;
    define(&mut m, &[0], "t.f", &[p, i32t], None, |b, _| {
        let (s, x) = (b.param(0).unwrap(), b.param(1).unwrap());
        let (k, d) = (b.int(7), b.f64(1.5));
        b.call(g, Some(i32t), &[s.into(), x.into(), k, d, s.into()]);
        b.call(g, Some(i32t), &[s.into()]);
        b.ret(None);
    });
    assert_eq!(errs(&m), Vec::<String>::new());
    define(&mut m, &[2], "t.h", &[p, u8t, f32t], None, |b, _| {
        let (s, c, x) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
        b.call(g, Some(i32t), &[s.into(), c.into()]);
        b.call(g, Some(i32t), &[s.into(), x.into()]);
        b.call(g, Some(i32t), &[]);
        b.ret(None);
    });
    let e = errs(&m);
    let bad = e.iter().filter(|s| s.contains("variadic argument 1: not a C variadic type")).count();
    assert_eq!(bad, 2, "{e:#?}");
    has(&m, "0 arguments, expected 1");
}

#[test]
fn compare_of_two_fn_items() {
    let mut m = Module::new();
    let bt = m.types.bool();
    let g = m.declare(inst(&[1]), "t.g", vec![], None);
    let h = m.declare(inst(&[2]), "t.h", vec![], None);
    define(&mut m, &[0], "t.f", &[], None, |b, _| {
        b.cmp(Op::Eq, bt, Operand::Func(g), Operand::Func(h));
        b.cmp(Op::Ne, bt, Operand::Func(g), Operand::Func(g));
        b.ret(None);
    });
    assert_eq!(errs(&m), Vec::<String>::new());
    define(&mut m, &[3], "t.k", &[], None, |b, _| {
        b.cmp(Op::Lt, bt, Operand::Func(g), Operand::Func(h));
        b.ret(None);
    });
    has(&m, "compare of mismatched constants");
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
    has(&m, "unknown intrinsic ``");
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

/// A type that contains itself by value is a verifier error, used or not; the interpreter, which
/// verifies what it runs, refuses it before running.
#[test]
fn infinite_struct_is_rejected() {
    let mut m = Module::new();
    let s = m.types.intern(TypeKey::Adt { def: stave_ir::DeclId(1), args: vec![] });
    m.types.set_shape(s, stave_ir::AdtShape::structure(vec![s]));
    let ps = m.types.ref_(true, s);
    let f = define(&mut m, &[0], "t.f", &[], None, |b, _| {
        b.slot(ps);
        b.ret(None);
    });
    has(&m, "type contains itself without indirection");
    let (r, _) = stave_ir::interp::run(&m, f, vec![], host(), stave_ir::Config::default());
    assert!(matches!(r, Err(stave_ir::InterpError::Verify(_))), "{r:?}");
}

/// A vector type's lane is an int or float lane type and its count a power of two in 2..=64; a
/// type whose size overflows `u64` is an error even unused.
#[test]
fn vector_shapes_and_sizes() {
    use stave_ir::Lanes;
    let base = || {
        let mut m = Module::new();
        let u8t = m.types.u8();
        (m, u8t)
    };
    let (mut m, u8t) = base();
    m.types.intern(TypeKey::Vec { lane: u8t, n: Lanes::Fixed(16) });
    assert!(errs(&m).is_empty(), "{:#?}", errs(&m));
    let (mut m, u8t) = base();
    let huge = m.types.array(u8t, 1 << 62);
    m.types.intern(TypeKey::Vec { lane: huge, n: Lanes::Fixed(8) });
    has(&m, "not a vector or mask shape");
    let (mut m, u8t) = base();
    m.types.intern(TypeKey::Vec { lane: u8t, n: Lanes::Fixed(3) });
    has(&m, "not a vector or mask shape");
    let (mut m, _) = base();
    m.types.intern(TypeKey::Mask { width: 12, n: Lanes::Fixed(4) });
    has(&m, "not a vector or mask shape");
    let (mut m, u8t) = base();
    let big = m.types.array(u8t, 1 << 62);
    m.types.array(big, 8);
    has(&m, "size overflows u64");
}

/// Every site is `Site::NONE` or a row of the site table: the fn's name site and each
/// instruction's.
#[test]
fn sites_in_range() {
    use stave_ir::{Site, SiteInfo};
    let mut m = Module::new();
    m.sites.push(SiteInfo { file: "a.stv".into(), line: 1, col: 1, func: "t.f".into() });
    let f = define(&mut m, &[0], "t.f", &[], None, |b, _| {
        b.set_site(Site(0));
        b.inst(Op::FenceStream, None, &[]);
        b.ret(None);
    });
    m.funcs[f.index()].name_site = Site(0);
    assert!(errs(&m).is_empty(), "{:#?}", errs(&m));
    let mut bad = m.clone();
    bad.funcs[f.index()].name_site = Site(1);
    has(&bad, "name site 1 out of range");
    let mut bad = m.clone();
    bad.funcs[f.index()].body.as_mut().unwrap().insts.site[0] = Site(7);
    has(&bad, "inst 0: site 7 out of range");
}

/// A fn stamped as a known item that a back end or the interpreter calls on its own has that
/// item's signature, and only a panic report entry runs on the alternate stack.
#[test]
fn known_signatures() {
    use stave_ir::Known;
    let mut m = Module::new();
    let u32t = m.types.u32();
    let usize_t = m.types.usize();
    let trap = define_known(&mut m, &[0], Known::Trap, &[u32t, u32t], None, |b, _| {
        b.panic(stave_ir::PanicKind::Assert);
        b.unreachable();
    });
    let ent = define_known(&mut m, &[1], Known::ThreadEntry, &[usize_t], Some(usize_t), |b, _| {
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
    });
    assert!(errs(&m).is_empty(), "{:#?}", errs(&m));
    let mut ok = m.clone();
    ok.funcs[trap.index()].alt_stack = false;
    assert!(errs(&ok).is_empty(), "{:#?}", errs(&ok));
    let mut bad = m.clone();
    bad.funcs[ent.index()].alt_stack = true;
    has(&bad, "alt_stack is true for `runtime.thread.entry`");
    let mut bad = m.clone();
    bad.funcs[trap.index()].params = vec![usize_t, u32t];
    has(&bad, "`runtime.panic.trap` must have the signature");
    let mut bad = m.clone();
    bad.funcs[ent.index()].known = Some(Known::CpuInit);
    has(&bad, "`runtime.cpu.init` must have the signature");
    let mut bad = m.clone();
    bad.funcs[ent.index()].ret = None;
    has(&bad, "`runtime.thread.entry` must have the signature");
}

/// Every known item with an ABI signature is checked against it; the others are not.
#[test]
fn every_abi_signature_is_enforced() {
    use stave_ir::Known;
    let mut checked = 0;
    for &k in Known::ALL {
        let mut m = Module::new();
        let u8t = m.types.u8();
        // Five `u8` parameters match no row.
        define_known(&mut m, &[0], k, &[u8t; 5], None, |b, _| b.unreachable());
        m.funcs[0].alt_stack = false;
        let e = errs(&m);
        let wrong = e.iter().any(|s| s.contains("must have the signature"));
        assert_eq!(wrong, stave_ir::known::abi_signature(k).is_some(), "{k}: {e:#?}");
        checked += usize::from(wrong);
    }
    assert_eq!(checked, 9);
}

/// Each operand and switch case belongs to one span at most.
#[test]
fn shared_spans() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    define(&mut m, &[0], "t.f", &[i32t], Some(i32t), |b, _| {
        let x = b.param(0).unwrap();
        let y = b.binary(Op::Add, i32t, x, x);
        let z = b.binary(Op::Add, i32t, y, x);
        b.ret(Some(z.into()));
    });
    assert!(errs(&m).is_empty(), "{:#?}", errs(&m));
    let body = m.funcs[0].body.as_mut().unwrap();
    body.insts.args[1] = body.insts.args[0];
    has(&m, "operand span shared");
}

/// The constants of `check.bounds` fit `u64`; an element index constant fits 64 bits.
#[test]
fn index_constants_fit_a_word() {
    let mut m = Module::new();
    let u8t = m.types.u8();
    let p = m.types.ptr(false, u8t);
    let one = |k: i128| {
        let mut m = m.clone();
        define(&mut m, &[0], "t.f", &[p], None, |b, _| {
            let x = b.param(0).unwrap();
            let (i, n) = (b.int(k), b.int(1));
            b.inst(Op::BoundsCheck, None, &[i, n]);
            let i = b.int(k);
            b.op(Op::ElemAddr, p, &[x.into(), i]);
            b.ret(None);
        });
        m
    };
    assert!(errs(&one(u64::MAX as i128)).is_empty(), "{:#?}", errs(&one(u64::MAX as i128)));
    has(&one(1 << 64), "check.bounds operands must be unsigned ints");
    has(&one(1 << 64), "element index must be an int (a constant in 64 bits)");
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
        m.statics.push(stave_ir::Static { path: "t.A".into(), ty: tup, init: vec![0], refs: Vec::new() });
        m.statics.push(stave_ir::Static {
            path: "t.B".into(),
            ty: arr,
            init: vec![1, 2, 3],
            refs: Vec::new(),
        });
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

#[test]
fn seal_unreached_closes_only_blocks_out_of_reach() {
    let mut b = FuncBuilder::new(&[]);
    let dead = b.create_block();
    b.unreachable();
    b.seal_unreached();
    let body = b.finish().expect("an open block out of reach is sealed");
    assert!(matches!(body.blocks[dead.index()].term, Term::Unreachable));

    let mut b = FuncBuilder::new(&[]);
    let live = b.create_block();
    b.br(live, &[]);
    b.seal_unreached();
    assert_eq!(b.finish().err(), Some(BuildError::Unterminated(live)));
}

/// A static's address words: 8-aligned, ascending, inside zero bytes of `init`, into an existing
/// string or static, at most one past its end.
#[test]
fn static_address_words() {
    use stave_ir::{DataRef, DataTarget, Static};
    let mut m = Module::new();
    let u64t = m.types.u64();
    let pair = m.types.array(u64t, 2);
    let s = m.add_string(b"abc".to_vec());
    let word = |offset, target, addend| DataRef { offset, target, addend };
    let refs = vec![word(0, DataTarget::Str(s), 3), word(8, DataTarget::Static(0), 16)];
    m.statics.push(Static { path: "t.S".into(), ty: pair, init: vec![0; 16], refs });
    assert!(errs(&m).is_empty(), "{:#?}", errs(&m));
    let bad = |refs: Vec<DataRef>, init: Vec<u8>, needle: &str| {
        let mut m = m.clone();
        m.statics.push(Static { path: "t.bad".into(), ty: pair, init, refs });
        has(&m, needle);
    };
    bad(vec![word(4, DataTarget::Str(s), 0)], vec![0; 16], "misplaced");
    bad(vec![word(8, DataTarget::Str(s), 0), word(8, DataTarget::Str(s), 0)], vec![0; 16], "misplaced");
    bad(vec![word(16, DataTarget::Str(s), 0)], vec![0; 16], "misplaced");
    bad(vec![word(0, DataTarget::Str(s), 0)], vec![1; 16], "misplaced");
    bad(vec![word(0, DataTarget::Str(s), 4)], vec![0; 16], "points past");
    bad(vec![word(0, DataTarget::Static(9), 0)], vec![0; 16], "no target");
}

/// A static's `init` is exactly its type's size; one with address words has a type aligned to
/// a multiple of 8.
#[test]
fn static_size_and_alignment() {
    use stave_ir::{DataRef, DataTarget, Static};
    let mut m = Module::new();
    let u64t = m.types.u64();
    let u8t = m.types.u8();
    let pair = m.types.array(u64t, 2);
    let bytes = m.types.array(u8t, 16);
    let s = m.add_string(b"abc".to_vec());
    let refs = vec![DataRef { offset: 0, target: DataTarget::Str(s), addend: 0 }];
    m.statics.push(Static { path: "t.S".into(), ty: pair, init: vec![0; 16], refs: refs.clone() });
    m.statics.push(Static { path: "t.B".into(), ty: bytes, init: vec![7; 16], refs: Vec::new() });
    assert!(errs(&m).is_empty(), "{:#?}", errs(&m));
    let bad = |st: Static, needle: &str| {
        let mut m = m.clone();
        m.statics.push(st);
        has(&m, needle);
    };
    bad(Static { path: "t.short".into(), ty: pair, init: vec![0; 8], refs: Vec::new() }, "8 initial bytes");
    bad(Static { path: "t.long".into(), ty: u8t, init: vec![0; 2], refs: Vec::new() }, "2 initial bytes");
    bad(Static { path: "t.byte_words".into(), ty: bytes, init: vec![0; 16], refs }, "aligned to 1");
}

/// An intrinsic instruction is checked against the one table: a name the table lacks, one
/// without a checked signature, and a wrong arity, operand, or result are errors.
#[test]
fn intrinsic_table() {
    let mut m = Module::new();
    let usize_t = m.types.usize();
    let u32t = m.types.u32();
    let u64t = m.types.u64();
    let cpu = m.types.tuple(vec![u32t; 4]);
    let one = |name: &str, ty: Option<stave_ir::TypeId>, n: usize| {
        let mut m = m.clone();
        let id = m.intrinsic(name);
        define(&mut m, &[0], "t.f", &[], None, |b, _| {
            let args: Vec<Operand> = (0..n).map(|_| b.int(1)).collect();
            b.intrinsic(id, ty, &args);
            b.ret(None);
        });
        m
    };
    for m in [one("cpuid", Some(cpu), 2), one("atomic_cas_u64", Some(u64t), 3), one("cpu_init", None, 0)] {
        assert!(errs(&m).is_empty(), "{:#?}", errs(&m));
    }
    assert!(errs(&one("ccall12", Some(usize_t), 13)).is_empty());
    has(&one("frobnicate", Some(usize_t), 0), "unknown intrinsic `frobnicate`");
    has(&one("syscall7", Some(usize_t), 8), "unknown intrinsic `syscall7`");
    has(&one("syscall03", Some(usize_t), 4), "unknown intrinsic `syscall03`");
    has(&one("tid_x", Some(u32t), 0), "no checked signature");
    has(&one("cpu_baseline", Some(u64t), 0), "lowering replaces it");
    has(&one("cpu_init", Some(u64t), 0), "intrinsic.cpu_init: must have no result");
    has(&one("main_addr", None, 0), "intrinsic.main_addr: wrong result type");
    has(&one("call_on_stack", Some(usize_t), 2), "intrinsic.call_on_stack: 2 arguments, expected 3");
    has(&one("cpuid", Some(cpu), 1), "intrinsic.cpuid: 1 arguments, expected 2");
    has(&one("cpuid", Some(usize_t), 2), "intrinsic.cpuid: wrong result type");
}

/// `static_addr <i>`: one int constant naming an existing static, and a raw-pointer result to
/// its type.
#[test]
fn static_addr_operands() {
    use stave_ir::Static;
    let mut m = Module::new();
    let u64t = m.types.u64();
    let u32t = m.types.u32();
    let pu64 = m.types.ptr(true, u64t);
    let pu32 = m.types.ptr(true, u32t);
    let ru64 = m.types.ref_(true, u64t);
    m.statics.push(Static { path: "t.S".into(), ty: u64t, init: vec![0; 8], refs: Vec::new() });
    let one = |k: i128, ty, extra: bool| {
        let mut m = m.clone();
        let id = m.intrinsic(stave_ir::intrinsics::STATIC_ADDR);
        define(&mut m, &[0], "t.f", &[], None, |b, _| {
            let mut args = vec![b.int(k)];
            if extra {
                args.push(b.int(0));
            }
            b.intrinsic(id, Some(ty), &args);
            b.ret(None);
        });
        m
    };
    assert!(errs(&one(0, pu64, false)).is_empty(), "{:#?}", errs(&one(0, pu64, false)));
    has(&one(1, pu64, false), "static 1 out of range");
    has(&one(-1, pu64, false), "static -1 out of range");
    has(&one(0, pu32, false), "must be a raw pointer to t");
    has(&one(0, ru64, false), "must be a raw pointer to t");
    has(&one(0, pu64, true), "2 operands, expected 1");
}
