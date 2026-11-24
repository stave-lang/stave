//! Golden tests of the checked-IR printer (contracts/checked-ir.md) over hand-built IR that
//! follows the contract's lowering rules (block order, SSA rule, slots). Each golden is also run
//! in the interpreter where it computes something.

mod common;

use common::*;
use stave_ir::interp::Val;
use stave_ir::{
    AdtShape, CastKind, DeclId, FuncBuilder, GenericArg, Module, Op, Operand, PanicKind, Site, SiteInfo,
    TypeKey, print_checked,
    types::{AdtKind, DeclInfo},
    verify_module,
};

/// Prints `m` and compares with `tests/print_golden/<name>.ir` (`STAVE_BLESS=1` rewrites it).
fn check(m: &mut Module, name: &str) {
    m.canonicalize_types().expect("canonical");
    if let Err(e) = verify_module(m) {
        panic!("verify: {e:#?}");
    }
    let got = print_checked(m).unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/print_golden")
        .join(format!("{name}.ir"));
    if std::env::var_os("STAVE_BLESS").is_some() {
        std::fs::write(&path, &got).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(got, expected, "\n--- got ---\n{got}\n--- expected ---\n{expected}");
}

/// The contract's own example (the first complete file in checked-ir.md), built by hand.
#[test]
fn contract_example() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let bt = m.types.bool();
    m.sites.push(SiteInfo { file: "src/app.stv".into(), line: 1, col: 4, func: "app.max".into() });
    let f = m.declare(inst(&[0]), "app.max", vec![i32t, i32t], Some(i32t));
    m.func_mut(f).unwrap().name_site = Site(0);
    let mut b = FuncBuilder::new(&[i32t, i32t]);
    let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
    let then = b.create_block();
    let els = b.create_block();
    let c = b.cmp(Op::Gt, bt, x, y);
    b.cbr(c, then, &[], els, &[]);
    b.switch_to(then);
    b.ret(Some(x.into()));
    b.switch_to(els);
    b.ret(Some(y.into()));
    m.define(f, b.finish().unwrap());
    verify_module(&m).unwrap();
    let doc = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/checked-ir.md"),
    )
    .unwrap();
    let start = doc.find("```\nstave.ir 4 ").expect("the contract example") + 4;
    let example = &doc[start..start + doc[start..].find("```").unwrap()];
    assert_eq!(print_checked(&m).unwrap(), example);
    assert_eq!(ret_val(run(&m, f, vec![Val::S(3), Val::S(9)])), Val::S(9));
}

/// `fn abs(x: i32) -> i32 { let mut r = x; if x < 0 { r = -x; } else { r = x; } return r; }`
/// if statement: then, else, join; one join param for `r`. Values are created out of print order.
#[test]
fn if_else() {
    let mut m = Module::new();
    let bt = m.types.bool();
    let i32t = m.types.i32();
    let f = m.declare(inst(&[1]), "app.abs", vec![i32t], Some(i32t));
    let mut b = FuncBuilder::new(&[i32t]);
    let x = b.param(0).unwrap();
    let then = b.create_block();
    let els = b.create_block();
    let (join, jp) = b.create_block_with(&[i32t]); // join param created before `neg`
    let z = b.int(0);
    let c = b.cmp(Op::Lt, bt, x, z);
    b.cbr(c, then, &[], els, &[]);
    b.switch_to(then);
    let n = b.unary(Op::Neg, i32t, x);
    b.br(join, &[n.into()]);
    b.switch_to(els);
    b.br(join, &[x.into()]);
    b.switch_to(join);
    b.ret(Some(jp[0].into()));
    m.define(f, b.finish().unwrap());
    check(&mut m, "if_else");
    assert_eq!(ret_val(run(&m, f, vec![Val::S((-5i64) as u64)])), Val::S(5));
    assert!(matches!(run(&m, f, vec![Val::S(i32::MIN as i64 as u64)]), stave_ir::Outcome::Trap(_)));
}

/// ```text
/// fn sum(n: u64) -> u64 {
///     let mut i: u64 = 0; let mut s: u64 = 0;
///     loop { if i == n { break; } s = s + i; i = i + 1; }
///     return s;
/// }
/// ```
/// loop: body (b1, carries i and s), exit (b2, carries s: live after the loop); the `if` inside:
/// then (b3), join (b4). Params by local index: n, i, s.
#[test]
fn loop_with_carried_local() {
    let mut m = Module::new();
    let u64t = m.types.u64();
    let bt = m.types.bool();
    let f = m.declare(inst(&[2]), "app.sum", vec![u64t], Some(u64t));
    let mut b = FuncBuilder::new(&[u64t]);
    let n = b.param(0).unwrap();
    let (body, bp) = b.create_block_with(&[u64t, u64t]);
    let (exit, ep) = b.create_block_with(&[u64t]);
    let then = b.create_block();
    let join = b.create_block();
    let z = b.int(0);
    let z2 = b.int(0);
    b.br(body, &[z, z2]);
    b.switch_to(body);
    let (i, s) = (bp[0], bp[1]);
    let c = b.cmp(Op::Eq, bt, i, n);
    b.cbr(c, then, &[], join, &[]);
    b.switch_to(then);
    b.br(exit, &[s.into()]);
    b.switch_to(join);
    let s2 = b.binary(Op::Add, u64t, s, i);
    let one = b.int(1);
    let i2 = b.binary(Op::Add, u64t, i, one);
    b.br(body, &[i2.into(), s2.into()]);
    b.switch_to(exit);
    b.ret(Some(ep[0].into()));
    m.define(f, b.finish().unwrap());
    check(&mut m, "loop_with_carried_local");
    assert_eq!(ret_val(run(&m, f, vec![Val::S(100)])), Val::S(4950));
}

/// ```text
/// fn count(n: u32) -> u32 {
///     let mut i: u32 = 0; let mut c: u32 = 0;
///     while i < n {
///         i = i + 1;
///         if i % 3 == 0 { continue; }
///         if c == 10 { break; }
///         c = c + 1;
///     }
///     return c;
/// }
/// ```
/// while: head (b1: i, c), body (b2), exit (b3: c); ifs: then b4, join b5; then b6, join b7.
#[test]
fn while_break_continue() {
    let mut m = Module::new();
    let bt = m.types.bool();
    let u32t = m.types.u32();
    let f = m.declare(inst(&[3]), "app.count", vec![u32t], Some(u32t));
    let mut b = FuncBuilder::new(&[u32t]);
    let n = b.param(0).unwrap();
    let (head, hp) = b.create_block_with(&[u32t, u32t]);
    let body = b.create_block();
    let (exit, ep) = b.create_block_with(&[u32t]);
    let then1 = b.create_block();
    let join1 = b.create_block();
    let then2 = b.create_block();
    let join2 = b.create_block();
    let (z, z2) = (b.int(0), b.int(0));
    b.br(head, &[z, z2]);
    b.switch_to(head);
    let (i, c) = (hp[0], hp[1]);
    let lt = b.cmp(Op::Lt, bt, i, n);
    b.cbr(lt, body, &[], exit, &[c.into()]);
    b.switch_to(body);
    let one = b.int(1);
    let i1 = b.binary(Op::Add, u32t, i, one);
    let three = b.int(3);
    let r = b.binary(Op::Rem, u32t, i1, three);
    let z3 = b.int(0);
    let e = b.cmp(Op::Eq, bt, r, z3);
    b.cbr(e, then1, &[], join1, &[]);
    b.switch_to(then1);
    b.br(head, &[i1.into(), c.into()]);
    b.switch_to(join1);
    let ten = b.int(10);
    let e2 = b.cmp(Op::Eq, bt, c, ten);
    b.cbr(e2, then2, &[], join2, &[]);
    b.switch_to(then2);
    b.br(exit, &[c.into()]);
    b.switch_to(join2);
    let one2 = b.int(1);
    let c1 = b.binary(Op::Add, u32t, c, one2);
    b.br(head, &[i1.into(), c1.into()]);
    b.switch_to(exit);
    b.ret(Some(ep[0].into()));
    m.define(f, b.finish().unwrap());
    check(&mut m, "while_break_continue");
    assert_eq!(ret_val(run(&m, f, vec![Val::S(9)])), Val::S(6));
    assert_eq!(ret_val(run(&m, f, vec![Val::S(100)])), Val::S(10));
}

/// ```text
/// enum Shape { Circle(f64), Rect { w: f64, h: f64 } }
/// fn area(s: Shape) -> f64 {
///     return match s { Shape.Circle(r) => r * r, Shape.Rect.{ w, h } => w * h };
/// }
/// ```
/// match expression: arm 0 test (b1), body (b2); arm 1 test (b3), body (b4); join (b5, value).
/// The last arm's test is the switch default.
#[test]
fn match_to_switch() {
    let mut m = Module::new();
    let f64t = m.types.f64();
    let shape = m.types.adt(DeclId(5), vec![], AdtShape::enumeration(vec![vec![f64t], vec![f64t, f64t]]));
    m.decls.push(DeclInfo { id: DeclId(5), kind: AdtKind::Enum, path: "app.Shape".into() });
    let u32t = m.types.u32();
    let f = m.declare(inst(&[4]), "app.area", vec![shape], Some(f64t));
    let mut b = FuncBuilder::new(&[shape]);
    let s = b.param(0).unwrap();
    let t0 = b.create_block();
    let b0 = b.create_block();
    let t1 = b.create_block();
    let b1 = b.create_block();
    let (join, jp) = b.create_block_with(&[f64t]);
    let tag = b.op(Op::EnumTag, u32t, &[s.into()]);
    b.switch(tag, &[(0, t0, &[])], t1, &[]);
    b.switch_to(t0);
    let (v0, f0) = (b.int(0), b.int(0));
    let r = b.op(Op::EnumGet, f64t, &[s.into(), v0, f0]);
    b.br(b0, &[]);
    b.switch_to(b0);
    let rr = b.binary(Op::Mul, f64t, r, r);
    b.br(join, &[rr.into()]);
    b.switch_to(t1);
    let (v1, f1) = (b.int(1), b.int(0));
    let w = b.op(Op::EnumGet, f64t, &[s.into(), v1, f1]);
    let (v1b, f1b) = (b.int(1), b.int(1));
    let h = b.op(Op::EnumGet, f64t, &[s.into(), v1b, f1b]);
    b.br(b1, &[]);
    b.switch_to(b1);
    let wh = b.binary(Op::Mul, f64t, w, h);
    b.br(join, &[wh.into()]);
    b.switch_to(join);
    b.ret(Some(jp[0].into()));
    m.define(f, b.finish().unwrap());
    check(&mut m, "match_to_switch");
    let circle = Val::agg(vec![Val::S(0), Val::f64(3.0)]);
    let rect = Val::agg(vec![Val::S(1), Val::f64(2.0), Val::f64(3.5)]);
    assert_eq!(ret_val(run(&m, f, vec![circle])), Val::f64(9.0));
    assert_eq!(ret_val(run(&m, f, vec![rect])), Val::f64(7.0));
}

/// ```text
/// fn bump(p: &mut i32) { *p = *p + 1; }
/// fn f() -> i32 {
///     let mut a: i32 = 1; let b: i32 = 5; let mut c: i32 = 2;
///     bump(&mut a); bump(&mut c);
///     return a + b + c;
/// }
/// ```
/// `a` and `c` are address-taken: slots in b0 in declaration order (placed first in b0 even when
/// created lazily); `b` is an SSA value (no instruction).
#[test]
fn slots_for_address_taken_locals() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let rt = m.types.ref_(true, i32t);
    // declared in reverse to check print order by instance id
    let fid = m.declare(inst(&[6]), "app.f", vec![], Some(i32t));
    let bump = m.declare(inst(&[5]), "app.bump", vec![rt], None);
    let mut b = FuncBuilder::new(&[rt]);
    let p = b.param(0).unwrap();
    let x = b.load(i32t, p);
    let one = b.int(1);
    let y = b.binary(Op::Add, i32t, x, one);
    b.store(p, y);
    b.ret(None);
    m.define(bump, b.finish().unwrap());

    let mut b = FuncBuilder::new(&[]);
    let a = b.slot(rt);
    let one = b.int(1);
    b.store(a, one);
    let c = b.slot(rt);
    let two = b.int(2);
    b.store(c, two);
    b.call(bump, None, &[a.into()]);
    b.call(bump, None, &[c.into()]);
    let av = b.load(i32t, a);
    let five = b.int(5);
    let ab = b.binary(Op::Add, i32t, av, five);
    let cv = b.load(i32t, c);
    let sum = b.binary(Op::Add, i32t, ab, cv);
    b.ret(Some(sum.into()));
    m.define(fid, b.finish().unwrap());
    check(&mut m, "slots_for_address_taken_locals");
    assert_eq!(ret_val(run(&m, fid, vec![])), Val::S(10));
}

/// Direct and indirect calls, fn values, and closures `@7.0` printed right after `@7`.
#[test]
fn calls_and_closure_order() {
    let mut m = Module::new();
    let i64t = m.types.i64();
    let fnt = m.types.fn_type(vec![i64t], i64t, stave_target::FeatureSet::EMPTY);
    let main = m.declare(inst(&[8]), "app.main", vec![], Some(i64t));
    let clo = m.declare(inst(&[7, 0]), "app.apply", vec![i64t], Some(i64t));
    let apply = m.declare(inst(&[7]), "app.apply", vec![fnt, i64t], Some(i64t));
    let inc = m.declare(inst(&[2]), "app.inc", vec![i64t], Some(i64t));

    let mut b = FuncBuilder::new(&[i64t]);
    let x = b.param(0).unwrap();
    let one = b.int(1);
    let r = b.binary(Op::Add, i64t, x, one);
    b.ret(Some(r.into()));
    m.define(inc, b.finish().unwrap());

    let mut b = FuncBuilder::new(&[fnt, i64t]);
    let (fv, x) = (b.param(0).unwrap(), b.param(1).unwrap());
    let r = b.call_ind(fv, Some(i64t), &[x.into()]).unwrap();
    let r2 = b.call(clo, Some(i64t), &[r.into()]).unwrap();
    b.ret(Some(r2.into()));
    m.define(apply, b.finish().unwrap());

    let mut b = FuncBuilder::new(&[i64t]);
    let y = b.param(0).unwrap();
    let two = b.int(2);
    let r = b.binary(Op::MulWrap, i64t, y, two);
    b.ret(Some(r.into()));
    m.define(clo, b.finish().unwrap());

    let mut b = FuncBuilder::new(&[]);
    let k = b.int(40);
    let r = b.call(apply, Some(i64t), &[Operand::Func(inc), k]).unwrap();
    b.ret(Some(r.into()));
    m.define(main, b.finish().unwrap());
    check(&mut m, "calls_and_closure_order");
    assert_eq!(ret_val(run(&m, main, vec![])), Val::S(82));
}

/// An intrinsic call (`intrinsic.syscall3`), a string literal operand, fat-pointer parts, and
/// pointer / int casts. Runs and writes to stdout.
#[test]
fn intrinsic_call() {
    let mut m = Module::new();
    let str_t = m.types.str();
    let sref = m.types.ref_(false, str_t);
    let isize_t = m.types.isize();
    let i32t = m.types.i32();
    let u8t = m.types.u8();
    let p8 = m.types.ptr(false, u8t);
    let usize_t = m.types.usize();
    let s0 = m.add_string(b"hi\n".to_vec());
    let sys3 = m.intrinsic("syscall3");
    let write = m.declare(inst(&[5]), "runtime.os.write", vec![i32t, sref], Some(isize_t));
    let main = m.declare(inst(&[6]), "app.main", vec![], Some(i32t));

    let mut b = FuncBuilder::new(&[i32t, sref]);
    let (fd, s) = (b.param(0).unwrap(), b.param(1).unwrap());
    let p = b.op(Op::SlicePtr, p8, &[s.into()]);
    let pa = b.cast(CastKind::Exact, usize_t, p);
    let n = b.op(Op::SliceLen, usize_t, &[s.into()]);
    let fdu = b.cast(CastKind::Wrap, usize_t, fd);
    let nr = b.int(1);
    let r = b.intrinsic(sys3, Some(isize_t), &[nr, fdu.into(), pa.into(), n.into()]).unwrap();
    b.ret(Some(r.into()));
    m.define(write, b.finish().unwrap());

    let mut b = FuncBuilder::new(&[]);
    let one = b.int(1);
    let st = b.str(s0);
    b.call(write, Some(isize_t), &[one, st]);
    let z = b.int(0);
    b.ret(Some(z));
    m.define(main, b.finish().unwrap());
    check(&mut m, "intrinsic_call");
    let (o, h) = run_host(&m, main, vec![]);
    assert_eq!(ret_val(o), Val::S(0));
    assert_eq!(h.stdout, b"hi\n");
}

/// Trapping ops are single instructions (no blocks): `add`, `div`, `shl`, `check.bounds`, a
/// dynamic array `get`, plus an explicit `panic.unwrap`. Also floats as IEEE bits.
#[test]
fn trap_ops() {
    let mut m = Module::new();
    let bt = m.types.bool();
    let arr = {
        let i32t = m.types.i32();
        m.types.array(i32t, 4)
    };
    let i32t = m.types.i32();
    let usize_t = m.types.usize();
    let f32t = m.types.f32();
    let f = m.declare(inst(&[9]), "app.traps", vec![i32t, i32t, arr, usize_t], Some(i32t));
    let mut b = FuncBuilder::new(&[i32t, i32t, arr, usize_t]);
    let p = b.params().to_vec();
    let s = b.binary(Op::Add, i32t, p[0], p[1]);
    let d = b.binary(Op::Div, i32t, s, p[1]);
    let sh = b.binary(Op::Shl, i32t, d, p[3]);
    let four = b.int(4);
    b.bounds_check(p[3], four);
    let e = b.op(Op::Get, i32t, &[p[2].into(), p[3].into()]);
    let w = b.binary(Op::AddWrap, i32t, sh, e);
    let two = b.int(2);
    let ms = b.binary(Op::MulSat, i32t, w, two);
    let seven = b.int(7);
    let r = b.binary(Op::Rem, i32t, ms, seven);
    let fl = b.cast(CastKind::Exact, f32t, p[0]);
    let half = b.f32(0.5);
    let _ = b.cmp(Op::Lt, bt, fl, half);
    let z = b.int(0);
    let c = b.cmp(Op::Eq, bt, r, z);
    let (pan, ok) = (b.create_block(), b.create_block());
    b.cbr(c, pan, &[], ok, &[]);
    b.switch_to(pan);
    b.panic(PanicKind::Unwrap);
    b.unreachable();
    b.switch_to(ok);
    b.ret(Some(r.into()));
    m.define(f, b.finish().unwrap());
    // i32 -> f32 is not lossless: the verifier rejects plain `as`; switch the op to `as~`.
    assert!(verify_module(&m).is_err());
    let body = m.funcs[f.index()].body.as_mut().unwrap();
    let k = body.insts.op.iter().position(|o| *o == Op::Cast(CastKind::Exact)).unwrap();
    body.insts.op[k] = Op::Cast(CastKind::Round);
    check(&mut m, "trap_ops");
    let xs = |v: [i64; 4]| Val::agg(v.iter().map(|x| Val::S(*x as u64)).collect());
    let si = |x: i64| Val::S(x as u64);
    // (3 + 4) / 4 = 1; 1 << 1 = 2; xs[1] = 5 → 7; 7 *| 2 = 14; 14 % 7 = 0 → panic.unwrap
    let o = run(&m, f, vec![si(3), si(4), xs([0, 5, 0, 0]), Val::S(1)]);
    assert!(matches!(o, stave_ir::Outcome::Trap(t) if t.kind == PanicKind::Unwrap));
    // (1 + 1) / 1 = 2; 2 << 0 = 2; + 1 = 3; *2 = 6; 6 % 7 = 6
    assert_eq!(ret_val(run(&m, f, vec![si(1), si(1), xs([1, 0, 0, 0]), Val::S(0)])), si(6));
    let trap = |args| match run(&m, f, args) {
        stave_ir::Outcome::Trap(t) => t.kind,
        o => panic!("{o:?}"),
    };
    assert_eq!(trap(vec![si(i32::MAX as i64), si(1), xs([0; 4]), Val::S(0)]), PanicKind::Overflow);
    assert_eq!(trap(vec![si(1), si(0), xs([0; 4]), Val::S(0)]), PanicKind::DivByZero);
    assert_eq!(trap(vec![si(1), si(1), xs([0; 4]), Val::S(32)]), PanicKind::Overflow);
    assert_eq!(trap(vec![si(1), si(1), xs([0; 4]), Val::S(4)]), PanicKind::Bounds);
}

/// Blocks unreachable from b0 are deleted and the rest renumbered; their values are not numbered.
#[test]
fn unreachable_blocks_dropped() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let f = m.declare(inst(&[1]), "app.g", vec![i32t], Some(i32t));
    let mut b = FuncBuilder::new(&[i32t]);
    let x = b.param(0).unwrap();
    let dead = b.create_block();
    let live = b.create_block();
    b.br(live, &[]);
    b.switch_to(dead);
    let one = b.int(1);
    let y = b.binary(Op::Add, i32t, x, one);
    b.ret(Some(y.into()));
    b.switch_to(live);
    let two = b.int(2);
    let z = b.binary(Op::Mul, i32t, x, two);
    b.ret(Some(z.into()));
    m.define(f, b.finish().unwrap());
    check(&mut m, "unreachable_blocks_dropped");
}

/// Instance type args and every TypeKey form.
#[test]
fn type_key_text() {
    let mut m = Module::new();
    let t = &mut m.types;
    let i32t = t.i32();
    let u8t = t.u8();
    let us = t.usize();
    let k = |t: &mut stave_ir::TypeTable, key| {
        let id = t.intern(key);
        t.key_text(id)
    };
    assert_eq!(k(t, TypeKey::Int { signed: false, bits: stave_ir::IntBits::Size }), "int u size");
    assert_eq!(k(t, TypeKey::Float(stave_ir::FloatBits::B16)), "float b16");
    assert_eq!(k(t, TypeKey::Char), "char");
    assert_eq!(k(t, TypeKey::Ref { mutable: true, inner: i32t }), format!("ref mut t{}", i32t.0));
    assert_eq!(k(t, TypeKey::Ref { mutable: false, inner: i32t }), format!("ref t{}", i32t.0));
    assert_eq!(k(t, TypeKey::RawPtr { mutable: false, inner: u8t }), format!("ptr const t{}", u8t.0));
    assert_eq!(k(t, TypeKey::RawPtr { mutable: true, inner: u8t }), format!("ptr mut t{}", u8t.0));
    assert_eq!(k(t, TypeKey::Array { elem: u8t, len: 16 }), format!("array t{} 16", u8t.0));
    assert_eq!(k(t, TypeKey::Slice { elem: u8t, align: Some(32) }), format!("slice t{} align 32", u8t.0));
    assert_eq!(k(t, TypeKey::Slice { elem: u8t, align: None }), format!("slice t{}", u8t.0));
    assert_eq!(k(t, TypeKey::Tuple(vec![])), "tuple");
    assert_eq!(k(t, TypeKey::Tuple(vec![i32t, us])), format!("tuple t{} t{}", i32t.0, us.0));
    assert_eq!(
        k(t, TypeKey::Adt { def: DeclId(17), args: vec![GenericArg::Type(i32t), GenericArg::Const(8)] }),
        format!("adt d17 t{} 8", i32t.0)
    );
    let avx2 = stave_target::FeatureSet::parse_list("avx2").unwrap();
    assert_eq!(
        k(t, TypeKey::FnType { params: vec![i32t, u8t], ret: us, features: avx2 }),
        format!("fntype t{} t{} -> t{} avx2", i32t.0, u8t.0, us.0)
    );
    let v4 = stave_target::FeatureSet::parse_list("avx512f,fma,bmi2").unwrap();
    assert_eq!(
        k(t, TypeKey::FnType { params: vec![], ret: us, features: v4 }),
        format!("fntype -> t{} bmi2 avx512f", us.0)
    );
    assert_eq!(k(t, TypeKey::Vec { lane: i32t, n: stave_ir::Lanes::Fixed(8) }), format!("vec t{} 8", i32t.0));
    assert_eq!(k(t, TypeKey::Vec { lane: i32t, n: stave_ir::Lanes::Native }), format!("vec t{} n", i32t.0));
    assert_eq!(k(t, TypeKey::Mask { width: 32, n: stave_ir::Lanes::Fixed(8) }), "mask 32 8");
    assert_eq!(k(t, TypeKey::Mask { width: 16, n: stave_ir::Lanes::Native }), "mask 16 n");
    assert_eq!(k(t, TypeKey::Param(0)), "param 0");
    assert_eq!(k(t, TypeKey::ConstParam(2)), "constparam 2");
}

/// A generic instance prints its type and const args; types used only by keys are listed too.
#[test]
fn instance_type_args() {
    let mut m = Module::new();
    let u8t = m.types.u8();
    let arr = m.types.array(u8t, 3);
    let r = m.types.ref_(false, arr);
    let f = m.declare(inst(&[12]), "std.mem.first", vec![r], Some(u8t));
    m.funcs[f.index()].type_args = vec![GenericArg::Type(u8t), GenericArg::Type(arr), GenericArg::Const(3)];
    let mut b = FuncBuilder::new(&[r]);
    let p = b.param(0).unwrap();
    let v = b.load(arr, p);
    let z = b.int(0);
    let e = b.op(Op::Get, u8t, &[v.into(), z]);
    b.ret(Some(e.into()));
    m.define(f, b.finish().unwrap());
    check(&mut m, "instance_type_args");
}
