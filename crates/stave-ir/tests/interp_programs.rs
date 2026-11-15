//! Whole-program behaviour of the interpreter: calls, recursion, memory, aggregates, host
//! syscalls, trap reports, limits.

mod common;

use common::*;
use stave_ir::interp::{self, Config, Interp, InterpError, Outcome, Val};
use stave_ir::{AdtShape, DeclId, FuncBuilder, Module, Op, Operand, PanicKind, Site, Trap};

#[test]
fn recursion_fib() {
    let mut m = Module::new();
    let u64t = m.types.u64();
    let bt = m.types.bool();
    let fib = m.declare(inst(&[1]), "app.fib", vec![u64t], Some(u64t));
    let mut b = FuncBuilder::new(&[u64t]);
    let n = b.param(0).unwrap();
    let (base, rec) = (b.create_block(), b.create_block());
    let two = b.int(2);
    let c = b.cmp(Op::Lt, bt, n, two);
    b.cbr(c, base, &[], rec, &[]);
    b.switch_to(base);
    b.ret(Some(n.into()));
    b.switch_to(rec);
    let one = b.int(1);
    let n1 = b.binary(Op::Sub, u64t, n, one);
    let two = b.int(2);
    let n2 = b.binary(Op::Sub, u64t, n, two);
    let a = b.call(fib, Some(u64t), &[n1.into()]).unwrap();
    let c2 = b.call(fib, Some(u64t), &[n2.into()]).unwrap();
    let s = b.binary(Op::Add, u64t, a, c2);
    b.ret(Some(s.into()));
    m.define(fib, b.finish().unwrap());
    assert_eq!(ret_val(run(&m, fib, vec![Val::S(20)])), Val::S(6765));
}

#[test]
fn trap_reports_site_and_fn() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let callee = m.declare(inst(&[2]), "app.boom", vec![i32t], Some(i32t));
    let main = m.declare(inst(&[1]), "app.main", vec![], Some(i32t));
    let mut b = FuncBuilder::new(&[i32t]);
    let x = b.param(0).unwrap();
    b.set_site(Site(41));
    let one = b.int(1);
    let y = b.binary(Op::Add, i32t, x, one);
    b.set_site(Site(42));
    let z = b.int(0);
    let q = b.binary(Op::Div, i32t, y, z);
    b.ret(Some(q.into()));
    m.define(callee, b.finish().unwrap());
    let mut b = FuncBuilder::new(&[]);
    b.set_site(Site(7));
    let k = b.int(i32::MAX as i128);
    let r = b.call(callee, Some(i32t), &[k]).unwrap();
    b.ret(Some(r.into()));
    m.define(main, b.finish().unwrap());
    assert_eq!(
        run(&m, main, vec![]),
        Outcome::Trap(Trap { kind: PanicKind::Overflow, site: Site(41), func: callee })
    );
    assert_eq!(
        run(&m, callee, vec![Val::S(1)]),
        Outcome::Trap(Trap { kind: PanicKind::DivByZero, site: Site(42), func: callee })
    );
}

#[test]
fn explicit_panic_kinds() {
    for kind in PanicKind::ALL {
        let mut m = Module::new();
        let f = define(&mut m, &[0], "t.f", &[], None, |b, _| {
            b.set_site(Site(3));
            b.panic(kind);
            b.unreachable();
        });
        assert_eq!(run(&m, f, vec![]), Outcome::Trap(Trap { kind, site: Site(3), func: f }));
    }
}

#[test]
fn stack_overflow_by_depth_and_by_slots() {
    let mut m = Module::new();
    let u64t = m.types.u64();
    let f = m.declare(inst(&[1]), "app.r", vec![u64t], Some(u64t));
    let mut b = FuncBuilder::new(&[u64t]);
    let x = b.param(0).unwrap();
    b.set_site(Site(9));
    let r = b.call(f, Some(u64t), &[x.into()]).unwrap();
    b.ret(Some(r.into()));
    m.define(f, b.finish().unwrap());
    let cfg = Config { max_depth: 1000, ..Config::default() };
    let (o, _) = interp::run(&m, f, vec![Val::S(0)], host(), cfg);
    assert_eq!(o.unwrap(), Outcome::Trap(Trap { kind: PanicKind::StackOverflow, site: Site(9), func: f }));

    // a large slot per frame exhausts the stack before the depth limit
    let mut m = Module::new();
    let u8t = m.types.u8();
    let big = m.types.array(u8t, 1 << 16);
    let pbig = m.types.ref_(true, big);
    let f = m.declare(inst(&[1]), "app.r", vec![], None);
    let mut b = FuncBuilder::new(&[]);
    b.slot(pbig);
    b.call(f, None, &[]);
    b.ret(None);
    m.define(f, b.finish().unwrap());
    let cfg = Config { stack_size: 1 << 20, ..Config::default() };
    let (o, _) = interp::run(&m, f, vec![], host(), cfg);
    assert!(matches!(o.unwrap(), Outcome::Trap(Trap { kind: PanicKind::StackOverflow, .. })));
}

#[test]
fn fuel_limit() {
    let mut m = Module::new();
    let f = define(&mut m, &[0], "t.spin", &[], None, |b, _| {
        let l = b.create_block();
        b.br(l, &[]);
        b.switch_to(l);
        b.br(l, &[]);
    });
    let (o, _) = interp::run(&m, f, vec![], host(), Config { max_steps: 1000, ..Config::default() });
    assert_eq!(o, Err(InterpError::Fuel));
}

#[test]
fn memory_fault_on_null_and_dangling() {
    let mut m = Module::new();
    let u64t = m.types.u64();
    let p = m.types.ptr(false, u64t);
    let f = define(&mut m, &[0], "t.f", &[p], Some(u64t), |b, _| {
        let x = b.param(0).unwrap();
        let v = b.load(u64t, x);
        b.ret(Some(v.into()));
    });
    let (o, _) = interp::run(&m, f, vec![Val::S(0)], host(), Config::default());
    assert!(matches!(o, Err(InterpError::Fault { .. })));
    // a pointer into a returned frame's slot is dangling
    let mut m = Module::new();
    let u64t = m.types.u64();
    let p = m.types.ptr(true, u64t);
    let mk = m.declare(inst(&[1]), "t.mk", vec![], Some(p));
    let mut b = FuncBuilder::new(&[]);
    let s = b.slot(p);
    b.ret(Some(s.into()));
    m.define(mk, b.finish().unwrap());
    let main = define(&mut m, &[2], "t.main", &[], Some(u64t), |b, _| {
        let q = b.call(mk, Some(p), &[]).unwrap();
        let v = b.load(u64t, q);
        b.ret(Some(v.into()));
    });
    let (o, _) = interp::run(&m, main, vec![], host(), Config::default());
    assert!(matches!(o, Err(InterpError::Fault { .. })));
}

#[test]
fn string_write_and_exit() {
    let mut m = Module::new();
    let usize_t = m.types.usize();
    let isize_t = m.types.isize();
    let u8t = m.types.u8();
    let str_t = m.types.str();
    let sref = m.types.ref_(false, str_t);
    let p8 = m.types.ptr(false, u8t);
    let _ = sref;
    let s = m.add_string(b"hello, world\n".to_vec());
    let e = m.add_string(b"oops\n".to_vec());
    let sys3 = m.intrinsic("syscall3");
    let sys1 = m.intrinsic("syscall1");
    let f = define(&mut m, &[0], "t.main", &[], None, |b, _| {
        for (fd, st) in [(1, s), (2, e)] {
            let o = b.str(st);
            let ptr = b.op(Op::SlicePtr, p8, &[o]);
            let pa = b.cast(stave_ir::CastKind::Exact, usize_t, ptr);
            let o2 = b.str(st);
            let n = b.op(Op::SliceLen, usize_t, &[o2]);
            let (nr, fdk) = (b.int(1), b.int(fd));
            b.intrinsic(sys3, Some(isize_t), &[nr, fdk, pa.into(), n.into()]);
        }
        let (nr, code) = (b.int(231), b.int(3));
        b.intrinsic(sys1, Some(isize_t), &[nr, code]);
        b.unreachable();
    });
    let (o, h) = run_host(&m, f, vec![]);
    assert_eq!(o, Outcome::Exit(3));
    assert_eq!(h.stdout, b"hello, world\n");
    assert_eq!(h.stderr, b"oops\n");
    assert_eq!(h.log.len(), 3);
}

#[test]
fn unknown_syscall_is_enosys_and_aarch64_numbering() {
    let mut m = Module::new();
    let usize_t = m.types.usize();
    let isize_t = m.types.isize();
    let _ = usize_t;
    let sys0 = m.intrinsic("syscall0");
    let f = define(&mut m, &[0], "t.f", &[], Some(isize_t), |b, _| {
        let nr = b.int(999);
        let r = b.intrinsic(sys0, Some(isize_t), &[nr]).unwrap();
        b.ret(Some(r.into()));
    });
    assert_eq!(ret_val(run(&m, f, vec![])), Val::S((-38i64) as u64));
    let sys1 = m.intrinsic("syscall1");
    let g = define(&mut m, &[1], "t.g", &[], Some(isize_t), |b, _| {
        let (nr, c) = (b.int(94), b.int(258)); // aarch64 exit_group; status & 0xff
        let r = b.intrinsic(sys1, Some(isize_t), &[nr, c]).unwrap();
        b.ret(Some(r.into()));
    });
    let (o, _) = interp::run(
        &m,
        g,
        vec![],
        stave_ir::BufferHost::new(stave_ir::SyscallAbi::Aarch64),
        Config::default(),
    );
    assert_eq!(o.unwrap(), Outcome::Exit(2));
}

#[test]
fn mmap_heap_store_load() {
    let mut m = Module::new();
    let usize_t = m.types.usize();
    let isize_t = m.types.isize();
    let u32t = m.types.u32();
    let p = m.types.ptr(true, u32t);
    let sys6 = m.intrinsic("syscall6");
    let f = define(&mut m, &[0], "t.f", &[], Some(u32t), |b, _| {
        let args =
            [b.int(9), b.int(0), b.int(8192), b.int(3), b.int(0x22), b.int(-1i128 as u64 as i128), b.int(0)];
        let r = b.intrinsic(sys6, Some(isize_t), &args).unwrap();
        let ru = b.cast(stave_ir::CastKind::Wrap, usize_t, r);
        let ptr = b.cast(stave_ir::CastKind::Exact, p, ru);
        let k = b.int(1000);
        let q = b.op(Op::ElemAddr, p, &[ptr.into(), k]);
        let v = b.int(0xdead_beef);
        b.store(q, v);
        let x = b.load(u32t, q);
        b.ret(Some(x.into()));
    });
    assert_eq!(ret_val(run(&m, f, vec![])), Val::S(0xdead_beef));
}

/// Struct in a slot: store the whole struct, address a field, update through it, reload.
#[test]
fn struct_slot_field_addr() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let f64t = m.types.f64();
    let s = m.types.adt(DeclId(3), vec![], AdtShape::structure(vec![i32t, f64t, i32t]));
    let ps = m.types.ref_(true, s);
    let pi = m.types.ref_(true, i32t);
    let f = define(&mut m, &[0], "t.f", &[], Some(s), |b, _| {
        let slot = b.slot(ps);
        let (a, c, d) = (b.int(1), b.f64(2.5), b.int(3));
        let v = b.op(Op::Make, s, &[a, c, d]);
        b.store(slot, v);
        let k = b.int(2);
        let fp = b.op(Op::FieldAddr, pi, &[slot.into(), k]);
        let old = b.load(i32t, fp);
        let ten = b.int(10);
        let nv = b.binary(Op::Mul, i32t, old, ten);
        b.store(fp, nv);
        let w = b.load(s, slot);
        let z = b.int(0);
        let x = b.int(-7);
        let w2 = b.op(Op::Set, s, &[w.into(), z, x]);
        b.ret(Some(w2.into()));
    });
    assert_eq!(
        ret_val(run(&m, f, vec![])),
        Val::agg(vec![Val::S((-7i64) as u64), Val::f64(2.5), Val::S(30)])
    );
}

/// Enum in memory: store a variant, read the tag through a load, a payload field through
/// `addr.enum_field`.
#[test]
fn enum_in_memory() {
    let mut m = Module::new();
    let u8t = m.types.u8();
    let u64t = m.types.u64();
    let e = m.types.adt(DeclId(4), vec![], AdtShape::enumeration(vec![vec![], vec![u8t, u64t]]));
    let pe = m.types.ref_(true, e);
    let pu = m.types.ref_(false, u64t);
    let u32t = m.types.u32();
    let f = define(&mut m, &[0], "t.f", &[], Some(u64t), |b, _| {
        let slot = b.slot(pe);
        let (v, x, y) = (b.int(1), b.int(9), b.int(1 << 40));
        let ev = b.op(Op::EnumMake, e, &[v, x, y]);
        b.store(slot, ev);
        let back = b.load(e, slot);
        let tag = b.op(Op::EnumTag, u32t, &[back.into()]);
        let (vv, ff) = (b.int(1), b.int(1));
        let fa = b.op(Op::EnumFieldAddr, pu, &[slot.into(), vv, ff]);
        let y2 = b.load(u64t, fa);
        let t64 = b.cast(stave_ir::CastKind::Exact, u64t, tag);
        let r = b.binary(Op::Add, u64t, y2, t64);
        b.ret(Some(r.into()));
    });
    assert_eq!(ret_val(run(&m, f, vec![])), Val::S((1 << 40) + 1));
}

/// Array slot → slice: `slice.make`, `slice.len`, `addr.elem` + `check.bounds`.
#[test]
fn slices_over_array_slot() {
    let mut m = Module::new();
    let u16t = m.types.u16();
    let usize_t = m.types.usize();
    let bt = m.types.bool();
    let arr = m.types.array(u16t, 5);
    let parr = m.types.ref_(true, arr);
    let sl = m.types.slice(u16t);
    let rsl = m.types.ref_(false, sl);
    let pe = m.types.ptr(false, u16t);
    let re = m.types.ref_(false, u16t);
    let pre = m.types.ref_(true, u16t);
    // sum(s: &[u16], i: usize) = s[i] + s.len()   (bounds-checked)
    let get = define(&mut m, &[1], "t.get", &[rsl, usize_t], Some(usize_t), |b, _| {
        let (s, i) = (b.param(0).unwrap(), b.param(1).unwrap());
        let n = b.op(Op::SliceLen, usize_t, &[s.into()]);
        b.bounds_check(i, n);
        let ep = b.op(Op::ElemAddr, re, &[s.into(), i.into()]);
        let v = b.load(u16t, ep);
        let vw = b.cast(stave_ir::CastKind::Exact, usize_t, v);
        let r = b.binary(Op::Add, usize_t, vw, n);
        b.ret(Some(r.into()));
    });
    let main = define(&mut m, &[2], "t.main", &[usize_t], Some(usize_t), |b, _| {
        let i = b.param(0).unwrap();
        let a = b.slot(parr);
        // a[k] = 100 * k for k in 0..5 via a counted loop
        let (head, hp) = b.create_block_with(&[usize_t]);
        let (body, exit) = (b.create_block(), b.create_block());
        let z = b.int(0);
        b.br(head, &[z]);
        b.switch_to(head);
        let five = b.int(5);
        let c = b.cmp(Op::Lt, bt, hp[0], five);
        b.cbr(c, body, &[], exit, &[]);
        b.switch_to(body);
        let ep = b.op(Op::ElemAddr, pre, &[a.into(), hp[0].into()]);
        let k = b.cast(stave_ir::CastKind::Wrap, u16t, hp[0]);
        let h = b.int(100);
        let v = b.binary(Op::Mul, u16t, k, h);
        b.store(ep, v);
        let one = b.int(1);
        let nx = b.binary(Op::Add, usize_t, hp[0], one);
        b.br(head, &[nx.into()]);
        b.switch_to(exit);
        let z2 = b.int(0);
        let p0 = b.op(Op::ElemAddr, pre, &[a.into(), z2]);
        let p0c = b.cast(stave_ir::CastKind::Exact, pe, p0);
        let n = b.int(5);
        let s = b.op(Op::SliceMake, rsl, &[p0c.into(), n]);
        let r = b.call(get, Some(usize_t), &[s.into(), i.into()]).unwrap();
        b.ret(Some(r.into()));
    });
    assert_eq!(ret_val(run(&m, main, vec![Val::S(3)])), Val::S(305));
    assert!(
        matches!(run(&m, main, vec![Val::S(5)]), Outcome::Trap(Trap { kind: PanicKind::Bounds, func, .. }) if func == get)
    );
}

#[test]
fn fn_values_through_memory() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let fnt = m.types.fn_type(vec![i32t], i32t, stave_target::FeatureSet::EMPTY);
    let pf = m.types.ref_(true, fnt);
    let dbl = define(&mut m, &[1], "t.dbl", &[i32t], Some(i32t), |b, _| {
        let x = b.param(0).unwrap();
        let r = b.binary(Op::Add, i32t, x, x);
        b.ret(Some(r.into()));
    });
    let main = define(&mut m, &[2], "t.main", &[], Some(i32t), |b, _| {
        let s = b.slot(pf);
        b.store(s, Operand::Func(dbl));
        let fv = b.load(fnt, s);
        let k = b.int(21);
        let r = b.call_ind(fv, Some(i32t), &[k]).unwrap();
        b.ret(Some(r.into()));
    });
    assert_eq!(ret_val(run(&m, main, vec![])), Val::S(42));
}

#[test]
fn switch_dispatch() {
    let mut m = Module::new();
    let i8t = m.types.i8();
    let f = define(&mut m, &[0], "t.f", &[i8t], Some(i8t), |b, _| {
        let x = b.param(0).unwrap();
        let (a, c, d) = (b.create_block(), b.create_block(), b.create_block());
        b.switch(x, &[(-1, a, &[]), (5, c, &[])], d, &[]);
        for (blk, v) in [(a, 10), (c, 20), (d, 30)] {
            b.switch_to(blk);
            let k = b.int(v);
            b.ret(Some(k));
        }
    });
    assert_eq!(ret_val(run(&m, f, vec![Val::S(u64::MAX)])), Val::S(10));
    assert_eq!(ret_val(run(&m, f, vec![Val::S(5)])), Val::S(20));
    assert_eq!(ret_val(run(&m, f, vec![Val::S(6)])), Val::S(30));
}

#[test]
fn unreachable_and_no_body_are_errors() {
    let mut m = Module::new();
    let f = define(&mut m, &[0], "t.f", &[], None, |b, _| b.unreachable());
    let (o, _) = interp::run(&m, f, vec![], host(), Config::default());
    assert!(matches!(o, Err(InterpError::Unreachable { .. })));
    let ext = m.declare(inst(&[1]), "c.puts", vec![], None);
    let g = define(&mut m, &[2], "t.g", &[], None, |b, _| {
        b.call(ext, None, &[]);
        b.ret(None);
    });
    let (o, _) = interp::run(&m, g, vec![], host(), Config::default());
    assert_eq!(o, Err(InterpError::NoBody(ext)));
    let (o, _) = interp::run(&m, g, vec![Val::S(1)], host(), Config::default());
    assert!(matches!(o, Err(InterpError::ArgCount { .. })));
}

#[test]
fn process_stack_for_runtime_entry() {
    // entry(sp: *const usize) -> usize returns argc + the first byte of argv[1]
    let mut m = Module::new();
    let usize_t = m.types.usize();
    let u8t = m.types.u8();
    let psz = m.types.ptr(false, usize_t);
    let pu8 = m.types.ptr(false, u8t);
    let f = define(&mut m, &[0], "runtime.start.entry", &[psz], Some(usize_t), |b, _| {
        let sp = b.param(0).unwrap();
        let argc = b.load(usize_t, sp);
        let two = b.int(2);
        let a1p = b.op(Op::ElemAddr, psz, &[sp.into(), two]);
        let a1 = b.load(usize_t, a1p);
        let a1ptr = b.cast(stave_ir::CastKind::Exact, pu8, a1);
        let c = b.load(u8t, a1ptr);
        let cw = b.cast(stave_ir::CastKind::Exact, usize_t, c);
        let r = b.binary(Op::Add, usize_t, argc, cw);
        b.ret(Some(r.into()));
    });
    let mut it = Interp::new(&m, host(), Config::default()).unwrap();
    let sp = it.process_stack(&[b"prog", b"A"], &[b"X=1"]).unwrap();
    assert_eq!(it.run(f, vec![Val::S(sp)]).unwrap(), Outcome::Return(Some(Val::S(2 + 65))));
}

#[test]
fn verify_failure_blocks_interpretation() {
    let mut m = Module::new();
    let i32t = m.types.i32();
    let f = define(&mut m, &[0], "t.f", &[], Some(i32t), |b, _| {
        let k = b.int(1 << 40);
        b.ret(Some(k));
    });
    let (o, h) = interp::run(&m, f, vec![], host(), Config::default());
    assert!(matches!(o, Err(InterpError::Verify(_))));
    assert!(h.is_none());
}
