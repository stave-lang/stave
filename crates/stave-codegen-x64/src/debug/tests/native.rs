//! Intrinsics the interpreter's `BufferHost` does not model (`call_on_stack`, `main_addr`,
//! `ccallN`, `cpuid`, `xgetbv`, the stubs and tables): run natively only. Also clean errors for
//! IR the backend does not support.

use stave_ir::{CastKind, FuncBuilder, Lanes, Op, Operand, Site, TypeKey};
use stave_obj::{Machine, SourceFile};

use super::harness::*;
use crate::debug::{Error, Options, SiteRow, compile_module};

fn release() -> Options {
    Options { image: stave_obj::Options::release(Machine::X86_64), release: true, ..Options::default() }
}

fn run_native(p: &Prog, name: &str, opts: &Options) -> Option<RunOut> {
    native(p, name, opts)
}

/// `runtime.start.entry` maps a 1 MiB stack and runs `main` on it through
/// `call_on_stack(main_addr(), 0, top)`; the exit status is `main`'s `i32`.
#[test]
fn call_on_stack_main() {
    let mut p = Prog::new();
    let t = p.t;
    let main = p.decl("t.main", &[], Some(t.i32));
    {
        let mut b = FuncBuilder::new(&[]);
        p.print_str(&mut b, "on the new stack\n");
        let c = b.int(-3);
        b.ret(Some(c));
        p.def(main, b);
    }
    let (sys6, maddr, cos) =
        (p.m.intrinsic("syscall6"), p.m.intrinsic("main_addr"), p.m.intrinsic("call_on_stack"));
    let mut b = FuncBuilder::new(&[t.pusize]);
    let args = [b.int(9), b.int(0), b.int(1 << 20), b.int(3), b.int(0x22), b.int(u64::MAX as i128), b.int(0)];
    let base = b.intrinsic(sys6, Some(t.isize), &args).unwrap();
    let base = b.cast(CastKind::Wrap, t.usize, base);
    let sz = b.int(1 << 20);
    let top = b.binary(Op::AddWrap, t.usize, base, sz);
    let f = b.intrinsic(maddr, Some(t.usize), &[]).unwrap();
    let z = b.int(0);
    let r = b.intrinsic(cos, Some(t.usize), &[f.into(), z, top.into()]).unwrap();
    let nr = b.int(231);
    b.intrinsic(p.sys1, Some(t.isize), &[nr, r.into()]);
    b.unreachable();
    p.def(p.entry, b);
    p.main = Some(main);
    for (n, o) in [("debug", Options::default()), ("release", release())] {
        if let Some(r) = run_native(&p, &format!("cos-{n}"), &o) {
            assert_eq!(r.stdout, b"on the new stack\n", "{n}");
            assert_eq!(r.code, 253, "{n}: -3 as exit status");
        }
    }
}

/// `ccallN` into Stave functions (System V for integer args, 7th+ on the stack), the C-ABI stubs
/// `__stave_thread_start` / `__stave_signal`, `cpuid`, `xgetbv`.
#[test]
fn ccall_stubs_cpuid() {
    let mut p = Prog::new();
    let t = p.t;
    let add3 = p.decl("t.add3", &[t.usize; 3], Some(t.usize));
    {
        let mut b = FuncBuilder::new(&[t.usize; 3]);
        let (x, y, z) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
        let k = b.int(100);
        let y2 = b.binary(Op::MulWrap, t.usize, y, k);
        let s = b.binary(Op::AddWrap, t.usize, x, y2);
        let k = b.int(10000);
        let z2 = b.binary(Op::MulWrap, t.usize, z, k);
        let s = b.binary(Op::AddWrap, t.usize, s, z2);
        b.ret(Some(s.into()));
        p.def(add3, b);
    }
    let many = p.decl("t.many", &[t.usize; 12], Some(t.usize));
    {
        let mut b = FuncBuilder::new(&[t.usize; 12]);
        let mut acc: Operand = b.int(0);
        for i in 0..12 {
            let x = b.param(i).unwrap();
            let k = b.int(31);
            let a2 = b.binary(Op::MulWrap, t.usize, acc, k);
            acc = b.binary(Op::AddWrap, t.usize, a2, x).into();
        }
        b.ret(Some(acc));
        p.def(many, b);
    }
    let te = p.decl("runtime.thread.entry", &[t.usize], Some(t.usize));
    {
        let mut b = FuncBuilder::new(&[t.usize]);
        let x = b.param(0).unwrap();
        let two = b.int(2);
        let r = b.binary(Op::MulWrap, t.usize, x, two);
        b.ret(Some(r.into()));
        p.def(te, b);
    }
    let os = p.decl("runtime.panic.on_signal", &[t.usize; 3], None);
    {
        let mut b = FuncBuilder::new(&[t.usize; 3]);
        let sig = b.param(0).unwrap();
        let nr = b.int(231);
        b.intrinsic(p.sys1, Some(t.isize), &[nr, sig.into()]);
        b.unreachable();
        p.def(os, b);
    }
    let names =
        ["ccall3", "ccall12", "ccall1", "thread_start_addr", "signal_handler_addr", "cpuid", "xgetbv"];
    let ids: Vec<_> = names.iter().map(|n| p.m.intrinsic(n)).collect();
    let cpu_t = p.m.types.tuple(vec![t.u32; 4]);
    p.main(|p, b| {
        let fv = b.func(add3);
        let f = b.cast(CastKind::Exact, t.usize, fv);
        let (x, y, z) = (b.int(1), b.int(2), b.int(3));
        let r = b.intrinsic(ids[0], Some(t.usize), &[f.into(), x, y, z]).unwrap();
        p.print(b, r.into(), t.usize);
        let fv = b.func(many);
        let f = b.cast(CastKind::Exact, t.usize, fv);
        let mut args = vec![f.into()];
        args.extend((0..12).map(|i| b.int(i * 3 + 1)));
        let r = b.intrinsic(ids[1], Some(t.usize), &args).unwrap();
        p.print(b, r.into(), t.usize);
        let ts = b.intrinsic(ids[3], Some(t.usize), &[]).unwrap();
        let k = b.int(21);
        let r = b.intrinsic(ids[2], Some(t.usize), &[ts.into(), k]).unwrap();
        p.print(b, r.into(), t.usize);
        let (l0, s0) = (b.int(0), b.int(0));
        let c = b.intrinsic(ids[5], Some(cpu_t), &[l0, s0]).unwrap();
        let k0 = b.int(0);
        let eax = b.op(Op::Get, t.u32, &[c.into(), k0]);
        let z = b.int(0);
        let pos = b.cmp(Op::Gt, t.bool, eax, z);
        p.print(b, pos.into(), t.bool);
        let x0 = b.int(0);
        let xcr = b.intrinsic(ids[6], Some(t.u64), &[x0]).unwrap();
        let three = b.int(3);
        let m = b.binary(Op::And, t.u64, xcr, three);
        p.print(b, m.into(), t.u64);
        // the signal stub zero-extends the C int: 0xffff_ffff_0000_0007 → 7
        let sh = b.intrinsic(ids[4], Some(t.usize), &[]).unwrap();
        let (sig, z1, z2) = (b.int(0xffff_ffff_0000_0007u64 as i128), b.int(0), b.int(0));
        b.intrinsic(ids[0], Some(t.usize), &[sh.into(), sig, z1, z2]);
    });
    let mut want = String::new();
    want.push_str(&format!("{:016x}\n", 1 + 200 + 30000));
    let mut acc: u64 = 0;
    for i in 0..12u64 {
        acc = acc.wrapping_mul(31).wrapping_add(i * 3 + 1);
    }
    want.push_str(&format!("{acc:016x}\n{:016x}\n{:016x}\n{:016x}\n", 42, 1, 3));
    for (n, o) in [("debug", Options::default()), ("release", release())] {
        if let Some(r) = run_native(&p, &format!("ccall-{n}"), &o) {
            assert_eq!(String::from_utf8_lossy(&r.stdout), want, "{n}");
            assert_eq!(r.code, 7, "{n}: on_signal exit status");
        }
    }
}

/// The site table rows, the PC table (sorted, one row per fn), and the `site` argument of
/// `runtime.panic.trap`.
#[test]
fn tables_and_sites() {
    let mut p = Prog::with_trap(false);
    let t = p.t;
    let trap = p.decl("runtime.panic.trap", &[t.u32, t.u32], None);
    {
        let mut b = FuncBuilder::new(&[t.u32, t.u32]);
        let site = b.param(1).unwrap();
        let s = b.cast(CastKind::Exact, t.usize, site);
        let nr = b.int(231);
        b.intrinsic(p.sys1, Some(t.isize), &[nr, s.into()]);
        b.unreachable();
        p.def(trap, b);
    }
    let (st, pc) = (p.m.intrinsic("site_table"), p.m.intrinsic("pc_table"));
    let pu32 = p.m.types.ptr(false, t.u32);
    let pu64 = p.m.types.ptr(false, t.u64);
    let main = p.main(|p, b| {
        let a = b.intrinsic(st, Some(t.usize), &[]).unwrap();
        let ap = b.cast(CastKind::Exact, pu64, a);
        let n = b.load(t.u64, ap);
        p.print(b, n.into(), t.u64);
        for off in [8 + 8, 8 + 12, 8 + 24, 8 + 28, 40 + 12] {
            let k = b.int(off);
            let x = b.binary(Op::AddWrap, t.usize, a, k);
            let xp = b.cast(CastKind::Exact, pu32, x);
            let v = b.load(t.u32, xp);
            p.print(b, v.into(), t.u32);
        }
        // first byte of the file path and of the fn path
        let pu8 = p.t.pu8;
        for off in [8i128, 8 + 16] {
            let k = b.int(off);
            let x = b.binary(Op::AddWrap, t.usize, a, k);
            let xp = b.cast(CastKind::Exact, pu64, x);
            let s = b.load(t.u64, xp);
            let sp = b.cast(CastKind::Exact, pu8, s);
            let c = b.load(t.u8, sp);
            p.print(b, c.into(), t.u8);
        }
        let a = b.intrinsic(pc, Some(t.usize), &[]).unwrap();
        let ap = b.cast(CastKind::Exact, pu64, a);
        let n = b.load(t.u64, ap);
        p.print(b, n.into(), t.u64);
        // rows ascending: pc_lo(row1) > pc_lo(row0) > 0, and row 0 (the entry fn) site
        let (k8, k24) = (b.int(8), b.int(24));
        let r0 = b.binary(Op::AddWrap, t.usize, a, k8);
        let r1 = b.binary(Op::AddWrap, t.usize, a, k24);
        let r0p = b.cast(CastKind::Exact, pu64, r0);
        let r1p = b.cast(CastKind::Exact, pu64, r1);
        let pc0 = b.load(t.u64, r0p);
        let pc1 = b.load(t.u64, r1p);
        let asc = b.cmp(Op::Lt, t.bool, pc0, pc1);
        p.print(b, asc.into(), t.bool);
        let k16 = b.int(16);
        let s0 = b.binary(Op::AddWrap, t.usize, a, k16);
        let s0p = b.cast(CastKind::Exact, pu32, s0);
        let site0 = b.load(t.u32, s0p);
        p.print(b, site0.into(), t.u32);
        // trap at site 1
        b.set_site(Site(1));
        let c = b.int(i32::MAX as i128);
        let x = p.opaque(b, c, t.i32);
        let one = b.int(1);
        let r = b.binary(Op::Add, t.i32, x, one);
        b.set_site(Site::NONE);
        p.print(b, r.into(), t.i32);
    });
    let nfuncs = p.m.funcs.iter().filter(|f| f.body.is_some()).count();
    let mut o = Options {
        files: vec![SourceFile { dir: "src".into(), name: "main.stv".into() }],
        sites: vec![
            SiteRow { file: 0, line: 3, col: 5, func: main },
            SiteRow { file: 0, line: 10, col: 17, func: main },
        ],
        ..Options::default()
    };
    o.fn_sites = vec![Site::NONE; p.m.funcs.len()];
    o.fn_sites[p.entry.index()] = Site(0);
    let want = format!(
        "{:016x}\n{:016x}\n{:016x}\n{:016x}\n{:016x}\n{:016x}\n{:016x}\n{:016x}\n{:016x}\n{:016x}\n{:016x}\n",
        2, 12, 3, 6, 5, 10, b's', b't', nfuncs, 1, 0
    );
    for (n, img) in [("debug", o.image.clone()), ("release", stave_obj::Options::release(Machine::X86_64))] {
        let o = Options { image: img, release: n == "release", ..o.clone() };
        if let Some(r) = run_native(&p, &format!("tables-{n}"), &o) {
            assert_eq!(String::from_utf8_lossy(&r.stdout), want, "{n}");
            assert_eq!(r.code, 1, "{n}: trap site");
        }
    }
}

/// Valid IR the backend does not support yields an error, not a panic.
#[test]
fn clean_errors() {
    let mut p = Prog::new();
    let t = p.t;
    let v4 = p.m.types.intern(TypeKey::Vec { lane: t.f32, n: Lanes::Fixed(4) });
    let pv = p.m.types.ref_(true, v4);
    let f = p.decl("t.vec", &[pv], None);
    let mut b = FuncBuilder::new(&[pv]);
    let x = b.param(0).unwrap();
    let l = b.load(v4, x);
    b.store(x, l);
    b.ret(None);
    p.def(f, b);
    let main = p.main(|_, _| {});
    let _ = main;
    match compile_module(&p.m, &Options::default()) {
        Err(Error::Unsupported { func: Some(g), .. }) => assert_eq!(g, f),
        r => panic!("expected Unsupported, got {:?}", r.map(|_| ())),
    }
    // no runtime entry
    let mut q = Prog::new();
    q.main(|_, _| {});
    q.m.funcs[q.entry.index()].path = "not.the.entry".into();
    assert!(compile_module(&q.m, &Options::default()).is_err());
    // unverified garbage: still no panic
    let mut r = Prog::new();
    r.main(|_, _| {});
    if let Some(body) = &mut r.m.funcs[r.print.index()].body {
        body.operands.truncate(1);
        body.values.truncate(2);
    }
    let o = Options { verify: false, ..Options::default() };
    let _ = compile_module(&r.m, &o);
    assert!(compile_module(&r.m, &Options::default()).is_err());
}

/// Parallel codegen gives the same image as one thread; line rows follow the sites.
#[test]
fn parallel_deterministic_and_lines() {
    let mut p = Prog::new();
    let mut last = None;
    for i in 0..120u64 {
        let (f, _, _) =
            super::randgen::Gen::new(&mut p, i + 77, 30, true, false).function(&format!("g.f{i}"));
        last = Some(f);
    }
    let t = p.t;
    let main = p.main(|p, b| {
        b.set_site(Site(0));
        let c = b.int(5);
        let x = p.opaque(b, c, t.u32);
        let one = b.int(1);
        let r = b.binary(Op::Add, t.u32, x, one);
        b.set_site(Site::NONE);
        p.print(b, r.into(), t.u32);
    });
    let _ = last;
    let o = Options {
        files: vec![SourceFile { dir: ".".into(), name: "m.stv".into() }],
        sites: vec![SiteRow { file: 0, line: 42, col: 9, func: main }],
        threads: 1,
        ..Options::default()
    };
    let one = compile_module(&p.m, &o).unwrap();
    let four = compile_module(&p.m, &Options { threads: 4, ..o.clone() }).unwrap();
    assert_eq!(one.bytes(), four.bytes(), "parallel codegen must be deterministic");
    let ms = one.lookup("t.main").unwrap();
    let code = one.func_code(ms).unwrap();
    assert!(code.lines.iter().any(|r| r.line == 42 && r.column == 9), "{:?}", code.lines);
    assert!(code.lines.windows(2).all(|w| w[0].offset <= w[1].offset));
    assert!(code.lines.iter().all(|r| (r.offset as usize) < code.code.len()));
}

/// `static mut` globals (`Module::statics`, `intrinsic.static_addr`) against the interpreter, and
/// the site table built from `Module::sites` / `Func::name_site`.
#[test]
fn statics_and_module_sites() {
    let mut p = Prog::new();
    let t = p.t;
    let arr = p.m.types.array(t.u8, 4);
    p.m.statics.push(stave_ir::Static { path: "t.COUNT".into(), ty: t.u64, init: vec![0; 8] });
    p.m.statics.push(stave_ir::Static { path: "t.MSG".into(), ty: arr, init: b"abc\n".to_vec() });
    p.m.sites.push(stave_ir::SiteInfo { file: "src/a.stv".into(), line: 7, col: 3, func: "t.main".into() });
    p.m.sites.push(stave_ir::SiteInfo { file: "src/a.stv".into(), line: 9, col: 11, func: "t.main".into() });
    let sa = p.m.intrinsic("static_addr");
    let pu64 = p.m.types.ptr(true, t.u64);
    let parr = p.m.types.ptr(true, arr);
    let main = p.main(|p, b| {
        let k0 = b.int(0);
        let c = b.intrinsic(sa, Some(pu64), &[k0]).unwrap();
        for _ in 0..3 {
            let x = b.load(t.u64, c);
            let k = b.int(5);
            let y = b.binary(Op::AddWrap, t.u64, x, k);
            b.store(c, y);
        }
        let x = b.load(t.u64, c);
        p.print(b, x.into(), t.u64);
        let k1 = b.int(1);
        let m = b.intrinsic(sa, Some(parr), &[k1]).unwrap();
        let mp = b.cast(CastKind::Exact, p.t.pu8, m);
        let n = b.int(4);
        b.call(p.write, None, &[mp.into(), n]);
        b.set_site(Site(1));
        let c = b.int(i64::MIN as i128);
        let v = p.opaque(b, c, t.i64);
        let r = b.unary(Op::Neg, t.i64, v);
        b.set_site(Site::NONE);
        p.print(b, r.into(), t.i64);
    });
    p.m.funcs[main.index()].name_site = Site(0);
    let r = super::harness::check(&p, "statics");
    assert_eq!(r.stdout, b"000000000000000f\nabc\n");
    assert_eq!(r.code, 101);
    // the table, natively: row 1 line and column
    let mut q = Prog::new();
    q.m.sites = p.m.sites.clone();
    let st = q.m.intrinsic("site_table");
    let pu32 = q.m.types.ptr(false, q.t.u32);
    let main = q.main(|q, b| {
        let a = b.intrinsic(st, Some(q.t.usize), &[]).unwrap();
        for off in [8 + 32 + 12, 8 + 32 + 28, 8 + 8] {
            let k = b.int(off);
            let x = b.binary(Op::AddWrap, q.t.usize, a, k);
            let xp = b.cast(CastKind::Exact, pu32, x);
            let v = b.load(q.t.u32, xp);
            q.print(b, v.into(), q.t.u32);
        }
    });
    let _ = main;
    if let Some(r) = native(&q, "module-sites", &Options::default()) {
        assert_eq!(String::from_utf8_lossy(&r.stdout), format!("{:016x}\n{:016x}\n{:016x}\n", 9, 11, 9));
    }
}
