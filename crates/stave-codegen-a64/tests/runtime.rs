//! Runtime-facing parts: statics, `__stave_main` + `call_on_stack`, `ccallN`, the site and PC
//! tables, the signal and thread stubs, big frames and copies, register pressure, jump-table
//! (debug) images, and errors for what the backend rejects.

mod common;

use common::*;
use stave_codegen_a64::debug::{BuildConfig, ErrorKind, build_image};
use stave_ir::{CastKind, Op, Operand, SiteInfo, Static};
use stave_obj::{Machine, Options};

/// `static mut` globals: initialized data and bss, read and written through `static_addr`.
#[test]
fn statics() {
    let mut p = Prog::new();
    let t = p.t;
    let arr = p.m.types.array(t.u32, 3);
    let parr = p.m.types.ptr(true, arr);
    let pu32 = p.m.types.ptr(true, t.u32);
    let pu64 = p.m.types.ptr(true, t.u64);
    let mut init = Vec::new();
    for x in [7u32, 8, 9] {
        init.extend_from_slice(&x.to_le_bytes());
    }
    p.m.statics.push(Static { path: "app.A".into(), ty: arr, init });
    p.m.statics.push(Static { path: "app.Z".into(), ty: t.u64, init: vec![0; 8] });
    let sa = p.m.intrinsic("static_addr");
    let bump = p.func("app.bump", &[], Some(t.u64), |b, p| {
        let one = b.int(1);
        let z = b.intrinsic(sa, Some(pu64), &[one]).unwrap();
        let v = b.load(p.t.u64, z);
        let k = b.int(5);
        let v2 = b.binary(Op::Add, p.t.u64, v, k);
        b.store(z, v2);
        b.ret(Some(v2.into()));
    });
    p.set_main(|b, p| {
        let z = b.int(0);
        let a = b.intrinsic(sa, Some(parr), &[z]).unwrap();
        for k in 0..3 {
            let kk = b.int(k);
            let e = b.binary(Op::ElemAddr, pu32, a, kk);
            let x = b.load(t.u32, e);
            p.out(b, x, t.u32);
            let w = b.binary(Op::MulWrap, t.u32, x, x);
            b.store(e, w);
        }
        let whole = b.load(arr, a);
        let two = b.int(2);
        let x = b.binary(Op::Get, t.u32, whole, two);
        p.out(b, x, t.u32);
        for _ in 0..3 {
            let r = b.call(bump, Some(t.u64), &[]).unwrap();
            p.out(b, r, t.u64);
        }
    });
    let r = check(&p, "statics");
    assert_eq!(r.stdout.len(), 7 * 8);
}

/// Large frames (probed), large aggregate copies (the copy loop), and a big slot.
#[test]
fn big_frames_and_copies() {
    let mut p = Prog::new();
    let t = p.t;
    let big = p.m.types.array(t.u64, 1000);
    let huge = p.m.types.array(t.u8, 100_000);
    let phuge = p.m.types.ref_(true, huge);
    let pu8 = p.m.types.ref_(true, t.u8);
    // fn fill(x: u64) -> [u64; 1000]
    let fill = p.func("app.fill", &[t.u64], Some(big), |b, _| {
        let x = b.param(0).unwrap();
        let mut ops = Vec::new();
        for k in 0..1000 {
            ops.push(if k % 7 == 0 { x.into() } else { b.int(k as i128 * 3) });
        }
        let a = b.op(Op::Make, big, &ops);
        b.ret(Some(a.into()));
    });
    // fn sum(a: [u64; 1000], b: [u64; 1000]) -> u64 (by-reference aggregate args, dynamic gets)
    let sum = p.func("app.sum", &[big, big], Some(t.u64), |b, p| {
        let (a, c) = (b.param(0).unwrap(), b.param(1).unwrap());
        let (head, hp) = b.create_block_with(&[t.usize, t.u64]);
        let body = b.create_block();
        let (exit, ep) = b.create_block_with(&[t.u64]);
        let (z, z2) = (b.int(0), b.int(0));
        b.br(head, &[z, z2]);
        b.switch_to(head);
        let n = b.int(1000);
        let cc = b.cmp(Op::Lt, p.t.bool, hp[0], n);
        b.cbr(cc, body, &[], exit, &[hp[1].into()]);
        b.switch_to(body);
        let x = b.binary(Op::Get, t.u64, a, hp[0]);
        let y = b.binary(Op::Get, t.u64, c, hp[0]);
        let s = b.binary(Op::AddWrap, t.u64, x, y);
        let s = b.binary(Op::AddWrap, t.u64, hp[1], s);
        let one = b.int(1);
        let i1 = b.binary(Op::Add, t.usize, hp[0], one);
        b.br(head, &[i1.into(), s.into()]);
        b.switch_to(exit);
        b.ret(Some(ep[0].into()));
    });
    // fn deep(n: u32) -> u64: a 100 KB slot per frame, a little recursion
    let deep = p.declare("app.deep", &[t.u32], Some(t.u64));
    p.define(deep, &[t.u32], |b, p| {
        let n = b.param(0).unwrap();
        let s = b.slot(phuge);
        let k = b.int(99_999);
        let e = b.binary(Op::ElemAddr, pu8, s, k);
        let nb = b.cast(CastKind::Wrap, t.u8, n);
        b.store(e, nb);
        let (rec, base) = (b.create_block(), b.create_block());
        let z = b.int(0);
        let c = b.cmp(Op::Eq, p.t.bool, n, z);
        b.cbr(c, base, &[], rec, &[]);
        b.switch_to(base);
        let r0 = b.int(1);
        b.ret(Some(r0));
        b.switch_to(rec);
        let one = b.int(1);
        let n1 = b.binary(Op::Sub, t.u32, n, one);
        let r = b.call(deep, Some(t.u64), &[n1.into()]).unwrap();
        let x = b.load(t.u8, e);
        let xw = b.cast(CastKind::Exact, t.u64, x);
        let s = b.binary(Op::Add, t.u64, r, xw);
        b.ret(Some(s.into()));
    });
    p.set_main(|b, p| {
        let (x, y) = (b.int(5), b.int(1 << 40));
        let a = b.call(fill, Some(big), &[x]).unwrap();
        let c = b.call(fill, Some(big), &[y]).unwrap();
        let k = b.int(999);
        let v = b.int(77);
        let c2 = b.op(Op::Set, big, &[c.into(), k, v]);
        let r = b.call(sum, Some(t.u64), &[a.into(), c2.into()]).unwrap();
        p.out(b, r, t.u64);
        let r = b.call(sum, Some(t.u64), &[a.into(), c.into()]).unwrap();
        p.out(b, r, t.u64);
        let n = b.int(20);
        let r = b.call(deep, Some(t.u64), &[n]).unwrap();
        p.out(b, r, t.u64);
    });
    check(&p, "big");
}

/// More simultaneously live values than registers: eviction and reload, across calls.
#[test]
fn register_pressure() {
    let mut p = Prog::new();
    let t = p.t;
    let id = p.func("app.id", &[t.f64], Some(t.f64), |b, _| {
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
    });
    let f = p.func("app.f", &[t.u64, t.f64], Some(t.u64), |b, p| {
        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
        let mut iv: Vec<Operand> = Vec::new();
        let mut fv: Vec<Operand> = Vec::new();
        for k in 0..40 {
            let kk = b.int(k * 7 + 1);
            let v = b.binary(Op::MulWrap, t.u64, x, kk);
            iv.push(v.into());
            let kf = b.f64(k as f64 + 0.5);
            let w = b.binary(Op::Mul, t.f64, y, kf);
            fv.push(w.into());
        }
        let mid = b.call(id, Some(t.f64), &[fv[3]]).unwrap();
        let mut acc = b.int(0);
        for (k, v) in iv.iter().enumerate().rev() {
            let s = b.binary(Op::Xor, t.u64, acc, *v);
            let r = b.int((k % 13) as i128);
            let s = b.binary(Op::ShlWrap, t.u64, s, r);
            acc = s.into();
        }
        let mut facc: Operand = mid.into();
        for v in &fv {
            let s = b.binary(Op::Add, t.f64, facc, *v);
            facc = s.into();
        }
        let fi = b.cast(CastKind::Sat, t.u64, facc);
        let r = b.binary(Op::AddWrap, t.u64, acc, fi);
        let _ = p;
        b.ret(Some(r.into()));
    });
    p.set_main(|b, p| {
        for (x, y) in [(3, 1.25), (u64::MAX as i128, -7.5), (12345, 1e10)] {
            let (x, y) = (b.int(x), b.f64(y));
            let r = b.call(f, Some(t.u64), &[x, y]).unwrap();
            p.out(b, r, t.u64);
        }
    });
    check(&p, "pressure");
}

/// The same programs linked with the debug (jump-table) options.
#[test]
fn debug_image_options() {
    let mut p = Prog::new();
    let t = p.t;
    let sq = p.func("app.sq", &[t.i64], Some(t.i64), |b, _| {
        let x = b.param(0).unwrap();
        let r = b.binary(Op::Mul, t.i64, x, x);
        b.ret(Some(r.into()));
    });
    let fnty = p.m.types.fn_type(vec![t.i64], t.i64, stave_target::FeatureSet::EMPTY);
    p.set_main(|b, p| {
        let _ = fnty;
        let fv = b.func(sq);
        let k = b.int(-12);
        let r = b.call_ind(fv, Some(t.i64), &[k]).unwrap();
        p.out(b, r, t.i64);
        let r = b.call(sq, Some(t.i64), &[k]).unwrap();
        p.out(b, r, t.i64);
    });
    let want = interp(&p.m, p.entry);
    let img = build_image(
        &p.m,
        &BuildConfig { opts: Options::debug(Machine::Aarch64), main: None, release: false },
    )
    .unwrap();
    if !can_run() {
        return;
    }
    let path = tmp("dbgopts");
    img.write_file(&path).unwrap();
    let out = output_retry(std::process::Command::new("qemu-aarch64").arg(&path));
    assert_eq!(out.stdout, want.stdout);
    assert_eq!(out.status.code(), Some(want.code));
}

/// Native-only: `main_addr` + `call_on_stack` onto a slot used as a stack, `__stave_main`'s
/// status, `ccall` of a Stave function (≤ 8 and 12 args), thread and signal stubs, the site and
/// PC tables.
#[test]
fn native_runtime_intrinsics() {
    let mut p = Prog::new();
    let t = p.t;
    let us = t.usize;
    let stack = p.m.types.array(t.u64, 8192);
    let pstack = p.m.types.ref_(true, stack);
    let (main_addr, cos) = (p.m.intrinsic("main_addr"), p.m.intrinsic("call_on_stack"));
    let (sig_addr, thr_addr) = (p.m.intrinsic("signal_handler_addr"), p.m.intrinsic("thread_start_addr"));
    let (site_tab, pc_tab) = (p.m.intrinsic("site_table"), p.m.intrinsic("pc_table"));
    let (cc3, cc12) = (p.m.intrinsic("ccall3"), p.m.intrinsic("ccall12"));
    let pus = p.m.types.ptr(false, us);
    let pu32 = p.m.types.ptr(false, t.u32);
    p.m.sites = vec![
        SiteInfo { file: "src/a.stv".into(), line: 3, col: 9, func: "app.main".into() },
        SiteInfo { file: "src/b.stv".into(), line: 40, col: 1, func: "app.k12".into() },
    ];
    // The program's real main (called through __stave_main): prints, returns 7 (i32).
    let pmain = p.func("app.real_main", &[], Some(t.i32), |b, p| {
        b.set_site(stave_ir::Site(0));
        p.mark(b, 0xaaaa);
        b.set_site(stave_ir::Site(1));
        let one = b.int(1);
        b.binary(Op::AddWrap, p.t.i32, one, one);
        let k = b.int(-9);
        b.ret(Some(k));
    });
    let k12 = p.func("app.k12", &[us; 12], Some(us), |b, _| {
        let mut acc: Operand = b.int(0);
        for k in 0..12 {
            let x = b.param(k).unwrap();
            let m = b.int(k as i128 + 1);
            let y = b.binary(Op::MulWrap, us, x, m);
            let s = b.binary(Op::AddWrap, us, acc, y);
            acc = s.into();
        }
        b.ret(Some(acc));
    });
    let k3 = p.func("app.k3", &[us, us, us], Some(us), |b, _| {
        let (x, y, z) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
        let s = b.binary(Op::SubWrap, us, x, y);
        let s = b.binary(Op::MulWrap, us, s, z);
        b.ret(Some(s.into()));
    });
    let thread_entry = p.func_known(stave_ir::Known::ThreadEntry, &[us], Some(us), |b, _| {
        let x = b.param(0).unwrap();
        let k = b.int(1000);
        let r = b.binary(Op::AddWrap, us, x, k);
        b.ret(Some(r.into()));
    });
    let sys1 = p.sys1;
    let on_signal = p.func_known(stave_ir::Known::OnSignal, &[us, us, us], None, |b, p| {
        for k in 0..3 {
            let x = b.param(k).unwrap();
            p.out(b, x, us);
        }
        let nr = b.int(94);
        let c = b.int(55);
        b.intrinsic(sys1, Some(t.isize), &[nr, c]);
        b.unreachable();
    });
    let _ = (thread_entry, on_signal);
    let out_table = p.func("app.tables", &[], None, |b, p| {
        let st = b.intrinsic(site_tab, Some(us), &[]).unwrap();
        let sp = b.cast(CastKind::Exact, pus, st);
        let n = b.load(us, sp);
        p.out(b, n, us);
        // row 1: file_len, line, fn_len, col
        for off in [8 + 32 + 8, 8 + 32 + 12, 8 + 32 + 24, 8 + 32 + 28] {
            let a = b.int(off);
            let addr = b.binary(Op::AddWrap, us, st, a);
            let pa = b.cast(CastKind::Exact, pu32, addr);
            let x = b.load(t.u32, pa);
            p.out(b, x, t.u32);
        }
        // row 1 file string: print it
        let a = b.int(8 + 32);
        let addr = b.binary(Op::AddWrap, us, st, a);
        let pa = b.cast(CastKind::Exact, pus, addr);
        let fp = b.load(us, pa);
        let a = b.int(8 + 32 + 8);
        let addr = b.binary(Op::AddWrap, us, st, a);
        let pa = b.cast(CastKind::Exact, pu32, addr);
        let fl = b.load(t.u32, pa);
        let fl = b.cast(CastKind::Exact, us, fl);
        let fpp = b.cast(CastKind::Exact, t.pu8, fp);
        let s = b.binary(Op::SliceMake, t.str_ref, fpp, fl);
        b.call(p.out_str, None, &[s.into()]);
        // PC table: count, and whether rows ascend
        let pt = b.intrinsic(pc_tab, Some(us), &[]).unwrap();
        let pp = b.cast(CastKind::Exact, pus, pt);
        let n = b.load(us, pp);
        p.out(b, n, us);
        let (a0, a1) = (b.int(8), b.int(24));
        let r0 = b.binary(Op::AddWrap, us, pt, a0);
        let r1 = b.binary(Op::AddWrap, us, pt, a1);
        let (p0, p1) = (b.cast(CastKind::Exact, pus, r0), b.cast(CastKind::Exact, pus, r1));
        let (x0, x1) = (b.load(us, p0), b.load(us, p1));
        let c = b.cmp(Op::Lt, p.t.bool, x0, x1);
        p.out(b, c, p.t.bool);
        b.ret(None);
    });
    p.set_main(|b, p| {
        // call_on_stack(main_addr, 0, top of a 64 KiB slot)
        let s = b.slot(pstack);
        let base = b.cast(CastKind::Exact, us, s);
        let sz = b.int(8192 * 8);
        let top = b.binary(Op::AddWrap, us, base, sz);
        let f = b.intrinsic(main_addr, Some(us), &[]).unwrap();
        let z = b.int(0);
        let st = b.intrinsic(cos, Some(us), &[f.into(), z, top.into()]).unwrap();
        p.out(b, st, us);
        // ccall of Stave functions through their addresses
        let fv = b.cast(CastKind::Exact, us, b.func(k3));
        let (x, y, w) = (b.int(10), b.int(3), b.int(5));
        let r = b.intrinsic(cc3, Some(us), &[fv.into(), x, y, w]).unwrap();
        p.out(b, r, us);
        let fv = b.cast(CastKind::Exact, us, b.func(k12));
        let mut args = vec![fv.into()];
        for k in 0..12 {
            args.push(b.int(100 + k));
        }
        let r = b.intrinsic(cc12, Some(us), &args).unwrap();
        p.out(b, r, us);
        // the thread stub (C ABI) through ccall
        let ta = b.intrinsic(thr_addr, Some(us), &[]).unwrap();
        let (x, y, w) = (b.int(5), b.int(0), b.int(0));
        let r = b.intrinsic(cc3, Some(us), &[ta.into(), x, y, w]).unwrap();
        p.out(b, r, us);
        b.call(out_table, None, &[]);
        // the signal stub: sig is a C int (upper bits garbage) → zero-extended
        let sa = b.intrinsic(sig_addr, Some(us), &[]).unwrap();
        let (x, y, w) = (b.int(0xdead_beef_0000_000b_u64 as i128), b.int(2), b.int(3));
        b.intrinsic(cc3, Some(us), &[sa.into(), x, y, w]);
    });
    let img = build_image(&p.m, &BuildConfig::new(Some(pmain))).unwrap();
    let ms = img.lookup("app.real_main").unwrap();
    let rows = &img.func_code(ms).unwrap().lines;
    assert_eq!(rows.iter().map(|r| (r.line, r.column)).collect::<Vec<_>>(), vec![(3, 9), (40, 1)]);
    if !can_run() {
        return;
    }
    let path = tmp("rt");
    img.write_file(&path).unwrap();
    if have("aarch64-linux-gnu-addr2line") {
        let a = img.symbol_addr(ms).unwrap() + u64::from(rows[1].offset);
        let out = std::process::Command::new("aarch64-linux-gnu-addr2line")
            .arg("-e")
            .arg(&path)
            .arg(format!("{a:#x}"))
            .output()
            .unwrap();
        let l = String::from_utf8_lossy(&out.stdout);
        assert!(l.trim().ends_with("src/b.stv:40"), "{l}");
    }
    let out = output_retry(std::process::Command::new("qemu-aarch64").arg(&path));
    let mut want: Vec<u8> = Vec::new();
    let w = |v: &mut Vec<u8>, x: u64| v.extend_from_slice(&x.to_le_bytes());
    w(&mut want, 0xaaaa);
    w(&mut want, (-9i64) as u64);
    w(&mut want, 35);
    w(&mut want, (0..12u64).map(|k| (100 + k) * (k + 1)).sum());
    w(&mut want, 1005);
    w(&mut want, 2);
    for x in [9u64, 40, 7, 1] {
        w(&mut want, x);
    }
    want.extend_from_slice(b"src/b.stv");
    let nfn = p.m.funcs.iter().filter(|f| f.body.is_some()).count() as u64;
    w(&mut want, nfn);
    w(&mut want, 1);
    w(&mut want, 0xb);
    w(&mut want, 2);
    w(&mut want, 3);
    assert_eq!(out.stdout, want, "stderr {:?}", out.stderr);
    assert_eq!(out.status.code(), Some(55));
}

#[test]
fn rejects_x86_intrinsics_and_unsupported_types() {
    let mut p = Prog::new();
    let t = p.t;
    let cpuid = p.m.intrinsic("cpuid");
    let four = p.m.types.tuple(vec![t.u32; 4]);
    let f = p.func("app.c", &[], Some(four), |b, _| {
        let (x, y) = (b.int(0), b.int(0));
        let r = b.intrinsic(cpuid, Some(four), &[x, y]).unwrap();
        b.ret(Some(r.into()));
    });
    let _ = f;
    p.set_main(|_, _| {});
    let e = build_image(&p.m, &BuildConfig::new(None)).unwrap_err();
    assert!(matches!(&e.kind, ErrorKind::Unsupported(s) if s.contains("x86-64 only")), "{e}");
    assert_eq!(e.func, Some(f));

    let mut p = Prog::new();
    let h = p.m.types.intern(stave_ir::TypeKey::Float(stave_ir::FloatBits::F16));
    let g = p.func("app.h", &[h], Some(h), |b, _| {
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
    });
    p.set_main(|_, _| {});
    let e = build_image(&p.m, &BuildConfig::new(None)).unwrap_err();
    assert!(matches!(&e.kind, ErrorKind::Unsupported(_)), "{e}");
    assert_eq!(e.func, Some(g));
}
