//! Regressions: scratch registers across far frame offsets, a fused compare's operand read later
//! in its block, frame-size overflow, functions over the conditional-branch range, user fns named
//! like compiler symbols, the thread child's stack alignment, references to fns without a body,
//! zero-size statics, whole-slot loads read in place.

mod common;

use common::*;
use stave_codegen_a64::debug::{BuildConfig, Codegen, ErrorKind, FuncScratch, build_image, prepare};
use stave_ir::{CastKind, Op, Operand};
use stave_obj::{ImageBuilder, Machine, Options};

/// fn values compared when their homes are beyond the scaled-offset range (≥ 32 KiB from sp):
/// loading the second operand must not clobber the first one's words.
#[test]
fn fn_value_compare_in_a_large_frame() {
    let mut p = Prog::new();
    let t = p.t;
    let fnty = p.m.types.fn_type(vec![t.i64], t.i64, stave_target::FeatureSet::EMPTY);
    let body = |b: &mut stave_ir::FuncBuilder, _: &Prog| {
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
    };
    let k = p.func("app.k", &[t.i64, t.usize], Some(t.i64), body);
    let k2 = p.func("app.k2", &[t.i64, t.usize], Some(t.i64), body);
    let buf = p.m.types.array(t.u8, 40000);
    let pbuf = p.m.types.ref_(true, buf);
    let pu8 = p.m.types.ref_(true, t.u8);
    p.set_main(|b, p| {
        let s = b.slot(pbuf);
        let z = b.int(0);
        let e = b.op(Op::ElemAddr, pu8, &[s.into(), z]);
        let one = b.int(1);
        b.store(e, one);
        let (e7, e8) = (b.int(7), b.int(8));
        let f = b.fn_make(fnty, k, e7);
        let g = b.fn_make(fnty, k, e7);
        let h = b.fn_make(fnty, k, e8);
        let h2 = b.fn_make(fnty, k2, e7);
        for (x, y) in [(f, g), (f, h), (f, h2), (h2, h2)] {
            for op in [Op::Eq, Op::Ne] {
                let c = b.cmp(op, t.bool, x, y);
                p.out(b, c, t.bool);
            }
        }
    });
    let r = check(&p, "fn_eq_far");
    let words: Vec<u8> = r.stdout.chunks(8).map(|c| c[0]).collect();
    assert_eq!(words, [1, 0, 0, 1, 0, 1, 1, 0]);
}

/// A compare fused into its block's `cbr` is emitted at the terminator, after the block's other
/// instructions: an operand that one of them reads later in the block must still be bound there,
/// not dropped at that instruction as its last use (its home was never written).
#[test]
fn fused_compare_operand_used_later_in_the_block() {
    let mut p = Prog::new();
    let t = p.t;
    let f = p.func("app.f", &[t.u64], Some(t.u64), |b, _| {
        let x = b.param(0).unwrap();
        let five = b.int(5);
        let y = b.binary(Op::AddWrap, t.u64, x, five);
        let seven = b.int(7);
        let c = b.cmp(Op::Gt, t.bool, y, seven);
        let one = b.int(1);
        let z = b.binary(Op::Xor, t.u64, y, one);
        let (yes, no) = (b.create_block(), b.create_block());
        b.cbr(c, yes, &[], no, &[]);
        b.switch_to(yes);
        b.ret(Some(z.into()));
        b.switch_to(no);
        let k = b.int(99);
        b.ret(Some(k));
    });
    p.set_main(|b, p| {
        for x in [10, 0, 3] {
            let x = b.int(x);
            let r = b.call(f, Some(t.u64), &[x]).unwrap();
            p.out(b, r, t.u64);
        }
    });
    let r = check(&p, "fused_cmp_later_use");
    let words: Vec<u8> = r.stdout.chunks(8).map(|c| c[0]).collect();
    assert_eq!(words, [14, 99, 9]);
}

/// Slot sizes add up past `u32`: rejected like any frame larger than 1 GiB (a frame the checker's
/// bound admits never gets there, so the error is an internal one), never wrapped.
#[test]
fn frame_size_overflow_is_rejected() {
    let mut p = Prog::new();
    let t = p.t;
    let big = p.m.types.array(t.u8, 1 << 30);
    let pbig = p.m.types.ref_(true, big);
    let pu8 = p.m.types.ref_(true, t.u8);
    let f = p.func("app.huge", &[], None, |b, _| {
        for _ in 0..4 {
            let s = b.slot(pbig);
            let z = b.int(0);
            let e = b.op(Op::ElemAddr, pu8, &[s.into(), z]);
            let one = b.int(1);
            b.store(e, one);
        }
        b.ret(None);
    });
    p.set_main(|b, _| {
        b.call(f, None, &[]);
    });
    let e = build_image(&p.m, &BuildConfig::new(None)).unwrap_err();
    assert!(matches!(&e.kind, ErrorKind::BadIr(s) if s.contains("larger than 1 GiB")), "{e}");
    assert_eq!(e.func, Some(f));
}

/// Frames past 16 MiB, beyond two `add` immediates: slot, value home, parameter, and
/// return-pointer offsets from `sp` are formed with `mov` + extended-register `add`. Both ends of
/// every slot are written and read back; ten parameters (two on the stack) and an indirect
/// return go through homes above the slots. Runs under `qemu-aarch64 -s` (the harness runtime
/// has no stack of its own).
#[test]
fn frames_over_16_mib_run() {
    let cases: [(&str, &[u64]); 2] =
        [("big26", &[1_000_000, 1_000_000, 8_000_000, 8_000_000, 8_000_000]), ("big200", &[200 << 20])];
    for (name, sizes) in cases {
        let mut p = Prog::new();
        let t = p.t;
        let pu8 = p.m.types.ref_(true, t.u8);
        let ret = p.m.types.array(t.u64, 4);
        let tys: Vec<_> = sizes
            .iter()
            .map(|&n| {
                let a = p.m.types.array(t.u8, n);
                p.m.types.ref_(true, a)
            })
            .collect();
        let f = p.func("app.big", &[t.u64; 10], Some(ret), |b, _| {
            let ps: Vec<_> = (0..10).map(|k| b.param(k).unwrap()).collect();
            let mut ends = Vec::new();
            for (k, (&pt, &n)) in tys.iter().zip(sizes).enumerate() {
                let s = b.slot(pt);
                for (j, i) in [0, n - 1].into_iter().enumerate() {
                    let idx = b.int(i128::from(i));
                    let e = b.op(Op::ElemAddr, pu8, &[s.into(), idx]);
                    let v = b.int((2 * k + j + 1) as i128);
                    b.store(e, v);
                    ends.push(e);
                }
            }
            let mut acc = ps[0];
            for &p in &ps[1..] {
                acc = b.binary(Op::AddWrap, t.u64, acc, p);
            }
            for e in ends {
                let x = b.load(t.u8, e);
                let xw = b.cast(CastKind::Exact, t.u64, x);
                acc = b.binary(Op::AddWrap, t.u64, acc, xw);
            }
            let seven = b.int(7);
            let r = b.op(Op::Make, ret, &[acc.into(), ps[9].into(), ps[0].into(), seven]);
            b.ret(Some(r.into()));
        });
        p.set_main(|b, p| {
            let args: Vec<Operand> = (1..=10).map(|k| b.int(k * 1000)).collect();
            let r = b.call(f, Some(ret), &args).unwrap();
            for k in 0..4 {
                let i = b.int(k);
                let x = b.binary(Op::Get, t.u64, r, i);
                p.out(b, x, t.u64);
            }
        });
        let ends = 2 * sizes.len() as u64;
        let want: Vec<u8> =
            [55_000 + ends * (ends + 1) / 2, 10_000, 1000, 7].iter().flat_map(|w| w.to_le_bytes()).collect();
        for release in [false, true] {
            let cfg = BuildConfig { release, ..BuildConfig::new(None) };
            let img = build_image(&p.m, &cfg).unwrap_or_else(|e| panic!("{name}: codegen: {e}"));
            if !have("qemu-aarch64") {
                eprintln!("SKIP running AArch64 ELF: no qemu-aarch64");
                continue;
            }
            let path = tmp(name);
            img.write_file(&path).unwrap();
            let out =
                output_retry(std::process::Command::new("qemu-aarch64").args(["-s", "320M"]).arg(&*path));
            assert_eq!(out.status.code(), Some(0), "{name} release={release}: stderr {:?}", out.stderr);
            assert_eq!(out.stdout, want, "{name} release={release}");
        }
    }
}

/// A function with more than 1 MiB of code: `b.cond` to its trap stub (and blocks) is out of
/// the ±1 MiB range, so the function is emitted with long branches.
#[test]
fn function_over_one_mib() {
    let mut p = Prog::new();
    let t = p.t;
    let big = p.func("app.big", &[t.u64], Some(t.u64), |b, _| {
        let (body, done) = (b.create_block(), b.create_block());
        let x0 = b.param(0).unwrap();
        let z = b.int(0);
        let c = b.cmp(Op::Eq, t.bool, x0, z);
        b.cbr(c, done, &[], body, &[]);
        b.switch_to(body);
        let mut x: Operand = x0.into();
        for _ in 0..200_000 {
            let one = b.int(1);
            x = b.binary(Op::Add, t.u64, x, one).into();
        }
        b.ret(Some(x));
        b.switch_to(done);
        let k = b.int(42);
        b.ret(Some(k));
    });
    p.set_main(|b, p| {
        for v in [5u64, 0, u64::MAX - 10] {
            let a = b.int(i128::from(v));
            let r = b.call(big, Some(t.u64), &[a]).unwrap();
            p.out(b, r, t.u64);
        }
    });
    let r = check(&p, "over_1mib");
    assert_eq!(r.code, 101, "the last call overflows");
    assert_eq!(r.stdout.len(), 16);
}

/// User functions whose paths are compiler symbol names get `path@instance` symbols.
#[test]
fn reserved_fn_names() {
    let mut p = Prog::new();
    let t = p.t;
    let mut fs = Vec::new();
    for (k, path) in ["__stave_fmod", "_start", "__stave_main", "__stave_strings"].into_iter().enumerate() {
        fs.push(p.func(path, &[], Some(t.u64), move |b, _| {
            let v = b.int(100 + k as i128);
            b.ret(Some(v));
        }));
    }
    p.set_main(|b, p| {
        for &f in &fs {
            let r = b.call(f, Some(t.u64), &[]).unwrap();
            p.out(b, r, t.u64);
        }
        // the real __stave_fmod still backs float `rem`
        let (x, y) = (b.f64(7.5), b.f64(2.0));
        let r = b.binary(Op::Rem, t.f64, x, y);
        p.out(b, r, t.f64);
    });
    check(&p, "reserved_names");
}

/// The thread child aligns its stack down to 16: the child's frame (a slot at `sp`) is aligned
/// even when the stack top passed to `clone` is not. Native only (the interpreter has one thread).
#[test]
fn thread_child_stack_is_aligned() {
    let mut p = Prog::new();
    let t = p.t;
    let us = t.usize;
    let sys6 = p.m.intrinsic("syscall6");
    let sys0 = p.m.intrinsic("syscall0");
    let clone = p.m.intrinsic("thread_clone");
    let load = p.m.intrinsic("atomic_load_u64");
    let xchg = p.m.intrinsic("atomic_xchg_u64");
    let pu64 = p.m.types.ref_(true, t.u64);
    p.func_known(stave_ir::Known::ThreadEntry, &[us], Some(us), |b, _| {
        let arg = b.param(0).unwrap();
        let s = b.slot(pu64);
        let a = b.cast(CastKind::Exact, us, s);
        let m = b.int(15);
        let lo = b.binary(Op::And, us, a, m);
        let one = b.int(1);
        let v = b.binary(Op::AddWrap, us, lo, one);
        b.intrinsic(xchg, Some(t.u64), &[arg.into(), v.into()]);
        let z = b.int(0);
        b.ret(Some(z));
    });
    p.set_main(|b, p| {
        let ops: Vec<Operand> =
            [222i128, 0, 65536, 3, 0x22, i128::from(u64::MAX), 0].iter().map(|&k| b.int(k)).collect();
        let base = b.intrinsic(sys6, Some(t.isize), &ops).unwrap();
        let base = b.cast(CastKind::Wrap, us, base);
        let off = b.int(65536 - 8);
        let sp = b.binary(Op::AddWrap, us, base, off);
        let (nr, flags, z) = (b.int(220), b.int(0x50f00), b.int(0));
        b.intrinsic(clone, Some(t.isize), &[nr, flags, sp.into(), z, z, z, base.into()]);
        let (head, wait, done) = (b.create_block(), b.create_block(), b.create_block());
        b.br(head, &[]);
        b.switch_to(head);
        let v = b.intrinsic(load, Some(t.u64), &[base.into()]).unwrap();
        let z = b.int(0);
        let c = b.cmp(Op::Eq, t.bool, v, z);
        b.cbr(c, wait, &[], done, &[]);
        b.switch_to(wait);
        let y = b.int(124);
        b.intrinsic(sys0, Some(t.isize), &[y]);
        b.br(head, &[]);
        b.switch_to(done);
        p.out(b, v, t.u64);
    });
    if let Some(r) = native(&p.m, "thread_align", false) {
        assert_eq!(r.code, 0, "{r:?}");
        assert_eq!(r.stdout, 1u64.to_le_bytes(), "child frame address mod 16, plus 1");
    }
}

/// A call of a fn without a body is rejected as unsupported before anything is linked, at the
/// calling fn, as on x86-64. The text gives the reason the fn has no body: an `extern "C"`
/// import, a fn the target does not build (its features belong to another arch), or neither.
#[test]
fn reference_to_bodiless_fn_is_unsupported() {
    let avx2 = stave_target::FeatureSet::of(stave_target::Feature::Avx2);
    let cases: [(Option<&str>, bool, &str); 3] = [
        (Some("puts"), false, "reference to fn `app.puts` without a body (extern imports need a C linker)"),
        (
            None,
            true,
            "reference to fn `app.puts`, which this target does not build: it needs(avx2), features of another arch",
        ),
        (None, false, "reference to fn `app.puts` without a body"),
    ];
    for (symbol, unbuilt, want) in cases {
        let mut p = Prog::new();
        let t = p.t;
        let ext = p.declare("app.puts", &[t.u64], Some(t.u64));
        let f = &mut p.m.funcs[ext.index()];
        f.symbol = symbol.map(String::from);
        f.unbuilt = unbuilt;
        if unbuilt {
            f.features = avx2;
        }
        p.set_main(|b, _| {
            let x = b.int(1);
            b.call(ext, Some(t.u64), &[x]);
        });
        let e = build_image(&p.m, &BuildConfig::new(Some(p.main))).unwrap_err();
        assert!(matches!(&e.kind, ErrorKind::Unsupported(s) if s == want), "{e}");
        assert_eq!(e.func, Some(p.main));
        assert!(e.to_string().starts_with("aarch64 backend: unsupported in fn #"), "{e}");
    }
}

/// A `load` of a whole slot whose one use is the next store, call argument, or `ret` reads the
/// slot in place (stave-opt `inplace`): the frame holds the slots and the `make`, and no home of
/// its own for either such load. A load with a store to its slot before its use keeps a copy of
/// the old bytes.
#[test]
fn whole_slot_load_is_read_in_place() {
    let mut p = Prog::new();
    let t = p.t;
    let arr = p.m.types.array(t.u64, 512);
    let parr = p.m.types.ref_(true, arr);
    let pu64 = p.m.types.ref_(true, t.u64);
    // fn pick(a: [u64; 512], i: usize) -> u64 { a[i] + a[511] }
    let pick = p.func("app.pick", &[arr, t.usize], Some(t.u64), |b, _| {
        let (a, i) = (b.param(0).unwrap(), b.param(1).unwrap());
        let x = b.binary(Op::Get, t.u64, a, i);
        let last = b.int(511);
        let y = b.binary(Op::Get, t.u64, a, last);
        let s = b.binary(Op::AddWrap, t.u64, x, y);
        b.ret(Some(s.into()));
    });
    // fn made(k: u64) -> [u64; 512]: s1 = [0; 512], s1[0] = k, s1[511] = k + 1, s2 = s1,
    // s1[0] = 9, return s2
    let made = p.func("app.made", &[t.u64], Some(arr), |b, _| {
        let k = b.param(0).unwrap();
        let (s1, s2) = (b.slot(parr), b.slot(parr));
        let zs: Vec<Operand> = (0..512).map(|_| b.int(0)).collect();
        let z = b.op(Op::Make, arr, &zs);
        b.store(s1, z);
        let (i0, i511) = (b.int(0), b.int(511));
        let e0 = b.op(Op::ElemAddr, pu64, &[s1.into(), i0]);
        b.store(e0, k);
        let e511 = b.op(Op::ElemAddr, pu64, &[s1.into(), i511]);
        let one = b.int(1);
        let k1 = b.binary(Op::AddWrap, t.u64, k, one);
        b.store(e511, k1);
        let v = b.load(arr, s1);
        b.store(s2, v);
        let nine = b.int(9);
        b.store(e0, nine);
        let w = b.load(arr, s2);
        b.ret(Some(w.into()));
    });
    p.set_main(|b, p| {
        let five = b.int(5);
        let r = b.call(made, Some(arr), &[five]).unwrap();
        let z = b.int(0);
        let q = b.call(pick, Some(t.u64), &[r.into(), z]).unwrap();
        p.out(b, q, t.u64);
        let s3 = b.slot(parr);
        b.store(s3, r);
        let old = b.load(arr, s3);
        let i0 = b.int(0);
        let e0 = b.op(Op::ElemAddr, pu64, &[s3.into(), i0]);
        let one = b.int(1);
        b.store(e0, one);
        let z = b.int(0);
        let q = b.call(pick, Some(t.u64), &[old.into(), z]).unwrap();
        p.out(b, q, t.u64);
        let new = b.load(arr, s3);
        let z = b.int(0);
        let q = b.call(pick, Some(t.u64), &[new.into(), z]).unwrap();
        p.out(b, q, t.u64);
    });
    let r = check(&p, "in_place_load");
    let words: Vec<u8> = r.stdout.chunks(8).map(|c| c[0]).collect();
    assert_eq!(words, [11, 11, 7]);
    let mut img = ImageBuilder::new(Machine::Aarch64, Options::debug(Machine::Aarch64));
    let syms = prepare(&mut img, &p.m, None).unwrap();
    let cg = Codegen::new(&p.m, &syms);
    let (_, made_use) = cg.compile_stack(made, &mut FuncScratch::default()).unwrap();
    // two slots and the `make`: 3 x 4 KiB; a home per load would add 8 KiB
    assert!(made_use.frame < 13 * 1024, "{}", made_use.frame);
}
