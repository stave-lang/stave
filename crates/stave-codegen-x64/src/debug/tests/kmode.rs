//! K-mode functions next to functions of other feature sets in one v2-baseline module.
//!
//! The calling convention follows the types, so a function that holds its masks in `k`
//! registers and one that holds them as vectors call each other with masks and vectors of
//! every width class, in registers and on the stack. The program's output is compared with the
//! interpreter's; every function's instructions are checked against its feature set (EVEX and
//! mask-register instructions only in K mode), and the upper parts of the vector registers are
//! clean at every call and return that carries no wide value. The program runs on the `v4` leg.

use stave_ir::{FuncBuilder, FuncId, Op, Operand, TypeId, Value, VecOp};
use stave_target::{Feature, FeatureSet};

use super::features::{Inst, check_image, check_upper_halves, functions, have_objdump};
use super::harness::*;
use super::simd::{Lane, Shape, Sp, baseline_fn, sh};
use crate::debug::{Options, compile_module};

fn k_set() -> FeatureSet {
    FeatureSet::from_features(V4_NEEDS)
}

fn v3_set() -> FeatureSet {
    FeatureSet::from_features([Feature::Avx2, Feature::Fma, Feature::Bmi2, Feature::F16c])
}

fn lanes(s: Shape, seed: u64) -> Vec<u64> {
    (0..u64::from(s.n)).map(|j| (seed * 0x9e37 + j * j * 77 + j) % 23).collect()
}

/// The program: per shape and per pair of feature sets a caller and two callees, and the
/// functions with stack arguments.
struct Mixed {
    sp: Sp,
    /// Every function with `needs`, its set, and whether it is straight-line code.
    fns: Vec<(FuncId, FeatureSet)>,
}

/// `callee(m, a, b) -> mask`: `m ^ (a < b)`; `pick(m, a, b) -> V`: `m.select(a, b)`.
fn callees(sp: &mut Sp, s: Shape, tag: &str) -> (FuncId, FuncId) {
    let (vt, mt) = (sp.vt(s), sp.mt(s.lane.bits(), s.n));
    let f = sp.p.decl(&format!("t.k_mask_{tag}"), &[mt, vt, vt], Some(mt));
    let mut b = FuncBuilder::new(&[mt, vt, vt]);
    let (m, x, y) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
    let lt = b.op(Op::Vec(VecOp::Lt), mt, &[x.into(), y.into()]);
    let r = b.op(Op::Vec(VecOp::MaskXor), mt, &[m.into(), lt.into()]);
    b.ret(Some(r.into()));
    sp.p.def(f, b);
    let g = sp.p.decl(&format!("t.k_pick_{tag}"), &[mt, vt, vt], Some(vt));
    let mut b = FuncBuilder::new(&[mt, vt, vt]);
    let (m, x, y) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
    let r = b.op(Op::Vec(VecOp::Select), vt, &[m.into(), x.into(), y.into()]);
    b.ret(Some(r.into()));
    sp.p.def(g, b);
    (f, g)
}

/// A caller of the two callees: prints the mask and the vector they return, and the mask again
/// after the calls (it lives across them).
fn caller(sp: &mut Sp, s: Shape, tag: &str, (f, g): (FuncId, FuncId)) -> FuncId {
    let (vt, mt) = (sp.vt(s), sp.mt(s.lane.bits(), s.n));
    let c = sp.p.decl(&format!("t.k_call_{tag}"), &[], None);
    let mut b = FuncBuilder::new(&[]);
    let (w, n) = (s.lane.bits(), s.n);
    let x = sp.vec(&mut b, s, &lanes(s, 3));
    let y = sp.vec(&mut b, s, &lanes(s, 8));
    let m = sp.mask(&mut b, w, n, 0x5a5a_c33c_a5a5_3cc3 & if n == 64 { u64::MAX } else { (1 << n) - 1 });
    let r = b.call(f, Some(mt), &[m.into(), x.into(), y.into()]).unwrap();
    sp.label(&mut b, &format!("{tag} mask"), None);
    sp.out_mask(&mut b, r, w, n);
    let v = b.call(g, Some(vt), &[r.into(), x.into(), y.into()]).unwrap();
    sp.label(&mut b, &format!("{tag} pick"), None);
    sp.out(&mut b, v, s);
    sp.label(&mut b, &format!("{tag} mask again"), None);
    sp.out_mask(&mut b, m, w, n);
    b.ret(None);
    sp.p.def(c, b);
    c
}

/// `f(v0..v7: i32x8, m8: mask32x8, m4: mask32x4, m2: mask32x2, x: i32x8) -> mask32x8`: the
/// eight vectors take the vector registers, so the masks and `x` are on the stack. Prints the
/// narrow masks and returns `m8 ^ (v7 < x)`.
fn stack_callee(sp: &mut Sp, tag: &str) -> (FuncId, Vec<TypeId>) {
    let s8 = sh(Lane::I32, 8);
    let v8 = sp.vt(s8);
    let (m8, m4, m2) = (sp.mt(32, 8), sp.mt(32, 4), sp.mt(32, 2));
    let mut params = vec![v8; 8];
    params.extend([m8, m4, m2, v8]);
    let f = sp.p.decl(&format!("t.k_stack_{tag}"), &params, Some(m8));
    let mut b = FuncBuilder::new(&params);
    let p: Vec<Value> = (0..params.len()).map(|i| b.param(i).unwrap()).collect();
    sp.label(&mut b, &format!("{tag} m4"), None);
    sp.out_mask(&mut b, p[9], 32, 4);
    sp.label(&mut b, &format!("{tag} m2"), None);
    sp.out_mask(&mut b, p[10], 32, 2);
    let lt = b.op(Op::Vec(VecOp::Lt), m8, &[p[7].into(), p[11].into()]);
    let r = b.op(Op::Vec(VecOp::MaskXor), m8, &[p[8].into(), lt.into()]);
    b.ret(Some(r.into()));
    sp.p.def(f, b);
    (f, params)
}

fn stack_caller(sp: &mut Sp, tag: &str, f: FuncId) -> FuncId {
    let s8 = sh(Lane::I32, 8);
    let m8 = sp.mt(32, 8);
    let c = sp.p.decl(&format!("t.k_stack_call_{tag}"), &[], None);
    let mut b = FuncBuilder::new(&[]);
    let mut args: Vec<Operand> = (0..8).map(|k| sp.vec(&mut b, s8, &lanes(s8, 20 + k)).into()).collect();
    args.push(sp.mask(&mut b, 32, 8, 0b1011_0010).into());
    args.push(sp.mask(&mut b, 32, 4, 0b0110).into());
    args.push(sp.mask(&mut b, 32, 2, 0b10).into());
    args.push(sp.vec(&mut b, s8, &lanes(s8, 40)).into());
    let r = b.call(f, Some(m8), &args).unwrap();
    sp.label(&mut b, &format!("{tag} stack mask"), None);
    sp.out_mask(&mut b, r, 32, 8);
    b.ret(None);
    sp.p.def(c, b);
    c
}

/// `f(g0..g5: u64, m: mask8x64, n: mask32x16, v, w: i32x16) -> mask32x16`: the six integers take
/// the general argument registers, so the two 512-bit masks arrive on the stack as their bits.
/// Prints `m`'s bits and returns `n ^ (v < w)`.
fn bits_callee(sp: &mut Sp) -> FuncId {
    let t = sp.t();
    let s = sh(Lane::I32, 16);
    let (vt, m64, m16) = (sp.vt(s), sp.mt(8, 64), sp.mt(32, 16));
    let mut params = vec![t.u64; 6];
    params.extend([m64, m16, vt, vt]);
    let f = sp.p.decl("t.k_bits", &params, Some(m16));
    let mut b = FuncBuilder::new(&params);
    let p: Vec<Value> = (0..params.len()).map(|i| b.param(i).unwrap()).collect();
    let bits = b.op(Op::Vec(VecOp::ToBits), t.u64, &[p[6].into()]);
    sp.label(&mut b, "bits of mask8x64", None);
    sp.out_scalar(&mut b, bits, Lane::U64);
    sp.label(&mut b, "last integer", None);
    sp.out_scalar(&mut b, p[5], Lane::U64);
    let lt = b.op(Op::Vec(VecOp::Lt), m16, &[p[8].into(), p[9].into()]);
    let r = b.op(Op::Vec(VecOp::MaskXor), m16, &[p[7].into(), lt.into()]);
    b.ret(Some(r.into()));
    sp.p.def(f, b);
    f
}

fn bits_caller(sp: &mut Sp, f: FuncId) -> FuncId {
    let s = sh(Lane::I32, 16);
    let m16 = sp.mt(32, 16);
    let c = sp.p.decl("t.k_bits_call", &[], None);
    let mut b = FuncBuilder::new(&[]);
    let mut args: Vec<Operand> = (0..6).map(|k| b.int(1000 + k)).collect();
    args.push(sp.mask(&mut b, 8, 64, 0x8000_0000_dead_beef).into());
    args.push(sp.mask(&mut b, 32, 16, 0xa5c3).into());
    args.push(sp.vec(&mut b, s, &lanes(s, 5)).into());
    args.push(sp.vec(&mut b, s, &lanes(s, 6)).into());
    let r = b.call(f, Some(m16), &args).unwrap();
    sp.label(&mut b, "bits call mask", None);
    sp.out_mask(&mut b, r, 32, 16);
    b.ret(None);
    sp.p.def(c, b);
    c
}

fn mixed() -> Mixed {
    let mut sp = Sp::new();
    let (k, v3) = (k_set(), v3_set());
    let mut fns = Vec::new();
    let mut callers = Vec::new();
    // (caller's set, callees' set): a K-mode fn calls and is called by one that is not
    let pairs = [("k_v3", k, v3), ("v3_k", v3, k), ("k_k", k, k)];
    for s in [sh(Lane::I32, 2), sh(Lane::I32, 4), sh(Lane::I16, 16), sh(Lane::F64, 4), sh(Lane::U8, 32)] {
        for (name, cs, fs) in pairs {
            let tag = format!("{}_{name}", s.name());
            let (f, g) = callees(&mut sp, s, &tag);
            let c = caller(&mut sp, s, &tag, (f, g));
            fns.extend([(f, fs), (g, fs), (c, cs)]);
            callers.push(c);
        }
    }
    // 512-bit values exist in K mode only
    for s in [sh(Lane::I32, 16), sh(Lane::U8, 64), sh(Lane::F64, 8)] {
        let tag = format!("{}_k_k", s.name());
        let (f, g) = callees(&mut sp, s, &tag);
        let c = caller(&mut sp, s, &tag, (f, g));
        fns.extend([(f, k), (g, k), (c, k)]);
        callers.push(c);
    }
    for (name, cs, fs) in pairs {
        let (f, _) = stack_callee(&mut sp, name);
        let c = stack_caller(&mut sp, name, f);
        fns.extend([(f, fs), (c, cs)]);
        callers.push(c);
    }
    let f = bits_callee(&mut sp);
    let c = bits_caller(&mut sp, f);
    fns.extend([(f, k), (c, k)]);
    callers.push(c);
    sp.p.main(|_, b| {
        for c in callers {
            b.call(c, None, &[]);
        }
    });
    // the helpers that build and print vectors of every width are K-mode code
    for f in &mut sp.p.m.funcs {
        if !baseline_fn(&f.path) {
            f.features = k;
        }
    }
    for &(f, set) in &fns {
        sp.p.m.funcs[f.index()].features = set;
    }
    Mixed { sp, fns }
}

#[test]
fn k_mode_and_other_fns_share_signatures() {
    let mx = mixed();
    let p = &mx.sp.p;
    let m = &p.m;
    stave_ir::verify_module(m).unwrap_or_else(|e| panic!("IR does not verify: {e:?}"));
    let want = interp(p);
    assert_eq!(want.code, 0, "interpreter exit");
    let leg = v4_leg();
    for (build, o) in [("debug", Options::default()), ("release", release_options())] {
        let what = format!("k-mixed [{build}]");
        let o = Options { main: p.main, ..o };
        let img = compile_module(m, &o).unwrap_or_else(|e| panic!("{what}: codegen: {e}"));
        if have_objdump() {
            check_image(m, &img, o.features, &what);
            let code = functions(m, &img, o.features);
            for &(f, set) in &mx.fns {
                let func = &m.funcs[f.index()];
                let c: &[Inst] = &code.iter().find(|x| x.0 == func.path).expect("code of a fn").2;
                let kmode = V4_NEEDS.iter().all(|x| set.contains(*x));
                let evex = c.iter().any(|i| i.evex || i.mnem.starts_with("kmov"));
                assert_eq!(evex, kmode, "{what}: `{}`: AVX-512 instructions and K mode", func.path);
                // straight-line code: the callers call, the callees of the stack test print
                check_upper_halves(m, func, c, (false, false), &what);
            }
        }
        let Some(leg) = &leg else {
            eprintln!("SKIP run of {what}: no CPU with the AVX-512 set and no STAVE_SDE");
            continue;
        };
        let path = tmp(&format!("k-mixed-{build}"));
        img.write_file(&path).unwrap();
        let out = v4_command(leg, &path).output().unwrap();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{what}: exit; stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&want.stdout),
            "{what}: output against the interpreter"
        );
    }
}

/// A 512-bit vector or mask in a function without the whole AVX-512 set is `Unsupported`, and
/// the message names the features the function lacks.
#[test]
fn zmm_outside_k_mode_is_unsupported() {
    use crate::debug::Error;
    let sets: [(&[Feature], &str); 3] = [
        (&[], "`avx512f`, `avx512bw`, `avx512vl`, `avx512dq`, `avx512cd`)"),
        (&[Feature::Avx512f, Feature::Avx512bw], "the fn lacks `avx512vl`, `avx512dq`, `avx512cd`)"),
        (
            &[Feature::Avx512f, Feature::Avx512bw, Feature::Avx512vl, Feature::Avx512dq],
            "the fn lacks `avx512cd`)",
        ),
    ];
    for (needs, lacks) in sets {
        let mut sp = Sp::new();
        let vt = sp.vt(sh(Lane::F32, 16));
        let f = sp.p.decl("t.zmm", &[vt, vt], Some(vt));
        let mut b = FuncBuilder::new(&[vt, vt]);
        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
        let r = b.op(Op::Vec(VecOp::Add), vt, &[x.into(), y.into()]);
        b.ret(Some(r.into()));
        sp.p.def(f, b);
        sp.p.main(|_, _| {});
        sp.p.m.funcs[f.index()].features = FeatureSet::from_features(needs.iter().copied());
        for o in [Options::default(), release_options()] {
            let o = Options { main: sp.p.main, features: v3(), ..o };
            match compile_module(&sp.p.m, &o) {
                Err(Error::Unsupported { what, .. }) => {
                    assert!(what.contains(lacks), "{needs:?}: message `{what}`")
                }
                r => panic!("{needs:?}: expected Unsupported, got {:?}", r.map(|_| ())),
            }
        }
    }
}
