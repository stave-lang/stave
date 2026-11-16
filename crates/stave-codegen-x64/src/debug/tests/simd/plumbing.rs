//! Vector and mask values outside the vector ops: parameters and returns (more than 8 vector
//! arguments mixed with floats and ints), block parameters, `select` on a `bool`, tuples /
//! `Option` / enums holding vectors, slots, and the 32-byte alignment of 256-bit slots.

use stave_ir::{CastKind, FuncBuilder, Op, Operand, TypeId, Value, VecOp};

use super::values::Rng;
use super::{Case, Lane, Shape, Sp, sh};

/// A parameter of a test callee.
#[derive(Clone, Copy)]
enum P {
    V(Shape),
    M(u32, u32),
    F64,
    I64,
}

fn lanes(s: Shape, seed: u64) -> Vec<u64> {
    let mut r = Rng::new(seed * 7 + 3);
    (0..s.n).map(|_| r.next() & s.lane.ones()).collect()
}

impl Sp {
    fn p_ty(&mut self, p: P) -> TypeId {
        match p {
            P::V(s) => self.vt(s),
            P::M(w, n) => self.mt(w, n),
            P::F64 => self.t().f64,
            P::I64 => self.t().i64,
        }
    }

    fn p_out(&mut self, b: &mut FuncBuilder, v: Value, p: P) {
        match p {
            P::V(s) => self.out(b, v, s),
            P::M(w, n) => self.out_mask(b, v, w, n),
            P::F64 => self.out_scalar(b, v, Lane::F64),
            P::I64 => self.out_scalar(b, v, Lane::I64),
        }
    }

    fn p_arg(&mut self, b: &mut FuncBuilder, p: P, seed: u64) -> Operand {
        match p {
            P::V(s) => self.vec(b, s, &lanes(s, seed)).into(),
            P::M(w, n) => self.mask(b, w, n, seed.wrapping_mul(0x9e37_79b9) & ((1 << n) - 1)).into(),
            P::F64 => b.f64(seed as f64 * -1.25 + 0.5),
            P::I64 => b.int(seed as i128 * -1_000_003),
        }
    }
}

/// `callee(params…)` prints every argument and returns the last vector parameter; the caller
/// prints what came back. A second callee returns a `(V, f64, i64)` of its arguments.
fn calls(tag: &'static str, params: Vec<P>) -> Case {
    let bits = params
        .iter()
        .map(|p| match p {
            P::V(s) => s.bits(),
            P::M(w, n) => w * n,
            _ => 0,
        })
        .max()
        .unwrap_or(0);
    Case::new(&[], bits, 400, move |sp, b| {
        let tys: Vec<TypeId> = params.iter().map(|&p| sp.p_ty(p)).collect();
        let (ri, rp) = params
            .iter()
            .enumerate()
            .rev()
            .find(|(_, p)| matches!(p, P::V(_)))
            .map(|(i, p)| (i, *p))
            .unwrap();
        let rt = tys[ri];
        let f = sp.p.decl(&format!("t.callee_{tag}"), &tys, Some(rt));
        let mut fb = FuncBuilder::new(&tys);
        for (k, &p) in params.iter().enumerate() {
            let v = fb.param(k).unwrap();
            sp.p_out(&mut fb, v, p);
        }
        let r = fb.param(ri).unwrap();
        fb.ret(Some(r.into()));
        sp.p.def(f, fb);
        // (V, f64, i64) from the first vector, f64 and i64 parameters
        let (vi, vp) =
            params.iter().enumerate().find(|(_, p)| matches!(p, P::V(_))).map(|(i, p)| (i, *p)).unwrap();
        let P::V(vs) = vp else { unreachable!() };
        let fi = params.iter().position(|p| matches!(p, P::F64)).unwrap();
        let ii = params.iter().position(|p| matches!(p, P::I64)).unwrap();
        let tup = sp.p.m.types.tuple(vec![tys[vi], sp.t().f64, sp.t().i64]);
        let g = sp.p.decl(&format!("t.agg_{tag}"), &tys, Some(tup));
        let mut gb = FuncBuilder::new(&tys);
        let (a, x, y) = (gb.param(vi).unwrap(), gb.param(fi).unwrap(), gb.param(ii).unwrap());
        let t = gb.op(Op::Make, tup, &[a.into(), x.into(), y.into()]);
        gb.ret(Some(t.into()));
        sp.p.def(g, gb);

        sp.label(b, &format!("call {tag}"), None);
        let args: Vec<Operand> =
            params.iter().enumerate().map(|(k, &p)| sp.p_arg(b, p, k as u64 + 1)).collect();
        let r = b.call(f, Some(rt), &args).unwrap();
        sp.label(b, &format!("call {tag} returned"), None);
        sp.p_out(b, r, rp);
        let t = b.call(g, Some(tup), &args).unwrap();
        sp.label(b, &format!("call {tag} returned a tuple"), None);
        let (k0, k1, k2) = (b.int(0), b.int(1), b.int(2));
        let a = b.op(Op::Get, tys[vi], &[t.into(), k0]);
        sp.out(b, a, vs);
        let x = b.op(Op::Get, sp.t().f64, &[t.into(), k1]);
        sp.out_scalar(b, x, Lane::F64);
        let y = b.op(Op::Get, sp.t().i64, &[t.into(), k2]);
        sp.out_scalar(b, y, Lane::I64);
    })
}

/// A loop carrying two vectors that swap every iteration (one also gains a constant) and a mask
/// that flips.
fn loop_swap(s: Shape) -> Case {
    Case::new(&[], s.bits(), 200, move |sp, b| {
        let t = sp.t();
        let vt = sp.vt(s);
        let (w, n) = (s.lane.bits(), s.n);
        let mt = sp.mt(w, n);
        let a = sp.vec(b, s, &lanes(s, 1));
        let c = sp.vec(b, s, &lanes(s, 2));
        let k = sp.vec(b, s, &vec![1; n as usize]);
        let m = sp.mask(b, w, n, 0b01);
        let (head, hv) = b.create_block_with(&[t.u32, vt, vt, mt]);
        let body = b.create_block();
        let exit = b.create_block();
        let z = b.int(0);
        b.br(head, &[z, a.into(), c.into(), m.into()]);
        b.switch_to(head);
        let (i, x, y, mm) = (hv[0], hv[1], hv[2], hv[3]);
        let five = b.int(5);
        let go = b.cmp(Op::Lt, t.bool, i, five);
        b.cbr(go, body, &[], exit, &[]);
        b.switch_to(body);
        sp.label(b, &format!("loop {}", s.name()), None);
        sp.out(b, x, s);
        sp.out(b, y, s);
        sp.out_mask(b, mm, w, n);
        let op = if s.lane.is_float() { VecOp::Add } else { VecOp::AddWrap };
        let x2 = b.op(Op::Vec(op), vt, &[x.into(), k.into()]);
        let m2 = b.op(Op::Vec(VecOp::MaskNot), mt, &[mm.into()]);
        let one = b.int(1);
        let i2 = b.binary(Op::AddWrap, t.u32, i, one);
        b.br(head, &[i2.into(), y.into(), x2.into(), m2.into()]);
        b.switch_to(exit);
        sp.label(b, &format!("loop {} done", s.name()), None);
        sp.out(b, x, s);
        sp.out(b, y, s);
        sp.out_mask(b, mm, w, n);
    })
}

/// `select` on a `bool` between two vectors and between two masks; tuples, `Option` and an enum
/// holding vectors, passed by value through calls; vectors in slots, arrays and structs.
fn aggregates(s: Shape) -> Case {
    Case::new(&[], s.bits(), 400, move |sp, b| {
        let t = sp.t();
        let vt = sp.vt(s);
        let (w, n) = (s.lane.bits(), s.n);
        let mt = sp.mt(w, n);
        let a = sp.vec(b, s, &lanes(s, 3));
        let c = sp.vec(b, s, &lanes(s, 4));
        let ma = sp.mask(b, w, n, 0b10);
        let mc = sp.mask(b, w, n, 0b01);
        for cond in [0, 1] {
            sp.label(b, &format!("select bool {cond} {}", s.name()), None);
            let k = b.int(cond);
            let cv = sp.p.opaque(b, k, t.bool);
            let r = b.select(vt, cv, a, c);
            sp.out(b, r, s);
            let rm = b.select(mt, cv, ma, mc);
            sp.out_mask(b, rm, w, n);
        }
        // tuple (i32, V, f64, V) through an opaque call
        let tup = sp.p.m.types.tuple(vec![t.i32, vt, t.f64, vt]);
        let (x1, x2) = (b.int(-7), b.f64(2.5));
        let tv = b.op(Op::Make, tup, &[x1, a.into(), x2, c.into()]);
        let tv = sp.p.opaque(b, tv.into(), tup);
        sp.label(b, &format!("tuple {}", s.name()), None);
        let k1 = b.int(1);
        let g1 = b.op(Op::Get, vt, &[tv.into(), k1]);
        sp.out(b, g1, s);
        let k3 = b.int(3);
        let g3 = b.op(Op::Get, vt, &[tv.into(), k3]);
        sp.out(b, g3, s);
        let k0 = b.int(0);
        let g0 = b.op(Op::Get, t.i32, &[tv.into(), k0]);
        sp.out_scalar(b, g0, Lane::I32);
        // Option<V>: Some and None through an opaque call
        let ot = sp.p.option(vt);
        for some in [true, false] {
            sp.label(b, &format!("option {} some {some}", s.name()), None);
            let o = if some {
                let v1 = b.int(1);
                b.op(Op::EnumMake, ot, &[v1, c.into()])
            } else {
                let v0 = b.int(0);
                b.op(Op::EnumMake, ot, &[v0])
            };
            let o = sp.p.opaque(b, o.into(), ot);
            sp.out_opt(b, o, s);
        }
        // an enum with a vector in each variant
        let half = sh(s.lane, (n / 2).max(2));
        let ht = sp.vt(half);
        let et = sp.p.enumeration(vec![vec![vt], vec![t.u8, ht]]);
        let hv = sp.vec(b, half, &lanes(half, 9));
        let v1 = b.int(1);
        let k = b.int(42);
        let e = b.op(Op::EnumMake, et, &[v1, k, hv.into()]);
        let e = sp.p.opaque(b, e.into(), et);
        sp.label(b, &format!("enum {}", s.name()), None);
        let tag = b.op(Op::EnumTag, t.u32, &[e.into()]);
        sp.out_scalar(b, tag, Lane::U32);
        let (v1, f1) = (b.int(1), b.int(1));
        let got = b.op(Op::EnumGet, ht, &[e.into(), v1, f1]);
        sp.out(b, got, half);
        // slot: store, load back
        sp.label(b, &format!("slot {}", s.name()), None);
        let pv = sp.p.m.types.ref_(true, vt);
        let sl = b.slot(pv);
        b.store(sl, c);
        let back = b.load(vt, sl);
        sp.out(b, back, s);
        // array of three vectors
        sp.label(b, &format!("array of {}", s.name()), None);
        let arr = sp.p.m.types.array(vt, 3);
        let pa = sp.p.m.types.ref_(true, arr);
        let asl = b.slot(pa);
        for (j, x) in [a, c, back].into_iter().enumerate() {
            let kj = b.int(j as i128);
            let p = b.op(Op::ElemAddr, pv, &[asl.into(), kj]);
            b.store(p, x);
        }
        for j in [2, 0, 1] {
            let kj = b.int(j);
            let p = b.op(Op::ElemAddr, pv, &[asl.into(), kj]);
            let x = b.load(vt, p);
            sp.out(b, x, s);
        }
        // struct { u8, V, u16 } in a slot
        sp.label(b, &format!("struct with {}", s.name()), None);
        let st = sp.p.structure(vec![t.u8, vt, t.u16]);
        let ps = sp.p.m.types.ref_(true, st);
        let ss = b.slot(ps);
        let k1 = b.int(1);
        let fp = b.op(Op::FieldAddr, pv, &[ss.into(), k1]);
        b.store(fp, a);
        let k1b = b.int(1);
        let fp2 = b.op(Op::FieldAddr, pv, &[ss.into(), k1b]);
        let x = b.load(vt, fp2);
        sp.out(b, x, s);
    })
}

/// The address of a slot of a 256-bit vector (alone, or in a struct) modulo 32, from frames of
/// several shapes and depths.
fn align32() -> Case {
    let mut c = Case::new(&[], 256, 300, move |sp, b| {
        let t = sp.t();
        let s = sh(Lane::F32, 8);
        let vt = sp.vt(s);
        let st = sp.p.structure(vec![t.u8, vt]);
        let leaf = sp.p.decl("t.align_leaf", &[t.u8], None);
        let mut lb = FuncBuilder::new(&[t.u8]);
        let pv = sp.p.m.types.ref_(true, vt);
        let ps = sp.p.m.types.ref_(true, st);
        let x = lb.param(0).unwrap();
        let pu8 = sp.p.m.types.ref_(true, t.u8);
        let s1 = lb.slot(pu8);
        lb.store(s1, x);
        let sv = lb.slot(pv);
        let ss = lb.slot(ps);
        for p in [sv, ss] {
            let a = lb.cast(CastKind::Exact, t.u64, p);
            let k = lb.int(31);
            let r = lb.binary(Op::And, t.u64, a, k);
            lb.call(sp.p.print, None, &[r.into()]);
        }
        let v = sp.vec(&mut lb, s, &lanes(s, 1));
        lb.store(sv, v);
        let back = lb.load(vt, sv);
        sp.out(&mut lb, back, s);
        lb.ret(None);
        sp.p.def(leaf, lb);
        // callers with 1 to 3 extra 8-byte slots, so the incoming stack offset differs
        let mut callee = leaf;
        for depth in 1..=3 {
            let f = sp.p.decl(&format!("t.align_mid{depth}"), &[t.u8], None);
            let mut fb = FuncBuilder::new(&[t.u8]);
            let x = fb.param(0).unwrap();
            let pu64 = sp.p.m.types.ref_(true, t.u64);
            for _ in 0..depth {
                let sl = fb.slot(pu64);
                let k = fb.int(depth);
                fb.store(sl, k);
            }
            fb.call(callee, None, &[x.into()]);
            fb.call(leaf, None, &[x.into()]);
            fb.ret(None);
            sp.p.def(f, fb);
            callee = f;
        }
        sp.label(b, "slot alignment f32x8", None);
        let k = b.int(1);
        b.call(leaf, None, &[k]);
        b.call(callee, None, &[k]);
    });
    c.check = Some(|out| {
        let at = out.find("# slot alignment f32x8\n").expect("alignment label");
        let lines: Vec<&str> = out[at..].lines().skip(1).collect();
        // five leaf calls, each: two addresses mod 32, then the vector's four words
        assert!(lines.len() >= 30, "alignment output: {lines:?}");
        for call in lines.chunks(6).take(5) {
            assert_eq!(&call[..2], &["0000000000000000"; 2], "a 256-bit slot is not 32-byte aligned");
        }
    });
    c
}

pub fn plumbing() -> Vec<Case> {
    let v = |l: Lane, n: u32| P::V(sh(l, n));
    let narrow = vec![
        v(Lane::F32, 4),
        P::F64,
        v(Lane::I32, 4),
        P::I64,
        v(Lane::U8, 16),
        v(Lane::F64, 2),
        P::F64,
        v(Lane::I16, 8),
        v(Lane::U64, 2),
        P::I64,
        v(Lane::F32, 2),
        v(Lane::I8, 16),
        P::F64,
        v(Lane::U32, 4),
        P::M(32, 4),
        v(Lane::I32, 4),
        P::I64,
        v(Lane::F64, 2),
        P::F64,
        v(Lane::U16, 8),
        v(Lane::U8, 4),
    ];
    let wide = vec![
        v(Lane::F32, 8),
        P::F64,
        v(Lane::I16, 16),
        P::I64,
        v(Lane::F64, 4),
        v(Lane::U8, 32),
        v(Lane::F64, 2),
        P::F64,
        v(Lane::F32, 8),
        v(Lane::I64, 4),
        P::M(32, 8),
        v(Lane::U32, 8),
        P::I64,
        P::F64,
        v(Lane::U64, 4),
        v(Lane::I8, 32),
        P::I64,
        v(Lane::F32, 4),
        v(Lane::F64, 4),
    ];
    let mut out = vec![calls("v2", narrow), calls("v3", wide), align32()];
    for s in [sh(Lane::I32, 4), sh(Lane::F64, 2), sh(Lane::U8, 8), sh(Lane::F64, 4), sh(Lane::U16, 16)] {
        out.push(loop_swap(s));
        out.push(aggregates(s));
    }
    out
}
