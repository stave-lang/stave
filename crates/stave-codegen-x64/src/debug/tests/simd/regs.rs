//! Vector values under register pressure: more live vectors than registers in one block and
//! across loops, accumulators rotated through block parameters, vectors live across calls and
//! across trap edges, many vector and float arguments in permuted order, aggregates, scalar float
//! code interleaved with vector code, and ops that need memory or fixed registers. Also the IR
//! shapes a code generator may fuse (a `vec.load` feeding the next op's second operand,
//! `check.bounds i (sub.sat n K)`) next to near-misses that must not fuse.
//!
//! Inputs are stored word by word into slots whose addresses pass through an opaque call, and are
//! loaded only after the last such call, so the loaded vectors are not constants and no call
//! separates their definitions. Float inputs are small multiples of 1/8 so results stay finite.

use stave_ir::{CastKind, FuncBuilder, FuncId, Op, Operand, PanicKind, TypeId, TypeKey, Value, VecOp};
use stave_target::FeatureSet;

use super::values::{Rng, fl, nans};
use super::{Case, Lane, Shape, Sp, sh};

fn rnd(s: Shape, seed: u64) -> Vec<u64> {
    let mut r = Rng::new(seed * 131 + s.n as u64 * 7 + s.lane.bits() as u64);
    (0..s.n)
        .map(|_| {
            if s.lane.is_float() {
                fl(s.lane, (r.below(129) as f64 - 64.0) / 8.0)
            } else {
                r.next() & s.lane.ones()
            }
        })
        .collect()
}

/// Lanes `0..s.n` from `f(j)`.
fn lanes_of(s: Shape, mut f: impl FnMut(u64) -> u64) -> Vec<u64> {
    (0..s.n as u64).map(|j| f(j) & s.lane.ones()).collect()
}

fn add(s: Shape) -> VecOp {
    if s.lane.is_float() { VecOp::Add } else { VecOp::AddWrap }
}

fn sub(s: Shape) -> VecOp {
    if s.lane.is_float() { VecOp::Sub } else { VecOp::SubWrap }
}

fn mul(s: Shape) -> VecOp {
    if s.lane.is_float() { VecOp::Mul } else { VecOp::MulWrap }
}

fn vop(sp: &mut Sp, b: &mut FuncBuilder, op: VecOp, s: Shape, args: &[Value]) -> Value {
    let vt = sp.vt(s);
    let a: Vec<Operand> = args.iter().map(|&v| v.into()).collect();
    b.op(Op::Vec(op), vt, &a)
}

fn vsplat(sp: &mut Sp, b: &mut FuncBuilder, s: Shape, x: Value) -> Value {
    vop(sp, b, VecOp::Splat, s, &[x])
}

/// The mask of a compare of two vectors of `s`.
fn vcmp(sp: &mut Sp, b: &mut FuncBuilder, op: VecOp, s: Shape, x: Value, y: Value) -> Value {
    let mt = sp.mt(s.lane.bits(), s.n);
    b.op(Op::Vec(op), mt, &[x.into(), y.into()])
}

fn nan(l: Lane) -> Option<u32> {
    l.is_float().then(|| l.bits())
}

fn show(sp: &mut Sp, b: &mut FuncBuilder, what: &str, v: Value, s: Shape) {
    sp.label(b, &format!("{what} {}", s.name()), nan(s.lane));
    sp.out(b, v, s);
}

fn show_mask(sp: &mut Sp, b: &mut FuncBuilder, what: &str, m: Value, s: Shape) {
    let (w, n) = (s.lane.bits(), s.n);
    sp.label(b, &format!("{what} mask{w}x{n}"), None);
    sp.out_mask(b, m, w, n);
}

fn show_scalar(sp: &mut Sp, b: &mut FuncBuilder, what: &str, x: Value, l: Lane) {
    sp.label(b, &format!("{what} {}", l.name()), nan(l));
    sp.out_scalar(b, x, l);
}

fn show_opt(sp: &mut Sp, b: &mut FuncBuilder, what: &str, o: Value, s: Shape) {
    sp.label(b, &format!("{what} Option<{}>", s.name()), nan(s.lane));
    sp.out_opt(b, o, s);
}

/// An opaque copy of the constant `c` of type `ty`.
fn opq(sp: &mut Sp, b: &mut FuncBuilder, c: Operand, ty: TypeId) -> Value {
    sp.p.opaque(b, c, ty)
}

fn opq_int(sp: &mut Sp, b: &mut FuncBuilder, ty: TypeId, x: i128) -> Value {
    let c = b.int(x);
    opq(sp, b, c, ty)
}

fn opq_f64(sp: &mut Sp, b: &mut FuncBuilder, x: f64) -> Value {
    let t = sp.t();
    let c = b.f64(x);
    opq(sp, b, c, t.f64)
}

fn opq_f32(sp: &mut Sp, b: &mut FuncBuilder, x: f32) -> Value {
    let t = sp.t();
    let c = b.f32(x);
    opq(sp, b, c, t.f32)
}

/// Vectors of the given shapes and lanes (each at least 8 bytes): see the module doc.
fn inputs(sp: &mut Sp, b: &mut FuncBuilder, items: &[(Shape, Vec<u64>)]) -> Vec<Value> {
    let t = sp.t();
    let pe = sp.p.m.types.ref_(true, t.u64);
    let mut ptrs = Vec::new();
    for (s, lanes) in items {
        assert!(s.bytes() >= 8, "input {} narrower than a word", s.name());
        let vt = sp.vt(*s);
        let pv = sp.p.m.types.ref_(true, vt);
        let arr = sp.p.m.types.array(t.u64, (s.bytes() / 8) as u64);
        let pa = sp.p.m.types.ref_(true, arr);
        let sl = b.slot(pv);
        let raw = b.cast(CastKind::Exact, pa, sl);
        for (j, w) in Sp::words(*s, lanes).into_iter().enumerate() {
            let kj = b.int(j as i128);
            let a = b.op(Op::ElemAddr, pe, &[raw.into(), kj]);
            let c = b.int(w as i128);
            b.store(a, c);
        }
        let p = opq(sp, b, sl.into(), pv);
        ptrs.push((p, vt));
    }
    ptrs.into_iter().map(|(p, vt)| b.load(vt, p)).collect()
}

/// Rough instruction count of [`inputs`] for `k` vectors of `bits` bits.
fn inputs_cost(k: usize, bits: u32) -> usize {
    k * (6 + 2 * (bits as usize / 64))
}

fn counted_loop(
    b: &mut FuncBuilder,
    t: super::Tys,
    params: &[TypeId],
) -> (stave_ir::Block, Vec<Value>, stave_ir::Block, stave_ir::Block) {
    let mut tys = vec![t.u32];
    tys.extend_from_slice(params);
    let (head, hv) = b.create_block_with(&tys);
    let body = b.create_block();
    let exit = b.create_block();
    (head, hv, body, exit)
}

fn next_i(b: &mut FuncBuilder, t: super::Tys, i: Value) -> Value {
    let one = b.int(1);
    b.binary(Op::AddWrap, t.u32, i, one)
}

/// 24 vectors of `s` live at once in one block (12 loaded, 12 derived without calls), combined in
/// reverse order, several printed after the sum.
fn pressure(s: Shape) -> Case {
    let ops = [add(s), sub(s), mul(s)];
    Case::new(&ops, s.bits(), inputs_cost(12, s.bits()) + 80, move |sp, b| {
        let tag = format!("pressure {}", s.name());
        let items: Vec<(Shape, Vec<u64>)> = (0..12).map(|k| (s, rnd(s, 10 + k))).collect();
        let mut vs = inputs(sp, b, &items);
        for k in 0..12 {
            let v = vop(sp, b, ops[k % 3], s, &[vs[k], vs[k + 1]]);
            vs.push(v);
        }
        let mut acc = vs[23];
        for k in (0..23).rev() {
            acc = vop(sp, b, ops[k % 3], s, &[acc, vs[k]]);
        }
        show(sp, b, &format!("{tag} sum"), acc, s);
        for k in [0, 5, 11, 12, 17, 22, 23] {
            show(sp, b, &format!("{tag} v{k}"), vs[k], s);
        }
    })
}

/// 8 `f32x4`, 8 `f32x8`, their 8 joins and sums, 8 reductions and 12 `f64` scalars live at once.
fn pressure_mixed() -> Case {
    let (n4, n8) = (sh(Lane::F32, 4), sh(Lane::F32, 8));
    let ops = [VecOp::Add, VecOp::Sub, VecOp::Join, VecOp::ReduceAdd, VecOp::Splat];
    Case::new(&ops, 256, inputs_cost(8, 128) + inputs_cost(8, 256) + 150, move |sp, b| {
        let t = sp.t();
        let tag = "pressure mixed";
        let fs: Vec<Value> = (0..6).map(|k| opq_f64(sp, b, 0.75 * k as f64 - 1.875)).collect();
        let mut items: Vec<(Shape, Vec<u64>)> = (0..8).map(|k| (n4, rnd(n4, 30 + k))).collect();
        items.extend((0..8).map(|k| (n8, rnd(n8, 40 + k))));
        let vs = inputs(sp, b, &items);
        let (nv, wv) = vs.split_at(8);
        let mut ws = Vec::new();
        for k in 0..8 {
            let j = vop(sp, b, VecOp::Join, n8, &[nv[k], nv[(k + 1) % 8]]);
            ws.push(vop(sp, b, VecOp::Add, n8, &[wv[k], j]));
        }
        let mut ss = Vec::new();
        for k in 0..6 {
            let p = b.binary(Op::Mul, t.f64, fs[k], fs[(k + 1) % 6]);
            ss.push(b.binary(Op::Add, t.f64, p, fs[k]));
        }
        let mut rs = Vec::new();
        for &v in nv {
            let r = b.op(Op::Vec(VecOp::ReduceAdd), t.f32, &[v.into()]);
            rs.push(b.cast(CastKind::Exact, t.f64, r));
        }
        let mut acc8 = ws[7];
        for k in (0..7).rev() {
            acc8 = vop(sp, b, if k % 2 == 0 { VecOp::Sub } else { VecOp::Add }, n8, &[acc8, ws[k]]);
        }
        let mut acc4 = nv[7];
        for k in (0..7).rev() {
            acc4 = vop(sp, b, if k % 2 == 0 { VecOp::Add } else { VecOp::Sub }, n4, &[acc4, nv[k]]);
        }
        let mut sacc = ss[5];
        for k in (0..5).rev() {
            sacc = b.binary(Op::Sub, t.f64, sacc, ss[k]);
        }
        for &r in &rs {
            sacc = b.binary(Op::Add, t.f64, sacc, r);
        }
        let sf = b.cast(CastKind::Round, t.f32, sacc);
        let sv = vsplat(sp, b, n8, sf);
        acc8 = vop(sp, b, VecOp::Add, n8, &[acc8, sv]);
        show(sp, b, &format!("{tag} acc"), acc8, n8);
        show(sp, b, &format!("{tag} acc"), acc4, n4);
        show_scalar(sp, b, &format!("{tag} acc"), sacc, Lane::F64);
        for (k, v) in [nv[0], nv[7]].into_iter().enumerate() {
            show(sp, b, &format!("{tag} narrow {k}"), v, n4);
        }
        for (k, v) in [wv[0], wv[7], ws[3]].into_iter().enumerate() {
            show(sp, b, &format!("{tag} wide {k}"), v, n8);
        }
        for (k, x) in [fs[0], fs[5], ss[2], rs[4]].into_iter().enumerate() {
            show_scalar(sp, b, &format!("{tag} scalar {k}"), x, Lane::F64);
        }
    })
}

/// 12 vectors defined in the entry block and read by every iteration of a loop carrying three
/// vector accumulators; all printed after the loop.
fn loop_pressure(s: Shape) -> Case {
    let ops = if s.lane.is_float() {
        vec![VecOp::Add, VecOp::Sub]
    } else {
        vec![VecOp::AddWrap, VecOp::SubWrap, VecOp::MulWrap, VecOp::Xor]
    };
    Case::new(&ops, s.bits(), inputs_cost(15, s.bits()) + 120, move |sp, b| {
        let t = sp.t();
        let vt = sp.vt(s);
        let tag = format!("loop pressure {}", s.name());
        let cnt = opq_int(sp, b, t.u32, 7);
        let items: Vec<(Shape, Vec<u64>)> = (0..15).map(|k| (s, rnd(s, 50 + k))).collect();
        let vs = inputs(sp, b, &items);
        let (head, hv, body, exit) = counted_loop(b, t, &[vt, vt, vt]);
        let z = b.int(0);
        b.br(head, &[z, vs[12].into(), vs[13].into(), vs[14].into()]);
        b.switch_to(head);
        let go = b.cmp(Op::Lt, t.bool, hv[0], cnt);
        b.cbr(go, body, &[], exit, &[]);
        b.switch_to(body);
        let mut a = [hv[1], hv[2], hv[3]];
        for (k, &v) in vs[..12].iter().enumerate() {
            let j = k / 4;
            let op = match j {
                0 => add(s),
                1 => sub(s),
                _ if s.lane.is_float() => {
                    if k % 2 == 0 {
                        VecOp::Sub
                    } else {
                        VecOp::Add
                    }
                }
                _ => {
                    if k % 2 == 0 {
                        VecOp::MulWrap
                    } else {
                        VecOp::Xor
                    }
                }
            };
            a[j] = vop(sp, b, op, s, &[a[j], v]);
        }
        let i2 = next_i(b, t, hv[0]);
        b.br(head, &[i2.into(), a[0].into(), a[1].into(), a[2].into()]);
        b.switch_to(exit);
        for k in 0..3 {
            show(sp, b, &format!("{tag} acc{k}"), hv[1 + k], s);
        }
        for (k, &v) in vs[..12].iter().enumerate() {
            show(sp, b, &format!("{tag} v{k}"), v, s);
        }
    })
}

/// A loop carrying four vectors whose block arguments permute them by `perm` (`perm[k] == k`:
/// recomputed from its neighbour instead), a fifth vector reading all four, an `f64` and an `i64`.
fn rotate(s: Shape, perm: [usize; 4]) -> Case {
    Case::new(&[add(s), sub(s)], s.bits(), inputs_cost(5, s.bits()) + 120, move |sp, b| {
        let t = sp.t();
        let vt = sp.vt(s);
        let tag = format!("rotate {perm:?} {}", s.name());
        let cnt = opq_int(sp, b, t.u32, 7);
        let f0 = opq_f64(sp, b, 1.25);
        let x0 = opq_int(sp, b, t.i64, -12345);
        let items: Vec<(Shape, Vec<u64>)> = (0..5).map(|k| (s, rnd(s, 70 + k))).collect();
        let vs = inputs(sp, b, &items);
        let (head, hv, body, exit) = counted_loop(b, t, &[vt, vt, vt, vt, vt, t.f64, t.i64]);
        let mut args: Vec<Operand> = vec![b.int(0)];
        args.extend(vs.iter().map(|&v| Operand::from(v)));
        args.extend([Operand::from(f0), Operand::from(x0)]);
        b.br(head, &args);
        b.switch_to(head);
        let go = b.cmp(Op::Lt, t.bool, hv[0], cnt);
        b.cbr(go, body, &[], exit, &[]);
        b.switch_to(body);
        let old = [hv[1], hv[2], hv[3], hv[4]];
        let new: Vec<Value> =
            (0..4)
                .map(|k| {
                    if perm[k] == k {
                        vop(sp, b, sub(s), s, &[old[k], old[(k + 1) % 4]])
                    } else {
                        old[perm[k]]
                    }
                })
                .collect();
        let ab = vop(sp, b, sub(s), s, &[old[0], old[1]]);
        let cd = vop(sp, b, add(s), s, &[old[2], old[3]]);
        let e1 = vop(sp, b, add(s), s, &[hv[5], ab]);
        let e2 = vop(sp, b, sub(s), s, &[e1, cd]);
        let half = b.f64(0.5);
        let fh = b.binary(Op::Mul, t.f64, hv[6], half);
        let fi = b.cast(CastKind::Exact, t.f64, hv[0]);
        let f2 = b.binary(Op::Add, t.f64, fh, fi);
        let three = b.int(3);
        let x3 = b.binary(Op::MulWrap, t.i64, hv[7], three);
        let xi = b.cast(CastKind::Exact, t.i64, hv[0]);
        let x2 = b.binary(Op::AddWrap, t.i64, x3, xi);
        let i2 = next_i(b, t, hv[0]);
        let mut args: Vec<Operand> = vec![i2.into()];
        args.extend(new.iter().map(|&v| Operand::from(v)));
        args.extend([Operand::from(e2), Operand::from(f2), Operand::from(x2)]);
        b.br(head, &args);
        b.switch_to(exit);
        for k in 0..5 {
            show(sp, b, &format!("{tag} v{k}"), hv[1 + k], s);
        }
        show_scalar(sp, b, &format!("{tag} f"), hv[6], Lane::F64);
        show_scalar(sp, b, &format!("{tag} x"), hv[7], Lane::I64);
    })
}

/// A loop carrying a float vector, an int vector of another shape and a mask: the mask is updated
/// from a compare of the float accumulator and selects its next value.
fn mixed_loop(fs: Shape, is: Shape) -> Case {
    let bits = fs.bits().max(is.bits());
    let ops = [VecOp::Add, VecOp::Lt, VecOp::MaskXor, VecOp::Select, VecOp::MulWrap, VecOp::AddWrap];
    Case::new(&ops, bits, inputs_cost(5, bits) + 120, move |sp, b| {
        let t = sp.t();
        let (ft, it) = (sp.vt(fs), sp.vt(is));
        let mt = sp.mt(32, fs.n);
        let tag = format!("mixed loop {} {}", fs.name(), is.name());
        let cnt = opq_int(sp, b, t.u32, 6);
        let m0 = sp.mask(b, 32, fs.n, 0b0110_1001);
        let items = vec![
            (fs, rnd(fs, 80)),
            (fs, rnd(fs, 81)),
            (fs, rnd(fs, 82)),
            (is, rnd(is, 83)),
            (is, rnd(is, 84)),
        ];
        let vs = inputs(sp, b, &items);
        let (fa0, fk, thr, ia0, ik) = (vs[0], vs[1], vs[2], vs[3], vs[4]);
        let (head, hv, body, exit) = counted_loop(b, t, &[ft, it, mt]);
        let z = b.int(0);
        b.br(head, &[z, fa0.into(), ia0.into(), m0.into()]);
        b.switch_to(head);
        let go = b.cmp(Op::Lt, t.bool, hv[0], cnt);
        b.cbr(go, body, &[], exit, &[]);
        b.switch_to(body);
        let (fa, ia, m) = (hv[1], hv[2], hv[3]);
        let fa1 = vop(sp, b, VecOp::Add, fs, &[fa, fk]);
        let lt = vcmp(sp, b, VecOp::Lt, fs, fa1, thr);
        let m1 = b.op(Op::Vec(VecOp::MaskXor), mt, &[m.into(), lt.into()]);
        let fa2 = vop(sp, b, VecOp::Select, fs, &[m1, fa1, fa]);
        let im = vop(sp, b, VecOp::MulWrap, is, &[ia, ik]);
        let ia1 = vop(sp, b, VecOp::AddWrap, is, &[im, ia]);
        let i2 = next_i(b, t, hv[0]);
        b.br(head, &[i2.into(), fa2.into(), ia1.into(), m1.into()]);
        b.switch_to(exit);
        show(sp, b, &format!("{tag} acc"), fa, fs);
        show(sp, b, &format!("{tag} acc"), ia, is);
        show_mask(sp, b, &format!("{tag} acc"), m, fs);
        show(sp, b, &format!("{tag} step"), fk, fs);
        show(sp, b, &format!("{tag} threshold"), thr, fs);
        show(sp, b, &format!("{tag} step"), ik, is);
    })
}

/// What a clobbering callee returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ret {
    Unit,
    F64,
    Narrow,
    Wide,
}

/// `callee(p: N, x: f64[, q: W])`: sums `p` with 6 loaded `N` vectors (and `q` with 6 loaded `W`
/// vectors), so it uses many vector registers of both widths.
fn clobber(
    sp: &mut Sp,
    name: &str,
    n: Shape,
    w: Option<Shape>,
    ret: Ret,
) -> (FuncId, Vec<TypeId>, Option<TypeId>) {
    let t = sp.t();
    let nt = sp.vt(n);
    let mut params = vec![nt, t.f64];
    if let Some(w) = w {
        params.push(sp.vt(w));
    }
    let rt = match ret {
        Ret::Unit => None,
        Ret::F64 => Some(t.f64),
        Ret::Narrow => Some(nt),
        Ret::Wide => Some(sp.vt(w.unwrap())),
    };
    let f = sp.p.decl(name, &params, rt);
    let mut fb = FuncBuilder::new(&params);
    let (p, x) = (fb.param(0).unwrap(), fb.param(1).unwrap());
    let mut items: Vec<(Shape, Vec<u64>)> = (0..6).map(|k| (n, rnd(n, 300 + k))).collect();
    if let Some(w) = w {
        items.extend((0..6).map(|k| (w, rnd(w, 310 + k))));
    }
    let vs = inputs(sp, &mut fb, &items);
    let mut na = p;
    for (k, &v) in vs[..6].iter().enumerate() {
        na = vop(sp, &mut fb, if k % 2 == 0 { add(n) } else { sub(n) }, n, &[na, v]);
    }
    let mut wa = None;
    if let Some(w) = w {
        let mut a = fb.param(2).unwrap();
        for (k, &v) in vs[6..].iter().enumerate() {
            a = vop(sp, &mut fb, if k % 2 == 0 { sub(w) } else { add(w) }, w, &[a, v]);
        }
        let pair = sp.p.m.types.tuple(vec![nt, nt]);
        let halves = fb.op(Op::Vec(VecOp::Split), pair, &[a.into()]);
        let k0 = fb.int(0);
        let lo = fb.op(Op::Get, nt, &[halves.into(), k0]);
        na = vop(sp, &mut fb, add(n), n, &[na, lo]);
        wa = Some(a);
    }
    let r: Option<Operand> = match ret {
        Ret::Unit => {
            sp.out(&mut fb, na, n);
            if let (Some(a), Some(w)) = (wa, w) {
                sp.out(&mut fb, a, w);
            }
            None
        }
        Ret::F64 => {
            let r = fb.op(Op::Vec(VecOp::ReduceAdd), t.f32, &[na.into()]);
            let r = fb.cast(CastKind::Exact, t.f64, r);
            let mut s = fb.binary(Op::Add, t.f64, r, x);
            if let Some(a) = wa {
                let r2 = fb.op(Op::Vec(VecOp::ReduceAdd), t.f32, &[a.into()]);
                let r2 = fb.cast(CastKind::Exact, t.f64, r2);
                s = fb.binary(Op::Sub, t.f64, s, r2);
            }
            Some(s.into())
        }
        Ret::Narrow => Some(na.into()),
        Ret::Wide => {
            let w = w.unwrap();
            let j = vop(sp, &mut fb, VecOp::Join, w, &[na, na]);
            Some(vop(sp, &mut fb, add(w), w, &[wa.unwrap(), j]).into())
        }
    };
    fb.ret(r);
    sp.p.def(f, fb);
    (f, params, rt)
}

/// Vectors live across a call to a callee that clobbers vector registers: before a loop, and
/// accumulators live across the call in the loop body (through a fn value when `ind`).
fn calls(wide: bool, ret: Ret, ind: bool) -> Case {
    let n = sh(Lane::F32, 4);
    let w = wide.then(|| sh(Lane::F32, 8));
    let bits = if wide { 256 } else { 128 };
    let k = if wide { 14 } else { 7 };
    let ops = [VecOp::Add, VecOp::Sub, VecOp::Split, VecOp::Join, VecOp::ReduceAdd];
    Case::new(&ops, bits, 2 * inputs_cost(k, bits) + 200, move |sp, b| {
        let t = sp.t();
        let tag =
            format!("calls {ret:?}{} {}", if ind { " indirect" } else { "" }, if wide { "v3" } else { "v2" });
        let name = format!("t.regs_clobber_{}_{ret:?}_{ind}", if wide { "w" } else { "n" });
        let (f, params, rt) = clobber(sp, &name, n, w, ret);
        let unit = sp.p.m.types.unit();
        let ft = sp.p.m.types.fn_type(params.clone(), rt.unwrap_or(unit), FeatureSet::EMPTY);
        let fv = if ind {
            let fo = b.func(f);
            Some(opq(sp, b, fo, ft))
        } else {
            None
        };
        let x = opq_f64(sp, b, 2.5);
        let cnt = opq_int(sp, b, t.u32, 4);
        let mut items: Vec<(Shape, Vec<u64>)> = (0..7).map(|k| (n, rnd(n, 90 + k))).collect();
        if let Some(w) = w {
            items.extend((0..7).map(|k| (w, rnd(w, 100 + k))));
        }
        let vs = inputs(sp, b, &items);
        let (us, ws) = vs.split_at(7);
        let call = |b: &mut FuncBuilder, args: &[Operand], indirect: bool| match fv {
            Some(g) if indirect => b.call_ind(g, rt, args),
            _ => b.call(f, rt, args),
        };
        let d0 = vop(sp, b, VecOp::Add, n, &[us[0], us[1]]);
        let mut args: Vec<Operand> = vec![d0.into(), x.into()];
        let dw = w.map(|w| vop(sp, b, VecOp::Add, w, &[ws[0], ws[1]]));
        if let Some(d) = dw {
            args.push(d.into());
        }
        sp.label(b, &format!("{tag} first call"), Some(32));
        let r = call(b, &args, false);
        let mut cn = d0;
        for &u in &us[..6] {
            cn = vop(sp, b, VecOp::Sub, n, &[cn, u]);
        }
        if ret == Ret::Narrow {
            cn = vop(sp, b, VecOp::Add, n, &[cn, r.unwrap()]);
        }
        let mut cw = None;
        if let (Some(w), Some(d)) = (w, dw) {
            let mut a = d;
            for &v in &ws[..6] {
                a = vop(sp, b, VecOp::Sub, w, &[a, v]);
            }
            if ret == Ret::Wide {
                a = vop(sp, b, VecOp::Add, w, &[a, r.unwrap()]);
            }
            cw = Some(a);
        }
        show(sp, b, &format!("{tag} after the first call"), cn, n);
        if let (Some(w), Some(a)) = (w, cw) {
            show(sp, b, &format!("{tag} after the first call"), a, w);
        }
        if ret == Ret::F64 {
            show_scalar(sp, b, &format!("{tag} first call returned"), r.unwrap(), Lane::F64);
        }
        // the loop: accumulators (and every input) live across the call in the body
        let nt = sp.vt(n);
        let mut ptys = vec![nt, t.f64];
        if let Some(w) = w {
            ptys.push(sp.vt(w));
        }
        let (head, hv, body, exit) = counted_loop(b, t, &ptys);
        let z = b.int(0);
        let mut init: Vec<Operand> = vec![z, us[6].into(), x.into()];
        if wide {
            init.push(ws[6].into());
        }
        b.br(head, &init);
        b.switch_to(head);
        let go = b.cmp(Op::Lt, t.bool, hv[0], cnt);
        b.cbr(go, body, &[], exit, &[]);
        b.switch_to(body);
        let xi = b.cast(CastKind::Exact, t.f64, hv[0]);
        let mut args: Vec<Operand> = vec![hv[1].into(), xi.into()];
        if wide {
            args.push(hv[3].into());
        }
        let r = call(b, &args, true);
        let mut an = vop(sp, b, VecOp::Add, n, &[hv[1], us[2]]);
        an = vop(sp, b, VecOp::Sub, n, &[an, us[3]]);
        if ret == Ret::Narrow {
            an = vop(sp, b, VecOp::Add, n, &[an, r.unwrap()]);
        }
        let fa = b.binary(Op::Add, t.f64, hv[2], if ret == Ret::F64 { r.unwrap() } else { xi });
        let i2 = next_i(b, t, hv[0]);
        let mut next: Vec<Operand> = vec![i2.into(), an.into(), fa.into()];
        if let Some(w) = w {
            let mut aw = vop(sp, b, VecOp::Add, w, &[hv[3], ws[2]]);
            if ret == Ret::Wide {
                aw = vop(sp, b, VecOp::Sub, w, &[aw, r.unwrap()]);
            }
            next.push(aw.into());
        }
        b.br(head, &next);
        b.switch_to(exit);
        show(sp, b, &format!("{tag} loop acc"), hv[1], n);
        show_scalar(sp, b, &format!("{tag} loop acc"), hv[2], Lane::F64);
        if let Some(w) = w {
            show(sp, b, &format!("{tag} loop acc"), hv[3], w);
        }
        for (k, &u) in us.iter().enumerate() {
            show(sp, b, &format!("{tag} u{k}"), u, n);
        }
        if let Some(w) = w {
            for (k, &v) in ws.iter().enumerate() {
                show(sp, b, &format!("{tag} w{k}"), v, w);
            }
        }
    })
}

/// A parameter of [`many_args`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum A {
    F64,
    I64,
    V(Shape),
    /// A mask of 32-bit lanes.
    M(u32),
}

/// `acc + acc - x` (`x` alone at first): a fold whose result depends on the argument order.
fn fold(sp: &mut Sp, b: &mut FuncBuilder, r: Shape, acc: Option<Value>, x: Value) -> Value {
    match acc {
        None => x,
        Some(a) => {
            let a2 = vop(sp, b, VecOp::Add, r, &[a, a]);
            vop(sp, b, VecOp::Sub, r, &[a2, x])
        }
    }
}

/// `g(params…) -> R` folds every parameter into an `R` (`f32` lanes) in parameter order;
/// `f(params…)` calls `g` with each class of same-typed parameters rotated by one (so the
/// argument moves form cycles) and again in order, prints both, and returns their difference.
fn many_args(tag: &'static str, r: Shape, params: Vec<A>) -> Case {
    let bits = params
        .iter()
        .map(|p| match p {
            A::V(s) => s.bits(),
            A::M(n) => 32 * n,
            _ => 0,
        })
        .chain([r.bits()])
        .max()
        .unwrap();
    let ops =
        [VecOp::Add, VecOp::Sub, VecOp::Join, VecOp::Cast(CastKind::Round), VecOp::ReduceAdd, VecOp::Select];
    Case::new(&ops, bits, 120 + params.len() * 20, move |sp, b| {
        let t = sp.t();
        let rt = sp.vt(r);
        let tys: Vec<TypeId> = params
            .iter()
            .map(|&p| match p {
                A::F64 => t.f64,
                A::I64 => t.i64,
                A::V(s) => sp.vt(s),
                A::M(n) => sp.mt(32, n),
            })
            .collect();
        let g = sp.p.decl(&format!("t.regs_args_g_{tag}"), &tys, Some(rt));
        let mut gb = FuncBuilder::new(&tys);
        let (mut acc, mut half, mut sacc, mut iacc) = (None, None, None::<Value>, None::<Value>);
        let mut masks = Vec::new();
        for (k, &p) in params.iter().enumerate() {
            let v = gb.param(k).unwrap();
            match p {
                A::F64 => {
                    sacc = Some(match sacc {
                        None => v,
                        Some(a) => {
                            let a2 = gb.binary(Op::Add, t.f64, a, a);
                            gb.binary(Op::Sub, t.f64, a2, v)
                        }
                    })
                }
                A::I64 => {
                    iacc = Some(match iacc {
                        None => v,
                        Some(a) => {
                            let c3 = gb.int(3);
                            let m = gb.binary(Op::MulWrap, t.i64, a, c3);
                            gb.binary(Op::AddWrap, t.i64, m, v)
                        }
                    })
                }
                A::V(s) if s == r => acc = Some(fold(sp, &mut gb, r, acc, v)),
                A::V(s) if s.lane == Lane::F32 && s.n * 2 == r.n => match half.take() {
                    None => half = Some(v),
                    Some(h) => {
                        let j = vop(sp, &mut gb, VecOp::Join, r, &[h, v]);
                        acc = Some(fold(sp, &mut gb, r, acc, j));
                    }
                },
                A::V(s) if !s.lane.is_float() && s.n == r.n => {
                    let c = gb.op(Op::Vec(VecOp::Cast(CastKind::Round)), rt, &[v.into()]);
                    acc = Some(fold(sp, &mut gb, r, acc, c));
                }
                A::V(s) if s.lane == Lane::F64 => {
                    let red = gb.op(Op::Vec(VecOp::ReduceAdd), t.f64, &[v.into()]);
                    sacc = Some(match sacc {
                        None => red,
                        Some(a) => gb.binary(Op::Sub, t.f64, a, red),
                    });
                }
                A::M(_) => masks.push(v),
                A::V(s) => panic!("many_args: no fold for {}", s.name()),
            }
        }
        assert!(half.is_none(), "many_args {tag}: odd number of half vectors");
        let si = gb.cast(CastKind::Round, t.f64, iacc.unwrap());
        let st = gb.binary(Op::Add, t.f64, sacc.unwrap(), si);
        let sf = gb.cast(CastKind::Round, t.f32, st);
        let sv = vsplat(sp, &mut gb, r, sf);
        let mut acc = vop(sp, &mut gb, VecOp::Add, r, &[acc.unwrap(), sv]);
        for m in masks {
            let up = vop(sp, &mut gb, VecOp::Add, r, &[acc, sv]);
            let s = vop(sp, &mut gb, VecOp::Select, r, &[m, up, acc]);
            acc = vop(sp, &mut gb, VecOp::Add, r, &[s, s]);
        }
        gb.ret(Some(acc.into()));
        sp.p.def(g, gb);

        let f = sp.p.decl(&format!("t.regs_args_f_{tag}"), &tys, Some(rt));
        let mut fb = FuncBuilder::new(&tys);
        let mut rot: Vec<Operand> = Vec::new();
        for (k, &p) in params.iter().enumerate() {
            let class: Vec<usize> =
                params.iter().enumerate().filter(|(_, q)| **q == p).map(|(j, _)| j).collect();
            let at = class.iter().position(|&j| j == k).unwrap();
            rot.push(fb.param(class[(at + 1) % class.len()]).unwrap().into());
        }
        let own: Vec<Operand> = (0..tys.len()).map(|k| fb.param(k).unwrap().into()).collect();
        let r1 = fb.call(g, Some(rt), &rot).unwrap();
        let r2 = fb.call(g, Some(rt), &own).unwrap();
        sp.out(&mut fb, r1, r);
        sp.out(&mut fb, r2, r);
        let d = vop(sp, &mut fb, VecOp::Sub, r, &[r1, r2]);
        fb.ret(Some(d.into()));
        sp.p.def(f, fb);

        let mut scalars: Vec<Option<Operand>> = Vec::new();
        for (k, &p) in params.iter().enumerate() {
            scalars.push(match p {
                A::F64 => Some(opq_f64(sp, b, k as f64 * 0.625 - 3.0).into()),
                A::I64 => Some(opq_int(sp, b, t.i64, k as i128 * 1_000_003 - 7).into()),
                A::M(n) => Some(sp.mask(b, 32, n, (k as u64 * 0x9e37_79b9) & ((1 << n) - 1)).into()),
                A::V(_) => None,
            });
        }
        let items: Vec<(Shape, Vec<u64>)> = params
            .iter()
            .enumerate()
            .filter_map(|(k, &p)| match p {
                A::V(s) => Some((s, rnd(s, 200 + k as u64))),
                _ => None,
            })
            .collect();
        let mut vs = inputs(sp, b, &items).into_iter();
        let args: Vec<Operand> =
            scalars.into_iter().map(|s| s.unwrap_or_else(|| vs.next().unwrap().into())).collect();
        let x = b.call(g, Some(rt), &args).unwrap();
        show(sp, b, &format!("args {tag} g"), x, r);
        sp.label(b, &format!("args {tag} f prints"), Some(32));
        let y = b.call(f, Some(rt), &args).unwrap();
        show(sp, b, &format!("args {tag} f"), y, r);
    })
}

/// A checked op of [`trap_edges`].
const CHECKS: [(&str, PanicKind); 4] = [
    ("check.bounds", PanicKind::Bounds),
    ("scalar add", PanicKind::Overflow),
    ("vec.shl", PanicKind::Overflow),
    ("vec.div_lanes", PanicKind::DivByZero),
];

/// Four checked ops (bounds, scalar overflow, vector shift amount, vector divisor) between the
/// definitions of float and int vectors (computed in registers, without calls) and their uses;
/// `fire`: the check that traps.
fn trap_edges(fs: Shape, is: Shape, fire: Option<usize>) -> Case {
    let bits = fs.bits().max(is.bits());
    let ops = [add(fs), sub(fs), VecOp::Xor, VecOp::AddWrap, VecOp::SubWrap, VecOp::Shl, VecOp::DivLanes];
    let c = Case::new(&ops, bits, inputs_cost(10, bits) + 150, move |sp, b| {
        let t = sp.t();
        let tag = format!("trap edges {} {} fire {fire:?}", fs.name(), is.name());
        let idx = opq_int(sp, b, t.usize, if fire == Some(0) { 9 } else { 3 });
        let len = opq_int(sp, b, t.usize, 8);
        let xa = opq_int(sp, b, t.i64, if fire == Some(1) { i64::MAX as i128 - 1 } else { 1000 });
        let xb = opq_int(sp, b, t.i64, 5);
        let ib = is.lane.bits() as u64;
        let mut r = Rng::new(ib + fs.n as u64);
        let mut amt: Vec<u64> = (0..is.n).map(|_| r.below(ib)).collect();
        if fire == Some(2) {
            amt[is.n as usize - 1] = ib;
        }
        let mut dv: Vec<u64> = (0..is.n).map(|_| 1 + r.below(100)).collect();
        if fire == Some(3) {
            dv[is.n as usize / 2] = 0;
        }
        let mut items: Vec<(Shape, Vec<u64>)> = (0..4).map(|k| (fs, rnd(fs, 110 + k))).collect();
        items.extend((0..4).map(|k| (is, rnd(is, 120 + k))));
        items.push((is, amt));
        items.push((is, dv));
        let vs = inputs(sp, b, &items);
        let (f, n, amt, dv) = (&vs[..4], &vs[4..8], vs[8], vs[9]);
        show(sp, b, &format!("{tag} before f0"), f[0], fs);
        show(sp, b, &format!("{tag} before n0"), n[0], is);
        let d1 = vop(sp, b, add(fs), fs, &[f[0], f[1]]);
        let e1 = vop(sp, b, VecOp::Xor, is, &[n[0], n[1]]);
        b.bounds_check(idx, len);
        let d2 = vop(sp, b, add(fs), fs, &[d1, f[2]]);
        let e2 = vop(sp, b, VecOp::AddWrap, is, &[e1, n[2]]);
        let s = b.binary(Op::Add, t.i64, xa, xb);
        let d3 = vop(sp, b, sub(fs), fs, &[d2, f[3]]);
        let e3 = vop(sp, b, VecOp::Xor, is, &[e2, n[3]]);
        let sv = vop(sp, b, VecOp::Shl, is, &[n[1], amt]);
        let d4 = vop(sp, b, add(fs), fs, &[d3, f[0]]);
        let e4 = vop(sp, b, VecOp::SubWrap, is, &[e3, n[0]]);
        let q = vop(sp, b, VecOp::DivLanes, is, &[n[2], dv]);
        for (k, &d) in [d1, d2, d3, d4].iter().enumerate() {
            show(sp, b, &format!("{tag} d{}", k + 1), d, fs);
        }
        for (k, &e) in [e1, e2, e3, e4].iter().enumerate() {
            show(sp, b, &format!("{tag} e{}", k + 1), e, is);
        }
        show_scalar(sp, b, &format!("{tag} sum"), s, Lane::I64);
        show(sp, b, &format!("{tag} shl"), sv, is);
        show(sp, b, &format!("{tag} div"), q, is);
        for (k, &v) in f.iter().enumerate() {
            show(sp, b, &format!("{tag} f{k}"), v, fs);
        }
        for (k, &v) in n.iter().enumerate() {
            show(sp, b, &format!("{tag} n{k}"), v, is);
        }
    });
    match fire {
        Some(k) => c.traps(CHECKS[k].1),
        None => c,
    }
}

/// Tuples, a struct and an array in slots, `Option`s and a tuple returned through a callee, and
/// `select` on a `bool`, all built from and mixed with 20 live vectors of shapes `a` and `c`.
fn aggregates(a: Shape, c: Shape) -> Case {
    let bits = a.bits().max(c.bits());
    Case::new(&[add(a), add(c), sub(c)], bits, inputs_cost(10, bits) + 250, move |sp, b| {
        let t = sp.t();
        let (at, ct) = (sp.vt(a), sp.vt(c));
        let tag = format!("aggregates {} {}", a.name(), c.name());
        let tup = sp.p.m.types.tuple(vec![ct, t.f64, at]);
        let ot = sp.p.option(at);
        let pair = sp.p.m.types.tuple(vec![at, ct]);
        // h(T, Option<A>) -> (A, C)
        let h = sp.p.decl(&format!("t.regs_agg_{}_{}", a.name(), c.name()), &[tup, ot], Some(pair));
        let mut hb = FuncBuilder::new(&[tup, ot]);
        let (tv, o) = (hb.param(0).unwrap(), hb.param(1).unwrap());
        let (k0, k2) = (hb.int(0), hb.int(2));
        let ga = hb.op(Op::Get, at, &[tv.into(), k2]);
        let gc = hb.op(Op::Get, ct, &[tv.into(), k0]);
        let tag_v = hb.op(Op::EnumTag, t.u32, &[o.into()]);
        let one = hb.int(1);
        let is_some = hb.cmp(Op::Eq, t.bool, tag_v, one);
        let (some, none) = (hb.create_block(), hb.create_block());
        hb.cbr(is_some, some, &[], none, &[]);
        hb.switch_to(some);
        let (v1, f0) = (hb.int(1), hb.int(0));
        let pl = hb.op(Op::EnumGet, at, &[o.into(), v1, f0]);
        let s = vop(sp, &mut hb, add(a), a, &[ga, pl]);
        let rv = hb.op(Op::Make, pair, &[s.into(), gc.into()]);
        hb.ret(Some(rv.into()));
        hb.switch_to(none);
        let cc = vop(sp, &mut hb, add(c), c, &[gc, gc]);
        let rv = hb.op(Op::Make, pair, &[ga.into(), cc.into()]);
        hb.ret(Some(rv.into()));
        sp.p.def(h, hb);

        let fsc = opq_f64(sp, b, 0.375);
        let c0 = opq_int(sp, b, t.bool, 0);
        let c1 = opq_int(sp, b, t.bool, 1);
        let mut items: Vec<(Shape, Vec<u64>)> = (0..5).map(|k| (a, rnd(a, 130 + k))).collect();
        items.extend((0..5).map(|k| (c, rnd(c, 140 + k))));
        let vs = inputs(sp, b, &items);
        let (av, cv) = vs.split_at(5);
        let xs: Vec<Value> = (0..5).map(|k| vop(sp, b, add(a), a, &[av[k], av[(k + 1) % 5]])).collect();
        let ys: Vec<Value> = (0..5).map(|k| vop(sp, b, sub(c), c, &[cv[k], cv[(k + 1) % 5]])).collect();
        // a tuple, through a slot
        let tv = b.op(Op::Make, tup, &[ys[0].into(), fsc.into(), xs[0].into()]);
        let pt = sp.p.m.types.ref_(true, tup);
        let ts = b.slot(pt);
        b.store(ts, tv);
        let tv2 = b.load(tup, ts);
        let (k0, k1, k2) = (b.int(0), b.int(1), b.int(2));
        let g0 = b.op(Op::Get, at, &[tv2.into(), k2]);
        let g1 = b.op(Op::Get, ct, &[tv2.into(), k0]);
        let g2 = b.op(Op::Get, t.f64, &[tv2.into(), k1]);
        // a struct { u8, A, C } in a slot
        let st = sp.p.structure(vec![t.u8, at, ct]);
        let ps = sp.p.m.types.ref_(true, st);
        let ss = b.slot(ps);
        let (pa, pc) = (sp.p.m.types.ref_(true, at), sp.p.m.types.ref_(true, ct));
        let k1 = b.int(1);
        let fa = b.op(Op::FieldAddr, pa, &[ss.into(), k1]);
        b.store(fa, xs[1]);
        let k2 = b.int(2);
        let fc = b.op(Op::FieldAddr, pc, &[ss.into(), k2]);
        b.store(fc, ys[1]);
        let sx = b.load(at, fa);
        let sy = b.load(ct, fc);
        // an array [A; 3] in a slot
        let arr = sp.p.m.types.array(at, 3);
        let parr = sp.p.m.types.ref_(true, arr);
        let asl = b.slot(parr);
        for (j, x) in [xs[2], av[0], xs[3]].into_iter().enumerate() {
            let kj = b.int(j as i128);
            let p = b.op(Op::ElemAddr, pa, &[asl.into(), kj]);
            b.store(p, x);
        }
        let mut ar = Vec::new();
        for j in [2, 0] {
            let kj = b.int(j);
            let p = b.op(Op::ElemAddr, pa, &[asl.into(), kj]);
            ar.push(b.load(at, p));
        }
        // Option<A> both ways through the callee
        let v1 = b.int(1);
        let some = b.op(Op::EnumMake, ot, &[v1, xs[4].into()]);
        let v0 = b.int(0);
        let none = b.op(Op::EnumMake, ot, &[v0]);
        let r1 = b.call(h, Some(pair), &[tv.into(), some.into()]).unwrap();
        let r2 = b.call(h, Some(pair), &[tv2.into(), none.into()]).unwrap();
        let sel_a = b.select(at, c0, xs[0], av[1]);
        let sel_c = b.select(ct, c1, ys[2], cv[3]);
        for (k, r) in [r1, r2].into_iter().enumerate() {
            let (k0, k1) = (b.int(0), b.int(1));
            let x = b.op(Op::Get, at, &[r.into(), k0]);
            let y = b.op(Op::Get, ct, &[r.into(), k1]);
            show(sp, b, &format!("{tag} call {k}"), x, a);
            show(sp, b, &format!("{tag} call {k}"), y, c);
        }
        show(sp, b, &format!("{tag} select false"), sel_a, a);
        show(sp, b, &format!("{tag} select true"), sel_c, c);
        show(sp, b, &format!("{tag} tuple get"), g0, a);
        show(sp, b, &format!("{tag} tuple get"), g1, c);
        show_scalar(sp, b, &format!("{tag} tuple get"), g2, Lane::F64);
        show(sp, b, &format!("{tag} struct"), sx, a);
        show(sp, b, &format!("{tag} struct"), sy, c);
        for (k, &x) in ar.iter().enumerate() {
            show(sp, b, &format!("{tag} array {k}"), x, a);
        }
        for (k, &x) in av.iter().chain(&xs).enumerate() {
            show(sp, b, &format!("{tag} a{k}"), x, a);
        }
        for (k, &x) in cv.iter().chain(&ys).enumerate() {
            show(sp, b, &format!("{tag} c{k}"), x, c);
        }
    })
}

/// Scalar `f32` / `f64` arithmetic, int-to-float casts and float compares feeding branches,
/// interleaved with vector ops whose results stay live, in one block and in a loop.
fn scalar_mix(fs: Shape, ds: Shape) -> Case {
    let ops = [VecOp::Add, VecOp::Sub, VecOp::Mul, VecOp::Splat, VecOp::ReduceAdd];
    Case::new(&ops, fs.bits().max(ds.bits()), inputs_cost(5, fs.bits()) + 200, move |sp, b| {
        let t = sp.t();
        let (ft, dt) = (sp.vt(fs), sp.vt(ds));
        let tag = format!("scalar mix {} {}", fs.name(), ds.name());
        let x = opq_f64(sp, b, 1.75);
        let y = opq_f32(sp, b, -0.625);
        let n = opq_int(sp, b, t.i64, 37);
        let k = opq_int(sp, b, t.i32, -3);
        let cnt = opq_int(sp, b, t.u32, 6);
        let items = vec![
            (fs, rnd(fs, 150)),
            (fs, rnd(fs, 151)),
            (fs, rnd(fs, 152)),
            (ds, rnd(ds, 153)),
            (ds, rnd(ds, 154)),
        ];
        let vs = inputs(sp, b, &items);
        let (v0, v1, v2, w0, w1) = (vs[0], vs[1], vs[2], vs[3], vs[4]);
        let yd = b.cast(CastKind::Exact, t.f64, y);
        let s1 = b.binary(Op::Add, t.f64, x, yd);
        let a1 = vop(sp, b, VecOp::Mul, fs, &[v0, v1]);
        let s2 = b.binary(Op::Mul, t.f64, s1, x);
        let nf = b.cast(CastKind::Round, t.f64, n);
        let s2v = vsplat(sp, b, ds, s2);
        let b1 = vop(sp, b, VecOp::Add, ds, &[w0, s2v]);
        let s3 = b.binary(Op::Sub, t.f64, s2, nf);
        let f1 = b.cast(CastKind::Round, t.f32, s3);
        let f1v = vsplat(sp, b, fs, f1);
        let a2 = vop(sp, b, VecOp::Add, fs, &[a1, f1v]);
        let kf = b.cast(CastKind::Round, t.f32, k);
        let f2 = b.binary(Op::Mul, t.f32, f1, kf);
        let f2v = vsplat(sp, b, fs, f2);
        let a3 = vop(sp, b, VecOp::Sub, fs, &[a2, f2v]);
        let c = b.cmp(Op::Lt, t.bool, s3, x);
        let (then, els) = (b.create_block(), b.create_block());
        let (join, jv) = b.create_block_with(&[t.f64, ft, dt]);
        b.cbr(c, then, &[], els, &[]);
        b.switch_to(then);
        let t1 = vop(sp, b, VecOp::Add, fs, &[a3, v2]);
        let one = b.f64(1.0);
        let ts = b.binary(Op::Add, t.f64, s3, one);
        b.br(join, &[ts.into(), t1.into(), b1.into()]);
        b.switch_to(els);
        let t1 = vop(sp, b, VecOp::Mul, fs, &[a3, v2]);
        let half = b.f64(0.5);
        let ts = b.binary(Op::Mul, t.f64, s3, half);
        let b2 = vop(sp, b, VecOp::Sub, ds, &[b1, w1]);
        b.br(join, &[ts.into(), t1.into(), b2.into()]);
        b.switch_to(join);
        let (head, hv, body, exit) = counted_loop(b, t, &[t.f64, ft, dt]);
        let z = b.int(0);
        b.br(head, &[z, jv[0].into(), jv[1].into(), jv[2].into()]);
        b.switch_to(head);
        let go = b.cmp(Op::Lt, t.bool, hv[0], cnt);
        b.cbr(go, body, &[], exit, &[]);
        b.switch_to(body);
        let (s, acc, accd) = (hv[1], hv[2], hv[3]);
        let fi = b.cast(CastKind::Exact, t.f64, hv[0]);
        let half = b.f64(0.5);
        let sh_ = b.binary(Op::Mul, t.f64, s, half);
        let s2 = b.binary(Op::Add, t.f64, sh_, fi);
        let sf = b.cast(CastKind::Round, t.f32, s2);
        let sfv = vsplat(sp, b, fs, sf);
        let acc1 = vop(sp, b, VecOp::Add, fs, &[acc, sfv]);
        let red = b.op(Op::Vec(VecOp::ReduceAdd), t.f32, &[acc1.into()]);
        let redd = b.cast(CastKind::Exact, t.f64, red);
        let big = b.cmp(Op::Gt, t.bool, redd, s2);
        let (bb, sb) = (b.create_block(), b.create_block());
        let (latch, lv) = b.create_block_with(&[ft, dt]);
        b.cbr(big, bb, &[], sb, &[]);
        b.switch_to(bb);
        let acc2 = vop(sp, b, VecOp::Sub, fs, &[acc1, v1]);
        let accd2 = vop(sp, b, VecOp::Add, ds, &[accd, w0]);
        b.br(latch, &[acc2.into(), accd2.into()]);
        b.switch_to(sb);
        let acc2 = vop(sp, b, VecOp::Add, fs, &[acc1, v2]);
        let s2v = vsplat(sp, b, ds, s2);
        let accd2 = vop(sp, b, VecOp::Sub, ds, &[accd, s2v]);
        b.br(latch, &[acc2.into(), accd2.into()]);
        b.switch_to(latch);
        let i2 = next_i(b, t, hv[0]);
        b.br(head, &[i2.into(), s2.into(), lv[0].into(), lv[1].into()]);
        b.switch_to(exit);
        show_scalar(sp, b, &format!("{tag} loop s"), s, Lane::F64);
        show(sp, b, &format!("{tag} loop acc"), acc, fs);
        show(sp, b, &format!("{tag} loop acc"), accd, ds);
        show_scalar(sp, b, &format!("{tag} branch s"), jv[0], Lane::F64);
        show(sp, b, &format!("{tag} branch"), jv[1], fs);
        show(sp, b, &format!("{tag} branch"), jv[2], ds);
        show(sp, b, &format!("{tag} a1"), a1, fs);
        show(sp, b, &format!("{tag} b1"), b1, ds);
        show(sp, b, &format!("{tag} v0"), v0, fs);
        show_scalar(sp, b, &format!("{tag} f2"), f2, Lane::F32);
    })
}

/// Gather, scatter, masked load and store, compress, expand and lookup on `f32` lanes with 10
/// vectors live around them and every result printed only after the last op.
fn mem_ops(fs: Shape) -> Case {
    let is = sh(Lane::I32, fs.n);
    let us = sh(Lane::U32, fs.n);
    let ops = [
        VecOp::Gather,
        VecOp::Scatter,
        VecOp::LoadMasked,
        VecOp::StoreMasked,
        VecOp::Compress,
        VecOp::Expand,
        VecOp::Lookup,
    ];
    Case::new(&ops, fs.bits(), inputs_cost(14, fs.bits()) + 150, move |sp, b| {
        let t = sp.t();
        let n = fs.n as u64;
        let tag = format!("mem ops {}", fs.name());
        const LEN: u64 = 64;
        let arr = sp.p.m.types.array(t.f32, LEN);
        let pa = sp.p.m.types.ref_(true, arr);
        let asl = b.slot(pa);
        let addr = b.cast(CastKind::Exact, t.usize, asl);
        let nb = b.int(LEN as i128 * 4);
        sp.fill(b, addr.into(), nb, 21);
        let i = opq_int(sp, b, t.usize, 5);
        let j = opq_int(sp, b, t.usize, 2 * n as i128 + 1);
        let m = sp.mask(b, 32, fs.n, 0b1011_0110);
        let mut items: Vec<(Shape, Vec<u64>)> = (0..6).map(|k| (fs, rnd(fs, 160 + k))).collect();
        items.extend((0..4).map(|k| (is, rnd(is, 170 + k))));
        items.push((us, lanes_of(us, |j| (j * 13 + 7) % LEN)));
        items.push((us, lanes_of(us, |j| LEN - 1 - j * 3)));
        items.push((us, lanes_of(us, |j| if j == 1 { 200 } else { (j * 5 + 1) % n })));
        let vs = inputs(sp, b, &items);
        let (l, k, gidx, sidx, lidx) = (&vs[..6], &vs[6..10], vs[10], vs[11], vs[12]);
        let d0 = vop(sp, b, VecOp::Add, fs, &[l[0], l[1]]);
        let d1 = vop(sp, b, VecOp::Sub, fs, &[l[2], l[3]]);
        let cm = vcmp(sp, b, VecOp::Lt, fs, l[4], l[5]);
        let ft = sp.vt(fs);
        let g = b.op(Op::Vec(VecOp::Gather), ft, &[asl.into(), gidx.into()]);
        b.inst(Op::Vec(VecOp::Scatter), None, &[asl.into(), sidx.into(), d0.into()]);
        let ml = b.op(Op::Vec(VecOp::LoadMasked), ft, &[asl.into(), i.into(), cm.into()]);
        b.inst(Op::Vec(VecOp::StoreMasked), None, &[asl.into(), j.into(), d1.into(), m.into()]);
        let cp = vop(sp, b, VecOp::Compress, fs, &[l[0], cm]);
        let ex = vop(sp, b, VecOp::Expand, fs, &[l[1], m]);
        let lk = b.op(Op::Vec(VecOp::Lookup), ft, &[l[2].into(), lidx.into()]);
        let g2 = b.op(Op::Vec(VecOp::Gather), ft, &[asl.into(), sidx.into()]);
        show(sp, b, &format!("{tag} gather"), g, fs);
        show(sp, b, &format!("{tag} masked load"), ml, fs);
        show(sp, b, &format!("{tag} compress"), cp, fs);
        show(sp, b, &format!("{tag} expand"), ex, fs);
        show(sp, b, &format!("{tag} lookup"), lk, fs);
        show(sp, b, &format!("{tag} gather scattered"), g2, fs);
        sp.label(b, &format!("{tag} array"), None);
        let nb = b.int(LEN as i128 * 4);
        sp.dump(b, addr.into(), nb);
        show_mask(sp, b, &format!("{tag} compare"), cm, fs);
        for (q, &v) in l.iter().chain([&d0, &d1]).enumerate() {
            show(sp, b, &format!("{tag} l{q}"), v, fs);
        }
        for (q, &v) in k.iter().enumerate() {
            show(sp, b, &format!("{tag} k{q}"), v, is);
        }
    })
}

/// Int division, lane-wise shifts of 8- and 16-bit lanes, blend, fma, `cast.try`, `checked_mul`,
/// ordered reduction and scans with 10 vectors live around them.
fn arith_ops(fs: Shape) -> Case {
    let n = fs.n;
    let is = sh(Lane::I32, n);
    let bs = sh(Lane::U8, n * 4);
    let hs = sh(Lane::I16, n * 2);
    let ops = [
        VecOp::DivLanes,
        VecOp::Shl,
        VecOp::Shr,
        VecOp::Select,
        VecOp::Fma,
        VecOp::Cast(CastKind::Try),
        VecOp::CheckedMul,
        VecOp::ReduceAddOrdered,
        VecOp::ScanAdd,
        VecOp::ScanAddWrap,
    ];
    Case::new(&ops, fs.bits(), inputs_cost(22, fs.bits()) + 200, move |sp, b| {
        let t = sp.t();
        let tag = format!("arith ops {}", fs.name());
        let init = opq_f32(sp, b, 0.5);
        let mut r = Rng::new(n as u64 + 5);
        let mut items: Vec<(Shape, Vec<u64>)> = (0..5).map(|k| (fs, rnd(fs, 180 + k))).collect();
        items.extend((0..5).map(|k| (is, rnd(is, 190 + k))));
        items.push((is, lanes_of(is, |_| r.next())));
        items.push((
            is,
            lanes_of(is, |j| {
                let d = 1 + (j * 37 + 2) % 1000;
                if j % 3 == 1 { d.wrapping_neg() } else { d }
            }),
        ));
        items.push((bs, rnd(bs, 200)));
        items.push((bs, lanes_of(bs, |j| j % 8)));
        items.push((hs, rnd(hs, 201)));
        items.push((hs, lanes_of(hs, |j| (j * 5) % 16)));
        items.push((fs, lanes_of(fs, |j| fl(Lane::F32, j as f64 * 3.0 - 7.0))));
        items.push((fs, lanes_of(fs, |j| fl(Lane::F32, if j == n as u64 - 1 { 2.5 } else { j as f64 }))));
        items.push((is, lanes_of(is, |j| (j * 77).wrapping_sub(300))));
        items.push((is, lanes_of(is, |j| j * 1000 + 3)));
        items.push((is, lanes_of(is, |j| 0x10000 + j)));
        items.push((is, lanes_of(is, |j| 0x8000 + j)));
        let vs = inputs(sp, b, &items);
        let (l, k) = (&vs[..5], &vs[5..10]);
        let [dd, dv, b8, a8, h16, a16, tok, tbad, c1, c2, c3, c4] = vs[10..22].try_into().unwrap();
        let q = vop(sp, b, VecOp::DivLanes, is, &[dd, dv]);
        let sl8 = vop(sp, b, VecOp::Shl, bs, &[b8, a8]);
        let sr8 = vop(sp, b, VecOp::Shr, bs, &[b8, a8]);
        let sr16 = vop(sp, b, VecOp::Shr, hs, &[h16, a16]);
        let sl16 = vop(sp, b, VecOp::Shl, hs, &[h16, a16]);
        let m2 = vcmp(sp, b, VecOp::Lt, fs, l[0], l[1]);
        let sel = vop(sp, b, VecOp::Select, fs, &[m2, l[2], l[3]]);
        let fm = vop(sp, b, VecOp::Fma, fs, &[l[0], l[1], l[2]]);
        let it = sp.vt(is);
        let oi = sp.p.option(it);
        let try_ok = b.op(Op::Vec(VecOp::Cast(CastKind::Try)), oi, &[tok.into()]);
        let try_bad = b.op(Op::Vec(VecOp::Cast(CastKind::Try)), oi, &[tbad.into()]);
        let cm_ok = b.op(Op::Vec(VecOp::CheckedMul), oi, &[c1.into(), c2.into()]);
        let cm_bad = b.op(Op::Vec(VecOp::CheckedMul), oi, &[c3.into(), c4.into()]);
        let ro = b.op(Op::Vec(VecOp::ReduceAddOrdered), t.f32, &[l[3].into(), init.into()]);
        let sc = vop(sp, b, VecOp::ScanAdd, fs, &[l[4]]);
        let sw = vop(sp, b, VecOp::ScanAddWrap, is, &[k[0]]);
        show(sp, b, &format!("{tag} div"), q, is);
        show(sp, b, &format!("{tag} shl"), sl8, bs);
        show(sp, b, &format!("{tag} shr"), sr8, bs);
        show(sp, b, &format!("{tag} shr"), sr16, hs);
        show(sp, b, &format!("{tag} shl"), sl16, hs);
        show(sp, b, &format!("{tag} select"), sel, fs);
        show(sp, b, &format!("{tag} fma"), fm, fs);
        show_opt(sp, b, &format!("{tag} cast.try ok"), try_ok, is);
        show_opt(sp, b, &format!("{tag} cast.try inexact"), try_bad, is);
        show_opt(sp, b, &format!("{tag} checked_mul ok"), cm_ok, is);
        show_opt(sp, b, &format!("{tag} checked_mul overflow"), cm_bad, is);
        show_scalar(sp, b, &format!("{tag} reduce_add_ordered"), ro, Lane::F32);
        show(sp, b, &format!("{tag} scan_add"), sc, fs);
        show(sp, b, &format!("{tag} scan_add_wrap"), sw, is);
        show_mask(sp, b, &format!("{tag} compare"), m2, fs);
        for (q, &v) in l.iter().enumerate() {
            show(sp, b, &format!("{tag} l{q}"), v, fs);
        }
        for (q, &v) in k.iter().enumerate() {
            show(sp, b, &format!("{tag} k{q}"), v, is);
        }
    })
}

/// The same value as several operands of one op (`v + v`, `fma(v, v, v)`, `select(m, v, v)`,
/// `v ^ v`, compares of `v` with itself, `m & m`), with 8 more vectors live.
fn same_operand(fs: Shape) -> Case {
    let is = sh(Lane::I32, fs.n);
    let ops = [
        VecOp::Add,
        VecOp::Sub,
        VecOp::Mul,
        VecOp::Fma,
        VecOp::Min,
        VecOp::Select,
        VecOp::Eq,
        VecOp::Lt,
        VecOp::Xor,
        VecOp::Andnot,
        VecOp::MulWrap,
        VecOp::SubWrap,
        VecOp::MaskAnd,
        VecOp::MaskXor,
    ];
    Case::new(&ops, fs.bits(), inputs_cost(10, fs.bits()) + 150, move |sp, b| {
        let tag = format!("same operand {}", fs.name());
        let mt = sp.mt(32, fs.n);
        let m = sp.mask(b, 32, fs.n, 0b1001_1100);
        let nn = nans(Lane::F32);
        let special = lanes_of(fs, |j| match j {
            0 => nn[0],
            1 => fl(Lane::F32, -0.0),
            2 => nn[3],
            _ => fl(Lane::F32, j as f64 * 1.5 - 4.0),
        });
        let mut items = vec![(fs, special), (is, rnd(is, 210))];
        items.extend((0..4).map(|k| (fs, rnd(fs, 211 + k))));
        items.extend((0..4).map(|k| (is, rnd(is, 215 + k))));
        let vs = inputs(sp, b, &items);
        let (v, u, rest) = (vs[0], vs[1], &vs[2..]);
        let fr: Vec<(&str, Value)> = vec![
            ("v + v", vop(sp, b, VecOp::Add, fs, &[v, v])),
            ("v - v", vop(sp, b, VecOp::Sub, fs, &[v, v])),
            ("v * v", vop(sp, b, VecOp::Mul, fs, &[v, v])),
            ("fma(v, v, v)", vop(sp, b, VecOp::Fma, fs, &[v, v, v])),
            ("min(v, v)", vop(sp, b, VecOp::Min, fs, &[v, v])),
            ("select(m, v, v)", vop(sp, b, VecOp::Select, fs, &[m, v, v])),
        ];
        let eq = vcmp(sp, b, VecOp::Eq, fs, v, v);
        let lt = vcmp(sp, b, VecOp::Lt, fs, v, v);
        let ir: Vec<(&str, Value)> = vec![
            ("u ^ u", vop(sp, b, VecOp::Xor, is, &[u, u])),
            ("andnot(u, u)", vop(sp, b, VecOp::Andnot, is, &[u, u])),
            ("u * u", vop(sp, b, VecOp::MulWrap, is, &[u, u])),
            ("u - u", vop(sp, b, VecOp::SubWrap, is, &[u, u])),
            ("select(m, u, u)", vop(sp, b, VecOp::Select, is, &[m, u, u])),
        ];
        let ueq = vcmp(sp, b, VecOp::Eq, is, u, u);
        let mand = b.op(Op::Vec(VecOp::MaskAnd), mt, &[m.into(), m.into()]);
        let mxor = b.op(Op::Vec(VecOp::MaskXor), mt, &[eq.into(), eq.into()]);
        for (what, x) in fr {
            show(sp, b, &format!("{tag} {what}"), x, fs);
        }
        for (what, x) in ir {
            show(sp, b, &format!("{tag} {what}"), x, is);
        }
        for (what, x) in [("v == v", eq), ("v < v", lt), ("u == u", ueq), ("m & m", mand), ("e ^ e", mxor)] {
            show_mask(sp, b, &format!("{tag} {what}"), x, fs);
        }
        for (q, &x) in rest.iter().enumerate() {
            let s = if q < 4 { fs } else { is };
            show(sp, b, &format!("{tag} live{q}"), x, s);
        }
    })
}

/// A loop whose body reads block parameter `x` in several ops and also passes it as the next
/// iteration's argument for another parameter.
fn param_reuse(s: Shape) -> Case {
    let ops = [
        if s.lane.is_float() { VecOp::Fma } else { VecOp::MulWrap },
        VecOp::Lt,
        VecOp::MaskXor,
        VecOp::Select,
    ];
    Case::new(&ops, s.bits(), inputs_cost(3, s.bits()) + 80, move |sp, b| {
        let t = sp.t();
        let vt = sp.vt(s);
        let (w, n) = (s.lane.bits(), s.n);
        let mt = sp.mt(w, n);
        let tag = format!("param reuse {}", s.name());
        let cnt = opq_int(sp, b, t.u32, 5);
        let m0 = sp.mask(b, w, n, 0b0101);
        let mut items = vec![(s, rnd(s, 220)), (s, rnd(s, 221))];
        items.push((
            s,
            if s.lane.is_float() { lanes_of(s, |j| fl(s.lane, 0.5 + j as f64 * 0.25)) } else { rnd(s, 222) },
        ));
        let vs = inputs(sp, b, &items);
        let (head, hv, body, exit) = counted_loop(b, t, &[vt, vt, mt]);
        let z = b.int(0);
        b.br(head, &[z, vs[0].into(), vs[1].into(), m0.into()]);
        b.switch_to(head);
        let go = b.cmp(Op::Lt, t.bool, hv[0], cnt);
        b.cbr(go, body, &[], exit, &[]);
        b.switch_to(body);
        let (x, y, m) = (hv[1], hv[2], hv[3]);
        let z = if s.lane.is_float() {
            vop(sp, b, VecOp::Fma, s, &[x, vs[2], y])
        } else {
            let p = vop(sp, b, VecOp::MulWrap, s, &[x, vs[2]]);
            vop(sp, b, VecOp::AddWrap, s, &[p, y])
        };
        let lt = vcmp(sp, b, VecOp::Lt, s, z, x);
        let m2 = b.op(Op::Vec(VecOp::MaskXor), mt, &[m.into(), lt.into()]);
        let wv = vop(sp, b, VecOp::Select, s, &[m2, z, x]);
        let i2 = next_i(b, t, hv[0]);
        b.br(head, &[i2.into(), wv.into(), x.into(), m2.into()]);
        b.switch_to(exit);
        show(sp, b, &format!("{tag} x"), x, s);
        show(sp, b, &format!("{tag} y"), y, s);
        show_mask(sp, b, &format!("{tag} m"), m, s);
        show(sp, b, &format!("{tag} k"), vs[2], s);
    })
}

/// Ops whose second operand may be a vector load in memory.
fn fold_ops(s: Shape) -> Vec<VecOp> {
    use VecOp::*;
    if s.lane.is_float() {
        vec![Add, Sub, Mul, Div, MinX86, MaxX86]
    } else {
        vec![AddWrap, SubWrap, And, Or, Xor]
    }
}

/// A `&mut [L]` of `lanes` in a slot, its address and length run-time values.
fn fold_slice(sp: &mut Sp, b: &mut FuncBuilder, l: Lane, lanes: &[u64]) -> Value {
    let t = sp.t();
    let len = lanes.len() as u64;
    let e = sp.lane_t(l);
    let arr = sp.p.m.types.array(e, len);
    let pa = sp.p.m.types.ref_(true, arr);
    let sl = b.slot(pa);
    let words = Sp::words(sh(l, len as u32), lanes);
    let warr = sp.p.m.types.array(t.u64, words.len() as u64);
    let pw = sp.p.m.types.ref_(true, warr);
    let pe = sp.p.m.types.ref_(true, t.u64);
    let raw = b.cast(CastKind::Exact, pw, sl);
    for (j, w) in words.into_iter().enumerate() {
        let kj = b.int(j as i128);
        let a = b.op(Op::ElemAddr, pe, &[raw.into(), kj]);
        let c = b.int(w as i128);
        b.store(a, c);
    }
    let addr = b.cast(CastKind::Exact, t.usize, sl);
    let addr = opq(sp, b, addr.into(), t.usize);
    let pl = sp.p.m.types.ptr(true, e);
    let p = b.cast(CastKind::Exact, pl, addr);
    let st = sp.p.m.types.intern(TypeKey::Slice { elem: e, align: None });
    let rst = sp.p.m.types.ref_(true, st);
    let n = opq_int(sp, b, t.usize, len as i128);
    b.op(Op::SliceMake, rst, &[p.into(), n.into()])
}

/// `check.bounds i (sub.sat (slice.len s) n-1)`, then `vec.load s i`: the load is the last
/// instruction emitted.
fn guarded_load(sp: &mut Sp, b: &mut FuncBuilder, s: Shape, sl: Value, i: Operand) -> Value {
    let t = sp.t();
    let len = b.op(Op::SliceLen, t.usize, &[sl.into()]);
    let k = b.int(s.n as i128 - 1);
    let lim = b.binary(Op::SubSat, t.usize, len, k);
    b.bounds_check(i, lim);
    let vt = sp.vt(s);
    b.op(Op::Vec(VecOp::Load), vt, &[sl.into(), i])
}

/// Slice lanes for [`fold_loads`]: floats mix NaNs, signed zeros and infinities with ordinary
/// values, so the operand order of `sub`, `div`, `min_x86` and `max_x86` shows.
fn fold_lanes(l: Lane, len: u64, seed: u64) -> Vec<u64> {
    let mut r = Rng::new(seed);
    if !l.is_float() {
        return (0..len).map(|_| r.next() & l.ones()).collect();
    }
    let nn = nans(l);
    let special =
        [nn[0], fl(l, -0.0), fl(l, 0.0), fl(l, f64::INFINITY), fl(l, f64::NEG_INFINITY), nn[3], fl(l, 1.5)];
    (0..len)
        .map(|j| {
            if j % 3 == seed % 3 {
                special[((j / 3) + seed) as usize % special.len()]
            } else {
                fl(l, (r.below(129) as f64 - 64.0) / 8.0)
            }
        })
        .collect()
}

/// `op(acc, vec.load s i)` with the load right before the op (constant and run-time index, in a
/// loop, the dot-product shape `acc + load a i * load b i`), and the shapes that must not fold:
/// the load as the first operand, used twice, as both operands, or with an instruction between;
/// 10 more vectors live throughout.
fn fold_loads(s: Shape) -> Case {
    let mut ops = fold_ops(s);
    ops.push(VecOp::Load);
    let len = 4 * s.n as u64 + 8;
    let cost = inputs_cost(11, s.bits()) + 2 * len as usize + fold_ops(s).len() * 110 + 120;
    Case::new(&ops, s.bits(), cost, move |sp, b| {
        let t = sp.t();
        let vt = sp.vt(s);
        let tag = format!("fold loads {}", s.name());
        let n = s.n as u64;
        let sa = fold_slice(sp, b, s.lane, &fold_lanes(s.lane, len, 1));
        let sb = fold_slice(sp, b, s.lane, &fold_lanes(s.lane, len, 2));
        let ir = opq_int(sp, b, t.usize, 1);
        let il = opq_int(sp, b, t.usize, (len - n) as i128);
        let cnt = opq_int(sp, b, t.usize, (len - n + 1) as i128);
        let items: Vec<(Shape, Vec<u64>)> = (0..11).map(|k| (s, rnd(s, 230 + k))).collect();
        let live = inputs(sp, b, &items);
        let mut acc = live[10];
        let mut between = Vec::new();
        for (q, op) in fold_ops(s).into_iter().enumerate() {
            let name = op.name();
            let k = b.int(2);
            let l = guarded_load(sp, b, s, sa, k);
            acc = vop(sp, b, op, s, &[acc, l]);
            show(sp, b, &format!("{tag} {name}(acc, load a 2)"), acc, s);
            let l = guarded_load(sp, b, s, sa, ir.into());
            acc = vop(sp, b, op, s, &[acc, l]);
            show(sp, b, &format!("{tag} {name}(acc, load a i)"), acc, s);
            let l = guarded_load(sp, b, s, sb, il.into());
            acc = vop(sp, b, op, s, &[l, acc]);
            show(sp, b, &format!("{tag} {name}(load b last, acc)"), acc, s);
            let l = guarded_load(sp, b, s, sa, ir.into());
            let t1 = vop(sp, b, op, s, &[acc, l]);
            acc = vop(sp, b, op, s, &[l, t1]);
            show(sp, b, &format!("{tag} {name}(load, {name}(acc, load))"), acc, s);
            let l = guarded_load(sp, b, s, sb, ir.into());
            let ll = vop(sp, b, op, s, &[l, l]);
            show(sp, b, &format!("{tag} {name}(load b i, same)"), ll, s);
            let l = guarded_load(sp, b, s, sa, il.into());
            let u = vop(sp, b, add(s), s, &[live[q], live[q + 1]]);
            acc = vop(sp, b, op, s, &[acc, l]);
            between.push(u);
            show(sp, b, &format!("{tag} {name}(acc, load a last) after another op"), acc, s);
        }
        let dot = if s.lane.is_float() { VecOp::Mul } else { VecOp::And };
        for (k, i) in [Operand::from(ir), Operand::from(il)].into_iter().enumerate() {
            let la = guarded_load(sp, b, s, sa, i);
            let lb = guarded_load(sp, b, s, sb, i);
            let p = vop(sp, b, dot, s, &[la, lb]);
            acc = vop(sp, b, add(s), s, &[acc, p]);
            show(sp, b, &format!("{tag} dot {k}"), acc, s);
        }
        let (op1, op2) = (fold_ops(s)[0], fold_ops(s)[1]);
        let (head, hv) = b.create_block_with(&[t.usize, vt, vt]);
        let body = b.create_block();
        let exit = b.create_block();
        let z = b.int(0);
        b.br(head, &[z, acc.into(), live[9].into()]);
        b.switch_to(head);
        let go = b.cmp(Op::Lt, t.bool, hv[0], cnt);
        b.cbr(go, body, &[], exit, &[]);
        b.switch_to(body);
        let l1 = guarded_load(sp, b, s, sa, hv[0].into());
        let a1 = vop(sp, b, op1, s, &[hv[1], l1]);
        let l2 = guarded_load(sp, b, s, sb, hv[0].into());
        let a2 = vop(sp, b, op2, s, &[hv[2], l2]);
        let one = b.int(1);
        let j2 = b.binary(Op::AddWrap, t.usize, hv[0], one);
        b.br(head, &[j2.into(), a1.into(), a2.into()]);
        b.switch_to(exit);
        show(sp, b, &format!("{tag} loop {}", op1.name()), hv[1], s);
        show(sp, b, &format!("{tag} loop {}", op2.name()), hv[2], s);
        for (k, &v) in live.iter().chain(&between).enumerate() {
            show(sp, b, &format!("{tag} live{k}"), v, s);
        }
    })
}

/// How a bounds check against `sub.sat n K` is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// `check.bounds i (sub.sat n K)` adjacent, `K` constant, the limit used once.
    Fused,
    /// The limit also printed after the check.
    TwoUses,
    /// Another instruction between the two.
    Between,
    /// `K` a run-time value.
    RuntimeK,
    /// `check.bounds (sub.sat n K) i`.
    Swapped,
    /// `u32` operands.
    U32,
}

#[derive(Clone, Copy, Debug)]
enum Len {
    Value(u64),
    /// `slice.len` of a slice of that length.
    Slice(u64),
}

#[derive(Clone, Copy, Debug)]
enum Idx {
    Value(u64),
    Const(u64),
}

#[derive(Clone, Copy, Debug)]
struct Check {
    form: Form,
    k: u64,
    n: Len,
    i: Idx,
}

impl Check {
    fn n(self) -> u64 {
        match self.n {
            Len::Value(x) | Len::Slice(x) => x,
        }
    }

    fn i(self) -> u64 {
        match self.i {
            Idx::Value(x) | Idx::Const(x) => x,
        }
    }

    fn traps(self) -> bool {
        let lim = self.n().saturating_sub(self.k);
        match self.form {
            Form::Swapped => lim >= self.i(),
            Form::U32 => self.i() as u32 >= (self.n() as u32).saturating_sub(self.k as u32),
            _ => self.i() >= lim,
        }
    }
}

/// A label, the check, then `passed` (and the values a near-miss form also uses).
fn emit_check(sp: &mut Sp, b: &mut FuncBuilder, c: Check) {
    let t = sp.t();
    sp.label(b, &format!("check.bounds {:?} K {} n {:?} i {:?}", c.form, c.k, c.n, c.i), None);
    let ty = if c.form == Form::U32 { t.u32 } else { t.usize };
    let n: Value = match c.n {
        Len::Value(x) => opq_int(sp, b, ty, x as i128),
        Len::Slice(x) => {
            let pb = sp.p.m.types.ptr(true, t.u8);
            let arr = sp.p.m.types.array(t.u8, 64);
            let pa = sp.p.m.types.ref_(true, arr);
            let sl = b.slot(pa);
            let p = b.cast(CastKind::Exact, pb, sl);
            let st = sp.p.m.types.intern(TypeKey::Slice { elem: t.u8, align: None });
            let rst = sp.p.m.types.ref_(true, st);
            let len = opq_int(sp, b, t.usize, x as i128);
            let s = b.op(Op::SliceMake, rst, &[p.into(), len.into()]);
            b.op(Op::SliceLen, t.usize, &[s.into()])
        }
    };
    let i: Operand = match c.i {
        Idx::Value(x) => opq_int(sp, b, ty, x as i128).into(),
        Idx::Const(x) => b.int(x as i128),
    };
    let k: Operand =
        if c.form == Form::RuntimeK { opq_int(sp, b, ty, c.k as i128).into() } else { b.int(c.k as i128) };
    let lim = b.binary(Op::SubSat, ty, n, k);
    let mut extra = None;
    match c.form {
        Form::Swapped => b.bounds_check(lim, i),
        Form::Between => {
            let one = b.int(1);
            extra = Some(b.binary(Op::AddWrap, ty, n, one));
            b.bounds_check(i, lim);
        }
        _ => b.bounds_check(i, lim),
    }
    sp.label(b, "passed", None);
    let l = if c.form == Form::U32 { Lane::U32 } else { Lane::U64 };
    if c.form == Form::TwoUses {
        sp.out_scalar(b, lim, l);
    }
    if let Some(x) = extra {
        sp.out_scalar(b, x, l);
    }
}

/// `check.bounds i (sub.sat n K)` in every [`Form`]: lengths below, at and above `K`, the last
/// passing index, the first failing one, indices past `n` and near `u64::MAX` (where `i + K`
/// wraps), constant indices, `slice.len` lengths.
fn fold_bounds() -> Vec<Check> {
    let mut out = Vec::new();
    let max = u64::MAX;
    for k in [0, 1, 7, 31] {
        let f = |n: Len, i: Idx| Check { form: Form::Fused, k, n, i };
        let n = k + 10;
        out.extend([
            f(Len::Value(n), Idx::Value(n - k - 1)),
            f(Len::Value(k + 1), Idx::Const(0)),
            f(Len::Value(n), Idx::Value(0)),
            f(Len::Value(n), Idx::Const(9)),
            f(Len::Slice(k + 5), Idx::Value(4)),
            f(Len::Value(k), Idx::Value(0)),
            f(Len::Value(n), Idx::Value(n - k)),
            f(Len::Value(n), Idx::Value(n + 5)),
            f(Len::Value(n), Idx::Value((max - k).wrapping_add(1))),
            f(Len::Value(n), Idx::Value((max - k).wrapping_add(3))),
            f(Len::Value(n), Idx::Value(max - 1)),
            f(Len::Value(n), Idx::Const(max)),
            f(Len::Value(n), Idx::Const(1 << 63)),
            f(Len::Slice(k + 5), Idx::Value(5)),
        ]);
        if k > 0 {
            out.push(f(Len::Value(k - 1), Idx::Value(0)));
        }
    }
    for form in [Form::TwoUses, Form::Between, Form::RuntimeK, Form::Swapped, Form::U32] {
        for k in [1, 7] {
            let n = k + 10;
            let top = if form == Form::U32 { u32::MAX as u64 } else { max };
            for i in [n - k - 1, n - k, n - k + 1, n + 3, top] {
                out.push(Check { form, k, n: Len::Value(n), i: Idx::Value(i) });
            }
            out.push(Check { form, k, n: Len::Value(n), i: Idx::Const(n - k - 1) });
            out.push(Check { form, k, n: Len::Value(k - 1), i: Idx::Value(0) });
        }
    }
    out
}

/// The non-trapping [`fold_bounds`] checks in one case, each trapping one in a case of its own.
fn bounds_cases() -> Vec<Case> {
    let all = fold_bounds();
    let pass: Vec<Check> = all.iter().copied().filter(|c| !c.traps()).collect();
    let mut out = vec![Case::new(&[], 0, pass.len() * 14, move |sp, b| {
        for &c in &pass {
            emit_check(sp, b, c);
        }
    })];
    for c in all.into_iter().filter(|c| c.traps()) {
        out.push(Case::new(&[], 0, 20, move |sp, b| emit_check(sp, b, c)).traps(PanicKind::Bounds));
    }
    out
}

pub fn registers() -> Vec<Case> {
    let f = |l: Lane, n: u32| sh(l, n);
    let narrow = [f(Lane::F32, 4), f(Lane::F64, 2), f(Lane::I32, 4), f(Lane::U8, 16)];
    let wide = [f(Lane::F32, 8), f(Lane::F64, 4), f(Lane::I64, 4), f(Lane::I16, 16)];
    let mut out = Vec::new();
    for s in narrow.into_iter().chain(wide) {
        out.push(pressure(s));
        out.push(loop_pressure(s));
        out.push(param_reuse(s));
    }
    out.push(pressure_mixed());
    for s in [f(Lane::F32, 4), f(Lane::I32, 4), f(Lane::F64, 4), f(Lane::I16, 16)] {
        for perm in [[1, 2, 0, 3], [1, 0, 3, 2], [3, 0, 1, 2], [0, 1, 2, 3]] {
            out.push(rotate(s, perm));
        }
    }
    out.push(mixed_loop(f(Lane::F32, 4), f(Lane::I16, 8)));
    out.push(mixed_loop(f(Lane::F32, 8), f(Lane::I32, 4)));
    for ret in [Ret::Unit, Ret::F64, Ret::Narrow] {
        out.push(calls(false, ret, ret == Ret::F64));
    }
    for ret in [Ret::Unit, Ret::F64, Ret::Narrow, Ret::Wide] {
        out.push(calls(true, ret, matches!(ret, Ret::Unit | Ret::Wide)));
    }
    let v = |l: Lane, n: u32| A::V(sh(l, n));
    out.push(many_args(
        "v2",
        f(Lane::F32, 4),
        vec![
            A::F64,
            v(Lane::F32, 4),
            A::F64,
            v(Lane::I32, 4),
            A::I64,
            v(Lane::F32, 4),
            A::F64,
            A::M(4),
            A::F64,
            v(Lane::F64, 2),
            v(Lane::F32, 4),
            A::I64,
            A::F64,
            v(Lane::I32, 4),
            v(Lane::F32, 2),
            A::M(4),
            A::F64,
            v(Lane::F64, 2),
            v(Lane::F32, 2),
        ],
    ));
    out.push(many_args(
        "v3",
        f(Lane::F32, 8),
        vec![
            A::F64,
            v(Lane::F32, 4),
            A::F64,
            v(Lane::F32, 8),
            A::I64,
            v(Lane::F32, 4),
            A::F64,
            A::M(8),
            A::F64,
            v(Lane::F32, 8),
            v(Lane::F32, 4),
            A::I64,
            A::F64,
            v(Lane::F32, 8),
            v(Lane::F32, 4),
            A::M(8),
            A::F64,
            v(Lane::I32, 8),
            v(Lane::F64, 4),
            v(Lane::I32, 8),
        ],
    ));
    for (fs, is) in [
        (f(Lane::F64, 2), f(Lane::I16, 8)),
        (f(Lane::F32, 8), f(Lane::I32, 4)),
        (f(Lane::F64, 4), f(Lane::U8, 32)),
    ] {
        out.push(trap_edges(fs, is, None));
        for k in 0..CHECKS.len() {
            out.push(trap_edges(fs, is, Some(k)));
        }
    }
    out.push(aggregates(f(Lane::F32, 4), f(Lane::I16, 8)));
    out.push(aggregates(f(Lane::F32, 8), f(Lane::I32, 4)));
    out.push(aggregates(f(Lane::I64, 4), f(Lane::F64, 2)));
    out.push(scalar_mix(f(Lane::F32, 4), f(Lane::F64, 2)));
    out.push(scalar_mix(f(Lane::F32, 8), f(Lane::F64, 4)));
    for fs in [f(Lane::F32, 4), f(Lane::F32, 8)] {
        out.push(mem_ops(fs));
        out.push(arith_ops(fs));
        out.push(same_operand(fs));
    }
    for s in narrow.into_iter().chain(wide) {
        out.push(fold_loads(s));
    }
    out.extend(bounds_cases());
    out
}
