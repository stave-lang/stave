//! Vector and mask ops, native (qemu) vs the interpreter, on edge-biased lanes: bit for bit,
//! except the NaN results of arithmetic (quiet; sign and payload unspecified).

mod common;

use common::vgen::*;
use stave_ir::simd::VecOp;

/// Whether a lane op takes float lanes, int lanes.
fn lane_op(op: VecOp) -> Option<(bool, bool, usize)> {
    use VecOp::*;
    Some(match op {
        Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc => (true, false, 1),
        Add | Sub | Mul | Div | Copysign | MinNum | MaxNum | MinX86 | MaxX86 => (true, false, 2),
        Min | Max => (true, true, 2),
        Fma => (true, false, 3),
        AbsWrap | Popcount | Clz | Not => (false, true, 1),
        AddWrap | SubWrap | MulWrap | AddSat | SubSat | MulSat | And | Or | Xor | Andnot => (false, true, 2),
        _ => return None,
    })
}

fn canon(op: VecOp) -> Canon {
    match op {
        VecOp::Neg | VecOp::Abs | VecOp::Copysign => Canon::Exact,
        _ => Canon::Quiet,
    }
}

fn lane_ops(name: &str, types: Vec<VT>, float: bool) {
    let mut cx = Cx::new(0x51_3d);
    for op in VecOp::ALL {
        let Some((f, i, arity)) = lane_op(op) else { continue };
        if !(if float { f } else { i }) {
            continue;
        }
        for &vt in &types {
            if op == VecOp::AbsWrap && !vt.l.signed() {
                continue;
            }
            let v = cx.vty(vt);
            let func = cx.wrap_all(op, &vec![v; arity], Some(v));
            let out = O::V(vt, canon(op));
            let what = format!("{} {}", op.name(), vt.name());
            let mut inputs: Vec<Vec<Vec<u64>>> = Vec::new();
            match arity {
                1 => inputs.extend(pack(&edges(vt.l), vt.n).into_iter().map(|a| vec![a])),
                2 => inputs.extend(pairs(&crit(vt.l), vt.n).into_iter().map(|(a, c)| vec![a, c])),
                _ => {}
            }
            for _ in 0..10 {
                inputs.push((0..arity).map(|_| cx.rng.lanes(vt)).collect());
            }
            for args in inputs {
                let args = args.into_iter().map(|a| A::V(vt, a)).collect();
                cx.case(&what, func, Some(v), args, out);
            }
        }
    }
    cx.run(name);
}

#[test]
fn float_lane_ops() {
    lane_ops("simd-float", float_vts(), true);
}

#[test]
fn int_lane_ops() {
    lane_ops("simd-int", int_vts(), false);
}

/// Systematic inputs of `arity` vectors (every edge value; every pairing of the critical
/// values), then `random` random ones.
fn inputs(cx: &mut Cx, vt: VT, arity: usize, random: usize) -> Vec<Vec<Vec<u64>>> {
    let mut v: Vec<Vec<Vec<u64>>> = match arity {
        1 => pack(&edges(vt.l), vt.n).into_iter().map(|a| vec![a]).collect(),
        2 => pairs(&crit(vt.l), vt.n).into_iter().map(|(a, c)| vec![a, c]).collect(),
        _ => Vec::new(),
    };
    for _ in 0..random {
        v.push((0..arity).map(|_| cx.rng.lanes(vt)).collect());
    }
    v
}

#[test]
fn compares_reductions_scans() {
    use VecOp::*;
    let mut cx = Cx::new(0xc0_3f);
    for vt in all_vts() {
        let v = cx.vty(vt);
        let m = cx.mask_of(vt);
        let lane = cx.lane(vt.l);
        let float = vt.l.is_float();
        for op in [Eq, Ne, Lt, Le, Gt, Ge] {
            let f = cx.wrap_all(op, &[v, v], Some(m));
            for a in inputs(&mut cx, vt, 2, 8) {
                let args = a.into_iter().map(|x| A::V(vt, x)).collect();
                cx.case(&format!("{} {}", op.name(), vt.name()), f, Some(m), args, O::M(vt.l.bits(), vt.n));
            }
        }
        let reductions: &[VecOp] = if float {
            &[ReduceAdd, ReduceMin, ReduceMax]
        } else {
            &[ReduceAddWrap, ReduceMin, ReduceMax, ReduceAnd, ReduceOr]
        };
        for &op in reductions {
            let f = cx.wrap_all(op, &[v], Some(lane));
            for a in inputs(&mut cx, vt, 1, 16) {
                let args = a.into_iter().map(|x| A::V(vt, x)).collect();
                cx.case(&format!("{} {}", op.name(), vt.name()), f, Some(lane), args, O::S(lane));
            }
        }
        let scan = if float { ScanAdd } else { ScanAddWrap };
        let f = cx.wrap_all(scan, &[v], Some(v));
        for a in inputs(&mut cx, vt, 1, 16) {
            let args = a.into_iter().map(|x| A::V(vt, x)).collect();
            cx.case(&format!("{} {}", scan.name(), vt.name()), f, Some(v), args, O::V(vt, Canon::Quiet));
        }
        if float {
            let f = cx.wrap_all(ReduceAddOrdered, &[v, lane], Some(lane));
            for a in inputs(&mut cx, vt, 1, 16) {
                let init = cx.rng.lane(vt.l);
                let args = vec![A::V(vt, a[0].clone()), A::F(vt.l, init)];
                cx.case(&format!("reduce_add_ordered {}", vt.name()), f, Some(lane), args, O::S(lane));
            }
        }
    }
    cx.run("simd-cmp-reduce");
}

fn vargs(vt: VT, a: Vec<Vec<u64>>) -> Vec<A> {
    a.into_iter().map(|x| A::V(vt, x)).collect()
}

/// Shifts and divisions on inputs that do not trap.
#[test]
fn shifts_and_divisions() {
    use VecOp::*;
    let mut cx = Cx::new(0x5d_17);
    let t = cx.p.t;
    for vt in int_vts() {
        let v = cx.vty(vt);
        let it = vt.l.int().unwrap();
        let bits = u64::from(it.bits);
        for op in [Shl, Shr, ShlWrap] {
            let what = format!("{} {}", op.name(), vt.name());
            // per-lane amounts
            let f = cx.wrap_all(op, &[v, v], Some(v));
            for k in 0..12 {
                let a = cx.rng.lanes(vt);
                let amount: Vec<u64> = (0..vt.n)
                    .map(|j| match (op, (k + j) % 4) {
                        (ShlWrap, 3) => cx.rng.lane(vt.l),
                        (_, 0) => 0,
                        (_, 1) => bits - 1,
                        _ => cx.rng.below(bits),
                    })
                    .collect();
                cx.case(&what, f, Some(v), vargs(vt, vec![a, amount]), O::V(vt, Canon::Exact));
            }
            // one amount: a register of two int types, a constant
            let mut amounts: Vec<i128> = vec![0, 1, i128::from(bits) / 2, i128::from(bits) - 1];
            if op == ShlWrap {
                amounts.extend([i128::from(bits), i128::from(bits) + 3, 255, -1]);
            }
            for (ty, signed) in [(t.u32, false), (t.i64, true)] {
                let f = cx.wrap_all(op, &[v, ty], Some(v));
                for &n in amounts.iter().filter(|n| signed || **n >= 0) {
                    let a = cx.rng.lanes(vt);
                    cx.case(&what, f, Some(v), vec![A::V(vt, a), A::I(n)], O::V(vt, Canon::Exact));
                }
            }
            for &n in amounts.iter().filter(|n| **n >= 0) {
                let f = cx.wrap(op, &[v], Some(v), &[P::Arg(0), P::K(n)]);
                let a = cx.rng.lanes(vt);
                cx.case(&what, f, Some(v), vec![A::V(vt, a)], O::V(vt, Canon::Exact));
            }
        }
        // divisions: no zero divisor, no MIN / -1
        let safe = |a: u64, d: u64| d != 0 && !(it.signed && it.value(a) == it.min() && it.value(d) == -1);
        let f = cx.wrap_all(DivLanes, &[v, v], Some(v));
        for mut a in inputs(&mut cx, vt, 2, 8) {
            for k in 0..vt.n as usize {
                if !safe(a[0][k], a[1][k]) {
                    a[1][k] = 1;
                }
            }
            cx.case(&format!("div_lanes {}", vt.name()), f, Some(v), vargs(vt, a), O::V(vt, Canon::Exact));
        }
        let mut divisors = vec![1, 3, 7, it.max(), it.max() - 1];
        if it.signed {
            divisors.extend([-1, -3, it.min()]);
        }
        for d in divisors {
            let f = cx.wrap(DivConst, &[v], Some(v), &[P::Arg(0), P::K(d)]);
            for a in inputs(&mut cx, vt, 1, 3) {
                let a: Vec<u64> =
                    a[0].iter().map(|&x| if safe(x, it.wrap(d)) { x } else { it.wrap(1) }).collect();
                let what = format!("div_const {} by {d}", vt.name());
                cx.case(&what, f, Some(v), vec![A::V(vt, a)], O::V(vt, Canon::Exact));
            }
        }
    }
    cx.run("simd-shift-div");
}

#[test]
fn mask_ops() {
    use VecOp::*;
    let mut cx = Cx::new(0x3a_5c);
    let t = cx.p.t;
    for (w, n) in mask_shapes() {
        let m = cx.mty(w, n);
        let name = format!("mask{w}x{n}");
        let out = O::M(w, n);
        for op in [MaskAnd, MaskOr, MaskXor] {
            let f = cx.wrap_all(op, &[m, m], Some(m));
            for _ in 0..10 {
                let args = vec![A::M(w, cx.rng.mask(w, n)), A::M(w, cx.rng.mask(w, n))];
                cx.case(&format!("{} {name}", op.name()), f, Some(m), args, out);
            }
        }
        let f = cx.wrap_all(MaskNot, &[m], Some(m));
        let sel = cx.wrap_all(Select, &[m, m, m], Some(m));
        let bits_ty = if n > 8 { t.u16 } else { t.u8 };
        let to_bits = cx.wrap_all(ToBits, &[m], Some(bits_ty));
        for _ in 0..12 {
            let x = cx.rng.mask(w, n);
            cx.case(&format!("mask.not {name}"), f, Some(m), vec![A::M(w, x.clone())], out);
            cx.case(
                &format!("mask.to_bits {name}"),
                to_bits,
                Some(bits_ty),
                vec![A::M(w, x.clone())],
                O::S(bits_ty),
            );
            let args = vec![A::M(w, x), A::M(w, cx.rng.mask(w, n)), A::M(w, cx.rng.mask(w, n))];
            cx.case(&format!("mask.select {name}"), sel, Some(m), args, out);
        }
        for signed in [false, true] {
            let vt = VT { l: L::I(w as u8, signed), n };
            let v = cx.vty(vt);
            let f = cx.wrap_all(ToVec, &[m], Some(v));
            for _ in 0..6 {
                let x = cx.rng.mask(w, n);
                cx.case(&format!("mask.to_vec {name}"), f, Some(v), vec![A::M(w, x)], O::V(vt, Canon::Exact));
            }
        }
        let f = cx.wrap_all(FirstN, &[t.usize], Some(m));
        let counts =
            [0, 1, i128::from(n) - 1, i128::from(n), i128::from(n) + 1, 255, 256, 1 << 32, u64::MAX as i128];
        for k in counts {
            cx.case(&format!("mask.first_n {name}"), f, Some(m), vec![A::I(k)], out);
            let g = cx.wrap(FirstN, &[], Some(m), &[P::K(k)]);
            cx.case(&format!("mask.first_n const {name}"), g, Some(m), vec![], out);
        }
        for w2 in [8, 16, 32, 64] {
            if w2 * n > 128 {
                continue;
            }
            let m2 = cx.mty(w2, n);
            let f = cx.wrap_all(Rewiden, &[m], Some(m2));
            for _ in 0..6 {
                let x = cx.rng.mask(w, n);
                cx.case(&format!("mask.rewiden {name} -> {w2}"), f, Some(m2), vec![A::M(w, x)], O::M(w2, n));
            }
        }
        for op in [MaskAll, MaskNone] {
            let f = cx.wrap(op, &[], Some(m), &[]);
            cx.case(&format!("{} {name}", op.name()), f, Some(m), vec![], out);
        }
    }
    // select on vectors
    for vt in all_vts() {
        let v = cx.vty(vt);
        let m = cx.mask_of(vt);
        let w = vt.l.bits();
        let f = cx.wrap_all(Select, &[m, v, v], Some(v));
        for _ in 0..10 {
            let args =
                vec![A::M(w, cx.rng.mask(w, vt.n)), A::V(vt, cx.rng.lanes(vt)), A::V(vt, cx.rng.lanes(vt))];
            cx.case(&format!("mask.select {}", vt.name()), f, Some(v), args, O::V(vt, Canon::Exact));
        }
    }
    cx.run("simd-mask");
}

#[test]
fn shape_ops() {
    use VecOp::*;
    let mut cx = Cx::new(0x54_a9);
    let all = all_vts();
    for &vt in &all {
        let v = cx.vty(vt);
        let lane = cx.lane(vt.l);
        let name = vt.name();
        let exact = O::V(vt, Canon::Exact);
        // splat: a register, a constant
        let f = cx.wrap_all(Splat, &[lane], Some(v));
        for x in edges(vt.l) {
            let arg = match vt.l.int() {
                Some(it) => A::I(it.value(x)),
                None => A::F(vt.l, x),
            };
            cx.case(&format!("vec.splat {name}"), f, Some(v), vec![arg], exact);
        }
        if let Some(it) = vt.l.int() {
            for x in [it.min(), it.max(), 5] {
                let g = cx.wrap(Splat, &[], Some(v), &[P::K(x)]);
                cx.case(&format!("vec.splat const {name}"), g, Some(v), vec![], exact);
            }
        }
        // bitcast to every type of the same size
        for &to in all.iter().filter(|to| to.bytes() == vt.bytes() && **to != vt) {
            let tv = cx.vty(to);
            let f = cx.wrap_all(Bitcast, &[v], Some(tv));
            for _ in 0..3 {
                let a = cx.rng.lanes(vt);
                cx.case(
                    &format!("vec.bitcast {name} -> {}", to.name()),
                    f,
                    Some(tv),
                    vec![A::V(vt, a)],
                    O::V(to, Canon::Exact),
                );
            }
        }
        if vt.n >= 4 {
            let half = VT { l: vt.l, n: vt.n / 2 };
            let h = cx.vty(half);
            let pair = cx.p.m.types.tuple(vec![h, h]);
            let f = cx.wrap_all(Split, &[v], Some(pair));
            let g = cx.wrap_all(Join, &[h, h], Some(v));
            for _ in 0..6 {
                let a = cx.rng.lanes(vt);
                cx.case(
                    &format!("vec.split {name}"),
                    f,
                    Some(pair),
                    vec![A::V(vt, a)],
                    O::Pair(half, Canon::Exact),
                );
                let args = vec![A::V(half, cx.rng.lanes(half)), A::V(half, cx.rng.lanes(half))];
                cx.case(&format!("vec.join {name}"), g, Some(v), args, exact);
            }
            if let Some(wide) = vt.l.wide() {
                let wt = VT { l: wide, n: vt.n / 2 };
                let wv = cx.vty(wt);
                for op in [WidenLo, WidenHi] {
                    let f = cx.wrap_all(op, &[v], Some(wv));
                    for a in inputs(&mut cx, vt, 1, 4) {
                        cx.case(
                            &format!("{} {name}", op.name()),
                            f,
                            Some(wv),
                            vargs(vt, a),
                            O::V(wt, Canon::Any),
                        );
                    }
                }
                if !vt.l.is_float() {
                    for op in [NarrowWrap, NarrowSat] {
                        let f = cx.wrap_all(op, &[wv, wv], Some(v));
                        for a in inputs(&mut cx, wt, 2, 6) {
                            cx.case(&format!("{} -> {name}", op.name()), f, Some(v), vargs(wt, a), exact);
                        }
                    }
                }
            }
        }
        // permute to the same, half and double lane count
        for m in [vt.n, vt.n / 2, vt.n * 2] {
            let to = VT { l: vt.l, n: m };
            if !to.fits() {
                continue;
            }
            let tv = cx.vty(to);
            for round in 0..5 {
                let idx: Vec<P> = (0..m)
                    .map(|j| match round {
                        0 => u64::from(j % vt.n),
                        1 => u64::from(vt.n - 1 - j % vt.n),
                        2 => u64::from(vt.n - 1),
                        _ => cx.rng.below(u64::from(vt.n)),
                    })
                    .map(|k| P::K(i128::from(k)))
                    .collect();
                let mut operands = vec![P::Arg(0)];
                operands.extend(idx);
                let f = cx.wrap(Permute, &[v], Some(tv), &operands);
                let a = cx.rng.lanes(vt);
                cx.case(
                    &format!("vec.permute {name} -> {m} {operands:?}"),
                    f,
                    Some(tv),
                    vec![A::V(vt, a)],
                    O::V(to, Canon::Exact),
                );
            }
        }
        // lookup with every index lane type of the lane count
        for il in INTS {
            let it = VT { l: il, n: vt.n };
            if !it.fits() {
                continue;
            }
            let iv = cx.vty(it);
            let f = cx.wrap_all(Lookup, &[v, iv], Some(v));
            for round in 0..8 {
                let idx: Vec<u64> = (0..vt.n)
                    .map(|j| match (round + j) % 4 {
                        0 => cx.rng.lane(il),
                        1 => il.norm(u64::from(vt.n)),
                        _ => cx.rng.below(u64::from(vt.n)),
                    })
                    .collect();
                let args = vec![A::V(vt, cx.rng.lanes(vt)), A::V(it, idx)];
                cx.case(&format!("vec.lookup {name} by {}", it.name()), f, Some(v), args, exact);
            }
        }
        let m = cx.mask_of(vt);
        for op in [Compress, Expand] {
            let f = cx.wrap_all(op, &[v, m], Some(v));
            for _ in 0..12 {
                let args =
                    vec![A::V(vt, cx.rng.lanes(vt)), A::M(vt.l.bits(), cx.rng.mask(vt.l.bits(), vt.n))];
                cx.case(&format!("{} {name}", op.name()), f, Some(v), args, exact);
            }
        }
    }
    cx.run("simd-shape");
}

/// How a memory op of `mem_fn` names its memory and its index.
#[derive(Clone, Copy, Default)]
struct Mem {
    /// The whole array by reference instead of the slice; `off` and `len` are then unused.
    array: bool,
    /// A constant index instead of the index param.
    at: Option<u64>,
}

/// `fn(d0, d1, d2: V, off, len: usize, ..)`: the three vectors fill an array of `3n` lanes; the op
/// runs on the slice of `len` lanes from lane `off` with the remaining params as its operands
/// after the slice; then a load's result and the whole array are printed.
fn mem_fn(cx: &mut Cx, op: VecOp, vt: VT, index: Option<VT>, form: Mem) -> stave_ir::FuncId {
    use VecOp::*;
    use stave_ir::types::TypeKey;
    use stave_ir::{CastKind, Op, Operand, Site};
    let v = cx.vty(vt);
    let lane = cx.lane(vt.l);
    let m = cx.mask_of(vt);
    let arr = cx.p.m.types.array(lane, u64::from(3 * vt.n));
    let (parr, pl, pv) = (cx.refm(arr), cx.ptr(lane), cx.ptr(v));
    let slice = match op {
        LoadAligned | StoreAligned | StoreStream => {
            cx.p.m.types.intern(TypeKey::Slice { elem: lane, align: Some(vt.bytes()) })
        }
        _ => cx.p.m.types.slice(lane),
    };
    let ps = cx.refm(slice);
    let us = cx.p.t.usize;
    let iv = index.map(|it| cx.vty(it));
    let mut params = vec![v, v, v, us, us];
    let idx = form.at.is_none().then_some(us);
    match op {
        Load | LoadAligned => params.extend(idx),
        LoadMasked => params.extend([us, m]),
        Store | StoreAligned | StoreStream => params.extend(idx.into_iter().chain([v])),
        StoreMasked => params.extend([us, v, m]),
        Gather => params.extend(iv),
        _ => params.extend([iv.unwrap(), v]),
    }
    let out = cx.out(vt, Canon::Exact);
    let ret = matches!(op, Load | LoadAligned | LoadMasked | Gather).then_some(v);
    let count = params.len();
    cx.funcs += 1;
    cx.p.func("t.mem", &params, None, |b, _| {
        let s = b.slot(parr);
        let q = b.cast(CastKind::Exact, pl, s);
        let mut cells = Vec::new();
        for k in 0..3 {
            let kk = b.int(i128::from(k * vt.n));
            let e = b.binary(Op::ElemAddr, pl, q, kk);
            let c = b.cast(CastKind::Exact, pv, e);
            let d = b.param(k as usize).unwrap();
            b.store(c, d);
            cells.push(c);
        }
        let (off, len) = (b.param(3).unwrap(), b.param(4).unwrap());
        let base = b.binary(Op::ElemAddr, pl, q, off);
        let sl = b.binary(Op::SliceMake, ps, base, len);
        let mut operands: Vec<Operand> = vec![if form.array { s } else { sl }.into()];
        let rest: Vec<Operand> = (5..count).map(|k| Operand::from(b.param(k).unwrap())).collect();
        if let Some(k) = form.at {
            operands.push(b.int(i128::from(k)));
        }
        operands.extend(rest);
        b.set_site(Site(SITE));
        if let Some(r) = b.inst(Op::Vec(op), ret, &operands) {
            b.call(out, None, &[r.into()]);
        }
        for c in cells {
            let y = b.load(v, c);
            b.call(out, None, &[y.into()]);
        }
        b.ret(None);
    })
}

fn mem_types() -> Vec<VT> {
    [
        (L::I(8, false), 16),
        (L::I(16, true), 8),
        (L::I(32, false), 4),
        (L::F(32), 4),
        (L::I(64, false), 2),
        (L::F(64), 2),
        (L::I(8, true), 8),
        (L::I(16, false), 4),
        (L::F(32), 2),
        (L::I(8, false), 4),
        (L::I(16, false), 2),
        (L::I(8, false), 2),
    ]
    .into_iter()
    .map(|(l, n)| VT { l, n })
    .collect()
}

/// The three vectors behind the slice, then `off` and `len`.
fn backing(cx: &mut Cx, vt: VT, off: u64, len: u64) -> Vec<A> {
    let mut a: Vec<A> = (0..3).map(|_| A::V(vt, cx.rng.lanes(vt))).collect();
    a.extend([A::I(i128::from(off)), A::I(i128::from(len))]);
    a
}

/// Memory ops on inputs that do not trap: bases and indices off the vector size, accesses that
/// end at the slice end, masked accesses whose inactive lanes are past the end, repeated
/// scatter indices.
#[test]
fn memory_ops() {
    use VecOp::*;
    let mut cx = Cx::new(0x3e_30);
    for vt in mem_types() {
        let n = u64::from(vt.n);
        let w = vt.l.bits();
        for op in [Load, Store, LoadAligned, StoreAligned, StoreStream] {
            let f = mem_fn(&mut cx, op, vt, None, Mem::default());
            let aligned = matches!(op, LoadAligned | StoreAligned | StoreStream);
            for round in 0..9usize {
                let off = if aligned { 0 } else { [0, 1, 3][round % 3] };
                let len = n + cx.rng.below(2 * n - off + 1);
                let i = match (aligned, round % 4) {
                    (true, _) => n * cx.rng.below(len / n),
                    (_, 0) => 0,
                    (_, 1) => len - n,
                    _ => cx.rng.below(len - n + 1),
                };
                let mut args = backing(&mut cx, vt, off, len);
                args.push(A::I(i128::from(i)));
                if matches!(op, Store | StoreAligned | StoreStream) {
                    args.push(A::V(vt, cx.rng.lanes(vt)));
                }
                cx.case(&format!("{} {}", op.name(), vt.name()), f, None, args, O::Unit);
            }
        }
        for op in [LoadMasked, StoreMasked] {
            let f = mem_fn(&mut cx, op, vt, None, Mem::default());
            for round in 0..24usize {
                let mask = cx.rng.mask(w, vt.n);
                let off = [0, 1, 2][round % 3];
                let (len, i) = match mask.iter().rposition(|&x| x != 0) {
                    // nothing is accessed, wherever the index points
                    None => {
                        let len = cx.rng.below(3 * n - off + 1);
                        (len, [0, len, len + 5, u64::MAX][round % 4])
                    }
                    // the highest active lane is inside
                    Some(h) => {
                        let h = h as u64;
                        let len = h + 1 + cx.rng.below(3 * n - off - h);
                        let last = len - 1 - h;
                        (len, [0, last, cx.rng.below(last + 1)][round % 3])
                    }
                };
                let mut args = backing(&mut cx, vt, off, len);
                args.push(A::I(i128::from(i)));
                if op == StoreMasked {
                    args.push(A::V(vt, cx.rng.lanes(vt)));
                }
                args.push(A::M(w, mask));
                cx.case(&format!("{} {}", op.name(), vt.name()), f, None, args, O::Unit);
            }
        }
        for il in INTS.into_iter().filter(|l| !l.signed()) {
            let it = VT { l: il, n: vt.n };
            if !it.fits() {
                continue;
            }
            for op in [Gather, Scatter] {
                let f = mem_fn(&mut cx, op, vt, Some(it), Mem::default());
                for round in 0..8usize {
                    let off = [0, 1][round % 2];
                    let len = 1 + cx.rng.below(3 * n - off);
                    let same = cx.rng.below(len);
                    let idx: Vec<u64> =
                        (0..vt.n).map(|_| if round % 3 == 0 { same } else { cx.rng.below(len) }).collect();
                    let mut args = backing(&mut cx, vt, off, len);
                    args.push(A::V(it, idx));
                    if op == Scatter {
                        args.push(A::V(vt, cx.rng.lanes(vt)));
                    }
                    cx.case(&format!("{} {} by {}", op.name(), vt.name(), it.name()), f, None, args, O::Unit);
                }
            }
        }
        // constant indices, through the slice and through the array
        for op in [Load, Store, LoadAligned, StoreAligned, StoreStream] {
            let aligned = matches!(op, LoadAligned | StoreAligned | StoreStream);
            let mut forms =
                vec![(false, if aligned { 0 } else { 1 }), (false, if aligned { 2 * n } else { n + 1 })];
            if !aligned {
                forms.push((true, 2 * n));
            }
            for (array, at) in forms {
                let f = mem_fn(&mut cx, op, vt, None, Mem { array, at: Some(at) });
                for _ in 0..3 {
                    let off = if aligned || array { 0 } else { cx.rng.below(2 * n - at + 1) };
                    let len = at + n + cx.rng.below(2 * n - off - at + 1);
                    let mut args = backing(&mut cx, vt, off, len);
                    if matches!(op, Store | StoreAligned | StoreStream) {
                        args.push(A::V(vt, cx.rng.lanes(vt)));
                    }
                    cx.case(&format!("{} {} at {at}", op.name(), vt.name()), f, None, args, O::Unit);
                }
            }
        }
        // the whole array by reference
        let all = 3 * n;
        let array = Mem { array: true, at: None };
        for op in [Load, Store] {
            let f = mem_fn(&mut cx, op, vt, None, array);
            for i in [0, 1, all - n, cx.rng.below(all - n + 1)] {
                let mut args = backing(&mut cx, vt, 0, all);
                args.push(A::I(i128::from(i)));
                if op == Store {
                    args.push(A::V(vt, cx.rng.lanes(vt)));
                }
                cx.case(&format!("{} {} array", op.name(), vt.name()), f, None, args, O::Unit);
            }
        }
        for op in [LoadMasked, StoreMasked] {
            let f = mem_fn(&mut cx, op, vt, None, array);
            for _ in 0..8 {
                let mask = cx.rng.mask(w, vt.n);
                let i = match mask.iter().rposition(|&x| x != 0) {
                    None => u64::MAX,
                    Some(h) => cx.rng.below(all - h as u64),
                };
                let mut args = backing(&mut cx, vt, 0, all);
                args.push(A::I(i128::from(i)));
                if op == StoreMasked {
                    args.push(A::V(vt, cx.rng.lanes(vt)));
                }
                args.push(A::M(w, mask));
                cx.case(&format!("{} {} array", op.name(), vt.name()), f, None, args, O::Unit);
            }
        }
        let it = VT { l: L::I(8, false), n: vt.n };
        for op in [Gather, Scatter] {
            let f = mem_fn(&mut cx, op, vt, Some(it), array);
            for _ in 0..4 {
                let idx: Vec<u64> = (0..vt.n).map(|_| cx.rng.below(all)).collect();
                let mut args = backing(&mut cx, vt, 0, all);
                args.push(A::V(it, idx));
                if op == Scatter {
                    args.push(A::V(vt, cx.rng.lanes(vt)));
                }
                cx.case(&format!("{} {} array", op.name(), vt.name()), f, None, args, O::Unit);
            }
        }
    }
    cx.run("simd-memory");
}

/// One trapping case per program: the kind and the site must be the interpreter's.
#[test]
fn traps() {
    use VecOp::*;
    use stave_ir::PanicKind::{Bounds, DivByZero, Overflow};
    let mut count = 0;
    let mut trap = |kind: stave_ir::PanicKind, build: &dyn Fn(&mut Cx)| {
        let mut cx = Cx::new(7);
        build(&mut cx);
        count += 1;
        cx.run_trap(&format!("simd-trap-{count}"), kind);
    };
    for vt in [VT { l: L::I(32, true), n: 4 }, VT { l: L::I(8, true), n: 4 }, VT { l: L::I(64, true), n: 2 }]
    {
        let it = vt.l.int().unwrap();
        let (min, m1, last) = (it.wrap(it.min()), it.wrap(-1), vt.n as usize - 1);
        let lanes = |pairs: &[(usize, u64, u64)]| {
            let (mut a, mut d) = (vec![it.wrap(6); vt.n as usize], vec![it.wrap(3); vt.n as usize]);
            for &(k, x, y) in pairs {
                (a[k], d[k]) = (x, y);
            }
            vec![A::V(vt, a), A::V(vt, d)]
        };
        // (lanes, kind): the lowest trapping lane decides
        let cases: [(Vec<A>, stave_ir::PanicKind); 5] = [
            (lanes(&[(last, 5, 0)]), DivByZero),
            (lanes(&[(0, 5, 0)]), DivByZero),
            (lanes(&[(last, min, m1)]), Overflow),
            (lanes(&[(0, min, m1), (1, 1, 0)]), Overflow),
            (lanes(&[(0, 1, 0), (1, min, m1)]), DivByZero),
        ];
        for (args, kind) in cases {
            trap(kind, &|cx| {
                let v = cx.vty(vt);
                let f = cx.wrap_all(DivLanes, &[v, v], Some(v));
                cx.case("div_lanes", f, Some(v), args.clone(), O::V(vt, Canon::Exact));
            });
        }
        for (d, x, kind) in [(0, it.wrap(5), DivByZero), (-1, min, Overflow)] {
            trap(kind, &|cx| {
                let v = cx.vty(vt);
                let f = cx.wrap(DivConst, &[v], Some(v), &[P::Arg(0), P::K(d)]);
                let mut a = vec![it.wrap(9); vt.n as usize];
                a[last] = x;
                cx.case("div_const", f, Some(v), vec![A::V(vt, a)], O::V(vt, Canon::Exact));
            });
        }
        let bits = i128::from(it.bits);
        for op in [Shl, Shr] {
            for (k, amount) in [(0, it.wrap(bits)), (last, it.wrap(bits)), (last, m1)] {
                trap(Overflow, &|cx| {
                    let v = cx.vty(vt);
                    let f = cx.wrap_all(op, &[v, v], Some(v));
                    let mut n = vec![it.wrap(1); vt.n as usize];
                    n[k] = amount;
                    let args = vec![A::V(vt, vec![it.wrap(3); vt.n as usize]), A::V(vt, n)];
                    cx.case("shift by lanes", f, Some(v), args, O::V(vt, Canon::Exact));
                });
            }
            for amount in [bits, -1] {
                trap(Overflow, &|cx| {
                    let v = cx.vty(vt);
                    let f = cx.wrap_all(op, &[v, cx.p.t.i64], Some(v));
                    let args = vec![A::V(vt, vec![it.wrap(3); vt.n as usize]), A::I(amount)];
                    cx.case("shift by a register", f, Some(v), args, O::V(vt, Canon::Exact));
                });
            }
            trap(Overflow, &|cx| {
                let v = cx.vty(vt);
                let f = cx.wrap(op, &[v], Some(v), &[P::Arg(0), P::K(bits)]);
                let args = vec![A::V(vt, vec![it.wrap(3); vt.n as usize])];
                cx.case("shift by a constant", f, Some(v), args, O::V(vt, Canon::Exact));
            });
        }
    }
    for vt in [VT { l: L::I(32, false), n: 4 }, VT { l: L::I(8, false), n: 2 }, VT { l: L::F(64), n: 2 }] {
        let n = u64::from(vt.n);
        let w = vt.l.bits();
        let ones = if w == 64 { u64::MAX } else { (1 << w) - 1 };
        for op in [LoadAligned, StoreAligned, StoreStream] {
            // (index, constant)
            for (i, konst) in [(1, false), (n + 1, false), (n + 1, true)] {
                trap(Bounds, &|cx| {
                    let form = Mem { array: false, at: konst.then_some(i) };
                    let f = mem_fn(cx, op, vt, None, form);
                    let mut args = backing(cx, vt, 0, 3 * n);
                    if !konst {
                        args.push(A::I(i128::from(i)));
                    }
                    if op != LoadAligned {
                        args.push(A::V(vt, cx.rng.lanes(vt)));
                    }
                    cx.case("aligned access", f, None, args, O::Unit);
                });
            }
        }
        for op in [LoadMasked, StoreMasked] {
            // (len, index, active lane): the lane is at or past the end
            for (len, i, lane) in
                [(n, 1, n - 1), (n, n, 0), (0, 0, 0), (2 * n, u64::MAX, 1), (n + 1, 2, n - 1)]
            {
                trap(Bounds, &|cx| {
                    let f = mem_fn(cx, op, vt, None, Mem::default());
                    let mut args = backing(cx, vt, 1, len);
                    args.push(A::I(i128::from(i)));
                    if op == StoreMasked {
                        args.push(A::V(vt, cx.rng.lanes(vt)));
                    }
                    let mut mask = vec![0; vt.n as usize];
                    mask[lane as usize] = ones;
                    args.push(A::M(w, mask));
                    cx.case("masked access", f, None, args, O::Unit);
                });
            }
        }
        for op in [Gather, Scatter] {
            for (len, lane, index) in [(n, 0, n), (n + 3, n - 1, n + 3), (0, 0, 0), (5, 1, 255)] {
                trap(Bounds, &|cx| {
                    let it = VT { l: L::I(8, false), n: vt.n };
                    let f = mem_fn(cx, op, vt, Some(it), Mem::default());
                    let mut args = backing(cx, vt, 0, len);
                    let mut idx = vec![0; vt.n as usize];
                    idx[lane as usize] = index;
                    args.push(A::V(it, idx));
                    if op == Scatter {
                        args.push(A::V(vt, cx.rng.lanes(vt)));
                    }
                    cx.case("gather / scatter", f, None, args, O::Unit);
                });
            }
        }
    }
    eprintln!("simd traps: {count} programs");
}

#[test]
fn checked_ops() {
    use VecOp::*;
    let mut cx = Cx::new(0xc4_ec);
    for vt in int_vts() {
        let v = cx.vty(vt);
        let o = cx.option(v);
        for op in [CheckedAdd, CheckedSub, CheckedMul] {
            let f = cx.wrap_all(op, &[v, v], Some(o));
            let mut ins = inputs(&mut cx, vt, 2, 8);
            // small lanes: no overflow
            for _ in 0..6 {
                let mut small = || (0..vt.n).map(|_| vt.l.norm(cx.rng.below(12))).collect::<Vec<u64>>();
                let pair = vec![small(), small()];
                ins.push(pair);
            }
            for a in ins {
                let args = a.into_iter().map(|x| A::V(vt, x)).collect();
                cx.case(&format!("{} {}", op.name(), vt.name()), f, Some(o), args, O::Opt(vt, Canon::Exact));
            }
        }
    }
    cx.run("simd-checked");
}

#[test]
fn casts() {
    use stave_ir::CastKind::*;
    use stave_ir::arith::cast_allowed;
    let lanes: Vec<L> = INTS.iter().chain(FLOATS.iter()).copied().collect();
    for kinds in [vec![Exact, Wrap, Sat, Round], vec![Try]] {
        let mut cx = Cx::new(0xca_57);
        for &kind in &kinds {
            for &from in &lanes {
                for &to in &lanes {
                    if !cast_allowed(kind, from.cast_ty(), to.cast_ty()) {
                        continue;
                    }
                    for n in [2u32, 4, 8, 16] {
                        let (fv, tv) = (VT { l: from, n }, VT { l: to, n });
                        if !fv.fits() || !tv.fits() {
                            continue;
                        }
                        let (ft, tt) = (cx.vty(fv), cx.vty(tv));
                        let (ret, out) = if kind == Try {
                            (cx.option(tt), O::Opt(tv, Canon::Any))
                        } else {
                            (tt, O::V(tv, Canon::Any))
                        };
                        let f = cx.wrap_all(VecOp::Cast(kind), &[ft], Some(ret));
                        for a in inputs(&mut cx, fv, 1, 3) {
                            let what = format!("{} {} -> {}", VecOp::Cast(kind).name(), fv.name(), tv.name());
                            cx.case(&what, f, Some(ret), vec![A::V(fv, a[0].clone())], out);
                        }
                    }
                }
            }
        }
        cx.run(if kinds[0] == Try { "simd-cast-try" } else { "simd-cast" });
    }
}

/// The test of this file that runs each op (the match is exhaustive: a new op must be placed).
fn tested_by(op: VecOp) -> &'static str {
    use VecOp::*;
    match op {
        Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc | Add | Sub | Mul | Div | Copysign | MinNum
        | MaxNum | MinX86 | MaxX86 | Fma => "float_lane_ops",
        Min | Max => "float_lane_ops, int_lane_ops",
        AbsWrap | Popcount | Clz | Not | AddWrap | SubWrap | MulWrap | AddSat | SubSat | MulSat | And
        | Or | Xor | Andnot => "int_lane_ops",
        Eq | Ne | Lt | Le | Gt | Ge | ReduceAdd | ReduceAddWrap | ReduceMin | ReduceMax | ReduceAnd
        | ReduceOr | ReduceAddOrdered | ScanAdd | ScanAddWrap => "compares_reductions_scans",
        Shl | Shr | ShlWrap | DivLanes | DivConst => "shifts_and_divisions, traps",
        CheckedAdd | CheckedSub | CheckedMul => "checked_ops",
        Cast(_) => "casts",
        Splat | Bitcast | Split | Join | WidenLo | WidenHi | NarrowWrap | NarrowSat | Permute | Lookup
        | Compress | Expand => "shape_ops",
        Load | Store => "memory_ops",
        LoadAligned | StoreAligned | StoreStream | LoadMasked | StoreMasked | Gather | Scatter => {
            "memory_ops, traps"
        }
        MaskAnd | MaskOr | MaskXor | MaskNot | Select | ToBits | ToVec | FirstN | Rewiden | MaskAll
        | MaskNone => "mask_ops",
    }
}

/// Ops the backend does not generate code for (none: every op is tested above).
const UNSUPPORTED: [VecOp; 0] = [];

#[test]
fn every_op_is_tested_or_listed_unsupported() {
    let tested =
        VecOp::ALL.iter().filter(|op| !UNSUPPORTED.contains(op) && !tested_by(**op).is_empty()).count();
    assert_eq!(tested + UNSUPPORTED.len(), VecOp::ALL.len());
    assert_eq!(VecOp::ALL.len(), 93);
}

/// Vector values outside the vector ops: arguments in registers and on the stack (ten vectors,
/// floats between them), block parameters that swap on a loop edge, fields of a tuple, the
/// scalar `select`, a mask argument and result.
#[test]
fn values_and_calls() {
    use stave_ir::{Op, Operand};
    let mut cx = Cx::new(0xab_1c);
    let t = cx.p.t;
    for vt in [VT { l: L::I(32, false), n: 4 }, VT { l: L::I(8, false), n: 4 }, VT { l: L::F(64), n: 2 }] {
        let v = cx.vty(vt);
        let m = cx.mask_of(vt);
        let w = vt.l.bits();
        let bits = VT { l: vt.l.uint(), n: vt.n };
        let bv = cx.vty(bits);
        let exact = O::V(vt, Canon::Exact);
        // ((a0 << 1) ^ a1) << 1 ^ ..: every argument at its own position
        let mut params = vec![v; 10];
        params.insert(3, t.f64);
        params.insert(9, t.f32);
        cx.funcs += 1;
        let many = cx.p.func("t.many", &params, Some(v), |b, _| {
            let mut acc: Option<stave_ir::Value> = None;
            for k in (0..12).filter(|k| *k != 3 && *k != 9) {
                let x = b.param(k).unwrap();
                let xb = b.op(Op::Vec(VecOp::Bitcast), bv, &[x.into()]);
                acc = Some(match acc {
                    None => xb,
                    Some(a) => {
                        let one = b.int(1);
                        let sh = b.op(Op::Vec(VecOp::ShlWrap), bv, &[a.into(), one]);
                        b.op(Op::Vec(VecOp::Xor), bv, &[sh.into(), xb.into()])
                    }
                });
            }
            let r = b.op(Op::Vec(VecOp::Bitcast), v, &[acc.unwrap().into()]);
            b.ret(Some(r.into()));
        });
        for _ in 0..4 {
            let mut args: Vec<A> = (0..10).map(|_| A::V(vt, cx.rng.lanes(vt))).collect();
            args.insert(3, A::F(L::F(64), 1.5f64.to_bits()));
            args.insert(9, A::F(L::F(32), u64::from(2.5f32.to_bits())));
            cx.case(&format!("ten vector arguments {}", vt.name()), many, Some(v), args, exact);
        }
        // (x, y) -> (y, x ^ y) around a loop: the two vector params swap on the back edge
        cx.funcs += 1;
        let swap = cx.p.func("t.swap", &[v, v, t.usize], Some(v), |b, _| {
            let (x, y, n) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
            let (head, hp) = b.create_block_with(&[t.usize, v, v]);
            let (body, exit) = (b.create_block(), b.create_block());
            let z = b.int(0);
            b.br(head, &[z, x.into(), y.into()]);
            b.switch_to(head);
            let c = b.cmp(Op::Lt, t.bool, hp[0], n);
            b.cbr(c, body, &[], exit, &[]);
            b.switch_to(body);
            let xb = b.op(Op::Vec(VecOp::Bitcast), bv, &[hp[1].into()]);
            let yb = b.op(Op::Vec(VecOp::Bitcast), bv, &[hp[2].into()]);
            let mix = b.op(Op::Vec(VecOp::Xor), bv, &[xb.into(), yb.into()]);
            let next = b.op(Op::Vec(VecOp::Bitcast), v, &[mix.into()]);
            let one = b.int(1);
            let i1 = b.binary(Op::AddWrap, t.usize, hp[0], one);
            b.br(head, &[i1.into(), hp[2].into(), next.into()]);
            b.switch_to(exit);
            b.ret(Some(hp[1].into()));
        });
        for n in [0, 1, 2, 5] {
            let args = vec![A::V(vt, cx.rng.lanes(vt)), A::V(vt, cx.rng.lanes(vt)), A::I(n)];
            cx.case(&format!("swapping block params {}", vt.name()), swap, Some(v), args, exact);
        }
        // a tuple (V, u8, M, V) made and taken apart; `select` of vectors by a bool
        let tup = cx.p.m.types.tuple(vec![v, t.u8, m, v]);
        cx.funcs += 1;
        let pick = cx.p.func("t.pick", &[v, v, m, t.u8], Some(v), |b, _| {
            let args: Vec<Operand> = (0..4).map(|k| Operand::from(b.param(k).unwrap())).collect();
            let seven = b.int(7);
            let tv = b.op(Op::Make, tup, &[args[0], seven, args[2], args[1]]);
            let (k0, k2, k3) = (b.int(0), b.int(2), b.int(3));
            let x = b.op(Op::Get, v, &[tv.into(), k0]);
            let mm = b.op(Op::Get, m, &[tv.into(), k2]);
            let y = b.op(Op::Get, v, &[tv.into(), k3]);
            let blend = b.op(Op::Vec(VecOp::Select), v, &[mm.into(), x.into(), y.into()]);
            let z = b.int(0);
            let c = b.cmp(Op::Ne, t.bool, args[3], z);
            let r = b.select(v, c, blend, y);
            b.ret(Some(r.into()));
        });
        cx.funcs += 1;
        let not = cx.p.func("t.maskfn", &[m, v], Some(m), |b, _| {
            let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
            let eq = b.op(Op::Vec(VecOp::Eq), m, &[y.into(), y.into()]);
            let r = b.op(Op::Vec(VecOp::MaskXor), m, &[x.into(), eq.into()]);
            b.ret(Some(r.into()));
        });
        for k in 0..6 {
            let mask = cx.rng.mask(w, vt.n);
            let args = vec![
                A::V(vt, cx.rng.lanes(vt)),
                A::V(vt, cx.rng.lanes(vt)),
                A::M(w, mask.clone()),
                A::I(k % 2),
            ];
            cx.case(&format!("tuple and select {}", vt.name()), pick, Some(v), args, exact);
            let args = vec![A::M(w, mask), A::V(vt, cx.rng.lanes(vt))];
            cx.case(&format!("mask argument and result {}", vt.name()), not, Some(m), args, O::M(w, vt.n));
        }
    }
    cx.run("simd-values");
}

/// Starts a counted loop: branches to a new block `head(i, vars..)` with `i = 0` and the given
/// initial values, and leaves the builder in it, for header instructions before [`loop_test`].
fn loop_head(
    b: &mut stave_ir::FuncBuilder,
    usize_ty: stave_ir::TypeId,
    vars: &[stave_ir::TypeId],
    init: &[stave_ir::Operand],
) -> (stave_ir::Block, Vec<stave_ir::Value>) {
    let mut tys = vec![usize_ty];
    tys.extend_from_slice(vars);
    let (head, hp) = b.create_block_with(&tys);
    let mut args = vec![b.int(0)];
    args.extend_from_slice(init);
    b.br(head, &args);
    b.switch_to(head);
    (head, hp)
}

/// Ends the head of [`loop_head`]: the loop runs its body, which the builder enters, while
/// `i < n`. Returns the exit block.
fn loop_test(
    b: &mut stave_ir::FuncBuilder,
    bool_ty: stave_ir::TypeId,
    i: stave_ir::Value,
    n: stave_ir::Value,
) -> stave_ir::Block {
    let (body, exit) = (b.create_block(), b.create_block());
    let c = b.cmp(stave_ir::Op::Lt, bool_ty, i, n);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    exit
}

/// Ends the body of a loop of [`loop_head`] with the next values of its variables and moves to
/// `exit`.
fn loop_next(
    b: &mut stave_ir::FuncBuilder,
    usize_ty: stave_ir::TypeId,
    head: stave_ir::Block,
    i: stave_ir::Value,
    next: &[stave_ir::Operand],
    exit: stave_ir::Block,
) {
    let one = b.int(1);
    let i1 = b.binary(stave_ir::Op::AddWrap, usize_ty, i, one);
    let mut args = vec![i1.into()];
    args.extend_from_slice(next);
    b.br(head, &args);
    b.switch_to(exit);
}

/// Vectors that live across blocks: a loop variable updated in place by a lane op, a value
/// defined in a loop's header and read in its body, an `fma` that adds to a loop variable and
/// one that multiplies it, three variables that rotate on the back edge next to two floats that
/// swap, a value live across a call in a loop, and more live vectors than registers. Vectors
/// below 16 bytes go through `not`, which sets the register bits past them.
#[test]
fn vectors_across_blocks() {
    use stave_ir::{Op, Operand};
    let mut cx = Cx::new(0x0a64_b10c);
    let t = cx.p.t;
    let shapes = [
        VT { l: L::I(32, false), n: 4 },
        VT { l: L::I(16, true), n: 4 },
        VT { l: L::I(8, false), n: 4 },
        VT { l: L::F(32), n: 4 },
        VT { l: L::F(32), n: 2 },
        VT { l: L::F(64), n: 2 },
    ];
    for vt in shapes {
        let v = cx.vty(vt);
        let bv = cx.vty(VT { l: vt.l.uint(), n: vt.n });
        let exact = O::V(vt, Canon::Exact);
        let name = vt.name();
        let two =
            |cx: &mut Cx, n: i128| vec![A::V(vt, cx.rng.lanes(vt)), A::V(vt, cx.rng.lanes(vt)), A::I(n)];

        // acc = not(acc) + x, then acc ^= y: the second op is the variable's next value
        cx.funcs += 1;
        let accum = cx.p.func("t.accum", &[v, v, t.usize], Some(v), |b, _| {
            let (x, y, n) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
            let xb = b.op(Op::Vec(VecOp::Bitcast), bv, &[x.into()]);
            let yb = b.op(Op::Vec(VecOp::Bitcast), bv, &[y.into()]);
            let (head, hp) = loop_head(b, t.usize, &[bv], &[xb.into()]);
            let exit = loop_test(b, t.bool, hp[0], n);
            let na = b.op(Op::Vec(VecOp::Not), bv, &[hp[1].into()]);
            let sum = b.op(Op::Vec(VecOp::AddWrap), bv, &[na.into(), xb.into()]);
            let next = b.op(Op::Vec(VecOp::Xor), bv, &[sum.into(), yb.into()]);
            loop_next(b, t.usize, head, hp[0], &[next.into()], exit);
            let r = b.op(Op::Vec(VecOp::Bitcast), v, &[hp[1].into()]);
            b.ret(Some(r.into()));
        });
        // h = not(acc) in the header; the body and the exit read it
        cx.funcs += 1;
        let header = cx.p.func("t.header", &[v, v, t.usize], Some(v), |b, _| {
            let (x, y, n) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
            let xb = b.op(Op::Vec(VecOp::Bitcast), bv, &[x.into()]);
            let yb = b.op(Op::Vec(VecOp::Bitcast), bv, &[y.into()]);
            let (head, hp) = loop_head(b, t.usize, &[bv], &[xb.into()]);
            let h = b.op(Op::Vec(VecOp::Not), bv, &[hp[1].into()]);
            let exit = loop_test(b, t.bool, hp[0], n);
            let next = b.op(Op::Vec(VecOp::AddWrap), bv, &[h.into(), yb.into()]);
            loop_next(b, t.usize, head, hp[0], &[next.into()], exit);
            let mix = b.op(Op::Vec(VecOp::Xor), bv, &[h.into(), hp[1].into()]);
            let both = b.op(Op::Vec(VecOp::SubWrap), bv, &[mix.into(), yb.into()]);
            let r = b.op(Op::Vec(VecOp::Bitcast), v, &[both.into()]);
            b.ret(Some(r.into()));
        });
        // (p, q, r) -> (q, r, p ^ q) and (f, g) -> (g, f) on the back edge
        cx.funcs += 1;
        let rotate = cx.p.func("t.rotate", &[v, v, v, t.f64, t.f64, t.usize], Some(v), |b, _| {
            let p: Vec<Operand> = (0..3).map(|k| Operand::from(b.param(k).unwrap())).collect();
            let (f, g, n) = (b.param(3).unwrap(), b.param(4).unwrap(), b.param(5).unwrap());
            let init = [p[0], p[1], p[2], f.into(), g.into()];
            let (head, hp) = loop_head(b, t.usize, &[v, v, v, t.f64, t.f64], &init);
            let exit = loop_test(b, t.bool, hp[0], n);
            let pb = b.op(Op::Vec(VecOp::Bitcast), bv, &[hp[1].into()]);
            let qb = b.op(Op::Vec(VecOp::Bitcast), bv, &[hp[2].into()]);
            let mix = b.op(Op::Vec(VecOp::Xor), bv, &[pb.into(), qb.into()]);
            let last = b.op(Op::Vec(VecOp::Bitcast), v, &[mix.into()]);
            let next: [Operand; 5] = [hp[2].into(), hp[3].into(), last.into(), hp[5].into(), hp[4].into()];
            loop_next(b, t.usize, head, hp[0], &next, exit);
            let c = b.cmp(Op::Lt, t.bool, hp[4], hp[5]);
            let r = b.select(v, c, hp[1], hp[3]);
            b.ret(Some(r.into()));
        });
        // acc = mix(acc, x) + y with x and y live across the call
        cx.funcs += 1;
        let mix = cx.p.func("t.mix", &[v, v], Some(v), |b, _| {
            let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
            let xb = b.op(Op::Vec(VecOp::Bitcast), bv, &[x.into()]);
            let yb = b.op(Op::Vec(VecOp::Bitcast), bv, &[y.into()]);
            let one = b.int(1);
            let sh = b.op(Op::Vec(VecOp::ShlWrap), bv, &[xb.into(), one]);
            let r = b.op(Op::Vec(VecOp::Xor), bv, &[sh.into(), yb.into()]);
            let r = b.op(Op::Vec(VecOp::Bitcast), v, &[r.into()]);
            b.ret(Some(r.into()));
        });
        cx.funcs += 1;
        let calls = cx.p.func("t.calls", &[v, v, t.usize], Some(v), |b, _| {
            let (x, y, n) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
            let yb = b.op(Op::Vec(VecOp::Bitcast), bv, &[y.into()]);
            let ny = b.op(Op::Vec(VecOp::Not), bv, &[yb.into()]);
            let (head, hp) = loop_head(b, t.usize, &[v], &[y.into()]);
            let exit = loop_test(b, t.bool, hp[0], n);
            let m = b.call(mix, Some(v), &[hp[1].into(), x.into()]).unwrap();
            let mb = b.op(Op::Vec(VecOp::Bitcast), bv, &[m.into()]);
            let sum = b.op(Op::Vec(VecOp::AddWrap), bv, &[mb.into(), ny.into()]);
            let next = b.op(Op::Vec(VecOp::Bitcast), v, &[sum.into()]);
            loop_next(b, t.usize, head, hp[0], &[next.into()], exit);
            let ab = b.op(Op::Vec(VecOp::Bitcast), bv, &[hp[1].into()]);
            let r = b.op(Op::Vec(VecOp::Xor), bv, &[ab.into(), ny.into()]);
            let r = b.op(Op::Vec(VecOp::Bitcast), v, &[r.into()]);
            b.ret(Some(r.into()));
        });
        // twelve values made before the loop, all read in it
        cx.funcs += 1;
        let many = cx.p.func("t.many_live", &[v, v, t.usize], Some(v), |b, _| {
            let (x, y, n) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
            let xb = b.op(Op::Vec(VecOp::Bitcast), bv, &[x.into()]);
            let yb = b.op(Op::Vec(VecOp::Bitcast), bv, &[y.into()]);
            let mut live = vec![xb];
            for _ in 1..12 {
                let one = b.int(1);
                let sh = b.op(Op::Vec(VecOp::ShlWrap), bv, &[live[live.len() - 1].into(), one]);
                live.push(b.op(Op::Vec(VecOp::Xor), bv, &[sh.into(), yb.into()]));
            }
            let (head, hp) = loop_head(b, t.usize, &[bv], &[yb.into()]);
            let exit = loop_test(b, t.bool, hp[0], n);
            let mut acc = hp[1];
            for (k, e) in live.iter().enumerate() {
                let op = if k % 2 == 0 { VecOp::AddWrap } else { VecOp::Xor };
                acc = b.op(Op::Vec(op), bv, &[acc.into(), (*e).into()]);
            }
            loop_next(b, t.usize, head, hp[0], &[acc.into()], exit);
            let r = b.op(Op::Vec(VecOp::Bitcast), v, &[hp[1].into()]);
            b.ret(Some(r.into()));
        });
        for n in [0, 1, 2, 5] {
            let args = two(&mut cx, n);
            cx.case(&format!("loop variable in place {name}"), accum, Some(v), args, exact);
            let args = two(&mut cx, n);
            cx.case(&format!("value of a loop header {name}"), header, Some(v), args, exact);
            let args = two(&mut cx, n);
            cx.case(&format!("vectors across a call {name}"), calls, Some(v), args, exact);
            let args = two(&mut cx, n);
            cx.case(&format!("more vectors than registers {name}"), many, Some(v), args, exact);
            for (f, g) in [(1.5f64, 2.5f64), (2.5, 1.5)] {
                let mut args: Vec<A> = (0..3).map(|_| A::V(vt, cx.rng.lanes(vt))).collect();
                args.extend([A::F(L::F(64), f.to_bits()), A::F(L::F(64), g.to_bits()), A::I(n)]);
                cx.case(&format!("rotating block params {name}"), rotate, Some(v), args, exact);
            }
        }
        if !vt.l.is_float() {
            continue;
        }
        // acc = x * y + acc in place; alt = alt * x + y, which reads alt after writing the result
        cx.funcs += 1;
        let fma = cx.p.func("t.fma_loop", &[v, v, t.usize], Some(v), |b, _| {
            let (x, y, n) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
            let (head, hp) = loop_head(b, t.usize, &[v, v], &[x.into(), y.into()]);
            let exit = loop_test(b, t.bool, hp[0], n);
            let acc = b.op(Op::Vec(VecOp::Fma), v, &[x.into(), y.into(), hp[1].into()]);
            let alt = b.op(Op::Vec(VecOp::Fma), v, &[hp[2].into(), x.into(), y.into()]);
            loop_next(b, t.usize, head, hp[0], &[acc.into(), alt.into()], exit);
            let r = b.op(Op::Vec(VecOp::Sub), v, &[hp[1].into(), hp[2].into()]);
            b.ret(Some(r.into()));
        });
        for n in [0, 1, 3] {
            let args = two(&mut cx, n);
            cx.case(&format!("fma on loop variables {name}"), fma, Some(v), args, O::V(vt, Canon::Quiet));
        }
    }
    cx.run("simd-blocks");
}

/// Vector shapes the backend does not generate code for are reported as unsupported.
#[test]
fn wide_vectors_are_unsupported() {
    use common::{Prog, inst};
    use stave_codegen_a64::debug::{BuildConfig, ErrorKind, build_image};
    use stave_ir::types::{Lanes, TypeKey};
    use stave_ir::{FuncBuilder, Op};
    for (lane_bits, n) in [(8u32, 32u8), (32, 8), (64, 4)] {
        let mut p = Prog::new();
        let lane = L::I(lane_bits as u8, false).tid(&p.t);
        let v = p.m.types.intern(TypeKey::Vec { lane, n: Lanes::Fixed(n) });
        let f = p.m.declare(inst(), "t.wide", vec![lane], None);
        let mut b = FuncBuilder::new(&[lane]);
        let x = b.param(0).unwrap();
        b.op(Op::Vec(VecOp::Splat), v, &[x.into()]);
        b.ret(None);
        p.m.define(f, b.finish().unwrap());
        p.set_main(|_, _| {});
        let Err(e) = build_image(&p.m, &BuildConfig::new(None)) else { panic!("a 256-bit vector compiled") };
        assert!(matches!(&e.kind, ErrorKind::Unsupported(s) if s.contains("wider than 128 bits")), "{e}");
    }
}
