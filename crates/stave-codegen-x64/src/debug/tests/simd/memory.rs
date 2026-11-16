//! Vector loads and stores (plain, aligned, masked), gathers and scatters, on slices in `mmap`ed
//! regions: element- and vector-misaligned bases, the last valid index, slices that end where the
//! next page is unmapped (a native over-read or over-write faults), and the bounds traps.

use stave_ir::{CastKind, FuncBuilder, Op, Operand, PanicKind, TypeId, TypeKey, Value, VecOp};

use super::values::*;
use super::{Case, Lane, Shape, Sp, full_shapes, sh, shapes};

/// A slice `&mut [L]` (or `&mut [L align A]`) of `len` elements at `base + off`.
struct Sl {
    v: Value,
    addr: Value,
    len: u64,
    s: Shape,
}

impl Sp {
    fn slice_t(&mut self, l: Lane, align: Option<u32>) -> TypeId {
        let e = self.lane_t(l);
        let st = self.p.m.types.intern(TypeKey::Slice { elem: e, align });
        self.p.m.types.ref_(true, st)
    }

    fn mk_slice(
        &mut self,
        b: &mut FuncBuilder,
        base: Value,
        off: u64,
        len: u64,
        s: Shape,
        align: Option<u32>,
    ) -> Sl {
        let addr = self.addr(b, base, off);
        let e = self.lane_t(s.lane);
        let pe = self.p.m.types.ptr(true, e);
        let p = b.cast(CastKind::Exact, pe, addr);
        let st = self.slice_t(s.lane, align);
        let n = b.int(len as i128);
        let v = b.op(Op::SliceMake, st, &[p.into(), n]);
        Sl { v, addr, len, s }
    }

    /// Stores lane bits `x` at `addr + k * lane bytes`.
    fn poke(&mut self, b: &mut FuncBuilder, addr: Value, l: Lane, k: u64, x: u64) {
        let u = self.lane_t(Lane::uint(l.bits()));
        let pu = self.p.m.types.ptr(true, u);
        let a = self.addr(b, addr, k * l.bytes() as u64);
        let p = b.cast(CastKind::Exact, pu, a);
        let c = b.int(x as i128);
        b.store(p, c);
    }

    /// Prints elements `from..to` of a slice (clamped to the slice).
    fn dump_elems(&mut self, b: &mut FuncBuilder, sl: &Sl, from: i64, to: u64) {
        let w = sl.s.lane.bytes() as u64;
        let from = from.max(0) as u64;
        let to = to.min(sl.len);
        let a = self.addr(b, sl.addr, from * w);
        let n = b.int(((to - from) * w) as i128);
        self.dump(b, a.into(), n);
    }

    /// `check.bounds i (sub.sat len n-1)`, the guard of a plain or aligned vector access.
    fn guard(&mut self, b: &mut FuncBuilder, sl: &Sl, i: Operand) {
        let t = self.t();
        let len = b.op(Op::SliceLen, t.usize, &[sl.v.into()]);
        let k = b.int(sl.s.n as i128 - 1);
        let lim = b.binary(Op::SubSat, t.usize, len, k);
        b.bounds_check(i, lim);
    }

    fn vload(&mut self, b: &mut FuncBuilder, op: VecOp, sl: &Sl, i: Operand, m: Option<u64>) -> Value {
        let vt = self.vt(sl.s);
        let mut args = vec![sl.v.into(), i];
        if let Some(m) = m {
            args.push(self.mask(b, sl.s.lane.bits(), sl.s.n, m).into());
        }
        b.op(Op::Vec(op), vt, &args)
    }

    fn vstore(&mut self, b: &mut FuncBuilder, op: VecOp, sl: &Sl, i: Operand, x: Value, m: Option<u64>) {
        let mut args = vec![sl.v.into(), i, x.into()];
        if let Some(m) = m {
            args.push(self.mask(b, sl.s.lane.bits(), sl.s.n, m).into());
        }
        b.inst(Op::Vec(op), None, &args);
    }
}

fn all_mask(n: u32) -> u64 {
    if n == 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// A vector of distinct recognizable lanes (NaN payloads for floats).
fn lanes(s: Shape, seed: u64) -> Vec<u64> {
    let mut r = Rng::new(seed + 991);
    let nn = if s.lane.is_float() { nans(s.lane) } else { vec![] };
    (0..s.n as usize)
        .map(|j| if j < nn.len() && seed.is_multiple_of(2) { nn[j] } else { r.next() & s.lane.ones() })
        .collect()
}

/// Plain, masked and aligned loads and stores at misaligned bases and the last valid index.
fn plain(s: Shape) -> Case {
    let n = s.n as u64;
    let w = s.lane.bytes() as u64;
    Case::new(
        &[VecOp::Load, VecOp::Store, VecOp::LoadMasked, VecOp::StoreMasked],
        s.bits(),
        400,
        move |sp, b| {
            let base = sp.region(b, 1);
            let all = b.int(4096);
            sp.fill(b, base.into(), all, s.n as u64 * 7 + w);
            let mut offs = vec![0, 1, w, 3];
            offs.dedup();
            for (k, &off) in offs.iter().enumerate() {
                let len = 3 * n + 3;
                let sl = sp.mk_slice(b, base, 64 + off, len, s, None);
                if s.lane.is_float() {
                    for (j, x) in nans(s.lane).into_iter().enumerate() {
                        sp.poke(b, sl.addr, s.lane, j as u64 + 1, x);
                    }
                }
                for (j, i) in [0, 1, len - n].into_iter().enumerate() {
                    sp.label(b, &format!("vec.load {} off {off} i {i}", s.name()), None);
                    let io = sp.usize(b, i, j % 2 == 0);
                    sp.guard(b, &sl, io);
                    let x = sp.vload(b, VecOp::Load, &sl, io, None);
                    sp.out(b, x, s);
                }
                for (j, i) in [1, len - n].into_iter().enumerate() {
                    sp.label(b, &format!("vec.store {} off {off} i {i}", s.name()), None);
                    let x = sp.vec(b, s, &lanes(s, j as u64 + k as u64));
                    let io = sp.usize(b, i, j % 2 == 1);
                    sp.guard(b, &sl, io);
                    sp.vstore(b, VecOp::Store, &sl, io, x, None);
                    sp.dump_elems(b, &sl, i as i64 - 1, i + n + 1);
                }
                for m in mask_patterns(s.n).into_iter().take(6) {
                    for i in [0, len - n] {
                        sp.label(
                            b,
                            &format!("vec.load_masked {} off {off} i {i} mask {m:x}", s.name()),
                            None,
                        );
                        let io = sp.usize(b, i, false);
                        let x = sp.vload(b, VecOp::LoadMasked, &sl, io, Some(m));
                        sp.out(b, x, s);
                    }
                    sp.label(b, &format!("vec.store_masked {} off {off} i 2 mask {m:x}", s.name()), None);
                    let x = sp.vec(b, s, &lanes(s, m));
                    let io = sp.usize(b, 2, false);
                    sp.vstore(b, VecOp::StoreMasked, &sl, io, x, Some(m));
                    sp.dump_elems(b, &sl, 1, n + 3);
                }
            }
            // an array operand in a slot
            let t = sp.t();
            let e = sp.lane_t(s.lane);
            let arr = sp.p.m.types.array(e, 2 * n);
            let pa = sp.p.m.types.ref_(true, arr);
            let slot = b.slot(pa);
            let pe = sp.p.m.types.ref_(true, e);
            for j in 0..2 * n {
                let kj = b.int(j as i128);
                let a = b.op(Op::ElemAddr, pe, &[slot.into(), kj]);
                let c = if s.lane.is_float() {
                    let bits = lanes(s, j)[0];
                    sp.scalar(b, s.lane, bits, true)
                } else {
                    b.int(s.lane.value(s.lane.raw(j as i128 * 37 - 5)))
                };
                b.store(a, c);
            }
            sp.label(b, &format!("vec.load {} array[{}] i {}", s.name(), 2 * n, n - 1), None);
            let vt = sp.vt(s);
            let i = sp.usize(b, n - 1, false);
            let x = b.op(Op::Vec(VecOp::Load), vt, &[slot.into(), i]);
            sp.out(b, x, s);
            sp.label(b, &format!("vec.store {} array[{}] i 1", s.name(), 2 * n), None);
            let i = sp.usize(b, 1, true);
            b.inst(Op::Vec(VecOp::Store), None, &[slot.into(), i, x.into()]);
            let au = b.cast(CastKind::Exact, t.usize, slot);
            let nb = b.int((2 * n * w) as i128);
            sp.dump(b, au.into(), nb);
            sp.unmap(b, base, 2);
        },
    )
}

/// A slice that ends where the next page is unmapped: full accesses at the last valid index,
/// masked accesses whose inactive lanes lie past the end.
fn flush(s: Shape) -> Case {
    let n = s.n as u64;
    let w = s.lane.bytes() as u64;
    Case::new(
        &[VecOp::Load, VecOp::Store, VecOp::LoadMasked, VecOp::StoreMasked],
        s.bits(),
        600,
        move |sp, b| {
            let base = sp.region(b, 1);
            let all = b.int(4096);
            sp.fill(b, base.into(), all, 3 + w);
            let len = 2 * n + 1;
            let sl = sp.mk_slice(b, base, 4096 - len * w, len, s, None);
            let i = len - n;
            sp.label(b, &format!("vec.load {} at the page end i {i}", s.name()), None);
            let io = sp.usize(b, i, false);
            sp.guard(b, &sl, io);
            let x = sp.vload(b, VecOp::Load, &sl, io, None);
            sp.out(b, x, s);
            sp.label(b, &format!("vec.store {} at the page end i {i}", s.name()), None);
            let y = sp.vec(b, s, &lanes(s, 4));
            sp.guard(b, &sl, io);
            sp.vstore(b, VecOp::Store, &sl, io, y, None);
            sp.dump_elems(b, &sl, 0, len);
            let mut tail: Vec<(u64, u64)> = (1..n).map(|k| (i + k, all_mask((n - k) as u32))).collect();
            tail.push((len - 1, 1));
            tail.push((len, 0));
            tail.push((len + 5, 0));
            tail.push((u64::MAX - 3, 0));
            tail.push((i + 1, all_mask(n as u32 - 1) & 0x5555_5555_5555_5555));
            for (k, &(i, m)) in tail.iter().enumerate() {
                sp.label(b, &format!("vec.load_masked {} at the page end i {i} mask {m:x}", s.name()), None);
                let io = sp.usize(b, i, k % 2 == 0);
                let x = sp.vload(b, VecOp::LoadMasked, &sl, io, Some(m));
                sp.out(b, x, s);
                sp.label(b, &format!("vec.store_masked {} at the page end i {i} mask {m:x}", s.name()), None);
                let y = sp.vec(b, s, &lanes(s, k as u64 + 1));
                sp.vstore(b, VecOp::StoreMasked, &sl, io, y, Some(m));
                sp.dump_elems(b, &sl, len as i64 - n as i64 - 1, len);
            }
            sp.unmap(b, base, 2);
        },
    )
}

/// `load_aligned` / `store_aligned` on a slice declared aligned to the vector size.
fn aligned(s: Shape) -> Case {
    let n = s.n as u64;
    let a = s.bytes();
    Case::new(&[VecOp::LoadAligned, VecOp::StoreAligned], s.bits(), 200, move |sp, b| {
        let base = sp.region(b, 1);
        let all = b.int(4096);
        sp.fill(b, base.into(), all, 11);
        let len = 4 * n;
        let sl = sp.mk_slice(b, base, 4096 - len * s.lane.bytes() as u64, len, s, Some(a));
        for (k, i) in [0, n, 3 * n].into_iter().enumerate() {
            sp.label(b, &format!("vec.load_aligned {} i {i}", s.name()), None);
            let io = sp.usize(b, i, k == 1);
            sp.guard(b, &sl, io);
            let x = sp.vload(b, VecOp::LoadAligned, &sl, io, None);
            sp.out(b, x, s);
        }
        for (k, i) in [2 * n, 3 * n].into_iter().enumerate() {
            sp.label(b, &format!("vec.store_aligned {} i {i}", s.name()), None);
            let y = sp.vec(b, s, &lanes(s, k as u64));
            let io = sp.usize(b, i, k == 0);
            sp.guard(b, &sl, io);
            sp.vstore(b, VecOp::StoreAligned, &sl, io, y, None);
            sp.dump_elems(b, &sl, 0, len);
        }
        sp.unmap(b, base, 2);
    })
}

/// The data slice length a gather or scatter with index lanes `il` runs on: past 255 for 8-bit
/// indices, past 65535 for wider ones (so indices with the top bit of their lane set are valid).
fn gather_len(il: Lane) -> u64 {
    if il.bits() == 8 { 300 } else { 0x1_0010 }
}

fn gather_pool(il: Lane, len: u64) -> Vec<u64> {
    let mut v =
        vec![0, 1, 5, 5, 200, 128, 129, 255, len - 1, 37, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff, 0x1_0000];
    v.retain(|&x| x < len && x <= il.ones());
    v
}

fn pages(bytes: u64) -> u64 {
    bytes.div_ceil(4096) + 1
}

/// Unsigned index lanes of `n` lanes that fit in 256 bits.
fn index_lanes(n: u32) -> Vec<Lane> {
    [Lane::U8, Lane::U16, Lane::U32, Lane::U64].into_iter().filter(|l| l.bits() * n <= 256).collect()
}

/// Gathers and scatters with every index lane width, duplicate scatter indices.
fn gather_scatter(s: Shape, il: Lane) -> Case {
    let is = sh(il, s.n);
    let len = gather_len(il);
    let w = s.lane.bytes() as u64;
    let pg = pages(len * w + 8);
    let bits = s.bits().max(is.bits());
    Case::new(&[VecOp::Gather, VecOp::Scatter], bits, 600, move |sp, b| {
        let base = sp.region(b, pg);
        let all = b.int(4096);
        sp.fill(b, base.into(), all, 5);
        let pool = gather_pool(il, len);
        for off in [0, 1] {
            let sl = sp.mk_slice(b, base, off, len, s, None);
            for (j, &k) in pool.iter().enumerate() {
                let x = lanes(s, j as u64)[j % s.n as usize];
                sp.poke(b, sl.addr, s.lane, k, x);
            }
            let vt = sp.vt(s);
            for idx in spread(&pool, s.n as usize) {
                sp.label(b, &format!("vec.gather {} off {off} idx {} {idx:?}", s.name(), is.name()), None);
                let iv = sp.vec(b, is, &idx);
                let x = b.op(Op::Vec(VecOp::Gather), vt, &[sl.v.into(), iv.into()]);
                sp.out(b, x, s);
            }
            let n = s.n as usize;
            let mut scat: Vec<Vec<u64>> = vec![vec![3; n], (0..n).map(|j| pool[j % pool.len()]).collect()];
            scat.push((0..n).map(|j| if j % 2 == 0 { 7 } else { len - 1 }).collect());
            scat.push((0..n).rev().map(|j| (j as u64 * 3) % len).collect());
            for (k, idx) in scat.into_iter().enumerate() {
                sp.label(b, &format!("vec.scatter {} off {off} idx {} {idx:?}", s.name(), is.name()), None);
                let iv = sp.vec(b, is, &idx);
                let x = sp.vec(b, s, &lanes(s, k as u64 + 2));
                b.inst(Op::Vec(VecOp::Scatter), None, &[sl.v.into(), iv.into(), x.into()]);
                let mut seen = idx.clone();
                seen.sort();
                seen.dedup();
                for e in seen {
                    sp.dump_elems(b, &sl, e as i64 - 1, e + 2);
                }
            }
        }
        sp.unmap(b, base, pg + 1);
    })
}

/// Gathers and scatters on a slice whose only mapped elements are the `NEAR` ones from index
/// `far` on: indices with the top bit of their low 32 bits set, or past 2^32.
fn gather_far(s: Shape, il: Lane, far: u64) -> Case {
    const NEAR: u64 = 40;
    let is = sh(il, s.n);
    let w = s.lane.bytes() as u64;
    let bits = s.bits().max(is.bits());
    Case::new(&[VecOp::Gather, VecOp::Scatter], bits, 300, move |sp, b| {
        let base = sp.region(b, 1);
        let all = b.int(4096);
        sp.fill(b, base.into(), all, 7);
        let sl = sp.mk_slice(b, base, (far * w).wrapping_neg(), far + NEAR, s, None);
        let pool: Vec<u64> = [0, 1, 19, 20, 21, 39, 5, 5, 30, 17].iter().map(|k| far + k).collect();
        let vt = sp.vt(s);
        for idx in spread(&pool, s.n as usize) {
            sp.label(b, &format!("vec.gather {} far idx {} {idx:x?}", s.name(), is.name()), None);
            let iv = sp.vec(b, is, &idx);
            let x = b.op(Op::Vec(VecOp::Gather), vt, &[sl.v.into(), iv.into()]);
            sp.out(b, x, s);
        }
        let idx: Vec<u64> = (0..s.n as u64).rev().map(|j| far + (j * 7) % NEAR).collect();
        sp.label(b, &format!("vec.scatter {} far idx {} {idx:x?}", s.name(), is.name()), None);
        let iv = sp.vec(b, is, &idx);
        let x = sp.vec(b, s, &lanes(s, 6));
        b.inst(Op::Vec(VecOp::Scatter), None, &[sl.v.into(), iv.into(), x.into()]);
        let n = b.int((NEAR * w) as i128);
        sp.dump(b, base.into(), n);
        sp.unmap(b, base, 2);
    })
}

/// The slice starts of [`gather_far`] for index lanes `il`: its elements straddle index 2^31,
/// end at the highest index of the lane, or lie past 2^32.
fn far_starts(il: Lane) -> Vec<u64> {
    match il.bits() {
        32 => vec![(1 << 31) - 20, (1 << 32) - 40],
        64 => vec![(1 << 31) - 20, (5 << 31) - 20],
        _ => vec![],
    }
}

/// A raw pointer to an array `[L; len]` at `addr`.
fn array_ptr(sp: &mut Sp, b: &mut FuncBuilder, addr: Value, l: Lane, len: u64) -> Value {
    let e = sp.lane_t(l);
    let at = sp.p.m.types.intern(TypeKey::Array { elem: e, len });
    let pt = sp.p.m.types.ptr(false, at);
    b.cast(CastKind::Exact, pt, addr)
}

/// Gathers from an array behind a raw pointer: the length is a constant.
fn gather_array(s: Shape, il: Lane) -> Case {
    let is = sh(il, s.n);
    let len = gather_len(il);
    let pg = pages(len * s.lane.bytes() as u64 + 8);
    let bits = s.bits().max(is.bits());
    Case::new(&[VecOp::Gather], bits, 300, move |sp, b| {
        let base = sp.region(b, pg);
        let all = b.int(4096);
        sp.fill(b, base.into(), all, 13);
        let addr = sp.addr(b, base, 1);
        let pool = gather_pool(il, len);
        for (j, &k) in pool.iter().enumerate() {
            let x = lanes(s, j as u64 + 1)[j % s.n as usize];
            sp.poke(b, addr, s.lane, k, x);
        }
        let arr = array_ptr(sp, b, addr, s.lane, len);
        let vt = sp.vt(s);
        for idx in spread(&pool, s.n as usize) {
            sp.label(b, &format!("vec.gather {} array idx {} {idx:?}", s.name(), is.name()), None);
            let iv = sp.vec(b, is, &idx);
            let x = b.op(Op::Vec(VecOp::Gather), vt, &[arr.into(), iv.into()]);
            sp.out(b, x, s);
        }
        sp.unmap(b, base, pg + 1);
    })
}

pub fn memory() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::ALL {
        for s in shapes(l) {
            out.push(plain(s));
            out.push(flush(s));
            out.push(aligned(s));
            for il in index_lanes(s.n) {
                out.push(gather_scatter(s, il));
                out.push(gather_array(s, il));
                for far in far_starts(il) {
                    out.push(gather_far(s, il, far));
                }
            }
        }
    }
    out
}

fn trap(ops: &[VecOp], bits: u32, k: PanicKind, emit: impl Fn(&mut Sp, &mut FuncBuilder) + 'static) -> Case {
    Case::new(ops, bits, 100, emit).traps(k)
}

/// Bounds traps of the aligned, masked, gather and scatter forms.
pub fn traps() -> Vec<Case> {
    let mut out = Vec::new();
    for l in Lane::ALL {
        let mut ss = full_shapes(l);
        ss.push(sh(l, 2));
        for s in ss {
            let n = s.n as u64;
            let w = s.lane.bytes() as u64;
            for op in [VecOp::LoadAligned, VecOp::StoreAligned] {
                for i in [1, n + 1] {
                    out.push(trap(&[op], s.bits(), PanicKind::Bounds, move |sp, b| {
                        let base = sp.region(b, 1);
                        let sl = sp.mk_slice(b, base, 0, 4 * n, s, Some(s.bytes()));
                        sp.label(b, &format!("{} {} i {i}", op.name(), s.name()), None);
                        let io = sp.usize(b, i, false);
                        sp.guard(b, &sl, io);
                        if op == VecOp::LoadAligned {
                            let x = sp.vload(b, op, &sl, io, None);
                            sp.out(b, x, s);
                        } else {
                            let y = sp.vec(b, s, &lanes(s, 1));
                            sp.vstore(b, op, &sl, io, y, None);
                            sp.dump_elems(b, &sl, 0, 4 * n);
                        }
                    }));
                }
            }
            // an active lane past the end: all lanes, or only the last one
            let len = 2 * n + 1;
            for op in [VecOp::LoadMasked, VecOp::StoreMasked] {
                for m in [all_mask(n as u32), 1 << (n - 1)] {
                    out.push(trap(&[op], s.bits(), PanicKind::Bounds, move |sp, b| {
                        let base = sp.region(b, 1);
                        let sl = sp.mk_slice(b, base, 4096 - len * w, len, s, None);
                        let i = len - n + 1;
                        sp.label(b, &format!("{} {} i {i} mask {m:x}", op.name(), s.name()), None);
                        let io = sp.usize(b, i, false);
                        if op == VecOp::LoadMasked {
                            let x = sp.vload(b, op, &sl, io, Some(m));
                            sp.out(b, x, s);
                        } else {
                            let y = sp.vec(b, s, &lanes(s, 1));
                            sp.vstore(b, op, &sl, io, y, Some(m));
                            sp.dump_elems(b, &sl, 0, len);
                        }
                    }));
                }
            }
            for il in index_lanes(s.n) {
                let is = sh(il, s.n);
                let bits = s.bits().max(is.bits());
                let glen = 40u64;
                let mut bad = vec![glen, il.ones()];
                if il.bits() >= 32 {
                    bad.push(1 << (il.bits() - 1));
                }
                for (k, &x) in bad.iter().enumerate() {
                    let pos = if k % 2 == 0 { n as usize - 1 } else { 0 };
                    let mut idx: Vec<u64> = (0..n).map(|j| j % glen).collect();
                    idx[pos] = x;
                    let gi = idx.clone();
                    let ai = idx.clone();
                    out.push(trap(&[VecOp::Gather], bits, PanicKind::Bounds, move |sp, b| {
                        let base = sp.region(b, 1);
                        let arr = array_ptr(sp, b, base, s.lane, glen);
                        sp.label(b, &format!("vec.gather {} array idx {} {ai:?}", s.name(), is.name()), None);
                        let iv = sp.vec(b, is, &ai);
                        let vt = sp.vt(s);
                        let x = b.op(Op::Vec(VecOp::Gather), vt, &[arr.into(), iv.into()]);
                        sp.out(b, x, s);
                    }));
                    out.push(trap(&[VecOp::Gather], bits, PanicKind::Bounds, move |sp, b| {
                        let base = sp.region(b, 1);
                        let sl = sp.mk_slice(b, base, 0, glen, s, None);
                        sp.label(b, &format!("vec.gather {} idx {} {gi:?}", s.name(), is.name()), None);
                        let iv = sp.vec(b, is, &gi);
                        let vt = sp.vt(s);
                        let x = b.op(Op::Vec(VecOp::Gather), vt, &[sl.v.into(), iv.into()]);
                        sp.out(b, x, s);
                    }));
                    // scatter: the in-range lanes must not be written before the trap
                    let mut c = trap(&[VecOp::Scatter], bits, PanicKind::Bounds, move |sp, b| {
                        let base = sp.region(b, 1);
                        let all = b.int(4096);
                        sp.fill(b, base.into(), all, 9);
                        let sl = sp.mk_slice(b, base, 0, glen, s, None);
                        sp.label(b, &format!("vec.scatter {} idx {} {idx:?}", s.name(), is.name()), None);
                        let iv = sp.vec(b, is, &idx);
                        let y = sp.vec(b, s, &lanes(s, 3));
                        let nb = b.int((n.min(glen) * w) as i128);
                        sp.set_dump(b, sl.addr.into(), nb);
                        sp.label(b, "dump", None);
                        sp.dump_elems(b, &sl, 0, n.min(glen));
                        b.inst(Op::Vec(VecOp::Scatter), None, &[sl.v.into(), iv.into(), y.into()]);
                        sp.dump_elems(b, &sl, 0, n.min(glen));
                    });
                    c.trap_dump = true;
                    out.push(c);
                }
            }
        }
    }
    out
}
