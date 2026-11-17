//! The IR passes of a release build keep what a program computes: the same result and the same
//! trap in the reference interpreter before and after, with verified IR in between. Plus the
//! shapes the passes exist for: one length, one address, and one bound per vector width per
//! slice, read outside the loop.

use stave_ir::bounds;
use stave_ir::{
    Body, BufferHost, Config, FuncBuilder, FuncId, InstanceId, Interp, Module, Op, Operand, Outcome,
    PanicKind, SyscallAbi, TypeKey, Val, verify_module,
};
use stave_opt::layout;

/// `t.sum(s: &mut [u64], k: usize) -> u64`: the sum of the elements plus `s[k]` (a `Bounds` trap
/// for `k >= len`), every access with its own `slice.len` and `check.bounds`; and
/// `t.main(k: usize) -> u64`, which calls it on eight elements `3 j + 1`.
fn module() -> (Module, FuncId, FuncId) {
    let mut m = Module::new();
    let (u64_, usize_, bool_) = (m.types.u64(), m.types.usize(), m.types.bool());
    let arr = m.types.array(u64_, 8);
    let parr = m.types.ref_(true, arr);
    let pel = m.types.ref_(true, u64_);
    let sl = m.types.slice(u64_);
    let psl = m.types.ref_(true, sl);

    let sum = m.declare(InstanceId(vec![1]), "t.sum", vec![psl, usize_], Some(u64_));
    let mut b = FuncBuilder::new(&[psl, usize_]);
    let (s, k) = (b.param(0).unwrap(), b.param(1).unwrap());
    let n = b.op(Op::SliceLen, usize_, &[s.into()]);
    let (head, hp) = b.create_block_with(&[usize_, u64_]);
    let (body, exit) = (b.create_block(), b.create_block());
    let (z, z2) = (b.int(0), b.int(0));
    b.br(head, &[z, z2]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, bool_, hp[0], n);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    let n2 = b.op(Op::SliceLen, usize_, &[s.into()]);
    b.bounds_check(hp[0], n2);
    let p = b.op(Op::ElemAddr, pel, &[s.into(), hp[0].into()]);
    let x = b.load(u64_, p);
    let acc = b.binary(Op::AddWrap, u64_, hp[1], x);
    let one = b.int(1);
    let i2 = b.binary(Op::AddWrap, usize_, hp[0], one);
    b.br(head, &[i2.into(), acc.into()]);
    b.switch_to(exit);
    let n3 = b.op(Op::SliceLen, usize_, &[s.into()]);
    b.bounds_check(k, n3);
    let p = b.op(Op::ElemAddr, pel, &[s.into(), k.into()]);
    let x = b.load(u64_, p);
    let r = b.binary(Op::AddWrap, u64_, hp[1], x);
    // dead: a pure chain nobody reads
    let d1 = b.binary(Op::MulWrap, u64_, r, x);
    let _ = b.binary(Op::Xor, u64_, d1, x);
    b.ret(Some(r.into()));
    assert!(m.define(sum, b.finish().unwrap()));

    let main = m.declare(InstanceId(vec![2]), "t.main", vec![usize_], Some(u64_));
    let mut b = FuncBuilder::new(&[usize_]);
    let k = b.param(0).unwrap();
    let a = b.slot(parr);
    for j in 0..8 {
        let idx = b.int(j);
        let q = b.op(Op::ElemAddr, pel, &[a.into(), idx]);
        let v = b.int(3 * j + 1);
        b.store(q, v);
    }
    let zero = b.int(0);
    let base = b.op(Op::ElemAddr, pel, &[a.into(), zero]);
    let len = b.int(8);
    let s = b.op(Op::SliceMake, psl, &[base.into(), len]);
    let r = b.call(sum, Some(u64_), &[s.into(), k.into()]).unwrap();
    b.ret(Some(r.into()));
    assert!(m.define(main, b.finish().unwrap()));
    verify_module(&m).unwrap();
    (m, sum, main)
}

fn run(m: &Module, f: FuncId, k: u64) -> Outcome {
    let mut it = Interp::new(m, BufferHost::new(SyscallAbi::X86_64), Config::default()).unwrap();
    it.run(f, vec![Val::S(k)]).unwrap()
}

fn optimized(m: &Module) -> Module {
    let mut c = m.clone();
    for f in &mut c.funcs {
        if let Some(b) = &mut f.body {
            stave_opt::before_elision(b);
        }
    }
    verify_module(&c).unwrap();
    bounds::elide(&mut c).unwrap_or_else(|r| panic!("witness refused: {r:?}"));
    stave_opt::slices::declare_elem_pointers(&mut c.types);
    let types = c.types.clone();
    for f in &mut c.funcs {
        if let Some(b) = &mut f.body {
            stave_opt::after_elision(&types, b);
        }
    }
    verify_module(&c).unwrap_or_else(|e| panic!("optimized IR does not verify: {e:?}"));
    c
}

fn count(b: &Body, op: Op) -> usize {
    b.blocks.iter().flat_map(|bd| bd.insts.indices()).filter(|&i| b.insts.op[i as usize] == op).count()
}

#[test]
fn same_results_and_traps() {
    let (m, _, main) = module();
    let o = optimized(&m);
    for k in [0u64, 3, 7] {
        let want = run(&m, main, k);
        assert_eq!(want, Outcome::Return(Some(Val::S(92 + 3 * k + 1))));
        assert_eq!(run(&o, main, k), want, "k = {k}");
    }
    for k in [8u64, 9, u64::MAX] {
        let (want, got) = (run(&m, main, k), run(&o, main, k));
        assert!(matches!(want, Outcome::Trap(t) if t.kind == PanicKind::Bounds), "{want:?}");
        assert_eq!(got, want, "k = {k}");
    }
}

/// `t.windows(s: &mut [u64], k: usize) -> u64`: the sum of `s[i]` for `i < k`, where every
/// iteration checks `i` against `len -| 3` twice and against `len -| 1` once, each check with
/// its own `slice.len` and `sub.sat`; and `t.main(k: usize) -> u64`, which calls it on eight
/// elements `3 j + 1`.
fn windows() -> (Module, FuncId, FuncId) {
    let mut m = Module::new();
    let (u64_, usize_, bool_) = (m.types.u64(), m.types.usize(), m.types.bool());
    let arr = m.types.array(u64_, 8);
    let parr = m.types.ref_(true, arr);
    let pel = m.types.ref_(true, u64_);
    let sl = m.types.slice(u64_);
    let psl = m.types.ref_(true, sl);

    let f = m.declare(InstanceId(vec![1]), "t.windows", vec![psl, usize_], Some(u64_));
    let mut b = FuncBuilder::new(&[psl, usize_]);
    let (s, k) = (b.param(0).unwrap(), b.param(1).unwrap());
    let (head, hp) = b.create_block_with(&[usize_, u64_]);
    let (body, exit) = (b.create_block(), b.create_block());
    let (z, z2) = (b.int(0), b.int(0));
    b.br(head, &[z, z2]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, bool_, hp[0], k);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    for lanes in [3, 3, 1] {
        let n = b.op(Op::SliceLen, usize_, &[s.into()]);
        let w = b.int(lanes);
        let lim = b.binary(Op::SubSat, usize_, n, w);
        b.bounds_check(hp[0], lim);
    }
    let p = b.op(Op::ElemAddr, pel, &[s.into(), hp[0].into()]);
    let x = b.load(u64_, p);
    let acc = b.binary(Op::AddWrap, u64_, hp[1], x);
    let one = b.int(1);
    let i2 = b.binary(Op::AddWrap, usize_, hp[0], one);
    b.br(head, &[i2.into(), acc.into()]);
    b.switch_to(exit);
    b.ret(Some(hp[1].into()));
    assert!(m.define(f, b.finish().unwrap()));

    let main = m.declare(InstanceId(vec![2]), "t.main", vec![usize_], Some(u64_));
    let mut b = FuncBuilder::new(&[usize_]);
    let k = b.param(0).unwrap();
    let a = b.slot(parr);
    for j in 0..8 {
        let idx = b.int(j);
        let q = b.op(Op::ElemAddr, pel, &[a.into(), idx]);
        let v = b.int(3 * j + 1);
        b.store(q, v);
    }
    let zero = b.int(0);
    let base = b.op(Op::ElemAddr, pel, &[a.into(), zero]);
    let len = b.int(8);
    let s = b.op(Op::SliceMake, psl, &[base.into(), len]);
    let r = b.call(f, Some(u64_), &[s.into(), k.into()]).unwrap();
    b.ret(Some(r.into()));
    assert!(m.define(main, b.finish().unwrap()));
    verify_module(&m).unwrap();
    (m, f, main)
}

#[test]
fn one_bound_per_slice_and_width() {
    let (m, f, main) = windows();
    let before = m.func(f).unwrap().body.as_ref().unwrap();
    assert_eq!(count(before, Op::SubSat), 3);
    let o = optimized(&m);
    let b = o.func(f).unwrap().body.as_ref().unwrap();
    assert_eq!(count(b, Op::SubSat), 2);
    assert_eq!(count(b, Op::SliceLen), 1);
    assert_eq!(count(b, Op::BoundsCheck), 3);
    let entry = b.blocks[0].insts;
    let in_entry = entry.indices().filter(|&i| b.insts.op[i as usize] == Op::SubSat).count();
    assert_eq!(in_entry, 2, "both bounds are computed where the slice is defined");
    for k in [0u64, 1, 5] {
        let want = run(&m, main, k);
        assert_eq!(want, Outcome::Return(Some(Val::S(3 * k * k.saturating_sub(1) / 2 + k))));
        assert_eq!(run(&o, main, k), want, "k = {k}");
    }
    for k in [6u64, 8, u64::MAX] {
        let (want, got) = (run(&m, main, k), run(&o, main, k));
        assert!(matches!(want, Outcome::Trap(t) if t.kind == PanicKind::Bounds), "{want:?}");
        assert_eq!(got, want, "k = {k}");
    }
}

#[test]
fn one_length_and_one_address_per_slice() {
    let (m, sum, _) = module();
    let before = m.func(sum).unwrap().body.as_ref().unwrap();
    assert_eq!(count(before, Op::SliceLen), 3);
    assert_eq!(count(before, Op::BoundsCheck), 2);
    let o = optimized(&m);
    let b = o.func(sum).unwrap().body.as_ref().unwrap();
    // the loop's check is proven by the loop bound once both name the same length
    assert_eq!(count(b, Op::BoundsCheck), 1);
    assert_eq!(count(b, Op::SliceLen), 1);
    assert_eq!(count(b, Op::SlicePtr), 1);
    // both parts sit in the entry block, where the slice is defined
    let entry = b.blocks[0].insts;
    for op in [Op::SliceLen, Op::SlicePtr] {
        assert!(entry.indices().any(|i| b.insts.op[i as usize] == op), "{op:?} is not in b0");
    }
    // every element address is on the element pointer
    for i in b.blocks.iter().flat_map(|bd| bd.insts.indices()) {
        if b.insts.op[i as usize] == Op::ElemAddr {
            let Operand::Value(p) = b.args(i)[0] else { panic!("addr.elem base") };
            let key = o.types.key(b.value_ty(p).unwrap());
            assert!(matches!(key, Some(TypeKey::RawPtr { .. })), "addr.elem on {key:?}");
        }
    }
    // the dead chain is gone
    assert_eq!(count(b, Op::MulWrap), 0);
    assert_eq!(count(b, Op::Xor), 0);
}

#[test]
fn loop_blocks_follow_the_header_and_cold_blocks_come_last() {
    let mut m = Module::new();
    let (u64_, bool_) = (m.types.u64(), m.types.bool());
    let mut b = FuncBuilder::new(&[u64_]);
    let n = b.param(0).unwrap();
    let (head, hp) = b.create_block_with(&[u64_]);
    let (body, exit, fail, back) = (b.create_block(), b.create_block(), b.create_block(), b.create_block());
    let z = b.int(0);
    b.br(head, &[z]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, bool_, hp[0], n);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    let k = b.int(1000);
    let bad = b.cmp(Op::Gt, bool_, hp[0], k);
    b.cbr(bad, fail, &[], back, &[]);
    b.switch_to(fail);
    b.panic(PanicKind::Assert);
    b.unreachable();
    b.switch_to(back);
    let one = b.int(1);
    let i2 = b.binary(Op::AddWrap, u64_, hp[0], one);
    b.br(head, &[i2.into()]);
    b.switch_to(exit);
    b.ret(Some(hp[0].into()));
    let body_ir = b.finish().unwrap();
    let order = layout::order(&body_ir);
    assert_eq!(order, [0, head.0, body.0, back.0, exit.0, fail.0]);
    let cold = layout::cold_blocks(&body_ir, &order);
    assert_eq!(cold, [false, false, false, false, true, false]);
}

/// `t.hints(s: &mut [u64], x: u64) -> u64`: prefetches `s` at `len + 64` (an address nothing
/// else reads), streams `x` to `s[0]`, fences, and returns `s[0]`. The passes keep the prefetch,
/// the arithmetic only it uses, the streaming store, and the fence.
#[test]
fn memory_hints_survive_the_passes() {
    let mut m = Module::new();
    let (u64_, usize_) = (m.types.u64(), m.types.usize());
    let arr = m.types.array(u64_, 4);
    let parr = m.types.ref_(true, arr);
    let pel = m.types.ref_(true, u64_);
    let pconst = m.types.ptr(false, u64_);
    let sl = m.types.slice(u64_);
    let psl = m.types.ref_(true, sl);
    let hint = stave_ir::Prefetch::from_method("prefetch_read_l1").unwrap();

    let hints = m.declare(InstanceId(vec![1]), "t.hints", vec![psl, u64_], Some(u64_));
    let mut b = FuncBuilder::new(&[psl, u64_]);
    let (s, x) = (b.param(0).unwrap(), b.param(1).unwrap());
    let n = b.op(Op::SliceLen, usize_, &[s.into()]);
    let k = b.int(64);
    let far = b.binary(Op::AddWrap, usize_, n, k);
    let p = b.op(Op::SlicePtr, pconst, &[s.into()]);
    b.inst(Op::Prefetch(hint), None, &[p.into(), far.into()]);
    let z = b.int(0);
    b.bounds_check(z, n);
    let e = b.op(Op::ElemAddr, pel, &[s.into(), z]);
    b.inst(Op::StoreStream, None, &[e.into(), x.into()]);
    b.inst(Op::FenceStream, None, &[]);
    let y = b.load(u64_, e);
    b.ret(Some(y.into()));
    assert!(m.define(hints, b.finish().unwrap()));

    let main = m.declare(InstanceId(vec![2]), "t.main", vec![u64_], Some(u64_));
    let mut b = FuncBuilder::new(&[u64_]);
    let x = b.param(0).unwrap();
    let a = b.slot(parr);
    let zero = b.int(0);
    let base = b.op(Op::ElemAddr, pel, &[a.into(), zero]);
    let len = b.int(4);
    let s = b.op(Op::SliceMake, psl, &[base.into(), len]);
    let r = b.call(hints, Some(u64_), &[s.into(), x.into()]).unwrap();
    b.ret(Some(r.into()));
    assert!(m.define(main, b.finish().unwrap()));
    verify_module(&m).unwrap();

    let o = optimized(&m);
    let body = o.func(hints).unwrap().body.as_ref().unwrap();
    assert_eq!(count(body, Op::Prefetch(hint)), 1);
    assert_eq!(count(body, Op::AddWrap), 1);
    assert_eq!(count(body, Op::StoreStream), 1);
    assert_eq!(count(body, Op::FenceStream), 1);
    for v in [0u64, 41, u64::MAX] {
        assert_eq!(run(&m, main, v), Outcome::Return(Some(Val::S(v))));
        assert_eq!(run(&o, main, v), Outcome::Return(Some(Val::S(v))));
    }
}
