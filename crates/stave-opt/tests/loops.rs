//! Loop bookkeeping in release IR: a chunk `while` loop keeps no bounds check, computes its step
//! once, and reads its slice's address once, outside the loop; only dominated repeats of an op are
//! shared. Results and traps stay those of the unoptimized IR in the reference interpreter.

use stave_ir::bounds;
use stave_ir::{
    Body, BufferHost, Config, FuncBuilder, FuncId, InstanceId, Interp, Lanes, Module, Op, Operand, Outcome,
    PanicKind, SyscallAbi, TypeKey, Val, VecOp, verify_module,
};
use stave_opt::layout;

fn run(m: &Module, f: FuncId, args: Vec<Val>) -> Outcome {
    let mut it = Interp::new(m, BufferHost::new(SyscallAbi::X86_64), Config::default()).unwrap();
    it.run(f, args).unwrap()
}

fn optimized(m: &Module) -> Module {
    let mut c = m.clone();
    bounds::elide(&mut c).unwrap_or_else(|r| panic!("witness refused: {r:?}"));
    verify_module(&c).unwrap();
    let before =
        element_accesses(&bounds::audit(&c).unwrap_or_else(|u| panic!("unguarded after removal: {u:?}")));
    stave_opt::slices::declare_elem_pointers(&mut c.types);
    let types = c.types.clone();
    for f in &mut c.funcs {
        if let Some(b) = &mut f.body {
            stave_opt::optimize(&types, b);
        }
    }
    verify_module(&c).unwrap_or_else(|e| panic!("optimized IR does not verify: {e:?}"));
    let after = bounds::audit(&c).unwrap_or_else(|u| panic!("unguarded after the release passes: {u:?}"));
    // the element pointer form the passes make (`slice.ptr` + the access) is audited as before
    assert_eq!(element_accesses(&after), before);
    c
}

/// The number of element accesses per kind, without the `slice.ptr`s.
fn element_accesses(acc: &[bounds::Access]) -> [usize; 4] {
    use bounds::AccessKind as K;
    [K::Elem, K::Get, K::Set, K::Vector].map(|k| acc.iter().filter(|a| a.kind == k).count())
}

fn count(b: &Body, op: Op) -> usize {
    b.blocks.iter().flat_map(|bd| bd.insts.indices()).filter(|&i| b.insts.op[i as usize] == op).count()
}

/// `t.double(s: &mut [u32]) -> u32`: `while i + 4 <= s.len() { v = u32x4.load(s, i) +% itself;
/// v.store(s, i); acc +%= v.reduce_add_wrap(); i = i + 4; }` as the front end lowers it (each
/// access with its own `slice.len`, `sub.sat`, and `check.bounds`; the increment its own trapping
/// `add`); and `t.main(k: usize) -> u32`, which calls it on the first `k` of eight elements
/// `3 j + 1`.
fn double() -> (Module, FuncId, FuncId) {
    let mut m = Module::new();
    let (u32_, usize_, bool_) = (m.types.u32(), m.types.usize(), m.types.bool());
    let v4 = m.types.intern(TypeKey::Vec { lane: u32_, n: Lanes::Fixed(4) });
    let arr = m.types.array(u32_, 8);
    let parr = m.types.ref_(true, arr);
    let pel = m.types.ref_(true, u32_);
    let sl = m.types.slice(u32_);
    let psl = m.types.ref_(true, sl);

    let f = m.declare(InstanceId(vec![1]), "t.double", vec![psl], Some(u32_));
    let mut b = FuncBuilder::new(&[psl]);
    let s = b.param(0).unwrap();
    let (head, hp) = b.create_block_with(&[usize_, u32_]);
    let (body, exit) = (b.create_block(), b.create_block());
    let (z, z2) = (b.int(0), b.int(0));
    b.br(head, &[z, z2]);
    b.switch_to(head);
    let four = b.int(4);
    let end = b.binary(Op::Add, usize_, hp[0], four);
    let n = b.op(Op::SliceLen, usize_, &[s.into()]);
    let c = b.cmp(Op::Le, bool_, end, n);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    let check = |b: &mut FuncBuilder| {
        let n = b.op(Op::SliceLen, usize_, &[s.into()]);
        let three = b.int(3);
        let lim = b.binary(Op::SubSat, usize_, n, three);
        b.bounds_check(hp[0], lim);
    };
    check(&mut b);
    let v = b.op(Op::Vec(VecOp::Load), v4, &[s.into(), hp[0].into()]);
    let w = b.op(Op::Vec(VecOp::AddWrap), v4, &[v.into(), v.into()]);
    check(&mut b);
    b.inst(Op::Vec(VecOp::Store), None, &[s.into(), hp[0].into(), w.into()]);
    let r = b.op(Op::Vec(VecOp::ReduceAddWrap), u32_, &[w.into()]);
    let acc = b.binary(Op::AddWrap, u32_, hp[1], r);
    let four = b.int(4);
    let i2 = b.binary(Op::Add, usize_, hp[0], four);
    b.br(head, &[i2.into(), acc.into()]);
    b.switch_to(exit);
    b.ret(Some(hp[1].into()));
    assert!(m.define(f, b.finish().unwrap()));

    let main = m.declare(InstanceId(vec![2]), "t.main", vec![usize_], Some(u32_));
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
    let s = b.op(Op::SliceMake, psl, &[base.into(), k.into()]);
    let r = b.call(f, Some(u32_), &[s.into()]).unwrap();
    b.ret(Some(r.into()));
    assert!(m.define(main, b.finish().unwrap()));
    verify_module(&m).unwrap();
    (m, f, main)
}

#[test]
fn chunk_while_loop_reads_its_slice_once_and_steps_once() {
    let (m, f, main) = double();
    let o = optimized(&m);
    let b = o.func(f).unwrap().body.as_ref().unwrap();
    // `i + 4 <= len` proves both accesses; the increment reuses the test's `i + 4`
    assert_eq!(count(b, Op::BoundsCheck), 0);
    assert_eq!(count(b, Op::Add), 1);
    assert_eq!(count(b, Op::SlicePtr), 1);
    let entry = b.blocks[0].insts;
    assert!(entry.indices().any(|i| b.insts.op[i as usize] == Op::SlicePtr), "slice.ptr is not in b0");
    for i in b.blocks.iter().flat_map(|bd| bd.insts.indices()) {
        if matches!(b.insts.op[i as usize], Op::Vec(VecOp::Load | VecOp::Store)) {
            let Operand::Value(p) = b.args(i)[0] else { panic!("vector memory operand") };
            let key = o.types.key(b.value_ty(p).unwrap());
            assert!(matches!(key, Some(TypeKey::RawPtr { .. })), "vector access on {key:?}");
        }
    }
    for k in 0..=8u64 {
        let whole: u64 = (0..k / 4 * 4).map(|j| 2 * (3 * j + 1)).sum();
        let want = run(&m, main, vec![Val::S(k)]);
        assert_eq!(want, Outcome::Return(Some(Val::S(whole))), "k = {k}");
        assert_eq!(run(&o, main, vec![Val::S(k)]), want, "k = {k}");
    }
}

/// `t.steps(n: u64, c: bool) -> u64`: `z = n + 1`; a loop `while i + 1 <= n` whose body steps
/// `i` by the test's `i + 1` when `c` (a repeat that only feeds the back edge), else by an
/// `i + 1` that an xor also reads; after the loop, `n + 1` again (outside every loop) feeds the
/// return block's param.
#[test]
fn only_loop_steps_that_repeat_a_dominating_op_are_shared_and_traps_stay() {
    let mut m = Module::new();
    let (u64_, bool_) = (m.types.u64(), m.types.bool());
    let f = m.declare(InstanceId(vec![1]), "t.steps", vec![u64_, bool_], Some(u64_));
    let mut b = FuncBuilder::new(&[u64_, bool_]);
    let (n, c) = (b.param(0).unwrap(), b.param(1).unwrap());
    let one = b.int(1);
    let z = b.binary(Op::Add, u64_, n, one);
    let (head, hp) = b.create_block_with(&[u64_, u64_]);
    let (body, by_test, by_own, exit) =
        (b.create_block(), b.create_block(), b.create_block(), b.create_block());
    let (out, op) = b.create_block_with(&[u64_]);
    let zero = b.int(0);
    b.br(head, &[zero, z.into()]);
    b.switch_to(head);
    let end = b.binary(Op::Add, u64_, hp[0], one);
    let go = b.cmp(Op::Le, bool_, end, n);
    b.cbr(go, body, &[], exit, &[]);
    b.switch_to(body);
    b.cbr(c, by_test, &[], by_own, &[]);
    b.switch_to(by_test);
    let i2 = b.binary(Op::Add, u64_, hp[0], one);
    let acc = b.binary(Op::Xor, u64_, hp[1], one);
    b.br(head, &[i2.into(), acc.into()]);
    b.switch_to(by_own);
    let i3 = b.binary(Op::Add, u64_, hp[0], one);
    let acc = b.binary(Op::Xor, u64_, hp[1], i3);
    b.br(head, &[i3.into(), acc.into()]);
    b.switch_to(exit);
    let z2 = b.binary(Op::Add, u64_, n, one);
    b.br(out, &[z2.into()]);
    b.switch_to(out);
    let r = b.binary(Op::Xor, u64_, op[0], hp[1]);
    b.ret(Some(r.into()));
    assert!(m.define(f, b.finish().unwrap()));
    verify_module(&m).unwrap();
    let o = optimized(&m);
    let ob = o.func(f).unwrap().body.as_ref().unwrap();
    assert_eq!(count(ob, Op::Add), 4, "only the step under `c` reuses the test's `i + 1`");
    for n in [0u64, 1, 5] {
        for c in [0u64, 1] {
            let want = run(&m, f, vec![Val::S(n), Val::S(c)]);
            assert!(matches!(want, Outcome::Return(_)), "{want:?}");
            assert_eq!(run(&o, f, vec![Val::S(n), Val::S(c)]), want, "n = {n}, c = {c}");
        }
    }
    let t = run(&o, f, vec![Val::S(u64::MAX), Val::S(1)]);
    assert!(matches!(t, Outcome::Trap(t) if t.kind == PanicKind::Overflow), "{t:?}");
}

/// `while i < n { if i != k { s = s +% i; } i = i +% 1; }` with the `if` joining at the latch
/// (`join`: the sum is a param of the latch), or the same loop whose `if` result is dropped (the
/// latch has no param).
fn counted(join: bool) -> (Body, [u32; 5]) {
    let mut m = Module::new();
    let (u64_, bool_) = (m.types.u64(), m.types.bool());
    let mut b = FuncBuilder::new(&[u64_, u64_]);
    let (n, k) = (b.param(0).unwrap(), b.param(1).unwrap());
    let (head, hp) = b.create_block_with(&[u64_, u64_]);
    let (body, add) = (b.create_block(), b.create_block());
    let (latch, lp) = if join { b.create_block_with(&[u64_]) } else { (b.create_block(), Vec::new()) };
    let exit = b.create_block();
    let z = b.int(0);
    b.br(head, &[z, z]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, bool_, hp[0], n);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    let ne = b.cmp(Op::Ne, bool_, hp[0], k);
    let skip: Vec<Operand> = if join { vec![hp[1].into()] } else { Vec::new() };
    b.cbr(ne, add, &[], latch, &skip);
    b.switch_to(add);
    let s2 = b.binary(Op::AddWrap, u64_, hp[1], hp[0]);
    let to: Vec<Operand> = if join { vec![s2.into()] } else { Vec::new() };
    b.br(latch, &to);
    b.switch_to(latch);
    let one = b.int(1);
    let i2 = b.binary(Op::AddWrap, u64_, hp[0], one);
    let s = if join { lp[0] } else { hp[1] };
    b.br(head, &[i2.into(), s.into()]);
    b.switch_to(exit);
    b.ret(Some(hp[1].into()));
    (b.finish().unwrap(), [head.0, body.0, add.0, latch.0, exit.0])
}

#[test]
fn loop_test_moves_after_the_body_unless_a_join_needs_the_registers() {
    let (b, [head, body, add, latch, exit]) = counted(false);
    assert_eq!(layout::order(&b), [0, head, body, add, latch, exit]);
    assert_eq!(layout::rotated_order(&b, |_| true), [0, body, add, latch, head, exit]);
    // the header's `i` is read in the body: without a register for it, no rotation
    assert_eq!(layout::rotated_order(&b, |_| false), [0, head, body, add, latch, exit]);
    let (b, [head, body, add, latch, exit]) = counted(true);
    assert_eq!(layout::rotated_order(&b, |_| true), [0, head, body, add, latch, exit]);
}
