//! Register assignment: hand-written shapes, and random control flow checked against an
//! independent liveness (iterative dataflow): values that share a register are never live in
//! the same block, and a value in a scratch register never lives across a clobber. Floats and
//! vectors share the float registers and differ in which of them survive a clobber.

use stave_ir::ir::Target;
use stave_ir::{Block, Body, FuncBuilder, IntrinsicId, Op, Operand, Term, TypeId, TypeTable, Value};
use stave_opt::regalloc::{self, Assignment, CLASS_FLOAT, CLASS_INT, CLASS_NONE, CLASS_VEC, NO_REG, Pools};

struct Tys {
    u64: TypeId,
    f64: TypeId,
    bool: TypeId,
}

fn tys() -> Tys {
    let mut t = TypeTable::new();
    Tys { u64: t.u64(), f64: t.f64(), bool: t.bool() }
}

/// `vecs`: every `f64` value with an odd number stands for a vector (the allocator sees
/// classes only).
fn classes(b: &Body, t: &Tys, vecs: bool) -> Vec<u8> {
    b.values
        .iter()
        .enumerate()
        .map(|(k, v)| {
            if v.ty == t.f64 {
                if vecs && k % 2 == 1 { CLASS_VEC } else { CLASS_FLOAT }
            } else if v.ty == t.u64 || v.ty == t.bool {
                CLASS_INT
            } else {
                CLASS_NONE
            }
        })
        .collect()
}

fn clobbers(b: &Body) -> Vec<bool> {
    b.insts.op.iter().map(|op| matches!(op, Op::Intrinsic(_) | Op::Call | Op::CallInd)).collect()
}

fn targets(b: &Body, t: &Term) -> Vec<Target> {
    match t {
        Term::Br(t) => vec![*t],
        Term::Cbr { then, els, .. } => vec![*then, *els],
        Term::Switch { cases, default, .. } => {
            let mut v: Vec<Target> = b.switch_cases(*cases).iter().map(|c| c.target).collect();
            v.push(*default);
            v
        }
        Term::Ret(_) | Term::Unreachable => Vec::new(),
    }
}

/// Reverse postorder of the reachable blocks.
fn rpo(b: &Body) -> Vec<u32> {
    fn visit(b: &Body, x: usize, seen: &mut [bool], post: &mut Vec<u32>) {
        seen[x] = true;
        for t in targets(b, &b.blocks[x].term) {
            if !seen[t.block.index()] {
                visit(b, t.block.index(), seen, post);
            }
        }
        post.push(x as u32);
    }
    let mut seen = vec![false; b.blocks.len()];
    let mut post = Vec::new();
    visit(b, 0, &mut seen, &mut post);
    post.reverse();
    post
}

fn uses(b: &Body, blk: usize) -> Vec<Value> {
    let bd = &b.blocks[blk];
    let mut v = Vec::new();
    let mut add = |o: Operand| {
        if let Operand::Value(x) = o {
            v.push(x);
        }
    };
    for i in bd.insts.indices() {
        b.args(i).iter().for_each(|o| add(*o));
    }
    match bd.term {
        Term::Cbr { cond, .. } => add(cond),
        Term::Switch { scrut, .. } => add(scrut),
        Term::Ret(Some(o)) => add(o),
        _ => {}
    }
    for t in targets(b, &bd.term) {
        b.target_args(&t).iter().for_each(|o| add(*o));
    }
    v
}

fn defs(b: &Body, blk: usize) -> Vec<Value> {
    let bd = &b.blocks[blk];
    let mut v: Vec<Value> = b.params(Block(blk as u32)).to_vec();
    v.extend(bd.insts.indices().filter_map(|i| b.insts.result[i as usize]));
    v
}

/// Per value: the blocks it occupies (its defining block and every block it is live into).
fn occupied(b: &Body, order: &[u32]) -> Vec<Vec<bool>> {
    let (nb, nv) = (b.blocks.len(), b.values.len());
    let mut live_in = vec![vec![false; nv]; nb];
    let mut changed = true;
    while changed {
        changed = false;
        for &blk in order.iter().rev() {
            let blk = blk as usize;
            let mut live = vec![false; nv];
            for t in targets(b, &b.blocks[blk].term) {
                let s = t.block.index();
                for v in 0..nv {
                    live[v] |= live_in[s][v] && !b.params(t.block).contains(&Value(v as u32));
                }
            }
            for d in defs(b, blk) {
                live[d.index()] = false;
            }
            let ds = defs(b, blk);
            for u in uses(b, blk) {
                if !ds.contains(&u) {
                    live[u.index()] = true;
                }
            }
            if live != live_in[blk] {
                live_in[blk] = live;
                changed = true;
            }
        }
    }
    let mut occ = vec![vec![false; nb]; nv];
    for &blk in order {
        let blk = blk as usize;
        for d in defs(b, blk) {
            occ[d.index()][blk] = true;
        }
        for v in 0..nv {
            if live_in[blk][v] {
                occ[v][blk] = true;
            }
        }
    }
    occ
}

fn check(b: &Body, t: &Tys, pools: Pools) -> Assignment {
    let class = classes(b, t, false);
    check_classes(b, &class, pools)
}

fn check_classes(b: &Body, class: &[u8], pools: Pools) -> Assignment {
    let order = rpo(b);
    let clob = clobbers(b);
    let a = regalloc::assign(b, &order, class, &clob, pools);
    let occ = occupied(b, &order);
    let nv = b.values.len();
    for v in 0..nv {
        let r = a.reg[v];
        if r == NO_REG {
            continue;
        }
        assert_ne!(class[v], CLASS_NONE, "%{v} has a register but no class");
        let floats = pools.float_saved + pools.float_scratch;
        let (saved, total) = match class[v] {
            CLASS_FLOAT => (pools.float_saved, floats),
            CLASS_VEC => (pools.vec_saved.min(pools.float_saved), floats),
            _ => (pools.int_saved, pools.int_saved + pools.int_scratch),
        };
        assert!(r < total, "%{v}: register {r} outside the pool");
        for w in v + 1..nv {
            if a.reg[w] == r && (class[w] == CLASS_INT) == (class[v] == CLASS_INT) {
                let both = (0..b.blocks.len()).find(|&k| occ[v][k] && occ[w][k]);
                assert_eq!(both, None, "%{v} and %{w} share register {r} in a block");
            }
        }
        if r >= saved {
            // scratch: no clobber after the definition in any occupied block
            let def_inst = (0..b.insts.len()).find(|&i| b.insts.result[i] == Some(Value(v as u32)));
            for (k, bd) in b.blocks.iter().enumerate() {
                if !occ[v][k] {
                    continue;
                }
                for i in bd.insts.indices() {
                    let after =
                        def_inst.is_none_or(|d| !bd.insts.indices().contains(&(d as u32)) || i as usize > d);
                    assert!(!(clob[i as usize] && after), "%{v} in a scratch register across clobber {i}");
                }
            }
        }
    }
    a
}

const POOLS: Pools = Pools { int_saved: 5, int_scratch: 2, float_saved: 0, float_scratch: 8, vec_saved: 0 };

/// `for i in 0..n { acc += x }; return acc`: the counter, the accumulator, and the invariants
/// all get registers.
#[test]
fn loop_values_get_registers() {
    let t = tys();
    let mut b = FuncBuilder::new(&[t.u64, t.f64]);
    let (n, x) = (b.param(0).unwrap(), b.param(1).unwrap());
    let (head, hp) = b.create_block_with(&[t.u64, t.f64]);
    let (body, exit) = (b.create_block(), b.create_block());
    let z = b.int(0);
    let fz = b.f64(0.0);
    b.br(head, &[z, fz]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, t.bool, hp[0], n);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    let one = b.int(1);
    let i2 = b.binary(Op::AddWrap, t.u64, hp[0], one);
    let acc2 = b.binary(Op::Add, t.f64, hp[1], x);
    b.br(head, &[i2.into(), acc2.into()]);
    b.switch_to(exit);
    b.ret(Some(hp[1].into()));
    let body = b.finish().unwrap();
    let a = check(&body, &t, POOLS);
    for v in [n, x, hp[0], hp[1]] {
        assert_ne!(a.reg[v.index()], NO_REG, "%{} should own a register", v.0);
    }
    // block-local values stay with the backend's cache
    for v in [c, i2, acc2] {
        assert_eq!(a.reg[v.index()], NO_REG, "%{} is block-local", v.0);
    }
    assert_ne!(a.reg[n.index()], a.reg[hp[0].index()]);
    assert_ne!(a.reg[x.index()], a.reg[hp[1].index()]);
    // no clobber anywhere: scratch registers first
    assert!(a.reg[n.index()] >= POOLS.int_saved);
}

/// A value live across a call takes a saved register; with no saved float registers a float
/// stays in memory.
#[test]
fn values_across_a_call_need_saved_registers() {
    let t = tys();
    let mut b = FuncBuilder::new(&[t.u64, t.f64]);
    let (n, x) = (b.param(0).unwrap(), b.param(1).unwrap());
    let next = b.create_block();
    b.br(next, &[]);
    b.switch_to(next);
    let r = b.intrinsic(IntrinsicId(0), Some(t.u64), &[]).unwrap();
    let last = b.create_block();
    b.br(last, &[]);
    b.switch_to(last);
    let s = b.binary(Op::AddWrap, t.u64, n, r);
    let _ = b.binary(Op::Add, t.f64, x, x);
    b.ret(Some(s.into()));
    let body = b.finish().unwrap();
    let a = check(&body, &t, POOLS);
    assert!(a.reg[n.index()] < POOLS.int_saved, "n crosses the call");
    assert_eq!(a.reg[x.index()], NO_REG, "no saved float register exists");
    // defined by the call itself, no clobber after: a scratch register
    assert!(a.reg[r.index()] >= POOLS.int_saved && a.reg[r.index()] != NO_REG);
}

/// More live values than registers: the ones used in the loop win.
#[test]
fn the_lowest_weights_stay_in_memory() {
    let t = tys();
    let mut b = FuncBuilder::new(&[t.u64; 6]);
    let ps: Vec<Value> = (0..6).map(|i| b.param(i).unwrap()).collect();
    let (head, hp) = b.create_block_with(&[t.u64]);
    let (body, exit) = (b.create_block(), b.create_block());
    let z = b.int(0);
    b.br(head, &[z]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, t.bool, hp[0], ps[0]);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    let i2 = b.binary(Op::AddWrap, t.u64, hp[0], ps[1]);
    b.br(head, &[i2.into()]);
    b.switch_to(exit);
    let mut acc = hp[0];
    for &p in &ps[2..] {
        acc = b.binary(Op::AddWrap, t.u64, acc, p);
    }
    b.ret(Some(acc.into()));
    let body = b.finish().unwrap();
    let pools = Pools { int_saved: 3, int_scratch: 0, float_saved: 0, float_scratch: 0, vec_saved: 0 };
    let a = check(&body, &t, pools);
    for v in [ps[0], ps[1], hp[0]] {
        assert_ne!(a.reg[v.index()], NO_REG, "%{} is used in the loop", v.0);
    }
    for &p in &ps[2..] {
        assert_eq!(a.reg[p.index()], NO_REG, "%{} is used once after the loop", p.0);
    }
    assert_eq!(a.int_used, 0b111);
}

/// A float and a vector cross a call, another pair lives in a loop without one. The float
/// registers that survive a call as floats do not keep a whole vector: the vector across the
/// call stays in memory, the one in the loop takes a scratch register first.
#[test]
fn vectors_share_the_float_registers() {
    let t = tys();
    let mut b = FuncBuilder::new(&[t.u64, t.f64, t.f64, t.f64, t.f64]);
    let n = b.param(0).unwrap();
    let (f_call, v_call, f_loop, v_loop) =
        (b.param(1).unwrap(), b.param(2).unwrap(), b.param(3).unwrap(), b.param(4).unwrap());
    let next = b.create_block();
    b.br(next, &[]);
    b.switch_to(next);
    b.intrinsic(IntrinsicId(0), Some(t.u64), &[]).unwrap();
    let after = b.create_block();
    b.br(after, &[]);
    b.switch_to(after);
    let s1 = b.binary(Op::Add, t.f64, f_call, v_call);
    let (head, hp) = b.create_block_with(&[t.u64, t.f64, t.f64]);
    let (body, exit) = (b.create_block(), b.create_block());
    let z = b.int(0);
    b.br(head, &[z, f_loop.into(), v_loop.into()]);
    b.switch_to(head);
    let c = b.cmp(Op::Lt, t.bool, hp[0], n);
    b.cbr(c, body, &[], exit, &[]);
    b.switch_to(body);
    let one = b.int(1);
    let i2 = b.binary(Op::AddWrap, t.u64, hp[0], one);
    let f2 = b.binary(Op::Add, t.f64, hp[1], s1);
    let v2 = b.binary(Op::Add, t.f64, hp[2], hp[2]);
    b.br(head, &[i2.into(), f2.into(), v2.into()]);
    b.switch_to(exit);
    let r = b.binary(Op::Add, t.f64, hp[1], hp[2]);
    b.ret(Some(r.into()));
    let body = b.finish().unwrap();
    let mut class = classes(&body, &t, false);
    for v in [v_call, v_loop, hp[2]] {
        class[v.index()] = CLASS_VEC;
    }
    let pools = Pools { int_saved: 4, int_scratch: 0, float_saved: 2, float_scratch: 3, vec_saved: 0 };
    let a = check_classes(&body, &class, pools);
    assert!(a.reg[f_call.index()] < pools.float_saved, "the float crosses the call in a saved register");
    assert_eq!(a.reg[v_call.index()], NO_REG, "no register keeps a vector across the call");
    assert!(a.reg[hp[2].index()] >= pools.float_saved && a.reg[hp[2].index()] != NO_REG);
    assert_ne!(a.reg[hp[1].index()], NO_REG);
    assert_ne!(a.reg[hp[1].index()], a.reg[hp[2].index()], "a float and a vector live together");
    // with a register that keeps a vector, the vector crosses the call in it
    let keep = Pools { vec_saved: 1, ..pools };
    let a = check_classes(&body, &class, keep);
    assert_eq!(a.reg[v_call.index()], 0);
    assert_eq!(a.reg[f_call.index()], 1);
}

#[test]
fn single_block_and_empty_pools_assign_nothing() {
    let t = tys();
    let mut b = FuncBuilder::new(&[t.u64]);
    let p = b.param(0).unwrap();
    let v = b.binary(Op::AddWrap, t.u64, p, p);
    b.ret(Some(v.into()));
    let body = b.finish().unwrap();
    assert!(check(&body, &t, POOLS).is_empty());
    assert!(check(&body, &t, Pools::default()).is_empty());
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

struct Gen<'a> {
    r: Rng,
    t: &'a Tys,
    b: FuncBuilder,
    /// Values in scope: (value, is float).
    vals: Vec<(Value, bool)>,
    budget: usize,
}

impl Gen<'_> {
    fn pick(&mut self, float: bool) -> Operand {
        let n = self.vals.iter().filter(|v| v.1 == float).count();
        if n == 0 {
            return if float { self.b.f64(1.5) } else { self.b.int(3) };
        }
        let k = self.r.below(n);
        self.vals.iter().filter(|v| v.1 == float).nth(k).unwrap().0.into()
    }

    fn ty(&self, float: bool) -> TypeId {
        if float { self.t.f64 } else { self.t.u64 }
    }

    fn step(&mut self, depth: u32) {
        if self.budget == 0 {
            return;
        }
        self.budget -= 1;
        match self.r.below(10) {
            0..=4 => {
                let float = self.r.below(3) == 0;
                let (x, y) = (self.pick(float), self.pick(float));
                let op = if float { Op::Add } else { Op::AddWrap };
                let v = self.b.binary(op, self.ty(float), x, y);
                self.vals.push((v, float));
            }
            5 => {
                let x = self.pick(false);
                let v = self.b.intrinsic(IntrinsicId(0), Some(self.t.u64), &[x]).unwrap();
                self.vals.push((v, false));
            }
            6 | 7 if depth < 4 => {
                let float = self.r.below(2) == 0;
                let (x, y) = (self.pick(false), self.pick(false));
                let c = self.b.cmp(Op::Lt, self.t.bool, x, y);
                let (bt, be) = (self.b.create_block(), self.b.create_block());
                let (join, jp) = self.b.create_block_with(&[self.ty(float)]);
                self.b.cbr(c, bt, &[], be, &[]);
                let mark = self.vals.len();
                for blk in [bt, be] {
                    self.b.switch_to(blk);
                    for _ in 0..self.r.below(4) {
                        self.step(depth + 1);
                    }
                    let x = self.pick(float);
                    self.b.br(join, &[x]);
                    self.vals.truncate(mark);
                }
                self.b.switch_to(join);
                self.vals.push((jp[0], float));
            }
            8 | 9 if depth < 4 => {
                let float = self.r.below(2) == 0;
                let init = self.pick(float);
                let n = self.pick(false);
                let (head, hp) = self.b.create_block_with(&[self.t.u64, self.ty(float)]);
                let (body, exit) = (self.b.create_block(), self.b.create_block());
                let z = self.b.int(0);
                self.b.br(head, &[z, init]);
                self.b.switch_to(head);
                let c = self.b.cmp(Op::Lt, self.t.bool, hp[0], n);
                self.b.cbr(c, body, &[], exit, &[]);
                self.b.switch_to(body);
                let mark = self.vals.len();
                self.vals.push((hp[0], false));
                self.vals.push((hp[1], float));
                for _ in 0..self.r.below(4) {
                    self.step(depth + 1);
                }
                let one = self.b.int(1);
                let i2 = self.b.binary(Op::AddWrap, self.t.u64, hp[0], one);
                // sometimes the params swap roles or feed each other
                let acc2 = self.pick(float);
                self.b.br(head, &[i2.into(), acc2]);
                self.vals.truncate(mark);
                self.b.switch_to(exit);
                self.vals.push((hp[1], float));
            }
            _ => {}
        }
    }
}

fn random_body(seed: u64, t: &Tys) -> Body {
    let mut g = Gen {
        r: Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1),
        t,
        b: FuncBuilder::new(&[t.u64, t.u64, t.f64]),
        vals: Vec::new(),
        budget: 0,
    };
    g.budget = 5 + g.r.below(120);
    for (i, float) in [false, false, true].into_iter().enumerate() {
        g.vals.push((g.b.param(i).unwrap(), float));
    }
    while g.budget > 0 {
        g.step(0);
    }
    let x = g.pick(false);
    g.b.ret(Some(x));
    g.b.finish().unwrap()
}

#[test]
fn random_control_flow() {
    let t = tys();
    let mut assigned = 0usize;
    for seed in 0..600u64 {
        let body = random_body(seed, &t);
        for pools in [
            POOLS,
            Pools { int_saved: 10, int_scratch: 0, float_saved: 8, float_scratch: 0, vec_saved: 0 },
            Pools { int_saved: 1, int_scratch: 1, float_saved: 1, float_scratch: 1, vec_saved: 0 },
            Pools { int_saved: 2, int_scratch: 1, float_saved: 4, float_scratch: 3, vec_saved: 2 },
        ] {
            let a = check(&body, &t, pools);
            assigned += a.reg.iter().filter(|r| **r != NO_REG).count();
            let class = classes(&body, &t, true);
            let a = check_classes(&body, &class, pools);
            assigned += a.reg.iter().filter(|r| **r != NO_REG).count();
        }
    }
    assert!(assigned > 2000, "the random bodies exercise the allocator: {assigned} assignments");
}
