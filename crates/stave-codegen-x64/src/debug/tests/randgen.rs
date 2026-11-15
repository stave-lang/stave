//! Random IR function generator for the property test and the codegen benchmark: straight-line
//! integer and float code with traps, casts, compares and selects, if/else diamonds with join
//! params, counted loops, and calls.

use stave_ir::arith::{self, CastTy, IntTy};
use stave_ir::{CastKind, FuncBuilder, FuncId, Op, Operand, TypeId, TypeKey, Value};

use super::harness::Prog;

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    pub fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum K {
    Int(IntTy),
    F32,
    F64,
    Bool,
}

pub struct Gen<'p> {
    p: &'p mut Prog,
    r: Rng,
    tys: Vec<(TypeId, K)>,
    avail: Vec<(Value, TypeId)>,
    budget: usize,
    /// Allow ops that may trap.
    traps: bool,
    print: bool,
    /// Aggregate values in scope and the shapes of their types.
    aggs: Vec<(Value, TypeId)>,
    shapes: Vec<(TypeId, Agg)>,
}

#[derive(Clone, Debug)]
enum Agg {
    Tuple(Vec<TypeId>),
    Array(TypeId, u64),
}

const WRAP_OPS: [Op; 10] = [
    Op::AddWrap,
    Op::SubWrap,
    Op::MulWrap,
    Op::AddSat,
    Op::SubSat,
    Op::MulSat,
    Op::And,
    Op::Or,
    Op::Xor,
    Op::ShlWrap,
];
const TRAP_OPS: [Op; 7] = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Rem, Op::Shl, Op::Shr];
const CMPS: [Op; 6] = [Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge];

impl<'p> Gen<'p> {
    pub fn new(p: &'p mut Prog, seed: u64, budget: usize, traps: bool, print: bool) -> Gen<'p> {
        let t = p.t;
        let mut tys: Vec<(TypeId, K)> = Vec::new();
        for ty in super::harness::int_types(&t) {
            let it = match p.m.types.key(ty) {
                Some(TypeKey::Int { signed, bits }) => IntTy::new(bits.width(), *signed),
                _ => IntTy::new(64, false),
            };
            tys.push((ty, K::Int(it)));
        }
        tys.push((t.f32, K::F32));
        tys.push((t.f64, K::F64));
        tys.push((t.bool, K::Bool));
        Gen {
            p,
            r: Rng::new(seed),
            tys,
            avail: Vec::new(),
            budget,
            traps,
            print,
            aggs: Vec::new(),
            shapes: Vec::new(),
        }
    }

    fn kind(&self, t: TypeId) -> K {
        self.tys.iter().find(|x| x.0 == t).map_or(K::Bool, |x| x.1)
    }

    fn rand_ty(&mut self) -> (TypeId, K) {
        let i = self.r.below(self.tys.len());
        self.tys[i]
    }

    fn int_val(&mut self, it: IntTy) -> i128 {
        let (lo, hi) = (it.min(), it.max());
        let x = match self.r.below(8) {
            0 => lo,
            1 => hi,
            2 => 0,
            3 => 1,
            4 => -1,
            5 => (self.r.next() % 16) as i128,
            _ => {
                let span = (hi - lo) as u128 + 1;
                lo + (u128::from(self.r.next()) % span) as i128
            }
        };
        x.clamp(lo, hi)
    }

    fn float_val(&mut self) -> f64 {
        const V: [f64; 14] = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            0.1,
            1e10,
            -3.75,
            1e300,
            f64::INFINITY,
            f64::NAN,
            65536.5,
            -1e-5,
            7.0,
            2.0e19,
        ];
        if self.r.chance(70) { V[self.r.below(V.len())] } else { (self.r.next() as i64 as f64) / 1e6 }
    }

    fn konst(&mut self, b: &mut FuncBuilder, k: K) -> Operand {
        match k {
            K::Int(it) => {
                let v = self.int_val(it);
                b.int(v)
            }
            K::F32 => {
                let v = self.float_val() as f32;
                b.f32(v)
            }
            K::F64 => {
                let v = self.float_val();
                b.f64(v)
            }
            K::Bool => {
                let v = self.r.below(2) as i128;
                b.int(v)
            }
        }
    }

    fn pick(&mut self, t: TypeId) -> Option<Value> {
        let n = self.avail.iter().filter(|x| x.1 == t).count();
        if n == 0 {
            return None;
        }
        let k = self.r.below(n);
        self.avail.iter().filter(|x| x.1 == t).nth(k).map(|x| x.0)
    }

    fn operand(&mut self, b: &mut FuncBuilder, t: TypeId, const_pct: u64) -> Operand {
        if !self.r.chance(const_pct)
            && let Some(v) = self.pick(t)
        {
            return v.into();
        }
        let k = self.kind(t);
        self.konst(b, k)
    }

    /// A value operand of type `t` (an opaque copy of a constant if none exists).
    fn value(&mut self, b: &mut FuncBuilder, t: TypeId) -> Value {
        if let Some(v) = self.pick(t) {
            return v;
        }
        let k = self.kind(t);
        let c = self.konst(b, k);
        let v = self.p.opaque(b, c, t);
        self.avail.push((v, t));
        v
    }

    fn push(&mut self, v: Value, t: TypeId) {
        self.avail.push((v, t));
    }

    /// Emits one instruction (or construct); returns false when the budget is used up.
    fn step(&mut self, b: &mut FuncBuilder, depth: u32) {
        if self.budget == 0 {
            return;
        }
        self.budget -= 1;
        let tb = self.p.t.bool;
        let r = self.r.below(112);
        match r {
            0..=34 => {
                let (t, k) = self.rand_ty();
                let K::Int(it) = k else { return self.float_bin(b) };
                let trap = self.traps && self.r.chance(25);
                let op = if trap { TRAP_OPS[self.r.below(7)] } else { WRAP_OPS[self.r.below(10)] };
                let a = self.value(b, t);
                let x = if matches!(op, Op::Shl | Op::Shr | Op::ShlWrap) {
                    if self.r.chance(60) {
                        let n = (self.r.next() % u64::from(it.bits)) as i128;
                        b.int(n)
                    } else {
                        let (at, _) = self.int_ty();
                        self.operand(b, at, 20)
                    }
                } else {
                    self.operand(b, t, 35)
                };
                let v = b.binary(op, t, a, x);
                self.push(v, t);
            }
            35..=44 => {
                let (t, k) = self.int_ty();
                let K::Int(it) = k else { return };
                let ops = [Op::Not, Op::Popcount, Op::Clz, Op::Ctz, Op::Neg];
                let op = ops[self.r.below(if it.signed && self.traps { 5 } else { 4 })];
                let a = self.value(b, t);
                let v = b.unary(op, t, a);
                self.push(v, t);
            }
            45..=54 => {
                let (t, _) = self.rand_ty();
                let op = CMPS[self.r.below(6)];
                let a = self.value(b, t);
                let x = self.operand(b, t, 30);
                let v = b.cmp(op, tb, a, x);
                self.push(v, tb);
            }
            55..=64 => self.cast(b),
            65..=79 => self.float_bin(b),
            80..=84 => {
                let (t, _) = self.rand_ty();
                let c = self.value(b, tb);
                let x = self.operand(b, t, 30);
                let y = self.operand(b, t, 30);
                let v = b.select(t, c, x, y);
                self.push(v, t);
            }
            85..=91 if depth < 3 => self.diamond(b, depth),
            92..=95 if depth < 3 => self.counted_loop(b, depth),
            100..=103 => self.agg_make(b),
            104..=107 => self.agg_use(b),
            108 => self.slot(b),
            109..=110 if depth < 3 => self.switch(b, depth),
            111 => self.agg_copy(b),
            _ => {
                let (t, _) = self.rand_ty();
                let x = self.value(b, t);
                let v = self.p.opaque(b, x.into(), t);
                self.push(v, t);
            }
        }
    }

    fn shape(&self, t: TypeId) -> Option<Agg> {
        self.shapes.iter().find(|x| x.0 == t).map(|x| x.1.clone())
    }

    fn push_any(&mut self, v: Value, t: TypeId) {
        if self.shape(t).is_some() { self.aggs.push((v, t)) } else { self.push(v, t) }
    }

    /// A scalar type, or (30%) the type of an aggregate in scope.
    fn any_ty(&mut self) -> TypeId {
        if !self.aggs.is_empty() && self.r.chance(30) {
            let i = self.r.below(self.aggs.len());
            return self.aggs[i].1;
        }
        self.rand_ty().0
    }

    fn pick_agg(&mut self, t: TypeId) -> Option<Value> {
        let n = self.aggs.iter().filter(|x| x.1 == t).count();
        let k = self.r.below(n.max(1));
        self.aggs.iter().filter(|x| x.1 == t).nth(k).map(|x| x.0)
    }

    fn join_operand(&mut self, b: &mut FuncBuilder, t: TypeId) -> Operand {
        if self.shape(t).is_some() {
            return match self.pick_agg(t) {
                Some(v) => v.into(),
                None => b.int(0),
            };
        }
        self.operand(b, t, 20)
    }

    fn agg_make(&mut self, b: &mut FuncBuilder) {
        let (t, ops) = if self.r.chance(65) {
            let n = 1 + self.r.below(4);
            let fs: Vec<TypeId> = (0..n).map(|_| self.rand_ty().0).collect();
            let t = self.p.m.types.tuple(fs.clone());
            let ops: Vec<Operand> = fs.iter().map(|&f| self.operand(b, f, 40)).collect();
            if self.shape(t).is_none() {
                self.shapes.push((t, Agg::Tuple(fs)));
            }
            (t, ops)
        } else {
            let (e, _) = self.rand_ty();
            let n = if self.r.chance(20) { 24 } else { 2 + self.r.below(5) as u64 };
            let t = self.p.m.types.array(e, n);
            let ops: Vec<Operand> = (0..n).map(|_| self.operand(b, e, 40)).collect();
            if self.shape(t).is_none() {
                self.shapes.push((t, Agg::Array(e, n)));
            }
            (t, ops)
        };
        let v = b.op(Op::Make, t, &ops);
        self.aggs.push((v, t));
    }

    fn agg_use(&mut self, b: &mut FuncBuilder) {
        if self.aggs.is_empty() {
            return self.agg_make(b);
        }
        let (v, t) = self.aggs[self.r.below(self.aggs.len())];
        let Some(shape) = self.shape(t) else { return };
        let usize_ = self.p.t.usize;
        let (ft, idx): (TypeId, Operand) = match shape {
            Agg::Tuple(fs) => {
                let k = self.r.below(fs.len());
                (fs[k], b.int(k as i128))
            }
            Agg::Array(e, n) => {
                let oob = self.traps && self.r.chance(5);
                let k = if oob { n as i128 } else { self.r.below(n as usize) as i128 };
                let c = b.int(k);
                if self.r.chance(60) || oob { (e, self.p.opaque(b, c, usize_).into()) } else { (e, c) }
            }
        };
        if self.r.chance(55) {
            let x = b.op(Op::Get, ft, &[v.into(), idx]);
            self.push(x, ft);
        } else {
            let nv = self.operand(b, ft, 30);
            let x = b.op(Op::Set, t, &[v.into(), idx, nv]);
            self.aggs.push((x, t));
        }
    }

    fn agg_copy(&mut self, b: &mut FuncBuilder) {
        if self.aggs.is_empty() {
            return;
        }
        let (v, t) = self.aggs[self.r.below(self.aggs.len())];
        let x = if self.r.chance(50) {
            self.p.opaque(b, v.into(), t)
        } else {
            let tb = self.p.t.bool;
            let c = self.value(b, tb);
            let w = self.pick_agg(t).unwrap_or(v);
            b.select(t, c, v, w)
        };
        self.aggs.push((x, t));
    }

    fn slot(&mut self, b: &mut FuncBuilder) {
        let (t, _) = self.rand_ty();
        let pt = self.p.m.types.ref_(true, t);
        let s = b.slot(pt);
        let x = self.operand(b, t, 30);
        b.store(s, x);
        let v = if self.r.chance(50) {
            b.load(t, s)
        } else {
            let rp = self.p.m.types.ptr(true, t);
            let q = b.cast(CastKind::Exact, rp, s);
            let q = self.p.opaque(b, q.into(), rp);
            b.load(t, q)
        };
        self.push(v, t);
    }

    fn switch(&mut self, b: &mut FuncBuilder, depth: u32) {
        let (st, sk) = self.int_ty();
        let K::Int(it) = sk else { return };
        let scrut = self.value(b, st);
        let rt = self.any_ty();
        let (join, jp) = b.create_block_with(&[rt]);
        let n = 1 + self.r.below(6);
        let mut vals: Vec<i128> = (0..n).map(|_| self.int_val(it)).collect();
        vals.sort();
        vals.dedup();
        let blocks: Vec<_> = vals.iter().map(|_| b.create_block()).collect();
        let dflt = b.create_block();
        let direct: Vec<Option<Operand>> = vals
            .iter()
            .map(|_| if self.r.chance(30) { Some(self.join_operand(b, rt)) } else { None })
            .collect();
        let cases: Vec<(i128, stave_ir::Block, Vec<Operand>)> = vals
            .iter()
            .zip(&blocks)
            .zip(&direct)
            .map(|((v, bl), d)| match d {
                Some(o) => (*v, join, vec![*o]),
                None => (*v, *bl, vec![]),
            })
            .collect();
        let refs: Vec<(i128, stave_ir::Block, &[Operand])> =
            cases.iter().map(|c| (c.0, c.1, c.2.as_slice())).collect();
        b.switch(scrut, &refs, dflt, &[]);
        let (mark, amark) = (self.avail.len(), self.aggs.len());
        for (bl, d) in blocks.iter().zip(&direct).map(|(b, d)| (*b, d.is_some())).chain([(dflt, false)]) {
            b.switch_to(bl);
            if !d {
                for _ in 0..self.r.below(3) {
                    self.step(b, depth + 1);
                }
            }
            let x = self.join_operand(b, rt);
            b.br(join, &[x]);
            self.avail.truncate(mark);
            self.aggs.truncate(amark);
        }
        b.switch_to(join);
        self.push_any(jp[0], rt);
    }

    fn int_ty(&mut self) -> (TypeId, K) {
        let i = self.r.below(10);
        self.tys[i]
    }

    fn float_bin(&mut self, b: &mut FuncBuilder) {
        let t = if self.r.chance(50) { self.p.t.f32 } else { self.p.t.f64 };
        if self.r.chance(10) {
            let a = self.value(b, t);
            let v = b.unary(Op::Neg, t, a);
            return self.push(v, t);
        }
        let op = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Rem][self.r.below(5)];
        let a = self.operand(b, t, 15);
        let x = self.operand(b, t, 30);
        if matches!((a, x), (Operand::Const(_), Operand::Const(_))) {
            let a = self.value(b, t);
            let v = b.binary(op, t, a, x);
            return self.push(v, t);
        }
        let v = b.binary(op, t, a, x);
        self.push(v, t);
    }

    fn cast_ty(&self, k: K) -> CastTy {
        match k {
            K::Int(it) => CastTy::Int(it),
            K::F32 => CastTy::Float(stave_ir::FloatBits::F32),
            K::F64 => CastTy::Float(stave_ir::FloatBits::F64),
            K::Bool => CastTy::Bool,
        }
    }

    fn cast(&mut self, b: &mut FuncBuilder) {
        let (ft, fk) = self.rand_ty();
        let (tt, tk) = self.rand_ty();
        if tk == K::Bool {
            return;
        }
        let kinds = [CastKind::Exact, CastKind::Wrap, CastKind::Sat, CastKind::Round, CastKind::Try];
        let ok: Vec<CastKind> = kinds
            .into_iter()
            .filter(|k| arith::cast_allowed(*k, self.cast_ty(fk), self.cast_ty(tk)))
            .collect();
        if ok.is_empty() {
            return;
        }
        let kind = ok[self.r.below(ok.len())];
        let x = self.value(b, ft);
        if kind == CastKind::Try {
            let ot = self.p.option(tt);
            let r = b.cast(kind, ot, x);
            let u32_ = self.p.t.u32;
            let tag = b.op(Op::EnumTag, u32_, &[r.into()]);
            self.push(tag, u32_);
        } else {
            let v = b.cast(kind, tt, x);
            self.push(v, tt);
        }
    }

    fn diamond(&mut self, b: &mut FuncBuilder, depth: u32) {
        let tb = self.p.t.bool;
        let t = self.any_ty();
        let c = self.value(b, tb);
        let (bt, be) = (b.create_block(), b.create_block());
        let (join, jp) = b.create_block_with(&[t]);
        b.cbr(c, bt, &[], be, &[]);
        let (mark, amark) = (self.avail.len(), self.aggs.len());
        for blk in [bt, be] {
            b.switch_to(blk);
            let n = self.r.below(4);
            for _ in 0..n {
                self.step(b, depth + 1);
            }
            let x = self.join_operand(b, t);
            b.br(join, &[x]);
            self.avail.truncate(mark);
            self.aggs.truncate(amark);
        }
        b.switch_to(join);
        self.push_any(jp[0], t);
    }

    fn counted_loop(&mut self, b: &mut FuncBuilder, depth: u32) {
        let t = self.p.t;
        let (acc_t, k) = self.rand_ty();
        if k == K::Bool {
            return;
        }
        let init = self.operand(b, acc_t, 50);
        let (head, hp) = b.create_block_with(&[t.u32, acc_t]);
        let (body, exit) = (b.create_block(), b.create_block());
        let z = b.int(0);
        b.br(head, &[z, init]);
        b.switch_to(head);
        let trips = self.r.below(6) as i128;
        let n = b.int(trips);
        let lt = b.cmp(Op::Lt, t.bool, hp[0], n);
        b.cbr(lt, body, &[], exit, &[]);
        b.switch_to(body);
        let (mark, amark) = (self.avail.len(), self.aggs.len());
        self.push(hp[1], acc_t);
        for _ in 0..self.r.below(3) {
            self.step(b, depth + 1);
        }
        let op = match k {
            K::Int(_) => WRAP_OPS[self.r.below(9)],
            _ => [Op::Add, Op::Mul, Op::Sub][self.r.below(3)],
        };
        let x = self.operand(b, acc_t, 40);
        let acc2 = b.binary(op, acc_t, hp[1], x);
        let one = b.int(1);
        let i2 = b.binary(Op::Add, t.u32, hp[0], one);
        b.br(head, &[i2.into(), acc2.into()]);
        self.avail.truncate(mark);
        self.aggs.truncate(amark);
        b.switch_to(exit);
        self.push(hp[1], acc_t);
    }

    /// Generates and defines `fn(params) -> u64` (prints every value when `print`); returns it
    /// with a fresh builder holding constant arguments for a call.
    pub fn function(mut self, name: &str) -> (FuncId, Vec<Operand>, FuncBuilder) {
        let nparams = self.r.below(8);
        let ptys: Vec<TypeId> = (0..nparams).map(|_| self.rand_ty().0).collect();
        let f = self.p.decl(name, &ptys, Some(self.p.t.u64));
        let mut b = FuncBuilder::new(&ptys);
        for (i, &t) in ptys.iter().enumerate() {
            if let Some(v) = b.param(i) {
                self.push(v, t);
            }
        }
        while self.budget > 0 {
            self.step(&mut b, 0);
        }
        let u64_ = self.p.t.u64;
        let aggs = std::mem::take(&mut self.aggs);
        for &(v, t) in &aggs {
            let Some(shape) = self.shape(t) else { continue };
            let (fields, n) = match shape {
                Agg::Tuple(fs) => (fs.clone(), fs.len()),
                Agg::Array(e, n) => (vec![e; n as usize], n as usize),
            };
            for (k, ft) in fields.into_iter().enumerate().take(n) {
                let kk = b.int(k as i128);
                let x = b.op(Op::Get, ft, &[v.into(), kk]);
                self.avail.push((x, ft));
            }
        }
        let mut acc: Operand = b.int(0);
        let avail = std::mem::take(&mut self.avail);
        for &(v, t) in &avail {
            if self.print {
                self.p.print(&mut b, v.into(), t);
            }
            let k = self.kind(t);
            let x: Operand = match k {
                K::Int(_) => b.cast(CastKind::Wrap, u64_, v).into(),
                K::Bool => b.cast(CastKind::Exact, u64_, v).into(),
                K::F32 | K::F64 => b.cast(CastKind::Sat, u64_, v).into(),
            };
            let m = b.int(31);
            let a2 = b.binary(Op::MulWrap, u64_, acc, m);
            acc = b.binary(Op::AddWrap, u64_, a2, x).into();
        }
        b.ret(Some(acc));
        self.p.def(f, b);
        let mut main_b = FuncBuilder::new(&[]);
        let args: Vec<Operand> = ptys
            .iter()
            .map(|&t| {
                let k = self.kind(t);
                self.konst(&mut main_b, k)
            })
            .collect();
        (f, args, main_b)
    }
}
