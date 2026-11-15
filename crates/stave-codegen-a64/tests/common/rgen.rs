//! Random IR program generator for the differential tests and the benchmark: straight-line
//! and branchy (if/else joins, switches, bounded loops with block params) int and float code with
//! casts, selects, aggregates, stack slots, and calls between the generated functions. The stress
//! shape ([`program_stress`]) adds frames of several pages, up to 22 params, nesting to depth 5,
//! and fn values (closures and fn items through a join, called and compared).

use super::*;
use stave_ir::arith::{self, CastTy, IntTy};
use stave_ir::{CastKind, FloatBits, FuncBuilder, FuncId, Op, Operand, Site, TypeId, Value, verify_module};

pub struct Rng(pub u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

#[derive(Clone, Copy, PartialEq)]
enum K {
    I(IntTy),
    F(bool),
    B,
    /// A tuple (aggregate in memory).
    A,
    /// `&str` (fat pointer).
    S,
}

struct G<'a> {
    p: &'a Prog,
    r: &'a mut Rng,
    pool: Vec<(Value, TypeId)>,
    tys: Vec<TypeId>,
    site: u32,
    callees: &'a [Callee],
    opt_of: &'a dyn Fn(TypeId) -> Option<TypeId>,
    strs: &'a [stave_ir::StrId],
    /// Nesting limit of diamonds, loops, and switches.
    max_depth: u32,
    /// Nesting depth of the region being generated.
    depth: u32,
    extra: Option<Extra>,
}

/// What the stress shape needs declared up front.
#[derive(Clone, Copy)]
struct Extra {
    /// `fn(u64) -> u64`.
    fn_ty: TypeId,
    /// `(x: u64, env: usize) -> u64`, a closure body.
    closure: FuncId,
    /// `(x: u64) -> u64`.
    item: FuncId,
    /// `&mut [u64; BIG]`.
    big_ref: TypeId,
}

const BIG: u64 = 1500;

/// A generated function: id, params, return type, fn type.
type Callee = (FuncId, Vec<TypeId>, TypeId, TypeId);

fn kind(p: &Prog, t: TypeId) -> K {
    use stave_ir::types::IntBits;
    match p.m.types.key(t) {
        Some(stave_ir::TypeKey::Int { signed, bits }) => K::I(IntTy::new(
            match bits {
                IntBits::Size => 64,
                b => b.width(),
            },
            *signed,
        )),
        Some(stave_ir::TypeKey::Float(FloatBits::F64)) => K::F(true),
        Some(stave_ir::TypeKey::Float(_)) => K::F(false),
        Some(stave_ir::TypeKey::Bool) => K::B,
        Some(stave_ir::TypeKey::Tuple(_)) => K::A,
        _ => K::S,
    }
}

fn fields(p: &Prog, t: TypeId) -> Vec<TypeId> {
    match p.m.types.key(t) {
        Some(stave_ir::TypeKey::Tuple(ts)) => ts.clone(),
        _ => vec![],
    }
}

fn scalar(k: K) -> bool {
    matches!(k, K::I(_) | K::F(_) | K::B)
}

fn cast_class(k: K) -> CastTy {
    match k {
        K::I(it) => CastTy::Int(it),
        K::F(true) => CastTy::Float(FloatBits::F64),
        K::F(false) => CastTy::Float(FloatBits::F32),
        _ => CastTy::Bool,
    }
}

impl G<'_> {
    fn konst(&mut self, b: &mut FuncBuilder, t: TypeId) -> Operand {
        match kind(self.p, t) {
            K::I(it) => {
                let v = match self.r.below(6) {
                    0 => it.min(),
                    1 => it.max(),
                    2 => self.r.below(8) as i128,
                    3 => -(self.r.below(4) as i128),
                    _ => it.value(self.r.next()),
                };
                b.int(if it.fits(v) { v } else { 0 })
            }
            K::F(d) => {
                let v = match self.r.below(8) {
                    0 => 0.0,
                    1 => -1.5,
                    2 => 1e30,
                    3 => f64::NAN,
                    4 => 0.1,
                    5 => (self.r.below(1000) as f64) - 500.0,
                    6 => f64::INFINITY,
                    _ => f64::from_bits(self.r.next() >> 2),
                };
                if d { b.f64(v) } else { b.f32(v as f32) }
            }
            K::B => b.int(self.r.below(2) as i128),
            K::A => {
                let ops: Vec<Operand> = fields(self.p, t).into_iter().map(|f| self.konst(b, f)).collect();
                b.op(Op::Make, t, &ops).into()
            }
            K::S => b.str(self.strs[self.r.below(self.strs.len())]),
        }
    }

    /// A value of type `t` from the pool, or a constant.
    fn arg(&mut self, b: &mut FuncBuilder, t: TypeId, const_pct: u64) -> Operand {
        let c: Vec<Value> = self.pool.iter().filter(|x| x.1 == t).map(|x| x.0).collect();
        if c.is_empty() || self.r.chance(const_pct) {
            return self.konst(b, t);
        }
        c[self.r.below(c.len())].into()
    }

    fn val(&mut self, b: &mut FuncBuilder, t: TypeId) -> Operand {
        self.arg(b, t, 15)
    }

    fn push(&mut self, v: Value, t: TypeId) {
        self.pool.push((v, t));
    }

    fn pick_ty(&mut self) -> TypeId {
        if !self.pool.is_empty() && self.r.chance(70) {
            return self.pool[self.r.below(self.pool.len())].1;
        }
        self.tys[self.r.below(self.tys.len())]
    }

    fn site(&mut self, b: &mut FuncBuilder) {
        self.site += 1;
        b.set_site(Site(self.site));
    }

    /// A slot of several pages: one element written and read back.
    fn big_slot(&mut self, b: &mut FuncBuilder, x: Extra) {
        let tt = self.p.t;
        let Some(pel) = self.p.m.types.find(&stave_ir::TypeKey::Ref { mutable: true, inner: tt.u64 }) else {
            return;
        };
        let s = b.slot(x.big_ref);
        let idx: Operand = if self.r.chance(50) {
            b.int(self.r.below(BIG as usize) as i128)
        } else {
            let i = self.val(b, tt.u32);
            let iz = b.cast(CastKind::Exact, tt.usize, i);
            let mask = b.int(1023);
            b.binary(Op::And, tt.usize, iz, mask).into()
        };
        let v = self.val(b, tt.u64);
        let q = b.op(Op::ElemAddr, pel, &[s.into(), idx]);
        b.store(q, v);
        let r = b.load(tt.u64, q);
        self.push(r, tt.u64);
    }

    /// A closure and a fn item joined into one fn value, which is called and compared.
    fn fn_value(&mut self, b: &mut FuncBuilder, x: Extra) {
        let tt = self.p.t;
        let env = self.val(b, tt.usize);
        let fv = b.fn_make(x.fn_ty, x.closure, env);
        let c = self.val(b, tt.bool);
        let arg = self.val(b, tt.u64);
        let (th, el) = (b.create_block(), b.create_block());
        let (join, jp) = b.create_block_with(&[x.fn_ty]);
        b.cbr(c, th, &[], el, &[]);
        b.switch_to(th);
        b.br(join, &[fv.into()]);
        b.switch_to(el);
        let fi = b.func(x.item);
        b.br(join, &[fi]);
        b.switch_to(join);
        let g = jp[0];
        self.site(b);
        let r = b.call_ind(g, Some(tt.u64), &[arg]).unwrap();
        self.push(r, tt.u64);
        let op = if self.r.chance(50) { Op::Eq } else { Op::Ne };
        let eq = b.op(op, tt.bool, &[g.into(), fv.into()]);
        self.push(eq, tt.bool);
    }

    /// One straight-line instruction.
    fn inst(&mut self, b: &mut FuncBuilder) {
        if let Some(x) = self.extra
            && self.r.chance(6)
        {
            self.site(b);
            return if self.r.chance(40) { self.big_slot(b, x) } else { self.fn_value(b, x) };
        }
        let t = self.pick_ty();
        let k = kind(self.p, t);
        let tt = self.p.t;
        self.site(b);
        let choice = self.r.below(12);
        if !scalar(k) {
            return self.inst_mem(b, t, k, choice);
        }
        match (k, choice) {
            (K::I(_), 0..=4) => {
                let ops = [
                    Op::AddWrap,
                    Op::SubWrap,
                    Op::MulWrap,
                    Op::AddSat,
                    Op::SubSat,
                    Op::MulSat,
                    Op::And,
                    Op::Or,
                    Op::Xor,
                    Op::Add,
                    Op::Sub,
                    Op::Mul,
                    Op::Div,
                    Op::Rem,
                ];
                let mut op = ops[self.r.below(ops.len())];
                if matches!(op, Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem) && !self.r.chance(35) {
                    op = Op::AddWrap;
                }
                let x = self.val(b, t);
                let y = self.val(b, t);
                let v = b.op(op, t, &[x, y]);
                self.push(v, t);
            }
            (K::I(it), 5) => {
                let op = [Op::Shl, Op::Shr, Op::ShlWrap][self.r.below(3)];
                let x = self.val(b, t);
                let n: Operand = if op == Op::ShlWrap || self.r.chance(50) {
                    let at = self.tys[self.r.below(self.tys.len())];
                    if matches!(kind(self.p, at), K::I(_)) && op == Op::ShlWrap {
                        self.val(b, at)
                    } else {
                        b.int(self.r.below(it.bits as usize + 1) as i128)
                    }
                } else {
                    b.int(self.r.below(it.bits as usize) as i128)
                };
                let v = b.op(op, t, &[x, n]);
                self.push(v, t);
            }
            (K::I(it), 6) => {
                let mut ops = vec![Op::Not, Op::Popcount, Op::Clz, Op::Ctz];
                if it.signed && self.r.chance(30) {
                    ops.push(Op::Neg);
                }
                let op = ops[self.r.below(ops.len())];
                let x = self.val(b, t);
                let v = b.unary(op, t, x);
                self.push(v, t);
            }
            (K::F(_), 0..=5) => {
                let op = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Rem, Op::Neg][self.r.below(6)];
                let x = self.val(b, t);
                let v = if op == Op::Neg {
                    b.unary(op, t, x)
                } else {
                    let y = self.val(b, t);
                    b.op(op, t, &[x, y])
                };
                self.push(v, t);
            }
            (K::B, 0..=2) => {
                let op = [Op::And, Op::Or, Op::Xor, Op::Not][self.r.below(4)];
                let x = self.val(b, t);
                let v = if op == Op::Not {
                    b.unary(op, t, x)
                } else {
                    let y = self.val(b, t);
                    b.op(op, t, &[x, y])
                };
                self.push(v, t);
            }
            (_, 7) => {
                let op = [Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge][self.r.below(6)];
                let x = self.val(b, t);
                let y = self.arg(b, t, 30);
                let v = b.op(op, tt.bool, &[x, y]);
                self.push(v, tt.bool);
            }
            (_, 8) => {
                let c = self.val(b, tt.bool);
                let x = self.val(b, t);
                let y = self.val(b, t);
                let v = b.select(t, c, x, y);
                self.push(v, t);
            }
            (_, 9) => {
                // a cast from t to a random type
                let to = self.tys[self.r.below(self.tys.len())];
                let kinds = [CastKind::Exact, CastKind::Wrap, CastKind::Sat, CastKind::Round, CastKind::Try];
                let kd = kinds[self.r.below(kinds.len())];
                let (fc, tc) = (cast_class(k), cast_class(kind(self.p, to)));
                if !scalar(kind(self.p, to)) || !arith::cast_allowed(kd, fc, tc) || to == tt.bool {
                    return;
                }
                let x: Operand = match self.pool.iter().filter(|x| x.1 == t).map(|x| x.0).next() {
                    Some(v) => v.into(),
                    None => return,
                };
                if kd == CastKind::Try {
                    let Some(o) = (self.opt_of)(to) else { return };
                    let r = b.cast(kd, o, x);
                    let tag = b.unary(Op::EnumTag, tt.u32, r);
                    self.push(tag, tt.u32);
                } else {
                    let v = b.cast(kd, to, x);
                    self.push(v, to);
                }
            }
            (_, 10) => {
                // aggregate round trip: tuple make/get, array with a dynamic (masked) index
                let u = self.pick_ty();
                let tup = self.p.m.types.find(&stave_ir::TypeKey::Tuple(vec![t, u, t]));
                let Some(tup) = tup else { return };
                let (x, y, z) = (self.val(b, t), self.val(b, u), self.val(b, t));
                let agg = b.op(Op::Make, tup, &[x, y, z]);
                let k = self.r.below(3);
                let ft = if k == 1 { u } else { t };
                let kk = b.int(k as i128);
                let v = b.binary(Op::Get, ft, agg, kk);
                self.push(v, ft);
                if let Some(arr) = self.p.m.types.find(&stave_ir::TypeKey::Array { elem: t, len: 4 }) {
                    let (a0, a1, a2, a3) = (self.val(b, t), self.val(b, t), self.val(b, t), self.val(b, t));
                    let ar = b.op(Op::Make, arr, &[a0, a1, a2, a3]);
                    let i = self.val(b, tt.u32);
                    let iz = b.cast(CastKind::Exact, tt.usize, i);
                    let i = if self.r.chance(90) {
                        let three = b.int(3);
                        b.binary(Op::And, tt.usize, iz, three)
                    } else {
                        iz
                    };
                    let nv = self.val(b, t);
                    self.site(b);
                    let ar2 = b.op(Op::Set, arr, &[ar.into(), i.into(), nv]);
                    let j = b.int(self.r.below(4) as i128);
                    let e = b.binary(Op::Get, t, ar2, j);
                    self.push(e, t);
                }
            }
            (_, 11) if !self.callees.is_empty() => self.call(b),
            _ => {
                // slot store / load
                let pt = self.p.m.types.find(&stave_ir::TypeKey::Ref { mutable: true, inner: t });
                let Some(pt) = pt else { return };
                let s = b.slot(pt);
                let x = self.val(b, t);
                b.store(s, x);
                let v = b.load(t, s);
                self.push(v, t);
            }
        }
    }

    fn call(&mut self, b: &mut FuncBuilder) {
        // Calls nest multiplicatively through the generated fns: none below depth 2, so the
        // run time stays bounded whatever the nesting limit is.
        if self.depth > 2 {
            return;
        }
        let (f, ps, rt, fnty) = self.callees[self.r.below(self.callees.len())].clone();
        let args: Vec<Operand> = ps.iter().map(|&pt| self.arg(b, pt, 25)).collect();
        let v = if self.r.chance(30) {
            // a fn item as the fn value (code, null environment)
            let _ = fnty;
            b.call_ind(b.func(f), Some(rt), &args).unwrap()
        } else {
            b.call(f, Some(rt), &args).unwrap()
        };
        self.push(v, rt);
    }

    /// Instructions on aggregates and fat pointers.
    fn inst_mem(&mut self, b: &mut FuncBuilder, t: TypeId, k: K, choice: usize) {
        let tt = self.p.t;
        match (k, choice) {
            (K::A, 0..=3) => {
                let fs = fields(self.p, t);
                let x = self.val(b, t);
                let i = self.r.below(fs.len());
                let ii = b.int(i as i128);
                let v = b.op(Op::Get, fs[i], &[x, ii]);
                self.push(v, fs[i]);
            }
            (K::A, 4..=5) => {
                let fs = fields(self.p, t);
                let ops: Vec<Operand> = fs.iter().map(|&f| self.val(b, f)).collect();
                let v = b.op(Op::Make, t, &ops);
                self.push(v, t);
            }
            (K::A, 6) => {
                let fs = fields(self.p, t);
                let x = self.val(b, t);
                let i = self.r.below(fs.len());
                let ii = b.int(i as i128);
                let nv = self.val(b, fs[i]);
                let v = b.op(Op::Set, t, &[x, ii, nv]);
                self.push(v, t);
            }
            (K::S, 0..=4) => {
                let x = self.val(b, t);
                let v = b.unary(Op::SliceLen, tt.usize, x);
                self.push(v, tt.usize);
            }
            (_, 7..=8) => {
                let c = self.val(b, tt.bool);
                let x = self.val(b, t);
                let y = self.val(b, t);
                let v = b.select(t, c, x, y);
                self.push(v, t);
            }
            (_, 9..=10) if !self.callees.is_empty() => self.call(b),
            _ => {
                let pt = self.p.m.types.find(&stave_ir::TypeKey::Ref { mutable: true, inner: t });
                let Some(pt) = pt else { return };
                let s = b.slot(pt);
                let x = self.val(b, t);
                b.store(s, x);
                let v = b.load(t, s);
                self.push(v, t);
            }
        }
    }

    fn straight(&mut self, b: &mut FuncBuilder, n: usize) {
        for _ in 0..n {
            self.inst(b);
        }
    }

    /// Straight-line code, sometimes with a nested construct.
    fn region(&mut self, b: &mut FuncBuilder, depth: u32) {
        self.depth = depth;
        let n = 2 + self.r.below(6);
        self.straight(b, n);
        if depth < self.max_depth && self.r.chance(60) {
            match self.r.below(3) {
                0 => self.diamond(b, depth),
                1 => self.loop_(b, depth),
                _ => self.switch(b, depth),
            }
            self.depth = depth;
            let n = self.r.below(4);
            self.straight(b, n);
        }
    }

    fn join_tys(&mut self) -> Vec<TypeId> {
        (0..1 + self.r.below(3)).map(|_| self.pick_ty()).collect()
    }

    fn diamond(&mut self, b: &mut FuncBuilder, depth: u32) {
        let tys = self.join_tys();
        let c = self.val(b, self.p.t.bool);
        let (th, el) = (b.create_block(), b.create_block());
        let (join, jp) = b.create_block_with(&tys);
        b.cbr(c, th, &[], el, &[]);
        let keep = self.pool.len();
        for arm in [th, el] {
            b.switch_to(arm);
            self.region(b, depth + 1);
            let args: Vec<Operand> = tys.iter().map(|&t| self.val(b, t)).collect();
            b.br(join, &args);
            self.pool.truncate(keep);
        }
        b.switch_to(join);
        for (v, t) in jp.into_iter().zip(tys) {
            self.push(v, t);
        }
    }

    fn switch(&mut self, b: &mut FuncBuilder, depth: u32) {
        let tt = self.p.t;
        let tys = self.join_tys();
        let x = self.val(b, tt.u8);
        let three = b.int(3);
        let s = b.binary(Op::And, tt.u8, x, three);
        let arms: Vec<_> = (0..3).map(|_| b.create_block()).collect();
        let (join, jp) = b.create_block_with(&tys);
        let dargs: Vec<Operand> = tys.iter().map(|&t| self.val(b, t)).collect();
        b.switch(s, &[(0, arms[0], &[]), (1, arms[1], &[]), (2, arms[2], &[])], join, &dargs);
        let keep = self.pool.len();
        for &arm in &arms {
            b.switch_to(arm);
            self.region(b, depth + 1);
            let args: Vec<Operand> = tys.iter().map(|&t| self.val(b, t)).collect();
            b.br(join, &args);
            self.pool.truncate(keep);
        }
        b.switch_to(join);
        for (v, t) in jp.into_iter().zip(tys) {
            self.push(v, t);
        }
    }

    /// `for i in 0..n { accs = body(accs) }` with n in 0..5 and 1–3 accumulators.
    fn loop_(&mut self, b: &mut FuncBuilder, depth: u32) {
        let tt = self.p.t;
        let mut tys = vec![tt.u32];
        tys.extend(self.join_tys());
        let (head, hp) = b.create_block_with(&tys);
        let body = b.create_block();
        let (exit, ep) = b.create_block_with(&tys[1..]);
        let mut init = vec![b.int(0)];
        for &t in &tys[1..] {
            init.push(self.val(b, t));
        }
        let n = b.int(self.r.below(5) as i128);
        b.br(head, &init);
        b.switch_to(head);
        let c = b.cmp(Op::Lt, tt.bool, hp[0], n);
        let exit_args: Vec<Operand> = hp[1..].iter().map(|v| (*v).into()).collect();
        b.cbr(c, body, &[], exit, &exit_args);
        let keep = self.pool.len();
        for (&v, &t) in hp.iter().zip(&tys) {
            self.push(v, t);
        }
        b.switch_to(body);
        self.region(b, depth + 1);
        let one = b.int(1);
        let i1 = b.binary(Op::Add, tt.u32, hp[0], one);
        let mut next = vec![i1.into()];
        // permute the accumulators sometimes (parallel copies)
        let mut accs: Vec<Operand> = tys[1..].iter().map(|&t| self.val(b, t)).collect();
        if accs.len() > 1 && tys[1..].iter().all(|x| *x == tys[1]) && self.r.chance(70) {
            // all accumulators have one type: pass the header params rotated
            accs = hp[1..].iter().map(|v| Operand::Value(*v)).collect();
            accs.rotate_left(1);
        }
        next.extend(accs);
        b.br(head, &next);
        self.pool.truncate(keep);
        b.switch_to(exit);
        for (v, t) in ep.into_iter().zip(tys[1..].iter().copied()) {
            self.push(v, t);
        }
    }
}

/// A random program of `nf` (default 2–6) functions.
pub fn program_n(seed: u64, nf: Option<usize>) -> Prog {
    program_shaped(seed, nf, false)
}

/// A random program in the stress shape.
pub fn program_stress(seed: u64) -> Prog {
    program_shaped(seed, None, true)
}

fn program_shaped(seed: u64, nf: Option<usize>, stress: bool) -> Prog {
    let mut p = Prog::new();
    let t = p.t;
    let mut tys: Vec<TypeId> = t.ints().to_vec();
    tys.extend([t.f32, t.f64, t.bool, t.f64, t.i32, t.u64]);
    let scalars = tys.clone();
    let aggs = [
        p.m.types.tuple(vec![t.i32, t.f64, t.i32]),
        p.m.types.tuple(vec![t.u8, t.u64, t.u8]),
        p.m.types.tuple(vec![t.bool, t.f32, t.bool]),
    ];
    let strs: Vec<stave_ir::StrId> =
        ["", "a", "hello", "0123456789abcdef0123"].iter().map(|s| p.m.add_string(*s)).collect();
    for &a in aggs.iter().chain([&t.str_ref]) {
        p.m.types.ref_(true, a);
    }
    // Types the generator looks up: tuples, arrays, refs, Options.
    for &a in &scalars {
        for &c in &scalars {
            p.m.types.tuple(vec![a, c, a]);
        }
        p.m.types.array(a, 4);
        p.m.types.ref_(true, a);
    }
    tys.extend(aggs);
    tys.push(t.str_ref);
    let mut opts = Vec::new();
    for (k, &a) in scalars.iter().enumerate() {
        let o = p.m.types.adt(
            stave_ir::DeclId(20_000 + k as u32),
            vec![],
            stave_ir::AdtShape::enumeration(vec![vec![], vec![a]]),
        );
        opts.push((a, o));
    }
    let opt_of = move |a: TypeId| opts.iter().find(|x| x.0 == a).map(|x| x.1);
    let extra = stress.then(|| {
        let fn_ty = p.m.types.fn_type(vec![t.u64], t.u64, stave_target::FeatureSet::EMPTY);
        let big = p.m.types.array(t.u64, BIG);
        let big_ref = p.m.types.ref_(true, big);
        let closure = p.declare("rnd.closure", &[t.u64, t.usize], Some(t.u64));
        let mut b = FuncBuilder::new(&[t.u64, t.usize]);
        let (x, e) = (b.param(0).unwrap(), b.param(1).unwrap());
        let three = b.int(3);
        let m = b.binary(Op::MulWrap, t.u64, x, three);
        let e64 = b.cast(CastKind::Wrap, t.u64, e);
        let v = b.binary(Op::AddWrap, t.u64, m, e64);
        b.ret(Some(v.into()));
        p.m.define(closure, b.finish().expect("builder"));
        let item = p.declare("rnd.item", &[t.u64], Some(t.u64));
        let mut b = FuncBuilder::new(&[t.u64]);
        let x = b.param(0).unwrap();
        let k = b.int(0x55);
        let v = b.binary(Op::Xor, t.u64, x, k);
        b.ret(Some(v.into()));
        p.m.define(item, b.finish().expect("builder"));
        Extra { fn_ty, closure, item, big_ref }
    });
    let (max_depth, max_params) = if stress { (5, 22) } else { (2, 11) };
    let mut r = Rng(seed);
    let mut funcs: Vec<Callee> = Vec::new();
    let nf = nf.unwrap_or(2 + r.below(5));
    let mut site = 0u32;
    for _ in 0..nf {
        let np = 1 + r.below(max_params);
        let params: Vec<TypeId> = (0..np).map(|_| tys[r.below(tys.len())]).collect();
        let ret = tys[r.below(tys.len())];
        let f = p.declare("rnd.f", &params, Some(ret));
        let fnty = p.m.types.fn_type(params.clone(), ret, stave_target::FeatureSet::EMPTY);
        let mut b = FuncBuilder::new(&params);
        let pool: Vec<(Value, TypeId)> = b.params().iter().copied().zip(params.iter().copied()).collect();
        let mut g = G {
            p: &p,
            r: &mut r,
            pool,
            tys: tys.clone(),
            site,
            callees: &funcs,
            opt_of: &opt_of,
            strs: &strs,
            max_depth,
            depth: 0,
            extra,
        };
        g.region(&mut b, 0);
        if g.r.chance(50) {
            g.region(&mut b, 1);
        }
        let rv = g.arg(&mut b, ret, 5);
        site = g.site;
        b.ret(Some(rv));
        p.m.define(f, b.finish().expect("builder"));
        funcs.push((f, params, ret, fnty));
    }
    let calls: Vec<Callee> = funcs.clone();
    let mut r2 = Rng(seed ^ 0xabcdef);
    p.set_main(|b, p| {
        for (f, ps, rt, _) in &calls {
            for _ in 0..3 {
                let mut g = G {
                    p,
                    r: &mut r2,
                    pool: vec![],
                    tys: vec![],
                    site: 0,
                    callees: &[],
                    opt_of: &|_| None,
                    strs: &strs,
                    max_depth: 0,
                    depth: 0,
                    extra: None,
                };
                let args: Vec<Operand> = ps.iter().map(|&pt| g.konst(b, pt)).collect();
                let v = b.call(*f, Some(*rt), &args).unwrap();
                match kind(p, *rt) {
                    K::A => {
                        for (i, ft) in fields(p, *rt).into_iter().enumerate() {
                            let ii = b.int(i as i128);
                            let x = b.op(Op::Get, ft, &[v.into(), ii]);
                            p.out(b, x, ft);
                        }
                    }
                    K::S => {
                        b.call(p.out_str, None, &[v.into()]);
                    }
                    _ => p.out(b, v, *rt),
                }
            }
        }
    });
    if let Err(e) = verify_module(&p.m) {
        panic!("generator produced invalid IR (seed {seed}): {:?}", &e[..e.len().min(3)]);
    }
    p
}

pub fn program(seed: u64) -> Prog {
    program_n(seed, None)
}
