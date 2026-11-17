//! Vector and mask ops, differential per op: small IR functions run in the interpreter and
//! natively under each feature [`Mode`] (a v2 baseline: vectors up to 128 bits; a v3 baseline,
//! and a v2 baseline whose test functions carry `needs`: up to 256 bits; a v4 baseline, and a v2
//! baseline whose test functions carry the AVX-512 set: up to 512 bits, masks in `k`
//! registers), each as a debug and as a release build. Every function of every build is
//! disassembled and its instructions are checked against the function's feature set
//! ([`super::features::check_image`]).
//!
//! The AVX-512 modes run on the `v4` leg ([`v4_leg`]: natively, or under Intel SDE), and only
//! the cases whose function compiles in K mode ([`k_ok`]): the ops K mode lacks are listed in
//! [`K_UNSUPPORTED`], and a 512-bit case of one of them runs nowhere.
//!
//! A program is a list of cases, one function each, called in order from `main`. Every input tuple
//! prints a label line (`# <op> <operands> -> <result>`), then the raw bits of its result as
//! 16-hex-digit words (vectors in little-endian 8-byte words, masks through `mask.to_vec`). A label
//! ending in ` ~32` / ` ~64` marks the following words as float lanes of that width whose NaNs
//! compare by NaN-ness only (arithmetic results); every other line compares bit-exactly.
//! Vectors are built from words through a stack slot and printed through one, so the op under test
//! never sees a constant. Traps end the program with exit status `100 + kind`; a trapping case is
//! a program of its own.

mod lanewise;
mod memory;
mod plumbing;
mod recipes;
mod regs;
mod reshape;
mod values;

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use stave_ir::arith::{CastTy, IntTy};
use stave_ir::simd::LaneTy;
use stave_ir::{
    CastKind, FloatBits, FuncBuilder, FuncId, IntrinsicId, Lanes, Op, Operand, PanicKind, TypeId, TypeKey,
    Value, VecOp,
};

use stave_target::{Feature, FeatureSet};

use super::harness::*;
use crate::debug::{Error, Options, bench_ctx, compile_func, compile_module};

/// A lane type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lane {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl Lane {
    pub const ALL: [Lane; 10] = [
        Lane::I8,
        Lane::I16,
        Lane::I32,
        Lane::I64,
        Lane::U8,
        Lane::U16,
        Lane::U32,
        Lane::U64,
        Lane::F32,
        Lane::F64,
    ];
    pub const INTS: [Lane; 8] =
        [Lane::I8, Lane::I16, Lane::I32, Lane::I64, Lane::U8, Lane::U16, Lane::U32, Lane::U64];
    pub const FLOATS: [Lane; 2] = [Lane::F32, Lane::F64];

    pub fn bits(self) -> u32 {
        match self {
            Lane::I8 | Lane::U8 => 8,
            Lane::I16 | Lane::U16 => 16,
            Lane::I32 | Lane::U32 | Lane::F32 => 32,
            Lane::I64 | Lane::U64 | Lane::F64 => 64,
        }
    }

    pub fn bytes(self) -> u32 {
        self.bits() / 8
    }

    pub fn is_float(self) -> bool {
        matches!(self, Lane::F32 | Lane::F64)
    }

    pub fn signed(self) -> bool {
        matches!(self, Lane::I8 | Lane::I16 | Lane::I32 | Lane::I64)
    }

    pub fn int(self) -> Option<IntTy> {
        if self.is_float() { None } else { Some(IntTy::new(self.bits() as u8, self.signed())) }
    }

    pub fn lane_ty(self) -> LaneTy {
        match self {
            Lane::F32 => LaneTy::Float(FloatBits::F32),
            Lane::F64 => LaneTy::Float(FloatBits::F64),
            _ => LaneTy::Int(IntTy::new(self.bits() as u8, self.signed())),
        }
    }

    pub fn cast_ty(self) -> CastTy {
        self.lane_ty().cast_ty()
    }

    pub fn name(self) -> &'static str {
        match self {
            Lane::I8 => "i8",
            Lane::I16 => "i16",
            Lane::I32 => "i32",
            Lane::I64 => "i64",
            Lane::U8 => "u8",
            Lane::U16 => "u16",
            Lane::U32 => "u32",
            Lane::U64 => "u64",
            Lane::F32 => "f32",
            Lane::F64 => "f64",
        }
    }

    pub fn uint(bits: u32) -> Lane {
        match bits {
            8 => Lane::U8,
            16 => Lane::U16,
            32 => Lane::U32,
            _ => Lane::U64,
        }
    }

    pub fn sint(bits: u32) -> Lane {
        match bits {
            8 => Lane::I8,
            16 => Lane::I16,
            32 => Lane::I32,
            _ => Lane::I64,
        }
    }

    /// The double-width lane (`widen_lo` / `narrow_*`).
    pub fn wide(self) -> Option<Lane> {
        Some(match self {
            Lane::I8 => Lane::I16,
            Lane::I16 => Lane::I32,
            Lane::I32 => Lane::I64,
            Lane::U8 => Lane::U16,
            Lane::U16 => Lane::U32,
            Lane::U32 => Lane::U64,
            Lane::F32 => Lane::F64,
            _ => return None,
        })
    }

    /// All ones in the lane's width.
    pub fn ones(self) -> u64 {
        if self.bits() == 64 { u64::MAX } else { (1u64 << self.bits()) - 1 }
    }

    /// Lane bits of the int `v` (two's complement, truncated).
    pub fn raw(self, v: i128) -> u64 {
        (v as u64) & self.ones()
    }

    /// The mathematical value of int lane bits.
    pub fn value(self, x: u64) -> i128 {
        self.int().map(|t| t.value(x)).unwrap_or(x as i128)
    }

    pub fn tid(self, t: &Tys) -> TypeId {
        match self {
            Lane::I8 => t.i8,
            Lane::I16 => t.i16,
            Lane::I32 => t.i32,
            Lane::I64 => t.i64,
            Lane::U8 => t.u8,
            Lane::U16 => t.u16,
            Lane::U32 => t.u32,
            Lane::U64 => t.u64,
            Lane::F32 => t.f32,
            Lane::F64 => t.f64,
        }
    }
}

/// A fixed-size vector shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Shape {
    pub lane: Lane,
    pub n: u32,
}

pub fn sh(lane: Lane, n: u32) -> Shape {
    Shape { lane, n }
}

impl Shape {
    pub fn bits(self) -> u32 {
        self.lane.bits() * self.n
    }

    pub fn bytes(self) -> u32 {
        self.bits() / 8
    }

    pub fn name(self) -> String {
        format!("{}x{}", self.lane.name(), self.n)
    }
}

/// Every shape of `lane` from 16 to 512 bits with at least 2 lanes.
pub fn shapes(lane: Lane) -> Vec<Shape> {
    [2, 4, 8, 16, 32, 64]
        .into_iter()
        .map(|n| sh(lane, n))
        .filter(|s| s.bits() >= 16 && s.bits() <= 512)
        .collect()
}

/// The 128-, 256- and 512-bit shapes of `lane`.
pub fn full_shapes(lane: Lane) -> Vec<Shape> {
    vec![sh(lane, 128 / lane.bits()), sh(lane, 256 / lane.bits()), sh(lane, 512 / lane.bits())]
}

/// An operand of a case.
#[derive(Clone, Debug)]
pub enum Arg {
    /// A vector, its lane bits.
    V(Shape, Vec<u64>),
    /// A mask of lane width `.0` and `.1` lanes; bit j of `.2` = lane j.
    M(u32, u32, u64),
    /// A lane-typed scalar (bits); `true`: a constant operand.
    S(Lane, u64, bool),
    /// A `usize`; `true`: a constant operand.
    U(u64, bool),
    /// An untyped int constant (lane indices, `div_const` divisors).
    K(i128),
}

/// The result of a case.
#[derive(Clone, Copy, Debug)]
pub enum Res {
    V(Shape),
    /// Mask: lane width, lanes.
    M(u32, u32),
    S(Lane),
    /// `Option<V>`.
    Opt(Shape),
    /// `(H, H)`.
    Pair(Shape),
}

fn hex_lanes(lane: Lane, v: &[u64]) -> String {
    let w = (lane.bits() / 4) as usize;
    v.iter().map(|x| format!("{:0w$x}", x & lane.ones())).collect::<Vec<_>>().join(",")
}

impl Arg {
    fn bits(&self) -> u32 {
        match self {
            Arg::V(s, _) => s.bits(),
            Arg::M(w, n, _) => w * n,
            _ => 0,
        }
    }

    fn label(&self) -> String {
        match self {
            Arg::V(s, v) => format!("{}[{}]", s.name(), hex_lanes(s.lane, v)),
            Arg::M(w, n, m) => format!("mask{w}x{n}[{m:x}]"),
            Arg::S(l, x, k) => {
                format!("{}{}{}", if *k { "const " } else { "" }, l.name(), hex_lanes(*l, &[*x]))
            }
            Arg::U(x, k) => format!("{}usize {x}", if *k { "const " } else { "" }),
            Arg::K(k) => format!("{k}"),
        }
    }
}

impl Res {
    fn bits(&self) -> u32 {
        match self {
            Res::V(s) | Res::Opt(s) => s.bits(),
            Res::Pair(s) => 2 * s.bits(),
            Res::M(w, n) => w * n,
            Res::S(_) => 0,
        }
    }

    fn label(&self) -> String {
        match self {
            Res::V(s) => s.name(),
            Res::M(w, n) => format!("mask{w}x{n}"),
            Res::S(l) => l.name().to_string(),
            Res::Opt(s) => format!("Option<{}>", s.name()),
            Res::Pair(s) => format!("({0}, {0})", s.name()),
        }
    }

    /// The float lane width of the result, if it has float lanes.
    fn float_bits(&self) -> Option<u32> {
        let l = match self {
            Res::V(s) | Res::Opt(s) | Res::Pair(s) => s.lane,
            Res::S(l) => *l,
            Res::M(..) => return None,
        };
        l.is_float().then(|| l.bits())
    }
}

/// Emits a case's IR into its own function.
pub type Emit = Box<dyn Fn(&mut Sp, &mut FuncBuilder)>;

/// One case: a function body, the ops it exercises, the widest vector type it uses, and its trap.
pub struct Case {
    pub ops: Vec<VecOp>,
    pub bits: u32,
    pub trap: Option<PanicKind>,
    /// The trap handler prints the dump window (`Sp::set_dump`) natively; the interpreter stops at
    /// the trap, so the expected native output is the interpreter's plus the case's last `# dump`.
    pub trap_dump: bool,
    /// Rough instruction count (program size limit).
    pub cost: usize,
    pub emit: Emit,
    /// A check of the interpreter's output of the program (it panics on failure).
    pub check: Option<fn(&str)>,
}

impl Case {
    pub fn new(
        ops: &[VecOp],
        bits: u32,
        cost: usize,
        emit: impl Fn(&mut Sp, &mut FuncBuilder) + 'static,
    ) -> Case {
        Case {
            ops: ops.to_vec(),
            bits,
            trap: None,
            trap_dump: false,
            cost,
            emit: Box::new(emit),
            check: None,
        }
    }

    pub fn traps(mut self, k: PanicKind) -> Case {
        self.trap = Some(k);
        self
    }
}

/// Tuples per case function.
const PER_CASE: usize = 24;

/// Cases applying `op` with result `res` to each tuple of operands. `nan`: float result lanes
/// compare by NaN-ness.
pub fn batch(op: VecOp, res: Res, nan: bool, tuples: Vec<Vec<Arg>>) -> Vec<Case> {
    let mut out = Vec::new();
    for chunk in tuples.chunks(PER_CASE) {
        let chunk: Vec<Vec<Arg>> = chunk.to_vec();
        let bits = chunk.iter().flatten().map(Arg::bits).chain([res.bits()]).max().unwrap_or(0);
        let cost = chunk.iter().map(|t| 8 + t.len() * 6).sum();
        out.push(Case::new(&[op], bits, cost, move |sp, b| {
            for t in &chunk {
                sp.apply(b, op, res, nan, t);
            }
        }));
    }
    out
}

/// A single tuple that traps with `k`.
pub fn trap_case(op: VecOp, res: Res, args: Vec<Arg>, k: PanicKind) -> Case {
    let bits = args.iter().map(Arg::bits).chain([res.bits()]).max().unwrap_or(0);
    Case::new(&[op], bits, 20, move |sp, b| sp.apply(b, op, res, false, &args)).traps(k)
}

/// A program under construction: the harness program plus vector helpers.
pub struct Sp {
    pub p: Prog,
    mk: HashMap<TypeId, FuncId>,
    out: HashMap<TypeId, FuncId>,
    dump: FuncId,
    fill: FuncId,
    pub sys6: IntrinsicId,
    static_addr: IntrinsicId,
    pub pu64: TypeId,
}

impl Sp {
    pub fn new() -> Sp {
        let mut p = Prog::with_trap(false);
        let t = p.t;
        let sys6 = p.m.intrinsic("syscall6");
        let static_addr = p.m.intrinsic("static_addr");
        p.m.statics.push(stave_ir::Static { path: "t.DUMP_PTR".into(), ty: t.u64, init: vec![0; 8] });
        p.m.statics.push(stave_ir::Static { path: "t.DUMP_LEN".into(), ty: t.u64, init: vec![0; 8] });
        let pu64 = p.m.types.ptr(true, t.u64);
        let dump = p.decl("t.dump", &[t.usize, t.usize], None);
        let fill = p.decl("t.fill", &[t.usize, t.usize, t.u64], None);
        let mut sp = Sp { p, mk: HashMap::new(), out: HashMap::new(), dump, fill, sys6, static_addr, pu64 };
        sp.def_dump();
        sp.def_fill();
        sp.def_trap();
        sp
    }

    pub fn t(&self) -> Tys {
        self.p.t
    }

    pub fn lane_t(&self, l: Lane) -> TypeId {
        l.tid(&self.p.t)
    }

    pub fn vt(&mut self, s: Shape) -> TypeId {
        let lane = self.lane_t(s.lane);
        self.p.m.types.intern(TypeKey::Vec { lane, n: Lanes::Fixed(s.n as u8) })
    }

    pub fn mt(&mut self, w: u32, n: u32) -> TypeId {
        self.p.m.types.intern(TypeKey::Mask { width: w as u8, n: Lanes::Fixed(n as u8) })
    }

    pub fn res_t(&mut self, r: Res) -> TypeId {
        match r {
            Res::V(s) => self.vt(s),
            Res::M(w, n) => self.mt(w, n),
            Res::S(l) => self.lane_t(l),
            Res::Opt(s) => {
                let v = self.vt(s);
                self.p.option(v)
            }
            Res::Pair(s) => {
                let h = self.vt(s);
                self.p.m.types.tuple(vec![h, h])
            }
        }
    }

    /// A label line; `nan`: the following words are float lanes of that width compared by NaN-ness.
    pub fn label(&mut self, b: &mut FuncBuilder, text: &str, nan: Option<u32>) {
        let s = match nan {
            Some(w) => format!("# {text} ~{w}\n"),
            None => format!("# {text}\n"),
        };
        self.p.print_str(b, &s);
    }

    /// The 8-byte words of a vector of `s` holding `lanes` (fewer than 8 bytes: one word).
    pub fn words(s: Shape, lanes: &[u64]) -> Vec<u64> {
        assert_eq!(lanes.len(), s.n as usize, "lane count of {}", s.name());
        let bytes = stave_ir::simd::lanes_to_bytes(
            &lanes.iter().map(|x| x & s.lane.ones()).collect::<Vec<_>>(),
            s.lane.bytes(),
        );
        bytes
            .chunks(8)
            .map(|c| {
                let mut w = [0u8; 8];
                w[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(w)
            })
            .collect()
    }

    /// The raw view of a vector slot: `ref mut [u64; k]` for 8 bytes and more, else `ref mut uB`.
    fn raw_view(&mut self, bytes: u32) -> (TypeId, Option<TypeId>) {
        let t = self.p.t;
        if bytes >= 8 {
            let arr = self.p.m.types.array(t.u64, (bytes / 8) as u64);
            let pa = self.p.m.types.ref_(true, arr);
            let pe = self.p.m.types.ref_(true, t.u64);
            (pa, Some(pe))
        } else {
            let u = Lane::uint(bytes * 8).tid(&t);
            (self.p.m.types.ref_(true, u), None)
        }
    }

    /// `t.mk<V>(words…) -> V`: a vector from its 8-byte words, through a slot of type V.
    fn mk_fn(&mut self, vt: TypeId, bytes: u32) -> FuncId {
        if let Some(&f) = self.mk.get(&vt) {
            return f;
        }
        let t = self.p.t;
        let k = (bytes / 8).max(1) as usize;
        let params = vec![t.u64; k];
        let f = self.p.decl(&format!("t.mk{}", vt.0), &params, Some(vt));
        let pv = self.p.m.types.ref_(true, vt);
        let (view, elem) = self.raw_view(bytes);
        let mut b = FuncBuilder::new(&params);
        let s = b.slot(pv);
        let r = b.cast(CastKind::Exact, view, s);
        match elem {
            Some(pe) => {
                for j in 0..k {
                    let w = b.param(j).unwrap();
                    let kj = b.int(j as i128);
                    let a = b.op(Op::ElemAddr, pe, &[r.into(), kj]);
                    b.store(a, w);
                }
            }
            None => {
                let u = Lane::uint(bytes * 8).tid(&t);
                let w = b.param(0).unwrap();
                let x = b.cast(CastKind::Wrap, u, w);
                b.store(r, x);
            }
        }
        let v = b.load(vt, s);
        b.ret(Some(v.into()));
        self.p.def(f, b);
        self.mk.insert(vt, f);
        f
    }

    /// `t.out<V>(v)`: prints the 8-byte words of a vector or mask value stored in a slot.
    fn out_fn(&mut self, vt: TypeId, bytes: u32) -> FuncId {
        if let Some(&f) = self.out.get(&vt) {
            return f;
        }
        let t = self.p.t;
        let f = self.p.decl(&format!("t.out{}", vt.0), &[vt], None);
        let pv = self.p.m.types.ref_(true, vt);
        let (view, elem) = self.raw_view(bytes);
        let mut b = FuncBuilder::new(&[vt]);
        let v = b.param(0).unwrap();
        let s = b.slot(pv);
        b.store(s, v);
        let r = b.cast(CastKind::Exact, view, s);
        match elem {
            Some(pe) => {
                for j in 0..(bytes / 8) {
                    let kj = b.int(j as i128);
                    let a = b.op(Op::ElemAddr, pe, &[r.into(), kj]);
                    let w = b.load(t.u64, a);
                    b.call(self.p.print, None, &[w.into()]);
                }
            }
            None => {
                let u = Lane::uint(bytes * 8).tid(&t);
                let x = b.load(u, r);
                let w = b.cast(CastKind::Exact, t.u64, x);
                b.call(self.p.print, None, &[w.into()]);
            }
        }
        b.ret(None);
        self.p.def(f, b);
        self.out.insert(vt, f);
        f
    }

    /// A vector of shape `s` with lanes `lanes` (built at run time).
    pub fn vec(&mut self, b: &mut FuncBuilder, s: Shape, lanes: &[u64]) -> Value {
        let vt = self.vt(s);
        let f = self.mk_fn(vt, s.bytes());
        let args: Vec<Operand> = Sp::words(s, lanes).into_iter().map(|w| b.int(w as i128)).collect();
        b.call(f, Some(vt), &args).unwrap()
    }

    /// A mask of lane width `w`, `n` lanes, lane j = bit j of `bits` (`vec.ne` of a 0/1 vector).
    pub fn mask(&mut self, b: &mut FuncBuilder, w: u32, n: u32, bits: u64) -> Value {
        let s = sh(Lane::uint(w), n);
        let lanes: Vec<u64> = (0..n).map(|j| (bits >> j) & 1).collect();
        let v = self.vec(b, s, &lanes);
        let z = self.vec(b, s, &vec![0; n as usize]);
        let mt = self.mt(w, n);
        b.op(Op::Vec(VecOp::Ne), mt, &[v.into(), z.into()])
    }

    /// Prints a vector of shape `s`.
    pub fn out(&mut self, b: &mut FuncBuilder, v: Value, s: Shape) {
        let vt = self.vt(s);
        let f = self.out_fn(vt, s.bytes());
        b.call(f, None, &[v.into()]);
    }

    /// Prints a mask through `mask.to_vec` (all-ones lanes where set).
    pub fn out_mask(&mut self, b: &mut FuncBuilder, m: Value, w: u32, n: u32) {
        let s = sh(Lane::uint(w), n);
        let vt = self.vt(s);
        let v = b.op(Op::Vec(VecOp::ToVec), vt, &[m.into()]);
        self.out(b, v, s);
    }

    /// Prints a lane-typed scalar (ints zero-extended, floats as their bits).
    pub fn out_scalar(&mut self, b: &mut FuncBuilder, v: Value, l: Lane) {
        let t = self.p.t;
        if l.is_float() {
            let ty = self.lane_t(l);
            self.p.print(b, v.into(), ty);
            return;
        }
        let u = self.lane_t(Lane::uint(l.bits()));
        let r = b.cast(CastKind::Wrap, u, v);
        let x = b.cast(CastKind::Exact, t.u64, r);
        b.call(self.p.print, None, &[x.into()]);
    }

    /// Prints an `Option<V>`: its tag, then the payload if `Some`.
    pub fn out_opt(&mut self, b: &mut FuncBuilder, r: Value, s: Shape) {
        let t = self.p.t;
        let tag = b.op(Op::EnumTag, t.u32, &[r.into()]);
        let w = b.cast(CastKind::Exact, t.u64, tag);
        b.call(self.p.print, None, &[w.into()]);
        let some = b.create_block();
        let join = b.create_block();
        let one = b.int(1);
        let is = b.cmp(Op::Eq, t.bool, tag, one);
        b.cbr(is, some, &[], join, &[]);
        b.switch_to(some);
        let vt = self.vt(s);
        let (v1, f0) = (b.int(1), b.int(0));
        let pl = b.op(Op::EnumGet, vt, &[r.into(), v1, f0]);
        self.out(b, pl, s);
        b.br(join, &[]);
        b.switch_to(join);
    }

    /// Prints a result of kind `r`.
    pub fn out_res(&mut self, b: &mut FuncBuilder, v: Value, r: Res) {
        match r {
            Res::V(s) => self.out(b, v, s),
            Res::M(w, n) => self.out_mask(b, v, w, n),
            Res::S(l) => self.out_scalar(b, v, l),
            Res::Opt(s) => self.out_opt(b, v, s),
            Res::Pair(s) => {
                let h = self.vt(s);
                for k in 0..2 {
                    let kk = b.int(k);
                    let x = b.op(Op::Get, h, &[v.into(), kk]);
                    self.out(b, x, s);
                }
            }
        }
    }

    /// A scalar operand: a constant, or the constant through an opaque call.
    pub fn scalar(&mut self, b: &mut FuncBuilder, l: Lane, x: u64, konst: bool) -> Operand {
        let c = match l {
            Lane::F32 => b.f32(f32::from_bits(x as u32)),
            Lane::F64 => b.f64(f64::from_bits(x)),
            _ => b.int(l.value(x & l.ones())),
        };
        if konst {
            return c;
        }
        let ty = self.lane_t(l);
        self.p.opaque(b, c, ty).into()
    }

    pub fn usize(&mut self, b: &mut FuncBuilder, x: u64, konst: bool) -> Operand {
        let c = b.int(x as i128);
        if konst {
            return c;
        }
        let us = self.p.t.usize;
        self.p.opaque(b, c, us).into()
    }

    pub fn arg(&mut self, b: &mut FuncBuilder, a: &Arg) -> Operand {
        match a {
            Arg::V(s, v) => self.vec(b, *s, v).into(),
            Arg::M(w, n, m) => self.mask(b, *w, *n, *m).into(),
            Arg::S(l, x, k) => self.scalar(b, *l, *x, *k),
            Arg::U(x, k) => self.usize(b, *x, *k),
            Arg::K(k) => b.int(*k),
        }
    }

    /// Label, operands, `op`, and the printed result.
    pub fn apply(&mut self, b: &mut FuncBuilder, op: VecOp, res: Res, nan: bool, args: &[Arg]) {
        let text = format!(
            "{} {} -> {}",
            op.name(),
            args.iter().map(Arg::label).collect::<Vec<_>>().join(" "),
            res.label()
        );
        let nan = if nan { res.float_bits() } else { None };
        self.label(b, &text, nan);
        let ops: Vec<Operand> = args.iter().map(|a| self.arg(b, a)).collect();
        let rt = self.res_t(res);
        let r = b.op(Op::Vec(op), rt, &ops);
        self.out_res(b, r, res);
    }

    fn static_ptr(&mut self, b: &mut FuncBuilder, k: i128) -> Value {
        let kk = b.int(k);
        b.intrinsic(self.static_addr, Some(self.pu64), &[kk]).unwrap()
    }

    /// The window the trap handler dumps natively.
    pub fn set_dump(&mut self, b: &mut FuncBuilder, addr: Operand, len: Operand) {
        let t = self.p.t;
        let p0 = self.static_ptr(b, 0);
        let a = b.cast(CastKind::Exact, t.u64, addr);
        b.store(p0, a);
        let p1 = self.static_ptr(b, 1);
        let n = b.cast(CastKind::Exact, t.u64, len);
        b.store(p1, n);
    }

    /// Prints `n` bytes at address `addr` as little-endian 8-byte words (the last one partial).
    pub fn dump(&mut self, b: &mut FuncBuilder, addr: Operand, n: Operand) {
        b.call(self.dump, None, &[addr, n]);
    }

    /// Fills `n` bytes at `addr` with a pattern derived from `seed`.
    pub fn fill(&mut self, b: &mut FuncBuilder, addr: Operand, n: Operand, seed: u64) {
        let s = b.int(seed as i128);
        b.call(self.fill, None, &[addr, n, s]);
    }

    /// `mmap` of `pages + 1` pages whose last page is then unmapped (natively; the interpreter
    /// keeps whole regions): returns the base address. `pages` bytes from the base end at a hole.
    pub fn region(&mut self, b: &mut FuncBuilder, pages: u64) -> Value {
        let t = self.p.t;
        let len = (pages + 1) * 4096;
        let args = [
            b.int(9),
            b.int(0),
            b.int(len as i128),
            b.int(3),
            b.int(0x22),
            b.int(u64::MAX as i128),
            b.int(0),
        ];
        let r = b.intrinsic(self.sys6, Some(t.isize), &args).unwrap();
        let base = b.cast(CastKind::Wrap, t.usize, r);
        let off = b.int((pages * 4096) as i128);
        let hole = b.binary(Op::AddWrap, t.usize, base, off);
        let args = [b.int(11), hole.into(), b.int(4096), b.int(0), b.int(0), b.int(0), b.int(0)];
        b.intrinsic(self.sys6, Some(t.isize), &args);
        base
    }

    pub fn unmap(&mut self, b: &mut FuncBuilder, base: Value, pages: u64) {
        let t = self.p.t;
        let args =
            [b.int(11), base.into(), b.int((pages * 4096) as i128), b.int(0), b.int(0), b.int(0), b.int(0)];
        b.intrinsic(self.sys6, Some(t.isize), &args);
    }

    /// `base + off` as a `usize`.
    pub fn addr(&mut self, b: &mut FuncBuilder, base: Value, off: u64) -> Value {
        let t = self.p.t;
        let k = b.int(off as i128);
        b.binary(Op::AddWrap, t.usize, base, k)
    }

    fn def_dump(&mut self) {
        let t = self.p.t;
        let pb = self.p.m.types.ptr(true, t.u8);
        let mut b = FuncBuilder::new(&[t.usize, t.usize]);
        let (p, n) = (b.param(0).unwrap(), b.param(1).unwrap());
        let (head, hv) = b.create_block_with(&[t.usize, t.u64]);
        let body = b.create_block();
        let word = b.create_block();
        let exit = b.create_block();
        let last = b.create_block();
        let done = b.create_block();
        let (z, z2) = (b.int(0), b.int(0));
        b.br(head, &[z, z2]);
        b.switch_to(head);
        let (i, acc) = (hv[0], hv[1]);
        let c = b.cmp(Op::Lt, t.bool, i, n);
        b.cbr(c, body, &[], exit, &[]);
        b.switch_to(body);
        let a = b.binary(Op::AddWrap, t.usize, p, i);
        let pa = b.cast(CastKind::Exact, pb, a);
        let by = b.load(t.u8, pa);
        let bw = b.cast(CastKind::Exact, t.u64, by);
        let c7 = b.int(7);
        let lo = b.binary(Op::And, t.usize, i, c7);
        let lo64 = b.cast(CastKind::Exact, t.u64, lo);
        let c8 = b.int(8);
        let sh = b.binary(Op::MulWrap, t.u64, lo64, c8);
        let shifted = b.binary(Op::ShlWrap, t.u64, bw, sh);
        let acc2 = b.binary(Op::Or, t.u64, acc, shifted);
        let one = b.int(1);
        let i2 = b.binary(Op::AddWrap, t.usize, i, one);
        let c7b = b.int(7);
        let full = b.cmp(Op::Eq, t.bool, lo, c7b);
        b.cbr(full, word, &[], head, &[i2.into(), acc2.into()]);
        b.switch_to(word);
        b.call(self.p.print, None, &[acc2.into()]);
        let z3 = b.int(0);
        b.br(head, &[i2.into(), z3]);
        b.switch_to(exit);
        let c7c = b.int(7);
        let rest = b.binary(Op::And, t.usize, n, c7c);
        let z4 = b.int(0);
        let partial = b.cmp(Op::Ne, t.bool, rest, z4);
        b.cbr(partial, last, &[], done, &[]);
        b.switch_to(last);
        b.call(self.p.print, None, &[acc.into()]);
        b.br(done, &[]);
        b.switch_to(done);
        b.ret(None);
        self.p.def(self.dump, b);
    }

    fn def_fill(&mut self) {
        let t = self.p.t;
        let pb = self.p.m.types.ptr(true, t.u8);
        let mut b = FuncBuilder::new(&[t.usize, t.usize, t.u64]);
        let (p, n, seed) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
        let (head, hv) = b.create_block_with(&[t.usize]);
        let body = b.create_block();
        let exit = b.create_block();
        let z = b.int(0);
        b.br(head, &[z]);
        b.switch_to(head);
        let i = hv[0];
        let c = b.cmp(Op::Lt, t.bool, i, n);
        b.cbr(c, body, &[], exit, &[]);
        b.switch_to(body);
        let i64_ = b.cast(CastKind::Exact, t.u64, i);
        let k = b.int(0x9d);
        let m = b.binary(Op::MulWrap, t.u64, i64_, k);
        let x = b.binary(Op::AddWrap, t.u64, m, seed);
        let c5 = b.int(5);
        let hi = b.binary(Op::Shr, t.u64, i64_, c5);
        let y = b.binary(Op::Xor, t.u64, x, hi);
        let by = b.cast(CastKind::Wrap, t.u8, y);
        let a = b.binary(Op::AddWrap, t.usize, p, i);
        let pa = b.cast(CastKind::Exact, pb, a);
        b.store(pa, by);
        let one = b.int(1);
        let i2 = b.binary(Op::AddWrap, t.usize, i, one);
        b.br(head, &[i2.into()]);
        b.switch_to(exit);
        b.ret(None);
        self.p.def(self.fill, b);
    }

    /// `runtime.panic.trap`: prints the dump window if one is set, then exits `100 + kind`.
    fn def_trap(&mut self) {
        let t = self.p.t;
        let f = self.p.decl("runtime.panic.trap", &[t.u32, t.u32], None);
        let mut b = FuncBuilder::new(&[t.u32, t.u32]);
        let k = b.param(0).unwrap();
        let p1 = self.static_ptr(&mut b, 1);
        let n = b.load(t.u64, p1);
        let dumpb = b.create_block();
        let exit = b.create_block();
        let z = b.int(0);
        let has = b.cmp(Op::Ne, t.bool, n, z);
        b.cbr(has, dumpb, &[], exit, &[]);
        b.switch_to(dumpb);
        self.p.print_str(&mut b, "# trap dump\n");
        let p0 = self.static_ptr(&mut b, 0);
        let a = b.load(t.u64, p0);
        let au = b.cast(CastKind::Exact, t.usize, a);
        let nu = b.cast(CastKind::Exact, t.usize, n);
        b.call(self.dump, None, &[au.into(), nu.into()]);
        b.br(exit, &[]);
        b.switch_to(exit);
        let kk = b.cast(CastKind::Exact, t.usize, k);
        let c100 = b.int(100);
        let code = b.binary(Op::AddWrap, t.usize, kk, c100);
        let nr = b.int(231);
        b.intrinsic(self.p.sys1, Some(t.isize), &[nr, code.into()]);
        b.unreachable();
        self.p.def(f, b);
        self.p.trap = Some(f);
    }
}

/// The widest vector or mask type of the module, in bits.
fn max_bits(m: &stave_ir::Module) -> u32 {
    let mut out = 0;
    for k in m.types.keys() {
        let b = match k {
            TypeKey::Vec { lane, n: Lanes::Fixed(n) } => match m.types.key(*lane) {
                Some(TypeKey::Int { bits, .. }) => bits.width() as u32 * *n as u32,
                Some(TypeKey::Float(f)) => f.width() as u32 * *n as u32,
                _ => 0,
            },
            TypeKey::Mask { width, n: Lanes::Fixed(n) } => *width as u32 * *n as u32,
            _ => 0,
        };
        out = out.max(b);
    }
    out
}

/// One way to compile a program: the module baseline, and the `needs` of its test functions.
#[derive(Clone, Copy, Debug)]
struct Mode {
    name: &'static str,
    /// The module baseline is v3 (else v2).
    v3: bool,
    /// The module baseline is v4.
    v4: bool,
    /// The features every function but the runtime's and `main` gets on top of the baseline:
    /// those functions are called from baseline code and call it back (`t.print_u64`).
    needs: &'static [Feature],
}

const MODES: [Mode; 9] = [
    Mode { name: "V2", v3: false, v4: false, needs: &[] },
    Mode { name: "v2+bmi2", v3: false, v4: false, needs: &[Feature::Bmi2] },
    Mode { name: "V3", v3: true, v4: false, needs: &[] },
    Mode {
        name: "v2+v3",
        v3: false,
        v4: false,
        needs: &[Feature::Avx2, Feature::Fma, Feature::Bmi2, Feature::F16c],
    },
    Mode { name: "v2+avx2", v3: false, v4: false, needs: &[Feature::Avx2] },
    Mode { name: "v2+avx2+fma", v3: false, v4: false, needs: &[Feature::Avx2, Feature::Fma] },
    Mode { name: "v2+avx", v3: false, v4: false, needs: &[Feature::Avx] },
    Mode { name: "V4", v3: false, v4: true, needs: &[] },
    Mode { name: "v2+v4", v3: false, v4: false, needs: &V4_NEEDS },
];

impl Mode {
    /// The test functions are compiled in K mode.
    fn k(self) -> bool {
        self.v4 || self.needs == V4_NEEDS
    }
}

/// The modes a program whose widest vector has `bits` bits runs under. `fma`: it has a
/// `vec.fma`, the one op `fma` on top of `avx2` changes; `mask_bits`: it has an op that takes
/// the bits of a mask, which `bmi2` changes; `kok`: its cases compile in K mode.
fn modes(bits: u32, fma: bool, mask_bits: bool, kok: bool) -> Vec<Mode> {
    MODES
        .into_iter()
        .filter(|m| match m.name {
            "V2" => bits <= 128,
            "v2+bmi2" => bits <= 128 && mask_bits,
            "v2+avx2+fma" => fma && bits <= 256,
            "V4" | "v2+v4" => kok,
            _ => bits <= 256,
        })
        .collect()
}

/// Ops without a K-mode lowering: every case of them is unsupported in a function compiled for
/// the AVX-512 set. (`vec.shl`, `vec.shl.wrap` and `vec.shr` lack only the form on 8-bit lanes
/// with a vector amount.)
pub const K_UNSUPPORTED: [VecOp; 14] = [
    VecOp::MulSat,
    VecOp::CheckedMul,
    VecOp::DivConst,
    VecOp::Popcount,
    VecOp::Clz,
    VecOp::ScanAdd,
    VecOp::ScanAddWrap,
    VecOp::Permute,
    VecOp::Lookup,
    VecOp::Compress,
    VecOp::Expand,
    VecOp::StoreStream,
    VecOp::Gather,
    VecOp::Scatter,
];

/// Whether the functions of `c` (its own and the helpers it calls) compile in K mode (debug
/// and release agree).
fn k_ok(c: &Case) -> bool {
    let b = build("probe".into(), &[c]);
    let mut m = b.sp.p.m;
    for f in &mut m.funcs {
        if !baseline_fn(&f.path) {
            f.features = FeatureSet::from_features(V4_NEEDS);
        }
    }
    let mut ok = [true; 2];
    for (release, slot) in [false, true].into_iter().zip(&mut ok) {
        let mut cx = bench_ctx(&m);
        cx.release = release;
        for (i, f) in m.funcs.iter().enumerate() {
            if f.body.is_none() || baseline_fn(&f.path) {
                continue;
            }
            match compile_func(&cx, FuncId(i as u32)) {
                Ok(_) => {}
                Err(Error::Unsupported { .. }) => *slot = false,
                Err(e) => panic!("K mode: {:?}: {}: {e}", c.ops, f.path),
            }
        }
    }
    assert_eq!(ok[0], ok[1], "K mode: {:?}: debug and release disagree", c.ops);
    ok[0]
}

/// Functions that stay at the module baseline under every mode.
pub fn baseline_fn(path: &str) -> bool {
    path.starts_with("runtime.") || matches!(path, "t.main" | "t.write" | "t.print_u64" | "t.dump" | "t.fill")
}

/// The module of `p` under `mode`. With `avx` alone a 256-bit integer op is unsupported: in
/// that mode each function of a 256-bit program that does not compile is given `avx2`, the
/// others run with `avx` alone. Also returns how many functions were tried and how many needed
/// `avx2`.
fn module_for(p: &Prog, mode: Mode, release: bool) -> (stave_ir::Module, usize, usize) {
    let mut m = p.m.clone();
    let needs = FeatureSet::from_features(mode.needs.iter().copied());
    for f in &mut m.funcs {
        if !baseline_fn(&f.path) {
            f.features = needs;
        }
    }
    let (mut tried, mut raised) = (0, 0);
    if mode.needs == [Feature::Avx] && max_bits(&m) > 128 {
        let mut raise = Vec::new();
        {
            let mut cx = bench_ctx(&m);
            cx.release = release;
            for (i, f) in m.funcs.iter().enumerate() {
                if f.body.is_none() || baseline_fn(&f.path) {
                    continue;
                }
                tried += 1;
                match compile_func(&cx, FuncId(i as u32)) {
                    Ok(_) => {}
                    Err(Error::Unsupported { .. }) => raise.push(i),
                    Err(e) => panic!("{}: {e}", f.path),
                }
            }
        }
        raised = raise.len();
        for i in raise {
            m.funcs[i].features = FeatureSet::of(Feature::Avx2);
        }
    }
    (m, tried, raised)
}

/// Program size limit (sum of case costs).
const PROGRAM_COST: usize = 6000;

/// A built program and what it must do.
struct Built {
    sp: Sp,
    name: String,
    trap: Option<PanicKind>,
    trap_dump: bool,
    /// A case has a `vec.fma`.
    fma: bool,
    /// A case has an op that reads the bits of a mask.
    mask_bits: bool,
    /// Every case compiles in K mode.
    kok: bool,
    checks: Vec<fn(&str)>,
}

fn build(name: String, cases: &[&Case]) -> Built {
    let mut sp = Sp::new();
    let mut fns = Vec::new();
    for (k, c) in cases.iter().enumerate() {
        let f = sp.p.decl(&format!("t.case{k}"), &[], None);
        let mut b = FuncBuilder::new(&[]);
        (c.emit)(&mut sp, &mut b);
        if b.is_open() {
            b.ret(None);
        }
        sp.p.def(f, b);
        fns.push(f);
    }
    sp.p.main(|_, b| {
        for f in fns {
            b.call(f, None, &[]);
        }
    });
    let bits = cases.iter().map(|c| c.bits).max().unwrap_or(0);
    let got = max_bits(&sp.p.m);
    assert!(got <= bits.max(128), "{name}: uses {got}-bit vectors, cases declare {bits}");
    let trap = cases.iter().find_map(|c| c.trap);
    let trap_dump = cases.iter().any(|c| c.trap_dump);
    let checks = cases.iter().filter_map(|c| c.check).collect();
    let fma = cases.iter().any(|c| c.ops.contains(&VecOp::Fma));
    let mask_ops = [VecOp::ToBits, VecOp::Compress, VecOp::Expand, VecOp::LoadMasked, VecOp::StoreMasked];
    let mask_bits = cases.iter().any(|c| c.ops.iter().any(|op| mask_ops.contains(op)));
    Built { sp, name, trap, trap_dump, fma, mask_bits, kok: false, checks }
}

/// Groups cases into programs: trap-free cases by level set (128, 256, 512 bits) and by whether
/// they compile in K mode, at most [`PROGRAM_COST`] each; every trapping case alone. A 512-bit
/// case that K mode lacks is in no program.
fn programs(test: &str, cases: &[Case]) -> Vec<Built> {
    let kok: Vec<bool> = cases.iter().map(k_ok).collect();
    let level = |c: &Case| match c.bits {
        0..=128 => 0,
        129..=256 => 1,
        _ => 2,
    };
    let mut out = Vec::new();
    let push = |cur: &[&Case], k: bool, out: &mut Vec<Built>| {
        let mut b = build(format!("{test}-{}", out.len()), cur);
        b.kok = k;
        out.push(b);
    };
    for wide in [0, 1, 2] {
        for k in [false, true] {
            if wide == 2 && !k {
                continue;
            }
            let sel: Vec<&Case> = cases
                .iter()
                .zip(&kok)
                .filter(|(c, ok)| c.trap.is_none() && level(c) == wide && **ok == k)
                .map(|(c, _)| c)
                .collect();
            let mut cur: Vec<&Case> = Vec::new();
            let mut cost = 0;
            for c in sel {
                if !cur.is_empty() && cost + c.cost > PROGRAM_COST {
                    push(&cur, k, &mut out);
                    cur.clear();
                    cost = 0;
                }
                cost += c.cost;
                cur.push(c);
            }
            if !cur.is_empty() {
                push(&cur, k, &mut out);
            }
        }
    }
    for (c, &k) in cases.iter().zip(&kok).filter(|(c, k)| c.trap.is_some() && (c.bits <= 256 || **k)) {
        push(&[c], k, &mut out);
    }
    out
}

/// Compiles, checks the instructions of every function against its features, and runs natively
/// (`leg`: on the `v4` leg); a signal is reported as exit status `-signal`.
fn native_run(
    m: &stave_ir::Module,
    main: Option<FuncId>,
    name: &str,
    opts: &Options,
    leg: Option<&V4Leg>,
) -> Option<RunOut> {
    let opts = Options { main, ..opts.clone() };
    let img = compile_module(m, &opts).unwrap_or_else(|e| panic!("{name}: codegen: {e}"));
    super::features::check_image(m, &img, opts.features, name);
    if !native_host() {
        return None;
    }
    let path = tmp(name);
    img.write_file(&path).unwrap();
    let out = loop {
        let mut cmd = match leg {
            Some(l) => v4_command(l, &path),
            None => std::process::Command::new(&*path),
        };
        match cmd.output() {
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
            r => break r.unwrap(),
        }
    };
    use std::os::unix::process::ExitStatusExt;
    let code = out.status.code().unwrap_or_else(|| -out.status.signal().unwrap_or(0));
    Some(RunOut { stdout: out.stdout, stderr: out.stderr, code })
}

fn is_nan(w: u32, x: u64) -> bool {
    match w {
        32 => {
            let x = x as u32;
            x & 0x7f80_0000 == 0x7f80_0000 && x & 0x007f_ffff != 0
        }
        _ => x & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 && x & 0x000f_ffff_ffff_ffff != 0,
    }
}

/// Equal words, or every float lane of width `nan` either equal or NaN on both sides.
fn same_word(a: &str, b: &str, nan: Option<u32>) -> bool {
    if a == b {
        return true;
    }
    let (Some(w), Ok(x), Ok(y)) = (nan, u64::from_str_radix(a, 16), u64::from_str_radix(b, 16)) else {
        return false;
    };
    let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    (0..64 / w).all(|k| {
        let (p, q) = ((x >> (k * w)) & mask, (y >> (k * w)) & mask);
        p == q || (is_nan(w, p) && is_nan(w, q))
    })
}

/// Compares native output with the expected (interpreter) output line by line.
fn compare(what: &str, want: &[u8], want_code: i32, got: &RunOut) {
    let (w, g) = (String::from_utf8_lossy(want), String::from_utf8_lossy(&got.stdout));
    let (wl, gl): (Vec<&str>, Vec<&str>) = (w.lines().collect(), g.lines().collect());
    let mut label = "(before the first label)";
    let mut nan = None;
    let stderr = String::from_utf8_lossy(&got.stderr);
    let fail = |label: &str, line: usize, msg: String| -> ! {
        panic!(
            "{what}: {label}\n  line {line}: {msg}\n  exit native {} interp {want_code}\n  native stderr: {}",
            got.code,
            stderr.chars().take(2000).collect::<String>()
        )
    };
    for i in 0..wl.len().max(gl.len()) {
        match (wl.get(i), gl.get(i)) {
            (Some(a), Some(b)) => {
                if let Some(l) = a.strip_prefix("# ") {
                    label = l;
                    nan = if l.ends_with(" ~32") {
                        Some(32)
                    } else if l.ends_with(" ~64") {
                        Some(64)
                    } else {
                        None
                    };
                    if a != b {
                        fail(label, i + 1, format!("native `{b}`, interp label"));
                    }
                } else if !same_word(a, b, nan) {
                    fail(label, i + 1, format!("native {b}, interp {a}"));
                }
            }
            (Some(a), None) => fail(label, i + 1, format!("native output ends; interp {a}")),
            (None, Some(b)) => fail(label, i + 1, format!("native prints {b} past the interp output")),
            (None, None) => {}
        }
    }
    if got.code != want_code {
        fail(label, wl.len(), format!("exit status native {} interp {want_code}", got.code));
    }
}

/// The expected native output of a `trap_dump` program: the interpreter's, plus its last `# dump`
/// block repeated by the trap handler.
fn with_trap_dump(want: &[u8]) -> Vec<u8> {
    let s = String::from_utf8_lossy(want).into_owned();
    let at = s.rfind("# dump\n").unwrap_or_else(|| panic!("trap-dump program prints no `# dump`"));
    let mut out = want.to_vec();
    out.extend_from_slice(b"# trap dump\n");
    out.extend_from_slice(&s.as_bytes()[at + "# dump\n".len()..]);
    out
}

fn run_built(b: &Built, stats: &Stats) {
    let p = &b.sp.p;
    if let Err(es) = stave_ir::verify_module(&p.m) {
        let msgs: Vec<String> = es.iter().take(5).map(|e| format!("{e:?}")).collect();
        panic!("{}: IR does not verify:\n{}", b.name, msgs.join("\n"));
    }
    let want = interp(p);
    let want_code = match b.trap {
        Some(k) => 100 + kind_index(k),
        None => 0,
    };
    if want.code != want_code {
        let s = String::from_utf8_lossy(&want.stdout);
        let last = s.lines().rev().find(|l| l.starts_with("# ")).unwrap_or("");
        panic!("{}: interpreter exit {} (expected {want_code}); last label: {last}", b.name, want.code);
    }
    for c in &b.checks {
        c(&String::from_utf8_lossy(&want.stdout));
    }
    let labels = want.stdout.split(|&c| c == b'\n').filter(|l| l.starts_with(b"# ")).count();
    let bits = max_bits(&p.m);
    stats.add(&stats.labels, labels);
    stats.add(if bits <= 128 { &stats.both } else { &stats.v3 }, 1);
    let expect = if b.trap_dump { with_trap_dump(&want.stdout) } else { want.stdout.clone() };
    for mode in modes(bits, b.fma, b.mask_bits, b.kok) {
        // the AVX-512 modes run where the `v4` leg does
        let leg = if mode.k() {
            match v4_leg() {
                Some(l) => Some(l),
                None => {
                    stats.add(&stats.no_v4, 1);
                    continue;
                }
            }
        } else {
            None
        };
        for (build, base) in [("debug", Options::default()), ("release", release_options())] {
            let what = format!("{} [{} {build}]", b.name, mode.name);
            let (m, tried, raised) = module_for(p, mode, base.release);
            stats.add(&stats.avx_fns, tried);
            stats.add(&stats.raised, raised);
            let features = if mode.v4 {
                v4()
            } else if mode.v3 {
                v3()
            } else {
                v2()
            };
            let opts = Options { features, ..base };
            let name = format!("{}-{}-{build}", b.name, mode.name);
            match native_run(&m, p.main, &name, &opts, leg.as_ref()) {
                Some(got) => {
                    compare(&what, &expect, want_code, &got);
                    stats.add(&stats.runs, 1);
                }
                None => eprintln!("SKIP native run of {what}: host is not x86-64 Linux"),
            }
        }
    }
}

/// Counts for `STAVE_TEST_VERBOSE`.
#[derive(Default)]
struct Stats {
    /// Programs of 128-bit vectors (every mode).
    both: AtomicUsize,
    /// Programs with 256-bit vectors (no v2 baseline without `needs`).
    v3: AtomicUsize,
    /// Functions of 256-bit programs compiled with `avx` alone, and those that needed `avx2`.
    avx_fns: AtomicUsize,
    raised: AtomicUsize,
    /// Label lines (input tuples) printed by the interpreter.
    labels: AtomicUsize,
    runs: AtomicUsize,
    /// AVX-512 mode runs skipped: the `v4` leg does not run here.
    no_v4: AtomicUsize,
}

impl Stats {
    fn add(&self, c: &AtomicUsize, k: usize) {
        c.fetch_add(k, Ordering::Relaxed);
    }
}

/// Builds and runs every program of `cases` (several worker threads).
pub fn run(test: &str, cases: Vec<Case>) {
    let t0 = std::time::Instant::now();
    let ncases = cases.len();
    let progs = programs(test, &cases);
    drop(cases);
    let next = AtomicUsize::new(0);
    let stats = Stats::default();
    let progs: Vec<Mutex<Option<Built>>> = progs.into_iter().map(|b| Mutex::new(Some(b))).collect();
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(slot) = progs.get(k) else { break };
                    let b = slot.lock().unwrap().take().unwrap();
                    run_built(&b, &stats);
                }
            });
        }
    });
    let no_v4 = stats.no_v4.load(Ordering::Relaxed);
    if no_v4 > 0 {
        eprintln!("SKIP {test}: {no_v4} AVX-512 runs: no CPU with the AVX-512 set and no STAVE_SDE");
    }
    if std::env::var_os("STAVE_TEST_VERBOSE").is_some() {
        eprintln!(
            "{test}: {ncases} case fns, programs {} of 128 bits and {} of 256, with avx alone {} of {} fns of 256-bit programs need avx2, {} labels, {} native runs, {:.1?}",
            stats.both.into_inner(),
            stats.v3.into_inner(),
            stats.raised.into_inner(),
            stats.avx_fns.into_inner(),
            stats.labels.into_inner(),
            stats.runs.into_inner(),
            t0.elapsed()
        );
    }
}

/// Ops the tables cannot exercise for int or f32/f64 lanes, with the reason.
const UNSUPPORTED: &[(VecOp, &str)] = &[];

fn all_cases() -> Vec<(&'static str, Vec<Case>)> {
    vec![
        ("float_arith", lanewise::float_arith()),
        ("int_arith", lanewise::int_arith()),
        ("int_div_shift", lanewise::int_div_shift()),
        ("compares", lanewise::compares()),
        ("reductions", lanewise::reductions()),
        ("casts", reshape::casts()),
        ("reshape", reshape::reshape()),
        ("compress", reshape::compress().into_iter().chain(memory::compress_store()).collect()),
        ("masks", reshape::masks()),
        ("memory", memory::memory()),
        ("traps", lanewise::traps().into_iter().chain(memory::traps()).collect()),
        ("plumbing", plumbing::plumbing()),
        ("registers", regs::registers()),
    ]
}

#[test]
fn coverage() {
    let mut seen = BTreeSet::new();
    let mut wide = BTreeSet::new();
    for (_, cases) in all_cases() {
        for c in &cases {
            for &op in &c.ops {
                seen.insert(op.name());
                if c.bits > 128 {
                    wide.insert(op.name());
                }
            }
        }
    }
    let mut missing = Vec::new();
    for op in VecOp::ALL {
        let listed = UNSUPPORTED.iter().any(|(o, _)| *o == op);
        if !seen.contains(op.name()) && !listed {
            missing.push(op.name());
        }
        if !wide.contains(op.name()) && !listed {
            missing.push(op.name());
        }
    }
    assert!(missing.is_empty(), "ops without (256-bit) cases: {missing:?}");
}

/// K mode covers what it claims: every op outside [`K_UNSUPPORTED`] has a 512-bit case that
/// compiles in K mode, and every op in the list has a case that does not (a case whose op has
/// no reader emits nothing and compiles anyway).
#[test]
fn k_coverage() {
    let mut wide = BTreeSet::new();
    let mut refused = BTreeSet::new();
    let mut lacking = BTreeSet::new();
    for (_, cases) in all_cases() {
        for c in &cases {
            let ok = k_ok(c);
            let listed = c.ops.iter().any(|o| K_UNSUPPORTED.contains(o));
            for &op in &c.ops {
                if ok && c.bits > 256 {
                    wide.insert(op.name());
                }
                if !ok && c.ops.len() == 1 {
                    refused.insert(op.name());
                }
            }
            if !ok && !listed {
                lacking.insert(format!("{:?}", c.ops.iter().map(|o| o.name()).collect::<Vec<_>>()));
            }
        }
    }
    let missing: Vec<&str> = VecOp::ALL
        .iter()
        .filter(|op| !K_UNSUPPORTED.contains(op) && !wide.contains(op.name()))
        .map(|op| op.name())
        .collect();
    assert!(missing.is_empty(), "ops without a 512-bit K-mode case: {missing:?}");
    let stale: Vec<&str> =
        K_UNSUPPORTED.iter().filter(|op| !refused.contains(op.name())).map(|op| op.name()).collect();
    assert!(stale.is_empty(), "ops listed as unsupported in K mode that compile: {stale:?}");
    // the one form of a supported op that K mode lacks: 8-bit lanes shifted by a vector
    for l in &lacking {
        assert!(l.contains("vec.sh"), "cases of supported ops that do not compile in K mode: {lacking:?}");
    }
}

#[test]
fn float_arith() {
    run("simd-float", lanewise::float_arith());
}

#[test]
fn int_arith() {
    run("simd-int", lanewise::int_arith());
}

#[test]
fn int_div_shift() {
    run("simd-divshift", lanewise::int_div_shift());
}

#[test]
fn compares() {
    run("simd-cmp", lanewise::compares());
}

#[test]
fn reductions() {
    run("simd-reduce", lanewise::reductions());
}

#[test]
fn casts() {
    run("simd-cast", reshape::casts());
}

#[test]
fn reshape() {
    run("simd-reshape", reshape::reshape());
}

#[test]
fn compress() {
    run("simd-compress", reshape::compress().into_iter().chain(memory::compress_store()).collect());
}

#[test]
fn masks() {
    run("simd-mask", reshape::masks());
}

#[test]
fn memory() {
    run("simd-mem", memory::memory());
}

#[test]
fn traps() {
    run("simd-trap", lanewise::traps().into_iter().chain(memory::traps()).collect());
}

#[test]
fn plumbing() {
    run("simd-plumb", plumbing::plumbing());
}

#[test]
fn registers() {
    run("simd-regs", regs::registers());
}

/// The comparator: words under a NaN-insensitive label match when both lanes are NaN, everything
/// else compares exactly.
#[test]
fn comparator() {
    let out = |s: &str, code: i32| RunOut { stdout: s.as_bytes().to_vec(), stderr: Vec::new(), code };
    let want = "# vec.add f32x4 ~32\n7fc000003f800000\n# vec.neg f32x4\n7fa5a5a500000000\n";
    compare("same", want.as_bytes(), 0, &out(want, 0));
    let quiet = "# vec.add f32x4 ~32\nffc123453f800000\n# vec.neg f32x4\n7fa5a5a500000000\n";
    compare("NaN payload and sign", want.as_bytes(), 0, &out(quiet, 0));
    let fails = |got: &'static str, code: i32| {
        std::panic::catch_unwind(move || compare("t", want.as_bytes(), 0, &out(got, code))).is_err()
    };
    assert!(fails("# vec.add f32x4 ~32\n7fc000003f800001\n# vec.neg f32x4\n7fa5a5a500000000\n", 0));
    assert!(fails("# vec.add f32x4 ~32\n000000003f800000\n# vec.neg f32x4\n7fa5a5a500000000\n", 0));
    assert!(fails("# vec.add f32x4 ~32\n7fc000003f800000\n# vec.neg f32x4\n7fe5a5a500000000\n", 0));
    assert!(fails("# vec.add f32x4 ~32\n7fc000003f800000\n", 0));
    assert!(fails(want, 101));
    assert!(same_word("7ff8000000000000", "fff0000000000001", Some(64)));
    assert!(!same_word("7ff8000000000000", "7ff0000000000000", Some(64)));
    assert!(!same_word("7ff8000000000000", "fff0000000000001", None));
    let dumped = with_trap_dump(b"# x\n# dump\n0000000000000001\n");
    assert_eq!(dumped, b"# x\n# dump\n0000000000000001\n# trap dump\n0000000000000001\n");
}

/// Vector types the backend does not generate code for yield `Error::Unsupported`: f16 lanes,
/// sizeless vectors, vectors and masks wider than the registers of the function's features.
#[test]
fn unsupported_shapes() {
    use stave_ir::types::TypeTable;
    type Mk = fn(&mut TypeTable) -> TypeId;
    let f16x8: Mk = |t| {
        let l = t.intern(TypeKey::Float(FloatBits::F16));
        t.intern(TypeKey::Vec { lane: l, n: Lanes::Fixed(8) })
    };
    let sizeless: Mk = |t| {
        let l = t.f32();
        t.intern(TypeKey::Vec { lane: l, n: Lanes::Native })
    };
    let f32x16: Mk = |t| {
        let l = t.f32();
        t.intern(TypeKey::Vec { lane: l, n: Lanes::Fixed(16) })
    };
    let i64x4: Mk = |t| {
        let l = t.i64();
        t.intern(TypeKey::Vec { lane: l, n: Lanes::Fixed(4) })
    };
    let mask32x8: Mk = |t| t.intern(TypeKey::Mask { width: 32, n: Lanes::Fixed(8) });
    // (type, baseline: v3 or v2)
    let cases: [(&str, Mk, bool); 7] = [
        ("f16x8", f16x8, false),
        ("f16x8", f16x8, true),
        ("f32xN", sizeless, true),
        ("f32x16", f32x16, true),
        ("f32x16", f32x16, false),
        ("i64x4", i64x4, false),
        ("mask32x8", mask32x8, false),
    ];
    for (what, mk, wide) in cases {
        let level = if wide { v3() } else { v2() };
        let mut p = Prog::new();
        let vt = mk(&mut p.m.types);
        let f = p.decl("t.vec", &[vt], Some(vt));
        let mut b = FuncBuilder::new(&[vt]);
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
        p.def(f, b);
        p.main(|_, _| {});
        stave_ir::verify_module(&p.m).unwrap_or_else(|e| panic!("{what}: IR does not verify: {e:?}"));
        for (mode, base) in [("debug", Options::default()), ("release", release_options())] {
            let o = Options { features: level, main: p.main, ..base };
            match compile_module(&p.m, &o) {
                Err(crate::debug::Error::Unsupported { .. }) => {}
                r => panic!("{what} [{level:?} {mode}]: expected Unsupported, got {:?}", r.map(|_| ())),
            }
        }
    }
}
