//! Vector test programs: vector and mask types, edge-biased lane values, and a case list that
//! becomes one program. A case calls a function under test with constant arguments and prints
//! its result as raw bytes, so many cases share one executable; a marker before each case names
//! the case when the native output differs from the interpreter's.

use std::collections::HashMap;

use stave_ir::arith::{CastTy, IntTy};
use stave_ir::simd::{LaneTy, VecOp};
use stave_ir::types::{Lanes, TypeKey};
use stave_ir::{
    AdtShape, CastKind, Const, DeclId, FloatBits, FuncBuilder, FuncId, Op, Operand, Site, TypeId, Value,
};

use super::{Prog, Run, T, interp, native};

/// A lane type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum L {
    /// Bits, signed.
    I(u8, bool),
    /// Bits (32, 64).
    F(u8),
}

pub const INTS: [L; 8] = [
    L::I(8, false),
    L::I(8, true),
    L::I(16, false),
    L::I(16, true),
    L::I(32, false),
    L::I(32, true),
    L::I(64, false),
    L::I(64, true),
];
pub const FLOATS: [L; 2] = [L::F(32), L::F(64)];

impl L {
    pub fn bits(self) -> u32 {
        match self {
            L::I(b, _) | L::F(b) => u32::from(b),
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, L::F(_))
    }

    pub fn signed(self) -> bool {
        matches!(self, L::I(_, true))
    }

    pub fn int(self) -> Option<IntTy> {
        match self {
            L::I(b, s) => Some(IntTy::new(b, s)),
            L::F(_) => None,
        }
    }

    pub fn float(self) -> Option<FloatBits> {
        match self {
            L::F(32) => Some(FloatBits::F32),
            L::F(_) => Some(FloatBits::F64),
            L::I(..) => None,
        }
    }

    pub fn lane_ty(self) -> LaneTy {
        match (self.int(), self.float()) {
            (Some(t), _) => LaneTy::Int(t),
            (_, Some(f)) => LaneTy::Float(f),
            _ => unreachable!(),
        }
    }

    pub fn cast_ty(self) -> CastTy {
        self.lane_ty().cast_ty()
    }

    /// The unsigned int lane of the same width.
    pub fn uint(self) -> L {
        L::I(self.bits() as u8, false)
    }

    /// The lane twice as wide (ints by signedness, `f32` to `f64`).
    pub fn wide(self) -> Option<L> {
        match self {
            L::I(b, s) if b < 64 => Some(L::I(b * 2, s)),
            L::F(32) => Some(L::F(64)),
            _ => None,
        }
    }

    pub fn name(self) -> String {
        match self {
            L::I(b, true) => format!("i{b}"),
            L::I(b, false) => format!("u{b}"),
            L::F(b) => format!("f{b}"),
        }
    }

    pub fn tid(self, t: &T) -> TypeId {
        match self {
            L::I(8, false) => t.u8,
            L::I(16, false) => t.u16,
            L::I(32, false) => t.u32,
            L::I(_, false) => t.u64,
            L::I(8, true) => t.i8,
            L::I(16, true) => t.i16,
            L::I(32, true) => t.i32,
            L::I(_, true) => t.i64,
            L::F(32) => t.f32,
            L::F(_) => t.f64,
        }
    }

    /// The constant operand of a lane with these (normalized) bits.
    pub fn konst(self, b: &mut FuncBuilder, bits: u64) -> Operand {
        match self {
            L::I(w, s) => b.int(IntTy::new(w, s).value(bits)),
            L::F(32) => b.konst(Const::F32(bits as u32)),
            L::F(_) => b.konst(Const::F64(bits)),
        }
    }

    /// Lane bits as the interpreter carries them.
    pub fn norm(self, bits: u64) -> u64 {
        self.lane_ty().norm(bits)
    }
}

/// A vector type of at most 128 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VT {
    pub l: L,
    pub n: u32,
}

impl VT {
    pub fn bytes(self) -> u32 {
        self.l.bits() / 8 * self.n
    }

    pub fn name(self) -> String {
        format!("{}x{}", self.l.name(), self.n)
    }

    pub fn with(self, l: L) -> VT {
        VT { l, n: self.n }
    }

    pub fn fits(self) -> bool {
        self.n >= 2 && self.l.bits() * self.n <= 128
    }
}

/// Every lane count of `l` up to 128 bits.
pub fn vts(l: L) -> Vec<VT> {
    [2, 4, 8, 16].into_iter().map(|n| VT { l, n }).filter(|v| v.fits()).collect()
}

pub fn int_vts() -> Vec<VT> {
    INTS.iter().flat_map(|l| vts(*l)).collect()
}

pub fn float_vts() -> Vec<VT> {
    FLOATS.iter().flat_map(|l| vts(*l)).collect()
}

pub fn all_vts() -> Vec<VT> {
    int_vts().into_iter().chain(float_vts()).collect()
}

/// Every mask shape `(lane width, lanes)` up to 128 bits.
pub fn mask_shapes() -> Vec<(u32, u32)> {
    [8, 16, 32, 64]
        .into_iter()
        .flat_map(|w| [2, 4, 8, 16].into_iter().map(move |n| (w, n)))
        .filter(|(w, n)| w * n <= 128)
        .collect()
}

/// Edge values of a lane type, as lane bits.
pub fn edges(l: L) -> Vec<u64> {
    match l {
        L::I(w, s) => {
            let it = IntTy::new(w, s);
            let (min, max) = (it.min(), it.max());
            let mut v: Vec<i128> = vec![
                0,
                1,
                2,
                3,
                7,
                100,
                max,
                max - 1,
                max / 2,
                max / 2 + 1,
                min,
                min + 1,
                min / 2,
                -1,
                -2,
                -100,
            ];
            v.push(0x5555_5555_5555_5555 & max);
            v.push(0x0f0f_0f0f_0f0f_0f0f & max);
            v.retain(|x| it.fits(*x));
            v.sort();
            v.dedup();
            v.into_iter().map(|x| it.wrap(x)).collect()
        }
        L::F(32) => {
            let mut v: Vec<u32> = [
                0.0f32,
                -0.0,
                1.0,
                -1.0,
                0.5,
                -0.5,
                1.5,
                2.5,
                -2.5,
                3.5,
                0.1,
                127.5,
                -128.5,
                255.0,
                256.0,
                32767.0,
                -32769.0,
                65535.5,
                2147483648.0,
                -2147483648.0,
                4294967296.0,
                9.223372e18,
                -9.223372e18,
                1.8446744e19,
                16777216.0,
                16777217.0,
                1e30,
                -1e30,
                1e-30,
                f32::MAX,
                f32::MIN,
                f32::MIN_POSITIVE,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ]
            .iter()
            .map(|x| x.to_bits())
            .collect();
            // denormals; quiet and signaling NaNs with payloads and signs
            v.extend([
                1,
                0x8000_0001,
                0x007f_ffff,
                0x7fc0_0000,
                0x7fc1_2345,
                0xffc0_0001,
                0x7fa0_0000,
                0xff80_0001,
            ]);
            v.into_iter().map(u64::from).collect()
        }
        L::F(_) => {
            let mut v: Vec<u64> = [
                0.0f64,
                -0.0,
                1.0,
                -1.0,
                0.5,
                -0.5,
                1.5,
                2.5,
                -2.5,
                3.5,
                0.1,
                127.5,
                -128.5,
                255.0,
                256.0,
                32767.0,
                -32769.0,
                65535.5,
                2147483647.5,
                -2147483648.5,
                4294967295.5,
                4294967296.0,
                9.223372036854776e18,
                -9.223372036854776e18,
                1.8446744073709552e19,
                9007199254740993.0,
                1e30,
                -1e30,
                1e300,
                -1e300,
                1e-300,
                3.5e38,
                1e-46,
                f64::MAX,
                f64::MIN,
                f64::MIN_POSITIVE,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ]
            .iter()
            .map(|x| x.to_bits())
            .collect();
            v.extend([
                1,
                0x8000_0000_0000_0001,
                0x000f_ffff_ffff_ffff,
                0x7ff8_0000_0000_0000,
                0x7ff8_0000_1234_5678,
                0xfff8_0000_0000_0001,
                0x7ff4_0000_0000_0000,
                0xfff0_0000_0000_0001,
            ]);
            v
        }
    }
}

/// The values whose pairings decide the semantics of two-operand ops.
pub fn crit(l: L) -> Vec<u64> {
    match l {
        L::I(w, s) => {
            let it = IntTy::new(w, s);
            let mut v: Vec<i128> = vec![0, 1, 2, it.max(), it.max() - 1, it.min(), it.min() + 1, -1, 100];
            v.retain(|x| it.fits(*x));
            v.dedup();
            v.into_iter().map(|x| it.wrap(x)).collect()
        }
        L::F(32) => [0.0f32, -0.0, 1.0, -1.0, 2.5, f32::INFINITY, f32::NEG_INFINITY]
            .iter()
            .map(|x| u64::from(x.to_bits()))
            .chain([0x7fc0_0000, 0x7fa0_1000, 0xffc0_0001, 1])
            .collect(),
        L::F(_) => [0.0f64, -0.0, 1.0, -1.0, 2.5, f64::INFINITY, f64::NEG_INFINITY]
            .iter()
            .map(|x| x.to_bits())
            .chain([0x7ff8_0000_0000_0000, 0x7ff4_0000_0100_0000, 0xfff8_0000_0000_0001, 1])
            .collect(),
    }
}

/// `vals` cut into vectors of `n` lanes (the last one filled up from the start).
pub fn pack(vals: &[u64], n: u32) -> Vec<Vec<u64>> {
    let n = n as usize;
    let count = vals.len().div_ceil(n);
    (0..count).map(|c| (0..n).map(|k| vals[(c * n + k) % vals.len()]).collect()).collect()
}

/// Every pair of `vals`, as vectors of `n` lanes: lane `k` of the two vectors is one pair.
pub fn pairs(vals: &[u64], n: u32) -> Vec<(Vec<u64>, Vec<u64>)> {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for &x in vals {
        for &y in vals {
            xs.push(x);
            ys.push(y);
        }
    }
    pack(&xs, n).into_iter().zip(pack(&ys, n)).collect()
}

/// xorshift64*.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n.max(1)
    }

    /// One lane: mostly an edge value, else random bits (floats: also values near 1 with a
    /// random mantissa, which make rounding and summation order visible).
    pub fn lane(&mut self, l: L) -> u64 {
        let e = edges(l);
        match (self.below(8), l) {
            (0..=4, _) => e[self.below(e.len() as u64) as usize],
            (5 | 6, L::F(32)) => {
                let exp = 125 + self.below(5);
                (self.next() & 0x8000_0000) | (exp << 23) | (self.next() & 0x007f_ffff)
            }
            (5 | 6, L::F(_)) => {
                let exp = 1021 + self.below(5);
                (self.next() & (1 << 63)) | (exp << 52) | (self.next() & 0x000f_ffff_ffff_ffff)
            }
            _ => l.norm(self.next()),
        }
    }

    pub fn lanes(&mut self, vt: VT) -> Vec<u64> {
        (0..vt.n).map(|_| self.lane(vt.l)).collect()
    }

    /// A mask as raw lane values: all, none, or a random mix. A lane is all zeros or all ones: any
    /// other pattern is undefined behaviour.
    pub fn mask(&mut self, w: u32, n: u32) -> Vec<u64> {
        let ones = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
        let kind = self.below(6);
        (0..n)
            .map(|_| match kind {
                0 => ones,
                1 => 0,
                _ if self.below(2) == 0 => 0,
                _ => ones,
            })
            .collect()
    }
}

/// How a float result is compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Canon {
    /// Bit for bit.
    Exact,
    /// A NaN as one of two patterns: quiet or signaling (arithmetic results: quiet NaN, sign and
    /// payload unspecified).
    Quiet,
    /// Every NaN as one pattern.
    Any,
}

/// An argument of a case.
#[derive(Clone, Debug)]
pub enum A {
    V(VT, Vec<u64>),
    /// Mask of lane width `.0` from raw lane values.
    M(u32, Vec<u64>),
    I(i128),
    /// A float scalar.
    F(L, u64),
}

/// What a case prints.
#[derive(Clone, Copy, Debug)]
pub enum O {
    V(VT, Canon),
    M(u32, u32),
    /// A scalar of this type.
    S(TypeId),
    /// `Option<V>`: the tag, then the payload of `Some`.
    Opt(VT, Canon),
    /// A pair of vectors.
    Pair(VT, Canon),
    /// Nothing (the function prints).
    Unit,
}

/// An operand of a wrapped op: a parameter of the wrapper or an int constant.
#[derive(Clone, Copy, Debug)]
pub enum P {
    Arg(usize),
    K(i128),
}

struct Case {
    f: FuncId,
    ret: Option<TypeId>,
    args: Vec<A>,
    out: O,
    desc: String,
}

const MAGIC: u64 = 0xC0DE_CA5E;
/// The site of every op under test.
pub const SITE: u32 = 700;

pub struct Cx {
    pub p: Prog,
    pub rng: Rng,
    cases: Vec<Case>,
    mk: HashMap<VT, FuncId>,
    mkm: HashMap<(u32, u32), FuncId>,
    outs: HashMap<(VT, Canon), FuncId>,
    outm: HashMap<(u32, u32), FuncId>,
    outo: HashMap<(VT, Canon), FuncId>,
    opts: HashMap<TypeId, TypeId>,
    decl: u32,
    /// Functions under test.
    pub funcs: usize,
}

/// `write(1, p, n)`.
pub fn write_out(p: &Prog, b: &mut FuncBuilder, ptr: Value, n: u32) {
    let pa = b.cast(CastKind::Exact, p.t.usize, ptr);
    let (nr, fd, n) = (b.int(64), b.int(1), b.int(i128::from(n)));
    b.intrinsic(p.sys3, Some(p.t.isize), &[nr, fd, pa.into(), n]);
}

impl Cx {
    pub fn new(seed: u64) -> Cx {
        Cx {
            p: Prog::new(),
            rng: Rng(seed | 1),
            cases: Vec::new(),
            mk: HashMap::new(),
            mkm: HashMap::new(),
            outs: HashMap::new(),
            outm: HashMap::new(),
            outo: HashMap::new(),
            opts: HashMap::new(),
            decl: 20_000,
            funcs: 0,
        }
    }

    pub fn lane(&self, l: L) -> TypeId {
        l.tid(&self.p.t)
    }

    pub fn vty(&mut self, vt: VT) -> TypeId {
        assert!(vt.fits(), "{vt:?}");
        let lane = self.lane(vt.l);
        self.p.m.types.intern(TypeKey::Vec { lane, n: Lanes::Fixed(vt.n as u8) })
    }

    pub fn mty(&mut self, w: u32, n: u32) -> TypeId {
        self.p.m.types.intern(TypeKey::Mask { width: w as u8, n: Lanes::Fixed(n as u8) })
    }

    /// The mask type of compares on `vt`.
    pub fn mask_of(&mut self, vt: VT) -> TypeId {
        self.mty(vt.l.bits(), vt.n)
    }

    /// `Option<t>`.
    pub fn option(&mut self, t: TypeId) -> TypeId {
        if let Some(o) = self.opts.get(&t) {
            return *o;
        }
        self.decl += 1;
        let o = self.p.m.types.adt(DeclId(self.decl), vec![], AdtShape::enumeration(vec![vec![], vec![t]]));
        self.opts.insert(t, o);
        o
    }

    /// `*mut t`.
    pub fn ptr(&mut self, t: TypeId) -> TypeId {
        self.p.m.types.ptr(true, t)
    }

    /// `&mut t`.
    pub fn refm(&mut self, t: TypeId) -> TypeId {
        self.p.m.types.ref_(true, t)
    }

    /// `fn(l0, ..) -> V`: the lanes stored as an array, read back as the vector.
    pub fn mk(&mut self, vt: VT) -> FuncId {
        if let Some(f) = self.mk.get(&vt) {
            return *f;
        }
        let lane = self.lane(vt.l);
        let v = self.vty(vt);
        let f = self.lanes_fn(lane, vt.n, v);
        self.mk.insert(vt, f);
        f
    }

    /// `fn(l0, ..) -> M` from unsigned lanes: a lane is set iff nonzero.
    pub fn mkm(&mut self, w: u32, n: u32) -> FuncId {
        if let Some(f) = self.mkm.get(&(w, n)) {
            return *f;
        }
        let lane = self.lane(L::I(w as u8, false));
        let m = self.mty(w, n);
        let f = self.lanes_fn(lane, n, m);
        self.mkm.insert((w, n), f);
        f
    }

    fn lanes_fn(&mut self, lane: TypeId, n: u32, v: TypeId) -> FuncId {
        let arr = self.p.m.types.array(lane, u64::from(n));
        let parr = self.refm(arr);
        let pv = self.ptr(v);
        self.p.func("t.mk", &vec![lane; n as usize], Some(v), |b, _| {
            let s = b.slot(parr);
            let args: Vec<Operand> = (0..n as usize).map(|k| b.param(k).unwrap().into()).collect();
            let a = b.op(Op::Make, arr, &args);
            b.store(s, a);
            let q = b.cast(CastKind::Exact, pv, s);
            let x = b.load(v, q);
            b.ret(Some(x.into()));
        })
    }

    /// `fn(v: V)`: prints the vector's bytes, floats by `canon`.
    pub fn out(&mut self, vt: VT, canon: Canon) -> FuncId {
        let canon = if vt.l.is_float() { canon } else { Canon::Exact };
        if let Some(f) = self.outs.get(&(vt, canon)) {
            return *f;
        }
        let v = self.vty(vt);
        let pv = self.refm(v);
        let lane = self.lane(vt.l);
        let bits = self.lane(vt.l.uint());
        let (plane, pbits) = (self.ptr(lane), self.ptr(bits));
        let wide = vt.l.bits() == 64;
        let f = self.p.func("t.out", &[v], None, |b, p| {
            let x = b.param(0).unwrap();
            let s = b.slot(pv);
            b.store(s, x);
            match canon {
                Canon::Exact => write_out(p, b, s, vt.bytes()),
                Canon::Any => {
                    let q = b.cast(CastKind::Exact, plane, s);
                    for k in 0..vt.n {
                        let kk = b.int(i128::from(k));
                        let e = b.binary(Op::ElemAddr, plane, q, kk);
                        let y = b.load(lane, e);
                        p.out(b, y, lane);
                    }
                }
                Canon::Quiet => {
                    let q = b.cast(CastKind::Exact, pbits, s);
                    let (abs, inf, quiet, qnan, snan) = if wide {
                        (
                            i64::MAX as i128,
                            0x7ff0_0000_0000_0000,
                            1i128 << 51,
                            0x7ff8_0000_0000_0000,
                            0x7ff4_0000_0000_0000,
                        )
                    } else {
                        (0x7fff_ffff, 0x7f80_0000, 1 << 22, 0x7fc0_0000, 0x7fa0_0000)
                    };
                    for k in 0..vt.n {
                        let kk = b.int(i128::from(k));
                        let e = b.binary(Op::ElemAddr, pbits, q, kk);
                        let y = b.load(bits, e);
                        let (c0, c1, c2, c3, c4, z) =
                            (b.int(abs), b.int(inf), b.int(quiet), b.int(qnan), b.int(snan), b.int(0));
                        let mag = b.binary(Op::And, bits, y, c0);
                        let nan = b.cmp(Op::Gt, p.t.bool, mag, c1);
                        let qb = b.binary(Op::And, bits, y, c2);
                        let is_quiet = b.cmp(Op::Ne, p.t.bool, qb, z);
                        let which = b.select(bits, is_quiet, c3, c4);
                        let r = b.select(bits, nan, which, y);
                        p.out(b, r, bits);
                    }
                }
            }
            b.ret(None);
        });
        self.outs.insert((vt, canon), f);
        f
    }

    /// `fn(m: M)`: prints the mask's bytes.
    pub fn out_mask(&mut self, w: u32, n: u32) -> FuncId {
        if let Some(f) = self.outm.get(&(w, n)) {
            return *f;
        }
        let m = self.mty(w, n);
        let pm = self.refm(m);
        let f = self.p.func("t.outm", &[m], None, |b, p| {
            let x = b.param(0).unwrap();
            let s = b.slot(pm);
            b.store(s, x);
            write_out(p, b, s, w / 8 * n);
            b.ret(None);
        });
        self.outm.insert((w, n), f);
        f
    }

    /// `fn(o: Option<V>)`: prints the tag, then the payload of `Some`.
    pub fn out_opt(&mut self, vt: VT, canon: Canon) -> FuncId {
        if let Some(f) = self.outo.get(&(vt, canon)) {
            return *f;
        }
        let v = self.vty(vt);
        let o = self.option(v);
        let out = self.out(vt, canon);
        let f = self.p.func("t.outo", &[o], None, |b, p| {
            let x = b.param(0).unwrap();
            let tag = b.unary(Op::EnumTag, p.t.u32, x);
            p.out(b, tag, p.t.u32);
            let (some, join) = (b.create_block(), b.create_block());
            b.switch(tag, &[(1, some, &[])], join, &[]);
            b.switch_to(some);
            let (one, z) = (b.int(1), b.int(0));
            let y = b.op(Op::EnumGet, v, &[x.into(), one, z]);
            b.call(out, None, &[y.into()]);
            b.br(join, &[]);
            b.switch_to(join);
            b.ret(None);
        });
        self.outo.insert((vt, canon), f);
        f
    }

    /// `fn(params) -> ret { op operands }` at [`SITE`].
    pub fn wrap(&mut self, op: VecOp, params: &[TypeId], ret: Option<TypeId>, operands: &[P]) -> FuncId {
        self.funcs += 1;
        self.p.func(&format!("t.{}", op.name()), params, ret, |b, _| {
            let args: Vec<Operand> = operands
                .iter()
                .map(|o| match o {
                    P::Arg(k) => b.param(*k).unwrap().into(),
                    P::K(v) => b.int(*v),
                })
                .collect();
            b.set_site(Site(SITE));
            let r = b.inst(Op::Vec(op), ret, &args);
            b.ret(r.map(Into::into));
        })
    }

    /// [`Self::wrap`] with the params as the operands, in order.
    pub fn wrap_all(&mut self, op: VecOp, params: &[TypeId], ret: Option<TypeId>) -> FuncId {
        let operands: Vec<P> = (0..params.len()).map(P::Arg).collect();
        self.wrap(op, params, ret, &operands)
    }

    /// Adds a case: `f(args)`, result printed as `out`.
    pub fn case(&mut self, what: &str, f: FuncId, ret: Option<TypeId>, args: Vec<A>, out: O) {
        for a in &args {
            match a {
                A::V(vt, lanes) => {
                    assert_eq!(lanes.len(), vt.n as usize, "{what}");
                    self.mk(*vt);
                }
                A::M(w, lanes) => {
                    self.mkm(*w, lanes.len() as u32);
                }
                A::I(_) | A::F(..) => {}
            }
        }
        match out {
            O::V(vt, c) | O::Pair(vt, c) => {
                self.out(vt, c);
            }
            O::M(w, n) => {
                self.out_mask(w, n);
            }
            O::Opt(vt, c) => {
                self.out_opt(vt, c);
            }
            O::S(_) | O::Unit => {}
        }
        let desc = format!("{what} {args:x?}");
        self.cases.push(Case { f, ret, args, out, desc });
    }

    pub fn cases(&self) -> usize {
        self.cases.len()
    }

    fn build(&mut self) {
        let cases = std::mem::take(&mut self.cases);
        let mut vtys = HashMap::new();
        let mut mtys = HashMap::new();
        for c in &cases {
            for a in &c.args {
                match a {
                    A::V(vt, _) => {
                        vtys.insert(*vt, self.vty(*vt));
                    }
                    A::M(w, lanes) => {
                        mtys.insert((*w, lanes.len() as u32), self.mty(*w, lanes.len() as u32));
                    }
                    _ => {}
                }
            }
            if let O::Pair(vt, _) = c.out {
                vtys.insert(vt, self.vty(vt));
            }
        }
        let canon = |vt: VT, c: Canon| if vt.l.is_float() { c } else { Canon::Exact };
        let (mk, mkm, outs, outm, outo) = (&self.mk, &self.mkm, &self.outs, &self.outm, &self.outo);
        let mut drivers = Vec::new();
        for (ci, chunk) in cases.chunks(48).enumerate() {
            let d = self.p.func("t.drv", &[], None, |b, p| {
                for (k, c) in chunk.iter().enumerate() {
                    p.mark(b, (MAGIC << 32) | (ci * 48 + k) as u64);
                    let args: Vec<Operand> = c
                        .args
                        .iter()
                        .map(|a| match a {
                            A::V(vt, lanes) => {
                                let ls: Vec<Operand> = lanes.iter().map(|&x| vt.l.konst(b, x)).collect();
                                b.call(mk[vt], Some(vtys[vt]), &ls).unwrap().into()
                            }
                            A::M(w, lanes) => {
                                let key = (*w, lanes.len() as u32);
                                let ls: Vec<Operand> = lanes.iter().map(|&x| b.int(i128::from(x))).collect();
                                b.call(mkm[&key], Some(mtys[&key]), &ls).unwrap().into()
                            }
                            A::I(v) => b.int(*v),
                            A::F(l, bits) => l.konst(b, *bits),
                        })
                        .collect();
                    let r = b.call(c.f, c.ret, &args);
                    match (c.out, r) {
                        (O::V(vt, cn), Some(r)) => {
                            b.call(outs[&(vt, canon(vt, cn))], None, &[r.into()]);
                        }
                        (O::M(w, n), Some(r)) => {
                            b.call(outm[&(w, n)], None, &[r.into()]);
                        }
                        (O::S(t), Some(r)) => p.out(b, r, t),
                        (O::Opt(vt, cn), Some(r)) => {
                            b.call(outo[&(vt, cn)], None, &[r.into()]);
                        }
                        (O::Pair(vt, cn), Some(r)) => {
                            for k in 0..2 {
                                let kk = b.int(k);
                                let h = b.op(Op::Get, vtys[&vt], &[r.into(), kk]);
                                b.call(outs[&(vt, canon(vt, cn))], None, &[h.into()]);
                            }
                        }
                        (O::Unit, None) => {}
                        (o, _) => panic!("{}: result does not match {o:?}", c.desc),
                    }
                }
                b.ret(None);
            });
            drivers.push(d);
        }
        self.p.set_main(|b, _| {
            for d in &drivers {
                b.call(*d, None, &[]);
            }
        });
        self.cases = cases;
    }

    /// The case whose output holds byte `at` of the interpreter's stdout.
    fn case_at(&self, out: &[u8], at: usize) -> String {
        let magic = (MAGIC as u32).to_le_bytes();
        let end = (at + 1).min(out.len());
        let found = (4..end.saturating_sub(3)).rev().find(|&o| out[o..o + 4] == magic);
        match found {
            Some(o) => {
                let k = u32::from_le_bytes([out[o - 4], out[o - 3], out[o - 2], out[o - 1]]) as usize;
                let from = o + 4;
                let desc = self.cases.get(k).map_or("?", |c| c.desc.as_str());
                format!(
                    "case {k}: {desc}\n interpreter bytes of the case: {:02x?}",
                    &out[from..(from + 96).min(out.len())]
                )
            }
            None => "before the first case".into(),
        }
    }

    /// Builds the program, whose last case must trap with `kind` at [`SITE`] in the interpreter,
    /// and compares the native runs with it.
    pub fn run_trap(mut self, name: &str, kind: stave_ir::PanicKind) {
        self.build();
        let want = interp(&self.p.m, self.p.entry);
        let what = self.cases.last().map_or("?", |c| c.desc.as_str());
        assert_eq!(want.code, 101, "{name}: no trap in the interpreter: {what}");
        assert_eq!(want.stderr, super::trap_bytes(kind, SITE), "{name}: {what}");
        for release in [false, true] {
            let mode = if release { "release" } else { "debug" };
            if let Some(got) = native(&self.p.m, &format!("{name}-{mode}"), release) {
                assert_eq!(got, want, "{name} ({mode}): {what}");
            }
        }
    }

    /// Builds the program and compares the native runs (debug and release code generation) with
    /// the interpreter, which must not trap. Returns the interpreter's run.
    pub fn run(mut self, name: &str) -> Run {
        self.build();
        let want = interp(&self.p.m, self.p.entry);
        assert_eq!(want.code, 0, "{name}: the interpreter run ended with {} {:?}", want.code, want.stderr);
        eprintln!(
            "{name}: {} functions, {} cases, {} output bytes",
            self.funcs,
            self.cases.len(),
            want.stdout.len()
        );
        for release in [false, true] {
            let mode = if release { "release" } else { "debug" };
            let Some(got) = native(&self.p.m, &format!("{name}-{mode}"), release) else { continue };
            if got == want {
                continue;
            }
            let at = got
                .stdout
                .iter()
                .zip(&want.stdout)
                .position(|(a, b)| a != b)
                .unwrap_or(got.stdout.len().min(want.stdout.len()));
            panic!(
                "{name} ({mode}): native differs from the interpreter at stdout byte {at}\n {}\n native bytes there: {:02x?}\n interp bytes there: {:02x?}\n stdout lengths: native {} interp {}\n native stderr {:?} code {}",
                self.case_at(&want.stdout, at),
                &got.stdout[at.min(got.stdout.len())..(at + 32).min(got.stdout.len())],
                &want.stdout[at.min(want.stdout.len())..(at + 32).min(want.stdout.len())],
                got.stdout.len(),
                want.stdout.len(),
                got.stderr,
                got.code
            );
        }
        want
    }
}
