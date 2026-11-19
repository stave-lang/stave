//! Instruction selection for every op of the stave-ir op table (crates/stave-ir/src/ir.rs).

use stave_ir::arith::{self, CastTy, IntTy};
use stave_ir::layout::{Layout, Shape};
use stave_ir::types::FloatBits;
use stave_ir::{CastKind, Const, Op, Operand, PanicKind, Site, TypeId, TypeKey, Value};

use super::abi::Ret;
use super::analysis::Vk;
use super::emit::{Gen, S15, S16, S17, SP, W, X, ZR, xr};
use super::support::Intr;
use super::ty::{self, Sc, Ty};
use super::{R, bad, unsupported};
use crate::enc::{
    AddSub, Addr, Arr, Cond, Ext, Fp, FpBin, FpUn, Label, Logic, MemSize, PrefKind, PrefLevel, Prefetch, Reg,
    Shift, VLogic, VReg, encode_logical,
};

/// Words of an aggregate `select` emitted straight-line; longer values use a counted loop.
const SELECT_UNROLL: i64 = 64;

/// A scalar in a register: the source or the result of a cast.
#[derive(Clone, Copy)]
pub(super) enum Payload {
    G(Reg),
    F(VReg, Fp),
}

fn icond(op: Op, signed: bool) -> Cond {
    match (op, signed) {
        (Op::Eq, _) => Cond::Eq,
        (Op::Ne, _) => Cond::Ne,
        (Op::Lt, true) => Cond::Lt,
        (Op::Le, true) => Cond::Le,
        (Op::Gt, true) => Cond::Gt,
        (Op::Ge, true) => Cond::Ge,
        (Op::Lt, false) => Cond::Lo,
        (Op::Le, false) => Cond::Ls,
        (Op::Gt, false) => Cond::Hi,
        _ => Cond::Hs,
    }
}

fn fcond(op: Op) -> Cond {
    match op {
        Op::Eq => Cond::Eq,
        Op::Ne => Cond::Ne,
        Op::Lt => Cond::Mi,
        Op::Le => Cond::Ls,
        Op::Gt => Cond::Gt,
        _ => Cond::Ge,
    }
}

fn mirror(op: Op) -> Op {
    match op {
        Op::Lt => Op::Gt,
        Op::Le => Op::Ge,
        Op::Gt => Op::Lt,
        Op::Ge => Op::Le,
        o => o,
    }
}

/// Bits of the float `x` in format `fp` (exact for the powers of two used here).
fn fbits(x: f64, fp: Fp) -> u64 {
    if fp == Fp::S { u64::from((x as f32).to_bits()) } else { x.to_bits() }
}

impl<'a, 'm> Gen<'a, 'm> {
    fn sc(&self, t: Option<TypeId>) -> R<Sc> {
        match t.map(|t| self.cg.ty(t)) {
            Some(Ty::Sc(s)) => Ok(s),
            Some(Ty::Bad(why)) => Err(unsupported(why)),
            _ => Err(bad("scalar result expected")),
        }
    }

    pub(super) fn layout(&self, t: TypeId) -> R<&'a Layout> {
        let cg: &'a super::support::Codegen<'m> = self.cg;
        cg.layouts.get_layout(t).map_err(|e| unsupported(format!("layout: {e}")))
    }

    pub(super) fn value_tid(&self, o: Operand) -> R<TypeId> {
        match o {
            Operand::Value(v) => self.b.value_ty(v).ok_or_else(|| bad("value out of range")),
            _ => Err(bad("value operand expected")),
        }
    }

    /// Machine class of an operand (`None` for an int constant, typed by context).
    fn otype(&self, o: Operand) -> Option<Ty> {
        match o {
            Operand::Value(v) => Some(self.vty(v)),
            Operand::Const(c) => match self.b.konst(c) {
                Some(Const::F32(_)) => Some(Ty::Sc(Sc::F32)),
                Some(Const::F64(_)) => Some(Ty::Sc(Sc::F64)),
                Some(Const::F16(_) | Const::BF16(_)) => Some(Ty::Bad("f16/bf16 values")),
                _ => None,
            },
            Operand::Str(_) | Operand::Func(_) => Some(Ty::Fat),
            Operand::Type(_) => None,
        }
    }

    pub(super) fn norm(&mut self, r: Reg, s: Sc) -> R<()> {
        match (s.bits(), s.signed()) {
            (32, true) => self.a.sxt(X, r, r, Ext::W)?,
            (32, false) => self.a.mov(W, r, r)?,
            (16, true) => self.a.sxt(X, r, r, Ext::H)?,
            (16, false) => self.a.uxt(r, r, Ext::H)?,
            (8, true) => self.a.sxt(X, r, r, Ext::B)?,
            (8, false) => self.a.uxt(r, r, Ext::B)?,
            (1, _) => self.a.logic_imm(Logic::And, X, r, r, 1)?,
            _ => {}
        }
        Ok(())
    }

    /// Traps to `l` unless the exact 64-bit value in `r` fits `s` (8/16/32-bit types).
    fn range_check(&mut self, r: Reg, s: Sc, l: Label) -> R<()> {
        let ext = match s.bits() {
            8 => Ext::B,
            16 => Ext::H,
            32 => Ext::W,
            _ => return Ok(()),
        };
        if s.signed() {
            let t = self.tmp_g()?;
            self.a.sxt(X, t, r, ext)?;
            self.a.cmp(X, t, r)?;
        } else {
            let mask = (1u64 << s.bits()) - 1;
            self.a.tst_imm(X, r, !mask)?;
        }
        self.a.b_cond(Cond::Ne, l)?;
        Ok(())
    }

    /// Clamps the exact 64-bit value in `r` to the range of `s` (signedness of the value: `vs`).
    pub(super) fn clamp(&mut self, r: Reg, from: (i128, i128), vs: bool, to: Sc) -> R<()> {
        let (tmin, tmax) = to.range();
        if from.1 > tmax {
            let t = self.tmp_g()?;
            self.a.mov_imm(X, t, tmax as u64)?;
            self.a.cmp(X, r, t)?;
            self.a.csel(X, r, t, r, if vs { Cond::Gt } else { Cond::Hi })?;
        }
        if from.0 < tmin {
            let t = self.tmp_g()?;
            self.a.mov_imm(X, t, tmin as i64 as u64)?;
            self.a.cmp(X, r, t)?;
            self.a.csel(X, r, t, r, Cond::Lt)?;
        }
        Ok(())
    }

    pub(super) fn res(&self, i: u32) -> R<Value> {
        self.b.insts.result.get(i as usize).copied().flatten().ok_or_else(|| bad("missing result"))
    }

    pub fn inst(&mut self, i: u32) -> R<()> {
        let b = self.b;
        let iu = i as usize;
        let op = b.insts.op[iu];
        let ty = b.insts.ty[iu];
        let args = b.args(i);
        let site = self.site(i);
        let arg = |k: usize| args.get(k).copied().ok_or_else(|| bad("missing operand"));
        match op {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem => {
                let s = self.sc(ty)?;
                let v = self.res(i)?;
                if s.is_fp() {
                    self.fbin(op, s, arg(0)?, arg(1)?, v)
                } else {
                    self.ibin(op, s, arg(0)?, arg(1)?, v, site)
                }
            }
            Op::AddWrap | Op::SubWrap | Op::MulWrap => {
                let s = self.sc(ty)?;
                let v = self.res(i)?;
                let ra = self.use_g(arg(0)?)?;
                let rb = self.use_g(arg(1)?)?;
                let rd = self.def_g(v)?;
                let w = if s.bits() == 32 { W } else { X };
                match op {
                    Op::AddWrap => self.a.add(w, rd, ra, rb)?,
                    Op::SubWrap => self.a.sub(w, rd, ra, rb)?,
                    _ => self.a.mul(w, rd, ra, rb)?,
                }
                self.norm(rd, s)
            }
            Op::AddSat | Op::SubSat | Op::MulSat => {
                let s = self.sc(ty)?;
                let v = self.res(i)?;
                self.sat(op, s, arg(0)?, arg(1)?, v)
            }
            Op::Shl | Op::Shr | Op::ShlWrap => {
                let s = self.sc(ty)?;
                let v = self.res(i)?;
                self.shift(op, s, arg(0)?, arg(1)?, v, site)
            }
            Op::Neg => {
                let s = self.sc(ty)?;
                let v = self.res(i)?;
                if s.is_fp() {
                    let fa = self.use_f(arg(0)?, s)?;
                    let rd = self.def_f(v)?;
                    self.a.fp_un(FpUn::Neg, s.fp(), rd, fa);
                    return Ok(());
                }
                let ra = self.use_g(arg(0)?)?;
                let rd = self.def_g(v)?;
                let l = self.trap(PanicKind::Overflow, site)?;
                match s.bits() {
                    64 => {
                        self.a.addsub_reg(AddSub::Subs, X, rd, ZR, ra, Shift::Lsl, 0)?;
                        self.a.b_cond(Cond::Vs, l)?;
                    }
                    32 => {
                        self.a.addsub_reg(AddSub::Subs, W, rd, ZR, ra, Shift::Lsl, 0)?;
                        self.a.b_cond(Cond::Vs, l)?;
                        self.a.sxt(X, rd, rd, Ext::W)?;
                    }
                    _ => {
                        self.a.neg(X, rd, ra)?;
                        self.range_check(rd, s, l)?;
                    }
                }
                Ok(())
            }
            Op::And | Op::Or | Op::Xor => {
                let v = self.res(i)?;
                let (mut a, mut c) = (arg(0)?, arg(1)?);
                if self.b.int_const(a).is_some() && self.b.int_const(c).is_none() {
                    std::mem::swap(&mut a, &mut c);
                }
                let lop = match op {
                    Op::And => Logic::And,
                    Op::Or => Logic::Orr,
                    _ => Logic::Eor,
                };
                let ra = self.use_g(a)?;
                match self.int_bits(c) {
                    Some(k) if encode_logical(k, 64).is_some() => {
                        let rd = self.def_g(v)?;
                        self.a.logic_imm(lop, X, rd, ra, k)?;
                    }
                    _ => {
                        let rb = self.use_g(c)?;
                        let rd = self.def_g(v)?;
                        self.a.logic_reg(lop, X, rd, ra, rb, Shift::Lsl, 0)?;
                    }
                }
                Ok(())
            }
            Op::Not => {
                let s = self.sc(ty)?;
                let v = self.res(i)?;
                let ra = self.use_g(arg(0)?)?;
                let rd = self.def_g(v)?;
                if s == Sc::Bool {
                    self.a.logic_imm(Logic::Eor, X, rd, ra, 1)?;
                } else {
                    self.a.mvn(X, rd, ra)?;
                    if !s.signed() {
                        self.norm(rd, s)?;
                    }
                }
                Ok(())
            }
            Op::Popcount | Op::Clz | Op::Ctz => {
                let s = self.sc(ty)?;
                let v = self.res(i)?;
                self.count(op, s, arg(0)?, v)
            }
            Op::MulWide => self.mul_wide(i, arg(0)?, arg(1)?),
            Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                let v = self.res(i)?;
                self.compare(op, arg(0)?, arg(1)?, v)
            }
            Op::Select => {
                let v = self.res(i)?;
                let t = ty.ok_or_else(|| bad("select type"))?;
                self.select(self.cg.ty(t), arg(0)?, arg(1)?, arg(2)?, v)
            }
            Op::Cast(kind) => {
                let v = self.res(i)?;
                let t = ty.ok_or_else(|| bad("cast type"))?;
                self.cast(kind, t, arg(0)?, v)
            }
            Op::Make => {
                let v = self.res(i)?;
                let t = ty.ok_or_else(|| bad("make type"))?;
                let h = self.home(v);
                let fields: Vec<(u64, TypeId)> = match &self.layout(t)?.shape {
                    Shape::Fields(fs) => fs.clone(),
                    Shape::Array { elem, stride, len } => (0..*len).map(|k| (k * stride, *elem)).collect(),
                    _ => return Err(bad("make of a non-record")),
                };
                for (k, (off, ft)) in fields.into_iter().enumerate() {
                    let ft = self.cg.ty(ft);
                    self.put(arg(k)?, ft, SP, h + off as i64)?;
                }
                Ok(())
            }
            Op::Get | Op::Set => self.get_set(op, i, site),
            Op::EnumMake => {
                let v = self.res(i)?;
                let t = ty.ok_or_else(|| bad("enum.make type"))?;
                let var = self.b.int_const(arg(0)?).ok_or_else(|| bad("variant index"))?;
                let l = self.layout(t)?;
                let size = l.size as u32;
                let fields: Vec<(u64, TypeId)> = match &l.shape {
                    Shape::Enum(vs) => usize::try_from(var)
                        .ok()
                        .and_then(|k| vs.get(k))
                        .cloned()
                        .ok_or_else(|| bad("variant"))?,
                    _ => return Err(bad("enum.make of a non-enum")),
                };
                let h = self.home(v);
                self.memzero(SP, h, size)?;
                if var != 0 {
                    self.a.mov_imm(W, S15, var as u64)?;
                    self.st(MemSize::W, S15, SP, h)?;
                }
                for (k, (off, ft)) in fields.into_iter().enumerate() {
                    let ft = self.cg.ty(ft);
                    self.put(arg(k + 1)?, ft, SP, h + off as i64)?;
                }
                Ok(())
            }
            Op::EnumTag => {
                let v = self.res(i)?;
                let Operand::Value(e) = arg(0)? else { return Err(bad("enum.tag of a non-value")) };
                let he = self.home(e);
                let rd = self.def_g(v)?;
                self.ld(MemSize::W, false, rd, SP, he)
            }
            Op::EnumGet => {
                let v = self.res(i)?;
                let Operand::Value(e) = arg(0)? else { return Err(bad("enum.get of a non-value")) };
                let et = self.value_tid(arg(0)?)?;
                let vi = self.b.int_const(arg(1)?).ok_or_else(|| bad("variant index"))?;
                let fi = self.b.int_const(arg(2)?).ok_or_else(|| bad("field index"))?;
                let (off, ft) =
                    ty::variant_field(self.layout(et)?, vi, fi).ok_or_else(|| bad("enum field"))?;
                let he = self.home(e);
                let ft = self.cg.ty(ft);
                self.take(v, ft, SP, he + off as i64)
            }
            Op::Slot => Ok(()),
            Op::Load => {
                let v = self.res(i)?;
                let t = ty.ok_or_else(|| bad("load type"))?;
                let rp = self.use_g(arg(0)?)?;
                let t = self.cg.ty(t);
                self.take(v, t, rp, 0)
            }
            // AArch64 has no single-register non-temporal store, and `stnp` is not used: a
            // streaming store is the plain store, so it needs no fence
            Op::Store | Op::StoreStream => {
                let pt = self.value_tid(arg(0)?)?;
                let inner =
                    ty::pointee(&self.cg.m.types, pt).ok_or_else(|| bad("store through a non-pointer"))?;
                let rp = self.use_g(arg(0)?)?;
                let t = self.cg.ty(inner);
                self.put(arg(1)?, t, rp, 0)
            }
            Op::FieldAddr | Op::EnumFieldAddr => {
                let v = self.res(i)?;
                let pt = self.value_tid(arg(0)?)?;
                let inner =
                    ty::pointee(&self.cg.m.types, pt).ok_or_else(|| bad("addr through a non-pointer"))?;
                let l = self.layout(inner)?;
                let off = if op == Op::FieldAddr {
                    let k = self.b.int_const(arg(1)?).ok_or_else(|| bad("field index"))?;
                    ty::field(l, k).ok_or_else(|| bad("field"))?.0
                } else {
                    let vi = self.b.int_const(arg(1)?).ok_or_else(|| bad("variant index"))?;
                    let fi = self.b.int_const(arg(2)?).ok_or_else(|| bad("field index"))?;
                    ty::variant_field(l, vi, fi).ok_or_else(|| bad("enum field"))?.0
                };
                let rp = self.use_g(arg(0)?)?;
                let rd = self.def_g(v)?;
                self.add_imm(rd, rp, off as i64)
            }
            Op::ElemAddr => self.elem_addr(i),
            Op::SliceMake => {
                let v = self.res(i)?;
                let h = self.home(v);
                self.put(arg(0)?, Ty::Sc(Sc::Ptr), SP, h)?;
                self.put(arg(1)?, Ty::Sc(Sc::USIZE), SP, h + 8)
            }
            Op::SliceLen | Op::SlicePtr => {
                let v = self.res(i)?;
                let word = if op == Op::SliceLen { 8 } else { 0 };
                match arg(0)? {
                    Operand::Value(s) => {
                        let h = self.home(s);
                        let rd = self.def_g(v)?;
                        self.ld(MemSize::X, false, rd, SP, h + word)
                    }
                    Operand::Str(s) => {
                        let (sym, so, len) = self.str_ref(s)?;
                        let rd = self.def_g(v)?;
                        if op == Op::SliceLen {
                            self.a.mov_imm(X, rd, len)?;
                        } else {
                            self.a.addr_of_sym(rd, sym, so)?;
                        }
                        Ok(())
                    }
                    _ => Err(bad("slice operand")),
                }
            }
            Op::BoundsCheck => self.bounds(arg(0)?, arg(1)?, site),
            Op::Prefetch(h) => self.prefetch(h, arg(0)?, arg(1)?),
            Op::FenceStream => Ok(()),
            Op::Call => {
                let Operand::Func(g) = arg(0)? else { return Err(bad("call callee")) };
                let m = self.cg.m;
                let gf = m.func(g).ok_or_else(|| bad("callee out of range"))?;
                let sym = self.cg.func_sym(g)?;
                let res = self.b.insts.result[iu];
                self.call(Ok(sym), &gf.params, ty, &args[1..], res)
            }
            Op::CallInd => {
                let m = self.cg.m;
                let res = self.b.insts.result[iu];
                if let Operand::Func(g) = arg(0)? {
                    let gf = m.func(g).ok_or_else(|| bad("callee out of range"))?;
                    let sym = self.cg.func_sym(g)?;
                    return self.call_env(Ok(sym), &gf.params, ty, &args[1..], res);
                }
                let ft = self.value_tid(arg(0)?)?;
                let Some(TypeKey::FnType { params, .. }) = m.types.key(ft) else {
                    return Err(bad("call.ind of a non-fn value"));
                };
                self.call_env(Err(arg(0)?), params, ty, &args[1..], res)
            }
            Op::FnMake => {
                let v = self.res(i)?;
                let h = self.home(v);
                self.put_code(arg(0)?, SP, h)?;
                self.put(arg(1)?, Ty::Sc(Sc::USIZE), SP, h + 8)
            }
            Op::Intrinsic(id) => self.intrinsic(id, i),
            Op::Vec(v) => self.vec_op(v, i),
            Op::Panic(kind) => {
                let l = self.trap(kind, site)?;
                self.a.b(l)?;
                Ok(())
            }
        }
    }

    // ---------------------------------------------------------------- integer arithmetic

    fn ibin(&mut self, op: Op, s: Sc, a: Operand, bo: Operand, v: Value, site: Site) -> R<()> {
        let bits = s.bits();
        let ra = self.use_g(a)?;
        let imm = match (op, self.b.int_const(bo)) {
            (Op::Add | Op::Sub, Some(c)) if (0..4096).contains(&c) => Some(c as u64),
            _ => None,
        };
        let rb = match imm {
            Some(_) => ZR,
            None => self.use_g(bo)?,
        };
        let rd = self.def_g(v)?;
        let w = if bits == 32 { W } else { X };
        match op {
            Op::Add | Op::Sub => {
                let flags = bits >= 32;
                let o = match (op == Op::Add, flags) {
                    (true, true) => AddSub::Adds,
                    (true, false) => AddSub::Add,
                    (false, true) => AddSub::Subs,
                    (false, false) => AddSub::Sub,
                };
                let ww = if flags { w } else { X };
                match imm {
                    Some(k) => self.a.addsub_imm(o, ww, rd, ra, k)?,
                    None => self.a.addsub_reg(o, ww, rd, ra, rb, Shift::Lsl, 0)?,
                }
                let l = self.trap(PanicKind::Overflow, site)?;
                if flags {
                    let c = if s.signed() {
                        Cond::Vs
                    } else if op == Op::Add {
                        Cond::Hs
                    } else {
                        Cond::Lo
                    };
                    self.a.b_cond(c, l)?;
                    if bits == 32 && s.signed() {
                        self.a.sxt(X, rd, rd, Ext::W)?;
                    }
                } else {
                    self.range_check(rd, s, l)?;
                }
            }
            Op::Mul => {
                let l = self.trap(PanicKind::Overflow, site)?;
                match (bits, s.signed()) {
                    (64, true) => {
                        self.a.mul(X, rd, ra, rb)?;
                        let t = self.tmp_g()?;
                        self.a.smulh(t, ra, rb)?;
                        self.a.addsub_reg(AddSub::Subs, X, ZR, t, rd, Shift::Asr, 63)?;
                        self.a.b_cond(Cond::Ne, l)?;
                    }
                    (64, false) => {
                        let t = self.tmp_g()?;
                        self.a.umulh(t, ra, rb)?;
                        self.a.cbnz(X, t, l)?;
                        self.a.mul(X, rd, ra, rb)?;
                    }
                    (32, true) => {
                        self.a.smull(rd, ra, rb)?;
                        self.range_check(rd, s, l)?;
                    }
                    (32, false) => {
                        self.a.umull(rd, ra, rb)?;
                        self.range_check(rd, s, l)?;
                    }
                    _ => {
                        self.a.mul(X, rd, ra, rb)?;
                        self.range_check(rd, s, l)?;
                    }
                }
            }
            _ => {
                let l0 = self.trap(PanicKind::DivByZero, site)?;
                self.a.cbz(X, rb, l0)?;
                if s.signed() {
                    let lo = self.trap(PanicKind::Overflow, site)?;
                    let t = self.tmp_g()?;
                    self.a.mov_imm(X, t, s.range().0 as i64 as u64)?;
                    self.a.cmn_imm(X, rb, 1)?;
                    self.a.ccmp(X, ra, t, 0, Cond::Eq)?;
                    self.a.b_cond(Cond::Eq, lo)?;
                }
                let div = |g: &mut Self, d: Reg| -> R<()> {
                    if s.signed() {
                        g.a.sdiv(X, d, ra, rb)?;
                    } else {
                        g.a.udiv(X, d, ra, rb)?;
                    }
                    Ok(())
                };
                if op == Op::Div {
                    div(self, rd)?;
                } else {
                    let q = self.tmp_g()?;
                    div(self, q)?;
                    self.a.msub(X, rd, q, rb, ra)?;
                }
            }
        }
        Ok(())
    }

    fn sat(&mut self, op: Op, s: Sc, a: Operand, bo: Operand, v: Value) -> R<()> {
        let ra = self.use_g(a)?;
        let rb = self.use_g(bo)?;
        let rd = self.def_g(v)?;
        const MAX: u64 = i64::MAX as u64;
        if s.bits() == 64 {
            match (op, s.signed()) {
                (Op::AddSat | Op::SubSat, true) => {
                    let t = self.tmp_g()?;
                    self.a.shift_imm(Shift::Asr, X, t, ra, 63)?;
                    self.a.logic_imm(Logic::Eor, X, t, t, MAX)?;
                    let o = if op == Op::AddSat { AddSub::Adds } else { AddSub::Subs };
                    self.a.addsub_reg(o, X, rd, ra, rb, Shift::Lsl, 0)?;
                    self.a.csel(X, rd, t, rd, Cond::Vs)?;
                }
                (Op::AddSat, false) => {
                    self.a.addsub_reg(AddSub::Adds, X, rd, ra, rb, Shift::Lsl, 0)?;
                    self.a.csinv(X, rd, rd, ZR, Cond::Lo)?;
                }
                (Op::SubSat, false) => {
                    self.a.addsub_reg(AddSub::Subs, X, rd, ra, rb, Shift::Lsl, 0)?;
                    self.a.csel(X, rd, rd, ZR, Cond::Hs)?;
                }
                (_, true) => {
                    let t = self.tmp_g()?;
                    self.a.logic_reg(Logic::Eor, X, t, ra, rb, Shift::Lsl, 0)?;
                    self.a.shift_imm(Shift::Asr, X, t, t, 63)?;
                    self.a.logic_imm(Logic::Eor, X, t, t, MAX)?;
                    self.a.mul(X, rd, ra, rb)?;
                    let hi = self.tmp_g()?;
                    self.a.smulh(hi, ra, rb)?;
                    self.a.addsub_reg(AddSub::Subs, X, ZR, hi, rd, Shift::Asr, 63)?;
                    self.a.csel(X, rd, rd, t, Cond::Eq)?;
                }
                (_, false) => {
                    let hi = self.tmp_g()?;
                    self.a.umulh(hi, ra, rb)?;
                    self.a.mul(X, rd, ra, rb)?;
                    self.a.cmp_imm(X, hi, 0)?;
                    self.a.csinv(X, rd, rd, ZR, Cond::Eq)?;
                }
            }
            return Ok(());
        }
        match op {
            Op::AddSat => self.a.add(X, rd, ra, rb)?,
            Op::SubSat => self.a.sub(X, rd, ra, rb)?,
            _ => self.a.mul(X, rd, ra, rb)?,
        }
        let tmax = s.range().1;
        if s.signed() {
            self.clamp(rd, (i128::MIN, i128::MAX), true, s)?;
        } else if op == Op::SubSat {
            self.a.cmp_imm(X, rd, 0)?;
            self.a.csel(X, rd, ZR, rd, Cond::Lt)?;
        } else {
            let t = self.tmp_g()?;
            self.a.mov_imm(X, t, tmax as u64)?;
            self.a.cmp(X, rd, t)?;
            self.a.csel(X, rd, t, rd, Cond::Hi)?;
        }
        Ok(())
    }

    fn shift(&mut self, op: Op, s: Sc, a: Operand, n: Operand, v: Value, site: Site) -> R<()> {
        let bits = i128::from(s.bits());
        let ra = self.use_g(a)?;
        let sh = match op {
            Op::Shr if s.signed() => Shift::Asr,
            Op::Shr => Shift::Lsr,
            _ => Shift::Lsl,
        };
        if let Some(k) = self.b.int_const(n) {
            let rd = self.def_g(v)?;
            let k = if op == Op::ShlWrap { k & (bits - 1) } else { k };
            if !(0..bits).contains(&k) {
                let l = self.trap(PanicKind::Overflow, site)?;
                self.a.b(l)?;
                self.a.mov(X, rd, ra)?;
                return Ok(());
            }
            self.a.shift_imm(sh, X, rd, ra, k as u32)?;
        } else {
            let rn = self.use_g(n)?;
            let rd = self.def_g(v)?;
            if op == Op::ShlWrap {
                match bits {
                    64 => self.a.shift_reg(Shift::Lsl, X, rd, ra, rn)?,
                    32 => self.a.shift_reg(Shift::Lsl, W, rd, ra, rn)?,
                    _ => {
                        let t = self.tmp_g()?;
                        self.a.logic_imm(Logic::And, X, t, rn, (bits - 1) as u64)?;
                        self.a.shift_reg(Shift::Lsl, X, rd, ra, t)?;
                    }
                }
            } else {
                let l = self.trap(PanicKind::Overflow, site)?;
                self.a.cmp_imm(X, rn, bits as u64)?;
                self.a.b_cond(Cond::Hs, l)?;
                self.a.shift_reg(sh, X, rd, ra, rn)?;
            }
        }
        if op != Op::Shr {
            let rd = xr(self.cached(v).ok_or_else(|| bad("shift result"))?);
            self.norm(rd, s)?;
        }
        Ok(())
    }

    fn count(&mut self, op: Op, s: Sc, a: Operand, v: Value) -> R<()> {
        let bits = s.bits();
        let ra = self.use_g(a)?;
        let rd = self.def_g(v)?;
        let masked = |g: &mut Self| -> R<Reg> {
            Ok(match bits {
                64 => ra,
                32 => {
                    let t = g.tmp_g()?;
                    g.a.mov(W, t, ra)?;
                    t
                }
                _ => {
                    let t = g.tmp_g()?;
                    g.a.logic_imm(Logic::And, X, t, ra, (1u64 << bits) - 1)?;
                    t
                }
            })
        };
        match op {
            Op::Popcount => {
                let src = masked(self)?;
                let v31 = VReg::vn(31);
                self.a.fmov_from_gpr(Fp::D, v31, src)?;
                self.a.cnt_8b(v31, v31);
                self.a.addv_8b(v31, v31);
                self.a.fmov_to_gpr(Fp::S, rd, v31)?;
            }
            Op::Clz => match bits {
                64 => self.a.clz(X, rd, ra)?,
                32 => self.a.clz(W, rd, ra)?,
                _ => {
                    let src = masked(self)?;
                    self.a.clz(W, rd, src)?;
                    self.a.addsub_imm(AddSub::Sub, W, rd, rd, 32 - u64::from(bits))?;
                }
            },
            _ => match bits {
                64 => {
                    self.a.rbit(X, rd, ra)?;
                    self.a.clz(X, rd, rd)?;
                }
                32 => {
                    self.a.rbit(W, rd, ra)?;
                    self.a.clz(W, rd, rd)?;
                }
                _ => {
                    let t = self.tmp_g()?;
                    self.a.logic_imm(Logic::Orr, W, t, ra, 1u64 << bits)?;
                    self.a.rbit(W, t, t)?;
                    self.a.clz(W, rd, t)?;
                }
            },
        }
        Ok(())
    }

    fn mul_wide(&mut self, i: u32, a: Operand, bo: Operand) -> R<()> {
        let v = self.res(i)?;
        let t = self.b.insts.ty[i as usize].ok_or_else(|| bad("mul.wide type"))?;
        let (o0, o1, et) = match &self.layout(t)?.shape {
            Shape::Fields(fs) if fs.len() == 2 => (fs[0].0 as i64, fs[1].0 as i64, fs[0].1),
            _ => return Err(bad("mul.wide result")),
        };
        let (wide, signed) = match self.cg.ty(et) {
            Ty::Sc(Sc::Int { bits, signed }) => (bits == 64, signed),
            _ => return Err(bad("mul.wide operand")),
        };
        let ra = self.use_g(a)?;
        let rb = self.use_g(bo)?;
        let lo = self.tmp_g()?;
        let hi = self.tmp_g()?;
        let h = self.home(v);
        if wide {
            if signed {
                self.a.smulh(hi, ra, rb)?;
            } else {
                self.a.umulh(hi, ra, rb)?;
            }
            self.a.mul(X, lo, ra, rb)?;
            self.st(MemSize::X, hi, SP, h + o0)?;
            self.st(MemSize::X, lo, SP, h + o1)
        } else {
            if signed {
                self.a.smull(lo, ra, rb)?;
            } else {
                self.a.umull(lo, ra, rb)?;
            }
            self.a.shift_imm(if signed { Shift::Asr } else { Shift::Lsr }, X, hi, lo, 32)?;
            self.st(MemSize::W, hi, SP, h + o0)?;
            self.st(MemSize::W, lo, SP, h + o1)
        }
    }

    // ---------------------------------------------------------------- floats

    fn fbin(&mut self, op: Op, s: Sc, a: Operand, bo: Operand, v: Value) -> R<()> {
        let fp = s.fp();
        if op == Op::Rem {
            self.flush_all()?;
            let (d0, d1) = (VReg::vn(0), VReg::vn(1));
            self.load_to_v(d0, a, s)?;
            self.load_to_v(d1, bo, s)?;
            if fp == Fp::S {
                self.a.fcvt(Fp::D, d0, Fp::S, d0)?;
                self.a.fcvt(Fp::D, d1, Fp::S, d1)?;
            }
            self.a.bl_sym(self.cg.syms.fmod);
            if fp == Fp::S {
                self.a.fcvt(Fp::S, d0, Fp::D, d0)?;
            }
            return self.result_regs(Ret::V0, Some(v));
        }
        let fa = self.use_f(a, s)?;
        let fb = self.use_f(bo, s)?;
        let rd = self.def_f(v)?;
        let o = match op {
            Op::Add => FpBin::Add,
            Op::Sub => FpBin::Sub,
            Op::Mul => FpBin::Mul,
            _ => FpBin::Div,
        };
        self.a.fp_bin(o, fp, rd, fa, fb);
        Ok(())
    }

    // ---------------------------------------------------------------- compare / select

    /// Emits compare instruction `i` (on int or float scalars) as flags only; the returned
    /// condition holds iff the compare is true. `b.cond` on its negation branches when the
    /// compare is false, also for unordered floats.
    pub(crate) fn compare_flags(&mut self, i: u32) -> R<Cond> {
        let op = self.b.insts.op[i as usize];
        let (mut a, mut bo, mut op) = match self.b.args(i) {
            &[a, bo] => (a, bo, op),
            _ => return Err(bad("compare operands")),
        };
        match self.otype(a).or(self.otype(bo)) {
            Some(Ty::Sc(s)) if s.is_fp() => {
                let fa = self.use_f(a, s)?;
                let fb = self.use_f(bo, s)?;
                self.a.fcmp(s.fp(), fa, fb);
                Ok(fcond(op))
            }
            Some(Ty::Sc(s)) => {
                if self.b.int_const(a).is_some() && self.b.int_const(bo).is_none() {
                    std::mem::swap(&mut a, &mut bo);
                    op = mirror(op);
                }
                let ra = self.use_g(a)?;
                match self.b.int_const(bo) {
                    Some(k) if (0..4096).contains(&k) => self.a.cmp_imm(X, ra, k as u64)?,
                    Some(k) if (-4095..0).contains(&k) => self.a.cmn_imm(X, ra, (-k) as u64)?,
                    _ => {
                        let rb = self.use_g(bo)?;
                        self.a.cmp(X, ra, rb)?;
                    }
                }
                Ok(icond(op, s.signed()))
            }
            _ => Err(bad("fused compare of a non-scalar")),
        }
    }

    fn compare(&mut self, op: Op, a: Operand, bo: Operand, v: Value) -> R<()> {
        let is_fn = |g: &Self, o: Operand| match o {
            Operand::Func(_) => true,
            Operand::Value(x) => {
                g.b.value_ty(x).is_some_and(|t| matches!(g.cg.m.types.key(t), Some(TypeKey::FnType { .. })))
            }
            _ => false,
        };
        if is_fn(self, a) || is_fn(self, bo) {
            // fn values: equal iff code and environment are equal
            use crate::enc::{Logic, Shift};
            let rd = self.def_g(v)?;
            // `ld` writes only its destination, so neither load disturbs the other's registers
            self.load_fat(S16, S17, a)?;
            self.load_fat(S15, rd, bo)?;
            self.a.logic_reg(Logic::Eor, X, S16, S16, S15, Shift::Lsl, 0)?;
            self.a.logic_reg(Logic::Eor, X, S17, S17, rd, Shift::Lsl, 0)?;
            self.a.logic_reg(Logic::Orr, X, S16, S16, S17, Shift::Lsl, 0)?;
            self.a.cmp_imm(X, S16, 0)?;
            self.a.cset(W, rd, if op == Op::Eq { crate::enc::Cond::Eq } else { crate::enc::Cond::Ne })?;
            return Ok(());
        }
        match self.otype(a).or(self.otype(bo)) {
            None => {
                let (x, y) = (self.b.int_const(a), self.b.int_const(bo));
                let (Some(x), Some(y)) = (x, y) else { return Err(bad("compare operands")) };
                let r = match op {
                    Op::Eq => x == y,
                    Op::Ne => x != y,
                    Op::Lt => x < y,
                    Op::Le => x <= y,
                    Op::Gt => x > y,
                    _ => x >= y,
                };
                let rd = self.def_g(v)?;
                self.a.mov_imm(X, rd, u64::from(r))?;
                Ok(())
            }
            Some(Ty::Sc(s)) if s.is_fp() => {
                let fa = self.use_f(a, s)?;
                let fb = self.use_f(bo, s)?;
                let rd = self.def_g(v)?;
                self.a.fcmp(s.fp(), fa, fb);
                self.a.cset(W, rd, fcond(op))?;
                Ok(())
            }
            Some(Ty::Sc(s)) => {
                let (mut a, mut bo, mut op) = (a, bo, op);
                if self.b.int_const(a).is_some() && self.b.int_const(bo).is_none() {
                    std::mem::swap(&mut a, &mut bo);
                    op = mirror(op);
                }
                let ra = self.use_g(a)?;
                let c = self.b.int_const(bo);
                let rb = match c {
                    Some(k) if (-4095..4096).contains(&k) => None,
                    _ => Some(self.use_g(bo)?),
                };
                let rd = self.def_g(v)?;
                match (rb, c) {
                    (Some(rb), _) => self.a.cmp(X, ra, rb)?,
                    (None, Some(k)) if k >= 0 => self.a.cmp_imm(X, ra, k as u64)?,
                    (None, Some(k)) => self.a.cmn_imm(X, ra, (-k) as u64)?,
                    (None, None) => return Err(bad("compare operand")),
                }
                self.a.cset(W, rd, icond(op, s.signed()))?;
                Ok(())
            }
            Some(Ty::Bad(why)) => Err(unsupported(why)),
            Some(_) => Err(bad("compare of an aggregate")),
        }
    }

    fn select(&mut self, t: Ty, c: Operand, a: Operand, bo: Operand, v: Value) -> R<()> {
        if let Some(k) = self.b.int_const(c) {
            let o = if k != 0 { a } else { bo };
            return self.copy_into(v, t, o);
        }
        match t {
            Ty::Sc(s) if s.is_fp() => {
                let rc = self.use_g(c)?;
                let fa = self.use_f(a, s)?;
                let fb = self.use_f(bo, s)?;
                let rd = self.def_f(v)?;
                self.a.cmp_imm(W, rc, 0)?;
                self.a.fcsel(s.fp(), rd, fa, fb, Cond::Ne);
                Ok(())
            }
            Ty::Sc(_) => {
                let rc = self.use_g(c)?;
                let ra = self.use_g(a)?;
                let rb = self.use_g(bo)?;
                let rd = self.def_g(v)?;
                self.a.cmp_imm(W, rc, 0)?;
                self.a.csel(X, rd, ra, rb, Cond::Ne)?;
                Ok(())
            }
            Ty::Vec(_) if self.vregs() => {
                // every bit of the result is taken from `a` where the condition is set
                let rc = self.use_g(c)?;
                let (va, _) = self.use_v(a)?;
                let (vb, _) = self.use_v(bo)?;
                let rd = self.dst_vec(v)?;
                self.a.cmp_imm(W, rc, 0)?;
                self.a.csetm(X, S15, Cond::Ne)?;
                self.a.dup_gpr(Arr::D2, rd, S15)?;
                self.a.vlogic(VLogic::Bsl, Arr::B16, rd, va, vb)?;
                self.set_v(v, rd)
            }
            Ty::Fat | Ty::Vec(_) | Ty::Mem { .. } => {
                let rc = self.use_g(c)?;
                let h = self.home(v);
                self.select_mem(rc, h, a, bo, t.size())
            }
            Ty::Bad(why) => Err(unsupported(why)),
        }
    }

    /// Chunk `k` (`sz` bytes at offset `k`) of an aggregate-class operand into `rt`. Changes
    /// no flags: loads, `adrp`/`add`, and `mov` only.
    fn mchunk(&mut self, o: Operand, sz: MemSize, k: i64, rt: Reg) -> R<()> {
        match o {
            Operand::Value(x) => {
                let h = self.home(x);
                self.ld(sz, false, rt, SP, h + k)
            }
            Operand::Str(s) if sz == MemSize::X => {
                let (sym, so, len) = self.str_ref(s)?;
                if k == 0 {
                    self.a.addr_of_sym(rt, sym, so)?;
                } else {
                    self.a.mov_imm(X, rt, len)?;
                }
                Ok(())
            }
            Operand::Func(f) if sz == MemSize::X => {
                if k == 0 {
                    let s = self.cg.func_sym(f)?;
                    self.a.addr_of_sym(rt, s, 0)?;
                } else {
                    self.a.mov_imm(X, rt, 0)?;
                }
                Ok(())
            }
            _ => Err(bad("aggregate operand must be a value")),
        }
    }

    /// `home h = rc ? a : bo` over `size` bytes, without a branch on `rc`: one `cmp`, then per
    /// 8-byte word both words loaded, `csel`, stored (the flags survive the loads and stores).
    /// Up to [`SELECT_UNROLL`] words are unrolled; a longer value loops over its words with
    /// post-incremented pointers and a `cbnz` counter (the loop branch depends on the size only,
    /// never on `rc`). The tail below 8 bytes is selected in 4/2/1-byte chunks.
    fn select_mem(&mut self, rc: Reg, h: i64, a: Operand, bo: Operand, size: u32) -> R<()> {
        let n8 = i64::from(size / 8);
        let tail = !size.is_multiple_of(8);
        let homes = match (a, bo) {
            (Operand::Value(av), Operand::Value(bv)) => Some((self.home(av), self.home(bv))),
            _ if !tail && n8 <= SELECT_UNROLL => None,
            _ => return Err(bad("aggregate operand must be a value")),
        };
        if n8 <= SELECT_UNROLL {
            self.a.cmp_imm(W, rc, 0)?;
            for k in 0..n8 {
                self.mchunk(bo, MemSize::X, 8 * k, S16)?;
                self.mchunk(a, MemSize::X, 8 * k, S15)?;
                self.a.csel(X, S15, S15, S16, Cond::Ne)?;
                self.st(MemSize::X, S15, SP, h + 8 * k)?;
            }
            return match homes {
                Some((ha, hb)) if tail => self.select_tail([(SP, ha), (SP, hb), (SP, h)], n8 * 8, size),
                _ => Ok(()),
            };
        }
        let Some((ha, hb)) = homes else { return Err(bad("aggregate operand must be a value")) };
        let (pa, pb, pd) = (self.tmp_g()?, self.tmp_g()?, self.tmp_g()?);
        self.add_imm(pa, SP, ha)?;
        self.add_imm(pb, SP, hb)?;
        self.add_imm(pd, SP, h)?;
        self.a.mov_imm(X, S17, n8 as u64)?;
        self.a.cmp_imm(W, rc, 0)?;
        let top = self.a.new_label();
        self.a.bind(top)?;
        self.a.ldr(MemSize::X, S15, Addr::Post(pa, 8))?;
        self.a.ldr(MemSize::X, S16, Addr::Post(pb, 8))?;
        self.a.csel(X, S15, S15, S16, Cond::Ne)?;
        self.a.str(MemSize::X, S15, Addr::Post(pd, 8))?;
        self.a.addsub_imm(AddSub::Sub, X, S17, S17, 1)?;
        self.a.cbnz(X, S17, top)?;
        if !tail {
            return Ok(());
        }
        // the pointers now stand at byte `n8 * 8` of their values
        let at = -n8 * 8;
        self.select_tail([(pa, at), (pb, at), (pd, at)], n8 * 8, size)
    }

    /// Bytes `from..size` of [`Self::select_mem`], after its `cmp`. `loc` holds where byte 0 of
    /// `a`, `b`, and the result is, as `(base, offset)`.
    fn select_tail(&mut self, loc: [(Reg, i64); 3], from: i64, size: u32) -> R<()> {
        let [(ab, ao), (bb, bo), (db, dof)] = loc;
        let mut k = from;
        for (c, sz) in [(4i64, MemSize::W), (2, MemSize::H), (1, MemSize::B)] {
            if i64::from(size) - k < c {
                continue;
            }
            self.ld(sz, false, S16, bb, bo + k)?;
            self.ld(sz, false, S15, ab, ao + k)?;
            self.a.csel(W, S15, S15, S16, Cond::Ne)?;
            self.st(sz, S15, db, dof + k)?;
            k += c;
        }
        Ok(())
    }

    fn copy_into(&mut self, v: Value, t: Ty, o: Operand) -> R<()> {
        match t {
            Ty::Sc(s) if s.is_fp() => {
                let fa = self.use_f(o, s)?;
                let rd = self.def_f(v)?;
                self.a.fmov(s.fp(), rd, fa);
                Ok(())
            }
            Ty::Sc(_) => {
                let ra = self.use_g(o)?;
                let rd = self.def_g(v)?;
                self.a.mov(X, rd, ra)?;
                Ok(())
            }
            Ty::Vec(_) if self.vregs() => {
                let (r, _) = self.use_v(o)?;
                self.set_v(v, r)
            }
            Ty::Fat | Ty::Vec(_) | Ty::Mem { .. } => {
                let h = self.home(v);
                self.put(o, t, SP, h)
            }
            Ty::Bad(why) => Err(unsupported(why)),
        }
    }

    // ---------------------------------------------------------------- casts

    fn cast(&mut self, kind: CastKind, t: TypeId, x: Operand, v: Value) -> R<()> {
        let opt = if kind == CastKind::Try {
            Some(ty::option_shape(&self.cg.m.types, t).ok_or_else(|| bad("cast.try result"))?)
        } else {
            None
        };
        let to_t = opt.map_or(t, |o| o.2);
        let to = match self.cg.ty(to_t) {
            Ty::Sc(s) => s,
            Ty::Bad(why) => return Err(unsupported(why)),
            _ => return Err(bad("cast to a non-scalar")),
        };
        // Constant operands fold with the interpreter's own conversion.
        let folded = match x {
            Operand::Const(c) => match self.b.konst(c) {
                Some(Const::Int(k)) => Some((CastTy::Int(IntTy::new(64, k < 0)), k as u64)),
                Some(Const::F32(b)) => Some((CastTy::Float(FloatBits::F32), u64::from(b))),
                Some(Const::F64(b)) => Some((CastTy::Float(FloatBits::F64), b)),
                _ => return Err(unsupported("f16/bf16 constants")),
            },
            _ => None,
        };
        if let Some((from, bits)) = folded {
            let r = arith::convert(kind, from, to.cast_ty(), bits);
            return match opt {
                None => {
                    let r = r.ok_or_else(|| bad("constant cast"))?;
                    self.def_const(v, to, r)
                }
                Some((none, some, _)) => self.option_const(v, t, none, some, r),
            };
        }
        let from = match x {
            Operand::Value(xv) => match self.vty(xv) {
                Ty::Sc(s) => s,
                _ => return Err(bad("cast of a non-scalar")),
            },
            Operand::Func(_) => Sc::Ptr,
            _ => return Err(bad("cast operand")),
        };
        let src = if from.is_fp() {
            Payload::F(self.use_f(x, from)?, from.fp())
        } else {
            Payload::G(self.use_g(x)?)
        };
        let Some((_, some, _)) = opt else {
            let dst =
                if to.is_fp() { Payload::F(self.def_f(v)?, to.fp()) } else { Payload::G(self.def_g(v)?) };
            self.cast_core(kind, from, to, src, dst)?;
            return Ok(());
        };
        let dst = if to.is_fp() { Payload::F(self.tmp_f()?, to.fp()) } else { Payload::G(self.tmp_g()?) };
        let ok = self.cast_core(kind, from, to, src, dst)?.ok_or_else(|| bad("cast.try"))?;
        self.option_write(v, t, some, ok, dst, to)
    }

    /// The scalar cast of `src` (of class `from`) to class `to`, into `dst`: a register of the
    /// result's class that is not `src`. For `Try`, returns a register holding 1 if the value
    /// converts and 0 if not (`dst` is then unspecified). Straight-line code.
    pub(super) fn cast_core(
        &mut self,
        kind: CastKind,
        from: Sc,
        to: Sc,
        src: Payload,
        dst: Payload,
    ) -> R<Option<Reg>> {
        match (src, dst) {
            (Payload::G(ra), Payload::G(rd)) => match kind {
                CastKind::Exact | CastKind::Wrap | CastKind::Round => {
                    self.a.mov(X, rd, ra)?;
                    if to != Sc::Ptr && to != Sc::Char {
                        self.norm(rd, to)?;
                    }
                    Ok(None)
                }
                CastKind::Sat => {
                    self.a.mov(X, rd, ra)?;
                    self.clamp(rd, from.range(), from.signed(), to)?;
                    Ok(None)
                }
                CastKind::Try => {
                    let ok = self.tmp_g()?;
                    if to == Sc::Char {
                        let t1 = self.tmp_g()?;
                        self.a.mov_imm(X, t1, 0x10_ffff)?;
                        self.a.cmp(X, ra, t1)?;
                        self.a.cset(W, ok, Cond::Ls)?;
                        self.a.mov_imm(X, t1, 0xd800)?;
                        self.a.sub(X, t1, ra, t1)?;
                        self.a.cmp_imm(X, t1, 0x7ff)?;
                        self.a.cset(W, t1, Cond::Hi)?;
                        self.a.logic_reg(Logic::And, W, ok, ok, t1, Shift::Lsl, 0)?;
                    } else {
                        self.fits_check(ok, ra, from, to)?;
                    }
                    self.a.mov(X, rd, ra)?;
                    Ok(Some(ok))
                }
            },
            (Payload::G(ra), Payload::F(rd, fp)) => {
                if from.signed() {
                    self.a.scvtf(fp, rd, X, ra)?;
                } else {
                    self.a.ucvtf(fp, rd, X, ra)?;
                }
                if kind != CastKind::Try {
                    return Ok(None);
                }
                let lim = self.tmp_f()?;
                let l =
                    if from.signed() { 9_223_372_036_854_775_808.0 } else { 18_446_744_073_709_551_616.0 };
                self.mat_f(lim, fp, fbits(l, fp))?;
                self.a.fcmp(fp, rd, lim);
                let ok = self.tmp_g()?;
                self.a.cset(W, ok, Cond::Mi)?;
                let back = self.tmp_g()?;
                if from.signed() {
                    self.a.fcvtzs(X, back, fp, rd)?;
                } else {
                    self.a.fcvtzu(X, back, fp, rd)?;
                }
                self.a.cmp(X, back, ra)?;
                self.a.cset(W, back, Cond::Eq)?;
                self.a.logic_reg(Logic::And, W, ok, ok, back, Shift::Lsl, 0)?;
                Ok(Some(ok))
            }
            (Payload::F(fx, fp), Payload::G(rd)) => match kind {
                CastKind::Try => {
                    let t1 = self.tmp_f()?;
                    self.a.fp_un(FpUn::RintZ, fp, t1, fx);
                    self.a.fcmp(fp, t1, fx);
                    let ok = self.tmp_g()?;
                    self.a.cset(W, ok, Cond::Eq)?;
                    let (tmin, tmax) = to.range();
                    let c2 = self.tmp_g()?;
                    self.mat_f(t1, fp, fbits(tmin as f64, fp))?;
                    self.a.fcmp(fp, fx, t1);
                    self.a.cset(W, c2, Cond::Ge)?;
                    self.a.logic_reg(Logic::And, W, ok, ok, c2, Shift::Lsl, 0)?;
                    self.mat_f(t1, fp, fbits((tmax + 1) as f64, fp))?;
                    self.a.fcmp(fp, fx, t1);
                    self.a.cset(W, c2, Cond::Mi)?;
                    self.a.logic_reg(Logic::And, W, ok, ok, c2, Shift::Lsl, 0)?;
                    if to.signed() {
                        self.a.fcvtzs(X, rd, fp, fx)?;
                    } else {
                        self.a.fcvtzu(X, rd, fp, fx)?;
                    }
                    Ok(Some(ok))
                }
                _ => {
                    match (to.bits(), to.signed()) {
                        (64, true) => self.a.fcvtzs(X, rd, fp, fx)?,
                        (64, false) => self.a.fcvtzu(X, rd, fp, fx)?,
                        (32, true) => {
                            self.a.fcvtzs(W, rd, fp, fx)?;
                            self.a.sxt(X, rd, rd, Ext::W)?;
                        }
                        (32, false) => self.a.fcvtzu(W, rd, fp, fx)?,
                        (_, true) => {
                            self.a.fcvtzs(W, rd, fp, fx)?;
                            self.a.sxt(X, rd, rd, Ext::W)?;
                            self.clamp(rd, Sc::Int { bits: 32, signed: true }.range(), true, to)?;
                        }
                        (_, false) => {
                            self.a.fcvtzu(W, rd, fp, fx)?;
                            self.clamp(rd, Sc::U32.range(), false, to)?;
                        }
                    }
                    Ok(None)
                }
            },
            (Payload::F(fx, ff), Payload::F(rd, tf)) => {
                if kind == CastKind::Try && ff == Fp::D && tf == Fp::S {
                    let ok = self.tmp_g()?;
                    self.a.fcvt(Fp::S, rd, Fp::D, fx)?;
                    let back = self.tmp_f()?;
                    self.a.fcvt(Fp::D, back, Fp::S, rd)?;
                    self.a.fcmp(Fp::D, back, fx);
                    self.a.cset(W, ok, Cond::Eq)?;
                    let c2 = self.tmp_g()?;
                    self.a.cset(W, c2, Cond::Vs)?;
                    self.a.logic_reg(Logic::Orr, W, ok, ok, c2, Shift::Lsl, 0)?;
                    return Ok(Some(ok));
                }
                if ff == tf {
                    self.a.fmov(tf, rd, fx);
                } else {
                    self.a.fcvt(tf, rd, ff, fx)?;
                }
                if kind != CastKind::Try {
                    return Ok(None);
                }
                let ok = self.tmp_g()?;
                self.a.mov_imm(W, ok, 1)?;
                Ok(Some(ok))
            }
        }
    }

    /// `ok = value in ra (of type from) fits to`.
    pub(super) fn fits_check(&mut self, ok: Reg, ra: Reg, from: Sc, to: Sc) -> R<()> {
        let (fmin, fmax) = from.range();
        let (tmin, tmax) = to.range();
        self.a.mov_imm(W, ok, 1)?;
        if fmax > tmax {
            let t = self.tmp_g()?;
            self.a.mov_imm(X, t, tmax as u64)?;
            self.a.cmp(X, ra, t)?;
            self.a.cset(W, ok, if from.signed() { Cond::Le } else { Cond::Ls })?;
        }
        if fmin < tmin {
            let t = self.tmp_g()?;
            self.a.mov_imm(X, t, tmin as i64 as u64)?;
            self.a.cmp(X, ra, t)?;
            self.a.cset(W, t, Cond::Ge)?;
            self.a.logic_reg(Logic::And, W, ok, ok, t, Shift::Lsl, 0)?;
        }
        Ok(())
    }

    fn def_const(&mut self, v: Value, to: Sc, bits: u64) -> R<()> {
        if to.is_fp() {
            let rd = self.def_f(v)?;
            self.mat_f(rd, to.fp(), bits)
        } else {
            let rd = self.def_g(v)?;
            Ok(self.a.mov_imm(X, rd, bits)?)
        }
    }

    pub(super) fn option_layout(&self, t: TypeId, some: u32) -> R<(u32, i64)> {
        let l = self.layout(t)?;
        let off = ty::variant_field(l, i128::from(some), 0).ok_or_else(|| bad("option payload"))?.0;
        Ok((l.size as u32, off as i64))
    }

    fn option_const(&mut self, v: Value, t: TypeId, none: u32, some: u32, r: Option<u64>) -> R<()> {
        let (size, off) = self.option_layout(t, some)?;
        let h = self.home(v);
        self.memzero(SP, h, size)?;
        let tag = if r.is_some() { some } else { none };
        if tag != 0 {
            self.a.mov_imm(W, S15, u64::from(tag))?;
            self.st(MemSize::W, S15, SP, h)?;
        }
        if let Some(bits) = r {
            let pt = ty::option_shape(&self.cg.m.types, t).map(|o| self.cg.ty(o.2));
            let Some(Ty::Sc(ps)) = pt else { return Err(bad("option payload")) };
            if bits != 0 {
                self.a.mov_imm(X, S15, bits)?;
                self.st(ps.msize(), S15, SP, h + off)?;
            }
        }
        Ok(())
    }

    /// Writes `Option` value `v`: `Some(payload)` if `ok` (0/1 in a W register), else `None`
    /// (payload bytes zero).
    #[allow(clippy::too_many_arguments)]
    fn option_write(&mut self, v: Value, t: TypeId, some: u32, ok: Reg, p: Payload, ps: Sc) -> R<()> {
        let (size, off) = self.option_layout(t, some)?;
        let h = self.home(v);
        let pv = match p {
            Payload::G(r) => r,
            Payload::F(f, fp) => {
                let r = self.tmp_g()?;
                self.a.fmov_to_gpr(fp, r, f)?;
                r
            }
        };
        let tag = self.tmp_g()?;
        if some == 1 {
            self.a.mov(W, tag, ok)?;
        } else {
            self.a.logic_imm(Logic::Eor, W, tag, ok, 1)?;
        }
        self.memzero(SP, h, size)?;
        self.st(MemSize::W, tag, SP, h)?;
        self.a.cmp_imm(W, ok, 0)?;
        self.a.csel(X, tag, pv, ZR, Cond::Ne)?;
        self.st(ps.msize(), tag, SP, h + off)
    }

    // ---------------------------------------------------------------- aggregates and memory

    fn index_addr(&mut self, base: Reg, idx: Reg, stride: u64) -> R<()> {
        if stride == 0 {
            return Ok(());
        }
        if stride.is_power_of_two() {
            self.a.addsub_reg(AddSub::Add, X, base, base, idx, Shift::Lsl, stride.trailing_zeros())?;
        } else {
            let t = self.tmp_g()?;
            self.a.mov_imm(X, t, stride)?;
            self.a.madd(X, base, idx, t, base)?;
        }
        Ok(())
    }

    fn get_set(&mut self, op: Op, i: u32, site: Site) -> R<()> {
        let args = self.b.args(i);
        let v = self.res(i)?;
        let agg = *args.first().ok_or_else(|| bad("get operand"))?;
        let Operand::Value(av) = agg else { return Err(bad("get of a non-value")) };
        let at = self.value_tid(agg)?;
        let idx = *args.get(1).ok_or_else(|| bad("index"))?;
        let l = self.layout(at)?;
        let size = l.size as u32;
        let ha = self.home(av);
        let hv = self.home(v);
        let (off, ft, dynamic) = match (&l.shape, self.b.int_const(idx)) {
            (Shape::Fields(fs), Some(k)) => {
                let f = usize::try_from(k).ok().and_then(|k| fs.get(k)).ok_or_else(|| bad("field"))?;
                (f.0 as i64, f.1, None)
            }
            (Shape::Array { elem, stride, len }, Some(k)) => {
                let k = u64::try_from(k).ok().filter(|k| k < len).ok_or_else(|| bad("constant index"))?;
                ((k * stride) as i64, *elem, None)
            }
            (Shape::Array { elem, stride, len }, None) => (0, *elem, Some((*stride, *len))),
            _ => return Err(bad("get/set index")),
        };
        let ft = self.cg.ty(ft);
        if op == Op::Get {
            let Some((stride, len)) = dynamic else { return self.take(v, ft, SP, ha + off) };
            let ri = self.use_g(idx)?;
            self.bounds_reg(ri, len, site)?;
            let base = self.tmp_g()?;
            self.add_imm(base, SP, ha)?;
            self.index_addr(base, ri, stride)?;
            return self.take(v, ft, base, 0);
        }
        let val = *args.get(2).ok_or_else(|| bad("set value"))?;
        match dynamic {
            None => {
                self.memcpy(SP, hv, SP, ha, size)?;
                self.put(val, ft, SP, hv + off)
            }
            Some((stride, len)) => {
                let ri = self.use_g(idx)?;
                self.bounds_reg(ri, len, site)?;
                self.memcpy(SP, hv, SP, ha, size)?;
                let base = self.tmp_g()?;
                self.add_imm(base, SP, hv)?;
                self.index_addr(base, ri, stride)?;
                self.put(val, ft, base, 0)
            }
        }
    }

    fn bounds_reg(&mut self, ri: Reg, len: u64, site: Site) -> R<()> {
        let l = self.trap(PanicKind::Bounds, site)?;
        if len < 4096 {
            self.a.cmp_imm(X, ri, len)?;
        } else {
            let t = self.tmp_g()?;
            self.a.mov_imm(X, t, len)?;
            self.a.cmp(X, ri, t)?;
        }
        self.a.b_cond(Cond::Hs, l)?;
        Ok(())
    }

    fn bounds(&mut self, i: Operand, n: Operand, site: Site) -> R<()> {
        let l = self.trap(PanicKind::Bounds, site)?;
        match (self.b.int_const(i), self.b.int_const(n)) {
            (Some(a), Some(c)) => {
                if a >= c {
                    self.a.b(l)?;
                }
            }
            (None, Some(k)) if (0..4096).contains(&k) => {
                let ri = self.use_g(i)?;
                self.a.cmp_imm(X, ri, k as u64)?;
                self.a.b_cond(Cond::Hs, l)?;
            }
            (Some(k), None) if (0..4096).contains(&k) => {
                let rn = self.use_g(n)?;
                self.a.cmp_imm(X, rn, k as u64)?;
                self.a.b_cond(Cond::Ls, l)?;
            }
            _ => {
                let ri = self.use_g(i)?;
                let rn = self.use_g(n)?;
                self.a.cmp(X, ri, rn)?;
                self.a.b_cond(Cond::Hs, l)?;
            }
        }
        Ok(())
    }

    /// `prfm` at `p + i * size` (the size of `p`'s pointee): `pld`/`pst` by intent, `l1keep l2keep
    /// l3keep l1strm` by level.
    fn prefetch(&mut self, h: stave_ir::Prefetch, p: Operand, i: Operand) -> R<()> {
        use stave_ir::PrefetchLoc as L;
        let (level, stream) = match h.loc {
            L::L1 => (PrefLevel::L1, false),
            L::L2 => (PrefLevel::L2, false),
            L::L3 => (PrefLevel::L3, false),
            L::Nta => (PrefLevel::L1, true),
        };
        let kind = if h.write { PrefKind::Pst } else { PrefKind::Pld };
        let pt = self.value_tid(p)?;
        let inner = ty::pointee(&self.cg.m.types, pt).ok_or_else(|| bad("prefetch of a non-pointer"))?;
        let stride = self.layout(inner)?.size;
        let rb = self.use_g(p)?;
        let t = self.tmp_g()?;
        if let Some(k) = self.b.int_const(i) {
            self.add_imm(t, rb, (k as i64).wrapping_mul(stride as i64))?;
        } else {
            let ri = self.use_g(i)?;
            if stride == 0 {
                self.a.mov(X, t, rb)?;
            } else if stride.is_power_of_two() {
                self.a.addsub_reg(AddSub::Add, X, t, rb, ri, Shift::Lsl, stride.trailing_zeros())?;
            } else {
                let s = self.tmp_g()?;
                self.a.mov_imm(X, s, stride)?;
                self.a.madd(X, t, ri, s, rb)?;
            }
        }
        self.a.prfm(Prefetch { kind, level, stream }, Addr::Off(t, 0))?;
        Ok(())
    }

    fn elem_addr(&mut self, i: u32) -> R<()> {
        let args = self.b.args(i);
        let v = self.res(i)?;
        let p = *args.first().ok_or_else(|| bad("addr.elem pointer"))?;
        let idx = *args.get(1).ok_or_else(|| bad("addr.elem index"))?;
        let pt = self.value_tid(p)?;
        let inner = ty::pointee(&self.cg.m.types, pt).ok_or_else(|| bad("addr.elem of a non-pointer"))?;
        let et = match self.cg.m.types.key(inner) {
            Some(TypeKey::Array { elem, .. } | TypeKey::Slice { elem, .. }) => *elem,
            _ => inner,
        };
        let stride = self.layout(et)?.size;
        let Operand::Value(pv) = p else { return Err(bad("addr.elem pointer")) };
        let rb = if matches!(self.vkind(pv), Vk::Mem(_)) {
            let t = self.tmp_g()?;
            let h = self.home(pv);
            self.ld(MemSize::X, false, t, SP, h)?;
            t
        } else {
            self.use_g(p)?
        };
        if let Some(k) = self.b.int_const(idx) {
            let off = (k as i64).wrapping_mul(stride as i64);
            let rd = self.def_g(v)?;
            return self.add_imm(rd, rb, off);
        }
        let ri = self.use_g(idx)?;
        let rd = self.def_g(v)?;
        if stride == 0 {
            self.a.mov(X, rd, rb)?;
        } else if stride.is_power_of_two() {
            self.a.addsub_reg(AddSub::Add, X, rd, rb, ri, Shift::Lsl, stride.trailing_zeros())?;
        } else {
            let t = self.tmp_g()?;
            self.a.mov_imm(X, t, stride)?;
            self.a.madd(X, rd, ri, t, rb)?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------- intrinsics

    fn intrinsic(&mut self, id: stave_ir::IntrinsicId, i: u32) -> R<()> {
        let args = self.b.args(i);
        let res = self.b.insts.result.get(i as usize).copied().flatten();
        let kind = self.cg.intr.get(id.index()).copied().unwrap_or(Intr::Unknown);
        let syms = self.cg.syms;
        let m = self.cg.m;
        let name = || m.intrinsic_name(id).unwrap_or("?").to_string();
        let addr_of = |g: &mut Self, sym: Option<stave_obj::SymbolId>, what: &str| -> R<()> {
            let sym = sym.ok_or_else(|| unsupported(format!("{what} is not emitted for this program")))?;
            let v = res.ok_or_else(|| bad("intrinsic result"))?;
            let rd = g.def_g(v)?;
            Ok(g.a.addr_of_sym(rd, sym, 0)?)
        };
        match kind {
            Intr::Syscall(n) => {
                self.flush_all()?;
                if args.len() != usize::from(n) + 1 {
                    return Err(bad("syscall argument count"));
                }
                for k in 0..usize::from(n) {
                    self.load_to_x(xr(k as u8), args[k + 1])?;
                }
                self.load_to_x(xr(8), args[0])?;
                self.a.svc(0);
                self.result_regs(Ret::X0, res)
            }
            Intr::Ccall(n) => {
                self.flush_all()?;
                if args.len() != usize::from(n) + 1 {
                    return Err(bad("ccall argument count"));
                }
                for k in 8..usize::from(n) {
                    self.load_to_x(S15, args[k + 1])?;
                    self.st(MemSize::X, S15, SP, 8 * (k as i64 - 8))?;
                }
                for k in 0..usize::from(n).min(8) {
                    self.load_to_x(xr(k as u8), args[k + 1])?;
                }
                self.load_to_x(S16, args[0])?;
                self.a.blr(S16)?;
                self.result_regs(Ret::X0, res)
            }
            Intr::CallOnStack => {
                self.flush_all()?;
                if args.len() != 3 {
                    return Err(bad("call_on_stack argument count"));
                }
                self.load_to_x(S16, args[0])?;
                self.load_to_x(xr(0), args[1])?;
                self.load_to_x(S17, args[2])?;
                let x19 = xr(19);
                self.a.stp(X, x19, crate::enc::Reg::LR, Addr::Pre(SP, -16))?;
                self.a.mov(X, x19, SP)?;
                self.a.mov(X, SP, S17)?;
                self.a.blr(S16)?;
                self.a.mov(X, SP, x19)?;
                self.a.ldp(X, x19, crate::enc::Reg::LR, Addr::Post(SP, 16))?;
                self.result_regs(Ret::X0, res)
            }
            Intr::MainAddr => addr_of(self, syms.main_stub, "__stave_main"),
            Intr::SignalAddr => addr_of(self, syms.signal_stub, "__stave_signal"),
            Intr::ThreadAddr => addr_of(self, syms.thread_stub, "__stave_thread_start"),
            Intr::SiteTable => addr_of(self, Some(syms.site_table), "__stave_site_table"),
            Intr::PcTable => addr_of(self, Some(syms.pc_table), "__stave_pc_table"),
            Intr::StackTable => addr_of(self, Some(syms.stack_table), "__stave_stack_table"),
            Intr::StaticAddr => {
                let k = args
                    .first()
                    .and_then(|o| self.b.int_const(*o))
                    .ok_or_else(|| bad("static_addr index"))?;
                let sym = usize::try_from(k).ok().and_then(|k| syms.statics.get(k)).copied();
                addr_of(self, Some(sym.ok_or_else(|| bad("static index out of range"))?), "static")
            }
            Intr::ThreadClone => {
                // thread_clone(nr, a1..a5, arg): x8 = nr, x0..x4 = a1..a5, x5 = arg; `svc #0`; the
                // child (x0 == 0, on its new stack) calls __stave_thread_start(arg), then `exit` (93)
                self.flush_all()?;
                if args.len() != 7 {
                    return Err(bad("thread_clone argument count"));
                }
                let stub = syms
                    .thread_stub
                    .ok_or_else(|| unsupported("thread_clone: the program has no `runtime.thread.entry`"))?;
                for k in 0..6usize {
                    self.load_to_x(xr(k as u8), args[k + 1])?;
                }
                self.load_to_x(xr(8), args[0])?;
                self.a.svc(0);
                let parent = self.a.new_label();
                self.a.cbnz(X, xr(0), parent)?;
                // the child's stack top is the caller's: align it down to 16 (sp-relative access
                // faults otherwise), as the x86-64 backend does
                self.a.mov(X, S16, SP)?;
                self.a.logic_imm(Logic::And, X, S16, S16, !15u64)?;
                self.a.mov(X, SP, S16)?;
                self.a.mov(X, xr(0), xr(5))?;
                self.a.mov(X, xr(29), ZR)?;
                self.a.mov(X, crate::enc::Reg::LR, ZR)?;
                self.a.bl_sym(stub);
                self.a.mov_imm(X, xr(8), 93)?;
                self.a.svc(0);
                self.a.brk(0);
                self.a.bind(parent)?;
                self.result_regs(Ret::X0, res)
            }
            Intr::ThreadBlock => {
                self.flush_all()?;
                if !args.is_empty() {
                    return Err(bad("thread_block argument count"));
                }
                self.a.word(0xD53B_D040); // mrs x0, tpidr_el0
                self.result_regs(Ret::X0, res)
            }
            Intr::SetThreadBlock => {
                self.flush_all()?;
                if args.len() != 1 {
                    return Err(bad("set_thread_block argument count"));
                }
                self.load_to_x(xr(0), args[0])?;
                self.a.word(0xD51B_D040); // msr tpidr_el0, x0
                Ok(())
            }
            Intr::CpuInit => match syms.cpu_init {
                Some(sym) => self.call(Ok(sym), &[], None, &[], None),
                None => Ok(()),
            },
            Intr::Atomic { op, wide } => {
                // x1 = p, x2 = value / new, x3 = expected; result x0 (sequentially consistent:
                // ldar, or an ldaxr/stlxr loop)
                self.flush_all()?;
                let n = [1usize, 2, 2, 3][usize::from(op)];
                if args.len() != n {
                    return Err(bad("atomic argument count"));
                }
                self.load_to_x(xr(1), args[0])?;
                if n >= 2 {
                    self.load_to_x(xr(2), args[1])?;
                }
                if n == 3 {
                    self.load_to_x(xr(3), args[1])?;
                    self.load_to_x(xr(2), args[2])?;
                }
                let sz: u32 = if wide { 0xC000_0000 } else { 0x8000_0000 };
                let w = if wide { X } else { crate::enc::Width::W };
                const LDAR: u32 = 0x08DF_FC00;
                const LDAXR: u32 = 0x085F_FC00;
                const STLXR: u32 = 0x0800_FC00;
                let rn1 = 1 << 5;
                if op == 0 {
                    self.a.word(sz | LDAR | rn1);
                    return self.result_regs(Ret::X0, res);
                }
                let top = self.a.new_label();
                let fail = self.a.new_label();
                let done = self.a.new_label();
                self.a.bind(top)?;
                self.a.word(sz | LDAXR | rn1); // ldaxr r0, [x1]
                let src = match op {
                    1 => 2,
                    2 => {
                        self.a.add(w, xr(5), xr(0), xr(2))?;
                        5
                    }
                    _ => {
                        self.a.cmp(w, xr(0), xr(3))?;
                        self.a.b_cond(crate::enc::Cond::Ne, fail)?;
                        2
                    }
                };
                self.a.word(sz | STLXR | (4 << 16) | rn1 | src); // stlxr w4, r<src>, [x1]
                self.a.cbnz(crate::enc::Width::W, xr(4), top)?;
                self.a.b(done)?;
                self.a.bind(fail)?;
                self.a.word(0xD503_305F); // clrex
                self.a.bind(done)?;
                self.result_regs(Ret::X0, res)
            }
            Intr::X86Only => Err(unsupported(format!("intrinsic.{} is x86-64 only", name()))),
            Intr::Unknown => Err(unsupported(format!("intrinsic.{}", name()))),
        }
    }
}
