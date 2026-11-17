//! Instruction selection for the stave-ir op table (contracts/checked-ir.md): integer ops in
//! their trapping / wrapping / saturating forms, f32/f64 via SSE2, compare, select, aggregates,
//! enums, memory and address ops, bounds checks, calls, panics. Casts are in `cast`, intrinsics
//! in `intrin`.
//!
//! Integer values are held normalized to 64 bits, so most ops run at 64 bits; trapping ops on
//! 32/64-bit types use the native width and `jo`/`jc`, narrower ones check that the 64-bit result
//! is unchanged by normalization.

use stave_ir::arith::IntTy;
use stave_ir::layout::Shape;
use stave_ir::{FloatBits, Op, Operand, PanicKind, TypeId, TypeKey, Value};

use super::abi::VCls;
use super::func::*;
use super::{Res, bad, unsup};
use crate::enc::*;

const MIN64: i64 = i64::MIN;

/// Words of an aggregate `select` emitted straight-line; longer values use a counted loop.
const SELECT_UNROLL: u32 = 64;

fn int_min(t: IntTy) -> i64 {
    t.min() as i64
}

fn int_max(t: IntTy) -> i64 {
    t.max() as u64 as i64
}

/// Never traps and has no effect besides its result (skipped when the result is unused).
fn pure(op: Op, float: bool, const_idx: bool) -> bool {
    match op {
        Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem | Op::Neg => float,
        Op::Get | Op::Set => const_idx,
        Op::AddWrap
        | Op::SubWrap
        | Op::MulWrap
        | Op::ShlWrap
        | Op::AddSat
        | Op::SubSat
        | Op::MulSat
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::Not
        | Op::Popcount
        | Op::Clz
        | Op::Ctz
        | Op::MulWide
        | Op::Eq
        | Op::Ne
        | Op::Lt
        | Op::Le
        | Op::Gt
        | Op::Ge
        | Op::Select
        | Op::Cast(_)
        | Op::Make
        | Op::EnumMake
        | Op::EnumTag
        | Op::EnumGet
        | Op::Slot
        | Op::FieldAddr
        | Op::EnumFieldAddr
        | Op::ElemAddr
        | Op::SliceMake
        | Op::SliceLen
        | Op::SlicePtr
        | Op::FnMake => true,
        _ => false,
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

impl Cg<'_, '_> {
    pub(crate) fn inst(&mut self, i: u32) -> Res<()> {
        let b = self.b;
        let iu = i as usize;
        let op = *b.insts.op.get(iu).ok_or_else(|| bad("instruction out of range"))?;
        let ty = b.insts.ty.get(iu).copied().flatten();
        let res = b.insts.result.get(iu).copied().flatten();
        let args = b.args(i);
        let arg = |k: usize| args.get(k).copied().ok_or_else(|| bad(format!("{op:?}: missing operand {k}")));
        if let Some(v) = res
            && !self.an.used.get(v.index()).copied().unwrap_or(true)
        {
            let float = ty.and_then(|t| self.sk(t)).is_some_and(|k| matches!(k, Sk::Float(_)));
            let const_idx = args.get(1).is_some_and(|o| self.int_const(*o).is_some());
            if pure(op, float, const_idx) {
                return Ok(());
            }
        }
        let rv = res.ok_or_else(|| bad(format!("{op:?}: missing result")));
        let rt = ty.ok_or_else(|| bad(format!("{op:?}: missing result type")));
        match op {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem | Op::Neg => match self.sk(rt?) {
                Some(Sk::Int(t)) => match op {
                    Op::Div | Op::Rem => self.int_divrem(op, t, arg(0)?, arg(1)?, rv?),
                    Op::Neg => self.int_neg(t, arg(0)?, rv?),
                    _ => self.int_bin(op, t, arg(0)?, arg(1)?, rv?),
                },
                Some(Sk::Float(FloatBits::F32)) => self.float_op(op, Prec::F32, args, rv?),
                Some(Sk::Float(FloatBits::F64)) => self.float_op(op, Prec::F64, args, rv?),
                _ => Err(unsup(format!("{op:?} on this type"))),
            },
            Op::AddWrap | Op::SubWrap | Op::MulWrap | Op::AddSat | Op::SubSat | Op::MulSat => {
                match self.sk(rt?) {
                    Some(Sk::Int(t)) => self.int_bin(op, t, arg(0)?, arg(1)?, rv?),
                    _ => Err(bad(format!("{op:?} on a non-int type"))),
                }
            }
            Op::And | Op::Or | Op::Xor => match self.sk(rt?) {
                Some(Sk::Int(t)) => self.int_bin(op, t, arg(0)?, arg(1)?, rv?),
                Some(Sk::Bool) => self.int_bin(op, IntTy::new(8, false), arg(0)?, arg(1)?, rv?),
                _ => Err(bad("bitwise op on a non-int type")),
            },
            Op::Shl | Op::Shr | Op::ShlWrap => match self.sk(rt?) {
                Some(Sk::Int(t)) => self.shift(op, t, arg(0)?, arg(1)?, rv?),
                _ => Err(bad("shift of a non-int type")),
            },
            Op::Not => {
                let v = rv?;
                match self.sk(rt?) {
                    Some(Sk::Int(t)) => {
                        let r = self.take_g(arg(0)?)?;
                        self.a.unary(UnaryOp::Not, Size::Q, r)?;
                        if !t.signed {
                            self.norm(r, t)?;
                        }
                        self.bind_g(v, r)
                    }
                    Some(Sk::Bool) => {
                        let r = self.take_g(arg(0)?)?;
                        self.a.alu_ri(AluOp::Xor, Size::D, r, 1)?;
                        self.bind_g(v, r)
                    }
                    _ => Err(bad("not on a non-int type")),
                }
            }
            Op::Popcount | Op::Clz | Op::Ctz => match self.sk(rt?) {
                Some(Sk::Int(t)) => self.count(op, t, arg(0)?, rv?),
                _ => Err(bad("bit count of a non-int type")),
            },
            Op::MulWide => self.mul_wide(arg(0)?, arg(1)?, rt?, rv?),
            Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => self.compare(op, arg(0)?, arg(1)?, rv?),
            Op::Select => self.select(arg(0)?, arg(1)?, arg(2)?, rt?, rv?),
            Op::Cast(k) => self.cast(k, arg(0)?, rt?, rv?),
            Op::Make => self.make(args, rt?, rv?),
            Op::Get => self.get(arg(0)?, arg(1)?, rt?, rv?),
            Op::Set => self.set(arg(0)?, arg(1)?, arg(2)?, rt?, rv?),
            Op::EnumMake => self.enum_make(args, rt?, rv?),
            Op::EnumTag => {
                let v = rv?;
                let Operand::Value(e) = arg(0)? else { return Err(bad("enum.tag of a non-value")) };
                let m = self.hmem(e)?;
                let r = self.alloc_g()?;
                self.a.mov_rm(Size::D, r, m)?;
                self.bind_g(v, r)
            }
            Op::EnumGet => {
                let v = rv?;
                let Operand::Value(e) = arg(0)? else { return Err(bad("enum.get of a non-value")) };
                let (off, ft) = self.enum_field(self.vty(e)?, arg(1)?, arg(2)?)?;
                self.part(v, e, off, ft)
            }
            Op::Slot => {
                let v = rv?;
                let pt = self.pointee(rt?)?;
                let l = self.layout(pt)?;
                let size = u32::try_from(l.size).map_err(|_| unsup("slot larger than 4 GiB"))?;
                if l.align > u64::from(self.fx.vector_bytes()) {
                    return Err(unsup("slot alignment above the widest vector register"));
                }
                let off = self.alloc_frame(size.max(1), l.align as u32)?;
                self.def_slot(v, off)
            }
            Op::Load => {
                let v = rv?;
                let c = self.vcls(rt?)?;
                let m = self.ptr_mem(arg(0)?)?;
                self.load_cls(v, c, m)
            }
            Op::Store => {
                let p = arg(0)?;
                let pt = self.oty(p).ok_or_else(|| bad("store through a non-value"))?;
                let c = self.vcls(self.pointee(pt)?)?;
                let m = self.ptr_mem(p)?;
                self.store_cls(arg(1)?, c, m)
            }
            Op::FieldAddr | Op::EnumFieldAddr => {
                let v = rv?;
                let p = arg(0)?;
                let pt = self.oty(p).ok_or_else(|| bad("addr of a non-value"))?;
                let inner = self.pointee(pt)?;
                let off = if op == Op::FieldAddr {
                    self.field(inner, arg(1)?)?.0
                } else {
                    self.enum_field(inner, arg(1)?, arg(2)?)?.0
                };
                self.addr_plus(v, p, off)
            }
            Op::ElemAddr => self.elem_addr(arg(0)?, arg(1)?, rv?),
            Op::SliceMake => {
                let v = rv?;
                let d = self.def_mem(v)?;
                let g8 = VCls::G { bytes: 8, signed: false };
                self.store_cls(arg(0)?, g8, d)?;
                self.store_cls(arg(1)?, g8, disp(d, 8)?)
            }
            Op::SliceLen | Op::SlicePtr => {
                let v = rv?;
                let r = self.alloc_g()?;
                match self.msrc(arg(0)?)? {
                    MSrc::Mem(m) => {
                        self.a.mov_rm(Size::Q, r, if op == Op::SliceLen { disp(m, 8)? } else { m })?
                    }
                    MSrc::Str(s) => {
                        if op == Op::SliceLen {
                            let n = self.str_len(s)?;
                            self.a.mov_ri(Size::Q, r, n)?;
                        } else {
                            let sym = self.str_sym(s)?;
                            self.a.lea(Size::Q, r, Mem::rip_sym(sym, 0))?;
                        }
                    }
                    MSrc::Func(_) => return Err(bad("slice.len / slice.ptr of a fn value")),
                }
                self.bind_g(v, r)
            }
            Op::BoundsCheck => self.bounds(arg(0)?, arg(1)?),
            Op::Prefetch(h) => self.prefetch(h, arg(0)?, arg(1)?),
            Op::StoreStream => {
                let p = arg(0)?;
                let pt = self.oty(p).ok_or_else(|| bad("mem.store_stream through a non-value"))?;
                let c = self.vcls(self.pointee(pt)?)?;
                let m = self.ptr_mem(p)?;
                self.store_stream_cls(arg(1)?, c, m)
            }
            Op::FenceStream => {
                self.a.sfence();
                Ok(())
            }
            Op::Call => {
                let Operand::Func(g) = arg(0)? else { return Err(bad("call callee must be a fn item")) };
                let gf = self.cx.m.func(g).ok_or_else(|| bad("callee out of range"))?;
                let abi = self.abi_of(&gf.params, gf.ret)?;
                let sym = self.func_sym(g)?;
                self.call(Callee::Sym(sym), &args[1..], &abi, res, false)
            }
            Op::CallInd => {
                let f = arg(0)?;
                let ft = match f {
                    Operand::Func(g) => {
                        let gf = self.cx.m.func(g).ok_or_else(|| bad("callee out of range"))?;
                        let abi = self.abi_of_env(&gf.params, gf.ret)?;
                        return self.call(Callee::Fn(f), &args[1..], &abi, res, false);
                    }
                    _ => self.oty(f).ok_or_else(|| bad("call.ind callee must be a value"))?,
                };
                let Some(TypeKey::FnType { params, ret, .. }) = self.key(ft) else {
                    return Err(bad("call.ind of a non-fn value"));
                };
                let abi = self.abi_of_env(params, Some(*ret))?;
                self.call(Callee::Fn(f), &args[1..], &abi, res, false)
            }
            Op::FnMake => {
                let v = rv?;
                let d = self.def_mem(v)?;
                let g8 = VCls::G { bytes: 8, signed: false };
                self.store_cls(arg(0)?, g8, d)?;
                self.store_cls(arg(1)?, g8, disp(d, 8)?)
            }
            Op::Intrinsic(id) => {
                let name = self.cx.m.intrinsic_name(id).ok_or_else(|| bad("intrinsic id out of range"))?;
                self.intrinsic(name, args, ty, res)
            }
            Op::Vec(v) => self.vec_op(v, args, ty, res),
            Op::Panic(k) => self.trap_now(k),
        }
    }

    // ---------------------------------------------------------------- helpers

    pub(crate) fn pointee(&self, pt: TypeId) -> Res<TypeId> {
        match self.key(pt) {
            Some(TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. }) => Ok(*inner),
            _ => Err(bad("pointer operand of a non-pointer type")),
        }
    }

    /// The displacement or index scale an address instruction adds to its base pointer, when
    /// that fits one x86 addressing mode: `Ok(scale)` for `base + index * scale`, `Err(disp)`
    /// for `base + disp`.
    fn addr_form(&self, i: u32) -> Res<Option<Result<Scale, i32>>> {
        let b = self.b;
        let args = b.args(i);
        let (Some(&p), Some(&k)) = (args.first(), args.get(1)) else { return Ok(None) };
        let Some(pt) = self.oty(p) else { return Ok(None) };
        let inner = self.pointee(pt)?;
        let off = match b.insts.op[i as usize] {
            Op::FieldAddr => self.field(inner, k)?.0,
            Op::EnumFieldAddr => match args.get(2) {
                Some(&f) => self.enum_field(inner, k, f)?.0,
                None => return Ok(None),
            },
            Op::ElemAddr => {
                let elem = match self.key(inner) {
                    Some(TypeKey::Array { elem, .. }) => *elem,
                    // a fat base is read from the slice value first
                    Some(TypeKey::Slice { .. }) => return Ok(None),
                    _ => inner,
                };
                let stride = self.layout(elem)?.size;
                match self.int_const(k) {
                    Some(c) => (c as i64).wrapping_mul(stride as i64),
                    None => {
                        return Ok(match stride {
                            1 => Some(Ok(Scale::S1)),
                            2 => Some(Ok(Scale::S2)),
                            4 => Some(Ok(Scale::S4)),
                            8 => Some(Ok(Scale::S8)),
                            _ => None,
                        });
                    }
                }
            }
            _ => return Ok(None),
        };
        Ok(i32::try_from(off).ok().map(Err))
    }

    /// Release: whether address instruction `i` is left to the load or store that reads its
    /// result: the result has that one use, a few instructions further in the block, as the
    /// address. The base and the index then stay alive up to that instruction.
    pub(crate) fn fold_addr(&mut self, i: u32, bd: &stave_ir::ir::BlockData) -> Res<bool> {
        const WINDOW: u32 = 4;
        let b = self.b;
        if !matches!(b.insts.op[i as usize], Op::ElemAddr | Op::FieldAddr | Op::EnumFieldAddr) {
            return Ok(false);
        }
        let Some(v) = b.insts.result[i as usize] else { return Ok(false) };
        if self.uses.get(v.index()).copied() != Some(1) {
            return Ok(false);
        }
        let args = b.args(i);
        let (Some(&base), Some(&index)) = (args.first(), args.get(1)) else { return Ok(false) };
        // a frame address needs no register as it is
        if !matches!(base, Operand::Value(_)) || self.slot_off(base).is_some() {
            return Ok(false);
        }
        let Some(form) = self.addr_form(i)? else { return Ok(false) };
        let end = bd.insts.start.saturating_add(bd.insts.len);
        let me = Operand::Value(v);
        for k in i + 1..end.min(i + 1 + WINDOW) {
            let ka = b.args(k);
            if !ka.contains(&me) {
                continue;
            }
            // the class of what the instruction reads or writes through the address
            let class = match b.insts.op[k as usize] {
                Op::Load if ka == [me] => b.insts.ty[k as usize].map(|t| self.vcls(t)),
                Op::Store if ka.len() == 2 && ka[0] == me && ka[1] != me => {
                    Some(self.pointee(self.vty(v)?).and_then(|t| self.vcls(t)))
                }
                _ => None,
            };
            let Some(class) = class else { return Ok(false) };
            // an aggregate is copied through `[base + disp]` only
            if matches!(class?, VCls::M { .. } | VCls::V { .. }) && form.is_ok() {
                return Ok(false);
            }
            let at = self.pos + (k - i);
            self.extend_use(base, at);
            if form.is_ok() {
                self.extend_use(index, at);
            }
            self.folded[v.index()] = i;
            return Ok(true);
        }
        Ok(false)
    }

    /// `[p]` for a thin pointer operand (a slot address needs no register).
    fn ptr_mem(&mut self, p: Operand) -> Res<Mem> {
        if let Some(off) = self.slot_off(p) {
            return Ok(Mem::base(RBP, off));
        }
        if let Operand::Value(v) = p
            && let Some(&i) = self.folded.get(v.index())
            && i != super::analysis::NONE
        {
            let b = self.b;
            let (Some(&base), Some(&index)) = (b.args(i).first(), b.args(i).get(1)) else {
                return Err(bad("internal: folded address"));
            };
            let base = self.read_g(base)?;
            return Ok(match self.addr_form(i)? {
                Some(Ok(scale)) => Mem::index(base, self.read_g(index)?, scale, 0),
                Some(Err(d)) => Mem::base(base, d),
                None => return Err(bad("internal: folded address")),
            });
        }
        let r = self.read_g(p)?;
        Ok(Mem::base(r, 0))
    }

    /// Offset and type of field `k` of a record type.
    fn field(&self, t: TypeId, k: Operand) -> Res<(i64, TypeId)> {
        let k = self.int_const(k).ok_or_else(|| bad("field index must be a constant"))?;
        match &self.layout(t)?.shape {
            Shape::Fields(fs) => {
                let (o, ft) =
                    usize::try_from(k).ok().and_then(|k| fs.get(k)).ok_or_else(|| bad("field index"))?;
                Ok((*o as i64, *ft))
            }
            Shape::Fat => match k {
                0 | 1 => Err(unsup("field of a fat pointer")),
                _ => Err(bad("field index")),
            },
            _ => Err(bad("field of a non-record type")),
        }
    }

    fn enum_field(&self, t: TypeId, v: Operand, f: Operand) -> Res<(i64, TypeId)> {
        let v = self.int_const(v).ok_or_else(|| bad("variant index must be a constant"))?;
        let f = self.int_const(f).ok_or_else(|| bad("field index must be a constant"))?;
        match &self.layout(t)?.shape {
            Shape::Enum(vs) => {
                let (o, ft) = usize::try_from(v)
                    .ok()
                    .and_then(|v| vs.get(v))
                    .and_then(|fs| usize::try_from(f).ok().and_then(|f| fs.get(f)))
                    .ok_or_else(|| bad("enum field index"))?;
                Ok((*o as i64, *ft))
            }
            _ => Err(bad("enum field of a non-enum type")),
        }
    }

    /// Defines `v` as the part at `off` (type `ft`) of aggregate `root`.
    fn part(&mut self, v: Value, root: Value, off: i64, ft: TypeId) -> Res<()> {
        let c = self.vcls(ft)?;
        if matches!(c, VCls::M { .. }) && self.alias(v, root, off)? {
            return Ok(());
        }
        let m = disp(self.hmem(root)?, off)?;
        self.load_cls(v, c, m)
    }

    /// `v = p + off` (a constant offset; stays a frame address for slots).
    fn addr_plus(&mut self, v: Value, p: Operand, off: i64) -> Res<()> {
        // address arithmetic wraps (mod 2^64), like the interpreter's
        if let Some(so) = self.slot_off(p)
            && let Ok(o) = i32::try_from(i64::from(so).wrapping_add(off))
        {
            return self.def_slot(v, o);
        }
        let r = self.take_g(p)?;
        if let Ok(o) = i32::try_from(off) {
            if o != 0 {
                self.a.lea(Size::Q, r, Mem::base(r, o))?;
            }
        } else {
            self.a.mov_ri(Size::Q, S0, off)?;
            self.a.alu_rr(AluOp::Add, Size::Q, r, S0);
        }
        self.bind_g(v, r)
    }

    fn alu_gi(&mut self, op: AluOp, s: Size, r: Gpr, b: GI) -> Res<()> {
        match b {
            GI::R(x) => self.a.alu_rr(op, s, r, x),
            GI::I(i) => self.a.alu_ri(op, s, r, i64::from(i))?,
        }
        Ok(())
    }

    fn imul_gi(&mut self, s: Size, r: Gpr, b: GI) -> Res<()> {
        match b {
            GI::R(x) => self.a.imul_rr(s, r, x)?,
            GI::I(i) => self.a.imul_rri(s, r, r, i64::from(i))?,
        }
        Ok(())
    }

    /// Traps `Overflow` unless `r` is a normalized value of `t`.
    fn fits_check(&mut self, r: Gpr, t: IntTy) -> Res<()> {
        self.norm_into(S0, r, t)?;
        self.a.alu_rr(AluOp::Cmp, Size::Q, S0, r);
        self.jtrap(Cond::Ne, PanicKind::Overflow)
    }

    /// Clamps `r` (a 64-bit result of exact arithmetic on normalized values) to `t`'s range;
    /// `signed_cmp` selects the comparison of the 64-bit value.
    pub(crate) fn clamp(&mut self, r: Gpr, t: IntTy, lo: bool, hi: bool, signed_cmp: bool) -> Res<()> {
        if hi {
            self.a.mov_ri(Size::Q, S0, int_max(t))?;
            self.a.alu_rr(AluOp::Cmp, Size::Q, r, S0);
            self.a.cmov(if signed_cmp { Cond::G } else { Cond::A }, Size::Q, r, S0)?;
        }
        if lo {
            self.a.mov_ri(Size::Q, S0, int_min(t))?;
            self.a.alu_rr(AluOp::Cmp, Size::Q, r, S0);
            self.a.cmov(if signed_cmp { Cond::L } else { Cond::B }, Size::Q, r, S0)?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------- integer arithmetic

    fn int_bin(&mut self, op: Op, t: IntTy, a: Operand, b: Operand, v: Value) -> Res<()> {
        if t.bits == 64 && !t.signed && matches!(op, Op::Mul | Op::MulSat) {
            return self.mul_u64(op, a, b, v);
        }
        let sat64 = t.bits == 64 && matches!(op, Op::AddSat | Op::SubSat | Op::MulSat);
        let bo = self.read_gi(b)?;
        let r = if sat64 { self.new_g(a)? } else { self.take_g(a)? };
        let alu = |op: Op| match op {
            Op::Add | Op::AddWrap | Op::AddSat => AluOp::Add,
            Op::Sub | Op::SubWrap | Op::SubSat => AluOp::Sub,
            Op::And => AluOp::And,
            Op::Or => AluOp::Or,
            _ => AluOp::Xor,
        };
        match op {
            Op::AddWrap | Op::SubWrap => {
                self.alu_gi(alu(op), Size::Q, r, bo)?;
                self.norm(r, t)?;
            }
            Op::MulWrap => {
                self.imul_gi(Size::Q, r, bo)?;
                self.norm(r, t)?;
            }
            Op::And | Op::Or | Op::Xor => self.alu_gi(alu(op), Size::Q, r, bo)?,
            Op::Add | Op::Sub => {
                let c = if t.signed { Cond::O } else { Cond::B };
                match t.bits {
                    64 => {
                        self.alu_gi(alu(op), Size::Q, r, bo)?;
                        self.jtrap(c, PanicKind::Overflow)?;
                    }
                    32 => {
                        self.alu_gi(alu(op), Size::D, r, bo)?;
                        self.jtrap(c, PanicKind::Overflow)?;
                        self.norm(r, t)?;
                    }
                    _ => {
                        self.alu_gi(alu(op), Size::Q, r, bo)?;
                        self.fits_check(r, t)?;
                    }
                }
            }
            Op::Mul => {
                if t.signed && t.bits >= 32 {
                    let s = if t.bits == 64 { Size::Q } else { Size::D };
                    self.imul_gi(s, r, bo)?;
                    self.jtrap(Cond::O, PanicKind::Overflow)?;
                    self.norm(r, t)?;
                } else {
                    // |a|, |b| < 2^32: the 64-bit product is exact (as unsigned for u32).
                    self.imul_gi(Size::Q, r, bo)?;
                    self.fits_check(r, t)?;
                }
            }
            Op::AddSat | Op::SubSat | Op::MulSat if t.bits < 64 => {
                if op == Op::MulSat {
                    self.imul_gi(Size::Q, r, bo)?;
                } else {
                    self.alu_gi(alu(op), Size::Q, r, bo)?;
                }
                if t.signed {
                    self.clamp(r, t, true, true, true)?;
                } else if op == Op::SubSat {
                    self.clamp(r, t, true, false, true)?;
                } else {
                    self.clamp(r, t, false, true, false)?;
                }
            }
            Op::AddSat | Op::SubSat if t.signed => {
                let done = self.a.new_label();
                self.a.mov_rr(Size::Q, S0, r);
                self.alu_gi(alu(op), Size::Q, r, bo)?;
                self.a.jcc(Cond::No, done)?;
                self.a.shift_ri(ShiftOp::Sar, Size::Q, S0, 63)?;
                self.a.mov_ri(Size::Q, r, i64::MAX)?;
                self.a.alu_rr(AluOp::Xor, Size::Q, r, S0);
                self.a.bind(done)?;
            }
            Op::MulSat => {
                let done = self.a.new_label();
                self.a.mov_rr(Size::Q, S0, r);
                self.alu_gi(AluOp::Xor, Size::Q, S0, bo)?;
                self.imul_gi(Size::Q, r, bo)?;
                self.a.jcc(Cond::No, done)?;
                self.a.shift_ri(ShiftOp::Sar, Size::Q, S0, 63)?;
                self.a.mov_ri(Size::Q, r, i64::MAX)?;
                self.a.alu_rr(AluOp::Xor, Size::Q, r, S0);
                self.a.bind(done)?;
            }
            Op::AddSat | Op::SubSat => {
                self.alu_gi(alu(op), Size::Q, r, bo)?;
                self.a.mov_ri(Size::Q, S0, if op == Op::AddSat { -1 } else { 0 })?;
                self.a.cmov(Cond::B, Size::Q, r, S0)?;
            }
            _ => return Err(bad(format!("{op:?} is not an int binary op"))),
        }
        self.bind_g(v, r)
    }

    /// `u64` `mul` / `mul.sat` through `rdx:rax`.
    fn mul_u64(&mut self, op: Op, a: Operand, b: Operand, v: Value) -> Res<()> {
        self.reserve_g(RAX)?;
        self.reserve_g(RDX)?;
        let br = self.read_g(b)?;
        self.copy_to_g(a, RAX)?;
        self.a.unary(UnaryOp::Mul, Size::Q, br)?;
        if op == Op::Mul {
            self.jtrap(Cond::O, PanicKind::Overflow)?;
        } else {
            self.a.mov_ri(Size::Q, S0, -1)?;
            self.a.cmov(Cond::O, Size::Q, RAX, S0)?;
        }
        self.bind_g(v, RAX)
    }

    fn int_divrem(&mut self, op: Op, t: IntTy, a: Operand, b: Operand, v: Value) -> Res<()> {
        let bc = self.int_const(b);
        if bc == Some(0) {
            self.trap_now(PanicKind::DivByZero)?;
            let r = self.alloc_g()?;
            return self.bind_g(v, r);
        }
        self.reserve_g(RAX)?;
        self.reserve_g(RDX)?;
        let br = self.read_g(b)?;
        let ar = self.read_g(a)?;
        if bc.is_none() {
            self.a.test_rr(Size::Q, br, br);
            self.jtrap(Cond::E, PanicKind::DivByZero)?;
        }
        if t.signed && (bc.is_none() || bc == Some(-1)) {
            let ok = self.a.new_label();
            if bc.is_none() {
                self.a.alu_ri(AluOp::Cmp, Size::Q, br, -1)?;
                self.a.jcc(Cond::Ne, ok)?;
            }
            let min = int_min(t);
            if min == MIN64 {
                self.a.mov_ri(Size::Q, S0, min)?;
                self.a.alu_rr(AluOp::Cmp, Size::Q, ar, S0);
            } else {
                self.a.alu_ri(AluOp::Cmp, Size::Q, ar, min)?;
            }
            self.jtrap(Cond::E, PanicKind::Overflow)?;
            self.a.bind(ok)?;
        }
        self.a.mov_rr(Size::Q, RAX, ar);
        if t.signed {
            self.a.cqo();
            self.a.unary(UnaryOp::Idiv, Size::Q, br)?;
        } else {
            self.a.alu_rr(AluOp::Xor, Size::D, RDX, RDX);
            self.a.unary(UnaryOp::Div, Size::Q, br)?;
        }
        self.bind_g(v, if op == Op::Div { RAX } else { RDX })
    }

    fn int_neg(&mut self, t: IntTy, a: Operand, v: Value) -> Res<()> {
        let r = self.take_g(a)?;
        let s = match t.bits {
            8 => Size::B,
            16 => Size::W,
            32 => Size::D,
            _ => Size::Q,
        };
        self.a.unary(UnaryOp::Neg, s, r)?;
        if t.signed {
            self.jtrap(Cond::O, PanicKind::Overflow)?;
        }
        self.norm(r, t)?;
        self.bind_g(v, r)
    }

    fn shift(&mut self, op: Op, t: IntTy, a: Operand, n: Operand, v: Value) -> Res<()> {
        let bits = i128::from(t.bits);
        let sop = match op {
            Op::Shr if t.signed => ShiftOp::Sar,
            Op::Shr => ShiftOp::Shr,
            _ => ShiftOp::Shl,
        };
        if let Some(k) = self.int_const(n) {
            let k = if op == Op::ShlWrap {
                k & (bits - 1)
            } else if !(0..bits).contains(&k) {
                self.trap_now(PanicKind::Overflow)?;
                let r = self.new_g(a)?;
                return self.bind_g(v, r);
            } else {
                k
            };
            let r = self.take_g(a)?;
            if k != 0 {
                self.a.shift_ri(sop, Size::Q, r, k as u8)?;
            }
            if sop == ShiftOp::Shl {
                self.norm(r, t)?;
            }
            return self.bind_g(v, r);
        }
        self.reserve_g(RCX)?;
        self.copy_to_g(n, RCX)?;
        if op == Op::ShlWrap {
            self.a.alu_ri(AluOp::And, Size::D, RCX, (bits - 1) as i64)?;
        } else {
            self.a.alu_ri(AluOp::Cmp, Size::Q, RCX, bits as i64)?;
            self.jtrap(Cond::Ae, PanicKind::Overflow)?;
        }
        let r = self.take_g(a)?;
        self.a.shift_cl(sop, Size::Q, r)?;
        if sop == ShiftOp::Shl {
            self.norm(r, t)?;
        }
        self.bind_g(v, r)
    }

    fn count(&mut self, op: Op, t: IntTy, a: Operand, v: Value) -> Res<()> {
        let r = self.take_g(a)?;
        if t.signed && t.bits < 64 {
            self.norm_into(r, r, IntTy::new(t.bits, false))?;
        }
        match op {
            // every level has `popcnt` (x86-64-v2)
            Op::Popcount => self.a.popcnt(Size::Q, r, r)?,
            Op::Clz => {
                self.a.bsr(Size::Q, S0, r)?;
                self.a.mov_ri(Size::Q, S1, -1)?;
                self.a.cmov(Cond::E, Size::Q, S0, S1)?;
                self.a.mov_ri(Size::Q, r, i64::from(t.bits) - 1)?;
                self.a.alu_rr(AluOp::Sub, Size::Q, r, S0);
            }
            _ => {
                self.a.bsf(Size::Q, S0, r)?;
                self.a.mov_ri(Size::Q, S1, i64::from(t.bits))?;
                self.a.cmov(Cond::E, Size::Q, S0, S1)?;
                self.a.mov_rr(Size::Q, r, S0);
            }
        }
        self.bind_g(v, r)
    }

    fn mul_wide(&mut self, a: Operand, b: Operand, rt: TypeId, v: Value) -> Res<()> {
        let (o0, o1, t) = match &self.layout(rt)?.shape {
            Shape::Fields(fs) if fs.len() == 2 => (fs[0].0 as i64, fs[1].0 as i64, fs[0].1),
            _ => return Err(bad("mul.wide result must be a pair")),
        };
        let bits = match self.sk(t) {
            Some(Sk::Int(it)) if !it.signed && (it.bits == 32 || it.bits == 64) => it.bits,
            _ => return Err(bad("mul.wide needs u32 or u64")),
        };
        if bits == 32 {
            let bo = self.read_gi(b)?;
            let r = self.new_g(a)?;
            self.imul_gi(Size::Q, r, bo)?;
            let d = self.def_mem(v)?;
            self.a.mov_rr(Size::Q, S0, r);
            self.a.shift_ri(ShiftOp::Shr, Size::Q, S0, 32)?;
            self.a.mov_mr(Size::D, disp(d, o0)?, S0)?;
            self.a.mov_mr(Size::D, disp(d, o1)?, r)?;
            return Ok(());
        }
        self.reserve_g(RAX)?;
        self.reserve_g(RDX)?;
        let br = self.read_g(b)?;
        self.copy_to_g(a, RAX)?;
        self.a.unary(UnaryOp::Mul, Size::Q, br)?;
        let d = self.def_mem(v)?;
        self.a.mov_mr(Size::Q, disp(d, o0)?, RDX)?;
        self.a.mov_mr(Size::Q, disp(d, o1)?, RAX)?;
        Ok(())
    }

    // ---------------------------------------------------------------- floats

    fn float_op(&mut self, op: Op, p: Prec, args: &[Operand], v: Value) -> Res<()> {
        let arg = |k: usize| args.get(k).copied().ok_or_else(|| bad("missing operand"));
        if op == Op::Neg {
            let r = self.take_x(arg(0)?, p)?;
            match p {
                Prec::F32 => self.a.mov_ri(Size::D, S0, 0x8000_0000)?,
                Prec::F64 => self.a.mov_ri(Size::Q, S0, MIN64)?,
            }
            self.a.mov_xg(Size::Q, XS0, S0)?;
            self.a.logic_p(PLogic::Xor, p, r, XS0)?;
            return self.bind(v, R::X(r));
        }
        if op == Op::Rem {
            return self.fmod(p, arg(0)?, arg(1)?, v);
        }
        let bx = self.read_x(arg(1)?, p)?;
        let r = self.take_x(arg(0)?, p)?;
        let sop = match op {
            Op::Add => SseOp::Add,
            Op::Sub => SseOp::Sub,
            Op::Mul => SseOp::Mul,
            _ => SseOp::Div,
        };
        self.a.sse(sop, p, r, bx)?;
        self.bind(v, R::X(r))
    }

    /// Truncated remainder (C `fmod`): exact via x87 `fprem`; a NaN operand propagates as in
    /// `x + y` (glibc's `(x * y) / (x * y)` gives the same NaN).
    fn fmod(&mut self, p: Prec, a: Operand, b: Operand, v: Value) -> Res<()> {
        self.reserve_g(RAX)?;
        let ax = self.read_x(a, p)?;
        let bx = self.read_x(b, p)?;
        let r = self.alloc_x()?;
        let s = self.fscratch()?;
        let (nan, done, again) = (self.a.new_label(), self.a.new_label(), self.a.new_label());
        self.a.ucomis(p, ax, bx)?;
        self.a.jcc(Cond::P, nan)?;
        self.a.movs_store(p, s, bx)?;
        self.a.movs_store(p, disp(s, 8)?, ax)?;
        self.a.fld(p, s)?;
        self.a.fld(p, disp(s, 8)?)?;
        self.a.bind(again)?;
        self.a.fprem();
        self.a.fnstsw_ax();
        self.a.test_ri(Size::D, RAX, 0x400)?;
        self.a.jcc(Cond::Ne, again)?;
        self.a.fstp(p, s)?;
        self.a.fstp_st0();
        self.a.movs(p, r, s)?;
        self.a.jmp(done)?;
        self.a.bind(nan)?;
        self.a.movaps(r, ax);
        self.a.sse(SseOp::Add, p, r, bx)?;
        self.a.bind(done)?;
        self.bind(v, R::X(r))
    }

    /// An f16 / bf16 / f32 / f64 operand as an f32 or f64 register value (f16 / bf16 widen to
    /// f32 exactly). Returns the register and its precision.
    pub(crate) fn float_in(&mut self, o: Operand, f: FloatBits) -> Res<(Xmm, Prec)> {
        match f {
            FloatBits::F32 => Ok((self.read_x(o, Prec::F32)?, Prec::F32)),
            FloatBits::F64 => Ok((self.read_x(o, Prec::F64)?, Prec::F64)),
            FloatBits::F16 | FloatBits::B16 => {
                let g = self.read_g(o)?;
                let x = self.alloc_x()?;
                self.half_to_f32(g, f == FloatBits::B16, x)?;
                Ok((x, Prec::F32))
            }
        }
    }

    /// Converts the f16 (or bf16) bits in `g` to f32 in `x` (exact; NaNs become the canonical
    /// quiet NaN with the sign kept, as the interpreter's conversion does).
    pub(crate) fn half_to_f32(&mut self, g: Gpr, bf16: bool, x: Xmm) -> Res<()> {
        let (special, sub, sign, inf) =
            (self.a.new_label(), self.a.new_label(), self.a.new_label(), self.a.new_label());
        self.a.movzx(Size::D, S0, Size::W, g)?;
        let (emask, mshift, bias_adj, mmask) =
            if bf16 { (0x7f80i64, 16u8, 0i64, 0x7fi64) } else { (0x7c00, 13, 0x3800_0000, 0x3ff) };
        self.a.mov_rr(Size::D, S1, S0);
        self.a.alu_ri(AluOp::And, Size::D, S1, emask)?;
        self.a.alu_ri(AluOp::Cmp, Size::D, S1, emask)?;
        self.a.jcc(Cond::E, special)?;
        if !bf16 {
            self.a.test_rr(Size::D, S1, S1);
            self.a.jcc(Cond::E, sub)?;
        }
        // normal (bf16: also zero and subnormals, which shift exactly)
        self.a.mov_rr(Size::D, S1, S0);
        self.a.alu_ri(AluOp::And, Size::D, S1, 0x7fff)?;
        self.a.shift_ri(ShiftOp::Shl, Size::D, S1, mshift)?;
        if bias_adj != 0 {
            self.a.alu_ri(AluOp::Add, Size::D, S1, bias_adj)?;
        }
        self.a.jmp(sign)?;
        // infinity or NaN
        self.a.bind(special)?;
        self.a.test_ri(Size::D, S0, mmask)?;
        self.a.jcc(Cond::E, inf)?;
        self.a.mov_ri(Size::D, S1, 0x7fc0_0000)?;
        self.a.jmp(sign)?;
        self.a.bind(inf)?;
        self.a.mov_ri(Size::D, S1, 0x7f80_0000)?;
        self.a.jmp(sign)?;
        // f16 subnormal: m * 2^-24
        self.a.bind(sub)?;
        if !bf16 {
            self.a.mov_rr(Size::D, S1, S0);
            self.a.alu_ri(AluOp::And, Size::D, S1, 0x3ff)?;
            self.a.cvtsi2s(Prec::F32, x, Size::D, S1)?;
            self.a.mov_ri(Size::D, S1, 0x3380_0000)?;
            self.a.mov_xg(Size::D, XS1, S1)?;
            self.a.sse(SseOp::Mul, Prec::F32, x, XS1)?;
            self.a.mov_gx(Size::D, S1, x)?;
        }
        self.a.bind(sign)?;
        self.a.alu_ri(AluOp::And, Size::D, S0, 0x8000)?;
        self.a.shift_ri(ShiftOp::Shl, Size::D, S0, 16)?;
        self.a.alu_rr(AluOp::Or, Size::D, S1, S0);
        self.a.mov_xg(Size::D, x, S1)?;
        Ok(())
    }

    // ---------------------------------------------------------------- compare, select

    /// Whether `o` is a fn value (a 16-byte `(code, environment)` pair) or a fn item.
    pub(crate) fn is_fn_operand(&self, o: Operand) -> bool {
        match o {
            Operand::Func(_) => true,
            Operand::Value(_) => {
                self.oty(o).is_some_and(|t| matches!(self.key(t), Some(TypeKey::FnType { .. })))
            }
            _ => false,
        }
    }

    /// Word `k` of a fn value into `dst`.
    fn fn_word(&mut self, o: Operand, k: i64, dst: Gpr) -> Res<()> {
        match self.msrc(o)? {
            MSrc::Mem(m) => Ok(self.a.mov_rm(Size::Q, dst, disp(m, 8 * k)?)?),
            MSrc::Func(f) if k == 0 => {
                let sym = self.func_sym(f)?;
                Ok(self.a.lea(Size::Q, dst, Mem::rip_sym(sym, 0))?)
            }
            MSrc::Func(_) => {
                self.a.alu_rr(AluOp::Xor, Size::D, dst, dst);
                Ok(())
            }
            MSrc::Str(_) => Err(bad("string literal compared as a fn value")),
        }
    }

    /// Compares two scalars (ints, pointers, `bool`, `char`, `f32`, `f64`) and leaves the result
    /// in the flags: the returned test holds iff `a op b`.
    pub(crate) fn compare_flags(&mut self, op: Op, a: Operand, b: Operand) -> Res<Jc> {
        match self.osk(a).or(self.osk(b)) {
            Some(Sk::Float(f)) => {
                let p = if f == FloatBits::F32 { Prec::F32 } else { Prec::F64 };
                let ax = self.read_x(a, p)?;
                let bx = self.read_x(b, p)?;
                Ok(match op {
                    Op::Eq | Op::Ne => {
                        self.a.ucomis(p, ax, bx)?;
                        if op == Op::Eq { Jc::FloatEq } else { Jc::FloatNe }
                    }
                    // unordered sets CF: `a` / `ae` are false, as the compare is
                    Op::Lt | Op::Le => {
                        self.a.ucomis(p, bx, ax)?;
                        Jc::Flag(if op == Op::Lt { Cond::A } else { Cond::Ae })
                    }
                    _ => {
                        self.a.ucomis(p, ax, bx)?;
                        Jc::Flag(if op == Op::Gt { Cond::A } else { Cond::Ae })
                    }
                })
            }
            Some(sk) => {
                let signed = matches!(sk, Sk::Int(t) if t.signed);
                let (a, b, op) = if matches!(a, Operand::Const(_)) && !matches!(b, Operand::Const(_)) {
                    (b, a, mirror(op))
                } else {
                    (a, b, op)
                };
                let ar = self.read_g(a)?;
                let bo = self.read_gi(b)?;
                self.alu_gi(AluOp::Cmp, Size::Q, ar, bo)?;
                Ok(Jc::Flag(match (op, signed) {
                    (Op::Eq, _) => Cond::E,
                    (Op::Ne, _) => Cond::Ne,
                    (Op::Lt, true) => Cond::L,
                    (Op::Le, true) => Cond::Le,
                    (Op::Gt, true) => Cond::G,
                    (Op::Ge, true) => Cond::Ge,
                    (Op::Lt, false) => Cond::B,
                    (Op::Le, false) => Cond::Be,
                    (Op::Gt, false) => Cond::A,
                    _ => Cond::Ae,
                }))
            }
            None => Err(bad("compare of non-scalar operands")),
        }
    }

    fn compare(&mut self, op: Op, a: Operand, b: Operand, v: Value) -> Res<()> {
        if self.is_fn_operand(a) || self.is_fn_operand(b) {
            // fn values: equal iff code and environment are equal
            let r = self.alloc_g()?;
            let r2 = self.alloc_g()?;
            self.fn_word(a, 0, r)?;
            self.fn_word(b, 0, S0)?;
            self.a.alu_rr(AluOp::Xor, Size::Q, r, S0);
            self.fn_word(a, 1, S0)?;
            self.fn_word(b, 1, r2)?;
            self.a.alu_rr(AluOp::Xor, Size::Q, S0, r2);
            self.a.alu_rr(AluOp::Or, Size::Q, r, S0);
            self.a.setcc(if op == Op::Eq { Cond::E } else { Cond::Ne }, r);
            self.a.movzx(Size::D, r, Size::B, r)?;
            return self.bind_g(v, r);
        }
        let k = self.osk(a).or(self.osk(b));
        match k {
            None => {
                let (Some(x), Some(y)) = (self.int_const(a), self.int_const(b)) else {
                    return Err(bad("compare of non-scalar operands"));
                };
                let r = match op {
                    Op::Eq => x == y,
                    Op::Ne => x != y,
                    Op::Lt => x < y,
                    Op::Le => x <= y,
                    Op::Gt => x > y,
                    _ => x >= y,
                };
                let g = self.alloc_g()?;
                self.a.mov_ri(Size::D, g, i64::from(r))?;
                self.bind_g(v, g)
            }
            Some(Sk::Float(f)) => {
                let (ax, p) = self.float_in(a, f)?;
                let (bx, _) = self.float_in(b, f)?;
                let r = self.alloc_g()?;
                match op {
                    Op::Eq | Op::Ne => {
                        self.a.ucomis(p, ax, bx)?;
                        let (c1, c2, lop) = if op == Op::Eq {
                            (Cond::E, Cond::Np, AluOp::And)
                        } else {
                            (Cond::Ne, Cond::P, AluOp::Or)
                        };
                        self.a.setcc(c1, r);
                        self.a.setcc(c2, S0);
                        self.a.alu_rr(lop, Size::B, r, S0);
                    }
                    Op::Lt | Op::Le => {
                        self.a.ucomis(p, bx, ax)?;
                        self.a.setcc(if op == Op::Lt { Cond::A } else { Cond::Ae }, r);
                    }
                    _ => {
                        self.a.ucomis(p, ax, bx)?;
                        self.a.setcc(if op == Op::Gt { Cond::A } else { Cond::Ae }, r);
                    }
                }
                self.a.movzx(Size::D, r, Size::B, r)?;
                self.bind_g(v, r)
            }
            Some(sk) => {
                let signed = matches!(sk, Sk::Int(t) if t.signed);
                let (a, b, op) = if matches!(a, Operand::Const(_)) && !matches!(b, Operand::Const(_)) {
                    (b, a, mirror(op))
                } else {
                    (a, b, op)
                };
                let ar = self.read_g(a)?;
                let bo = self.read_gi(b)?;
                let r = self.alloc_g()?;
                self.alu_gi(AluOp::Cmp, Size::Q, ar, bo)?;
                let c = match (op, signed) {
                    (Op::Eq, _) => Cond::E,
                    (Op::Ne, _) => Cond::Ne,
                    (Op::Lt, true) => Cond::L,
                    (Op::Le, true) => Cond::Le,
                    (Op::Gt, true) => Cond::G,
                    (Op::Ge, true) => Cond::Ge,
                    (Op::Lt, false) => Cond::B,
                    (Op::Le, false) => Cond::Be,
                    (Op::Gt, false) => Cond::A,
                    _ => Cond::Ae,
                };
                self.a.setcc(c, r);
                self.a.movzx(Size::D, r, Size::B, r)?;
                self.bind_g(v, r)
            }
        }
    }

    /// `c ? a : b` without a branch on `c`: one `cmov` for an integer-class value; a float goes
    /// through GPRs (`movd`/`movq`, `cmov`, back), which needs nothing beyond the baseline; an
    /// aggregate or fat pointer is selected word by word (both words loaded, `cmov`, stored), so
    /// the memory accesses do not depend on `c` either.
    fn select(&mut self, c: Operand, a: Operand, b: Operand, rt: TypeId, v: Value) -> Res<()> {
        let cls = self.vcls(rt)?;
        if let Some(k) = self.int_const(c) {
            let o = if k != 0 { a } else { b };
            return match cls {
                VCls::G { .. } => {
                    let r = self.new_g(o)?;
                    self.bind_g(v, r)
                }
                VCls::X(p) => {
                    let r = self.new_x(o, p)?;
                    self.bind(v, R::X(r))
                }
                VCls::M { size, .. } | VCls::V { size } => {
                    let s = self.msrc(o)?;
                    let d = self.def_mem(v)?;
                    self.copy_mem(d, s, size)
                }
            };
        }
        match cls {
            VCls::G { .. } => {
                let cr = self.read_g(c)?;
                let ar = self.read_g(a)?;
                let r = self.new_g(b)?;
                self.a.test_rr(Size::D, cr, cr);
                self.a.cmov(Cond::Ne, Size::Q, r, ar)?;
                self.bind_g(v, r)
            }
            VCls::X(p) => {
                let cr = self.read_g(c)?;
                let ax = self.read_x(a, p)?;
                let bx = self.read_x(b, p)?;
                let r = self.alloc_x()?;
                let s = if p == Prec::F32 { Size::D } else { Size::Q };
                self.a.mov_gx(s, S0, bx)?;
                self.a.mov_gx(s, S1, ax)?;
                self.a.test_rr(Size::D, cr, cr);
                self.a.cmov(Cond::Ne, s, S0, S1)?;
                self.a.mov_xg(s, r, S0)?;
                self.bind(v, R::X(r))
            }
            VCls::M { size, .. } | VCls::V { size } => {
                let cr = self.read_g(c)?;
                let (sa, sb) = (self.msrc(a)?, self.msrc(b)?);
                let t = self.alloc_g()?;
                let d = self.def_mem(v)?;
                self.select_mem(cr, t, d, sa, sb, size)
            }
        }
    }

    /// Word `k` (8 bytes at offset `8 * k`) of an aggregate-class operand into `dst`, without
    /// changing the flags.
    fn mword(&mut self, s: MSrc, k: i64, dst: Gpr) -> Res<()> {
        match s {
            MSrc::Mem(m) => Ok(self.a.mov_rm(Size::Q, dst, disp(m, 8 * k)?)?),
            MSrc::Str(st) if k == 0 => {
                let sym = self.str_sym(st)?;
                Ok(self.a.lea(Size::Q, dst, Mem::rip_sym(sym, 0))?)
            }
            MSrc::Str(st) => {
                let n = self.str_len(st)?;
                Ok(self.a.mov_ri(Size::Q, dst, n)?)
            }
            MSrc::Func(f) if k == 0 => {
                let sym = self.func_sym(f)?;
                Ok(self.a.lea(Size::Q, dst, Mem::rip_sym(sym, 0))?)
            }
            MSrc::Func(_) => Ok(self.a.mov_ri(Size::Q, dst, 0)?),
        }
    }

    /// `d = cr ? sa : sb` over `size` bytes: per word `S0 = b`, `t = a`, `test`, `cmovne`,
    /// store. Up to [`SELECT_UNROLL`] words are unrolled; a longer value loops over its words
    /// with a counter (the loop branch depends on the size only, never on `cr`). The tail below
    /// 8 bytes uses 4/2/1-byte loads and a 32-bit `cmov`.
    fn select_mem(&mut self, cr: Gpr, t: Gpr, d: Mem, sa: MSrc, sb: MSrc, size: u32) -> Res<()> {
        let n8 = size / 8;
        if n8 <= SELECT_UNROLL {
            for k in 0..i64::from(n8) {
                self.mword(sb, k, S0)?;
                self.mword(sa, k, t)?;
                self.a.test_rr(Size::D, cr, cr);
                self.a.cmov(Cond::Ne, Size::Q, S0, t)?;
                self.a.mov_mr(Size::Q, disp(d, 8 * k)?, S0)?;
            }
        } else {
            let (
                MSrc::Mem(Mem::Sib { base: Some(ab), index: None, disp: ad }),
                MSrc::Mem(Mem::Sib { base: Some(bb), index: None, disp: bd }),
                Mem::Sib { base: Some(db), index: None, disp: dd },
            ) = (sa, sb, d)
            else {
                return Err(bad("internal: word select needs base+disp operands"));
            };
            if [ab, bb, db].iter().any(|r| *r == S0 || *r == S1) {
                return Err(bad("internal: word select base is a scratch register"));
            }
            let at = |base: Gpr, dsp: i32| -> Res<Mem> {
                Ok(Mem::index(base, S1, Scale::S8, dsp.checked_sub(8).ok_or_else(|| unsup("disp"))?))
            };
            self.a.mov_ri(Size::Q, S1, i64::from(n8))?;
            let top = self.a.new_label();
            self.a.bind(top)?;
            self.a.mov_rm(Size::Q, S0, at(bb, bd)?)?;
            self.a.mov_rm(Size::Q, t, at(ab, ad)?)?;
            self.a.test_rr(Size::D, cr, cr);
            self.a.cmov(Cond::Ne, Size::Q, S0, t)?;
            self.a.mov_mr(Size::Q, at(db, dd)?, S0)?;
            self.a.alu_ri(AluOp::Sub, Size::Q, S1, 1)?;
            self.a.jcc(Cond::Ne, top)?;
        }
        let mut rest = size % 8;
        if rest == 0 {
            return Ok(());
        }
        let mut off = i64::from(n8) * 8;
        let (MSrc::Mem(ma), MSrc::Mem(mb)) = (sa, sb) else {
            return Err(bad("internal: a string or fn operand selected as a non-fat value"));
        };
        for (w, s) in [(4u32, Size::D), (2, Size::W), (1, Size::B)] {
            if rest >= w {
                let bytes = w as u8;
                self.sized_load(S0, disp(mb, off)?, bytes, false)?;
                self.sized_load(t, disp(ma, off)?, bytes, false)?;
                self.a.test_rr(Size::D, cr, cr);
                self.a.cmov(Cond::Ne, Size::D, S0, t)?;
                self.a.mov_mr(s, disp(d, off)?, S0)?;
                off += i64::from(w);
                rest -= w;
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- aggregates

    fn make(&mut self, args: &[Operand], rt: TypeId, v: Value) -> Res<()> {
        let l = self.layout(rt)?;
        let d = self.def_mem(v)?;
        match &l.shape {
            Shape::Fields(fs) => {
                if fs.len() != args.len() {
                    return Err(bad("make: operand count differs from the field count"));
                }
                for (&(off, ft), &o) in fs.iter().zip(args) {
                    let c = self.vcls(ft)?;
                    self.store_cls(o, c, disp(d, off as i64)?)?;
                }
            }
            Shape::Array { elem, stride, len } => {
                if *len != args.len() as u64 {
                    return Err(bad("make: operand count differs from the array length"));
                }
                let c = self.vcls(*elem)?;
                for (k, &o) in args.iter().enumerate() {
                    self.store_cls(o, c, disp(d, (k as u64 * stride) as i64)?)?;
                }
            }
            _ => return Err(bad("make of a non-aggregate type")),
        }
        Ok(())
    }

    /// Bounds-checks a dynamic array index and returns the element address in a fresh register.
    fn elem_ptr(&mut self, base: Mem, i: Operand, stride: u64, len: u64) -> Res<Mem> {
        let ir = self.read_g(i)?;
        if let Ok(n) = i32::try_from(len) {
            self.a.alu_ri(AluOp::Cmp, Size::Q, ir, i64::from(n))?;
        } else {
            self.a.mov_ri(Size::Q, S0, len as i64)?;
            self.a.alu_rr(AluOp::Cmp, Size::Q, ir, S0);
        }
        self.jtrap(Cond::Ae, PanicKind::Bounds)?;
        let r = self.alloc_g()?;
        self.a.lea(Size::Q, r, base)?;
        self.index_into(r, ir, stride)?;
        Ok(Mem::base(r, 0))
    }

    /// `r += i * stride`.
    fn index_into(&mut self, r: Gpr, ir: Gpr, stride: u64) -> Res<()> {
        let sc = match stride {
            1 => Some(Scale::S1),
            2 => Some(Scale::S2),
            4 => Some(Scale::S4),
            8 => Some(Scale::S8),
            _ => None,
        };
        match sc {
            Some(s) => self.a.lea(Size::Q, r, Mem::index(r, ir, s, 0))?,
            None if stride == 0 => {}
            None => {
                let st = i64::try_from(stride).ok().filter(|s| i32::try_from(*s).is_ok());
                match st {
                    Some(st) => self.a.imul_rri(Size::Q, S0, ir, st)?,
                    None => {
                        self.a.mov_ri(Size::Q, S0, stride as i64)?;
                        self.a.imul_rr(Size::Q, S0, ir)?;
                    }
                }
                self.a.alu_rr(AluOp::Add, Size::Q, r, S0);
            }
        }
        Ok(())
    }

    fn get(&mut self, agg: Operand, idx: Operand, rt: TypeId, v: Value) -> Res<()> {
        let Operand::Value(e) = agg else { return Err(bad("get of a non-value")) };
        let et = self.vty(e)?;
        let l = self.layout(et)?;
        match &l.shape {
            Shape::Array { elem, stride, len } => {
                if let Some(k) = self.int_const(idx) {
                    if k < 0 || k as u128 >= u128::from(*len) {
                        return Err(bad("get: constant index out of range"));
                    }
                    return self.part(v, e, k as i64 * *stride as i64, *elem);
                }
                let base = self.hmem(e)?;
                let m = self.elem_ptr(base, idx, *stride, *len)?;
                let c = self.vcls(*elem)?;
                self.load_cls(v, c, m)
            }
            _ => {
                let (off, ft) = self.field(et, idx)?;
                if ft != rt {
                    return Err(bad("get: result type differs from the field type"));
                }
                self.part(v, e, off, ft)
            }
        }
    }

    fn set(&mut self, agg: Operand, idx: Operand, val: Operand, rt: TypeId, v: Value) -> Res<()> {
        let src = self.msrc(agg)?;
        let l = self.layout(rt)?;
        let size = u32::try_from(l.size).map_err(|_| unsup("aggregate too large"))?;
        let (dst, ft) = match &l.shape {
            Shape::Array { elem, stride, len } => {
                let (stride, len, elem) = (*stride, *len, *elem);
                if let Some(k) = self.int_const(idx) {
                    if k < 0 || k as u128 >= u128::from(len) {
                        return Err(bad("set: constant index out of range"));
                    }
                    let d = self.def_mem(v)?;
                    self.copy_mem(d, src, size)?;
                    (disp(d, k as i64 * stride as i64)?, elem)
                } else {
                    let d = self.def_mem(v)?;
                    self.copy_mem(d, src, size)?;
                    (self.elem_ptr(d, idx, stride, len)?, elem)
                }
            }
            _ => {
                let (off, ft) = self.field(rt, idx)?;
                let d = self.def_mem(v)?;
                self.copy_mem(d, src, size)?;
                (disp(d, off)?, ft)
            }
        };
        let c = self.vcls(ft)?;
        self.store_cls(val, c, dst)
    }

    fn enum_make(&mut self, args: &[Operand], rt: TypeId, v: Value) -> Res<()> {
        let tag = args.first().and_then(|o| self.int_const(*o)).ok_or_else(|| bad("enum.make variant"))?;
        let l = self.layout(rt)?;
        let Shape::Enum(vs) = &l.shape else { return Err(bad("enum.make of a non-enum type")) };
        let fs = usize::try_from(tag).ok().and_then(|t| vs.get(t)).ok_or_else(|| bad("enum.make variant"))?;
        if fs.len() + 1 != args.len() {
            return Err(bad("enum.make: operand count differs from the variant's fields"));
        }
        let size = u32::try_from(l.size).map_err(|_| unsup("enum too large"))?;
        let d = self.def_mem(v)?;
        self.zero_mem(d, size)?;
        self.store_imm(d, 4, tag as i64)?;
        for (&(off, ft), &o) in fs.iter().zip(&args[1..]) {
            let c = self.vcls(ft)?;
            self.store_cls(o, c, disp(d, off as i64)?)?;
        }
        Ok(())
    }

    fn elem_addr(&mut self, p: Operand, i: Operand, v: Value) -> Res<()> {
        let pt = self.oty(p).ok_or_else(|| bad("addr.elem of a non-value"))?;
        let inner = self.pointee(pt)?;
        let (elem, fat) = match self.key(inner) {
            Some(TypeKey::Array { elem, .. }) => (*elem, false),
            Some(TypeKey::Slice { elem, .. }) => (*elem, true),
            _ => (inner, false),
        };
        let stride = self.layout(elem)?.size;
        if let Some(k) = self.int_const(i) {
            let off = (k as i64).wrapping_mul(stride as i64);
            if !fat {
                return self.addr_plus(v, p, off);
            }
        }
        let r = if fat {
            let r = self.alloc_g()?;
            match self.msrc(p)? {
                MSrc::Mem(m) => self.a.mov_rm(Size::Q, r, m)?,
                MSrc::Str(_) | MSrc::Func(_) => return Err(bad("addr.elem of a literal")),
            }
            r
        } else {
            self.new_g(p)?
        };
        if let Some(k) = self.int_const(i) {
            let off = (k as i64).wrapping_mul(stride as i64);
            if let Ok(o) = i32::try_from(off) {
                self.a.lea(Size::Q, r, Mem::base(r, o))?;
            } else {
                self.a.mov_ri(Size::Q, S0, off)?;
                self.a.alu_rr(AluOp::Add, Size::Q, r, S0);
            }
        } else {
            let ir = self.read_g(i)?;
            self.index_into(r, ir, stride)?;
        }
        self.bind_g(v, r)
    }

    /// `prefetcht0/1/2/nta` at `p + i * size` (the size of `p`'s pointee). A write hint is the
    /// read hint of its level: `prefetchw` needs PRFCHW, which neither v2 nor v3 implies.
    fn prefetch(&mut self, h: stave_ir::Prefetch, p: Operand, i: Operand) -> Res<()> {
        use stave_ir::PrefetchLoc as L;
        let hint = match h.loc {
            L::L1 => Prefetch::T0,
            L::L2 => Prefetch::T1,
            L::L3 => Prefetch::T2,
            L::Nta => Prefetch::Nta,
        };
        let pt = self.oty(p).ok_or_else(|| bad("prefetch of a non-value"))?;
        let stride = self.layout(self.pointee(pt)?)?.size;
        let r = self.new_g(p)?;
        let m = match self.int_const(i) {
            Some(k) => match i32::try_from((k as i64).wrapping_mul(stride as i64)) {
                Ok(d) => Mem::base(r, d),
                Err(_) => {
                    self.a.mov_ri(Size::Q, S0, (k as i64).wrapping_mul(stride as i64))?;
                    self.a.alu_rr(AluOp::Add, Size::Q, r, S0);
                    Mem::base(r, 0)
                }
            },
            None => {
                let ir = self.read_g(i)?;
                self.index_into(r, ir, stride)?;
                Mem::base(r, 0)
            }
        };
        Ok(self.a.prefetch(hint, m)?)
    }

    /// The store of `o` (class `c`) to `dst` with a non-temporal hint: `movnti` for 32- and 64-bit
    /// scalars (a float through a general register), `movntps/pd/dq` for 16- and 32-byte vectors;
    /// other values have no non-temporal store and are stored plainly.
    fn store_stream_cls(&mut self, o: Operand, c: VCls, dst: Mem) -> Res<()> {
        let s = match c {
            VCls::G { bytes: 4, .. } | VCls::X(Prec::F32) => Size::D,
            VCls::G { bytes: 8, .. } | VCls::X(Prec::F64) => Size::Q,
            VCls::V { size: 16 | 32 } => {
                let sh = self.voshape(o)?;
                let x = self.vread(o, sh.size)?;
                return self.vstore_nt(dst, x, sh);
            }
            _ => return self.store_cls(o, c, dst),
        };
        self.copy_to_g(o, S0)?;
        Ok(self.a.movnti(s, dst, S0)?)
    }

    fn bounds(&mut self, i: Operand, n: Operand) -> Res<()> {
        match (self.int_const(i), self.int_const(n)) {
            (Some(x), Some(y)) => {
                if x >= y {
                    self.trap_now(PanicKind::Bounds)?;
                }
                Ok(())
            }
            (Some(x), None) => {
                let nr = self.read_g(n)?;
                let x = x as u64 as i64;
                if let Ok(xi) = i32::try_from(x) {
                    self.a.alu_ri(AluOp::Cmp, Size::Q, nr, i64::from(xi))?;
                } else {
                    self.a.mov_ri(Size::Q, S0, x)?;
                    self.a.alu_rr(AluOp::Cmp, Size::Q, nr, S0);
                }
                self.jtrap(Cond::Be, PanicKind::Bounds)
            }
            _ => {
                let ir = self.read_g(i)?;
                let no = self.read_gi(n)?;
                self.alu_gi(AluOp::Cmp, Size::Q, ir, no)?;
                self.jtrap(Cond::Ae, PanicKind::Bounds)
            }
        }
    }
}
