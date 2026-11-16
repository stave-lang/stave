//! Lane-wise execution of the vector and mask ops (module doc of [`crate::simd`]). A vector or
//! mask value is an `Agg` of its lanes: normalized lane bits, mask lanes 0 / 1.

use super::{Interp, InterpError, R, malformed};
use crate::arith;
use crate::host::Host;
use crate::ir::{Body, CastKind, FuncId, Operand, PanicKind, Site};
use crate::mem::Fault;
use crate::simd::{self, LaneTy, VecOp};
use crate::types::TypeId;

use super::Val;

/// What a vector op produced.
pub(super) enum VOut {
    Val(Val),
    /// No result (stores).
    None,
    Trap(PanicKind),
}

fn unsupported(what: String) -> InterpError {
    InterpError::Unsupported(what)
}

fn lanes_val(v: Vec<u64>) -> Val {
    Val::agg(v.into_iter().map(Val::S).collect())
}

fn mask_val(v: impl IntoIterator<Item = bool>) -> Val {
    Val::agg(v.into_iter().map(Val::bool).collect())
}

impl<H: Host> Interp<'_, H> {
    fn lanes(&self, body: &Body, regs: &[Val], o: Operand) -> R<Vec<u64>> {
        match self.eval(body, regs, o)? {
            Val::Agg(ls) => ls.iter().map(|l| l.bits().ok_or(malformed("vector lane"))).collect(),
            _ => Err(malformed("vector value expected")),
        }
    }

    fn mask(&self, body: &Body, regs: &[Val], o: Operand) -> R<Vec<bool>> {
        Ok(self.lanes(body, regs, o)?.into_iter().map(|b| b != 0).collect())
    }

    fn vshape(&self, t: TypeId) -> R<(TypeId, LaneTy, u32)> {
        simd::vec_shape(&self.m.types, t).ok_or(malformed("vector type"))
    }

    fn operand_vshape(&self, body: &Body, o: Operand) -> R<(TypeId, LaneTy, u32)> {
        match o {
            Operand::Value(v) => self.vshape(body.value_ty(v).ok_or(malformed("value"))?),
            _ => Err(malformed("vector operand")),
        }
    }

    /// (address, length) of a vector memory operand.
    fn mem_of(&self, body: &Body, regs: &[Val], o: Operand) -> R<(u64, u64)> {
        let t = match o {
            Operand::Value(v) => body.value_ty(v).ok_or(malformed("value"))?,
            _ => return Err(malformed("memory operand")),
        };
        if simd::elem_ptr_operand(&self.m.types, t).is_some() {
            // unchecked: its bounds check is a separate instruction
            return match self.eval(body, regs, o)? {
                Val::S(a) => Ok((a, u64::MAX)),
                _ => Err(malformed("memory operand value")),
            };
        }
        let mo = simd::mem_operand(&self.m.types, t).ok_or(malformed("memory operand type"))?;
        match (self.eval(body, regs, o)?, mo.len) {
            (Val::Pair(a, n), None) => Ok((a, n)),
            (Val::S(a), Some(n)) => Ok((a, n)),
            _ => Err(malformed("memory operand value")),
        }
    }

    fn read_lane(&self, lane: LaneTy, addr: u64, func: FuncId, site: Site) -> R<u64> {
        let raw = self.mem.read_uint(addr, (lane.bits() / 8) as u64).map_err(Self::fault(func, site))?;
        Ok(lane.norm(raw))
    }

    fn write_lane(&mut self, lane: LaneTy, addr: u64, v: u64, func: FuncId, site: Site) -> R<()> {
        self.mem.write_uint(addr, (lane.bits() / 8) as u64, v).map_err(Self::fault(func, site))
    }

    /// Loads a vector or mask value of type `t` stored at `addr` (layout: lanes in order).
    pub(super) fn load_bytes(&self, t: TypeId, addr: u64, func: FuncId, site: Site) -> R<Val> {
        if let Some((_, lane, n)) = simd::vec_shape(&self.m.types, t) {
            let w = (lane.bits() / 8) as u64;
            let mut out = Vec::with_capacity(n as usize);
            for k in 0..n as u64 {
                out.push(self.read_lane(lane, addr.wrapping_add(k * w), func, site)?);
            }
            return Ok(lanes_val(out));
        }
        if let Some((bits, n)) = simd::mask_shape(&self.m.types, t) {
            let w = (bits / 8) as u64;
            let ones = u64::MAX >> (64 - bits);
            let mut out = Vec::with_capacity(n as usize);
            for k in 0..n as u64 {
                let at = addr.wrapping_add(k * w);
                let raw = self.mem.read_uint(at, w).map_err(Self::fault(func, site))?;
                // a mask lane is all zeros or all ones; any other pattern is undefined behaviour
                if raw != 0 && raw != ones {
                    return Err(InterpError::Fault {
                        fault: Fault { addr: at, len: w, write: false },
                        func,
                        site,
                    });
                }
                out.push(raw != 0);
            }
            return Ok(mask_val(out));
        }
        Err(unsupported("sizeless vector in memory".into()))
    }

    pub(super) fn store_bytes(&mut self, t: TypeId, addr: u64, v: &Val, func: FuncId, site: Site) -> R<()> {
        let ls: Vec<u64> = match v {
            Val::Agg(ls) => ls.iter().map(|l| l.bits().ok_or(malformed("vector lane"))).collect::<R<_>>()?,
            _ => return Err(malformed("stored vector value")),
        };
        if let Some((_, lane, _)) = simd::vec_shape(&self.m.types, t) {
            let w = (lane.bits() / 8) as u64;
            for (k, &x) in ls.iter().enumerate() {
                self.write_lane(lane, addr.wrapping_add(k as u64 * w), x, func, site)?;
            }
            return Ok(());
        }
        if let Some((w, _)) = simd::mask_shape(&self.m.types, t) {
            let w = (w / 8) as u64;
            for (k, &x) in ls.iter().enumerate() {
                let bits = if x != 0 { u64::MAX } else { 0 };
                self.mem
                    .write_uint(addr.wrapping_add(k as u64 * w), w, bits)
                    .map_err(Self::fault(func, site))?;
            }
            return Ok(());
        }
        Err(unsupported("sizeless vector in memory".into()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn exec_vec(
        &mut self,
        body: &Body,
        fid: FuncId,
        regs: &[Val],
        op: VecOp,
        ty: Option<TypeId>,
        args: &[Operand],
        site: Site,
    ) -> R<VOut> {
        use VecOp::*;
        let arg = |k: usize| args.get(k).copied().ok_or(malformed("operand"));
        let rt = || ty.ok_or(malformed("vector result type"));
        let lane_err = |lane: LaneTy| unsupported(format!("{} on {lane:?} lanes", op.name()));
        let v = match op {
            Splat => {
                let (_, _, n) = self.vshape(rt()?)?;
                let x = self.scalar(body, regs, arg(0)?)?;
                lanes_val(vec![x; n as usize])
            }
            Add | Sub | Mul | Div | Copysign | Min | Max | MinNum | MaxNum | MinX86 | MaxX86 | AddWrap
            | SubWrap | MulWrap | AddSat | SubSat | MulSat | And | Or | Xor | Andnot | DivLanes => {
                let (_, lane, _) = self.vshape(rt()?)?;
                let (a, b) = (self.lanes(body, regs, arg(0)?)?, self.lanes(body, regs, arg(1)?)?);
                let mut out = Vec::with_capacity(a.len());
                for (&x, &y) in a.iter().zip(&b) {
                    out.push(match lane {
                        LaneTy::Float(f) => simd::float_binary(op, f, x, y).ok_or_else(|| lane_err(lane))?,
                        LaneTy::Int(t) => match simd::int_binary(op, t, x, y) {
                            Ok(r) => r,
                            Err(k) => return Ok(VOut::Trap(k)),
                        },
                    });
                }
                lanes_val(out)
            }
            CheckedAdd | CheckedSub | CheckedMul => {
                let t = rt()?;
                let (none, some, p) = self.m.types.option_shape(t).ok_or(malformed("checked result"))?;
                let (_, lane, _) = self.vshape(p)?;
                let LaneTy::Int(it) = lane else { return Err(lane_err(lane)) };
                let (a, b) = (self.lanes(body, regs, arg(0)?)?, self.lanes(body, regs, arg(1)?)?);
                let mut out = Vec::with_capacity(a.len());
                for (&x, &y) in a.iter().zip(&b) {
                    match simd::int_binary(op, it, x, y) {
                        Ok(r) => out.push(r),
                        Err(_) => return Ok(VOut::Val(Val::agg(vec![Val::S(none as u64)]))),
                    }
                }
                Val::agg(vec![Val::S(some as u64), lanes_val(out)])
            }
            DivConst => {
                let (_, lane, _) = self.vshape(rt()?)?;
                let LaneTy::Int(it) = lane else { return Err(lane_err(lane)) };
                let a = self.lanes(body, regs, arg(0)?)?;
                let k = body.int_const(arg(1)?).ok_or(malformed("div_const divisor"))?;
                let mut out = Vec::with_capacity(a.len());
                for &x in &a {
                    match simd::int_binary(op, it, x, it.wrap(k)) {
                        Ok(r) => out.push(r),
                        Err(k) => return Ok(VOut::Trap(k)),
                    }
                }
                lanes_val(out)
            }
            Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc | AbsWrap | Popcount | Clz | Not => {
                let (_, lane, _) = self.vshape(rt()?)?;
                let a = self.lanes(body, regs, arg(0)?)?;
                let mut out = Vec::with_capacity(a.len());
                for &x in &a {
                    out.push(match lane {
                        LaneTy::Float(f) => simd::float_unary(op, f, x).ok_or_else(|| lane_err(lane))?,
                        LaneTy::Int(t) => simd::int_unary(op, t, x),
                    });
                }
                lanes_val(out)
            }
            Fma => {
                let (_, lane, _) = self.vshape(rt()?)?;
                let LaneTy::Float(f) = lane else { return Err(lane_err(lane)) };
                let (a, b, c) = (
                    self.lanes(body, regs, arg(0)?)?,
                    self.lanes(body, regs, arg(1)?)?,
                    self.lanes(body, regs, arg(2)?)?,
                );
                let mut out = Vec::with_capacity(a.len());
                for k in 0..a.len() {
                    out.push(simd::float_fma(f, a[k], b[k], c[k]).ok_or_else(|| lane_err(lane))?);
                }
                lanes_val(out)
            }
            Shl | Shr | ShlWrap => {
                let t = rt()?;
                let (_, lane, n) = self.vshape(t)?;
                let LaneTy::Int(it) = lane else { return Err(lane_err(lane)) };
                let a = self.lanes(body, regs, arg(0)?)?;
                let amt = arg(1)?;
                let amounts: Vec<i128> = match amt {
                    Operand::Value(v) if body.value_ty(v) == Some(t) => {
                        self.lanes(body, regs, amt)?.into_iter().map(|x| it.value(x)).collect()
                    }
                    _ => vec![self.ival(body, regs, amt)?; n as usize],
                };
                let s = op.scalar().ok_or(malformed("shift"))?;
                let mut out = Vec::with_capacity(a.len());
                for (&x, &k) in a.iter().zip(&amounts) {
                    match arith::int_shift(s, it, x, k) {
                        Ok(r) => out.push(r),
                        Err(k) => return Ok(VOut::Trap(k)),
                    }
                }
                lanes_val(out)
            }
            Eq | Ne | Lt | Le | Gt | Ge => {
                let (_, lane, _) = self.operand_vshape(body, arg(0)?)?;
                let (a, b) = (self.lanes(body, regs, arg(0)?)?, self.lanes(body, regs, arg(1)?)?);
                mask_val(a.iter().zip(&b).map(|(&x, &y)| simd::compare(op, lane, x, y)))
            }
            ReduceAdd | ReduceAddWrap | ReduceMin | ReduceMax | ReduceAnd | ReduceOr => {
                let (_, lane, _) = self.operand_vshape(body, arg(0)?)?;
                let mut a = self.lanes(body, regs, arg(0)?)?;
                let r = simd::reduce_tree(&mut a, |x, y| simd::reduce_step(op, lane, x, y))
                    .ok_or_else(|| lane_err(lane))?;
                Val::S(r)
            }
            ReduceAddOrdered => {
                let (_, lane, _) = self.operand_vshape(body, arg(0)?)?;
                let a = self.lanes(body, regs, arg(0)?)?;
                let mut acc = self.scalar(body, regs, arg(1)?)?;
                for &x in &a {
                    acc = simd::lane_add(lane, acc, x).ok_or_else(|| lane_err(lane))?;
                }
                Val::S(acc)
            }
            ScanAdd | ScanAddWrap => {
                let (_, lane, _) = self.vshape(rt()?)?;
                let a = self.lanes(body, regs, arg(0)?)?;
                let mut out = Vec::with_capacity(a.len());
                let mut acc: Option<u64> = None;
                for &x in &a {
                    let s = match acc {
                        None => x,
                        Some(p) => simd::lane_add(lane, p, x).ok_or_else(|| lane_err(lane))?,
                    };
                    out.push(s);
                    acc = Some(s);
                }
                lanes_val(out)
            }
            Cast(kind) => {
                let t = rt()?;
                let (_, from, _) = self.operand_vshape(body, arg(0)?)?;
                let a = self.lanes(body, regs, arg(0)?)?;
                let to_t = if kind == CastKind::Try {
                    self.m.types.option_shape(t).ok_or(malformed("vec.cast.try type"))?.2
                } else {
                    t
                };
                let (_, to, _) = self.vshape(to_t)?;
                let mut out = Vec::with_capacity(a.len());
                for &x in &a {
                    match arith::convert(kind, from.cast_ty(), to.cast_ty(), x) {
                        Some(r) => out.push(r),
                        None if kind == CastKind::Try => {
                            let (none, _, _) = self.m.types.option_shape(t).ok_or(malformed("option"))?;
                            return Ok(VOut::Val(Val::agg(vec![Val::S(none as u64)])));
                        }
                        None => return Err(malformed("vector cast")),
                    }
                }
                if kind == CastKind::Try {
                    let (_, some, _) = self.m.types.option_shape(t).ok_or(malformed("option"))?;
                    Val::agg(vec![Val::S(some as u64), lanes_val(out)])
                } else {
                    lanes_val(out)
                }
            }
            Bitcast => {
                let (_, to, _) = self.vshape(rt()?)?;
                let (_, from, _) = self.operand_vshape(body, arg(0)?)?;
                let a = self.lanes(body, regs, arg(0)?)?;
                let bytes = simd::lanes_to_bytes(&a, from.bits() / 8);
                lanes_val(simd::bytes_to_lanes(&bytes, to))
            }
            Split => {
                let a = self.lanes(body, regs, arg(0)?)?;
                let h = a.len() / 2;
                Val::agg(vec![lanes_val(a[..h].to_vec()), lanes_val(a[h..].to_vec())])
            }
            Join => {
                let mut a = self.lanes(body, regs, arg(0)?)?;
                a.extend(self.lanes(body, regs, arg(1)?)?);
                lanes_val(a)
            }
            WidenLo | WidenHi => {
                let (_, to, _) = self.vshape(rt()?)?;
                let (_, from, _) = self.operand_vshape(body, arg(0)?)?;
                let a = self.lanes(body, regs, arg(0)?)?;
                let h = a.len() / 2;
                let part = if op == WidenLo { &a[..h] } else { &a[h..] };
                let mut out = Vec::with_capacity(h);
                for &x in part {
                    out.push(
                        arith::convert(CastKind::Exact, from.cast_ty(), to.cast_ty(), x)
                            .ok_or(malformed("widen"))?,
                    );
                }
                lanes_val(out)
            }
            NarrowWrap | NarrowSat => {
                let (_, to, _) = self.vshape(rt()?)?;
                let (_, from, _) = self.operand_vshape(body, arg(0)?)?;
                let mut a = self.lanes(body, regs, arg(0)?)?;
                a.extend(self.lanes(body, regs, arg(1)?)?);
                let kind = if op == NarrowWrap { CastKind::Wrap } else { CastKind::Sat };
                let mut out = Vec::with_capacity(a.len());
                for &x in &a {
                    out.push(
                        arith::convert(kind, from.cast_ty(), to.cast_ty(), x).ok_or(malformed("narrow"))?,
                    );
                }
                lanes_val(out)
            }
            Permute => {
                let a = self.lanes(body, regs, arg(0)?)?;
                let mut out = Vec::with_capacity(args.len().saturating_sub(1));
                for &k in &args[1..] {
                    let k = body.int_const(k).ok_or(malformed("permute index"))?;
                    out.push(
                        *usize::try_from(k).ok().and_then(|k| a.get(k)).ok_or(malformed("permute index"))?,
                    );
                }
                lanes_val(out)
            }
            Lookup => {
                let (_, il, _) = self.operand_vshape(body, arg(1)?)?;
                let tbl = self.lanes(body, regs, arg(0)?)?;
                let idx = self.lanes(body, regs, arg(1)?)?;
                let LaneTy::Int(it) = il else { return Err(malformed("lookup index")) };
                lanes_val(
                    idx.iter()
                        .map(|&k| {
                            usize::try_from(it.value(k)).ok().and_then(|k| tbl.get(k)).copied().unwrap_or(0)
                        })
                        .collect(),
                )
            }
            Compress | Expand => {
                let a = self.lanes(body, regs, arg(0)?)?;
                let m = self.mask(body, regs, arg(1)?)?;
                let mut out = vec![0u64; a.len()];
                let mut k = 0;
                for (j, &on) in m.iter().enumerate() {
                    if on {
                        if op == Compress {
                            out[k] = a[j];
                        } else {
                            out[j] = a[k];
                        }
                        k += 1;
                    }
                }
                lanes_val(out)
            }
            Load | LoadAligned | LoadMasked => {
                let (_, lane, n) = self.vshape(rt()?)?;
                let (base, len) = self.mem_of(body, regs, arg(0)?)?;
                let i = self.scalar(body, regs, arg(1)?)? as u128;
                let mask = if op == LoadMasked { Some(self.mask(body, regs, arg(2)?)?) } else { None };
                if op == LoadAligned && !i.is_multiple_of(n as u128) {
                    return Ok(VOut::Trap(PanicKind::Bounds));
                }
                if self.bad_lane(i, n, len, mask.as_deref()).is_some() {
                    if mask.is_some() {
                        return Ok(VOut::Trap(PanicKind::Bounds));
                    }
                    return Err(malformed("vector load past the end without a bounds check"));
                }
                let w = (lane.bits() / 8) as u128;
                let mut out = Vec::with_capacity(n as usize);
                for k in 0..n as usize {
                    if mask.as_ref().is_some_and(|m| !m[k]) {
                        out.push(0);
                        continue;
                    }
                    let a = (base as u128).wrapping_add((i + k as u128) * w) as u64;
                    out.push(self.read_lane(lane, a, fid, site)?);
                }
                lanes_val(out)
            }
            Store | StoreAligned | StoreMasked => {
                let (_, lane, n) = self.operand_vshape(body, arg(2)?)?;
                let (base, len) = self.mem_of(body, regs, arg(0)?)?;
                let i = self.scalar(body, regs, arg(1)?)? as u128;
                let v = self.lanes(body, regs, arg(2)?)?;
                let mask = if op == StoreMasked { Some(self.mask(body, regs, arg(3)?)?) } else { None };
                if op == StoreAligned && !i.is_multiple_of(n as u128) {
                    return Ok(VOut::Trap(PanicKind::Bounds));
                }
                if self.bad_lane(i, n, len, mask.as_deref()).is_some() {
                    if mask.is_some() {
                        return Ok(VOut::Trap(PanicKind::Bounds));
                    }
                    return Err(malformed("vector store past the end without a bounds check"));
                }
                let w = (lane.bits() / 8) as u128;
                for (k, &x) in v.iter().enumerate() {
                    if mask.as_ref().is_some_and(|m| !m[k]) {
                        continue;
                    }
                    let a = (base as u128).wrapping_add((i + k as u128) * w) as u64;
                    self.write_lane(lane, a, x, fid, site)?;
                }
                return Ok(VOut::None);
            }
            Gather | Scatter => {
                let (_, lane, _) =
                    if op == Gather { self.vshape(rt()?)? } else { self.operand_vshape(body, arg(2)?)? };
                let (_, il, _) = self.operand_vshape(body, arg(1)?)?;
                let LaneTy::Int(it) = il else { return Err(malformed("gather index")) };
                if it.signed {
                    return Err(malformed("signed gather index"));
                }
                let (base, len) = self.mem_of(body, regs, arg(0)?)?;
                let idx: Vec<i128> =
                    self.lanes(body, regs, arg(1)?)?.into_iter().map(|x| it.value(x)).collect();
                if idx.iter().any(|&k| k < 0 || k >= len as i128) {
                    return Ok(VOut::Trap(PanicKind::Bounds));
                }
                let w = (lane.bits() / 8) as u64;
                let at = |k: i128| base.wrapping_add(k as u64 * w);
                if op == Gather {
                    let mut out = Vec::with_capacity(idx.len());
                    for &k in &idx {
                        out.push(self.read_lane(lane, at(k), fid, site)?);
                    }
                    lanes_val(out)
                } else {
                    let v = self.lanes(body, regs, arg(2)?)?;
                    // in lane order: on a duplicate index the highest lane is written last
                    for (&k, &x) in idx.iter().zip(&v) {
                        self.write_lane(lane, at(k), x, fid, site)?;
                    }
                    return Ok(VOut::None);
                }
            }
            MaskAnd | MaskOr | MaskXor => {
                let (a, b) = (self.mask(body, regs, arg(0)?)?, self.mask(body, regs, arg(1)?)?);
                mask_val(a.iter().zip(&b).map(|(&x, &y)| match op {
                    MaskAnd => x && y,
                    MaskOr => x || y,
                    _ => x != y,
                }))
            }
            MaskNot => mask_val(self.mask(body, regs, arg(0)?)?.into_iter().map(|x| !x)),
            Select => {
                let m = self.mask(body, regs, arg(0)?)?;
                let (a, b) = (self.lanes(body, regs, arg(1)?)?, self.lanes(body, regs, arg(2)?)?);
                let out =
                    m.iter().zip(a.iter().zip(&b)).map(|(&on, (&x, &y))| if on { x } else { y }).collect();
                lanes_val(out)
            }
            ToBits => {
                let m = self.mask(body, regs, arg(0)?)?;
                Val::S(m.iter().enumerate().fold(0u64, |acc, (k, &on)| acc | ((on as u64) << k)))
            }
            ToVec => {
                let (_, lane, _) = self.vshape(rt()?)?;
                let LaneTy::Int(it) = lane else { return Err(malformed("to_vec lane")) };
                let ones = it.norm(u64::MAX);
                lanes_val(
                    self.mask(body, regs, arg(0)?)?.into_iter().map(|on| if on { ones } else { 0 }).collect(),
                )
            }
            FirstN => {
                let (_, n) = simd::mask_shape(&self.m.types, rt()?).ok_or(malformed("mask type"))?;
                let k = self.scalar(body, regs, arg(0)?)?;
                mask_val((0..n as u64).map(|j| j < k))
            }
            Rewiden => self.eval(body, regs, arg(0)?)?,
            MaskAll | MaskNone => {
                let (_, n) = simd::mask_shape(&self.m.types, rt()?).ok_or(malformed("mask type"))?;
                mask_val((0..n).map(|_| op == MaskAll))
            }
        };
        Ok(VOut::Val(v))
    }

    /// The first lane of an `n`-lane access at `i` that is out of `0..len` (active lanes only
    /// when `mask` is given).
    fn bad_lane(&self, i: u128, n: u32, len: u64, mask: Option<&[bool]>) -> Option<usize> {
        (0..n as usize).find(|&k| {
            mask.is_none_or(|m| m.get(k).copied().unwrap_or(false)) && i + k as u128 >= len as u128
        })
    }
}
