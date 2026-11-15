//! Typing rules of the vector and mask ops (module doc of [`crate::simd`]).

use super::{Cls, FnVerifier, option_shape};
use crate::arith;
use crate::ir::{Body, CastKind, Operand};
use crate::simd::{self, LaneTy, VecOp};
use crate::types::{IntBits, Lanes, TypeId, TypeKey};

type R = Result<(), String>;

impl FnVerifier<'_> {
    fn vshape(&self, t: TypeId) -> Result<(TypeId, LaneTy, u32), String> {
        simd::vec_shape(&self.m.types, t)
            .ok_or_else(|| format!("{} is not a fixed-size vector", self.tname(t)))
    }

    fn mshape(&self, t: TypeId) -> Result<(u32, u32), String> {
        simd::mask_shape(&self.m.types, t)
            .ok_or_else(|| format!("{} is not a fixed-size mask", self.tname(t)))
    }

    fn value_ty(&self, b: &Body, o: Operand) -> Result<TypeId, String> {
        self.operand_ty(b, o).ok_or_else(|| "operand must be a value".to_string())
    }

    fn find_ty(&self, k: TypeKey) -> Result<TypeId, String> {
        self.m.types.find(&k).ok_or_else(|| format!("type {k:?} not in the module"))
    }

    /// The mask type of compares on vector type `t` must be `mask <bits of lane> <n>`.
    fn mask_for(&self, m: TypeId, lane: LaneTy, n: u32) -> R {
        let (w, k) = self.mshape(m)?;
        if w != lane.bits() || k != n {
            return Err(format!("mask {} does not match {n} lanes of {} bits", self.tname(m), lane.bits()));
        }
        Ok(())
    }

    /// A slice / array memory operand of lane `lane`; returns the operand's shape.
    fn mem(&self, b: &Body, s: Operand, lane: TypeId, store: bool) -> Result<simd::MemOperand, String> {
        let st = self.value_ty(b, s)?;
        let mo = simd::mem_operand(&self.m.types, st).ok_or_else(|| {
            format!("vector memory operand {} is not a slice or array reference", self.tname(st))
        })?;
        if mo.elem != lane {
            return Err(format!("vector memory operand {} has another element type", self.tname(st)));
        }
        if store && !mo.mutable {
            return Err("vector store through a shared reference".into());
        }
        Ok(mo)
    }

    fn usize_operand(&self, b: &Body, o: Operand) -> R {
        let us = self.find_ty(TypeKey::Int { signed: false, bits: IntBits::Size })?;
        self.check_operand(b, o, us)
    }

    fn int_lanes(&self, t: TypeId) -> Result<(TypeId, arith::IntTy, u32), String> {
        match self.vshape(t)? {
            (l, LaneTy::Int(it), n) => Ok((l, it, n)),
            _ => Err(format!("int vector op on {}", self.tname(t))),
        }
    }

    fn float_lanes(&self, t: TypeId) -> Result<(), String> {
        match self.vshape(t)? {
            (_, LaneTy::Float(_), _) => Ok(()),
            _ => Err(format!("float vector op on {}", self.tname(t))),
        }
    }

    pub(super) fn check_vec(&self, b: &Body, op: VecOp, ty: Option<TypeId>, args: &[Operand]) -> R {
        use VecOp::*;
        let n = args.len();
        let arity = |k: usize| if n == k { Ok(()) } else { Err(format!("{n} operands, expected {k}")) };
        let res = || ty.ok_or_else(|| "missing result type".to_string());
        if op.is_store() && ty.is_some() {
            return Err("must have no result".into());
        }
        match op {
            Splat => {
                arity(1)?;
                let t = res()?;
                let (lane, _, _) = self.vshape(t)?;
                self.check_args(b, args, &[lane])
            }
            Add | Sub | Mul | Div | Copysign | MinNum | MaxNum | MinX86 | MaxX86 => {
                arity(2)?;
                let t = res()?;
                self.float_lanes(t)?;
                self.check_args(b, args, &[t, t])
            }
            Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc => {
                arity(1)?;
                let t = res()?;
                self.float_lanes(t)?;
                self.check_args(b, args, &[t])
            }
            Fma => {
                arity(3)?;
                let t = res()?;
                self.float_lanes(t)?;
                self.check_args(b, args, &[t, t, t])
            }
            Min | Max => {
                arity(2)?;
                let t = res()?;
                self.vshape(t)?;
                self.check_args(b, args, &[t, t])
            }
            AddWrap | SubWrap | MulWrap | AddSat | SubSat | MulSat | DivLanes | And | Or | Xor | Andnot => {
                arity(2)?;
                let t = res()?;
                self.int_lanes(t)?;
                self.check_args(b, args, &[t, t])
            }
            CheckedAdd | CheckedSub | CheckedMul => {
                arity(2)?;
                let t = res()?;
                let (_, _, p) =
                    option_shape(&self.m.types, t).ok_or("checked op result must be an Option")?;
                self.int_lanes(p)?;
                self.check_args(b, args, &[p, p])
            }
            DivConst => {
                arity(2)?;
                let t = res()?;
                let (_, it, _) = self.int_lanes(t)?;
                self.check_operand(b, args[0], t)?;
                match b.int_const(args[1]) {
                    Some(k) if it.fits(k) => Ok(()),
                    _ => Err("div_const divisor must be a lane-typed constant".into()),
                }
            }
            AbsWrap | Popcount | Clz | Not => {
                arity(1)?;
                let t = res()?;
                self.int_lanes(t)?;
                self.check_args(b, args, &[t])
            }
            ScanAdd => {
                arity(1)?;
                let t = res()?;
                self.vshape(t)?;
                self.check_args(b, args, &[t])
            }
            Shl | Shr | ShlWrap => {
                arity(2)?;
                let t = res()?;
                self.int_lanes(t)?;
                self.check_operand(b, args[0], t)?;
                if self.operand_ty(b, args[1]) == Some(t) {
                    return Ok(());
                }
                match self.operand_cls(b, args[1]) {
                    Some(Cls::Int(_)) => Ok(()),
                    None if b.int_const(args[1]).is_some() => Ok(()),
                    _ => Err("vector shift amount must be the vector type or an int".into()),
                }
            }
            Eq | Ne | Lt | Le | Gt | Ge => {
                arity(2)?;
                let t = res()?;
                let vt = self.value_ty(b, args[0])?;
                let (_, lane, k) = self.vshape(vt)?;
                self.mask_for(t, lane, k)?;
                self.check_args(b, args, &[vt, vt])
            }
            ReduceAdd | ReduceAddWrap | ReduceMin | ReduceMax | ReduceAnd | ReduceOr => {
                arity(1)?;
                let t = res()?;
                let vt = self.value_ty(b, args[0])?;
                let (lane, lc, _) = self.vshape(vt)?;
                if t != lane {
                    return Err(format!("reduction result {} is not the lane type", self.tname(t)));
                }
                if matches!(op, ReduceAnd | ReduceOr) && !matches!(lc, LaneTy::Int(_)) {
                    return Err("bitwise reduction of float lanes".into());
                }
                if op == ReduceAdd && !matches!(lc, LaneTy::Float(_)) {
                    return Err("`vec.reduce_add` of int lanes (use `vec.reduce_add_wrap`)".into());
                }
                if op == ReduceAddWrap && !matches!(lc, LaneTy::Int(_)) {
                    return Err("wrapping reduction of float lanes".into());
                }
                Ok(())
            }
            ReduceAddOrdered => {
                arity(2)?;
                let t = res()?;
                let vt = self.value_ty(b, args[0])?;
                let (lane, _, _) = self.vshape(vt)?;
                if t != lane {
                    return Err(format!("reduction result {} is not the lane type", self.tname(t)));
                }
                self.check_args(b, args, &[vt, lane])
            }
            Cast(kind) => {
                arity(1)?;
                let t = res()?;
                let to_t = if kind == CastKind::Try {
                    option_shape(&self.m.types, t).ok_or("vec.cast.try result must be an Option")?.2
                } else {
                    t
                };
                let (_, to, k) = self.vshape(to_t)?;
                let (_, from, k0) = self.vshape(self.value_ty(b, args[0])?)?;
                if k != k0 {
                    return Err("vector cast changes the lane count".into());
                }
                if !arith::cast_allowed(kind, from.cast_ty(), to.cast_ty()) {
                    return Err(format!("{kind:?} lane cast from {from:?} to {to:?}"));
                }
                Ok(())
            }
            Bitcast => {
                arity(1)?;
                let t = res()?;
                let (_, to, k) = self.vshape(t)?;
                let (_, from, k0) = self.vshape(self.value_ty(b, args[0])?)?;
                if to.bits() * k != from.bits() * k0 {
                    return Err("bitcast between vectors of different widths".into());
                }
                Ok(())
            }
            Split => {
                arity(1)?;
                let t = res()?;
                let vt = self.value_ty(b, args[0])?;
                let (lane, _, k) = self.vshape(vt)?;
                let h = self.find_ty(TypeKey::Vec { lane, n: Lanes::Fixed((k / 2) as u8) })?;
                if k < 4 || self.key(t) != Some(&TypeKey::Tuple(vec![h, h])) {
                    return Err("split result must be the pair of half vectors".into());
                }
                Ok(())
            }
            Join => {
                arity(2)?;
                let t = res()?;
                let (lane, _, k) = self.vshape(t)?;
                let h = self.find_ty(TypeKey::Vec { lane, n: Lanes::Fixed((k / 2) as u8) })?;
                if k < 4 {
                    return Err("join of 1-lane halves".into());
                }
                self.check_args(b, args, &[h, h])
            }
            WidenLo | WidenHi => {
                arity(1)?;
                let t = res()?;
                let (_, wide, k) = self.vshape(t)?;
                let (_, lane, k0) = self.vshape(self.value_ty(b, args[0])?)?;
                if simd::wide_lane(lane) != Some(wide) || k * 2 != k0 {
                    return Err("widen result must be half as many lanes of the double-width lane".into());
                }
                Ok(())
            }
            NarrowWrap | NarrowSat => {
                arity(2)?;
                let t = res()?;
                let (_, lane, k) = self.vshape(t)?;
                let at = self.value_ty(b, args[0])?;
                let (_, wide, k0) = self.vshape(at)?;
                if !matches!(lane, LaneTy::Int(_)) || simd::wide_lane(lane) != Some(wide) || k0 * 2 != k {
                    return Err("narrow operands must be two vectors of the double-width int lane".into());
                }
                self.check_args(b, args, &[at, at])
            }
            Permute => {
                let t = res()?;
                let (lane, _, k) = self.vshape(t)?;
                if n != k as usize + 1 {
                    return Err(format!("permute needs {k} lane indices"));
                }
                let vt = self.value_ty(b, args[0])?;
                let (l0, _, k0) = self.vshape(vt)?;
                if l0 != lane {
                    return Err("permute changes the lane type".into());
                }
                for &a in &args[1..] {
                    match b.int_const(a) {
                        Some(x) if (0..k0 as i128).contains(&x) => {}
                        _ => return Err("permute index must be a constant below the lane count".into()),
                    }
                }
                Ok(())
            }
            Lookup => {
                arity(2)?;
                let t = res()?;
                let (_, _, k) = self.vshape(t)?;
                self.check_operand(b, args[0], t)?;
                let (_, _, k1) = self.int_lanes(self.value_ty(b, args[1])?)?;
                if k1 != k {
                    return Err("lookup index lane count differs".into());
                }
                Ok(())
            }
            Compress | Expand => {
                arity(2)?;
                let t = res()?;
                let (_, lane, k) = self.vshape(t)?;
                self.check_operand(b, args[0], t)?;
                self.mask_for(self.value_ty(b, args[1])?, lane, k)
            }
            Load | LoadAligned | LoadMasked => {
                arity(if op == LoadMasked { 3 } else { 2 })?;
                let t = res()?;
                let (lane, lc, k) = self.vshape(t)?;
                let mo = self.mem(b, args[0], lane, false)?;
                if op == LoadAligned {
                    self.aligned(mo, lc, k)?;
                }
                self.usize_operand(b, args[1])?;
                if op == LoadMasked { self.mask_for(self.value_ty(b, args[2])?, lc, k) } else { Ok(()) }
            }
            Store | StoreAligned | StoreMasked => {
                arity(if op == StoreMasked { 4 } else { 3 })?;
                let vt = self.value_ty(b, args[2])?;
                let (lane, lc, k) = self.vshape(vt)?;
                let mo = self.mem(b, args[0], lane, true)?;
                if op == StoreAligned {
                    self.aligned(mo, lc, k)?;
                }
                self.usize_operand(b, args[1])?;
                if op == StoreMasked { self.mask_for(self.value_ty(b, args[3])?, lc, k) } else { Ok(()) }
            }
            Gather => {
                arity(2)?;
                let t = res()?;
                let (lane, _, k) = self.vshape(t)?;
                self.mem(b, args[0], lane, false)?;
                let (_, _, k1) = self.int_lanes(self.value_ty(b, args[1])?)?;
                if k1 != k { Err("gather index lane count differs".into()) } else { Ok(()) }
            }
            Scatter => {
                arity(3)?;
                let vt = self.value_ty(b, args[2])?;
                let (lane, _, k) = self.vshape(vt)?;
                self.mem(b, args[0], lane, true)?;
                let (_, _, k1) = self.int_lanes(self.value_ty(b, args[1])?)?;
                if k1 != k { Err("scatter index lane count differs".into()) } else { Ok(()) }
            }
            MaskAnd | MaskOr | MaskXor => {
                arity(2)?;
                let t = res()?;
                self.mshape(t)?;
                self.check_args(b, args, &[t, t])
            }
            MaskNot => {
                arity(1)?;
                let t = res()?;
                self.mshape(t)?;
                self.check_args(b, args, &[t])
            }
            Select => {
                arity(3)?;
                let t = res()?;
                let mt = self.value_ty(b, args[0])?;
                let (w, k) = self.mshape(mt)?;
                let (w1, k1) = match self.vshape(t) {
                    Ok((_, lane, k1)) => (lane.bits(), k1),
                    Err(_) => self.mshape(t)?,
                };
                if w != w1 || k != k1 {
                    return Err(format!("select mask {} does not match {}", self.tname(mt), self.tname(t)));
                }
                self.check_args(b, &args[1..], &[t, t])
            }
            ToBits => {
                arity(1)?;
                let t = res()?;
                let (_, k) = self.mshape(self.value_ty(b, args[0])?)?;
                match self.cls(t) {
                    Cls::Int(it)
                        if !it.signed
                            && it.bits as u32 == simd::to_bits_width(k)
                            && self.key(t) != Some(&TypeKey::Int { signed: false, bits: IntBits::Size }) =>
                    {
                        Ok(())
                    }
                    _ => Err(format!(
                        "to_bits result {} is not the smallest uint holding {k} bits",
                        self.tname(t)
                    )),
                }
            }
            ToVec => {
                arity(1)?;
                let t = res()?;
                let (_, _, k) = self.int_lanes(t)?;
                let (_, lane, _) = self.vshape(t)?;
                self.mask_for(self.value_ty(b, args[0])?, lane, k)
            }
            FirstN => {
                arity(1)?;
                self.mshape(res()?)?;
                self.usize_operand(b, args[0])
            }
            Rewiden => {
                arity(1)?;
                let (_, k) = self.mshape(res()?)?;
                let (_, k0) = self.mshape(self.value_ty(b, args[0])?)?;
                if k != k0 { Err("rewiden changes the lane count".into()) } else { Ok(()) }
            }
            MaskAll | MaskNone => {
                arity(0)?;
                self.mshape(res()?).map(|_| ())
            }
        }
    }

    /// An aligned access needs a slice declared aligned to at least the vector's size.
    fn aligned(&self, mo: simd::MemOperand, lane: LaneTy, n: u32) -> R {
        let bytes = lane.bits() / 8 * n;
        match mo.align {
            Some(a) if a >= bytes => Ok(()),
            _ => Err(format!("aligned vector access needs a slice aligned to {bytes} bytes")),
        }
    }
}
