//! Lowering of vector and mask expressions to the `vec.*` / `mask.*` ops of stave-ir
//! (contracts/simd-ops.md): operators, compares (mask results), casts, the built-in methods and
//! type-relative fns of vector and mask types, and `s.align_to::<A>()`.
//!
//! `TxN.load(s, i)` and `v.store(s, i)` (and the aligned forms) get the bounds check of
//! stave-ir's `bounds` module: `check.bounds i (sub.sat len N-1)` right before the access.
//! Masked forms, gathers, and scatters check their active lanes themselves.

use stave_ir::{CastKind, Op, Operand, VecOp};

use crate::lower_fn::{E, FnCx, Place, Ty};
use crate::prim::Lanes;
use crate::types::{GArg, Interner, TypeKind};

impl FnCx<'_> {
    pub fn is_simd(&self, t: Ty) -> bool {
        matches!(self.kind(t), TypeKind::Vector { .. } | TypeKind::Mask { .. })
    }

    /// A vector or mask type of a fixed lane count (sizeless ones are not lowered yet).
    fn fixed_simd(&self, t: Ty) -> bool {
        matches!(
            self.kind(t),
            TypeKind::Vector { n: Lanes::Fixed(_), .. } | TypeKind::Mask { n: Lanes::Fixed(_), .. }
        )
    }

    /// Emits vector op `v` with result type `ty`; a trapping op gets the site of `tok`.
    pub fn vop(&mut self, tok: u32, v: VecOp, ty: Option<Ty>, args: &[Operand]) -> Option<Operand> {
        let op = Op::Intrinsic(self.intrinsic(v.name()));
        if v.trap_kinds().is_empty() {
            self.f.b.inst(op, ty.map(Self::ir), args).map(Operand::Value)
        } else {
            self.trapping(tok, op, ty, args)
        }
    }

    fn vval(&mut self, tok: u32, v: VecOp, ty: Ty, args: &[Operand]) -> E {
        match self.vop(tok, v, Some(ty), args) {
            Some(r) => E::V(r, ty),
            None => E::Never,
        }
    }

    fn sizeless(&mut self, tok: u32) -> E {
        self.vector_unsupported(Some(tok), "sizeless vectors")
    }

    /// `a op b` on vectors or masks of type `ty` (scalar op `o` of the operator).
    pub fn vec_binop(&mut self, tok: u32, o: Op, a: Operand, b: Operand, ty: Ty) -> Option<Operand> {
        if !self.fixed_simd(ty) {
            self.sizeless(tok);
            return None;
        }
        let mask = matches!(self.kind(ty), TypeKind::Mask { .. });
        let v = match o {
            Op::Add => VecOp::Add,
            Op::Sub => VecOp::Sub,
            Op::Mul => VecOp::Mul,
            Op::Div => VecOp::Div,
            Op::AddWrap => VecOp::AddWrap,
            Op::SubWrap => VecOp::SubWrap,
            Op::MulWrap => VecOp::MulWrap,
            Op::AddSat => VecOp::AddSat,
            Op::SubSat => VecOp::SubSat,
            Op::MulSat => VecOp::MulSat,
            Op::Shl => VecOp::Shl,
            Op::Shr => VecOp::Shr,
            Op::ShlWrap => VecOp::ShlWrap,
            Op::And if mask => VecOp::MaskAnd,
            Op::Or if mask => VecOp::MaskOr,
            Op::Xor if mask => VecOp::MaskXor,
            Op::And => VecOp::And,
            Op::Or => VecOp::Or,
            Op::Xor => VecOp::Xor,
            _ => {
                self.fail_at(Some(tok), "vector operator");
                return None;
            }
        };
        self.vop(tok, v, Some(ty), &[a, b])
    }

    /// `-v` / `~v` on a vector or mask.
    pub fn vec_unop(&mut self, tok: u32, o: Op, v: Operand, ty: Ty) -> E {
        if !self.fixed_simd(ty) {
            return self.sizeless(tok);
        }
        let op = match (o, self.kind(ty)) {
            (Op::Neg, _) => VecOp::Neg,
            (_, TypeKind::Mask { .. }) => VecOp::MaskNot,
            _ => VecOp::Not,
        };
        self.vval(tok, op, ty, &[v])
    }

    /// `a op b` compare of vectors (or masks, `==`/`!=` only) of type `ot`, result mask `rt`.
    pub fn vec_compare(&mut self, tok: u32, o: Op, a: Operand, b: Operand, ot: Ty, rt: Ty) -> E {
        if !self.fixed_simd(ot) {
            return self.sizeless(tok);
        }
        if matches!(self.kind(ot), TypeKind::Mask { .. }) {
            let Some(x) = self.vop(tok, VecOp::MaskXor, Some(rt), &[a, b]) else { return E::Never };
            return if o == Op::Ne { E::V(x, rt) } else { self.vval(tok, VecOp::MaskNot, rt, &[x]) };
        }
        let v = match o {
            Op::Eq => VecOp::Eq,
            Op::Ne => VecOp::Ne,
            Op::Lt => VecOp::Lt,
            Op::Le => VecOp::Le,
            Op::Gt => VecOp::Gt,
            _ => VecOp::Ge,
        };
        self.vval(tok, v, rt, &[a, b])
    }

    /// `v as~ T` (any `as` form) of a vector: lane-wise.
    pub fn vec_cast(&mut self, tok: u32, kind: CastKind, v: Operand, rt: Ty) -> Option<Operand> {
        self.vop(tok, VecOp::Cast(kind), Some(rt), &[v])
    }

    /// The lane type and lane count of a fixed vector type.
    fn lanes_of(&self, t: Ty) -> Option<(Ty, u32)> {
        match self.kind(t) {
            TypeKind::Vector { lane, n: Lanes::Fixed(k) } => Some((lane, k as u32)),
            _ => None,
        }
    }

    /// `check.bounds i (sub.sat (slice.len s) N-1)`: traps unless `i + N <= len`.
    fn vec_bounds(&mut self, tok: u32, s: Operand, i: Operand, lanes: u32) {
        let us = self.t_usize();
        let n = self.op1(Op::SliceLen, us, s);
        let k = self.int(lanes as i128 - 1);
        let lim = self.op2(Op::SubSat, us, n, k);
        self.trapping(tok, Op::BoundsCheck, None, &[i, lim]);
    }

    fn slice_ref(&mut self, mutable: bool, elem: Ty) -> Ty {
        let s = self.tys.intern(TypeKind::Slice { elem, align: 0 });
        self.t_ref(mutable, s)
    }

    /// A built-in method of a vector or mask receiver `p` (result type `rt`, name token `seg`).
    pub fn vec_method(&mut self, rt: Ty, p: Place, name: &str, args: &[u32], seg: u32, consts: &[u64]) -> E {
        let t = p.ty;
        if !self.fixed_simd(t) || (self.is_simd(rt) && !self.fixed_simd(rt)) {
            return self.sizeless(seg);
        }
        let Some(recv) = self.read(&p) else { return E::Never };
        let arg = |s: &mut Self, k: usize, want: Option<Ty>| -> Option<Operand> {
            let a = *args.get(k)?;
            s.val(a, want)
        };
        if let TypeKind::Mask { .. } = self.kind(t) {
            let op = match name {
                "select" => {
                    let Some(a) = arg(self, 0, Some(rt)) else { return E::Never };
                    let Some(b) = arg(self, 1, Some(rt)) else { return E::Never };
                    return self.vval(seg, VecOp::Select, rt, &[recv, a, b]);
                }
                "to_bits" => VecOp::ToBits,
                "to_vec" => VecOp::ToVec,
                "rewiden" => VecOp::Rewiden,
                _ => {
                    return self
                        .unsupported_at(Some(seg), format!("mask method `{name}` is not supported yet"));
                }
            };
            return self.vval(seg, op, rt, &[recv]);
        }
        let Some((lane, lanes)) = self.lanes_of(t) else { return self.sizeless(seg) };
        let unary = match name {
            "sqrt" => Some(VecOp::Sqrt),
            "abs" => Some(VecOp::Abs),
            "floor" => Some(VecOp::Floor),
            "ceil" => Some(VecOp::Ceil),
            "round" => Some(VecOp::Round),
            "trunc" => Some(VecOp::Trunc),
            "abs_wrap" => Some(VecOp::AbsWrap),
            "popcount" => Some(VecOp::Popcount),
            "clz" => Some(VecOp::Clz),
            "scan_add" => Some(VecOp::ScanAdd),
            "reduce_add" => Some(VecOp::ReduceAdd),
            "reduce_add_wrap" => Some(VecOp::ReduceAddWrap),
            "reduce_min" => Some(VecOp::ReduceMin),
            "reduce_max" => Some(VecOp::ReduceMax),
            "reduce_and" => Some(VecOp::ReduceAnd),
            "reduce_or" => Some(VecOp::ReduceOr),
            "split" => Some(VecOp::Split),
            "bitcast" => Some(VecOp::Bitcast),
            "widen_lo" => Some(VecOp::WidenLo),
            "widen_hi" => Some(VecOp::WidenHi),
            _ => None,
        };
        if let Some(op) = unary {
            return self.vval(seg, op, rt, &[recv]);
        }
        let binary = match name {
            "copysign" => Some(VecOp::Copysign),
            "min" => Some(VecOp::Min),
            "max" => Some(VecOp::Max),
            "min_num" => Some(VecOp::MinNum),
            "max_num" => Some(VecOp::MaxNum),
            "min_x86" => Some(VecOp::MinX86),
            "max_x86" => Some(VecOp::MaxX86),
            "andnot" => Some(VecOp::Andnot),
            "div_lanes" => Some(VecOp::DivLanes),
            "lookup" => Some(VecOp::Lookup),
            "narrow_wrap" => Some(VecOp::NarrowWrap),
            "narrow_sat" => Some(VecOp::NarrowSat),
            "checked_add" => Some(VecOp::CheckedAdd),
            "checked_sub" => Some(VecOp::CheckedSub),
            "checked_mul" => Some(VecOp::CheckedMul),
            _ => None,
        };
        if let Some(op) = binary {
            let Some(b) = arg(self, 0, Some(t)) else { return E::Never };
            return self.vval(seg, op, rt, &[recv, b]);
        }
        match name {
            "fma" => {
                let Some(b) = arg(self, 0, Some(t)) else { return E::Never };
                let Some(c) = arg(self, 1, Some(t)) else { return E::Never };
                self.vval(seg, VecOp::Fma, rt, &[recv, b, c])
            }
            "reduce_add_ordered" => {
                let Some(init) = arg(self, 0, Some(lane)) else { return E::Never };
                self.vval(seg, VecOp::ReduceAddOrdered, rt, &[recv, init])
            }
            "compress" | "expand" => {
                let Some(m) = arg(self, 0, None) else { return E::Never };
                let op = if name == "compress" { VecOp::Compress } else { VecOp::Expand };
                self.vval(seg, op, rt, &[recv, m])
            }
            "div_const" => {
                let Some(&k) = consts.first() else { return self.fail_at(Some(seg), "div_const divisor") };
                let fits = match self.kind(lane) {
                    TypeKind::Int { signed, bits } => {
                        let w = bits.bits();
                        let max = if signed { (1u128 << (w - 1)) - 1 } else { (1u128 << w) - 1 };
                        (k as u128) <= max
                    }
                    _ => false,
                };
                if !fits {
                    return self.unsupported_at(Some(seg), "a `div_const` divisor outside the lane type");
                }
                let k = self.int(k as i128);
                self.vval(seg, VecOp::DivConst, rt, &[recv, k])
            }
            "permute" => {
                let mut ops = vec![recv];
                for &k in consts {
                    ops.push(self.int(k as i128));
                }
                self.vval(seg, VecOp::Permute, rt, &ops)
            }
            "store" | "store_aligned" | "store_stream" | "store_masked" => {
                let want = if name == "store" || name == "store_masked" {
                    Some(self.slice_ref(true, lane))
                } else {
                    None
                };
                let Some(s) = arg(self, 0, want) else { return E::Never };
                let us = self.t_usize();
                let Some(i) = arg(self, 1, Some(us)) else { return E::Never };
                if name == "store_masked" {
                    let Some(m) = arg(self, 2, None) else { return E::Never };
                    self.vop(seg, VecOp::StoreMasked, None, &[s, i, recv, m]);
                    return E::Unit;
                }
                self.vec_bounds(seg, s, i, lanes);
                let op = if name == "store" { VecOp::Store } else { VecOp::StoreAligned };
                self.vop(seg, op, None, &[s, i, recv]);
                E::Unit
            }
            "scatter" => {
                let want = self.slice_ref(true, lane);
                let Some(s) = arg(self, 0, Some(want)) else { return E::Never };
                let Some(idx) = arg(self, 1, None) else { return E::Never };
                self.vop(seg, VecOp::Scatter, None, &[s, idx, recv]);
                E::Unit
            }
            _ => self.unsupported_at(Some(seg), format!("vector method `{name}` is not supported yet")),
        }
    }

    /// A type-relative built-in of vector or mask type `t` (`f32x4.splat(x)`) with result type `rt`.
    pub fn vec_assoc(&mut self, rt: Ty, t: Ty, name: &str, args: &[u32], tok: u32) -> E {
        if !self.fixed_simd(t) {
            return self.sizeless(tok);
        }
        let arg = |s: &mut Self, k: usize, want: Option<Ty>| -> Option<Operand> {
            let a = *args.get(k)?;
            s.val(a, want)
        };
        if let TypeKind::Mask { .. } = self.kind(t) {
            return match name {
                "first_n" => {
                    let us = self.t_usize();
                    let Some(k) = arg(self, 0, Some(us)) else { return E::Never };
                    self.vval(tok, VecOp::FirstN, rt, &[k])
                }
                "all" => self.vval(tok, VecOp::MaskAll, rt, &[]),
                "none" => self.vval(tok, VecOp::MaskNone, rt, &[]),
                _ => self.unsupported_at(Some(tok), format!("mask fn `{name}` is not supported yet")),
            };
        }
        let Some((lane, lanes)) = self.lanes_of(t) else { return self.sizeless(tok) };
        let us = self.t_usize();
        match name {
            "splat" => {
                let Some(x) = arg(self, 0, Some(lane)) else { return E::Never };
                self.vval(tok, VecOp::Splat, rt, &[x])
            }
            "join" => {
                let Some(a) = arg(self, 0, None) else { return E::Never };
                let Some(b) = arg(self, 1, None) else { return E::Never };
                self.vval(tok, VecOp::Join, rt, &[a, b])
            }
            "load" | "load_aligned" | "load_masked" => {
                let want = if name == "load_aligned" { None } else { Some(self.slice_ref(false, lane)) };
                let Some(s) = arg(self, 0, want) else { return E::Never };
                let Some(i) = arg(self, 1, Some(us)) else { return E::Never };
                if name == "load_masked" {
                    let Some(m) = arg(self, 2, None) else { return E::Never };
                    return self.vval(tok, VecOp::LoadMasked, rt, &[s, i, m]);
                }
                self.vec_bounds(tok, s, i, lanes);
                let op = if name == "load" { VecOp::Load } else { VecOp::LoadAligned };
                self.vval(tok, op, rt, &[s, i])
            }
            "gather" => {
                let want = self.slice_ref(false, lane);
                let Some(s) = arg(self, 0, Some(want)) else { return E::Never };
                let Some(idx) = arg(self, 1, None) else { return E::Never };
                self.vval(tok, VecOp::Gather, rt, &[s, idx])
            }
            _ => self.vector_unsupported(Some(tok), &format!("type-relative built-in `{name}`")),
        }
    }

    /// `s.align_to::<A>()` on slice value `s` of type `st`: `Option.Some` of `s` as `&[T align A]`
    /// when its address is a multiple of A, else `Option.None` (no blocks: a `select`).
    pub fn align_to(&mut self, rt: Ty, s: Operand, st: Ty, tok: u32) -> E {
        let Some(GArg::Type(at)) = (match self.kind(rt) {
            TypeKind::Adt { args, .. } => args.first().copied(),
            _ => None,
        }) else {
            return self.fail_at(Some(tok), "align_to result");
        };
        let Some((inner, _, _)) = self.pointee(at) else { return self.fail_at(Some(tok), "align_to result") };
        let TypeKind::Slice { elem, align } = self.kind(inner) else {
            return self.fail_at(Some(tok), "align_to result");
        };
        let us = self.t_usize();
        let bt = self.t_bool();
        let ep = self.t_ref(false, elem);
        let p = self.op1(Op::SlicePtr, ep, s);
        let addr = self.op1(Op::Cast(CastKind::Exact), us, p);
        let c = if align <= 1 {
            self.int(1)
        } else {
            let a = self.int(align as i128);
            let Some(r) = self.trapping(tok, Op::Rem, Some(us), &[addr, a]) else { return E::Never };
            let z = self.int(0);
            self.op2(Op::Eq, bt, r, z)
        };
        let s2 = self.convert_ptr(s, st, at);
        let (some, none) = (self.int(self.g.opt_some as i128), self.int(self.g.opt_none as i128));
        let yes = self.opn(Op::EnumMake, rt, &[some, s2]);
        let no = self.opn(Op::EnumMake, rt, &[none]);
        E::V(self.opn(Op::Select, rt, &[c, yes, no]), rt)
    }
}
