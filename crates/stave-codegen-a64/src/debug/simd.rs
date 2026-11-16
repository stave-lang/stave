//! Vector and mask ops (`stave_ir::simd`) on Advanced SIMD.
//!
//! A vector or mask value lives in its frame home ([`VTy`]: lanes in order, a mask lane all ones
//! or zero). An op loads its operands into temporary V registers, computes into another one, and
//! stores the result. Operand registers are only read, never written.
//!
//! Register bits past the vector's size are unspecified (a `u8x4` is the low 4 bytes of a 64-bit
//! arrangement). Lane-wise instructions compute garbage there, which nothing reads: an op that
//! folds lanes together or tests them (reductions, `to_bits`, trap and overflow checks) looks at
//! the valid lanes only.
//!
//! Lane ops are one instruction where Advanced SIMD has one. The others:
//! - `min_num`/`max_num`: `fmin`/`fmax`, then a NaN lane whose other operand is a number takes
//!   the number (`fminnm` would keep a signaling NaN). `min_x86`/`max_x86`: `fcmgt` + `bsl`.
//! - 64-bit lanes have no `mul`, `smin`/`umin`, `clz`: `mul` and `clz` go through general
//!   registers, `min`/`max` are `cmgt`/`cmhi` + `bsl`.
//! - `mul.sat` and `checked_mul`: a widening multiply narrowed back (64-bit lanes: `smulh`/
//!   `umulh` per lane). `checked_add`/`checked_sub` overflow where the saturating result
//!   differs from the wrapping one.
//! - `div_lanes`, `div_const`: `sdiv`/`udiv` per lane in lane order, each lane with the scalar
//!   checks, so the lowest trapping lane decides the trap kind.
//! - Reductions: `addv`/`smaxv`/... where the register holds exactly the vector and the order
//!   does not matter; otherwise the fixed tree (upper half folded onto the lower half).
//!   `reduce_add_ordered` and the float `scan_add` add lane by lane; the int scan is
//!   `log2(n)` shifted adds.
//! - Casts: widen / narrow chains, `scvtf`/`ucvtf`/`fcvtzs`/`fcvtzu` at the float's lane width,
//!   saturating narrows. `cast.try` and int64 → f32 run the scalar cast per lane.
//! - `permute`: `tbl` with a constant index vector. `lookup`: `tbl` for byte lanes, else a load
//!   per lane from the table spilled to the work area. `compress`/`expand`: lane by lane
//!   through the work area with a running position.
//! - `load_masked`, `store_masked`, `gather`, `scatter`: one scalar access per active lane.
//!   Stores check every lane before writing any.
//! - `to_bits`: lanes narrowed to bytes, bit 0 of each byte gathered by one multiply.

use stave_ir::layout::Shape;
use stave_ir::simd::{self, VecOp};
use stave_ir::{CastKind, Operand, PanicKind, Site, TypeId, Value};

use super::emit::{Gen, S15, S16, S17, SP, W, X, ZR};
use super::ops::Payload;
use super::ty::{self, Ty, VTy, half_arr, lane_of};
use super::{R, bad, unsupported};
use crate::enc::{
    Across, AddSub, Addr, Arr, CmpZ, Cond, FcvtRound, Fp, FpBin, IdxExt, ImmShift, Label, Lane, Logic,
    MemSize, Narrow, Reg, Shift, Sign, Tbl, VBin, VFpBin, VFpUn, VIntUn, VLogic, VReg, VShift, VecSize,
};

const B16: Arr = Arr::B16;
const NO_SHIFT: ImmShift = ImmShift::Lsl(0);

fn msize(lane: Lane) -> MemSize {
    match lane {
        Lane::B => MemSize::B,
        Lane::H => MemSize::H,
        Lane::S => MemSize::W,
        Lane::D => MemSize::X,
    }
}

fn wider(lane: Lane) -> Lane {
    match lane {
        Lane::B => Lane::H,
        Lane::H => Lane::S,
        _ => Lane::D,
    }
}

fn narrower(lane: Lane) -> Lane {
    match lane {
        Lane::D => Lane::S,
        Lane::S => Lane::H,
        _ => Lane::B,
    }
}

/// The byte arrangement covering a vector of `size` bytes.
fn bytes_arr(size: u8) -> Arr {
    if size == 16 { Arr::B16 } else { Arr::B8 }
}

impl<'a, 'm> Gen<'a, 'm> {
    fn vt(&self, o: Operand) -> R<VTy> {
        match o {
            Operand::Value(v) => match self.vty(v) {
                Ty::Vec(t) => Ok(t),
                Ty::Bad(why) => Err(unsupported(why)),
                _ => Err(bad("vector operand expected")),
            },
            _ => Err(bad("vector operand must be a value")),
        }
    }

    fn vt_of(&self, t: Option<TypeId>) -> R<VTy> {
        match t.map(|t| self.cg.ty(t)) {
            Some(Ty::Vec(t)) => Ok(t),
            Some(Ty::Bad(why)) => Err(unsupported(why)),
            _ => Err(bad("vector type expected")),
        }
    }

    /// Operand `o` in a temporary V register, to read only.
    fn use_v(&mut self, o: Operand) -> R<(VReg, VTy)> {
        let vt = self.vt(o)?;
        let r = self.tmp_f()?;
        self.load_vec(r, o, vt)?;
        Ok((r, vt))
    }

    /// Stores `r` as the vector value `v`.
    fn set_v(&mut self, v: Value, r: VReg) -> R<()> {
        let Ty::Vec(vt) = self.vty(v) else { return Err(bad("vector result expected")) };
        let h = self.home(v);
        self.stf(vt.home_size(), r, SP, h)
    }

    /// Loads exactly `size` bytes at `[base, #off]` into `r` (uses `x15`; `base` is not `x15`,
    /// `x17`).
    fn ld_bytes(&mut self, r: VReg, size: u8, base: Reg, off: i64) -> R<()> {
        match size {
            16 => self.ldf(VecSize::Q, r, base, off),
            8 => self.ldf(VecSize::D, r, base, off),
            4 => self.ldf(VecSize::S, r, base, off),
            _ => {
                self.ld(MemSize::H, false, S15, base, off)?;
                Ok(self.a.fmov_from_gpr(Fp::S, r, S15)?)
            }
        }
    }

    /// Stores exactly the low `size` bytes of `r` at `[base, #off]` (uses `x15`; `base` is not
    /// `x15`, `x17`).
    fn st_bytes(&mut self, r: VReg, size: u8, base: Reg, off: i64) -> R<()> {
        match size {
            16 => self.stf(VecSize::Q, r, base, off),
            8 => self.stf(VecSize::D, r, base, off),
            4 => self.stf(VecSize::S, r, base, off),
            _ => {
                self.a.umov(Lane::H, S15, r, 0)?;
                self.st(MemSize::H, S15, base, off)
            }
        }
    }

    /// Lane `k` of `v` in `rd`, extended to 64 bits by `signed`.
    fn lane_get(&mut self, rd: Reg, v: VReg, lane: Lane, k: u32, signed: bool) -> R<()> {
        if signed && lane != Lane::D {
            self.a.smov(lane, X, rd, v, k)?;
        } else {
            self.a.umov(lane, rd, v, k)?;
        }
        Ok(())
    }

    /// `rd` = nonzero iff any of the low `bytes` bytes of `v` is (uses `x15`; `rd` is not `x15`).
    fn any_bits(&mut self, rd: Reg, v: VReg, bytes: u32) -> R<()> {
        match bytes {
            16 => {
                self.a.umov(Lane::D, rd, v, 0)?;
                self.a.umov(Lane::D, S15, v, 1)?;
                self.a.logic_reg(Logic::Orr, X, rd, rd, S15, Shift::Lsl, 0)?;
            }
            8 => self.a.umov(Lane::D, rd, v, 0)?,
            4 => self.a.umov(Lane::S, rd, v, 0)?,
            _ => self.a.umov(Lane::H, rd, v, 0)?,
        }
        Ok(())
    }

    /// The upper half of the low `bytes` bytes of `v`, at the bottom of `rd`.
    fn high_half(&mut self, rd: VReg, v: VReg, bytes: u32) -> R<()> {
        if bytes == 16 {
            self.a.vext(B16, rd, v, v, 8)?;
        } else {
            self.a.vshift(VShift::Ushr, Arr::D2, rd, v, bytes * 4)?;
        }
        Ok(())
    }

    /// Every lane of `rd` = `val` (uses `x15`).
    fn splat_imm(&mut self, rd: VReg, arr: Arr, val: u64) -> R<()> {
        self.a.mov_imm(X, S15, val)?;
        Ok(self.a.dup_gpr(arr, rd, S15)?)
    }

    /// `rd` = the constant `lo` (low 8 bytes) and `hi` (uses `x15`).
    fn mat_v(&mut self, rd: VReg, lo: u64, hi: Option<u64>) -> R<()> {
        self.a.mov_imm(X, S15, lo)?;
        self.a.fmov_from_gpr(Fp::D, rd, S15)?;
        if let Some(hi) = hi {
            self.a.mov_imm(X, S15, hi)?;
            self.a.ins_gpr(Lane::D, rd, 1, S15)?;
        }
        Ok(())
    }

    /// Lane-wise int `min`/`max` of `a`, `b` into `rd` (not `a` or `b` for 64-bit lanes).
    fn int_minmax(&mut self, max: bool, vt: VTy, rd: VReg, a: VReg, b: VReg) -> R<()> {
        let arr = vt.arr();
        if vt.lane != Lane::D {
            let op = match (max, vt.signed()) {
                (true, true) => VBin::Smax,
                (true, false) => VBin::Umax,
                (false, true) => VBin::Smin,
                (false, false) => VBin::Umin,
            };
            return Ok(self.a.vbin(op, arr, rd, a, b)?);
        }
        self.a.vbin(if vt.signed() { VBin::Cmgt } else { VBin::Cmhi }, arr, rd, a, b)?;
        let (x, y) = if max { (a, b) } else { (b, a) };
        Ok(self.a.vlogic(VLogic::Bsl, B16, rd, x, y)?)
    }

    /// Lanes of size `fl` in `a` resized to `tl` in `rd`: widened by `signed`, or truncated.
    fn resize(&mut self, signed: bool, fl: Lane, tl: Lane, rd: VReg, a: VReg) -> R<()> {
        if fl == tl {
            return Ok(self.a.vmov(B16, rd, a)?);
        }
        let sign = if signed { Sign::Signed } else { Sign::Unsigned };
        let (mut cur, mut lane) = (a, fl);
        while (lane as u32) < (tl as u32) {
            self.a.vwiden(sign, half_arr(lane), rd, cur)?;
            cur = rd;
            lane = wider(lane);
        }
        while (lane as u32) > (tl as u32) {
            lane = narrower(lane);
            self.a.vnarrow(Narrow::Xtn, half_arr(lane), rd, cur)?;
            cur = rd;
        }
        Ok(())
    }

    /// Lanes of size `fl` in `a` narrowed to `tl` in `rd`: the first halving with `ops.0`, the
    /// others with `ops.1`.
    fn narrow(&mut self, ops: (Narrow, Narrow), fl: Lane, tl: Lane, rd: VReg, a: VReg) -> R<()> {
        let (mut cur, mut lane, mut op) = (a, fl, ops.0);
        while (lane as u32) > (tl as u32) {
            lane = narrower(lane);
            self.a.vnarrow(op, half_arr(lane), rd, cur)?;
            cur = rd;
            op = ops.1;
        }
        Ok(())
    }

    /// Lanes of `r` with the top bit set become the signed maximum.
    fn clamp_smax(&mut self, arr: Arr, r: VReg) -> R<()> {
        let (m, max) = (self.tmp_f()?, self.tmp_f()?);
        self.a.vcmp_zero(CmpZ::Lt, arr, m, r)?;
        self.a.movi(B16, max, 0xff, NO_SHIFT)?;
        self.a.vshift(VShift::Ushr, arr, max, max, 1)?;
        Ok(self.a.vlogic(VLogic::Bit, B16, r, max, m)?)
    }

    /// Writes the `Option` value `v` of type `t`: `Some(payload)` if `fail` is zero, else `None`
    /// with zero payload bytes. `fail` is not a scratch register.
    fn option_vec(&mut self, v: Value, t: TypeId, fail: Reg, payload: VReg, pvt: VTy) -> R<()> {
        let (none, some, _) =
            ty::option_shape(&self.cg.m.types, t).ok_or_else(|| bad("vector op result must be an Option"))?;
        let (size, off) = self.option_layout(t, some)?;
        let h = self.home(v);
        let keep = self.tmp_f()?;
        self.memzero(SP, h, size)?;
        self.a.cmp_imm(X, fail, 0)?;
        self.a.mov_imm(W, S15, u64::from(some))?;
        self.a.mov_imm(W, S16, u64::from(none))?;
        self.a.csel(W, S15, S15, S16, Cond::Eq)?;
        self.a.csetm(X, S16, Cond::Eq)?;
        self.st(MemSize::W, S15, SP, h)?;
        self.a.dup_gpr(Arr::D2, keep, S16)?;
        self.a.vlogic(VLogic::And, B16, keep, keep, payload)?;
        self.st_bytes(keep, pvt.size, SP, h + off)
    }

    pub(super) fn vec_op(&mut self, op: VecOp, i: u32) -> R<()> {
        use VecOp::*;
        let b = self.b;
        let ty = b.insts.ty[i as usize];
        let args = b.args(i);
        let site = self.site(i);
        let arg = |k: usize| args.get(k).copied().ok_or_else(|| bad("missing operand"));
        match op {
            Splat => {
                let v = self.res(i)?;
                let vt = self.vt_of(ty)?;
                let rd = if vt.is_float() {
                    let fx = self.use_f(arg(0)?, vt.sc())?;
                    let rd = self.tmp_f()?;
                    self.a.dup_elem(vt.arr(), rd, fx, 0)?;
                    rd
                } else {
                    let rx = self.use_g(arg(0)?)?;
                    let rd = self.tmp_f()?;
                    self.a.dup_gpr(vt.arr(), rd, rx)?;
                    rd
                };
                self.set_v(v, rd)
            }
            Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc | AbsWrap | Popcount | Clz | Not | MaskNot => {
                let v = self.res(i)?;
                let (a, vt) = self.use_v(arg(0)?)?;
                let rd = self.tmp_f()?;
                self.unary(op, vt, rd, a)?;
                self.set_v(v, rd)
            }
            Add | Sub | Mul | Div | Copysign | Min | Max | MinNum | MaxNum | MinX86 | MaxX86 | AddWrap
            | SubWrap | MulWrap | AddSat | SubSat | MulSat | And | Or | Xor | Andnot | MaskAnd | MaskOr
            | MaskXor => {
                let v = self.res(i)?;
                let (a, vt) = self.use_v(arg(0)?)?;
                let (c, _) = self.use_v(arg(1)?)?;
                let rd = self.tmp_f()?;
                self.binary(op, vt, rd, a, c)?;
                self.set_v(v, rd)
            }
            Fma => {
                let v = self.res(i)?;
                let (a, vt) = self.use_v(arg(0)?)?;
                let (c, _) = self.use_v(arg(1)?)?;
                let (d, _) = self.use_v(arg(2)?)?;
                let rd = self.tmp_f()?;
                self.a.vmov(B16, rd, d)?;
                self.a.vfp_bin(VFpBin::Fmla, vt.arr(), rd, a, c)?;
                self.set_v(v, rd)
            }
            CheckedAdd | CheckedSub | CheckedMul => self.checked(op, i, arg(0)?, arg(1)?),
            DivLanes | DivConst => self.div(op, i, arg(0)?, arg(1)?, site),
            Shl | Shr | ShlWrap => self.vshift(op, i, arg(0)?, arg(1)?, site),
            Eq | Ne | Lt | Le | Gt | Ge => self.vcompare(op, i, arg(0)?, arg(1)?),
            ReduceAdd | ReduceAddWrap | ReduceMin | ReduceMax | ReduceAnd | ReduceOr => {
                self.reduce(op, i, arg(0)?)
            }
            ReduceAddOrdered => {
                let v = self.res(i)?;
                let (a, vt) = self.use_v(arg(0)?)?;
                let sc = vt.sc();
                let init = self.use_f(arg(1)?, sc)?;
                let t = self.tmp_f()?;
                let rd = self.def_f(v)?;
                for k in 0..vt.lanes() {
                    self.a.dup_scalar(vt.lane, t, a, k)?;
                    self.a.fp_bin(FpBin::Add, sc.fp(), rd, if k == 0 { init } else { rd }, t);
                }
                Ok(())
            }
            ScanAdd => {
                let v = self.res(i)?;
                let (a, vt) = self.use_v(arg(0)?)?;
                let fp = vt.sc().fp();
                let (rd, acc, t) = (self.tmp_f()?, self.tmp_f()?, self.tmp_f()?);
                self.a.vmov(B16, rd, a)?;
                self.a.dup_scalar(vt.lane, acc, a, 0)?;
                for k in 1..vt.lanes() {
                    self.a.dup_scalar(vt.lane, t, a, k)?;
                    self.a.fp_bin(FpBin::Add, fp, acc, acc, t);
                    self.a.ins_elem(vt.lane, rd, k, acc, 0)?;
                }
                self.set_v(v, rd)
            }
            ScanAddWrap => {
                let v = self.res(i)?;
                let (a, vt) = self.use_v(arg(0)?)?;
                let (rd, zero, t) = (self.tmp_f()?, self.tmp_f()?, self.tmp_f()?);
                self.a.vmov(B16, rd, a)?;
                self.a.movi(B16, zero, 0, NO_SHIFT)?;
                // lane j += lane j - s, for s = 1, 2, 4, ..: the lanes shifted up by s, added
                let (bytes, reg) = (bytes_arr(vt.size), if vt.size == 16 { 16 } else { 8 });
                let mut s = vt.lane_bytes();
                while s < u32::from(vt.size) {
                    self.a.vext(bytes, t, zero, rd, reg - s)?;
                    self.a.vbin(VBin::Add, vt.arr(), rd, rd, t)?;
                    s *= 2;
                }
                self.set_v(v, rd)
            }
            Cast(kind) => self.vcast(kind, i, arg(0)?),
            Bitcast | ToVec => {
                let v = self.res(i)?;
                let (a, _) = self.use_v(arg(0)?)?;
                self.set_v(v, a)
            }
            Split => {
                let v = self.res(i)?;
                let t = ty.ok_or_else(|| bad("vec.split type"))?;
                let (a, vt) = self.use_v(arg(0)?)?;
                let (o0, o1) = match &self.layout(t)?.shape {
                    Shape::Fields(fs) if fs.len() == 2 => (fs[0].0 as i64, fs[1].0 as i64),
                    _ => return Err(bad("vec.split result")),
                };
                let h = self.home(v);
                let hi = self.tmp_f()?;
                self.st_bytes(a, vt.size / 2, SP, h + o0)?;
                self.high_half(hi, a, u32::from(vt.size))?;
                self.st_bytes(hi, vt.size / 2, SP, h + o1)
            }
            Join => {
                let v = self.res(i)?;
                let (a, half) = self.use_v(arg(0)?)?;
                let (c, _) = self.use_v(arg(1)?)?;
                let rd = self.tmp_f()?;
                self.a.vmov(B16, rd, a)?;
                self.a.ins_elem(lane_of(u32::from(half.size)), rd, 1, c, 0)?;
                self.set_v(v, rd)
            }
            WidenLo | WidenHi => {
                let v = self.res(i)?;
                let (a, from) = self.use_v(arg(0)?)?;
                let rd = self.tmp_f()?;
                // the upper half: the `2` form of a 128-bit source, else moved down first
                let (src, arr) = if op == WidenLo {
                    (a, half_arr(from.lane))
                } else if from.size == 16 {
                    (a, from.arr())
                } else {
                    self.high_half(rd, a, u32::from(from.size))?;
                    (rd, half_arr(from.lane))
                };
                if from.is_float() {
                    self.a.fcvtl(arr, rd, src)?;
                } else {
                    let sign = if from.signed() { Sign::Signed } else { Sign::Unsigned };
                    self.a.vwiden(sign, arr, rd, src)?;
                }
                self.set_v(v, rd)
            }
            NarrowWrap | NarrowSat => {
                let v = self.res(i)?;
                let to = self.vt_of(ty)?;
                let (a, _) = self.use_v(arg(0)?)?;
                let (c, _) = self.use_v(arg(1)?)?;
                let (rd, t) = (self.tmp_f()?, self.tmp_f()?);
                let nop = match (op == NarrowWrap, to.signed()) {
                    (true, _) => Narrow::Xtn,
                    (false, true) => Narrow::Sqxtn,
                    (false, false) => Narrow::Uqxtn,
                };
                let arr = half_arr(to.lane);
                self.a.vnarrow(nop, arr, rd, a)?;
                self.a.vnarrow(nop, arr, t, c)?;
                self.a.ins_elem(lane_of(u32::from(to.size) / 2), rd, 1, t, 0)?;
                self.set_v(v, rd)
            }
            Permute => {
                let v = self.res(i)?;
                let to = self.vt_of(ty)?;
                let (a, _) = self.use_v(arg(0)?)?;
                let w = to.lane_bytes() as usize;
                if args.len() != to.lanes() as usize + 1 {
                    return Err(bad("vec.permute lane indices"));
                }
                // byte indices into the source; the bytes past the result select zero
                let mut idx = [0xffu8; 16];
                for (j, o) in args[1..].iter().enumerate() {
                    let k = self.b.int_const(*o).ok_or_else(|| bad("vec.permute index"))? as usize;
                    for c in 0..w {
                        idx[j * w + c] = (k * w + c) as u8;
                    }
                }
                let (ix, rd) = (self.tmp_f()?, self.tmp_f()?);
                let word =
                    |at: usize| idx[at..at + 8].iter().rev().fold(0u64, |acc, &x| (acc << 8) | u64::from(x));
                self.mat_v(ix, word(0), (to.size == 16).then(|| word(8)))?;
                self.a.vtbl(Tbl::Tbl, bytes_arr(to.size), rd, a, 1, ix)?;
                self.set_v(v, rd)
            }
            Lookup => self.lookup(i, arg(0)?, arg(1)?),
            Compress | Expand => self.compress(op, i, arg(0)?, arg(1)?),
            Load | LoadAligned => {
                let v = self.res(i)?;
                let vt = self.vt_of(ty)?;
                let (p, _) = self.vmem(arg(0)?, false)?;
                let pa = self.lane_ptr(p, arg(1)?, vt, op == LoadAligned, site)?;
                let rd = self.tmp_f()?;
                self.ld_bytes(rd, vt.size, pa, 0)?;
                self.set_v(v, rd)
            }
            Store | StoreAligned => {
                let (p, _) = self.vmem(arg(0)?, false)?;
                let (x, vt) = self.use_v(arg(2)?)?;
                let pa = self.lane_ptr(p, arg(1)?, vt, op == StoreAligned, site)?;
                self.st_bytes(x, vt.size, pa, 0)
            }
            LoadMasked => self.load_masked(i, arg(0)?, arg(1)?, arg(2)?, site),
            StoreMasked => self.store_masked(arg(0)?, arg(1)?, arg(2)?, arg(3)?, site),
            Gather => self.gather(i, arg(0)?, arg(1)?, site),
            Scatter => self.scatter(arg(0)?, arg(1)?, arg(2)?, site),
            Select => {
                let v = self.res(i)?;
                let (m, _) = self.use_v(arg(0)?)?;
                let (a, _) = self.use_v(arg(1)?)?;
                let (c, _) = self.use_v(arg(2)?)?;
                let rd = self.tmp_f()?;
                self.a.vmov(B16, rd, m)?;
                self.a.vlogic(VLogic::Bsl, B16, rd, a, c)?;
                self.set_v(v, rd)
            }
            ToBits => self.mask_bits(i, arg(0)?),
            FirstN => {
                let v = self.res(i)?;
                let mt = self.vt_of(ty)?;
                let n = mt.lanes();
                let rk = self.use_g(arg(0)?)?;
                let (iota, count, rd) = (self.tmp_f()?, self.tmp_f()?, self.tmp_f()?);
                // lane j holds j; it is set iff j < min(k, n)
                let w = mt.lane_bytes();
                let per = 8 / w;
                let lanes =
                    |from: u32| (0..per).fold(0u64, |acc, j| acc | (u64::from(from + j) << (8 * w * j)));
                self.mat_v(iota, lanes(0), (mt.size == 16).then(|| lanes(per)))?;
                self.a.mov_imm(X, S15, u64::from(n))?;
                self.a.cmp(X, rk, S15)?;
                self.a.csel(X, S15, rk, S15, Cond::Lo)?;
                self.a.dup_gpr(mt.arr(), count, S15)?;
                self.a.vbin(VBin::Cmhi, mt.arr(), rd, count, iota)?;
                self.set_v(v, rd)
            }
            Rewiden => {
                let v = self.res(i)?;
                let to = self.vt_of(ty)?;
                let (m, from) = self.use_v(arg(0)?)?;
                let rd = self.tmp_f()?;
                self.resize(true, from.lane, to.lane, rd, m)?;
                self.set_v(v, rd)
            }
            MaskAll | MaskNone => {
                let v = self.res(i)?;
                let rd = self.tmp_f()?;
                self.a.movi(B16, rd, if op == MaskAll { 0xff } else { 0 }, NO_SHIFT)?;
                self.set_v(v, rd)
            }
        }
    }

    /// A lane-wise op of one operand: `a` into `rd` (not `a`).
    fn unary(&mut self, op: VecOp, vt: VTy, rd: VReg, a: VReg) -> R<()> {
        use VecOp::*;
        let arr = vt.arr();
        let fp = |o: VecOp| match o {
            Neg => VFpUn::Neg,
            Abs => VFpUn::Abs,
            Sqrt => VFpUn::Sqrt,
            Floor => VFpUn::RintM,
            Ceil => VFpUn::RintP,
            Round => VFpUn::RintN,
            _ => VFpUn::RintZ,
        };
        match op {
            Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc => self.a.vfp_un(fp(op), arr, rd, a)?,
            AbsWrap if vt.signed() => self.a.vint_un(VIntUn::Abs, arr, rd, a)?,
            Not | MaskNot => self.a.vint_un(VIntUn::Not, B16, rd, a)?,
            Popcount => {
                // per byte, then adjacent counts added up to the lane width
                self.a.vint_un(VIntUn::Cnt, bytes_arr(vt.size), rd, a)?;
                let mut lane = Lane::B;
                while lane != vt.lane {
                    let src = if vt.size == 16 { lane.arr() } else { half_arr(lane) };
                    self.a.vaddlp(Sign::Unsigned, src, rd, rd)?;
                    lane = wider(lane);
                }
            }
            Clz if vt.lane != Lane::D => self.a.vint_un(VIntUn::Clz, arr, rd, a)?,
            Clz => {
                for k in 0..2 {
                    self.a.umov(Lane::D, S15, a, k)?;
                    self.a.clz(X, S15, S15)?;
                    self.a.ins_gpr(Lane::D, rd, k, S15)?;
                }
            }
            _ => return Err(bad("vector unary op")),
        }
        Ok(())
    }

    /// A lane-wise op of two operands of the result's type: `a`, `c` into `rd` (neither of them).
    fn binary(&mut self, op: VecOp, vt: VTy, rd: VReg, a: VReg, c: VReg) -> R<()> {
        use VecOp::*;
        let arr = vt.arr();
        let signed = vt.signed();
        match op {
            Add => self.a.vfp_bin(VFpBin::Add, arr, rd, a, c)?,
            Sub => self.a.vfp_bin(VFpBin::Sub, arr, rd, a, c)?,
            Mul => self.a.vfp_bin(VFpBin::Mul, arr, rd, a, c)?,
            Div => self.a.vfp_bin(VFpBin::Div, arr, rd, a, c)?,
            Min if vt.is_float() => self.a.vfp_bin(VFpBin::Min, arr, rd, a, c)?,
            Max if vt.is_float() => self.a.vfp_bin(VFpBin::Max, arr, rd, a, c)?,
            Min | Max => self.int_minmax(op == Max, vt, rd, a, c)?,
            MinNum | MaxNum => {
                let (na, nc, pick) = (self.tmp_f()?, self.tmp_f()?, self.tmp_f()?);
                let o = if op == MinNum { VFpBin::Min } else { VFpBin::Max };
                self.a.vfp_bin(o, arr, rd, a, c)?;
                self.a.vfp_bin(VFpBin::Cmeq, arr, na, a, a)?;
                self.a.vfp_bin(VFpBin::Cmeq, arr, nc, c, c)?;
                // a NaN lane whose other operand is a number takes the number
                self.a.vlogic(VLogic::Bic, B16, pick, nc, na)?;
                self.a.vlogic(VLogic::Bit, B16, rd, c, pick)?;
                self.a.vlogic(VLogic::Bic, B16, pick, na, nc)?;
                self.a.vlogic(VLogic::Bit, B16, rd, a, pick)?;
            }
            MinX86 | MaxX86 => {
                let (x, y) = if op == MinX86 { (c, a) } else { (a, c) };
                self.a.vfp_bin(VFpBin::Cmgt, arr, rd, x, y)?;
                self.a.vlogic(VLogic::Bsl, B16, rd, a, c)?;
                // a NaN that came through is made quiet
                self.a.vfp_bin(VFpBin::Min, arr, rd, rd, rd)?;
            }
            Copysign => {
                // the sign bits: -0.0 in every lane
                self.a.movi(B16, rd, 0, NO_SHIFT)?;
                self.a.vfp_un(VFpUn::Neg, arr, rd, rd)?;
                self.a.vlogic(VLogic::Bsl, B16, rd, c, a)?;
            }
            AddWrap => self.a.vbin(VBin::Add, arr, rd, a, c)?,
            SubWrap => self.a.vbin(VBin::Sub, arr, rd, a, c)?,
            MulWrap if vt.lane != Lane::D => self.a.vbin(VBin::Mul, arr, rd, a, c)?,
            MulWrap => {
                for k in 0..2 {
                    self.a.umov(Lane::D, S15, a, k)?;
                    self.a.umov(Lane::D, S16, c, k)?;
                    self.a.mul(X, S15, S15, S16)?;
                    self.a.ins_gpr(Lane::D, rd, k, S15)?;
                }
            }
            AddSat => self.a.vbin(if signed { VBin::SqAdd } else { VBin::UqAdd }, arr, rd, a, c)?,
            SubSat => self.a.vbin(if signed { VBin::SqSub } else { VBin::UqSub }, arr, rd, a, c)?,
            MulSat if vt.lane != Lane::D => {
                // the exact double-width products, narrowed with saturation
                let sign = if signed { Sign::Signed } else { Sign::Unsigned };
                let nop = if signed { Narrow::Sqxtn } else { Narrow::Uqxtn };
                let half = half_arr(vt.lane);
                let wide = self.tmp_f()?;
                self.a.vmull(sign, half, wide, a, c)?;
                self.a.vnarrow(nop, half, rd, wide)?;
                if vt.size == 16 {
                    self.a.vmull(sign, arr, wide, a, c)?;
                    self.a.vnarrow(nop, arr, rd, wide)?;
                }
            }
            MulSat => {
                let (lo, hi, sat) = (self.tmp_g()?, self.tmp_g()?, self.tmp_g()?);
                for k in 0..2 {
                    self.a.umov(Lane::D, S15, a, k)?;
                    self.a.umov(Lane::D, S16, c, k)?;
                    if signed {
                        // the bound on overflow: MAX for equal signs, else MIN
                        self.a.logic_reg(Logic::Eor, X, sat, S15, S16, Shift::Lsl, 0)?;
                        self.a.shift_imm(Shift::Asr, X, sat, sat, 63)?;
                        self.a.logic_imm(Logic::Eor, X, sat, sat, i64::MAX as u64)?;
                        self.a.mul(X, lo, S15, S16)?;
                        self.a.smulh(hi, S15, S16)?;
                        self.a.addsub_reg(AddSub::Subs, X, ZR, hi, lo, Shift::Asr, 63)?;
                        self.a.csel(X, lo, lo, sat, Cond::Eq)?;
                    } else {
                        self.a.umulh(hi, S15, S16)?;
                        self.a.mul(X, lo, S15, S16)?;
                        self.a.cmp_imm(X, hi, 0)?;
                        self.a.csinv(X, lo, lo, ZR, Cond::Eq)?;
                    }
                    self.a.ins_gpr(Lane::D, rd, k, lo)?;
                }
            }
            And | MaskAnd => self.a.vlogic(VLogic::And, B16, rd, a, c)?,
            Or | MaskOr => self.a.vlogic(VLogic::Orr, B16, rd, a, c)?,
            Xor | MaskXor => self.a.vlogic(VLogic::Eor, B16, rd, a, c)?,
            Andnot => self.a.vlogic(VLogic::Bic, B16, rd, a, c)?,
            _ => return Err(bad("vector binary op")),
        }
        Ok(())
    }

    fn checked(&mut self, op: VecOp, i: u32, a: Operand, c: Operand) -> R<()> {
        let v = self.res(i)?;
        let t = self.b.insts.ty[i as usize].ok_or_else(|| bad("checked vector op type"))?;
        let (a, vt) = self.use_v(a)?;
        let (c, _) = self.use_v(c)?;
        let arr = vt.arr();
        let signed = vt.signed();
        let (rd, m) = (self.tmp_f()?, self.tmp_f()?);
        // nonzero iff a lane overflows
        let fail = self.tmp_g()?;
        match op {
            VecOp::CheckedAdd | VecOp::CheckedSub => {
                let (wrap, sat) = match (op == VecOp::CheckedAdd, signed) {
                    (true, true) => (VBin::Add, VBin::SqAdd),
                    (true, false) => (VBin::Add, VBin::UqAdd),
                    (false, true) => (VBin::Sub, VBin::SqSub),
                    (false, false) => (VBin::Sub, VBin::UqSub),
                };
                // a lane overflows iff saturation changes its result
                self.a.vbin(wrap, arr, rd, a, c)?;
                self.a.vbin(sat, arr, m, a, c)?;
                self.a.vbin(VBin::Cmeq, arr, m, m, rd)?;
                self.a.vint_un(VIntUn::Not, B16, m, m)?;
                self.any_bits(fail, m, u32::from(vt.size))?;
            }
            _ if vt.lane != Lane::D => {
                // a lane overflows iff its exact double-width product is not the truncated one
                // extended back
                let sign = if signed { Sign::Signed } else { Sign::Unsigned };
                let half = half_arr(vt.lane);
                let wide_arr = wider(vt.lane).arr();
                let (wide, back) = (self.tmp_f()?, self.tmp_f()?);
                self.a.vmull(sign, half, wide, a, c)?;
                self.a.vnarrow(Narrow::Xtn, half, rd, wide)?;
                self.a.vwiden(sign, half, back, rd)?;
                self.a.vbin(VBin::Cmeq, wide_arr, m, back, wide)?;
                if vt.size == 16 {
                    self.a.vmull(sign, arr, wide, a, c)?;
                    self.a.vnarrow(Narrow::Xtn, arr, rd, wide)?;
                    self.a.vwiden(sign, arr, back, rd)?;
                    self.a.vbin(VBin::Cmeq, wide_arr, back, back, wide)?;
                    self.a.vlogic(VLogic::And, B16, m, m, back)?;
                }
                self.a.vint_un(VIntUn::Not, B16, m, m)?;
                self.any_bits(fail, m, (2 * u32::from(vt.size)).min(16))?;
            }
            _ => {
                let (lo, hi) = (self.tmp_g()?, self.tmp_g()?);
                self.a.mov(X, fail, ZR)?;
                for k in 0..2 {
                    self.a.umov(Lane::D, S15, a, k)?;
                    self.a.umov(Lane::D, S16, c, k)?;
                    self.a.mul(X, lo, S15, S16)?;
                    if signed {
                        self.a.smulh(hi, S15, S16)?;
                        self.a.addsub_reg(AddSub::Subs, X, ZR, hi, lo, Shift::Asr, 63)?;
                    } else {
                        self.a.umulh(hi, S15, S16)?;
                        self.a.cmp_imm(X, hi, 0)?;
                    }
                    self.a.cset(W, hi, Cond::Ne)?;
                    self.a.logic_reg(Logic::Orr, X, fail, fail, hi, Shift::Lsl, 0)?;
                    self.a.ins_gpr(Lane::D, rd, k, lo)?;
                }
            }
        }
        self.option_vec(v, t, fail, rd, vt)
    }

    /// `div_lanes` / `div_const`: lane by lane with the scalar checks, so the lowest trapping
    /// lane decides the trap.
    fn div(&mut self, op: VecOp, i: u32, a: Operand, c: Operand, site: Site) -> R<()> {
        let v = self.res(i)?;
        let (a, vt) = self.use_v(a)?;
        let divisors = if op == VecOp::DivConst { None } else { Some(self.use_v(c)?.0) };
        let rd = self.tmp_f()?;
        let (ra, rc) = (self.tmp_g()?, self.tmp_g()?);
        let signed = vt.signed();
        let zero = self.trap(PanicKind::DivByZero, site)?;
        let over = if signed { Some(self.trap(PanicKind::Overflow, site)?) } else { None };
        if divisors.is_none() {
            let k = self.int_bits(c).ok_or_else(|| bad("vec.div_const divisor"))?;
            self.a.mov_imm(X, rc, k)?;
        }
        if signed {
            let min = (-1i64) << (vt.lane.bits() - 1);
            self.a.mov_imm(X, S15, min as u64)?;
        }
        for k in 0..vt.lanes() {
            self.lane_get(ra, a, vt.lane, k, signed)?;
            if let Some(d) = divisors {
                self.lane_get(rc, d, vt.lane, k, signed)?;
            }
            self.a.cbz(X, rc, zero)?;
            if let Some(over) = over {
                self.a.cmn_imm(X, rc, 1)?;
                self.a.ccmp(X, ra, S15, 0, Cond::Eq)?;
                self.a.b_cond(Cond::Eq, over)?;
                self.a.sdiv(X, ra, ra, rc)?;
            } else {
                self.a.udiv(X, ra, ra, rc)?;
            }
            self.a.ins_gpr(vt.lane, rd, k, ra)?;
        }
        self.set_v(v, rd)
    }

    fn vshift(&mut self, op: VecOp, i: u32, a: Operand, amount: Operand, site: Site) -> R<()> {
        let v = self.res(i)?;
        let t = self.b.insts.ty[i as usize];
        let vt = self.vt_of(t)?;
        let arr = vt.arr();
        let bits = u64::from(vt.lane.bits());
        let (a, _) = self.use_v(a)?;
        let (count, rd) = (self.tmp_f()?, self.tmp_f()?);
        // `ushl`/`sshl` shift left by the count in each lane, right by a negative one
        let per_lane = matches!(amount, Operand::Value(x) if self.b.value_ty(x) == t);
        if per_lane {
            let (n, _) = self.use_v(amount)?;
            if op == VecOp::ShlWrap {
                self.splat_imm(count, arr, bits - 1)?;
                self.a.vlogic(VLogic::And, B16, count, count, n)?;
            } else {
                let l = self.trap(PanicKind::Overflow, site)?;
                let any = self.tmp_g()?;
                self.splat_imm(count, arr, bits)?;
                self.a.vbin(VBin::Cmhs, arr, count, n, count)?;
                self.any_bits(any, count, u32::from(vt.size))?;
                self.a.cbnz(X, any, l)?;
                if op == VecOp::Shr {
                    self.a.vint_un(VIntUn::Neg, arr, count, n)?;
                } else {
                    self.a.vmov(B16, count, n)?;
                }
            }
        } else {
            let rn = self.use_g(amount)?;
            if op == VecOp::ShlWrap {
                self.a.logic_imm(Logic::And, X, S15, rn, bits - 1)?;
            } else {
                let l = self.trap(PanicKind::Overflow, site)?;
                self.a.cmp_imm(X, rn, bits)?;
                self.a.b_cond(Cond::Hs, l)?;
                if op == VecOp::Shr {
                    self.a.neg(X, S15, rn)?;
                } else {
                    self.a.mov(X, S15, rn)?;
                }
            }
            self.a.dup_gpr(arr, count, S15)?;
        }
        let sh = if op == VecOp::Shr && vt.signed() { VBin::Sshl } else { VBin::Ushl };
        self.a.vbin(sh, arr, rd, a, count)?;
        self.set_v(v, rd)
    }

    fn vcompare(&mut self, op: VecOp, i: u32, a: Operand, c: Operand) -> R<()> {
        use VecOp::*;
        let v = self.res(i)?;
        let (a, vt) = self.use_v(a)?;
        let (c, _) = self.use_v(c)?;
        let rd = self.tmp_f()?;
        let arr = vt.arr();
        // `lt`/`le` are `gt`/`ge` of the swapped operands; `ne` is `eq` inverted (true for
        // unordered floats)
        let (x, y) = if matches!(op, Lt | Le) { (c, a) } else { (a, c) };
        if vt.is_float() {
            let o = match op {
                Eq | Ne => VFpBin::Cmeq,
                Lt | Gt => VFpBin::Cmgt,
                _ => VFpBin::Cmge,
            };
            self.a.vfp_bin(o, arr, rd, x, y)?;
        } else {
            let o = match (op, vt.signed()) {
                (Eq | Ne, _) => VBin::Cmeq,
                (Lt | Gt, true) => VBin::Cmgt,
                (Lt | Gt, false) => VBin::Cmhi,
                (_, true) => VBin::Cmge,
                (_, false) => VBin::Cmhs,
            };
            self.a.vbin(o, arr, rd, x, y)?;
        }
        if op == Ne {
            self.a.vint_un(VIntUn::Not, B16, rd, rd)?;
        }
        self.set_v(v, rd)
    }

    fn reduce(&mut self, op: VecOp, i: u32, a: Operand) -> R<()> {
        use VecOp::*;
        let v = self.res(i)?;
        let (a, vt) = self.use_v(a)?;
        let arr = vt.arr();
        let float = vt.is_float();
        let max = op == ReduceMax;
        let t = self.tmp_f()?;
        // across-lanes instructions fold the whole register: only when it holds exactly the
        // vector, and never for the float sum, whose order is the tree's
        let across = vt.size >= 8 && matches!(arr, Arr::B8 | Arr::B16 | Arr::H4 | Arr::H8 | Arr::S4);
        let done = match op {
            ReduceAddWrap if across => {
                self.a.vacross(Across::Addv, arr, t, a)?;
                true
            }
            ReduceAddWrap if arr == Arr::S2 => {
                self.a.vbin(VBin::Addp, arr, t, a, a)?;
                true
            }
            ReduceAddWrap if arr == Arr::D2 => {
                self.a.addp_scalar(t, a)?;
                true
            }
            ReduceMin | ReduceMax if !float && across => {
                let o = match (max, vt.signed()) {
                    (true, true) => Across::Smaxv,
                    (true, false) => Across::Umaxv,
                    (false, true) => Across::Sminv,
                    (false, false) => Across::Uminv,
                };
                self.a.vacross(o, arr, t, a)?;
                true
            }
            ReduceMin | ReduceMax if !float && arr == Arr::S2 => {
                let o = match (max, vt.signed()) {
                    (true, true) => VBin::Smaxp,
                    (true, false) => VBin::Umaxp,
                    (false, true) => VBin::Sminp,
                    (false, false) => VBin::Uminp,
                };
                self.a.vbin(o, arr, t, a, a)?;
                true
            }
            ReduceMin | ReduceMax if float && arr == Arr::S4 => {
                self.a.vacross(if max { Across::Fmaxv } else { Across::Fminv }, arr, t, a)?;
                true
            }
            ReduceMin | ReduceMax if float => {
                self.a.vfp_bin(if max { VFpBin::Maxp } else { VFpBin::Minp }, arr, t, a, a)?;
                true
            }
            _ => false,
        };
        if !done {
            // the fixed tree: lane i with lane i + n/2, on the lower half each round
            let hi = self.tmp_f()?;
            let mut cur = a;
            let mut bytes = u32::from(vt.size);
            while bytes > vt.lane_bytes() {
                self.high_half(hi, cur, bytes)?;
                match op {
                    ReduceAdd => self.a.vfp_bin(VFpBin::Add, arr, t, cur, hi)?,
                    ReduceAddWrap => self.a.vbin(VBin::Add, arr, t, cur, hi)?,
                    ReduceAnd => self.a.vlogic(VLogic::And, B16, t, cur, hi)?,
                    ReduceOr => self.a.vlogic(VLogic::Orr, B16, t, cur, hi)?,
                    ReduceMin | ReduceMax if float => {
                        self.a.vfp_bin(if max { VFpBin::Max } else { VFpBin::Min }, arr, t, cur, hi)?
                    }
                    ReduceMin | ReduceMax => self.int_minmax(max, vt, t, cur, hi)?,
                    _ => return Err(bad("vector reduction")),
                }
                cur = t;
                bytes /= 2;
            }
        }
        if float {
            let rd = self.def_f(v)?;
            self.a.dup_scalar(vt.lane, rd, t, 0)?;
        } else {
            let rd = self.def_g(v)?;
            self.lane_get(rd, t, vt.lane, 0, vt.signed())?;
        }
        Ok(())
    }

    fn vcast(&mut self, kind: CastKind, i: u32, a: Operand) -> R<()> {
        let v = self.res(i)?;
        let t = self.b.insts.ty[i as usize].ok_or_else(|| bad("vec.cast type"))?;
        let try_ = kind == CastKind::Try;
        let to_t = if try_ {
            ty::option_shape(&self.cg.m.types, t).ok_or_else(|| bad("vec.cast.try result"))?.2
        } else {
            t
        };
        let to = self.vt_of(Some(to_t))?;
        let (a, from) = self.use_v(a)?;
        let rd = self.tmp_f()?;
        if !try_ && self.cast_simd(kind, from, to, rd, a)? {
            return self.set_v(v, rd);
        }
        // the scalar cast, lane by lane
        let fail = self.tmp_g()?;
        self.a.movi(B16, rd, 0, NO_SHIFT)?;
        self.a.mov(X, fail, ZR)?;
        let (fs, ts) = (from.sc(), to.sc());
        for k in 0..from.lanes() {
            let mark = self.tmp_mark();
            let src = if from.is_float() {
                let f = self.tmp_f()?;
                self.a.dup_scalar(from.lane, f, a, k)?;
                Payload::F(f, fs.fp())
            } else {
                let g = self.tmp_g()?;
                self.lane_get(g, a, from.lane, k, from.signed())?;
                Payload::G(g)
            };
            let dst =
                if to.is_float() { Payload::F(self.tmp_f()?, ts.fp()) } else { Payload::G(self.tmp_g()?) };
            let ok = self.cast_core(kind, fs, ts, src, dst)?;
            match dst {
                Payload::G(g) => self.a.ins_gpr(to.lane, rd, k, g)?,
                Payload::F(f, _) => self.a.ins_elem(to.lane, rd, k, f, 0)?,
            }
            if let Some(ok) = ok {
                self.a.logic_imm(Logic::Eor, W, ok, ok, 1)?;
                self.a.logic_reg(Logic::Orr, X, fail, fail, ok, Shift::Lsl, 0)?;
            }
            self.tmp_reset(mark);
        }
        if try_ { self.option_vec(v, t, fail, rd, to) } else { self.set_v(v, rd) }
    }

    /// The cast `a: from → to` as vector instructions into `rd` (not `a`). False: there is no
    /// such sequence (nothing is emitted).
    fn cast_simd(&mut self, kind: CastKind, from: VTy, to: VTy, rd: VReg, a: VReg) -> R<bool> {
        let (fl, tl) = (from.lane, to.lane);
        match (from.is_float(), to.is_float()) {
            (false, false) if kind == CastKind::Sat => self.int_sat(from, to, rd, a)?,
            (false, false) => self.resize(from.signed(), fl, tl, rd, a)?,
            (true, true) => match (fl, tl) {
                (Lane::S, Lane::D) => self.a.fcvtl(Arr::S2, rd, a)?,
                (Lane::D, Lane::S) => self.a.fcvtn(Arr::S2, rd, a)?,
                _ => self.a.vmov(B16, rd, a)?,
            },
            (false, true) => {
                // one rounding: the conversion runs at the float's width on the widened ints
                if (fl as u32) > (tl as u32) {
                    return Ok(false);
                }
                self.resize(from.signed(), fl, tl, rd, a)?;
                if from.signed() {
                    self.a.scvtf_v(to.arr(), rd, rd)?;
                } else {
                    self.a.ucvtf_v(to.arr(), rd, rd)?;
                }
            }
            (true, false) => {
                // saturating at the float's width (f32 → 64-bit ints: at f64, exactly), then a
                // saturating narrow
                let signed = to.signed();
                let (mut cur, mut lane) = (a, fl);
                if fl == Lane::S && tl == Lane::D {
                    self.a.fcvtl(Arr::S2, rd, a)?;
                    (cur, lane) = (rd, Lane::D);
                }
                let arr = if lane == Lane::D { Arr::D2 } else { from.arr() };
                self.a.fcvt_v(if signed { FcvtRound::Zs } else { FcvtRound::Zu }, arr, rd, cur)?;
                let nop = if signed { Narrow::Sqxtn } else { Narrow::Uqxtn };
                self.narrow((nop, nop), lane, tl, rd, rd)?;
            }
        }
        Ok(true)
    }

    /// The saturating int cast `a: from → to` into `rd` (not `a`).
    fn int_sat(&mut self, from: VTy, to: VTy, rd: VReg, a: VReg) -> R<()> {
        let (fs, ts) = (from.signed(), to.signed());
        let (fb, tb) = (from.lane as u32, to.lane as u32);
        if tb > fb {
            if fs && !ts {
                // negative lanes are 0
                self.a.vcmp_zero(CmpZ::Lt, from.arr(), rd, a)?;
                self.a.vlogic(VLogic::Bic, B16, rd, a, rd)?;
                return self.resize(false, from.lane, to.lane, rd, rd);
            }
            return self.resize(fs, from.lane, to.lane, rd, a);
        }
        if tb < fb {
            let ops = match (fs, ts) {
                (true, true) => (Narrow::Sqxtn, Narrow::Sqxtn),
                (true, false) => (Narrow::Sqxtun, Narrow::Uqxtn),
                (false, _) => (Narrow::Uqxtn, Narrow::Uqxtn),
            };
            self.narrow(ops, from.lane, to.lane, rd, a)?;
            if !fs && ts {
                self.clamp_smax(to.arr(), rd)?;
            }
            return Ok(());
        }
        match (fs, ts) {
            (true, false) => {
                self.a.vcmp_zero(CmpZ::Lt, from.arr(), rd, a)?;
                self.a.vlogic(VLogic::Bic, B16, rd, a, rd)?;
            }
            (false, true) => {
                self.a.vmov(B16, rd, a)?;
                self.clamp_smax(to.arr(), rd)?;
            }
            _ => self.a.vmov(B16, rd, a)?,
        }
        Ok(())
    }

    fn lookup(&mut self, i: u32, table: Operand, index: Operand) -> R<()> {
        let v = self.res(i)?;
        let (tbl, tt) = self.use_v(table)?;
        let (idx, it) = self.use_v(index)?;
        let rd = self.tmp_f()?;
        if tt.lane == Lane::B && it.lane == Lane::B {
            // `tbl` yields 0 for an index past the table register: its bytes past the table are
            // cleared
            let clean = self.tmp_f()?;
            let table = if tt.size == 16 {
                tbl
            } else {
                self.a.dup_scalar(lane_of(u32::from(tt.size)), clean, tbl, 0)?;
                clean
            };
            self.a.vtbl(Tbl::Tbl, bytes_arr(tt.size), rd, table, 1, idx)?;
            return self.set_v(v, rd);
        }
        let base = self.tmp_g()?;
        self.add_imm(base, SP, i64::from(self.an.vtmp_off))?;
        self.a.str_fp(VecSize::Q, tbl, Addr::Off(base, 0))?;
        let n = u64::from(tt.lanes());
        for j in 0..tt.lanes() {
            // an index past the table loads lane 0 and selects 0
            self.a.umov(it.lane, S15, idx, j)?;
            self.a.cmp_imm(X, S15, n)?;
            self.a.csel(X, S15, S15, ZR, Cond::Lo)?;
            self.a.ldr(
                msize(tt.lane),
                S16,
                Addr::Idx { base, index: S15, ext: IdxExt::Lsl, scaled: true },
            )?;
            self.a.csel(X, S16, S16, ZR, Cond::Lo)?;
            self.a.ins_gpr(tt.lane, rd, j, S16)?;
        }
        self.set_v(v, rd)
    }

    fn compress(&mut self, op: VecOp, i: u32, a: Operand, mask: Operand) -> R<()> {
        let v = self.res(i)?;
        let (a, vt) = self.use_v(a)?;
        let (m, _) = self.use_v(mask)?;
        let (t, rd) = (self.tmp_f()?, self.tmp_f()?);
        let base = self.tmp_g()?;
        self.add_imm(base, SP, i64::from(self.an.vtmp_off))?;
        let (lane, sz) = (vt.lane, msize(vt.lane));
        // x16: the packed position
        let packed = Addr::Idx { base, index: S16, ext: IdxExt::Lsl, scaled: true };
        self.a.mov(X, S16, ZR)?;
        if op == VecOp::Compress {
            // Every lane, zeroed when inactive, is stored at the packed position, which moves on
            // only past an active one: a zero is overwritten by the next active lane or is part
            // of the zero tail.
            self.a.vlogic(VLogic::And, B16, t, a, m)?;
            self.a.movi(B16, rd, 0, NO_SHIFT)?;
            self.a.str_fp(VecSize::Q, rd, Addr::Off(base, 0))?;
            for j in 0..vt.lanes() {
                self.a.umov(lane, S15, t, j)?;
                self.a.str(sz, S15, packed)?;
                self.a.umov(Lane::B, S15, m, j * vt.lane_bytes())?;
                self.a.logic_imm(Logic::And, X, S15, S15, 1)?;
                self.a.add(X, S16, S16, S15)?;
            }
            self.a.ldr_fp(VecSize::Q, rd, Addr::Off(base, 0))?;
        } else {
            self.a.str_fp(VecSize::Q, a, Addr::Off(base, 0))?;
            for j in 0..vt.lanes() {
                self.a.ldr(sz, S15, packed)?;
                self.a.umov(lane, S17, m, j)?;
                self.a.logic_reg(Logic::And, X, S15, S15, S17, Shift::Lsl, 0)?;
                self.a.ins_gpr(lane, rd, j, S15)?;
                self.a.logic_imm(Logic::And, X, S17, S17, 1)?;
                self.a.add(X, S16, S16, S17)?;
            }
        }
        self.set_v(v, rd)
    }

    /// The data pointer of the slice or array operand `s` of a vector memory op, and with
    /// `want_len` its length, in registers of the current instruction.
    fn vmem(&mut self, s: Operand, want_len: bool) -> R<(Reg, Option<Reg>)> {
        let Operand::Value(sv) = s else { return Err(bad("vector memory operand")) };
        let st = self.b.value_ty(sv).ok_or_else(|| bad("value out of range"))?;
        let mo = simd::mem_operand(&self.cg.m.types, st).ok_or_else(|| bad("vector memory operand type"))?;
        let (p, len) = match mo.len {
            Some(n) => {
                let p = self.use_g(s)?;
                let len = if want_len {
                    let r = self.tmp_g()?;
                    self.a.mov_imm(X, r, n)?;
                    Some(r)
                } else {
                    None
                };
                (p, len)
            }
            None => {
                let h = self.home(sv);
                let p = self.tmp_g()?;
                self.ld(MemSize::X, false, p, SP, h)?;
                let len = if want_len {
                    let r = self.tmp_g()?;
                    self.ld(MemSize::X, false, r, SP, h + 8)?;
                    Some(r)
                } else {
                    None
                };
                (p, len)
            }
        };
        Ok((p, len))
    }

    /// The address of lane `i` of the memory at `p`. `aligned`: traps `Bounds` unless `i` is a
    /// multiple of the lane count.
    fn lane_ptr(&mut self, p: Reg, i: Operand, vt: VTy, aligned: bool, site: Site) -> R<Reg> {
        let n = u64::from(vt.lanes());
        let shift = vt.lane as u32;
        if let Some(k) = self.int_bits(i) {
            if aligned && !k.is_multiple_of(n) {
                let l = self.trap(PanicKind::Bounds, site)?;
                self.a.b(l)?;
            }
            let pa = self.tmp_g()?;
            self.add_imm(pa, p, k.wrapping_shl(shift) as i64)?;
            return Ok(pa);
        }
        let ri = self.use_g(i)?;
        if aligned {
            let l = self.trap(PanicKind::Bounds, site)?;
            self.a.tst_imm(X, ri, n - 1)?;
            self.a.b_cond(Cond::Ne, l)?;
        }
        let pa = self.tmp_g()?;
        self.a.addsub_reg(AddSub::Add, X, pa, p, ri, Shift::Lsl, shift)?;
        Ok(pa)
    }

    /// The bounds check of lane `k` of a masked access at `(index, length)`: branches to `skip`
    /// when the lane is inactive, to `trap` when it is past the end; else the lane's index is in
    /// `x16`.
    fn masked_lane(&mut self, m: VReg, vt: VTy, k: u32, at: (Reg, Reg), skip: Label, trap: Label) -> R<()> {
        let (ri, len) = at;
        self.a.umov(Lane::B, S15, m, k * vt.lane_bytes())?;
        self.a.cbz(W, S15, skip)?;
        // carry: the index wrapped
        self.a.addsub_imm(AddSub::Adds, X, S16, ri, u64::from(k))?;
        self.a.b_cond(Cond::Hs, trap)?;
        self.a.cmp(X, S16, len)?;
        self.a.b_cond(Cond::Hs, trap)?;
        Ok(())
    }

    fn load_masked(&mut self, i: u32, s: Operand, index: Operand, mask: Operand, site: Site) -> R<()> {
        let v = self.res(i)?;
        let vt = self.vt_of(self.b.insts.ty[i as usize])?;
        let (p, len) = self.vmem(s, true)?;
        let len = len.ok_or_else(|| bad("vector memory length"))?;
        let ri = self.use_g(index)?;
        let (m, _) = self.use_v(mask)?;
        let rd = self.tmp_f()?;
        let trap = self.trap(PanicKind::Bounds, site)?;
        let at = Addr::Idx { base: p, index: S16, ext: IdxExt::Lsl, scaled: true };
        self.a.movi(B16, rd, 0, NO_SHIFT)?;
        for k in 0..vt.lanes() {
            let skip = self.a.new_label();
            self.masked_lane(m, vt, k, (ri, len), skip, trap)?;
            self.a.ldr(msize(vt.lane), S15, at)?;
            self.a.ins_gpr(vt.lane, rd, k, S15)?;
            self.a.bind(skip)?;
        }
        self.set_v(v, rd)
    }

    fn store_masked(&mut self, s: Operand, index: Operand, x: Operand, mask: Operand, site: Site) -> R<()> {
        let (p, len) = self.vmem(s, true)?;
        let len = len.ok_or_else(|| bad("vector memory length"))?;
        let ri = self.use_g(index)?;
        let (x, vt) = self.use_v(x)?;
        let (m, _) = self.use_v(mask)?;
        let trap = self.trap(PanicKind::Bounds, site)?;
        let at = Addr::Idx { base: p, index: S16, ext: IdxExt::Lsl, scaled: true };
        // every active lane is checked before any is written
        for k in 0..vt.lanes() {
            let skip = self.a.new_label();
            self.masked_lane(m, vt, k, (ri, len), skip, trap)?;
            self.a.bind(skip)?;
        }
        for k in 0..vt.lanes() {
            let skip = self.a.new_label();
            self.a.umov(Lane::B, S15, m, k * vt.lane_bytes())?;
            self.a.cbz(W, S15, skip)?;
            self.a.addsub_imm(AddSub::Add, X, S16, ri, u64::from(k))?;
            self.a.umov(vt.lane, S15, x, k)?;
            self.a.str(msize(vt.lane), S15, at)?;
            self.a.bind(skip)?;
        }
        Ok(())
    }

    fn gather(&mut self, i: u32, s: Operand, index: Operand, site: Site) -> R<()> {
        let v = self.res(i)?;
        let vt = self.vt_of(self.b.insts.ty[i as usize])?;
        let (p, len) = self.vmem(s, true)?;
        let len = len.ok_or_else(|| bad("vector memory length"))?;
        let (idx, it) = self.use_v(index)?;
        let rd = self.tmp_f()?;
        let trap = self.trap(PanicKind::Bounds, site)?;
        for j in 0..vt.lanes() {
            self.a.umov(it.lane, S15, idx, j)?;
            self.a.cmp(X, S15, len)?;
            self.a.b_cond(Cond::Hs, trap)?;
            self.a.ldr(
                msize(vt.lane),
                S16,
                Addr::Idx { base: p, index: S15, ext: IdxExt::Lsl, scaled: true },
            )?;
            self.a.ins_gpr(vt.lane, rd, j, S16)?;
        }
        self.set_v(v, rd)
    }

    fn scatter(&mut self, s: Operand, index: Operand, x: Operand, site: Site) -> R<()> {
        let (p, len) = self.vmem(s, true)?;
        let len = len.ok_or_else(|| bad("vector memory length"))?;
        let (idx, it) = self.use_v(index)?;
        let (x, vt) = self.use_v(x)?;
        let trap = self.trap(PanicKind::Bounds, site)?;
        // every index is checked before any lane is written
        for j in 0..vt.lanes() {
            self.a.umov(it.lane, S15, idx, j)?;
            self.a.cmp(X, S15, len)?;
            self.a.b_cond(Cond::Hs, trap)?;
        }
        // in lane order: on equal indices the highest lane is written last
        for j in 0..vt.lanes() {
            self.a.umov(it.lane, S15, idx, j)?;
            self.a.umov(vt.lane, S16, x, j)?;
            self.a.str(
                msize(vt.lane),
                S16,
                Addr::Idx { base: p, index: S15, ext: IdxExt::Lsl, scaled: true },
            )?;
        }
        Ok(())
    }

    fn mask_bits(&mut self, i: u32, mask: Operand) -> R<()> {
        const ONES: u64 = 0x0101_0101_0101_0101;
        // byte k times 2^(56 - 7k) lands on bit 56 + k; no two products share a bit
        const GATHER: u64 = 0x0102_0408_1020_4080;
        let v = self.res(i)?;
        let (m, mt) = self.use_v(mask)?;
        let n = mt.lanes();
        let t = self.tmp_f()?;
        // one byte per lane
        let (mut cur, mut lane) = (m, mt.lane);
        while lane != Lane::B {
            lane = narrower(lane);
            self.a.vnarrow(Narrow::Xtn, half_arr(lane), t, cur)?;
            cur = t;
        }
        let rd = self.def_g(v)?;
        self.a.umov(Lane::D, rd, cur, 0)?;
        if n >= 8 {
            self.a.logic_imm(Logic::And, X, rd, rd, ONES)?;
        } else {
            self.a.mov_imm(X, S15, ONES & ((1u64 << (8 * n)) - 1))?;
            self.a.logic_reg(Logic::And, X, rd, rd, S15, Shift::Lsl, 0)?;
        }
        self.a.mov_imm(X, S16, GATHER)?;
        self.a.mul(X, rd, rd, S16)?;
        self.a.shift_imm(Shift::Lsr, X, rd, rd, 56)?;
        if n == 16 {
            self.a.umov(Lane::D, S15, cur, 1)?;
            self.a.logic_imm(Logic::And, X, S15, S15, ONES)?;
            self.a.mul(X, S15, S15, S16)?;
            self.a.shift_imm(Shift::Lsr, X, S15, S15, 56)?;
            self.a.logic_reg(Logic::Orr, X, rd, rd, S15, Shift::Lsl, 8)?;
        }
        Ok(())
    }
}
