//! Vector loads, stores, gathers, and scatters.

use stave_ir::simd::{self as vs, VecOp};
use stave_ir::{Op, Operand, PanicKind, TypeId, Value};

use super::{Form, Sh, lane, pmovx, scale_of, size_of, y};
use crate::debug::func::*;
use crate::debug::{Res, bad};
use crate::enc::*;

/// The length of a vector memory operand: a slice's, in a register, or an array's.
#[derive(Clone, Copy)]
enum VLen {
    R(Gpr),
    K(u64),
}

impl Cg<'_, '_> {
    /// The address of element `i` (lanes of `w` bytes) of the slice or array behind `s`.
    pub(in crate::debug) fn vslice(&mut self, s: Operand, i: Operand, w: u32) -> Res<Mem> {
        let st = self.oty(s).ok_or_else(|| bad("vector memory operand must be a value"))?;
        let ptr = vs::elem_ptr_operand(&self.cx.m.types, st).is_some();
        let slice = !ptr
            && vs::mem_operand(&self.cx.m.types, st)
                .ok_or_else(|| bad("vector memory operand type"))?
                .len
                .is_none();
        let base = if slice {
            // a slice reference: (address, length)
            let fat = self.vmem(s)?;
            let g = self.alloc_g()?;
            self.a.mov_rm(Size::Q, g, fat)?;
            g
        } else {
            self.read_g(s)?
        };
        if let Some(k) = self.int_const(i)
            && let Ok(d) = i32::try_from(k * i128::from(w))
        {
            return Ok(Mem::base(base, d));
        }
        let ig = self.read_g(i)?;
        Ok(Mem::index(base, ig, scale_of(w), 0))
    }

    /// Traps `Bounds` unless `i` is a multiple of the lane count `n`.
    fn valigned(&mut self, i: Operand, n: u32) -> Res<()> {
        if let Some(k) = self.int_const(i) {
            return if k % i128::from(n) != 0 { self.trap_now(PanicKind::Bounds) } else { Ok(()) };
        }
        let g = self.read_g(i)?;
        self.a.test_ri(Size::D, g, i64::from(n - 1))?;
        self.jtrap(Cond::Ne, PanicKind::Bounds)
    }

    /// Release: whether the `vec.load` `i` of a whole `xmm` / `ymm` is left to the vector op
    /// right after it, its only reader, as that op's second operand. VEX only: a legacy SSE
    /// memory operand must be aligned. The slice and the index then stay alive up to that op.
    pub(crate) fn fold_vload(&mut self, i: u32, bd: &stave_ir::ir::BlockData) -> Res<bool> {
        let b = self.b;
        let iu = i as usize;
        if b.insts.op[iu] != Op::Vec(VecOp::Load)
            || !self.vex()
            || i + 1 >= bd.insts.start.saturating_add(bd.insts.len)
        {
            return Ok(false);
        }
        let (Some(v), Some(t)) = (b.insts.result[iu], b.insts.ty[iu]) else { return Ok(false) };
        let reads = matches!(
            b.insts.op[iu + 1],
            Op::Vec(
                VecOp::Add
                    | VecOp::Sub
                    | VecOp::Mul
                    | VecOp::Div
                    | VecOp::MinX86
                    | VecOp::MaxX86
                    | VecOp::AddWrap
                    | VecOp::SubWrap
                    | VecOp::And
                    | VecOp::Or
                    | VecOp::Xor
            )
        );
        let second = matches!(b.args(i + 1), &[x, y] if y == Operand::Value(v) && x != y);
        // an op without a reader is not emitted, and would take the load with it
        let emitted =
            b.insts.result[iu + 1].is_some_and(|r| self.an.used.get(r.index()).copied().unwrap_or(true));
        if !reads
            || !second
            || !emitted
            || b.insts.ty[iu + 1] != Some(t)
            || self.uses.get(v.index()).copied() != Some(1)
            || self.vshape(t)?.size < 16
        {
            return Ok(false);
        }
        for &o in b.args(i) {
            self.extend_use(o, self.pos + 1);
        }
        self.folded[v.index()] = i;
        Ok(true)
    }

    pub(super) fn v_load(&mut self, aligned: bool, s: Operand, i: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        if aligned {
            self.valigned(i, sh.n)?;
        }
        let m = self.vslice(s, i, sh.w)?;
        let x = self.alloc_x()?;
        self.vload(x, m, sh.size)?;
        self.vput(rv, x, sh.size)
    }

    pub(super) fn v_store(&mut self, aligned: bool, s: Operand, i: Operand, v: Operand) -> Res<()> {
        let sh = self.voshape(v)?;
        if aligned {
            self.valigned(i, sh.n)?;
        }
        let m = self.vslice(s, i, sh.w)?;
        let x = self.vread(v, sh.size)?;
        self.vstore(m, x, sh.size)
    }

    /// The address and the length of the slice or array behind `s`.
    fn vbase(&mut self, s: Operand) -> Res<(Gpr, VLen)> {
        let st = self.oty(s).ok_or_else(|| bad("vector memory operand must be a value"))?;
        let mo = vs::mem_operand(&self.cx.m.types, st).ok_or_else(|| bad("vector memory operand type"))?;
        match mo.len {
            Some(n) => Ok((self.read_g(s)?, VLen::K(n))),
            None => {
                let fat = self.vmem(s)?;
                let (base, len) = (self.alloc_g()?, self.alloc_g()?);
                self.a.mov_rm(Size::Q, base, fat)?;
                self.a.mov_rm(Size::Q, len, disp(fat, 8)?)?;
                Ok((base, VLen::R(len)))
            }
        }
    }

    /// Traps `Bounds` when an active lane of the mask bits `bits` at index `i` is at or past
    /// `len`: with no active lane nothing is checked, else the highest one decides. Clobbers
    /// `S0`, `S1`.
    fn vmasked_check(&mut self, bits: Gpr, i: Gpr, len: VLen) -> Res<()> {
        let ok = self.a.new_label();
        self.a.test_rr(Size::D, bits, bits);
        self.a.jcc(Cond::E, ok)?;
        self.a.bsr(Size::D, S1, bits)?;
        match len {
            VLen::R(l) => self.a.mov_rr(Size::Q, S0, l),
            VLen::K(n) => self.a.mov_ri(Size::Q, S0, n as i64)?,
        }
        // S0 = lanes left from `i` to the end
        self.a.alu_rr(AluOp::Sub, Size::Q, S0, i);
        self.jtrap(Cond::Be, PanicKind::Bounds)?;
        self.a.alu_rr(AluOp::Cmp, Size::Q, S1, S0);
        self.jtrap(Cond::Ae, PanicKind::Bounds)?;
        Ok(self.a.bind(ok)?)
    }

    /// The masked load and store of a shape's lanes (`vmaskmov*`, `vpmaskmov*`): 32- and 64-bit
    /// lanes at the VEX level. A lane is active by its sign bit, and an inactive lane neither
    /// faults nor is written.
    fn vmaskmov(&self, sh: Sh) -> Option<(VBin, VMaskStore)> {
        if !self.vex() || sh.w < 4 {
            return None;
        }
        Some(match (sh.float().is_some(), sh.w) {
            (true, 4) => (VBin::Maskmovps, VMaskStore::Maskmovps),
            (true, _) => (VBin::Maskmovpd, VMaskStore::Maskmovpd),
            (false, 4) => (VBin::Pmaskmovd, VMaskStore::Pmaskmovd),
            (false, _) => (VBin::Pmaskmovq, VMaskStore::Pmaskmovq),
        })
    }

    /// Active lanes only: inactive lanes read as zero and their memory is not touched.
    pub(super) fn v_load_masked(
        &mut self,
        s: Operand,
        i: Operand,
        m: Operand,
        rt: TypeId,
        rv: Value,
    ) -> Res<()> {
        let sh = self.vshape(rt)?;
        let packed = self.vmaskmov(sh);
        let mm = if packed.is_none() { Some(self.vmem(m)?) } else { None };
        let (base, len) = self.vbase(s)?;
        let ig = self.read_g(i)?;
        let x = self.vget(m, sh.size)?;
        let bits = self.alloc_g()?;
        self.vmask_bits(bits, x, sh)?;
        self.vmasked_check(bits, ig, len)?;
        let mem = Mem::index(base, ig, scale_of(sh.w), 0);
        let Some(mm) = mm else {
            let Some((op, _)) = packed else { return Err(bad("internal: masked load")) };
            // a mask of fewer than 16 bytes has zeros above it: those lanes are inactive
            if self.vform(sh.wide())? == Form::Y {
                self.a.vex_bin(op, y(x), y(x), mem)?;
            } else {
                self.a.vex_bin(op, x, x, mem)?;
            }
            return self.vput(rv, x, sh.size);
        };
        let d = self.def_mem(rv)?;
        self.zero_mem(d, sh.size)?;
        for k in 0..sh.n {
            let skip = self.a.new_label();
            self.a.alu_mi(AluOp::Cmp, Size::B, lane(mm, k, sh.w)?, 0)?;
            self.a.jcc(Cond::E, skip)?;
            self.sized_load(S0, lane(mem, k, sh.w)?, sh.w as u8, false)?;
            self.a.mov_mr(size_of(sh.w), lane(d, k, sh.w)?, S0)?;
            self.a.bind(skip)?;
        }
        Ok(())
    }

    /// Active lanes only: the memory of inactive lanes is not touched. Without a masked store
    /// instruction: up to 8 lanes, every lane is stored without a branch, an inactive one back
    /// to its own bytes in the home of `v`; more lanes, a loop over the set mask bits.
    pub(super) fn v_store_masked(&mut self, s: Operand, i: Operand, v: Operand, m: Operand) -> Res<()> {
        let sh = self.voshape(v)?;
        let packed = match self.vmaskmov(sh) {
            Some((_, op)) => Some((op, self.vread(v, sh.size)?)),
            None => None,
        };
        let home = if packed.is_none() { Some(self.vmem(v)?) } else { None };
        let (base, len) = self.vbase(s)?;
        let ig = self.read_g(i)?;
        let x = self.vread(m, sh.size)?;
        let bits = self.alloc_g()?;
        self.vmask_bits(bits, x, sh)?;
        self.vmasked_check(bits, ig, len)?;
        let mem = Mem::index(base, ig, scale_of(sh.w), 0);
        if let Some((op, xv)) = packed {
            if self.vform(sh.wide())? == Form::Y {
                self.a.vex_maskstore(op, mem, y(x), y(xv))?;
            } else {
                self.a.vex_maskstore(op, mem, x, xv)?;
            }
            return Ok(());
        }
        let Some(mv) = home else { return Err(bad("internal: masked store")) };
        let (t, u) = (self.alloc_g()?, self.alloc_g()?);
        let (wb, ws) = (sh.w as u8, size_of(sh.w));
        if sh.n <= 8 {
            for k in 0..sh.n {
                let src = lane(mv, k, sh.w)?;
                self.a.lea(Size::Q, t, src)?;
                self.a.lea(Size::Q, u, lane(mem, k, sh.w)?)?;
                self.a.test_ri(Size::D, bits, 1 << k)?;
                self.a.cmov(Cond::Ne, Size::Q, t, u)?;
                self.sized_load(S0, src, wb, false)?;
                self.a.mov_mr(ws, Mem::base(t, 0), S0)?;
            }
            return Ok(());
        }
        let (top, done) = (self.a.new_label(), self.a.new_label());
        let sc = scale_of(sh.w);
        self.a.lea(Size::Q, t, mv)?;
        self.a.lea(Size::Q, u, mem)?;
        self.a.bind(top)?;
        self.a.test_rr(Size::D, bits, bits);
        self.a.jcc(Cond::E, done)?;
        self.a.bsf(Size::D, S1, bits)?;
        self.sized_load(S0, Mem::index(t, S1, sc, 0), wb, false)?;
        self.a.mov_mr(ws, Mem::index(u, S1, sc, 0), S0)?;
        // clear the lowest set bit
        self.a.lea(Size::D, S0, Mem::base(bits, -1))?;
        self.a.alu_rr(AluOp::And, Size::D, bits, S0);
        self.a.jmp(top)?;
        Ok(self.a.bind(done)?)
    }

    /// Traps `Bounds` unless every index lane (zero-extended) is below `len`. Clobbers `S0`,
    /// `S1`.
    fn vindex_check(&mut self, mi: Mem, ish: Sh, len: VLen) -> Res<()> {
        for j in 0..ish.n {
            self.sized_load(S0, lane(mi, j, ish.w)?, ish.w as u8, false)?;
            match len {
                VLen::R(l) => self.a.alu_rr(AluOp::Cmp, Size::Q, S0, l),
                VLen::K(n) => {
                    self.a.mov_ri(Size::Q, S1, n as i64)?;
                    self.a.alu_rr(AluOp::Cmp, Size::Q, S0, S1);
                }
            }
            self.jtrap(Cond::Ae, PanicKind::Bounds)?;
        }
        Ok(())
    }

    /// `XS0` = the mask of a gather into a vector of shape `sh`: all ones in its lanes.
    fn vgather_mask(&mut self, sh: Sh) -> Res<()> {
        self.vones(sh.wide(), XS0)?;
        if sh.size < 16 {
            self.vu(VUn::Movq, false, XS0, XS0)?;
        }
        Ok(())
    }

    /// `d = [base + i * w]` in the lanes the mask `XS0` selects (the instruction clears it).
    /// `yd` / `yi`: the data / the index is 256 bits wide.
    fn vgather(&mut self, op: VGather, d: Xmm, base: Gpr, i: Xmm, w: u32, yd: bool, yi: bool) -> Res<()> {
        if yd || yi {
            self.vform(true)?;
        }
        let (dr, mr): (VReg, VReg) = if yd { (y(d).into(), y(XS0).into()) } else { (d.into(), XS0.into()) };
        let ir: VReg = if yi { y(i).into() } else { i.into() };
        Ok(self.a.vex_gather(op, dr, Some(base), ir, scale_of(w), 0, mr)?)
    }

    /// A gather of 32- or 64-bit lanes with `vpgather*`, after the bounds check of every index.
    /// The instruction sign-extends its index lanes: 32-bit ones serve only where no index can
    /// reach 2^31 (narrower index lanes, or a constant length up to 2^31); other indices are
    /// zero-extended to 64 bits.
    fn vgather_packed(&mut self, sh: Sh, ish: Sh, mi: Mem, base: Gpr, len: VLen, rv: Value) -> Res<()> {
        let short = ish.w < 4 || (ish.w == 4 && matches!(len, VLen::K(n) if n <= 1 << 31));
        // eight 32-bit lanes through 64-bit indices: two gathers of four
        let split = !short && sh.n > 4;
        let (xi, x) = (self.alloc_x()?, self.alloc_x()?);
        let x2 = if split { Some(self.alloc_x()?) } else { None };
        let d = self.def_mem(rv)?;
        self.vindex_check(mi, ish, len)?;
        self.vload(xi, mi, ish.size)?;
        let wide = sh.wide();
        if short {
            let yi = sh.w == 4 && wide;
            if ish.w < 4 {
                self.vu_x(pmovx(ish.w, 4, false), yi, xi, xi)?;
            }
            self.vgather_mask(sh)?;
            let op = if sh.w == 4 { VGather::Pgatherdd } else { VGather::Pgatherdq };
            self.vgather(op, x, base, xi, sh.w, wide, yi)?;
        } else if let Some(x2) = x2 {
            self.vhigh(XS1, xi)?;
            for (dst, src) in [(x, xi), (x2, XS1)] {
                self.vu_x(VUn::Pmovzxdq, true, xi, src)?;
                self.vones(false, XS0)?;
                self.vgather(VGather::Pgatherqd, dst, base, xi, sh.w, false, true)?;
            }
            self.vform(true)?;
            self.a.vex_bin_i(VBinI::Inserti128, y(x), y(x), x2, 1)?;
        } else {
            let yi = sh.n == 4;
            if ish.w == 4 {
                self.vu_x(VUn::Pmovzxdq, yi, xi, xi)?;
            }
            self.vgather_mask(sh)?;
            let op = if sh.w == 4 { VGather::Pgatherqd } else { VGather::Pgatherqq };
            self.vgather(op, x, base, xi, sh.w, wide, yi)?;
        }
        self.vstore(d, x, sh.size)
    }

    pub(super) fn v_gather(&mut self, s: Operand, idx: Operand, rt: TypeId, rv: Value) -> Res<()> {
        let sh = self.vshape(rt)?;
        let ish = self.voshape(idx)?;
        let mi = self.vmem(idx)?;
        let (base, len) = self.vbase(s)?;
        if self.vex() && sh.w >= 4 {
            return self.vgather_packed(sh, ish, mi, base, len, rv);
        }
        let d = self.def_mem(rv)?;
        self.vindex_check(mi, ish, len)?;
        for j in 0..sh.n {
            self.sized_load(S0, lane(mi, j, ish.w)?, ish.w as u8, false)?;
            self.sized_load(S1, Mem::index(base, S0, scale_of(sh.w), 0), sh.w as u8, false)?;
            self.a.mov_mr(size_of(sh.w), lane(d, j, sh.w)?, S1)?;
        }
        Ok(())
    }

    /// Every index is checked before the first write; lanes are written in order, so the highest
    /// lane of a repeated index wins.
    pub(super) fn v_scatter(&mut self, s: Operand, idx: Operand, v: Operand) -> Res<()> {
        let sh = self.voshape(v)?;
        let ish = self.voshape(idx)?;
        let (mi, mv) = (self.vmem(idx)?, self.vmem(v)?);
        let (base, len) = self.vbase(s)?;
        self.vindex_check(mi, ish, len)?;
        for j in 0..sh.n {
            self.sized_load(S0, lane(mi, j, ish.w)?, ish.w as u8, false)?;
            self.sized_load(S1, lane(mv, j, sh.w)?, sh.w as u8, false)?;
            self.a.mov_mr(size_of(sh.w), Mem::index(base, S0, scale_of(sh.w), 0), S1)?;
        }
        Ok(())
    }
}
