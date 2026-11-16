//! The single isel + register allocation + encoding pass: register file state, memory and copy
//! helpers, prologue/epilogue, blocks, terminators with block-argument copies, calls.

use stave_ir::ir::{Target, ValueDef};
use stave_ir::{Body, Const, Func, Op, Operand, PanicKind, Site, Term, Value};
use stave_obj::{FuncCode, SymbolId};

use super::abi::{self, Loc, Ret};
use super::analysis::{Analysis, NONE, Vk};
use super::support::Codegen;
use super::ty::{LaneKind, Sc, Ty, VTy};
use super::{R, bad, unsupported};
use crate::enc::{
    AddSub, Addr, Asm, Cond, Fp, Label, MemSize, Reg, Shift, VBin, VReg, VecSize, Width, encode_fp8,
};

/// Allocatable GPRs: `x0–x7`, `x9–x14`.
const GPOOL: u32 = 0x0000_7eff;
/// Allocatable FP registers: `v0–v7`, `v16–v29`.
const FPOOL: u32 = 0x3fff_00ff;
pub(crate) const NOREG: u8 = 0xff;
/// Parallel-move scratch: parks one source of a cycle.
const PARK_G: u8 = 16;
const PARK_F: u8 = 30;

pub(crate) const X: Width = Width::X;
pub(crate) const W: Width = Width::W;
pub(crate) const SP: Reg = Reg::SP;
pub(crate) const ZR: Reg = Reg::ZR;
/// Scratch: data.
pub(crate) const S15: Reg = Reg::xn(15);
/// Scratch: copy source, indirect call target.
pub(crate) const S16: Reg = Reg::xn(16);
/// Scratch: address computation, copy destination.
pub(crate) const S17: Reg = Reg::xn(17);
pub(crate) const V31: VReg = VReg::vn(31);

pub(crate) fn xr(n: u8) -> Reg {
    Reg::xn(n)
}

pub(crate) fn kind_code(k: PanicKind) -> u32 {
    match k {
        PanicKind::Bounds => 0,
        PanicKind::Overflow => 1,
        PanicKind::DivByZero => 2,
        PanicKind::Assert => 3,
        PanicKind::OutOfMemory => 4,
        PanicKind::StackOverflow => 5,
        PanicKind::Unwrap => 6,
    }
}

/// Whether `[base, #off]` is encodable for an access of `1 << scale` bytes.
fn fits(off: i64, scale: u32) -> bool {
    let unit = 1i64 << scale;
    (off >= 0 && off % unit == 0 && off / unit < 4096) || (-256..=255).contains(&off)
}

/// A place a parallel move reads or writes: a register or an 8-byte home slot (`sp` offset).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PLoc {
    G(u8),
    F(u8),
    M(i64),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PSrc {
    Loc(PLoc),
    Imm(u64),
    Func(stave_ir::FuncId),
    /// The address `sp + off`.
    Addr(i64),
}

struct PMove {
    dst: PLoc,
    src: PSrc,
    sc: Sc,
}

/// How a conditional branch tests its condition.
#[derive(Clone, Copy)]
enum Test {
    /// A bool in a register.
    Reg(Reg),
    /// The flags of a compare.
    Flags(Cond),
}

pub(crate) struct Gen<'a, 'm> {
    pub cg: &'a Codegen<'m>,
    pub f: &'a Func,
    pub b: &'a Body,
    pub an: &'a Analysis,
    pub a: Asm,
    vreg: &'a mut Vec<u8>,
    gval: [u32; 32],
    fval: [u32; 32],
    gocc: u32,
    focc: u32,
    gdirty: u32,
    fdirty: u32,
    gpin: u32,
    fpin: u32,
    gtmp: u32,
    ftmp: u32,
    gstamp: [u32; 32],
    fstamp: [u32; 32],
    clock: u32,
    labels: Vec<Label>,
    traps: Vec<(u32, u32, Label)>,
    /// RPO index of the block being compiled.
    pub cur: u32,
    /// Its index in the emission order.
    at: usize,
    /// Release: the owned register of a target param that the current instruction's result is
    /// the next value of ([`Self::edge_hint`]); the result is defined there. `NOREG` otherwise.
    hint: u8,
    line_site: u32,
}

impl<'a, 'm> Gen<'a, 'm> {
    pub fn new(
        cg: &'a Codegen<'m>,
        f: &'a Func,
        b: &'a Body,
        an: &'a Analysis,
        vreg: &'a mut Vec<u8>,
        long_branches: bool,
    ) -> Self {
        vreg.clear();
        vreg.resize(b.values.len(), NOREG);
        let mut a = Asm::new();
        a.set_long_branches(long_branches);
        Gen {
            cg,
            f,
            b,
            an,
            a,
            vreg,
            gval: [NONE; 32],
            fval: [NONE; 32],
            gocc: 0,
            focc: 0,
            gdirty: 0,
            fdirty: 0,
            gpin: 0,
            fpin: 0,
            gtmp: 0,
            ftmp: 0,
            gstamp: [0; 32],
            fstamp: [0; 32],
            clock: 0,
            labels: Vec::new(),
            traps: Vec::new(),
            cur: 0,
            at: 0,
            hint: NOREG,
            line_site: NONE,
        }
    }

    // ---------------------------------------------------------------- addressing

    /// `[base, #off]` for an access of `1 << scale` bytes, via `x17` when not encodable.
    fn addr(&mut self, base: Reg, off: i64, scale: u32) -> R<Addr> {
        if fits(off, scale) {
            return Ok(Addr::Off(base, off));
        }
        self.add_imm(S17, base, off)?;
        Ok(Addr::Off(S17, 0))
    }

    /// `rd = rn + imm` (any imm; uses `rd` or `x17` as a temporary for large ones).
    pub fn add_imm(&mut self, rd: Reg, rn: Reg, imm: i64) -> R<()> {
        let (op, mag) = if imm >= 0 { (AddSub::Add, imm as u64) } else { (AddSub::Sub, imm.unsigned_abs()) };
        if mag < 4096 {
            if mag == 0 && rd == rn {
                return Ok(());
            }
            self.a.addsub_imm(op, X, rd, rn, mag)?;
        } else if mag < (1 << 24) {
            self.a.addsub_imm(op, X, rd, rn, mag & !0xfff)?;
            if mag & 0xfff != 0 {
                self.a.addsub_imm(op, X, rd, rd, mag & 0xfff)?;
            }
        } else {
            let t = if rd != rn && !rd.is_sp() { rd } else { S17 };
            self.a.mov_imm(X, t, imm as u64)?;
            self.a.addsub_reg(AddSub::Add, X, rd, rn, t, Shift::Lsl, 0)?;
        }
        Ok(())
    }

    /// Integer load (`signed`: sign-extends to 64 bits). An unencodable offset is formed in `rt`
    /// itself, so a load writes no register but `rt` (unless `rt` is `base` and |off| ≥ 16 MiB,
    /// which uses `x17`); `st`, `ldf` and `stf` form the address in `x17`.
    pub fn ld(&mut self, sz: MemSize, signed: bool, rt: Reg, base: Reg, off: i64) -> R<()> {
        let ad = if fits(off, sz as u32) || rt.is_zr() {
            self.addr(base, off, sz as u32)?
        } else {
            self.add_imm(rt, base, off)?;
            Addr::Off(rt, 0)
        };
        if signed && sz != MemSize::X {
            self.a.ldrs(sz, X, rt, ad)?;
        } else {
            self.a.ldr(sz, rt, ad)?;
        }
        Ok(())
    }

    pub fn st(&mut self, sz: MemSize, rt: Reg, base: Reg, off: i64) -> R<()> {
        let ad = self.addr(base, off, sz as u32)?;
        self.a.str(sz, rt, ad)?;
        Ok(())
    }

    fn vscale(vs: VecSize) -> u32 {
        match vs {
            VecSize::S => 2,
            VecSize::D => 3,
            VecSize::Q => 4,
        }
    }

    pub fn ldf(&mut self, vs: VecSize, vt: VReg, base: Reg, off: i64) -> R<()> {
        let ad = self.addr(base, off, Self::vscale(vs))?;
        self.a.ldr_fp(vs, vt, ad)?;
        Ok(())
    }

    pub fn stf(&mut self, vs: VecSize, vt: VReg, base: Reg, off: i64) -> R<()> {
        let ad = self.addr(base, off, Self::vscale(vs))?;
        self.a.str_fp(vs, vt, ad)?;
        Ok(())
    }

    /// Copies `size` bytes. Bases must not be `x15`–`x17`; uses only scratch registers.
    pub fn memcpy(&mut self, db: Reg, doff: i64, sb: Reg, soff: i64, size: u32) -> R<()> {
        if size == 0 {
            return Ok(());
        }
        let n = i64::from(size);
        let small = n <= 128;
        let in_range = |off: i64| off >= 0 && off + n <= 4095;
        let (db, doff) = if small && in_range(doff) {
            (db, doff)
        } else {
            self.add_imm(S17, db, doff)?;
            (S17, 0)
        };
        let (sb, soff) = if small && in_range(soff) {
            (sb, soff)
        } else {
            self.add_imm(S16, sb, soff)?;
            (S16, 0)
        };
        if small {
            return self.copy_chunks(db, doff, sb, soff, n);
        }
        // Loop over 16-byte chunks with post-increment, then the tail.
        self.a.mov_imm(X, S15, (n / 16) as u64)?;
        let l = self.a.new_label();
        self.a.bind(l)?;
        self.a.ldr_fp(VecSize::Q, V31, Addr::Post(S16, 16))?;
        self.a.str_fp(VecSize::Q, V31, Addr::Post(S17, 16))?;
        self.a.addsub_imm(AddSub::Subs, X, S15, S15, 1)?;
        self.a.b_cond(Cond::Ne, l)?;
        self.copy_chunks(S17, 0, S16, 0, n % 16)
    }

    fn copy_chunks(&mut self, db: Reg, doff: i64, sb: Reg, soff: i64, n: i64) -> R<()> {
        let mut k = 0i64;
        while k < n {
            let rem = n - k;
            let al = ((doff + k) | (soff + k) | 16).trailing_zeros().min(4);
            let mut c = 1i64 << al;
            while c > rem {
                c >>= 1;
            }
            let (d, s) = (Addr::Off(db, doff + k), Addr::Off(sb, soff + k));
            match c {
                16 => {
                    self.a.ldr_fp(VecSize::Q, V31, s)?;
                    self.a.str_fp(VecSize::Q, V31, d)?;
                }
                _ => {
                    let sz = match c {
                        8 => MemSize::X,
                        4 => MemSize::W,
                        2 => MemSize::H,
                        _ => MemSize::B,
                    };
                    self.a.ldr(sz, S15, s)?;
                    self.a.str(sz, S15, d)?;
                }
            }
            k += c;
        }
        Ok(())
    }

    /// Zeroes `size` bytes at `base + off` (base not `x15`–`x17`).
    pub fn memzero(&mut self, base: Reg, off: i64, size: u32) -> R<()> {
        let n = i64::from(size);
        if n == 0 {
            return Ok(());
        }
        let (base, off, n) = if n <= 128 && off >= 0 && off + n <= 4095 {
            (base, off, n)
        } else {
            self.add_imm(S17, base, off)?;
            if n > 128 {
                // 8-byte stores with post-increment, then the tail.
                self.a.mov_imm(X, S15, (n / 8) as u64)?;
                let l = self.a.new_label();
                self.a.bind(l)?;
                self.a.str(MemSize::X, ZR, Addr::Post(S17, 8))?;
                self.a.addsub_imm(AddSub::Subs, X, S15, S15, 1)?;
                self.a.b_cond(Cond::Ne, l)?;
            }
            (S17, 0, if n > 128 { n % 8 } else { n })
        };
        let mut k = 0i64;
        while k < n {
            let rem = n - k;
            let al = ((off + k) | 8).trailing_zeros().min(3);
            let mut c = 1i64 << al;
            while c > rem {
                c >>= 1;
            }
            let sz = match c {
                8 => MemSize::X,
                4 => MemSize::W,
                2 => MemSize::H,
                _ => MemSize::B,
            };
            self.a.str(sz, ZR, Addr::Off(base, off + k))?;
            k += c;
        }
        Ok(())
    }

    // ---------------------------------------------------------------- registers

    pub fn home(&self, v: Value) -> i64 {
        i64::from(self.an.home.get(v.index()).copied().unwrap_or(0))
    }

    pub fn vkind(&self, v: Value) -> Vk {
        self.an.kind.get(v.index()).copied().unwrap_or(Vk::Dead)
    }

    pub fn vty(&self, v: Value) -> Ty {
        self.b.values.get(v.index()).map_or(Ty::Bad("value out of range"), |d| self.cg.ty(d.ty))
    }

    fn touch_g(&mut self, r: u8) {
        self.clock += 1;
        self.gstamp[r as usize] = self.clock;
        self.gpin |= 1 << r;
    }

    fn touch_f(&mut self, r: u8) {
        self.clock += 1;
        self.fstamp[r as usize] = self.clock;
        self.fpin |= 1 << r;
    }

    fn lru(mask: u32, stamps: &[u32; 32]) -> Option<u8> {
        let mut best = None;
        let mut bs = u32::MAX;
        let mut m = mask;
        while m != 0 {
            let r = m.trailing_zeros() as u8;
            m &= m - 1;
            if stamps[r as usize] < bs {
                bs = stamps[r as usize];
                best = Some(r);
            }
        }
        best
    }

    fn store_home(&mut self, v: u32, r: u8, fp: bool) -> R<()> {
        let v = Value(v);
        let off = self.home(v);
        if fp {
            let vs = match self.vkind(v) {
                Vk::Fpr(s) => s.vsize(),
                _ => VecSize::D,
            };
            self.stf(vs, VReg::vn(r), SP, off)
        } else {
            self.st(MemSize::X, xr(r), SP, off)
        }
    }

    fn evict_g(&mut self, r: u8) -> R<()> {
        let v = self.gval[r as usize];
        if v != NONE {
            if self.gdirty & (1 << r) != 0 {
                self.store_home(v, r, false)?;
            }
            self.vreg[v as usize] = NOREG;
        }
        self.gval[r as usize] = NONE;
        self.gocc &= !(1 << r);
        self.gdirty &= !(1 << r);
        Ok(())
    }

    fn evict_f(&mut self, r: u8) -> R<()> {
        let v = self.fval[r as usize];
        if v != NONE {
            if self.fdirty & (1 << r) != 0 {
                self.store_home(v, r, true)?;
            }
            self.vreg[v as usize] = NOREG;
        }
        self.fval[r as usize] = NONE;
        self.focc &= !(1 << r);
        self.fdirty &= !(1 << r);
        Ok(())
    }

    /// A free (or freed) allocatable GPR, pinned for the current instruction.
    pub fn alloc_g(&mut self) -> R<Reg> {
        let free = GPOOL & !self.gocc & !self.gtmp & !self.gpin;
        let r = if free != 0 {
            let pref = free & 0x7e00;
            (if pref != 0 { pref } else { free }).trailing_zeros() as u8
        } else {
            let r = Self::lru(GPOOL & !self.gpin & !self.gtmp, &self.gstamp)
                .ok_or_else(|| unsupported("out of general registers"))?;
            self.evict_g(r)?;
            r
        };
        self.touch_g(r);
        Ok(xr(r))
    }

    pub fn alloc_f(&mut self) -> R<VReg> {
        let free = FPOOL & !self.focc & !self.ftmp & !self.fpin;
        let r = if free != 0 {
            let pref = free & 0x3fff_0000;
            (if pref != 0 { pref } else { free }).trailing_zeros() as u8
        } else {
            let r = Self::lru(FPOOL & !self.fpin & !self.ftmp, &self.fstamp)
                .ok_or_else(|| unsupported("out of FP registers"))?;
            self.evict_f(r)?;
            r
        };
        self.touch_f(r);
        Ok(VReg::vn(r))
    }

    /// A temporary GPR for the current instruction.
    pub fn tmp_g(&mut self) -> R<Reg> {
        let r = self.alloc_g()?;
        self.gtmp |= 1 << r.num();
        Ok(r)
    }

    pub fn tmp_f(&mut self) -> R<VReg> {
        let r = self.alloc_f()?;
        self.ftmp |= 1 << r.num();
        Ok(r)
    }

    fn bind_g(&mut self, v: Value, r: u8, dirty: bool) {
        self.gval[r as usize] = v.0;
        self.vreg[v.index()] = r;
        self.gocc |= 1 << r;
        if dirty {
            self.gdirty |= 1 << r;
        }
    }

    fn bind_f(&mut self, v: Value, r: u8, dirty: bool) {
        self.fval[r as usize] = v.0;
        self.vreg[v.index()] = r;
        self.focc |= 1 << r;
        if dirty {
            self.fdirty |= 1 << r;
        }
    }

    /// The register `v` owns (release builds), if any. An owned register is outside the cache
    /// and holds its value from the definition on.
    pub fn owned(&self, v: Value) -> Option<u8> {
        self.an.own.get(v.index()).copied().filter(|&r| r != NOREG)
    }

    /// The register currently holding `v`, if any.
    pub fn cached(&self, v: Value) -> Option<u8> {
        self.owned(v).or_else(|| self.vreg.get(v.index()).copied().filter(|&r| r != NOREG))
    }

    /// The bits of an int constant operand (normalized two's complement).
    pub fn int_bits(&self, o: Operand) -> Option<u64> {
        self.b.int_const(o).map(|v| v as u64)
    }

    /// `o` in a GPR, pinned for the current instruction.
    pub fn use_g(&mut self, o: Operand) -> R<Reg> {
        match o {
            Operand::Value(v) => {
                if let Some(r) = self.cached(v)
                    && matches!(self.vkind(v), Vk::Gpr(_) | Vk::Slot)
                {
                    self.touch_g(r);
                    return Ok(xr(r));
                }
                match self.vkind(v) {
                    Vk::Gpr(_) => {
                        let r = self.alloc_g()?;
                        let h = self.home(v);
                        self.ld(MemSize::X, false, r, SP, h)?;
                        self.bind_g(v, r.num(), false);
                        Ok(r)
                    }
                    Vk::Slot => {
                        let r = self.alloc_g()?;
                        let h = self.home(v);
                        self.add_imm(r, SP, h)?;
                        self.bind_g(v, r.num(), false);
                        Ok(r)
                    }
                    k => Err(bad(format!("%{} ({k:?}) used as an integer register", v.0))),
                }
            }
            Operand::Const(_) => {
                let bits = self.int_bits(o).ok_or_else(|| bad("float constant in an integer position"))?;
                let r = self.tmp_g()?;
                self.a.mov_imm(X, r, bits)?;
                Ok(r)
            }
            Operand::Func(fid) => {
                let r = self.tmp_g()?;
                let s = self.cg.func_sym(fid)?;
                self.a.addr_of_sym(r, s, 0)?;
                Ok(r)
            }
            Operand::Str(_) | Operand::Type(_) => Err(bad("string or type operand in a scalar position")),
        }
    }

    /// Materializes float bits into `vd` (may use `x15`).
    pub fn mat_f(&mut self, vd: VReg, fp: Fp, bits: u64) -> R<()> {
        let val = if fp == Fp::S { f64::from(f32::from_bits(bits as u32)) } else { f64::from_bits(bits) };
        if bits == 0 {
            self.a.fmov_from_gpr(fp, vd, ZR)?;
        } else if encode_fp8(val).is_some() && !val.is_nan() {
            self.a.fmov_imm(fp, vd, val)?;
        } else {
            let w = if fp == Fp::S { W } else { X };
            self.a.mov_imm(w, S15, bits)?;
            self.a.fmov_from_gpr(fp, vd, S15)?;
        }
        Ok(())
    }

    pub fn float_bits(&self, o: Operand) -> Option<u64> {
        match o {
            Operand::Const(c) => match self.b.konst(c) {
                Some(Const::F32(b)) => Some(u64::from(b)),
                Some(Const::F64(b)) => Some(b),
                _ => None,
            },
            _ => None,
        }
    }

    /// `o` in an FP register, pinned for the current instruction.
    pub fn use_f(&mut self, o: Operand, sc: Sc) -> R<VReg> {
        match o {
            Operand::Value(v) => {
                if let Some(r) = self.cached(v)
                    && matches!(self.vkind(v), Vk::Fpr(_))
                {
                    self.touch_f(r);
                    return Ok(VReg::vn(r));
                }
                match self.vkind(v) {
                    Vk::Fpr(s) => {
                        let r = self.alloc_f()?;
                        let h = self.home(v);
                        self.ldf(s.vsize(), r, SP, h)?;
                        self.bind_f(v, r.num(), false);
                        Ok(r)
                    }
                    k => Err(bad(format!("%{} ({k:?}) used as an FP register", v.0))),
                }
            }
            Operand::Const(_) => {
                let bits = self.float_bits(o).ok_or_else(|| bad("int constant in a float position"))?;
                let r = self.tmp_f()?;
                self.mat_f(r, sc.fp(), bits)?;
                Ok(r)
            }
            _ => Err(bad("non-float operand in a float position")),
        }
    }

    /// The hinted register for result `v`, which then lives there (outside the cache).
    fn take_hint(&mut self, v: Value) -> Option<u8> {
        let r = std::mem::replace(&mut self.hint, NOREG);
        if r == NOREG {
            return None;
        }
        self.vreg[v.index()] = r;
        Some(r)
    }

    pub fn def_g(&mut self, v: Value) -> R<Reg> {
        if let Some(r) = self.owned(v) {
            return Ok(xr(r));
        }
        if let Some(r) = self.take_hint(v) {
            return Ok(xr(r));
        }
        let r = self.alloc_g()?;
        self.bind_g(v, r.num(), true);
        Ok(r)
    }

    pub fn def_f(&mut self, v: Value) -> R<VReg> {
        if let Some(r) = self.owned(v) {
            return Ok(VReg::vn(r));
        }
        if let Some(r) = self.take_hint(v) {
            return Ok(VReg::vn(r));
        }
        let r = self.alloc_f()?;
        self.bind_f(v, r.num(), true);
        Ok(r)
    }

    /// Drops `v`'s register binding without storing it.
    pub fn release(&mut self, v: Value) {
        if self.owned(v).is_some() {
            return;
        }
        let Some(r) = self.cached(v) else { return };
        // a hinted result sits in a param's owned register, which is not the cache's
        let held = match self.vkind(v) {
            Vk::Fpr(_) => self.fval[r as usize],
            _ => self.gval[r as usize],
        };
        if held != v.0 {
            self.vreg[v.index()] = NOREG;
            return;
        }
        let bit = 1u32 << r;
        match self.vkind(v) {
            Vk::Fpr(_) => {
                self.fval[r as usize] = NONE;
                self.focc &= !bit;
                self.fdirty &= !bit;
            }
            _ => {
                self.gval[r as usize] = NONE;
                self.gocc &= !bit;
                self.gdirty &= !bit;
            }
        }
        self.vreg[v.index()] = NOREG;
    }

    /// The temporaries and pins of the current instruction, for [`Self::tmp_reset`].
    pub fn tmp_mark(&self) -> [u32; 4] {
        [self.gtmp, self.ftmp, self.gpin, self.fpin]
    }

    /// Frees the temporaries taken since `mark`. Only between straight-line code: a register
    /// taken after the reset may be one of them.
    pub fn tmp_reset(&mut self, mark: [u32; 4]) {
        [self.gtmp, self.ftmp, self.gpin, self.fpin] = mark;
    }

    /// Ends an instruction: unpins, frees temporaries, drops values at their last use.
    pub fn end_inst(&mut self, i: u32, result: Option<Value>) {
        self.gpin = 0;
        self.fpin = 0;
        self.gtmp = 0;
        self.ftmp = 0;
        let span = self.b.insts.args.get(i as usize).copied().unwrap_or_default();
        for oi in span.indices() {
            if self.an.last_use.get(oi as usize).copied().unwrap_or(false)
                && let Some(Operand::Value(v)) = self.b.operands.get(oi as usize)
                && self.an.end.get(v.index()).copied() == Some(self.cur)
            {
                self.release(*v);
            }
        }
        if let Some(r) = result
            && self.an.unused.get(r.index()).copied().unwrap_or(false)
            && self.an.end.get(r.index()).copied() == Some(self.cur)
        {
            self.release(r);
        }
    }

    /// Stores every dirty register and forgets all bindings (before calls).
    pub fn flush_all(&mut self) -> R<()> {
        let mut m = self.gdirty & self.gocc;
        while m != 0 {
            let r = m.trailing_zeros() as u8;
            m &= m - 1;
            self.store_home(self.gval[r as usize], r, false)?;
        }
        let mut m = self.fdirty & self.focc;
        while m != 0 {
            let r = m.trailing_zeros() as u8;
            m &= m - 1;
            self.store_home(self.fval[r as usize], r, true)?;
        }
        self.forget_all();
        Ok(())
    }

    pub fn forget_all(&mut self) {
        for r in 0..32 {
            let v = self.gval[r];
            if v != NONE {
                self.vreg[v as usize] = NOREG;
                self.gval[r] = NONE;
            }
            let v = self.fval[r];
            if v != NONE {
                self.vreg[v as usize] = NOREG;
                self.fval[r] = NONE;
            }
        }
        self.gocc = 0;
        self.focc = 0;
        self.gdirty = 0;
        self.fdirty = 0;
        self.gpin = 0;
        self.fpin = 0;
        self.gtmp = 0;
        self.ftmp = 0;
    }

    /// Stores dirty values that are live after this block (bindings stay valid).
    fn flush_live_out(&mut self) -> R<()> {
        let cur = self.cur;
        let mut m = self.gdirty & self.gocc;
        while m != 0 {
            let r = m.trailing_zeros() as u8;
            m &= m - 1;
            let v = self.gval[r as usize];
            if self.an.end[v as usize] > cur {
                self.store_home(v, r, false)?;
                self.gdirty &= !(1 << r);
            }
        }
        let mut m = self.fdirty & self.focc;
        while m != 0 {
            let r = m.trailing_zeros() as u8;
            m &= m - 1;
            let v = self.fval[r as usize];
            if self.an.end[v as usize] > cur {
                self.store_home(v, r, true)?;
                self.fdirty &= !(1 << r);
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- traps, lines

    /// The out-of-line stub label for a trap.
    pub fn trap(&mut self, kind: PanicKind, site: Site) -> R<Label> {
        let k = kind_code(kind);
        if let Some(&(_, _, l)) = self.traps.iter().find(|t| t.0 == k && t.1 == site.0) {
            return Ok(l);
        }
        let l = self.a.new_label();
        self.traps.push((k, site.0, l));
        Ok(l)
    }

    pub fn site(&self, i: u32) -> Site {
        self.b.insts.site.get(i as usize).copied().unwrap_or(Site::NONE)
    }

    fn line(&mut self, s: Site) {
        if s.0 == self.line_site || s == Site::NONE {
            return;
        }
        self.line_site = s.0;
        if let Some(Some((file, line, col))) = self.cg.syms.lines.get(s.index()) {
            self.a.line(*file, *line, *col);
        }
    }

    // ---------------------------------------------------------------- copies (scratch only)

    /// Writes operand `o` of type `t` to `base + off`. Changes no register binding (uses `x15`,
    /// `x16`, `x17`, `v30`, `v31` only), so it is safe inside conditional code. `base` must not
    /// be a scratch register.
    pub fn put(&mut self, o: Operand, t: Ty, base: Reg, off: i64) -> R<()> {
        match t {
            Ty::Sc(s) => {
                let sz = s.msize();
                if let Operand::Value(v) = o
                    && let Some(r) = self.cached(v)
                {
                    match self.vkind(v) {
                        Vk::Fpr(s) => return self.stf(s.vsize(), VReg::vn(r), base, off),
                        _ => return self.st(sz, xr(r), base, off),
                    }
                }
                match o {
                    Operand::Value(v) => match self.vkind(v) {
                        Vk::Slot => {
                            let h = self.home(v);
                            self.add_imm(S15, SP, h)?;
                        }
                        Vk::Gpr(_) | Vk::Fpr(_) => {
                            let h = self.home(v);
                            self.ld(MemSize::X, false, S15, SP, h)?;
                        }
                        k => return Err(bad(format!("%{} ({k:?}) as a scalar", v.0))),
                    },
                    Operand::Const(_) => {
                        let bits = self.int_bits(o).or(self.float_bits(o)).ok_or_else(|| bad("constant"))?;
                        if bits == 0 {
                            return self.st(sz, ZR, base, off);
                        }
                        self.a.mov_imm(X, S15, bits)?;
                    }
                    Operand::Func(f) => {
                        let s = self.cg.func_sym(f)?;
                        self.a.addr_of_sym(S15, s, 0)?;
                    }
                    _ => return Err(bad("string or type operand as a scalar")),
                }
                self.st(sz, S15, base, off)
            }
            Ty::Fat | Ty::Vec(_) | Ty::Mem { .. } => match o {
                Operand::Value(v) => {
                    let h = self.home(v);
                    self.memcpy(base, off, SP, h, t.size())
                }
                Operand::Str(s) if t == Ty::Fat => {
                    let (sym, so, len) = self.str_ref(s)?;
                    self.a.addr_of_sym(S15, sym, so)?;
                    self.st(MemSize::X, S15, base, off)?;
                    self.a.mov_imm(X, S15, len)?;
                    self.st(MemSize::X, S15, base, off + 8)
                }
                Operand::Func(_) if t == Ty::Fat => {
                    // a fn item as a fn value: (code, null environment)
                    self.put_code(o, base, off)?;
                    self.st(MemSize::X, ZR, base, off + 8)
                }
                _ => Err(bad("aggregate operand must be a value")),
            },
            Ty::Bad(why) => Err(unsupported(why)),
        }
    }

    /// Stores the code address of fn item `o` (one word).
    pub fn put_code(&mut self, o: Operand, base: Reg, off: i64) -> R<()> {
        let Operand::Func(f) = o else { return Err(bad("fn.make code must be a fn item")) };
        let s = self.cg.func_sym(f)?;
        self.a.addr_of_sym(S15, s, 0)?;
        self.st(MemSize::X, S15, base, off)
    }

    pub fn str_ref(&self, s: stave_ir::StrId) -> R<(SymbolId, i64, u64)> {
        let off = self.cg.syms.str_off.get(s.index()).ok_or_else(|| bad("string id out of range"))?;
        let len = self.cg.m.strings.get(s.index()).map_or(0, |b| b.len() as u64);
        Ok((self.cg.syms.strings, i64::from(*off), len))
    }

    /// Loads a value of type `t` at `base + off` into the new value `v`.
    pub fn take(&mut self, v: Value, t: Ty, base: Reg, off: i64) -> R<()> {
        match t {
            Ty::Sc(s) if s.is_fp() => {
                let r = self.def_f(v)?;
                self.ldf(s.vsize(), r, base, off)
            }
            Ty::Sc(s) => {
                let r = self.def_g(v)?;
                self.ld(s.msize(), s.signed(), r, base, off)?;
                if s == Sc::Bool {
                    self.a.cmp_imm(W, r, 0)?;
                    self.a.cset(W, r, Cond::Ne)?;
                }
                Ok(())
            }
            Ty::Fat | Ty::Mem { .. } => {
                let h = self.home(v);
                self.memcpy(SP, h, base, off, t.size())
            }
            Ty::Vec(vt) => {
                let h = self.home(v);
                self.memcpy(SP, h, base, off, t.size())?;
                if vt.kind == LaneKind::Mask {
                    // a mask lane in memory is set iff any of its bits is
                    self.ldf(vt.home_size(), V31, SP, h)?;
                    self.a.vbin(VBin::Cmtst, vt.arr(), V31, V31, V31)?;
                    self.stf(vt.home_size(), V31, SP, h)?;
                }
                Ok(())
            }
            Ty::Bad(why) => Err(unsupported(why)),
        }
    }

    /// Loads vector operand `o` from its home into `vd`.
    pub fn load_vec(&mut self, vd: VReg, o: Operand, vt: VTy) -> R<()> {
        let Operand::Value(v) = o else { return Err(bad("vector operand must be a value")) };
        let h = self.home(v);
        self.ldf(vt.home_size(), vd, SP, h)
    }

    // ---------------------------------------------------------------- function

    pub fn run(mut self) -> R<FuncCode> {
        let nb = self.b.blocks.len();
        self.labels.clear();
        for _ in 0..nb {
            let l = self.a.new_label();
            self.labels.push(l);
        }
        self.prologue()?;
        for at in 0..self.an.emit.len() {
            let blk = self.an.emit[at] as usize;
            self.at = at;
            self.cur = self.an.rpo_of[blk];
            self.forget_all();
            let l = self.labels[blk];
            self.a.bind(l)?;
            let bd = self.b.blocks[blk];
            let fused = self.an.fused.get(blk).copied().unwrap_or(NONE);
            for k in bd.insts.indices() {
                let s = self.site(k);
                self.line(s);
                if k != fused {
                    if !self.an.own.is_empty() {
                        self.hint = self.edge_hint(blk, k);
                    }
                    self.inst(k)?;
                    self.hint = NOREG;
                }
                let res = self.b.insts.result.get(k as usize).copied().flatten();
                self.end_inst(k, res);
            }
            self.term(blk)?;
        }
        let traps = std::mem::take(&mut self.traps);
        for (k, site, l) in traps {
            self.a.bind(l)?;
            match self.cg.syms.trap {
                Some(t) => {
                    self.a.mov_imm(W, xr(0), u64::from(k))?;
                    self.a.mov_imm(W, xr(1), u64::from(site))?;
                    self.a.bl_sym(t);
                    self.a.brk(0);
                }
                None => self.a.brk(0x100 + k as u16),
            }
        }
        Ok(self.a.finish()?)
    }

    fn prologue(&mut self) -> R<()> {
        self.a.stp(X, Reg::FP, Reg::LR, Addr::Pre(SP, -16))?;
        self.a.mov(X, Reg::FP, SP)?;
        let frame = i64::from(self.an.frame);
        if frame > 4096 {
            let pages = frame / 4096;
            if pages <= 16 {
                for _ in 0..pages {
                    self.a.addsub_imm(AddSub::Sub, X, SP, SP, 4096)?;
                    self.a.str(MemSize::X, ZR, Addr::Off(SP, 0))?;
                }
            } else {
                self.a.mov_imm(X, S16, pages as u64)?;
                let l = self.a.new_label();
                self.a.bind(l)?;
                self.a.addsub_imm(AddSub::Sub, X, SP, SP, 4096)?;
                self.a.str(MemSize::X, ZR, Addr::Off(SP, 0))?;
                self.a.addsub_imm(AddSub::Subs, X, S16, S16, 1)?;
                self.a.b_cond(Cond::Ne, l)?;
            }
            let rest = (frame % 4096) as u64;
            if rest != 0 {
                self.a.addsub_imm(AddSub::Sub, X, SP, SP, rest)?;
            }
        } else if frame > 0 {
            self.a.addsub_imm(AddSub::Sub, X, SP, SP, frame as u64)?;
        }
        if matches!(self.f.ret.map(|t| self.cg.ty(t)), Some(Ty::Mem { .. })) {
            let off = i64::from(self.an.x8_off);
            self.st(MemSize::X, xr(8), SP, off)?;
        }
        for k in 0..self.an.saved.len() {
            let (fp, r) = self.an.saved[k];
            let off = -8 * (k as i64 + 1);
            if fp {
                self.stf(VecSize::D, VReg::vn(r), Reg::FP, off)?;
            } else {
                self.st(MemSize::X, xr(r), Reg::FP, off)?;
            }
        }
        let params: Vec<Value> = self.b.params(stave_ir::Block(0)).to_vec();
        let mut args = Vec::new();
        abi::classify(params.iter().map(|&p| self.vty(p)), &mut args);
        // Register arguments first (stack copies below use x9 and the scratch registers).
        for (k, &p) in params.iter().enumerate() {
            let t = self.vty(p);
            let h = self.home(p);
            if self.vkind(p) == Vk::Dead {
                continue;
            }
            if let Some(r) = self.owned(p) {
                // owned registers are callee-saved: none is an argument register
                match (args[k].loc, t) {
                    (Loc::X(n), _) => self.a.mov(X, xr(r), xr(n))?,
                    (Loc::V(n), Ty::Sc(s)) => self.a.fmov(s.fp(), VReg::vn(r), VReg::vn(n)),
                    _ => {}
                }
                continue;
            }
            match (args[k].loc, t) {
                (Loc::X(n), Ty::Mem { size, .. }) => self.memcpy(SP, h, xr(n), 0, size)?,
                (Loc::X(n), _) => self.st(MemSize::X, xr(n), SP, h)?,
                (Loc::V(n), Ty::Sc(s)) => self.stf(s.vsize(), VReg::vn(n), SP, h)?,
                (Loc::V(n), Ty::Vec(vt)) => self.stf(vt.home_size(), VReg::vn(n), SP, h)?,
                (Loc::XPair(n), _) => {
                    self.st(MemSize::X, xr(n), SP, h)?;
                    self.st(MemSize::X, xr(n + 1), SP, h + 8)?;
                }
                _ => {}
            }
        }
        for (k, &p) in params.iter().enumerate() {
            let t = self.vty(p);
            let h = self.home(p);
            if self.vkind(p) == Vk::Dead {
                continue;
            }
            if let Loc::Stack(so) = args[k].loc {
                let src = 16 + i64::from(so);
                if let Some(r) = self.owned(p) {
                    match t {
                        Ty::Sc(s) if s.is_fp() => self.ldf(s.vsize(), VReg::vn(r), Reg::FP, src)?,
                        _ => self.ld(MemSize::X, false, xr(r), Reg::FP, src)?,
                    }
                    continue;
                }
                match t {
                    Ty::Mem { size, .. } => {
                        let x9 = xr(9);
                        self.ld(MemSize::X, false, x9, Reg::FP, src)?;
                        self.memcpy(SP, h, x9, 0, size)?;
                    }
                    Ty::Fat => self.memcpy(SP, h, Reg::FP, src, 16)?,
                    Ty::Vec(vt) => self.memcpy(SP, h, Reg::FP, src, u32::from(vt.size))?,
                    _ => {
                        self.ld(MemSize::X, false, S15, Reg::FP, src)?;
                        self.st(MemSize::X, S15, SP, h)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn epilogue(&mut self) -> R<()> {
        for k in 0..self.an.saved.len() {
            let (fp, r) = self.an.saved[k];
            let off = -8 * (k as i64 + 1);
            if fp {
                self.ldf(VecSize::D, VReg::vn(r), Reg::FP, off)?;
            } else {
                self.ld(MemSize::X, false, xr(r), Reg::FP, off)?;
            }
        }
        self.a.mov(X, SP, Reg::FP)?;
        self.a.ldp(X, Reg::FP, Reg::LR, Addr::Post(SP, 16))?;
        self.a.ret();
        Ok(())
    }

    /// The owned register of a param of the block's only successor, when the result of
    /// instruction `k` is nothing but that param's next value (a loop counter's increment): the
    /// op then defines its result in the param's register and the branch has nothing to move.
    ///
    /// Writing the register early is safe when the param's old value is dead: the block has one
    /// successor (which redefines the param), nothing after `k` reads the param, and the block
    /// lies in the param's interval, so no other value lives in that register here. Only ops that
    /// are one instruction reading all operands qualify.
    fn edge_hint(&self, blk: usize, k: u32) -> u8 {
        let b = self.b;
        let bd = &b.blocks[blk];
        let Some(v) = b.insts.result[k as usize] else { return NOREG };
        if self.an.uses.get(v.index()).copied() != Some(1) || self.owned(v).is_some() {
            return NOREG;
        }
        let Term::Br(t) = bd.term else { return NOREG };
        let me = Operand::Value(v);
        let Some(j) = b.target_args(&t).iter().position(|o| *o == me) else { return NOREG };
        let Some(&p) = b.params(t.block).get(j) else { return NOREG };
        let Some(rp) = self.owned(p) else { return NOREG };
        let at = self.at as u32;
        let (first, last) = (self.an.own_first.get(p.index()), self.an.own_last.get(p.index()));
        if !matches!((first, last), (Some(&f), Some(&l)) if f <= at && at <= l) {
            return NOREG;
        }
        let pv = Operand::Value(p);
        let end = bd.insts.start + bd.insts.len;
        if (k + 1..end).any(|i| b.args(i).contains(&pv)) || b.target_args(&t).contains(&pv) {
            return NOREG;
        }
        let float = b.insts.ty[k as usize].is_some_and(|ty| matches!(self.cg.ty(ty), Ty::Sc(s) if s.is_fp()));
        let simple = match b.insts.op[k as usize] {
            Op::AddWrap | Op::SubWrap | Op::MulWrap | Op::And | Op::Or | Op::Xor | Op::Not | Op::Neg => true,
            Op::Add | Op::Sub => true,
            Op::Mul | Op::Div => float,
            _ => false,
        };
        if simple { rp } else { NOREG }
    }

    /// Whether the edge to `t` moves anything. Release: an argument already in the param's
    /// owned register moves nothing.
    fn has_copies(&self, t: &Target) -> bool {
        if self.an.own.is_empty() {
            return t.args.len > 0;
        }
        let (params, args) = (self.b.params(t.block), self.b.target_args(t));
        params.len() != args.len()
            || params.iter().zip(args).any(|(&p, &o)| {
                if o == Operand::Value(p) || self.vkind(p) == Vk::Dead {
                    return false;
                }
                match (self.owned(p), self.vkind(p), self.psrc(o)) {
                    (Some(r), Vk::Fpr(_), Ok(src)) => src != PSrc::Loc(PLoc::F(r)),
                    (Some(r), Vk::Gpr(_), Ok(src)) => src != PSrc::Loc(PLoc::G(r)),
                    _ => true,
                }
            })
    }

    fn branch_if(&mut self, t: Test, l: Label) -> R<()> {
        match t {
            Test::Reg(r) => self.a.cbnz(W, r, l)?,
            Test::Flags(c) => self.a.b_cond(c, l)?,
        }
        Ok(())
    }

    fn branch_unless(&mut self, t: Test, l: Label) -> R<()> {
        match t {
            Test::Reg(r) => self.a.cbz(W, r, l)?,
            Test::Flags(c) => self.a.b_cond(c.negate(), l)?,
        }
        Ok(())
    }

    fn next_block(&self) -> Option<u32> {
        self.an.emit.get(self.at + 1).copied()
    }

    fn jump(&mut self, blk: stave_ir::Block) -> R<()> {
        if self.next_block() != Some(blk.0) {
            let l = self.labels[blk.index()];
            self.a.b(l)?;
        }
        Ok(())
    }

    fn term(&mut self, blk: usize) -> R<()> {
        let bd = self.b.blocks[blk];
        match bd.term {
            Term::Br(t) => {
                self.flush_live_out()?;
                self.edge(&t)?;
                self.jump(t.block)
            }
            Term::Cbr { cond, then, els } => {
                if let Some(c) = self.b.int_const(cond) {
                    let t = if c != 0 { then } else { els };
                    self.flush_live_out()?;
                    self.edge(&t)?;
                    return self.jump(t.block);
                }
                let fused = self.an.fused.get(blk).copied().unwrap_or(NONE);
                let test = if fused == NONE {
                    let rc = self.use_g(cond)?;
                    self.flush_live_out()?;
                    Test::Reg(rc)
                } else {
                    // stores and loads keep the flags, so the compare comes last
                    self.flush_live_out()?;
                    Test::Flags(self.compare_flags(fused)?)
                };
                let (ta, ea) = (self.has_copies(&then), self.has_copies(&els));
                let (lt, le) = (self.labels[then.block.index()], self.labels[els.block.index()]);
                if !ta && !ea {
                    if self.next_block() == Some(els.block.0) {
                        self.branch_if(test, lt)?;
                    } else if self.next_block() == Some(then.block.0) {
                        self.branch_unless(test, le)?;
                    } else {
                        self.branch_if(test, lt)?;
                        self.a.b(le)?;
                    }
                    return Ok(());
                }
                if !ta {
                    self.branch_if(test, lt)?;
                    self.edge(&els)?;
                    return self.jump(els.block);
                }
                let l = self.a.new_label();
                self.branch_unless(test, l)?;
                self.edge(&then)?;
                self.a.b(lt)?;
                self.a.bind(l)?;
                self.edge(&els)?;
                self.jump(els.block)
            }
            Term::Switch { scrut, cases, default } => {
                let rs = self.use_g(scrut)?;
                self.flush_live_out()?;
                let mut pending = Vec::new();
                for c in self.b.switch_cases(cases) {
                    let v = c.value;
                    if (0..4096).contains(&v) {
                        self.a.cmp_imm(X, rs, v as u64)?;
                    } else if (-4095..0).contains(&v) {
                        self.a.cmn_imm(X, rs, (-v) as u64)?;
                    } else {
                        self.a.mov_imm(X, S15, v as u64)?;
                        self.a.cmp(X, rs, S15)?;
                    }
                    if c.target.args.len == 0 {
                        let l = self.labels[c.target.block.index()];
                        self.a.b_cond(Cond::Eq, l)?;
                    } else {
                        let l = self.a.new_label();
                        self.a.b_cond(Cond::Eq, l)?;
                        pending.push((l, c.target));
                    }
                }
                self.edge(&default)?;
                if pending.is_empty() {
                    return self.jump(default.block);
                }
                let ld = self.labels[default.block.index()];
                self.a.b(ld)?;
                for (l, t) in pending {
                    self.a.bind(l)?;
                    self.edge(&t)?;
                    let lt = self.labels[t.block.index()];
                    self.a.b(lt)?;
                }
                Ok(())
            }
            Term::Ret(o) => {
                self.ret(o)?;
                self.epilogue()
            }
            Term::Unreachable => {
                self.a.brk(1);
                Ok(())
            }
        }
    }

    fn ret(&mut self, o: Option<Operand>) -> R<()> {
        let rt = self.f.ret.map(|t| self.cg.ty(t));
        let Some(o) = o else { return Ok(()) };
        match abi::ret(rt) {
            Ret::None => Ok(()),
            Ret::X0 => self.load_to_x(xr(0), o),
            Ret::V0 => {
                let s = match rt {
                    Some(Ty::Sc(s)) => s,
                    _ => Sc::F64,
                };
                self.load_to_v(VReg::vn(0), o, s)
            }
            Ret::X0X1 => self.load_fat(xr(0), xr(1), o),
            Ret::Q0 => match rt {
                Some(Ty::Vec(vt)) => self.load_vec(VReg::vn(0), o, vt),
                _ => Err(bad("vector return type")),
            },
            Ret::Indirect => {
                let Operand::Value(v) = o else { return Err(bad("aggregate return of a non-value")) };
                let size = rt.map_or(0, |t| t.size());
                let x9 = xr(9);
                let off = i64::from(self.an.x8_off);
                self.ld(MemSize::X, false, x9, SP, off)?;
                let h = self.home(v);
                self.memcpy(x9, 0, SP, h, size)
            }
        }
    }

    /// Block-argument copies for one edge, as a parallel copy: arguments that read a param of
    /// the target (at another index) from memory are saved to the scratch area first.
    fn edge(&mut self, t: &Target) -> R<()> {
        let args = self.b.target_args(t);
        if args.is_empty() {
            return Ok(());
        }
        let params = self.b.params(t.block);
        // Release: scalars are one parallel move (below); only aggregates are saved here.
        let release = !self.an.own.is_empty();
        let mut moves: Vec<PMove> = Vec::new();
        let mut saved: Vec<(usize, i64)> = Vec::new();
        let mut soff = i64::from(self.an.scratch_off);
        for (i, &o) in args.iter().enumerate() {
            if let Operand::Value(v) = o
                && let Some(vd) = self.b.values.get(v.index())
                && matches!(vd.def, ValueDef::Param { block, index } if block == t.block && index as usize != i)
                && (self.cached(v).is_none()
                    || self.vkind(v) == Vk::Slot
                    || matches!(self.vkind(v), Vk::Mem(_)))
                && (!release || matches!(self.vkind(v), Vk::Mem(_)))
            {
                let ty = self.vty(v);
                let size = ty.size().max(8).div_ceil(8) * 8;
                match ty {
                    Ty::Fat | Ty::Vec(_) | Ty::Mem { .. } => {
                        let h = self.home(v);
                        self.memcpy(SP, soff, SP, h, ty.size())?;
                    }
                    _ => {
                        let h = self.home(v);
                        self.ld(MemSize::X, false, S15, SP, h)?;
                        self.st(MemSize::X, S15, SP, soff)?;
                    }
                }
                saved.push((i, soff));
                soff += i64::from(size);
            }
        }
        for (i, &o) in args.iter().enumerate() {
            let Some(&p) = params.get(i) else { return Err(bad("branch argument count")) };
            if o == Operand::Value(p) || self.vkind(p) == Vk::Dead {
                continue;
            }
            let ty = self.vty(p);
            let h = self.home(p);
            if release && let Ty::Sc(sc) = ty {
                let dst = match self.owned(p) {
                    Some(r) if sc.is_fp() => PLoc::F(r),
                    Some(r) => PLoc::G(r),
                    None => PLoc::M(h),
                };
                moves.push(PMove { dst, src: self.psrc(o)?, sc });
                continue;
            }
            if let Some(&(_, so)) = saved.iter().find(|s| s.0 == i) {
                match ty {
                    Ty::Fat | Ty::Vec(_) | Ty::Mem { .. } => self.memcpy(SP, h, SP, so, ty.size())?,
                    _ => {
                        self.ld(MemSize::X, false, S15, SP, so)?;
                        self.st(MemSize::X, S15, SP, h)?;
                    }
                }
            } else {
                match ty {
                    // Scalar homes are written as whole words.
                    Ty::Sc(s) if !s.is_fp() => self.put(o, Ty::Sc(Sc::USIZE), SP, h)?,
                    Ty::Sc(s) => self.put(o, Ty::Sc(if s == Sc::F32 { Sc::F32 } else { Sc::F64 }), SP, h)?,
                    _ => self.put(o, ty, SP, h)?,
                }
            }
        }
        self.par_moves(moves)
    }

    /// Where a scalar operand is right now (no binding change).
    fn psrc(&self, o: Operand) -> R<PSrc> {
        match o {
            Operand::Value(v) => {
                let k = self.vkind(v);
                let h = self.home(v);
                match (self.cached(v), k) {
                    (Some(r), Vk::Fpr(_)) => Ok(PSrc::Loc(PLoc::F(r))),
                    (Some(r), Vk::Gpr(_) | Vk::Slot) => Ok(PSrc::Loc(PLoc::G(r))),
                    (None, Vk::Slot) => Ok(PSrc::Addr(h)),
                    (None, Vk::Gpr(_) | Vk::Fpr(_)) => Ok(PSrc::Loc(PLoc::M(h))),
                    _ => Err(bad(format!("%{} ({k:?}) as a scalar", v.0))),
                }
            }
            Operand::Const(_) => {
                Ok(PSrc::Imm(self.int_bits(o).or(self.float_bits(o)).ok_or_else(|| bad("constant"))?))
            }
            Operand::Func(f) => Ok(PSrc::Func(f)),
            _ => Err(bad("string or type operand as a scalar")),
        }
    }

    /// Emits the moves as one parallel move: every source is read before its place is written.
    /// A cycle is broken by parking one source in `x16` / `v30`; memory-to-memory moves and
    /// constants go through `x15`, unencodable addresses through `x17`.
    fn par_moves(&mut self, mut ms: Vec<PMove>) -> R<()> {
        ms.retain(|m| m.src != PSrc::Loc(m.dst));
        while !ms.is_empty() {
            let read = |ms: &[PMove], l: PLoc, but: usize| {
                ms.iter().enumerate().any(|(j, o)| j != but && o.src == PSrc::Loc(l))
            };
            if let Some(k) = (0..ms.len()).find(|&k| !read(&ms, ms[k].dst, k)) {
                let m = ms.swap_remove(k);
                self.emit_move(m.dst, m.src, m.sc)?;
                continue;
            }
            // Only cycles are left: every destination is still read. Park one of them.
            let l = ms[0].dst;
            let sc = ms[0].sc;
            let tmp = if sc.is_fp() { PLoc::F(PARK_F) } else { PLoc::G(PARK_G) };
            self.emit_move(tmp, PSrc::Loc(l), sc)?;
            for o in ms.iter_mut() {
                if o.src == PSrc::Loc(l) {
                    o.src = PSrc::Loc(tmp);
                }
            }
        }
        Ok(())
    }

    fn emit_move(&mut self, dst: PLoc, src: PSrc, sc: Sc) -> R<()> {
        let fsize = if sc.is_fp() { sc.vsize() } else { VecSize::D };
        match (dst, src) {
            (PLoc::G(d), PSrc::Loc(PLoc::G(s))) => self.a.mov(X, xr(d), xr(s))?,
            (PLoc::G(d), PSrc::Loc(PLoc::M(o))) => self.ld(MemSize::X, false, xr(d), SP, o)?,
            (PLoc::G(d), PSrc::Imm(b)) => self.a.mov_imm(X, xr(d), b)?,
            (PLoc::G(d), PSrc::Func(f)) => {
                let s = self.cg.func_sym(f)?;
                self.a.addr_of_sym(xr(d), s, 0)?;
            }
            (PLoc::G(d), PSrc::Addr(o)) => self.add_imm(xr(d), SP, o)?,
            (PLoc::F(d), PSrc::Loc(PLoc::F(s))) => self.a.fmov(sc.fp(), VReg::vn(d), VReg::vn(s)),
            (PLoc::F(d), PSrc::Loc(PLoc::M(o))) => self.ldf(fsize, VReg::vn(d), SP, o)?,
            (PLoc::F(d), PSrc::Imm(b)) => self.mat_f(VReg::vn(d), sc.fp(), b)?,
            (PLoc::M(o), PSrc::Loc(PLoc::G(s))) => self.st(MemSize::X, xr(s), SP, o)?,
            (PLoc::M(o), PSrc::Loc(PLoc::F(s))) => self.stf(fsize, VReg::vn(s), SP, o)?,
            (PLoc::M(o), PSrc::Loc(PLoc::M(o2))) => {
                self.ld(MemSize::X, false, S15, SP, o2)?;
                self.st(MemSize::X, S15, SP, o)?;
            }
            (PLoc::M(o), PSrc::Imm(0)) => self.st(MemSize::X, ZR, SP, o)?,
            (PLoc::M(o), PSrc::Imm(b)) => {
                self.a.mov_imm(X, S15, b)?;
                self.st(MemSize::X, S15, SP, o)?;
            }
            (PLoc::M(o), PSrc::Func(f)) => {
                let s = self.cg.func_sym(f)?;
                self.a.addr_of_sym(S15, s, 0)?;
                self.st(MemSize::X, S15, SP, o)?;
            }
            (PLoc::M(o), PSrc::Addr(a)) => {
                self.add_imm(S15, SP, a)?;
                self.st(MemSize::X, S15, SP, o)?;
            }
            _ => return Err(bad("parallel move across register classes")),
        }
        Ok(())
    }

    // ---------------------------------------------------------------- calls

    /// Loads a scalar operand into a specific GPR (all bindings flushed, or `o` cached).
    pub fn load_to_x(&mut self, rd: Reg, o: Operand) -> R<()> {
        match o {
            Operand::Value(v) => {
                if let Some(r) = self.cached(v)
                    && !matches!(self.vkind(v), Vk::Fpr(_))
                {
                    if r != rd.num() {
                        self.a.mov(X, rd, xr(r))?;
                    }
                    return Ok(());
                }
                let h = self.home(v);
                match self.vkind(v) {
                    Vk::Gpr(_) => self.ld(MemSize::X, false, rd, SP, h),
                    Vk::Slot => self.add_imm(rd, SP, h),
                    Vk::Mem(_) => self.add_imm(rd, SP, h),
                    k => Err(bad(format!("%{} ({k:?}) into a GPR", v.0))),
                }
            }
            Operand::Const(_) => {
                let bits = self.int_bits(o).or(self.float_bits(o)).ok_or_else(|| bad("constant"))?;
                Ok(self.a.mov_imm(X, rd, bits)?)
            }
            Operand::Func(f) => {
                let s = self.cg.func_sym(f)?;
                Ok(self.a.addr_of_sym(rd, s, 0)?)
            }
            _ => Err(bad("string or type operand into a GPR")),
        }
    }

    pub fn load_to_v(&mut self, vd: VReg, o: Operand, s: Sc) -> R<()> {
        match o {
            Operand::Value(v) => {
                if let Some(r) = self.cached(v)
                    && matches!(self.vkind(v), Vk::Fpr(_))
                {
                    if r != vd.num() {
                        self.a.fmov(s.fp(), vd, VReg::vn(r));
                    }
                    return Ok(());
                }
                let h = self.home(v);
                self.ldf(s.vsize(), vd, SP, h)
            }
            Operand::Const(_) => {
                let bits = self.float_bits(o).ok_or_else(|| bad("float constant"))?;
                self.mat_f(vd, s.fp(), bits)
            }
            _ => Err(bad("non-float operand into an FP register")),
        }
    }

    pub fn load_fat(&mut self, r0: Reg, r1: Reg, o: Operand) -> R<()> {
        match o {
            Operand::Value(v) => {
                let h = self.home(v);
                self.ld(MemSize::X, false, r0, SP, h)?;
                self.ld(MemSize::X, false, r1, SP, h + 8)
            }
            Operand::Str(s) => {
                let (sym, so, len) = self.str_ref(s)?;
                self.a.addr_of_sym(r0, sym, so)?;
                Ok(self.a.mov_imm(X, r1, len)?)
            }
            Operand::Func(f) => {
                let s = self.cg.func_sym(f)?;
                self.a.addr_of_sym(r0, s, 0)?;
                Ok(self.a.mov(X, r1, ZR)?)
            }
            _ => Err(bad("fat pointer operand")),
        }
    }

    /// A Stave call: `callee` = symbol, or an operand holding the fn value.
    pub fn call(
        &mut self,
        callee: Result<SymbolId, Operand>,
        params: &[stave_ir::TypeId],
        ret: Option<stave_ir::TypeId>,
        args: &[Operand],
        res: Option<Value>,
    ) -> R<()> {
        self.flush_all()?;
        if params.len() != args.len() {
            return Err(bad("call argument count"));
        }
        let mut locs = Vec::with_capacity(params.len());
        abi::classify(params.iter().map(|t| self.cg.ty(*t)), &mut locs);
        // Stack arguments first (they use x15–x17), then registers.
        for (k, (&pt, &o)) in params.iter().zip(args).enumerate() {
            let Loc::Stack(so) = locs[k].loc else { continue };
            let so = i64::from(so);
            match self.cg.ty(pt) {
                Ty::Fat => self.put(o, Ty::Fat, SP, so)?,
                t @ Ty::Vec(_) => self.put(o, t, SP, so)?,
                Ty::Mem { .. } => {
                    self.load_to_x(S15, o)?;
                    self.st(MemSize::X, S15, SP, so)?;
                }
                Ty::Sc(s) if s.is_fp() => {
                    self.put(o, Ty::Sc(if s == Sc::F32 { Sc::F32 } else { Sc::F64 }), SP, so)?;
                }
                Ty::Sc(_) => self.put(o, Ty::Sc(Sc::USIZE), SP, so)?,
                Ty::Bad(why) => return Err(unsupported(why)),
            }
        }
        for (k, (&pt, &o)) in params.iter().zip(args).enumerate() {
            match locs[k].loc {
                Loc::X(n) => self.load_to_x(xr(n), o)?,
                Loc::V(n) => match self.cg.ty(pt) {
                    Ty::Vec(vt) => self.load_vec(VReg::vn(n), o, vt)?,
                    Ty::Sc(s) => self.load_to_v(VReg::vn(n), o, s)?,
                    _ => return Err(bad("argument in an FP register")),
                },
                Loc::XPair(n) => self.load_fat(xr(n), xr(n + 1), o)?,
                Loc::Stack(_) => {}
            }
        }
        let rt = ret.map(|t| self.cg.ty(t));
        let rk = abi::ret(rt);
        if rk == Ret::Indirect {
            let Some(r) = res else { return Err(bad("aggregate call result without a value")) };
            let h = self.home(r);
            self.add_imm(xr(8), SP, h)?;
        }
        match callee {
            Ok(sym) => self.a.bl_sym(sym),
            Err(o) => {
                self.load_to_x(S16, o)?;
                self.a.blr(S16)?;
            }
        }
        self.result_regs(rk, res)
    }

    /// A fn-value call (`call.ind`): the arguments, then the environment word of `callee` as one
    /// more integer argument (a closure's last param; a fn item ignores it), then `blr` through
    /// the code word. `callee` = a fn item's symbol (null environment) or the fn value operand.
    pub fn call_env(
        &mut self,
        callee: Result<SymbolId, Operand>,
        params: &[stave_ir::TypeId],
        ret: Option<stave_ir::TypeId>,
        args: &[Operand],
        res: Option<Value>,
    ) -> R<()> {
        self.flush_all()?;
        if params.len() != args.len() {
            return Err(bad("call argument count"));
        }
        let mut locs = Vec::with_capacity(params.len() + 1);
        abi::classify(params.iter().map(|t| self.cg.ty(*t)).chain([Ty::Sc(Sc::USIZE)]), &mut locs);
        let env_loc = locs[params.len()].loc;
        // the environment word: from the fn value's home (+8), or 0 for a fn item
        let env_home = match callee {
            Err(Operand::Value(v)) => Some(self.home(v) + 8),
            Err(Operand::Func(_)) | Ok(_) => None,
            Err(_) => return Err(bad("fn value operand")),
        };
        if let Loc::Stack(so) = env_loc {
            match env_home {
                Some(h) => self.ld(MemSize::X, false, S15, SP, h)?,
                None => self.a.mov(X, S15, ZR)?,
            }
            self.st(MemSize::X, S15, SP, i64::from(so))?;
        }
        for (k, (&pt, &o)) in params.iter().zip(args).enumerate() {
            let Loc::Stack(so) = locs[k].loc else { continue };
            let so = i64::from(so);
            match self.cg.ty(pt) {
                Ty::Fat => self.put(o, Ty::Fat, SP, so)?,
                t @ Ty::Vec(_) => self.put(o, t, SP, so)?,
                Ty::Mem { .. } => {
                    self.load_to_x(S15, o)?;
                    self.st(MemSize::X, S15, SP, so)?;
                }
                Ty::Sc(s) if s.is_fp() => {
                    self.put(o, Ty::Sc(if s == Sc::F32 { Sc::F32 } else { Sc::F64 }), SP, so)?;
                }
                Ty::Sc(_) => self.put(o, Ty::Sc(Sc::USIZE), SP, so)?,
                Ty::Bad(why) => return Err(unsupported(why)),
            }
        }
        for (k, (&pt, &o)) in params.iter().zip(args).enumerate() {
            match locs[k].loc {
                Loc::X(n) => self.load_to_x(xr(n), o)?,
                Loc::V(n) => match self.cg.ty(pt) {
                    Ty::Vec(vt) => self.load_vec(VReg::vn(n), o, vt)?,
                    Ty::Sc(s) => self.load_to_v(VReg::vn(n), o, s)?,
                    _ => return Err(bad("argument in an FP register")),
                },
                Loc::XPair(n) => self.load_fat(xr(n), xr(n + 1), o)?,
                Loc::Stack(_) => {}
            }
        }
        if let Loc::X(n) = env_loc {
            match env_home {
                Some(h) => self.ld(MemSize::X, false, xr(n), SP, h)?,
                None => self.a.mov(X, xr(n), ZR)?,
            }
        }
        let rt = ret.map(|t| self.cg.ty(t));
        let rk = abi::ret(rt);
        if rk == Ret::Indirect {
            let Some(r) = res else { return Err(bad("aggregate call result without a value")) };
            let h = self.home(r);
            self.add_imm(xr(8), SP, h)?;
        }
        match callee {
            Ok(sym) => self.a.bl_sym(sym),
            Err(o) => {
                match o {
                    Operand::Value(v) => {
                        let h = self.home(v);
                        self.ld(MemSize::X, false, S16, SP, h)?;
                    }
                    _ => self.load_fat(S16, S17, o)?,
                }
                self.a.blr(S16)?;
            }
        }
        self.result_regs(rk, res)
    }

    /// Binds a call's result registers to `res` (after a call; nothing is bound).
    pub fn result_regs(&mut self, rk: Ret, res: Option<Value>) -> R<()> {
        let Some(r) = res else { return Ok(()) };
        if self.vkind(r) == Vk::Dead {
            return Ok(());
        }
        match rk {
            Ret::X0 | Ret::V0 if self.owned(r).is_some() => {
                let own = self.owned(r).unwrap_or(NOREG);
                match self.vkind(r) {
                    Vk::Fpr(s) => self.a.fmov(s.fp(), VReg::vn(own), VReg::vn(0)),
                    _ => self.a.mov(X, xr(own), xr(0))?,
                }
            }
            Ret::X0 => self.bind_g(r, 0, true),
            Ret::V0 => self.bind_f(r, 0, true),
            Ret::X0X1 => {
                let h = self.home(r);
                self.st(MemSize::X, xr(0), SP, h)?;
                self.st(MemSize::X, xr(1), SP, h + 8)?;
            }
            Ret::Q0 => {
                let Ty::Vec(vt) = self.vty(r) else { return Err(bad("vector call result")) };
                let h = self.home(r);
                self.stf(vt.home_size(), VReg::vn(0), SP, h)?;
            }
            Ret::Indirect | Ret::None => {}
        }
        Ok(())
    }
}
