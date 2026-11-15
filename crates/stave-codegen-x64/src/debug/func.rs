//! Per-function code generation: frame homes, the greedy local register cache, block and
//! terminator emission, calls, prologue/epilogue, and trap stubs. Instruction selection for the
//! individual ops is in `ops`, `cast`, and `intrin`.

use stave_ir::arith::IntTy;
use stave_ir::ir::Target;
use stave_ir::layout::Layout;
use stave_ir::{
    Body, Const, FloatBits, Func, FuncId, Operand, PanicKind, Site, StrId, Term, TypeId, TypeKey, Value,
};
use stave_obj::{FuncCode, SymbolId};

use super::abi::{self, CallAbi, Loc, Piece, RetLoc, VCls};
use super::analysis::{self, Analysis, NONE};
use super::{Ctx, Res, bad, unsup};
use crate::enc::*;

/// Scratch GPRs, never allocated to values.
pub(crate) const S0: Gpr = Gpr::R11;
pub(crate) const S1: Gpr = Gpr::R10;
/// Scratch XMMs, never allocated to values.
pub(crate) const XS0: Xmm = Xmm::masked(15);
pub(crate) const XS1: Xmm = Xmm::masked(14);
pub(crate) const RBP: Gpr = Gpr::RBP;
pub(crate) const RSP: Gpr = Gpr::RSP;
pub(crate) const RAX: Gpr = Gpr::RAX;
pub(crate) const RCX: Gpr = Gpr::RCX;
pub(crate) const RDX: Gpr = Gpr::RDX;
pub(crate) const RBX: Gpr = Gpr::RBX;
pub(crate) const RSI: Gpr = Gpr::RSI;
pub(crate) const RDI: Gpr = Gpr::RDI;

const ALLOC_G: [Gpr; 12] = [
    Gpr::RSI,
    Gpr::RDI,
    Gpr::R8,
    Gpr::R9,
    Gpr::RAX,
    Gpr::RCX,
    Gpr::RDX,
    Gpr::RBX,
    Gpr::R12,
    Gpr::R13,
    Gpr::R14,
    Gpr::R15,
];
pub(crate) const CALLER_SAVED: [Gpr; 7] =
    [Gpr::RSI, Gpr::RDI, Gpr::R8, Gpr::R9, Gpr::RAX, Gpr::RCX, Gpr::RDX];
const CALLEE_SAVED: [Gpr; 5] = [Gpr::RBX, Gpr::R12, Gpr::R13, Gpr::R14, Gpr::R15];
/// `xmm0`..`xmm13` hold values.
const NXMM: u8 = 14;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum R {
    G(Gpr),
    X(Xmm),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Occ {
    Free,
    Val(Value),
    Temp,
}

/// An integer operand: a register or a sign-extended 32-bit immediate.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GI {
    R(Gpr),
    I(i32),
}

/// Where an aggregate's bytes are.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MSrc {
    Mem(Mem),
    /// A string literal (`&str` = address, length).
    Str(StrId),
    /// A fn item as a fn value (code address, null environment).
    Func(FuncId),
}

/// Scalar kinds for op selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sk {
    Int(IntTy),
    Bool,
    Char,
    Float(FloatBits),
    /// Thin pointer or fn value.
    Ptr,
}

pub(crate) enum Callee {
    Sym(SymbolId),
    /// A code address in an integer-class operand.
    Val(Operand),
    /// A fn value `(code, environment)`: the environment is the last argument (`call.ind`).
    Fn(Operand),
}

pub(crate) struct Cg<'a, 'm> {
    pub cx: &'a Ctx<'m>,
    pub f: &'a Func,
    pub b: &'a Body,
    pub an: Analysis,
    pub a: Asm,
    vreg: Vec<Option<R>>,
    in_home: Vec<bool>,
    home: Vec<i32>,
    has_home: Vec<bool>,
    is_slot: Vec<bool>,
    /// The value's home is never rewritten (not a non-entry block param), so `get`/`enum.get`
    /// results may alias into it.
    stable: Vec<bool>,
    last_use: Vec<u32>,
    touched: Vec<u32>,
    gocc: [Occ; 16],
    xocc: [Occ; 16],
    glock: u16,
    xlock: u16,
    used_callee: u16,
    frame: u32,
    labels: Vec<Label>,
    traps: Vec<(PanicKind, u32, Label)>,
    trap_index: rustc_hash::FxHashMap<(u8, u32), Label>,
    epilogue: Label,
    sret_home: Option<i32>,
    fscratch: Option<i32>,
    pub pos: u32,
    cur_site: u32,
    line_site: u32,
    ret_abi: RetLoc,
}

pub(crate) fn compile(cx: &Ctx<'_>, fid: FuncId) -> Res<FuncCode> {
    let f = cx.m.func(fid).ok_or_else(|| bad("unknown function"))?;
    let b = f.body.as_ref().ok_or_else(|| bad("function has no body"))?;
    let an = analysis::analyze(b);
    let nv = b.values.len();
    let mut stable = vec![true; nv];
    for (bi, bd) in b.blocks.iter().enumerate() {
        if bi != 0 {
            for &p in bd.params.of(&b.block_params) {
                if let Some(s) = stable.get_mut(p.index()) {
                    *s = false;
                }
            }
        }
    }
    let mut a = Asm::with_capacity(32 * b.insts.len() + 64, b.blocks.len() * 2 + 8);
    let epilogue = a.new_label();
    let labels = (0..b.blocks.len()).map(|_| a.new_label()).collect();
    let cg = Cg {
        cx,
        f,
        b,
        an,
        a,
        vreg: vec![None; nv],
        in_home: vec![false; nv],
        home: vec![0; nv],
        has_home: vec![false; nv],
        is_slot: vec![false; nv],
        stable,
        last_use: vec![NONE; nv],
        touched: Vec::new(),
        gocc: [Occ::Free; 16],
        xocc: [Occ::Free; 16],
        glock: 0,
        xlock: 0,
        used_callee: 0,
        frame: 0,
        labels,
        traps: Vec::new(),
        trap_index: rustc_hash::FxHashMap::default(),
        epilogue,
        sret_home: None,
        fscratch: None,
        pos: 0,
        cur_site: Site::NONE.0,
        line_site: Site::NONE.0,
        ret_abi: RetLoc::None,
    };
    let fs = cx.fn_sites.get(fid.index()).copied().unwrap_or(Site::NONE);
    cg.run(fs)
}

fn bit(r: Gpr) -> u16 {
    1 << r.num()
}

fn xbit(x: Xmm) -> u16 {
    1 << x.num()
}

pub(crate) fn disp(m: Mem, d: i64) -> Res<Mem> {
    match m {
        Mem::Sib { base, index, disp } => {
            let nd = i32::try_from(i64::from(disp) + d).map_err(|_| unsup("displacement out of range"))?;
            Ok(Mem::Sib { base, index, disp: nd })
        }
        Mem::Rip { target, disp } => {
            let nd = i32::try_from(i64::from(disp) + d).map_err(|_| unsup("displacement out of range"))?;
            Ok(Mem::Rip { target, disp: nd })
        }
    }
}

pub(crate) fn size_of_bytes(bytes: u8) -> Size {
    match bytes {
        1 => Size::B,
        2 => Size::W,
        4 => Size::D,
        _ => Size::Q,
    }
}

impl<'a, 'm> Cg<'a, 'm> {
    // ---------------------------------------------------------------- types

    pub fn key(&self, t: TypeId) -> Option<&'m TypeKey> {
        self.cx.m.types.key(t)
    }

    pub fn vcls(&self, t: TypeId) -> Res<VCls> {
        self.cx
            .cls
            .get(t.index())
            .copied()
            .flatten()
            .ok_or_else(|| unsup(format!("values of type t{} ({})", t.0, self.cx.m.types.key_text(t))))
    }

    pub fn vty(&self, v: Value) -> Res<TypeId> {
        self.b.value_ty(v).ok_or_else(|| bad(format!("value %{} out of range", v.0)))
    }

    pub fn layout(&self, t: TypeId) -> Res<&'a Layout> {
        let cx: &'a Ctx<'m> = self.cx;
        cx.layouts.get_layout(t).map_err(|e| unsup(format!("layout of t{}: {e}", t.0)))
    }

    pub fn sk(&self, t: TypeId) -> Option<Sk> {
        Some(match self.key(t)? {
            TypeKey::Int { signed, bits } => Sk::Int(IntTy::new(bits.width(), *signed)),
            TypeKey::Bool => Sk::Bool,
            TypeKey::Char => Sk::Char,
            TypeKey::Float(f) => Sk::Float(*f),
            TypeKey::FnType { .. } => Sk::Ptr,
            TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. } => {
                if stave_ir::layout::is_unsized_pointee(&self.cx.m.types, *inner) {
                    return None;
                }
                Sk::Ptr
            }
            _ => return None,
        })
    }

    /// The type of a value operand.
    pub fn oty(&self, o: Operand) -> Option<TypeId> {
        match o {
            Operand::Value(v) => self.b.value_ty(v),
            _ => None,
        }
    }

    /// The scalar kind of an operand (`None` for an int constant, typed by context).
    pub fn osk(&self, o: Operand) -> Option<Sk> {
        match o {
            Operand::Value(v) => self.b.value_ty(v).and_then(|t| self.sk(t)),
            Operand::Const(c) => match self.b.konst(c) {
                Some(Const::F32(_)) => Some(Sk::Float(FloatBits::F32)),
                Some(Const::F64(_)) => Some(Sk::Float(FloatBits::F64)),
                Some(Const::F16(_)) => Some(Sk::Float(FloatBits::F16)),
                Some(Const::BF16(_)) => Some(Sk::Float(FloatBits::B16)),
                _ => None,
            },
            Operand::Func(_) => Some(Sk::Ptr),
            _ => None,
        }
    }

    /// The class of an operand; `ctx` types an int constant.
    pub fn ocls(&self, o: Operand, ctx: Option<TypeId>) -> Res<VCls> {
        match o {
            Operand::Value(v) => self.vcls(self.vty(v)?),
            Operand::Const(c) => match self.b.konst(c) {
                Some(Const::F32(_)) => Ok(VCls::X(Prec::F32)),
                Some(Const::F64(_)) => Ok(VCls::X(Prec::F64)),
                Some(Const::F16(_) | Const::BF16(_)) => Ok(VCls::G { bytes: 2, signed: false }),
                Some(Const::Int(_)) => match ctx {
                    Some(t) => self.vcls(t),
                    None => Ok(VCls::G { bytes: 8, signed: true }),
                },
                None => Err(bad("constant out of range")),
            },
            Operand::Str(_) | Operand::Func(_) => Ok(VCls::M { size: 16, align: 8 }),
            Operand::Type(_) => Err(bad("type operand used as a value")),
        }
    }

    /// The 64-bit register image of an integer-like constant (`Int` as normalized bits, f16 /
    /// bf16 bits).
    pub fn const_bits(&self, o: Operand) -> Option<i64> {
        match o {
            Operand::Const(c) => match self.b.konst(c)? {
                Const::Int(x) => Some(x as u64 as i64),
                Const::F16(b) | Const::BF16(b) => Some(i64::from(b)),
                Const::F32(b) => Some(i64::from(b)),
                Const::F64(b) => Some(b as i64),
            },
            _ => None,
        }
    }

    pub fn int_const(&self, o: Operand) -> Option<i128> {
        self.b.int_const(o)
    }

    pub fn str_len(&self, s: StrId) -> Res<i64> {
        self.cx.m.strings.get(s.index()).map(|x| x.len() as i64).ok_or_else(|| bad("string out of range"))
    }

    pub fn str_sym(&self, s: StrId) -> Res<SymbolId> {
        self.cx.str_sym.get(s.index()).copied().ok_or_else(|| bad("string out of range"))
    }

    pub fn func_sym(&self, f: FuncId) -> Res<SymbolId> {
        self.cx
            .func_sym
            .get(f.index())
            .copied()
            .flatten()
            .ok_or_else(|| unsup("reference to a fn without a body"))
    }

    // ---------------------------------------------------------------- frame

    pub fn alloc_frame(&mut self, size: u32, align: u32) -> Res<i32> {
        let al = align.clamp(1, 16);
        let end = u64::from(self.frame) + u64::from(size);
        let end = end.div_ceil(u64::from(al)) * u64::from(al);
        if end > (1 << 30) {
            return Err(unsup("stack frame larger than 1 GiB"));
        }
        self.frame = end as u32;
        Ok(-(end as i32))
    }

    /// The value's home (allocated on first use).
    pub fn home(&mut self, v: Value) -> Res<i32> {
        let i = v.index();
        if i >= self.home.len() {
            return Err(bad(format!("value %{} out of range", v.0)));
        }
        if !self.has_home[i] {
            let h = match self.vcls(self.vty(v)?)? {
                VCls::G { .. } | VCls::X(_) => self.alloc_frame(8, 8)?,
                VCls::M { size, align } => self.alloc_frame(size.div_ceil(8) * 8, align.max(8))?,
            };
            self.home[i] = h;
            self.has_home[i] = true;
        }
        Ok(self.home[i])
    }

    pub fn hmem(&mut self, v: Value) -> Res<Mem> {
        Ok(Mem::base(RBP, self.home(v)?))
    }

    /// Defines an aggregate-class result: allocates its home.
    pub fn def_mem(&mut self, v: Value) -> Res<Mem> {
        let m = self.hmem(v)?;
        self.in_home[v.index()] = true;
        Ok(m)
    }

    /// Makes `v`'s home alias `root`'s home + `off` (an immutable aggregate part), when allowed.
    pub fn alias(&mut self, v: Value, root: Value, off: i64) -> Res<bool> {
        let (vi, ri) = (v.index(), root.index());
        if vi >= self.home.len() || ri >= self.home.len() || !self.stable[ri] || self.is_slot[ri] {
            return Ok(false);
        }
        let h = self.home(root)?;
        let nh = i32::try_from(i64::from(h) + off).map_err(|_| unsup("frame offset out of range"))?;
        self.home[vi] = nh;
        self.has_home[vi] = true;
        self.in_home[vi] = true;
        Ok(true)
    }

    /// Marks `v` as a remateralizable frame address `rbp + off` (a `slot`, or a constant offset
    /// into one).
    pub fn def_slot(&mut self, v: Value, off: i32) -> Res<()> {
        let i = v.index();
        if i >= self.home.len() {
            return Err(bad("slot result out of range"));
        }
        self.home[i] = off;
        self.has_home[i] = true;
        self.in_home[i] = true;
        self.is_slot[i] = true;
        Ok(())
    }

    pub fn slot_off(&self, o: Operand) -> Option<i32> {
        match o {
            Operand::Value(v) if self.is_slot.get(v.index()).copied().unwrap_or(false) => {
                Some(self.home[v.index()])
            }
            _ => None,
        }
    }

    /// 16 bytes of scratch memory for the x87 remainder.
    pub fn fscratch(&mut self) -> Res<Mem> {
        let h = match self.fscratch {
            Some(h) => h,
            None => {
                let h = self.alloc_frame(16, 16)?;
                self.fscratch = Some(h);
                h
            }
        };
        Ok(Mem::base(RBP, h))
    }

    // ---------------------------------------------------------------- register cache

    fn occ(&self, r: R) -> Occ {
        match r {
            R::G(g) => self.gocc[g.num() as usize],
            R::X(x) => self.xocc[x.num() as usize],
        }
    }

    fn set_occ(&mut self, r: R, o: Occ) {
        match r {
            R::G(g) => self.gocc[g.num() as usize] = o,
            R::X(x) => self.xocc[x.num() as usize] = o,
        }
    }

    fn locked(&self, r: R) -> bool {
        match r {
            R::G(g) => self.glock & bit(g) != 0,
            R::X(x) => self.xlock & xbit(x) != 0,
        }
    }

    fn lock(&mut self, r: R) {
        match r {
            R::G(g) => {
                self.glock |= bit(g);
                if CALLEE_SAVED.contains(&g) {
                    self.used_callee |= bit(g);
                }
            }
            R::X(x) => self.xlock |= xbit(x),
        }
    }

    /// Still needed after the current instruction (or by it).
    fn needed(&self, v: Value) -> bool {
        let lu = self.last_use.get(v.index()).copied().unwrap_or(NONE);
        lu != NONE && lu >= self.pos
    }

    pub fn store_home(&mut self, v: Value, r: R) -> Res<()> {
        let m = self.hmem(v)?;
        match r {
            R::G(g) => self.a.mov_mr(Size::Q, m, g)?,
            R::X(x) => {
                let p = match self.vcls(self.vty(v)?)? {
                    VCls::X(p) => p,
                    _ => Prec::F64,
                };
                self.a.movs_store(p, m, x)?
            }
        }
        self.in_home[v.index()] = true;
        Ok(())
    }

    fn free_reg(&mut self, r: R) {
        if let Occ::Val(v) = self.occ(r)
            && let Some(s) = self.vreg.get_mut(v.index())
        {
            *s = None;
        }
        self.set_occ(r, Occ::Free);
    }

    /// Frees `r`, saving its value first if it is still needed and not in its home.
    pub fn evict(&mut self, r: R) -> Res<()> {
        match self.occ(r) {
            Occ::Free => {}
            Occ::Temp => return Err(bad("internal: evicting a temporary register")),
            Occ::Val(v) => {
                let vi = v.index();
                if !self.in_home[vi] && (self.needed(v) || self.an.escapes[vi]) {
                    self.store_home(v, r)?;
                }
                self.free_reg(r);
            }
        }
        Ok(())
    }

    fn victim_score(&self, r: R) -> Option<u32> {
        if self.locked(r) {
            return None;
        }
        match self.occ(r) {
            Occ::Free => Some(0),
            Occ::Temp => None,
            Occ::Val(v) => {
                let vi = v.index();
                if !self.needed(v) {
                    Some(1)
                } else if self.in_home[vi] {
                    Some(2)
                } else {
                    let d = self.an.def_block.get(vi).and_then(|&b| self.an.loop_depth.get(b as usize));
                    Some(3 + u32::from(d.copied().unwrap_or(0)))
                }
            }
        }
    }

    pub fn alloc_g(&mut self) -> Res<Gpr> {
        let mut best: Option<(u32, Gpr)> = None;
        for g in ALLOC_G {
            if let Some(s) = self.victim_score(R::G(g)) {
                if s == 0 {
                    best = Some((0, g));
                    break;
                }
                if best.is_none_or(|(bs, _)| s < bs) {
                    best = Some((s, g));
                }
            }
        }
        let (_, g) = best.ok_or_else(|| bad("internal: out of general registers"))?;
        self.evict(R::G(g))?;
        self.set_occ(R::G(g), Occ::Temp);
        self.lock(R::G(g));
        Ok(g)
    }

    pub fn alloc_x(&mut self) -> Res<Xmm> {
        let mut best: Option<(u32, Xmm)> = None;
        for n in 0..NXMM {
            let x = Xmm::masked(n);
            if let Some(s) = self.victim_score(R::X(x)) {
                if s == 0 {
                    best = Some((0, x));
                    break;
                }
                if best.is_none_or(|(bs, _)| s < bs) {
                    best = Some((s, x));
                }
            }
        }
        let (_, x) = best.ok_or_else(|| bad("internal: out of vector registers"))?;
        self.evict(R::X(x))?;
        self.set_occ(R::X(x), Occ::Temp);
        self.lock(R::X(x));
        Ok(x)
    }

    /// Takes a specific register as a temporary (its value is saved / moved out first).
    pub fn reserve(&mut self, r: R) -> Res<()> {
        if self.locked(r) {
            return Err(bad("internal: fixed register already in use"));
        }
        self.evict(r)?;
        self.set_occ(r, Occ::Temp);
        self.lock(r);
        Ok(())
    }

    pub fn reserve_g(&mut self, g: Gpr) -> Res<()> {
        self.reserve(R::G(g))
    }

    /// Makes `r` hold value `v` (the instruction's result). Values used in other blocks are also
    /// written to their home.
    pub fn bind(&mut self, v: Value, r: R) -> Res<()> {
        let vi = v.index();
        if vi >= self.vreg.len() {
            return Err(bad("result value out of range"));
        }
        if let Occ::Val(old) = self.occ(r)
            && old != v
        {
            self.free_reg(r);
        }
        self.set_occ(r, Occ::Val(v));
        self.vreg[vi] = Some(r);
        self.lock(r);
        self.in_home[vi] = false;
        if self.an.escapes[vi] {
            self.store_home(v, r)?;
        }
        Ok(())
    }

    pub fn bind_g(&mut self, v: Value, g: Gpr) -> Res<()> {
        self.bind(v, R::G(g))
    }

    /// Saves every caller-saved register (a call follows).
    pub fn spill_caller_saved(&mut self) -> Res<()> {
        for g in CALLER_SAVED {
            self.evict(R::G(g))?;
        }
        for n in 0..NXMM {
            self.evict(R::X(Xmm::masked(n)))?;
        }
        Ok(())
    }

    fn reset_regs(&mut self) {
        for n in 0..16u8 {
            if let Some(g) = Gpr::new(n) {
                self.free_reg(R::G(g));
            }
            self.free_reg(R::X(Xmm::masked(n)));
        }
        self.glock = 0;
        self.xlock = 0;
    }

    fn end_inst(&mut self, i: u32) {
        let pos = self.pos;
        let b = self.b;
        for &o in b.args(i) {
            if let Operand::Value(v) = o
                && self.last_use.get(v.index()).copied() == Some(pos)
                && let Some(r) = self.vreg[v.index()]
            {
                self.free_reg(r);
            }
        }
        if let Some(Some(v)) = b.insts.result.get(i as usize) {
            let vi = v.index();
            if vi < self.vreg.len()
                && !self.needed_after(*v)
                && let Some(r) = self.vreg[vi]
            {
                self.free_reg(r);
            }
        }
        for n in 0..16usize {
            if self.gocc[n] == Occ::Temp {
                self.gocc[n] = Occ::Free;
            }
            if self.xocc[n] == Occ::Temp {
                self.xocc[n] = Occ::Free;
            }
        }
        self.glock = 0;
        self.xlock = 0;
    }

    fn needed_after(&self, v: Value) -> bool {
        let lu = self.last_use.get(v.index()).copied().unwrap_or(NONE);
        lu != NONE && lu > self.pos
    }

    // ---------------------------------------------------------------- operand access

    /// Puts the current value of `v` into `dst` without changing the cache.
    fn load_val(&mut self, v: Value, dst: R) -> Res<()> {
        let vi = v.index();
        let cur = self.vreg.get(vi).copied().flatten();
        if let Some(r) = cur {
            match (r, dst) {
                (R::G(s), R::G(d)) => {
                    if s != d {
                        self.a.mov_rr(Size::Q, d, s);
                    }
                }
                (R::X(s), R::X(d)) => {
                    if s != d {
                        self.a.movaps(d, s);
                    }
                }
                (R::X(s), R::G(d)) => self.a.mov_gx(Size::Q, d, s)?,
                (R::G(s), R::X(d)) => self.a.mov_xg(Size::Q, d, s)?,
            }
            return Ok(());
        }
        if !self.in_home.get(vi).copied().unwrap_or(false) {
            return Err(bad(format!("internal: %{} has no location", v.0)));
        }
        let h = self.home[vi];
        let m = Mem::base(RBP, h);
        if self.is_slot[vi] {
            return match dst {
                R::G(d) => Ok(self.a.lea(Size::Q, d, m)?),
                R::X(_) => Err(bad("slot address in a vector register")),
            };
        }
        match dst {
            R::G(d) => self.a.mov_rm(Size::Q, d, m)?,
            R::X(d) => {
                let p = match self.vcls(self.vty(v)?)? {
                    VCls::X(p) => p,
                    _ => Prec::F64,
                };
                self.a.movs(p, d, m)?
            }
        }
        Ok(())
    }

    /// Copies an integer-class operand into `dst` (no cache change).
    pub fn copy_to_g(&mut self, o: Operand, dst: Gpr) -> Res<()> {
        match o {
            Operand::Value(v) => self.load_val(v, R::G(dst)),
            Operand::Const(_) => {
                let x = self.const_bits(o).ok_or_else(|| bad("constant"))?;
                if x == 0 {
                    self.a.alu_rr(AluOp::Xor, Size::D, dst, dst);
                } else {
                    self.a.mov_ri(Size::Q, dst, x)?;
                }
                Ok(())
            }
            Operand::Func(f) => {
                let s = self.func_sym(f)?;
                Ok(self.a.lea(Size::Q, dst, Mem::rip_sym(s, 0))?)
            }
            Operand::Str(_) => Err(bad("string literal where a scalar is expected")),
            Operand::Type(_) => Err(bad("type operand used as a value")),
        }
    }

    /// Copies a float operand into `dst` (no cache change).
    pub fn copy_to_x(&mut self, o: Operand, p: Prec, dst: Xmm) -> Res<()> {
        match o {
            Operand::Value(v) => self.load_val(v, R::X(dst)),
            Operand::Const(_) => {
                let x = self.const_bits(o).ok_or_else(|| bad("constant"))?;
                if x == 0 {
                    self.a.logic_p(PLogic::Xor, p, dst, dst)?;
                } else {
                    self.a.mov_ri(Size::Q, S0, x)?;
                    self.a.mov_xg(Size::Q, dst, S0)?;
                }
                Ok(())
            }
            _ => Err(bad("non-float operand where a float is expected")),
        }
    }

    /// An integer-class operand in a register (cached for later uses in the block); locked
    /// until the end of the instruction.
    pub fn read_g(&mut self, o: Operand) -> Res<Gpr> {
        if let Operand::Value(v) = o {
            let vi = v.index();
            if let Some(R::G(g)) = self.vreg.get(vi).copied().flatten() {
                self.lock(R::G(g));
                return Ok(g);
            }
            let g = self.alloc_g()?;
            self.load_val(v, R::G(g))?;
            self.set_occ(R::G(g), Occ::Val(v));
            self.vreg[vi] = Some(R::G(g));
            return Ok(g);
        }
        let g = self.alloc_g()?;
        self.copy_to_g(o, g)?;
        Ok(g)
    }

    /// A register or an immediate (int constants whose 64-bit image fits `i32`).
    pub fn read_gi(&mut self, o: Operand) -> Res<GI> {
        if matches!(o, Operand::Const(_))
            && let Some(x) = self.const_bits(o)
            && let Ok(i) = i32::try_from(x)
        {
            return Ok(GI::I(i));
        }
        Ok(GI::R(self.read_g(o)?))
    }

    pub fn read_x(&mut self, o: Operand, p: Prec) -> Res<Xmm> {
        if let Operand::Value(v) = o {
            let vi = v.index();
            if let Some(R::X(x)) = self.vreg.get(vi).copied().flatten() {
                self.lock(R::X(x));
                return Ok(x);
            }
            let x = self.alloc_x()?;
            self.load_val(v, R::X(x))?;
            self.set_occ(R::X(x), Occ::Val(v));
            self.vreg[vi] = Some(R::X(x));
            return Ok(x);
        }
        let x = self.alloc_x()?;
        self.copy_to_x(o, p, x)?;
        Ok(x)
    }

    /// A fresh temporary holding a copy of `o`.
    pub fn new_g(&mut self, o: Operand) -> Res<Gpr> {
        let g = self.alloc_g()?;
        self.copy_to_g(o, g)?;
        Ok(g)
    }

    /// A register holding `o` that the instruction may overwrite: `o`'s own register when this
    /// is its last use in the block, else a fresh copy.
    pub fn take_g(&mut self, o: Operand) -> Res<Gpr> {
        if let Operand::Value(v) = o {
            let vi = v.index();
            if let Some(R::G(g)) = self.vreg.get(vi).copied().flatten()
                && self.last_use.get(vi).copied() == Some(self.pos)
                && (self.in_home[vi] || !self.an.escapes[vi])
            {
                self.vreg[vi] = None;
                self.set_occ(R::G(g), Occ::Temp);
                self.lock(R::G(g));
                return Ok(g);
            }
        }
        self.new_g(o)
    }

    pub fn new_x(&mut self, o: Operand, p: Prec) -> Res<Xmm> {
        let x = self.alloc_x()?;
        self.copy_to_x(o, p, x)?;
        Ok(x)
    }

    pub fn take_x(&mut self, o: Operand, p: Prec) -> Res<Xmm> {
        if let Operand::Value(v) = o {
            let vi = v.index();
            if let Some(R::X(x)) = self.vreg.get(vi).copied().flatten()
                && self.last_use.get(vi).copied() == Some(self.pos)
                && (self.in_home[vi] || !self.an.escapes[vi])
            {
                self.vreg[vi] = None;
                self.set_occ(R::X(x), Occ::Temp);
                self.lock(R::X(x));
                return Ok(x);
            }
        }
        self.new_x(o, p)
    }

    /// The bytes of an aggregate-class operand.
    pub fn msrc(&mut self, o: Operand) -> Res<MSrc> {
        match o {
            Operand::Value(v) => {
                let vi = v.index();
                if vi >= self.in_home.len() || !self.in_home[vi] {
                    return Err(bad(format!("internal: aggregate %{} has no home", v.0)));
                }
                Ok(MSrc::Mem(Mem::base(RBP, self.home[vi])))
            }
            Operand::Str(s) => Ok(MSrc::Str(s)),
            Operand::Func(f) => Ok(MSrc::Func(f)),
            _ => Err(bad("scalar operand where an aggregate is expected")),
        }
    }

    // ---------------------------------------------------------------- memory helpers

    /// `mov dst, [m]` of `bytes` bytes, normalized to 64 bits.
    pub fn sized_load(&mut self, dst: Gpr, m: Mem, bytes: u8, signed: bool) -> Res<()> {
        match (bytes, signed) {
            (1, true) => self.a.movsx(Size::Q, dst, Size::B, m)?,
            (1, false) => self.a.movzx(Size::D, dst, Size::B, m)?,
            (2, true) => self.a.movsx(Size::Q, dst, Size::W, m)?,
            (2, false) => self.a.movzx(Size::D, dst, Size::W, m)?,
            (4, true) => self.a.movsx(Size::Q, dst, Size::D, m)?,
            (4, false) => self.a.mov_rm(Size::D, dst, m)?,
            _ => self.a.mov_rm(Size::Q, dst, m)?,
        }
        Ok(())
    }

    /// Normalizes the low `bits` of `src` into `dst` (sign- or zero-extension to 64 bits).
    pub fn norm_into(&mut self, dst: Gpr, src: Gpr, t: IntTy) -> Res<()> {
        match (t.bits, t.signed) {
            (8, true) => self.a.movsx(Size::Q, dst, Size::B, src)?,
            (8, false) => self.a.movzx(Size::D, dst, Size::B, src)?,
            (16, true) => self.a.movsx(Size::Q, dst, Size::W, src)?,
            (16, false) => self.a.movzx(Size::D, dst, Size::W, src)?,
            (32, true) => self.a.movsx(Size::Q, dst, Size::D, src)?,
            (32, false) => self.a.mov_rr(Size::D, dst, src),
            _ => {
                if dst != src {
                    self.a.mov_rr(Size::Q, dst, src);
                }
            }
        }
        Ok(())
    }

    pub fn norm(&mut self, r: Gpr, t: IntTy) -> Res<()> {
        if t.bits < 64 { self.norm_into(r, r, t) } else { Ok(()) }
    }

    /// Copies `size` bytes (scratch registers only; no cache change). `dst` and a memory `src`
    /// must be `[base + disp]` with a base register other than the scratch registers.
    pub fn copy_mem(&mut self, dst: Mem, src: MSrc, size: u32) -> Res<()> {
        let src = match src {
            MSrc::Str(s) => {
                if size != 16 {
                    return Err(bad("string literal copied as a non-fat value"));
                }
                let sym = self.str_sym(s)?;
                let n = self.str_len(s)?;
                self.a.lea(Size::Q, S0, Mem::rip_sym(sym, 0))?;
                self.a.mov_mr(Size::Q, dst, S0)?;
                self.store_imm(disp(dst, 8)?, 8, n)?;
                return Ok(());
            }
            MSrc::Func(f) => {
                if size != 16 {
                    return Err(bad("fn item copied as a non-fn value"));
                }
                let sym = self.func_sym(f)?;
                self.a.lea(Size::Q, S0, Mem::rip_sym(sym, 0))?;
                self.a.mov_mr(Size::Q, dst, S0)?;
                self.store_imm(disp(dst, 8)?, 8, 0)?;
                return Ok(());
            }
            MSrc::Mem(m) => m,
        };
        if src == dst || size == 0 {
            return Ok(());
        }
        let n8 = size / 8;
        let mut off = 0i64;
        if n8 <= 16 {
            for _ in 0..n8 {
                self.a.mov_rm(Size::Q, S0, disp(src, off)?)?;
                self.a.mov_mr(Size::Q, disp(dst, off)?, S0)?;
                off += 8;
            }
        } else {
            let (
                Mem::Sib { base: Some(sb), index: None, disp: sd },
                Mem::Sib { base: Some(db), index: None, disp: dd },
            ) = (src, dst)
            else {
                return Err(bad("internal: block copy needs base+disp operands"));
            };
            if [sb, db].iter().any(|r| *r == S0 || *r == S1) {
                return Err(bad("internal: block copy base is a scratch register"));
            }
            self.a.mov_ri(Size::Q, S1, i64::from(n8))?;
            let top = self.a.new_label();
            self.a.bind(top)?;
            self.a.mov_rm(
                Size::Q,
                S0,
                Mem::index(sb, S1, Scale::S8, sd.checked_sub(8).ok_or_else(|| unsup("disp"))?),
            )?;
            self.a.mov_mr(
                Size::Q,
                Mem::index(db, S1, Scale::S8, dd.checked_sub(8).ok_or_else(|| unsup("disp"))?),
                S0,
            )?;
            self.a.alu_ri(AluOp::Sub, Size::Q, S1, 1)?;
            self.a.jcc(Cond::Ne, top)?;
            off = i64::from(n8) * 8;
        }
        let mut rest = size % 8;
        for (w, s) in [(4u32, Size::D), (2, Size::W), (1, Size::B)] {
            if rest >= w {
                self.a.mov_rm(s, S0, disp(src, off)?)?;
                self.a.mov_mr(s, disp(dst, off)?, S0)?;
                off += i64::from(w);
                rest -= w;
            }
        }
        Ok(())
    }

    /// Zeroes `size` bytes at `dst`.
    pub fn zero_mem(&mut self, dst: Mem, size: u32) -> Res<()> {
        let n8 = size / 8;
        let mut off = 0i64;
        if n8 <= 16 {
            for _ in 0..n8 {
                self.a.mov_mi(Size::Q, disp(dst, off)?, 0)?;
                off += 8;
            }
        } else {
            let Mem::Sib { base: Some(db), index: None, disp: dd } = dst else {
                return Err(bad("internal: block zero needs a base+disp operand"));
            };
            self.a.alu_rr(AluOp::Xor, Size::D, S0, S0);
            self.a.mov_ri(Size::Q, S1, i64::from(n8))?;
            let top = self.a.new_label();
            self.a.bind(top)?;
            self.a.mov_mr(
                Size::Q,
                Mem::index(db, S1, Scale::S8, dd.checked_sub(8).ok_or_else(|| unsup("disp"))?),
                S0,
            )?;
            self.a.alu_ri(AluOp::Sub, Size::Q, S1, 1)?;
            self.a.jcc(Cond::Ne, top)?;
            off = i64::from(n8) * 8;
        }
        let mut rest = size % 8;
        for (w, s) in [(4u32, Size::D), (2, Size::W), (1, Size::B)] {
            if rest >= w {
                self.a.mov_mi(s, disp(dst, off)?, 0)?;
                off += i64::from(w);
                rest -= w;
            }
        }
        Ok(())
    }

    /// Stores the low `bytes` of the immediate `x`.
    pub fn store_imm(&mut self, dst: Mem, bytes: u8, x: i64) -> Res<()> {
        let s = size_of_bytes(bytes);
        let v = match bytes {
            1 => i64::from(x as i8),
            2 => i64::from(x as i16),
            4 => i64::from(x as i32),
            _ => x,
        };
        if i32::try_from(v).is_ok() {
            self.a.mov_mi(s, dst, v)?;
        } else {
            self.a.mov_ri(Size::Q, S0, x)?;
            self.a.mov_mr(s, dst, S0)?;
        }
        Ok(())
    }

    /// Stores operand `o` of class `c` at `dst` (natural size).
    pub fn store_cls(&mut self, o: Operand, c: VCls, dst: Mem) -> Res<()> {
        match c {
            VCls::G { bytes, .. } => match o {
                Operand::Const(_) => {
                    let x = self.const_bits(o).ok_or_else(|| bad("constant"))?;
                    self.store_imm(dst, bytes, x)
                }
                Operand::Func(_) => {
                    self.copy_to_g(o, S0)?;
                    Ok(self.a.mov_mr(Size::Q, dst, S0)?)
                }
                _ => {
                    let g = self.read_g(o)?;
                    self.a.mov_mr(size_of_bytes(bytes), dst, g)?;
                    self.release(R::G(g), dst);
                    Ok(())
                }
            },
            VCls::X(p) => match o {
                Operand::Const(_) => {
                    let x = self.const_bits(o).ok_or_else(|| bad("constant"))?;
                    self.store_imm(dst, if p == Prec::F32 { 4 } else { 8 }, x)
                }
                _ => {
                    let x = self.read_x(o, p)?;
                    self.a.movs_store(p, dst, x)?;
                    self.release(R::X(x), dst);
                    Ok(())
                }
            },
            VCls::M { size, .. } => {
                let s = self.msrc(o)?;
                self.copy_mem(dst, s, size)
            }
        }
    }

    /// Unlocks a cached operand register that the rest of the instruction does not need (so
    /// `make` of many operands cannot run out of registers). Keeps registers used by `keep`.
    fn release(&mut self, r: R, keep: Mem) {
        if let (R::G(g), Mem::Sib { base, index, .. }) = (r, keep)
            && (base == Some(g) || index.is_some_and(|(i, _)| i == g))
        {
            return;
        }
        if let Occ::Val(_) = self.occ(r) {
            match r {
                R::G(g) => self.glock &= !bit(g),
                R::X(x) => self.xlock &= !xbit(x),
            }
        }
    }

    /// Defines `res` (class `c`) from memory at `src`; aggregates are copied to a fresh home.
    pub fn load_cls(&mut self, res: Value, c: VCls, src: Mem) -> Res<()> {
        match c {
            VCls::G { bytes, signed } => {
                let g = self.alloc_g()?;
                self.sized_load(g, src, bytes, signed)?;
                // a bool byte read from memory is true iff nonzero (as in the interpreter)
                if matches!(self.key(self.vty(res)?), Some(TypeKey::Bool)) {
                    self.a.test_rr(Size::D, g, g);
                    self.a.setcc(Cond::Ne, g);
                    self.a.movzx(Size::D, g, Size::B, g)?;
                }
                self.bind_g(res, g)
            }
            VCls::X(p) => {
                let x = self.alloc_x()?;
                self.a.movs(p, x, src)?;
                self.bind(res, R::X(x))
            }
            VCls::M { size, .. } => {
                let d = self.def_mem(res)?;
                self.copy_mem(d, MSrc::Mem(src), size)
            }
        }
    }

    // ---------------------------------------------------------------- traps

    pub fn trap_label(&mut self, k: PanicKind) -> Label {
        let site = self.cur_site;
        let key = (k as u8, site);
        if let Some(&l) = self.trap_index.get(&key) {
            return l;
        }
        let l = self.a.new_label();
        self.trap_index.insert(key, l);
        self.traps.push((k, site, l));
        l
    }

    /// `jcc` to the trap stub of kind `k` at the current site.
    pub fn jtrap(&mut self, c: Cond, k: PanicKind) -> Res<()> {
        let l = self.trap_label(k);
        Ok(self.a.jcc(c, l)?)
    }

    /// An unconditional trap.
    pub fn trap_now(&mut self, k: PanicKind) -> Res<()> {
        let l = self.trap_label(k);
        Ok(self.a.jmp(l)?)
    }

    // ---------------------------------------------------------------- calls

    pub fn abi_of(&self, params: &[TypeId], ret: Option<TypeId>) -> Res<CallAbi> {
        abi::call_abi(&self.cx.m.types, &self.cx.layouts, &self.cx.cls, params, ret, false)
            .ok_or_else(|| unsup("a parameter or return type has no value class"))
    }

    /// [`Self::abi_of`] plus the trailing environment argument of a fn-value call.
    pub fn abi_of_env(&self, params: &[TypeId], ret: Option<TypeId>) -> Res<CallAbi> {
        abi::call_abi(&self.cx.m.types, &self.cx.layouts, &self.cx.cls, params, ret, true)
            .ok_or_else(|| unsup("a parameter or return type has no value class"))
    }

    /// `sub rsp, n`, touching every page on the way down (as the prologue does), so a large
    /// stack-argument area cannot jump the guard page. Uses `S0`.
    fn sub_rsp_probed(&mut self, n: u32) -> Res<()> {
        if n > (1 << 30) {
            return Err(unsup("stack argument area larger than 1 GiB"));
        }
        let pages = n / 4096;
        if pages <= 8 {
            for _ in 0..pages {
                self.a.alu_ri(AluOp::Sub, Size::Q, RSP, 4096)?;
                self.a.alu_mi(AluOp::Or, Size::Q, Mem::base(RSP, 0), 0)?;
            }
        } else {
            self.a.mov_ri(Size::Q, S0, i64::from(pages))?;
            let top = self.a.new_label();
            self.a.bind(top)?;
            self.a.alu_ri(AluOp::Sub, Size::Q, RSP, 4096)?;
            self.a.alu_mi(AluOp::Or, Size::Q, Mem::base(RSP, 0), 0)?;
            self.a.alu_ri(AluOp::Sub, Size::Q, S0, 1)?;
            self.a.jcc(Cond::Ne, top)?;
        }
        let rest = n % 4096;
        if rest > 0 {
            self.a.alu_ri(AluOp::Sub, Size::Q, RSP, i64::from(rest))?;
        }
        Ok(())
    }

    /// A call with the Stave calling convention (`ccall`: also `eax = 0`).
    pub fn call(
        &mut self,
        callee: Callee,
        args: &[Operand],
        abi: &CallAbi,
        res: Option<Value>,
        ccall: bool,
    ) -> Res<()> {
        let env = match callee {
            Callee::Fn(o) => Some(self.msrc(o)?),
            _ => None,
        };
        if args.len() + usize::from(env.is_some()) != abi.args.len() {
            return Err(bad("call argument count differs from the callee's parameters"));
        }
        self.spill_caller_saved()?;
        let res_home = match (&abi.ret, res) {
            (RetLoc::Sret, Some(v)) => Some(self.home(v)?),
            (RetLoc::Sret, None) => return Err(bad("internal: sret call without a result value")),
            _ => None,
        };
        if abi.stack > 0 {
            self.sub_rsp_probed(abi.stack)?;
        }
        for (&o, loc) in args.iter().zip(&abi.args) {
            if let Loc::Stack(off) = *loc {
                let dst = Mem::base(RSP, off as i32);
                match self.ocls(o, None)? {
                    VCls::G { .. } => {
                        self.copy_to_g(o, S0)?;
                        self.a.mov_mr(Size::Q, dst, S0)?;
                    }
                    VCls::X(p) => {
                        self.copy_to_x(o, p, XS0)?;
                        self.a.movs_store(p, dst, XS0)?;
                    }
                    VCls::M { size, .. } => {
                        let s = self.msrc(o)?;
                        self.copy_mem(dst, s, size)?;
                    }
                }
            }
        }
        // the environment word of a fn-value call (after the declared arguments)
        let env_loc = env.and_then(|e| abi.args.last().map(|l| (e, l.clone())));
        if let Some((e, Loc::Stack(off))) = &env_loc {
            self.env_word(*e, S0)?;
            self.a.mov_mr(Size::Q, Mem::base(RSP, *off as i32), S0)?;
        }
        for (&o, loc) in args.iter().zip(&abi.args) {
            match loc {
                Loc::Reg(Piece::G(g)) => self.copy_to_g(o, *g)?,
                Loc::Reg(Piece::X(x)) => {
                    let p = match self.ocls(o, None)? {
                        VCls::X(p) => p,
                        _ => Prec::F64,
                    };
                    self.copy_to_x(o, p, *x)?
                }
                Loc::Regs(ps) => {
                    let s = self.msrc(o)?;
                    self.load_pieces(s, ps)?;
                }
                Loc::Ignore | Loc::Stack(_) => {}
            }
        }
        if let Some((e, Loc::Reg(Piece::G(g)))) = &env_loc {
            self.env_word(*e, *g)?;
        }
        if let Some(h) = res_home {
            self.a.lea(Size::Q, RDI, Mem::base(RBP, h))?;
        }
        if ccall {
            self.a.alu_rr(AluOp::Xor, Size::D, RAX, RAX);
        }
        match callee {
            Callee::Sym(s) => self.a.call_sym(s),
            Callee::Val(o) => {
                self.copy_to_g(o, S0)?;
                self.a.call_rm(S0)?;
            }
            Callee::Fn(o) => match self.msrc(o)? {
                MSrc::Func(f) => {
                    let sym = self.func_sym(f)?;
                    self.a.call_sym(sym);
                }
                MSrc::Mem(m) => {
                    self.a.mov_rm(Size::Q, S0, m)?;
                    self.a.call_rm(S0)?;
                }
                MSrc::Str(_) => return Err(bad("string literal called as a fn value")),
            },
        }
        if abi.stack > 0 {
            self.a.alu_ri(AluOp::Add, Size::Q, RSP, i64::from(abi.stack))?;
        }
        let Some(v) = res else { return Ok(()) };
        match &abi.ret {
            RetLoc::None => {
                // zero-sized result
                self.def_mem(v)?;
            }
            RetLoc::Reg(Piece::G(g)) => {
                self.set_occ(R::G(*g), Occ::Temp);
                self.bind_g(v, *g)?;
            }
            RetLoc::Reg(Piece::X(x)) => {
                self.set_occ(R::X(*x), Occ::Temp);
                self.bind(v, R::X(*x))?;
            }
            RetLoc::Regs(ps) => {
                let d = self.def_mem(v)?;
                self.store_pieces(d, ps)?;
            }
            RetLoc::Sret => {
                self.in_home[v.index()] = true;
            }
        }
        Ok(())
    }

    /// The environment word (offset 8) of a fn value into `dst`.
    fn env_word(&mut self, e: MSrc, dst: Gpr) -> Res<()> {
        match e {
            MSrc::Mem(m) => Ok(self.a.mov_rm(Size::Q, dst, disp(m, 8)?)?),
            MSrc::Func(_) => {
                self.a.alu_rr(AluOp::Xor, Size::D, dst, dst);
                Ok(())
            }
            MSrc::Str(_) => Err(bad("string literal called as a fn value")),
        }
    }

    /// Loads an aggregate's eightbytes into registers.
    pub fn load_pieces(&mut self, s: MSrc, ps: &[Piece]) -> Res<()> {
        match s {
            MSrc::Mem(m) => {
                for (k, p) in ps.iter().enumerate() {
                    let pm = disp(m, 8 * k as i64)?;
                    match p {
                        Piece::G(g) => self.a.mov_rm(Size::Q, *g, pm)?,
                        Piece::X(x) => self.a.movs(Prec::F64, *x, pm)?,
                    }
                }
            }
            MSrc::Str(st) => {
                let sym = self.str_sym(st)?;
                let n = self.str_len(st)?;
                for (k, p) in ps.iter().enumerate() {
                    let Piece::G(g) = p else { return Err(bad("string literal in a vector register")) };
                    if k == 0 {
                        self.a.lea(Size::Q, *g, Mem::rip_sym(sym, 0))?;
                    } else {
                        self.a.mov_ri(Size::Q, *g, n)?;
                    }
                }
            }
            MSrc::Func(f) => {
                let sym = self.func_sym(f)?;
                for (k, p) in ps.iter().enumerate() {
                    let Piece::G(g) = p else { return Err(bad("fn value in a vector register")) };
                    if k == 0 {
                        self.a.lea(Size::Q, *g, Mem::rip_sym(sym, 0))?;
                    } else {
                        self.a.alu_rr(AluOp::Xor, Size::D, *g, *g);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn store_pieces(&mut self, d: Mem, ps: &[Piece]) -> Res<()> {
        for (k, p) in ps.iter().enumerate() {
            let pm = disp(d, 8 * k as i64)?;
            match p {
                Piece::G(g) => self.a.mov_mr(Size::Q, pm, *g)?,
                Piece::X(x) => self.a.movs_store(Prec::F64, pm, *x)?,
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- blocks

    fn prescan(&mut self, bi: u32) {
        for v in self.touched.drain(..) {
            if let Some(l) = self.last_use.get_mut(v as usize) {
                *l = NONE;
            }
        }
        let b = self.b;
        let Some(bd) = b.blocks.get(bi as usize) else { return };
        let nv = self.last_use.len();
        for (k, i) in bd.insts.indices().enumerate() {
            for &o in b.args(i) {
                if let Operand::Value(v) = o
                    && v.index() < nv
                {
                    self.last_use[v.index()] = k as u32;
                    self.touched.push(v.0);
                }
            }
        }
        let n = bd.insts.len;
        let (lu, touched) = (&mut self.last_use, &mut self.touched);
        analysis::for_term_operands(b, &bd.term, |o| {
            if let Operand::Value(v) = o
                && v.index() < nv
            {
                lu[v.index()] = n;
                touched.push(v.0);
            }
        });
    }

    fn line_for(&mut self, i: u32) {
        let site = self.b.insts.site.get(i as usize).copied().unwrap_or(Site::NONE).0;
        self.cur_site = site;
        if site != self.line_site
            && let Some(Some((file, line, col))) = self.cx.lines.get(site as usize)
        {
            self.a.line(*file, *line, *col);
            self.line_site = site;
        }
    }

    fn run(mut self, fn_site: Site) -> Res<FuncCode> {
        let b = self.b;
        let f = self.f;
        if self.an.rpo.first() != Some(&0) {
            return Err(bad("body has no entry block"));
        }
        let abi = self.abi_of(&f.params, f.ret)?;
        self.ret_abi = abi.ret.clone();
        let params = b.params(stave_ir::Block(0));
        if params.len() != abi.args.len() {
            return Err(bad("b0 params differ from the fn params"));
        }
        self.prescan(0);
        // Entry: params arrive per the ABI.
        if let RetLoc::Sret = abi.ret {
            let h = self.alloc_frame(8, 8)?;
            self.a.mov_mr(Size::Q, Mem::base(RBP, h), RDI)?;
            self.sret_home = Some(h);
        }
        for (&p, loc) in params.iter().zip(&abi.args) {
            let pi = p.index();
            if pi >= self.vreg.len() {
                return Err(bad("param out of range"));
            }
            match loc {
                Loc::Ignore => {
                    self.def_mem(p)?;
                }
                Loc::Reg(Piece::G(g)) => self.bind(p, R::G(*g))?,
                Loc::Reg(Piece::X(x)) => self.bind(p, R::X(*x))?,
                Loc::Regs(ps) => {
                    let d = self.def_mem(p)?;
                    self.store_pieces(d, ps)?;
                }
                Loc::Stack(off) => {
                    self.home[pi] = 16 + *off as i32;
                    self.has_home[pi] = true;
                    self.in_home[pi] = true;
                }
            }
        }
        self.glock = 0;
        self.xlock = 0;
        let rpo = std::mem::take(&mut self.an.rpo);
        for (k, &bi) in rpo.iter().enumerate() {
            let next = rpo.get(k + 1).copied();
            let Some(bd) = b.blocks.get(bi as usize) else { return Err(bad("block out of range")) };
            let l = self.labels[bi as usize];
            self.a.bind(l)?;
            if bi != 0 {
                self.reset_regs();
                self.prescan(bi);
                for &p in bd.params.of(&b.block_params) {
                    self.home(p)?;
                    if let Some(h) = self.in_home.get_mut(p.index()) {
                        *h = true;
                    }
                }
            }
            for (pos, i) in bd.insts.indices().enumerate() {
                self.pos = pos as u32;
                self.line_for(i);
                self.inst(i)?;
                self.end_inst(i);
            }
            self.pos = bd.insts.len;
            self.term(&bd.term, next)?;
        }
        self.an.rpo = rpo;
        self.finish(fn_site)
    }

    fn block_label(&self, b: stave_ir::Block) -> Res<Label> {
        self.labels.get(b.index()).copied().ok_or_else(|| bad("branch target out of range"))
    }

    fn jump_block(&mut self, b: stave_ir::Block, next: Option<u32>) -> Res<()> {
        if next == Some(b.0) {
            return Ok(());
        }
        let l = self.block_label(b)?;
        Ok(self.a.jmp(l)?)
    }

    fn has_copies(&self, t: &Target) -> bool {
        t.args.len > 0
    }

    /// Writes a branch's arguments into the target's param homes (parallel copy). Uses scratch
    /// registers and memory only, so the register cache is unchanged.
    fn edge(&mut self, t: &Target) -> Res<()> {
        let b = self.b;
        let params = b.params(t.block);
        let args = b.target_args(t);
        if params.len() != args.len() {
            return Err(bad("branch argument count differs from the target's params"));
        }
        // Sources that are params of the target (other than in place) are saved first.
        let mut saved: Vec<Option<i32>> = vec![None; args.len()];
        for (j, &o) in args.iter().enumerate() {
            if let Operand::Value(v) = o
                && params.contains(&v)
                && params[j] != v
            {
                let c = self.vcls(self.vty(v)?)?;
                let size = match c {
                    VCls::M { size, .. } => size.div_ceil(8) * 8,
                    _ => 8,
                };
                let tmp = self.alloc_frame(size, 16)?;
                self.copy_value_to(o, c, Mem::base(RBP, tmp))?;
                saved[j] = Some(tmp);
            }
        }
        for (j, (&p, &o)) in params.iter().zip(args).enumerate() {
            if o == Operand::Value(p) {
                continue;
            }
            let c = self.vcls(self.vty(p)?)?;
            let dst = self.hmem(p)?;
            match saved[j] {
                Some(tmp) => {
                    let src = Mem::base(RBP, tmp);
                    match c {
                        VCls::M { size, .. } => self.copy_mem(dst, MSrc::Mem(src), size)?,
                        _ => {
                            self.a.mov_rm(Size::Q, S0, src)?;
                            self.a.mov_mr(Size::Q, dst, S0)?;
                        }
                    }
                }
                None => self.copy_value_to(o, c, dst)?,
            }
        }
        Ok(())
    }

    /// Writes operand `o` (class `c`) to a home-format location (8 bytes for scalars) without
    /// changing the register cache.
    fn copy_value_to(&mut self, o: Operand, c: VCls, dst: Mem) -> Res<()> {
        match c {
            VCls::G { .. } => {
                if let Operand::Value(v) = o
                    && let Some(R::G(g)) = self.vreg.get(v.index()).copied().flatten()
                {
                    return Ok(self.a.mov_mr(Size::Q, dst, g)?);
                }
                if matches!(o, Operand::Const(_)) {
                    let x = self.const_bits(o).ok_or_else(|| bad("constant"))?;
                    return self.store_imm(dst, 8, x);
                }
                self.copy_to_g(o, S0)?;
                Ok(self.a.mov_mr(Size::Q, dst, S0)?)
            }
            VCls::X(p) => {
                if let Operand::Value(v) = o
                    && let Some(R::X(x)) = self.vreg.get(v.index()).copied().flatten()
                {
                    return Ok(self.a.movs_store(p, dst, x)?);
                }
                if matches!(o, Operand::Const(_)) {
                    let x = self.const_bits(o).ok_or_else(|| bad("constant"))?;
                    return self.store_imm(dst, 8, x);
                }
                if let Operand::Value(v) = o
                    && self.in_home.get(v.index()).copied().unwrap_or(false)
                {
                    self.a.mov_rm(Size::Q, S0, Mem::base(RBP, self.home[v.index()]))?;
                    return Ok(self.a.mov_mr(Size::Q, dst, S0)?);
                }
                Err(bad("internal: float value has no location"))
            }
            VCls::M { size, .. } => {
                let s = self.msrc(o)?;
                self.copy_mem(dst, s, size)
            }
        }
    }

    fn term(&mut self, t: &Term, next: Option<u32>) -> Res<()> {
        match *t {
            Term::Br(tg) => {
                self.edge(&tg)?;
                self.jump_block(tg.block, next)
            }
            Term::Cbr { cond, then, els } => {
                if let Some(c) = self.int_const(cond) {
                    let tg = if c != 0 { then } else { els };
                    self.edge(&tg)?;
                    return self.jump_block(tg.block, next);
                }
                let c = self.read_g(cond)?;
                self.a.test_rr(Size::D, c, c);
                let (tl, el) = (self.block_label(then.block)?, self.block_label(els.block)?);
                match (self.has_copies(&then), self.has_copies(&els)) {
                    (false, false) => {
                        if next == Some(then.block.0) {
                            self.a.jcc(Cond::E, el)?;
                        } else {
                            self.a.jcc(Cond::Ne, tl)?;
                            self.jump_block(els.block, next)?;
                        }
                    }
                    (false, true) => {
                        self.a.jcc(Cond::Ne, tl)?;
                        self.edge(&els)?;
                        self.jump_block(els.block, next)?;
                    }
                    (true, false) => {
                        self.a.jcc(Cond::E, el)?;
                        self.edge(&then)?;
                        self.jump_block(then.block, next)?;
                    }
                    (true, true) => {
                        let l = self.a.new_label();
                        self.a.jcc(Cond::E, l)?;
                        self.edge(&then)?;
                        self.a.jmp(tl)?;
                        self.a.bind(l)?;
                        self.edge(&els)?;
                        self.jump_block(els.block, next)?;
                    }
                }
                Ok(())
            }
            Term::Switch { scrut, cases, default } => self.switch(scrut, cases, default, next),
            Term::Ret(o) => {
                self.ret(o)?;
                let l = self.epilogue;
                Ok(self.a.jmp(l)?)
            }
            Term::Unreachable => {
                self.a.ud2();
                Ok(())
            }
        }
    }

    fn switch(
        &mut self,
        scrut: Operand,
        cases: stave_ir::ir::Span,
        default: Target,
        next: Option<u32>,
    ) -> Res<()> {
        let b = self.b;
        let cs = b.switch_cases(cases);
        if let Some(x) = self.int_const(scrut) {
            let tg = cs.iter().find(|c| c.value == x).map_or(default, |c| c.target);
            self.edge(&tg)?;
            return self.jump_block(tg.block, next);
        }
        let signed = matches!(self.osk(scrut), Some(Sk::Int(t)) if t.signed);
        let s = self.read_g(scrut)?;
        // (value, label, target needing copies)
        let mut list: Vec<(i64, Label, Option<Target>)> = Vec::with_capacity(cs.len());
        for c in cs {
            let (l, pending) = if self.has_copies(&c.target) {
                (self.a.new_label(), Some(c.target))
            } else {
                (self.block_label(c.target.block)?, None)
            };
            list.push((c.value as u64 as i64, l, pending));
        }
        if signed {
            list.sort_by_key(|x| x.0);
        } else {
            list.sort_by_key(|x| x.0 as u64);
        }
        let (dl, dpending) = if self.has_copies(&default) {
            (self.a.new_label(), Some(default))
        } else {
            (self.block_label(default.block)?, None)
        };
        let keys: Vec<(i64, Label)> = list.iter().map(|x| (x.0, x.1)).collect();
        self.switch_tree(s, &keys, dl, signed)?;
        for (_, l, p) in list {
            if let Some(tg) = p {
                self.a.bind(l)?;
                self.edge(&tg)?;
                let tl = self.block_label(tg.block)?;
                self.a.jmp(tl)?;
            }
        }
        if let Some(tg) = dpending {
            self.a.bind(dl)?;
            self.edge(&tg)?;
            let tl = self.block_label(tg.block)?;
            self.a.jmp(tl)?;
        }
        Ok(())
    }

    fn cmp_imm(&mut self, s: Gpr, v: i64) -> Res<()> {
        if let Ok(i) = i32::try_from(v) {
            self.a.alu_ri(AluOp::Cmp, Size::Q, s, i64::from(i))?;
        } else {
            self.a.mov_ri(Size::Q, S0, v)?;
            self.a.alu_rr(AluOp::Cmp, Size::Q, s, S0);
        }
        Ok(())
    }

    /// Binary search over sorted case values, then the default.
    fn switch_tree(&mut self, s: Gpr, cases: &[(i64, Label)], default: Label, signed: bool) -> Res<()> {
        if cases.len() <= 4 {
            for &(v, l) in cases {
                self.cmp_imm(s, v)?;
                self.a.jcc(Cond::E, l)?;
            }
            return Ok(self.a.jmp(default)?);
        }
        let mid = cases.len() / 2;
        let (v, l) = cases[mid];
        self.cmp_imm(s, v)?;
        self.a.jcc(Cond::E, l)?;
        let left = self.a.new_label();
        self.a.jcc(if signed { Cond::L } else { Cond::B }, left)?;
        self.switch_tree(s, &cases[mid + 1..], default, signed)?;
        self.a.bind(left)?;
        self.switch_tree(s, &cases[..mid], default, signed)
    }

    fn ret(&mut self, o: Option<Operand>) -> Res<()> {
        let Some(o) = o else { return Ok(()) };
        match self.ret_abi.clone() {
            RetLoc::None => Ok(()),
            RetLoc::Reg(Piece::G(g)) => self.copy_to_g(o, g),
            RetLoc::Reg(Piece::X(x)) => {
                let p = match self.ocls(o, self.f.ret)? {
                    VCls::X(p) => p,
                    _ => Prec::F64,
                };
                self.copy_to_x(o, p, x)
            }
            RetLoc::Regs(ps) => {
                let s = self.msrc(o)?;
                self.load_pieces(s, &ps)
            }
            RetLoc::Sret => {
                let h = self.sret_home.ok_or_else(|| bad("internal: no sret home"))?;
                let size = match self.ocls(o, self.f.ret)? {
                    VCls::M { size, .. } => size,
                    _ => return Err(bad("sret of a scalar")),
                };
                let s = self.msrc(o)?;
                self.a.mov_rm(Size::Q, RAX, Mem::base(RBP, h))?;
                self.copy_mem(Mem::base(RAX, 0), s, size)
            }
        }
    }

    // ---------------------------------------------------------------- prologue, epilogue

    fn finish(mut self, fn_site: Site) -> Res<FuncCode> {
        let locals = self.frame.div_ceil(16) * 16;
        let saved: Vec<Gpr> =
            CALLEE_SAVED.iter().copied().filter(|g| self.used_callee & bit(*g) != 0).collect();
        let total = (locals + 8 * saved.len() as u32).div_ceil(16) * 16;
        let save_off = |k: usize| -((locals + 8 * (k as u32 + 1)) as i32);
        let ep = self.epilogue;
        self.a.bind(ep)?;
        for (k, g) in saved.iter().enumerate() {
            self.a.mov_rm(Size::Q, *g, Mem::base(RBP, save_off(k)))?;
        }
        self.a.leave();
        self.a.ret();
        let traps = std::mem::take(&mut self.traps);
        let trap_sym = self.cx.trap_sym;
        for (k, site, l) in traps {
            self.a.bind(l)?;
            if let Some(Some((file, line, col))) = self.cx.lines.get(site as usize) {
                self.a.line(*file, *line, *col);
            }
            let kind = PanicKind::ALL.iter().position(|x| *x == k).unwrap_or(0) as i64;
            self.a.mov_ri(Size::D, RDI, kind)?;
            self.a.mov_ri(Size::D, RSI, i64::from(site))?;
            self.a.call_sym(trap_sym);
            self.a.ud2();
        }
        let body = self.a.finish()?;

        let mut p = Asm::new();
        if let Some(Some((file, line, col))) = self.cx.lines.get(fn_site.0 as usize) {
            p.line(*file, *line, *col);
        }
        p.push(RBP);
        p.mov_rr(Size::Q, RBP, RSP);
        if total > 0 {
            let pages = total / 4096;
            if pages == 0 {
                p.alu_ri(AluOp::Sub, Size::Q, RSP, i64::from(total))?;
            } else {
                // Probe page by page so a large frame cannot jump the guard page.
                if pages <= 8 {
                    for _ in 0..pages {
                        p.alu_ri(AluOp::Sub, Size::Q, RSP, 4096)?;
                        p.alu_mi(AluOp::Or, Size::Q, Mem::base(RSP, 0), 0)?;
                    }
                } else {
                    p.mov_ri(Size::Q, S0, i64::from(pages))?;
                    let top = p.new_label();
                    p.bind(top)?;
                    p.alu_ri(AluOp::Sub, Size::Q, RSP, 4096)?;
                    p.alu_mi(AluOp::Or, Size::Q, Mem::base(RSP, 0), 0)?;
                    p.alu_ri(AluOp::Sub, Size::Q, S0, 1)?;
                    p.jcc(Cond::Ne, top)?;
                }
                let rest = total % 4096;
                if rest > 0 {
                    p.alu_ri(AluOp::Sub, Size::Q, RSP, i64::from(rest))?;
                }
            }
        }
        for (k, g) in saved.iter().enumerate() {
            p.mov_mr(Size::Q, Mem::base(RBP, save_off(k)), *g)?;
        }
        let pro = p.finish()?;
        let shift = pro.code.len() as u32;
        let mut code = pro.code;
        code.extend_from_slice(&body.code);
        let mut relocs = pro.relocs;
        relocs.extend(body.relocs.into_iter().map(|mut r| {
            r.offset += shift;
            r
        }));
        let mut lines = pro.lines;
        lines.extend(body.lines.into_iter().map(|mut r| {
            r.offset += shift;
            r
        }));
        Ok(FuncCode { code, relocs, lines })
    }
}
