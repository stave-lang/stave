//! Per-function code generation: frame homes, the greedy local register cache, block and
//! terminator emission, calls, prologue/epilogue, and trap stubs. Instruction selection for the
//! individual ops is in `ops`, `cast`, and `intrin`.

use stave_ir::arith::IntTy;
use stave_ir::ir::{Target, ValueDef};
use stave_ir::layout::Layout;
use stave_ir::{
    Body, Const, FloatBits, Func, FuncId, Op, Operand, PanicKind, Site, StrId, Term, TypeId, TypeKey, Value,
};
use stave_obj::{FuncCode, SymbolId};
use stave_opt::regalloc;

use super::abi::{self, CallAbi, Loc, Piece, RetLoc, VCls};
use super::analysis::{self, Analysis, NONE};
use super::{Ctx, Error, Res, bad, unsup};
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
/// Release: call-clobbered registers a value may own for its whole life (the rest of the
/// allocatable ones always stay with the block-local cache; `rax`, `rcx`, `rdx` have fixed uses).
const OWN_G_SCRATCH: [Gpr; 2] = [Gpr::R9, Gpr::R8];
/// Release: `xmm6`..`xmm13` may be owned (all vector registers are call-clobbered).
const OWN_X_FIRST: u8 = 6;
const OWN_X_COUNT: u8 = 8;
const OUT_OF_G: &str = "internal: out of general registers";
const OUT_OF_X: &str = "internal: out of vector registers";

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

/// A place a parallel move reads or writes: a register or an 8-byte frame home (`rbp` offset).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PLoc {
    G(Gpr),
    X(Xmm),
    M(i32),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PSrc {
    Loc(PLoc),
    Imm(i64),
    Sym(SymbolId),
    /// The frame address `rbp + off`.
    Addr(i32),
}

struct PMove {
    dst: PLoc,
    src: PSrc,
    /// `Some`: a float-class value.
    prec: Option<Prec>,
}

fn ploc(r: R) -> PLoc {
    match r {
        R::G(g) => PLoc::G(g),
        R::X(x) => PLoc::X(x),
    }
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

/// How a conditional branch tests its condition.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Jc {
    /// One flag condition.
    Flag(Cond),
    /// Float `==` after `ucomis`: equal and ordered (`ZF` and not `PF`).
    FloatEq,
    /// Float `!=` after `ucomis`: not equal or unordered.
    FloatNe,
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
    /// Release code generation. The callee-saved registers in use are then known up front (the
    /// owned ones; the cache keeps to call-clobbered registers) and saved at fixed slots at the
    /// top of the frame, so the epilogue can be emitted at a `ret`.
    rel: bool,
    /// Release: the register a value owns from its definition to its last use (stave-opt
    /// `regalloc`). Empty in a debug build. An owned register is outside the block-local cache:
    /// its `Occ` is `Free` between instructions and `vreg` of its value stays set across blocks.
    own: Vec<Option<R>>,
    /// Registers owned by some value of this function; the cache never allocates them.
    gown: u16,
    xown: u16,
    /// General registers the cache does not allocate: the owned ones and, in a release build,
    /// every callee-saved one.
    gskip: u16,
    /// Release: the epilogue label is bound (the first `ret` emitted it).
    epilogue_bound: bool,
    /// Release: per owned value, the first and last block (emission index) of its interval. No
    /// other value has its register in those blocks.
    first: Vec<u32>,
    last: Vec<u32>,
    /// Emission index of the block being compiled.
    cur_at: u32,
    /// The register the current instruction's result should end up in (its owned register, or
    /// [`Self::edge_hint`]): the next temporary an op asks for lands there.
    hint: Option<R>,
    /// Release: uses per value, saturating. Empty in a debug build.
    pub(crate) uses: Vec<u8>,
    /// Release: per value, the address instruction (`addr.elem`, `addr.field`,
    /// `addr.enum_field`) that was not emitted because the one load or store reading the value
    /// uses it as its addressing mode; `NONE` otherwise. Empty in a debug build.
    pub(crate) folded: Vec<u32>,
    /// Release: the compare of the current block that its `cbr` emits (`cmp` + `jcc`, no bool
    /// value); `NONE` otherwise.
    fused: u32,
    /// The frame holds a 32-byte-aligned value: the prologue realigns the stack so that `rbp` is
    /// 16 above a multiple of 32, keeps the entry frame address at `[rbp - 8]`, and the
    /// epilogue restores `rsp` from there.
    pub(crate) realign: bool,
    /// A `ymm` register was written since the last `vzeroupper`.
    pub(crate) ydirty: bool,
    /// The function has 256-bit code (its trap stubs clear the upper halves before the call).
    pub(crate) yused: bool,
    /// Vector constants, emitted after the code and read `rip`-relative.
    pub(crate) vpool: Vec<(Label, Vec<u8>)>,
}

pub(crate) fn compile(cx: &Ctx<'_>, fid: FuncId) -> Res<FuncCode> {
    let f = cx.m.func(fid).ok_or_else(|| bad("unknown function"))?;
    let b = f.body.as_ref().ok_or_else(|| bad("function has no body"))?;
    // `cpuid` writes `rbx`, which a release build may have given to a value.
    let cpuid = b
        .insts
        .op
        .iter()
        .any(|op| matches!(op, Op::Intrinsic(id) if cx.m.intrinsic_name(*id) == Some("cpuid")));
    if cx.release && !cpuid {
        match compile_with(cx, fid, true) {
            // An instruction that needs more temporaries than the cache keeps next to the owned
            // registers: this function is compiled without them.
            Err(Error::Malformed { what, .. }) if what == OUT_OF_G || what == OUT_OF_X => {}
            r => return r,
        }
    }
    compile_with(cx, fid, false)
}

/// Release register ownership: the owned register per value, the masks of owned general and
/// vector registers, and the intervals.
struct Owned {
    reg: Vec<Option<R>>,
    gown: u16,
    xown: u16,
    first: Vec<u32>,
    last: Vec<u32>,
}

fn ownership(cx: &Ctx<'_>, b: &Body, an: &Analysis) -> Owned {
    let nv = b.values.len();
    let mut class = vec![regalloc::CLASS_NONE; nv];
    for (c, vd) in class.iter_mut().zip(&b.values) {
        *c = match cx.cls.get(vd.ty.index()).copied().flatten() {
            Some(VCls::G { .. }) => regalloc::CLASS_INT,
            Some(VCls::X(_)) => regalloc::CLASS_FLOAT,
            _ => regalloc::CLASS_NONE,
        };
    }
    let mut clobbers = vec![false; b.insts.len()];
    // Frame addresses (a `slot` and constant offsets into one) are rematerialized, not held.
    let mut frame_addr = vec![false; nv];
    for (i, &op) in b.insts.op.iter().enumerate() {
        let res = b.insts.result.get(i).copied().flatten().filter(|v| v.index() < nv);
        match op {
            Op::Call | Op::CallInd | Op::Intrinsic(_) | Op::Vec(_) => clobbers[i] = true,
            Op::Slot => {
                if let Some(v) = res {
                    frame_addr[v.index()] = true;
                }
            }
            Op::FieldAddr | Op::EnumFieldAddr | Op::ElemAddr => {
                let args = b.args(i as u32);
                let base = matches!(args.first(), Some(Operand::Value(p))
                    if frame_addr.get(p.index()).copied().unwrap_or(false));
                let fixed = op != Op::ElemAddr || args.get(1).is_some_and(|o| b.int_const(*o).is_some());
                if base
                    && fixed
                    && let Some(v) = res
                {
                    frame_addr[v.index()] = true;
                }
            }
            _ => {}
        }
    }
    for (c, fa) in class.iter_mut().zip(&frame_addr) {
        if *fa {
            *c = regalloc::CLASS_NONE;
        }
    }
    let saved = CALLEE_SAVED.len() as u8;
    let pools = regalloc::Pools {
        int_saved: saved,
        int_scratch: OWN_G_SCRATCH.len() as u8,
        float_saved: 0,
        float_scratch: OWN_X_COUNT,
    };
    let asg = regalloc::assign(b, &an.rpo, &class, &clobbers, pools);
    let (mut gown, mut xown) = (0u16, 0u16);
    let reg = asg
        .reg
        .iter()
        .zip(&class)
        .map(|(&r, &c)| {
            if r == regalloc::NO_REG {
                return None;
            }
            if c == regalloc::CLASS_FLOAT {
                let x = Xmm::masked(OWN_X_FIRST + r % OWN_X_COUNT);
                xown |= xbit(x);
                Some(R::X(x))
            } else {
                let g = if r < saved {
                    CALLEE_SAVED[usize::from(r)]
                } else {
                    OWN_G_SCRATCH[usize::from(r - saved) % OWN_G_SCRATCH.len()]
                };
                gown |= bit(g);
                Some(R::G(g))
            }
        })
        .collect();
    Owned { reg, gown, xown, first: asg.first, last: asg.last }
}

fn use_counts(b: &Body) -> Vec<u8> {
    let mut n = vec![0u8; b.values.len()];
    let mut add = |o: Operand| {
        if let Operand::Value(v) = o
            && let Some(c) = n.get_mut(v.index())
        {
            *c = c.saturating_add(1);
        }
    };
    b.operands.iter().for_each(|o| add(*o));
    for bd in &b.blocks {
        match bd.term {
            Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => add(o),
            _ => {}
        }
    }
    n
}

fn compile_with(cx: &Ctx<'_>, fid: FuncId, release: bool) -> Res<FuncCode> {
    let f = cx.m.func(fid).ok_or_else(|| bad("unknown function"))?;
    let b = f.body.as_ref().ok_or_else(|| bad("function has no body"))?;
    let mut an = analysis::analyze(b);
    let nv = b.values.len();
    if release {
        an.rpo = stave_opt::layout::order(b);
    }
    let Owned { reg: own, gown, xown, first, last } = if release {
        ownership(cx, b, &an)
    } else {
        Owned { reg: Vec::new(), gown: 0, xown: 0, first: Vec::new(), last: Vec::new() }
    };
    let uses = if release { use_counts(b) } else { Vec::new() };
    let callee_saved = CALLEE_SAVED.iter().fold(0, |m, g| m | bit(*g));
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
    let wide = |t: TypeId| match cx.cls.get(t.index()).copied().flatten() {
        Some(VCls::V { size }) => size > 16,
        Some(VCls::M { align, .. }) => align > 16,
        _ => false,
    };
    let wide_slot = |i: usize| {
        b.insts.op[i] == Op::Slot
            && matches!(b.insts.ty.get(i).copied().flatten().and_then(|t| cx.m.types.key(t)),
                Some(TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. })
                    if cx.layouts.get_layout(*inner).is_ok_and(|l| l.align > 16))
    };
    let realign = b.values.iter().any(|v| wide(v.ty)) || (0..b.insts.len()).any(wide_slot);
    let saved_bytes = 8 * (gown & callee_saved).count_ones();
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
        used_callee: gown & callee_saved,
        // the entry frame address of a realigned frame, then (release) the save slots of the
        // owned callee-saved registers
        frame: saved_bytes + if realign { 8 } else { 0 },
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
        rel: release,
        own,
        gown,
        xown,
        gskip: if release { gown | callee_saved } else { 0 },
        epilogue_bound: false,
        first,
        last,
        cur_at: 0,
        hint: None,
        uses,
        folded: if release { vec![NONE; nv] } else { Vec::new() },
        fused: NONE,
        realign,
        ydirty: false,
        yused: false,
        vpool: Vec::new(),
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
            .ok_or_else(|| {
                let text = self.cx.m.types.key_text(t);
                match self.key(t) {
                    Some(TypeKey::Vec { .. } | TypeKey::Mask { .. }) => unsup(format!(
                        "vector type {text}: sizeless, f16 / bf16 lanes, or wider than the target's {}-bit vector registers",
                        8 * self.cx.level.vector_bytes()
                    )),
                    _ => unsup(format!("values of type t{} ({text})", t.0)),
                }
            })
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

    /// Frame bytes at the returned `rbp` offset. Alignment above 16 needs a realigned frame.
    pub fn alloc_frame(&mut self, size: u32, align: u32) -> Res<i32> {
        let al = align.clamp(1, 32);
        let end = u64::from(self.frame) + u64::from(size);
        let end = if al == 32 {
            if !self.realign {
                return Err(bad("internal: a 32-byte-aligned value in a frame that is not realigned"));
            }
            // `rbp` is 16 above a multiple of 32
            (end + 16).div_ceil(32) * 32 - 16
        } else {
            end.div_ceil(u64::from(al)) * u64::from(al)
        };
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
            let (size, align) = self.vcls(self.vty(v)?)?.home();
            let h = self.alloc_frame(size, align)?;
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

    /// Keeps `o` alive up to position `at` of the current block (a later instruction reads it
    /// though it is not one of its operands).
    pub(crate) fn extend_use(&mut self, o: Operand, at: u32) {
        if let Operand::Value(v) = o
            && let Some(lu) = self.last_use.get_mut(v.index())
            && (*lu == NONE || *lu < at)
        {
            *lu = at;
            self.touched.push(v.0);
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

    /// The register `v` owns, if any.
    fn owned(&self, v: Value) -> Option<R> {
        self.own.get(v.index()).copied().flatten()
    }

    /// The hinted register as a temporary, when it is idle.
    fn take_hint(&mut self, want_x: bool) -> Option<R> {
        let h = self.hint?;
        if matches!(h, R::X(_)) != want_x || self.locked(h) || self.occ(h) != Occ::Free {
            return None;
        }
        self.set_occ(h, Occ::Temp);
        self.lock(h);
        Some(h)
    }

    pub fn alloc_g(&mut self) -> Res<Gpr> {
        if let Some(R::G(g)) = self.take_hint(false) {
            return Ok(g);
        }
        let mut best: Option<(u32, Gpr)> = None;
        for g in ALLOC_G {
            if self.gskip & bit(g) != 0 {
                continue;
            }
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
        let (_, g) = best.ok_or_else(|| bad(OUT_OF_G))?;
        self.evict(R::G(g))?;
        self.set_occ(R::G(g), Occ::Temp);
        self.lock(R::G(g));
        Ok(g)
    }

    pub fn alloc_x(&mut self) -> Res<Xmm> {
        if let Some(R::X(x)) = self.take_hint(true) {
            return Ok(x);
        }
        let mut best: Option<(u32, Xmm)> = None;
        for n in 0..NXMM {
            let x = Xmm::masked(n);
            if self.xown & xbit(x) != 0 {
                continue;
            }
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
        let (_, x) = best.ok_or_else(|| bad(OUT_OF_X))?;
        self.evict(R::X(x))?;
        self.set_occ(R::X(x), Occ::Temp);
        self.lock(R::X(x));
        Ok(x)
    }

    /// A cache register for an operand: never the hinted one, which the result is about to
    /// overwrite.
    fn alloc_g_cache(&mut self) -> Res<Gpr> {
        let h = self.hint.take();
        let r = self.alloc_g();
        self.hint = h;
        r
    }

    fn alloc_x_cache(&mut self) -> Res<Xmm> {
        let h = self.hint.take();
        let r = self.alloc_x();
        self.hint = h;
        r
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
        if self.an.escapes[vi] && self.owned(v).is_none() {
            self.store_home(v, r)?;
        }
        Ok(())
    }

    /// Moves the result `v` of the instruction just emitted into the register `rv` it owns and
    /// takes that register out of the cache.
    fn settle(&mut self, v: Value, rv: R) -> Res<()> {
        let vi = v.index();
        match self.vreg[vi] {
            Some(r) if r == rv => {}
            Some(r) => {
                if let Occ::Val(u) = self.occ(rv)
                    && u != v
                {
                    self.evict(rv)?;
                }
                match (rv, r) {
                    (R::G(d), R::G(s)) => self.a.mov_rr(Size::Q, d, s),
                    (R::X(d), R::X(s)) => self.a.movaps(d, s),
                    _ => return Err(bad("internal: result in a register of the wrong class")),
                }
                self.free_reg(r);
            }
            None if self.is_slot[vi] => {
                // a frame address after all: rematerialized, the register stays unused
                self.own[vi] = None;
                return Ok(());
            }
            None if self.in_home[vi] => {
                if let Occ::Val(_) = self.occ(rv) {
                    self.evict(rv)?;
                }
                self.load_val(v, rv)?;
                self.in_home[vi] = false;
            }
            // an unused pure op: nothing was emitted
            None => return Ok(()),
        }
        self.set_occ(rv, Occ::Free);
        self.vreg[vi] = Some(rv);
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

    /// Release, before a call: stores the cached values that are read after the call. The
    /// bindings stay, so the arguments can move straight out of their registers.
    fn save_for_call(&mut self) -> Res<()> {
        let regs = CALLER_SAVED.into_iter().map(R::G).chain((0..NXMM).map(|n| R::X(Xmm::masked(n))));
        for r in regs {
            match self.occ(r) {
                Occ::Free => {}
                Occ::Temp => return Err(bad("internal: a temporary register across a call")),
                Occ::Val(v) => {
                    if !self.in_home[v.index()] && self.needed_after(v) {
                        self.store_home(v, r)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Forgets what the call-clobbered registers held.
    fn drop_caller_saved(&mut self) {
        for g in CALLER_SAVED {
            self.free_reg(R::G(g));
        }
        for n in 0..NXMM {
            self.free_reg(R::X(Xmm::masked(n)));
        }
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

    fn end_inst(&mut self, i: u32) -> Res<()> {
        let pos = self.pos;
        let b = self.b;
        self.hint = None;
        if let Some(Some(v)) = b.insts.result.get(i as usize)
            && let Some(rv) = self.owned(*v)
        {
            self.settle(*v, rv)?;
        }
        for &o in b.args(i) {
            // an owned register is not the cache's to free (it may hold this result by now)
            if let Operand::Value(v) = o
                && self.last_use.get(v.index()).copied() == Some(pos)
                && self.owned(v).is_none()
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
        Ok(())
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
            let g = self.alloc_g_cache()?;
            self.load_val(v, R::G(g))?;
            self.set_occ(R::G(g), Occ::Val(v));
            self.vreg[vi] = Some(R::G(g));
            return Ok(g);
        }
        let g = self.alloc_g_cache()?;
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
            let x = self.alloc_x_cache()?;
            self.load_val(v, R::X(x))?;
            self.set_occ(R::X(x), Occ::Val(v));
            self.vreg[vi] = Some(R::X(x));
            return Ok(x);
        }
        let x = self.alloc_x_cache()?;
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
                && self.owned(v).is_none()
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
                && self.owned(v).is_none()
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
            VCls::M { size, .. } | VCls::V { size } => {
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
            VCls::M { size, .. } | VCls::V { size } => {
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
        // Release: register arguments are one parallel move from where the values are. (A code
        // address operand is read after the arguments are in place, so it takes the other way.)
        let direct = self.rel && !matches!(callee, Callee::Val(_));
        if direct {
            self.save_for_call()?;
        } else {
            self.spill_caller_saved()?;
        }
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
                    VCls::M { size, .. } | VCls::V { size } => {
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
        if direct {
            let mut moves = Vec::new();
            for (&o, loc) in args.iter().zip(&abi.args) {
                match loc {
                    Loc::Reg(Piece::G(g)) => {
                        moves.push(PMove { dst: PLoc::G(*g), src: self.psrc(o)?, prec: None })
                    }
                    Loc::Reg(Piece::X(x)) => {
                        let p = match self.ocls(o, None)? {
                            VCls::X(p) => p,
                            _ => Prec::F64,
                        };
                        moves.push(PMove { dst: PLoc::X(*x), src: self.psrc(o)?, prec: Some(p) });
                    }
                    _ => {}
                }
            }
            self.par_moves(moves)?;
            self.drop_caller_saved();
        }
        for (&o, loc) in args.iter().zip(&abi.args) {
            match loc {
                Loc::Reg(_) if direct => {}
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
                Loc::Vec(x) => {
                    let VCls::V { size } = self.ocls(o, None)? else {
                        return Err(bad("internal: a non-vector argument in a vector register"));
                    };
                    let MSrc::Mem(m) = self.msrc(o)? else { return Err(bad("vector argument")) };
                    self.vload(*x, m, size)?;
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
        // The callee returns with clean upper halves unless `ymm0` carries its result.
        self.ydirty = false;
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
            RetLoc::Vec => {
                let VCls::V { size } = self.vcls(self.vty(v)?)? else {
                    return Err(bad("internal: a non-vector result in a vector register"));
                };
                let d = self.def_mem(v)?;
                self.vstore(d, abi::xmm(0), size)?;
                self.vclean();
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
        self.fused = if self.rel { self.fusable(bd) } else { NONE };
        let n = bd.insts.len;
        for (k, i) in bd.insts.indices().enumerate() {
            // the operands of a fused compare are read by the terminator
            let at = if i == self.fused { n } else { k as u32 };
            for &o in b.args(i) {
                if let Operand::Value(v) = o
                    && v.index() < nv
                {
                    let lu = &mut self.last_use[v.index()];
                    if *lu == NONE || *lu < at {
                        *lu = at;
                    }
                    self.touched.push(v.0);
                }
            }
        }
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

    /// The compare instruction of block `bd` that its `cbr` can emit itself: defined in the
    /// block, used by nothing else, on scalars.
    fn fusable(&self, bd: &stave_ir::ir::BlockData) -> u32 {
        let b = self.b;
        let Term::Cbr { cond: Operand::Value(c), .. } = bd.term else { return NONE };
        let Some(ValueDef::Inst(j)) = b.values.get(c.index()).map(|v| v.def) else { return NONE };
        if !bd.insts.indices().contains(&j) || self.uses.get(c.index()).copied() != Some(1) {
            return NONE;
        }
        let op = b.insts.op.get(j as usize).copied();
        if !matches!(op, Some(Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge)) {
            return NONE;
        }
        let &[x, y] = b.args(j) else { return NONE };
        if self.is_fn_operand(x) || self.is_fn_operand(y) {
            return NONE;
        }
        match self.osk(x).or(self.osk(y)) {
            Some(Sk::Float(FloatBits::F32 | FloatBits::F64) | Sk::Int(_) | Sk::Bool | Sk::Char | Sk::Ptr) => {
                j
            }
            _ => NONE,
        }
    }

    /// The owned register of a param of the block's only successor, when the result of
    /// instruction `i` is nothing but that param's next value (a loop counter's increment): the
    /// op then computes into the param's register and the branch has nothing to move.
    ///
    /// Writing the register early is safe when the param's old value is dead: the block has one
    /// successor (which redefines the param), nothing after `i` reads the param, and the block
    /// lies in the param's interval, so no other value lives in that register here. Only ops that
    /// read all their operands before they write the result qualify.
    fn edge_hint(&self, bd: &stave_ir::ir::BlockData, i: u32) -> Option<R> {
        let b = self.b;
        let v = b.insts.result.get(i as usize).copied().flatten()?;
        if self.uses.get(v.index()).copied() != Some(1) {
            return None;
        }
        let Term::Br(t) = bd.term else { return None };
        let j = b.target_args(&t).iter().position(|o| *o == Operand::Value(v))?;
        let p = *b.params(t.block).get(j)?;
        let rp = self.owned(p)?;
        let pi = p.index();
        if self.cur_at < self.first.get(pi).copied()? || self.cur_at > self.last.get(pi).copied()? {
            return None;
        }
        let lu = self.last_use.get(pi).copied().unwrap_or(NONE);
        if lu != NONE && lu > self.pos {
            return None;
        }
        let simple = matches!(
            b.insts.op.get(i as usize)?,
            Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Neg
                | Op::AddWrap
                | Op::SubWrap
                | Op::MulWrap
                | Op::And
                | Op::Or
                | Op::Xor
                | Op::Not
                | Op::Shl
                | Op::Shr
                | Op::ShlWrap
        );
        simple.then_some(rp)
    }

    /// `jcc` to `l` when the condition holds.
    fn jump_if(&mut self, jc: Jc, l: Label) -> Res<()> {
        match jc {
            Jc::Flag(c) => self.a.jcc(c, l)?,
            Jc::FloatEq => {
                let skip = self.a.new_label();
                self.a.jcc(Cond::P, skip)?;
                self.a.jcc(Cond::E, l)?;
                self.a.bind(skip)?;
            }
            Jc::FloatNe => {
                self.a.jcc(Cond::P, l)?;
                self.a.jcc(Cond::Ne, l)?;
            }
        }
        Ok(())
    }

    /// `jcc` to `l` when the condition does not hold.
    fn jump_unless(&mut self, jc: Jc, l: Label) -> Res<()> {
        match jc {
            Jc::Flag(c) => Ok(self.a.jcc(c.negate(), l)?),
            Jc::FloatEq => self.jump_if(Jc::FloatNe, l),
            Jc::FloatNe => self.jump_if(Jc::FloatEq, l),
        }
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
        let mut entry_base = false;
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
                Loc::Vec(x) => {
                    let VCls::V { size } = self.vcls(self.vty(p)?)? else {
                        return Err(bad("internal: a non-vector param in a vector register"));
                    };
                    let d = self.def_mem(p)?;
                    self.vstore(d, *x, size)?;
                }
                // A realigned frame is at no fixed distance from the stack arguments: they are
                // copied into homes (`rax` carries nothing at entry).
                Loc::Stack(off) if self.realign => {
                    let size = match self.vcls(self.vty(p)?)? {
                        VCls::M { size, .. } | VCls::V { size } => size,
                        _ => 8,
                    };
                    if !entry_base {
                        self.a.mov_rm(Size::Q, RAX, Mem::base(RBP, -8))?;
                        entry_base = true;
                    }
                    let d = self.def_mem(p)?;
                    self.copy_mem(d, MSrc::Mem(Mem::base(RAX, 16 + *off as i32)), size)?;
                }
                Loc::Stack(off) => {
                    self.home[pi] = 16 + *off as i32;
                    self.has_home[pi] = true;
                    self.in_home[pi] = true;
                }
            }
        }
        self.vclean();
        self.glock = 0;
        self.xlock = 0;
        if self.rel {
            self.settle_params(params)?;
        }
        let rpo = std::mem::take(&mut self.an.rpo);
        for (k, &bi) in rpo.iter().enumerate() {
            self.cur_at = k as u32;
            let next = rpo.get(k + 1).copied();
            let Some(bd) = b.blocks.get(bi as usize) else { return Err(bad("block out of range")) };
            let l = self.labels[bi as usize];
            self.a.bind(l)?;
            if bi != 0 {
                self.reset_regs();
                self.prescan(bi);
                for &p in bd.params.of(&b.block_params) {
                    if let Some(r) = self.owned(p) {
                        // written by the branches into this block
                        self.vreg[p.index()] = Some(r);
                        continue;
                    }
                    self.home(p)?;
                    if let Some(h) = self.in_home.get_mut(p.index()) {
                        *h = true;
                    }
                }
            }
            for (pos, i) in bd.insts.indices().enumerate() {
                self.pos = pos as u32;
                self.line_for(i);
                if self.rel {
                    let res = b.insts.result.get(i as usize).copied().flatten();
                    self.hint = res.and_then(|v| self.owned(v)).or_else(|| self.edge_hint(bd, i));
                }
                if i != self.fused && !(self.rel && self.fold_addr(i, bd)?) {
                    self.inst(i)?;
                }
                self.end_inst(i)?;
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
        if !self.rel {
            return t.args.len > 0;
        }
        // Release: an argument already where the param lives (its own register) moves nothing.
        let b = self.b;
        let (params, args) = (b.params(t.block), b.target_args(t));
        params.len() != args.len()
            || params.iter().zip(args).any(|(&p, &o)| {
                if o == Operand::Value(p) {
                    return false;
                }
                match (self.owned(p), self.psrc(o)) {
                    (Some(r), Ok(src)) => src != PSrc::Loc(ploc(r)),
                    _ => true,
                }
            })
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
        if self.rel {
            return self.edge_moves(params, args);
        }
        // Sources that are params of the target (other than in place) are saved first.
        let mut saved: Vec<Option<i32>> = vec![None; args.len()];
        for (j, &o) in args.iter().enumerate() {
            if let Operand::Value(v) = o
                && params.contains(&v)
                && params[j] != v
            {
                let c = self.vcls(self.vty(v)?)?;
                let tmp = self.alloc_frame(c.home().0, 16)?;
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
                        VCls::M { size, .. } | VCls::V { size } => {
                            self.copy_mem(dst, MSrc::Mem(src), size)?
                        }
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

    /// Release form of [`Self::edge`]: aggregates go home to home; scalars are one parallel move
    /// into the params' owned registers and homes. Uses scratch registers only.
    fn edge_moves(&mut self, params: &[Value], args: &[Operand]) -> Res<()> {
        // An aggregate source that is another param of the target is saved first.
        let mut saved: Vec<Option<i32>> = vec![None; args.len()];
        for (j, &o) in args.iter().enumerate() {
            if let Operand::Value(v) = o
                && params[j] != v
                && let VCls::M { size, .. } | VCls::V { size } = self.vcls(self.vty(v)?)?
                && params.contains(&v)
            {
                let tmp = self.alloc_frame(size.div_ceil(8) * 8, 16)?;
                let src = self.msrc(o)?;
                self.copy_mem(Mem::base(RBP, tmp), src, size)?;
                saved[j] = Some(tmp);
            }
        }
        let mut moves = Vec::new();
        for (j, (&p, &o)) in params.iter().zip(args).enumerate() {
            if o == Operand::Value(p) {
                continue;
            }
            match self.vcls(self.vty(p)?)? {
                VCls::M { size, .. } | VCls::V { size } => {
                    let dst = self.hmem(p)?;
                    let src = match saved[j] {
                        Some(tmp) => MSrc::Mem(Mem::base(RBP, tmp)),
                        None => self.msrc(o)?,
                    };
                    self.copy_mem(dst, src, size)?;
                }
                c => {
                    let dst = match self.owned(p) {
                        Some(r) => ploc(r),
                        None => PLoc::M(self.home(p)?),
                    };
                    let prec = match c {
                        VCls::X(p) => Some(p),
                        _ => None,
                    };
                    moves.push(PMove { dst, src: self.psrc(o)?, prec });
                }
            }
        }
        self.par_moves(moves)
    }

    /// Where a scalar operand is right now (no cache change).
    fn psrc(&self, o: Operand) -> Res<PSrc> {
        match o {
            Operand::Value(v) => {
                let vi = v.index();
                if let Some(r) = self.vreg.get(vi).copied().flatten() {
                    return Ok(PSrc::Loc(ploc(r)));
                }
                if !self.in_home.get(vi).copied().unwrap_or(false) {
                    return Err(bad(format!("internal: %{} has no location", v.0)));
                }
                Ok(if self.is_slot[vi] {
                    PSrc::Addr(self.home[vi])
                } else {
                    PSrc::Loc(PLoc::M(self.home[vi]))
                })
            }
            Operand::Const(_) => Ok(PSrc::Imm(self.const_bits(o).ok_or_else(|| bad("constant"))?)),
            Operand::Func(f) => Ok(PSrc::Sym(self.func_sym(f)?)),
            Operand::Str(_) => Err(bad("string literal where a scalar is expected")),
            Operand::Type(_) => Err(bad("type operand used as a value")),
        }
    }

    /// Emits the moves as one parallel move: every source is read before its place is written.
    /// A cycle is broken by parking one source in `r10` / `xmm14`; memory-to-memory moves and
    /// wide immediates go through `r11`. Flags are not preserved.
    fn par_moves(&mut self, mut ms: Vec<PMove>) -> Res<()> {
        ms.retain(|m| m.src != PSrc::Loc(m.dst));
        while !ms.is_empty() {
            let read = |ms: &[PMove], l: PLoc, but: usize| {
                ms.iter().enumerate().any(|(j, o)| j != but && o.src == PSrc::Loc(l))
            };
            if let Some(k) = (0..ms.len()).find(|&k| !read(&ms, ms[k].dst, k)) {
                let m = ms.swap_remove(k);
                self.emit_move(m.dst, m.src, m.prec)?;
                continue;
            }
            // Only cycles are left: every destination is still read. Park one of them.
            let l = ms[0].dst;
            let prec = ms[0].prec;
            let tmp = if prec.is_some() { PLoc::X(XS1) } else { PLoc::G(S1) };
            self.emit_move(tmp, PSrc::Loc(l), prec)?;
            for o in ms.iter_mut() {
                if o.src == PSrc::Loc(l) {
                    o.src = PSrc::Loc(tmp);
                }
            }
        }
        Ok(())
    }

    fn emit_move(&mut self, dst: PLoc, src: PSrc, prec: Option<Prec>) -> Res<()> {
        let p = prec.unwrap_or(Prec::F64);
        let at = |off: i32| Mem::base(RBP, off);
        match (dst, src) {
            (PLoc::G(d), PSrc::Loc(PLoc::G(s))) => self.a.mov_rr(Size::Q, d, s),
            (PLoc::G(d), PSrc::Loc(PLoc::M(o))) => self.a.mov_rm(Size::Q, d, at(o))?,
            (PLoc::G(d), PSrc::Loc(PLoc::X(s))) => self.a.mov_gx(Size::Q, d, s)?,
            (PLoc::G(d), PSrc::Imm(0)) => self.a.alu_rr(AluOp::Xor, Size::D, d, d),
            (PLoc::G(d), PSrc::Imm(x)) => self.a.mov_ri(Size::Q, d, x)?,
            (PLoc::G(d), PSrc::Sym(s)) => self.a.lea(Size::Q, d, Mem::rip_sym(s, 0))?,
            (PLoc::G(d), PSrc::Addr(o)) => self.a.lea(Size::Q, d, at(o))?,
            (PLoc::X(d), PSrc::Loc(PLoc::X(s))) => self.a.movaps(d, s),
            (PLoc::X(d), PSrc::Loc(PLoc::M(o))) => self.a.movs(p, d, at(o))?,
            (PLoc::X(d), PSrc::Loc(PLoc::G(s))) => self.a.mov_xg(Size::Q, d, s)?,
            (PLoc::X(d), PSrc::Imm(0)) => self.a.logic_p(PLogic::Xor, p, d, d)?,
            (PLoc::X(d), PSrc::Imm(x)) => {
                self.a.mov_ri(Size::Q, S0, x)?;
                self.a.mov_xg(Size::Q, d, S0)?;
            }
            (PLoc::M(o), PSrc::Loc(PLoc::G(s))) => self.a.mov_mr(Size::Q, at(o), s)?,
            (PLoc::M(o), PSrc::Loc(PLoc::X(s))) => self.a.movs_store(p, at(o), s)?,
            (PLoc::M(o), PSrc::Loc(PLoc::M(o2))) => {
                self.a.mov_rm(Size::Q, S0, at(o2))?;
                self.a.mov_mr(Size::Q, at(o), S0)?;
            }
            (PLoc::M(o), PSrc::Imm(x)) => self.store_imm(at(o), 8, x)?,
            (PLoc::M(o), PSrc::Sym(s)) => {
                self.a.lea(Size::Q, S0, Mem::rip_sym(s, 0))?;
                self.a.mov_mr(Size::Q, at(o), S0)?;
            }
            (PLoc::M(o), PSrc::Addr(o2)) => {
                self.a.lea(Size::Q, S0, at(o2))?;
                self.a.mov_mr(Size::Q, at(o), S0)?;
            }
            (PLoc::X(_), PSrc::Sym(_) | PSrc::Addr(_)) => {
                return Err(bad("internal: an address moved into a vector register"));
            }
        }
        Ok(())
    }

    /// Release: moves the entry params that own a register out of their ABI locations, and the
    /// other params out of registers owned by some value.
    fn settle_params(&mut self, params: &[Value]) -> Res<()> {
        for n in 0..16u8 {
            for r in [Gpr::new(n).map(R::G), Some(R::X(Xmm::masked(n)))].into_iter().flatten() {
                let is_owned = match r {
                    R::G(g) => self.gown & bit(g) != 0,
                    R::X(x) => self.xown & xbit(x) != 0,
                };
                if is_owned
                    && let Occ::Val(q) = self.occ(r)
                    && self.owned(q).is_none()
                {
                    self.evict(r)?;
                }
            }
        }
        let mut moves = Vec::new();
        for &p in params {
            let Some(rp) = self.owned(p) else { continue };
            let pi = p.index();
            let src = match self.vreg[pi] {
                Some(r) => {
                    self.set_occ(r, Occ::Free);
                    ploc(r)
                }
                None if self.in_home[pi] => PLoc::M(self.home[pi]),
                None => continue,
            };
            let prec = match self.vcls(self.vty(p)?)? {
                VCls::X(p) => Some(p),
                _ => None,
            };
            moves.push(PMove { dst: ploc(rp), src: PSrc::Loc(src), prec });
            self.vreg[pi] = Some(rp);
            self.in_home[pi] = false;
        }
        self.par_moves(moves)
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
            VCls::M { size, .. } | VCls::V { size } => {
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
                let jc = if self.fused == NONE {
                    let c = self.read_g(cond)?;
                    self.a.test_rr(Size::D, c, c);
                    Jc::Flag(Cond::Ne)
                } else {
                    let b = self.b;
                    let op = b.insts.op.get(self.fused as usize).copied();
                    match (op, b.args(self.fused)) {
                        (Some(op), &[x, y]) => self.compare_flags(op, x, y)?,
                        _ => return Err(bad("internal: fused compare")),
                    }
                };
                let (tl, el) = (self.block_label(then.block)?, self.block_label(els.block)?);
                match (self.has_copies(&then), self.has_copies(&els)) {
                    (false, false) => {
                        if next == Some(then.block.0) {
                            self.jump_unless(jc, el)?;
                        } else {
                            self.jump_if(jc, tl)?;
                            self.jump_block(els.block, next)?;
                        }
                    }
                    (false, true) => {
                        self.jump_if(jc, tl)?;
                        self.edge(&els)?;
                        self.jump_block(els.block, next)?;
                    }
                    (true, false) => {
                        self.jump_unless(jc, el)?;
                        self.edge(&then)?;
                        self.jump_block(then.block, next)?;
                    }
                    (true, true) => {
                        let l = self.a.new_label();
                        self.jump_unless(jc, l)?;
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
                // Release: the first `ret` carries the epilogue, and so does every other one
                // while the epilogue is as short as a jump to it.
                if self.rel && (!self.epilogue_bound || self.used_callee.count_ones() <= 1) {
                    return self.emit_epilogue();
                }
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
            // a 32-byte result leaves in `ymm0`: its upper half stays as loaded
            RetLoc::Vec => {
                let VCls::V { size } = self.ocls(o, self.f.ret)? else {
                    return Err(bad("internal: a non-vector result in a vector register"));
                };
                let MSrc::Mem(m) = self.msrc(o)? else { return Err(bad("vector result")) };
                self.vload(abi::xmm(0), m, size)?;
                // control leaves here: the code emitted next starts clean
                self.ydirty = false;
                Ok(())
            }
            RetLoc::Sret => {
                let h = self.sret_home.ok_or_else(|| bad("internal: no sret home"))?;
                let size = match self.ocls(o, self.f.ret)? {
                    VCls::M { size, .. } => size,
                    _ => return Err(bad("sret of a non-aggregate")),
                };
                let s = self.msrc(o)?;
                self.a.mov_rm(Size::Q, RAX, Mem::base(RBP, h))?;
                self.copy_mem(Mem::base(RAX, 0), s, size)
            }
        }
    }

    // ---------------------------------------------------------------- prologue, epilogue

    /// Release: the epilogue, here (the save slots are fixed and `leave` needs no frame size).
    fn emit_epilogue(&mut self) -> Res<()> {
        if !self.epilogue_bound {
            let ep = self.epilogue;
            self.a.bind(ep)?;
            self.epilogue_bound = true;
        }
        let saved = CALLEE_SAVED.iter().copied().filter(|g| self.used_callee & bit(*g) != 0);
        let first = if self.realign { 8 } else { 0 };
        for (k, g) in saved.enumerate() {
            self.a.mov_rm(Size::Q, g, Mem::base(RBP, -(first + 8 * (k as i32 + 1))))?;
        }
        self.leave_ret()
    }

    /// Drops the frame and returns. A realigned frame is left through the entry frame address.
    fn leave_ret(&mut self) -> Res<()> {
        if self.realign {
            self.a.mov_rm(Size::Q, RSP, Mem::base(RBP, -8))?;
            self.a.pop(RBP);
        } else {
            self.a.leave();
        }
        self.a.ret();
        Ok(())
    }

    fn finish(mut self, fn_site: Site) -> Res<FuncCode> {
        let saved: Vec<Gpr> =
            CALLEE_SAVED.iter().copied().filter(|g| self.used_callee & bit(*g) != 0).collect();
        // Debug: the registers the cache happened to use are saved below the locals. Release:
        // the owned ones, at the top of the frame.
        let (total, first_save) = if self.rel {
            (self.frame.div_ceil(16) * 16, if self.realign { 8 } else { 0 })
        } else {
            let locals = self.frame.div_ceil(16) * 16;
            ((locals + 8 * saved.len() as u32).div_ceil(16) * 16, locals)
        };
        let save_off = |k: usize| -((first_save + 8 * (k as u32 + 1)) as i32);
        let ep = self.epilogue;
        if !self.rel {
            self.a.bind(ep)?;
            for (k, g) in saved.iter().enumerate() {
                self.a.mov_rm(Size::Q, *g, Mem::base(RBP, save_off(k)))?;
            }
            self.leave_ret()?;
        } else if !self.epilogue_bound {
            // no `ret` in the function
            self.a.bind(ep)?;
        }
        let traps = std::mem::take(&mut self.traps);
        let trap_sym = self.cx.trap_sym;
        for (k, site, l) in traps {
            self.a.bind(l)?;
            if let Some(Some((file, line, col))) = self.cx.lines.get(site as usize) {
                self.a.line(*file, *line, *col);
            }
            let kind = PanicKind::ALL.iter().position(|x| *x == k).unwrap_or(0) as i64;
            if self.yused {
                self.a.vzeroupper();
            }
            self.a.mov_ri(Size::D, RDI, kind)?;
            self.a.mov_ri(Size::D, RSI, i64::from(site))?;
            self.a.call_sym(trap_sym);
            self.a.ud2();
        }
        for (l, bytes) in std::mem::take(&mut self.vpool) {
            self.a.bind(l)?;
            self.a.raw(&bytes);
        }
        let body = self.a.finish()?;

        let mut p = Asm::new();
        if let Some(Some((file, line, col))) = self.cx.lines.get(fn_site.0 as usize) {
            p.line(*file, *line, *col);
        }
        p.push(RBP);
        if self.realign {
            // The return address and the caller's `rbp` are copied below the aligned boundary,
            // so the frame chain reads as in any other function.
            p.mov_rr(Size::Q, S1, RSP);
            p.alu_ri(AluOp::And, Size::Q, RSP, -32)?;
            p.alu_ri(AluOp::Sub, Size::Q, RSP, 16)?;
            for off in [8, 0] {
                p.mov_rm(Size::Q, S0, Mem::base(S1, off))?;
                p.mov_mr(Size::Q, Mem::base(RSP, off), S0)?;
            }
        }
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
        if self.realign {
            p.mov_mr(Size::Q, Mem::base(RBP, -8), S1)?;
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
