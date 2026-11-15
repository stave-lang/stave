//! Reference interpreter: the semantic oracle for differential tests.
//!
//! Executes a verified [`Module`] from an entry function over a simulated [`Memory`], with the
//! exact trap semantics (returns [`Outcome::Trap`] with the kind,
//! the trapping instruction's [`Site`], and the function), wrapping and saturating ops, the `as`
//! family, and IEEE f32 / f64 via Rust's native operations (round to nearest even).
//! System calls and other intrinsics go to a pluggable [`Host`].
//!
//! Values ([`Val`]): scalars are raw `u64` bits (ints normalized to their type, floats as IEEE
//! bits, `bool` 0/1, `char` code point, pointers and fn values as addresses / function indices);
//! fat pointers are `(address, length)`; tuples, structs and arrays are `Agg` of their fields;
//! an enum is `Agg[tag, fields of that variant..]`; a vector or mask is `Agg` of its lanes (mask
//! lanes 0 / 1).
//!
//! The interpreter never recurses on the host stack per Stave call (explicit frame stack), and
//! malformed input yields an [`InterpError`], never a panic.

use std::rc::Rc;

mod simd;

use crate::arith::{self, IntTy};
use crate::host::{Host, SysResult};
use crate::ir::{
    Block, Body, CastKind, Const, FuncId, Module, Op, Operand, PanicKind, Site, Target, Term, Value,
};
use crate::layout::{LayoutError, Layouts, Shape};
use crate::mem::{Fault, Memory};
use crate::types::{FloatBits, MAX_TYPE_DEPTH, TypeId};
use crate::verify::{self, Cls, VerifyError};

/// A runtime value.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    S(u64),
    Pair(u64, u64),
    Agg(Rc<[Val]>),
}

impl Val {
    pub fn agg(v: Vec<Val>) -> Val {
        Val::Agg(v.into())
    }
    pub fn unit() -> Val {
        Val::Agg(Rc::from(Vec::new()))
    }
    pub fn i64(x: i64) -> Val {
        Val::S(x as u64)
    }
    pub fn bool(b: bool) -> Val {
        Val::S(b as u64)
    }
    pub fn f32(x: f32) -> Val {
        Val::S(x.to_bits() as u64)
    }
    pub fn f64(x: f64) -> Val {
        Val::S(x.to_bits())
    }
    /// The scalar bits, if this is a scalar.
    pub fn bits(&self) -> Option<u64> {
        match self {
            Val::S(x) => Some(*x),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        self.bits().map(|b| f32::from_bits(b as u32))
    }
    pub fn as_f64(&self) -> Option<f64> {
        self.bits().map(f64::from_bits)
    }
    pub fn fields(&self) -> Option<&[Val]> {
        match self {
            Val::Agg(v) => Some(v),
            _ => None,
        }
    }
}

/// Limits of one run.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Instructions + terminators executed before [`InterpError::Fuel`].
    pub max_steps: u64,
    /// Call depth at which a call traps `StackOverflow`.
    pub max_depth: usize,
    /// Stack bytes (slots) at which a slot traps `StackOverflow`.
    pub stack_size: usize,
    /// Heap bytes (`mmap`) the host may allocate.
    pub heap_limit: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config { max_steps: u64::MAX, max_depth: 10_000, stack_size: 8 << 20, heap_limit: 1 << 30 }
    }
}

/// A trap (panic): kind, the trapping instruction's site, the function containing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trap {
    pub kind: PanicKind,
    pub site: Site,
    pub func: FuncId,
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The entry function returned.
    Return(Option<Val>),
    /// `exit` / `exit_group` with this status.
    Exit(i32),
    Trap(Trap),
}

/// A run that could not complete (not a Stave-level outcome).
#[derive(Clone, Debug, PartialEq)]
pub enum InterpError {
    Verify(Vec<VerifyError>),
    UnknownFunc(FuncId),
    /// The function has no body (declared only).
    NoBody(FuncId),
    ArgCount {
        func: FuncId,
        got: usize,
        want: usize,
    },
    /// `max_steps` exhausted.
    Fuel,
    /// An access outside any live region (a native run would fault).
    Fault {
        fault: Fault,
        func: FuncId,
        site: Site,
    },
    Unreachable {
        func: FuncId,
        block: Block,
    },
    /// `enum.get` / `addr.enum_field` read of a variant the value does not hold, or a bad tag.
    WrongVariant {
        func: FuncId,
        site: Site,
    },
    /// A fn value that names no function.
    BadFnValue(u64),
    Layout {
        ty: TypeId,
        err: LayoutError,
    },
    /// An operation the interpreter does not support yet (vector ops, f16 arithmetic).
    Unsupported(String),
    Host(String),
    /// Inconsistent IR that the verifier did not reject (an interpreter bug).
    Malformed(&'static str),
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for InterpError {}

struct Frame {
    func: FuncId,
    regs: Vec<Val>,
    block: u32,
    pc: u32,
    mark: usize,
    dest: Option<Value>,
}

enum Step {
    Next,
    Call(FuncId, Vec<Val>, Option<Value>),
    Trap(PanicKind),
    Exit(i32),
}

type R<T> = Result<T, InterpError>;

const MAX_AGG_LEN: u64 = 1 << 24;

/// The interpreter over one module.
pub struct Interp<'m, H: Host> {
    m: &'m Module,
    cls: Vec<Cls>,
    pointee: Vec<Option<TypeId>>,
    layouts: Layouts,
    strs: Vec<u64>,
    /// Address of each `static mut` (`Module::statics`).
    statics: Vec<u64>,
    pub host: H,
    pub mem: Memory,
    cfg: Config,
    steps: u64,
    scratch: Vec<Val>,
    /// The vector op of each intrinsic id.
    vops: Vec<Option<crate::simd::VecOp>>,
}

fn malformed(s: &'static str) -> InterpError {
    InterpError::Malformed(s)
}

impl<'m, H: Host> Interp<'m, H> {
    /// Verifies `m` and prepares memory (string literals in the read-only data region).
    pub fn new(m: &'m Module, host: H, cfg: Config) -> R<Interp<'m, H>> {
        verify::verify_module(m).map_err(InterpError::Verify)?;
        let n = m.types.len();
        let cls = (0..n).map(|i| verify::classify(&m.types, TypeId(i as u32))).collect();
        let pointee = (0..n).map(|i| verify::pointee(&m.types, TypeId(i as u32))).collect();
        let mut mem = Memory::new(cfg.stack_size, cfg.heap_limit);
        let strs = m.strings.iter().map(|s| mem.add_data(s)).collect();
        let layouts = Layouts::compute(&m.types);
        let mut statics = Vec::with_capacity(m.statics.len());
        for st in &m.statics {
            let l = layouts.get_layout(st.ty).map_err(|err| InterpError::Layout { ty: st.ty, err })?;
            let oom = || InterpError::Host("static data: out of memory".into());
            let a = mem.alloc(l.size.max(1), l.align).ok_or_else(oom)?;
            let n = (st.init.len() as u64).min(l.size) as usize;
            mem.write(a, &st.init[..n]).map_err(|_| oom())?;
            statics.push(a);
        }
        let vops = m.vec_ops();
        Ok(Interp {
            m,
            cls,
            pointee,
            layouts,
            strs,
            statics,
            host,
            mem,
            cfg,
            steps: 0,
            scratch: Vec::new(),
            vops,
        })
    }

    /// Instructions + terminators executed so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn into_host(self) -> H {
        self.host
    }

    /// Address of `static mut` number `i` in simulated memory.
    pub fn static_addr(&self, i: usize) -> Option<u64> {
        self.statics.get(i).copied()
    }

    /// Address of a string literal in simulated memory.
    pub fn string_addr(&self, s: crate::ir::StrId) -> Option<u64> {
        self.strs.get(s.index()).copied()
    }

    /// Builds a Linux-style initial process stack (argc, argv.., 0, envp.., 0, AT_NULL) in heap
    /// memory and returns its address: the `sp` for `runtime.start.entry(sp)`.
    pub fn process_stack(&mut self, args: &[&[u8]], env: &[&[u8]]) -> R<u64> {
        let oom = || InterpError::Host("process stack: out of memory".into());
        let mut ptrs = Vec::new();
        for s in args.iter().chain(env) {
            let a = self.mem.alloc(s.len() as u64 + 1, 1).ok_or_else(oom)?;
            self.mem.write(a, s).map_err(|_| oom())?;
            ptrs.push(a);
        }
        let mut words = vec![args.len() as u64];
        words.extend_from_slice(&ptrs[..args.len()]);
        words.push(0);
        words.extend_from_slice(&ptrs[args.len()..]);
        words.extend_from_slice(&[0, 0, 0]);
        let sp = self.mem.alloc(words.len() as u64 * 8, 16).ok_or_else(oom)?;
        for (k, w) in words.iter().enumerate() {
            self.mem.write_uint(sp + 8 * k as u64, 8, *w).map_err(|_| oom())?;
        }
        Ok(sp)
    }

    fn cls(&self, t: TypeId) -> Cls {
        self.cls.get(t.index()).copied().unwrap_or(Cls::Other)
    }

    fn push_frame(&mut self, frames: &mut Vec<Frame>, f: FuncId, args: Vec<Val>) -> R<()> {
        let func = self.m.func(f).ok_or(InterpError::UnknownFunc(f))?;
        let body = func.body.as_ref().ok_or(InterpError::NoBody(f))?;
        if args.len() != func.params.len() {
            return Err(InterpError::ArgCount { func: f, got: args.len(), want: func.params.len() });
        }
        let mut regs = vec![Val::S(0); body.values.len()];
        for (p, a) in body.params(Block(0)).iter().zip(args) {
            if let Some(r) = regs.get_mut(p.index()) {
                *r = a;
            }
        }
        frames.push(Frame { func: f, regs, block: 0, pc: 0, mark: self.mem.stack_mark(), dest: None });
        Ok(())
    }

    /// Runs `entry` with `args` to completion.
    pub fn run(&mut self, entry: FuncId, args: Vec<Val>) -> R<Outcome> {
        let m = self.m;
        let mut frames: Vec<Frame> = Vec::new();
        self.push_frame(&mut frames, entry, args)?;
        loop {
            let fr = frames.last_mut().ok_or(malformed("no frame"))?;
            let fid = fr.func;
            let body = m.func(fid).and_then(|f| f.body.as_ref()).ok_or(InterpError::NoBody(fid))?;
            let (step, inst) = loop {
                self.steps += 1;
                if self.steps > self.cfg.max_steps {
                    return Err(InterpError::Fuel);
                }
                let blk = body.blocks.get(fr.block as usize).ok_or(malformed("block"))?;
                if fr.pc < blk.insts.len {
                    let i = blk.insts.start + fr.pc;
                    fr.pc += 1;
                    match self.exec(body, fid, &mut fr.regs, i)? {
                        Step::Next => continue,
                        s => break (Some(s), i),
                    }
                }
                match blk.term {
                    Term::Br(t) => self.jump(body, fr, &t)?,
                    Term::Cbr { cond, then, els } => {
                        let c = self.scalar(body, &fr.regs, cond)?;
                        self.jump(body, fr, if c != 0 { &then } else { &els })?;
                    }
                    Term::Switch { scrut, cases, default } => {
                        let v = self.ival(body, &fr.regs, scrut)?;
                        let t = body
                            .switch_cases(cases)
                            .iter()
                            .find(|c| c.value == v)
                            .map_or(default, |c| c.target);
                        self.jump(body, fr, &t)?;
                    }
                    Term::Ret(o) => {
                        let v = match o {
                            Some(o) => Some(self.eval(body, &fr.regs, o)?),
                            None => None,
                        };
                        let done = frames.pop().ok_or(malformed("frame"))?;
                        self.mem.stack_release(done.mark);
                        match frames.last_mut() {
                            None => return Ok(Outcome::Return(v)),
                            Some(caller) => {
                                if let Some(d) = caller.dest.take()
                                    && let Some(r) = caller.regs.get_mut(d.index())
                                {
                                    *r = v.unwrap_or_else(Val::unit);
                                }
                            }
                        }
                        break (None, 0);
                    }
                    Term::Unreachable => {
                        return Err(InterpError::Unreachable { func: fid, block: Block(fr.block) });
                    }
                }
            };
            let site = |i: u32| body.insts.site.get(i as usize).copied().unwrap_or(Site::NONE);
            match step {
                None => {}
                Some(Step::Call(f, args, dest)) => {
                    if let Some(fr) = frames.last_mut() {
                        fr.dest = dest;
                    }
                    if frames.len() >= self.cfg.max_depth {
                        return Ok(Outcome::Trap(Trap {
                            kind: PanicKind::StackOverflow,
                            site: site(inst),
                            func: fid,
                        }));
                    }
                    self.push_frame(&mut frames, f, args)?;
                }
                Some(Step::Trap(kind)) => {
                    return Ok(Outcome::Trap(Trap { kind, site: site(inst), func: fid }));
                }
                Some(Step::Exit(c)) => return Ok(Outcome::Exit(c)),
                Some(Step::Next) => {}
            }
        }
    }

    fn jump(&mut self, body: &Body, fr: &mut Frame, t: &Target) -> R<()> {
        let mut tmp = std::mem::take(&mut self.scratch);
        tmp.clear();
        for &o in body.target_args(t) {
            match self.eval(body, &fr.regs, o) {
                Ok(v) => tmp.push(v),
                Err(e) => {
                    self.scratch = tmp;
                    return Err(e);
                }
            }
        }
        for (&p, v) in body.params(t.block).iter().zip(tmp.drain(..)) {
            if let Some(r) = fr.regs.get_mut(p.index()) {
                *r = v;
            }
        }
        self.scratch = tmp;
        fr.block = t.block.0;
        fr.pc = 0;
        Ok(())
    }

    #[inline]
    fn eval(&self, body: &Body, regs: &[Val], o: Operand) -> R<Val> {
        match o {
            Operand::Value(v) => regs.get(v.index()).cloned().ok_or(malformed("value")),
            Operand::Const(c) => match body.konst(c) {
                Some(Const::Int(x)) => Ok(Val::S(x as u64)),
                Some(Const::F16(b) | Const::BF16(b)) => Ok(Val::S(b as u64)),
                Some(Const::F32(b)) => Ok(Val::S(b as u64)),
                Some(Const::F64(b)) => Ok(Val::S(b)),
                None => Err(malformed("const")),
            },
            Operand::Str(s) => {
                let a = self.strs.get(s.index()).ok_or(malformed("string"))?;
                let n = self.m.strings.get(s.index()).map_or(0, |b| b.len());
                Ok(Val::Pair(*a, n as u64))
            }
            // a fn value is (code, environment); a fn item has no environment
            Operand::Func(f) => Ok(Val::Pair(f.0 as u64, 0)),
            Operand::Type(_) => Err(malformed("type operand")),
        }
    }

    #[inline]
    fn scalar(&self, body: &Body, regs: &[Val], o: Operand) -> R<u64> {
        match o {
            Operand::Value(v) => match regs.get(v.index()) {
                Some(Val::S(x)) => Ok(*x),
                _ => Err(malformed("scalar expected")),
            },
            _ => self.eval(body, regs, o)?.bits().ok_or(malformed("scalar expected")),
        }
    }

    fn pair(&self, body: &Body, regs: &[Val], o: Operand) -> R<(u64, u64)> {
        match self.eval(body, regs, o)? {
            Val::Pair(a, n) => Ok((a, n)),
            _ => Err(malformed("fat pointer expected")),
        }
    }

    /// The class of an operand (`None` = an int constant).
    fn ocls(&self, body: &Body, o: Operand) -> Option<Cls> {
        match o {
            Operand::Value(v) => body.value_ty(v).map(|t| self.cls(t)),
            Operand::Const(c) => match body.konst(c) {
                Some(Const::F16(_)) => Some(Cls::Float(FloatBits::F16)),
                Some(Const::BF16(_)) => Some(Cls::Float(FloatBits::B16)),
                Some(Const::F32(_)) => Some(Cls::Float(FloatBits::F32)),
                Some(Const::F64(_)) => Some(Cls::Float(FloatBits::F64)),
                _ => None,
            },
            Operand::Str(_) => Some(Cls::Fat),
            Operand::Func(_) => Some(Cls::Fn),
            Operand::Type(_) => Some(Cls::Other),
        }
    }

    /// The mathematical value of an int-like operand.
    fn ival(&self, body: &Body, regs: &[Val], o: Operand) -> R<i128> {
        if let Some(x) = body.int_const(o) {
            return Ok(x);
        }
        let bits = self.scalar(body, regs, o)?;
        Ok(match self.ocls(body, o) {
            Some(Cls::Int(t)) => t.value(bits),
            _ => bits as i128,
        })
    }

    fn layout(&self, t: TypeId) -> R<&crate::layout::Layout> {
        self.layouts.get_layout(t).map_err(|err| InterpError::Layout { ty: t, err })
    }

    fn fault(func: FuncId, site: Site) -> impl Fn(Fault) -> InterpError + use<H> {
        move |fault| InterpError::Fault { fault, func, site }
    }

    fn load_val(&self, t: TypeId, addr: u64, depth: usize, func: FuncId, site: Site) -> R<Val> {
        if depth > MAX_TYPE_DEPTH {
            return Err(InterpError::Unsupported("load nesting too deep".into()));
        }
        let fault = Self::fault(func, site);
        let l = self.layout(t)?;
        Ok(match &l.shape {
            Shape::Scalar => {
                let raw = self.mem.read_uint(addr, l.size).map_err(fault)?;
                Val::S(match self.cls(t) {
                    Cls::Int(it) => it.norm(raw),
                    Cls::Bool => (raw != 0) as u64,
                    _ => raw,
                })
            }
            Shape::Fat => {
                let a = self.mem.read_uint(addr, 8).map_err(&fault)?;
                let n = self.mem.read_uint(addr.wrapping_add(8), 8).map_err(&fault)?;
                Val::Pair(a, n)
            }
            Shape::Fields(fs) => {
                let mut v = Vec::with_capacity(fs.len());
                for &(off, ft) in fs {
                    v.push(self.load_val(ft, addr.wrapping_add(off), depth + 1, func, site)?);
                }
                Val::agg(v)
            }
            Shape::Array { elem, stride, len } => {
                if *len > MAX_AGG_LEN {
                    return Err(InterpError::Unsupported("array value too large".into()));
                }
                let mut v = Vec::with_capacity(*len as usize);
                for k in 0..*len {
                    v.push(self.load_val(*elem, addr.wrapping_add(k * stride), depth + 1, func, site)?);
                }
                Val::agg(v)
            }
            Shape::Enum(vs) => {
                let tag = self.mem.read_uint(addr, 4).map_err(&fault)?;
                let fs = vs.get(tag as usize).ok_or(InterpError::WrongVariant { func, site })?;
                let mut v = Vec::with_capacity(fs.len() + 1);
                v.push(Val::S(tag));
                for &(off, ft) in fs {
                    v.push(self.load_val(ft, addr.wrapping_add(off), depth + 1, func, site)?);
                }
                Val::agg(v)
            }
            Shape::Bytes => self.load_bytes(t, addr, func, site)?,
        })
    }

    fn store_val(&mut self, t: TypeId, addr: u64, v: &Val, depth: usize, func: FuncId, site: Site) -> R<()> {
        if depth > MAX_TYPE_DEPTH {
            return Err(InterpError::Unsupported("store nesting too deep".into()));
        }
        let fault = Self::fault(func, site);
        let l = self.layout(t)?.clone();
        match (&l.shape, v) {
            (Shape::Scalar, Val::S(x)) => self.mem.write_uint(addr, l.size, *x).map_err(fault),
            (Shape::Fat, Val::Pair(a, n)) => {
                self.mem.write_uint(addr, 8, *a).map_err(&fault)?;
                self.mem.write_uint(addr.wrapping_add(8), 8, *n).map_err(&fault)
            }
            (Shape::Fields(fs), Val::Agg(vs)) if fs.len() == vs.len() => {
                for (&(off, ft), fv) in fs.iter().zip(vs.iter()) {
                    self.store_val(ft, addr.wrapping_add(off), fv, depth + 1, func, site)?;
                }
                Ok(())
            }
            (Shape::Array { elem, stride, len }, Val::Agg(vs)) if *len == vs.len() as u64 => {
                for (k, fv) in vs.iter().enumerate() {
                    self.store_val(*elem, addr.wrapping_add(k as u64 * stride), fv, depth + 1, func, site)?;
                }
                Ok(())
            }
            (Shape::Enum(vars), Val::Agg(vs)) => {
                let tag = vs.first().and_then(|t| t.bits()).ok_or(malformed("enum value"))?;
                let fs = vars.get(tag as usize).ok_or(InterpError::WrongVariant { func, site })?;
                if fs.len() + 1 != vs.len() {
                    return Err(malformed("enum value arity"));
                }
                // Zero the whole enum first, so the bytes beyond this variant are deterministic.
                self.mem.write(addr, &vec![0u8; l.size as usize]).map_err(&fault)?;
                self.mem.write_uint(addr, 4, tag).map_err(&fault)?;
                for (&(off, ft), fv) in fs.iter().zip(vs[1..].iter()) {
                    self.store_val(ft, addr.wrapping_add(off), fv, depth + 1, func, site)?;
                }
                Ok(())
            }
            (Shape::Bytes, _) => self.store_bytes(t, addr, v, func, site),
            _ => Err(malformed("stored value does not match its type")),
        }
    }

    fn set(regs: &mut [Val], body: &Body, i: u32, v: Val) {
        if let Some(Some(r)) = body.insts.result.get(i as usize)
            && let Some(slot) = regs.get_mut(r.index())
        {
            *slot = v;
        }
    }

    fn exec(&mut self, body: &Body, fid: FuncId, regs: &mut [Val], i: u32) -> R<Step> {
        let m = self.m;
        let iu = i as usize;
        let op = *body.insts.op.get(iu).ok_or(malformed("inst"))?;
        let ty = body.insts.ty.get(iu).copied().flatten();
        let args = body.args(i);
        let site = body.insts.site.get(iu).copied().unwrap_or(Site::NONE);
        let rc = ty.map(|t| self.cls(t));
        let arg = |k: usize| args.get(k).copied().ok_or(malformed("operand"));
        macro_rules! s {
            ($k:expr) => {
                self.scalar(body, regs, arg($k)?)?
            };
        }
        let out: Val = match op {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem => match rc {
                Some(Cls::Int(t)) => match arith::int_binary(op, t, s!(0), s!(1)) {
                    Ok(v) => Val::S(v),
                    Err(k) => return Ok(Step::Trap(k)),
                },
                Some(Cls::Float(f @ (FloatBits::F32 | FloatBits::F64))) => {
                    Val::S(arith::float_binary(op, f, s!(0), s!(1)))
                }
                _ => return Err(InterpError::Unsupported(format!("{op:?} on this type"))),
            },
            Op::AddWrap
            | Op::SubWrap
            | Op::MulWrap
            | Op::AddSat
            | Op::SubSat
            | Op::MulSat
            | Op::And
            | Op::Or
            | Op::Xor => match rc {
                Some(Cls::Int(t)) => match arith::int_binary(op, t, s!(0), s!(1)) {
                    Ok(v) => Val::S(v),
                    Err(k) => return Ok(Step::Trap(k)),
                },
                Some(Cls::Bool) => {
                    let (a, b) = (s!(0), s!(1));
                    Val::S(match op {
                        Op::And => a & b,
                        Op::Or => a | b,
                        _ => a ^ b,
                    })
                }
                _ => return Err(malformed("int op type")),
            },
            Op::Shl | Op::Shr | Op::ShlWrap => {
                let Some(Cls::Int(t)) = rc else { return Err(malformed("shift type")) };
                let a = s!(0);
                let n = self.ival(body, regs, arg(1)?)?;
                match arith::int_shift(op, t, a, n) {
                    Ok(v) => Val::S(v),
                    Err(k) => return Ok(Step::Trap(k)),
                }
            }
            Op::Neg => match rc {
                Some(Cls::Int(t)) => match arith::int_neg(t, s!(0)) {
                    Ok(v) => Val::S(v),
                    Err(k) => return Ok(Step::Trap(k)),
                },
                Some(Cls::Float(f)) => Val::S(arith::float_neg(f, s!(0))),
                _ => return Err(malformed("neg type")),
            },
            Op::Not => match rc {
                Some(Cls::Int(t)) => Val::S(arith::int_not(t, s!(0))),
                Some(Cls::Bool) => Val::S(s!(0) ^ 1),
                _ => return Err(malformed("not type")),
            },
            Op::Popcount | Op::Clz | Op::Ctz => {
                let Some(Cls::Int(t)) = rc else { return Err(malformed("count type")) };
                Val::S(arith::int_count(op, t, s!(0)))
            }
            Op::MulWide => {
                let bits = match self.ocls(body, arg(0)?).or(self.ocls(body, arg(1)?)) {
                    Some(Cls::Int(t)) => t.bits,
                    _ => match ty.and_then(|t| self.m.types.key(t)) {
                        Some(crate::types::TypeKey::Tuple(ts)) => match ts.first().map(|t| self.cls(*t)) {
                            Some(Cls::Int(t)) => t.bits,
                            _ => 64,
                        },
                        _ => 64,
                    },
                };
                let (hi, lo) = arith::mul_wide(bits, s!(0), s!(1));
                Val::agg(vec![Val::S(hi), Val::S(lo)])
            }
            Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                let c = self.ocls(body, arg(0)?).or(self.ocls(body, arg(1)?));
                let r = match c {
                    Some(Cls::Float(f)) => arith::float_cmp(op, f, s!(0), s!(1)),
                    // fn values: (code, environment) both equal
                    Some(Cls::Fn) => {
                        let same = self.pair(body, regs, arg(0)?)? == self.pair(body, regs, arg(1)?)?;
                        if op == Op::Eq { same } else { !same }
                    }
                    _ => {
                        let (x, y) = (self.ival(body, regs, arg(0)?)?, self.ival(body, regs, arg(1)?)?);
                        match op {
                            Op::Eq => x == y,
                            Op::Ne => x != y,
                            Op::Lt => x < y,
                            Op::Le => x <= y,
                            Op::Gt => x > y,
                            _ => x >= y,
                        }
                    }
                };
                Val::bool(r)
            }
            Op::Select => {
                let c = s!(0);
                self.eval(body, regs, arg(if c != 0 { 1 } else { 2 })?)?
            }
            Op::Cast(kind) => {
                let t = ty.ok_or(malformed("cast type"))?;
                let src = arg(0)?;
                // a fn item casts as a pointer: its code address, which here is the fn index
                // (what `call.ind` and `call_on_stack` decode)
                let (from, x) = match (src, self.ocls(body, src)) {
                    (Operand::Func(f), _) => (arith::CastTy::Ptr, u64::from(f.0)),
                    (_, None) => {
                        let v = body.int_const(src).ok_or(malformed("cast operand"))?;
                        (arith::CastTy::Int(IntTy::new(64, v < 0)), s!(0))
                    }
                    (_, Some(c)) => (verify::cast_ty(c).ok_or(malformed("cast source"))?, s!(0)),
                };
                if kind == CastKind::Try {
                    let (none, some, p) =
                        verify::option_shape(&self.m.types, t).ok_or(malformed("cast.try type"))?;
                    let to = verify::cast_ty(self.cls(p)).ok_or(malformed("cast target"))?;
                    match arith::convert(kind, from, to, x) {
                        Some(v) => Val::agg(vec![Val::S(some as u64), Val::S(v)]),
                        None => Val::agg(vec![Val::S(none as u64)]),
                    }
                } else {
                    let to = verify::cast_ty(self.cls(t)).ok_or(malformed("cast target"))?;
                    Val::S(arith::convert(kind, from, to, x).ok_or(malformed("cast"))?)
                }
            }
            Op::Make => {
                let mut v = Vec::with_capacity(args.len());
                for &a in args {
                    v.push(self.eval(body, regs, a)?);
                }
                Val::agg(v)
            }
            Op::Get => {
                let agg = self.eval(body, regs, arg(0)?)?;
                let k = self.ival(body, regs, arg(1)?)?;
                let fs = agg.fields().ok_or(malformed("get of a scalar"))?;
                match usize::try_from(k).ok().and_then(|k| fs.get(k)) {
                    Some(v) => v.clone(),
                    None => return Ok(Step::Trap(PanicKind::Bounds)),
                }
            }
            Op::Set => {
                let agg = self.eval(body, regs, arg(0)?)?;
                let k = self.ival(body, regs, arg(1)?)?;
                let nv = self.eval(body, regs, arg(2)?)?;
                let mut fs = agg.fields().ok_or(malformed("set of a scalar"))?.to_vec();
                match usize::try_from(k).ok().and_then(|k| fs.get_mut(k)) {
                    Some(slot) => *slot = nv,
                    None => return Ok(Step::Trap(PanicKind::Bounds)),
                }
                Val::agg(fs)
            }
            Op::EnumMake => {
                let mut v = Vec::with_capacity(args.len());
                for &a in args {
                    v.push(self.eval(body, regs, a)?);
                }
                Val::agg(v)
            }
            Op::EnumTag => {
                let e = self.eval(body, regs, arg(0)?)?;
                e.fields().and_then(|f| f.first()).cloned().ok_or(malformed("enum value"))?
            }
            Op::EnumGet => {
                let e = self.eval(body, regs, arg(0)?)?;
                let v = self.ival(body, regs, arg(1)?)?;
                let f = self.ival(body, regs, arg(2)?)?;
                let fs = e.fields().ok_or(malformed("enum value"))?;
                if fs.first().and_then(|t| t.bits()).map(|t| t as i128) != Some(v) {
                    return Err(InterpError::WrongVariant { func: fid, site });
                }
                usize::try_from(f + 1).ok().and_then(|k| fs.get(k)).cloned().ok_or(malformed("enum field"))?
            }
            Op::Slot => {
                let t = ty.ok_or(malformed("slot type"))?;
                let p = self.pointee.get(t.index()).copied().flatten().ok_or(malformed("slot pointee"))?;
                let l = self.layout(p)?;
                match self.mem.stack_alloc(l.size, l.align) {
                    Some(a) => Val::S(a),
                    None => return Ok(Step::Trap(PanicKind::StackOverflow)),
                }
            }
            Op::Load => {
                let t = ty.ok_or(malformed("load type"))?;
                let a = s!(0);
                self.load_val(t, a, 0, fid, site)?
            }
            Op::Store => {
                let p = arg(0)?;
                let pt = match p {
                    Operand::Value(v) => body.value_ty(v),
                    _ => None,
                };
                let t = pt
                    .and_then(|t| self.pointee.get(t.index()).copied().flatten())
                    .ok_or(malformed("store type"))?;
                let a = s!(0);
                let v = self.eval(body, regs, arg(1)?)?;
                self.store_val(t, a, &v, 0, fid, site)?;
                return Ok(Step::Next);
            }
            Op::FieldAddr | Op::EnumFieldAddr | Op::ElemAddr => {
                let p = arg(0)?;
                let pt = match p {
                    Operand::Value(v) => body.value_ty(v).ok_or(malformed("addr operand"))?,
                    _ => return Err(malformed("addr operand")),
                };
                let inner =
                    self.pointee.get(pt.index()).copied().flatten().ok_or(malformed("addr pointee"))?;
                let off: u64 = match op {
                    Op::FieldAddr => {
                        let k = self.ival(body, regs, arg(1)?)?;
                        match &self.layout(inner)?.shape {
                            Shape::Fields(fs) => {
                                usize::try_from(k).ok().and_then(|k| fs.get(k)).ok_or(malformed("field"))?.0
                            }
                            _ => return Err(malformed("addr.field layout")),
                        }
                    }
                    Op::EnumFieldAddr => {
                        let v = self.ival(body, regs, arg(1)?)?;
                        let f = self.ival(body, regs, arg(2)?)?;
                        match &self.layout(inner)?.shape {
                            Shape::Enum(vs) => {
                                usize::try_from(v)
                                    .ok()
                                    .and_then(|v| vs.get(v))
                                    .and_then(|fs| usize::try_from(f).ok().and_then(|f| fs.get(f)))
                                    .ok_or(malformed("enum field"))?
                                    .0
                            }
                            _ => return Err(malformed("addr.enum_field layout")),
                        }
                    }
                    _ => {
                        let k = self.ival(body, regs, arg(1)?)?;
                        let stride = match self.m.types.key(inner) {
                            Some(
                                crate::types::TypeKey::Array { elem, .. }
                                | crate::types::TypeKey::Slice { elem, .. },
                            ) => self.layout(*elem)?.size,
                            _ => self.layout(inner)?.size,
                        };
                        (k as i64 as u64).wrapping_mul(stride)
                    }
                };
                let base = match self.eval(body, regs, p)? {
                    Val::S(a) | Val::Pair(a, _) => a,
                    _ => return Err(malformed("pointer value")),
                };
                Val::S(base.wrapping_add(off))
            }
            Op::SliceMake => Val::Pair(s!(0), s!(1)),
            Op::FnMake => {
                let Operand::Func(f) = arg(0)? else { return Err(malformed("fn.make code")) };
                Val::Pair(f.0 as u64, s!(1))
            }
            Op::SliceLen => Val::S(self.pair(body, regs, arg(0)?)?.1),
            Op::SlicePtr => Val::S(self.pair(body, regs, arg(0)?)?.0),
            Op::BoundsCheck => {
                let (k, n) = (self.ival(body, regs, arg(0)?)?, self.ival(body, regs, arg(1)?)?);
                if k >= n {
                    return Ok(Step::Trap(PanicKind::Bounds));
                }
                return Ok(Step::Next);
            }
            Op::Call | Op::CallInd => {
                let mut env = None;
                let f = match op {
                    Op::Call => match arg(0)? {
                        Operand::Func(f) => f,
                        _ => return Err(malformed("callee")),
                    },
                    _ => {
                        let (x, e) = self.pair(body, regs, arg(0)?)?;
                        env = Some(e);
                        let idx = u32::try_from(x).map_err(|_| InterpError::BadFnValue(x))?;
                        if idx as usize >= self.m.funcs.len() {
                            return Err(InterpError::BadFnValue(x));
                        }
                        FuncId(idx)
                    }
                };
                let mut vals = Vec::with_capacity(args.len());
                for &a in args.iter().skip(1) {
                    vals.push(self.eval(body, regs, a)?);
                }
                // `call.ind` passes the fn value's environment as a trailing argument; a callee
                // without the environment param (a fn item) ignores it
                if let Some(e) = env
                    && m.funcs.get(f.index()).is_some_and(|g| g.params.len() == vals.len() + 1)
                {
                    vals.push(Val::S(e));
                }
                let dest = body.insts.result.get(iu).copied().flatten();
                return Ok(Step::Call(f, vals, dest));
            }
            Op::Intrinsic(id) if self.vops.get(id.index()).copied().flatten().is_some() => {
                let v = self.vops.get(id.index()).copied().flatten().ok_or(malformed("vector op"))?;
                match self.exec_vec(body, fid, regs, v, ty, args, site)? {
                    simd::VOut::Val(x) => x,
                    simd::VOut::None => return Ok(Step::Next),
                    simd::VOut::Trap(k) => return Ok(Step::Trap(k)),
                }
            }
            Op::Intrinsic(id) => {
                let name = m.intrinsic_name(id).ok_or(malformed("intrinsic"))?;
                let mut vals = Vec::with_capacity(args.len());
                for &a in args {
                    vals.push(self.eval(body, regs, a)?);
                }
                if name == "call_on_stack" {
                    // `call_on_stack(f, arg, top)`: calls fn value `f` with `arg` (the stack switch
                    // is not modelled: interpreter frames have no machine stack).
                    let x = vals.first().and_then(|v| v.bits()).ok_or(malformed("call_on_stack callee"))?;
                    let idx = u32::try_from(x).map_err(|_| InterpError::BadFnValue(x))?;
                    if idx as usize >= self.m.funcs.len() {
                        return Err(InterpError::BadFnValue(x));
                    }
                    let a = vals.get(1).cloned().unwrap_or(Val::S(0));
                    let dest = body.insts.result.get(iu).copied().flatten();
                    return Ok(Step::Call(FuncId(idx), vec![a], dest));
                }
                if name == "static_addr" {
                    let k = vals.first().and_then(|v| v.bits()).ok_or(malformed("static_addr index"))?;
                    let a = self.statics.get(k as usize).copied().ok_or(malformed("static_addr index"))?;
                    Self::set(regs, body, i, Val::S(a));
                    return Ok(Step::Next);
                }
                let res = if let Some(k) = name.strip_prefix("syscall").and_then(|d| d.parse::<usize>().ok())
                {
                    let mut a6 = [0u64; 6];
                    for (slot, v) in a6.iter_mut().zip(vals.iter().skip(1)).take(k) {
                        *slot = v.bits().unwrap_or(0);
                    }
                    let nr = vals.first().and_then(|v| v.bits()).unwrap_or(0);
                    match self.host.syscall(nr, a6, &mut self.mem) {
                        SysResult::Ret(r) => Some(Val::S(r)),
                        SysResult::Exit(c) => return Ok(Step::Exit(c)),
                    }
                } else {
                    self.host.intrinsic(name, &vals, &mut self.mem).map_err(InterpError::Host)?
                };
                match res {
                    Some(v) => v,
                    None => return Ok(Step::Next),
                }
            }
            Op::Panic(k) => return Ok(Step::Trap(k)),
        };
        Self::set(regs, body, i, out);
        Ok(Step::Next)
    }
}

/// Convenience: verify, run `entry` with `args` on `host`, return the outcome and the host.
pub fn run<H: Host>(
    m: &Module,
    entry: FuncId,
    args: Vec<Val>,
    host: H,
    cfg: Config,
) -> (R<Outcome>, Option<H>) {
    match Interp::new(m, host, cfg) {
        Ok(mut it) => {
            let r = it.run(entry, args);
            (r, Some(it.into_host()))
        }
        Err(e) => (Err(e), None),
    }
}
