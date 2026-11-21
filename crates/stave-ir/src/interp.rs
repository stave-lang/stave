//! Reference interpreter: the semantic oracle for differential tests.
//!
//! Executes a verified [`Module`] from an entry function over a simulated [`Memory`], with the
//! exact trap semantics (returns [`Outcome::Trap`] with the kind,
//! the trapping instruction's [`Site`], and the function), wrapping and saturating ops, the `as`
//! family, and IEEE f32 / f64 via Rust's native operations (round to nearest even).
//! System calls and other intrinsics go to a pluggable [`Host`], except `copy_bytes`, which the
//! interpreter runs itself on its memory (native code calls `std.slice.copy_raw` instead, a fn
//! no IR calls, so the interpreter never runs its body).
//!
//! Values ([`Val`]): scalars are raw `u64` bits (ints normalized to their type, floats as IEEE
//! bits, `bool` 0/1, `char` code point, pointers and fn values as addresses / function indices);
//! fat pointers are `(address, length)`; tuples, structs and arrays are `Agg` of their fields;
//! an enum is `Agg[tag, fields of that variant..]`; a vector or mask is `Agg` of its lanes (mask
//! lanes 0 / 1).
//!
//! The interpreter never recurses on the host stack per Stave call (explicit frame stack), and
//! malformed input yields an [`InterpError`], never a panic.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

mod signature;
mod simd;

use crate::arith::{self, IntTy};
use crate::borrowtrack::{Ev, Kind, NO_EV, ROOT, Report, Tag, Tracker};
use crate::host::{Host, SysResult};
use crate::ir::{
    Block, Body, CastKind, Const, EventKind, FuncId, Module, Op, Operand, PanicKind, Site, Target, Term,
    Value, ValueDef,
};
use crate::layout::{LayoutError, Layouts, Shape};
use crate::mem::{Fault, Memory};
use crate::memcheck::{At, MemReport, TRACE_DEPTH, Trace};
use crate::types::{FloatBits, MAX_TYPE_DEPTH, TypeId, TypeKey};
use crate::verify::{self, Cls, VerifyError};

/// A runtime value.
#[derive(Clone, Debug)]
pub enum Val {
    S(u64),
    Pair(u64, u64),
    Agg(Rc<[Val]>),
    /// Borrow tracking: a 64-bit word with the tag of the reference it was made from.
    T(u64, Tag),
    /// Borrow tracking: a fat pointer whose address word has a tag.
    PairT(u64, u64, Tag),
}

/// A tag is no part of a value: `==` compares the bits.
impl PartialEq for Val {
    fn eq(&self, o: &Val) -> bool {
        match (self, o) {
            (Val::Agg(a), Val::Agg(b)) => a == b,
            (Val::S(a) | Val::T(a, _), Val::S(b) | Val::T(b, _)) => a == b,
            (Val::Pair(a, n) | Val::PairT(a, n, _), Val::Pair(b, k) | Val::PairT(b, k, _)) => {
                a == b && n == k
            }
            _ => false,
        }
    }
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
            Val::S(x) | Val::T(x, _) => Some(*x),
            _ => None,
        }
    }
    /// The address word of a thin or fat pointer and its tag ([`ROOT`]: none).
    pub fn word(&self) -> Option<(u64, Tag)> {
        match *self {
            Val::S(x) | Val::Pair(x, _) => Some((x, ROOT)),
            Val::T(x, t) | Val::PairT(x, _, t) => Some((x, t)),
            Val::Agg(_) => None,
        }
    }
    fn tagged(x: u64, t: Tag) -> Val {
        if t == ROOT { Val::S(x) } else { Val::T(x, t) }
    }
    fn untagged(self) -> Val {
        match self {
            Val::T(x, _) => Val::S(x),
            Val::PairT(a, n, _) => Val::Pair(a, n),
            v => v,
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
    /// Checking mode ([`crate::memcheck`]): a memory error ends the run with
    /// [`Outcome::MemError`]. The runtime allocator's `try_alloc`, `free`, and `try_realloc` are
    /// replaced by checked allocations that are never reused.
    pub check: bool,
    /// Borrow tracking ([`crate::borrowtrack`]), on top of `check`: a violation of the dynamic
    /// borrow model ends the run with [`Outcome::BorrowError`].
    pub borrow: bool,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            max_steps: u64::MAX,
            max_depth: 10_000,
            stack_size: 8 << 20,
            heap_limit: 1 << 30,
            check: false,
            borrow: false,
        }
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
    /// Checking mode: a memory error.
    MemError(Box<MemReport>),
    /// Borrow tracking: a violation of the dynamic borrow model.
    BorrowError(Box<BorrowReport>),
}

/// A violation of contracts/borrow-model.md with its sites.
#[derive(Clone, Debug, PartialEq)]
pub struct BorrowReport {
    pub report: Report,
    /// The violating event, innermost frame first.
    pub at: Trace,
    /// The event that made the failing tag `Frozen` or `Dead`.
    pub cause_at: Option<At>,
    /// Where the failing tag was made (a root: where its allocation began).
    pub made_at: Option<At>,
}

impl BorrowReport {
    /// Multi-line text: `borrow violation [<class>] <rule>: ...`, then the three sites.
    pub fn render(&self, m: &Module) -> String {
        let r = &self.report;
        let one = |at: &Option<At>| match at {
            Some(a) => MemReport::trace_text(m, &vec![*a]),
            None => "?".into(),
        };
        format!(
            "borrow violation [{}] {}: {:?} of bytes {}..{} of allocation {} through a {:?} {:?} tag\n  at {}\n  cause ({:?}) at {}\n  tag made at {}",
            r.class.name(),
            r.rule(),
            r.event,
            r.lo,
            r.hi,
            r.alloc,
            r.perm,
            r.kind,
            MemReport::trace_text(m, &self.at),
            r.cause,
            one(&self.cause_at),
            one(&self.made_at),
        )
    }
}

/// The state of borrow tracking.
struct Borrows {
    tracker: Tracker,
    /// The tag of the 64-bit word stored at an address, with the word: the tag holds while
    /// the word is there.
    shadow: BTreeMap<u64, (u64, Tag)>,
    /// The side table: per function, the row of each value ([`Mark`] as a byte; a function
    /// without rows has no entries).
    marks: Vec<Vec<u8>>,
    /// The site of the vector instruction being executed.
    at: Ev,
    /// The bytes of a value of each type that are inside an `UnsafeCell`, as sorted ranges.
    cells: HashMap<TypeId, Rc<[(u64, u64)]>>,
    report: Option<Box<Report>>,
    /// Whether a value of each type holds a reference word.
    refs: HashMap<TypeId, bool>,
    /// The signature check's plan of each function called so far.
    plans: Vec<Option<Rc<signature::Plan>>>,
    /// The allocations of the statics.
    static_ids: HashSet<u32>,
    /// The first of the `from` clauses that a call did not keep (M24), and how many there were.
    signatures: Vec<signature::SignatureReport>,
    signature_count: usize,
    /// The address the next store writes without an event: a constant temporary evaluated
    /// again.
    quiet: Option<u64>,
}

fn ev(f: FuncId, inst: u32) -> Ev {
    (u64::from(f.0) + 1) << 32 | u64::from(inst)
}

/// A row of the side table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Mark {
    None,
    Shared,
    Mut,
    Begin,
    End,
    Gone,
    Repeat,
    Env,
    /// A borrow through a raw pointer.
    SharedRaw,
    MutRaw,
    Const,
}

impl Mark {
    const ALL: [Mark; 11] = [
        Mark::None,
        Mark::Shared,
        Mark::Mut,
        Mark::Begin,
        Mark::End,
        Mark::Gone,
        Mark::Repeat,
        Mark::Env,
        Mark::SharedRaw,
        Mark::MutRaw,
        Mark::Const,
    ];

    fn of(e: EventKind) -> Mark {
        match e {
            EventKind::Begin => Mark::Begin,
            EventKind::End => Mark::End,
            EventKind::Gone => Mark::Gone,
            EventKind::Repeat => Mark::Repeat,
            EventKind::Env => Mark::Env,
            EventKind::Const => Mark::Const,
        }
    }
}

/// Whether the borrow that instruction `i` of `body` carries is made from a raw pointer: the
/// address it borrows is that of a place reached through one (`&*p`, `&(*p).f`, `&(*p)[k]`,
/// a fat reference put together from one), with no load in between.
fn from_raw(m: &Module, body: &Body, i: u32) -> bool {
    let raw = |o: Operand| match o {
        Operand::Value(v) => {
            matches!(body.value_ty(v).and_then(|t| m.types.key(t)), Some(TypeKey::RawPtr { .. }))
        }
        _ => false,
    };
    let def = |o: Operand| match o {
        Operand::Value(v) => match body.values.get(v.index()).map(|d| d.def) {
            Some(ValueDef::Inst(k)) => Some(k),
            _ => None,
        },
        _ => None,
    };
    let mut cur = i;
    for _ in 0..16 {
        let address = matches!(
            body.insts.op.get(cur as usize),
            Some(
                Op::Cast(CastKind::Exact)
                    | Op::FieldAddr
                    | Op::EnumFieldAddr
                    | Op::ElemAddr
                    | Op::SliceMake
                    | Op::SlicePtr
            )
        );
        let Some(&a) = body.args(cur).first().filter(|_| address) else { return false };
        if raw(a) {
            return true;
        }
        let Some(k) = def(a) else { return false };
        cur = k;
    }
    false
}

/// The rows of a body by value.
fn marks_of(m: &Module, body: &Body) -> Vec<u8> {
    if body.borrows.is_empty() && body.events.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Mark::None as u8; body.values.len()];
    let mut set = |v: Value, m: Mark| {
        if let Some(x) = out.get_mut(v.index()) {
            *x = m as u8;
        }
    };
    for b in &body.borrows {
        let raw = match body.values.get(b.value.index()).map(|d| d.def) {
            Some(ValueDef::Inst(i)) => from_raw(m, body, i),
            _ => false,
        };
        set(
            b.value,
            match (b.mutable, raw) {
                (true, false) => Mark::Mut,
                (false, false) => Mark::Shared,
                (true, true) => Mark::MutRaw,
                (false, true) => Mark::SharedRaw,
            },
        );
    }
    for e in &body.events {
        set(e.value, Mark::of(e.kind));
    }
    out
}

impl Borrows {
    /// The allocation and offset of `addr`; the tracker learns of an allocation at its first
    /// event.
    fn locate(&mut self, mem: &Memory, addr: u64) -> Option<(u32, u64)> {
        let (id, base, size) = mem.alloc_span(addr)?;
        if !self.tracker.has(id) {
            // an allocation that ended before any event of it: the access is a memory error
            if !mem.alloc_of(id).is_some_and(|a| a.2) {
                return None;
            }
            self.tracker.begin(id, size, false, NO_EV);
        }
        addr.checked_sub(base).map(|off| (id, off))
    }

    /// Allocation `id` ends (M3): every tag of it is `Dead`, and no word in it holds a tag. An
    /// end under a protected tag (M23) leaves its report in `report`.
    fn end_alloc(&mut self, mem: &Memory, id: u32, at: Ev) {
        if let Err(r) = self.tracker.end_alloc(id, at) {
            self.report.get_or_insert(r);
        }
        if let Some((base, size, _)) = mem.alloc_of(id) {
            self.forget(base, size);
        }
    }

    /// No word of the `size` bytes at `base` holds a tag any more.
    fn forget(&mut self, base: u64, size: u64) {
        if self.shadow.is_empty() {
            return;
        }
        let gone: Vec<u64> = self.shadow.range(base..base.saturating_add(size)).map(|(&a, _)| a).collect();
        for a in gone {
            self.shadow.remove(&a);
        }
    }

    /// Frees the tags no word carries: the registers of every frame and the tagged words in
    /// memory are the only holders of a tag.
    fn collect(&mut self, frames: &[Frame]) {
        let mut held: HashSet<Tag> = self.shadow.values().map(|w| w.1).collect();
        let mut scanned = self.shadow.len();
        let mut todo: Vec<&Val> = Vec::new();
        for fr in frames {
            todo.extend(fr.regs.iter());
            while let Some(v) = todo.pop() {
                scanned += 1;
                match v {
                    Val::T(_, t) | Val::PairT(_, _, t) => {
                        held.insert(*t);
                    }
                    Val::Agg(vs) => todo.extend(vs.iter()),
                    Val::S(_) | Val::Pair(..) => {}
                }
            }
        }
        self.tracker.collect(&held, scanned);
    }

    fn stored(&mut self, addr: u64, size: u64, bits: u64, tag: Tag) {
        if tag != ROOT && size == 8 {
            self.shadow.insert(addr, (bits, tag));
        } else if !self.shadow.is_empty() {
            self.shadow.remove(&addr);
        }
    }

    /// A byte copy of `src..src + n` to `dst`: each tagged word wholly inside the source keeps
    /// its tag at the same offset from `dst`; a tagged word that the destination covers or cuts
    /// into loses its tag (its bytes are the copy's now).
    fn copied(&mut self, dst: u64, src: u64, n: u64) {
        if self.shadow.is_empty() || n == 0 {
            return;
        }
        let moved: Vec<(u64, (u64, Tag))> = if n >= 8 {
            let last = src.saturating_add(n - 8);
            self.shadow.range(src..=last).map(|(&a, &w)| (a - src, w)).collect()
        } else {
            Vec::new()
        };
        let gone: Vec<u64> =
            self.shadow.range(dst.saturating_sub(7)..dst.saturating_add(n)).map(|(&a, _)| a).collect();
        for a in gone {
            self.shadow.remove(&a);
        }
        for (off, w) in moved {
            self.shadow.insert(dst.wrapping_add(off), w);
        }
    }

    fn loaded(&self, addr: u64, bits: u64) -> Tag {
        match self.shadow.get(&addr) {
            Some(&(b, t)) if b == bits => t,
            _ => ROOT,
        }
    }
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
    /// An access outside any live region (a native run would fault), or a load of a mask whose
    /// lane at `fault.addr` is neither all zeros nor all ones (undefined behaviour).
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
    /// Borrow tracking: the tags the call protects (M23).
    guards: Vec<Tag>,
    /// Borrow tracking: what the signature check keeps of the entry (M24).
    sig: Option<Box<signature::Entry>>,
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
    /// Checking mode: the runtime allocator fns whose calls are replaced.
    alloc_fns: Option<Box<AllocFns>>,
    bt: Option<Box<Borrows>>,
}

/// The runtime allocator entry points (`runtime/alloc.stv`); `alloc` and `realloc` reach them.
struct AllocFns {
    try_alloc: Option<FuncId>,
    free: Option<FuncId>,
    try_realloc: Option<FuncId>,
}

/// An aggregate being loaded: its parts (type, address), the next part, the values so far.
type Open = (Vec<(TypeId, u64)>, usize, Vec<Val>);

/// A part of an aggregate a reborrow walk has still to do: its type (`None`: kept as it is)
/// and value.
type Part = (Option<TypeId>, Val);

/// One level of a load: a value, or an aggregate's parts and the values before them.
enum Loaded {
    Val(Val),
    Parts(Vec<(TypeId, u64)>, Vec<Val>),
}

/// Adds the bytes `lo..hi` to sorted ranges; a range that starts where the last one ends
/// extends it.
fn push_range(out: &mut Vec<(u64, u64)>, lo: u64, hi: u64) {
    match out.last_mut() {
        _ if lo >= hi => {}
        Some(last) if last.1 == lo => last.1 = hi,
        _ => out.push((lo, hi)),
    }
}

/// Adds the ranges `one` of an element, for `n` elements at `stride` from `base`.
fn repeat_ranges(out: &mut Vec<(u64, u64)>, one: &[(u64, u64)], base: u64, stride: u64, n: u64) {
    if one == [(0, stride)] {
        return push_range(out, base, base.saturating_add(n.saturating_mul(stride)));
    }
    if one.is_empty() {
        return;
    }
    for k in 0..n {
        let at = base.saturating_add(k.saturating_mul(stride));
        for &(a, b) in one {
            push_range(out, at.saturating_add(a), at.saturating_add(b));
        }
    }
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
        let mut mem = if cfg.check {
            Memory::new_checked(cfg.stack_size, cfg.heap_limit)
        } else {
            Memory::new(cfg.stack_size, cfg.heap_limit)
        };
        let alloc_fns = cfg.check.then(|| {
            Box::new(AllocFns {
                try_alloc: m.find_func("runtime.alloc.try_alloc"),
                free: m.find_func("runtime.alloc.free"),
                try_realloc: m.find_func("runtime.alloc.try_realloc"),
            })
        });
        let bt = (cfg.check && cfg.borrow).then(|| {
            let marks = m.funcs.iter().map(|f| f.body.as_ref().map_or(Vec::new(), |b| marks_of(m, b)));
            Box::new(Borrows {
                tracker: Tracker::new(),
                shadow: BTreeMap::new(),
                marks: marks.collect(),
                at: NO_EV,
                cells: HashMap::new(),
                report: None,
                refs: HashMap::new(),
                plans: vec![None; m.funcs.len()],
                static_ids: HashSet::new(),
                signatures: Vec::new(),
                signature_count: 0,
                quiet: None,
            })
        });
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
        let mut it = Interp {
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
            alloc_fns,
            bt,
        };
        it.freeze_statics()?;
        Ok(it)
    }

    /// The instruction executing in each frame, innermost first, at most [`TRACE_DEPTH`].
    fn trace(&self, frames: &[Frame]) -> Trace {
        frames
            .iter()
            .rev()
            .take(TRACE_DEPTH)
            .filter_map(|fr| {
                let b = self.m.func(fr.func)?.body.as_ref()?;
                let blk = b.blocks.get(fr.block as usize)?;
                Some(At::of(fr.func, b, blk.insts.start + fr.pc.saturating_sub(1)))
            })
            .collect()
    }

    /// An error of [`Interp::run`]: the memory error it is in checking mode, else `e`.
    fn failed(&mut self, e: InterpError, frames: &[Frame]) -> R<Outcome> {
        if let Some(r) = self.bt.as_mut().and_then(|bt| bt.report.take()) {
            let at = |e: Ev| {
                let f = FuncId(u32::try_from((e >> 32).checked_sub(1)?).ok()?);
                Some(At::of(f, self.m.func(f)?.body.as_ref()?, e as u32))
            };
            let (cause_at, made_at) = (at(r.cause_at), at(r.made_at));
            return Ok(Outcome::BorrowError(Box::new(BorrowReport {
                report: *r,
                at: self.trace(frames),
                cause_at,
                made_at,
            })));
        }
        match self.mem.take_report() {
            Some(mut r) => {
                r.at = self.trace(frames);
                Ok(Outcome::MemError(r))
            }
            None => Err(e),
        }
    }

    fn alloc_block(&mut self, size: u64, align: u64, frames: &[Frame]) -> u64 {
        if size == 0 {
            return align;
        }
        let at = self.trace(frames);
        self.mem.heap_alloc(size, align, at).unwrap_or(0)
    }

    /// Checking mode: a call of a runtime allocator entry point, done on checked allocations
    /// with the entry point's contract. `None`: not such a call (or one whose alignment the
    /// runtime itself rejects), run it.
    fn alloc_call(
        &mut self,
        f: FuncId,
        args: &[Val],
        frames: &[Frame],
    ) -> Option<Result<Val, Box<MemReport>>> {
        let af = self.alloc_fns.as_deref()?;
        let (try_alloc, free, try_realloc) = (af.try_alloc, af.free, af.try_realloc);
        let u = |k: usize| args.get(k).and_then(Val::bits);
        let pow2 = |a: u64| a.is_power_of_two();
        let check_free =
            |it: &mut Self, p: u64, size: u64, align: u64| match it.mem.heap_check_free(p, size, align) {
                Ok(()) => Ok(()),
                Err(mut r) => {
                    r.at = it.trace(frames);
                    Err(r)
                }
            };
        if Some(f) == try_alloc {
            let (size, align) = (u(0)?, u(1)?);
            if !pow2(align) {
                return None;
            }
            return Some(Ok(Val::S(self.alloc_block(size, align, frames))));
        }
        if Some(f) == free {
            let (p, size, align) = (u(0)?, u(1)?, u(2)?);
            if size != 0 {
                if let Err(r) = check_free(self, p, size, align) {
                    return Some(Err(r));
                }
                let at = self.trace(frames);
                self.end_block(p, &at);
                self.mem.heap_free(p, at);
            }
            return Some(Ok(Val::unit()));
        }
        if Some(f) == try_realloc {
            let (p, old, align, new) = (u(0)?, u(1)?, u(2)?, u(3)?);
            if !pow2(align) {
                return None;
            }
            if old == 0 {
                return Some(Ok(Val::S(self.alloc_block(new, align, frames))));
            }
            if let Err(r) = check_free(self, p, old, align) {
                return Some(Err(r));
            }
            let q = if new == 0 { align } else { self.alloc_block(new, align, frames) };
            if q == 0 {
                return Some(Ok(Val::S(0)));
            }
            if new != 0 && self.mem.copy(q, p, old.min(new)).is_err() {
                return self.mem.take_report().map(Err);
            }
            let at = self.trace(frames);
            if let Some(bt) = &mut self.bt {
                // the new block takes the old bytes with their tags, and the places that were
                // moved out of stay moved out
                bt.copied(q, p, old.min(new));
                let empty = self.mem.alloc_span(p).map_or(Vec::new(), |(id, _, _)| bt.tracker.moved(id));
                if !empty.is_empty()
                    && new != 0
                    && let Some((id, _)) = bt.locate(&self.mem, q)
                {
                    bt.tracker.set_moved(id, &empty);
                }
            }
            self.end_block(p, &at);
            self.mem.heap_free(p, at);
            return Some(Ok(Val::S(q)));
        }
        None
    }

    /// Borrow tracking: the root of each `static` that is not `mut` is `Frozen`, and `Cell` on
    /// the bytes its type puts inside an `UnsafeCell`.
    fn freeze_statics(&mut self) -> R<()> {
        if self.bt.is_none() {
            return Ok(());
        }
        let m = self.m;
        for (k, st) in m.statics.iter().enumerate() {
            let Some((id, _, size)) = self.statics.get(k).and_then(|&a| self.mem.alloc_span(a)) else {
                continue;
            };
            if let Some(bt) = &mut self.bt {
                bt.static_ids.insert(id);
            }
            if m.frozen.get(k) != Some(&true) {
                continue;
            }
            let cells = self.cells(st.ty)?;
            if let Some(bt) = &mut self.bt {
                bt.tracker.begin_static(id, size, &cells, NO_EV);
            }
        }
        Ok(())
    }

    /// M18: the fn value `out` of a closure literal, with an environment of its own: a copy,
    /// in a slot of the current frame, of the one `fn.make` instruction `i` names, with the
    /// capture references it holds. Without stack left the fn value keeps the environment of
    /// its literal.
    fn own_env(&mut self, body: &Body, i: u32, out: Val) -> R<Val> {
        let Val::Pair(code, env) = out else { return Ok(out) };
        let t = match body.args(i).get(1) {
            Some(&Operand::Value(e)) => body.value_ty(e).and_then(|t| self.pointee.get(t.index()).copied()),
            _ => None,
        };
        let Some(t) = t.flatten() else { return Ok(out) };
        let (size, align) = {
            let l = self.layout(t)?;
            (l.size, l.align)
        };
        if size == 0 {
            return Ok(out);
        }
        let Some(copy) = self.mem.stack_alloc(size, align) else { return Ok(out) };
        if self.mem.copy(copy, env, size).is_err() {
            let _ = self.mem.take_report();
            return Ok(out);
        }
        if let Some(bt) = &mut self.bt {
            let held: Vec<_> =
                bt.shadow.range(env..env.saturating_add(size)).map(|(&a, &w)| (a, w)).collect();
            for (a, w) in held {
                bt.shadow.insert(copy + (a - env), w);
            }
        }
        Ok(Val::Pair(code, copy))
    }

    /// Borrow tracking: the allocation at `addr` ends (M3): every tag of it is `Dead`.
    fn end_block(&mut self, addr: u64, at: &Trace) {
        if let Some(bt) = &mut self.bt
            && let Some((id, _, _)) = self.mem.alloc_span(addr)
        {
            bt.end_alloc(&self.mem, id, at.first().map_or(NO_EV, |a| ev(a.func, a.inst)));
        }
    }

    /// Checking mode: the runtime-allocator blocks live now (at exit: the leaks), as
    /// (address, size, allocation site).
    pub fn live_heap_blocks(&self) -> Vec<(u64, u64, Trace)> {
        self.mem.live_heap_blocks()
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

    /// The tags of the reference words of `v`, a value of type `t`, found by the type: of an
    /// enum the live variant; not behind a pointer, not in a fn value.
    fn ref_tags(&self, t: TypeId, v: &Val, out: &mut Vec<Tag>) -> R<()> {
        // a list, not recursion: a value may nest as deep as the type table allows
        let mut work: Vec<(TypeId, &Val)> = vec![(t, v)];
        while let Some((t, v)) = work.pop() {
            match self.m.types.key(t) {
                Some(TypeKey::Ref { .. }) => {
                    if let Some((_, tag)) = v.word().filter(|w| w.1 != ROOT) {
                        out.push(tag);
                    }
                    continue;
                }
                Some(TypeKey::Tuple(_) | TypeKey::Array { .. } | TypeKey::Adt { .. }) => {}
                _ => continue,
            }
            let Val::Agg(vs) = v else { continue };
            match &self.layout(t)?.shape {
                Shape::Fields(fs) => work.extend(fs.iter().zip(vs.iter()).map(|(f, fv)| (f.1, fv))),
                Shape::Array { elem, .. } => work.extend(vs.iter().map(|fv| (*elem, fv))),
                Shape::Enum(vars) => {
                    let live = vs.first().and_then(Val::bits).and_then(|k| vars.get(k as usize));
                    let fields = live.map_or(&[][..], |f| f.as_slice());
                    work.extend(fields.iter().zip(vs.iter().skip(1)).map(|(f, fv)| (f.1, fv)));
                }
                Shape::Scalar | Shape::Fat | Shape::Bytes => {}
            }
        }
        Ok(())
    }

    /// Enters `f`. Borrow tracking: the call at `at` protects the references it passes until
    /// it returns (M23).
    fn push_frame(&mut self, frames: &mut Vec<Frame>, f: FuncId, args: Vec<Val>, at: Ev) -> R<()> {
        let m = self.m;
        let func = m.func(f).ok_or(InterpError::UnknownFunc(f))?;
        let body = func.body.as_ref().ok_or(InterpError::NoBody(f))?;
        if args.len() != func.params.len() {
            return Err(InterpError::ArgCount { func: f, got: args.len(), want: func.params.len() });
        }
        let mut guards = Vec::new();
        if self.bt.is_some() {
            for (&t, a) in func.params.iter().zip(&args) {
                self.ref_tags(t, a, &mut guards)?;
            }
        }
        if let Some(bt) = &mut self.bt {
            for &g in &guards {
                bt.tracker.protect(g, at);
            }
        }
        let sig = if self.bt.is_some() { self.signature_entry(f, &args) } else { None };
        let mut regs = vec![Val::S(0); body.values.len()];
        for (p, a) in body.params(Block(0)).iter().zip(args) {
            if let Some(r) = regs.get_mut(p.index()) {
                *r = a;
            }
        }
        let mark = self.mem.stack_mark();
        frames.push(Frame { func: f, regs, block: 0, pc: 0, mark, dest: None, guards, sig });
        Ok(())
    }

    /// Runs `entry` with `args` to completion.
    pub fn run(&mut self, entry: FuncId, args: Vec<Val>) -> R<Outcome> {
        let m = self.m;
        if let Some(bt) = &mut self.bt {
            // the frames of a run that ended in a trap protect nothing
            bt.tracker.unprotect_all();
        }
        let mut frames: Vec<Frame> = Vec::new();
        self.push_frame(&mut frames, entry, args, NO_EV)?;
        loop {
            if let Some(bt) = &mut self.bt
                && bt.tracker.due()
            {
                bt.collect(&frames);
            }
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
                    match self.exec(body, fid, &mut fr.regs, i) {
                        Ok(Step::Next) => continue,
                        Ok(s) => break (Some(s), i),
                        Err(e) => return self.failed(e, &frames),
                    }
                }
                // every word that holds a tag is in a frame or in memory here
                if self.bt.as_ref().is_some_and(|bt| bt.tracker.due()) {
                    self.steps -= 1;
                    break (None, 0);
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
                        let at = fr.sig.is_some().then(|| self.trace(&frames));
                        let done = frames.pop().ok_or(malformed("frame"))?;
                        if let (Some(e), Some(at)) = (&done.sig, &at) {
                            self.signature_return(fid, e, v.as_ref(), at);
                        }
                        if let Some(bt) = &mut self.bt {
                            for &g in &done.guards {
                                bt.tracker.unprotect(g);
                            }
                            // the frame's locals and temporaries end (M13)
                            let at = ev(fid, blk.insts.start + done.pc.saturating_sub(1));
                            for id in self.mem.slots_since(done.mark) {
                                bt.end_alloc(&self.mem, id, at);
                            }
                            if bt.report.is_some() {
                                return self.failed(malformed("borrow violation"), &frames);
                            }
                        }
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
                    if self.alloc_fns.is_some()
                        && let Some(res) = self.alloc_call(f, &args, &frames)
                    {
                        if self.bt.as_ref().is_some_and(|bt| bt.report.is_some()) {
                            return self.failed(malformed("borrow violation"), &frames);
                        }
                        let v = match res {
                            Ok(v) => v,
                            Err(r) => return Ok(Outcome::MemError(r)),
                        };
                        if let (Some(caller), Some(d)) = (frames.last_mut(), dest)
                            && let Some(r) = caller.regs.get_mut(d.index())
                        {
                            *r = v;
                        }
                        continue;
                    }
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
                    self.push_frame(&mut frames, f, args, ev(fid, inst))?;
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
                Some(Val::S(x) | Val::T(x, _)) => Ok(*x),
                _ => Err(malformed("scalar expected")),
            },
            _ => self.eval(body, regs, o)?.bits().ok_or(malformed("scalar expected")),
        }
    }

    fn pair(&self, body: &Body, regs: &[Val], o: Operand) -> R<(u64, u64)> {
        match self.eval(body, regs, o)? {
            Val::Pair(a, n) | Val::PairT(a, n, _) => Ok((a, n)),
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

    /// The value of type `t` at `addr`. Aggregates are read part by part with an explicit stack
    /// (a value nested [`MAX_TYPE_DEPTH`] levels deep needs no deeper native stack).
    fn load_val(&self, t: TypeId, addr: u64, depth: usize, func: FuncId, site: Site) -> R<Val> {
        let mut stack: Vec<Open> = Vec::new();
        let mut todo = Some((t, addr));
        let mut done: Option<Val> = None;
        loop {
            if let Some((t, addr)) = todo.take() {
                if depth + stack.len() > MAX_TYPE_DEPTH {
                    return Err(InterpError::Unsupported("load nesting too deep".into()));
                }
                match self.load_one(t, addr, func, site)? {
                    Loaded::Val(v) => done = Some(v),
                    Loaded::Parts(parts, out) => stack.push((parts, 0, out)),
                }
            }
            if let Some(v) = done.take() {
                match stack.last_mut() {
                    Some(top) => top.2.push(v),
                    None => return Ok(v),
                }
            }
            let Some(top) = stack.last_mut() else { return Err(malformed("load")) };
            if let Some(&part) = top.0.get(top.1) {
                top.1 += 1;
                todo = Some(part);
            } else if let Some((_, _, out)) = stack.pop() {
                done = Some(Val::agg(out));
            }
        }
    }

    /// One level of [`Interp::load_val`]: a scalar's value, or an aggregate's parts and the
    /// values that precede them (an enum's tag).
    fn load_one(&self, t: TypeId, addr: u64, func: FuncId, site: Site) -> R<Loaded> {
        let fault = Self::fault(func, site);
        let l = self.layout(t)?;
        Ok(match &l.shape {
            Shape::Scalar => {
                let raw = self.mem.read_uint(addr, l.size).map_err(fault)?;
                let x = match self.cls(t) {
                    Cls::Int(it) => it.norm(raw),
                    Cls::Bool => (raw != 0) as u64,
                    _ => raw,
                };
                let tag = match &self.bt {
                    Some(bt) if l.size == 8 => bt.loaded(addr, x),
                    _ => ROOT,
                };
                Loaded::Val(Val::tagged(x, tag))
            }
            Shape::Fat => {
                let a = self.mem.read_uint(addr, 8).map_err(&fault)?;
                let n = self.mem.read_uint(addr.wrapping_add(8), 8).map_err(&fault)?;
                match self.bt.as_ref().map_or(ROOT, |bt| bt.loaded(addr, a)) {
                    ROOT => Loaded::Val(Val::Pair(a, n)),
                    tag => Loaded::Val(Val::PairT(a, n, tag)),
                }
            }
            Shape::Fields(fs) => {
                let parts = fs.iter().map(|&(off, ft)| (ft, addr.wrapping_add(off))).collect();
                Loaded::Parts(parts, Vec::with_capacity(fs.len()))
            }
            Shape::Array { elem, stride, len } => {
                if *len > MAX_AGG_LEN {
                    return Err(InterpError::Unsupported("array value too large".into()));
                }
                let parts = (0..*len).map(|k| (*elem, addr.wrapping_add(k * stride))).collect();
                Loaded::Parts(parts, Vec::with_capacity(*len as usize))
            }
            Shape::Enum(vs) => {
                let tag = self.mem.read_uint(addr, 4).map_err(&fault)?;
                let fs = vs.get(tag as usize).ok_or(InterpError::WrongVariant { func, site })?;
                let mut out = Vec::with_capacity(fs.len() + 1);
                out.push(Val::S(tag));
                let parts = fs.iter().map(|&(off, ft)| (ft, addr.wrapping_add(off))).collect();
                Loaded::Parts(parts, out)
            }
            Shape::Bytes => Loaded::Val(self.load_bytes(t, addr, func, site)?),
        })
    }

    /// Stores `v` of type `t` at `addr`: each aggregate's own bytes (an enum's zeroing and tag)
    /// before its parts, parts in order, with an explicit stack (no deep native stack).
    fn store_val(&mut self, t: TypeId, addr: u64, v: &Val, depth: usize, func: FuncId, site: Site) -> R<()> {
        let mut stack: Vec<(TypeId, u64, &Val, usize)> = vec![(t, addr, v, depth)];
        while let Some((t, addr, v, depth)) = stack.pop() {
            if depth > MAX_TYPE_DEPTH {
                return Err(InterpError::Unsupported("store nesting too deep".into()));
            }
            let fault = Self::fault(func, site);
            let l = self.layout(t)?.clone();
            let first = stack.len();
            match (&l.shape, v) {
                (Shape::Scalar, &(Val::S(x) | Val::T(x, _))) => {
                    self.mem.write_uint(addr, l.size, x).map_err(fault)?;
                    if let Some(bt) = &mut self.bt {
                        bt.stored(addr, l.size, x, v.word().map_or(ROOT, |w| w.1));
                    }
                }
                (Shape::Fat, &(Val::Pair(a, n) | Val::PairT(a, n, _))) => {
                    self.mem.write_uint(addr, 8, a).map_err(&fault)?;
                    self.mem.write_uint(addr.wrapping_add(8), 8, n).map_err(&fault)?;
                    if let Some(bt) = &mut self.bt {
                        bt.stored(addr, 8, a, v.word().map_or(ROOT, |w| w.1));
                    }
                }
                (Shape::Fields(fs), Val::Agg(vs)) if fs.len() == vs.len() => {
                    for (&(off, ft), fv) in fs.iter().zip(vs.iter()) {
                        stack.push((ft, addr.wrapping_add(off), fv, depth + 1));
                    }
                }
                (Shape::Array { elem, stride, len }, Val::Agg(vs)) if *len == vs.len() as u64 => {
                    for (k, fv) in vs.iter().enumerate() {
                        stack.push((*elem, addr.wrapping_add(k as u64 * stride), fv, depth + 1));
                    }
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
                        stack.push((ft, addr.wrapping_add(off), fv, depth + 1));
                    }
                }
                (Shape::Bytes, _) => self.store_bytes(t, addr, v, func, site)?,
                _ => return Err(malformed("stored value does not match its type")),
            }
            // the first part is stored first
            stack[first..].reverse();
        }
        Ok(())
    }

    /// The bytes of a value of type `t` that are inside an `UnsafeCell` (M21): of an enum with
    /// such a payload, every byte.
    fn cells(&mut self, t: TypeId) -> R<Rc<[(u64, u64)]>> {
        // Each type after the types of its parts, with a list of the types still open instead
        // of recursion: a type may nest as deep as the type table allows.
        let mut open = vec![t];
        while let Some(&x) = open.last() {
            let known = |t: &TypeId| self.bt.as_ref().is_some_and(|bt| bt.cells.contains_key(t));
            if known(&x) {
                open.pop();
                continue;
            }
            let l = self.layout(x)?;
            let cell =
                matches!(self.m.types.key(x), Some(TypeKey::Adt { def, .. }) if Some(*def) == self.m.cell);
            let parts: Vec<TypeId> = match &l.shape {
                _ if cell => Vec::new(),
                Shape::Fields(fs) => fs.iter().map(|f| f.1).collect(),
                Shape::Array { elem, .. } => vec![*elem],
                Shape::Enum(vs) => vs.iter().flatten().map(|f| f.1).collect(),
                Shape::Scalar | Shape::Fat | Shape::Bytes => Vec::new(),
            };
            let before = open.len();
            open.extend(parts.iter().filter(|p| !known(p)));
            if open.len() > before {
                continue;
            }
            let of = |t: &TypeId| self.bt.as_ref().and_then(|bt| bt.cells.get(t)).cloned();
            let mut out = Vec::new();
            match &l.shape {
                _ if cell => push_range(&mut out, 0, l.size),
                Shape::Fields(fs) => {
                    for (off, ft) in fs {
                        for &(a, b) in of(ft).iter().flat_map(|c| c.iter()) {
                            push_range(&mut out, off.saturating_add(a), off.saturating_add(b));
                        }
                    }
                }
                Shape::Array { elem, stride, len } => {
                    let one = of(elem).unwrap_or_else(|| Rc::from(Vec::new()));
                    repeat_ranges(&mut out, &one, 0, *stride, *len);
                }
                Shape::Enum(_) => {
                    if parts.iter().any(|p| of(p).is_some_and(|c| !c.is_empty())) {
                        push_range(&mut out, 0, l.size);
                    }
                }
                Shape::Scalar | Shape::Fat | Shape::Bytes => {}
            }
            if let Some(bt) = &mut self.bt {
                bt.cells.insert(x, out.into());
            }
            open.pop();
        }
        let none = || Rc::from(Vec::new());
        Ok(self.bt.as_ref().and_then(|bt| bt.cells.get(&t)).cloned().unwrap_or_else(none))
    }

    /// The tag of the word an operand holds.
    fn tag_of(regs: &[Val], o: Operand) -> Tag {
        match o {
            Operand::Value(v) => regs.get(v.index()).and_then(Val::word).map_or(ROOT, |w| w.1),
            _ => ROOT,
        }
    }

    fn violation(&mut self, r: Box<Report>) -> InterpError {
        if let Some(bt) = &mut self.bt {
            bt.report.get_or_insert(r);
        }
        malformed("borrow violation")
    }

    /// Borrow tracking: the read or write of `len` bytes at `addr` through `tag`.
    fn track(&mut self, tag: Tag, addr: u64, len: u64, write: bool, f: FuncId, i: u32) -> R<()> {
        self.track_at(tag, addr, len, write, ev(f, i))
    }

    fn track_at(&mut self, tag: Tag, addr: u64, len: u64, write: bool, at: Ev) -> R<()> {
        let Some(bt) = &mut self.bt else { return Ok(()) };
        if len == 0 {
            return Ok(());
        }
        let Some((id, off)) = bt.locate(&self.mem, addr) else { return Ok(()) };
        let hi = off.saturating_add(len);
        let r = if write {
            bt.tracker.write(id, tag, off, hi, at)
        } else {
            bt.tracker.read(id, tag, off, hi, at)
        };
        r.map_err(|r| self.violation(r))
    }

    /// Borrow tracking: the access of a vector instruction to `len` bytes at `addr`, through
    /// the tag of its memory operand `o`.
    pub(super) fn track_vec(&mut self, regs: &[Val], o: Operand, addr: u64, len: u64, write: bool) -> R<()> {
        let Some(bt) = &self.bt else { return Ok(()) };
        self.track_at(Self::tag_of(regs, o), addr, len, write, bt.at)
    }

    /// `intrinsic.copy_bytes(dst, src, n)`: `n` bytes move from `src` to `dst` with their
    /// initialisation, as bytes: no typed value is read or written, so no byte needs to be
    /// initialised. Borrow tracking: a read of the source range through the tag of `src`, then a
    /// write of the destination range through the tag of `dst` (M1, M2, M20), and the tagged
    /// words move with their bytes ([`Borrows::copied`]). Checking mode: ranges that share a
    /// byte are a `copy-overlap` error (the native copy may run in either direction).
    #[allow(clippy::too_many_arguments)]
    fn copy_bytes(
        &mut self,
        regs: &[Val],
        args: &[Operand],
        dst: u64,
        src: u64,
        n: u64,
        f: FuncId,
        i: u32,
        site: Site,
    ) -> R<()> {
        if n == 0 {
            return Ok(());
        }
        if self.bt.is_some() {
            let (Some(&d), Some(&s)) = (args.first(), args.get(1)) else {
                return Err(malformed("copy_bytes"));
            };
            self.track(Self::tag_of(regs, s), src, n, false, f, i)?;
            self.track(Self::tag_of(regs, d), dst, n, true, f, i)?;
        }
        let fault = Self::fault(f, site);
        if self.mem.is_checked() && dst < src.saturating_add(n) && src < dst.saturating_add(n) {
            self.mem.fail_overlap(dst, n);
            return Err(fault(Fault { addr: dst, len: n, write: true }));
        }
        self.mem.copy(dst, src, n).map_err(fault)?;
        if let Some(bt) = &mut self.bt {
            bt.copied(dst, src, n);
        }
        Ok(())
    }

    /// Borrow tracking: the access of a CPU atomic intrinsic through the tag of its pointer
    /// operand: a read, and a write when it stores (a `cas` that fails stores nothing).
    fn track_atomic(
        &mut self,
        name: &str,
        regs: &[Val],
        args: &[Operand],
        vals: &[Val],
        f: FuncId,
        i: u32,
    ) -> R<()> {
        let Some((op, size)) = name.strip_prefix("atomic_").and_then(|r| r.rsplit_once('_')) else {
            return Ok(());
        };
        let size = match size {
            "u32" => 4u64,
            "u64" => 8,
            _ => return Ok(()),
        };
        let (Some(&o), Some(p)) = (args.first(), vals.first().and_then(Val::bits)) else { return Ok(()) };
        let mask = if size == 8 { u64::MAX } else { u64::from(u32::MAX) };
        let write = match op {
            "load" => false,
            "cas" => match (self.mem.read_uint(p, size), vals.get(1).and_then(Val::bits)) {
                (Ok(old), Some(want)) => old == want & mask,
                _ => false,
            },
            _ => true,
        };
        self.track(Self::tag_of(regs, o), p, size, write, f, i)
    }

    /// Borrow tracking: the result of instruction `i` with the tag it carries. A marked result
    /// is a new borrow (M4); the sum or difference of a tagged word and an untagged one keeps
    /// the tag (M20).
    fn retag(&mut self, body: &Body, f: FuncId, regs: &[Val], i: u32, out: Val) -> R<Val> {
        let Some(bt) = &self.bt else { return Ok(out) };
        let res = body.insts.result.get(i as usize).copied().flatten();
        let op = body.insts.op.get(i as usize).copied();
        let carried = match (op, &out, body.args(i)) {
            (Some(Op::Add | Op::AddWrap), &Val::S(x), &[a, b]) => {
                match (Self::tag_of(regs, a), Self::tag_of(regs, b)) {
                    (t, ROOT) | (ROOT, t) => Some(Val::tagged(x, t)),
                    _ => None,
                }
            }
            (Some(Op::Sub | Op::SubWrap), &Val::S(x), &[a, b]) if Self::tag_of(regs, b) == ROOT => {
                Some(Val::tagged(x, Self::tag_of(regs, a)))
            }
            _ => None,
        };
        let out = carried.unwrap_or(out);
        let Some(res) = res else { return Ok(out) };
        let code = bt.marks.get(f.index()).and_then(|m| m.get(res.index())).copied().unwrap_or(0);
        if code == Mark::None as u8 {
            return Ok(out);
        }
        let Some(rt) = body.value_ty(res) else { return Ok(out) };
        let at = ev(f, i);
        match Mark::ALL.get(code as usize).copied().unwrap_or(Mark::None) {
            Mark::None => Ok(out),
            Mark::Env => self.own_env(body, i, out),
            Mark::Shared => self.borrow_word(rt, out, false, false, at),
            Mark::Mut => self.borrow_word(rt, out, true, false, at),
            Mark::SharedRaw => self.borrow_word(rt, out, false, true, at),
            Mark::MutRaw => self.borrow_word(rt, out, true, true, at),
            Mark::Repeat => Err(self.violation(Tracker::repeat(at))),
            m => {
                self.place_event(m, rt, &out, at)?;
                Ok(out)
            }
        }
    }

    /// An event of the side table on the place the pointer `v` of type `pt` names: a local or
    /// temporary begins or ends in its slot (M13, M14), or the value leaves the place (M9).
    fn place_event(&mut self, m: Mark, pt: TypeId, v: &Val, at: Ev) -> R<()> {
        let Some((addr, tag)) = v.word() else { return Ok(()) };
        let size = match self.pointee.get(pt.index()).copied().flatten() {
            Some(t) if m == Mark::Gone => self.layout(t)?.size,
            _ => 0,
        };
        let Some(bt) = &mut self.bt else { return Ok(()) };
        let r = match m {
            Mark::Begin => {
                let Some((id, base, len)) = self.mem.alloc_span(addr) else { return Ok(()) };
                bt.forget(base, len);
                bt.tracker.restart(id, at)
            }
            Mark::End => {
                let Some((id, base, len)) = self.mem.alloc_span(addr) else { return Ok(()) };
                bt.forget(base, len);
                bt.tracker.end_alloc(id, at)
            }
            Mark::Const => {
                // static data from its first evaluation on: no evaluation after it is an event
                let Some((id, _, _)) = self.mem.alloc_span(addr) else { return Ok(()) };
                if !bt.static_ids.insert(id) {
                    bt.quiet = Some(addr);
                }
                Ok(())
            }
            _ => {
                let Some((id, off)) = bt.locate(&self.mem, addr) else { return Ok(()) };
                bt.tracker.gone(id, tag, off, off.saturating_add(size), at)
            }
        };
        r.map_err(|r| self.violation(r))
    }

    /// M4: the reference `v` of type `rt` with a new tag, a child of the tag it carries, for
    /// the place its address names: the pointee of `rt`, of a fat reference its elements.
    fn borrow_word(&mut self, rt: TypeId, v: Val, mutable: bool, raw: bool, at: Ev) -> R<Val> {
        let Some((addr, parent)) = v.word() else { return Ok(v) };
        // the place: `n` values of type `elem` (no type: bytes)
        let inner = self.pointee.get(rt.index()).copied().flatten();
        let (elem, n) = match (inner.and_then(|t| self.m.types.key(t)), &v) {
            (Some(TypeKey::Slice { elem, .. }), Val::Pair(_, n) | Val::PairT(_, n, _)) => (Some(*elem), *n),
            (_, Val::Pair(_, n) | Val::PairT(_, n, _)) => (None, *n),
            _ => (inner, 1),
        };
        let unit = match elem {
            Some(t) => self.layout(t)?.size,
            None => 1,
        };
        let one = match elem {
            Some(t) if !mutable => self.cells(t)?,
            _ => Rc::from(Vec::new()),
        };
        let Some(bt) = &mut self.bt else { return Ok(v) };
        let Some((id, off)) = bt.locate(&self.mem, addr) else { return Ok(v) };
        if bt.tracker.exhausted() {
            return Err(InterpError::Unsupported("borrow tracking: every tag is used up".into()));
        }
        let mut cells = Vec::new();
        repeat_ranges(&mut cells, &one, off, unit, n);
        let kind = if mutable { Kind::Mut } else { Kind::Shared };
        let bytes = off..off.saturating_add(n.saturating_mul(unit));
        match bt.tracker.borrow(id, parent, kind, bytes, &cells, at) {
            Ok(tag) => {
                if raw {
                    bt.tracker.mark_raw(tag);
                }
                Ok(match v {
                    Val::Pair(a, n) | Val::PairT(a, n, _) => Val::PairT(a, n, tag),
                    _ => Val::T(addr, tag),
                })
            }
            Err(r) => Err(self.violation(r)),
        }
    }

    /// M20: a value of type `t` read through a raw pointer, with a new tag for each reference
    /// in it (found by the type: of an enum the live variant), a child of the tag stored there.
    /// The value is walked with a list, not by recursion: it may nest as deep as the type table
    /// allows, deeper than a small native stack holds frames.
    fn reborrow_refs(&mut self, t: TypeId, v: Val, at: Ev) -> R<Val> {
        // per open aggregate: its parts still to do (`None`: a part kept as it is), and the
        // values done
        let mut stack: Vec<(std::vec::IntoIter<Part>, Vec<Val>)> = Vec::new();
        let mut todo = Some((t, v));
        loop {
            let mut done = None;
            if let Some((t, v)) = todo.take() {
                if stack.len() > MAX_TYPE_DEPTH {
                    return Err(InterpError::Unsupported("type nesting too deep".into()));
                }
                match self.reborrow_one(t, v, at)? {
                    Ok(v) => done = Some(v),
                    Err(parts) => {
                        let n = parts.len();
                        stack.push((parts.into_iter(), Vec::with_capacity(n)));
                    }
                }
            }
            if done.is_none() {
                let Some(top) = stack.last_mut() else { return Err(malformed("reborrow")) };
                match top.0.next() {
                    Some((Some(pt), pv)) => todo = Some((pt, pv)),
                    Some((None, pv)) => top.1.push(pv),
                    None => done = stack.pop().map(|(_, out)| Val::agg(out)),
                }
            }
            if let Some(v) = done {
                match stack.last_mut() {
                    Some(top) => top.1.push(v),
                    None => return Ok(v),
                }
            }
        }
    }

    /// One level of [`Interp::reborrow_refs`]: the value done (a reference with its new tag, a
    /// value that holds none), or the parts of an aggregate with their types.
    fn reborrow_one(&mut self, t: TypeId, v: Val, at: Ev) -> R<Result<Val, Vec<Part>>> {
        let types = &self.m.types;
        match types.key(t) {
            Some(&TypeKey::Ref { mutable, .. }) => return self.borrow_word(t, v, mutable, true, at).map(Ok),
            Some(TypeKey::Tuple(_) | TypeKey::Array { .. } | TypeKey::Adt { .. }) => {}
            _ => return Ok(Ok(v)),
        }
        let Val::Agg(vs) = &v else { return Ok(Ok(v)) };
        let tys: Vec<Option<TypeId>> = match &self.layout(t)?.shape {
            Shape::Fields(fs) => fs.iter().map(|f| Some(f.1)).collect(),
            Shape::Array { elem, .. } => vec![Some(*elem); vs.len()],
            Shape::Enum(vars) => {
                let live = vs.first().and_then(Val::bits).and_then(|k| vars.get(k as usize));
                std::iter::once(None).chain(live.into_iter().flatten().map(|f| Some(f.1))).collect()
            }
            Shape::Scalar | Shape::Fat | Shape::Bytes => return Ok(Ok(v)),
        };
        Ok(Err(vs.iter().enumerate().map(|(k, fv)| (tys.get(k).copied().flatten(), fv.clone())).collect()))
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
                let it = match self.ocls(body, arg(0)?).or(self.ocls(body, arg(1)?)) {
                    Some(Cls::Int(t)) => t,
                    _ => match ty.and_then(|t| self.m.types.key(t)) {
                        Some(TypeKey::Tuple(ts)) => match ts.first().map(|t| self.cls(*t)) {
                            Some(Cls::Int(t)) => t,
                            _ => IntTy::new(64, false),
                        },
                        _ => IntTy::new(64, false),
                    },
                };
                let (hi, lo) = arith::mul_wide(it, s!(0), s!(1));
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
                    let r = arith::convert(kind, from, to, x).ok_or(malformed("cast"))?;
                    // the same word keeps its tag
                    let tag = if r == x { Self::tag_of(regs, src) } else { ROOT };
                    Val::tagged(r, tag)
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
                    Some(a) => {
                        if self.cfg.check {
                            self.mem.note_alloc_site(a, vec![At::of(fid, body, i)]);
                        }
                        Val::S(a)
                    }
                    None => return Ok(Step::Trap(PanicKind::StackOverflow)),
                }
            }
            Op::Load => {
                let t = ty.ok_or(malformed("load type"))?;
                let a = s!(0);
                if self.bt.is_some() {
                    let n = self.layout(t)?.size;
                    self.track(Self::tag_of(regs, arg(0)?), a, n, false, fid, i)?;
                }
                let v = self.load_val(t, a, 0, fid, site)?;
                let raw = |p: Operand| match p {
                    Operand::Value(p) => {
                        matches!(body.value_ty(p).and_then(|t| m.types.key(t)), Some(TypeKey::RawPtr { .. }))
                    }
                    _ => false,
                };
                if self.bt.is_some() && raw(arg(0)?) { self.reborrow_refs(t, v, ev(fid, i))? } else { v }
            }
            // hints: no effect on the program's state
            Op::Prefetch(_) | Op::FenceStream => return Ok(Step::Next),
            Op::Store | Op::StoreStream => {
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
                let quiet = self.bt.as_mut().and_then(|bt| bt.quiet.take()) == Some(a);
                if self.bt.is_some() && !quiet {
                    let n = self.layout(t)?.size;
                    self.track(Self::tag_of(regs, p), a, n, true, fid, i)?;
                }
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
                            Some(TypeKey::Array { elem, .. } | TypeKey::Slice { elem, .. }) => {
                                self.layout(*elem)?.size
                            }
                            _ => self.layout(inner)?.size,
                        };
                        (k as i64 as u64).wrapping_mul(stride)
                    }
                };
                let (base, tag) = self.eval(body, regs, p)?.word().ok_or(malformed("pointer value"))?;
                Val::tagged(base.wrapping_add(off), tag)
            }
            Op::SliceMake => match Self::tag_of(regs, arg(0)?) {
                ROOT => Val::Pair(s!(0), s!(1)),
                tag => Val::PairT(s!(0), s!(1), tag),
            },
            Op::FnMake => {
                let Operand::Func(f) = arg(0)? else { return Err(malformed("fn.make code")) };
                Val::Pair(f.0 as u64, s!(1))
            }
            Op::SliceLen => Val::S(self.pair(body, regs, arg(0)?)?.1),
            Op::SlicePtr => Val::tagged(self.pair(body, regs, arg(0)?)?.0, Self::tag_of(regs, arg(0)?)),
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
            Op::Vec(v) => {
                if let Some(bt) = &mut self.bt {
                    bt.at = ev(fid, i);
                }
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
                    vals.push(self.eval(body, regs, a)?.untagged());
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
                if name == "copy_bytes" {
                    let u = |k: usize| vals.get(k).and_then(Val::bits).ok_or(malformed("copy_bytes operand"));
                    let (dst, src, n) = (u(0)?, u(1)?, u(2)?);
                    self.copy_bytes(regs, args, dst, src, n, fid, i, site)?;
                    return Ok(Step::Next);
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
                    if self.bt.is_some() {
                        self.track_atomic(name, regs, args, &vals, fid, i)?;
                    }
                    self.host.intrinsic(name, &vals, &mut self.mem).map_err(InterpError::Host)?
                };
                // a host access that failed a check (the host may have answered `EFAULT`)
                if self.mem.has_report() {
                    return Err(malformed("memory error in a host call"));
                }
                match res {
                    Some(v) => v,
                    None => return Ok(Step::Next),
                }
            }
            Op::Panic(k) => return Ok(Step::Trap(k)),
        };
        let out = if self.bt.is_some() { self.retag(body, fid, regs, i, out)? } else { out };
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
