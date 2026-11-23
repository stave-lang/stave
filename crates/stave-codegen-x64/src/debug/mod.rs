//! x86-64 debug backend: TPDE-style, two passes per function.
//!
//! 1. [`analysis`]: reverse postorder, loop depth, and liveness (which values cross a block
//!    boundary, which are used at all).
//! 2. `func` (with `ops`, `cast`, `intrin`): one pass over the blocks in RPO doing instruction
//!    selection, greedy local register allocation (a per-block register cache over per-value
//!    frame homes), and encoding.
//!
//! Every value has a frame *home* (an 8-byte slot, or the bytes of an aggregate). Inside a block
//! values live in registers (12 GPRs, 14 XMMs); values used in another block are written to their
//! home when defined, so block boundaries need no register state. Block parameters are written by
//! the branches (parallel copy through scratch memory when a source is overwritten). Calls follow
//! [`abi`] (the Stave calling convention = System V AMD64, documented there).
//!
//! Release code generation ([`Options::release`]) keeps this structure and changes where values
//! live and how blocks are laid out:
//!
//! - A value used in another block, and every param of a later block, owns a register from its
//!   definition to its last use when stave-opt's `regalloc` finds one: `rbx`, `r12`–`r15` across
//!   calls; `r8`, `r9`, `xmm6`–`xmm13` where no call intervenes. The cache keeps to the other
//!   call-clobbered registers; the rest of the values live in homes as above.
//! - Vector and mask values live in `xmm` / `ymm` registers like floats: in the cache inside a
//!   block, in `xmm6`–`xmm13` (their `ymm` views for 32 bytes) across blocks. No vector register
//!   survives a call, so a vector that is live across one is in its home there.
//! - Branch arguments and the register arguments of a call are parallel moves from where the
//!   values are.
//! - Blocks are emitted loop by loop with the blocks that cannot return last (stave-opt
//!   `layout`); callee-saved registers have fixed save slots and the first `ret` carries the
//!   epilogue.
//! - A compare read only by its block's `cbr` is flags and a jump; an address read by one load or
//!   store is its addressing mode; a loop counter's or an accumulator's next value computes into
//!   the param's register; a `vec.load` read only by the next vector op is that op's memory
//!   operand (VEX).
//!
//! A function that uses `cpuid`, or needs more temporaries in one instruction than the cache has
//! left, is compiled the debug way.
//!
//! Traps (overflow `jo`/`jc`/range checks, division by zero, `MIN / -1`, shift amount ≥ width,
//! bounds, `panic.*`) jump to one out-of-line stub per (kind, site) that calls
//! `runtime.panic.trap(kind, site)` (contracts/intrinsics.md, runtime/README.md). Without that
//! function in the module the backend emits `__stave_panic_fallback`, which writes
//! `panic: <Kind>` to fd 2 and exits 101.
//!
//! Module level ([`emit_module`], [`link`], [`compile_module`]): one symbol per function with a
//! body in canonical instance order, string literals, the compiler-emitted stubs `_start`
//! (stave-obj `stub::start`), `__stave_main`, `__stave_signal`, `__stave_thread_start`, and the tables
//! `__stave_site_table`, `__stave_pc_table`, and `__stave_stack_table` (layouts in
//! contracts/intrinsics.md).
//!
//! Features. Every function is compiled for its own feature set: the module baseline
//! ([`Options::features`]) and the function's `needs` (`Func::features`), closed under
//! implication ([`Feats`]). Each CPUID feature decides on its own: `avx` the VEX encoding of
//! the function's vector code and 256-bit float vectors, `avx2` 256-bit integer vectors and the
//! AVX2-only instructions at 128 bits (broadcasts from a register, per-lane shift counts,
//! gathers, integer masked moves), `fma` the fused multiply-add (else `__stave_fma32/64`),
//! `bmi2` `pext`. The assembler records the feature of every instruction it encodes
//! ([`crate::enc::Asm::features`]); a function whose code would need a feature outside its set
//! is [`Error::Unsupported`]. The stubs, the software routines and the tables are the same for
//! every function and use baseline instructions only.
//!
//! Vector and mask values (`simd`): every `vec.*` / `mask.*` op, 128-bit and narrower vectors
//! with SSE4.2 instructions (VEX-encoded with `avx`), 256-bit ones with AVX and AVX2. In a
//! debug build a vector value lives in its frame home: an op loads its operands into scratch
//! `xmm` / `ymm` registers and stores the result, and `ymm` upper halves are clean between
//! instructions. A release build keeps vectors in registers (above); a function with 32-byte
//! values then cleans the upper halves before a call and before a return that carries no such
//! value. The scalar float instructions of a function with `avx` are VEX-encoded in a release
//! build, and in a debug build when the module baseline has no `avx`. A function with a
//! 32-byte-aligned value realigns its frame (`func`).
//!
//! K mode. A function whose set has `avx512f`, `avx512bw`, `avx512vl`, `avx512dq` and `avx512cd`
//! ([`Feats::v4`]) holds 512-bit vectors (in `zmm0`–`zmm13`) and its masks as one bit per lane
//! in mask registers: a register class of its own next to the general and vector ones (`k1`–`k7`
//! in the cache, `k4`–`k7` ownable across blocks in a release build, `k0` scratch; 8-byte
//! homes). Its vector ops are the v4 recipes of the lowering table, EVEX-encoded (`simd::k`).
//! Masks keep their format outside the function, the vector of all-ones lanes: in memory and
//! across calls (`abi`), where the function converts. Every EVEX and mask-register instruction
//! records its AVX-512 features like the others.
//!
//! Not implemented (clean [`Error::Unsupported`]): sizeless vectors, vectors with f16 / bf16
//! lanes, 256-bit vectors in a function without `avx`, 256-bit ops whose instructions need a
//! feature the function lacks, vectors wider than 256 bits outside K mode, the ops K mode does
//! not lower (`simd::k`), conversions *to* f16 / bf16,
//! alignment above the widest vector, intrinsics outside the x86-64 table, references to
//! functions without a body (C imports need an external linker). Stack overflow is detected by the
//! runtime's guard page, not by an inline check; a function with an
//! [`Options::reserve_slots`] slot reads its reserve below its frame at entry, so the guard
//! faults there. When `ObjectImage::patch_func` moves a function,
//! its PC-table row follows it (`Abs64Body`) but keeps its position, so the table is no longer
//! sorted by address until the next full link.

pub mod abi;
pub mod analysis;
mod cast;
mod func;
mod intrin;
mod ops;
mod simd;
mod softfma;
mod stubs;
#[cfg(test)]
mod tests;

use std::fmt;

use rustc_hash::FxHashMap;
use stave_ir::layout::Layouts;
use stave_ir::{Func, FuncId, Known, Module, Operand, Site, VerifyError};
use stave_obj::{
    FileId, FuncCode, ImageBuilder, Machine, ObjError, ObjectImage, SourceFile, StackUse, SymbolId,
};
use stave_target::{Feature, FeatureSet};

use crate::enc::EncError;
use abi::VCls;

/// One row of the site table (contracts/intrinsics.md): `Site(n)` of the IR is row `n`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteRow {
    /// Index into [`Options::files`].
    pub file: u32,
    /// 1-based.
    pub line: u32,
    /// 1-based, in bytes.
    pub col: u32,
    /// The enclosing function (its checked-IR path is the row's `fn`).
    pub func: FuncId,
}

/// The CPU features one function is compiled for, one predicate per CPUID feature. SSE4.2 is
/// assumed everywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Feats {
    /// The closed set the predicates come from.
    pub set: FeatureSet,
    /// VEX encoding; 256-bit vectors with float instructions.
    pub avx: bool,
    /// 256-bit integer instructions, and the AVX2-only ones at 128 bits.
    pub avx2: bool,
    pub fma: bool,
    pub bmi2: bool,
    /// K mode: the set has `avx512f`, `avx512bw`, `avx512vl`, `avx512dq` and `avx512cd`. Masks
    /// live in `k` registers, vector ops are EVEX-encoded, and 512-bit vectors exist.
    pub v4: bool,
}

/// The features of K mode.
pub(crate) const V4_SET: [Feature; 5] =
    [Feature::Avx512f, Feature::Avx512bw, Feature::Avx512vl, Feature::Avx512dq, Feature::Avx512cd];

impl Feats {
    pub(crate) fn of(set: FeatureSet) -> Feats {
        Feats {
            set,
            avx: set.contains(Feature::Avx),
            avx2: set.contains(Feature::Avx2),
            fma: set.contains(Feature::Fma),
            bmi2: set.contains(Feature::Bmi2),
            v4: V4_SET.iter().all(|f| set.contains(*f)),
        }
    }

    /// The widest vector register in bytes.
    pub(crate) fn vector_bytes(self) -> u32 {
        if self.v4 {
            64
        } else if self.avx {
            32
        } else {
            16
        }
    }
}

/// Backend options.
#[derive(Clone, Debug)]
pub struct Options {
    /// Image layout ([`stave_obj::Options::debug`] by default: padded slots + jump table).
    pub image: stave_obj::Options,
    /// Source files of the site table and the DWARF line table (with [`Options::sites`]).
    pub files: Vec<SourceFile>,
    /// The site table: `Site(n)` = row `n` (front end order: by fn instance, then offset).
    /// Empty: `Module::sites` is used.
    pub sites: Vec<SiteRow>,
    /// Per `FuncId`: the site of the fn's name token (PC table). Empty: `Func::name_site`.
    pub fn_sites: Vec<Site>,
    /// The program's `main` (for `__stave_main`); `None`: no `__stave_main` stub (`main_addr` is
    /// then rejected).
    pub main: Option<FuncId>,
    /// Run the IR verifier first (codegen assumes verified IR; unverified input still never
    /// panics but may give a less precise error).
    pub verify: bool,
    /// Worker threads for per-function codegen (0 = available parallelism).
    pub threads: usize,
    /// Release code generation: values used across blocks get registers for their whole life
    /// (`func`, stave-opt `regalloc`). Off: every such value lives in its frame home.
    pub release: bool,
    /// The module baseline: the features every function may use (a target's
    /// `Triple::baseline`). A function adds its own `Func::features`.
    pub features: FeatureSet,
    /// Per `FuncId`: the slot of `__stave_reserve_table` whose bytes the function's entry probes
    /// below its frame, page by page, or `u32::MAX` (no probe; also every function past the end).
    pub reserve_slots: Vec<u32>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            image: stave_obj::Options::debug(Machine::X86_64),
            files: Vec::new(),
            sites: Vec::new(),
            fn_sites: Vec::new(),
            main: None,
            verify: true,
            threads: 0,
            release: false,
            features: FeatureSet::of(Feature::Sse42),
            reserve_slots: Vec::new(),
        }
    }
}

/// Codegen errors. Verified IR either compiles or yields one of these; nothing panics.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    Verify(Vec<VerifyError>),
    /// Valid IR this backend does not implement yet (sizeless vectors, f16 rounding casts, ...).
    Unsupported {
        func: Option<FuncId>,
        what: String,
    },
    /// IR the backend cannot make sense of (should have been rejected by the verifier).
    Malformed {
        func: Option<FuncId>,
        what: String,
    },
    Encode {
        func: Option<FuncId>,
        err: EncError,
    },
    Obj(ObjError),
}

impl Error {
    fn in_func(self, f: FuncId) -> Error {
        match self {
            Error::Unsupported { func: None, what } => Error::Unsupported { func: Some(f), what },
            Error::Malformed { func: None, what } => Error::Malformed { func: Some(f), what },
            Error::Encode { func: None, err } => Error::Encode { func: Some(f), err },
            e => e,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let at = |func: &Option<FuncId>| func.map(|x| format!(" in fn #{}", x.0)).unwrap_or_default();
        match self {
            Error::Verify(es) => {
                write!(f, "IR verification failed")?;
                for e in es.iter().take(5) {
                    write!(f, "; {e}")?;
                }
                Ok(())
            }
            Error::Unsupported { func, what } => {
                write!(f, "x86-64 debug backend: unsupported{}: {what}", at(func))
            }
            Error::Malformed { func, what } => {
                write!(f, "x86-64 debug backend: malformed IR{}: {what}", at(func))
            }
            Error::Encode { func, err } => write!(f, "x86-64 encoder{}: {err}", at(func)),
            Error::Obj(e) => write!(f, "ELF writer: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<EncError> for Error {
    fn from(err: EncError) -> Error {
        Error::Encode { func: None, err }
    }
}

impl From<ObjError> for Error {
    fn from(e: ObjError) -> Error {
        Error::Obj(e)
    }
}

pub(crate) fn unsup(what: impl Into<String>) -> Error {
    Error::Unsupported { func: None, what: what.into() }
}

pub(crate) fn bad(what: impl Into<String>) -> Error {
    Error::Malformed { func: None, what: what.into() }
}

pub(crate) type Res<T> = Result<T, Error>;

/// Module-wide facts shared (read-only) by the per-function code generators.
pub struct Ctx<'m> {
    pub(crate) m: &'m Module,
    pub(crate) layouts: Layouts,
    /// The class of every type, for a function without (`[0]`) and with (`[1]`) 256-bit
    /// vector registers, and for one in K mode (`[2]`). They differ in which types have a
    /// class and in how a function holds a mask, not in how a value crosses a call
    /// (`VCls::boundary`), so the ABI of a signature is the same for every function that can
    /// hold its types.
    pub(crate) cls: [Vec<Option<VCls>>; 3],
    pub(crate) func_sym: Vec<Option<SymbolId>>,
    pub(crate) str_sym: Vec<SymbolId>,
    pub(crate) trap_sym: SymbolId,
    pub(crate) main_sym: Option<SymbolId>,
    pub(crate) signal_sym: Option<SymbolId>,
    pub(crate) thread_sym: Option<SymbolId>,
    pub(crate) site_table_sym: SymbolId,
    pub(crate) pc_table_sym: SymbolId,
    pub(crate) stack_table_sym: SymbolId,
    /// `__stave_reserve_table` and [`Options::reserve_slots`].
    pub(crate) reserve_sym: Option<SymbolId>,
    pub(crate) reserve_slots: Vec<u32>,
    /// Line-table position per site row.
    pub(crate) lines: Vec<Option<(FileId, u32, u32)>>,
    /// Per `FuncId`: site of the name token.
    pub(crate) fn_sites: Vec<Site>,
    /// Per `Module::statics` index.
    pub(crate) static_sym: Vec<SymbolId>,
    /// [`Options::release`].
    pub(crate) release: bool,
    /// [`Options::features`].
    pub(crate) baseline: FeatureSet,
    /// `__stave_fma32`, `__stave_fma64`: the software `fma` of one lane, where a function
    /// without the `fma` feature has a `vec.fma`.
    pub(crate) fma_sym: Option<[SymbolId; 2]>,
    /// The `compress` / `expand` tables the module's shapes use (`simd::compress::Tab`).
    pub(crate) ctab: [Option<SymbolId>; simd::compress::NTAB],
}

/// What [`emit_module`] defined: the symbols of the functions, and the PC table rows still to
/// be written by [`link`].
#[derive(Clone, Debug)]
pub struct Emitted {
    /// Per `FuncId`: the function's symbol (`None` for functions without a body).
    pub func_syms: Vec<Option<SymbolId>>,
    pub pc_table: SymbolId,
    /// PC table rows in address order: (function symbol, site of its name token).
    pub pc_rows: Vec<(SymbolId, u32)>,
    /// Per-function codegen time in nanoseconds (by `FuncId`, 0 for no body).
    pub nanos: Vec<u64>,
    /// Per `FuncId`: what the function's code takes of the stack (default for no body).
    pub stack: Vec<StackUse>,
    /// Declared, defined by the caller ([`stave_obj::stub::stack_table`]) or else by [`link`].
    pub stack_table: SymbolId,
    /// Declared, defined by the caller ([`stave_obj::stub::reserve_table`]) or else by [`link`]
    /// (every slot 0).
    pub reserve_table: SymbolId,
    /// Slots of the reserve table the code reads.
    pub reserve_count: usize,
}

impl Ctx<'_> {
    /// The features `f` is compiled for.
    pub(crate) fn feats(&self, f: &Func) -> Feats {
        Feats::of(self.baseline.union(f.features))
    }

    /// The type classes of a function with features `fx`.
    pub(crate) fn cls_of(&self, fx: Feats) -> &[Option<VCls>] {
        &self.cls[if fx.v4 { 2 } else { usize::from(fx.avx) }]
    }
}

fn classify(m: &Module, layouts: &Layouts) -> [Vec<Option<VCls>>; 3] {
    [
        abi::classify_all(&m.types, layouts, 16, false),
        abi::classify_all(&m.types, layouts, 32, false),
        abi::classify_all(&m.types, layouts, 64, true),
    ]
}

fn find_main(m: &Module, o: &Options) -> Option<FuncId> {
    o.main.filter(|&f| m.func(f).is_some_and(|x| x.body.is_some()))
}

/// Declares and defines everything of `m` in `b`: functions, strings, stubs, and the site table.
/// The PC table is declared but defined by [`link`] (it needs final function addresses).
pub fn emit_module(m: &Module, o: &Options, b: &mut ImageBuilder) -> Result<Emitted, Error> {
    if b.machine() != Machine::X86_64 {
        return Err(unsup("image machine is not x86-64"));
    }
    if o.verify {
        stave_ir::verify_module(m).map_err(Error::Verify)?;
    }
    // Functions with bodies, in canonical instance order (stave_ir::link).
    let order = stave_ir::link::text_order(m);
    let names = stave_ir::link::func_symbols(m, &order);
    let mut func_sym = vec![None; m.funcs.len()];
    for f in &order {
        let Some(name) = &names[f.index()] else { continue };
        if b.lookup(name).is_some() {
            return Err(Error::Obj(ObjError::Redefined(name.clone())));
        }
        func_sym[f.index()] = Some(b.declare(name));
    }
    // Calls / fn values naming a function without a body cannot be linked statically.
    for f in &order {
        if let Some(body) = &m.funcs[f.index()].body {
            for o in &body.operands {
                if let Operand::Func(g) = o
                    && func_sym.get(g.index()).copied().flatten().is_none()
                {
                    let path = m.func(*g).map_or("?", |x| x.path.as_str());
                    return Err(Error::Unsupported {
                        func: Some(*f),
                        what: format!(
                            "reference to fn `{path}` without a body (extern imports need a C linker)"
                        ),
                    });
                }
            }
        }
    }
    let mut str_sym = Vec::with_capacity(m.strings.len());
    for (i, s) in m.strings.iter().enumerate() {
        str_sym.push(b.rodata(&format!("__stave_str.{i}"), stave_obj::DataObj::new(s.clone(), 1))?);
    }
    let sym_of = |k: Known| m.known_body(k).and_then(|f| func_sym[f.index()]);
    let trap_sym = match sym_of(Known::Trap) {
        Some(s) => s,
        None => stubs::panic_fallback(b)?,
    };
    let main = find_main(m, o);
    let main_sym = match main {
        Some(f) => Some(stubs::stave_main(m, b, f, func_sym[f.index()])?),
        None => None,
    };
    let signal_sym = match sym_of(Known::OnSignal) {
        Some(s) => Some(stubs::tail_stub(b, stubs::SIGNAL, s, true)?),
        None => None,
    };
    let thread_sym = match sym_of(Known::ThreadEntry) {
        Some(s) => Some(stubs::tail_stub(b, stubs::THREAD, s, false)?),
        None => None,
    };
    // The site table: from the options, else from the module (`Module::sites`).
    let (files, rows) = site_rows(m, o);
    let mut file_ids = Vec::with_capacity(files.len());
    for f in &files {
        file_ids.push(b.add_file(&f.dir, &f.name));
    }
    let site_table_sym = stubs::site_table(&files, &rows, b)?;
    let pc_table_sym = b.declare(stubs::PC_TABLE);
    let stack_table_sym = b.declare(stave_obj::stub::STACK_TABLE);
    let reserve_sym = b.declare(stave_obj::stub::RESERVE_TABLE);
    let reserve_count =
        o.reserve_slots.iter().filter(|&&s| s != u32::MAX).map(|&s| s as usize + 1).max().unwrap_or(0);
    let lines = rows.iter().map(|r| file_ids.get(r.file as usize).map(|&fi| (fi, r.line, r.col))).collect();
    let fn_sites: Vec<Site> = if o.fn_sites.is_empty() {
        m.funcs.iter().map(|f| f.name_site).collect()
    } else {
        o.fn_sites.clone()
    };
    let layouts = Layouts::compute(&m.types);
    let static_sym = stubs::statics(m, &layouts, &str_sym, b)?;
    let cls = classify(m, &layouts);
    let soft_fma = order.iter().map(|f| &m.funcs[f.index()]).any(|f| {
        !Feats::of(o.features.union(f.features)).fma
            && f.body.as_ref().is_some_and(|b| {
                b.insts.op.iter().any(|op| matches!(op, stave_ir::Op::Vec(stave_ir::simd::VecOp::Fma)))
            })
    });
    let fma_sym = if soft_fma { Some(softfma::define(b)?) } else { None };
    let ctab = simd::compress::define(m, o.features, b)?;
    let cx = Ctx {
        m,
        layouts,
        cls,
        func_sym: func_sym.clone(),
        str_sym,
        trap_sym,
        main_sym,
        signal_sym,
        thread_sym,
        site_table_sym,
        pc_table_sym,
        stack_table_sym,
        reserve_sym: Some(reserve_sym),
        reserve_slots: o.reserve_slots.clone(),
        lines,
        fn_sites: fn_sites.clone(),
        static_sym,
        release: o.release,
        baseline: o.features,
        fma_sym,
        ctab,
    };
    let (codes, nanos) = compile_all(&cx, &order, o.threads);
    let mut stack = vec![StackUse::default(); m.funcs.len()];
    for (f, code) in order.iter().zip(codes) {
        let (code, used) = code.map_err(|e| e.in_func(*f))?;
        stack[f.index()] = used;
        if let Some(s) = func_sym[f.index()] {
            b.define_func(s, code)?;
        }
    }
    if let Some(entry) = sym_of(Known::StartEntry) {
        b.add_start_stub(entry)?;
    }
    let pc_rows = order
        .iter()
        .filter_map(|f| {
            let site = fn_sites.get(f.index()).copied().unwrap_or(Site::NONE).0;
            func_sym[f.index()].map(|s| (s, site))
        })
        .collect();
    let mut all_nanos = vec![0u64; m.funcs.len()];
    for (f, n) in order.iter().zip(nanos) {
        all_nanos[f.index()] = n;
    }
    Ok(Emitted {
        func_syms: func_sym,
        pc_table: pc_table_sym,
        pc_rows,
        nanos: all_nanos,
        stack,
        stack_table: stack_table_sym,
        reserve_table: reserve_sym,
        reserve_count,
    })
}

/// Compiles one function (exposed for incremental patching and benchmarks).
pub fn compile_func(cx: &Ctx<'_>, f: FuncId) -> Result<FuncCode, Error> {
    compile_func_stack(cx, f).map(|(c, _)| c)
}

/// [`compile_func`] with what the function's code takes of the stack.
pub fn compile_func_stack(cx: &Ctx<'_>, f: FuncId) -> Result<(FuncCode, StackUse), Error> {
    func::compile(cx, f).map_err(|e| e.in_func(f))
}

type Compiled = Res<(FuncCode, StackUse)>;

fn compile_all(cx: &Ctx<'_>, order: &[FuncId], threads: usize) -> (Vec<Compiled>, Vec<u64>) {
    let n = order.len();
    let threads =
        if threads == 0 { std::thread::available_parallelism().map_or(1, |x| x.get()) } else { threads };
    let one = |f: FuncId| {
        let t = std::time::Instant::now();
        let r = compile_func_stack(cx, f);
        (r, t.elapsed().as_nanos() as u64)
    };
    if threads <= 1 || n < 64 {
        let (codes, nanos) = order.iter().map(|&f| one(f)).unzip();
        return (codes, nanos);
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    type Done = (Compiled, u64);
    let mut slots: Vec<Option<Done>> = (0..n).map(|_| None).collect();
    let results: Vec<Vec<(usize, Done)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.min(n))
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        out.push((i, one(order[i])));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    for v in results {
        for (i, r) in v {
            slots[i] = Some(r);
        }
    }
    slots.into_iter().map(|s| s.unwrap_or_else(|| (Err(bad("codegen worker failed")), 0))).unzip()
}

/// Compiles and links `m` into a static executable (entry `_start` → `runtime.start.entry`).
pub fn compile_module(m: &Module, o: &Options) -> Result<ObjectImage, Error> {
    let mut b = ImageBuilder::new(Machine::X86_64, o.image.clone());
    let e = emit_module(m, o, &mut b)?;
    if m.known_body(Known::StartEntry).is_none() {
        return Err(unsup(format!("no `{}` in the module: nothing to start", Known::StartEntry)));
    }
    link(b, &e)
}

/// Defines the PC table and links. The table's `pc_lo` words are [`stave_obj::RelocKind::Abs64Body`]
/// relocations: they resolve to each function's first instruction even when the jump table is
/// enabled (and follow a function that `ObjectImage::patch_func` moves). A stack table the caller
/// did not define is the empty one: every stack gets the runtime's default; a reserve table the
/// caller did not define has every slot 0.
pub fn link(mut b: ImageBuilder, e: &Emitted) -> Result<ObjectImage, Error> {
    b.define_rodata(e.pc_table, stubs::pc_table(&e.pc_rows))?;
    if !b.is_defined(e.stack_table) {
        b.define_rodata(e.stack_table, stave_obj::stub::stack_table(0, 0, &[]))?;
    }
    if !b.is_defined(e.reserve_table) {
        b.define_rodata(e.reserve_table, stave_obj::stub::reserve_table(&vec![0; e.reserve_count]))?;
    }
    Ok(b.link()?)
}

/// Builds the per-function context without an image (benchmarks, tests): every function gets a
/// placeholder symbol id.
pub fn bench_ctx(m: &Module) -> Ctx<'_> {
    let layouts = Layouts::compute(&m.types);
    let cls = classify(m, &layouts);
    let n = m.funcs.len() as u32;
    Ctx {
        m,
        layouts,
        cls,
        func_sym: (0..n).map(|i| Some(SymbolId(i))).collect(),
        str_sym: (0..m.strings.len() as u32).map(|i| SymbolId(n + i)).collect(),
        trap_sym: SymbolId(u32::MAX - 1),
        main_sym: Some(SymbolId(u32::MAX - 2)),
        signal_sym: Some(SymbolId(u32::MAX - 3)),
        thread_sym: Some(SymbolId(u32::MAX - 4)),
        site_table_sym: SymbolId(u32::MAX - 5),
        pc_table_sym: SymbolId(u32::MAX - 6),
        stack_table_sym: SymbolId(u32::MAX - 9),
        reserve_sym: None,
        reserve_slots: Vec::new(),
        lines: Vec::new(),
        fn_sites: Vec::new(),
        static_sym: (0..m.statics.len() as u32).map(|i| SymbolId(u32::MAX - 10 - i)).collect(),
        release: false,
        baseline: FeatureSet::of(Feature::Sse42),
        fma_sym: Some([SymbolId(u32::MAX - 7), SymbolId(u32::MAX - 8)]),
        ctab: simd::compress::placeholders(n + m.strings.len() as u32),
    }
}

/// Site table source: [`Options::sites`] when given, else `Module::sites` (files in first-use
/// order, split by `stave_ir::link::split_file`).
fn site_rows(m: &Module, o: &Options) -> (Vec<SourceFile>, Vec<stubs::Row>) {
    if !o.sites.is_empty() || m.sites.is_empty() {
        let rows = o
            .sites
            .iter()
            .map(|r| stubs::Row {
                file: r.file,
                line: r.line,
                col: r.col,
                func: m.func(r.func).map(|f| f.path.clone()).unwrap_or_default(),
            })
            .collect();
        return (o.files.clone(), rows);
    }
    let mut files: Vec<SourceFile> = Vec::new();
    let mut index: FxHashMap<&str, u32> = FxHashMap::default();
    let mut rows = Vec::with_capacity(m.sites.len());
    for s in &m.sites {
        let fi = match index.get(s.file.as_str()) {
            Some(&i) => i,
            None => {
                let (dir, name) = stave_ir::link::split_file(&s.file);
                files.push(SourceFile { dir: dir.to_owned(), name: name.to_owned() });
                let i = files.len() as u32 - 1;
                index.insert(s.file.as_str(), i);
                i
            }
        };
        rows.push(stubs::Row { file: fi, line: s.line, col: s.col, func: s.func.clone() });
    }
    (files, rows)
}
