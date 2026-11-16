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
//! - Branch arguments and the register arguments of a call are parallel moves from where the
//!   values are.
//! - Blocks are emitted loop by loop with the blocks that cannot return last (stave-opt
//!   `layout`); callee-saved registers have fixed save slots and the first `ret` carries the
//!   epilogue.
//! - A compare read only by its block's `cbr` is flags and a jump; an address read by one load or
//!   store is its addressing mode; a loop counter's increment computes into the param's register.
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
//! `__stave_site_table` and `__stave_pc_table` (layouts in contracts/intrinsics.md).
//!
//! Vector and mask values (`simd`): every `vec.*` / `mask.*` op at the target's [`Level`], 128-bit
//! and narrower vectors with SSE4.2 instructions, 256-bit ones with AVX2. A vector value lives in
//! its frame home; an op loads its operands into scratch `xmm` / `ymm` registers and stores the
//! result. A function with a 32-byte-aligned value realigns its frame (`func`), and `ymm` upper
//! halves are clean between instructions.
//!
//! Not implemented (clean [`Error::Unsupported`]): sizeless vectors, vectors with f16 / bf16
//! lanes, vectors wider than the level's registers, conversions *to* f16 / bf16, alignment above
//! the widest vector, intrinsics outside the x86-64 table, references to functions without a
//! body (C imports need an external linker). Stack overflow is detected by the
//! runtime's guard page, not by an inline check. When `ObjectImage::patch_func` moves a function,
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
use stave_ir::{FuncId, Module, Operand, Site, VerifyError};
use stave_obj::{FileId, FuncCode, ImageBuilder, Machine, ObjError, ObjectImage, SourceFile, SymbolId};

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

/// The x86-64 feature level vector code is generated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Level {
    /// SSE4.2: 128-bit vectors.
    #[default]
    V2,
    /// AVX2, FMA, BMI2: 128- and 256-bit vectors.
    V3,
}

impl Level {
    /// The widest vector register in bytes.
    pub(crate) fn vector_bytes(self) -> u32 {
        match self {
            Level::V2 => 16,
            Level::V3 => 32,
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
    /// The instruction set of vector code.
    pub level: Level,
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
            level: Level::V2,
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
    pub(crate) cls: Vec<Option<VCls>>,
    pub(crate) func_sym: Vec<Option<SymbolId>>,
    pub(crate) str_sym: Vec<SymbolId>,
    pub(crate) trap_sym: SymbolId,
    pub(crate) main_sym: Option<SymbolId>,
    pub(crate) signal_sym: Option<SymbolId>,
    pub(crate) thread_sym: Option<SymbolId>,
    pub(crate) site_table_sym: SymbolId,
    pub(crate) pc_table_sym: SymbolId,
    /// Line-table position per site row.
    pub(crate) lines: Vec<Option<(FileId, u32, u32)>>,
    /// Per `FuncId`: site of the name token.
    pub(crate) fn_sites: Vec<Site>,
    /// Per `Module::statics` index.
    pub(crate) static_sym: Vec<SymbolId>,
    /// [`Options::release`].
    pub(crate) release: bool,
    /// [`Options::level`].
    pub(crate) level: Level,
    /// `__stave_fma32`, `__stave_fma64`: the software `fma` of one lane, where the level has no
    /// FMA instruction and the module has a `vec.fma`.
    pub(crate) fma_sym: Option<[SymbolId; 2]>,
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
}

const RUNTIME_TRAP: &str = "runtime.panic.trap";
const RUNTIME_SIGNAL: &str = "runtime.panic.on_signal";
const RUNTIME_THREAD: &str = "runtime.thread.entry";

fn find_body(m: &Module, path: &str) -> Option<FuncId> {
    m.funcs.iter().position(|f| f.path == path && f.body.is_some()).map(|i| FuncId(i as u32))
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
    let sym_of = |p: &str| find_body(m, p).and_then(|f| func_sym[f.index()]);
    let trap_sym = match sym_of(RUNTIME_TRAP) {
        Some(s) => s,
        None => stubs::panic_fallback(b)?,
    };
    let main = find_main(m, o);
    let main_sym = match main {
        Some(f) => Some(stubs::stave_main(m, b, f, func_sym[f.index()])?),
        None => None,
    };
    let signal_sym = match sym_of(RUNTIME_SIGNAL) {
        Some(s) => Some(stubs::tail_stub(b, stubs::SIGNAL, s, true)?),
        None => None,
    };
    let thread_sym = match sym_of(RUNTIME_THREAD) {
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
    let lines = rows.iter().map(|r| file_ids.get(r.file as usize).map(|&fi| (fi, r.line, r.col))).collect();
    let fn_sites: Vec<Site> = if o.fn_sites.is_empty() {
        m.funcs.iter().map(|f| f.name_site).collect()
    } else {
        o.fn_sites.clone()
    };
    let layouts = Layouts::compute(&m.types);
    let static_sym = stubs::statics(m, &layouts, b)?;
    let cls = abi::classify_all(&m.types, &layouts, o.level.vector_bytes());
    let soft_fma = o.level == Level::V2
        && order
            .iter()
            .filter_map(|f| m.funcs[f.index()].body.as_ref())
            .any(|b| b.insts.op.iter().any(|op| matches!(op, stave_ir::Op::Vec(stave_ir::simd::VecOp::Fma))));
    let fma_sym = if soft_fma { Some(softfma::define(b)?) } else { None };
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
        lines,
        fn_sites: fn_sites.clone(),
        static_sym,
        release: o.release,
        level: o.level,
        fma_sym,
    };
    let (codes, nanos) = compile_all(&cx, &order, o.threads);
    for (f, code) in order.iter().zip(codes) {
        let code = code.map_err(|e| e.in_func(*f))?;
        if let Some(s) = func_sym[f.index()] {
            b.define_func(s, code)?;
        }
    }
    if let Some(entry) = sym_of(stave_obj::stub::RUNTIME_ENTRY) {
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
    Ok(Emitted { func_syms: func_sym, pc_table: pc_table_sym, pc_rows, nanos: all_nanos })
}

/// Compiles one function (exposed for incremental patching and benchmarks).
pub fn compile_func(cx: &Ctx<'_>, f: FuncId) -> Result<FuncCode, Error> {
    func::compile(cx, f).map_err(|e| e.in_func(f))
}

fn compile_all(cx: &Ctx<'_>, order: &[FuncId], threads: usize) -> (Vec<Res<FuncCode>>, Vec<u64>) {
    let n = order.len();
    let threads =
        if threads == 0 { std::thread::available_parallelism().map_or(1, |x| x.get()) } else { threads };
    let one = |f: FuncId| {
        let t = std::time::Instant::now();
        let r = compile_func(cx, f);
        (r, t.elapsed().as_nanos() as u64)
    };
    if threads <= 1 || n < 64 {
        let (codes, nanos) = order.iter().map(|&f| one(f)).unzip();
        return (codes, nanos);
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    type Done = (Res<FuncCode>, u64);
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
    if find_body(m, stave_obj::stub::RUNTIME_ENTRY).is_none() {
        return Err(unsup(format!(
            "no `{}` in the module: nothing to start",
            stave_obj::stub::RUNTIME_ENTRY
        )));
    }
    link(b, &e)
}

/// Defines the PC table and links. The table's `pc_lo` words are [`stave_obj::RelocKind::Abs64Body`]
/// relocations: they resolve to each function's first instruction even when the jump table is
/// enabled (and follow a function that `ObjectImage::patch_func` moves).
pub fn link(mut b: ImageBuilder, e: &Emitted) -> Result<ObjectImage, Error> {
    b.define_rodata(e.pc_table, stubs::pc_table(&e.pc_rows))?;
    Ok(b.link()?)
}

/// Builds the per-function context without an image (benchmarks, tests): every function gets a
/// placeholder symbol id.
pub fn bench_ctx(m: &Module) -> Ctx<'_> {
    let layouts = Layouts::compute(&m.types);
    let cls = abi::classify_all(&m.types, &layouts, Level::V2.vector_bytes());
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
        lines: Vec::new(),
        fn_sites: Vec::new(),
        static_sym: (0..m.statics.len() as u32).map(|i| SymbolId(u32::MAX - 9 - i)).collect(),
        release: false,
        level: Level::V2,
        fma_sym: Some([SymbolId(u32::MAX - 7), SymbolId(u32::MAX - 8)]),
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
