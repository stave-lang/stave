//! AArch64 debug backend.
//!
//! Structure (TPDE style, the same as the x86-64 debug backend):
//! 1. `analysis`: one pass over the function computing the reverse postorder, natural loops,
//!    liveness intervals (in RPO block indices, extended over loops), last uses, and the frame
//!    layout (every value's home slot, assigned by a linear scan over the intervals).
//! 2. `emit` + `ops`: one pass in RPO that does instruction selection, greedy block-local
//!    register allocation (values are cached in caller-saved registers, written back to their
//!    home slots when evicted, before calls, and at block ends when live out), and encoding.
//!
//! Register classes: GPR and FP scalar. Aggregates (tuples, structs, arrays, enums), fat
//! pointers, vectors and masks live in memory (their home slot); scalars live in registers.
//! Vector ops load their operands into temporary V registers and store their result (`simd`); a
//! 128-bit vector's home is 16-byte aligned. Release code generation keeps vectors and masks in
//! registers instead (below).
//!
//! Release code generation ([`Codegen::release`]) keeps this structure. A value used in another
//! block, and every param of a later block, owns a callee-saved register (`x19`–`x28`, `v8`–`v15`)
//! from its definition to its last use when stave-opt's `regalloc` finds one; those registers are
//! saved below the frame pointer. Vectors and masks live in V registers too: in `v8`–`v15` across
//! blocks where no call intervenes (a callee keeps only their low halves), else in the block
//! cache, with the home slot as the spill slot. Branch arguments are one parallel move. Blocks
//! are emitted loop by loop, a loop's test after its body when every value the test defines has a
//! register, and the blocks that cannot return last (stave-opt `layout`). A compare read only by
//! its block's `cbr` is flags and a `b.cond`; a loop variable's next value (a counter's
//! increment, a vector accumulator's sum) is defined in the param's register.
//!
//! # Calling convention (Stave → Stave calls, AAPCS64-based)
//!
//! - Integer-class scalars (ints, `bool`, `char`, thin pointers, fn values) go in the next free
//!   `x0..x7`, `f32`/`f64` in the next free `v0..v7` (`s`/`d`), counted independently (AAPCS64
//!   NGRN/NSRN). Ints narrower than 64 bits are passed and returned sign- or zero-extended to 64
//!   bits by their signedness (`bool` as 0/1, `char` zero-extended); this is stronger than
//!   AAPCS64, which leaves the upper bits unspecified.
//! - A fat pointer (`&[T]`, `&str`: address, length) takes two consecutive GPRs when at least
//!   two are free; otherwise it goes to the stack (16 bytes) and no further GPR is used (AAPCS64
//!   C.11/C.13 for a 16-byte composite).
//! - Every aggregate (tuple, struct, array, enum; any size, also zero-sized) is passed by
//!   reference: the caller passes the address of a caller-owned copy (its home slot) in the next
//!   GPR (or a stack word); the callee copies it into its own frame and never writes through the
//!   pointer (AAPCS64 B.4 for composites > 16 bytes, applied to all sizes).
//! - A vector or mask (at most 128 bits) goes in the next free `v0..v7` as a whole register, like
//!   an AAPCS64 short vector: lanes in order from bit 0, bits past its size unspecified, a mask as
//!   full-width lanes of all ones or zero. Without a free register it goes to the stack by value
//!   (16 bytes, 16-byte aligned, for a 128-bit vector; one word for a narrower one).
//! - Arguments that find no register go to the stack in order, each 8-byte aligned and rounded up
//!   to 8 bytes (16 for a fat pointer), from `[sp]` at the call; the callee finds them at
//!   `[x29 + 16]`.
//! - Results: integer-class scalars in `x0`, floats in `v0`, a fat pointer in `x0, x1`, a vector
//!   or mask in `v0`; an aggregate result is written to the buffer whose address the caller passes
//!   in `x8` (AAPCS64 indirect result location); unit returns nothing.
//! - Caller-saved: `x0–x18`, `v0–v7`, `v16–v31`, flags. Callee-saved: `x19–x28`, `x29` (frame
//!   pointer), `v8–v15` (low 64 bits: no register keeps a vector across a call). A debug build
//!   never allocates callee-saved registers;
//!   every frame saves `x29, x30` with `stp x29, x30, [sp, #-16]!; mov x29, sp`.
//! - `sp` is 16-byte aligned at every call. Frames larger than 4 KiB are probed page by page in
//!   the prologue (runtime/README.md, guard page).
//! - Reserved scratch registers: `x15` (data), `x16` (IP0: indirect call target, copy source),
//!   `x17` (IP1: address computation, copy destination), `v30`, `v31`. Allocatable: `x0–x7`,
//!   `x9–x14`, `v0–v7`, `v16–v29`.
//!
//! # Traps
//!
//! Every failing check branches to an out-of-line stub per (kind, site) at the end of the
//! function: `mov w0, #kind; mov w1, #site; bl runtime.panic.trap; brk #0` (kind numbering of
//! runtime/README.md, `site` = the [`stave_ir::Site`] row index, `0xFFFF_FFFF` unknown). Without a
//! `runtime.panic.trap` function in the module the stub is `brk #(0x100 + kind)`.
//!
//! - `add`/`sub` (32/64-bit): `adds`/`subs` + `b.vs` (signed) or `b.hs`/`b.lo` (unsigned);
//!   8/16-bit: exact 64-bit result + range check.
//! - `mul` 64-bit: `smulh`/`umulh` against the low half; 32-bit: `smull`/`umull` + range check.
//! - `div`/`rem`: `cbz` divisor → `DivByZero`; signed `MIN / -1`, `MIN % -1` → `Overflow`
//!   (`cmn b, #1; ccmp a, MIN, #0, eq; b.eq`).
//! - `shl`/`shr`: amount (unsigned compare) ≥ width → `Overflow`.
//! - `check.bounds`, dynamic `get`/`set`: `cmp i, n; b.hs` → `Bounds`.
//! - `panic.<kind>`: a branch to the stub.
//!
//! # Scalars
//!
//! Integers are kept normalized in 64-bit registers (sign- or zero-extended from their width, as
//! the interpreter carries them), so compares and divisions use 64-bit forms and narrow results
//! are re-normalized (`sxt*`/`uxt*`). `mul.wide`: `umulh`/`smulh` + `mul` (64-bit), `umull`/`smull` (32-bit).
//! `popcount`: `cnt`/`addv` on `v31`; `ctz`: `rbit` + `clz`. Floats are IEEE (FPCR default: round
//! to nearest even, no flush to zero); `rem` calls `__stave_fmod` (exact, integer algorithm; f32
//! via exact f64 widening). `as|` float → int is `fcvtzs`/`fcvtzu` (saturating, NaN → 0) plus a
//! clamp for 8/16-bit targets; `as~` is `scvtf`/`ucvtf`/`fcvt` under round-to-nearest-even;
//! `as?` checks exactness (`frintz`, range compares against powers of two, convert-back) and
//! writes the `Option` with zeroed padding. Constant operands of casts fold through
//! `stave_ir::arith::convert` (the interpreter's own code).
//!
//! # Intrinsics (contracts/intrinsics.md)
//!
//! `syscallN`: `x8 = nr`, `x0..x5`, `svc #0`, result `x0`. `ccallN`: `x0..x7`, `a9..a12` at
//! `[sp]`, `blr x16`. `call_on_stack`: `stp x19, x30, [sp, #-16]!; mov x19, sp; mov sp, top;
//! blr x16; mov sp, x19; ldp x19, x30, [sp], #16`. `main_addr`, `signal_handler_addr`,
//! `thread_start_addr`, `site_table`, `pc_table`, `static_addr k`: `adrp` + `add`. `cpu_init`:
//! `bl runtime.cpu.init` when the module defines it, else nothing. `copy_bytes`: `bl
//! std.slice.copy_raw` (the Stave convention, the three operands as its arguments). `cpuid`,
//! `xgetbv` (x86-64 only) and unknown intrinsics are [`ErrorKind::Unsupported`].
//!
//! # Emitted symbols ([`prepare`])
//!
//! One symbol per function, named and ordered as on x86-64 (`stave_ir::link`: its `path`;
//! `path@instance` when paths repeat or the path is a compiler symbol name; instance order);
//! `_start` (stave-obj `stub::start` → `runtime.start.entry`), `__stave_main` (calls `main`,
//! returns its status), `__stave_signal` (`mov w0, w0; b runtime.panic.on_signal`),
//! `__stave_thread_start` (`b runtime.thread.entry`), `__stave_site_table` and `__stave_pc_table`
//! (contracts/intrinsics.md row layouts, from `Module::sites` and `Func::name_site`),
//! `__stave_stack_table` (declared; the caller defines it from [`Codegen::compile_stack`]'s
//! frame sizes),
//! `__stave_strings` (all string literals), `__stave_fmod`, and one `.data`/`.bss` object per
//! `static mut` (`stave_ir::link::static_symbol`).
//!
//! # Limits
//!
//! - Branches to trap stubs and between blocks are `b.cond`/`cbz` (±1 MiB); a function where
//!   one does not reach is emitted again with each as an inverted branch over a `b` (±128 MiB).
//! - Frames are limited to 16 MiB; aggregates must be aligned to at most 16 bytes.
//! - With stave-obj's jump table (debug [`stave_obj::Options`]), every function reference resolves
//!   to the function's jump-table stub, except the `pc_lo` words of the PC table
//!   ([`stave_obj::RelocKind::Abs64Body`]: the body's first instruction).
//! - Vectors and masks wider than 128 bits, sizeless vectors, f16/bf16 values and lanes:
//!   [`ErrorKind::Unsupported`].

mod abi;
mod analysis;
mod emit;
mod ops;
mod simd;
mod support;
mod ty;

use std::fmt;

use stave_ir::FuncId;
use stave_obj::ObjError;

pub use support::{BuildConfig, Codegen, FuncScratch, Symbols, build_image, prepare};

/// A codegen failure: the function (if any) and what went wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub func: Option<FuncId>,
    pub kind: ErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// Valid IR this backend does not support (wide vectors, f16 arithmetic, x86-only intrinsics,
    /// ...).
    Unsupported(String),
    /// IR the verifier should have rejected.
    BadIr(String),
    /// An encoding error (immediate or branch out of range).
    Enc(crate::enc::EncError),
    /// An object-writer error.
    Obj(ObjError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(func) = self.func {
            write!(f, "fn #{}: ", func.0)?;
        }
        match &self.kind {
            ErrorKind::Unsupported(s) => write!(f, "unsupported: {s}"),
            ErrorKind::BadIr(s) => write!(f, "malformed IR: {s}"),
            ErrorKind::Enc(e) => write!(f, "encoding: {e}"),
            ErrorKind::Obj(e) => write!(f, "object: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<crate::enc::EncError> for Error {
    fn from(e: crate::enc::EncError) -> Error {
        Error { func: None, kind: ErrorKind::Enc(e) }
    }
}

impl From<ObjError> for Error {
    fn from(e: ObjError) -> Error {
        Error { func: None, kind: ErrorKind::Obj(e) }
    }
}

pub(crate) fn unsupported(s: impl Into<String>) -> Error {
    Error { func: None, kind: ErrorKind::Unsupported(s.into()) }
}

pub(crate) fn bad(s: impl Into<String>) -> Error {
    Error { func: None, kind: ErrorKind::BadIr(s.into()) }
}

pub(crate) type R<T> = Result<T, Error>;
