//! SSA IR, reference interpreter, checked-IR printer.
//!
//! - [`types`]: the type key ([`TypeKey`]), interned in a [`TypeTable`]; TypeKey
//!   text; canonical renumbering.
//! - [`ir`]: SSA with block params, struct-of-arrays bodies, `u32` ids; the op set ([`Op`]).
//! - [`build`]: [`FuncBuilder`] for the front end's lowering.
//! - [`fence`]: the placement of `mem.fence_stream` after streaming stores.
//! - [`verify`]: structure, dominance, block params, terminators, op typing.
//! - [`interp`]: the scalar reference interpreter (oracle) over [`mem`] with a pluggable [`host`];
//!   [`memcheck`]: its checking mode (memory errors as an outcome).
//! - [`borrowtrack`]: the dynamic borrow model of contracts/borrow-model.md as a state machine
//!   over allocation ids and byte offsets.
//! - [`print_checked`]: the canonical checked-IR text of contracts/checked-ir.md.
//! - [`simd`]: vector and mask ops ([`VecOp`]) and their lane semantics.
//! - [`arith`]: scalar op and cast semantics; [`layout`]: memory layout of IR types.
//! - [`bounds`]: the audit of element accesses and the pass that proves bounds checks removable.
//! - [`witness`]: the proofs of that pass as data, and their independent checker, which alone
//!   deletes a check.
//! - [`traps`]: the instructions that give a function the `trap` effect.
//! - [`link`]: text order and symbol names of a module's functions and statics (both backends).
//! - [`known`]: the compiler-known items ([`Known`]) functions are stamped with.
//! - [`intrinsics`]: the intrinsic names, arches, and signatures (front end and verifier).

pub mod arith;
pub mod borrowtrack;
pub mod bounds;
pub mod build;
pub mod divmagic;
pub mod fence;
pub mod host;
pub mod interp;
pub mod intrinsics;
pub mod ir;
pub mod known;
pub mod layout;
pub mod link;
pub mod mem;
pub mod memcheck;
pub mod print_checked;
pub mod simd;
pub mod traps;
pub mod types;
pub mod verify;
pub mod witness;

pub use build::{BuildError, FuncBuilder};
pub use host::{BufferHost, Host, SysResult, SyscallAbi};
pub use interp::{Config, Interp, InterpError, Outcome, Trap, Val};
pub use ir::{
    Block, Body, BorrowMark, CastKind, ClauseKind, Clauses, Const, DataRef, DataTarget, EventKind, EventMark,
    Func, FuncId, InstanceId, IntrinsicId, Module, Op, Operand, PanicKind, Prefetch, PrefetchLoc, Site,
    SiteInfo, Static, StrId, Term, Value,
};
pub use known::Known;
pub use print_checked::print_checked;
pub use simd::VecOp;
pub use types::{AdtShape, DeclId, FloatBits, GenericArg, IntBits, Lanes, TypeId, TypeKey, TypeTable};
pub use verify::{VerifyError, verify_func, verify_module};
