//! Pipeline, dumps, conformance runner.
//!
//! - [`package`]: checks the build root argument (a package directory or a `.stv` file);
//!   packages are loaded by stave-sema ([`stave_sema::Program::load`]).
//! - [`shipped`]: locates the shipped `std` and `runtime` (`STAVE_STD_DIR`, the install layout,
//!   the repository root; contracts/cli.md).
//! - [`pipeline`]: the stage0 pipeline: stave-sema's loader (read, lex, items,
//!   parse, intern, resolve), type checker, lowering, and [`build`] (code generation and the
//!   static ELF link); `--timings` stage names.
//! - [`build`]: the back end (lower + verify, x86-64 / AArch64 debug codegen, link).
//! - [`effects`]: the effect pass and its deny-clause checks (E1401, E1403), between borrowck and
//!   lowering.
//! - [`exec`]: running built executables (natively or through a `--runner`); `test fn` discovery.
//! - [`harness`]: the `test fn` harness of `stave test PKG_DIR`.
//! - [`interp`]: runs a lowered program in the stave-ir reference interpreter (the oracle).
//! - [`dump`]: `--dump=STAGE --dump-dir DIR` printers.
//! - [`conformance`]: the `stave test` runner and its stage0 compiler.
//! - [`select`]: `--compiler auto|cpu|gpu` and the stage1 probe (contracts/cli.md).
//! - [`prof`]: the `STAVE_PROFILE` step profile (wall time and allocations per pipeline step).

pub mod build;
pub mod conformance;
pub mod dump;
pub mod effects;
pub mod exec;
pub mod harness;
pub mod interp;
pub mod package;
pub mod pipeline;
pub mod prof;
pub mod select;
pub mod shipped;
