//! Release optimization passes and analyses on the checked IR.
//!
//! - [`slices`]: slice addresses, lengths, and vector-access bounds become values defined once,
//!   at the slice.
//! - [`cse`]: an integer op that repeats a dominating one reads its result.
//! - [`dce`]: dead pure instructions are removed.
//! - [`regalloc`]: which values live in a register for their whole life (both backends).
//! - [`layout`]: the block order of release code (cold blocks last).
//! - [`inplace`]: loads of a whole slot that a backend reads in place (both backends, every
//!   build).
//!
//! [`optimize`] runs the IR passes of a release build on one body. It runs after the removal of
//! bounds checks, which every build shares (stave-ir `bounds`), and removes no check itself.

mod cfg;
pub mod cse;
pub mod dce;
mod edit;
pub mod inplace;
pub mod layout;
pub mod regalloc;
pub mod slices;

use stave_ir::{Body, Op, TypeTable};

/// The IR passes of a release build, in order. `types` must have been through
/// [`slices::declare_elem_pointers`].
pub fn optimize(types: &TypeTable, b: &mut Body) {
    slices::unify_parts(b, Op::SliceLen);
    if slices::split_elem_addrs(types, b) {
        slices::unify_parts(b, Op::SlicePtr);
    }
    slices::unify_chunk_bounds(b);
    cse::remove_redundant(types, b);
    dce::remove_dead(types, b);
}
