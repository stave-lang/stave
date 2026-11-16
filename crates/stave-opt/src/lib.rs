//! Release optimization passes and analyses on the checked IR.
//!
//! - [`slices`]: slice addresses, lengths, and vector-access bounds become values defined once,
//!   at the slice.
//! - [`cse`]: an integer op that repeats a dominating one reads its result.
//! - [`dce`]: dead pure instructions are removed.
//! - [`regalloc`]: which values live in a register for their whole life (both backends).
//! - [`layout`]: the block order of release code (cold blocks last).
//!
//! [`before_elision`] and [`after_elision`] run the IR passes of a release build around
//! stave-ir's bounds-check elision.

mod cfg;
pub mod cse;
pub mod dce;
mod edit;
pub mod layout;
pub mod regalloc;
pub mod slices;

use stave_ir::{Body, Op, TypeTable};

/// The passes that run before bounds-check elision: one length value per slice, so a check and
/// the loop bound that proves it name the same operand.
pub fn before_elision(b: &mut Body) {
    slices::unify_parts(b, Op::SliceLen);
}

/// The passes that run after bounds-check elision. `types` must have been through
/// [`slices::declare_elem_pointers`].
pub fn after_elision(types: &TypeTable, b: &mut Body) {
    if slices::split_elem_addrs(types, b) {
        slices::unify_parts(b, Op::SlicePtr);
    }
    slices::unify_chunk_bounds(b);
    cse::remove_redundant(types, b);
    dce::remove_dead(types, b);
}
