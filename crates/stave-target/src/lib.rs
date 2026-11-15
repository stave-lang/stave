//! Features, triples, lowering tables (contracts/simd-ops.md §Features).

pub mod features;
pub mod triple;

pub use features::{Arch, ArchSet, Feature, FeatureError, FeatureSet};
pub use triple::{A64Ext, Target, TargetError, Triple, TripleError, X86Level};
