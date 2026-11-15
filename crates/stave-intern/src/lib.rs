//! Intern pool and canonical ids.
//!
//! - [`pool`]: sharded, thread-safe [`Interner`] mapping keys (strings, type keys, consts) to dense
//!   `u32` ids; equal keys get equal ids. Insertion order (and so the raw id) may depend on thread
//!   timing: renumber with [`canon`] before anything id-dependent becomes visible.
//! - [`canon`]: canonical renumbering, [`rank_by_key`] + [`Remap::apply`].
//! - [`hash`]: xxHash32 (canonical string order) and XXH3-128 (file content hash).

pub mod canon;
pub mod hash;
pub mod pool;

pub use canon::{CanonError, RawId, Remap, rank_by_key, rank_by_unique_key, string_rank_key};
pub use pool::{FrozenPool, Interner, PoolFull, StringPool};
