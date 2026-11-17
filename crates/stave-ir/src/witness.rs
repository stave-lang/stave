//! Witnesses of removed bounds checks: what the check-removal pass ([`crate::bounds`]) claims
//! about each `check.bounds` it wants gone, in a form the independent checker ([`check`])
//! validates without any search.
//!
//! A `check.bounds i n` may be removed only if `i < n` holds whenever control reaches it. Its
//! witness is a chain of inequalities between integers the IR names, each read off one place of
//! the IR:
//!
//! - A [`Node`] is an integer: zero, an unsigned SSA value, or the length of an SSA slice value
//!   (what every `slice.len` of that value reads).
//! - An operand denotes `node + k`: an int constant `c` (`0 <= c <=` [`MAX_CONST`]) is
//!   `Zero + c`, an unsigned int value `v` is `Val(v) + 0`.
//! - An [`Edge`] states `from <= to + w` and names its [`Source`].
//!
//! The chain of a check starts at the node of `i`, each edge starts where the one before ends,
//! and the last ends at the node of `n`; the weights must sum to at most `k_n - k_i - 1`. Then
//! `i <= n - 1`. A chain may be empty (two constants).
//!
//! | Source | Reads | States |
//! |---|---|---|
//! | `Branch { block }` | the `cbr c` whose one edge is the only way into `block`; `c` is a compare of unsigned ints under at most [`NOT_DEPTH`] `not`s | the compare (or its negation on the false edge), in every block `block` dominates |
//! | `Assert { call, arg }` | a `call @f` whose callee returns only through a branch on parameter `arg` being true (bodies of at most [`ASSERT_BLOCKS`] blocks) | the compare passed as that argument, after the call |
//! | `Check { inst }` | an earlier `check.bounds x y` (kept, or removed with its own witness) | `x < y`, after it |
//! | `Arith { inst }` | `v = add x c` or `v = sub x c`, unsigned and trapping, `c` a constant | `v = x + c` / `v = x - c` |
//! | `SliceLen { inst }` | `v = slice.len s` | `v = Len(s)` |
//! | `SliceMake { inst }` | `s = slice.make p n` | `Len(s) = n` |
//! | `SubSat { inst }` | `h = sub.sat n c`, unsigned, `c` a constant | `n <= h + c` |
//!
//! A compare `x < y`, `x <= y`, `x == y` states `x <= y - 1`, `x <= y`, or both `x <= y` and
//! `y <= x`. Where the larger side `y` of a strict compare is `sub.sat n c`, it also states
//! `x + c < n`; where the smaller side `x` of a non-strict one is `sub.sat n c`, also
//! `n <= y + c`. An equality source (`Arith`, `SliceLen`, `SliceMake`) gives its edge in both
//! directions. The instruction of every source but `Branch` must dominate the check.
//!
//! Dominance is certified, not computed, by the checker: [`FnWitness::dom`] names for every
//! reachable block one block that dominates it, and the checker accepts the table when every
//! CFG edge `u → v` has `dom[v]` among the ancestors of `u` in that tree (then every tree
//! ancestor of a block is on every path to it).
//!
//! Instruction indices are those of the body the pass ran on, before any removal.

pub mod check;

use crate::ir::{FuncId, Site, Span, Value};

/// The largest constant and the largest offset a witness may use (every `u64` fits).
pub const MAX_CONST: i128 = 1 << 64;
/// The most `not`s between a branch or an asserting call and the compare it tests.
pub const NOT_DEPTH: u32 = 8;
/// The largest body, in blocks, of a callee that counts as asserting a parameter.
pub const ASSERT_BLOCKS: usize = 16;
/// [`FnWitness::dom`] of a block that is not reachable.
pub const NO_BLOCK: u32 = u32::MAX;

/// An integer a witness talks about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    Zero,
    /// An unsigned int SSA value.
    Val(Value),
    /// The length of an SSA slice value.
    Len(Value),
}

/// Where an inequality is read off the IR (see the module table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Branch { block: u32 },
    Assert { call: u32, arg: u32 },
    Check { inst: u32 },
    Arith { inst: u32 },
    SliceLen { inst: u32 },
    SliceMake { inst: u32 },
    SubSat { inst: u32 },
}

/// `from <= to + w`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    pub from: Node,
    pub to: Node,
    pub w: i128,
    pub src: Source,
}

/// One removed `check.bounds`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Removed {
    /// The instruction index of the check.
    pub inst: u32,
    /// The check's source site (for reports).
    pub site: Site,
    /// Its chain, a span of [`FnWitness::edges`].
    pub edges: Span,
}

/// The removed checks of one function: flat tables, rows in instruction order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnWitness {
    pub func: FuncId,
    /// Per block, a block that dominates it (the entry: itself; unreachable: [`NO_BLOCK`]).
    pub dom: Vec<u32>,
    pub removed: Vec<Removed>,
    pub edges: Vec<Edge>,
}
