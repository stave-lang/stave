//! Witnesses of removed checks: what the check-removal pass ([`crate::bounds`]) claims about
//! each `check.bounds` and each contract check it wants gone, and about each contract check it
//! finds false on every path, in a form the independent checker ([`check`]) validates without any
//! search.
//!
//! **Checks.** A row names one check by an instruction index:
//!
//! - `check.bounds i n` (both unsigned): the goal is `i <= n - 1` right before it.
//! - a contract check: the instruction is a `call @f ..` with `f` = `runtime.panic.requires_failed`
//!   or `runtime.panic.ensures_failed`, the only instruction of its block `F`, which ends in
//!   `unreachable` and is entered only by the false edge of a `cbr c T F` (`T` not `F`) ending
//!   block `P`. The goal is `c` at the end of `P`, `c` a compare of ints under at most
//!   [`NOT_DEPTH`] `not`s: `x < y` is `x <= y - 1`, `x <= y` itself, `x == y` the two goals
//!   `x <= y` and `y <= x`; `!=` has none. Removing it turns the `cbr` into a `br T` and deletes
//!   the call: whenever `P` ends, `c` holds, so control went to `T` anyway.
//!
//! **Integers.** A [`Node`] is an integer: zero, an int SSA value, or the length of an SSA slice
//! value (what every `slice.len` of that value reads). A value's node is its mathematical value
//! under its own type: unsigned plain, signed two's complement. An operand denotes `node + k`: an
//! int constant `c` (`|c| <=` [`MAX_CONST`]) is `Zero + c`, an int value `v` is `Val(v) + 0`.
//! Since every node has one reading, an inequality between a signed and an unsigned node is an
//! ordinary statement about integers; what keeps the two kinds apart is that a source is read
//! only with the signedness its op computes in: both operands of a compare, `add`, or `sub` have
//! one int type and a constant operand must be a value of it; `check.bounds`, `sub.sat`, the bit
//! ops, and slice lengths are unsigned; a `Cast` relates a signed and an unsigned node only where
//! every value of the source type is one of the target type.
//!
//! **Rows.** An [`Edge`] states `from <= to + w` and names its [`Source`]; its multiplier `mul`
//! (at least 1) says how often it is added. A goal `a <= b + w` is proved by edges whose sum
//! `Σ mul·(from - to) <= Σ mul·w` has the left side `a - b` (every other node, zero aside,
//! cancels) and `Σ mul·w <= w`. A chain `a → .. → b` is the case with every multiplier 1; the
//! pass writes only chains. The edges of a goal may be empty (`0 <= w`).
//!
//! | Source | Reads | States |
//! |---|---|---|
//! | `Branch { block }` | the `cbr c` whose one edge is the only way into `block`; `c` a compare of ints under at most [`NOT_DEPTH`] `not`s | the compare (or its negation on the false edge), in every block `block` dominates |
//! | `Assert { call, arg }` | a `call @f` whose callee returns only through a branch on parameter `arg` being true (bodies of at most [`ASSERT_BLOCKS`] blocks) | the compare passed as that argument, after the call |
//! | `Check { inst }` | an earlier `check.bounds x y` (kept, or removed with its own witness) | `x < y`, after it |
//! | `Arith { inst }` | `v = add x c` or `v = sub x c`, int and trapping, `c` a constant of `v`'s type | `v = x + c` / `v = x - c` |
//! | `SliceLen { inst }` | `v = slice.len s` | `v = Len(s)` |
//! | `SliceMake { inst }` | `s = slice.make p n` | `Len(s) = n` |
//! | `SubSat { inst }` | `h = sub.sat n c`, unsigned, `c` a constant | `n <= h + c` |
//! | `Range { value }` | the type of `value`, an int of `bits` bits | unsigned: `0 <= value <= 2^bits - 1`; signed: `-2^(bits-1) <= value <= 2^(bits-1) - 1` |
//! | `Bits { inst }` | `v = and x y`, `v = rem x y`, or `v = shr x k`, unsigned; a constant operand `c` within the type (`rem`: `1 <= c`) | `and`: `v <= x`, `v <= y`; `rem`: `v <= x`, `v <= y - 1`; `shr`: `v <= x` |
//! | `Cast { inst }` | `v = cast x`, `cast.wrap x`, or `cast.sat x`: `x` an int value whose every value `v`'s int type holds (same signedness and at least as wide, or unsigned to a wider signed type) | `v = x` |
//!
//! A compare `x < y`, `x <= y`, `x == y` states `x <= y - 1`, `x <= y`, or both `x <= y` and
//! `y <= x`. Where the larger side `y` of an unsigned strict compare is `sub.sat n c`, it also
//! states `x + c < n`; where the smaller side `x` of an unsigned non-strict one is `sub.sat n c`,
//! also `n <= y + c`. An equality source (`Arith`, `SliceLen`, `SliceMake`, `Cast`) gives its
//! edge in both directions. The instruction of every source but `Branch` and `Range` must run
//! before the check on every path to it; for `Range`, the definition of `value` (an instruction,
//! or the block of a parameter). For a contract check "the check" is the end of block `P`, never
//! `F` (which only the failing edge enters).
//!
//! Why each `Bits` and `Cast` row holds of the values as mathematical integers: on unsigned ints
//! `x & y` is at most either operand; `rem` traps when `y = 0`, so a result that exists is
//! `x mod y`, below `y` and at most `x`; an unsigned `shr` is logical, `floor(x / 2^k)`, at most
//! `x` (an amount `>= bits` traps). A cast to a type holding every value of the source type keeps
//! the value under every one of the three kinds. A trapping `add`/`sub` that returns has the exact
//! sum. A constant operand is read as itself, so it must already be a value of the op's type.
//!
//! **Refutations.** A [`Refuted`] row claims that a contract check fails whenever its point is
//! reached: its edges prove the negation `b <= a - w - 1` of one goal `a <= b + w` (for `==`, of
//! either half). The checker validates it like a removal and removes nothing; the driver reports
//! it as an error.
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
/// The largest multiplier of an edge.
pub const MAX_MUL: u32 = 1 << 16;
/// The checked-IR paths of the fns a failed `requires` / `ensures` check calls.
pub const REQUIRES_FAILED: &str = "runtime.panic.requires_failed";
pub const ENSURES_FAILED: &str = "runtime.panic.ensures_failed";

/// An integer a witness talks about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Node {
    Zero,
    /// An int SSA value, read under its own type.
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
    Range { value: Value },
    Bits { inst: u32 },
    Cast { inst: u32 },
}

/// `from <= to + w`, added `mul` times.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    pub from: Node,
    pub to: Node,
    pub w: i128,
    pub src: Source,
    pub mul: u32,
}

/// One removed check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Removed {
    /// The instruction index of the check (a contract check: its failure call).
    pub inst: u32,
    /// The check's source site (for reports).
    pub site: Site,
    /// The proof of its goal, a span of [`FnWitness::edges`].
    pub edges: Span,
    /// The proof of the second goal of an `==` (empty otherwise).
    pub also: Span,
}

/// One contract check that fails whenever it is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Refuted {
    /// The failure call of the check.
    pub inst: u32,
    pub site: Site,
    /// The proof of the negation of goal `half` (0, or 1 for the second goal of an `==`).
    pub edges: Span,
    pub half: u8,
}

/// The removed and the refuted checks of one function: flat tables, rows in instruction order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnWitness {
    pub func: FuncId,
    /// Per block, a block that dominates it (the entry: itself; unreachable: [`NO_BLOCK`]).
    pub dom: Vec<u32>,
    pub removed: Vec<Removed>,
    pub refuted: Vec<Refuted>,
    pub edges: Vec<Edge>,
}
