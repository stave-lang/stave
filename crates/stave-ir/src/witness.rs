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
//! **Integers.** A [`Node`] is an integer: zero, an int SSA value, the length of an SSA slice
//! value (what every `slice.len` of that value reads), or an int field behind a `&` parameter
//! (what every load of it reads, [`Node::Field`]). A value's node is its mathematical value
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
//! pass writes chains, and around a contradiction (a walk of negative weight) a chain with that
//! walk counted several times. The edges of a goal may be empty (`0 <= w`).
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
//! | `FieldLoad { inst }` | `v = load a`, `a = addr.field p k`, `p` a parameter of the entry block of type `&T` (not `&mut`), T not `std.cell.UnsafeCell`, `v` an int | `v = Field(p, k)` |
//! | `Getter { call }` | `v = call @g p`, `p` such a parameter, `g` of the one parameter of `p`'s type and the one-block body `a = addr.field %0 k; r = load a; ret r`, `r` an int | `v = Field(p, k)` |
//! | `Requires { clause }` | row `clause` of the function's own [`Summary::pre`] | the row about the function's parameters; only in the proofs of [`Summary::entry`] and [`Summary::safe`] |
//! | `Ensures { call, clause }` | a `call @g ..` and row `clause` of `g`'s [`Summary::post`] | the row about the call's arguments and result, after the call |
//! | `Arg { call, clause }` | a `call @g ..` and row `clause` of `g`'s [`Summary::pre`] that reads a `requires` check ([`Pre::entry`]) | the row about the call's arguments, after the call |
//!
//! A compare `x < y`, `x <= y`, `x == y` states `x <= y - 1`, `x <= y`, or both `x <= y` and
//! `y <= x`. Where the larger side `y` of an unsigned strict compare is `sub.sat n c`, it also
//! states `x + c < n`; where the smaller side `x` of an unsigned non-strict one is `sub.sat n c`,
//! also `n <= y + c`. An equality source (`Arith`, `SliceLen`, `SliceMake`, `Cast`, `FieldLoad`,
//! `Getter`) gives its edge in both directions. The instruction of every source but `Branch` and
//! `Range` must run before the check on every path to it; for `Range`, the definition of `value`
//! (an instruction, or the block of a parameter). For a contract check "the check" is the end of
//! block `P`, never `F` (which only the failing edge enters).
//!
//! Why a `Field` node is one value: a write to the bytes behind a `&` parameter of a running
//! function, outside an `UnsafeCell`, is undefined behaviour (LANG.md §3), and safe code cannot
//! make one (it needs a `&mut` while the `&` lives). An int field of a T that is not `UnsafeCell`
//! lies outside every `UnsafeCell`, so every load of it while the function runs, the getter's
//! during a call included, reads one value. `std.soa.Soa` keeps `len` in such a field and the
//! lowering builds every column as `slice.make (..) len`, so all columns of a `&Soa` parameter have
//! one length.
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
//! **Summaries.** A [`Summary`] is what a function states to its callers, as [`Row`]s over its
//! parameters and result ([`Term`]), each certified against the function's own IR:
//!
//! - [`Summary::pre`]: what a call must prove of its arguments to skip the `requires` checks. With
//!   every row true at entry, each `requires` check of the function holds ([`Summary::entry`]: one
//!   proof per check, in which `Requires` edges stand for the rows), and each instruction of
//!   [`Summary::safe`] (a trapping `add`/`sub` by a constant) stays in range. A row that is one
//!   check's goal read through equalities alone ([`Pre::entry`]) also holds whenever that check
//!   passed, so it is a fact after the call (`Arg`) and can be refuted at a call.
//! - [`Summary::post`]: what holds of the parameters and the result whenever the function
//!   returns: proved at the end of every reachable `ret` block ([`Summary::post_at`]), like a
//!   goal there, with `Result` read as the returned operand.
//!
//! A row is about a call by reading `Param(i)` as argument `i` (a constant argument: a value of
//! the parameter's type) and `Result` as the call's result. A function with a certified entry
//! has an **unchecked entry**: its body with the failure call of every `requires` check deleted
//! (the branch stays and its false side is an empty `unreachable` block, so the condition is
//! still a fact of the body). [`Calls::proved`] lists the calls that prove every `pre` row of
//! their callee right before the call; the checker retargets exactly those to the unchecked
//! entry. [`Calls::refuted`] lists calls where the negation of a row that reads a check is
//! proved: such a call fails that check, or stops before it, whenever it is reached.
//!
//! Why this holds across functions, recursion included: take one execution and the first
//! moment a stated fact is false (a deleted check's condition, a `post` row at a return, a `pre`
//! row at an unchecked entry). Its proof uses only facts stated earlier in that execution (a
//! call that returned did so before; parameters are SSA values and keep their entry values), so
//! all of them hold and so does the proof: a contradiction. The claim is partial correctness:
//! nothing is said of a call that does not return.
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
/// The most reachable `ret` blocks of a function whose summary has `post` rows.
pub const SUMMARY_RETURNS: usize = 16;

/// An integer a witness talks about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Node {
    Zero,
    /// An int SSA value, read under its own type.
    Val(Value),
    /// The length of an SSA slice value.
    Len(Value),
    /// Int field `.1` of what parameter `.0` of the entry block, a `&T` (T not
    /// `std.cell.UnsafeCell`), points to: one value while the function runs.
    Field(Value, u32),
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
    FieldLoad { inst: u32 },
    Getter { call: u32 },
    Requires { clause: u32 },
    Ensures { call: u32, clause: u32 },
    Arg { call: u32, clause: u32 },
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

/// An integer a summary row talks about: of the function's parameters (by index) and result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    Zero,
    /// Int parameter `i`.
    Param(u32),
    /// The length of slice parameter `i`.
    LenParam(u32),
    /// The int result.
    Result,
    /// The length of the slice result.
    LenResult,
}

/// `a <= b + w`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Row {
    pub a: Term,
    pub b: Term,
    pub w: i128,
}

/// One row a call proves of its arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pre {
    pub row: Row,
    /// The row of [`Summary::entry`] whose goal `half` this row is, read through equalities
    /// alone ([`NO_BLOCK`]: a row only other proofs need).
    pub entry: u32,
    pub half: u8,
}

/// One trapping `add`/`sub` by a constant that stays in its type's range when the `pre` rows
/// hold: the proofs, right before it, of `lo <= x ± c` and of `x ± c <= hi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Safe {
    pub inst: u32,
    pub lo: Span,
    pub hi: Span,
}

/// The proof of `post` row `row` at the end of `ret` block `block`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PostAt {
    pub row: u32,
    pub block: u32,
    pub edges: Span,
}

/// What one function states to its callers (see the module text). Spans are of
/// [`FnWitness::edges`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub pre: Vec<Pre>,
    /// Per reachable `requires` check, in instruction order: its goals proved from `pre`.
    /// Empty: no unchecked entry.
    pub entry: Vec<Removed>,
    /// In instruction order.
    pub safe: Vec<Safe>,
    pub post: Vec<Row>,
    /// Per row of `post`, per reachable `ret` block in block order.
    pub post_at: Vec<PostAt>,
}

/// A call that proves every `pre` row of its callee.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CallProved {
    /// The `call` instruction.
    pub inst: u32,
    pub site: Site,
    /// Per `pre` row of the callee, its proof: a span of [`Calls::goals`].
    pub rows: Span,
}

/// A call whose arguments make `pre` row `row` of its callee false whenever it is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CallRefuted {
    pub inst: u32,
    pub site: Site,
    pub row: u32,
    /// The proof of the row's negation.
    pub edges: Span,
}

/// The calls of one function the pass decided, rows in instruction order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Calls {
    pub proved: Vec<CallProved>,
    /// Spans of [`FnWitness::edges`].
    pub goals: Vec<Span>,
    pub refuted: Vec<CallRefuted>,
}

/// The removed and the refuted checks of one function, its summary, and its decided calls: flat
/// tables, rows in instruction order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnWitness {
    pub func: FuncId,
    /// Per block, a block that dominates it (the entry: itself; unreachable: [`NO_BLOCK`]).
    pub dom: Vec<u32>,
    pub removed: Vec<Removed>,
    pub refuted: Vec<Refuted>,
    pub edges: Vec<Edge>,
    pub summary: Summary,
    pub calls: Calls,
}
