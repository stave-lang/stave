# Contract: the release passes (level `checked` -> level `opt`)

The release passes turn the canonical `checked` module of a `release` build into the module of
level `opt` ([checked-ir.md](checked-ir.md) "Levels and forms"). This file is their whole
specification: two implementations that follow it print byte-equal `opt` text from the same input.
The full pipeline is stage1's (`stage1 ir opt`, [cli-stage1.md](cli-stage1.md)); stage0
implements the base passes only (rule P3), for `stave build --release` and `stave ir opt`.

Every number below that is not a rule of the IR comes from the table
`crates/stave-ir/tables/opt.toml` ("Constants"); `tools/gen-tables` generates it into both
compilers.

## 0. Terms and kinds

- **Module, body, block, instruction, operand, value, terminator, target, site**: as in
  checked-ir.md. A *target* is `b<N>(<args>)` in a terminator; an *edge* into block X is one target
  naming X, counted with multiplicity (`cbr c X() X()` is two edges into X; a `switch` case and its
  default to one block are two).
- **Print order** of a body: blocks in stored order, a block's instructions in order, then its
  terminator. *Input numbering* of a phase: the canonical numbering (checked-ir.md "Canonical
  module") of the module the phase reads: `b<N>`, and `<k>` = the 0-based index of an instruction
  in its block.
- **Same operand**: the same value; or two constants of the same kind with the same bits
  (checked-ir.md "Canonical module"); or the same `s<S>`; or the same `@<inst>`; or the same
  `t<T>`.
- **Int constant**: an operand that is a decimal int constant. Its value is a mathematical
  integer (the operand position gives its type; `bool` is 0 or 1, `char` its code point).
- **Weight**: `w(i)` of an instruction is the `weight` of its op's row in the table; `w(t)` of a
  terminator is its `[term]` weight. The weight of a body is the sum over its instructions and
  terminators.
- **Reachable**: from `b0` over edges.

Every rule below is one of three deterministic kinds, named after the rule:

- **[N]** the normal form of a terminating, confluent rewrite system (a pointwise definition is
  the case where no rewrite enables another);
- **[F]** the least fixpoint of a monotone function on a finite lattice whose join is
  associative, commutative, and idempotent;
- **[A]** a minimum, or a prefix of a sorted list, under a total order.

The result of each is a function of its input alone. Rules apply to fns independently except
where a rule reads another fn's result (only the decision and the expansion, rules D5 and X1,
which read callees at lower levels), so neither evaluation order nor a thread schedule that keeps
the level order can change the output.

## 1. Pipeline

**P1.** `opt(M) = canon(B(R^n(M)))`: `n = ROUNDS` rounds `R(M') = canon(C*(X(D(M'))))`, then
the base passes on the canonical result. `M` is the loaded `checked` module of a `release` file,
canonical. `D` decides (section 3), `X` expands (section 4), `C*` cleans up (section 5), `B` is
the base passes (section 6), `canon` is checked-ir.md "Canonical module". Each phase reads only
the module the previous one wrote. [N: a composition of functions.]

**P2.** Every phase keeps every fn, its instance id, header, and `claim` lines, and the site,
string, static, `main`, and `cell` lines and the stack section. Only bodies change, and with them
the type table and decls that `canon` derives from the bodies.

**P3.** With `--passes=base` (and in stage0) the pipeline is `canon(B(M))`.

**P4.** After each phase, and always on the output, the module satisfies the IR verifier and the
invariants of section 7. A violation is a compiler bug. A `switch` whose scrutinee is an int
constant is legal at level `opt` (checked-ir.md "Loading" step 3): X writes one when it
substitutes a constant argument for a param a callee switches on, and C1 when it substitutes a
constant for a param. C3 of the same clean-up round turns every such `switch` into a `br`, and
nothing after C3 (C4-C6, `B`, `canon`) puts a constant into a scrutinee, so the module after `C*`
and the output never hold one; only the module after `X` (`--dump-phase=inline`) may.

## 2. Constants

**K1.** The table `crates/stave-ir/tables/opt.toml`:

| Name | Table key | Value | Meaning |
|---|---|---|---|
| `T` | `limits.threshold` | 32 | largest closed size `S` of an inlined callee |
| `CAP` | `limits.cap` | 2000 | the closed size up to which a caller still grows |
| `W_CALL` | `limits.call_saved` | 3 | the weight an inlined call removes |
| `SLOT_CALLEE` | `limits.slot_callee` | 64 | largest closed slot bytes `SB` of an inlined callee |
| `SLOT_CALLER` | `limits.slot_caller` | 1024 | most slot bytes one caller gains in a round |
| `ROUNDS` | `limits.rounds` | 1 | rounds of rule P1 (`stage1 ir opt --limit NAME=N` overrides a limit for one run) |
| `w(term)` | `term.*` | `cbr` 1, `switch` 2, `br` `ret` `unreachable` 0 | terminator weights |
| `w(op)`, `removable`, `fold` | `[[op]]` | per op | instruction weight; rules C2, C6, B5 |
| inlined known items | `known.inline` | `SetUserSite` (`runtime.panic.set_user_site`) | rule F8 |

Op weights: `slot` 0; `call`, `call.ind` 3; `intrinsic.*` 2; every other op 1. The weights,
`T`, `CAP`, and the slot limits are tunable data: changing them changes no rule, within K3.

**K2.** An op row's `removable`: `yes`; `no`; `float` (yes iff the result type is a float type);
`const-index` (yes iff operand 1 is an int constant); `lanes` (yes iff the `vec.*` or `mask.*`
op is not marked † in checked-ir.md "Vector and mask ops" and is none of the memory ops
`vec.load`, `vec.store`, `vec.load_aligned`, `vec.store_aligned`, `vec.store_stream`,
`vec.load_masked`, `vec.store_masked`, `vec.gather`, `vec.scatter`). A row whose name ends in `*`
covers every op name with that prefix that no other row names; every op has exactly one row.

**K3.** The table satisfies `W_CALL <= w(call)` and `w(br) = w(ret)`. `tools/gen-tables` refuses
a table that does not, and `stage1 ir opt --limit` refuses an override that does not. The first
keeps every `S` non-negative (D1); the second keeps `S` exact (D1), since X turns each copied
`ret` into a `br`.

## 3. Decision `D`

Input: the canonical module `M'` of the round. All facts are tables over `M'` in its input
numbering.

**F1. Site table.** One row `(f, b, k, g)` per instruction `call` of fn `f` in block `b` at
index `k` whose operand 0 is `@g`. `call.ind` is never a site. [N: a filter.]

**F2. `size0(f)`**: the weight of `f`'s body. [N: a sum; addition is associative and
commutative.]

**F3. `slot0(f)`**: the sum, over `f`'s `slot` instructions, of the layout size on the header's
target of the slot's pointee type (`slot t<ref mut T>`: the size of T). [N: a sum.]

**F4. `ret(f, b)`**: block `b` of `f` reaches, in zero or more edges, a block that ends in `ret`.
[F: backward reachability over the edges; the lattice is the set of blocks, the join is union.]

**F5. Graph `G`**: an edge `f -> g` per distinct pair of a site row. [N: a set.]

**F6. `cyc(g)`**: the strongly connected component of `g` in `G` has two or more fns, or `g -> g`
is an edge. [N: a property of `G`, not of the algorithm that finds the components.]

**F7. `level(f)`**: on the condensation of `G` (one node per component), 0 when `f`'s component
has no edge to another component, else 1 + the largest level of a component it has an edge to.
Fns without a body have level 0. [F: longest path in a finite DAG, a least fixpoint of
`level(c) = max(0, 1 + level(c'))` over the edges, join = max.]

**F8. Callee tests**, each a predicate of `g` alone, in this order (the verdict names in bold):

1. **`nobody`**: `g` has no body.
2. **`noret`**: no block of `g` ends in `ret`.
3. **`cyclic`**: `cyc(g)`.
4. **`symbol`**: `g` has a `symbol`.
5. **`known`**: `g` has the `known` mark and its known item is not one of `known.inline`
   (an item with an ABI signature, or a panic report entry, is never listed).
6. **`indirect`**: `g`'s body holds a `call.ind` whose operand 0 is a value.
7. **`gpu`**: `g`'s body holds a `gpu.*` op or an `intrinsic.<name>` of arch `ptx`. No body
   holds either today (the verifier refuses a `ptx` intrinsic on a CPU target, and no `gpu.*` op
   exists); the test becomes reachable when `gpu.*` ops or a PTX target reach the IR.

A callee with the `altstack` mark needs no test of its own: the verifier allows the mark only on
a panic report entry, which `known.inline` never lists, so it fails `noret` or `known`. An
`unbuilt` callee has no body (`nobody`) or a trap for a body (`noret`). [N: pointwise.]

**F9. Caller tests** of a row `(f, b, k, g)`, in this order:

- **`altstack`** (between `cyclic` and `symbol` of F8): `f` has the `altstack` mark: its frame
  must stay on the report stack.
- **`features`**: the features of `g`'s `needs` are not all in the closure, under the
  implications of simd-ops.md "Features", of the target baseline of the header and `f`'s
  `needs`.
- **`cold`**: not `ret(f, b)`.

[N: pointwise.]

**D1. Closed size and slots.** For a fn `f` with a body, with `A(f)` its accepted rows (D4):

    S(f)  = size0(f) + sum over (f, b, k, g) in A(f) of (S(g) - W_CALL)
    SB(f) = slot0(f) + sum over (f, b, k, g) in A(f) of SB(g)

`SB(f)` is the slot bytes of the expanded body `F(f)` of rule X1. `S(f)` is its weight when
`W_CALL` equals `w(call)` (with K3's `w(br) = w(ret)`); under K3 it is never negative. Both are
defined by recursion on `level`: an accepted
callee lies in a lower level (D5). [N: sums over a well-founded recursion.]

**D2. Candidates** of `f`: its rows that pass F8 and F9 and whose callee has `S(g) <= T` and
`SB(g) <= SLOT_CALLEE`.

**D3. Order.** Candidates sort by the key `(S(g), b, k)` ascending. The key is total: `(b, k)`
names one instruction of `f`. [A.]

**D4. Budget cut.** For the j-th candidate in that order let `a_j = max(S(g_j) - W_CALL, 0)` and
`z_j = SB(g_j)`. Candidate j is accepted iff `a_1 + .. + a_j <= max(CAP - size0(f), 0)` and
`z_1 + .. + z_j <= SLOT_CALLER`. Both sums grow with j, so `A(f)` is a prefix of the sorted list.
[A: a prefix of a sorted list.]

**D5. Schedule.** Fns decide by level, ascending; all fns of one level may decide at once. A
callee that passes F8 is not `cyclic`, so it lies in another component and, `G` having an edge
to it, in a lower level: its `S` and `SB` are final before any reader needs them. Members of a
cyclic component decide like other callers; their callees inside the component fail
`cyclic`. [F over levels; within a level, A per fn.]

**D6. Verdict** of a row: the name of the first test it fails, in the order `nobody`, `noret`,
`cyclic`, `altstack` (caller), `symbol`, `known`, `indirect`, `gpu`, `features`, `cold`, then
**`size`** (`S(g) > T`), **`slots`** (`SB(g) > SLOT_CALLEE`), **`budget`** (a candidate D4 does
not accept); else **`inline`**. [N: the first failing test of a fixed list.]

**D7. Decisions read the input only.** No fact is read from an expanded or cleaned body: `S` is
arithmetic on the tables of `M'`.

## 4. Expansion `X`

**X1. Expanded bodies.** `F(f)` is `f`'s body with every accepted row of `f` expanded by X2-X7,
each callee copied from `F(g)` (the callee expanded, not cleaned up). Fns by level, ascending;
fns of one level are independent. [N: each `F(f)` is a function of `f` and of `F(g)` for its
accepted callees.]

**X2. One site.** For an accepted row whose instruction is `[%r = ]call t @g a1..an[ !s]` at index
`k` of block `B` of `f`, with `F(g)` = blocks `c0..cm`, `c0`'s params `p1..pn`, `c0`'s leading
`slot` instructions `q1..qj`, the substitution `σ` maps `pi -> ai` (an argument is any operand:
value, constant, `s<S>`, `@<inst>`), every other value of `F(g)` (the slots included) to a fresh
value of `f` of the same type, and every block `ci` (i >= 1) to a fresh block. `c0` is never a
target, so each param has one binding.

**X3. Head.** `B` keeps its label and params. Its instructions are those before index `k`, then
`c0`'s instructions after its slots, under `σ`; its terminator is `c0`'s under `σ`.

**X4. Callee blocks.** `c1..cm` under `σ`, in `F(g)`'s stored order, directly after the head.

**X5. Continuation `K`.** A new block directly after them. Its params are `(%r: t)` when the call
has a result, else none; its instructions are those of `B` after index `k`; its terminator is
`B`'s. In the copied blocks (the head included) `ret v` becomes `br K(σ(v))`, `ret` becomes
`br K()`. `%r` keeps its identity as `K`'s param, so no use of it changes.

**X6. Several rows of one block** are expanded left to right: the continuation of one is the
block the next is split from. The call's own `!s` goes with the call. Every copied instruction
keeps its `!<row>`; types, strings, sites, statics, intrinsic names, and fns are module-wide and
copied by id.

**X7. Slots.** `σ(q1)..σ(qj)` of all accepted rows of `f` go to `f`'s `b0` directly after `f`'s
own leading slots: rows in `(b, k)` order of the input, each callee's slots in their order in
`F(g)`. So the slots of `F(f)` are again a prefix of `b0`, as X2 assumes for `F(g)`.

X2-X7 fix every block's place and every instruction's place; the numbering of `F(f)` is then
`canon`'s.

## 5. Clean-up `C*`

`C*` runs on every body, whether or not it received inlined code. Each phase is a function of one
body; fns are independent. Every block of the body is reachable at the start of a round (X keeps
reachability, C4 restores it).

**C0. Rounds.** One clean-up round is C1, C2, C3, C4, C5, C6 in this order, each run to its own
normal form or fixpoint. Rounds repeat until a round changes nothing. Every change lowers
`edges + blocks + instructions + block params + cbr and switch terminators` and none raises it,
so the loop ends. [N: iterating a function to its fixpoint is a function.]

**C1. Single-edge params.** Every block other than `b0` with exactly one incoming edge loses its
params: each is replaced everywhere by that edge's argument at its position, and the edge's
arguments are dropped. All such blocks at once; when an argument is itself a replaced param, the
replacement is followed to its end (no chain is a cycle: a cycle of single-edge blocks is not
reachable). [N: one simultaneous substitution.]

**C2. Constant folding.** An instruction whose op's `fold` is `int`, whose operands are all int
constants, and whose result below is defined, is deleted, and every use of its result reads the
result constant. A `select` (`fold` `cond`) whose operand 0 is an int constant is deleted, and its
result is replaced by operand 1 when the constant is 1, by operand 2 when it is 0. Repeat until no
instruction qualifies. Results, with `x`, `y` the operand values and the result type R (an int
type, `bool`, or `char`; `bits` is R's width, 64 for `int s|u size` on every current target):

- `add sub mul`: `x + y`, `x - y`, `x * y`; undefined (stays) when outside R's range.
- `div rem`: truncating quotient and remainder (the remainder has the sign of `x`); undefined
  when `y = 0`, or when R is signed, `x` is R's minimum, and `y = -1`.
- `neg`: `-x`; undefined when outside R's range.
- `add.wrap sub.wrap mul.wrap`: the exact result reduced into R's range modulo 2^bits.
  `add.sat sub.sat mul.sat`: the exact result clamped to R's range.
- `shl shr`: undefined when `y < 0` or `y >= bits`; `shl` is `x * 2^y` reduced modulo 2^bits into
  R's range; `shr` is `floor(x / 2^y)` (arithmetic when R is signed, `x` read in R).
  `shl.wrap`: `shl` by `y mod bits` (the amount's two's-complement bits masked by `bits - 1`).
- `and or xor not`: on `bool` the logical operation; on ints the bitwise operation on the
  two's-complement `bits`-wide representations, read back in R.
- `popcount clz ctz`: the count over the `bits`-wide representation of `x` (`clz` and `ctz` of 0
  are `bits`).
- `eq ne lt le gt ge`: the comparison of the two mathematical values; the result is 1 or 0.
- `cast`, `cast.wrap`, `cast.sat` with R an int type or `char`: `x`; `x` reduced modulo 2^bits
  into R's range; `x` clamped to R's range. Other casts (and these with another R) do not fold.

This is checked-ir.md "Ops" applied to constants; stave-ir `arith` computes the same. Floats,
vectors, masks, aggregates, `mul.wide`, `cast.round`, `cast.try`, and `check.bounds` never fold.
[N: a fold never turns a constant back into a value, so the set of folded instructions is the
least set closed under "every operand is a constant or a folded result"; the order of folds
cannot change it.]

**C3. Branch folding.** `cbr k T F` with an int constant `k` becomes `br T` when `k = 1`, `br F`
when `k = 0`. `switch k ..` with an int constant `k` becomes `br` of the first case whose value is
`k`, else of the default. `cbr c X(a1..) X(b1..)` whose two targets name one block with the same
operands, position by position, becomes `br X(a1..)`. Targets keep their arguments. [N:
pointwise; a `br` is not rewritten again.]

**C4. Unreachable blocks.** Blocks not reachable from `b0` are deleted with their instructions and
edges; the others keep their order. A reachable block uses no value of a deleted one: a
definition dominates its uses, and a path to the use that avoids the deleted block would have
existed before. [F: reachability.]

**C5. Block merge.** For a block `X` other than `b0` whose one incoming edge comes from a block
`P != X` that ends in `br X()`: `X`'s instructions are appended to `P`'s, `P` takes `X`'s
terminator, and `X` is deleted. The merged block stands where `P` stood. Maximal chains merge
into their first block. [N: merging `P, X` keeps the condition of every other mergeable pair, and
a cycle of such pairs is unreachable, so the normal form is unique.]

**C6. Dead code and dead params.** The live values are the least set that holds: every operand of
an instruction that is not removable (K2); the `cbr` condition, the `switch` scrutinee, and the
`ret` operand of every terminator; every operand of an instruction whose result is live; the
argument at position i of every edge into a block whose param i is live. An instruction is
deleted when it is removable and its result is not live. A param of a block other than `b0` that
is not live is deleted, with the argument at its position on every edge into the block; the other
params keep their order. [F: a least fixpoint over sets of values.]

## 6. Base passes `B`

The base passes run on every body, in this order, on a canonical module (P1) read in its input
numbering. stage0 implements them in stave-opt (`optimize`); the base gate (section 9) holds the
two implementations to each other.

**B1. Slice lengths.** The *parts* are the instructions `%v = slice.len t s` whose operand is a
value `s`. Parts with equal `(s, t)` form a group. A group *moves* when it has two or more parts,
or one of its parts is in a block of greater loop depth (B7) than the block that defines `s`.
Moving: the group's first part in print order is taken out of its block and inserted at the
*place* of `s`: directly after `s`'s defining instruction; or, when `s` is a param of block `x`,
directly before the first instruction of `x` that is not a `slot` (in `b0` after its slots,
elsewhere at the block's start). It keeps its result, type, operand, and site; every other part of
the group is deleted and its result replaced by the first part's. Groups that one step inserts at
one place go in ascending `(s, t, k)` order: `s` by value number in B's input (so the params of
one block in param order), `t` by type id in B's input with a type B2 added after every other,
`k` the group's constant (B3). An insertion goes directly at its place, before the instructions
an earlier step inserted there. A place after an instruction that moves or is deleted in the same
step is that instruction's old position.
[A: the survivor of a group is its minimum in print order; groups are disjoint.]

**B2. Element addresses.** Every `addr.elem s i`, `vec.load s i`, and `vec.store s i v` (exactly
those operand counts) whose operand 0 is a value `s` of type `ref` or `ptr` (`mut` or `const`)
`slice T` (with or without `align`), with `T` neither an array nor a slice, gets
`%p = slice.ptr t<ptr mut T> s` directly before it, with the access's site, and reads `%p` as
operand 0. The type `ptr mut T` is added to the type table when the module lacks it. [N: pointwise.]

**B2'.** Only when B2 changed the body: B1 with `slice.ptr` in place of `slice.len`.

**B3. Chunk bounds.** B1 with the parts `%v = sub.sat t n k`, `n` a value and `k` an int constant,
grouped by `(n, t, k)`.

**B4. Repeated loop steps.** A *candidate* is an instruction whose op is one of `add sub mul neg
add.wrap sub.wrap mul.wrap shl.wrap add.sat sub.sat mul.sat shl shr and or xor`, whose result type
is an int type, and which has at most two operands. Its *key* is `(op, result type, operands)`,
an int constant compared by value, a value by identity, any other operand as the same operand
(section 0). A candidate is a *repeat* when every use of its result, in B4's input, is an edge
argument. The walk visits the dominator tree of the body in preorder from `b0`, a block's
children in ascending block number, and keeps a scope: the keys added by the blocks on the
current path. At each candidate `i` of the visited block `x`, in order: if its key is in scope
with result `p` added by block `a`, and `i` is a repeat, and `a` is reached from `x` (below), `i`
is deleted and its result replaced by `p`; if its key is in scope otherwise, nothing happens; if
its key is not in scope, it enters the scope with `i`'s result and block `x`. *Reached*: a search
starts with the successors of `x` on a stack (targets in terminator order: `then`, `else`; the
cases in order, then the default; one entry per edge), pops the last entry `y`, answers yes when
`y = a`, skips `y` when already marked, else marks it, answers no once more than 4096 blocks are
marked, and pushes `y`'s successors the same way; an empty stack answers no. [A: among equal keys
in scope the survivor is the first in dominator preorder and instruction order; the search is a
fixed sequential procedure on one body.]

**B5. Dead instructions.** The live values are the least set that holds: every operand of an
instruction that is not removable (K2); every operand of every terminator (condition, scrutinee,
returned value, every edge argument); every operand of an instruction whose result is live. A
removable instruction whose result is not live is deleted. Params are not removed. [F.]

**B6.** Then `canon` (P1).

**B7. Loop depth** of a block `x`: the number of edges `l -> h` (with multiplicity, section 0)
such that `h` dominates `l` and `x` is in their natural loop (`h`, and every block that reaches `l`
without passing through `h`), at most 255. The bodies here are reducible (lowering builds
structured control flow; expansion substitutes reducible bodies for an instruction; clean-up only
deletes edges and merges single-edge pairs), so these are the back edges of any depth-first
search. [F.]

## 7. Order and invariants of level `opt`

**O1. Block order** is the stored order the rules give: input blocks keep their relative order
(X3); an expanded call puts, directly after the block holding the code before it, the callee's
blocks other than its entry in their order, then the block holding the code after it (X4, X5);
calls of one block are expanded left to right (X6); a deleted block leaves the others in order
(C4); a merged block stands where its first part stood and holds the first part's instructions,
then the second's (C5). B adds, moves, and deletes instructions only (B1-B5). `canon` keeps the
stored block order and numbers values and pools.

**O2. Instruction order** inside a block: X3, X5, X7, C5, B1-B3. Every `slot` of a body is in
`b0` before every other instruction: first the fn's own in their order, then those of expanded
calls, rows in `(b, k)` order of the round's input, each callee's slots in their order.

**I1. Invariants** of every module a phase writes: verifier-clean; canonical after `canon`; every
block reachable; no fn added or removed; site table, strings, statics, `claim` lines equal to the
input's; slots a prefix of `b0`; no `call.ind` through a value in a fn whose `checked` body has
none. No pass removes a trap an execution would reach: a check (`check.bounds`, a trapping op, a
`requires` or `ensures` test) goes only with a block no longer reachable (C4), or when an equal
instruction that dominates it already ran on the same operands (B4); C2 never folds an op that
would trap.

## 8. The `inline` dump

`stage1 ir opt --dump=inline` prints the decision tables instead of the module text, per round:
a line `round <r>` (`r` from 1), then per fn with a body, by instance id, its line and then one
line per site row of the fn in `(b, k)` order:

```
round <r>
fn @<inst> size <size0> closed <S> slots <SB> level <level>[ cyclic]
site @<inst> b<b> <k> @<callee> <S(callee)> <verdict>
```

`<S(callee)>` is `-` for a callee without a body; `b<b>` and `<k>` are in the round's input
numbering; `cyclic` is printed when `cyc(f)`. Each line ends with one `\n`; nothing else is
printed.

## 9. Gates

- **Goldens**: hand-written `checked` texts with their `inline` dump, their text after `X`
  (`--dump-phase=inline`), after `C*` (`--dump-phase=clean`), and the output, byte-equal; the
  expected texts are derived from this file, not from an implementation. The cases include the
  examples of section 10 and at least: a single-block callee; several `ret`s; a callee with a
  loop; slots; nesting three deep; self and mutual recursion; `features` both ways; a cold site;
  a budget cut and its tie-break; constant, string, and `@fn` arguments; a unit call and an unused
  result; a site in `b0`; two sites in one block; an `@n.u` callee; every reachable verdict.
- **Base**: on every `release` cell of the stage1 IR corpus, `stage1 ir opt --passes=base` equals
  `stave ir opt` on the same `checked` text, both in the same form (gate or emit).
- **Verifiers**: stage1's verifier after every phase (`--verify-phases`) and stage0's
  (`stave ir check`) on every full `opt` output: no fault.
- **Behaviour**: `stave ir run` with `STAVE_INTERP_CHECK=1` on the `checked` text and on its full
  `opt` text: equal stdout, stderr, and exit status, no memory error; the conformance run suite on
  binaries built by `stave ir emit --release`, `stage1 ir opt`, `stave ir build`.
- **Determinism**: the same output under one thread and under all threads; `opt(fmt(t)) =
  opt(t)` and `fmt(opt(t)) = opt(t)`; the same for the `inline` dump.
- **Folding**: every `fold` op on every int type against the stave-ir interpreter, exhaustive on
  8 bits, boundary and random values wider, traps included.
- **Mutants** (at acceptance and on every change of the optimizer, not in every `check`): a fixed
  list of seeded faults in the optimizer (a swapped argument in `σ`, a dropped slot move, a
  missing `K` param, an off-by-one block place, a skipped tie-break, a wrong prefix cut, a fold
  with the wrong signedness, ...), each built and run through the goldens, the verifiers, and the
  behaviour gate: every mutant is killed.

## 10. Worked examples

Target `x86_64-linux-v2` (baseline `sse42`), the constants of K1. Each input is a complete
`release checked` text; each output repeats the input's lines except the header (`release opt`)
and the bodies shown. `canon` keeps every type in 10.1-10.6 and 10.9-10.10, so type ids do not
change there; 10.7 and 10.8 show their whole output.

### 10.1 A getter in a loop, nested three deep (F, D, X, C1, C5)

Input:

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 64
type t1 int s 32
type t2 bool
type t3 ref mut t0
type t4 ref mut t5
type t5 tuple t0 t0
site 0 "src/main.stv" 1 4 main.len
site 1 "src/main.stv" 3 4 main.count
site 2 "src/main.stv" 5 15 main.count
site 3 "src/main.stv" 6 13 main.count
site 4 "src/main.stv" 11 4 main.main
site 5 "src/main.stv" 13 12 main.main
main @2
fn @0 main.len(t4) -> t0 site 0 {
b0(%0: t4):
  %1 = addr.field t3 %0 1
  %2 = load t0 %1
  ret %2
}
fn @1 main.count(t4) -> t0 site 1 {
b0(%0: t4):
  br b1(0, 0)
b1(%1: t0, %2: t0):
  %3 = call t0 @0 %0 !2
  %4 = lt t2 %1 %3
  cbr %4 b2() b3()
b2():
  %5 = add t0 %2 %1 !3
  %6 = add.wrap t0 %1 1
  br b1(%6, %5)
b3():
  ret %2
}
fn @2 main.main() -> t1 site 4 {
b0():
  %0 = slot t4
  %1 = make t5 7 4
  store () %0 %1
  %2 = call t0 @1 %0 !5
  %3 = cast.wrap t1 %2
  ret %3
}
```

Decision (`size0`: `len` 1 + 1; `count` 3 + 1 + 1 + 1 + 1; `main` 1 + 1 + 3 + 1; `slot0(main)` =
16; `S(count)` = 7 + (2 - 3); `S(main)` = 6 + (6 - 3)):

```
round 1
fn @0 size 2 closed 2 slots 0 level 0
fn @1 size 7 closed 6 slots 0 level 1
site @1 b1 0 @0 2 inline
fn @2 size 6 closed 9 slots 16 level 2
site @2 b0 3 @1 6 inline
```

After `X` (`--dump-phase=inline`; `len` unchanged):

```
fn @1 main.count(t4) -> t0 site 1 {
b0(%0: t4):
  br b1(0, 0)
b1(%1: t0, %2: t0):
  %3 = addr.field t3 %0 1
  %4 = load t0 %3
  br b2(%4)
b2(%5: t0):
  %6 = lt t2 %1 %5
  cbr %6 b3() b4()
b3():
  %7 = add t0 %2 %1 !3
  %8 = add.wrap t0 %1 1
  br b1(%8, %7)
b4():
  ret %2
}
fn @2 main.main() -> t1 site 4 {
b0():
  %0 = slot t4
  %1 = make t5 7 4
  store () %0 %1
  br b1(0, 0)
b1(%2: t0, %3: t0):
  %4 = addr.field t3 %0 1
  %5 = load t0 %4
  br b2(%5)
b2(%6: t0):
  %7 = lt t2 %2 %6
  cbr %7 b3() b4()
b3():
  %8 = add t0 %3 %2 !3
  %9 = add.wrap t0 %2 1
  br b1(%9, %8)
b4():
  br b5(%3)
b5(%10: t0):
  %11 = cast.wrap t1 %10
  ret %11
}
```

The weights of these bodies are 6 and 9: `S(count)` and `S(main)` (D1). `main` copies
`F(count)`, which already holds `len`. Clean-up, round 1: C1 substitutes `count`'s `b2` param
(one edge, from `b1`) and `main`'s `b2` and `b5` params; C5 merges `b2` into `b1` in both, and
`b5` into `b4` in `main`; C2, C3, C4, C6 change nothing. Round 2 changes nothing. `B` changes
nothing (no slice, no repeated key, nothing dead). Output:

```
fn @1 main.count(t4) -> t0 site 1 {
b0(%0: t4):
  br b1(0, 0)
b1(%1: t0, %2: t0):
  %3 = addr.field t3 %0 1
  %4 = load t0 %3
  %5 = lt t2 %1 %4
  cbr %5 b2() b3()
b2():
  %6 = add t0 %2 %1 !3
  %7 = add.wrap t0 %1 1
  br b1(%7, %6)
b3():
  ret %2
}
fn @2 main.main() -> t1 site 4 {
b0():
  %0 = slot t4
  %1 = make t5 7 4
  store () %0 %1
  br b1(0, 0)
b1(%2: t0, %3: t0):
  %4 = addr.field t3 %0 1
  %5 = load t0 %4
  %6 = lt t2 %2 %5
  cbr %6 b2() b3()
b2():
  %7 = add t0 %3 %2 !3
  %8 = add.wrap t0 %2 1
  br b1(%8, %7)
b3():
  %9 = cast.wrap t1 %3
  ret %9
}
```

### 10.2 Constant arguments, two sites in one block, an unused result (C1-C6)

Input:

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 64
type t1 int s 32
type t2 bool
site 0 "src/main.stv" 1 4 main.pick
site 1 "src/main.stv" 2 26 main.pick
site 2 "src/main.stv" 2 41 main.pick
site 3 "src/main.stv" 5 4 main.main
site 4 "src/main.stv" 6 13 main.main
site 5 "src/main.stv" 7 13 main.main
site 6 "src/main.stv" 8 5 main.main
main @1
fn @0 main.pick(t2, t0) -> t0 site 0 {
b0(%0: t2, %1: t0):
  cbr %0 b1() b2()
b1():
  %2 = mul t0 %1 3 !1
  ret %2
b2():
  %3 = add t0 %1 1 !2
  ret %3
}
fn @1 main.main() -> t1 site 3 {
b0():
  %0 = call t0 @0 1 5 !4
  %1 = call t0 @0 0 %0 !5
  %2 = call t0 @0 0 7 !6
  %3 = cast.wrap t1 %1
  ret %3
}
```

Decision: the three rows have the key `(3, 0, k)`, `a_j = 0`; all are accepted, `S(main)` = 10.

```
round 1
fn @0 size 3 closed 3 slots 0 level 0
fn @1 size 10 closed 10 slots 0 level 1
site @1 b0 0 @0 3 inline
site @1 b0 1 @0 3 inline
site @1 b0 2 @0 3 inline
```

After `X` (`pick` unchanged):

```
fn @1 main.main() -> t1 site 3 {
b0():
  cbr 1 b1() b2()
b1():
  %0 = mul t0 5 3 !1
  br b3(%0)
b2():
  %1 = add t0 5 1 !2
  br b3(%1)
b3(%2: t0):
  cbr 0 b4() b5()
b4():
  %3 = mul t0 %2 3 !1
  br b6(%3)
b5():
  %4 = add t0 %2 1 !2
  br b6(%4)
b6(%5: t0):
  cbr 0 b7() b8()
b7():
  %6 = mul t0 7 3 !1
  br b9(%6)
b8():
  %7 = add t0 7 1 !2
  br b9(%7)
b9(%8: t0):
  %9 = cast.wrap t1 %5
  ret %9
}
```

Clean-up of `main`, round 1: C1 nothing (`b3`, `b6`, `b9` have two edges); C2 folds `mul 5 3` =
15, `add 5 1` = 6, `mul 7 3` = 21, `add 7 1` = 8; C3 makes `b0` `br b1()`, `b3` `br b5()`, `b6`
`br b8()`; C4 deletes `b2`, `b4`, `b7`; C5 merges `b1` into `b0`, `b5` into `b3`, `b8` into `b6`,
leaving `b0: br b3(15)`, `b3(%2): %4 = add t0 %2 1 !2; br b6(%4)`, `b6(%5): br b9(8)`,
`b9(%8): %9 = cast.wrap t1 %5; ret %9`; C6 deletes `b9`'s param `%8` (unused) and its argument.
Round 2: C1 substitutes 15 for `%2` and `%4` for `%5`; C2 folds `add 15 1` = 16, then
`cast.wrap t1 16` = 16; C5 merges the chain into `b0`. Round 3 changes nothing. Output (`pick`
unchanged):

```
fn @1 main.main() -> t1 site 3 {
b0():
  ret 16
}
```

### 10.3 Slots, nested callees, two sites in one block (X7)

Input:

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 64
type t1 int s 32
type t2 ref mut t0
type t3 ref mut t4
type t4 tuple t0 t0
site 0 "src/main.stv" 1 4 main.h
site 1 "src/main.stv" 6 4 main.g
site 2 "src/main.stv" 8 12 main.g
site 3 "src/main.stv" 11 4 main.main
site 4 "src/main.stv" 13 13 main.main
site 5 "src/main.stv" 14 13 main.main
main @2
fn @0 main.h(t0) -> t0 site 0 {
b0(%0: t0):
  %1 = slot t2
  store () %1 %0
  %2 = load t0 %1
  %3 = add.wrap t0 %2 1
  ret %3
}
fn @1 main.g(t0) -> t0 site 1 {
b0(%0: t0):
  %1 = slot t3
  %2 = make t4 %0 %0
  store () %1 %2
  %3 = addr.field t2 %1 0
  %4 = load t0 %3
  %5 = call t0 @0 %4 !2
  ret %5
}
fn @2 main.main() -> t1 site 3 {
b0():
  %0 = slot t2
  store () %0 3
  %1 = load t0 %0
  %2 = call t0 @1 %1 !4
  %3 = call t0 @0 %2 !5
  %4 = cast.wrap t1 %3
  ret %4
}
```

Decision: `main`'s candidates sort `h` (key `(3, 0, 4)`) before `g` (key `(7, 0, 3)`); slot sums
8 and 32. The `site` lines stay in `(b, k)` order:

```
round 1
fn @0 size 3 closed 3 slots 8 level 0
fn @1 size 7 closed 7 slots 24 level 1
site @1 b0 5 @0 3 inline
fn @2 size 9 closed 13 slots 40 level 2
site @2 b0 3 @1 7 inline
site @2 b0 4 @0 3 inline
```

After `X` (`h` unchanged). In `main`'s `b0`: its own slot, then `F(g)`'s slots in their order (`g`'s
own, then the one `g` took from `h`), then `h`'s for the second row:

```
fn @1 main.g(t0) -> t0 site 1 {
b0(%0: t0):
  %1 = slot t3
  %2 = slot t2
  %3 = make t4 %0 %0
  store () %1 %3
  %4 = addr.field t2 %1 0
  %5 = load t0 %4
  store () %2 %5
  %6 = load t0 %2
  %7 = add.wrap t0 %6 1
  br b1(%7)
b1(%8: t0):
  ret %8
}
fn @2 main.main() -> t1 site 3 {
b0():
  %0 = slot t2
  %1 = slot t3
  %2 = slot t2
  %3 = slot t2
  store () %0 3
  %4 = load t0 %0
  %5 = make t4 %4 %4
  store () %1 %5
  %6 = addr.field t2 %1 0
  %7 = load t0 %6
  store () %2 %7
  %8 = load t0 %2
  %9 = add.wrap t0 %8 1
  br b1(%9)
b1(%10: t0):
  br b2(%10)
b2(%11: t0):
  store () %3 %11
  %12 = load t0 %3
  %13 = add.wrap t0 %12 1
  br b3(%13)
b3(%14: t0):
  %15 = cast.wrap t1 %14
  ret %15
}
```

`main`'s `b1` is `F(g)`'s `b1` (the continuation of `g`'s own call), `b2` the continuation of the
first row and head of the second, `b3` the continuation of the second. Clean-up: C1 substitutes
every single-edge param (`%10`, `%11` by `%9`, `%14` by `%13`, `g`'s `%8` by `%7`), C5 merges each
body into `b0`; `B` changes nothing. Output (`h` unchanged):

```
fn @1 main.g(t0) -> t0 site 1 {
b0(%0: t0):
  %1 = slot t3
  %2 = slot t2
  %3 = make t4 %0 %0
  store () %1 %3
  %4 = addr.field t2 %1 0
  %5 = load t0 %4
  store () %2 %5
  %6 = load t0 %2
  %7 = add.wrap t0 %6 1
  ret %7
}
fn @2 main.main() -> t1 site 3 {
b0():
  %0 = slot t2
  %1 = slot t3
  %2 = slot t2
  %3 = slot t2
  store () %0 3
  %4 = load t0 %0
  %5 = make t4 %4 %4
  store () %1 %5
  %6 = addr.field t2 %1 0
  %7 = load t0 %6
  store () %2 %7
  %8 = load t0 %2
  %9 = add.wrap t0 %8 1
  store () %3 %9
  %10 = load t0 %3
  %11 = add.wrap t0 %10 1
  %12 = cast.wrap t1 %11
  ret %12
}
```

### 10.4 Every callee verdict but `gpu` (decision only)

Input:

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 32
type t1 int u 64
type t2 int s 32
type t3 bool
type t4 ref mut t5
type t5 array t1 9
type t6 fntype t1 -> t1
main @13
fn @0 main.leaf(t1) -> t1 {
b0(%0: t1):
  %1 = add.wrap t1 %0 1
  ret %1
}
fn @1 main.rec(t1) -> t1 {
b0(%0: t1):
  %1 = call t1 @1 %0
  ret %1
}
fn @2 main.ping(t1) -> t1 {
b0(%0: t1):
  %1 = call t1 @3 %0
  ret %1
}
fn @3 main.pong(t1) -> t1 {
b0(%0: t1):
  %1 = call t1 @2 %0
  ret %1
}
fn @4 main.stop(t1) -> t1 {
b0(%0: t1):
  panic.assert ()
  unreachable
}
fn @5 main.ext(t1) -> t1 symbol "ext"
fn @6 main.exp(t1) -> t1 symbol "exp" {
b0(%0: t1):
  ret %0
}
fn @7 runtime.panic.set_user_site(t0) known {
b0(%0: t0):
  ret
}
fn @8 runtime.panic.test_skip() -> t0 known {
b0():
  ret 0
}
fn @9 main.ind(t6) -> t1 {
b0(%0: t6):
  %1 = call.ind t1 %0 5
  ret %1
}
fn @10 main.wide(t1) -> t1 needs(avx2) {
b0(%0: t1):
  %1 = add.wrap t1 %0 2
  ret %1
}
fn @11 main.big(t1) -> t1 {
b0(%0: t1):
  %1 = call.ind t1 @0 %0
  %2 = call.ind t1 @0 %1
  %3 = call.ind t1 @0 %2
  %4 = call.ind t1 @0 %3
  %5 = call.ind t1 @0 %4
  %6 = call.ind t1 @0 %5
  %7 = call.ind t1 @0 %6
  %8 = call.ind t1 @0 %7
  %9 = call.ind t1 @0 %8
  %10 = call.ind t1 @0 %9
  %11 = call.ind t1 @0 %10
  ret %11
}
fn @12 main.fat(t1) -> t1 {
b0(%0: t1):
  %1 = slot t4
  ret %0
}
fn @13 main.main() -> t2 {
b0():
  call () @7 3
  %0 = call t1 @0 1
  %1 = call t1 @1 %0
  %2 = call t1 @2 %1
  %3 = call t1 @5 %2
  %4 = call t1 @6 %3
  %5 = call t0 @8
  %6 = call t1 @9 @0
  %7 = call t1 @10 %4
  %8 = call t1 @11 %7
  %9 = call t1 @12 %8
  %10 = call t1 @14
  %11 = eq t3 %9 0
  cbr %11 b1() b2()
b1():
  %12 = call t1 @4 %9
  %13 = call t1 @0 %12
  panic.assert ()
  unreachable
b2():
  %14 = cast.wrap t2 %9
  ret %14
}
fn @14 main.vec() -> t1 needs(avx2) {
b0():
  %0 = call t1 @10 7
  ret %0
}
```

`big`'s `call.ind` operand is the constant `@0`, so it passes `indirect` and fails `size` (11 x
3 = 33); `fat`'s slot holds 72 bytes; `vec` may take `wide` (its `needs` covers `avx2`), `main`
may not; `b1` of `main` reaches no `ret`. `main`: `size0` = 12 x 3 + 1 + 1 (b0) + 2 x 3 + 1 (b1)
+ 1 (b2) = 46; accepted `set_user_site` (key `(0, 0, 0)`) and `leaf` (`(1, 0, 1)`); `S` = 46 +
(0 - 3) + (1 - 3) = 41. `vec`: `S` = 3 + (1 - 3) = 1.

```
round 1
fn @0 size 1 closed 1 slots 0 level 0
fn @1 size 3 closed 3 slots 0 level 0 cyclic
site @1 b0 0 @1 3 cyclic
fn @2 size 3 closed 3 slots 0 level 0 cyclic
site @2 b0 0 @3 3 cyclic
fn @3 size 3 closed 3 slots 0 level 0 cyclic
site @3 b0 0 @2 3 cyclic
fn @4 size 1 closed 1 slots 0 level 0
fn @6 size 0 closed 0 slots 0 level 0
fn @7 size 0 closed 0 slots 0 level 0
fn @8 size 0 closed 0 slots 0 level 0
fn @9 size 3 closed 3 slots 0 level 0
fn @10 size 1 closed 1 slots 0 level 0
fn @11 size 33 closed 33 slots 0 level 0
fn @12 size 0 closed 0 slots 72 level 0
fn @13 size 46 closed 41 slots 0 level 2
site @13 b0 0 @7 0 inline
site @13 b0 1 @0 1 inline
site @13 b0 2 @1 3 cyclic
site @13 b0 3 @2 3 cyclic
site @13 b0 4 @5 - nobody
site @13 b0 5 @6 0 symbol
site @13 b0 6 @8 0 known
site @13 b0 7 @9 3 indirect
site @13 b0 8 @10 1 features
site @13 b0 9 @11 33 size
site @13 b0 10 @12 0 slots
site @13 b0 11 @14 1 features
site @13 b1 0 @4 1 noret
site @13 b1 1 @0 1 cold
fn @14 size 3 closed 1 slots 0 level 1
site @14 b0 0 @10 1 inline
```

### 10.5 The budget cut on slots and its order (decision only)

Input:

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 64
type t1 int s 32
type t2 ref mut t3
type t3 array t0 8
main @2
fn @0 main.a(t0) -> t0 {
b0(%0: t0):
  %1 = slot t2
  ret %0
}
fn @1 main.b(t0) -> t0 {
b0(%0: t0):
  %1 = slot t2
  %2 = add.wrap t0 %0 1
  ret %2
}
fn @2 main.main() -> t1 {
b0():
  %0 = call t0 @1 1
  %1 = call t0 @1 %0
  %2 = call t0 @1 %1
  %3 = call t0 @1 %2
  %4 = call t0 @1 %3
  %5 = call t0 @1 %4
  %6 = call t0 @1 %5
  %7 = call t0 @1 %6
  %8 = call t0 @0 %7
  %9 = call t0 @0 %8
  %10 = call t0 @0 %9
  %11 = call t0 @0 %10
  %12 = call t0 @0 %11
  %13 = call t0 @0 %12
  %14 = call t0 @0 %13
  %15 = call t0 @0 %14
  %16 = call t0 @0 %15
  %17 = cast.wrap t1 %16
  ret %17
}
```

Each callee holds 64 slot bytes, so 16 of the 17 rows fit `SLOT_CALLER`. The order (D3) puts the
nine calls of `a` (`S` = 0) first, then the eight of `b` (`S` = 1), each group by `k`; the
seventeenth is `b`'s last call, `k` = 7, not the last call of the block. `S(main)` = 52 + 9 x (0 -
3) + 7 x (1 - 3) = 11:

```
round 1
fn @0 size 0 closed 0 slots 64 level 0
fn @1 size 1 closed 1 slots 64 level 0
fn @2 size 52 closed 11 slots 1024 level 1
site @2 b0 0 @1 1 inline
site @2 b0 1 @1 1 inline
site @2 b0 2 @1 1 inline
site @2 b0 3 @1 1 inline
site @2 b0 4 @1 1 inline
site @2 b0 5 @1 1 inline
site @2 b0 6 @1 1 inline
site @2 b0 7 @1 1 budget
site @2 b0 8 @0 0 inline
site @2 b0 9 @0 0 inline
site @2 b0 10 @0 0 inline
site @2 b0 11 @0 0 inline
site @2 b0 12 @0 0 inline
site @2 b0 13 @0 0 inline
site @2 b0 14 @0 0 inline
site @2 b0 15 @0 0 inline
site @2 b0 16 @0 0 inline
```

### 10.6 A caller with `altstack` (decision only)

The verifier allows `altstack` only on a panic report entry, here `runtime.panic.abort`. Its call
of `leaf` fails `altstack` (the caller test comes before `cold`, which it would also fail):

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 8
type t1 int u 64
type t2 int s 32
type t3 ref t4
type t4 slice t0
fn @0 main.leaf(t1) -> t1 {
b0(%0: t1):
  %1 = add.wrap t1 %0 1
  ret %1
}
fn @1 main.main() -> t2 {
b0():
  %0 = call t1 @0 1
  %1 = cast.wrap t2 %0
  ret %1
}
fn @2 runtime.panic.abort(t3) known altstack {
b0(%0: t3):
  %1 = call t1 @0 7
  panic.assert ()
  unreachable
}
```

```
round 1
fn @0 size 1 closed 1 slots 0 level 0
fn @1 size 4 closed 2 slots 0 level 1
site @1 b0 0 @0 1 inline
fn @2 size 4 closed 4 slots 0 level 1
site @2 b0 0 @0 1 altstack
```

### 10.7 `switch` on a constant, `select`, equal targets, a fold that would trap, dead code

Input:

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 8
type t1 int u 64
type t2 int u size
type t3 int s 32
type t4 bool
type t5 str
type t6 ref t5
str s0 "hi"
fn @0 main.sel(t1) -> t1 {
b0(%0: t1):
  switch %0 0:b1() 1:b2() _:b3()
b1():
  ret 10
b2():
  ret 20
b3():
  ret 30
}
fn @1 main.note(t6) {
b0(%0: t6):
  %1 = slice.len t2 %0
  ret
}
fn @2 main.ovf(t0) -> t0 {
b0(%0: t0):
  %1 = add t0 %0 9
  ret %1
}
fn @3 main.main() -> t3 {
b0():
  %0 = call t1 @0 1
  call () @1 s0
  %1 = call t0 @2 250
  %2 = eq t4 %1 0
  %3 = select t1 1 %0 7
  cbr %2 b1(%3) b1(%3)
b1(%4: t1):
  %5 = cast.wrap t3 %4
  ret %5
}
```

Decision: `size0`: `sel` 2 (`switch`), `note` 1, `ovf` 1, `main` 3 x 3 + 1 + 1 + 1 + 1 = 13.
The candidates sort `note` `(1, 0, 1)`, `ovf` `(1, 0, 2)`, `sel` `(2, 0, 0)`, every `a_j` = 0;
`S(main)` = 13 + (2 - 3) + (1 - 3) + (1 - 3) = 8.

```
round 1
fn @0 size 2 closed 2 slots 0 level 0
fn @1 size 1 closed 1 slots 0 level 0
fn @2 size 1 closed 1 slots 0 level 0
fn @3 size 13 closed 8 slots 0 level 1
site @3 b0 0 @0 2 inline
site @3 b0 1 @1 1 inline
site @3 b0 2 @2 1 inline
```

After `X` (other fns unchanged). The string argument and the constants reach the callee code;
`switch 1` is legal here only because the level is `opt` (P4); the call of `note` has no result,
so its continuation `b5` has no param:

```
fn @3 main.main() -> t3 {
b0():
  switch 1 0:b1() 1:b2() _:b3()
b1():
  br b4(10)
b2():
  br b4(20)
b3():
  br b4(30)
b4(%0: t1):
  %1 = slice.len t2 s0
  br b5()
b5():
  %2 = add t0 250 9
  br b6(%2)
b6(%3: t0):
  %4 = eq t4 %3 0
  %5 = select t1 1 %0 7
  cbr %4 b7(%5) b7(%5)
b7(%6: t1):
  %7 = cast.wrap t3 %6
  ret %7
}
```

Clean-up of `main`, round 1: C1 replaces `%3` by `%2` (`b6` has one edge); C2 leaves `add t0 250 9`
(259 is outside `u8`: it would trap) and replaces the `select` by `%0`; C3 makes `b0` `br b2()`
(case 1) and the `cbr` with two equal targets `br b7(%0)`; C4 deletes `b1` and `b3`; C5 merges
`b2` into `b0` and `b5`, `b6` into `b4`; C6 deletes `slice.len` and `eq` (removable, unused) and
keeps the `add` (not removable). Round 2: C1 replaces `%0` and `%6` by 20, C2 folds
`cast.wrap t3 20` to 20, C5 merges every block into `b0`. Round 3 changes nothing. `note`'s
`slice.len` goes by C6 too, so `usize` and `bool` lose their last use and `canon` drops them.
`B` changes nothing. Output (`--dump-phase=clean` prints the same text):

```
stave.ir 4 "x86_64-linux-v2" release opt
type t0 int u 8
type t1 int u 64
type t2 int s 32
type t3 str
type t4 ref t3
str s0 "hi"
fn @0 main.sel(t1) -> t1 {
b0(%0: t1):
  switch %0 0:b1() 1:b2() _:b3()
b1():
  ret 10
b2():
  ret 20
b3():
  ret 30
}
fn @1 main.note(t4) {
b0(%0: t4):
  ret
}
fn @2 main.ovf(t0) -> t0 {
b0(%0: t0):
  %1 = add t0 %0 9
  ret %1
}
fn @3 main.main() -> t2 {
b0():
  %0 = add t0 250 9
  ret 20
}
```

### 10.8 The base passes B1-B7 (`--passes=base`)

Input:

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 64
type t1 int u size
type t2 bool
type t3 ref t0
type t4 ref t6
type t5 ref mut t0
type t6 slice t0
fn @0 main.sum(t4, t4) -> t0 {
b0(%0: t4, %1: t4):
  %2 = slot t5
  %3 = slice.len t1 %1
  %4 = slice.len t1 %0
  %5 = sub.sat t1 %4 3
  %6 = sub.sat t1 %4 3
  store () %2 0
  br b1(0)
b1(%7: t1):
  %8 = lt t2 %7 %4
  cbr %8 b2() b3()
b2():
  %9 = slice.len t1 %0
  %10 = addr.elem t3 %0 %7
  %11 = load t0 %10
  %12 = load t0 %2
  %13 = add.wrap t0 %12 %11
  store () %2 %13
  %14 = add.wrap t1 %7 1
  %15 = lt t2 %14 %9
  %16 = add.wrap t1 %7 1
  br b1(%16)
b3():
  %17 = add.wrap t1 %5 %6
  %18 = cast.wrap t0 %17
  %19 = load t0 %2
  %20 = add.wrap t0 %19 %18
  ret %20
}
```

Loop depth (B7): the edge `b2 -> b1` is a back edge, so `b1` and `b2` have depth 1. B1: group `%0`
has two parts (`%4`, `%9`) and moves: `%4` goes before the first non-slot instruction of `b0`,
`%9` is deleted and reads `%4`; group `%1` has one part at depth 0 and stays. B2: `%10`'s slice
operand gets `slice.ptr` of the new type `ptr mut t0` before it. B2': its one part is at depth 1,
so it moves to the same place as `%4` and goes before it (an earlier step's insertion). B3: `%5`,
`%6` form one group and move to directly after `%4`; `%6` reads `%5`. B4: in `b2`, `%16` has the
key of `%14`, only branch arguments read it, and the search from `b2` pops `b1`, `b3`, then `b2`:
it reads `%14`. B5 deletes `%15` and `%3`. Output, whole:

```
stave.ir 4 "x86_64-linux-v2" release opt
type t0 int u 64
type t1 int u size
type t2 bool
type t3 ref t0
type t4 ref t7
type t5 ref mut t0
type t6 ptr mut t0
type t7 slice t0
fn @0 main.sum(t4, t4) -> t0 {
b0(%0: t4, %1: t4):
  %2 = slot t5
  %3 = slice.ptr t6 %0
  %4 = slice.len t1 %0
  %5 = sub.sat t1 %4 3
  store () %2 0
  br b1(0)
b1(%6: t1):
  %7 = lt t2 %6 %4
  cbr %7 b2() b3()
b2():
  %8 = addr.elem t3 %3 %6
  %9 = load t0 %8
  %10 = load t0 %2
  %11 = add.wrap t0 %10 %9
  store () %2 %11
  %12 = add.wrap t1 %6 1
  br b1(%12)
b3():
  %13 = add.wrap t1 %5 %5
  %14 = cast.wrap t0 %13
  %15 = load t0 %2
  %16 = add.wrap t0 %15 %14
  ret %16
}
```

### 10.9 Two slice params at one place (`--passes=base`)

Both groups move to the start of `b0`, after the slot. They are ordered by `s`: the group of `%0`
comes first although the first part in print order reads `%1`.

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u size
type t1 ref t3
type t2 ref mut t0
type t3 slice t0
fn @0 main.f(t1, t1) -> t0 {
b0(%0: t1, %1: t1):
  %2 = slot t2
  %3 = slice.len t0 %1
  %4 = slice.len t0 %0
  %5 = slice.len t0 %1
  %6 = slice.len t0 %0
  %7 = add.wrap t0 %3 %4
  %8 = add.wrap t0 %5 %6
  %9 = add.wrap t0 %7 %8
  store () %2 %9
  %10 = load t0 %2
  ret %10
}
```

Output (`B4` keeps both `add.wrap`: an instruction, not a branch argument, reads each):

```
fn @0 main.f(t1, t1) -> t0 {
b0(%0: t1, %1: t1):
  %2 = slot t2
  %3 = slice.len t0 %0
  %4 = slice.len t0 %1
  %5 = add.wrap t0 %4 %3
  %6 = add.wrap t0 %4 %3
  %7 = add.wrap t0 %5 %6
  store () %2 %7
  %8 = load t0 %2
  ret %8
}
```

### 10.10 The budget cut on `CAP` (decision only, `--limit cap=20`)

With `CAP` = 20, `main`'s budget is `20 - size0(main)` = `20 - 10` = 10. The candidates sort `h8`
`(8, 0, 1)`, `h8` `(8, 0, 2)`, `h9` `(9, 0, 0)` with `a_j` = 5, 5, 6; the prefix sums 5 and 10
fit, 16 does not. With the table's `CAP` all three are inlined and `S(main)` is 26.

```
stave.ir 4 "x86_64-linux-v2" release checked
type t0 int u 64
type t1 int s 32
main @2
fn @0 main.h8(t0) -> t0 {
b0(%0: t0):
  %1 = add.wrap t0 %0 1
  %2 = add.wrap t0 %1 1
  %3 = add.wrap t0 %2 1
  %4 = add.wrap t0 %3 1
  %5 = add.wrap t0 %4 1
  %6 = add.wrap t0 %5 1
  %7 = add.wrap t0 %6 1
  %8 = add.wrap t0 %7 1
  ret %8
}
fn @1 main.h9(t0) -> t0 {
b0(%0: t0):
  %1 = add.wrap t0 %0 1
  %2 = add.wrap t0 %1 1
  %3 = add.wrap t0 %2 1
  %4 = add.wrap t0 %3 1
  %5 = add.wrap t0 %4 1
  %6 = add.wrap t0 %5 1
  %7 = add.wrap t0 %6 1
  %8 = add.wrap t0 %7 1
  %9 = add.wrap t0 %8 1
  ret %9
}
fn @2 main.main() -> t1 {
b0():
  %0 = call t0 @1 1
  %1 = call t0 @0 %0
  %2 = call t0 @0 %1
  %3 = cast.wrap t1 %2
  ret %3
}
```

```
round 1
fn @0 size 8 closed 8 slots 0 level 0
fn @1 size 9 closed 9 slots 0 level 0
fn @2 size 10 closed 20 slots 0 level 1
site @2 b0 0 @1 9 budget
site @2 b0 1 @0 8 inline
site @2 b0 2 @0 8 inline
```
