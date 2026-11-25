# Contract: canonical checked IR and its text (v0.4, `--dump=lir|ir|opt`, `stave ir`)

The checked IR is the SSA IR as built by lowering: after type check, borrow check, and
monomorphization. Values carry Stave types (canonical type ids). The back ends lay out the
checked IR themselves; no other IR lies between it and machine code. Both compilers print it as
one text and read that text back: its syntax is [ir-text.ebnf](ir-text.ebnf) (the only place
for syntax), its meaning is this file (the only place for semantics). The stage1 gate compares
the printed bytes (stage-gates `lir`, `ir`, `opt`). A dump is printed only for a build with no
error diagnostics; with errors, its file is not written.

## Levels and forms

| Level | The module after | Dump |
|---|---|---|
| `lowered` | lowering and the IR verifier, before the removal of proven checks ("Bounds checks"), which reads this canonical module | `lir` |
| `checked` | the removal of the checks that have a witness, which keeps the module canonical, then the IR verifier: the input of the interpreter and of both back ends | `ir` |
| `opt` | the release passes ([opt-passes.md](opt-passes.md)), which keep the module canonical, then the IR verifier on their output; release builds only | `opt` |

Each level is printed from the canonical module of that point ("Canonical module"), so ids of
one level do not correspond to those of another. stage0's own release build also runs the bounds
audit on the output of its base passes, as a self-check of stage0; that audit is not part of level
`opt`, and loading an `opt` text does not run it. It checks that every element access the base
passes leave, those rule B2 (opt-passes.md) puts behind `slice.ptr` included, still has a
`check.bounds` or a proof the audit accepts ("Loading" step 5).

The **gate form** is what the dumps hold. The **emit form** (`stave ir emit`) is the gate form
plus the stack section: the `runstack`, `truncated`, `node`, `edge` and `nodes` lines. Rule: the
gate form is exactly the emit form with the stack-section lines filtered out, byte for byte; no
other line differs between the two forms. The stack section is not in a gate: its node numbers are the front
end's call-graph order, which no contract fixes.

## Text

- ASCII, one `\n` after every line, no blank lines, no comments, no trailing spaces. Lines of
  instructions and terminators begin with two spaces, every other line in column 1.
- Tokens are separated by one space, except where the spellings below join them.
- A path (of a decl, fn, static, or site) prints bare when it has the `Word` shape of
  ir-text.ebnf, else as a string. A string prints between double quotes with the `strings` dump
  escape rule (stage-gates.md), so every byte of the text is printable ASCII.
- The reader also accepts runs of spaces and tabs between tokens and at either end of a line,
  `#` comments, and blank lines; `stave ir fmt` prints the canonical text.

Spellings, `<x>` a field, `[..]` present only with its datum, `{..}` repeated zero or more times:

| Line | Spelling |
|---|---|
| header | `stave.ir 4 "<target>" <mode> <level>` |
| decl | `decl d<D> struct\|enum <path>` |
| type | `type t<T> <key>` (TypeKey text, below) |
| shape | `shape t<T> struct{ t<F>}` or `shape t<T> enum{ \|{ t<F>}}` (`shape t7 enum \| \| t3`) |
| string | `str s<S> "<bytes>"` |
| site | `site <row> "<file>" <line> <col> <fn path>` |
| static | `static <i> t<T> <path>` |
| static bytes | `data <i> <offset> x<hex>` (two lowercase hex digits per byte) |
| address word | `ref <i> <offset> s<S> <addend>` or `ref <i> <offset> static <j> <addend>` |
| entry | `main @<inst>` |
| cell | `cell d<D>` |
| run stack | `runstack <bytes>` |
| incomplete graph | `truncated` |
| call-graph node | `node <n>[ rec]` |
| call-graph edge | `edge <from> <to> call\|indirect\|spawn` |
| fn nodes | `nodes @<inst> <n>{ <n>}` |
| fn header | `fn @<inst> <path>[<<arg>{,<arg>}>](<t>{, <t>})[ -> t<R>][ needs(<f>{, <f>})][ symbol "<sym>"][ variadic][ known][ altstack][ unbuilt][ site <row>][ {]` |
| block header | `b<N>(<param>{, <param>}):`, a param `%<v>: t<T>` |
| instruction | `  [%<v> = ]<op> t<T>\|()[ <operand>]..[ !<row>]` |
| terminator | `  br <target>`, `  cbr <o> <target> <target>`, `  switch <o>{ <int>:<target>} _:<target>`, `  ret[ <o>]`, `  unreachable`; a target `b<N>(<o>{, <o>})` |
| end of body | `}` |
| claim | `claim @<inst> pre[ checked] <a> <= <b> <w>` or `claim @<inst> post <a> <= <b> <w>` |

Operands: `%<v>` a value; a decimal int constant (`bool` as `0`/`1`, `char` as its code point;
the operand position gives its type; every decimal integer of the text, here, in a `switch`
case, or a `claim` weight, lies in the range of `i128`, -2^127 to 2^127-1, and one outside it
is a step 1 error); a float constant as `0x` and its IEEE bits in lowercase hex,
4 digits for f16, 8 for f32, 16 for f64, and `bf16 0x` with 4 digits for bf16 (the digit count
and the prefix give the format); `s<S>` the `&str` of string S; `@<inst>` a fn item (a direct
callee, or a fn value with a null environment); `t<T>` a type operand. Constants are inline
operands, never values.

```
stave.ir 4 "x86_64-linux-v2" debug checked
type t0 int s 32
type t1 bool
site 0 "src/app.stv" 1 4 app.max
fn @0 app.max(t0, t0) -> t0 site 0 {
b0(%0: t0, %1: t0):
  %2 = gt t1 %0 %1
  cbr %2 b1() b2()
b1():
  ret %0
b2():
  ret %1
}
```

## Sections

A file is these sections in this order; each kind of line has one section.

1. The header, once, first.
2. `decl` lines, by ascending D.
3. `type` lines, `t0` up, dense.
4. `shape` lines, by ascending T.
5. `str` lines, `s0` up, dense.
6. `site` lines, row 0 up, dense.
7. Per static, `0` up, dense: its `static` line, its `data` lines by ascending offset, its `ref`
   lines by ascending offset.
8. `main`, then `cell`, each at most once.
9. The stack section (emit form only), in this order: `runstack` and `truncated`, each at most
   once; `node` lines, `0` up, dense; `edge` lines by ascending (from, to); `nodes` lines by
   instance id.
10. Per fn, by instance id ("Ids"): its header; with a body, the body's blocks (each a block
    header, its instructions, one terminator) and `}`; then its `claim` lines, the `pre` rows
    then the `post` rows, each in the summary's order.

**Header.** `4` is this contract's version (v0.4); a reader refuses any other. `<target>` is the
canonical name of the target the lowering ran for (`x86_64-linux-v2`, `aarch64-linux+sve`, ...;
`x86_64-linux` prints as `x86_64-linux-v2`): lowering depends on it (`unbuilt`, dispatch arms,
the `cpu_baseline` constant, the root set). `<mode>` is `debug` or `release` (the root set, and
whether the release passes run). `<level>`: "Levels and forms"; `opt` only with `release`.

**decl.** One line per decl a `d<D>` of the file names (`adt` keys, `cell`): its kind and its
decl path. The kind of a decl equals that of the shapes of its `adt` types.

**type.** Every type of the module, by id; no two lines have equal keys. The key's components
name `type` lines and never form a cycle (a recursive ADT recurses only through its shape).
The module is monomorphized: no type is a generic parameter, which has no key text. The IR
verifier refuses a module with one, or with an `adt` key or `cell` whose decl the module does not
record, a recorded decl nothing names, or a shape of another kind than its decl; the printer
refuses every module whose text a reader would refuse in steps 1 and 2 of "Loading" (an id that
names no line, one of these keys), so it prints no line that cannot load.

**shape.** The monomorphized shape of an `adt` type: `struct` and its field types in field
order, or `enum` and one `|` group per variant in variant index order, each holding the
variant's field types in order. A type without a shape line has no shape: an aggregate op on it,
or its layout, is a verify error.

**str.** The bytes of string constant S. They are read-only program data.

**site.** Row `<row>` of the site table (intrinsics.md "site table", same rows in the same
order): the file path diagnostics use, the 1-based line, the 1-based column in bytes, and the
checked-IR path of the enclosing fn. `!<row>` and the header's `site <row>` name rows.

**static, data, ref.** Static `<i>` (the index `intrinsic.static_addr <i>` takes, below) has type T
and its decl path. Its initial bytes are as many as T's layout size, zero except where `data` lines
say otherwise. A `data` line holds the bytes at `<offset>`. Canonically the bytes are cut into
64-byte blocks at offsets 0, 64, 128, ...; each block with a nonzero byte prints one `data` line,
from its first to its last nonzero byte. A `ref` line is an address word: the 8 bytes at `<offset>`
(zero in the initial bytes) hold, in the image, the address of the first byte of string `s<S>` or of
static `<j>`, plus `<addend>`, at most the target's size. The address-word rules are those of "Ids"
`intrinsic.static_addr`.

**main.** The program's `fn main` (the fn whose path is the root module's `main` item path,
the first in instance order), which `__stave_main` calls (intrinsics.md). Absent for a library.

**cell.** The decl of `std.cell.UnsafeCell` when the program has one: the bytes of such a value
are not read-only behind a `&`. Absent: no field behind a `&` parameter is read as one value
("Bounds checks").

**Stack section.** What the stack bound (cli.md, intrinsics.md "stack table") reads besides the
module and the frame sizes. `runstack`: the root package's `[run] stack` in bytes (absent: not
set). `truncated`: the call graph is incomplete (an instantiation chain too deep, too many
instances), so no stack has a bound. `node <n>`:
call-graph node n; `rec`: the node of a `rec fn`. `edge`: the call-graph edge from node `<from>`
to node `<to>` (at most one per pair): `spawn` a `spawn` edge, `indirect` a call through a fn
value, `call` every other kind (a call, a dispatch arm, a member, a drop). `nodes @<inst>`: the
nodes fn `@<inst>` stands for, ascending; a fn without the line stands for none. A module loaded without the
section builds with an empty stack table: every stack gets the `[run] stack` default. The
release passes keep the nodes and edges, and never give a fn a `call.ind` through a value that
its `checked` body lacks: the bound stays sound when a caller holds a callee's code.

**fn header.** Every fn of the module, with or without a body.

- `@<inst>`: "Ids". `<path>`: the module path and item name joined by `.`; a method is
  `<head>.<name>`, `<head>` = the impl's self-type ADT path (`main.Counter.bump`), or the impl's
  module path plus a built-in type name (`std.core.str.fmt`); drop glue is `drop_glue` with its
  type as the one generic argument (`fn @200 drop_glue<t135>(t140) {`); closures: "Closures".
- `<..>`: the instance's generic arguments, `t<T>` or a decimal const argument.
- `(..)`: the param types. `-> t<R>`: the return type; absent: the fn returns no value (`ret`
  without an operand). A fn that returns a value of the unit type prints `-> t<unit>`.
- `needs(..)`: the fn's own features ("Types and fntype").
- `symbol "<sym>"`: the link symbol of an `extern "C"` import or export, unmangled.
- `variadic`: an `extern "C"` import declared with `...` (`call`, below).
- `known`: the fn is an instance of the compiler-known item (stave-ir `Known`; intrinsics.md
  "Paths fixed by the compiler") whose path is the fn's path. Stages find known fns by this
  mark, never by path; a fn without it is never known, whatever its path.
- `altstack`: the fn runs on its thread's alternate signal stack (intrinsics.md "Alternate
  stack").
- `unbuilt`: the target does not build the fn (its features belong to another arch); its body is
  a trap, or absent where the target cannot pass its params or return value.
- `site <row>`: the site of the fn's name token (the PC-table row).
- `{`: the fn has a body; without it the fn is declared only (an `extern "C"` import, an
  `unbuilt` fn without a body).

**Body.** `b0` is the entry; its params are the fn's params, with the header's types. Op names
are the stave-ir `Op` names (scalar ops, one per simd-ops portable op, `intrinsic.<name>`); the
sections "Ops" below give their operands and meaning. Trapping operations (overflow, bounds,
division, `tail exact`) are single instructions and create no blocks. An instruction defines
its result value exactly when it prints a result type; an instruction with no result prints
`()` there (`store () %3 %1`). `!<row>`: the instruction's site, absent when it has none;
lowering gives one to every trapping instruction, every call, and every `store` of a user
assignment. Terminators have no site. At level `opt` an instruction's `!<row>` may name a row
whose fn path is not the fn's: inlined code keeps the rows of its source.

**claim.** The summary the witness checker certified for the fn ("Contracts across calls"),
rows only: `<a> <= <b> + <w>` over the terms `zero`, `p <i>` (int param i), `lenp <i>` (the
length of slice param i), `result` (the int result), `lenresult` (the length of the slice
result). `pre` rows are what a call proves of its arguments; `checked` marks a `pre` row that
reads a `requires` check of the fn, which is also a fact about the arguments after a call.
`post` rows hold at every reachable `ret`. The proofs are not printed. A fn that states nothing
has no `claim` line; at level `lowered` no fn has one. The release passes do not change `claim`
lines.

## Ids

The ids are local to the printed module, never the `types` dump's or another stage's numbering,
except `d<D>`.

- `t<T>`: dense from `t0` over every type of the module. The *P* set comes first: the closure over
  key components of the types the fns with a body use (their signatures and type arguments, block
  param and value types, instruction result types, type operands), ranked in the canonical
  type-key order. The *S* set follows: the other types that the shapes of P and S types, the
  statics, and the bodiless fns' signatures reach over key components and shapes, ranked the
  same way. A module holds no type outside P and S. An op rule that names a scalar type (`bool`
  for a condition, `int u size` for a length) checks a constant operand by that key, so the
  table need not hold a type that no value has. Instance types such as `Pair<u16>` exist only here. `d<D>` in `adt` keys is the canonical decl
  id D.
- `@<inst>`: the rank of the instance among the build's instances (the root set's closure,
  below), ordered by (decl D, generic args by `t<T>`), drop glue after every decl instance, ordered
  by its type; closures `@<enclosing>.<k>`; the unchecked entry of `@<n>` (Contracts, below) is
  `@<n>.u`, ordered after `@<n>` and its closures, with the path of `@<n>`. Bodiless instances
  (`extern "C"` imports) have ids like every other. Ids differ between debug and `--release`
  (different root sets).
- `s<S>`: the rank of the string constant among the module's distinct string constants by
  (xxHash32 seed 0 over the bytes, bytes), like interned strings; literals are not interned
  strings, so these ids are not the `strings` dump's S.
- Site rows: canonical order (fn instance id, then source offset), intrinsics.md "site table".
- `intrinsic.static_addr <i>` (contracts/intrinsics.md): `i` is the rank of the decl by D among
  the program's `static` and `static mut` decls (one sequence over both forms).
  Every static occupies at least one byte in the image (a zero-size static gets one zero
  byte, aligned as its type), so distinct statics have distinct addresses on every target.
  A static's initial bytes may hold address words: an 8-byte word at an 8-aligned offset is the
  address of a string constant's bytes or of another static, plus an addend of at most that
  target's size. A `&str` in a static initializer is such a word (to the literal's bytes) then
  the length. The image holds the resolved addresses (a static binary: link-time constants, no
  dynamic relocation); the interpreter writes them before the program starts. A load of such a
  word is a pointer to read-only program data, like a string operand.
- `%<v>` and `b<N>`: per fn, dense in print order ("Order").
- `intrinsic.copy_bytes () %dst %src %n` (contracts/intrinsics.md "Untyped copy"): operands
  `*mut u8`, `*const u8`, `usize`, no result. A copy of `n` bytes with their initialization,
  not a `load` or `store` of any type: the IR's one untyped memory access.

## Canonical module

At every printed point the driver canonicalizes the module before it prints, builds, or runs
it: fns in instance-id order (a fn's index is its rank), unreachable blocks deleted, blocks and
values numbered densely in print order, each body's constant pool holding one entry per
distinct constant, numbered by first use in print order, the module's intrinsic names listed once
each, sorted by bytes, and the type table holding exactly the P and S sets of that point ("Ids"),
renumbered (types that removed checks or the release passes left unreached are dropped; types
the release passes add are ranked like the others), with the decls its `adt` keys and `cell`
name. The interpreter and the back ends get this module. Then `read(print(m))`
equals `m`, and `print(read(t))` equals `t` for every printed `t`.

- Two constants are the same constant when they have the same kind (`int`, `f16`, `bf16`, `f32`,
  `f64`) and the same bits: an int by its value (the operand position gives its type, so `0` is
  one entry wherever it appears); a float by its IEEE bits, so `0.0` and `-0.0` are two
  entries, NaNs with different payloads are two, and `0` (int) and `0x00000000` (f32) are two,
  as are `f16` and `bf16` constants of equal bits. The pool is per body, numbered over the
  blocks in order: a block's instruction operands, then its terminator's (the `cbr` condition,
  `switch` scrutinee, or `ret` value first, then the operands of each target as printed).
- The bounds audit calls two operands the same when they are the same value or the same pool
  entry; with this pool that is the same constant. A pool with one entry per operand would make
  the guard `check.bounds () 0 %n` and the access `addr.elem t %s 0` use different constants.
- The text has no pool and no intrinsic table: a reader builds both from the operands and the
  `intrinsic.<name>` ops it reads, so a loaded module is canonical in both. The table holds the
  names the bodies use, not the names only removed code used.

The text leaves out what no consumer of a lowered module reads: the borrow side table (only the
interpreter's tracked lowering has it, which is never printed), the witnesses of removed checks,
and the proofs inside summaries (only their rows are printed). Two modules are equal when what
their text carries is equal: every section row by row, and of each fn's summary its `claim`
rows (a `pre` row with its `checked` mark), so a summary without rows equals none. The parts the
text leaves out are not part of equality.

## Loading

A reader (`stave ir`, each compiler's IR library) loads a file in this order; a failure at any
step stops before the next.

1. **Lines.** Each line is lexed and parsed with ir-text.ebnf. An error prints
   `FILE:LINE:COL: error: MSG` on stderr, one per bad line, at most 100.
2. **Structure.** One sequential pass checks "Sections" and the rules below. Errors print as in
   step 1, one per fault (a line with two undefined values has two), at most 100 in all:
   - ids of a dense section run from 0 without gaps; ascending sections have no repeats;
   - every `t<T>`, `s<S>`, site row, static index, `d<D>`, node number, and `@<inst>` names an
     existing line; at most one `nodes` line per fn; each `@<inst>` heads one fn; fn headers may come in any order (they are
     sorted by instance id);
   - a body begins with block `b0`; block labels and values are unique within their fn, with any
     numbers (they are renumbered in text order: blocks by their headers, values by their
     definitions); every value and block a fn names is its own; a block is its header, its
     instructions, and one terminator, and `}` follows the last block's terminator; every block
     is reachable from `b0`;
   - `claim` lines follow the body of the fn they name; none at level `lowered`; a fn without a
     body has none;
   - a `known` mark names a known item by the fn's path; `data` bytes do not overlap; `opt` only
     with `release`.
3. **Verification.** The IR verifier on the type table, shapes and statics (each static's `data`
   bytes lie inside its type's size), then on every fn with a body (fns in parallel). An error
   prints `FILE:LINE: verify: MSG`, at the line of the instruction, terminator, block, or fn it
   is about: a block error at the block's terminator, except the entry block's params, at its
   header (a module error: the line of the `type` or `static` it names, else line 1). The
   verifier requires every `intrinsic.<name>` to be an intrinsic of intrinsics.md with its
   operand and result types, every `p <i>` and `lenp <i>` of a fn's `claim` rows to name a param
   of the fn (`i` below its param count; an error at the fn's line), the `slot` instructions of
   a body to be a prefix of `b0` (every level), a `switch` scrutinee to be a value except at
   level `opt`, where it may also be an int constant (its case values are then checked only for
   duplicates; the release passes write one only between inlining and the branch folding that
   removes it, opt-passes.md P4), and a `known` fn to have its item's signature; against the
   header, every intrinsic must be built for the target's arch (intrinsics.md, the rule E0704
   applies to a declaration).
4. **Checks.** At level `lowered`: the removal of proven checks, as after lowering ("Bounds
   checks"); it computes the summaries. A contract check that fails whenever reached, and a call
   that fails a `requires` whenever reached, are reported as in step 3 where a build reports them
   as errors (E0215, E0217, E0218: not in an instance of a generic fn).
5. **Audit.** At levels `lowered` and `checked`: the bounds audit of every fn, reading calls by
   the loaded `claim` rows (`checked`) or the computed ones (`lowered`); an unguarded access is
   reported as in step 3. Every element access has a `check.bounds` or a proof the audit
   accepts, or traps itself (`get` and `set` with a dynamic index; masked, gather and scatter
   vector ops). The element accesses: every `addr.elem` but the pointer arithmetic below, a `get`
   or `set` of an array, and every vector memory op. One through the element pointer
   `p = slice.ptr s` of a slice is an access of `s` (refused when the elements of `s` are arrays
   or slices); one through any other reference or raw pointer to an array or slice is checked
   against that type's length. A `slice.ptr` result points to the slice's element type, and its
   every use is such an access of its slice, a prefetch, a cast to an int, or `slice.make` of it
   with `slice.len s` whose type is that of `s` with mutability or alignment dropped (raised
   alignment only as the `Some` of `align_to`'s `select`); any other is refused as an unguarded
   access. An `addr.elem` on a raw pointer to one element that is, through pointer-to-pointer
   casts, the result of a `load` or a call is pointer arithmetic, the `unsafe` code's own
   obligation, and no access; any other access through a pointer to one element is refused. At
   level `opt` no audit runs: the release passes inline calls and substitute arguments for
   params, and the audit takes facts from calls and params.

After steps 3-5 the loaded module is renumbered canonically. At level `lowered` the removal of
step 4 follows, which keeps the module canonical, and then the IR verifier, as in a build; at the
other levels the verifier runs on the renumbered module.

What loading proves depends on the level:
- Every level: the module is well formed (steps 1-3: types, shapes, statics, signatures,
  `known` items, intrinsics and their arch, SSA and op typing).
- `lowered`, `checked`: every element access (step 5) has a `check.bounds` or a proof the audit
  accepts.
- `lowered`: no `claim` line. Step 4 computes the summaries from the bodies; a check is removed
  only by a proof the witness checker accepts. This is the one level at which loading proves
  the element accesses of slices and arrays in bounds. The IR carries no `unsafe` marker, and
  `unsafe` code emits the following forms, so they are trusted, not proved: pointer arithmetic
  on a raw pointer to one element from a `load` or a call, a cast that changes a pointer's
  pointee, `slice.make` with a length taken from data, and a cast of an int to a pointer.
- `checked`: the `claim` rows are assertions of the text, not facts loading proves; the text
  holds no proof to check them by. The audit trusts them as it trusts computed summaries: a
  `pre` row lets an access of the fn go without a check, a `post` row lets a caller's access go
  without one. So a text at this level may omit checks by construction, and loading does not
  establish its memory safety.
- `opt`: well-formedness only. The text is trusted as the output of the release passes; its
  `claim` rows are carried, not read.
- A claim does nothing else: no step and no later pass (`ir opt`'s release passes, `ir build`,
  `ir run`) reads a claim to remove a check, retarget a call, or change code; the removal of
  proven checks runs only at level `lowered`.

Building from `checked` or `opt` text is therefore a developer path, not a safety boundary.

Every failure of steps 1-5 is an error in the input (exit 1), except a witness the witness
checker refuses in step 4, which is a compiler bug (exit 3).

## Ops (scalar set, stave-ir `Op`)

Integer ops take signedness from the result type. Trapping ops (marked †) are single
instructions; the trap kind is the matching panic kind (`Overflow`, `Bounds`, `DivByZero`).

| Op | Operands | Meaning |
|---|---|---|
| `add sub mul` † | `a b` | int: trap `Overflow`; float: IEEE |
| `div rem` † | `a b` | int: `b == 0` traps `DivByZero`; `MIN / -1` and `MIN % -1` trap `Overflow`; float: IEEE, `rem` truncated (fmod) |
| `neg` † | `a` | signed int: `-MIN` traps `Overflow`; float: sign flip |
| `add.wrap sub.wrap mul.wrap shl.wrap` | `a b` | `+% -% *% <<%` (`shl.wrap` masks the amount by bits-1) |
| `add.sat sub.sat mul.sat` | `a b` | `+\| -\| *\|` |
| `shl shr` † | `a amount` | amount ≥ bits traps `Overflow`; `shr` arithmetic iff signed. The front end gives the amount the left operand's type |
| `and or xor`, `not` | `a b`, `a` | bitwise on ints, logical on `bool` |
| `popcount clz ctz` | `a` | bit counts within the type's width; result type = operand type |
| `mul.wide` | `a b` | `i32`/`i64`/`u32`/`u64` → `tuple (hi, lo)`: the exact double-width product, `hi` signed iff the type is, `lo` its low bits |
| `eq ne lt le gt ge` | `a b` | result `bool`; ints by signedness, floats IEEE (unordered: only `ne` true); pointers and fn values `eq`/`ne` only |
| `select` | `c a b` | `c ? a : b`: `c` is `bool`, `a` and `b` have the result type (any type, aggregates included). Both operands are already computed; a backend emits no branch on `c` and no memory access whose address depends on `c` (a conditional move per scalar or per 8-byte word). A mask `c` with vector operands of its shape is the lane-wise form, verified when vector lowering lands |
| `cast cast.wrap cast.sat cast.round cast.try` | `x` | `as as% as\| as~ as?` (allowed pairs: below); `cast.try` yields the `Option` instance |
| `make` | elems.. | tuple, struct, or array value (SSA aggregate) |
| `get` † / `set` † | `agg idx` / `agg idx v` | field (constant index) or array element; `set` yields the updated aggregate; a dynamic array index traps `Bounds` |
| `enum.make` / `enum.tag` / `enum.get` | `v fields..` / `e` / `e v f` | enum value of variant `v` (constant) with its fields; its variant index (result `u32`; the verifier accepts any int type holding the variant count); field `f` of variant `v` |
| `slot` | — | `%k = slot t<ref mut T>`: a stack slot for a local of type T that lives in memory (SSA rule), result `&mut T` |
| `load` / `store` | `p` / `p v` | memory read / write (`store` has no result); a `bool` byte read from memory is `1` iff nonzero |
| `addr.field` / `addr.enum_field` / `addr.elem` | `p i` / `p v f` / `p i` | address arithmetic, unchecked (wrapping: `p + i * size` mod 2^64) |
| `slice.make` / `slice.len` / `slice.ptr` | `p n` / `s` / `s` | fat pointer (`&[T]`, `&str`) parts |
| `check.bounds` † | `i n` | `i ≥ n` (unsigned) traps `Bounds` |
| `mem.prefetch_read_l1` … `mem.prefetch_write_nta` | `p i` | the eight prefetch hints (simd-ops.md "Memory hints") of the address `p + i * size_of(pointee)` (wrapping); `p` a thin pointer (`slice.ptr` of a slice, or the raw pointer with `i = 0`); no access, never traps |
| `mem.store_stream` / `mem.fence_stream` | `p v` / — | `store` with a non-temporal hint (`v` a lane type or a vector) / orders the streaming stores before it |
| `fn.make` | `@f env` | fn value `(code, environment)` of closure body `@f` (see Closures) |
| `call` / `call.ind` | `@f args..` / `f args..` | direct / fn-value call (`call.ind` also passes the value's environment, see Closures); a direct call of an `extern "C"` import declared with `...` passes C variadic args after its params: `i32 u32 i64 u64 isize usize f64` values, thin raw pointers, or int constants |
| `intrinsic.<name>` | args.. | contracts/intrinsics.md |
| `panic.<kind>` | — | unconditional trap: `panic.bounds panic.overflow panic.div_by_zero panic.assert panic.out_of_memory panic.stack_overflow panic.unwrap panic.unsupported_cpu panic.deadlock panic.contract` (lowering emits no `panic.contract`: a false clause calls the runtime, see Contracts) |

GPU ops (`gpu.<name>`, gpu-ops.md) extend this list, and ir-text.ebnf, when their lowering lands.

**Vector and mask ops** (one per simd-ops.md portable op; full table in stave-ir `simd`). Lane-wise
ops take the lane semantics of the scalar op; V = the vector result type, M = `mask W n`:

| Op | Operands | Meaning |
|---|---|---|
| `vec.splat` | `x` | every lane `x` |
| `vec.add sub mul div neg abs copysign sqrt floor ceil round trunc fma` | lanes | float lanes, IEEE (`round` ties to even, `fma` one rounding) |
| `vec.min max min_num max_num min_x86 max_x86` | `a b` | IEEE minimum / maximum, minimumNumber / maximumNumber (a quiet or signaling NaN is missing: the other operand; two NaNs: a quiet NaN), `a < b ? a : b` (a NaN result is some NaN: quiet bit and payload unspecified); int `min max` by value |
| `vec.add.wrap sub.wrap mul.wrap add.sat sub.sat mul.sat checked_add checked_sub checked_mul` | `a b` | int lanes; `checked_*` yield `Option<V>` |
| `vec.mul_hi` | `a b` | int lanes: the high half of the exact double-width product, signed iff the lanes are; never traps |
| `vec.div_lanes` † `vec.div_const` † | `a b` / `a K` | int division per lane (traps as scalar `div`); `K` a constant of the lane type |
| `vec.and or xor andnot not abs_wrap popcount clz` | lanes | int lanes (`abs_wrap`: signed); `andnot` = `a & ~b` |
| `vec.shl shr` † `vec.shl.wrap` | `a amount` | amount: V or an int; as scalar shifts |
| `vec.eq ne lt le gt ge` | `a b` | result M |
| `vec.reduce_add reduce_add_wrap reduce_min reduce_max reduce_and reduce_or` | `v` | lane type; tree: lane i with lane i+N/2 until one lane remains (`reduce_add`: float lanes; `reduce_add_wrap`: int lanes, wrapping) |
| `vec.reduce_add_ordered` / `vec.scan_add scan_add_wrap` | `v init` / `v` | sequential from lane 0; `reduce_add_ordered` and `scan_add`: float lanes; `scan_add_wrap`: int lanes, wrapping |
| `vec.cast cast.wrap cast.sat cast.round cast.try` | `v` | `as as% as\| as~ as?` per lane (`try`: `Option<V>`, `Some` iff every lane converts) |
| `vec.bitcast split join widen_lo widen_hi narrow_wrap narrow_sat permute lookup compress expand` | | shape ops (`permute v k0 ..`: constant lane indices) |
| `vec.load` / `vec.store` | `s i` / `s i v` | lanes `s[i..i+N]`, after `check.bounds i (sub.sat (slice.len s) N-1)` or under a chunk loop's head test; release passes (which run after the removal of proven bounds checks) may replace a slice `s` by its `slice.ptr` (unchecked `*T`) |
| `vec.load_aligned` † / `vec.store_aligned` † | `s i` / `s i v` | as `load` / `store`; `i % N != 0` traps `Bounds` |
| `vec.store_stream` † | `s i v` | `vec.store_aligned` with a non-temporal hint, after the same `check.bounds` |
| `vec.load_masked` † / `vec.store_masked` † | `s i m` / `s i v m` | active lanes only (inactive read 0); an active lane past the end traps `Bounds` before any lane is written |
| `vec.gather` † / `vec.scatter` † | `s idx` / `s idx v` | `idx`: unsigned lanes of any width, N of them, zero-extended; any index ≥ len traps `Bounds` before any write; duplicate indices: highest lane wins |
| `mask.and or xor not select to_bits to_vec first_n rewiden all none` | | mask ops (`mask.select m a b`); a mask lane is all zeros or all ones, a load of any other pattern is undefined behaviour (the interpreter faults) |

Vector and mask ops are ops of their own, not intrinsics: `vec.*` and `mask.*` are not intrinsic
names (`intrinsic.vec.add` is rejected).

Mask `==` / `!=` are `mask.not (mask.xor a b)` / `mask.xor a b`. `s.align_to::<A>()` is
`slice.ptr`, `cast` to `usize`, `rem` by A, `eq 0`, and a `select` between the two `enum.make`s.

**Casts** (`arith::cast_allowed`): `cast` = lossless only (ints when the target holds every
value; `bool` → int; `char` → `u32 u64 i64 usize isize`; `u8` → `char`; int → float when the
significand holds it; f16/bf16 → f32/f64, f32 → f64; pointer ↔ pointer; pointer ↔ 64-bit int
`u64 usize i64 isize`). A fn item operand `@f` casts like a pointer (its code address); a fn
value operand and a fn-type target are rejected. `cast.wrap`: int, `bool`, `char`, or pointer → int (float → int is
rejected: use `cast.sat` or `cast.try`). `cast.sat`: int, `char`, or float → int, NaN → 0
(float → float is rejected: use `cast.round`). `cast.round`: int or float → float, nearest even.
`cast.try`: int, `char`, or float → int; int → `char` or float; float → float; `Some` only when
the value converts exactly (float → int: integral and in range). A float cast to its own type
(any kind, scalar or per lane) is a bit copy: a signaling NaN keeps its bits.

## TypeKey text (canonical type-key printer)

One printer, used by `type` lines here and by the `types` dump (stage-gates). It prints the
canonical type key: the variant keyword, then its fields in key order, separated by single
spaces. Component types print as `t<T>` (canonical type id); decls as `d<D>`.

| Variant | Text | Examples |
|---|---|---|
| `Int{signed, bits}` | `int s\|u 8\|16\|32\|64\|size` | `int s 32`, `int u size` |
| `Float{bits}` | `float 16\|32\|64\|b16` (`b16` = bf16, sorts last) | `float 64` |
| `Bool`, `Char`, `Str` | `bool`, `char`, `str` | |
| `Ref{mut, inner}` | `ref t<T>` / `ref mut t<T>` | `ref mut t4` |
| `RawPtr{mut, inner}` | `ptr const t<T>` / `ptr mut t<T>` | `ptr const t2` |
| `Array{elem, len}` | `array t<T> <len>` | `array t3 16` |
| `Slice{elem, align}` | `slice t<T>`, or `slice t<T> align <A>` | `slice t5 align 32` |
| `Tuple[..]` | `tuple` then ` t<T>` per element; unit is `tuple` | `tuple t3 t9` |
| `Adt{def, args}` | `adt d<D>` then ` t<T>` per type argument, a decimal integer per const argument | `adt d17 t4 8` |
| `FnType{params, ret, features}` | `fntype` then ` t<T>` per param, ` -> t<T>`, then ` <feature>` per feature in contracts/simd-ops.md §Features order | `fntype t4 t5 -> t2 avx2` |
| `Vec{lane, n}` | `vec t<T> <n>` or `vec t<T> n` (sizeless) | `vec t3 8` |
| `Mask{width, n}` | `mask <W> <n>` or `mask <W> n` | `mask 32 8` |
| `Param(idx)`, `ConstParam(idx)` | `param <idx>`, `constparam <idx>` (`types` dump only) | `param 0` |
| `Never` | `never` (return type of a fn that never returns; `types` dump only) | |

The IR has no generic types and no `never` (lowering makes a `-> never` fn type's return the
unit type), so the IR text has no `param`, `constparam`, or `never` key. There is no key for the
error type: nothing of error type is printed (IR is not printed with errors; the `types` dump
rule is in stage-gates.md).

## Order

1. Lines by "Sections". Functions by canonical instance id (Ids above). A closure or `spawn`
   body is a function `@<enclosing>.<k>` (k = its source order within the enclosing body),
   printed directly after the enclosing function.
2. Blocks: `b0` is the entry, and its params are the fn params. The other blocks are numbered by a
   preorder walk of the body's control-flow nesting: on reaching a construct, the walk numbers its
   blocks in the order below, then walks the construct's parts in source order.

   | Construct | Blocks, in order |
   |---|---|
   | `if` statement, `if then else` | then, else (if present), join |
   | `while` | head, body, exit |
   | `loop` | body, exit |
   | `for` (element or chunk) | head, body, tail (one per tail block or check), exit |
   | `match` statement or expression | per arm in source order: test (one per alternative of an or-pattern, in source order), guard (if present), body; then join |
   | `&&`, `\|\|` | right operand, join |
   | `?` | error return, continue |
   | a tested drop ("Drop flags") | drop, continue |
   | anything else | none |

   Blocks unreachable from `b0` (e.g. code after `return`) are deleted, and the rest are renumbered
   densely in the same order.
3. Values: `%0..` dense per function, in block order; within a block, its params, then its
   instruction results in order.
4. Instructions within a block follow source evaluation order: operands left to right, then the
   operation; call arguments left to right; an autoref `&mut self` borrow is two-phase: in
   `v.push(v.len())` the arguments are evaluated before the `&mut` loan activates.

## Order at level `opt`

Rules 2 and 4 of "Order" state levels `lowered` and `checked`. At level `opt` the block order is
the stored order the release passes give (opt-passes.md "Order and invariants of level `opt`"),
which "Canonical module" keeps: (1) blocks of the input keep their relative order; (2) an inlined call
in a block puts, directly after the block that holds the code before the call, the callee's
blocks other than its entry in the callee's order, then the block that holds the code after the
call; the entry's instructions follow the code before the call; calls of one block are expanded
left to right; (3) a removed block leaves the others in order; (4) a merged block stands where
its first part stood and holds the first part's instructions, then the second's. Values and
pools are numbered by "Canonical module". Every `slot` of a body is in `b0` before every other
instruction: first the fn's own in declaration order, then those of inlined calls, calls in
(block, instruction) order of the body before inlining, each callee's slots in its own order.

## SSA rule (pruned SSA)

This section states levels `lowered` and `checked`. At level `opt` the release passes
(opt-passes.md) may have folded, merged, deleted, and moved code, and a block may have any params;
the **Slots** rule still holds.

- Every `let` binding is its own local. A local lives in a `slot` created in `b0`, in declaration
  order, and is read and written with `load`/`store`, when its address is taken (`&`, `&mut`,
  autoref, a sub-slice), a closure captures it, its type has drop glue, or an assignment stores
  into a part of it (`a[i] = v`, `s.f += v`, `t.0[j].k = v`): such a store writes the part in
  place. Every other local is an SSA value: `let` and assignment rebind it and emit no
  instruction.
- A join (if/match join, loop head, loop exit, `for` head and exit, `&&`/`||` join) has exactly one
  block param per SSA local that is assigned inside the construct (a branch, arm, or loop body) and
  is live at the join's entry. Liveness is computed by path exploration. Params are ordered by local index (fn params, then `let`s in source order). An expression
  join (`if then else`, match expression, `&&`/`||`) has one more param, first, for the value.
- **Or-pattern arm** (`P1 | P2 =>`): every alternative has its own test block, which binds the
  arm's locals from that alternative and branches to the arm's guard block (its body block when
  the arm has no guard). That block has one param per SSA local the alternatives bind, ordered by
  local index (the first alternative's bindings are the locals), and is pruned like a join's: a
  bound name that is not used has no param. Locals in a `slot` are stored by each test block. The
  guard and the body are lowered once; a false guard continues with the next arm.
- No other block params. No folding, copy propagation, or dead-code removal beyond unreachable
  blocks, except: a literal `-` is folded; `size_of` / `align_of` / `needs_drop::<T>()` are the
  instance's constant; an `if` (statement or expression) whose condition is a constant condition
  (lang-details.md, "§1 Type queries") ends in `br` to the taken arm (or the join), and the other
  arm, unreachable, is deleted.
- **Slots**: every `slot` is in `b0`, before any other instruction of `b0`, one per local in a
  slot in declaration order (fn params first, then `let`s in source order). The drop flags
  follow them ("Drop flags").
- **Panics**: an unconditional trap (`unwrap` on `None`, a failed `std.debug.assert`, ...) is
  `panic.<kind> ()` followed by the terminator `unreachable`; code after it in source is
  unreachable and deleted with its blocks. A call of a `-> never` fn is followed by `unreachable`
  the same way.
- **Contracts**: after the params are bound, each `requires` clause in order is checked: each operand
  of its top-level `&&` in order is its expression, then `cbr c ok fail`; `fail` is
  `call runtime.panic.requires_failed (s)` (at the site of the operand's first token) +
  `unreachable`, with `s` the `str` constant of the whole clause's source text (one line: white
  space holding a line break becomes one space). Then each `old(e)` of the `ensures` clauses, in
  source order, is `e` evaluated once; the `old(e)` uses read that value. At each `return`, `?`
  error return, and the end of a unit body, after the scope drops, `result` is bound to the
  returned value (an int or `bool` no clause borrows: that SSA value; else stored to its slot, the
  last of the locals' slots, none for a unit return) and each `ensures` clause is checked the
  same way, `fail` calling `runtime.panic.ensures_failed (site, s)` with the site of the `return`
  token (the `?`, the body's `}`). No `panic.<kind>` op: the report holds the clause text. A
  `requires` clause of a `test fn` is not checked. The removal pass below treats each such check like a bounds check:
  a proven one goes (`cbr` becomes `br ok`, the call and its block go); one that fails whenever
  reached is a compile error (E0215, E0217) and stays.
- **Contracts across calls** (stave-ir `witness`, "Summaries"): a fn states rows `a <= b + w`
  over zero, its int params, the lengths of its slice params, and its result, each read off its
  own IR with a proof the witness checker validates. *Pre* rows: every reachable `requires`
  check tests a compare whose sides fold (trapping `add`/`sub` by a constant, widening casts,
  `slice.len`) to those terms; one row per goal, plus a row per bound that keeps such an
  `add`/`sub` in its type's range where the other rows and the params' types do not. *Post*
  rows: the goals of the `ensures` checks, with the returned value as `result`, that hold at
  every reachable `ret` (at most 16). After `%v = call @f ..` the post rows of `@f`, and its pre
  rows that read a check, are facts about the arguments and `%v`; a fn may use its own after a
  recursive call (what is proved is partial correctness: nothing is said of a call that does not
  return). A `call @f ..` whose facts prove every pre row of `@f` about its arguments goes to the
  unchecked entry of `@f`: the body of `@f` after its own removals, without the failure calls
  of its `requires` checks (each `cbr c ok fail` stays, `fail` being an empty `unreachable`
  block, so `c` is still a fact of the body). It is the fn `@<n>.u`, which never has the link
  symbol of an exported `@f`; when `@f` is an instance of a generic fn and every operand naming
  it in the program is the callee of such a call, `@f` itself becomes it and no `@<n>.u` exists. `call.ind`, `fn.make`, and every other use of `@f` keep `@f`. A call
  whose facts prove the negation of a pre row that reads a check is a compile error (E0218).
  Only the witness checker makes an unchecked entry or a call of one.
- **Bounds checks** (stave-ir `bounds`): `xs[i]` on a place in memory (a
  slice, or an array behind a reference or in a `slot`) is `check.bounds i n` immediately before
  `addr.elem p i` (`n` = `slice.len p`, or the array length as a constant); `a[i]` read from an
  array SSA value is `get`, which checks a dynamic index itself. A constant index ≥ a constant array
  length is `panic.bounds ()` + `unreachable` (the trap is kept; sub-slices fold the same way;
  the code after it is lowered into a block nothing reaches, which the post-pass deletes).
  One pass after lowering removes the checks that have a proof (constant index < length, a
  dominating check on the same index and length, a dominating `lt i n` branch,
  and `LenEq`/`Offset`: `i + k < n` derived from dominating int compares (signed values read
  as their own mathematical value), asserting calls such as `std.debug.assert`, trapping
  `add`/`sub` of constants, unsigned bit ops (`and x y` is at most `x` and `y`, `rem x y` below
  `y` and at most `x`, `shr x k` at most `x`), the range of an int type, and
  `cast`/`cast.wrap`/`cast.sat` to a type holding every value of the source type as the same
  value; a check or a strict unsigned compare against `sub.sat n c` reads as `i + c < n`; every
  `load (addr.field p k)` of an int field behind `p`, a `&T` (not `&mut`) parameter of the entry
  block with T not `std.cell.UnsafeCell`, and every call `@g p` of a getter (one parameter, the
  body `addr.field %0 k; load; ret`) of that field, as one value for the whole call, so the
  columns of a `&soa.Soa` parameter have the length its `len()` returns: this one rests on the
  memory model, not on an earlier event, see soundness.md "Removed checks"). The
  same facts decide the contract checks (above). The pass runs in every build and never looks at
  the build mode: a function keeps the same checks in debug and release, and the checked IR (the
  `ir` dump, the interpreter's input) is its result. Each removed check has a witness, the proof
  as inequalities that each name their place in the IR and that, times their multipliers, add
  up to the goal (stave-ir `witness`; mostly chains, every multiplier 1); an
  independent checker validates every witness, and every refutation of a contract check, before
  the check is deleted or the error reported, and a witness it refuses is a compiler bug (exit
  3). Release passes remove no check an execution would reach (opt-passes.md "Order and
  invariants of level `opt`"); the IR verifier runs again on their output.
  Compiler-made accesses (`for` over an array or slice, drop loops, array → slice coercion
  `addr.elem p 0` + `slice.make`) index below a length the generated code compares against.
- **Memory hints**: `s.prefetch_<rw>_<loc>(i)` is `slice.ptr s` and the prefetch of it with `i`;
  `x.store_stream(s, i)` is `check.bounds i (slice.len s)`, `addr.elem s i`, `mem.store_stream`.
  After the SSA post-pass, `mem.fence_stream` instructions are placed by the rule of simd-ops.md
  "Ordering" (stave-ir `fence`): at the start of a block a streaming loop exits to, right before a
  call, or right after the last streaming store of straight-line code before a call or return.
- **Sub-slices** (`xs[a..b]`, a slice place; `n` as for `xs[i]`): `cbr (gt b n)` to a block
  `panic.bounds ()` + `unreachable`, then `cbr (gt a b)` the same way, then
  `slice.make (addr.elem p a) (sub.wrap b a)`. `xs[a..=b]` tests `ge b n` first and uses
  `e = add b 1` for `b`. A constant `a` makes the length `e - a` a constant, `e` itself (`a = 0`),
  or a trapping `sub`, so bounds facts see it. No test between two constants is emitted; a range
  out of a constant length is `panic.bounds ()` + `unreachable`. That `addr.elem` feeds only the
  `slice.make` and may be one past the end (`a = n`); the audit accepts it with `a <= n` proven
  from the two tests (`SliceStart`).
- **Chunk loops** (`for v: TxN in s`): `s` becomes a slice (an array: `addr.elem p 0` +
  `slice.make`), then `lim = sub.sat (slice.len s) N-1` before the head. The head tests
  `lt i lim` (`i + N <= len`: the proof of the body's `vec.load s i`), the body steps `i` by N
  (`add.wrap`). `tail masked (v, m)`: a block testing `ne (sub.wrap len i) 0`, then a block with
  `m = mask.first_n (len - i)` and `v = vec.load_masked s i m`. `tail scalar (x)`: a counted loop
  (head join with the counter, body binding `x = load (addr.elem s j)`). `tail exact;`: one block
  with `check.bounds (sub.wrap len i) 1`. A `&mut` binding is a `slot` holding the vector, stored
  back (`vec.store`, in a masked tail `vec.store_masked`) wherever the block is left.
- **`Range<T>`** values are the tuple `(T, T)` (`type` lines show `tuple t<T> t<T>`).
- **Fn values**: a fn item bound to an SSA local stays the constant `@<inst>`, so a call through it
  is `call.ind @<inst> args..` (the rebinding emits nothing). The stage0 verifier accepts an
  `@<inst>` callee.
- **Exhaustive match**: a `match` whose arms test a discriminant (enum variant, int, `bool`,
  `char`) lowers its arm tests to one `switch` on the discriminant (`enum.tag` for an enum): a
  case per tested value to the test block of the first alternative testing it, and the last
  alternative's test block as the default `_` (the match is exhaustive, E0300, so no extra trap
  block exists). A test block that fails (a nested pattern, a false guard) branches to the next
  alternative that tests the same value or accepts every value. A guarded arm must make that
  target independent of the value: its alternatives test one value, or the alternative after
  the arm accepts every value. Otherwise the match has no `switch`: every alternative tests the
  discriminant itself, a failed test branches to the next alternative, and a false guard to the
  first alternative of the next arm.

### Drop flags

An owner is a local or a temporary whose type has drop glue in the instance. Whether a drop of
an owner runs where it is reached is decided once, by the forward init dataflow of the borrow
checker's CFG of the generic body (may-init: the owner holds its value on some path to the drop;
must-init: on every path), and the lowering only reads the answer:

| At the drop | The drop |
|---|---|
| must-init | is emitted |
| not may-init | is left out |
| may-init, not must-init | is tested: it runs where the owner's flag is set |

The drop of the value an assignment overwrites is a drop of the local assigned. A drop reached
at more than one place (a statement lowered at each `return`) is tested unless every place gives
the same answer, and a drop no path from the entry reaches is left out.

- An owner has a flag exactly when one of its drops is tested. A flag is a `bool` `slot`; no
  flag is an SSA value or a block param, and a temporary with a flag holds its value in a slot.
- The slots of a fn, in order: the locals' slots ("Slots"), the flags of its flagged locals in
  declaration order, the flags of its flagged temporaries in the order the temporaries are made,
  then every other slot. Right after the slots `b0` stores `0` into each flag, in the same order.
- A flag is stored `1` where its owner gets its value (its binding, an assignment of the whole
  owner, the making of a temporary), `0` where the value is moved out, and `0` after each drop
  of the owner at its end or on an exit that is emitted or tested.
- A tested drop is `%f = load t<bool> %flag` and `cbr %f d c`, with block `d` holding the drop
  and `br c`; the code after the drop continues in block `c` ("Order", rule 2).

## Types and fntype

- `type` lines list every type of the module, by canonical id T (the P and S sets, "Ids").
- `fntype` prints only the features not implied by another member of the set (the set is closed
  under implication: `needs(avx2)` prints `avx2`, not `sse42 avx avx2`), in contracts/simd-ops.md
  §Features order. In the canonical type order a feature set compares as its `u64` bitset (bit i
  = the i-th feature of §Features) read as a number.
- A fn header prints the fn's `needs(..)` (the same feature list, comma-separated) when they are not
  empty: `fn @3 main.sum_v3(t4) -> t2 needs(avx2) {`. A closure has the needs of its fn and prints
  them too; the target baseline is never printed.

## Closures

A closure or `spawn` body is a function `@<enclosing>.<k>` whose path is the enclosing fn's
path (`@7.0` after `@7`, both `app.apply`); `k` numbers the closure literals and `spawn`
bodies of the enclosing body together, by the position of their `fn` / `spawn` token (nested
bodies included). Captures (stave-sema `closures.rs`): every outer local used in the body, by
reference (`Ref` / `RefMut`); `copy` captures are reserved (no syntax yet, LANG.md §8).

- **Fn value** = two words `(code, environment)` (`fntype` layout: 16 bytes, align 8, like a fat
  pointer). A fn item, or a closure body without captures, is the constant `@<inst>` (null
  environment).
- **Environment**: a body with captures has one more param, last: `%e: t<ref mut (tuple of ref
  mut T_i)>`, one element per capture in capture order (binding declaration order), pointing at
  the captured local's slot (a captured local always lives in a `slot`). Its `b0` loads each
  element (`addr.field %e i`, `load`) and uses it as that local's slot. The enclosing fn fills a
  `slot` of the tuple (one `addr.field` + `store` per capture, at the closure expression) and
  makes the value with `%v = fn.make t<fntype> @<enclosing>.<k> %env`.
- **`call.ind f args..`** passes `f`'s environment word as one extra trailing argument; a callee
  without the environment param (a fn item) ignores it (the Stave calling convention lets the
  caller own the argument area). `eq`/`ne` compare both words.
- **`scope { .. spawn { B } .. }`**: `%h = slot t<ref mut usize>` stored `0` at the scope; each
  `spawn` is `call @runtime.thread.scope_spawn %h %v` (`%v` = B's fn value of type `fn()`); every
  exit of the scope block (its end, `return`, `break`, `continue`, `?`) first calls
  `@runtime.thread.scope_join %h`, before the block's drops.

## Instance root set

- **debug**: every non-generic fn of the package and its dependencies (not fns dropped for
  needing another arch's features; `test fn`s only under `stave test`), plus the release roots
  below, plus every instance they request, transitively. Fns of the built-in packages (`std`,
  `runtime`) are not roots: they are lowered only when reachable, as in release.
- **release**: only instances reachable through calls and fn-value uses from the roots `main`,
  `extern "C"` exports, the `allocator` impl, and, under `stave test` (also `stave test --release`),
  `test fn`s, plus the runtime entry points the compiler's own startup, trap, and signal code calls
  (contracts/intrinsics.md "Paths fixed by the compiler": `runtime.start.entry`,
  `runtime.panic.trap`, `runtime.panic.trap_msg`, `runtime.panic.on_signal`).
- **`dispatch fn`** (both modes): a reachable dispatch has a body that calls only its live arms,
  so only those are reachable through it. An arm of another arch is dropped. With K = the target
  baseline ∪ the dispatch's needs, the first remaining arm whose features are ⊆ K always matches:
  the body ends with a direct call of it and the arms below it (and `_`) are not called; if it is
  the first arm, the body is that one call. Otherwise the body is `%p = intrinsic.static_addr S`,
  `%w = load t<u64> %p` (S = the index of the static `runtime.cpu.WORD`; no call), then, per arm
  top-down, `%m = and t<u64> %w M`, `%c = eq t<bool> %m M`, `cbr` to a block
  returning the arm target's call result (M = the arm's closed feature set as a `u64`, bit i =
  feature i). Each such block, and the final one, begins with the trace test: `%t = and t<u64> %w
  T`, `%c = ne t<bool> %t 0`, `cbr` to a block that calls `@runtime.cpu.trace` with the string
  `dispatch: <dispatch path> -> <arm target path>\n` and branches on (T = bit 63, intrinsics.md
  "CPU feature detection"; no test without a `runtime.cpu.trace`). A runtime library without
  `runtime.cpu.WORD` (the dump goldens' stand-in) has no word: the body is the call of that
  last arm alone. A call of a dispatch from a fn whose
  needs ∪ the baseline cover the first remaining arm's features calls that arm target instead
  (a fn value of the dispatch is the dispatch).
- In both modes a lowered `intrinsic.thread_clone` or `intrinsic.thread_start_addr` requests
  `runtime.thread.entry`, which only the compiler's `__stave_thread_start` stub calls
  (intrinsics.md "Thread start"), and a request of `runtime.cpu.features` (`std.cpu.features`) or
  a dispatch body that loads `runtime.cpu.WORD` also requests `runtime.cpu.init`, which only `intrinsic.cpu_init` in
  `runtime.start.entry` calls. A lowered `intrinsic.cpu_init` requests `runtime.cpu.init` itself
  when the target baseline is above its arch's lowest (`x86_64-linux-v2`, `aarch64-linux`): start
  code then checks the CPU against it. Without either request (no reachable dispatch with a
  tested arm, no `std.cpu.features`, a lowest-baseline target) a program has no `runtime.cpu`
  code. `intrinsic.cpu_baseline` is never in the IR: lowering replaces it by its constant.
- In both modes a lowered `intrinsic.copy_bytes` requests `std.slice.copy_raw`, the fn the
  backends call in its place (no IR calls it).
- Generic decls are never printed, only their instances. `kernel fn`s are printed like fns; GPU ops print as
  `gpu.<name>` (gpu-ops.md "Checked-IR spelling"). Intrinsic calls print as `intrinsic.<name>`.
