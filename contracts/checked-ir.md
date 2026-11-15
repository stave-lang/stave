# Contract: canonical checked IR (v0.3, `--dump=ir`)

The checked IR is the SSA IR as built by lowering: after type check, borrow
check, and monomorphization, before any optimization pass. Values carry Stave types (canonical type
ids). Layout lowering to the machine types of `IrFunc` comes after this point and is not
compared. Both compilers print exactly this text; the stage1 gate compares the bytes (stage-gates `ir`).
It is printed only for a build with no error diagnostics; with errors, `ir.txt` is not written. UTF-8, one `\n` per line, single spaces,
no trailing spaces, no blank lines.

## Text (sketch)

```
IrFile  = TypeLine* Func* ;
TypeLine= "type " TId " " TypeKey "\n" ;          (* every type a printed instance uses, by T *)
TypeKey = variant (" " field)* ;                  (* canonical type key, fields in key order:
                                                     `int s 32`, `ref mut t4`, `adt d17 t4`, `fntype t4 -> t2 avx2` *)
Func    = "fn @" Inst " " Path ("<" GArg ("," GArg)* ">")? " {" "\n" Block+ "}" "\n" ;
GArg    = TId | IntC ;                            (* type arg, or const generic arg in decimal (as in `adt` keys) *)
Block   = Blk "(" (Val ": " TId ("," " " Val ": " TId)*)? "):" "\n" InstL* Term ;
InstL   = "  " (Val " = ")? Op " " (TId | "()") (" " Operand)* "\n" ;   (* result type; `()` if none *)
Term    = "  " ( "br " Target | "cbr " Operand " " Target " " Target
               | "switch " Operand (" " IntC ":" Target)* " _:" Target
               | "ret" (" " Operand)? | "unreachable" ) "\n" ;
Target  = Blk "(" (Operand (", " Operand)*)? ")" ;
Operand = Val | IntC | FloatC | SId | "@" Inst | TId ;
Val = "%" Nat ; Blk = "b" Nat ; TId = "t" Nat ; SId = "s" Nat ; Inst = Nat ("." Nat)* ;
IntC = "-"? Nat ; FloatC = "0x" Hex ;             (* floats as their exact IEEE bits *)
```

- `Op` names are the stave-ir `Op` names (scalar ops plus one per simd-ops portable op).
  Trapping operations (overflow, bounds, division, `tail exact`) are single instructions and create
  no blocks.
- `Path` is the module path and item name joined by `.`; a method is `<head>.<name>`, `<head>` =
  the impl's self-type ADT path (`main.Counter.bump`), or the impl's module path plus a built-in
  type name (`std.core.str.fmt`); drop glue is `drop_glue` with its type as the one generic arg
  (`fn @200 drop_glue<t135>`). Strings appear only as `s` ids.
- Constants are inline operands, never values: ints in decimal (`bool` as `0`/`1`, `char` as its
  code point), floats as `0x` + their IEEE bits in lowercase hex, zero-padded to the format's
  width (4 digits for f16/bf16, 8 for f32, 16 for f64), strings as `s<S>`, fn items as `@<inst>`,
  types as `t<T>`.
- An instruction with no result prints `()` as its type (`store () %3 %1`).

**Ids** (v0.3; the dump's ids are local to the printed module, never the `types` dump's or other
stages' numbering, except `d<D>`):

- `t<T>`: the rank, in the canonical type-key order, among the types
  the printed functions use (the `type` lines' set below); dense from `t0`. Instance types such as
  `Pair<u16>` exist only here. `d<D>` in `adt` keys is the canonical decl id D.
- `@<inst>`: the rank of the instance among the build's instances (the root set's closure,
  below), ordered by (decl D, generic args by `t<T>`), drop glue after every decl instance, ordered
  by its type; closures `@<enclosing>.<k>`. Bodiless instances (`extern "C"` imports; `dispatch
  fn`s until their lowering lands) take ids but are not printed, so printed ids may have gaps;
  calls still name them (`call t2 @0 %1`). Ids differ between debug and `--release` (different
  root sets).
- `s<S>`: the rank of the string constant among the module's distinct string constants by
  (xxHash32 seed 0 over the bytes, bytes), like interned strings; literals are not interned
  strings, so these ids are not the `strings` dump's S.
- `intrinsic.static_addr <i>` (contracts/intrinsics.md): `i` is the rank of the decl by D among
  the program's `static` and `static mut` decls (one sequence over both forms).
  Every static occupies at least one byte in the image (a zero-size static gets one zero
  byte, aligned as its type), so distinct statics have distinct addresses on every target.

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
| `mul.wide` | `a b` | `u32`/`u64` → `tuple (hi, lo)` |
| `eq ne lt le gt ge` | `a b` | result `bool`; ints by signedness, floats IEEE (unordered: only `ne` true); pointers and fn values `eq`/`ne` only |
| `select` | `c a b` | `c ? a : b`: `c` is `bool`, `a` and `b` have the result type (any type, aggregates included). Both operands are already computed; a backend emits no branch on `c` and no memory access whose address depends on `c` (a conditional move per scalar or per 8-byte word). A mask `c` with vector operands of its shape is the lane-wise form, verified when vector lowering lands |
| `cast cast.wrap cast.sat cast.round cast.try` | `x` | `as as% as\| as~ as?` (allowed pairs: below); `cast.try` yields the `Option` instance |
| `make` | elems.. | tuple, struct, or array value (SSA aggregate) |
| `get` † / `set` † | `agg idx` / `agg idx v` | field (constant index) or array element; `set` yields the updated aggregate; a dynamic array index traps `Bounds` |
| `enum.make` / `enum.tag` / `enum.get` | `v fields..` / `e` / `e v f` | enum value of variant `v` (constant) with its fields; its variant index (result `u32`; the verifier accepts any int type holding the variant count); field `f` of variant `v` |
| `slot` | — | `%k = slot t<ref mut T>`: a stack slot for an address-taken local of type T, result `&mut T` |
| `load` / `store` | `p` / `p v` | memory read / write (`store` has no result); a `bool` byte read from memory is `1` iff nonzero |
| `addr.field` / `addr.enum_field` / `addr.elem` | `p i` / `p v f` / `p i` | address arithmetic, unchecked (wrapping: `p + i * size` mod 2^64) |
| `slice.make` / `slice.len` / `slice.ptr` | `p n` / `s` / `s` | fat pointer (`&[T]`, `&str`) parts |
| `check.bounds` † | `i n` | `i ≥ n` (unsigned) traps `Bounds` |
| `fn.make` | `@f env` | fn value `(code, environment)` of closure body `@f` (see Closures) |
| `call` / `call.ind` | `@f args..` / `f args..` | direct / fn-value call (`call.ind` also passes the value's environment, see Closures) |
| `intrinsic.<name>` | args.. | contracts/intrinsics.md |
| `panic.<kind>` | — | unconditional trap: `panic.bounds panic.overflow panic.div_by_zero panic.assert panic.out_of_memory panic.stack_overflow panic.unwrap` |

GPU ops (`gpu.<name>`, gpu-ops.md) extend this list when their lowering lands.

**Vector and mask ops** (one per simd-ops.md portable op; full table in stave-ir `simd`). Lane-wise
ops take the lane semantics of the scalar op; V = the vector result type, M = `mask W n`:

| Op | Operands | Meaning |
|---|---|---|
| `vec.splat` | `x` | every lane `x` |
| `vec.add sub mul div neg abs copysign sqrt floor ceil round trunc fma` | lanes | float lanes, IEEE (`round` ties to even, `fma` one rounding) |
| `vec.min max min_num max_num min_x86 max_x86` | `a b` | IEEE minimum / maximum, minimumNumber / maximumNumber, `a < b ? a : b`; int `min max` by value |
| `vec.add.wrap sub.wrap mul.wrap add.sat sub.sat mul.sat checked_add checked_sub checked_mul` | `a b` | int lanes; `checked_*` yield `Option<V>` |
| `vec.div_lanes` † `vec.div_const` † | `a b` / `a K` | int division per lane (traps as scalar `div`) |
| `vec.and or xor andnot not abs_wrap popcount clz` | lanes | int lanes; `andnot` = `a & ~b` |
| `vec.shl shr` † `vec.shl.wrap` | `a amount` | amount: V or an int; as scalar shifts |
| `vec.eq ne lt le gt ge` | `a b` | result M |
| `vec.reduce_add reduce_add_wrap reduce_min reduce_max reduce_and reduce_or` | `v` | lane type; tree: lane i with lane i+N/2 until one lane remains (`reduce_add`: float lanes; `reduce_add_wrap`: int lanes, wrapping) |
| `vec.reduce_add_ordered` / `vec.scan_add` | `v init` / `v` | sequential from lane 0 |
| `vec.cast cast.wrap cast.sat cast.round cast.try` | `v` | `as as% as\| as~ as?` per lane (`try`: `Option<V>`, `Some` iff every lane converts) |
| `vec.bitcast split join widen_lo widen_hi narrow_wrap narrow_sat permute lookup compress expand` | | shape ops (`permute v k0 ..`: constant lane indices) |
| `vec.load` / `vec.store` | `s i` / `s i v` | lanes `s[i..i+N]`, after `check.bounds i (sub.sat (slice.len s) N-1)` or under a chunk loop's head test |
| `vec.load_aligned` † / `vec.store_aligned` † | `s i` / `s i v` | as `load` / `store`; `i % N != 0` traps `Bounds` |
| `vec.load_masked` † / `vec.store_masked` † | `s i m` / `s i v m` | active lanes only (inactive read 0); an active lane past the end traps `Bounds` |
| `vec.gather` † / `vec.scatter` † | `s idx` / `s idx v` | `idx`: unsigned lanes of any width, N of them, zero-extended; any index ≥ len traps `Bounds` before any write; duplicate indices: highest lane wins |
| `mask.and or xor not select to_bits to_vec first_n rewiden all none` | | mask ops (`mask.select m a b`) |

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
the value converts exactly (float → int: integral and in range).

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
| `Param(idx)`, `ConstParam(idx)` | `param <idx>`, `constparam <idx>` | `param 0` |
| `Never` | `never` (return type of a fn that never returns) | |

There is no key for the error type: nothing of error type is printed (IR is not printed with
errors; the `types` dump rule is in stage-gates.md).

```
type t3 int s 32
type t9 bool
fn @41 app.max {
b0(%0: t3, %1: t3):
  %2 = gt t9 %0 %1
  cbr %2 b1() b2()
b1():
  ret %0
b2():
  ret %1
}
```

## Order

1. `type` lines by canonical type id T. Functions by canonical instance id (Ids
   above). A closure or `spawn` body is a function `@<enclosing>.<k>` (k = its source order within
   the enclosing body), printed directly after the enclosing function.
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
   | anything else | none |

   Blocks unreachable from `b0` (e.g. code after `return`) are deleted, and the rest are renumbered
   densely in the same order.
3. Values: `%0..` dense per function, in block order; within a block, its params, then its
   instruction results in order.
4. Instructions within a block follow source evaluation order: operands left to right, then the
   operation; call arguments left to right; an autoref `&mut self` borrow is two-phase: in
   `v.push(v.len())` the arguments are evaluated before the `&mut` loan activates.

## SSA rule (pruned SSA)

- Every `let` binding is its own local. A local whose address is taken (`&`, `&mut`, autoref) lives
  in a `slot` created in `b0`, in declaration order, and is read and written with `load`/`store`.
  Every other local is an SSA value: `let` and assignment rebind it and emit no instruction.
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
- No other block params. No folding (except literal `-`), copy propagation, or dead-code
  removal beyond unreachable blocks.
- **Slots**: every `slot` is in `b0`, before any other instruction of `b0`, one per address-taken
  local in declaration order (fn params first, then `let`s in source order).
- **Panics**: an unconditional trap (`unwrap` on `None`, a failed `std.debug.assert`, ...) is
  `panic.<kind> ()` followed by the terminator `unreachable`; code after it in source is
  unreachable and deleted with its blocks. A call of a `-> never` fn is followed by `unreachable`
  the same way.
- **Bounds checks** (stave-ir `bounds`): `xs[i]` on a place in memory (a
  slice, or an array behind a reference or in a `slot`) is `check.bounds i n` immediately before
  `addr.elem p i` (`n` = `slice.len p`, or the array length as a constant); `a[i]` on an array SSA
  value is `get`/`set`, which check a dynamic index themselves. A constant index ≥ a constant array
  length is `panic.bounds ()` + `unreachable` (the only index folding; the trap is kept). No check is
  ever removed in the checked IR; release code generation may drop a check only with a recorded
  proof (constant index < length, a dominating identical check, a dominating `lt i n` branch,
  and `LenEq`/`Offset`: `i + k < n` derived from dominating unsigned compares, asserting calls
  such as `std.debug.assert`, and trapping `add`/`sub` of constants).
  Compiler-made accesses (`for` over an array or slice, drop loops, array → slice coercion
  `addr.elem p 0` + `slice.make`) index below a length the generated code compares against.
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

## Types and fntype

- `type` lines list the transitive closure, over key components, of every type a printed function
  references: its signature (param, return, and type-argument types), its block-param and value
  types, its instruction result types, and its type operands. By canonical id T.
- `fntype` prints only the features not implied by another member of the set (the set is closed
  under implication: `needs(avx2)` prints `avx2`, not `sse42 avx avx2`), in contracts/simd-ops.md
  §Features order. In the canonical type order a feature set compares as its `u64` bitset (bit i
  = the i-th feature of §Features) read as a number.

## Closures

A closure or `spawn` body is a function `@<enclosing>.<k>` whose `Path` is the enclosing fn's
path (`fn @7.0 app.apply` after `fn @7 app.apply`); `k` numbers the closure literals and `spawn`
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
  needing another arch's features; `test fn`s only under `stave test`), plus every instance they
  request, transitively.
- **release**: only instances reachable through calls and fn-value uses from the roots `main`,
  `extern "C"` exports, the `allocator` impl, and, under `stave test` (also `stave test --release`),
  `test fn`s, plus the runtime entry points the compiler's own startup, trap, and signal code calls
  (contracts/intrinsics.md "Paths fixed by the compiler": `runtime.start.entry`,
  `runtime.panic.trap`, `runtime.panic.trap_msg`, `runtime.panic.on_signal`). A reachable
  `dispatch` makes its arm targets reachable (an arm of another arch is not).
- Generic decls are never printed, only their instances. `kernel fn`s are printed like fns; GPU ops print as
  `gpu.<name>` (gpu-ops.md "Checked-IR spelling"). Intrinsic calls print as `intrinsic.<name>`.
