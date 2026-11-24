# Contract: language details behind `LANG.md`

Fine print for rules that `LANG.md` states in one sentence: output formats, per-case lists, and
reasons. Sections are named after the `LANG.md` section the rule is in; `LANG.md` stays the
reference for what a program may write and what it means.

## §2 Fn types

- A closure may return a borrow of a capture it holds by `&`, but not of one it holds by `&mut`,
  even through a dereference (E0512): the next call could change it or lend it again.
- A closure value with captures points to its environment (the captured places' addresses),
  which is in the frame of the body that evaluated the literal: a value holding it is not
  returned from that body (E0512 case `env`), also when every capture is one an enclosing
  closure holds by `&` (`fn() -> fn() -> u64 { return fn() -> u64 { return x; }; }`). Write the
  inner closure where the captured locals live, or return a value. What a call of the inner
  closure returns may leave if it holds no fn value (`return get();` with `get` returning `&x`, a
  slice, an `Option<&T>`): captures are references, so it points to the captured places.
- A `fn(..) needs(F)` value is usable as `fn(..) needs(G)` only if F ⊆ G ∪ the enclosing fn's
  needs ∪ the target baseline (E0707). Reason: CPU features never change while a program runs, so
  a value made where F is proven stays callable. A call through a value whose type keeps
  `needs(G)` needs G proven where the call is, as a direct call does (E0704 case `value`).
- No operator applies to a fn value: the operator traits and fn types both belong to `std`, which
  implements none for a fn type, so `impl core.Add for fn()` elsewhere is E0420 and `f += g` on a
  fn-typed `f` has no impl. An operator over fn values goes through a Copy struct that holds them
  (`view struct H { f: fn() }`, `impl Copy for H {}`, `impl core.Add for H`, §2 Operator traits):
  `H.f` is a field declared with a fn type,
  a tracked slot of the call graph (contracts/soundness.md), fed by every `H` literal, the impl's
  results included.

## §1 Closure capture modes

- A closure holds a capture by `&` only when every use of it in the body (nested closures
  included) is a read, whatever `*`, field, element, or auto-deref steps lead from the local to
  the use: a `&self` or by-value `self` method receiver, a built-in method's receiver, a fn value
  called, the operand of `&`, an f-string hole, a by-value use of a type other than `&mut T` (a
  `&mut T` passed as `&T` counts), the scrutinee of a `match` or destructuring `let` whose type
  holds no `&mut` outside a `&`, or the sequence of a `for` over a range, an array value, `&[T]`, or
  `&[T; N]` whose bindings hold no `&mut` (an array or `[T]` place is no source: E0219). Any other use (`*t.0 = ..`, `*a[0] += ..`,
  `w.v.push(..)` with `w.v: &mut Vec<T>`, `&mut`, a `&mut` chunk or element binding, a `for`
  over an iterator) makes it `&mut`, and so does a `&mut T` type. The thread rules (E0802) and
  the borrow checker read the same mode. Reason: a mode the thread rule and the borrow checker
  derive separately can disagree, and a `&` capture written from many threads races.

## §2 Type nesting depth

- A type without components (a scalar, `str`, a mask, a generic parameter) is 1 level deep. A
  reference, raw pointer, array, slice, tuple, fn type, vector, or range is one level above its
  deepest component; a struct or enum is one level above its deepest type argument and its
  deepest field (after substituting its type arguments). A pointee behind `&`, `*`, a slice, or a
  fn type adds nothing to its holder's depth: it is measured as a type of its own.
- Deeper than 1,000 levels is E0005 case `type`, wherever the type comes from: written, through
  an alias, by the fields of a generic struct (`struct D1<T> { x: D0<D0<T>> }` doubles), or built
  by a generic instance (`W<T>` in `f<T>` called with a 1,000-level `T`), also in a family of types
  a pointer field instantiates without end (`struct P<T> { n: *const P<(T,)> }`). A type deeper
  than the limit is reported once, where it is first reached, not again in the types holding it.
- Reason: lowering, layout, and the IR handle types up to that depth, and every recursion of the
  checker over type structure (trait queries, `size_of`, drop analysis, substitution) stops at the
  same limit and answers the rejecting side past it (not Copy, not Send / Sync / Flat / Frozen, no
  layout: E0911, needs a drop).

## §2 Size limits

Each limit is decided in one place; every reader of a size takes a size past `u64` as a reject.

| limited | bound | diagnostic | decided in |
|---|---|---|---|
| height of a type (a type is one level above its tallest component; written, from an alias, or built by substitution) | 1,000 | E0005 `type` | the type table: a taller type is never interned (the error type); the refusal is reported at the span that owned it (the written type, expression, statement, body, declaration, or call-graph instance being checked), and `typedepth` reports instances whose body types would pass it. A closure's environment `&mut (&mut T0, ..)` is such a type too (`closures::env_type`), so a capture more than 997 high is E0005 |
| bytes of a type | `u64` | E0911 `overflow` (a value of it) | `traits::type_bytes`, the one size function |
| compile-time data: every tuple, array, struct, or variant value a const-expr builds (so every `const` and `static`) | 2^26 bytes | E0904 `size`, at the literal | `consteval::data_size`, on the literal's type before the value is built |
| all compile-time data of a program: every `static` and `const` (file data and zero fill, each rounded up to its alignment) | 2^30 bytes | E0904 `total`, at the program's largest one (unused `const`s count; so do the built-in packages') | `typeck::data_total` |
| a value a fn holds: parameter, binding, array / tuple / struct literal, value `if` / `match`, return value | 2^28 bytes | E0911 `size` (concrete), `instance` (per instance) | `typeck::frame_of` |
| frame sum of a fn instance: its parameters, bindings, and return value; every expression value but a place read in place (a local, `self`, a `static`, a part of a place, a dereference that is borrowed, assigned to, split into parts, or initializes a binding), two slots for a value `if` / `match` (its result and the live arm, whatever the arm count); each call's result, by-value receiver, and arguments, whatever the callee (fn item, method, fn value, closure, built-in) | 704,643,072 bytes | E0911 `frame`, per instance naming it | `typeck::frame_of` |

- `[e; n]` is one value repeated, in const evaluation (never `n` values) and in a fn body (a
  fill at run time: `e` once, stored into each element; no count limit of its own). A const
  used twice is one value (`(C, C)` costs a step, not a copy), so doubling consts stop at the
  data limit without building it. An all-zero `static` takes no file bytes; a const of more
  than 64 value cells is read from its image where it is used.
- Reason: const evaluation and the image hold compile-time data. Both targets reach a static
  PC-relatively from code laid out before all data: x86-64 by a `rel32` (±2 GiB), AArch64 by
  `ADRP` (±4 GiB); half of the x86-64 range is kept for code and the rest of the image, so the
  program's data bound is 2^30 bytes on both. A back end keeps each value of
  a fn in a slot of its frame, and a frame holds at most 1 GiB: the measured frames are at most
  1.5 times the frame sum (a filled local, a whole aggregate loaded to read a part), on x86-64
  and AArch64, debug and release, so the frame bound is (1 GiB - 16 MiB) / 1.5. A trait query
  has no depth limit: a goal met again while in progress is cut, and the types are finite. The
  height bound makes the types of a program finitely many, so every walk over them with a
  visited set ends; polymorphic recursion (`P<T>` holding a `P<(T, T)>`) grows one level per
  instance and stops there.

## §2 Contracts

- A false `requires` panics at the call: the last call user code made into a fn with contracts, a
  `dispatch fn`, or a fn value. A false `ensures` panics at the `return`. The stderr line after
  the panic line is the clause's source text.
- Each operand of a clause's top-level `&&` is its own check.
- The compiler proves checks from what holds before them: earlier clauses, branches, type ranges,
  `+` / `-` by constants, widening casts.
  - A check that always holds is removed. A `requires` implied by the clauses before it is
    warning E0216 (not in generic fns).
  - A check that fails whenever reached is an error: E0215 for a `requires` (the fn can never be
    called), E0217 for an `ensures`, at that `return`. Neither is reported in generic instances
    or in provably dead code.
  - Any other check stays a runtime check.
- Clauses that compare int params, slice param lengths, and constants also hold across calls: a
  direct call that proves every `requires` of its callee skips them, a call that always fails a
  clause is E0218, and after a call its callee's `requires` and such `ensures` are known.
- Effects: a kept `ensures` check gives its fn `trap`; a kept `requires` check gives `trap` to the
  callers that do not prove it, not to its fn: `requires(..) without(trap)` means no trap when
  called inside the precondition, and such a fn is no value of a fn type that denies `trap`
  (E1403).

## §2 Deny clause: `block` and `park`

- `block` means "may wait for something outside the fn", not a fixed list of std fns, so a new
  kind of wait in std needs no language change. In std: a `scope` end, `thread.join`,
  `par_*`, and the file reads, writes, and opens of `std.io` (a pipe or FIFO may wait on another
  process or thread; std/README.md lists them).
- `park` is a wait on a completion queue. It is a part of `block`: whatever has `park` also has
  `block`, and `without(block)` denies `park`. So `fn(..) without(block)` is usable as
  `fn(..) without(park)`, not the other way (every fn type denies `park`, below). `without(park)` alone allows other waits: a stage
  may join threads (`par_for`) and still never wait on a queue.
- A fn type never has `park`: every `fn(..)` type denies it, so `without(park)` on a fn type is
  accepted and changes nothing (`fn(..) without(park)` is the type `fn(..)`), and a closure or fn
  item with `park` made into a fn value is an error (E1403 case `made`). Reason: a fn that parks
  keeps its frame while it waits, and its caller lays that frame out; a call through a fn value
  does not know the callee's frame size. So an `extern "C"` import and code the call graph does
  not show, which have every other effect, do not have `park`, and neither does a `with_lock` or
  `par_for` body.
- A call cycle may not contain a fn with `park`: a `rec fn` cannot wait. Reason: the frames of
  waiting fns nest statically, which a cycle breaks. A `rec fn` called from a waiting fn is
  allowed while the cycle itself cannot wait.
- No std fn has `park`, so nothing has it: neither rule above, nor a `without(block)` fn calling
  a parking fn, nor a denied `park` (E1401), has anything to check or a negative conformance
  test.

## §2 Operator traits

The operator traits of `std.core` are `Add WrapAdd SatAdd Sub WrapSub SatSub Mul WrapMul SatMul
Div Rem Shl WrapShl Shr BitAnd BitOr BitXor Neg Not`; a compound assignment (`a += b`) uses the
trait of its operator, and the comparisons, `!`, `&&`, and `||` are no traits. Every method is
`fn op(self, rhs: Self) -> Self` (unary `fn op(self) -> Self`; E0905 keeps an impl to it), so
every operand and the result have the impl's self type. That type must be Copy in every instance
the impl covers (E0422), decided by the query that decides Copy in a generic body: the impl's
params are rigid and carry exactly the impl's bounds. `impl<T: Copy> Add for W<T>` with
`impl<T: Copy> Copy for W<T>` passes; `impl<T> Add for W<T>` does not, nor `impl<T: Copy> Add
for F<T>` when F's `impl Copy` asks `T: Copy + Frozen`, nor an impl for a type (or an instance,
`impl<T: Copy> Div for K<T>` with only `impl Copy for K<u32>`) that has no `impl Copy`. Reason:
an operator is a call that is not written as one, and on operands that are not Copy it would
move them unseen (`d + d`, `a += b` out of a place that cannot be moved from, the old value of
`a` dropped inside the call). A type that owns memory (a big integer, a heap matrix, a string)
takes a method: `fn add(&self, rhs: &Self) -> Self`, called as `a.add(&b)`.

An operator copies its operands, so every operand is Copy where the operator is used (E0422
case `operand`, at the operator), by the Copy query of the checked fn's own bounds. A bound on an
operator trait does not imply `Copy`: in `fn f<T: Add>(a: T, b: T)`, `a + b` is E0422 and the
fix is `T: Add + Copy`. The rule covers binary and unary operators and compound assignment, an
operand behind a reference (`*r + e`, `*r += e`), a field, an element, a capture, and a type
whose operator impl is itself E0422. With it no operator moves an operand: the borrow checker
reads operator operands as copies, and the lowering drops nothing for them.

## §3 Operand order and moves

Every expression with several operands evaluates them left to right, each to a value, before it
acts: a call (direct, method, through a fn value or closure: the fn value, then the receiver,
then the arguments), a struct, tuple, array, or variant literal (fields in source order), a
binary operator (left, then right), `c.select(a, b)`, and `p op= e` (below). A by-value
operand whose type is not Copy is moved when it is evaluated, so every later operand sees the
place moved: using it again, a part of it (`s.n`, Copy or not), or borrowing it there is
E0502 (case `operand`), also inside a nested call, a closure literal, or an f-string hole. A
Copy operand does not move: a later operand may still use the place (`f(s.n, s)` is fine; the
checker takes the Copy value as read at the call, which may report a conflict a later operand
makes with it).
The fix names the value first: `let n = s.n; f(s, n);`.

An assignment `p = e` evaluates the operands of the place `p` first (an index `xs[i]`, the call
a dereference `*pick(&mut a)` or a field `sref(&mut s).f` goes through), then `e`, then makes
the moves of `e` and writes `p`: `xs[i] = bump(&mut i)` writes the element of the old `i`, and
`s = make(s, dup(&s.d))` is E0502 at `&s.d`. `p op= e` is `p = p op e`: the value of `p` is a
by-value operand evaluated after the place's operands and before `e`. It is a copy (§2
Operator traits), so `e` may use `p` and may change it: `c += set(&mut c, 4)` adds the old value
of `c`.

## §3 Whole moves and destructuring patterns

A non-Copy value moves whole. A by-value use of a non-Copy part of a place or of a temporary
outside a pattern (`let x = s.f;`, `f(t.0)`, `mk().a`, `s.v.into_iter()`) is E0504 case
`part`; Copy parts are read freely, borrows of parts (`&s.f`,
`&mut s.f`) and assignments to parts (`s.f = v`, which drops the old part) are unchanged.

Which patterns move. A `let`, `match` arm, or `for` pattern moves its scrutinee if it binds some
non-Copy part (at any depth) by value. Every alternative of an or-pattern binds the same names
with the same modes, so either all of them move or none does. A pattern that binds only Copy
parts, binds by reference (through a `&` / `&mut` scrutinee), or binds nothing (`_`, a literal,
`let _ = s;`, `match s { _ => .. }`, `S.{ n, b: _ }` with `n: u32`) only reads: the scrutinee
stays alive and is dropped with its owner. A struct pattern, also of a struct-like variant's
payload, names every field (E0208 case `pattern-missing-field`) and each field once (E0105), so
each part of the value has exactly one written sub-pattern: it is bound, or ignored, never both,
never neither. This holds for a pattern through a reference too, which moves and drops nothing:
a field added to a type makes every pattern over it say what happens to the field. There is no
rest marker; `f: _` ignores one field.

Decided on the definition. "Non-Copy" is the type as written where the pattern is: in a generic
fn a part of type `T` is Copy only under a `T: Copy` bound. So `let G.{ a, d: _ } = x;` with
`a: T` moves `x` and drops `d` at the `let` in every instance, also for `T = u32`, where the
same pattern in a fn written for `G<u32>` only reads and leaves `d` to the end of `x`'s scope.
The type checker decides this once per binding; the borrow checker (what is moved, which
borrows and lock guards end at the binding point) and the code of each instance read that
answer, so an instance never keeps alive what the check saw end.

When and where. A moving pattern takes the whole value at its binding point: the end of the
`let` statement's pattern, the entry of the arm (after the guard passed), the start of each
`for` iteration. There the bindings take their parts, the scrutinee becomes moved as a whole
(a local is moved; a temporary is consumed, never dropped as a whole at its statement end), and
each non-Copy part the pattern does not bind by value (a `_`, so `f: _` for a field, a
sub-pattern that binds nothing such as a literal) is dropped there,
before the arm body or the next statement, in declaration order of the fields, a nested part
in place of its field (depth first); of an enum, only the matched variant's payload. No state
"moved except a part" remains after the binding point.

1. `let S.{ a, b: _ } = s;`: `a` takes `s.a`, then `s.b` is dropped, at the `let`; `s` is moved.
   With a temporary (`let S.{ a, b: _ } = mk();`) the same: `b` is dropped at the `let`.
2. `match s { S.{ a, b: _ } => .., other => .. }`: in the first arm `b` is dropped at arm entry
   and `s` is moved; the second arm moves `s` into `other`. A temporary scrutinee: the same;
   an arm that does not move it leaves it to be dropped at the end of the `match` statement.
3. With a guard (`S.{ a, b: _ } if p(&a) => ..`): inside the guard the names are read-only
   views of the scrutinee, nothing is moved or dropped; if the guard fails the value is intact
   for the next arm; the move and the drop of `b` happen only when it passes. A guard only
   reads (E0505), so the value it sees is the value the arm takes. A guard that would move,
   write, or `?` is a test in the arm: `3 if eat(o) => a` is `3 => { if eat(o) { a } else {
   .. } }`, the `else` standing for the later arms.
4. Nested patterns and payloads: `E.Two(a, _)` drops the second payload at arm entry;
   `W.{ p: P.{ x, y: _ }, q: _ }` drops `p.y`, then `q`. An or-pattern drops what the
   alternative that matched ignores.
5. `for P.{ a, b: _ } in src`: each iteration's element is moved into the pattern at the start
   of the iteration and its `b` dropped there. An item the pattern only reads (`for _ in src`,
   `for H.{ n, d: _ } in src` with `n: u32`) is a value of its iteration and is dropped when
   the iteration ends (also by `continue`, `break`, `return`, `?`). The elements not reached
   are dropped by the loop's source: an iterator by its own drop, an array value by the loop
   when it ends or is left, in order, after the item of the iteration that left.

A value whose type has an `impl Drop` is never taken apart, by a pattern either (E0504 case
`drop`): its destructor runs on the whole value.

## §3 Storing calls: a view left borrowed

A storing call (a call of a fn whose `&mut` view param has a `from` clause) for a view it reaches
by a `&mut` borrow that is among what it stores leaves the whole view borrowed for as long as the
view is used (E0501 on a later conflicting use), because the callee may split that pointer and
store one part through the other. The cases:

- the source `&mut b` (after `set_link(&mut l, &mut b);`, `b` is not used again);
- a receiver the clause names (`b.give(&mut l)`);
- a view `a` held as `&mut a` by a view whose content is stored (`from copy l`).

The caller rule (every view the target reaches holds all the clause's sources from the call on)
applies also when the target holds no borrow at this instance (`Slot<u32>`): the sources still
take the store.

## §3 A borrow of a static

A `&` of an immutable `static` or a part of it (`&A`, `&T[i]`, `&S.f`, the auto-ref of a
`&self` method call on it) borrows nothing: the storage never ends, is never moved from, and
changes only through interior mutability, so the reference is a constant like a string literal,
wherever it flows: returned without `from`, stored through a `&mut` param, into a captured
reference, a struct field, an `Option`, or a value another thread reads. Still rejected: `&mut A`
(E0503), a `static mut` (E0809), a local or param that shadows the static, and a `const` (a
temporary) are borrows of locals (E0512, E0513).

## §3 Drops

Every value with drop glue is dropped exactly once, on every path, by the owner that holds it
when that owner ends.

- **Whole values.** A value is moved whole or taken apart by a pattern ("Whole moves and
  destructuring patterns" above): the parts a moving pattern ignores are dropped at its
  binding point, and nothing of the value is left to its former owner.
- **Assignment.** `x = e;` evaluates `e`, drops the value `x` holds, then writes; a field
  (`p.a = v;`) likewise drops the field's old value. The drop is a use of what the old value
  borrows when its type has drop glue: those borrows are live up to the assignment (E0501 where
  a borrowed local ends before it, and where the old value borrows `x` itself). An `x` that
  holds no value there on any path (moved out, not assigned since) drops nothing and uses
  nothing. A captured place assigned inside a closure or `spawn` body is no exception: a
  capture always holds a value when the body runs, so each run of the assignment drops the
  old value (`o = Option.None;` in a closure releases a guard `o` held).
- **`impl Drop`, order inside a value.** A value whose type has an `impl Drop` is dropped as a
  whole: its `drop` runs once, then its fields are dropped, in declaration order (tuple and
  array elements in index order; of an enum, the fields of the value's variant). No pattern
  takes it apart; one that only reads it (`match mk() { F.B => .. }`) leaves it to be dropped
  where its owner ends. A part is alive until its turn: a `MutexGuard` in a field is held while
  the type's `drop` and the drops of the fields before it run ("Locks and completion waits").
- **Compound assignment.** `p op= e` drops nothing: every operand of an operator is Copy where
  it is used (§2 Operator traits), so the old value of `p` has no drop glue.
- **Assignment targets.** The target of `=` and of every `op=` is a place rooted in a local, a
  `static`, or a dereference (`x.f`, `a[i]`, `S0.n`, `*r`, `f().x` with `f` returning `&mut S`).
  A target rooted in a temporary (`mk().a`, `[1u32, 2u32][0]`, `S.{ a: 1u32 }.a`, `arr()[0]`,
  `(if c then a else b).f`) or in a `const` (`C.a`, `C`) is E0503 case `temporary`, with or
  without drop glue: the statement drops the value, so the write would be lost. Bind it with
  `let mut t = mk();` first. A `&mut` of a temporary is a borrow, not a target: `mk().set(1)`
  (a `&mut self` call), `f(&mut arr()[1])`, `*&mut arr()[0] = 5` write through the reference
  into the temporary, which the statement then drops.
- **Consts.** A `const` is a new copy of its value at each use, so its type has no drop glue
  (E0413 case `drop`, at the declaration): a destructor would run at each use of the name, on a
  value no binding owns. One value with a destructor is a `static` (never dropped); a new one
  at each use is a fn call (`fn k() -> Q { return Q.{ n: 1 }; }`).
- **Temporaries.** A temporary with drop glue is dropped at the end of its statement, the last
  made first, and only if the operand that makes it ran (the right side of `&&` / `||`). These
  are statements of their own for this rule: the condition of an `if` or `while` (each
  evaluation), a `match` guard, a branch of a value `if` (`then`, `else`; an `else if` is a value
  `if` in the `else` branch), the body of an arm of a value `match`, and each check of a contract
  clause (an operand of its top-level `&&`) and the `e` of `old(e)`. A branch or arm drops its
  temporaries before its value goes to the rest of the statement. The
  temporaries of a `match` scrutinee and of a `for` header belong to the `match` / `for`
  statement: they live across the arms and the loop, and are dropped once after it. The
  iterator value a `for` is given is moved into the loop and, with drop glue, dropped once when
  the loop ends or is left ("`for` sources"). An array value a `for` is given is the loop's too:
  each element is handed to its iteration, and the elements the loop has not handed out when it
  ends or is left are dropped there, in order. Both belong to the loop's scope, around the
  body's: a `break`, `return`, or `?` in the body drops the body's locals first, while the
  iterator and the rest of the array are alive.
- **Rvalues under patterns.** An rvalue that a pattern only reads (`let P.{ n, a: _ } = mk();`
  with `n: u32`, a scrutinee that is a call in an arm that binds nothing by value) is a
  temporary of the statement and is dropped at its end. `let _ = e;` with an rvalue `e` drops
  it at the end of the statement; with a place `e` it moves nothing and drops nothing. A
  scrutinee one arm takes (by a name, `x if c => ..` or `other => ..`, or by a moving pattern)
  is that temporary on every way through the `match` that leaves it: a guard that fails,
  another arm. What it borrows stays borrowed there, to the end of the statement. The value a
  `for` iteration is given is the iteration's when the loop's pattern does not move it: it is
  dropped where the body ends or is left.
- **`select`.** `c.select(a, b)` evaluates both operands; the one it does not pick is a
  temporary of the statement, made when the `select` is evaluated. Which operand that is is
  known only when the program runs: what either operand borrows stays borrowed to the end of
  the statement, and the temporary's drop is checked as either operand's.
- **Leaving early.** `return`, `break`, `continue`, and `?` drop what they leave in the order
  of its normal end, innermost first: scope by scope from the innermost, the temporaries of the
  statements opened in it (each statement's after the scopes inside it), then its locals. So a
  `match` scrutinee's or a `for` header's temporaries are dropped after the locals of the arm or
  body that is left, and the locals of an arm or body end before the temporaries of the
  statement around it: a borrow of such a local that such a temporary holds, read by its drop,
  is E0501 on every way out of the arm, as at its end. A `break` or `continue` leaves only the
  statements inside the loop's body, so the temporaries of a `for` header are dropped after the
  loop. `?` also drops the operands
  of the enclosing expressions that were already evaluated (earlier arguments, a by-value
  receiver, earlier fields of a literal), the last evaluated first.
- **One schedule.** Where each drop runs is decided once per body, as its drop schedule
  (stave-sema `drops`): one row per drop, naming the point (the owner's end, the exit taken, a
  pattern's binding point, an assignment) and what is dropped there (a local, a temporary of a
  statement, the part of a pattern with a given index, an overwritten value). The rows are made
  with the borrow checker's control-flow graph, one per drop statement, so the borrow check,
  the move check, the lock-rank and join rules read exactly the scheduled drops; the code
  generator asks the schedule for every drop it emits, and a drop without a row, or the drops
  of one end or exit in another order than their rows, stop the build as a compiler error. So
  does a row the generator never decided: every scheduled drop must have been emitted or left
  out for a stated reason (the owner holds no value there, the type has no drop glue in this
  instance) where the generator lowered that point; a statement lowered at each return (an
  `ensures` clause) does so in every copy. A row is one value: a temporary is named by its
  statement and the first and last token of the expression whose value it holds (through
  parentheses; a step of a postfix chain by the chain up to the step's last token, the item of
  a `for` iteration by the `for` token, the operand `select` does not pick by the method's
  name), which both sides compute from the syntax, so `&e` and `e` are two values; and each row
  is consumed by at most one emitted drop. The parts a pattern ignores and the steps of a value's drop glue are each
  computed by one function both sides call. Whether a scheduled drop runs is the owner's init
  bit: one per owner, a temporary included.

## §3 Temporaries end with their statement

A temporary is the place an rvalue gets when its address is needed: `&e` and `&mut e` of a value
that is no place, an auto-referenced receiver (`f().m()`), an element or field of such a value
(`&mk().a`, `&arr()[1]`), and an rvalue with drop glue ("Drops"). Every temporary ends with its
statement (the statements of "Drops", "Temporaries"), after the drops of the ones with drop
glue; a `break`, `continue`, `return`, or `?` that leaves the statement ends it there. A borrow
of a temporary may be used inside its statement; one used after it is E0501 case `temporary`,
whatever holds it: a `let` (`let r = &f();`), an assignment, a field of a literal (`S.{ r: &f()
}`), a tuple or array element, a call or method result that borrows it (`let r =
mk().first();`), a binding of a `match` over `&mk()`, a value `if` / `match`. Write `let t =
f(); let r = &t;`. A branch of a value `if` and an arm of a value `match` are statements, so
`ok(if c then &f() else &x)` is E0501 too: bind `let t = f();` before the `if`. There is no exception for constants: `&4u32`, `&[1, 2, 3]`, `&S.{ a: 1 }`,
`&()`, `&C` of a `const` item (no drop glue, "Drops"), `&size_of::<T>()`, and a variant without payload make a temporary
at each evaluation, as `&f()` does; data that outlives the statement is a local or a `static`. A
string literal is no temporary: it is a `&str` to static data. A `static` is a place.

The header of a `for` and the scrutinee of a `match` belong to the `for` / `match` statement, so
their temporaries live through the loop and the arms: `for x in &mk()` and a `for` over an array
value (the loop's own temporary, "`for` sources") end with the loop, which is their statement,
not past it.

## §3 Must-consume

A pending-operation type of std stands for work that is still running (device work, a queued
operation). Dropping it would lose its result or error and free what the work still uses, so it
has no implicit drop: a value is consumed only by passing it to a method that takes it by `self`,
and an implicit drop on any path is an error, also the early return of `?`. The types are a closed
list of fixed paths in contracts/intrinsics.md; each has its own diagnostic (`gpu.Launch`: E1104).
No syntax marks such a type.

## §3 Locks and completion waits

Locks are ranked so that no lock-order cycle can form: safe code cannot deadlock on locks and
joins. The rank rule (LANG.md §3, E1405) is checked at every drop of the drop schedule ("Drops"), not
only at calls: a drop runs the dropped type's drop glue, which may take a lock. Two sets of
guards are held there. The guards of the other values alive at that point; and the guards
inside the dropped value itself, each until the glue reaches it (case `own`): `(L, guard)`
drops `L` while the guard is held, `(guard, L)` releases the guard first; a type's own `drop`
runs while every field is alive. A guard a `vec.Vec` owns counts as held while the `Vec` is
dropped. A destructor that joins threads (a `scope` in `drop`) takes what those threads take.
The steps are read from the type itself, for any type (a tuple or array made in place too); a
step the check cannot follow counts as taking any lock (case `opaque`). Array elements are
judged by their type: `[E.G(guard), E.K(l)]` is rejected although this order is safe, since the
check does not know which element holds which variant.

Completion-queue waits (`park`) are `block` waits: like file waits they are outside the claim that
safe code cannot deadlock on locks and joins. No `MutexGuard` may be held across a call that has
`park`: the guard stays with the OS thread, another task on that thread that wants the lock
sleeps the thread, and the task that holds it never resumes. No std fn has `park`, so no
diagnostic checks this rule.

## §3 Scope threads

A `scope`'s threads run until it is left: after a `spawn`, its captures stay borrowed on every
path through the block, also on one that never leaves it (a `-> never` call, a loop without
`break`), so main writing, moving, or `&mut`-borrowing a captured local there is E0501.

## §3 Error trace

`Result.unwrap` of an `Err` prints the task's trace under the panic line, oldest first:

- `  from <site>`: the `Result.Err(..)` that started it;
- `  (<k> earlier via sites not kept)`: past 32;
- `  via <site>` per `?` that passed an `Err` on (the last 32).

`<site>` = `<file>:<line>:<col> in <fn>`. Each evaluated `Result.Err(..)` in a user package
restarts it (std restarts it at the user site where it makes an error: I/O, `AllocError`,
`ok_or`); `?` on a `Result` adds its via before the scope's drops. One trace per task (main, each
`spawn` body, `thread.join` arm, `par_*` body call; a task leaves its caller's trace as it was): the
task's latest error, not necessarily the unwrapped one. The trace is out of band: `Result` layout
and fn ABI are unchanged, and it has no effect (`without`).

## §3 A `&mut` place cast to a raw pointer

- `p as *mut T` with `p` a `&mut` place reads `p` and makes the pointer as the reborrow
  `&mut *p` would (borrow-model M6, M11): it needs what that reborrow needs, so behind a `&`
  (`r.m as *mut T` with `r: &S`, `*r as *mut T` with `r: &&mut T`) it is E0503
  `shared-ref-cast`, as `&mut *r.m as *mut T` is E0503 `shared-ref`. Reason: a `&mut` reached
  through a `&` is read-only (M7), and a `*mut` made from it may not write.
- `p as *const T` reads the place: it is allowed behind a `&`.
- A cast chain in one expression is judged by its final type: `r.m as *const T as *mut T` is
  the `*mut` cast (E0503 behind a `&`). A `*const T` stored first and cast later is accepted
  by the checker; a write through the pointer made from it is reported by borrow tracking (the
  pointer carries the `&` tag of the read, M7).

## §3 Reasons

- Slice methods of `std.slice` are SIMD inside: regular code, explicit SIMD stays in std, no
  auto-vectorization cliffs.
- Release builds treat `&mut T` as noalias and `&T` (T `Frozen`) as read-only: borrow rules
  become free optimizer facts.

## §3 Unspecified results

- No op with an int or `bool` result is on the closed list: such a result sees an unspecified
  value only through a float's bits (`mem.f32_bits`, `bitcast`) or the effects of `std.slice`
  predicates, `par_for` / `par_for_mut` bodies, and threads.
- Tests (conformance goldens, cross-target and CPU / GPU comparisons) count every NaN equal to
  every NaN and compare everything else bit for bit (`-0.0` is not `0.0`).

## §4 `for` sources

The kind is decided once, by the type checker, from the source's type as written in the
definition (`stave_sema::loops::for_source`), and stored with the loop; the borrow checker, the
capture analysis, and the lowering of every instance read the stored kind (and an iterator's
resolved `next`), never the instance type. The list is closed: a type not listed is no source.

| source type | element loop `for x in s` | chunk loop `for v: TxN in s` / `v: &mut TxN` |
|---|---|---|
| a range `a..b`, `a..=b` | the values | E0219 `not-source` |
| `&a`, `&mut a`, a `&[T]` / `&mut [T]` / `&[T; N]` value | element `i` of `*s` borrowed per iteration: `&T` / `&mut T` | the chunks of `*s`; `&mut v` needs a `&mut` source (E0503 over `&`) |
| an array value (a temporary: a call, a literal, a const, a value `if` / `match`, a cast) | the loop's temporary: each element is moved out to its iteration, the ones not reached are dropped right after the loop on every path out (end, `break`, `return`, `?`) | the temporary, read only (`&mut v`: E0219 `mut-chunk`) |
| an iterator value or place (`it`, `mk()`, `s.it`: a type with `next(&mut self)`) | moved into the loop (copied if Copy) at the head; the loop owns it, advances it, and drops it right after the loop on every path out (end, `break`, `return`, `?`); a place is unusable after, as after any move (E0502, E0504 behind a reference) | not a chunk source |
| `&mut it`, `r: &mut It` (also a `&mut It` view field) | `*r` advanced in place: after the loop (also after `break`) the iterator has moved on; the loop drops nothing | not a chunk source |
| `&it`, `r: &It` | E0219 `shared-iter` (`next` takes `&mut self`) | not a chunk source |
| an array or `[T]` place | E0219: write `&a` / `&mut a` | E0219 |
| a reference to a reference (`&mut &mut It`, `&&mut It`, `& &[T]`, `&mut &[T; N]`, ..) | E0219 `ref-ref`: iterate the inner reference | E0219 `ref-ref` |
| any other type (a scalar, vector, mask, `str`, tuple, raw pointer, fn value, or a reference to one) | E0219 `not-source` | E0219 `not-source` |

The source is evaluated once, at the head, and the loop holds that value to its end, the tail
clause included: a reference source is a loan of its referent for the whole loop (`&mut` when the
loop writes through it: `&mut` elements or chunks; else `&`), whatever the body then assigns to
the place it was read from (`s = &b;` in the body is legal; writing, freeing, or `&mut`-borrowing
the old referent while the loop runs is E0501, also through `s` before it is reassigned). The
checker reads that one value, as the code does.

Chunk loop or element loop is decided once, by the type checker, on the definition, and stored
with the loop: a chunk loop is a `for` whose annotation, as written, is a vector type or a
reference to one. A generic `T` or `&T` annotation makes an element loop in every instance, a
vector `T` included. A tail clause belongs to a chunk loop only (on any other loop: E0601
`element-loop`). `tail scalar` runs its body once per element left over; each pass is an
iteration for the borrow rules (a borrow made in one pass and used in the next is live in
between).

An iterator type is a struct, enum, or generic parameter whose `next` (method lookup, for `T`
its bounds) is exactly `fn next(&mut self) -> Option<X>`: the loop passes the source and nothing
else. Another receiver (`&self`, `self`, none), another param, generic params of its own, or
another result is E0219 `next-shape`; no `next` at all is E0219 `not-source` (a `vec.Vec` or a
reference to one: `as-slice`, iterate `.as_slice()`). Two `next`s (two traits, or two instances
of one trait: `T: Nx<u8> + Nx<u16>`) are E0908, as for a call, when one of them has that shape.
The loop's call of `next` is checked as `recv.next()` is, through the same check of a method
call: its `needs` against the enclosing fn's (E0704), its privacy (E0106), an `unsafe fn next`
outside `unsafe(..)` (E0803: the whole `for` must be inside, as the calls happen at each
iteration; the source alone in `unsafe(..)` is not enough), the instance's bounds; it is an edge
of the call graph (effects, `without`, `rec`, lock ranks) and its `requires` runs at its entry.
A generic `T` (or `&mut T`) source is always an iterator through its bound's `next`: each
instance runs that trait method, whatever the instance type is (`T = [u64; 2]` does not become
an array loop, `T = &mut It` runs `impl Next for &mut It`); a `T` without such a bound is E0219
at the definition. Element and binding types, a sequence's length, and the resolved
`next` come from the instance; the kind, by reference or by value, and the chunk width come from
the definition. A source behind a view field is classified by the field's
type (`v.s: &mut [T]`: a reference to a slice; `v.it: &mut It`: a `&mut` to an iterator).

A *place* is: a local, param, `self`, or `static` (also through a module path, `m.TABLE`); `*e`;
an expression whose last step is a field, tuple field, element `e[i]`, sub-slice `e[a..b]`, or
`std.soa` column `ps.x` (of any base, `f().a` included); such an expression in parentheses or
after `unsafe(..)`. Reason: a loop over a place reads (or writes) it while the body may change
it; a reference makes the borrow visible and gives it one meaning in the checker and the code.

By value: `for x in &a { let v = *x; .. }` reads each element of a Copy `T`. To move the
elements of an array place out, make it a temporary: `for x in mem.replace(&mut a, b) { .. }`
(or pass it through a fn that returns it). A `vec.Vec` gives up its elements by move through
`pop()`, last first: `loop { match v.pop() { Option.Some(x) => { .. } Option.None => { break; } } }`.

A `&` or `&mut` chunk binding (also the first `tail masked` binding) borrows a copy of the chunk
that lives in its block; the chunk of the sequence stays borrowed until the block is left (its end, `break`, `continue`, `return`, `?`), where a
`&mut` copy is stored back. So the binding cannot leave its block (`return Option.Some(v)`:
E0512; kept in an outer local: E0501). Reason: the chunk is loaded into a vector register or
slot (a tail is partial, a chunk need not be aligned), never addressed in place.

## §4 Memory hints

Reason for explicit hints: bandwidth-bound loops need them, and auto-hinting is guesswork.

## §4 Hot loops over one field

Reason for `std.soa`: SIMD over a field of `Vec<Struct>` needs gathers; a column is one contiguous
slice.

Reason the collection, not the struct, picks the layout: one record passed, returned, or sent over
FFI is best as a plain struct, while a loop over many records that reads few fields wants only
those fields' bytes; a layout keyword or layout type parameter would give `&v[i]` two meanings and
hide the cost, so `Vec` and `Soa` stay two containers of one record type and code over columns names
the columns it reads (`ps.x[i]`). Reason for indices between records: an `idx.Id<T>` (u32) stays
valid when a container grows or changes layout, and crosses threads, files, and the GPU as Flat
data; a pointer does none of these.

Limit: Flat and Frozen count an ADT's type arguments, also one that occurs in no field, so
`idx.Id<T>` (a `u32` and no `T`) is Flat only when T is; a Soa row that refers to a record owning
a `Vec` keeps the id's `raw()` u32. Not changed: a type argument is the only way an author ties a
type whose fields hide a `T` (behind an integer or a byte pointer) to T's Flat and Frozen, since
both are derived and never implemented (E1408); counting only fields would make such types Flat
with no way out. Whether a marker param should stop counting is open.

## §4 Feature names

Each architecture owns its feature names (contracts/simd-ops.md "Features"): `sse42` to `avx512bf16`
and the groups `v3` / `v4` belong to x86-64, `neon`, `fp16`, `dotprod`, `bf16`, `sve`, and `sve2` to
AArch64, and `cuda` to every target. The set of names is the target's, not the language's: another
target may bring its own. A fn whose `needs` names a feature the target's architecture does not own
is not built for that target: its body is not checked, a `dispatch` arm that names it is dropped
from the arm chain, and no call of it runs there. A name no target has is E0708. Code that must
build on both architectures keeps the specialised bodies behind a `dispatch` whose `_` arm needs
nothing.

## §4 Features of generic instances

An instance of a generic fn or method (one per generic and type arguments) is compiled with the
features its generic declares plus those it inherits from its type arguments; its closures have
the same features. The type checker decides them once per instance on the call graph
(`stave_sema::inherit`), and lowering and codegen read that one set:

- An instance holds a vector by value when a fixed vector or mask, or a tuple, array, struct, or
  enum with one inside not behind a reference, raw pointer, slice, or fn type, is the type of a
  param, the return, a local, or an expression of its body after substitution, or is a value a
  postfix chain makes or reads between steps: a built-in's vector receiver and result, a
  literal, the operand of `?`, the result of a call through a fn or closure value. Not counted:
  an expression that is only a place there (the operand of `&` / `&mut`, the head of a postfix
  chain, the target of a plain `=`, a `for` source; parentheses do not change that), the result
  of a call to a known fn, method, operator, or dispatch (its signature is the callee's, checked
  by E0703 / E0704 / this rule), and a part that does not depend on a type parameter (the body
  gate below checks it against the generic's own `needs`). A `match` scrutinee counts the enums
  in it, through references too: matching reads an enum's tag from the whole value.
- It inherits the features whose registers hold each such vector (256 bits: `avx2`; 512 bits:
  the AVX-512 part of `v4`), beyond its declared `needs` and the target baseline, and what each
  generic instance it calls, drops (through drop glue), or makes into a fn value inherits.
- Drop glue reads the enums it drops whole: the glue of a type holding a wider vector than the
  baseline is compiled with the features that hold it, and every dropper inherits them.
- A non-generic fn that calls, drops, or makes a fn value of such an instance needs those
  features as for a declared `needs` (E0704 case `inherited`, naming the type argument); where no
  feature of the target's arch holds the vector (256 bits on AArch64), E0702 case `inherited` at
  that use. Where the fn's own body or signature holds the same vectors, its E0702 is the only
report. A
  generic instance never fails this rule: it inherits. A dispatch arm with `needs(avx2)` reaches
  `id::<f32x8>`; the `_` arm reaches `id::<f32x4>`, another instance, with no features.
- So at `f32x8`: `id<T>(x: T) -> T` inherits `avx2`, and so do `mem.swap` (it copies a `T`) and
  the drop of `vec.Vec` (it reads each element); `first<T>(xs: &[T]) -> Option<&T>` and
  `Vec.len` inherit nothing.
- A fn value of an instance has the instance's declared features in its type, not the inherited
  ones: it is made only where they are proven (the rule above), and a value made there whose
  type drops them stays callable anywhere, as for E0707: CPU features never change while a
  program runs.

Every fn body, generic or not, closures included, is gated the same way (E0702): the values it
holds by value, as above, whether their type is written or only inferred (a built-in's receiver
and result inside a postfix chain count), hold no vector wider than the registers of the fn's
own `needs` plus the target baseline: `needs(avx)` and `needs(fma)` give 128 bits, `needs(avx2)`
256, any AArch64 feature 128. A fn value whose type has no `needs` is callable without its fn's features (§2 Fn types), so the
result of a call through one is the caller's to hold. Lowering checks every fn, instance, and glue against
its features again: a vector left outside them is an internal error, never code.

Reason the instance and not the generic: a declared `needs` on a generic would make it uncallable
without the features at every type argument (`Vec.push` at `u8` included), and a rule on the
written signature misses a vector reached through a type parameter in the body and rejects
generics that only borrow one.

## §4 Entry points and dispatch arm targets

An entry point runs where no call written in the program names it, so on any CPU of the target:
the start code calls `main` of the root package, `stave test` each `test fn` (with parameters:
once per generated input), foreign code an `extern "C"` export, the `alloc` / `free` / `realloc`
of the `allocator` declaration's module (roots of a release build; no call binds to them yet), and
start code, the backend's stubs (signal handler, thread start), and code lowering inserts the fns
the compiler knows by path (contracts/intrinsics.md "Paths fixed by the compiler"). None declares `needs` (E0709; a `test fn`:
case `test`), and neither does a trait method (a call through a bound names no impl). Every other
way code runs has a typed call site the feature rules check: a direct call (E0704), a fn value or
closure of a type with `needs` (E0704 case `value`, E0707: a thread's body, a `Once` or `with_lock`
closure, an iterator adapter), a dispatch arm (E0705), drop glue (E0704 cases `inherited` and
`drops`; a `Drop` impl is a trait method), a contract clause (it runs with its fn's features).
Statics and consts are const-exprs: no initializer code runs. A test reaches feature code through
a dispatch whose `_` arm skips the work:

```text
fn wide_sum(xs: &[f32]) -> Option<f32> needs(avx2) { return Option.Some(f32x8.load(xs, 0).reduce_add()); }
fn no_sum(xs: &[f32]) -> Option<f32> { return Option.None; }
dispatch fn try_wide_sum(xs: &[f32]) -> Option<f32> { avx2 => wide_sum, _ => no_sum, }
test fn wide() {
    let a: [f32; 8] = [1.0; 8];
    match try_wide_sum(&a) {
        Option.Some(s) => { debug.assert(s == 8.0, "sum"); }
        Option.None => { }
    }
}
```

A dispatch arm target is a fn or method with a body: not a `dispatch fn` (E0705 case `dispatch`;
name its arms in the outer dispatch instead), an extern import, or a trait signature.

A fn or dispatch needing another arch's features is not built, but code the target builds may
make and pass on a fn value of it: every call of it or through its type needs those features
(E0704, E0707), so none runs there, and its code there is a trap.
