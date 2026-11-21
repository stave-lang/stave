# Contract: language details behind `LANG.md`

Fine print for rules that `LANG.md` states in one sentence: output formats, per-case lists, and
reasons. Sections are named after the `LANG.md` section the rule is in; `LANG.md` stays the
reference for what a program may write and what it means.

## §2 Fn types

- A closure may return a borrow of a capture it holds by `&`, but not of one it holds by `&mut`,
  even through a dereference (E0512): the next call could change it or lend it again.
- A `fn(..) needs(F)` value is usable as `fn(..) needs(G)` only if F ⊆ G ∪ the enclosing fn's
  needs ∪ the target baseline (E0707). Reason: CPU features never change while a program runs, so
  a value made where F is proven stays callable.

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
by-value operand evaluated after the place's operands and before `e`. With a user operator on
a non-Copy type it is a move, so `e` may not use `p` (E0502) and the write re-initializes `p`;
a `p` that is not a whole local (a part, a place behind a reference or an index, a capture, a
static) is E0504, like `*r = *r + e`.

## §3 Whole moves and destructuring patterns

A non-Copy value moves whole. A by-value use of a non-Copy part of a place or of a temporary
outside a pattern (`let x = s.f;`, `f(t.0)`, `mk().a`, `s.v.into_iter()`, `s.f += e` with a
user operator) is E0504 case `part`; Copy parts are read freely, borrows of parts (`&s.f`,
`&mut s.f`) and assignments to parts (`s.f = v`, which drops the old part) are unchanged.

Which patterns move. A `let`, `match` arm, or `for` pattern moves its scrutinee if it binds some
non-Copy part (at any depth) by value. Every alternative of an or-pattern binds the same names
with the same modes, so either all of them move or none does. A pattern that binds only Copy
parts, binds by reference (through a `&` / `&mut` scrutinee), or binds nothing (`_`, a literal,
`let _ = s;`, `match s { _ => .. }`, `S.{ n, b: _ }` with `n: u32`) only reads: the scrutinee
stays alive and is dropped with its owner.

When and where. A moving pattern takes the whole value at its binding point: the end of the
`let` statement's pattern, the entry of the arm (after the guard passed), the start of each
`for` iteration. There the bindings take their parts, the scrutinee becomes moved as a whole
(a local is moved; a temporary is consumed, never dropped as a whole at its statement end), and
each non-Copy part the pattern does not bind by value (a `_`, a field the struct
pattern does not name, a sub-pattern that binds nothing such as a literal) is dropped there,
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
   reads (E0505), so the value it sees is the value the arm takes.
4. Nested patterns and payloads: `E.Two(a, _)` drops the second payload at arm entry;
   `W.{ p: P.{ x, y: _ }, q: _ }` drops `p.y`, then `q`. An or-pattern drops what the
   alternative that matched ignores.
5. `for P.{ a, b: _ } in src`: each iteration's element is moved into the pattern at the start
   of the iteration and its `b` dropped there; the elements not reached are dropped by the
   loop's source as before.

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

## §3 Drops

Every value with drop glue is dropped exactly once, on every path, by the owner that holds it
when that owner ends.

- **Whole values.** A value is moved whole or taken apart by a pattern ("Whole moves and
  destructuring patterns" above): the parts a moving pattern ignores are dropped at its
  binding point, and nothing of the value is left to its former owner. Assigning a field
  (`p.a = v;`) drops the field's old value.
- **`impl Drop`.** A value whose type has an `impl Drop` is dropped as a whole: its `drop`
  runs once, then its fields are dropped. No pattern takes it apart; one that only reads it
  (`match mk() { F.B => .. }`) leaves it to be dropped where its owner ends.
- **Compound assignment.** `p op= e` with a user operator on a value with drop glue: the
  operator call takes the old value of `p` (the callee owns it), and the result is written into
  `p` without a drop. When `e` leaves by `?`, the old value is dropped once, with `p`, on that
  exit.
- **Temporaries.** A temporary with drop glue is dropped at the end of its statement, the last
  made first, and only if the operand that makes it ran (the right side of `&&` / `||`, a branch
  of a value `if`). These are statements of their own for this rule: the condition of an `if` or
  `while` (each evaluation), a `match` guard, and the body of an arm of a value `match`. The
  temporaries of a `match` scrutinee and of a `for` header belong to the `match` / `for`
  statement: they live across the arms and the loop, and are dropped once after it. The
  iterator of a `for` (an rvalue the loop calls `next` on) is not a temporary: with drop glue
  it lives to the end of the enclosing block, like a local declared before the loop.
- **Rvalues under patterns.** An rvalue that a pattern only reads (`let P.{ n, a: _ } = mk();`
  with `n: u32`, a scrutinee that is a call in an arm that binds nothing by value) is a
  temporary of the statement and is dropped at its end. `let _ = e;` with an rvalue `e` drops
  it at the end of the statement; with a place `e` it moves nothing and drops nothing.
- **Leaving early.** `return`, `break`, `continue`, and `?` drop the temporaries of the
  statements they leave, then the locals of the scopes they leave. `?` also drops the operands
  of the enclosing expressions that were already evaluated (earlier arguments, a by-value
  receiver, earlier fields of a literal), the last evaluated first.

## §3 Must-consume

A pending-operation type of std stands for work that is still running (device work, a queued
operation). Dropping it would lose its result or error and free what the work still uses, so it
has no implicit drop: a value is consumed only by passing it to a method that takes it by `self`,
and an implicit drop on any path is an error, also the early return of `?`. The types are a closed
list of fixed paths in contracts/intrinsics.md; each has its own diagnostic (`gpu.Launch`: E1104).
No syntax marks such a type.

## §3 Locks and completion waits

Completion-queue waits (`park`) are `block` waits: like file waits they are outside the claim that
safe code cannot deadlock on locks and joins. No `MutexGuard` may be held across a call that has
`park`: the guard stays with the OS thread, another task on that thread that wants the lock
sleeps the thread, and the task that holds it never resumes. No std fn has `park`, so no
diagnostic checks this rule.

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

## §3 Unspecified results

- No op with an int or `bool` result is on the closed list: such a result sees an unspecified
  value only through a float's bits (`mem.f32_bits`, `bitcast`) or the effects of `std.slice`
  predicates, `par_for` / `par_for_mut` bodies, and threads.
- Tests (conformance goldens, cross-target and CPU / GPU comparisons) count every NaN equal to
  every NaN and compare everything else bit for bit (`-0.0` is not `0.0`).

## §4 Memory hints

Reason for explicit hints: bandwidth-bound loops need them, and auto-hinting is guesswork.

## §4 Hot loops over one field

Reason for `std.soa`: SIMD over a field of `Vec<Struct>` needs gathers; a column is one contiguous
slice.
