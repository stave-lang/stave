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
