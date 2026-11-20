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
