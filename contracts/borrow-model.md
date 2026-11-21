# Contract: dynamic borrow model

What a program may do with references and owned values at run time. The rules read one
execution as a sequence of events. They know no loans, origins, liveness, or signatures: the
static rules of `LANG.md` §3 exist to make every execution of an accepted program follow this
text ("Relation to the static rules"). The borrow-tracking mode of the IR interpreter implements
it and reports each violation.

A *violation* is an event a rule below forbids. Safe code never reaches one. In `unsafe` code
it is undefined behaviour (the reference cases of the closed list of `LANG.md` §3 are these
rules). An event no rule forbids is allowed. Only uses count: a
reference that is never used again conflicts with nothing, except a param while its fn runs
(M23).

## State

**Allocations.** Memory is a set of allocations. Each is a run of bytes with a begin and an end.

| Allocation | Begins | Ends |
|---|---|---|
| local (`let`, pattern binding, param) | when its declaration runs (param: frame entry) | exit of the block that declares it, after that block's drops (param: frame exit) |
| temporary | at each evaluation of its expression | the next evaluation of the same expression in the same frame; the end of its statement, after its drop, when its type has drop glue; frame exit |
| heap block | `try_alloc` or `try_realloc` returns it | `free`; a `try_realloc` of it that succeeds, also when the address stays (the new block takes the old bytes with their tags and initialization) |
| `static`, string literal data, constant temporary | program start | never |

A `let` that runs again (a loop) begins a new allocation, whatever its address. Two expressions
never share a temporary. A fn value is not an allocation (M18).

A *constant temporary* is a temporary that a constant initializes and only `&` borrows: the
value of a literal, of `()`, of `size_of` / `align_of`, or of a path that names no local (a
`const` item or const generic, a variant without payload, a fn item), borrowed by `&e`, by `&`
of a field or element of it, or as an auto-referenced `&self` receiver, of a type without drop
glue (a temporary with drop glue ends with its statement, M14). Its value is fixed at compile time,
so it is static data like a string literal: every evaluation of the expression gives the same
allocation, its root is `Frozen` (`Cell` on the bytes inside an `UnsafeCell`), and a borrow of
it is never ended by a later evaluation (M14 does not apply). Nothing writes its bytes outside
an `UnsafeCell`. Its `Cell` bytes a `&` may write (M21), and each later evaluation writes the
constant's value into them again, through the root: the backends keep one slot per expression
and store the value at each evaluation, so a `const C: atomic.U32` borrowed in a loop starts
each round from its initial value. A temporary that `&mut` borrows, and one whose
value is computed (`&(2 + 2)`, `&-4`, `&f()`), is an ordinary temporary.

**Tags.** A tag is an id made by one event and never reused. The tags of an allocation form a
tree. The root is made with the allocation and stands for the place itself: an access that names
the local (`x = 1`, `x.f`, `x[i]`) goes through the root. Every other tag has a parent, a kind
(`&` or `&mut`), and a byte range inside its parent's range, all fixed when it is made (M4).
An access through tag `a` is a *child access* for `a` and each ancestor of `a`, and a *foreign
access* for every other tag of the allocation, the descendants of `a` included.

**Tagged words.** A reference value carries its tag in its pointer word (the length word of a
fat reference has none). The tag belongs to the 64-bit word, not to its type: a raw pointer or
integer made from the word keeps it (M20). Memory keeps the tag of each stored word, and a load
returns it; a byte copy (`copy_bytes`, M20) moves it with the word. A view value (view struct,
tuple, `Option<&T>`) is the words of its fields and has no tag of its own.

**Permissions.** A tag has one permission per byte of its range.

| Permission | Held by | Allows |
|---|---|---|
| `Reserved` | a `&mut` not yet written through | read, write |
| `Active` | a `&mut` written through; every root but a `static`'s (a `static mut` root is `Active`) | read, write |
| `Frozen` | a `&`; the root of a `static` | read |
| `Cell` | a `&`, or the root of a `static`, on the bytes of an `UnsafeCell` | read, write, shared with other tags |
| `Dead` | | nothing |

**Initialization.** A byte is initialized or *moved-out* (M9). A write initializes it.

**Granularity.** Permissions are per byte. A coarser unit is wrong: fields of one struct,
elements and sub-slices of one array, and the columns of one `Soa` block are borrowed side by
side (`&mut ps.x` with `&ps.y`, `par_for_mut`), and a vector load reads a run of elements. An
implementation may keep one permission per tag and run of bytes.

## Accesses

Every use of memory is a read, a write, or an end of some bytes through one tag. For each byte
`b` and each tag `t` with `b` in its range:

| `t` at `b` | child read | child write | foreign read | foreign write | foreign end |
|---|---|---|---|---|---|
| `Reserved` | `Reserved` | `Active` | `Reserved` | `Dead` | `Dead` |
| `Active` | `Active` | `Active` | `Frozen` | `Dead` | `Dead` |
| `Frozen` | `Frozen` | violation | `Frozen` | `Dead` | `Dead` |
| `Cell` | `Cell` | `Cell` | `Cell` | `Cell` | `Dead` |
| `Dead` | violation | violation | `Dead` | `Dead` | `Dead` |

- **M1 (read).** A read of `b` through `a` is a child read for `a` and its ancestors and a
  foreign read for the other tags. `b` must not be moved-out.
- **M2 (write).** A write of `b` through `a` is a child write for `a` and its ancestors and a
  foreign write for the other tags.
- **M3 (end).** The value in `b` ends through `a`: `a` and its ancestors must not be `Dead`,
  and it is a foreign end for the other tags. The end of an allocation ends all its bytes and
  makes every tag of it `Dead`, the root included.

The access that made a permission `Frozen` or `Dead` is its *cause*; a report names it.
```stave
fn stale() -> u32 {
    let mut x: u32 = 1;
    let r: &u32 = &x;
    x = 2;                  // foreign write through the root: `r` is Dead
    return *r;              // M1: read through a Dead `&`
}
fn frozen() -> u32 {
    let mut x: u32 = 1;
    let m: &mut u32 = &mut x;
    *m = 2;                 // `m` is Active
    let y: u32 = x;         // foreign read through the root: `m` is Frozen
    *m = y + 1;             // M2: write through a Frozen `&mut`
    return x;
}
```

## Events

References M4 to M7, values M8 to M12, scope M13 and M14, calls M15 to M17 and M23, closures
M18 and M19, unsafe code M20 and M21, threads M22. M24 is the oracle's check of signatures.

- **M4 (borrow).** `&p` and `&mut p` make a tag. Its parent is the tag `p` is reached through:
  the root when the path of `p` has no dereference, else the tag of the last reference the path
  dereferences; when that tag's range does not hold the bytes of `p` (a raw pointer moved past
  it, M20), the nearest ancestor whose range holds them. Its range is the bytes of `p`: of a
  slice place its elements, of a zero-sized place none. Its permission is `Reserved` for
  `&mut`, `Frozen` for `&`, and `Cell` on the bytes that are inside an `UnsafeCell` by the type
  of `p` (M21). The parent and its ancestors must not be `Dead` on the range. A borrow is not an
  access: it changes no permission. So the moment an auto-referenced receiver is borrowed needs
  no rule: in `v.push(v.len())` the `&mut v` is first used inside `push`.
```stave
fn through_dead() -> u32 {
    let mut x: u32 = 1;
    let m: &mut u32 = &mut x;
    x = 2;                  // `m` is Dead
    let r: &u32 = &*m;      // M4: borrow through a Dead tag
    return *r;
}
```
- **M5 (projection).** `&p.f`, `&p[i]`, `&p[a..b]`, `&mut ps.x` (a column), and a pattern that
  binds a field or payload of a borrowed scrutinee are borrows (M4) of exactly those bytes; a
  column is reached through the tag of the `Soa`'s block pointer. Borrows of disjoint bytes
  never meet: two fields, two elements, two halves of a slice, two columns.
```stave
fn overlap(xs: &mut [u32]) -> u32 {
    let a: &mut [u32] = &mut xs[0..2];
    let b: &mut [u32] = &mut xs[1..3];
    b[0] = 7;               // the bytes of `xs[1]`: `a` is Dead there
    return a[1];            // M1: read through a Dead `&mut`
}
```
- **M6 (reborrow).** `&*r`, `&mut *r`, `&r.f` are borrows (M4) through the tag of `r`. The
  implicit forms are the same event: a `&mut T` place given where a `&mut T` parameter (fn,
  method receiver, closure, built-in), variant constructor argument, or struct literal field
  takes it is `&mut *p`; a `&mut T` given where `&T` is expected is `&*p`; a `&mut T` place
  cast to a raw pointer is `&mut *p` (M11). Every other by-value use of a `&mut T` place moves
  it (M9).
```stave
fn child(p: &mut u32) -> u32 {
    let c: &mut u32 = &mut *p;
    *c = 1;
    *p = 2;                 // foreign write for the child: `c` is Dead
    *c = 3;                 // M2: write through a Dead `&mut`
    return *p;
}
```
- **M7 (use).** An expression that reads or writes a place accesses the bytes it reads or
  writes, through the tag the place is reached by (as in M4): a field its bytes, `xs[i]` one
  element, a vector load or store its lanes, a masked form its active lanes, `fill`, `copy_from`,
  `sum` the elements they touch, `xs.len()` none. A reference the path dereferences is first
  read from its own place. A chunk loop with a `&mut` binding reads the chunk when its block is
  entered and writes it when the block is left. Violations: M1, M2 (the examples above).
- **M8 (copy).** A by-value use of a place whose type is Copy is a read. The words of the result
  keep their tags: two copies of a `&`, or of a Copy view, hold the same tags, and no rule tells
  them apart. A copy has no violation of its own; the copies become `Dead` together.
- **M9 (move).** A by-value use of a place whose type is not Copy (a `&mut T`, a view holding
  one, a value with drop glue, any type without `impl Copy`) is a read, then an end (M3) of the
  bytes of the place through the tag it is reached by; the bytes are then moved-out. The value
  keeps the tags of its words: moving a view moves its references unchanged. Any read of a
  moved-out byte is a violation; a write (assignment, `p.write(v)`) initializes it again.
  `p.read()` is this event through a raw pointer.
```stave
struct S { a: u32 }
fn moved_under() -> u32 {
    let s = S.{ a: 1 };
    let r: &u32 = &s.a;
    let t = s;              // the value in `s` ends: `r` is Dead
    return *r + t.a;        // M1: read through a tag ended by a move
}
```
- **M10 (array repeat).** `[e; n]` evaluates `e` once; each element takes the value by one
  by-value use of it. For a type that is not Copy the second use reads a moved-out value (M9):
  the literal makes `n` owners of one value. `n = 1` is one move. `n = 0` uses it never.
  No oracle run shows this rule: such a literal is E0418, a type error, and a program with a
  type error is not lowered (the oracle lowers past borrow errors only).
```stave
fn twice(x: &mut u32) -> u32 {
    let t = [x; 2];         // M10: the second element is a second owner of one `&mut`
    *t[0] = 5;
    return *t[1];
}
```
- **M11 (casts and coercions).**

| Form | Event |
|---|---|
| `v as T`, `T` the type of `v` | a by-value use of `v` (M8 or M9); the words keep their tags |
| `r as *mut T`, `r as *const T`, `r` a `&mut T` place | a reborrow `&mut *r` (M6) whose word the pointer is: `r` is read, not moved, and stays usable; a later use of `r` is a foreign access for the pointer's tag |
| `&x as *const T`, `&mut x as *mut T`, pointer to pointer, pointer to and from a 64-bit integer, a `&T` to a raw pointer | the same word with the same tag (M20) |
| `&mut T` to `&T` | a reborrow (M6) |
| `&[T; N]` to `&[T]`, `&[T align A]` to `&[T]` | the same tag and range |
| `&String` to `&str`, an f-string to `&str` | a call of `as_str` (the f-string is a temporary with drop glue) |
| a closure or fn item to a fn type; a fn type to one with other `needs` / `without` | the same fn value with the same capture tags (M19) |

```stave
fn two_mut() -> u32 {
    let mut x: u32 = 1;
    let r: &mut u32 = &mut x;
    let s: &mut u32 = r as &mut u32;   // a move of `r`
    *r = 2;                            // M9: read of the moved-out `r`
    *s = 3;
    return x;
}
```
- **M12 (drop, assignment).** A drop of a place runs its drop glue on a `&mut` borrow (M4) of
  the place, then ends its bytes (M3); they are moved-out. A drop of a moved-out place is a
  violation. `p = v` evaluates `v`, drops the value in `p` when `p` is initialized and has drop
  glue, then writes `p` (M2): a reference into the old value is `Dead` after the statement,
  whoever holds it, the new value included.
```stave
view struct Chain { v: u32, prev: &u32 }
fn self_borrow(zero: &u32) -> u32 {
    let mut x = Chain.{ v: 1, prev: zero };
    x = Chain.{ v: 2, prev: &x.v };   // the write of `x` makes the tag of `&x.v` Dead
    return *x.prev;                   // M1: read through a Dead `&`
}
```
- **M13 (block and frame exit).** Leaving a block (its end, `return`, `break`, `continue`, `?`)
  drops its initialized locals in reverse declaration order, then ends their allocations. A
  `scope` block joins its threads first. Leaving a fn does this for the body, then drops and
  ends the params and ends every temporary of the frame. The returned value is evaluated before.
```stave
fn block_exit() -> u32 {
    let zero: u32 = 0;
    let mut r: &u32 = &zero;
    {
        let x: u32 = 7;
        r = &x;
    }                       // the allocation of `x` ends: every tag of it is Dead
    return *r;              // M1: read after the end of the referent
}
```
- **M14 (temporaries).** An evaluation of an expression whose value needs a place (`&f()`,
  `&(a + 1)`, the receiver in `f().m()`) first evaluates its operands, which may read the
  allocation of the evaluation before; then that allocation ends and a new one begins. Not a
  constant temporary ("State"): `lit = &4u32` or `tab = &TABLE` in a loop borrows the same
  static data in every round, and a reference of the round before stays usable. A
  temporary whose type has drop glue is dropped and ends with its statement, in reverse order
  of creation. Any other one stays until the next evaluation or the frame exit: a borrow of it
  may be used after its statement.
```stave
fn next(i: u32) -> u32 { return i + 1; }
fn temp_loop() -> u32 {
    let zero: u32 = 0;
    let mut r: &u32 = &zero;
    let mut sum: u32 = 0;
    for i in 0u32..3u32 {
        let before: &u32 = r;
        r = &next(i);          // the temporary of the round before ends here
        sum = sum + *before;   // M1 in the second round: `before` points to it
    }
    return sum;
}
```
- **M15 (call).** Arguments are evaluated left to right and passed by copy or move with their
  tags; `&mut` places are reborrowed (M6). A param is a local of the callee that holds the
  argument. A call makes no other tag. The references it passes are protected until it returns
  (M23, with the example). A call through a fn value is the same.
- **M16 (store into a view).** Writing a reference into a view reached through `&mut V` is a
  write (M2) of the view's bytes; the stored word keeps its tag and stays under every rule. A
  parameter `from` clause has no run-time effect: the callee may store any reference it holds,
  and a store the clause lacks is no violation by itself (M24 reports it). The violation is the
  later use of the stored reference after a conflicting access or after the end of its
  referent. After the call, each reference in a view the callee could write holds its word from
  before the call or a word the callee held. A signature does not say which, so a static rule
  must keep both (a weak update); the model follows the one word that is there. A pointer that
  may point to several views writes the one it points to.
```stave
view struct Pick { best: &u32, count: usize }
fn put(dst: &mut Pick from x, x: &u32) { dst.best = x; }
fn outlives(zero: &u32) -> u32 {
    let mut p = Pick.{ best: zero, count: 0 };
    {
        let y: u32 = 7;
        put(&mut p, &y);    // `p.best` holds the tag of `&y`
    }                       // M13: the allocation of `y` ends
    return *p.best;         // M1: read after the end of the referent
}
```
- **M17 (return).** A returned value moves to the caller with its tags; no tag is made. `from`
  and `from copy` have no run-time effect. A result `from x` of a `&mut V` param is a reborrow
  made in the callee, below the tag of `x`. A result `from copy x` is a word read out of the
  view: it does not hang below the tag of `x` and stays usable when that tag is `Dead`. A
  reference into the callee's frame is `Dead` when the caller gets it (M13).
```stave
fn local(z: &u32) -> &u32 from z {
    let x: u32 = *z;
    return &x;              // the tag of `&x` leaves; `x` ends with the frame
}
fn caller(zero: &u32) -> u32 {
    return *local(zero);    // M1: read after the end of the referent
}
```
- **M23 (protection).** From the entry of a call to the callee's return, the references passed
  are *protected*: the tag of each `&T` / `&mut T` param (of a reborrowed `&mut` argument: the
  reborrow's tag, M6) and of each reference word inside a by-value view param (found by its
  type; of an enum the live variant). Not protected: raw pointers; references reached only
  through a pointer (what the view behind a `&mut V` param holds); the captures of a fn value,
  when it is passed and when its body runs (a fn type does not show them; the params of the
  body are protected like any). While a tag is protected, an access or end that would make it
  `Dead` on a byte, and a foreign read of a `&mut` that is `Reserved` or `Active`, are violations
  at that access, whether or not the param is used again. Arguments are evaluated before the
  entry: in `v.push(v.len())` the read of `v` meets no protection. What this gives an optimizer:
  from entry to return, not only up to the last use, the bytes of a `&` param outside
  `UnsafeCell` are not written and stay allocated, and the bytes of a `&mut` param are read and
  written through it alone.
```stave
fn quiet(m: &mut u32, r: &u32) -> u32 {
    let a: u32 = *r;        // M23 when `r` covers the bytes of `m`: a foreign read of the
                            // protected `&mut`, which is still `Reserved`
    *m = a + 1;             // without M23 no rule fails: `r` is Dead here but not used again
    return a;
}
fn caller() -> u32 {
    let mut x: u32 = 1;
    return quiet(&mut x, &x);
}
```
- **M18 (closure literal).** Evaluating a closure literal borrows (M4) each captured local once:
  `&mut` when the body writes it or borrows it mutably, else `&`. The fn value is the code and
  these references, held by value like the fields of a view. It is not a temporary: values made
  by earlier evaluations of the literal keep their tags and stay usable. In the body a use of a
  captured local goes through its capture tag, and a borrow made there is a child of that tag.
  Such a borrow may leave in the result; it then falls under M1 to M3 like any reference: the
  next call that writes the capture, or whose `&mut` result is written through, makes it `Dead`.
```stave
fn two_results() -> u32 {
    let mut n: u32 = 1;
    let f = fn() -> &mut u32 { return &mut n; };
    let a: &mut u32 = f();
    let b: &mut u32 = f();  // a second child of the capture tag
    *a = 1;                 // foreign write for `b`: `b` is Dead
    *b = 2;                 // M2: write through a Dead `&mut`
    return *a;
}
```
- **M19 (fn values).** A fn value is Copy: a copy holds the same capture tags, a `&mut` capture
  included (the one case of a `&mut` tag with several holders; they are one borrower). A change
  of its fn type keeps them (M11). A call runs the body with them (M15): a capture whose
  referent has ended is a violation at its first use in the body.
```stave
fn widened() -> u32 {
    let zero: u32 = 0;
    let mut f: fn() -> u32 without(alloc) = fn() -> u32 { return zero; };
    {
        let x: u32 = 7;
        f = fn() -> u32 { return x; };   // holds the tag of a borrow of `x`
    }                                    // M13: the allocation of `x` ends
    return f();                          // M1 in the body: read after the end of the referent
}
```
- **M20 (raw pointers).** A raw pointer has no kind and no tag of its own: it carries the tag
  of the word it was made from. A cast keeps the tag (M11); so do `+` and `-` of a tagged word
  and an untagged integer; other arithmetic gives an untagged word. An allocator result carries
  the root of its block. `*p`, `p.read()`, `p.write(v)`, and the intrinsics that take a pointer
  are accesses through that tag (M1, M2, M9); a byte outside the tag's range is accessed through
  the nearest ancestor whose range holds it. `&*p` and `&mut *p` are borrows (M4) through it. A
  value read through a raw pointer gets a new tag for each reference in it, a child of the tag
  stored there. The byte copy `copy_bytes(dst, src, n)` (intrinsics.md "Untyped copy") is a read
  of `src..src+n` through the tag of `src`, then a write of `dst..dst+n` through the tag of
  `dst`; it reads no value, so it makes no tag: each stored word it copies whole keeps its tag
  at its new address, and a word of the destination it overwrites or cuts into loses its own. An untagged word, and a word whose tag belongs to another allocation, is used
  as the root of the allocation that holds its address. Nothing is suspended in `unsafe` code:
  M1 to M3 and M9 stay errors. What the model stops tracking is the pointer's type: a
  `*const T` made from a `&mut T` may write, and a `*mut T` made from a `&T` may not, outside
  `Cell` bytes.
```stave
fn behind() -> u32 {
    let mut x: u32 = 1;
    let p = &mut x as *mut u32;
    let r: &u32 = &x;
    unsafe("p points to x") { *p = 2; }   // write through the tag of `&mut x`: `r` is Dead
    return *r;                            // M1: read through a Dead `&`
}
```
- **M21 (`UnsafeCell`).** A `&` has permission `Cell` on the bytes that are inside an
  `UnsafeCell` by the type it was made at (every byte of an enum with such a payload). Writes
  through it or through a raw pointer made from it are allowed and leave other `Cell`
  permissions as they are; every tag that is not `Cell` there follows the table: a reference
  into the content (made at the content's type, so `Frozen` or `Reserved`) is `Dead` after the
  next write through the cell. `Cell`, atomics, `Mutex`, `Once`, `Arena` rest on this rule alone.
```stave
import std.cell;

fn into_cell() -> u32 {
    let c = cell.Cell::<u32>.new(1);
    let p = &c as *const cell.Cell<u32> as *const u32;
    let r: &u32 = unsafe("a Cell<u32> is one u32") &*p;   // Frozen child of a Cell tag
    c.set(2);                                             // foreign write: `r` is Dead
    return *r;                                            // M1: read through a Dead `&`
}
```
- **M22 (threads).** Tags do not belong to threads. The rules read the accesses of all threads
  in the one order the execution had. `Send` and `Sync` add no tag rule: they keep `Cell`
  permissions that are not atomic or locked, and fn values (whose `&mut` captures the type does
  not show), on one thread, which is what freedom from data races needs. A `spawn` body,
  `thread.join` arm, or `par_*` body is a closure (M18, M19); its threads are joined before the
  block that made it is left (M13), so a capture ends only after them.
```stave
fn interleaved() -> u32 {
    let mut x: u32 = 1;
    let mut y: u32 = 0;
    scope {
        spawn { x = 2; x = 3; }   // two writes through the `&mut` capture
        y = x;                    // between them: the capture is Frozen, the second write is M2
    }
    return y;
}
```

## Signature check

`from` clauses have no run-time rule (M16, M17). The oracle checks them all the same, as a
fifth report. It is a check of a signature against what one call did, not one of the four
aliasing classes and not a violation: the execution stays defined, and the reported program may
have no caller that misuses the reference.

- **M24 (signature).** At the return of a fn item or method. *Reachable from a value*: its
  reference words (of a fn value: its captures) and those in the views they point to, by type.
  *Writable through a param*: the reference words reachable from it behind `&mut` words only
  (`*p` of a `&mut V` param and on; what a `&mut` word of a by-value view or a `&mut` capture
  points to). A word *derives from* another when its tag is that word's tag or a descendant.
  - Result: each reference word in the returned value derives from a source of the result's
    clause; without a clause there is no source.
  - Stores: each word writable through a param `p` at return, whose tag was not writable
    through `p` at entry, derives from a word that was, from a source of the clause of `p`, or
    from a source of a clause that names `p` without `copy`.
  - Sources: of `from x`, the words reachable from the argument `x`, at entry or at return; of
    `from copy x`, the same without the word of `x` itself.
  - Not judged: a tag of a `static` or of string literal data; a tag made through a raw pointer
    (M20) or below such a tag (its origin is the claim of the unsafe code).
  - A closure body: every reference param and capture is a source of its result; no param has a
    clause; a capture is a param for the stores.

  The report is "signature": the return site, the param or result with its clause, the site that
  made the tag, and the site of the store when the word was written during the call. The check
  is weaker than E0510 / E0513: it does not tell a borrow through a `&mut` field of `*x` from a
  copy out of it, and it sees one execution.
```stave
view struct Pick { best: &u32, count: usize }
fn put(dst: &mut Pick, x: &u32) {
    dst.best = x;           // the word of `x` goes into `*dst`
}                           // M24: `dst` has no clause that names `x`
fn harmless(zero: &u32) -> usize {
    let y: u32 = 7;
    let mut p = Pick.{ best: zero, count: 0 };
    put(&mut p, &y);        // `y` outlives `p`: no rule of the model is broken
    return p.count;
}
```

## Reports

A report gives the class, the rule, the place (allocation and bytes), the site of the violating
event, the site of its cause, and the site where the failing tag was made. The failing tag is
the first tag from the accessing tag towards the root whose permission forbids the access. A
protection violation (M23) has the class its tag and the access give (a `&` written: (a); a
`&mut` read or written: (b); an end: (c)); its two sites are the call that protects the tag and
the access. The site of a frame exit as a cause is the last instruction the returning frame
ran.

| Class | Violation | Cause | Rules |
|---|---|---|---|
| (a) write under a `&` | a use through a `&` tag that a write made `Dead`; a write through a `Frozen` `&` or the `Frozen` root of a `static` | that write; for the write through a `&`, the borrow that made it; for the root of a `static`, its begin | M1, M2, M4 |
| (b) two exclusive users | a use through a `&mut` tag that is `Dead`, or a write through one that is `Frozen` | the foreign write, or the foreign read, through another `&mut`, a `&`, a raw pointer, or the place itself | M1, M2, M4 |
| (c) use after the referent | a use through a tag of either kind made `Dead` by an end | the end: block or frame exit, a temporary evaluated again or dropped, a move out of the place, `free` | M3, M9, M13, M14 |
| (d) second owner | a read or drop of a moved-out place; an array repeat of a value that is not Copy | the move that took the value (for the array repeat: the literal itself) | M9, M10, M12 |

In (d) the copy is the report: the later double drop, or the two `&mut` to one place, is its
symptom. Two holders of one `&mut` tag made by a copy cannot be told apart by M1 to M3, so
(b) never reports them; M9 does, at the first use of the place the value left. The fifth form,
"signature" (M24), is reported beside these and never instead of them.

## Relation to the static rules

The static rules must imply the model: no execution of an accepted program has a violation. A
program the checker accepts with an execution the model rejects is a hole in the static rules,
unless the execution breaks an obligation of its `unsafe` code ("Obligations of `unsafe` code").
A program the checker rejects with no execution the model rejects is a precision note, not a
hole. The model never reads a static rule to decide.

| Static rule (`LANG.md` §3, `diagnostic-codes.md`) | Model rules it enforces |
|---|---|
| E0501: an access, borrow, move, drop, or assignment under a live loan | M1, M2, M3, M4, M9, M12 |
| E0501: a local dies, or a temporary is evaluated again or dropped, under a live loan | M13, M14 |
| E0501 after a storing call, after a write through `&mut V`, for a view that borrows itself | M16 with M2, M13 |
| E0501 at a call: an argument's referent reachable by the callee another way (a second argument, a capture of a fn-value argument, a thread) | M23 |
| two-phase receiver borrows | M4 (a borrow is not an access), M23 (protection starts at the entry) |
| E0502: use after move; assignment into a part of a moved value | M9 |
| E0503: assignment through `&`; `&mut` of a place behind `&`; writes of a `static` | M2 on `Frozen` |
| E0503: assignment to, or `&mut` of, an immutable binding | none (the root of a local is `Active`) |
| E0504: move out of `*r`, `r.f`, `v[i]`, a capture | M9, M12 (the owner later reads or drops the moved-out place) |
| E0510: a returned borrow the `from` clause lacks | M24 (result); M13, M2 in the caller, which reads the clause |
| E0512: a returned borrow of a local | M13; M24 (result) |
| E0512 `mut-capture` | M18 |
| E0513: a store the view or the clause cannot take | M24 (stores); M16 with M13, M2 in the caller |
| E0514, E0515, E0520, E0521, E0905, E0807: form of clauses, fn values of fns with clauses | none (they keep M24 and the callers' reading of a signature well-defined) |
| E0410 to E0414: where references and views may be held | M13, heap `free` (a reference where no rule sees its holder) |
| E0415, E0416: Copy only for types without `&mut` and drop glue | M8, M9 |
| E0418: array repeat (a type error: no oracle run shows it) | M10 |
| E0211: an `ensures` reads no param whose `&mut` the result holds; no clause writes | M1 (the clause's read would freeze the result's tag before the caller writes through it) |
| temporaries: lifetime, one place per expression | M14 |
| E0801, E0808, `scope` / `join` structure | M13, M22 (threads end before the captured locals) |
| E0802, E0806, E0417, E1405, E1406 | none (data races and deadlocks, not tags) |
| E0803: unsafe operations | marks where M1 to M23 are the author's obligation |
| `Frozen` (E1410) | M21, M2 (which `&` are read-only) |

Precision notes that follow from the model, each accepted by it: borrows of disjoint elements
or sub-slices of one array; a `&mut` made under a live `&` and never written through; a view
that borrows a part of its own local when that part is not written again; a source of a storing
call used after the last use of the view that took it; `[e; 0]` of a value that is not Copy; an
assignment to an immutable binding.

## Shapes a checker must reject

Each shape below has an execution with a violation, so a checker that accepts it has a hole.
The oracle's regression set holds one program per shape. The rule that rejects the execution:

| Program shape | Violation | Class |
|---|---|---|
| a closure literal given a fn type with a deny clause, or a fn value whose fn type changes later, called after the block of a captured local was left | M19 with M13 | (c) |
| `a as &T` with `T` the type of `a`, the result used after the referent's block; the same result returned from the referent's frame | M11 with M13 | (c) |
| `r as &mut T`, then `r` and the result both used | M11 with M9 | (d) |
| a closure returns `&mut` of a capture and is called twice; a closure changes a capture and returns `&` of it | M18 with M2 | (b), (a) |
| a `&` of a temporary kept across the next evaluation of its expression (under `&`, as a call argument, in a literal's field, behind `if`, as a receiver, around a part of it) | M14 | (c) |
| `[e; n]`, `n >= 2`, of a `&mut`, a `Vec`, a struct that owns one | M10 | (d) |
| a view takes a borrow of another field of its own local (by a field assignment, through a pointer to the view or to a part of it, by a storing call on a field or tuple element, by a pattern binding), then that field is written | M16 with M2 | (a) |
| a storing call whose `from copy` source, or `from copy` result, is what the same call stored: the taker is used after the block of the stored referent | M16, M17 with M13 | (c) |
| a callee splits its `&mut` source into two pointers and stores one through the other (the source wrapped in a tuple, struct, array, `Option`, a by-value `self`, a pattern, a closure, a generic param); the caller then writes the part pointed to | M16 with M2 | (a) |
| a `&mut` source stored in a view; the source's local and the view's pointer both used | M16 with M2 | (b); (a) when the stale user is a `&` of the local |
| a storing call stores `&n` and its result is assigned to `n` | M16 with M2 | (a) |
| an assignment whose new value borrows the value it replaces (`x = V.{ r: &x.f }`, `*q = ..`, `w.h = ..`); with a heap buffer in the old value | M12 with M2; M12 with `free` | (a); (c) |
| a temporary's new value, or a view stored by the same statement, borrows the temporary's earlier value | M14 | (c) |
| a temporary with drop glue borrowed by a storing call and read after the statement | M14 | (c) |
| a `&mut` pointer exchanged out of a view through raw `read` / `write`, then split and stored into itself | M20, M16 with M2 | (a) |

A program that only shows a callee body the static rules must reject (a store its clause lacks,
a clause a trait impl drops) has no violating execution until a caller uses the stored
reference: the model then rejects that execution by the rows above. Before that, M24 reports
the call of the body.

## Obligations of `unsafe` code

Each shape below has an execution with a violation that the static rules do not see, because
it goes through a raw pointer. A checker may accept it; the author of the `unsafe` code must
rule the execution out, and the borrow-tracking mode reports it when a run has it.

| Program shape | Violation | Class |
|---|---|---|
| a `&mut V` made from a raw pointer under a `&` anchor takes a borrow of a shorter-lived local (`tests/borrowcheck/anchor_shorter_local.stv`) | M20, M16 with M13 | (c) |

## What an implementation over the checked IR needs

The checked IR (`checked-ir.md`) does not show every event. `&x` of a local is its `slot`
address; a move of an SSA local is a rebinding; a block exit of a local without drop glue and
the next evaluation of a temporary are nothing; one environment `slot` serves every evaluation
of a closure literal. An implementation therefore needs from lowering, per function and outside
the canonical dump: each borrow with its kind, range, and the implicit forms of M6 and M18; each
by-value use that is a move (M9), also of an SSA local; the begin and end of each local and
temporary (M13, M14); the capture references of each `fn.make` as a value of their own (M18);
the `from` clauses of each fn (M24).
`runtime.alloc.{try_alloc, try_realloc, free}` are primitive events. Everything else is read off
the IR: `load`, `store`, `vec.*` memory ops, and intrinsics that take a pointer are M1 and M2
(`intrinsic.copy_bytes`: M1 on its source range, then M2 on its destination range);
`addr.*` and casts keep the tag (M20). What the tracking mode gets of this list, and what it
does without, is in "Side table".

## Side table

The tracking mode runs a module lowered for it (`Options::borrow_marks` of the driver's
pipeline; `Options::tracked`, the default under `STAVE_INTERP_CHECK=borrow`, lowers it beside
the canonical module; no `stave` command asks for it). That lowering differs from the canonical
one in what this section lists, and in nothing else. Its module is not the canonical checked IR:
it is not dumped, not compared with a golden, and not given to a backend.

- **A borrow is a value of its own.** In the canonical IR `&x` is the `slot` value of `x`, and
  `&*r` is `r`: the same value names the reference and the place, so no table beside the
  instructions can tell a use of one from a use of the other. Here each borrow expression (`&p`,
  `&mut p`, the implicit reborrows of M6, `&mut T` given as `&T`, an auto-referenced receiver, a
  pattern binding by reference) ends in one more instruction: an identity cast for a thin
  reference; for a fat one the pointer and length taken apart and put together again.
- **`Body::borrows`**, one row per such value: the value, and `&` or `&mut`. The interpreter
  makes the tag (M4) when the instruction that defines the value runs: the parent is the tag its
  address word carries; the range starts at the address and has the size of the pointee of the
  value's type, for a fat reference its length times the element size. More borrows than the
  source spells have a row: a `&mut T` place given as a method receiver (M6) or cast to a raw
  pointer (M11); the capture
  references of a closure literal, one per captured local, `&` when the body only reads it
  (M18); the binding of a `for` over a reference to an array or slice, a borrow of its element
  through that reference; a `&mut` chunk binding, a borrow of the chunk's temporary; the
  `&mut` a drop glue runs on (M12).
- **`Body::events`**, one row per event that is no borrow, each on an instruction of its own
  (an identity cast of the pointer to the place; for *repeat* and *environment* the instruction named there):
  - *begin*: a local at its declaration, a temporary at each evaluation of its expression
    (M13, M14). The allocation that was in the slot ends with every tag of it, and the root
    stands for the new one.
  - *end*: the locals of a block when it is left by its end, `break`, or `continue`, after
    their drops; a temporary with drop glue after its drop at the end of its statement. A
    `return` and a `?` emit none: the return of the frame ends every slot of it. Between its
    end and its next begin the root of a slot is `Dead`.
  - *gone*: a by-value use of a place whose type is not Copy, after its read; `p.read()` of
    such a type; the end of a drop, after the glue (M9, M12). The bytes end through the tag of
    the pointer and are moved-out until the next write.
  - *repeat*, on a unit value made where the literal is evaluated, after `e`: `[e; n]`,
    `n >= 2`, of a type that is not Copy (M10).
  - *environment*, on `fn.make`: the interpreter copies the environment into a slot of its
    own in the current frame, so each evaluation of a closure literal holds its own capture
    references (M18).
  - *constant*, in place of *begin*: the slot of a constant temporary (every form "State"
    lists). Its first evaluation
    begins it; the store of a later one is no event on the bytes outside an `UnsafeCell` (they
    hold the value already) and a write through the root on its `Cell` bytes. Its tags are
    not judged by M24, like those of a `static`.
- **A local a value is moved out of lives in a slot**, also when the canonical lowering keeps
  it as an SSA value: a move is an event on the bytes of its place. A local that is never
  borrowed, captured, or moved out of stays an SSA value and is no allocation of the run.
- **Which by-value uses are moves.** Every one of a place whose type is not Copy, but a
  reference that the path of a place dereferences, a reference an implicit reborrow or a
  coercion takes (M6, M11), a `&mut T` place cast to a raw pointer (M11), and the reference a
  `for` loop iterates through.
- **`Module::cell`**, the declaration of `std.cell.UnsafeCell`. A `&` gets `Cell` on the bytes
  that the layout of its pointee type puts inside a value of such a type (M21).
- **`Module::frozen`**, per `static`: it is not `static mut`. Its root is `Frozen`, and `Cell`
  on its `UnsafeCell` bytes.
- **`Module::clauses`**, per function: a fn item or method with the `from` clauses of its
  result and of its params, or a closure body (M24).
- **Sites.** An instruction without a site of its own has the site of the statement it belongs
  to, so both sites of a report are source lines.

What the interpreter reads off the module without a row: a call protects the reference words of
its arguments, found by the types of the params (M23); a load through a raw pointer gives each
reference word of the loaded value a new tag (M20); the return of a frame ends its locals and
temporaries, `free` and `try_realloc` end the block (M3); a borrow whose instruction takes a raw
pointer is a tag made through a raw pointer (M24).

What the tracking mode does not see, or sees later than the rules say:

- A constant temporary is a slot of its frame, as in the code the backends get (one slot per
  expression, which each evaluation stores the same bytes into): the mode ends it with the
  frame, and its root stays `Active`.
- The moves of a `match` arm that has a guard happen when the guard holds (the names are bound
  before it runs). Of an or-pattern with a guard they have no row: the block the guard runs in
  has one predecessor per alternative.
- A drop of a moved-out place (M12) is reported by the first read its glue makes of the place,
  as a read of a moved-out place; a glue that reads nothing is not reported.
- *repeat* rows are never reached today: E0418 is a type error, and a program with one is not
  lowered, with or without the borrow checker.
- M24 walks the views a value points to by type. It does not see the captures of a fn value
  that is a param, a field, or a result (a fn type does not show them): when a source of a
  clause holds a fn value with captures, a reference that derives from no other source is not
  judged, and a fn value in a result or stored into a view is not judged either. A closure
  body's own environment it walks. A value with more than 4096 reference words (with the
  words in the views it reaches) is not judged: a clause whose source it is, and the stores
  into it, are not; the other clauses of the call are. The report
  names the param by its index and a closure body as `the closure at <file>:<line>:<col> in
  <fn>` (the site of its literal; its environment is its last param), and has no site of the
  store. It does not end the run: a run
  without a violation fails with its signature reports at its end.

## Limits

- **Data races.** Two threads that use one tag, or two `Cell` permissions, at once are outside
  the model: it orders accesses and judges tags only.
- **Executions, not programs.** A verdict says nothing about other inputs or interleavings.
  M24 reports a broken clause only for a call that breaks it, and not through raw pointers.
- **Captures.** The captures of a fn value are not protected (M23): a fact about one holds up
  to its last use.
- **Untagged pointers.** A word that lost its tag (arithmetic other than `+` / `-` with an
  integer, bytes of a reference copied one by one, a pointer from `extern "C"` or a system call)
  acts as the root of its allocation: its accesses kill references a finer rule would keep,
  and two such pointers are never in conflict with each other.
- **Memory errors.** Out-of-bounds and misaligned accesses, uninitialized bytes other than
  moved-out ones, leaks, a second `free`, and `p.read()` of a Copy type are the memory checking
  mode's, not this model's.
- **Foreign code.** Accesses made by `extern "C"` code and by the kernel are not seen.
