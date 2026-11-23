# Known limits

What the compiler does not guarantee, what it rejects without need, and what it rejects on
purpose. The README has a summary; this file has every item, with the example and the rewrite.
A safe program that breaks a guarantee below, or is rejected for no good reason, and is not listed
here is a bug.

The checker was hardened in three rounds of adversarial testing: hand-written probes and generated
programs over closures and threads, patterns, moves and drops, views, traits, function values, and
the verification rules (effects, recursion, stack bound, lock ranks, contracts). The rounds found
27, 21, and 3 root causes of programs that `check` accepted and that then misbehaved (memory
errors, data races, deadlocks, a denied effect or trap that ran, a crash on a CPU without a
feature), plus a compiler hang and 4 internal errors after a passing `check`; each was fixed and
the fixes were reviewed. The third round was the last, so its fixes were not hunted again: holes
of the same kinds may remain, and the first round ran without the interpreter's borrow tracking.

## 1. Guarantees and where they have gaps

### Memory safety

Memory safety is the aim, not a proof: the borrow checker is young, and soundness bugs are
expected. A safe program that corrupts memory is a compiler bug.

- Unsafe code in `std` and `runtime` was checked against the borrow model by reading it. Thread
  start (`clone`) and the system-call paths do not run under the interpreter, so their unsafe code
  was read, not run under borrow tracking or the memory mode.
- Unsafe code in the built-in packages, and C code, are trusted not to write function values into
  places the compiler tracks. This is an obligation, not a check.
- Races through raw pointers or `UnsafeCell` are the author's obligation (the undefined-behaviour
  list of `LANG.md` section 3). Nothing checks them, in the compiler or in the interpreter.

### Deadlocks

"Safe code cannot deadlock" covers ranked locks and thread joins only. Not covered: a file wait
(`std.io` read, write, open: the `block` effect) on a pipe or FIFO whose other end this process
holds, a loop that spins on `try_lock` or an atomic until another thread acts (livelock), and a
panic report stuck on a full stderr pipe. The table is in [std/README.md](../std/README.md).

Lock-order tracking at run time and an interpreter scheduler are not built. The interpreter runs a
spawned body on its one thread, at the join, in spawn order (see 2).

### Stack bound

The checker computes a stack size for every thread whose calls cannot reach a call cycle.

- A `rec fn` cycle, and the drop of a recursive value, have no bound. Their stack is `[run] stack`,
  and the program traps `StackOverflow` (exit 101) when it is exceeded. How deep a recursive value
  can be dropped is a run-time limit.
- Signal handlers and the panic report run on stacks of their own that no bound counts: the
  handler and the kernel's signal frame on a 32 KiB signal stack, the report on a 64 KiB report
  stack, per thread, each with a guard page below. The build keeps the report's depth within its
  stack and the handler's own use within half of the signal stack; the size of the kernel's
  signal frame depends on the CPU and is not checked. A frame that does not fit the signal stack
  ends the process by SIGSEGV; a handler that runs past its stack ends it with exit 101 and no
  report line.
- A stack overflow inside the compiler-made drop of a recursive value can be reported without a
  function: `panic: StackOverflow at <unknown>:0:0 in <unknown> (thread 0)`. Which fn the report
  names depends on where the stack ends: the same program can name `std.vec.Vec.drop` or
  `std.vec.Vec.truncate` on another target or in the other build. The exit code is 101 as always.
- There is no stack bound for the GPU target (it is design only).
- `stave test` runs each `test fn` on the default 8 MiB stack and does not read `[run] stack`. With
  `[run] stack = "64MiB"` and `rec fn depth(n: u64) -> u64`, `depth(400000)` runs under `stave run`
  and traps `StackOverflow` in a `test fn`; with `[run] stack = "64KiB"` and `depth(50000)` the
  program traps and the test passes. Keep a test's recursion within 8 MiB and run the deep case in
  `main`.

### Effects and `without(..)`

The effect sets over-approximate: a denied effect is never run, but the checker may refuse a
program that does not run it.

- The effects of the `runtime` boundary functions (allocator, panic, lock word, futex, thread start
  and join) come from a table in the compiler: `runtime` has no syntax to state them. `io` does not
  include the allocator's, a lock's or a thread's system calls, so a function with `without(io)`
  may still make those.
- Effect checks run only on programs without other errors: a type or borrow error hides an E1401
  until it is fixed.
- A value that is moved out is still charged the drop where it was made. A fn that makes a
  `vec.Vec` and returns it is E1401 under `without(alloc)`: the drop of the local reaches
  `runtime.alloc.free`.
- A call that cannot run still counts. Each of these is E1401, `noisy` being a fn that prints:
  `fn f(n: u32) -> u32 without(io) { return n; noisy(); }`; the same after a call to a `-> never`
  fn, after `break` in a loop, and `if false { noisy(); }`. Delete the dead call.

## 2. What the interpreter's checking does not see

The test harness and the fuzzer run programs in the IR interpreter with borrow tracking (a tag per
reference, a permission per byte) and a memory mode (use after free, double free, out of bounds,
uninitialised reads). `check` runs the positive conformance programs under it and the fuzzer runs
generated programs under it: it is the oracle that finds checker holes. Its limits:

- A verdict covers the run it saw: this input, this schedule. Other inputs and interleavings are
  not judged. A broken `from` clause is reported only for a call that breaks it, and not through
  raw pointers.
- The interpreter runs spawned bodies on one thread, each at its join. There is no data-race
  detection. A program in which one thread waits for another before the join (a spin on an atomic)
  never ends there; the gate lists such programs as not runnable.
- Leaks, destructors that never run, and hangs are reported by neither mode.
- The memory mode, not the tracker, finds out-of-bounds and misaligned accesses, uninitialised
  bytes other than moved-out ones, a second `free`, and `p.read()` of a Copy type.
- The captures of a function value are not walked: a function value that is a param, a field or a
  result is not judged for what it captures, and its captures are not protected by a call. A value
  with more than 4096 reference words is not judged for a `from` clause.
- A pointer that lost its tag (integer arithmetic other than `+` and `-`, the bytes of a reference
  copied one by one, a pointer from `extern "C"` or a system call) acts as the root of its
  allocation: its accesses can hide conflicts a finer rule would report, and two such pointers
  never conflict. Accesses made by `extern "C"` code and the kernel are not seen at all.
- Moves are tracked less exactly than the rules say: the moves of a guarded or-pattern have no row,
  and a drop of a moved-out place is reported only when the destructor's glue reads it.
- Programs that contain `unsafe` are not required to run clean in `check`, and the fuzzer's
  programs come from fixed classes of borrow constructs: a hole outside them is not found by it.
- The interpreter does not read `[run] stack`. It has two limits of its own, both trap
  `StackOverflow` (exit 101, the same report line as a native overflow): 20,000 live frames, and
  8 MiB for the slots of all live frames together. `rec fn depth(n: u64) -> u64` with
  `depth(30000)` runs natively and exits 101 there; a fn with `let mut a: [u8; 9000000] = [0;
  9000000];` runs natively and traps there, and so do two nested calls that hold 4,300,000 bytes
  each.

## 3. Valid programs the checker rejects

- `v.push(*v.get_ref(0));` (a Copy read in the argument of a `&mut self` call) is E0501. Write
  `let x = *v.get_ref(0); v.push(x);`.
- `*g.get_mut() = *g.get() + 1;` on a `MutexGuard` is E0501: the left place is borrowed before the
  right side is read. Write `let x = *g.get(); *g.get_mut() = x + 1;`.
- Lock holding is judged per local type, not per value (E1405). A local whose type can hold a guard
  (`Option<MutexGuard<..>>`, `vec.Vec<MutexGuard<..>>`, an enum with a guard variant) counts as
  holding one while it is in scope: `g = Option.None;` does not release it; an empty
  `vec.Vec<MutexGuard<..>>` blocks the lock that is pushed into it; a closure that takes the guard
  out of a captured `Option` and drops it leaves its caller holding it; an array
  `[E.G(guard), E.K(l)]` is judged as if the guard could be held while `l` drops, though elements
  drop in index order. Release a guard by moving it: `mem.drop(g)`, or a `match` whose every arm
  moves the `Option` (`other => { mem.drop(other); }` in place of `Option.None => {}`). Make a
  `vec.Vec` of guards after the lock is taken (`let g = m.lock(); v.push(g);`), and keep a guard
  and a value whose drop locks in separate locals of their own types, the guard declared last.
- A closure that passes a closure to a helper `fn call_ref(f: fn() -> &u64) -> &u64 from f`, which
  calls it, is E1409 (a call cycle through the closure literal).
- A closure that returns what an inner closure returns is E0512 when that is a reference to a
  function value or a view struct with a function-valued field (even a closure without captures):
  the type may hold a function value, so the inner closure's environment loan rides on the result.
- A view keeps every loan stored into it, also after the field that held it is overwritten:
  `h.r = &z; h.r = &x; z = 4; ok(h.r);` is E0501 (`cannot assign to z`), and
  `put(&mut h, &f()); h.r = &x; ok(h.r);` is E0501 `temporary` (`put` stores its second
  argument). A reference local that is rebound (`r = &x;`) lets go. Assign the whole view:
  `h = H.{ r: &x };`.
- A `match` that moves its scrutinee in some arms only leaves it in place in the others, holding
  every loan its type can hold, also in a variant that holds none: with `o` an `Option` of a view
  that has a `Drop` impl and borrows `x`, `match o { Option.Some(w) => { .. } Option.None => {} }`
  and then moving `x` is E0501. Move it in every arm: `other => { mem.drop(other); }`.
- Dropping a value whose type has a destructor that can reach a reference counts as an access to
  every reference the value holds, also one no destructor reaches. With `view struct W { r: &mut
  P }` that has `impl Drop` and `view struct H { w: W, r: &mut P }`,
  `fn f(h: H) -> &mut P from h { return &mut *h.r; }` is E0501 `cannot drop h`; with
  `w: vec.Vec<u32>` (no reference behind the drop) it is accepted. Take the value apart and
  return the field itself: `let H.{ w: _, r } = h; return r;`.
- A `&mut` to a view with drop glue stored into a container counts as stored into that view as
  well: after `root.kids.push(&mut leaf);` with `view struct N { v: i64, kids: vec.Vec<&mut N> }`,
  `leaf` holds a borrow of itself, and its drop at the end of the fn is E0501 `cannot drop leaf`,
  also after `mem.drop(root)`. The same holds without recursion (`vec.Vec<&mut L>`, `L` a view
  with a `vec.Vec` field); a child type without drop glue (`next: Option<&mut N>`) or `&`
  children (`vec.Vec<&N>`) are accepted. Keep the nodes in one `vec.Vec<N>` and name children by
  index (`kids: vec.Vec<u32>`).
- `c.select(f, g)` of two function items is a call result, which has the plain fn type: where
  `fn() without(io)` is expected, `run(c.select(p1, p2))` is E1403 even when neither prints, though
  `run(p1)` and `run(if c then p1 else p2)` are accepted. Bind each item to a local of the wanted
  type first (`let a: fn() -> u32 without(io) = p1;`) and select between the locals. A value
  `if` or `select` that mixes an item with the clause and one without is E0201: convert the one
  without it alone first.
- A fn value read out of an array, a tuple or the arms of an `if` is not followed: a call through it
  reaches every function value of its type. Given `fn a() {}` and `fn b() { call(); }`, with `b`
  used as a value somewhere (`let keep: fn() = b;`),
  ```
  fn call() {
      let fs: [fn(); 1] = [a];
      let f = fs[0];
      f();
  }
  ```
  is E1402 (`b -> call -> b`); with `fn call() without(trap)` and a trapping `b` it is E1401. A
  closure that makes the call, when the closure has the fn type it calls, is a cycle through
  itself, E1409:
  ```
  let c = fn() {
      let fs: [fn(); 2] = [a, b];
      let f = fs[0];
      f();
  };
  ```
  A `spawn` body and a `thread.join` arm are such closures (`spawn` captures no collection of fn
  values: fn values are not `Sync`, E0802, so the collection is made inside the body). A fn item
  that makes the call for the closure does not help. Rewrites: bind the value to a plain local
  (`let f: fn() = a;` is followed), call by `if k { a(); } else { b(); }`, give the fns a param so
  their type differs from the closure's (`[fn(u32); 2]` in a `fn()` closure). For the E1402 case,
  also mark the cycle `rec fn`.
  The value need not come from a collection: a fn value that a call returns is not followed either.
  With `fn pick() -> fn() -> u64`, the closure `fn() -> u64 { let g = pick(); return g(); }` is a
  cycle through itself, E1409; the same body in a fn item is accepted. Of the rewrites above, the
  plain local does not apply when the value really comes from `pick`: give the closure a different
  type, or move the body into a fn item.
- One slot stands for the function field of every `iter.Map` of one instance type: a
  `.map(loud)` anywhere gives its fn to every other `Map<u32, u32>`, a dead one included. With
  `fn loud(x: &u32) -> u32 { io.print("loud\n"); return *x; }` mapped in `main`,
  `fn total(xs: &[u32]) -> u32 without(io)` that runs `for v in iter.iter(xs).map(quiet)` over a
  quiet `fn quiet(x: &u32) -> u32` is E1401. `loud` returning `u64` (another instance) is
  accepted, and so is `for x in iter.iter(xs) { s = s +% quiet(x); }`.
- Generic code that stores what a `from copy` call returns, through a `&mut` view param, is
  E0513 when the clause names the iterator by `copy`:
  ```
  fn drain<T: view>(it: &mut iter.SliceIter<T>, out: &mut vec.Vec<&T> from copy it) {
      loop {
          match it.next() {
              Option.Some(x) => { out.push(x); }
              Option.None => { return; }
          }
      }
  }
  ```
  The same fn with `T` replaced by `u32` is accepted. Write `from it` (the caller keeps `it`
  borrowed while `out` is used), or do not make the fn generic.
- Generic code over masks or vectors: only ints, floats and `bool` have the built-in operator
  impls, so `T: Not + Copy` at `mask32x4` or `u32x4`, and `T: Add + Copy` at `f32x4`, are E0901.
  The operators work on the vector or mask directly; for generic code write a trait of your own
  and implement it for each type
  (`impl Inv for mask32x4 { fn inv(self) -> mask32x4 { return ~self; } }`).
- A vector type wider than the target's registers cannot be named in a fn whose `needs(..)` do not
  cover it, nor in a struct or enum field: `vec.Vec::<f32x8>.new()` in a plain fn is E0702 at the
  type argument, although the `Vec` holds no vector by value. Put `needs(avx2)` on the fn and reach
  it by a `dispatch`, or build for `x86_64-linux-v3`. On AArch64 no feature covers 256 bits.
- The exhaustiveness search has a limit (patterns nested more than 2,000 levels, or over
  1,000,000 search steps): a `match` past it is E0300 `limit`. Every `let`, even one that binds a
  plain name, is checked like a `match`: a struct or tuple with 2,000 fields or more in all (a
  nested struct or tuple counts as a field) is E0300 at `let s = S.{ f0: 0, f1: 1, ... };` and at
  `let (a0, a1, ..., a1999) = t;`; with 1,999 fields it is accepted, and a fn param of that type
  is accepted. Keep bulk data in an array (`[u32; 5000]` in a tuple is accepted) or a `vec.Vec`.
- The frame of a fn is limited to 704,643,072 bytes: E0911 when the values it holds (parameters,
  bindings, every expression value but a place read in place, two slots for a value `if` / `match`,
  each call's result, receiver and arguments) take more together. A back end's frame holds 1 GiB
  and measures up to 1.5 times that sum for some shapes (a local filled from an array literal, an
  array passed by value), and the bound allows for that for every shape. A fn whose frame is
  smaller than 1.5 times its sum can be rejected although it would run: `let a: [u8; N] = if c
  then [1; N] else [2; N];` is rejected from N = 140,929,000 and its frame is the sum, 5N + 44
  bytes. Keep a large array in a `vec.Vec`, or in a part of the fn that is split off.

## 4. Rules that are deliberate

These keep the checker small and its rules checkable; each has a rewrite. They reject programs
that would run.

- **Moves go strictly left to right.** An operand that reads a value another operand of the same
  expression moves is E0502: `s.consume(s.len())` and `f(s, s.n)` are rejected. Write
  `let n = s.len(); s.consume(n)`.
- **No partial moves.** A value that is not Copy moves whole, or is taken apart whole by a pattern:
  `let x = s.f;` of a non-Copy field is E0504. Use `mem.replace`, `mem.swap`, or
  `let S.{ f, g: _ } = s;`. A value whose type has `impl Drop` is never taken apart.
- **A `static` is never moved from** (E0504).
- **`for` names its borrow.** The source of a `for` is a range, a reference, an iterator or an
  array value; an array or slice place is E0219: write `for x in &arr` or `for x in &mut arr`.
  A chunk binding borrows a per-block copy of the chunk and cannot leave its block.
- **A `match` guard only reads**: no move, no assignment, no `&mut` borrow, no `?` (E0505).
- **A temporary ends with its statement; there is no constant promotion.** `let r = &f();` is
  E0501: write `let t = f(); let r = &t;`. `&4u32` cannot be returned.
- **A write through a reference to a temporary is not an error.** An assignment to a temporary or
  a `const` (`mk().x = 1`, `C.x = 1`, `arr()[0] = 5`) is E0503, but a write through a `&mut` of
  one is accepted and lost with the value at the end of the statement: `hh().pm().x = 1` (its
  destructor sees the new value), `*&mut C.x = 1` and `C.set(2)` (the const keeps its value).
  Bind the value first: `let mut t = hh(); t.pm().x = 1;`.
- **Each cast of a `&mut` to a raw pointer is a new reborrow.** `m as *mut T` is `&mut *m`, so
  two casts of one `m` are two exclusive borrows: after `let p = m as *mut u32; let q = m as *mut
  u32;`, a write through `q` (or through `m`) makes `p` unusable, and using `p` is undefined
  behaviour that borrow tracking reports. The compiler accepts it and the native program runs.
  Cast once and copy the pointer: `let q = p;`.
- **`Drop` impls only on a struct or enum of the same package** (E0419). A `Drop` impl may not
  overlap a `Copy` impl by head (E0416), even if no instance is both. A destructor is no function
  value (E0909).
- **Operator traits only on `Copy` types** (E0422); an operator copies its operands, so a generic
  one needs `T: Add + Copy`.
- **No operator impl on a fn type** (E0420): the operator traits and fn types both belong to
  `std`, so `impl Add for fn(u32) -> u32` is outside both packages. A trait of your own may be
  implemented for a fn type: `impl Twice for fn(u32) -> u32` gives `f.twice(1)`.
- **A call through a function value is as precise as its flow.** A function-typed place is
  followed only while every use of it is one of: its declaration, a plain assignment to it, a copy
  of its value as a callee, a call argument, a `let` initializer, the source of an assignment, a
  struct-literal field, or a returned value. Any other use (a borrow `&x` / `&mut x`, a method
  receiver, an operator operand, an `if` or `match` arm, an array or tuple element, a struct
  pattern) makes its calls reach every function value of the same type. With `without(..)`, a
  lock rank, `rec` or the stack bound this can give a false error.
- **A method call on a param with two bounds of one trait is ambiguous** (E0908):
  `U: Conv<u32> + Conv<u64>` needs `Conv::<u64>.c(a)`.
- **Const parameters are unsigned integers** (E0201), and a const argument has the type of its
  parameter: a `usize` const cannot be a `Mutex` rank, which is a `u32`.
- **A `Drop` impl on a call cycle needs `rec fn drop`** (E1402).
- **One type depth limit (1,000)**: a type nested deeper is E0005.

## 5. Targets and tooling

- Linux only.
- `check` accepts some programs that `build` cannot build; the build exits 2 with a reason:
  - `dispatch fn main() -> i32 { _ => m0, }` and `fn main<T>() -> i32 { return 0; }` build to "the
    program has no `fn main` in its root module"; `fn main(x: u32) -> i32 { return 0; }` to
    "`codegen` is not implemented yet: `main` with parameters". Write a plain `fn main() -> i32`
    that returns `pick()` of a `dispatch fn pick() -> i32 { _ => base, }`.
  - A fn with a 256-bit vector param and `needs(avx2)` used as a value builds on x86-64 only.
    `fn w(v: f32x8) -> f32 needs(avx2)` with `sink(w)`, `fn sink<T: view>(x: T) -> u32`, passes
    `check --target aarch64-linux`, and the AArch64 build fails with "reference to fn `main.w`
    without a body". A written `fn(f32x8) -> f32 needs(avx2)` type is E0702 there. Call `w`
    where the fn has `needs(avx2)`, or use `f32x4` (a value of it builds).
- Generated code is slow next to `-O2`: `--release` assigns registers and cleans up slice
  accesses, but nothing is inlined, hoisted out of loops, or vectorized.
- Two searches of `--release` give up on a huge function. The result is the same program, slower:
  - Register assignment stops after 50 million block visits of its liveness walk and keeps every
    value of the function in its stack frame (the speed of a debug build). A function of values
    `let a0: u32 = n +% 1;`, each read at the end, then `if n == 7 { s = s +% 1; }` (a different
    constant each) and a loop of 10^9 steps: 0.65 s with 500 values and 500 `if`s, 0.59 s with
    5,000 and 4,000, 1.65 s with 10,000 and 4,000 (a debug build: 1.54 s). Split the function.
  - The merge of a repeated loop step (`let nx = i + 8;` after the test `i + 8 <= n`) looks at
    most 4,096 blocks for the loop head: with 4,096 `if`s between it and the back edge the
    `add 8` appears once in the code, with 4,500 twice.
- No C interop yet: `extern "C"` imports check, but a program that reaches one cannot be built
  (there is no C linker step), and `[target.<T>.link]` is reserved ([cli.md](cli.md)).
- GPU: design only. There is no PTX back end and no kernel checks beyond the front end. The
  optimizer is not started.
