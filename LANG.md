# Stave language reference for agents (v0.7; grammar: contracts/grammar.ebnf)

Stave = Rust-family syntax + borrow checker, **no lifetimes, no macros, no implicit anything**, SIMD in type system.
Read whole file before writing Stave. Assume Rust **only** where this file is silent. Compiler is truth; error codes are in contracts/diagnostic-codes.md. Stage0 generates vector code on x86-64 and AArch64 (§4 Status); the optimizer and the GPU are not built. Items marked *planned*, §4, and §7 describe design.

## 1. Differences from Rust (most common agent mistakes first)

| Rust | Stave | Note | Why |
|---|---|---|---|
| `x` tail expr returns | `return x;` | Blocks are statements, never values (E0021). Non-unit fn: a path without `return` = E0207; unit fns may fall off the end | one way to return |
| `let v = if c { a } else { b };` | `let v = if c then a else b;` | Value-if: `then`/`else`, no braces. Chains: `if a then 1 else if b then 2 else 3` | blocks are not values |
| `match` as value | `match x { P => e, P2 => e2 }` | Value arms: expressions, comma-separated | blocks are not values |
| `match` as statement | `match x { P => { ... } P2 => { ... } }` | Statement arms: blocks, **no commas** | one spelling per construct |
| `Some(x)`, `None`, `Ok(v)` | `Option.Some(x)`, `Option.None`, `Result.Ok(v)`, `Result.Err(e)` | Variants **always qualified** (E0103), in patterns too | a bare pattern name is always a binding |
| `a::b::C`, `use a::b;` | `a.b.C`, `import a.b;` / `import a.b as c;` | Sole path separator (`a::b` = E0023). `import a.b;` binds `b`; `as c` binds `c`. `::` only before generic args: `f::<T>()`, `v.m::<4>()` (on a built-in that takes none: E0201). First segment: own package, else a dependency, else a module of this package; dependency and module both named `x` = E0107 (write `import mypkg.x;`); a module path starting with a built-in package name (`src/std/x.stv` is `std.x`) = E0107, never that package's | one separator; parse with no symbol table |
| `pub use`, `geo::shapes` via `use geo` | none | No re-exports: `import geo;` gives `geo`'s items only, not `geo.shapes`: `import geo.shapes;` | one path per item |
| prelude `Vec`, `String`, `drop` | `import std.vec;` → `vec.Vec::<u8>.new()`; `import std.string;` → `string.String` | Prelude = `pub` items of `std.core` (`Option Result Copy Send Sync Flat Frozen Drop Display Write`, operator traits) + built-in type names | small, explicit prelude |
| `mod`, `use {..}`, `crate::` | none | Module = file. `src/a/b.stv` is `a.b` (a dot in a file or directory name = E0107); `src/a.stv` and `src/a/` may coexist. Imports acyclic (E0102) | file tree is the module tree |
| `use Trait;` to call its methods | no import needed | `x.m()` looks in inherent impls, then every trait impl of x's type. Two traits with `m` = E0908: call `Trait.m(x)` | method lookup needs no import state |
| `let k = 7;` | `let k: i32 = 7;` or `7i32` | Literals have **no default type** (E0202). Type comes from annotation, param, field, variant payload, tuple element, other operand, or return type; an index is `usize`; a generic arg position takes the type inferred from the non-literal args, else a suffix | no inference across uses |
| `S { a: 1, b: 2 }` | `S.{ a: 1, b: 2 }` | Struct literals **and** struct patterns use `.{` (one token, no space): `Shape.Rect.{ w, h }`. `S { … }` = E0014. Generic: `Cell::<u32>.{ v: 1u32 }`, or args inferred from the expected type or fields. `Self.{ .. }` inside impls | `{` after a name always opens a block: LL(1), parallel parse |
| `S { a, b }` (shorthand) | `S.{ a: a, b: b }` | No field-init shorthand in literals. Pattern shorthand `S.{ a, b: p }` is allowed. Declarations keep plain `{`: `struct S { a: i32 }` | one spelling per construct |
| `x as usize < y` | `(x as usize) < y` | E0011. Also `a < b < c` is E0012 | `<` after a cast type is always generic args |
| `x as u8` (truncates) | `as` lossless only | Lossy `as` = E0204. Other casts: below table | every lossy conversion is named |
| `i32 + i64` | `a as i64 + b` | No implicit numeric conversion (E0201) | conversions are always visible |
| `+` wraps in release | `+ - *` **trap** on overflow, every build | Wrapping/saturating forms: below table. `/` by 0, `MIN / -1`, `MIN % -1` trap. `255u8 + 1` compiles and traps at run time (no folding outside `const`) | debug = release; no silent wrap |
| `x << 40` on u32 | traps (const amount: E0205) | Amount has the left operand's type (`x << 3`: `3` is u32). `<<%` masks amount by bits-1. `>>` arithmetic if signed, logical if unsigned | CPUs disagree on big shifts; trap is portable |
| `!x` on int | `~x` | `~` = bitwise NOT. `!` = bool NOT | `!` is bool-only: no truthiness |
| `'a` lifetimes | none (E0024) | References are second-class (§3) | borrow check per function, no lifetime solver |
| `static mut X: T` | `static X: T = e;`, `const X: T = e;` | A `static` is one immutable global shared by all threads: reads and `&X` are safe, assignment, `&mut X`, and `&mut self` calls are E0503. T must be `Sync` (E0417): plain data, `atomic.U64`, `sync.Mutex<T, R>`, `sync.Once<T, R>` (changed through `&self` methods). `e` is a const-expr; no ref/view types (E0413) except `&str` literals (a `&str` static checks but does not build yet: use `const`). `static mut` = E0809 (std/runtime only) | shared globals must be Sync |
| `-> !`, `loop {}` as a value | `-> never` | `never` only after `->` (E0209). A call to a `-> never` fn ends the path (no `return` after it, E0207); a `-> never` fn must end every path in such a call or a `loop` without `break` | `!` stays bool-only |
| `impl Drop`, `drop(x)`, `x.drop()` | `impl Drop for S { fn drop(&mut self) { .. } }`; `mem.drop(x)` (`import std.mem;`) | Runs at scope end (reverse declaration order) and when an initialized place is overwritten (`x = v`, `*p = v`). `x.drop()` = E0909. `impl Drop` on a Copy type = E0416 | destruction is never written by hand |
| `ptr::read(p)`, `ptr::write(p, v)` | `p.read()`, `p.write(v)` (unsafe) | `*p = v` drops the old pointee; `p.write(v)` does not (uninitialized slot); `p.read()` moves out, the slot is then logically uninitialized. `*p` by value only for Copy | the drop is visible at the call |
| `std::mem::size_of::<T>()` | `size_of::<T>()`, `align_of::<T>()` | Built-in, no import, const-evaluable. A concrete type without a layout (unsized) = E0911 | layout queries without a module |
| `-> &T` | `-> &T from x` | Returning a borrow of a param **must** name it (E0510); no `from` = result borrows nothing (literals, fn items) | names the origin without lifetime params |
| `unsafe { }` | `unsafe("why this is sound") { }` | Obligation string mandatory (E0013). Also before an expr, `fn`, `trait`, `impl`: `unsafe("..") fn` | every obligation is written and inventoried |
| `where T: X` | `<T: X + Y>` | Bounds inline only (E0022) | one spelling per construct |
| closure syntax | `fn(x: i32) -> i32 { return x + 1; }` | Closures: explicit param types, captures by ref only (`&mut` if the body mutates it). Type is the fn type `fn(i32) -> i32`, a view (§3) | no inference; non-escaping needs no heap |
| `println!`, `vec!`, `format!`, `assert!`, `#[derive]` | none | No macros, no attributes. Output: `io.print(f"n = {n}\n")` | no hidden code; linear compile time |
| `#[derive(Clone, Copy)]` | `impl Copy for S {}` | Empty impl; every field must be Copy (E0415). Built-in Copy: ints, floats, `bool`, `char`, `&T`, raw pointers, arrays/tuples of Copy. `&mut T` never | no attributes; one spelling |
| `format!("{}", a + b)` | `f"{name}"`, `f"{p.x}"` | f-string holes: **names or field paths only** (E0006, Notes); value must impl `Display` (ints, floats; `bool` → `true`/`false`, `char` → the character). An f-string is an owned `String`, borrowed automatically where `&str` is expected (`&String` → `&str`). Keep it: `let s = f"..";`, pass `&s` | lexer stays regular (parallel lexing) |
| `/* */` | `//` only | No block comments (E0029; write `a / *p`, not `a/*p`) | no nesting state: regular, parallel lexer |
| `0..`, `..n`, `..` | `a..b`, `a..=b` | Both ends required, same type | only two range types |
| `'label: loop`, `break v` | `loop { }`, `break;`, `continue;` | No labels, no break-with-value | blocks are statements; simpler CFG |
| default trait methods | none | Trait bodies hold signatures only: `fn f(&self) -> T;` | per-function checking |
| overloading, truthiness, null | none | `if n != 0`, not `if n` | one meaning per name; explicit tests |
| `mut self` param | `self`, `&self`, `&mut self` only | Rebind: `let mut s = self;` | params are immutable bindings |
| `x @ P`, `..` rest | not in grammar | Write the full pattern; bind the value by name to reuse it | not admitted yet (simplicity) |
| `v.pop();` (value ignored) | `let _ = v.pop();` | A non-`()`, non-`never` expression statement is E0210 | an ignored `Result` is always visible |

Casts and overflow operators (outside the table: they contain `|`):
- `as` lossless only. `as%` wraps or truncates. `as|` saturates (float NaN → 0). `as?` returns `Option`. `as~` converts to a float type, rounding to nearest even (int→float, f64→f32, f32→f16 lanes). To its own type (bit copy): scalars, vectors, masks, refs, raw pointers only (else E0204).
- `+% -% *% <<%` wrap. `+| -| *|` saturate. Compound forms: `+%= -%= *%= <<%= +|= -|= *|=`.

**Not in Stave** (write the right column):

| Rust form | Stave |
|---|---|
| `if let`, `while let` (E0019), `let … else` (E0020) | `match` (statement or value form) |
| `let x;` (no initializer) | `let x: T = e;` (initializer required) |
| `ref x`, `&x` patterns, slice patterns `[a, b]` | bind by value, then borrow; index the slice |
| tuple structs `S(i32)`, unit structs `S;` (E0026) | `struct S { v: i32 }`, `struct S {}` |
| `mut x: T` params (E0025) | `x: T` then `let mut y = x;` |
| supertraits `trait A: B`, associated types/consts | separate bounds `<T: A + B>`; generic params; top-level `const` |
| nested `fn` items | top-level `fn` |
| `!` (never type, E0027), `impl Trait for !` | `-> never` |
| `dyn Trait`, `impl Trait` (E0028) | generics `<T: Trait>`, or fn types |
| `pub(crate)`, struct update `..base`, `A = 1` discriminants | `pub`; write every field; undecided (§8) |
| `b'a'`, `b".."`, `r".."` | the byte value (`97u8`); `"…"` with `\` escapes |
| `i128`, `u128` (no such types) | multi-word math on `a.mul_wide(b) -> (hi, lo)` (i32/i64/u32/u64), `a.mul_hi(b)` (high half; any fixed-width int) |

Notes:
- `unsafe("..") expr` is for value positions only. At statement start `unsafe` always opens a block: `unsafe("..") { *p = 1; }`, not `unsafe("..") *p = 1;`.
- An `unsafe("..") fn` states the **caller's** obligation (calls need `unsafe`). Its body is not unsafe: unsafe ops inside need their own `unsafe("..")` (E0803). Methods too: `pub unsafe("..") fn` inside `impl`.
- `c.select(a, b)` on a `bool` (built in, no import): `a` if `c`, else `b`. `a` and `b` have one type, any type (E0201). Both are always evaluated, left to right, so `(i < n).select(xs[i], 0)` traps for `i >= len`; then one is picked without a branch (constant time); the other is dropped at the end of the statement. The mask form `m.select(a, b)` is lane-wise (§4); `if`/`match` evaluate one arm.
- Or-patterns, at the top of a `match` arm only (not nested, not in `let`/`for`): `Tok.Plus | Tok.Minus => 1u8,`. Every alternative binds the same names with the same types and modes (E0302). The first alternative that matches binds them; the guard then runs once (false: next arm). An alternative that never matches is E0301.
- A pattern binds a name once: `(a, a)` or `P.{ x, y: x }` is E0105.
- f-string holes: a name, `self`, or a field path (`{n}`, `{self.n}`, `{p.x}`). No tuple steps (`{t.0}`) or expressions: bind to a local first (`let a = t.0;` then `{a}`).

Same as Rust: `fn`, `let`/`let mut`, `struct`, `enum` (tagged), `impl T`, `impl Trait for T`, `trait`, `pub` (else private to the module, E0106; none on trait impl methods, E0907), method auto-ref (`x.m()` with `&self`/`&mut self`), auto-deref on `.` through any number of `&`/`&mut`, implicit reborrow of `&mut` args, `&T`/`&mut T` (shared xor exclusive), moves, `?`, `[T; N]`, `[T]`, tuples (`t.0`), `[e; n]` (`e` Copy unless `n` is the constant 1, E0418), `while`, `for x in a..b`, `match` exhaustive (E0300), guards `P if c =>`, `*const T`/`*mut T` (deref unsafe; casts `&x as *const T`, `&mut x as *mut T` safe), `type A = B;`, `const X: T = e;`, `Self`.

## 2. Lexical, keywords, items

- Strings and chars must end on their line: no raw newline inside `"…"`, `f"…"`, `'…'` (E0015). Use `\n`.
- Identifiers are ASCII: `[A-Za-z_][A-Za-z0-9_]*`. Non-ASCII only inside literals and comments; `@ $ \ #`, backtick, control bytes elsewhere = E0018.
- A first line starting `#!` is a comment (shebang: `#!/usr/bin/env -S stave run`). `#!` anywhere else = E0018.
- Float text (`Display`, f-strings): shortest round-trip digits: `6.0`, `0.1`, `1e-6`, `1e21`, `inf`, `-inf`, `NaN`, `-0.0` (exact form: std/README.md Floats).
- Ints: `1_000`, `0xff`, `0b1010`, `0o17`, suffix `5u8`. Floats need a digit after `.` (`1.0`, not `1.`), suffix `1.0f32`. `-128i8` legal (minus folded in), `128i8` = E0203.
- Statements end with `;`. `if`/`while`/`for`/`loop`/`match` statements take no `;` after the block.
- Syntax errors: a file with unbalanced brackets gets one E0016 and nothing else. Otherwise each item shows only its **first** syntax error and is dropped.
- Reserved (never use as names): `fn let mut const static struct enum trait impl for in while loop if then else match return break continue pub import as from needs dispatch unsafe extern self Self true false tail masked scalar exact align copy type view scope spawn strict emulate test allocator kernel requires ensures rec without invariant decreases ghost pure spec forall exists derive async await yield` (the last eleven have no meaning yet; `result` is not: Contracts bullet).
- Integer types: `i8..i64 u8..u64 isize usize`. Also `f32 f64 bool char str`, and `never` (return type only). Built-in names, not keywords: a local of the same name shadows them. `usize`/`isize` are 64-bit on every target (`u64 as usize` and back are lossless `as`; `usize as u32` needs `as%`/`as|`/`as?`); `&[T]`, `&mut [T]`, `&str` are two words (pointer, length).
- `test fn name() { }` runs under `stave test`, not in builds. With params it runs once per generated input: `test fn t(xs: &[u32], k: u32) requires(xs.len() <= 17 && k < 8) { .. }`. Param types: ints, `bool`, `char`, `f32`, `f64`, and arrays, tuples, visible structs, `&[T]` (with `requires(p.len() <= N)`) of those; return `()` (else E0214). `requires` clauses bound and filter the inputs; a failing input prints with the command that reruns it (generation, report: contracts/cli.md `stave test`).
- Fn types: `fn(i32, &T) -> &T needs(avx2)`. Closures and fn items coerce to fn types. No `from` in fn types: a call through a fn value borrows from **all** its ref/view args and from the fn value itself (a closure may return a borrow of a capture held by `&`; of one held by `&mut`, even through a dereference: E0512). `needs` and `without` after a fn type belong to it: `fn f() -> fn() needs(x)` = returned fn needs x; for f write `-> (fn()) needs(x)`. A `fn(..) needs(F)` value is usable as `fn(..) needs(G)` only if F ⊆ G ∪ the enclosing fn's needs ∪ the target baseline (E0707).
- C FFI: `unsafe("sigs from zlib.h 1.3") extern "C" { fn crc32(c: u64, p: *const u8, n: u32) -> u64 without(alloc, io, lock, block); fn printf(f: *const u8, ...) -> i32; }`. The obligation is required, the ABI string is `"C"` (E0811); calls are unsafe; an import has every effect except those its `without` lists (a claim of the obligation). C signature types: ints, `f32 f64 bool`, raw pointers (not to `[T]`/`str`), `()`/`never` returns (else E0804: pass structs by pointer). Variadic args: `i32 u32 i64 u64 isize usize f64` or such pointers (E0813: write `x as f64`). Export: `extern "C" fn add(a: i32, b: i32) -> i32 { return a + b; }`, also with `unsafe("..")`, `without`, `requires`, `ensures`; callable from Stave, not a fn value (E0807). Symbol = the fn name, unmangled, once per program (E0812). No libc is linked by default (std reaches the OS through the target's `runtime` package: raw syscalls on Linux), so imports cannot be linked yet: how is undecided (§8).
- Struct layout: fields in declaration order with C alignment and padding, the same on every target. Enums and tuples have no fixed layout.
- Planned: `allocator my.alloc;` replaces the global allocator (its module provides `try_alloc try_realloc free`; see contracts/intrinsics.md). Today it only parses and is checked (at most one per binary, E0104); std keeps the runtime allocator. Allocation failure panics `OutOfMemory`; the `try_` methods of `Vec`, `String` (`try_with_capacity try_reserve try_push`, `try_extend_from_slice`/`try_push_str`), and `std.arena` (`try_new try_alloc`) return `Result.Err(vec.AllocError)` instead, except `Vec.try_push(x)`, which returns `Result.Err(x)`.
- Recursion marker: `rec fn walk(n: &Node) -> u32 { .. }`, also on methods, `test rec fn`, `unsafe("..") rec fn`; not on trait signatures, closures, `dispatch`, `kernel`, `extern` fns. Every fn of a call cycle must carry it (E1402 names the cycle); calls through fn values count, `dispatch` counts as its arms, each instance is checked. A cycle through a closure literal is E1409: make the recursive part a `rec fn`. Dropping a recursive value (`struct N { kids: vec.Vec<N> }`) needs no marker. Stack: each thread or task stack is sized from the compiler's bound of what it can call; one that reaches a call cycle (recursion or a drop cycle) gets `[run] stack` and traps `StackOverflow` past it (contracts/cli.md). A `rec fn` on no cycle is warning E1411.
- Deny clause, after `needs(..)`, before `requires`: `fn f(xs: &[u32]) -> u32 without(alloc, trap, lock) { .. }`. Effects: `alloc trap recursive lock unsafe io block park` (else E1404); neither the fn nor anything it can call may have a denied one (E1401 names the call chain). `trap`: an op that can still trap after the proven bounds checks are removed (the same in every build), or a call cycle; `unsafe`: an `unsafe(..)` outside std and runtime; `io`: a system call, except the allocator's, a lock's, or a thread's; `block`: may wait for something outside the fn: a child task, a file, a completion queue, a device (std: a `scope` end, `thread.join`, `par_*`, a file read, write, or open); `park`: waits on a completion queue, a part of `block` (denied with it); never in a fn type (no fn value parks), an `extern "C"` import, or a call cycle; an `extern "C"` import has every effect its `without` does not list. A generic fn is checked per instance (not when never instantiated). Fn types carry it: `fn(..) without(D1)` is usable as `fn(..) without(D2)` only if D2 ⊆ D1, a closure or fn item only if it has no effect in D2 (E1403); a call through it has none of D2. Not on trait methods.
- Contracts, after `-> R needs(..) without(..)`, on fn items and inherent methods (also `unsafe(..) fn`): `fn push(v: &mut vec.Vec<u32>, x: u32) requires(x < 1000) ensures(v.len() == old(v.len()) + 1) { .. }`. Checked in every build: `requires` clauses in order at entry, `ensures` clauses in order at each `return`, `?` error return, and the end of a unit body. A false clause panics `Contract`, located at the call for `requires`, at the `return` for `ensures`. Inside `ensures`, `result` is the returned value and `old(e)` the value of `e` at entry (after `requires`); `e` must be Copy (E0213). Both are contextual names inside clauses only (a param named `result` / `old` in a fn with `ensures`: E0212 / E0213). Operands (E0211): calls only of safe fns whose params are `&` or Copy; no `unsafe`, closures, f-strings, `?`, `&mut` borrows, statics, intrinsics; a `&mut` param may be read (never written or passed as `&mut`); an `ensures` reads params only of `&`, `&mut`, or Copy type (the body may have moved the others; a `&mut` param it moved is E0502 there), and not `v` when the return type holds a `&mut` `from v` (a Copy `old(v..)` is fine). A clause is borrow-checked as fn code where it runs, removed or not: what `old(e)` borrows stays borrowed until the `ensures` clauses (`old(&*p)` then `*p = 1;` = E0501: write `old(*p)`). A fn called in a clause must not have `io`, `lock`, or `block` (E0211 `impure-call`; `alloc`, `trap` are fine). Every value a clause touches (each expression, call result, receiver, argument, field read, fn value called) must be `Frozen` (§3), and no callee may reach a `static` that is not: no atomics (`load()` included), `Cell`, `Mutex`, or fn-value calls (E0211 `frozen`, `static`); a generic `T` there needs `T: Frozen`. A check the compiler proves true is removed (a `requires` implied by the clauses before it: warning E0216); one that always fails is an error (`requires`: E0215; `ensures`: E0217; a call that always fails its callee's `requires`: E0218); any other stays a runtime check (what is proved: contracts/lang-details.md). A kept `ensures` check gives its fn `trap`; a kept `requires` check gives `trap` to the callers that do not prove it, not to its fn: `requires(..) without(trap)` means no trap when called inside the precondition, and such a fn is no value of a fn type that denies `trap` (E1403). Not on trait methods or trait impl methods; a `requires` on a `test fn` never panics (it selects inputs, above).
- const-expr (`const`, `static` initializers, array lengths): literals, consts, arithmetic, compare, `as`-family, `size_of::<T>()`, `align_of::<T>()`, `if then else`, array/tuple/struct/variant literals of const-exprs, and the constructors `atomic.U64.new(e)` (also `U32 Usize Bool`), `sync.Mutex::<T, R>.new(e)`, `sync.Once::<T, R>.new()`. No other calls, no loops (E0904).
- Generics are monomorphized and checked at the definition (E0901). Unbounded instantiation (`f::<[T; 1]>(..)` inside `f<T>`) is E0902. A type nests at most 1,000 levels, a struct one above its fields (E0005). Const generics: `<const N: usize>`. Generic args: types, int literals, `{const-expr}`, `[i0, i1, ..]`. Operator traits: one per mode, `Add` (`+`), `WrapAdd` (`+%`), `SatAdd` (`+|`), same for Sub/Mul/shifts, plus `Div Rem BitAnd BitOr BitXor`; all `fn add(self, rhs: Self) -> Self`, unary `Neg` (`-`), `Not` (`~`): `fn neg(self) -> Self`.

## 3. Ownership, references, views

- Single owner. Assignment and pass-by-value move unless the type is Copy (E0502 use after move). Moving a non-Copy value out of `*r`, `r.f` through a ref, `v[i]`, or a closure capture = E0504: borrow it, or `mem.replace(&mut x, v)` / `mem.swap(&mut a, &mut b)`. `match` on an owned value moves only the parts bound by value. A guard (`P if c =>`) runs after P's names are bound: in `c` they are read-only (moving one = E0504, writing or `&mut` = E0503) and the scrutinee is borrowed (`&mut` or writing it = E0501); a borrow of such a name ends with the guard (`v if p.take(&v) =>` then using `p` = E0501). Assigning into a part of a moved value (`s.f = v` after `let t = s;`) = E0502; assigning the whole value re-initializes it.
- `&T` xor `&mut T` (E0501). Checked per function, from signatures only. A temporary (`&f()`, the receiver in `f().m()`) lives until its expression is evaluated again (the next loop iteration) or the fn returns, one whose type has a drop only until its statement ends: a borrow of it used after that is E0501. An assignment ends the value it replaces, so the new value may not borrow it (`x = V.{ r: &x.f };`, `r = &wrap(r);` in a loop when the result borrows `*r` = E0501).
- References may appear **only** in locals, params, returns, and fields of `view` types. Exception: string literals are static `&str` (program data, never a loan): `const G: &str = "hi";` is allowed, and a literal satisfies any `from` clause.
- `view struct` / `view enum` may hold references. It is non-escapable: never a field of a non-view type (E0410), never an argument of a std generic without the `view` bound (`sync.Mutex`, `cell.Cell`: E0414), never passed to `forget` (E0412), never `const` or `static` (E0413). A tuple or Option that contains a ref or view is a view too, and so is **every fn value** (closure or fn item used as `fn(..)`): only a local, a param, or a view field. Writing a borrow through `&mut V` (V a view) makes the local view it points to hold that borrow from there on, like `v.f = &x` does: `*q = Option.Some(&x);` then `x` dying or being written while the view is used = E0501; a pointer that may point to several views (`q = if c then &mut a else &mut b`) makes each of them hold it. A pointer held in a by-value view param or a captured view takes only values that borrow nothing (E0513); so does assigning a captured ref or view inside a closure (`r = &z;` with `z` the closure's local = E0513). For the view behind a `&mut` param see the parameter `from` rule below.
- Return a ref or view with `from`:

| param `x` | `-> R from x` | `-> R from copy x` |
|---|---|---|
| `&T` / `&mut T` (T not view) | borrows `*x`; caller's loan lives while result lives | E0521 |
| view `V` by value | inherits x's origins | E0521 |
| `&V` / `&mut V` (V view) | reborrows the view object | inherits V's origins and may outlive the borrow (iterator `next`) |
| owned non-view by value | E0520 | E0521 |

  Multiple origins: `from a + b`. Returning an uncovered origin = E0510. A borrow through a `&mut` field of the view (`&mut *x.buf`, `&x.buf[k]`) is not a copy: `from x`. Borrowing a local = E0512.
- A `&mut` view param declares what the fn stores into the view: `fn reset(&mut self from s, s: &str) { self.src = s; }`, `fn set(dst: &mut V from a + copy b, a: &u32, b: &V)` (origins as in the table; a later param may be named; `copy b` is what `*b` holds after the call's own stores, and so is a `-> R from copy dst` result). Callee: without a clause the view takes only what it already reaches (`h.r = h.other;`); a store the clause lacks = E0513, and the fix names the clause. A borrow made through a `&` param (`&xs[a..b]`, `&other.f`) counts as that param; a `&mut` param is stored by the plain move `l.to = q;` (in a struct literal, in `Option.Some(q)`, or passed on to another storing call it is a reborrow: E0513). The declared origins may also go into a view reached from `*dst` through `&mut` fields (`self.out.take(x)`), and through any other param the clause names without `copy` (`fn f(k: &mut K from q + x, q: &mut V, x: &u32) { q.best = x; }`); otherwise an inner view never takes what the outer one holds (E0513). Only on `&mut V` params, V any type that may hold a borrow (`[V]`, `Option<&T>`, a tuple, `T: view`), of fn items and methods (E0514; not closure, `extern`, `kernel` params), never naming the param itself (E0515); a trait impl repeats the trait method's clauses, also when its `Self` holds no borrow (E0905). A fn with a clause is not a fn value (E0807): a call through a fn value stores nothing, so a closure stores nothing new into a view.
- Caller of such a fn: every view the target reaches through `&mut`, and every view a `&mut`, by-value view, or fn-value source reaches, holds all the clause's sources from the call on, while it or a pointer to it is used: after `p.take(&x);`, `x` going out of scope or being written while `p` is used = E0501. A view that holds a borrow of another part of itself keeps that part borrowed (`o.inner.take(&o.n);` then `o.n = 2;` = E0501); a borrow made through a pointer to the view (`q.cur = &q.n;`) leaves the whole view borrowed for as long as it is used, and so does a storing call for a view it reaches by a `&mut` borrow that is among what it stores (cases: contracts/lang-details.md).
- Unsafe code: a `*mut V` field of a view counts as a `&mut V` field. A `&mut V` made from a raw pointer that belongs to no view param or local the checker can see (also the result of an unsafe fn `from` a `&` anchor) takes only what that view already holds (E0513). The anchor of such a fn must come from a `&mut` borrow of the view it stands for: a store through the result reaches only the views its origins borrow mutably.
- `vec.Vec`, `Option`, `Result`, the `std.iter` adapters, `mem.replace` / `mem.swap` take `T: view`; an instance over a view is a view (`vec.Vec<&u32>` in a local or a view field; a user type needs its own `T: view`). Storing methods declare it: `fn push(&mut self from x, x: T)`, `pop(&mut self) -> Option<T> from copy self`, `swap(a: &mut T from copy b, b: &mut T from copy a)`. A borrow pushed into a container lives as long as the container; a write through `get_mut` / `as_mut_slice` stores too; dropping the container is a use, so declare the data first. A `MutexGuard` in a `vec.Vec` counts as held (E1405).
- Generic params reject views unless bound `T: view` (E0414).
- Must-consume: a pending-operation type of std (closed list: contracts/intrinsics.md) is consumed only by a `self` method; an implicit drop on any path, `?` included, is an error (`gpu.Launch`: E1104).
- Threads borrow stack data only through structured forms: `scope { spawn { .. } spawn { .. } }` (spawns join at `}` and wherever the scope is left: `return`, `break`, `continue`, `?`; a spawn in a loop runs alongside the earlier iterations' threads; `spawn` outside `scope` = E0801, in a closure inside `scope` = E0808; a target may run spawned work on one thread) and `import std.thread;`:
  - `thread.join<A: Send, B: Send>(a: fn() -> A, b: fn() -> B) -> (A, B)`: runs both concurrently.
  - `thread.par_for(lo: usize, hi: usize, body: fn(usize))`: `body(i)` for `i` in `lo..hi`, concurrently, any order.
  - `thread.par_for_mut<T: Send>(xs: &mut [T], body: fn(usize, &mut T))`: `body(i, &mut xs[i])`, concurrently.
  - `spawn` and `join` args: captured `&T` needs `T: Sync`, captured `&mut T` or copied `T` needs `Send` (E0802). A `par_for`/`par_for_mut` body runs on many threads at once: captures need `Sync`, no `&mut` captures (E0802). The argument itself must be a closure literal or fn item: not a fn-typed param, a local, or an `if`/`match` choosing one (E0806). `thread.join`, `par_for`, `par_for_mut` are never fn values (E0807).
  - Locks are ranked: `sync.Mutex<T, R>`, `sync.Once<T, R>`, R a `u32` const-expr: a literal, a named `u32` const, or `{..}` (no default; a const arg has its param's type). While a guard of rank r may be held (until moved or dropped, also by `mem.drop(guard)`; a `try_lock` guard too), every lock taken must rank above r: `m.lock()`, `o.get_or_init(f)` (which runs `f` holding R), through any call, fn value, or drop (E1405), and every lock the threads joined at a `scope` end, `thread.join`, or `par_*` may take (E1406). Two of one rank only together: `let (a, b) = sync.lock_pair(&m1, &m2);` (address order). `try_lock` takes no rank: it never waits, so it is no edge of a wait cycle (its guard is held as above; on a mutex already held, this thread's too, it returns `None`). Why: no lock-order cycle, so safe code cannot deadlock on locks and joins. Not covered: a loop that retries `try_lock` or reads an atomic until another thread acts can spin forever (livelock), and file and completion waits (`block`). No guard may be held across a `park` call.
  - Send/Sync are structural (an ADT is Send/Sync if its fields are) unless the ADT has an `unsafe("..") impl` of that trait: then only its impls count, e.g. `unsafe("..") impl<T: Sync> Sync for G<T> {}` makes `G<Cell<u32>>` not Sync. An impl of `Send`, `Sync`, or any `unsafe("..") trait` must be an `unsafe("..") impl` (E0810).
- `Flat` (prelude) = no pointer inside, so a value may be copied byte for byte to a thread, the GPU, or a file. Derived per type (per instance for generics), never implemented: any `impl Flat` is E1408. A type is Flat iff no field, enum payload, array / tuple element, or type argument, at any depth, is a reference, view, raw pointer, fn value, `UnsafeCell` (so no `Cell`, atomics, `Mutex`), or type with `Drop` (`vec.Vec`); `str` and slices are not Flat, a generic param only through `T: Flat`. Demand it with a bound `<T: Flat>`: E1407 names the first non-flat component and its path (`a.p`, `[_]`, `Some.0`). Copy does not imply Flat (`&T`, raw pointers).
- `Frozen` (prelude) = nothing reachable from a value changes while it is only shared. Derived like Flat (`impl Frozen` = E1408): no `UnsafeCell` and no fn value (captures are not in its type) through fields, payloads, elements, type arguments, or the pointee of a reference or raw pointer. `&u32`, `&[T]` of Frozen T, views, `vec.Vec<u32>` are Frozen; `Cell`, atomics, `Mutex`, `&atomic.U64` are not. Flat implies Frozen. `<T: Frozen>` (or `T: Flat`): E1410 names the first such component (`*` = pointee). Every "`&` is read-only" rule asks for it.
- Slice methods of `std.slice` (no import) are SIMD inside: `xs.sum()` (ints trap `Overflow` where a sequential loop would; floats: 32 (f32) / 16 (f64) logical lanes, §4 Float semantics), on floats `xs.dot(ys)` (same rule; lengths differ: `Assert`), `min max` (→ `Option`), `find(x)`, `eq(ys)` (floats: `==`), `fill(x)`, `copy_from(src)`, on `[u8]` `find_byte find_any_byte(set)`, on any Copy T (a view too) `count any all` (pred `fn(T) -> bool`, called on any elements, in any order, any number of times). Planned: `map_into(dst, f)` with `f: fn(f32xn) -> f32xn`, `sort`, `par_sum par_map_into par_sort`. Why: regular code, explicit SIMD stays in std, no auto-vectorization cliffs.
- Planned (no optimizer yet): release builds treat `&mut T` as noalias and `&T` as read-only when T is `Frozen` (`std.cell.UnsafeCell` is the only way to mutate through `&`; Cell/Mutex wrap it; no run-time-checked borrow type exists: share mutable structure by index, `Mutex`, or `unsafe` + UnsafeCell). A type that writes through `&T` or a raw pointer reached from it, outside `UnsafeCell` is wrongly `Frozen` (its author's bug, like a wrong `unsafe impl Sync`). Why: borrow rules become free optimizer facts.
- Every build, debug or release, drops a bounds check only with a proof, and both drop the same ones, so a program traps at the same places in both (the compiler re-checks each proof with a separate checker before it removes the check): `for i in 0..xs.len()`, chunk loops, lengths proven equal earlier on every path (`debug.assert(a.len() == b.len(), "..")`, or `if a.len() != b.len() { return ..; }`, then `b[i]` in `for i in 0..a.len()`), and constant offsets from the bound (`xs[i - 1]` in `for i in 1..xs.len()`, `xs[i + 1]` in `for i in 0..xs.len() - 1` or `while i + 1 < xs.len()`, `f32x8.load(xs, i)` and `xs[i + 7]` in `while i + 8 <= xs.len()`). Facts come from `<`/`<=`/`==`/`!=` tests on unsigned values, sub-slices (`&xs[0..n]` has length `n` and proves `n <= xs.len()`), and `+`/`-` of constants (they trap on overflow, so `xs.len() - 1` on an empty slice traps `Overflow`); `+%`, casts, and signed values prove nothing, and a slice assigned after the test is a new slice. Planned: `stave bounds-report` lists the checks kept.
- Sub-slice: `&xs[a..b]`, `&mut xs[a..=b]` (range written in the brackets, `usize` ends) on an array or slice is `&[T]` / `&mut [T]` of elements `a` up to `b` (exclusive / inclusive); `xs[a..b]` alone is a `[T]` place (`xs[1..4][0]`, `.len()`), never a value (E0504). Traps `Bounds` unless `a <= b <= len` (`a..=b`: `b < len` and `a <= b + 1`). It borrows the whole operand, like `&xs[i]`: even disjoint sub-slices of one array conflict. `Vec`, `String`, `str` take no index: `&v.as_slice()[a..b]`.
- Slice index out of range traps at runtime. Never UB. A trap (panic) prints `panic: <Kind> at <file>:<line>:<col> in <fn> (thread <n>)` (`<file>` of a std site: `std/vec.stv`) to the target's diagnostic channel (stderr on Linux) and exits with code 101 on every target. Kinds: `Bounds Overflow DivByZero Assert OutOfMemory StackOverflow Unwrap UnsupportedCpu Deadlock Contract` (`UnsupportedCpu` at start: the CPU lacks a `--target` baseline feature; `Deadlock`: a wait that can never end: `get_or_init` re-entered from its own `f` (only unsafe code gets past the rank check), `lock_pair` given one mutex twice; `Contract`: a false `requires`/`ensures` clause).
- Error trace: `Result.unwrap` of an `Err` prints the task's latest error trace under the panic line: the `Result.Err(..)` that made it, then a via per `?` that passed it on (format, restarts, tasks: contracts/lang-details.md). Out of band (no effect), the same in debug and release. `std.debug.err_trace()` copies it; `r.map_err(f)` keeps it and adds a via.
- Assertions: `std.debug.assert(cond: bool, msg: &str)` panics with `Assert` when `cond` is false (`msg` on the next stderr line).
- Safe code has no undefined behaviour (UB). Every op gives a value, a trap above, or, in `unsafe` code only, a UB of this closed list: a mask lane neither all zeros nor all ones (§4); an access to the pointee of a `&mut` that is used again or a param of a running fn, by another reference, the place or a foreign raw pointer; a write behind a `&` that is live or such a param, outside `UnsafeCell`; a raw pointer used when the reference it was cast from is unusable; dereferencing an invalid or unaligned raw pointer; reading a slot left uninitialized by `p.read()`; a data race through raw pointers or `UnsafeCell`.
- Unspecified results, a closed list: the sign and payload of a NaN result (also its quiet bit from `min_x86`/`max_x86`, §4); the order and number of calls of a `std.slice` predicate; the order of `par_for`/`par_for_mut` bodies; the interleaving of effects across threads (`spawn` and `thread.join` work, lock acquisition order, atomic ops). Targets, build modes, CPU and GPU agree modulo NaN sign and payload (no NaN encoding is pinned).

## 4. SIMD

**Status.** Stage0 checks vector code and lowers it to checked IR (fixed lane counts; the IR interpreter runs it). Vector machine code: x86-64 (`x86_64-linux`: vectors up to 128 bits, SSE4.2; `x86_64-linux-v3`: up to 256 bits, AVX2) and `aarch64-linux` (Advanced SIMD, up to 128 bits); no f16/bf16 lanes. Planned (not enforced yet): E0710-E0717, E0700/E0701 on `gpu-sm86`, sizeless vectors past type checking. E0601-E0604, E0607, E0704 are checked in part.

**Types.** `TxN`: T ∈ `i8..i64 u8..u64 f32 f64 f16 bf16`, N a power of two from 2 to 64. Total width ≤ target register: 128 bits everywhere (portable), 256 needs `avx2`, 512 needs `v4` (`avx512f` alone is not enough). Too wide = E0702. Masks `maskWxN` are typed by lane **width** and count: `i32x8`, `u32x8` and `f32x8` compares all give `mask32x8`. Generic spelling `vec<T, N>` / `mask<W, N>` only when an argument is generic.

**Strictness.**
- Only identical vector types combine (E0603). Scalar with vector = E0602: write `f32x4.splat(1.0)`.
- Int vectors: plain `+ - *` = E0604. Use `+%`, `+|`, or `a.checked_add(b) -> Option`.
- Floats: plain `+ - * /`. `+%` on float = E0206.
- Wrong mask width or count = E0605: use `m.rewiden::<64>()`.
- Int vector division: `a.div_const(k)` or `a.div_lanes(b)`. `k` is a nonzero const-expr of the lane type, as the literal of `v << 3` (`a.div_const(-3)`; run-time or 0: E0607); both truncate like `/`, and `MIN / -1` traps `Overflow`.

**Ops** (method form unless shown):
- arithmetic: `sqrt abs copysign(s) floor ceil round trunc fma(b, c) min max min_num max_num min_x86 max_x86` (floats; `round` ties to even), `div_lanes(b) div_const(k)` (ints), `mul_hi(b)` (ints: high half of the double-width product, signed iff the lanes are), `abs_wrap` (signed ints), `min max` (both)
  - `min`/`max` propagate NaN. `min_num`/`max_num` treat any NaN (quiet or signaling) as missing; two NaNs give a quiet NaN. `min_x86`/`max_x86` use x86 semantics (`a < b ? a : b`, `a > b ? a : b`); a NaN result has unspecified quiet bit and payload.
- bitwise: `& | ^ ~ andnot(b)` (`a & ~b`), shifts `<< >> <<%` by a vector or a literal (`v << 3`: a lane value)
- lane counting: `popcount clz`
- compare `a < b` → mask. Masks: `& | ^ ~`, `m.select(a, b)` (a where set), `m.to_bits()` → smallest uint, `m.to_vec::<i32>()`, `m.rewiden::<16>()`; constructors `maskWxN.first_n(k)`, `maskWxN.all()`, `maskWxN.none()`. A mask lane is all zeros or all ones: any other pattern (only `unsafe` code can read one from memory) is UB
- reductions (fixed tree): `reduce_add` (floats), `reduce_add_wrap` (ints), `reduce_min reduce_max reduce_and reduce_or`, `reduce_add_ordered(init)` (floats; sequential, init + v[0] + ..)
- `scan_add` (floats), `scan_add_wrap` (ints)
- shape: `TxN.splat(x)`, `v.bitcast::<UxM>()`, `v.split()`, `TxN.join(a, b)`, `widen_lo widen_hi`, `narrow_wrap(w) narrow_sat(w)`, `permute::<[i0, ..]>()`, `tbl.lookup(idx)` (out of range → 0), `compress(m) expand(m)`
- convert: `v as f64x4` (exact), `v as~ f32x4` (rounding), `v as| i32x4`, `v as? i32x4`
- memory: `TxN.load(s, i)`, `v.store(s, i)` (trap if `i+N > len`), `TxN.load_masked(s, i, m)`, `v.store_masked(s, i, m)`, `TxN.gather(s, idx)`, `v.scatter(s, idx)` (`idx`: unsigned lanes of any width, same lane count, each zero-extended to `usize`: `u32x4` for `f64x4`; sizeless `Txn` takes `uWxn` of its own width; else E0201; bounds-checked; duplicate index → highest lane wins). Masked forms touch active lanes only: inactive lanes read 0, are never written, and are not bounds-checked; an out-of-range active lane traps before any lane is written
- aligned: `TxN.load_aligned(s, i)`, `v.store_aligned(s, i)` need `s: &[T align A]` with A ≥ vector bytes (E0606) and trap unless `i % N == 0`. Get one with `s.align_to::<32>() -> Option<&[T align 32]>` (`Some` iff the address is a multiple of 32); `&[T align A]` coerces to `&[T]`.
- memory hints (no effect on results): `s.prefetch_{read,write}_{l1,l2,l3,nta}(i)` on a slice (any `i`, even ≥ len: never faults, no check) or raw pointer (`p.prefetch_read_l1()`), both safe. Streaming store bypassing caches: `v.store_stream(s, i)` (`s: &mut [T align A]`, A ≥ vector bytes; traps unless `i % N == 0` and `i+N ≤ len`), scalar `x.store_stream(s, i)`, unsafe `p.write_stream(v)`. The compiler fences them (contracts/simd-ops.md); unsafe `fence_stream()` fences earlier.

**Chunk loops.** A tail clause is required (E0601), except over an array whose length is a multiple of N. Syntax summary (`..` = elided; complete examples in §5):
```text
for v: f32x8 in xs { .. } tail masked (v, m) { .. }   // inactive lanes read 0; &mut binding writes active lanes only
for v: u8x16 in xs { .. } tail scalar (x) { .. }      // remaining elements one by one
for v: f32x4 in xs { .. } tail exact;                 // traps if len % N != 0
for v: &mut f32x4 in xs { *v = *v * k; } tail ..      // in-place
```
The tail runs after the full chunks and only when elements remain (`masked`: once, `scalar`: per element). `break`/`continue` inside a tail belong to the enclosing loop. A `&mut` binding is written back whenever its block is left (end, `break`, `continue`, `return`, `?`).

**Hot loops over one field**: store records as columns, `import std.soa;` → `soa.Soa::<P>.new()`, P a `Flat` struct (else E0910 / E1407). Methods as on `vec.Vec` plus `get(i) -> P` (P: Copy), `set(i, p)` (list: std/README.md). Each field of P is a column of the same name: `ps.x` is a `[F]` place of `len` values (`ps.x[i]`, `ps.x.len()`, `&mut ps.x`, `for v: f32x8 in ps.x { .. } tail ..`). Columns borrow disjointly like fields (`&mut ps.x` with `&ps.y`); `push`, `clear` conflict with a live column borrow. A private field gives a column private to P's module (E0106); a name P lacks is E0208. A struct field of P is one column. `Soa<P>` is not Flat; Send / Sync when P is.

**Features.** Signature form: `fn f(..) -> R needs(avx2, fma) { }`. Each target owns a closed set of feature names; a fn that needs another target's feature is left out of the build. A callee's needs must be ⊆ the caller's needs ∪ the target baseline (E0704). A closure has the needs of its fn. A vector in a signature (fn, fn type, closure) needs features covering its width, baseline counting (E0703). `main`, trait methods, and extern exports cannot have `needs` (E0709).
- x86-64: `sse42 ⊂ avx ⊂ avx2`, `fma bmi2 f16c`, `avx512f ⊂ avx512bw/vl/dq/cd`, extras `avx512vbmi avx512vbmi2 avx512vpopcntdq avx512bitalg avx512fp16 avx512bf16`; the names `v3` = avx2+fma+bmi2+f16c and `v4` = v3+avx512f/bw/vl/dq/cd (the `-v3` / `-v4` target baselines) stand for their features in `needs` and arms, not in `--features` or `STAVE_CPU`.
- AArch64: `neon` (baseline), `fp16 dotprod bf16`, `sve ⊂ sve2`.

**Dispatch** (runtime CPU pick, arms tried top-down, `_` required last; full example in §5):
```text
dispatch fn sum(xs: &[f32]) -> f32 { v4 => sum_v4, avx2 => sum_v3, sve => sum_sve, _ => sum_base, }
```
- An arm may name several features: `avx2 + fma => sum_fma,`. The dispatch fn may have its own `needs(..)`. An arm whose features (implied ones counted) include an earlier arm's is never picked (E0718): `avx2 => a, avx512f => b`.
- Each arm target has the dispatch's signature, with needs = dispatch needs + the arm's features (else E0705). The `_` target needs ⊆ dispatch needs ∪ the target baseline (E0706).
- The CPU's features are read once at start; each call tests the arms, but an arm the `--target` baseline (plus the dispatch's needs) covers ends the chain at compile time. At run time `STAVE_CPU=-avx2,-fma` hides features above the baseline; `STAVE_DISPATCH_TRACE=1` traces each pick to stderr (contracts/intrinsics.md).
- Arms of another arch are dropped. A `cuda` arm picks a GPU path when a usable NVIDIA device exists (planned, §7).

**Lowering gates.** Some ops are unavailable (class U) on some targets (`fma` on v2, scatter below v4; all: contracts/simd-ops.md Operation table, `stave lowering`). Using one is E0700, unless it is inside `emulate { }` (scalar loop, the only opt-in; listed in the build summary). `strict { }` also rejects multi-instruction ops and ops the table does not list (E0701). Both blocks are lexical (a closure inside counts) and check the `--target` baseline; fns with `needs(..)` are not gated, nor is `gpu-sm86` (E0702 only, at 128 bits).

**Sizeless (SVE).** `f32xn`, `mask32xn`: lane count fixed per run, legal only under `needs(sve)` (E0710). Allowed only in locals, params, and returns: not fields, array elements, globals, closure captures, or sized generic args (E0711–E0716). Loops need no tail clause. Float `reduce_add` on them = E0717: use `acc = v.reduce_add_ordered(acc);`.

**Float semantics.** IEEE, round-nearest-even, no FTZ, no auto-FMA. NaN results are quiet (sign and payload unspecified, §3). A float cast to its own type is a bit copy (a signaling NaN stays signaling). Lane op = scalar op per lane.
- Float sums follow the logical-lane rule: with L logical lanes (a power of two), element i adds into lane i % L, each lane in index order from `-0.0`, then the L lanes reduce by the `reduce_add` tree; the bits depend only on L and the input, never on the vector width or a thread split along the tree (contracts/simd-ops.md Semantics). Pad a tail with `-0.0`, never `0.0` (inactive lanes of a masked load read `0.0`: select `-0.0` into them). Std `sum` and `dot` (each product rounded, never fused) fix L = 32 (f32) / 16 (f64) on every target, dispatch arm, and the GPU, and go two-level past 4096 elements (blocks, then a block tree). An empty sum is `-0.0`; a NaN result has the same class everywhere (§3).

## 5. Examples

Every code block tagged `stave` is a complete source file: it passes `stave check`; blocks with `main` also run. SIMD blocks are check-only.

```stave
import std.io;

enum Shape { Circle(f64), Rect { w: f64, h: f64 } }

trait Area { fn area(self) -> f64; }

impl Area for Shape {
    fn area(self) -> f64 {                  // by value: consumes the shape
        return match self {
            Shape.Circle(r) => 3.14159 * r * r,
            Shape.Rect.{ w, h } => w * h,
        };
    }
}

struct Counter { pub n: u32 }

impl Counter {
    pub fn bump(&mut self) { self.n = self.n +| 1; }
}

enum ParseErr { Empty, Bad(u8) }

fn digit(c: u8) -> Result<u32, ParseErr> {
    if c < 48 || c > 57 { return Result.Err(ParseErr.Bad(c)); }
    return Result.Ok((c - 48) as u32);
}

fn parse(s: &[u8]) -> Result<u32, ParseErr> {
    if s.len() == 0 { return Result.Err(ParseErr.Empty); }
    let mut acc: u32 = 0;
    for i in 0..s.len() {
        acc = acc * 10 + digit(s[i])?;
    }
    return Result.Ok(acc);
}

fn report(s: &[u8]) -> i32 {
    match parse(s) {
        Result.Ok(v) => { io.print(f"v = {v}\n"); }
        Result.Err(_) => { return 1; }
    }
    return 0;
}

fn main() -> i32 {
    let s = Shape.Rect.{ w: 2.0, h: 3.0 };
    let a = s.area();
    io.print(f"area = {a}\n");
    let mut c = Counter.{ n: 0 };
    c.bump();
    let big: i64 = 300;
    let b: u8 = big as| u8;                 // 255
    let t: Option<u8> = big as? u8;         // Option.None
    return 0;
}
```

```stave
// views: borrowed data without lifetimes
view struct Lexer { src: &str, pos: usize }

fn lexer(src: &str) -> Lexer from src { return Lexer.{ src: src, pos: 0 }; }

fn first<T>(xs: &[T]) -> Option<&T> from xs {
    return if xs.len() == 0 then Option.None else Option.Some(&xs[0]);
}

unsafe("p is valid for reads") fn read_raw(p: *const u8) -> u8 {
    return unsafe("the caller guarantees p is valid") *p;
}
```

```stave
// globals, never, Drop
import std.atomic;
import std.io;
import std.mem;

static LIMIT: u32 = 100;
static CALLS: atomic.U64 = atomic.U64.new(0);

fn count() -> u64 {
    return CALLS.fetch_add(1) + 1;      // fetch_add returns the previous value
}

fn halt() -> never {
    loop { }
}

fn checked(x: u32) -> u32 {
    if x <= LIMIT { return x; }
    halt();
}

struct Guard { pub id: u32 }

impl Drop for Guard {
    fn drop(&mut self) { io.print(f"drop {self.id}\n"); }
}

fn guards() {
    let a = Guard.{ id: 1 };
    let b = Guard.{ id: 2 };
    mem.drop(a);                        // "drop 1" now
}                                       // "drop 2" at scope end
```

```stave
// portable 128-bit kernels: no needs, run on every target
fn sum_base(xs: &[f32]) -> f32 {            // L = 8 in two f32x4: the bits of sum_v3
    let z = f32x4.splat(-0.0);
    let mut a = z;
    let mut b = z;
    let mut i: usize = 0;
    while i + 8 <= xs.len() {
        a = a + f32x4.load(xs, i);
        b = b + f32x4.load(xs, i + 4);
        i = i + 8;
    }
    let r = xs.len() - i;
    let ma = mask32x4.first_n((r < 4).select(r, 4));
    let mb = mask32x4.first_n(r -| 4);
    a = a + ma.select(f32x4.load_masked(xs, i, ma), z);
    b = b + mb.select(f32x4.load_masked(xs, i + 4, mb), z);
    return (a + b).reduce_add();
}

fn count_eq(xs: &[u8], b: u8) -> usize {
    let mut n: usize = 0;
    let key = u8x16.splat(b);
    for v: u8x16 in xs {
        n = n + ((v == key).to_bits().popcount() as usize);
    } tail scalar (x) {
        if x == b { n = n + 1; }
    }
    return n;
}

fn add_wrap(a: &mut [i32], k: i32) {
    let kk = i32x4.splat(k);
    for v: &mut i32x4 in a {
        *v = *v +% kk;
    } tail masked (v, m) {
        *v = *v +% kk;
    }
}

// x86 256-bit variant + dispatch
fn sum_v3(xs: &[f32]) -> f32 needs(avx2) {
    let z = f32x8.splat(-0.0);
    let mut acc = z;
    for v: f32x8 in xs { acc = acc + v; } tail masked (v, m) { acc = acc + m.select(v, z); }
    return acc.reduce_add();
}

fn sum_sve(xs: &[f32]) -> f32 needs(sve) {  // sequential (L = 1): other bits than sum
    let mut acc: f32 = -0.0;
    for v: f32xn in xs { acc = v.reduce_add_ordered(acc); }
    return acc;
}

dispatch fn sum(xs: &[f32]) -> f32 { avx2 => sum_v3, _ => sum_base, }
```

```stave
// threads: closures passed as fn values
import std.thread;

fn total(xs: &[u64]) -> u64 {
    let mut s: u64 = 0;
    for i in 0..xs.len() { s = s + xs[i]; }
    return s;
}

fn sum_halves(a: &[u64], b: &[u64]) -> u64 {
    let (x, y) = thread.join(fn() -> u64 { return total(a); }, fn() -> u64 { return total(b); });
    return x + y;
}

fn double_all(xs: &mut [u64]) {
    thread.par_for_mut(xs, fn(i: usize, x: &mut u64) { *x = *x * 2; });   // each element borrowed by one call
}
```

## 6. Tooling

- Package: `stave.toml` (`name` required, `kind = "bin" | "lib"`, `version`, path-only `[dependencies]`, per-target `[target.<name>.dependencies]` merged over them, no `cfg`; contracts/cli.md) plus `src/main.stv` or `src/lib.stv`. Bad manifest, or a package or dependency named like a built-in package (`std`), = E1002.
- `stave check` (fast, no codegen), `stave build [--release] [--target x86_64-linux-v3|aarch64-linux+sve]`, `stave run`, `stave test [--filter S [--exact]] [--runner CMD] [--input N]`, `stave unsafe-report`, `stave effects` (each fn's effects), `stave fmt [--check]` (the one layout), `stave fix [--dry-run]` (applies fixes whose `applicability` is `exact`, re-checks), `stave lowering --target T [--op OP]` (the target's lowering table). Planned: `stave explain E0601`, `stave grammar --format lark|gbnf|ebnf` (for constrained decoding of the code region), `stave caps-report` (OS capabilities each package reaches).
- `--diagnostics json` emits one JSON diagnostic per line, each with a fix and its `applicability` (`exact`, `maybe`, `placeholder`). Exit codes: 0 ok, 1 diagnostics, 2 usage, 3 compiler bug.
- The default target is the baseline (v2 / neon), never auto-upgraded.
- No warnings in v1 for unused imports, variables, `mut`, or private fns. Lint: E1411 (§2 `rec`).

## 7. GPU (designed, not yet implemented)

NVIDIA GPUs through PTX (sm_86). Don't write GPU Stave code yet. Already fixed:
- `--target gpu-sm86` runs the whole program on the GPU. `kernel fn` keeps the kernel subset on every target: no heap or I/O (E1101), no recursion or fn-value calls (E1102).
- `extern "C"` is not allowed in GPU programs (E1201). Panics exit 101, as on the CPU; 125 = the GPU program cannot start.
- Recursion and fn-value calls inside `par_for` bodies are slow on the GPU (lint warning): prefer loops and `match`.

## 8. Not yet specified: don't guess, check `std/` or ask

Stdlib API beyond `io.print`, `vec.Vec` (`push`, `get_ref`), `String.as_str`, `.len()`, `.unwrap()`, `mul_wide`, `mul_hi`, `std.thread` (§3), `std.debug.assert`, `std.debug.err_trace() -> ErrTrace` (`origin() -> Option<Site>`, `len`, `skipped`, `via(i) -> Site`, `Display`; `Site`: `file line col func`), `Result.map_err`, `std.mem` (`drop forget replace swap`), `std.soa` (§4); `std.atomic` (`U32 U64 Usize Bool`: `new load store swap fetch_add compare_exchange`, sequentially consistent; `Bool` has no `fetch_add`), `std.sync` (`Mutex<T, R>`: `new lock try_lock with_lock(f)`; `lock_pair`; `Once<T, R>`: `new`, `get_or_init(f: fn() -> T) -> &T`, `f` runs once), `std.cpu` (`features() -> u64`, test with a mask: `cpu.features() & cpu.AVX2 != 0`; one `u64` mask per §4 feature name, upper case); `std.arena` (`Arena.new(bytes)`, fixed capacity, Copy values: `alloc(&self, v) -> &mut T from self`, `try_alloc`, `reset(&mut self)`); planned: `std.idx` (`IdVec`, `Id<T>`); scalar int methods such as `u16.popcount()` (used above) are assumed, not yet specified. Full trait/impl conformance rules (signature matching, missing or extra methods). How C objects and libraries are linked (`stave build` has no flag yet). Closure copy or move captures (none in v1: closures capture by reference only, no copy-capture syntax). Array→slice coercion (`&[T; N]` as `&[T]`). Binding types when matching on `&self`. Element binding type of `for x in slice` (`tail scalar (x)` binds by value). Enum↔int casts and explicit discriminants. String escapes and format specs. Moving a field out of a value whose type implements `Drop`.
