# Contract: diagnostic code registry (v0.8)

This is the human-readable table of codes for `crates/stave-diag`, `stave explain`, and
every `//~ ERROR` / `//~ WARNING` expectation in the conformance suite
([conformance-annotations.md](conformance-annotations.md)). It holds E00xx–E10xx, E13xx, and E14xx; the
E11xx rows (GPU kernels) live in [gpu-diagnostics.md](gpu-diagnostics.md) and the E12xx rows
(the `gpu-sm86` target) in [gpu-program-diagnostics.md](gpu-program-diagnostics.md). The registry is the union of
these three tables. Codes are stable: a code is never reused and never changes meaning. A retired
code stays listed as "retired".

Each code's message, related labels, and fix text are **templates** kept in the machine-readable
registry [diagnostics.toml](diagnostics.toml) (format in its header). `tools/diaggen` reads that
file and generates the stage0 (Rust) and stage1 (Stave) tables. diaggen fails unless every code of
the three tables, retired ones included, has exactly one entry there, and vice versa. Holes
take only canonical data (names, canonically printed types, token kinds), so both compilers print
the same bytes. Location-less (NOLOC) codes, with no primary span, are exactly E1001,
E1301, E1302, E1303. CI fails if a generated table is stale. E0004's expected set is printed in the
terminal order of the generated parse table (`tools/ll1gen`).

| Range | Area |
|---|---|
| E00xx | lexing and parsing |
| E01xx | modules, imports, name resolution |
| E02xx | types, literals, conversions, operators |
| E03xx | patterns and match |
| E04xx | kinds: views, Copy, sized |
| E05xx | ownership, borrowing, origins |
| E06xx | SIMD value typing and chunk loops |
| E07xx | target features, lowering classes, dispatch, sizeless vectors |
| E08xx | threads, unsafe, FFI, allocator |
| E09xx | generics, traits, const evaluation |
| E10xx | build and driver (packages, targets, cache) |
| E11xx | GPU kernels ([gpu-diagnostics.md](gpu-diagnostics.md)) |
| E12xx | GPU programs, target `gpu-sm86` ([gpu-program-diagnostics.md](gpu-program-diagnostics.md)) |
| E13xx | GPU compiler, stage1 ([cli-stage1.md](cli-stage1.md)); listed below |
| E14xx | effects, recursion, stack, ranks, flat; listed below |

| Code | Rule |
|---|---|
| E0001 | source is not valid UTF-8 |
| E0002 | source file exceeds 4 GiB (u32 offsets); files up to that size are accepted |
| E0003 | unterminated string, char, or f-string literal: it reaches end of file before its closing quote, with no newline in between (a raw newline first is E0015). At the literal's start |
| E0004 | unexpected token (the message lists the expected set). Only the first per item; the item is dropped |
| E0005 | bracket nesting depth over 10,000 (counting `( [ { .{`), at the first opener reaching depth 10,001. One per file; no other parse diagnostics for the file; E0016 takes precedence. Case `expr`: expressions, blocks, and patterns nested deeper than 20,000 levels (nesting without brackets, e.g. a run of unary operators), once per fn body, at the first node over the limit |
| E0006 | f-string interpolation is not a name or field path |
| E0011 | `<` after a cast type (rule D2). Write `(x as T) < y` |
| E0012 | chained comparison `a < b < c` |
| E0013 | `unsafe` without an obligation string, or with an empty one (`unsafe("")`), at every unsafe site (block, expr, fn, trait, impl) |
| E0014 | struct literal or struct pattern written `Name { … }`. Write `Name.{ … }` |
| E0015 | raw newline inside a string, f-string, or char literal, before its closing quote. At the literal's start; the literal ends at the newline (end of file first is E0003). Write `\n`; the literal must end on its line |
| E0016 | unbalanced brackets: unmatched closer, or closer of another kind (at the first such closer), else the innermost opener unclosed at end of file. One per file; no other parse diagnostics for the file |
| E0017 | byte-order mark at file start. Remove it (UTF-8 without BOM) |
| E0018 | unexpected character: a byte that starts no token (`@ $ \ #`, backtick, a control byte other than tab/LF/CR, DEL) or any non-ASCII byte outside string, char, and f-string literals and comments (identifiers are ASCII). One per run of consecutive such bytes, at its start; the item is dropped without E0004 |
| E0019 | `if let` / `while let` (Rust): write a `match` (statement or value form). Parse rule 5 (parse-errors.md) |
| E0020 | `let P = e else { .. };` (Rust `let … else`): write a `match`. Parse rule 6 |
| E0021 | expression statement without `;` directly before `}` (Rust tail expression, no assignment): blocks are not values; write `return e;`, or end it with `;`. Parse rule 7 |
| E0022 | `where` clause (Rust): bounds go inline, `<T: A + B>`. Parse rule 8 |
| E0023 | `::` as a path separator (`a::b`, `Vec::new()`, `Option::Some`): write `.`; `::` only before generic args. Exact fix. Parse rule 9 |
| E0024 | lifetime `'a`: a `'` followed by an identifier with no closing `'` is one lexical error token (like E0018, no E0004); Stave has no lifetimes. Case `ref`: directly after `&` (`&'a T`), where removing it is an exact fix |
| E0025 | `mut x: T` parameter (Rust): params are immutable; write `x: T` and `let mut y = x;`. Parse rule 10 |
| E0026 | tuple struct `struct S(T);` or (case `unit`) unit struct `struct S;`: structs have named fields, `struct S { v: T }`, `struct S {}`. Exact fix for case `unit`. Parse rule 11 |
| E0027 | `!` as a type (Rust never type): write `never` (only after `->`). Exact fix after `->` (case `ret`). Parse rule 12 |
| E0028 | `dyn Trait` / `impl Trait` in a type position: use a generic `<T: Trait>` or a fn type. Parse rule 13 |
| E0029 | block comment: the token `/*` is a lexical error through the first `*/` on its line or the end of the line (newlines stay lexer restart points; `a/*p` must be written `a / *p`); write `//` |
| E0101 | unresolved name |
| E0102 | import cycle. One per strongly connected component; the cycle is listed starting at its lexicographically smallest module path, following import order |
| E0103 | unqualified enum variant in a pattern or expression |
| E0104 | more than one `allocator` declaration in a binary |
| E0105 | duplicate declaration of `<name>` in `<scope>` (module items and import names, fields, variants, methods of a type, params, generic params, names bound by one pattern: `(a, a)`). Reported at each later declaration; related span: the first declaration |
| E0106 | `<name>` is private to module `<module>`: an item, field, or method used outside its module without `pub` (trait impl methods have the trait's visibility). Related span: the declaration |
| E0107 | import path is ambiguous: its first segment is both a dependency name of the package and a module (`src/x.stv`) or module directory (`src/x/`) of the package. At the first segment; the import binds an error decl. Fix: `import <package>.x` for the module (the package's own name first), or rename the module or the dependency |
| E0108 | a `bin` package has no `fn main` in its root module (`stave build`, `stave run`, `stave test`; `stave check` does not need it). At offset 0 of the root module; reported only when there is no other error (a dropped or invalid item may be `main`) |
| E0201 | mismatched types, with no implicit conversion |
| E0202 | literal has no type context (no default type). Add a suffix or an annotation |
| E0203 | literal out of range for its type |
| E0204 | lossy `as`. Use `as%`, `as|`, `as?` or `as~` |
| E0205 | constant shift amount ≥ bit width |
| E0206 | operator not defined for type (e.g. `+%` on a float) |
| E0207 | non-unit function can reach its end without `return` (a `()` function falling off its end is an implicit `return;`). Case `never`: a `-> never` function with a reachable `return` or a reachable end of body |
| E0208 | no method or field `<name>` on type `<T>` (e.g. `mul_wide` on `u16`: it exists on `i32`, `i64`, `u32` and `u64` only) |
| E0209 | `never` used outside a return type: it is legal only after `->` of a fn signature or a fn type |
| E0210 | expression statement whose type is not `()` or `never` (discarded value, e.g. an ignored `Result`). Discard explicitly with `let _ = e;` (drops at once). Exact fix |
| E0211 | contract operand not allowed: `requires`/`ensures` take a `bool` expression over params, fields, `.len()`, literals, arithmetic, comparison, `&&` `||` `!`, and calls of safe fns whose args are `&` or Copy; no `unsafe`, `&mut`, `static`/`static mut`. Case `trait-impl`: a contract on a trait impl method. Planned |
| E0212 | parameter named `result` in a fn with `ensures` (`result` is a contextual name inside `ensures`, not a keyword). Planned |
| E0300 | non-exhaustive match. Case `limit`: the exhaustiveness check exceeds its search limit (patterns nested more than 2,000 levels deep, or over 1,000,000 search steps); the match is rejected |
| E0301 | unreachable match arm. Case `alternative`: an alternative of an or-pattern (`A | B =>`) that never matches: covered by earlier arms or by earlier alternatives of its arm (reported at each such alternative) |
| E0302 | or-pattern alternatives (`A | B =>`) bind different names, or a name with a different type or binding mode (by value or by reference, `mut` or not). Reported at each alternative that differs from the first, once per name. Related span: the first alternative |
| E0410 | reference- or view-typed field in a non-view ADT |
| E0411 | view in a heap or container type (Box, Vec, Map, ...). To be retired: `vec.Vec`, `Box`, `Option`, `Result` take `T: view` and become conditional views; a container without the bound is E0414 case `container` |
| E0412 | view passed to `forget` |
| E0413 | view or reference in a `const`, `static`, or `static mut` (except `&str` initialized by a string literal or another such const: static strings) |
| E0414 | generic parameter instantiated with a view without the `T: view` bound. Case `container` (planned): a heap type that does not accept views, e.g. `Rc`/`Arc` |
| E0415 | `impl Copy for T {}` where a field or variant payload of T is not Copy (Copy: scalars, `&T`, raw pointers, arrays and tuples of Copy types, ADTs with `impl Copy`; `&mut T` never). Related span: the field |
| E0416 | `impl Drop` for a Copy type (built-in Copy, or an ADT with `impl Copy`; related span: the `impl Copy`) |
| E0417 | `static X: T` where T is not `Sync` (a `static` is shared by every thread): use an atomic (`atomic.U64`), `sync.Mutex<T>`, or `sync.Once<T>` |
| E0501 | borrow conflict (borrow site, conflicting access, later use). When an access conflicts with several live loans, the loan with the smallest creation offset is reported (ties: `&mut` before `&`, then place path); the later use is the first use after the access that keeps that loan live |
| E0502 | use after move |
| E0503 | assignment to an immutable binding or through `&`; assignment to, `&mut` of, or a `&mut self` call on a `static` or a part of one (case `static`, reported by the type checker) |
| E0504 | move out of a borrow or an index: a non-Copy value taken by value from `*r`, a field reached through a reference (`r.f`), an element `v[i]`, or a closure capture (captures are references) |
| E0510 | returned reference or view not covered by the `from` clause |
| E0511 | retired: a missing `from` clause declares the empty origin set; a returned borrow of a parameter not named is E0510 |
| E0512 | returned value borrows a local |
| E0513 | write through `&mut View` with origins not contained in the target's. Planned: for a `&mut` rooted at a parameter, the origins must be in that parameter's `from` set plus its own origins; fix "add `from x` to parameter `p`" |
| E0514 | `from` on a parameter that is not `&mut V` (V a view) of a fn item or method (closure, `extern`, and `kernel` params take none). Planned |
| E0515 | a parameter's `from` clause names the parameter itself. Planned |
| E0520 | `from x` names a by-value non-view parameter |
| E0521 | `from copy x` where x is not a reference to a view |
| E0601 | chunk loop without a tail policy |
| E0602 | scalar and vector mixed (use `TxN.splat`) |
| E0603 | vector types differ in lane type or count (use widen/narrow/split/join/bitcast) |
| E0604 | plain `+ - *` on an integer vector (choose `+%`, `+|`, or `checked_add`) |
| E0605 | mask width or count does not match the vector (use `m.rewiden::<W>()`) |
| E0606 | aligned load or store on a slice whose type lacks that alignment (use `align_to`) |
| E0607 | vector integer division by a non-constant: `a / b`, or `a.div_const(k)` with `k` not a const-expr (use `div_lanes`). Case `zero`: `a.div_const(k)` with `k` = 0 |
| E0608 | permute or table index out of range at compile time |
| E0609 | generic vector spelling `vec<T, N>` / `mask<W, N>` with no generic argument. Write `f32x8` / `mask32x8` |
| E0700 | operation unavailable (class U) on the target, outside `emulate { }` |
| E0701 | class M or U operation inside `strict { }` |
| E0702 | vector type wider than the target's register |
| E0703 | vector in the signature of a fn, fn type, or closure whose `needs` (plus the target baseline) do not cover its width (`extern "C"` signatures: E0804) |
| E0704 | callee needs features neither the caller nor the target baseline has |
| E0705 | dispatch arm signature differs from the dispatch signature |
| E0706 | dispatch without a final `_` arm (the grammar requires it: a syntax error comes first), or a `_` target needing extra features |
| E0707 | fn value coercion would drop features not proven where it happens (by the `needs` of the enclosing fn or the target baseline) |
| E0708 | unknown feature name, or a feature not owned by any arch (`cuda` is owned by x86-64, AArch64, and `gpu-sm86`) |
| E0709 | `main`, a trait method, or an `extern` export declares `needs` |
| E0710 | sizeless vector used without `needs(sve)`. Planned |
| E0711 | sizeless vector as an ADT field. Planned |
| E0712 | sizeless vector as an array element. Planned |
| E0713 | sizeless vector as a global or const. Planned |
| E0714 | `size_of` or pointer arithmetic on a sizeless type. Planned |
| E0715 | sizeless vector captured by a closure. Planned |
| E0716 | sizeless vector as a generic argument that needs a size. Planned |
| E0717 | VL-dependent float reduction on a sizeless vector (use `reduce_add_ordered`). Planned |
| E0718 | dispatch arm never picked: its features (implied ones counted, with the dispatch's) include those of an earlier arm |
| E0801 | `spawn` outside a `scope` block |
| E0802 | captured value is not `Send` (by value or `&mut`) or not `Sync` (`&`); for a `par_for`/`par_for_mut` body (shared by many threads): a capture that is not `Sync`, or a `&mut` capture |
| E0803 | unsafe operation outside `unsafe(..)`: raw-pointer `*p`, `p.read()`, `p.write(v)`, `p.write_stream(v)`, `fence_stream()`; a call of an `unsafe` fn or `extern` import; any use (read, write, borrow) of a `static mut` |
| E0804 | vector or view type in an `extern "C"` signature |
| E0805 | `extern "stave-intrinsic"` outside the shipped `std`, `runtime`, and `runtime-gpu` packages, or a name or signature not in `contracts/intrinsics.md` |
| E0806 | argument of `thread.join`, `thread.par_for`, or `thread.par_for_mut` is a fn value whose captures are unknown (not a closure literal or fn item) |
| E0807 | an `unsafe` fn coerced to a fn value (fn types carry no `unsafe` marker). Case `thread`: `thread.join`, `thread.par_for`, or `thread.par_for_mut` used as a fn value (a call through it would skip E0802/E0806) |
| E0808 | `spawn` inside a closure body: a closure is not part of the `scope` block around it, so its `spawn` has no scope to join |
| E0809 | `static mut` outside the shipped `std`, `runtime`, and `runtime-gpu` packages (user code: `static` with a `Sync` type) |
| E0810 | impl of an unsafe trait (`Send`, `Sync`, or any trait declared `unsafe("..") trait`) without `unsafe("..")`. At the `impl` |
| E0901 | method or operator not provided by the generic's bounds (reported at the definition) |
| E0902 | instantiation depth limit (64) exceeded: an instance of a generic fn or method, reached from a non-generic fn through calls, would be the 65th instance of that generic on its own instantiation chain (polymorphic recursion such as `f::<[T; 1]>(..)` inside `f<T>`). At the call that requests it, once per generic |
| E0903 | overlapping impls |
| E0904 | const-expr uses a call (other than a constructor fixed by path, intrinsics.md), a loop, a non-const item, or a const that depends on itself (the cycle is listed, starting at its smallest decl id) |
| E0905 | impl method signature differs from the trait's (params, return, `from`, `needs`, or `unsafe`). The fix shows the trait's signature |
| E0906 | impl lacks a trait method, or defines a method the trait does not declare |
| E0907 | `pub` on a method of a trait impl (it has the trait's visibility). At the `pub` |
| E0908 | ambiguous method: lookup (inherent impls, then every trait impl of the receiver type) finds it in two traits. Fix: the qualified call `Trait.m(x)` |
| E0909 | explicit call of `Drop.drop` (`x.drop()` or `Drop.drop(&mut x)`): drop runs only automatically (scope end, overwrite of a place). Fix: let the value go out of scope, or move it into `std.mem.drop(x)` |
| E1001 | target not runnable on this host (binary built, tests reported as not run, never as passed). Reported by `stave test` and `stave run`. No location |
| E1002 | invalid `stave.toml`, located in it. Cases: unknown key or table; `wrong-type`; `non-path-dependency`; `unknown-target` (`[target.<t>]` naming no known target); `syntax` (TOML syntax error, at the error, rest of the manifest ignored); `duplicate-key` (at the second definition); `bad-kind` (`kind` not `"bin"`/`"lib"`, at the value); `missing-name` (at offset 0); `bad-name` (`name` empty, `.`, `..`, containing `/`, `\`, or NUL, or starting with `-`: it names the default executable; at the value); `missing-root` (no `src/main.stv` / `src/lib.stv` for the kind, at the `kind` value or offset 0); `missing-dependency` (a path dependency directory without `stave.toml`, at the `path` value); `unreadable` (the manifest exists but cannot be read, at offset 0) |
| E1003 | a source file of a package cannot be read (removed after discovery, permissions, I/O error). At offset 0 of that file; the file is treated as empty |
| E1301 | project needs more device memory than the daemon may use (estimate after lexing: needed vs the daemon's `heap_max`, bounded by free device memory). Shows needed and available bytes. No location. Fix: use `--compiler cpu`, raise `[compiler] gpu_mem`, or free GPU memory (`stave daemon stop` frees the daemon's own) |
| E1302 | `--compiler gpu` but no usable GPU (reason as in the summary line). No location. Fix: use `--compiler auto` or `--compiler cpu`; or make the GPU usable: install the NVIDIA driver, check the device, or reinstall Stave with the GPU compiler image (compute capability below 8.6: stage1 needs 8.6 or higher) |
| E1303 | `--compiler gpu` with a code-generating command (`build`, `run`, `test`) for a target stage1 does not support (AArch64 in v1); `stave check` is not affected. Reported instead of E1302 when both apply (probe step 1, cli.md). No location. Fix: use `--compiler auto` or `--compiler cpu`: stage0 builds this target |
| E1304 | `--compiler gpu` and the package links C objects (through mold/lld), which stage1 v1 does not support. Located at the package's first C-link input in stave.toml (or the first `extern "C"` import block if none is declared). `stave check` is not affected. Under `auto` such packages go to stage0 (reported in the summary line). Fix: use `--compiler auto` or `--compiler cpu`: stage0 links C objects |
| E1407 | a `Flat` bound (`<T: Flat>`) given a type that is not `Flat`. Names the first non-flat component (reference, view, raw pointer, fn value, `UnsafeCell`, type with `Drop`, generic param without a `Flat` bound, unsized type) and its path: field names, `Variant.field` for an enum payload, `0` for a tuple element, `[_]` for an array element, `<T>` for a type argument. Case `self`: the type itself is that component. Case `depth`: the check passed its depth cutoff (1024 levels) and fails closed |
| E1408 | an `impl Flat`, safe or `unsafe(..)`: `Flat` is derived by the compiler from the type's structure and never implemented. At the `impl`; no other impl check runs on it |
