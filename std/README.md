# std (minimal)

The Stave standard library, written in Stave. A built-in package (no
manifest, supplied by the driver; stave-sema `package.rs`): `std/io.stv` is `std.io`; it depends on
the built-in `runtime`, and every package depends on it implicitly. The prelude is the
`pub` items of `std.core` plus the built-in type names (ints, floats, `bool`, `char`, `str`,
`never`); `Vec`, `String`, and the rest need an import (`import std.vec;` → `vec.Vec`).

| Module | Contents |
|---|---|
| `std.core` | `Option<T: view>`, `Result<T: view, E: view>`, `Copy`, `unsafe` `Send`/`Sync`, `Drop`, operator traits `fn op(self, rhs: Self) -> Self` (`Add WrapAdd SatAdd Sub WrapSub SatSub Mul WrapMul SatMul Div Rem Shl WrapShl Shr BitAnd BitOr BitXor`; unary `Neg Not`), `Write`, `Display` (+ `impl Display for str`); all `pub` items are the prelude |
| `std.mem` | raw-memory helpers: `type_size type_align` (over `size_of`/`align_of`), `drop forget replace swap` (`forget` runs no destructor; views rejected, E0412), raw pointers `dangling add add_mut`, slice/str parts `slice_ptr str_ptr str_bytes` and unsafe `slice slice_mut str_from`, float bits `f64_bits f64_from_bits f32_bits f32_from_bits`; **every representation assumption std makes is here** (A3, A4) |
| `std.vec` | `Vec<T>`: `new with_capacity len is_empty capacity reserve push pop get get_ref get_mut as_slice as_mut_slice clear spare_ptr`, unsafe `set_len`, `extend_from_slice` (T: Copy), `Drop`; Send / Sync when T is; allocation failure (capacity overflow included) panics `OutOfMemory`, except in `try_with_capacity try_reserve try_extend_from_slice`, which return `Result.Err(AllocError)`, and `try_push(x) -> Result<(), T>`, which returns `Result.Err(x)`; all leave the Vec unchanged; `AllocError` (Copy, `Display`) |
| `std.soa` | struct-of-arrays `Soa2<A, B>`, `Soa3`, `Soa4`, one `Vec` per column (type-checks and builds; `tests/conformance/run/soa_cols.stv` runs it): `new with_capacity len is_empty reserve push pop clear set get_ref get` (`get` when every column is Copy), per-column slices `col0..col3`, `col0_mut..`, `cols`, `cols_mut`; Send / Sync through the columns |
| `std.string` | `String` (UTF-8 over `Vec<u8>`; Send + Sync): `new with_capacity from_str from_utf8 len is_empty clear push_str push as_str as_bytes into_bytes`, `try_with_capacity try_reserve try_push_str try_push` (`Result<_, vec.AllocError>`); `Write`, `Display`; `valid_utf8` |
| `std.io` | `print`, `eprint`, `write_all`, `FdWriter`/`stdout()`/`stderr()`, `read_all`, `read_file`, `read_to_string`, `Error` (a path with a NUL byte is `Error.InvalidInput`) |
| `std.debug` | `assert(cond: bool, msg: &str)` |
| `std.cell` | `UnsafeCell<T>` (fixed path: the only interior mutability) `new get get_mut into_inner`; `Cell<T: Copy>` `new get set replace`. None is Sync |
| `std.sync` | `Mutex<T>` on a futex (`new lock try_lock with_lock get_mut into_inner`, Send + Sync for T: Send), guard view `MutexGuard<T>` (`get get_mut`, unlocks in its destructor; never Send, Sync when T is); `Once<T>` (`new`, `get_or_init(f) -> &T`: the first caller runs `f`, callers arriving meanwhile sleep on a futex; Sync for T: Send + Sync). `Mutex.new` and `Once.new` are const-exprs |
| `std.atomic` | `U32 U64 Usize Bool` over the runtime's CPU atomics, sequentially consistent: `new load store swap fetch_add compare_exchange` (`Bool` without `fetch_add`); Sync, usable as `static`; `new` is a const-expr |
| `std.thread` | `join<A: Send, B: Send>(a: fn() -> A, b: fn() -> B) -> (A, B)`, `par_for(lo, hi, body: fn(usize))`, `par_for_mut<T: Send>(xs: &mut [T], body: fn(usize, &mut T))`, `workers()`: one worker per CPU of the affinity mask, chunks claimed from an atomic counter; threads created per call and joined before return |
| `std.cpu` | `features() -> u64`: the CPU features of the running machine as a feature word (CPUID + XCR0 on x86-64, AT_HWCAP/AT_HWCAP2 on AArch64), closed under implication; one mask constant per feature (`SSE42 AVX AVX2 FMA BMI2 F16C AVX512F … NEON FP16 DOTPROD BF16 SVE SVE2`), bit order of contracts/intrinsics.md "CPU feature detection" |
| `std.env` | `count() -> usize`, `arg(i) -> Option<String>` (argv copied; None past the end or for non-UTF-8) |
| `std.iter` | `SliceIter<T>` (`iter(xs)`, `next(&mut self) -> Option<&T> from copy self`, `len`, `map(f)`), `Map<T, U>`, `zip(xs, ys)` → `Zip<A, B>` (items `(&A, &B)`); over slices only (no iterator trait) |
| `std.slice` | SIMD methods on the built-in slice types (`impl [T]` blocks, so no import): `sum min max find eq fill` on every int and float element type (`find eq fill` also on `bool`), `find_byte find_any_byte` on `[u8]`, `count any all copy_from` on any Copy T. 128-bit vectors, and on x86-64 256-bit loops (`needs(avx2)`, behind a `dispatch fn` per method and element type) for slices of 1 KiB or more (512 B or more for 4- and 8-byte elements, except `eq` and `copy_from` of ints and `bool`) on an AVX2 CPU, with the same results (float `sum` stays 128-bit); float `min`/`max` propagate NaN and return it as the canonical quiet NaN (`0x7fc00000` / `0x7ff8000000000000`) whatever NaNs the slice holds, with -0.0 < +0.0; int `sum` traps `Overflow` at the element the sequential loop would; float `sum` is a fixed 16-lane (f32) / 8-lane (f64) tree; `eq` is element-wise `==` (floats: NaN never equal, -0.0 == +0.0); the `count any all` predicate may run on any elements, in any order, any number of times. Reinterprets same-size element slices through `mem.slice` (A3, A4; a `bool` is one byte, 0 or 1). The file head has the full rules |
| `std.fmt` | `Display` for `u8..u64 usize i8..i64 isize bool char f32 f64`; `write_u64`, `write_i64`, `write_char`, `f64_text`, `f32_text` → `Text` (`len as_str write_to`, at most `TEXT_CAP` bytes); `Dec` |
| `std.num` | `parse_u64 parse_i64 parse_u32 parse_i32 parse_usize parse_f64 parse_f32` (all `-> Option<_>`) |
| `std.pow10` | generated 128-bit power-of-ten table shared by fmt and num |

## f-string lowering (contract for the front end)

`Display` is `fn fmt<W: Write>(&self, w: &mut W)`, `Write` is `fn write_str(&mut self, s: &str)`.
An f-string `f"a{x}b"` evaluates to an **owned** `std.string.String`:

```
let mut t = String.new(); t.push_str("a"); Display.fmt(&x, &mut t); t.push_str("b");  // value: t
```

Where `&str` is expected, an f-string argument is borrowed automatically and `&String` converts
to `&str` (`String` derefs to `str`: the compiler calls `String.as_str`); the temporary lives to
the end of the enclosing statement. So `io.print(f"n = {n}\n")`, `io.print("hi\n")`, and
`debug.assert(ok, f"bad {n}")` all type-check against `&str` parameters. To keep the text, bind
it: `let s = f"..";` (a `String`), then pass `&s`.

## Floats

- **f64/f32 → text** (`std.fmt`): Schubfach (Giulietti), integer only on `u64.mul_wide`;
  shortest round-trip digits, nearest to the value among the shortest; then the Stave text form
  (decimal for first-digit exponent −5..16, else `d.ddde-x`; `inf -inf NaN -0.0`). f32 uses the
  same code with its own significand and exponent.
- **text → f64/f32** (`std.num`): Eisel–Lemire (fast_float `compute_float`) on the first 19
  significant digits; if digits were dropped and w, w + 1 round differently, an exact big-integer
  comparison (80 × u64 limbs, first 800 digits plus a sticky bit) against the halfway point.
  Designed to round correctly, ties to even. Accepts `-?d+(.d+)?([eE][+-]?d+)?`, `inf`, `-inf`, `NaN`.
- **Tests in the tree**: `tests/conformance/run/json_lite` (`parse_f64` and `Display` on a
  handful of values: `0.1`, `1e21`, `-0.0`, `2.5e-7`, `6.02e23`),
  `tests/conformance/run/num_float_long.stv` (2·10^5-digit inputs with six-digit exponents, a
  halfway case decided by a digit past the 800th, a tie to even, inf and 0 from huge exponents,
  f32), `tests/conformance/std/num_float_exp7.stv` (10^6 digits, seven-digit exponents), and the LANG.md examples (`tests/conformance/langmd`).
  These are spot checks: correct rounding on every input rests on the algorithms above, not on
  an exhaustive test in the tree.

## Language rules std relies on

- `size_of::<T>()` / `align_of::<T>()` are built-in generic fns, used through
  `mem.type_size` / `mem.type_align`.
- Raw pointers: `*p = v` drops the old pointee (a place assignment); `p.write(v)`
  stores without dropping (Vec fills uninitialized slots with it); `p.read()` moves the pointee
  out bitwise (Vec `pop`, `clear`). Both are unsafe.
- `Drop` runs at scope end and on overwrite, never explicitly (E0909); `mem.drop(x)`
  ends a value's life early.
- `-> never`: `runtime.panic.trap`/`trap_msg`/`abort` and `runtime.os.exit_group`
  return `never`, so a call to them ends a path for E0207 (no `loop { }` after a trap).

## Assumptions (not yet specified)

- **A2** `&*p` / `&mut *p` on a raw pointer gives a reference whose origin is the enclosing fn's
  `from` clause.
- **A3** `&[T]`, `&mut [T]`, `&str` are (data pointer, length) word pairs; pointer casts between
  `*const &str`, `*const &[T]`, `*const [usize; 2]` reinterpret them.
- **A4** a value read through a raw pointer carries no loan, so it satisfies any `from` clause.
- std modules use Option/Result unqualified (prelude) and name traits as `core.Display` etc.
  Generic params of Option/Result are `view`-bounded.
- `impl Display for str` (an impl on the unsized `str`) is allowed; `x.fmt(w)` on `&str` finds it.
- A panic raised by std for its caller uses `runtime.panic.user_site()` (runtime/README.md).
