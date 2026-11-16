# Contract: portable SIMD operations and lowering classes (v0.4)

This file is the source for `stave lowering`, the declarative target tables
(`crates/stave-target/tables/*.toml`), and the SIMD conformance suite. If the
tables and this file disagree, the generated table is the build-time truth, and CI checks that the
two match.

Classes: **N** = native (≤ 2 instructions). **M** = multi-instruction sequence, with its cost
listed in the generated table. **U** = unavailable. A **U** op is E0700 unless it is inside
`emulate { }`, where it is a scalar loop over lanes. Inside `strict { }`, **M** and **U** are
E0701.

## Features

Feature sets are closed under implication, and every feature is owned by one architecture,
except `cuda` (a usable NVIDIA device), owned by x86-64, AArch64, and `gpu-sm86`:
`--features cuda` is accepted on CPU targets.

- **x86-64**: `sse42` ⊂ `avx` ⊂ `avx2`, plus `fma`, `bmi2`, `f16c` (target `v3` = avx2+fma+bmi2+f16c),
  then `avx512f` ⊂ {`avx512bw`, `avx512vl`, `avx512dq`, `avx512cd`} (target `v4` = all five).
  Named extras: `avx512vbmi`, `avx512vbmi2`, `avx512vpopcntdq`, `avx512bitalg`, `avx512fp16`,
  `avx512bf16`.
- **AArch64**: `neon` (baseline, includes fp16 arithmetic on ARMv8.2+ via `fp16`), `dotprod`, then
  `sve` ⊂ `sve2`, plus `bf16`.
- **Implications** beyond the chains above (they follow the hardware): `fma` ⇒ `avx`, `f16c` ⇒
  `avx` (VEX encoding); `avx512f` ⇒ `avx2` + `fma` + `f16c`; every AVX-512 extra ⇒ `avx512f`
  (`avx512vbmi`, `avx512vbmi2`, `avx512bitalg`, `avx512bf16` ⇒ `avx512bw`; `avx512fp16` ⇒
  `avx512bw` + `avx512vl` + `avx512dq`; `avx512vpopcntdq` ⇒ `avx512f`); every AArch64 extra
  (`fp16`, `dotprod`, `bf16`, `sve`, `sve2`) ⇒ `neon`; `sve` ⇒ `fp16`.
- **Target tiers below**: `v2` = sse42 baseline, `v3`, `v4`, `neon` = baseline AArch64, `sve`.

## Types

- `TxN`, where T ∈ {i8..i64, u8..u64, f32, f64, f16, bf16} and N ∈ {2,4,…,64}. `isize`/`usize`
  are not lane types.
  - A type exists on a target iff N·bits(T) ≤ the register width (v2/neon 128, v3 256, v4 512).
    Otherwise it is E0702: there are no silent multi-register types. 128-bit types exist on every
    tier, so they are the portable baseline.
  - f16/bf16 lanes support load/store/convert everywhere (`f16c` on x86 for conversions). For
    arithmetic they are **U** unless `avx512fp16`/`fp16` (f16) or `avx512bf16`/`bf16` (bf16 dot
    products only).
- `maskWxN`, where W is the lane width in bits and N the lane count, e.g. `mask32x8`. Its layout is
  unspecified and it is not addressable.
  - Leaving registers: `m.to_bits() -> u8|u16|u32|u64` (the smallest that holds N bits, high bits
    zero) and `m.to_vec::<iW>()`.
  - `m.rewiden::<W2>()` is N on v4/sve and M elsewhere. Constructors: `maskWxN.first_n(k)`,
    `maskWxN.all()`, `maskWxN.none()`.
- Generic spelling: `vec<T, N>` / `mask<W, N>`, legal only with generic arguments (E0609).
- `Txn` / `maskWxn` (sizeless) need `needs(sve)` in v1 (E0710). Restrictions are
  E0711–E0716 (see diagnostic-codes.md).
- **ABI rule:** a vector type in a signature requires `needs` covering its width
  (E0703). Every function and `dispatch` arm of one signature has one ABI. `extern "C"`
  signatures may not contain vectors (E0804).

## Semantics

- A lane op is the scalar op applied to each lane. Floats are IEEE 754 binary16/32/64, round to
  nearest even, with no FTZ/DAZ (the runtime never changes MXCSR/FPCR) and no automatic FMA
  contraction.
- An arithmetic op that yields NaN yields a quiet NaN with unspecified sign and payload. Scalars
  follow the same rule. `neg`, `abs`, `copysign`, loads, stores, `select`, and `bitcast` are
  bit-exact. Differential tests compare NaN-ness, not NaN bits.
- `select`: one method spelling, `m.select(a, b)` on a mask (lanes of `a` where `m` is set,
  else `b`; planned) and `c.select(a, b)` on a `bool` (any type; implemented). Both operands are
  evaluated first. It is always branchless: v2 `pblendvb`/`blendvps` (class N), v3 `vblendv*`, v4
  mask-register moves, NEON `bsl`, SVE `sel`, PTX `selp`; the bool form is `cmov`/`csel` per
  8-byte word (aggregates word by word; f32/f64 through GPRs on x86-64, `fcsel` on AArch64). No
  pass turns it into a branch.
- Reductions use a fixed tree: halve by lane index (lane i with i+N/2) until one lane remains.
  `reduce_add` and `scan_add` are float-only; the integer forms are `reduce_add_wrap` and
  `scan_add_wrap` (wrapping, as `+%`). `reduce_add_ordered` is float-only: a wrapping integer sum
  does not depend on the order (`init +% v.reduce_add_wrap()`).
  `v.reduce_add_ordered(init)` is strictly sequential (`init + v[0] + v[1] + …`). Threading the
  accumulator through a loop makes a whole-slice sum independent of chunk boundaries, and so of VL. On sizeless vectors, float
  `reduce_add`/`reduce_mul` are E0717.
- Integer vector arithmetic needs a mode (`+%`, `+|`, `checked_add`). Plain `+ - *` is E0604.
- Shifts: per-lane amounts ≥ bits trap (`<<`, `>>`) or are masked (`<<%`), as for scalars.
  A constant out-of-range amount is E0205.
- Memory and bounds:
  - `TxN.load(s, i)` / `v.store(s, i)` trap if `i + N > s.len()`. Checks are elided only when
    proved (chunk loops prove them).
  - The index vector of `TxN.gather(s, idx)` / `v.scatter(s, idx)` is any unsigned-int vector with
    N lanes (`f64x4.gather(s, idx: u32x4)`, `u8x16.gather(s, idx: u16x16)`); each lane is
    zero-extended to `usize`. A sizeless `Txn` takes `uWxn` of its own lane width W. Another index
    type is E0201.
  - Gather traps if any active index is ≥ len. Scatter traps on an out-of-range active index. On
    duplicate indices the highest lane wins (defined, not UB).
  - Aligned forms require `s: &[T align A]` with A ≥ the vector size in bytes (E0606), obtained
    through `s.align_to::<A>() -> Option<&[T align A]>`.

## Chunk loops and tails

| Construct | v2 | v3 | v4 | neon | sve |
|---|---|---|---|---|---|
| `for v: TxN in s` body (full chunks) | N | N | N | N | N |
| `tail masked (v, m)`, 32/64-bit lanes | M (a) | N | N | M (a) | N |
| `tail masked (v, m)`, 8/16-bit lanes | M (a) | M (a) | N | M (a) | N |
| `tail masked` with a `&mut` binding: write-back of active lanes only | M (b) | N 32/64, M (b) 8/16 | N | M (b) | N |
| `tail scalar (x)` | N | N | N | N | N |
| `tail exact;` | N (one compare plus a trap branch) | N | N | N | N |

- (a) The remaining `len % N` elements are copied into a zeroed stack buffer of N lanes, and a full
  vector is loaded from it. This never touches memory past the slice.
- (b) The result vector is stored to the stack buffer, and only the active elements are copied
  back. Inactive lanes are never written to the slice. A `&mut` chunk binding in a masked tail
  therefore cannot cause an out-of-bounds store.
- `for v: Txn in s` (sizeless) uses `whilelt` predicates and needs no separate tail (the predicate
  covers it). A `tail` clause on a sizeless loop is optional and, when present, receives the last
  partial predicate.

## Operation table

| Op | Syntax | v2 | v3 | v4 | neon | sve |
|---|---|---|---|---|---|---|
| float add/sub/mul/div | `a + b` … | N | N | N | N | N |
| float sqrt | `v.sqrt()` | N | N | N | N | N |
| float neg/abs/copysign | `-v`, `v.abs()`, `v.copysign(s)` | N | N | N | N | N |
| float floor/ceil/round/trunc | `v.floor()` … | N | N | N | N | N |
| int add/sub wrapping | `a +% b` | N | N | N | N | N |
| int neg wrapping / abs | `0 -% v`, `v.abs_wrap()` (signed) | N | N | N | N | N |
| int mul wrapping i16/i32 | `a *% b` | N | N | N | N | N |
| int mul wrapping i8 | `a *% b` | M | M | M | N | N |
| int mul wrapping i64 | `a *% b` | M | M | N | M | N |
| sat add/sub i8/i16 | `a +| b` | N | N | N | N | N |
| sat add/sub i32/i64 | `a +| b` | M | M | M | N | N |
| checked add/sub/mul | `a.checked_add(b) -> Option<TxN>` | M | M | M | M | M |
| bitwise and/or/xor/not, andnot | `a & b`, `a | b`, `a ^ b`, `~a`, `a.andnot(b)` | N | N | N | N | N |
| shift by const amount | `v << 3`, `v <<% 3` | N (16/32/64), M (8) | N (16/32/64), M (8) | N (16/32/64), M (8) | N | N |
| shift by vector amount (16/32/64) | `v << s`, `v <<% s` | M | N 32/64, M 16 | N | N | N |
| shift by vector amount (8) | same | M | M | M | N | N |
| int div by const splat | `a.div_const::<K>()` | M | M | M | M | M |
| int div by vector | `a.div_lanes(b)` | U | U | U | U | N (i32/i64), M (i8/i16) |
| fma | `a.fma(b, c)` | U | N | N | N | N |
| min/max (IEEE minimum, NaN propagates) | `a.min(b)`, `a.max(b)` | M | M | M | N | N |
| min_num/max_num (NaN = missing) | `a.min_num(b)` | M | M | M | N | N |
| min/max x86 semantics (named, order-dependent) | `a.min_x86(b)` | N | N | N | M | M |
| int min/max | `a.min(b)` | N (8–32), M (64) | N (8–32), M (64) | N | N (8–32), M (64) | N |
| compare → mask, signed and float | `a < b` | N | N | N | N | N |
| compare → mask, unsigned | `a < b` | M | M | N | N | N |
| mask and/or/xor/not | `m & n` … | N | N | N | N | N |
| mask rewiden | `m.rewiden::<W2>()` | M | M | N | M | N |
| mask to_bits | `m.to_bits()` | N (movmsk) | N | N | M (≈3) | M |
| select | `m.select(a, b)` | N | N | N | N | N |
| load/store unaligned | `TxN.load(s, i)` / `v.store(s, i)` | N | N | N | N | N |
| load/store aligned | `TxN.load_aligned(s, i)` | N | N | N | N | N |
| masked load/store 32/64-bit | `TxN.load_masked(s, i, m)`, `v.store_masked(s, i, m)` | M (a/b) | N | N | M (a/b) | N |
| masked load/store 8/16-bit | same | M (a/b) | M (a/b) | N | M (a/b) | N |
| first-fault load | `TxN.load_first_fault(s, i)` | U | U | U | U | N |
| gather 32/64 | `TxN.gather(s, idx)` | U | N | N | U | N |
| scatter 32/64 | `v.scatter(s, idx)` | U | U | N | U | N |
| compress 32/64 | `v.compress(m)` | M | M | N | M | N |
| compress 8/16 | same | M | M | N (+avx512vbmi2), M | M | M |
| expand | `v.expand(m)` | M | M | N | M | M |
| reduce add/min/max/and/or (fixed tree) | `v.reduce_add()`, `v.reduce_add_wrap()` (int) | M | M | M | N int, M fp | N int, E0717 fp |
| reduce add ordered (float) | `v.reduce_add_ordered(init)` | M | M | M | M | N (fadda) |
| prefix scan add | `v.scan_add()`, `v.scan_add_wrap()` (int) | M | M | M | M | M |
| permute (const indices) | `v.permute::<[i0, …]>()` | N (≤16 B), M | M (cross-lane) | N | N | N |
| table lookup bytes (out-of-range → 0) | `tbl.lookup(idx)` | N (16 B) | M | N (+avx512vbmi), M | N | N |
| popcount | `v.popcount()` | M | M | N 32/64 (+avx512vpopcntdq), N 8/16 (+avx512bitalg), else M | N u8, M wider | N |
| leading zeros | `v.clz()` | M | M | N 32/64, M 8/16 | N ≤32, M 64 | N |
| widen | `v.widen_lo()`, `v.widen_hi()` | N | N | N | N | N |
| narrow truncating | `v.narrow_wrap(w)` | M | M | N | N | M |
| narrow saturating | `v.narrow_sat(w)` | N | M | N | N | M |
| int → float exact (e.g. i32 → f64) | `v as f64x4` | N | N | N | N | N |
| int → float rounding (i32 → f32, i64 → f64) | `v as~ f32x4` | N (i32), M (u32/i64) | N (i32), M | N | N | N |
| float → int saturating | `v as| i32x4` | M (fixup) | M | M | N | N |
| float → int checked | `v as? i32x4` (Option) | M | M | M | M | M |
| f16 ↔ f32 convert | `v as f32x8` / `v as~ f16x8` | U | N (+f16c) | N | N | N |
| splat / bitcast / split / join | `TxN.splat(x)`, `v.bitcast::<UxM>()`, `v.split()`, `TxN.join(a, b)` | N | N | N | N | N |
| tail mask | `maskWxN.first_n(k)` | M | M | N | M | N (whilelt) |

## Memory hints

Prefetch and streaming (non-temporal) stores. Both are hints: the program's results never depend
on them. Class N everywhere (one or two instructions, or none), so `strict { }` accepts them and
no `emulate` is needed. "plain" = an ordinary store, hint dropped (same semantics).

- **Prefetch** `s.prefetch_<rw>_<loc>(i)` (`s: &[T]` / `&mut [T]`) and `p.prefetch_<rw>_<loc>()`
  (`p: *const T` / `*mut T`), `<rw>` ∈ `read write`, `<loc>` ∈ `l1 l2 l3 nta`. Safe in both forms:
  no access, never faults or traps, no bounds check (`i ≥ len` allowed, wrapping address
  arithmetic). Kept in program order by the optimizer. On `gpu-sm86` the slice form is guarded by
  `i < len` (one predicate; PTX gives no fault guarantee for a prefetch of an unmapped address,
  unverified), the raw-pointer form emits nothing.
- **Streaming store** `v.store_stream(s, i)` (`v: TxN`, `s: &mut [T align A]`, A ≥ vector bytes,
  else E0606; traps `Bounds` if `i + N > len` or `i % N != 0`), `x.store_stream(s, i)` (scalar lane
  type, `s: &mut [T]`, traps if `i ≥ len`), and unsafe `p.write_stream(v)` (`p: *mut T`, aligned to
  `size_of::<T>()`; no drop of the old pointee).
- **Ordering**: a streaming store is a plain store for the program. The compiler places
  `fence_stream()` after the last streaming store on each path to a return or a call (checked IR,
  before inlining; one per loop exit). `fence_stream()` is also a built-in fn for unsafe code.

| Op | v2 | v3 | v4 | neon | sve | gpu-sm86 |
|---|---|---|---|---|---|---|
| `prefetch_read_l1` | `prefetcht0` | `prefetcht0` | `prefetcht0` | `prfm pldl1keep` | `prfm pldl1keep` | `prefetch.L1` |
| `prefetch_read_l2` | `prefetcht1` | `prefetcht1` | `prefetcht1` | `prfm pldl2keep` | `prfm pldl2keep` | `prefetch.L2` |
| `prefetch_read_l3` | `prefetcht2` | `prefetcht2` | `prefetcht2` | `prfm pldl3keep` | `prfm pldl3keep` | `prefetch.L2` (no L3) |
| `prefetch_read_nta` | `prefetchnta` | `prefetchnta` | `prefetchnta` | `prfm pldl1strm` | `prfm pldl1strm` | none |
| `prefetch_write_l1` | `prefetcht0` (a) | `prefetcht0` (a) | `prefetchw` | `prfm pstl1keep` | `prfm pstl1keep` | `prefetch.L1` |
| `prefetch_write_l2` | `prefetcht1` (a) | `prefetcht1` (a) | `prefetchw` | `prfm pstl2keep` | `prfm pstl2keep` | `prefetch.L2` |
| `prefetch_write_l3` | `prefetcht2` (a) | `prefetcht2` (a) | `prefetchw` | `prfm pstl3keep` | `prfm pstl3keep` | `prefetch.L2` |
| `prefetch_write_nta` | `prefetchnta` (a) | `prefetchnta` (a) | `prefetchw` | `prfm pstl1strm` | `prfm pstl1strm` | none |
| `v.store_stream(s, i)` float lanes | `movntps/pd` | `vmovntps/pd` | `vmovntps/pd` (EVEX) | `stnp` of the two d halves (2) | `stnt1w/d` (all-true pred) | `st.global.cs` (`.v2/.v4` per 128 bits) |
| `v.store_stream(s, i)` int lanes | `movntdq` | `vmovntdq` | `vmovntdq` (EVEX) | `stnp` of the two d halves (2) | `stnt1b/h/w/d` | `st.global.cs` |
| `x.store_stream(s, i)` 32/64-bit int | `movnti` | `movnti` | `movnti` | plain `str` (b) | plain `str` (b) | `st.global.cs` |
| `x.store_stream(s, i)` f32/f64 | `movd/movq` + `movnti` (2) | same | same | plain `str` (b) | plain `str` (b) | `st.global.cs` |
| `x.store_stream(s, i)` 8/16-bit | plain `mov` (c) | plain | plain | plain `strb/strh` | plain | `st.global.cs` |
| `fence_stream()` | `sfence` | `sfence` | `sfence` | none (d) | none (d) | none (e) |

- (a) `prefetchw` needs PRFCHW (Intel Broadwell+, every AVX-512 CPU), which v2/v3 do not imply
  (Haswell lacks it), so v2/v3 use the read hint of the same level.
- (b) AArch64 has no single-register non-temporal store (`stnp` needs a pair).
- (c) x86 has no 8/16-bit non-temporal store.
- (d) `stnp`/`stnt1` follow normal store ordering; the non-temporal relaxation concerns `ldnp`
  address dependencies only (unverified: recheck against the Arm ARM before this lowering lands).
- (e) `.cs` is a cache operator on a weak store; PTX ordering is unchanged. The `.global` space
  is used when the slice is known global; otherwise the generic `st.cs` / `prefetch.L1` forms.

Lowering notes the backend MUST respect:
- **compress on v4:** compress to a register, then a masked store. Never use the memory form,
  which is microcoded on Zen 4.
- **min/max on x86:** `min`/`min_num` use the fixed sequence. `minps` semantics are only
  available as `min_x86`, whose NaN result has an unspecified quiet bit and payload, so x86 emits
  the bare instruction and other targets need no payload-exact sequence.
- **fixed reduce tree on neon:** pairwise `faddp` needs extra shuffles to match the tree, so the
  op is class M.
- **x86 VEX/EVEX transitions:** VZEROUPPER is inserted before calls and returns out of 256/512-bit
  code.
