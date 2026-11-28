# Contract: portable SIMD operations and lowering classes (v0.5)

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
- **Groups** (x86-64): `v3` = `avx2` + `fma` + `bmi2` + `f16c`; `v4` = `v3` + `avx512f` +
  `avx512bw` + `avx512vl` + `avx512dq` + `avx512cd`. A group name is accepted wherever source
  code names a feature (`needs(..)`, fn types, `dispatch` arms, `a + b` arms) and stands for the
  closure of its members: feature sets, types, diagnostics, and dumps only hold features (printed
  as the features not implied by another, in the order above). The `x86_64-linux-v3` / `-v4`
  baselines are these groups. `--features` and `STAVE_CPU` take feature names only.
- **Target tiers below**: `v2` = sse42 baseline, `v3`, `v4`, `neon` = baseline AArch64, `sve`.

### Per-function feature sets (x86-64 code generation)

- A function is compiled for F = the closure of the target baseline ∪ its `needs`. Each feature
  of F enables its own instructions and nothing else:
  - `avx`: the VEX encoding of the function's vector and scalar float instructions, and 256-bit
    float instructions. (Where the baseline has `avx`, a debug build keeps the legacy encoding of
    scalar float instructions: it holds no 256-bit value in a register between two IR
    instructions.)
  - `avx2`: 256-bit integer instructions, and the AVX2-only instructions at 128 bits (broadcast
    from a register, per-lane shift counts, gathers, integer masked moves).
  - `fma`, `bmi2`, `f16c`: their instructions; without them the baseline sequence (software
    `fma`).
- A function never contains an instruction outside F. An op with no lowering under F (a 256-bit
  integer op with `avx` alone) is a code generation error, never another instruction. The tier
  columns of the generated table describe functions whose F is a tier baseline.
- The calling convention follows the types of the signature only, never F: a caller and a callee
  with different feature sets agree. A 256-bit vector travels in a `ymm` register, so a signature
  with one needs `avx` on both sides (E0703).
- `ymm` upper halves are clean at every call and every return that carries no 256-bit value in a
  register. A function with `avx` in F that may have dirtied them executes `vzeroupper` before
  such a return and before such a call, direct or through a fn value; so code without `avx` (a
  baseline callee or caller, the trap path) never runs with dirty upper halves.
- Compiler-emitted stubs, routines, and tables use target-baseline instructions only.

### K mode (x86-64 functions with the AVX-512 set)

- A function is in **K mode** iff F ⊇ {`avx512f`, `avx512bw`, `avx512vl`, `avx512dq`,
  `avx512cd`} (the v4 set). Any other function is compiled as above, and a 512-bit vector or
  an EVEX instruction in it is a code generation error that names the features F lacks.
- In K mode every mask value is one bit per lane in a mask register `k1`–`k7` (`k0` is never a
  write mask; all are caller-saved; a spilled mask is its bits in 8 bytes). Bit j is lane j;
  the bits from the lane count up are unspecified, and what reads a mask as an integer
  (`to_bits`, a bounds check) narrows first. Every vector op is the v4 recipe of the generated
  table, EVEX-encoded at the length of the value (`xmm`, `ymm`, `zmm`; VEX on `xmm` where the
  recipe names it). Vectors use `zmm0`–`zmm13`, numbered as the `xmm` / `ymm` registers of the
  other functions; `zmm16`–`zmm31` are not used.
- The mask ABI follows the type, so K-mode and other functions share signatures. A mask whose
  lanes fill at most 256 bits crosses a call as the vector of its lanes (all ones or all
  zeros, in the vector argument sequence) for every function: a K-mode function converts at
  its params, call arguments and returns (`vpmov*2m` in, `vpmovm2*` out). A mask of a 512-bit
  type crosses as its bits, an integer argument (zero above the lane count), returned in
  `rax`. A 512-bit vector travels in a `zmm` register as a 256-bit one in a `ymm`. A mask in
  memory (behind a pointer, in an aggregate) is the vector of its lanes on every target.
- `vzeroupper`: the rule above with `zmm` state counted as dirty upper parts: the bits above
  127 of `zmm0`–`zmm15` are clean at every call and return that carries no 256- or 512-bit
  value in a register.
- Backed by code on v4: every op (f16 / bf16 lanes aside). Without VBMI / VBMI2 / BITALG /
  VPOPCNTDQ, byte and word permutes, lookups, compress / expand and bit counts take sequences of
  word, doubleword and byte-shuffle instructions. Lane by lane on the home bytes, as on v3:
  float scans, 64-bit `mul.sat`, `checked_mul` and `div_const`, gathers and scatters of 8- and
  16-bit lanes, and lookups whose index lanes differ in width from the table's.
- Test leg `v4`: binaries with AVX-512 code run natively on a CPU with the v4 set, else under
  Intel SDE when the environment variable `STAVE_SDE` names its `sde64` (`$STAVE_SDE -icx --
  program`), else the leg is skipped (a failure when `STAVE_REQUIRE_LEGS` lists `v4`).

## Types

- `TxN`, where T ∈ {i8..i64, u8..u64, f32, f64, f16, bf16} and N ∈ {2,4,…,64}. `isize`/`usize`
  are not lane types.
  - A type exists on a target iff N·bits(T) ≤ the register width (v2/neon 128, v3 256, v4 512).
    Otherwise it is E0702: there are no silent multi-register types. 128-bit types exist on every
    tier, so they are the portable baseline.
  - Features covering a width: 256 bits `avx2`; 512 bits the AVX-512 part of `v4`: `avx512f`,
    `avx512bw`, `avx512vl`, `avx512dq`, `avx512cd` (`avx512f` alone does not).
  - f16/bf16 lanes support load/store/convert everywhere (`f16c` on x86 for conversions). For
    arithmetic they are **U** unless `avx512fp16`/`fp16` (f16) or `avx512bf16`/`bf16` (bf16 dot
    products only).
- `maskWxN`, where W is the lane width in bits and N the lane count, e.g. `mask32x8`. Its layout is
  unspecified and it is not addressable (in a register: a vector of all-ones lanes, or one bit
  per lane in a `k` register in K mode).
  - Leaving registers: `m.to_bits() -> u8|u16|u32|u64` (the smallest that holds N bits, high bits
    zero) and `m.to_vec::<iW>()`.
  - `m.rewiden::<W2>()` is N on v4/sve and M elsewhere. Constructors: `maskWxN.first_n(k)`,
    `maskWxN.all()`, `maskWxN.none()`.
- Generic spelling: `vec<T, N>` / `mask<W, N>` is E0609 (write `f32x8` / `mask32x8`); a generic lane
  type T is planned (case `generic-lane` today).
- `Txn` / `maskWxn` (sizeless) need a scalable-vector feature of the target, in v1 `needs(sve)`
  (E0710). The lane count is fixed per run for the fn's feature set; two feature sets of one
  process may have different counts. Restrictions are E0711–E0716 (see diagnostic-codes.md).
- **ABI rule:** a vector type in a signature (fn, fn type, closure) requires `needs` covering
  its width, the target baseline counting (E0703). Every function and `dispatch` arm of one signature has one ABI. `extern "C"`
  signatures may not contain vectors (E0804).

## Semantics

- A lane op is the scalar op applied to each lane. Floats are IEEE 754 binary16/32/64, round to
  nearest even, with no FTZ/DAZ (the runtime never changes MXCSR/FPCR) and no automatic FMA
  contraction.
- An arithmetic op that yields NaN yields a quiet NaN with unspecified sign and payload. Scalars
  follow the same rule. `neg`, `abs`, `copysign`, loads, stores, `select`, and `bitcast` are
  bit-exact. Differential tests compare NaN-ness, not NaN bits.
- A float cast to its own type, scalar or lane-wise (`as`, `as~`, `as?`), is a bit copy: a
  signaling NaN keeps its signaling bit and payload.
- `min_num`/`max_num`: a NaN operand, quiet or signaling, is missing, and the result is the other
  operand bit for bit; two NaNs give a quiet NaN.
- A mask lane is all zeros (clear) or all ones (set). Any other bit pattern, which only `unsafe`
  code can make (reading a mask from memory), is undefined behaviour: a backend may test any bit
  of the lane, and the IR interpreter stops with a fault when it loads one.
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
  accumulator through a loop makes a whole-slice sum independent of chunk boundaries, and so of
  VL. On sizeless vectors, float `reduce_add` is E0717.
- **Logical-lane rule** (whole-slice float sums; normative for std and every backend). A sum of
  terms t_0 .. t_(n-1) over L logical lanes (L a power of two) is:
  1. lane l = `((-0.0 + t_l) + t_(l+L)) + …`, every i < n with i % L == l, in increasing i;
  2. while more than one lane is left: lane j = lane j + lane (j + L/2) for j < L/2, L halves
     (the `reduce_add` tree on L lanes).

  The bits depend only on L and the terms. Composition: K vectors of N lanes (KN = L) hold
  logical lane kN + j in lane j of vector k; step 2 on them is vector k + vector (k + K/2),
  halving K, then `reduce_add` of the last vector, so 128-, 256- and 512-bit loops, a scalar
  loop with L accumulators, and a GPU warp (L = 32: lane l of the warp is logical lane l, and
  the warp reduce of gpu-ops.md is the 32-lane tree; L < 32: lanes l ≥ L hold `-0.0`) give the
  same bits. `-0.0` is the additive identity (`x + -0.0` is x bit for bit for every non-NaN x,
  `+0.0` included), so a lane with no term stays `-0.0`, an empty sum is `-0.0`, and a tail
  pads its missing terms with `-0.0`: never `+0.0`, which turns a `-0.0` lane into `+0.0`
  (inactive lanes of a masked load read `+0.0`: select `-0.0` into them, or add only active
  lanes). No backend may reassociate or contract; it may skip only additions of `-0.0`. A NaN
  result is NaN on every target (which inputs give one is fixed
  by L); its sign and payload are unspecified.
  std: `xs.sum()` on `[f32]` / `[f64]` has terms t_i = x_i, and `xs.dot(ys)` has t_i = the
  rounded product x_i * y_i (never fused); both use L = 32 for f32 and L = 16 for f64 (1024 bits)
  on every target, dispatch arm (128-bit `_` and NEON, `avx2`, `v4`) and the GPU, in two levels
  (below). A program that sums one slice with loops of different L gets different bits; one L
  per sum is the portable form.
- **Two-level rule** (std `sum` and `dot`). With block size B (4096 elements, f32 and f64):
  1. block k holds terms kB .. min((k + 1)B, n) - 1; there are m = ceil(n / B) blocks, the last
     possibly short. S_k = the block's sum by the logical-lane rule above (its own lanes from
     `-0.0`, its own tail, its own tree).
  2. The block tree combines S_0 .. S_(m-1): level 0 is S; level j + 1 entry i = level j entry
     2i + level j entry 2i + 1, or level j entry 2i alone when 2i + 1 is past the end; the
     result is the single entry of the first level that has one. m = 0: `-0.0`; m = 1: S_0, so
     a sum of at most B terms is the one-level sum, bit for bit. Equivalently: tree(S_0 ..
     S_(m-1)) = tree(first p) + tree(rest), p the largest power of two below m.
  3. Streaming form (one pass, at most 64 entries of state): a stack of node sums; after block
     c (counting from 0) is summed, pop and add (popped + new) once per trailing 1 bit of c,
     then push; at the end add the entries from the top down (lower + sum so far).

  The bits depend only on n and the terms. Legal splits: a tree node is the aligned run of
  blocks i·2^j .. (i + 1)·2^j - 1 (clipped to m); any thread, warp or pass may compute any node
  alone (its sum is the two-level rule on that run of terms), and any cover of the blocks by
  nodes, combined by the levels above them, gives the same bits. Splitting the slice at multiples
  of 2^j·B elements (one j for all pieces) and combining the pieces' sums by the block tree is
  legal; any other cut (inside a block, or between nodes the tree does not have, e.g. 8 blocks
  as 0..2 and 3..7) is not.
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
  - `TxN.load_masked(s, i, m)` / `v.store_masked(s, i, m)` touch active lanes only: inactive lanes
    read 0, are never written, and are not checked. If an active lane is out of range
    (`i + k ≥ s.len()`), the op traps `Bounds`; a masked store checks every active lane before it
    writes any.
  - Gather traps if any active index is ≥ len. Scatter traps on an out-of-range active index. On
    duplicate indices the highest lane wins (defined, not UB).
  - Aligned forms require `s: &[T align A]` with A ≥ the vector size in bytes (E0606), obtained
    through `s.align_to::<A>() -> Option<&[T align A]>`.

## Chunk loops and tails

| Construct | v2 | v3 | v4 | neon | sve |
|---|---|---|---|---|---|
| `for v: TxN in s` body (full chunks) | N | N | N | N | N |
| `tail masked (v, m)`, 32/64-bit lanes | M (a) | M | M | M (a) | N |
| `tail masked (v, m)`, 8/16-bit lanes | M (a) | M (a) | M | M (a) | N |
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
| float neg/abs | `-v`, `v.abs()` | N | N | N | N | N |
| float copysign | `v.copysign(s)` | M | M | N | N | M |
| float floor/ceil/round/trunc | `v.floor()` … | N | N | N | N | N |
| int add/sub wrapping | `a +% b` | N | N | N | N | N |
| int neg wrapping / abs | `0 -% v`, `v.abs_wrap()` (signed) | N (8–32), M (64) | N (8–32), M (64) | N | N | N |
| int mul wrapping i16/i32 | `a *% b` | N | N | N | N | N |
| int mul wrapping i8 | `a *% b` | M | M | M | N | N |
| int mul wrapping i64 | `a *% b` | M | M | N | M | N |
| int mul saturating | `a *| b` | M | M | M | M | M |
| int mul high half | `a.mul_hi(b)` | N (16), M | N (16), M | N (16), M | M | N |
| sat add/sub i8/i16 | `a +| b` | N | N | N | N | N |
| sat add i32/i64 | `a +| b` | M | M | M | N | N |
| sat sub i32/i64 | `a -| b` | M | M | N u32/u64, M | N | N |
| checked add/sub/mul | `a.checked_add(b) -> Option<TxN>` | M | M | M | M | M |
| bitwise and/or/xor/not, andnot | `a & b`, `a | b`, `a ^ b`, `~a`, `a.andnot(b)` | N | N | N | N | N |
| shift left by const amount | `v << 3`, `v <<% 3` | N (16/32/64), M (8) | N (16/32/64), M (8) | N | N | N |
| shift right by const amount | `v >> 3` | M i8/u8/i64, N | M i8/u8/i64, N | M i8, N | N | N |
| shift left by vector amount (16/32/64) | `v << s`, `v <<% s` | M | N 32/64, M 16 | N | N | N |
| shift right by vector amount (16/32/64) | `v >> s` | M | N i32/u32/u64, M | N | N | N |
| shift by vector amount (8) | `v << s`, `v >> s`, `v <<% s` | M | M | M | N | N |
| int div by const splat | `a.div_const(k)` | M | M | M | M | M |
| int div by vector | `a.div_lanes(b)` | U | U | U | U | N (i32/i64), M (i8/i16) |
| fma | `a.fma(b, c)` | U | N | N | N | N |
| min/max (IEEE minimum, NaN propagates) | `a.min(b)`, `a.max(b)` | M | M | M | N | N |
| min_num/max_num (NaN = missing) | `a.min_num(b)` | M | M | M | M | N |
| min/max x86 semantics (named, order-dependent) | `a.min_x86(b)` | N | N | N | M | N |
| int min/max | `a.min(b)` | N (8–32), M (64) | N (8–32), M (64) | N | N | N |
| compare → mask, signed and float | `a < b` | N | N | N | N | N |
| compare → mask, unsigned | `a < b` | M | M | N | N | N |
| mask and/or/xor/not | `m & n` … | N | N | N | N | N |
| mask constants | `maskWxN.all()`, `maskWxN.none()` | N | N | N | N | N |
| mask to vector | `m.to_vec::<iW>()` | N | N | N | N | N |
| mask rewiden | `m.rewiden::<W2>()` | N | M | N | M | M |
| mask to_bits | `m.to_bits()` | M 16, N (movmsk) | N | N | M | N 8, M |
| select | `m.select(a, b)` | N | N | N | N | N |
| load/store unaligned | `TxN.load(s, i)` / `v.store(s, i)` | N | N | N | N | N |
| load/store aligned | `TxN.load_aligned(s, i)` | N | N | N | N | N |
| masked load/store 32/64-bit | `TxN.load_masked(s, i, m)`, `v.store_masked(s, i, m)` | M | N | N | M | N |
| masked load/store 8/16-bit | same | M | M | N | M | N |
| first-fault load | `TxN.load_first_fault(s, i)` (reserved: E0208 in stage0) | U | U | U | U | M |
| gather 32/64 | `TxN.gather(s, idx)` | U | M 32, N | N | U | N |
| scatter 32/64 | `v.scatter(s, idx)` | U | U | N | U | N |
| compress 32/64 | `v.compress(m)` | M | M | N | M | N |
| compress 8/16 | same | M | M | N (+avx512vbmi2), M | M | M |
| expand | `v.expand(m)` | M | M | M 8/16, N | M | M |
| float reduce add (fixed tree) | `v.reduce_add()` | N f64, M | M | M | N f64, M | N f64, M |
| int reduce add (wrapping) | `v.reduce_add_wrap()` | M | M | M | N | N |
| reduce and/or | `v.reduce_and()`, `v.reduce_or()` | M | M | M | M | N |
| reduce min/max (fixed tree) | `v.reduce_min()`, `v.reduce_max()` | M | M | M | M i64/u64, N | N |
| reduce add ordered (float) | `v.reduce_add_ordered(init)` | N f64, M | M | M | M | N (fadda) |
| prefix scan add | `v.scan_add()`, `v.scan_add_wrap()` (int) | N i64/u64, M | M | M | N i64/u64, M | N i64/u64, M |
| permute (const indices) | `v.permute::<[i0, …]>()` | N | M 8/16, N | M 8, N | N | N |
| table lookup bytes (out-of-range → 0) | `tbl.lookup(idx)` | N (16 B) | M | N (+avx512vbmi), M | N | N |
| popcount | `v.popcount()` | M | M | N 32/64 (+avx512vpopcntdq), N 8/16 (+avx512bitalg), else M | N 8/16, M | N |
| leading zeros | `v.clz()` | M | M | N 32/64, M 8/16 | N ≤32, M 64 | N |
| widen | `v.widen_lo()`, `v.widen_hi()` | N | N | N | N | N |
| narrow truncating | `v.narrow_wrap(w)` | N 64, M | N 64, M | M 16, N | M | N |
| narrow saturating | `v.narrow_sat(w)` | N i16/i32, M | N i16/i32, M | N i16/i32, M | M | M |
| int widen one step | `v as i32x4`, `v as% i32x4` (from 16-bit lanes) | N | N | N | N | N |
| int narrow one step, wrapping | `v as% i16x4` (from 32-bit lanes) | N | M | N | N | N |
| int → float exact (e.g. i32 → f64) | `v as f64x4` | M u32, N | M u32, N | N | M 8, N | M 8, N |
| int → float rounding (i32 → f32, i64 → f64) | `v as~ f32x4` | N (i32), M (u32/i64) | N (i32), M | N | N | N |
| f32 ↔ f64, float to its own type | `v as f64x2`, `v as~ f32x4` | N | N | N | N | N |
| float → int saturating | `v as| i32x4` | M (fixup) | M | N u32/u64, M | N | N |
| float → int checked | `v as? i32x4` (Option) | M | M | M | M | M |
| f16 ↔ f32 convert | `v as f32x8` / `v as~ f16x8` | U | N (+f16c) | N | N | N |
| splat | `TxN.splat(x)` | M 16, N | N | N | N | N |
| bitcast / split / join | `v.bitcast::<UxM>()`, `v.split()`, `TxN.join(a, b)` | N | N | N | N | N |
| tail mask | `maskWxN.first_n(k)` | M | M | M | M | N (whilelt) |

A plain `as` to the identical vector type is a bit copy of every lane (no op, no table entry, no
gate). The other lane casts (`as|` and `as?` between ints, int casts across two or more widths, `as?`
between floats) have a class per (source, target) pair: their table entry is the contract
(`stave lowering --op vec.cast.sat`). Class N is a recipe of at most 2 instructions, M one of
more; `tools/gen-tables` rejects a table cell whose class disagrees with its recipe.

## Memory hints

Prefetch and streaming (non-temporal) stores. Both are hints: the program's results never depend
on them, and a target may compile one to nothing (a prefetch) or to the plain store. Class N
everywhere (one or two instructions, or none), so `strict { }` accepts them and no `emulate` is
needed. "plain" = an ordinary store, hint dropped (same semantics).

- **Prefetch** `s.prefetch_<rw>_<loc>(i)` (`s: &[T]` / `&mut [T]`) and `p.prefetch_<rw>_<loc>()`
  (`p: *const T` / `*mut T`), `<rw>` ∈ `read write`, `<loc>` ∈ `l1 l2 l3 nta`. Safe in both forms:
  no access, never faults or traps, no bounds check (`i ≥ len` allowed, wrapping address
  arithmetic). Kept in program order by the optimizer. On `gpu-sm86` the slice form is guarded by
  `i < len` (one predicate; PTX gives no fault guarantee for a prefetch of an unmapped address,
  unverified), the raw-pointer form emits nothing.
- **Streaming store** `v.store_stream(s, i)` (`v: TxN`, `s: &mut [T align A]`, A ≥ vector bytes,
  else E0606; traps `Bounds` if `i + N > len`, then if `i % N != 0`), `x.store_stream(s, i)` (`x`
  of a lane type T, `s: &mut [T]`, traps `Bounds` if `i ≥ len`), and unsafe `p.write_stream(v)`
  (`p: *mut T`, T a lane type or a vector; obligation: `p` valid for writes and aligned to
  `size_of::<T>()`; no drop of the old pointee).
- **Ordering**: a streaming store is a plain store for the program. The compiler places
  `fence_stream()` after the last streaming store on each path to a return or a call (`call`,
  `call.ind`, any intrinsic), where no further streaming store can come before it: one at each exit
  of a streaming loop, one before a call inside it, never one per store. It works on the checked
  IR (before inlining), and a fence already on the path (an explicit `fence_stream()`) is not
  repeated. A path that ends in a trap gets none. `fence_stream()` is also a built-in fn, an unsafe
  operation (E0803 outside `unsafe(..)`).

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
| `v.store_stream(s, i)` float lanes | `movntps/pd` | `vmovntps/pd` | `vmovntps/pd` (EVEX) | plain `str q` (b) | plain `st1w/d` (b) | `st.global.cs` (`.v2/.v4` per 128 bits) |
| `v.store_stream(s, i)` int lanes | `movntdq` | `vmovntdq` | `vmovntdq` (EVEX) | plain `str q` (b) | plain `st1b/h/w/d` (b) | `st.global.cs` |
| `x.store_stream(s, i)` 32/64-bit int | `movnti` | `movnti` | `movnti` | plain `str` (b) | plain `str` (b) | `st.global.cs` |
| `x.store_stream(s, i)` f32/f64 | `movq` + `movnti` (2) | same | `vmovd/vmovq` + `movnti` (2) | plain `str` (b) | plain `str` (b) | `st.global.cs` |
| `x.store_stream(s, i)` 8/16-bit | plain `mov` (c) | plain | plain | plain `strb/strh` | plain | `st.global.cs` |
| `fence_stream()` | `sfence` | `sfence` | `sfence` | none (b) | none (b) | none (d) |

- (a) `prefetchw` needs PRFCHW (Intel Broadwell+, every AVX-512 CPU), which v2/v3 do not imply
  (Haswell lacks it), so v2/v3 use the read hint of the same level.
- (b) AArch64 code uses no non-temporal store (`stnp` needs a register pair, and its ordering
  against later stores is not relied on): every streaming store is the plain store with the same
  checks, so no fence is needed.
- (c) x86 has no 8/16-bit non-temporal store, and none for a vector under 16 bytes (stored
  plainly too). The float rows are f32/f64 lanes; f16/bf16 lanes take the int row.
- (d) `.cs` is a cache operator on a weak store; PTX ordering is unchanged. The `.global` space
  is used when the slice is known global; otherwise the generic `st.cs` / `prefetch.L1` forms.

Lowering notes the backend MUST respect:
- **compress on v4:** compress to a register, then a masked store. Never use the memory form,
  which is microcoded on Zen 4.
- **min/max on x86:** `min`/`min_num` use the fixed sequence. `minps` semantics are only
  available as `min_x86`, whose NaN result has an unspecified quiet bit and payload, so x86 emits
  the bare instruction and other targets need no payload-exact sequence.
- **fixed reduce tree on neon:** pairwise `faddp` does not follow the tree on more than two
  lanes: float `reduce_add` folds the upper half onto the lower half (`ext` + `fadd` per step),
  class M on f32 and N on f64.
- **x86 VEX/EVEX transitions:** VZEROUPPER is inserted before calls and returns out of 256/512-bit
  code.
