# Contract: GPU operations and availability (v0.2)

This file is the source for the GPU target tables in `crates/stave-target/tables/ptx-*.toml`, for
`stave lowering --target ptx-sm86`, and for the GPU conformance suite. It uses the same classes as
[simd-ops.md](simd-ops.md): **N** = native (a few
instructions), **M** = multi-instruction sequence, **U** = unavailable (E0700 unless inside
`emulate { }`; inside `strict { }` M and U are E0701).

Tier 1 target: `ptx-sm86` (RTX A6000). Columns for other compute capabilities are added when they
are supported.

## Scalars in device code

| Op | Syntax | sm_86 | Lowering notes |
|---|---|---|---|
| f32 add/sub/mul | `a + b` … | N | `add.rn.f32` etc. Explicit rounding prevents contraction |
| f32 div / sqrt (IEEE) | `a / b`, `a.sqrt()` | M | `div.rn.f32`, `sqrt.rn.f32` |
| f32/f64 fma (explicit) | `a.fma(b, c)` | N | `fma.rn` |
| f64 arithmetic | `a + b` … | N | `add.rn.f64` etc. Low throughput on GA102 (1/64 of f32), still class N |
| int add/sub/mul, checked | `a + b` … | M | Op plus overflow check, cold failure block |
| int wrapping / saturating | `a +% b`, `a +| b` | N | `add.s32`, `add.sat.s32` (saturating i8/i16/u*: M) |
| int div / rem | `a / b`, `a % b` | M | Software sequence from ptxas, plus a zero check |
| shifts | `a << b` … | M | Amount check (as on the CPU) plus `shl` |
| conversions | `as`, `as%`, `as|`, `as?`, `as~` | N–M | `cvt` with explicit rounding |
| approximate math (named only) | `gpu.fast_exp(x)`, `gpu.fast_log`, `gpu.fast_sin`, `gpu.fast_cos`, `gpu.fast_rsqrt`, `gpu.fast_div` | N | `.approx` forms. Never used implicitly |
| f16/bf16 arithmetic | on `f16` lanes | N | `add.rn.f16` etc. |

## Vectors in device code

Fixed vectors up to 128 bits (`f32x4`, `i32x4`, `u8x16`, `f16x8`, …) are allowed in registers. Lane
operations lower to per-lane scalar instructions, plus `ld.v2`/`ld.v4` and `st.v2`/`st.v4` for
aligned loads and stores. Wider vectors and sizeless vectors in device code are E0702/E0710.
Horizontal operations (`reduce_*`, `permute`, `compress`) are M.

## Device memory

| Op | Syntax | sm_86 | Lowering notes |
|---|---|---|---|
| read input | `x[i]` on `gpu.In<T>` | N | `ld.global.nc`, plus a bounds check (hoisted when possible) |
| vector read input | `f32x4.load_in(x, i)` | N | `ld.global.nc.v4.f32`, alignment from the buffer type |
| per-thread write | `*y.at(t) = v` | N | `st.global`. The `at` check is one compare against the length. `Thread2`: index `gy * nx + gx`, compares against `nx`, `ny`, and the length (gpu-lang.md "2-D grids") |
| atomic add i32/u32/f32/i64/u64 | `h.atomic_add(i, v)` | N | `red.global.add` when the result is unused, else `atom.global.add`. f32 sums in hardware order: not reproducible (see below) |
| atomic min/max i32/u32/i64/u64 | `h.atomic_min(i, v)` | N | `atom.global.min/max` |
| atomic min/max f32 | `h.atomic_min(i, v)` | M | CAS loop (no native float min/max on sm_86) |

## Cooperative operations (checked, safe; `std.gpu`)

All are collectives: block-uniform control flow (warp-uniform for `warp_*`), E1105 otherwise.

| Op | Syntax | sm_86 | Notes |
|---|---|---|---|
| block reduce add/min/max/and/or | `gpu.block_reduce_add(&t, v) -> T` | M | Warp shuffle tree, then shared memory. Float add uses the fixed order below (reproducible) |
| block inclusive/exclusive scan add | `gpu.block_scan_add(&t, v) -> T` | M | Warp scan plus a shared-memory pass |
| warp reduce add/min/max | `gpu.warp_reduce_add(&t, v) -> T` | N–M | `shfl.sync` tree, or `redux.sync` for integers |
| warp shuffle | `gpu.warp_shuffle(&t, v, lane)`, `_up`, `_down`, `_xor` | N | `shfl.sync` with a full mask |
| warp ballot / any / all | `gpu.warp_ballot(&t, p) -> mask32x32`, `warp_any`, `warp_all` | N | `vote.sync.ballot` |
| tile load | `gpu.tile_load::<T, N>(&t, x, start) -> gpu.Tile<T, N>` | M | Cooperative coalesced `ld.global` to `st.shared`, then `bar.sync`. Out-of-range elements read as zero. Read with `tile.get(i)` |
| 2-D tile load | `gpu.tile_load_2d::<T, R, C>(&t, x, width, row0, col0) -> gpu.Tile2<T, R, C>` | M | Element `(r, c)` is `x[(row0 + r) * width + col0 + c]`, zero when `col0 + c ≥ width` or the index is ≥ `x.len()`. Coalesced row loads to `st.shared` with one pad column (`R × (C + 1)` elements, counted in `shared_bytes`), then `bar.sync`. Read with `tile.get(r, c)` |

Block collectives count as warp operations: the launcher requires a block size that is a multiple
of 32. A tile's borrow ends at the next collective on any path (another `tile_load`, or a call to a
device fn that runs a collective, included); a later use of the tile is E1111.

### Reduction and scan order

The IR interpreter and CPU references replay this order bit-exactly. Integer adds are
checked in the same order.
- **Warp reduce:** `shfl.sync.bfly` with lane offsets 16, 8, 4, 2, 1; each step adds
  `own + partner`. IEEE addition is commutative, so every lane ends with the same bits: the
  pairwise tree `((v0 + v16) + (v8 + v24)) + …` by lane index.
- **Block reduce:** each warp reduces as above; warp w's result goes to shared slot w; after the
  barrier, warp 0 loads slot l in lane l (lanes ≥ warp count load `-0.0`, the IEEE additive
  identity) and reduces again with the same tree; the result is broadcast through shared memory.
- **Multi-block (reference kernels):** per-block partials are written to `gpu.Out` by block
  index, then reduced by a second launch with the same tree, so the order is by lane, then by
  block index.
- **Block scan:** inclusive Hillis–Steele per warp (`shfl.sync.up`, offsets 1, 2, 4, 8, 16, lane
  ≥ offset adds `partner + own`); warp totals are scanned the same way by warp 0; lane result =
  `exclusive warp prefix + lane value` (warp 0: the lane value alone). Exclusive scan = the
  inclusive result of the previous thread; thread 0 gets `0.0` (integer: 0).
- **f32 `atomic_add`** sums in hardware order and is outside the reproducibility guarantee (which
  covers collectives). Conformance goldens use it only with exactly representable sums
  (small integers); deterministic float sums use a collective or a two-pass reduction.

## Unsafe device operations

| Op | Syntax | Obligation the programmer states |
|---|---|---|
| raw barrier | `unsafe("…") gpu.barrier()` | Reached by every thread of the block |
| raw shared memory | `gpu.Shared<[T; N]>` access | No two threads write the same element between barriers; reads after barriers |
| unchecked index | `unsafe("…") x.get_unchecked(i)` | `i < x.len()` |
| raw warp intrinsics with a custom mask | `unsafe("…") gpu.shfl_sync(mask, …)` | Every lane in `mask` participates |

## Checked-IR spelling

In the canonical checked IR ([checked-ir.md](checked-ir.md)) a GPU op prints as `gpu.<name>`,
where `<name>` is the `std.gpu` method or function it implements: `gpu.global`, `gpu.local`,
`gpu.block` (witness reads), `gpu.load_in` (`x[i]` on `gpu.In`), `gpu.at`, `gpu.atomic_add`,
`gpu.atomic_min`, `gpu.atomic_max`, `gpu.barrier`, `gpu.shfl_sync`, `gpu.warp_ballot`,
`gpu.warp_any`, `gpu.warp_all`, `gpu.shared_load`, `gpu.shared_store`. Checked collectives are
`std.gpu` Stave code and print as calls. A failed check stays one instruction, as on the CPU; the
error-record path is PTX lowering, not checked IR.

## Unavailable in device code (E1101 / E1102)

Heap allocation (`Vec`, `String`, `Map`, allocator), file and console I/O, threads and `scope`,
`extern "C"` calls, `dispatch`, recursion, calls through fn values, `f"…"` strings, and sizeless
vectors. Closures that are inlined are allowed.

## Host operations (`std.gpu`; all return `Result`)

| Op | Syntax |
|---|---|
| open device | `gpu.Device.open(i)` |
| device info | `dev.compute_capability()`, `dev.max_threads_per_block()`, `dev.shared_mem_per_block()`. On `gpu-sm86` they report the virtual limits ([gpu-program.md](gpu-program.md)): R and the shared-memory carve-out |
| stream | `dev.stream()` |
| allocate | `dev.alloc::<T>(n)` (zero-filled with `cuMemsetD8_v2`, so no kernel reads uninitialized device memory), `dev.alloc_copy(&[T])` (synchronous) |
| async copy | `s.copy_to_device(&[T], &mut gpu.Buf<T>) -> gpu.Launch`, `s.copy_to_host(&gpu.Buf<T>, &mut [T]) -> gpu.Launch` |
| launch | `k.launch(&s, grid, args...) -> gpu.Launch` |
| wait | `l.wait()` |
| synchronous copy back | `buf.to_host(&mut [T])` |
| reset after a failure | none: the first `wait` that reports a kernel failure resets the stream's error record (gpu-lang.md "Waiting") |

Host operations need no `needs(cuda)`. Without a usable device they return the probe's error
(`NoDriver`, `DriverTooOld`, `NoDevice`, `CcTooLow`). `needs(cuda)` only marks code that runs
after a successful probe (gpu-lang.md "Dispatch").

## Errors (`gpu.Error`)

`gpu.Error` is an enum with these variants. It implements `Display` with the texts below (`{x}` is a
field). On CPU-program targets a `gpu.Error` is an ordinary `Err`: there is no fixed exit code,
and `main` chooses its status (the examples and goldens print the text to stderr and return 1).
Exit code 125 ([gpu-program.md](gpu-program.md)) applies only to `--target gpu-sm86` programs that cannot start.

| Variant | Fields | Raised by | `Display` |
|---|---|---|---|
| `NoDriver` | `reason: String` (`dlerror` text) | probe: `dlopen("libcuda.so.1")` fails; always on AArch64 in v1 | `no CUDA driver: {reason}` |
| `DriverTooOld` | `found: i32`, `needed: i32` | probe: `cuDriverGetVersion` below 11010 | `CUDA driver too old: version {found}, need {needed} or newer` |
| `NoDevice` | none | probe: `cuInit` returns 100, or the device count is 0 | `no CUDA device` |
| `CcTooLow` | `device: u32`, `found: (u32, u32)`, `needed: (u32, u32)` | probe: compute capability below the profile's | `CUDA device {device} has compute capability {found}, need {needed} or higher` (as `8.6`) |
| `OutOfMemory` | `bytes: u64` | allocation: `CUDA_ERROR_OUT_OF_MEMORY` | `CUDA device out of memory: allocating {bytes} bytes` |
| `LaunchConfig` | `kernel: String`, `limit: gpu.Limit`, `value: u64`, `max: u64` | launcher, before launching | `launch of kernel {kernel}: ` + the limit text |
| `KernelError` | `kernel: String`, `span: gpu.Span { file: String, line: u32, col: u32 }`, `kind: gpu.CheckKind`, `block: [u32; 3]`, `thread: [u32; 3]` | `wait`: a failed check in the launch, or in an earlier launch on the stream (skipped launch) | `kernel {kernel} failed: {kind} at {file}:{line}:{col} (block [x, y, z], thread [x, y, z])` |
| `Driver` | `call: String`, `code: i32`, `name: String`, `text: String` | any other `CUresult` ≠ 0, including a lost device (sticky: every later operation on the device fails too) | `CUDA driver error in {call}: {name} ({code}): {text}` (`cuGetErrorName`, `cuGetErrorString`) |

`gpu.Limit` texts: `GridShape` "grid shape does not match the kernel's thread witness",
`BlockSize` "block size {value} exceeds the device limit {max}", `WarpMultiple` "block size {value}
is not a multiple of 32", `GridSize` "grid size {value} exceeds the limit {max}" (a grid
dimension, or 2^40 threads in total), `SharedMem` "{value} bytes of shared memory exceed the device
limit {max}".

`gpu.CheckKind` texts: `Bounds` "index out of bounds", `Overflow` "integer overflow", `DivZero`
"division by zero", `Shift` "shift amount out of range".
