# Contract: GPU language additions (v0.2)

Additions to the Stave language for the PTX target. Everything not listed here is Stave as in
`LANG.md`. The `LANG.md` GPU section is generated from this contract and must stay within 1,500
tokens.

## Grammar delta (grammar.ebnf: `kernel` is in the grammar since v0.3)

```ebnf
Keywords    += "kernel" ;
ItemBody     = ... | KernelDecl ;
KernelDecl   = "kernel" "fn" Ident "(" Params? ")" Block ;   (* no generics, return type, or needs in v1 *)
```

- LL(1): `kernel` is a new leading keyword, so it is decided by the first token. The rule is local,
  so it has no cost to parallel parsing.
- No other syntax is added. Launch, copies, and cooperative operations are ordinary method and
  function calls on `std.gpu` types. `<<<>>>` does not exist.

## Kinds of functions

| Kind | Declared as | Runs on | Rules |
|---|---|---|---|
| Kernel | `kernel fn k(t: gpu.Thread1, ...)` | GPU, once per thread | First parameter is the thread witness (`gpu.Thread1` or `gpu.Thread2`). Other parameters are device-parameter types (below). Returns unit. No `needs`, no generics (v1). E1108 otherwise |
| Device function | plain `fn` | GPU, when reachable from a kernel | Must be GPU-compilable: no heap, I/O, threads, `extern`, `dispatch`, recursion, or calls through fn values (E1101, E1102). The same fn may also be used on the CPU. **These restrictions are the kernel subset. They hold on every target, including the whole-program target `gpu-sm86`; GPU programs get heap, recursion, fn values, and I/O only outside kernels** |
| Host function | plain `fn` | CPU (on `gpu-sm86`: the device) | May use `std.gpu` host types. `needs(cuda)` marks code that requires a device (see Dispatch) |

## Types (`std.gpu`)

Host side:

| Type | Meaning |
|---|---|
| `gpu.Device` | An opened CUDA device. `gpu.Device.open(i) -> Result<gpu.Device, gpu.Error>` |
| `gpu.Stream` | An ordered queue of device work. `dev.stream() -> Result<gpu.Stream, gpu.Error>` |
| `gpu.Buf<T>` | Owned buffer in device global memory, `T: gpu.Plain`. `dev.alloc::<T>(n)` (zero-filled), `dev.alloc_copy(&[T])` |
| `gpu.Launch` | Pending device work: a **view** whose origins are its buffer arguments, and **must-consume**: only `l.wait() -> Result<(), gpu.Error>` ends it (E1104) |
| `gpu.Grid1`, `gpu.Grid2` | Launch shape. `gpu.Grid1.cover(n, block)` gives enough blocks of `block` threads to cover `n`. `gpu.Grid2.cover(nx, ny, bx, by)` covers an `nx × ny` grid with blocks of `bx × by` threads (see "2-D grids") |
| `gpu.Error` | Probe, driver, out-of-memory, launch-precondition error, or `KernelError { kernel, span, kind, block, thread }`. Variants and `Display` texts: [gpu-ops.md](gpu-ops.md) "Errors" |

Device side (only valid in kernels and device functions):

| Type | Meaning |
|---|---|
| `gpu.Thread1`, `gpu.Thread2` | Thread witness. Created by the compiler at kernel entry. A non-escapable, non-copyable view (E1107). `t.global() -> usize` (or `(usize, usize)`), `t.local()`, `t.block()` |
| `gpu.In<T>` | Read-only device global memory. Indexed `x[i]` with bounds checks. `x.len()` |
| `gpu.Out<T>` | Writable device global memory, written only through `y.at(t) -> Option<&mut T>`. `None` means the thread is past the end (the grid tail). Element index: `t.global()` for `Thread1`, row-major for `Thread2` ("2-D grids") |
| `gpu.AtomicOut<T>` | Device global memory accessed only through `h.atomic_add(i, v)`, `atomic_min`, `atomic_max` (`T`: i32, u32, i64, u64, f32) |
| `gpu.Tile<T, N>`, `gpu.Tile2<T, R, C>` | Read-only block-shared view returned by `gpu.tile_load` (1-D) or `gpu.tile_load_2d` (2-D), read with `tile.get(i)` / `tile.get(r, c)` (bounds-checked). Its borrow ends at the next collective, including another `tile_load` and a call to a device fn that runs one; a later use is E1111 |
| `gpu.Shared<T>` | Raw block-shared memory. Every access is `unsafe("...")` |
| `mask32x32` | Warp-wide mask (32 lanes). Returned by `gpu.warp_ballot`, used with Stave mask operations. `to_bits() -> u32` |

`gpu.Plain` (marker trait): numeric scalars, `bool`, fixed vectors, arrays and structs of those.
No references, views, heap types, or raw pointers. A buffer or kernel parameter of another type is
E1110.

Memory spaces are distinct types. Passing one where another is expected (host slice for `gpu.In`,
`gpu.In` for `gpu.Out`, and so on) is E1106.

## Launching

For each kernel `k` the compiler provides `k.launch(stream, grid, args...)`. Its parameters are
derived from the kernel's, minus the witness:

| Kernel parameter | Launcher argument |
|---|---|
| scalar or vector `T: gpu.Plain` | `T` by value |
| `gpu.In<T>` | `&gpu.Buf<T>` |
| `gpu.Out<T>`, `gpu.AtomicOut<T>` | `&mut gpu.Buf<T>` |

The launcher returns `Result<gpu.Launch, gpu.Error> from <all buffer arguments>`. Its checks are:
- **At compile time:** argument types match the kernel exactly (E0201/E1106). A constant grid that
  exceeds device limits is E1109.
- **On the host before launch:** the grid shape matches the witness kind; the block size is within
  device limits (and a multiple of 32 when the kernel uses warp operations or collectives); the
  grid has fewer than 2^40 threads; the device is open. Each failure is `Err`, and nothing is
  launched.

Async copies use the same loan model: `s.copy_to_device(&host, &mut buf) -> Result<gpu.Launch, ..>`
and `s.copy_to_host(&buf, &mut host) -> Result<gpu.Launch, ..>`, both `from` their buffers.

While a `gpu.Launch` is alive, its buffers are borrowed, `&mut` buffers exclusively, so host code
touching them is E0501, with the launch shown as the borrow site.

### Waiting

`l.wait()` returns `Ok` unless the launch had a failed check or was skipped. A failed launch gives
`Err(gpu.Error.KernelError { .. })` naming the failing thread with the smallest linear block
index, then the smallest linear thread index (deterministic). A failure
poisons the stream: launches enqueued after it are skipped (async copies still run), and their
`wait` returns the same `Err`. The first `wait` that reports the failure (on the failing launch or
a later one) resets the stream, so launches enqueued after that call run normally. Driver errors
(`gpu.Error.Driver`) are reported by the operation or `wait` that sees them.

### 2-D grids

`gpu.Grid2.cover(nx, ny, bx, by)` launches `ceil(nx / bx) × ceil(ny / by)` blocks of `bx × by`
threads, only for `gpu.Thread2` kernels (otherwise `Err`, limit `GridShape`). `bx × by` is the
block size for every limit (E1109, `BlockSize`, `WarpMultiple`). For `t: gpu.Thread2`,
`t.global()` is `(gx, gy)` with `gx = block.x × bx + local.x` and `gy = block.y × by + local.y`;
`t.local()` and `t.block()` are `(x, y)` pairs. `y.at(t)` on `gpu.Out<T>` is element
`gy * nx + gx` (row-major, row width `nx` from the grid) and `None` when `gx ≥ nx`, `gy ≥ ny`, or
`gy * nx + gx ≥ y.len()`. Distinct threads get distinct elements, so rule 1 holds. `nx` and `ny`
are hidden kernel parameters.

Transpose (reference kernel, no `unsafe`): an `h × w` row-major input `x` becomes the `w × h`
output `y`. The launch covers the output, `gpu.Grid2.cover(h, w, 32, 32)`, so `gx` is an input row
and `gy` an input column. With `(ix, iy) = t.block()` each block loads its input tile
`gpu.tile_load_2d::<f32, 32, 32>(&t, x, w, ix * 32, iy * 32)` (coalesced along input rows, uniform
arguments), and with `(lx, ly) = t.local()` each thread writes `tile.get(lx, ly)` through
`y.at(t)` (coalesced along `gx`).

## Safety rules (device code)

1. **Writes:** in safe code, mutable device memory is written only through `out.at(t)` (one
   element per thread) or atomic operations. Any other write to `gpu.Out`, `gpu.AtomicOut`, or
   `gpu.Shared` is E1103 (or needs `unsafe`).
2. **Witness:** the thread witness cannot be copied, stored, returned, or captured (E1107). It can
   be passed down as `&gpu.Thread1` to device functions.
3. **Uniformity:** collectives (`block_*`, `warp_*`, `tile_load`, `gpu.barrier` in unsafe code)
   must run in block-uniform control flow (also for `warp_*` in v1: there are no warp-only uniform
   sources). Values derived
   from `t.global()`, `t.local()`, loads, or lane values are non-uniform. A branch or loop on them
   that contains a collective, or a return or break under them before a collective, is E1105.
4. **Runtime checks:** bounds, overflow, and division by zero behave as on the CPU (a failed
   check is an error, never wrapped). On the GPU they are reported at the next `wait` as
   `gpu.KernelError` (see "Waiting"), and the device stays usable: a failing thread writes an
   error record and exits. PTX `trap` is never used.
5. **Unsafe:** `gpu.Shared`, `gpu.barrier()`, raw warp intrinsics, and unchecked indexing are
   `unsafe("obligation")` and appear in the unsafe inventory (`stave unsafe-report`).

## Dispatch

`cuda` is a runtime feature (owned by both x86-64 and AArch64 hosts; in v1 GPU code runs only on
x86-64, and on AArch64 the probe always fails). `needs(cuda)` on a host function
means "a usable device of the target compute capability is present". Calling it from a context
without `cuda` is E0704, and `main` cannot declare it (E0709), so a program reaches `needs(cuda)`
code through a `dispatch` arm. `std.gpu` host operations need no `needs(cuda)`: without a device
they return the probe's error (`gpu.Error.NoDevice`, ...). `dispatch` arms may name `cuda`:

```stave
dispatch fn sum(xs: &[f32]) -> Result<f32, gpu.Error> { cuda => sum_gpu, avx2 => sum_v3, _ => sum_base, }
```

The probe is lazy, runs once, and is cached.

## Float and integer semantics on the GPU

Same as on the CPU: IEEE round-to-nearest-even, no flush-to-zero, no automatic FMA contraction
(explicit `fma` only), NaN results are quiet NaNs with unspecified sign and payload, and integer
`+ - *` are checked for overflow (a checked-arithmetic failure is reported per rule 4). Approximate GPU math exists only as named
operations: `gpu.fast_exp`, `gpu.fast_rsqrt`, and so on ([gpu-ops.md](gpu-ops.md)).

## Admission table

| Feature | Compile-time cost | Lowering | `LANG.md` cost (tokens, est.) |
|---|---|---|---|
| `kernel fn` | O(1) parse; kernel signature check O(params) | PTX `.entry`, with `.maxntid N` when every launch passes the same constant block size N | 120 |
| Thread witness | View rules (`LANG.md` §3), O(1) per use | `%tid`/`%ctaid`/`%ntid` reads | 120 |
| `gpu.In/Out/AtomicOut` | Type check O(1) per access | `ld.global.nc` / `st.global` / `atom.global` | 200 |
| Generated launcher | O(params) per kernel | `cuLaunchKernel` with a packed parameter array | 150 |
| `gpu.Launch` view + must-consume | Existing origin analysis + one "no implicit drop" check per local | Stream event and error-record read at `wait` | 150 |
| Uniformity analysis | Linear in function size (graph reachability, callee summaries) | None | 150 |
| Checked collectives | Ordinary calls | `std.gpu` code using shared memory and `bar.sync` | 250 |
| `cuda` feature in `dispatch` | Same as existing features | Cached device probe | 80 |
| Device error reporting | None at compile time | Error record + `exit` | 80 |
| **Total** | | | **≈1,300 (budget 1,500)** |

## Example (normative; compiled in CI)

```stave
import std.gpu;

kernel fn saxpy(t: gpu.Thread1, a: f32, x: gpu.In<f32>, y: gpu.Out<f32>) {
    let i = t.global();
    match y.at(t) {
        Option.Some(yi) => { *yi = a.fma(x[i], *yi); }
        Option.None => {}
    }
}

kernel fn total(t: gpu.Thread1, x: gpu.In<f32>, out: gpu.AtomicOut<f32>) {
    let i = t.global();
    let v: f32 = if i < x.len() then x[i] else 0.0;
    let s = gpu.block_reduce_add(&t, v);          // every thread of the block calls it (uniform)
    if t.local() == 0 { out.atomic_add(0, s); }
}

fn run_gpu(hx: &[f32], hy: &mut [f32]) -> Result<(), gpu.Error> needs(cuda) {
    let dev = gpu.Device.open(0)?;
    let s = dev.stream()?;
    let dx = dev.alloc_copy(hx)?;
    let mut dy = dev.alloc_copy(hy)?;
    let l = saxpy.launch(&s, gpu.Grid1.cover(hx.len(), 256), 2.0, &dx, &mut dy)?;
    l.wait()?;                                    // kernel errors surface here; loans end
    s.copy_to_host(&dy, hy)?.wait()?;
    return Result.Ok(());
}

fn run_cpu(hx: &[f32], hy: &mut [f32]) -> Result<(), gpu.Error> {
    let a: f32 = 2.0;
    for i in 0..hx.len() { hy[i] = a.fma(hx[i], hy[i]); }
    return Result.Ok(());
}

// `main` cannot declare needs(cuda) (E0709): it calls `run`, which picks the arm at run time
dispatch fn run(hx: &[f32], hy: &mut [f32]) -> Result<(), gpu.Error> { cuda => run_gpu, _ => run_cpu, }
```
