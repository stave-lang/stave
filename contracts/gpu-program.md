# Contract: the GPU program target (v0.2)

User-visible behaviour of `--target gpu-sm86`. The `LANG.md` GPU-program section is generated from
this contract and must stay within 400 tokens.

## Selecting the target

```text
stave build --target gpu-sm86 [--release] [--alloc-check] [--timings]
stave run   --target gpu-sm86 [-- ARGS...]
stave test  --target gpu-sm86        # E1001 (not run) on a host without a usable GPU
```

- The source is the same as for CPU targets. There is no new syntax and no new keyword.
- The output is one dynamic ELF for x86-64 Linux. It contains the generic host shim and the device
  image (`.stave.gpu`; [host-channel.md](host-channel.md) "Shim and image").
- `gpu-sm86` is never picked automatically (no silent target choice).
- `--alloc-check` exists only for `gpu-sm86` in v1. With another target it is a usage error
  (exit 2: `--alloc-check needs --target gpu-sm86`).

## Manifest keys (`stave.toml`)

| Key | Default | Meaning | Targets |
|---|---|---|---|
| `[run] stack` | `"8MiB"` | Per-thread stack limit. GPU: the Stave frame stack. CPU: the thread stack size | all |
| `[gpu] heap_max` | `"90%"` | Device heap cap: a size (`"40GiB"`, at least `"64MiB"`) or a percentage of free device memory at start. Growing beyond it is OutOfMemory. Resident programs set it explicitly. `STAVE_GPU_HEAP_MAX` overrides it for one run (below) | gpu-sm86 |
| `[gpu] block` | `256` | Runtime block size R: a multiple of 32, at most 1024. Kernels used by the program need block size B: a multiple of 32, B ≤ R, R % B == 0, and R/B ≤ 15 when B > 32 (E1202) | gpu-sm86 |
| `[gpu] poll` | `"adaptive"` | Host poller: `"adaptive"` (busy, then 50 µs sleeps after 10 ms idle, 1 ms sleeps after 1 s) or `"busy"` | gpu-sm86 |

Value grammar:
- A **size** is a TOML string: a decimal integer with no sign, point, `_`, or leading zero, then
  directly one unit `B`, `KiB`, `MiB`, `GiB`, or `TiB` (`"8MiB"`, `"4096B"`). No space, no SI units
  (`KB`, `MB`), no fractions (`"1.5GiB"`). The value must be > 0 and fit in `u64`. `stack` is at
  least `"4KiB"`, `heap_max` at least `"64MiB"`.
- A **percentage** (only `heap_max`) is a TOML string: an integer 1–100, then `%` (`"50%"`). `"0%"`
  and `"120%"` are invalid.
- `block` is a TOML integer. `poll` is the string `"adaptive"` or `"busy"`.

Invalid values are E1203, located in `stave.toml`.

**Environment** (read by the shim at start; they stay visible to the program):

| Variable | Effect |
|---|---|
| `STAVE_GPU_HEAP_MAX=<size\|N%>` | Overrides `[gpu] heap_max` for this run, same grammar. The stage1 compiler daemon is started with `50%`. Invalid, or below 64 MiB once resolved: exit 125 |
| `STAVE_GPU_TIMINGS=1` | Prints the run-time timings at exit, as a `--timings` build does |

## What runs where

| Construct | On `gpu-sm86` |
|---|---|
| `main` and everything it calls | Device, on one thread |
| `par_for`, `join`, `scope { spawn }` | Device, as regions across the resident warps. They are `std.thread` functions, and the `gpu-sm86` backend of `std.thread` creates the regions. Same Send/Sync and borrow rules (E0801, E0802) |
| `kernel fn` + `launch` + `wait` ([gpu-lang.md](gpu-lang.md)) | Device, as a virtual grid. No CPU round trip. Kernel bodies keep the kernel subset (E1101, E1102). A block size outside the `[gpu] block` rules is a launch-precondition `Err` (E1202 when B is constant). A failing kernel thread aborts its virtual block, and `wait` returns `Err(KernelError)` as on CPU targets |
| Heap types (`Vec`, `String`, `Map`, `Box`) | Device heap (slab allocator) |
| Recursion, fn values, closures | Device, outside kernels. Deep recursion uses the Stave frame stack |
| `std.io`, `std.fs`, `std.env`, `std.time`, `std.process.exit` | Host shim services. Same `Result` values as on the CPU |
| `dispatch` | A `cuda` arm is always taken: `cuda` is owned by `gpu-sm86` and always true there. Otherwise the `_` arm. CPU-architecture arms are dropped |
| `extern "C"` declarations and calls | E1201 |
| `allocator path;` | Allowed. The replacement allocator runs on the device and must be safe for concurrent use (it is trusted `unsafe` code) |

## Semantics

Identical to the CPU targets, except for:
- the order of unsynchronised output between threads;
- the thread field of panic reports;
- the exact point at which a resource limit (heap, stack) is hit.

Each `io.print` or `write` call of at most one staging chunk (1 MiB) reaches the host as one
`write(2)` and is never split. Larger calls are split into chunk-sized `write(2)` calls in order,
with the same semantics as the CPU std `write_all` ([host-channel.md](host-channel.md)).

## `std.gpu` on `gpu-sm86`

`std.gpu` host code (host functions, [gpu-lang.md](gpu-lang.md)) runs on the device here.

| API | On `gpu-sm86` |
|---|---|
| `gpu.Device.open(i)` | `Ok` for device 0, the device the program runs on. Other indices: `Err` (multi-GPU is out of scope) |
| `dev.stream()` | A runtime queue on the resident grid. Work on one stream runs in order; streams run concurrently. No CUDA stream is created |
| `dev.alloc`, `dev.alloc_copy` | Device heap memory |
| `s.copy_to_device`, `s.copy_to_host`, `buf.to_host` | Device memory copies run as regions. No PCIe transfer |
| `k.launch` + `wait` | Virtual grid. The prologue skips the kernel if an earlier kernel on the same stream failed (gpu-lang.md "Waiting") |
| Thread witness `t.local()`, `t.block()`, `t.global()` | Virtual thread and block indices, never the resident hardware ones |
| Collectives (`block_*`, `tile_load`, `gpu.barrier`) | Named barrier of the virtual block (`bar.sync id, B`), or `bar.warp.sync` when B = 32, followed by the abort-flag check |
| `warp_*` | Unchanged: virtual blocks are whole warps |
| `dev.compute_capability()` | `(8, 6)` |
| `dev.max_threads_per_block()` | R: the largest valid virtual block |
| `dev.shared_mem_per_block()` | The carve-out C per resident block, fixed at build time. A virtual block of size B gets C × B / R |

## Exit status

| Situation | Exit status (both targets) |
|---|---|
| `main` returns `n` | `n` |
| Panic (bounds, overflow, division by zero, assertion, OutOfMemory, StackOverflow) | 101, with a report on stderr: `panic: <kind> at <file>:<line>:<col> in <fn> (thread <id>)`. GPU thread id: `block (x,y,z) thread (x,y,z)`. For OutOfMemory the location is the innermost call outside the shipped `std` and runtime packages |
| Debug-build hardware-stack check fails (`gpu-sm86` only) | 101, `panic: HwStackOverflow at …`: the static bound was wrong, a compiler bug |
| `--alloc-check` finds a block error | 101, `panic: AllocCheck(<what>) at …` ("Allocation checker") |
| Device fault (compiler or `unsafe` bug) | 101, with `device fault: <CUDA error name> (unsafe code or compiler bug)` |
| SIGINT / SIGTERM | Killed by the signal (shell shows 128 + signal number) |
| Cannot start | 125, with `stave: cannot start GPU program: <reason>` (reasons below) |

Cannot-start reasons, each on one line after the prefix:
- no usable GPU: `no CUDA device of compute capability ≥ 8.6: <probe text>`, where `<probe text>`
  is the `Display` text of the `gpu.Error` `NoDriver`, `NoDevice`, `DriverTooOld`, or
  `CcTooLow` ([gpu-ops.md](gpu-ops.md) "Errors"). The shim runs the probe steps of that table in
  the same order;
- image: `missing or truncated .stave.gpu image`, `image ABI version <n>, shim expects 1`,
  `image needs the ptx_compile shim`;
- environment: `invalid STAVE_GPU_HEAP_MAX "<value>": …`, `heap_max resolves to <n> MiB, below the
  64 MiB minimum`;
- driver error during start: the `gpu.Error.Driver` text, `CUDA driver error in <call>: <name>
  (<code>): <text>`.

Exit codes are those of CPU targets: 101 for a panic, and 125 for "cannot start" (GPU programs only). `main`
may itself return 101 or 125. The report line on stderr tells the cases apart. A panic is also sent
to the program's panic sink, if one is registered.

## Allocation checker (`--alloc-check`, `gpu-sm86` only in v1)

The allocator adds 16 B red zones around each block (the front one also holds the block's size
and allocation site), fills freed blocks with `0xDB`, and checks the bitmap state on each free. On
stderr:
- a block error stops the program at once: `panic: AllocCheck(<what>) at <file>:<line>:<col> in
  <fn> (thread …)`, exit 101. `<what>` is `double free`, `invalid free`, or `red zone overwritten
  (block of <size> B allocated at <file>:<line>:<col>)`. The location is the free;
- at a normal exit, one summary line `alloc-check: <e> errors, <n> leaks (<bytes> B)`, then one
  line per leak, largest first, at most 20: `alloc-check: leak <size> B allocated at
  <file>:<line>:<col> in <fn>`. Leaks never change the exit status.

Reads of freed memory are not detected; they return the poison pattern.

## Build reports

`--timings` on `gpu-sm86` adds three sections:

```text
device: hw stack 4 KiB (bound), frame stack limit 8 MiB, R=256, regs 64
device: ptxas 312 ms (whole program, 16 threads), cubin 2.1 MiB
parallelism: 41 fns sequential, 118 parallel, 9 both
  sequential: main, parse_args, load_config, report_totals, ...
  longest sequential: load_config 18.2k instrs, main 6.1k, report_totals 2.4k, ...
  recursion (frame stack): {parse_expr, parse_unary} [in region: no], {walk} [in region: yes, E1205]
  indirect calls in regions: sort_by at src/sort.stv:41 (fn(&T, &T) -> i32) [E1206]
```

"Longest sequential" lists the largest sequential sections: functions reachable only outside
regions (on thread 0), by PTX instruction count after inlining, largest first.

`--timings --diagnostics json` emits `{"kind":"parallelism","sequential":[...],"parallel":[...],"both":[...],"longest_sequential":[{"fn":...,"instrs":...}],"recursion":[{"fns":[...],"in_region":...}],"indirect_in_region":[{"span":...,"fn_type":...}]}`.
E1205 and E1206 are printed as warnings in every build, with or without `--timings`. The JSON
line is the `parallelism` kind of `diagnostics.schema.json`.

## Run-time timings

A program built with `--timings`, or run with `STAVE_GPU_TIMINGS=1`, prints these lines on stderr
when it exits, after its own output (and after a panic report):

```text
gpu run: driver init 118.4 ms (not gated)
gpu run: start→main 4.61 ms (stack limit 2.20, heap 0.41, module 1.18, slots 0.31, staging 0.22, launch→main 0.07)
gpu run: grid 336x256 (R=256, regs 64), slots 2688, shared 0 B/block
gpu run: heap mapped 256 MiB, max 43.2 GiB, grow steps 0; frame stack high water 12 KiB/thread
gpu run: regions 0, virtual grids 0, idle enter/leave 0/0, stop→grid exit 21 µs
```

`start→main` is the start latency that CI gates (driver init is reported, not gated). Its end is
the device's `main_t0` converted to host time with the calibrated clock offset, accurate to about
2 µs. The grid shape is known only
at run time, so it is here and not in the build report.

## Admission table

| Feature | Compile-time cost | Lowering | `LANG.md` cost (tokens, est.) |
|---|---|---|---|
| `gpu-sm86` target | Linear analyses: SCCs, address-taken locals, may-block, stack bound | Whole-program PTX plus the runtime entry | 120 |
| Manifest keys `stack`, `heap_max`, `block`, `poll` | O(1) | `RuntimeConfig` | 100 |
| E1201–E1206 | One reachability pass (shared with the device-fn checks); SCCs and region bodies are already computed | None | 80 |
| Parallelism report | One pass over the call graph | None | 60 |
| Exit status table | None | Shim | 30 |
| **Total** | | | **≈390 (budget 400)** |

The `std.gpu` table above is not copied into `LANG.md`: the "What runs where" kernel row covers
it.

## Example (**non-normative** until the `gpu-sm86` target is implemented)

```stave
import std.io;
import std.thread;

fn fib(n: u64) -> u64 {
    return if n < 2 then n else fib(n - 1) + fib(n - 2);   // recursion: Stave frame stack; reachable from a join arm, so warning E1205
}

fn main() -> i32 {
    let r = thread.join::<u64, u64>(fn() -> u64 { return fib(30u64); }, fn() -> u64 { return fib(31u64); });
    let a = r.0;
    let b = r.1;
    io.print(f"fib(30) = {a}, fib(31) = {b}\n");
    return 0;
}
```

`thread.join` has the signature `join<A, B>(a: fn() -> A, b: fn() -> B) -> (A, B)` (`LANG.md` §3).
The example becomes normative, and is compiled and run for both targets in CI, once the target
exists, before `LANG.md` is updated.
