# Contract: host-service channel (internal ABI v1)

This is the protocol between the device runtime (`runtime-gpu/gpu/channel.stv`) and the host shim
(`crates/stave-gpu-shim`). It is internal: user code never sees it. It is versioned by
`shim_abi_version` in the image, and the shim refuses a mismatch with exit 125. Requests go
through mapped host memory, replies through device memory.

## Memory

**Rule: the device polls device memory only.** Mapped host memory is written by the device and
read by the host. When 504 blocks polled mapped memory, host-to-device copies fell from 12.9 to
1.4–1.8 GB/s.

| Area | Where | Writer → reader | Size |
|---|---|---|---|
| Request slots | `cuMemHostAlloc(DEVICEMAP \| PORTABLE)` | device → host | One per resident warp (grid blocks × R/32: 2,688 with the default 64 registers, 336 × 256 threads; 4,032 at 40 registers). 4,096 B each, aligned to 128 B |
| Reply slots | device memory, allocated before the launch | host (copy stream only) → device | One per request slot. 2,560 B each, aligned to 128 B |
| Control block | device memory, allocated and initialised before the launch | see "Control block" | 384 B, aligned to 128 B |
| Staging | pinned host chunks | host only | 1 MiB chunks, 2 per worker (16 for 8 workers). 256 KB–1 MB chunks measured equally well |
| Copy streams | non-blocking streams | host only | One per host worker and one for the poller |

`stop` is written by the device (`main` returned, panic) or by the host with
`cuStreamWriteValue32` (signal). Workers poll it in their scheduler loop. Panic reports
travel in the `panic_report` request's inline payload. There is no mirror in mapped memory.

## Control block (device memory)

The PTX entry `__stave_rt(ctl: u64)` passes the control block's address to
`runtime.gpu.start.entry` (contracts/intrinsics.md). The shim writes the host fields with one
`cuMemcpyHtoD` before the launch, and reads the whole block back with `cuMemcpyDtoH` after the grid
has ended (a blocking copy is allowed then).

| Offset | Field | Writer → reader | Notes |
|---|---|---|---|
| 0 | `stop: u32` | device (thread 0 after `main`, panic path) or host (`cuStreamWriteValue32`, signal) → device | 0 = run |
| 4 | `exit_code: i32` | device thread 0 → host | `main`'s return value, written before `stop` with `fence_sc_sys()` between |
| 8 | `timings: u32` | host → device | 1 when run-time timings are on (`RuntimeConfig.timings` or `STAVE_GPU_TIMINGS=1`) |
| 12 | `n_slots: u32` | host → device | Request/reply slot count (grid blocks × R/32) |
| 16 | `req_slots: u64` | host → device | Device pointer of the request slots (`cuMemHostGetDevicePointer_v2`) |
| 24 | `reply_slots: u64` | host → device | Device pointer of the reply slots |
| 32 | `heap_base: u64` | host → device | Start of the reserved heap range |
| 40 | `heap_mapped: u64` | host → device | Initially mapped bytes, `min(heap_initial, heap_max)` |
| 48 | `heap_max: u64` | host → device | Resolved cap in bytes (`STAVE_GPU_HEAP_MAX` or the manifest) |
| 56 | `main_t0: u64` | device thread 0 → host | `globaltimer()` immediately before `main` |
| 64 | `clock_offset: i64` | device (calibration warp) → host | Host `CLOCK_MONOTONIC` ns − `globaltimer` ns; only when `timings` = 1 |
| 128 | `TimingBlock` (128 B) | device → host | Counters for "Run-time timings" (contracts/gpu-program.md); only when `timings` = 1 |
| 256 | program `ErrorRecord` (128 B) | device panic path → host (copy in op 41) | `status`, `site_id`, `kind`, `block`, `thread`, `operands` |

## Request slot layout (mapped host memory)

| Offset | Field | Notes |
|---|---|---|
| 0 | `req_seq: u64` | Release store, `.sys` scope, **after** all other request fields |
| 8 | `opcode: u16`, `flags: u16` | |
| 12 | `lane_mask: u32` | Lanes taking part in this request (warp batching) |
| 16 | `args: [[u64; 8]; 32]` | Per lane: 64 B of arguments |
| 2064 | `inline_out: [u8; 2032]` | Small payloads device → host (paths, small writes, options) |

## Reply slot layout (device memory)

| Offset | Field | Notes |
|---|---|---|
| 0 | `reply_seq: u64` | Written last, with `cuStreamWriteValue64` |
| 16 | `results: [[u64; 2]; 32]` | Per lane: `(value, errno)`. `errno == 0` means success |
| 528 | `inline_in: [u8; 2032]` | Small payloads host → device |

## Protocol

- **Single writer per word.** No atomic read-modify-write on either area from either side.
  `HOST_NATIVE_ATOMIC_SUPPORTED = 0` on PCIe hosts.
- **Device** (one elected lane per warp):
  1. write `opcode`, `lane_mask`, `args`, and `inline_out`;
  2. `st.release.sys` `req_seq = seq`, where `seq` is the slot's last `reply_seq` + 1;
  3. poll `ld.acquire.sys reply_seq == seq` in the reply slot (device memory) with `nanosleep`
     backoff (cap 20 µs; 1 ms in idle mode);
  4. read `results` and `inline_in`, and broadcast them to the lanes with `shfl`.
- **Host**:
  1. the poller acquire-loads `req_seq` and compares it with the last served value it keeps;
  2. it serves the request itself, or hands a blocking call to a worker;
  3. the server writes `results` and `inline_in` into a pinned per-slot reply mirror, then issues on
     its own copy stream, in order: the bulk data copy (if any), `cuMemcpyHtoDAsync` of the reply
     mirror, and `cuStreamWriteValue64(reply_seq = seq)`. Stream order makes the data and results
     visible before `reply_seq`;
  4. a staging chunk is reused only after the stream's event completes.
- A warp has at most one request in flight, so a slot needs no queue.

## Services (opcode → per-lane args → result)

| Op | Name | Args | Result `value` | Notes |
|---|---|---|---|---|
| 1 | `write` | fd, dev_ptr, len | bytes written | `len ≤ 2032 / lanes`: payload in `inline_out`. Otherwise DtoH into staging, then `write(2)`. A request of at most one staging chunk (1 MiB) is **one** `write(2)`. Larger requests are split into chunk-sized `write(2)` calls in order, with the CPU std `write_all` semantics |
| 2 | `read` | fd, dev_ptr, len | bytes read | Host `read(2)` into staging, then HtoD on the copy stream; the reply follows the copy in stream order |
| 3 | `pread` | fd, dev_ptr, len, offset | bytes read | As `read` |
| 4 | `pwrite` | fd, dev_ptr, len, offset | bytes written | As `write` |
| 5 | `open` | path in inline, flags, mode | fd | `openat(AT_FDCWD, …)` |
| 6 | `close` | fd | 0 | Also closes socket handles (ops 50–55) |
| 7 | `fstat` | fd, dev_ptr | 0 | Writes the `Stat` wire layout (below) to `dev_ptr` |
| 8 | `lseek` | fd, offset, whence | new offset | |
| 9 | `getdents` | fd, dev_ptr, len | bytes | `getdents64(2)`: Linux `linux_dirent64` records, unchanged (below) |
| 10 | `unlink` / 11 `mkdir` / 12 `rename` | paths in inline | 0 | |
| 13 | `fsync` / 14 `fdatasync` | fd | 0 | Durability for databases and logs |
| 15 | `ftruncate` | fd, len | 0 | |
| 16 | `lock` | fd, cmd, type, start, len | 0 | `fcntl` record locks (`F_SETLK`/`F_SETLKW`/`F_GETLK`) and `flock`. `F_SETLKW` runs on a worker thread |
| 20 | `args` | dev_ptr, cap | total bytes | NUL-separated argv, copied once and cached by the device runtime |
| 21 | `env` | dev_ptr, cap | total bytes | NUL-separated environment, cached |
| 22 | `clock` | clock id | nanoseconds | `clock_gettime` |
| 23 | `getrandom` | dev_ptr, len | bytes | `getrandom(2)` |
| 24 | `mem_info` | dev_ptr | 0 | Writes `{free: u64, total: u64}` (`cuMemGetInfo`) to `dev_ptr`. Used by the stage1 compiler daemon for E1301 |
| 30 | `heap_grow` | bytes | new mapped end | `cuMemCreate` + `cuMemMap` + `cuMemSetAccess` inside the reserved range; errno ENOMEM on failure or `heap_max` |
| 31 | `heap_release` | offset, bytes | 0 | Unmaps and releases whole 2 MiB chunks the allocator no longer uses (`cuMemUnmap` + `cuMemRelease`, from the helper context). The range stays reserved and is remapped by `heap_grow`. **To validate** that it never blocks while the grid runs. Until then the runtime never sends it |
| 40 | `exit` | code | — | `process.exit(code)` from any thread. Flush acknowledged output, then `exit(code)` at once. The shim does not wait for the grid (`main` may still run); process exit tears it down, as for op 41. A normal return from `main` sends no request: it uses `exit_code` and `stop` in the control block |
| 41 | `panic_report` | — | — | Payload is the `ErrorRecord` copy in `inline_out`. The host formats it with the image's site table, writes it to stderr, sends it to the panic sink if one is registered, then `exit(101)` |
| 42 | `panic_sink` | handle | 0 | Registers a connected socket handle and a frame prefix (in `inline_out`, ≤ 256 B) as the panic sink. On a panic the host sends one frame: `u32` LE length, the prefix, then the report as a CBOR text string. Best effort: 100 ms send timeout, errors ignored. A later call replaces the sink; handle 0 removes it |
| 50 | `listen_unix` | path in inline, mode, backlog | handle | Creates the parent directory with `mode` if missing (it must then be owned by the process uid with exactly `mode`, else EACCES), unlinks a stale socket, then `socket(AF_UNIX, SOCK_STREAM \| SOCK_CLOEXEC)`, `bind`, `listen` |
| 51 | `accept` | handle | connection handle | `accept4(SOCK_CLOEXEC)`. Runs on a worker |
| 52 | `recv` | handle, dev_ptr, len | bytes (0 = end of stream) | Into staging or `inline_in`, then HtoD. Runs on a worker |
| 53 | `send` | handle, dev_ptr, len | bytes sent | As `write`, with `MSG_NOSIGNAL` |
| 54 | `peer_uid` | handle | uid | `getsockopt(SO_PEERCRED)` |
| 55 | `poll` | count (≤ 64), timeout_ms; handles as `u32[count]` in inline | bit mask of ready handles (0 on timeout) | `poll(2)` for readable or acceptable. Runs on a worker. The runtime polls before `accept`/`recv` so a worker is never held longer than the timeout |
| 60 | `ptx_compile` | src_ptr, src_len, out_ptr, out_cap, log_ptr, log_cap; options NUL-separated in inline | cubin bytes | nvPTXCompiler (`libnvptxcompiler_static`) on a worker, the one library call of the shim. Compile error: errno EINVAL, `value` = log bytes written to `log_ptr`. `out_cap` too small: errno ENOSPC, `value` = needed size. Other library failures: errno EIO, with the `nvPTXCompileResult` code in the log |

`errno` values are Linux's. The device turns them into the same `io.Error` variants as the CPU
std. Ops 24, 42, 50–55, and 60 are reached only through an internal std module
(`std.rt.host`, below; not in `LANG.md`, not a stable API in v1). Socket handles are host fds in
the shim's fd table.

**`Stat` wire layout** (op 7, 88 B, little-endian, the same on every host): 0 `dev: u64`, 8
`ino: u64`, 16 `mode: u32`, 20 `nlink: u32`, 24 `uid: u32`, 28 `gid: u32`, 32 `rdev: u64`, 40
`size: i64`, 48 `blksize: u64`, 56 `blocks: u64`, 64 `atime_ns: i64`, 72 `mtime_ns: i64`, 80
`ctime_ns: i64` (times as `sec × 10^9 + nsec`). The device std converts it to the `std.fs` type.

**`getdents` records** (op 9): Linux `linux_dirent64` as the kernel writes it: 0 `d_ino: u64`, 8
`d_off: i64`, 16 `d_reclen: u16`, 18 `d_type: u8`, 19 `d_name` (NUL-terminated, record padded to 8
B). This layout is the same on x86-64 and AArch64, and the CPU std parses the same records.

## Internal module `std.rt.host`

Signatures (Stave; `vec.Vec` is `std.vec.Vec`). On `gpu-sm86` they call the services above through
`runtime-gpu/host.stv`. On CPU targets `runtime/host.stv` makes every function return `Result.Err`
with the `io.Error` for ENOSYS (`PtxError.Io(..)` for `ptx_compile`); `std.rt.host` is
`gpu-sm86`-only in v1.

```text
pub struct Handle { fd: i32 }                 // host fd in the shim's fd table; drop closes it (op 6)
pub struct MemInfo { pub free: u64, pub total: u64 }
pub enum PtxError { Invalid(vec.Vec<u8>), Io(io.Error) }   // Invalid: the nvPTXCompiler log (EINVAL)

pub fn listen_unix(path: &str, dir_mode: u32, backlog: u32) -> Result<Handle, io.Error>   // op 50
pub fn accept(h: &Handle) -> Result<Handle, io.Error>                                     // op 51
pub fn recv(h: &Handle, buf: &mut [u8]) -> Result<usize, io.Error>                        // op 52; Ok(0) = end of stream
pub fn send(h: &Handle, buf: &[u8]) -> Result<usize, io.Error>                            // op 53
pub fn close(h: Handle) -> Result<(), io.Error>                                           // op 6
pub fn peer_uid(h: &Handle) -> Result<u32, io.Error>                                      // op 54
pub fn poll(handles: &[Handle], timeout_ms: u32) -> Result<u64, io.Error>                 // op 55; bit i = handles[i] ready; > 64 handles: EINVAL
pub fn mem_info() -> Result<MemInfo, io.Error>                                            // op 24
pub fn panic_sink(h: &Handle, prefix: &[u8]) -> Result<(), io.Error>                      // op 42; prefix > 256 B: EINVAL; replaces an earlier sink
pub fn panic_sink_clear() -> Result<(), io.Error>                                         // op 42 with handle 0
pub fn ptx_compile(ptx: &[u8], options: &[u8]) -> Result<vec.Vec<u8>, PtxError>           // op 60; options NUL-separated
```

`ptx_compile` handles ENOSPC itself: it grows the output buffer to the reported size and retries
once. Closing the sink's handle also clears the sink in the shim.

## Shim and image

- **Install**: release packaging puts the two prebuilt shims at
  `<stave install>/lib/stave/shim-base-<build id>` and `<stave install>/lib/stave/shim-ptx-<build id>`.
  `<stave install>` is the parent of the directory that holds the running `stave` executable, and
  `<build id>` is that compiler's build id, so a compiler only finds the shims built with it.
  `STAVE_SHIM_DIR` overrides the directory (tests). A missing shim is an internal compiler error
  (exit 3) naming the path.
- **Embedding**: `stave build` writes a byte copy of the selected shim and appends the GPU image as
  the section `.stave.gpu` (`SHT_PROGBITS`, not `SHF_ALLOC`, so it is never loaded into memory),
  rewriting the section header table at the end of the file. The output is mode 0755 and
  byte-identical for identical inputs. The section name is the same `.stave.gpu` as the kernel
  image of CPU programs ([gpu-lang.md](gpu-lang.md)). Unlike that image (`SHF_ALLOC`, read-only
  `PT_LOAD`), it is not loaded: the shim is prebuilt, so
  appending a section is simpler than adding a program header, and the shim reads it once.
- **Finding the image**: the shim opens `/proc/self/exe`, reads the section headers, and reads
  `.stave.gpu`. A missing or truncated section, an unknown `shim_abi_version`, or a `PtxCompile`
  image in the base shim exits 125 with `stave: cannot start GPU program: <reason>`.

## Shim lifecycle

0. Read `.stave.gpu` from `/proc/self/exe` ("Shim and image"), check `shim_abi_version` and
   `shim_variant`, and parse `STAVE_GPU_HEAP_MAX` and `STAVE_GPU_TIMINGS` from the environment. An
   invalid `STAVE_GPU_HEAP_MAX` exits 125 (`invalid STAVE_GPU_HEAP_MAX "<value>": expected a size
   like "40GiB" or a percentage like "50%"`). The variables stay visible to the program.
1. Probe the driver (same steps and texts as [gpu-ops.md](gpu-ops.md) "Errors"). On failure: exit 125.
2. Retain the primary context and make it current (`cuCtxSetCurrent`). **The start-latency window
   (`start→main`, [gpu-program.md](gpu-program.md)) starts when `cuCtxSetCurrent` returns.** Set `CU_LIMIT_STACK_SIZE = hw_stack_bytes`. The limit is
   never changed again.
3. Resolve `heap_max`: `STAVE_GPU_HEAP_MAX` if set, else the image's `RuntimeConfig.heap_max` (the
   stage1 compiler daemon sets `50%`). A percentage is of `cuMemGetInfo` free
   memory, read now. A result below 64 MiB exits 125. Reserve the heap range and map
   `min(heap_initial, heap_max)`.
4. Load the cubin and get `__stave_rt` (`cuModuleGetFunction`). If `kernel_shared_bytes` > 48 KiB,
   set `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. Compute the grid with
   `cuOccupancyMaxActiveBlocksPerMultiprocessor` (block R, dynamic shared `kernel_shared_bytes`)
   × SM count. Allocate request slots (mapped), reply slots and the control block (device), and
   staging (pinned). Write the control block's host fields. Create the copy streams and the
   helper context (`cuCtxCreate`, runs no kernels).
5. Install the SIGINT/SIGTERM handlers. A handler only sets a host atomic. The poller sees it,
   writes `stop` with `cuStreamWriteValue32`, restores the default disposition, and re-raises.
6. Launch `__stave_rt(ctl)` cooperatively with that grid, start the poller and workers, and serve
   requests. **The start-latency window ends at `main`'s first instruction**: `main_t0` in the
   control block, converted with `clock_offset`.
7. Normal end: the grid ends after `main` returned (`exit_code`, then `stop`). The shim copies the
   control block back, flushes output, prints run-time timings if enabled, and exits with
   `exit_code`.
8. On `exit` (op 40), `panic_report` (op 41), or a stream error: report and exit without waiting
   for the grid. A device fault prints `device fault: <name>` and exits 101.

Rules while the grid runs:
- **No blocking free.** `cuMemFree`, `cuMemFreeHost`, and `cuMemHostUnregister` block until the
  resident grid exits. The shim frees with `cuMemFreeAsync`, or unregisters and releases in
  the helper context. Whether `cuMemUnmap`/`cuMemRelease` block is unverified (op 31).
- **Idle**: in `"adaptive"` poll mode the poller busy-polls while requests arrive, sleeps 50 µs
  after 10 ms without a request, and 1 ms after 1 s. An idle grid still holds every SM.
