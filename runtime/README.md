# runtime (CPU)

The libc-free CPU runtime, written in Stave. Package
`runtime`; std depends on it. It serves the CPU targets only: for `gpu-sm86` the name `runtime`
is bound to the `runtime-gpu/` package instead (contracts/intrinsics.md "PTX"), with the same
module paths and `pub` signatures. The runtime does not depend on std, so it sees no prelude and uses no
Option, Result, or std traits (only the built-in type names and `size_of`/`align_of`).

| Module | File | Role |
|---|---|---|
| `runtime.start` | `start.stv` | `entry(sp) -> never`: argc/argv/envp/auxv, signal handler, main's stack + guard, run main, `exit_group` |
| `runtime.os` | `os.stv` | arch-neutral wrappers of the `syscalls.list` syscalls (raw result, `-errno` on failure); the only importer of `arch.sys` |
| `runtime_x86_64.sys`, `runtime_aarch64.sys` | `{x86_64,aarch64}/sys.stv` | syscall numbers, `syscall0`–`syscall6` intrinsics, ucontext PC offset (one arch package per build, below) |
| `runtime.alloc` | `alloc.stv` | global allocator over mmap/munmap/mremap |
| `runtime.panic` | `panic.stv` | panic report, exit 101, SIGSEGV guard-page handler |
| `runtime.state` | `state.stv` | the process-wide state block, one `static mut`; allocator lock, thread registry; the per-thread `Block` behind the thread pointer (`block()`: thread id, user site, error trace) |
| `runtime.errtrace` | `errtrace.stv` | the calling thread's error trace in its `state.Block`: `restart`, `push` (the compiler's hooks of `Result.Err(..)` and `?`), `clear`, `save` / `restore` (a `Trace` copy), all trap-free |
| `runtime.atomic` | `atomic.stv` | CPU atomic intrinsics (load/xchg/add/cas u32/u64) and a three-state futex lock |
| `runtime.thread` | `thread.stv` | threads: mapping = guard + stack + altstack + record, clone3 (clone if ENOSYS), futex join on CLEARTID, ids, registry; `scope_spawn`/`scope_join` for `scope { spawn { } }`; `start`/`join`/`cpu_count` for std.thread |
| `runtime.config` | `config.stv` | per-binary settings (`[run] stack`) |
| `runtime.cpu` | `cpu.stv` | feature word (contracts/intrinsics.md "CPU feature detection"): `detect()` probes (through `arch.cpu`) and closes under implication; `init()` checks it against the target baseline (`cpu_baseline()`; UnsupportedCpu), applies `STAVE_CPU` and `STAVE_DISPATCH_TRACE`, and stores it once in `WORD`, a `static mut` (start code, through the `cpu_init` intrinsic, only in programs that read the word, a dispatch body or `features()`, or whose baseline is above the arch's lowest); dispatch bodies load `WORD` directly, `features()` returns it, `trace(line)` writes a dispatch trace line |
| `runtime_x86_64.cpu`, `runtime_aarch64.cpu` | `{x86_64,aarch64}/cpu.stv` | `probe(hwcap, hwcap2)`: raw feature bits from CPUID + XCR0 (x86-64) or AT_HWCAP/AT_HWCAP2 (AArch64); imported only by `runtime.cpu` |

`syscalls.list`: the syscall set, one per line. Line i + 1 is call id i in `os.stv` (`SYS_*`) and the
index into `NR` in each `sys.stv` (a test compares them). A row with an arch directory after the name
(`arch_prctl x86_64`, last) is that arch's own: no call id, no `NR` entry; that arch's `sys.stv`
names its `NR_*` constant in the call.

## Per-arch selection: per-target dependencies

A built-in package may have a `stave.toml`; the loader reads it as for any package. Its modules
stay at the directory root (`os.stv` is `runtime.os`, no `src/`), and a subdirectory with its own
`stave.toml` is a nested package, not part of `runtime`'s modules. `runtime/stave.toml`:

```toml
name = "runtime"
kind = "lib"
[target.x86_64-linux.dependencies]      # every x86-64 level
arch = { path = "x86_64" }
[target.aarch64-linux.dependencies]     # every AArch64 target (+sve, +sve2 too)
arch = { path = "aarch64" }
```

`runtime/x86_64/stave.toml` names the package `runtime_x86_64`, `runtime/aarch64/stave.toml`
`runtime_aarch64`; each has two modules, `sys.stv` and `cpu.stv`. Dependencies of a built-in package are
built-in too (same layout; intrinsics allowed, E0805). `os.stv` does `import arch.sys;`, so a
build contains exactly one arch's syscall layer and every call is static: no `dispatch`, no
`needs`, no `cfg`. A `gpu-sm86` build uses `runtime-gpu/` instead of this package.

## Compiler contract (backends)

`contracts/intrinsics.md` is the source of truth for the intrinsics, stubs, and table layouts
below; this section restates what the runtime expects.

**Entry.** `_start` calls `runtime.start.entry(sp: *const usize) -> never` (declared
`pub unsafe(..) fn`; only `_start` calls it).

**Traps.** Every failing check calls

```
runtime.panic.trap(kind: u32, site: u32) -> never
runtime.panic.trap_msg(kind: u32, site: u32, msg: &[u8]) -> never  // + one line `msg` (std.debug.assert)
runtime.panic.trap_err(kind: u32, site: u32) -> never  // + the thread's error trace (std's Result.unwrap)
runtime.panic.requires_failed(clause: &str) -> never           // Contract at the user site, + the clause
runtime.panic.ensures_failed(site: u32, clause: &str) -> never  // Contract at the return, + the clause
```

`kind`: `KIND_BOUNDS 0, KIND_OVERFLOW 1, KIND_DIV_BY_ZERO 2, KIND_ASSERT 3, KIND_OUT_OF_MEMORY 4,
KIND_STACK_OVERFLOW 5, KIND_UNWRAP 6, KIND_UNSUPPORTED_CPU 7, KIND_DEADLOCK 8, KIND_CONTRACT 9` (panic kind names; shift amount >= width
is `Overflow`; `UnsupportedCpu` only from `runtime.cpu.init`, at `SITE_NONE`).
`site`: index into the compiler-emitted **site table**, `SITE_NONE = 0xFFFF_FFFF` for unknown.
Output: `panic: <Kind> at <file>:<line>:<col> in <fn> (thread <n>)` on fd 2, then `exit_group(101)`.
A panic during a panic exits 101 at once.

**User site**: before a call from a non-std/runtime package into a
std fn that may allocate (OutOfMemory) or panics for its caller (`std.debug.assert`,
`Option/Result.unwrap`, `Vec.get_ref/get_mut` Bounds), store the call's site id with
`runtime.panic.set_user_site(site: u32)` (or the same store inline: the `user_site` field of
the calling thread's `runtime.state.Block`). std reports `runtime.panic.user_site()`. In a program where some fn checks
a `requires` clause, user code also stores the site before a direct call of such a fn or of a
`dispatch fn`, and before every call through a fn value: `requires_failed` reports there.

**Error trace**: in fns of user packages, `runtime.errtrace.push(site)` where `?` on a `Result`
takes the error path (before the scope drops and `ensures` checks; site: the `?` token) and
`runtime.errtrace.restart(site)` after each evaluated `Result.Err(..)` constructor (site: the
`Err` token). Neither is a call-graph edge (no effect).

**Signals.** The compiler emits a C-ABI stub `__stave_signal(sig: int, info: *siginfo, uctx: *void)`
that calls `runtime.panic.on_signal(sig: usize, info: usize, uctx: usize) -> never`; its address is
the `signal_handler_addr()` intrinsic. A SIGSEGV with `si_addr` in `[guard_lo, guard_hi)` reports
`StackOverflow` at the faulting fn's name token (PC table); any other SIGSEGV/SIGBUS re-raises with
the default action. Frames larger than `runtime.config.GUARD` (64 KiB) must be probed page by page
by the prologue, or an overflow can jump the guard.

**Tables** (read-only data; little-endian; addresses from `site_table()`, `pc_table()`, and
`stack_table()`):

- site table: `u64 count`, then rows of 32 bytes: `+0 file ptr u64, +8 file_len u32, +12 line u32,
  +16 fn ptr u64, +24 fn_len u32, +28 col u32` (line/col 1-based, col in bytes; fn = checked-IR path);
- PC table: `u64 count`, then rows of 16 bytes sorted by `pc_lo`: `+0 pc_lo u64, +8 site u32, +12 pad`
  (site = the fn's name token).
- stack table: `+0 default u64` (`[run] stack`; 0: `RUN_STACK`), `+8 main u64` (0: no bound),
  `+16 count u64`, then rows of 16 bytes: `+0 code u64` (a fn value's code address), `+8 bytes
  u64` (0: no bound). `runtime.config.main_stack` and `thread_stack(code)` read it.

**Main.** `main_addr()` is the address of the compiler-emitted C-ABI stub
`__stave_main(arg: usize) -> usize` (calls `main`, returns its status: i32 sign-extended, 0 for a
unit main). `call_on_stack(f, arg, top)` runs it on main's own stack.

**Allocator.** `runtime.alloc`:

```
pub fn alloc(size: usize, alignment: usize) -> *mut u8                 // never null; OOM panics
pub unsafe(..) fn free(p: *mut u8, size: usize, alignment: usize)
pub unsafe(..) fn realloc(p: *mut u8, old_size: usize, alignment: usize, new_size: usize) -> *mut u8
pub fn try_alloc(size: usize, alignment: usize) -> *mut u8             // null instead of OOM
pub unsafe(..) fn try_realloc(p: *mut u8, old_size: usize, alignment: usize, new_size: usize) -> *mut u8
```

Sized deallocation (no headers); `alignment` a power of two; size 0 → a dangling aligned pointer.
`try_alloc` / `try_realloc` (std's `try_` methods) return null where `alloc` / `realloc` panic
`OutOfMemory`; a failed `try_realloc` leaves its block unchanged.
`alloc` and `realloc` are derived: `try_alloc` / `try_realloc` plus the `OutOfMemory` trap.
Planned: `allocator my.alloc;` provides only the fallible forms and `free` (`try_alloc`,
`try_realloc`, `free`, the signatures above) and the compiler binds `runtime.alloc`'s calls of
them to its module; `alloc` and `realloc` stay derived. Today the declaration is only parsed and
checked (at most one per binary, E0104), and release builds keep its module's `alloc`, `free`,
and `realloc` as roots; std's calls still go to `runtime.alloc`.

## Assumptions (not yet specified)

- **State block**: `runtime.state.STATE` is a zero-initialized `static mut` (.bss). Every module
  reaches it through `state.get() -> *mut State`; each dereference is `unsafe` and follows the
  field rules in the `state.stv` header. Its initializer uses a struct literal and `[0; NCLASS]`,
  both const-exprs.
- **Stack sizes**: main's and each thread's stack size come from the compiler-emitted stack
  table (`stack_table()`, contracts/intrinsics.md): the computed bound, else `[run] stack`;
  `runtime.config.RUN_STACK` (8 MiB) is the default where the manifest sets none or there is no
  table (the IR interpreter).
- **Threads**: main runs on the start thread, switched onto its own mmap'd stack by
  `call_on_stack` (id 0, not in the registry). Other threads: runtime/thread.stv header. The
  allocator's shared lists are guarded by one futex lock (`state.alloc_lock`); `panicking` is
  process-wide (a second concurrent panic waits for the first report's exit). Each thread's id,
  user site, and error trace are in its own `state.Block`, found through the thread pointer
  (contracts/intrinsics.md "Thread pointer"): main's in `STATE`, set first by `start.entry`; a
  thread's in its record page, set by `CLONE_SETTLS`. If a thread cannot be created (no memory, the
  guard pages cannot be protected, EAGAIN) the program panics with "cannot create thread"; only in the IR interpreter (clone is ENOSYS and the image has no stack table)
  the body is queued and runs on the joining thread at the join (`join`, `scope_join`), oldest
  first.
- **A5** a fn value is two words, (code, environment) (contracts/checked-ir.md "Closures");
  runtime.thread stores and rebuilds one through `*const [usize; 2]`.
- **Layouts used by pointer casts**: `&str`/`&[u8]` = (data, len) words; kernel structs are
  written as `u64` arrays (sigaction 4 words, stack_t 3 words, sigset 1 word, timespec `[i64; 2]`).
- Imports inside `runtime` are package-qualified (`import runtime.os;`), matching the module
  rules in crates/stave-sema `package.rs` / `modules.rs`; the arch package is reached through its
  dependency name (`import arch.sys;`).
