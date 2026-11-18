# Contract: compiler intrinsics (v0.5)

Stave has no inline assembly. The few machine operations a libc-free runtime needs are **compiler
intrinsics**, declared with the existing `extern` syntax and a reserved ABI string. The grammar does not change:

```
extern "stave-intrinsic" {
    fn syscall3(nr: usize, a1: usize, a2: usize, a3: usize) -> isize;
    fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32);
}
```

Rules:
- `extern "stave-intrinsic"` is allowed only in the `std`, `runtime`, and `runtime-gpu` packages
  shipped with the compiler. Anywhere else, or with a name or signature that is not in the
  table, it is **E0805**.
- Calls are `unsafe` (as for every `extern` import). Each intrinsic lowers to the listed
  instruction sequence inline. No call, no symbol, no relocation.
- An intrinsic whose arch column does not match the target is not compiled, like any function
  needing another arch's features. Calling it on the wrong arch is E0704.
- Both compilers (stage0 and stage1) implement exactly this table. The checked IR
  prints an intrinsic call as `intrinsic.<name>` (checked-ir.md).

| Intrinsic | Arch | Lowering |
|---|---|---|
| `syscall0(nr) … syscall6(nr, a1..a6)`, all `usize`, `-> isize` | x86-64 | `rax=nr, rdi rsi rdx r10 r8 r9 = a1..a6; syscall`; clobbers `rcx r11`; result `rax` |
| same | AArch64 | `x8=nr, x0..x5 = a1..a6; svc #0`; result `x0` |
| `cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32)` | x86-64 | `eax=leaf, ecx=subleaf; cpuid` → `(eax, ebx, ecx, edx)` |
| `xgetbv(xcr: u32) -> u64` | x86-64 | `ecx=xcr; xgetbv` → `edx:eax`. Callers check `cpuid` OSXSAVE first |
| `ccall0(f) … ccall12(f, a1..a12)`, all `usize`, `-> usize` | x86-64 | System V indirect call of the C function at address `f`: `a1..a6` in `rdi rsi rdx rcx r8 r9`, `a7..a12` on the stack (right to left, 8 bytes each, `rsp` 16-byte aligned at the `call`, padding added for an odd count), `eax=0` (no vector args, so variadic callees work), `call` through a scratch register; clobbers every caller-saved register (`rax rcx rdx rsi rdi r8–r11`, all vector registers, flags). Result is `rax` (x86-64) / `x0` (AArch64); for a C function returning `int` (e.g. `CUresult`) only the low 32 bits are defined |
| same | AArch64 | AAPCS64 indirect call: `a1..a8` in `x0..x7`, `a9..a12` on the stack (8 bytes each, `sp` 16-byte aligned), `blr` through `x16`; clobbers `x0–x18 x30`, `v0–v7 v16–v31` and the upper halves of `v8–v15` (all SVE `z`/`p` state), flags; result `x0` (low 32 bits only for an `int` result, as on x86-64) |

AArch64 feature detection needs no intrinsic: the runtime reads `AT_HWCAP`/`AT_HWCAP2` from the
auxiliary vector on the initial stack.

**CPU feature detection**: `runtime.cpu.detect() -> u64` returns the **feature word**: bit i is
feature i of the compiler's closed feature list (`crates/stave-target/src/features.rs`, the order of LANG.md §4 Features:
`sse42` = bit 0 … `sve2` = bit 22, `cuda` = bit 23, never set by the CPU probe). The constants
`runtime.cpu.SSE42 …` and the `std.cpu` names are these masks; a test checks them against the
compiler. The probe (`arch.cpu.probe`, per arch) sets a feature when its row below holds; then
**closure**: a feature stays set only if every feature it implies (transitively, per
features.rs) is set too, so testing one mask is sound. `runtime.cpu.init()` stores the word once
in the `static mut` `runtime.cpu.WORD` (before main, before other threads); a dispatch body
loads it directly (one load, no call), `runtime.cpu.features()` returns it.
`runtime.start.entry` calls `init` through `cpu_init()` (below) right after it stores the auxv
values, and the program has `init` exactly when it reads the word (a `dispatch fn` body that tests
an arm, `std.cpu.features`) or its target baseline is above its arch's lowest (`x86_64-linux-v3`,
`-v4`, `aarch64-linux+sve`, `+sve2`; checked-ir.md "Instance root set"). The IR interpreter's
`cpu_init` does nothing, so there the word is 0 and every dispatch takes its `_` arm (unless the
baseline prunes it).

`init`, in order:
1. **Baseline**: B = `cpu_baseline()`, the `--target` baseline as a closed feature word (a
   constant: lowering replaces the call by it; it never reaches the IR). If the detected word
   lacks a feature of B, the program stops before main: `panic: UnsupportedCpu at
   <unknown>:0:0 in <unknown> (thread 0)`, then one line `the CPU lacks features of the build
   target: <names>`, exit 101 (runtime.panic `trap_msg`, kind 7, `SITE_NONE`).
2. **Mask**: `STAVE_CPU` in the environment, read once: empty or unset changes nothing;
   otherwise comma-separated items `-<name>`, no spaces. An item naming a feature of B does
   nothing (the program's code assumes it); any other removes feature `<name>` and every feature
   implying it (CLOSURE rows containing it). The removed set R is closed upward and disjoint
   from B, so `detected & ~R` stays closed downward. A name of another arch's feature is
   accepted and changes nothing. An unknown name prints the one line `stave: STAVE_CPU: unknown feature
   '<name>'` (name cut at 64 bytes), any other malformed value `stave: STAVE_CPU: bad syntax`,
   to the diagnostic channel; exit 101 before main. Features can only be removed.
3. **Trace bit**: bit 63, never a feature, is set when `STAVE_DISPATCH_TRACE` is set to a value
   other than empty or `0`. Each block of a dispatch body (checked-ir.md "Instance root set")
   that calls an arm target first tests it on the word already loaded (one branch, no extra
   load) and, when set, calls `runtime.cpu.trace(line: &str)` with the literal
   `dispatch: <dispatch fn path> -> <arm target path>\n`, written with one `write` (lines of
   threads do not interleave). A call resolved at compile time (pruned to one arm, or the
   call-site redirect) runs no dispatch body and prints nothing. `runtime.cpu.features()`
   returns the word with the bit; `std.cpu.features()` clears it.

x86-64 (Intel SDM Vol. 2A, CPUID: Table 3-10 for leaf 01H ECX, Table 3-8 for leaf 07H; the same
positions as Linux `arch/x86/include/asm/cpufeatures.h` words 4, 9, 12, 16, 18). Leaf 07H is
read only if leaf 0 EAX ≥ 7; sub-leaf 1 only if leaf 07H.0 EAX ≥ 1.

| Feature | CPUID bit | | Feature | CPUID bit |
|---|---|---|---|---|
| `sse42` | 01H ECX[20] | | `avx512dq` | 07H.0 EBX[17] |
| `avx` | 01H ECX[28] | | `avx512cd` | 07H.0 EBX[28] |
| `fma` | 01H ECX[12] | | `avx512bw` | 07H.0 EBX[30] |
| `f16c` | 01H ECX[29] | | `avx512vl` | 07H.0 EBX[31] |
| `avx2` | 07H.0 EBX[5] | | `avx512vbmi` | 07H.0 ECX[1] |
| `bmi2` | 07H.0 EBX[8] | | `avx512vbmi2` | 07H.0 ECX[6] |
| `avx512f` | 07H.0 EBX[16] | | `avx512bitalg` | 07H.0 ECX[12] |
| `avx512fp16` | 07H.0 EDX[23] | | `avx512vpopcntdq` | 07H.0 ECX[14] |
| `avx512bf16` | 07H.1 EAX[5] | | | |

OS state: `xgetbv(0)` (XCR0) is read only if 01H ECX[26] (XSAVE) and ECX[27] (OSXSAVE) are set,
else XCR0 counts as 0. AVX-class features (`avx avx2 fma f16c` and every `avx512*`) need
`XCR0 & 0x6 == 0x6` (SSE, YMM state); AVX-512-class features (`avx512*`) also need
`XCR0 & 0xe6 == 0xe6` (opmask, ZMM_Hi256, Hi16_ZMM). `sse42` and `bmi2` need no OS state.

AArch64 (Linux `arch/arm64/include/uapi/asm/hwcap.h`):

| Feature | Condition |
|---|---|
| `neon` | `AT_HWCAP` bit 1 (`HWCAP_ASIMD`) |
| `fp16` | `AT_HWCAP` bits 9 and 10 (`HWCAP_FPHP`, `HWCAP_ASIMDHP`) |
| `dotprod` | `AT_HWCAP` bit 20 (`HWCAP_ASIMDDP`) |
| `bf16` | `AT_HWCAP2` bit 14 (`HWCAP2_BF16`) |
| `sve` | `AT_HWCAP` bit 22 (`HWCAP_SVE`) |
| `sve2` | `AT_HWCAP2` bit 1 (`HWCAP2_SVE2`) |

**`ccallN`** calls C functions whose address comes from `dlsym` (e.g. the
CUDA driver table of `std.gpu`). Only integer-class arguments and results: pointers and integers,
passed as `usize`. No floats, vectors, or structs by value; a C function that needs them cannot be
called this way. The callee defines only the low bits of a narrower C result (`int`, `bool`):
truncate with `as%`. Callers pass sign- or zero-extended values for narrower C parameters (`x as%
usize`). Calls are allowed only from threads that glibc knows (the main thread, or threads created
with `pthread_create`), since the callee may use glibc TLS. Typed `extern "C" fn`
pointer types are a later addition.

**Program entry**: the compiler emits `_start`, which passes the initial stack pointer (pointing at
`argc`) to the runtime function `runtime.start.entry(sp: *const usize)`. That function reads argv,
envp, and auxv, runs `main`, and ends with the `exit_group` system call. If it ever returns,
`_start` executes `ud2` (x86-64) or `brk #0` (AArch64). The function name is fixed by this
contract. It is not an attribute. The same plain `_start` (no `__libc_start_main`) is used by
static and dynamic binaries alike: in a dynamic binary (GPU use)
`ld.so` has already run libc's initializers and set up TLS before it jumps to `_start`.
In dynamic ELF mode `_start` is the same: ld.so's `rdx` (`rtld_fini`) is ignored
and the program still ends with `exit_group`; no crt1, no `__libc_start_main`.
This is not yet validated on hardware; if it fails, the dynamic start path falls back
to calling `__libc_start_main`.

**Thread start** (dynamic binaries, where threads come from `pthread_create`): the compiler emits a C-ABI stub `__stave_thread_start(arg: void*) ->
void*` that calls `runtime.thread.entry(arg: usize) -> usize` and returns its result.

| Intrinsic | Arch | Lowering |
|---|---|---|
| `thread_start_addr() -> usize` | x86-64 | `lea rax, [rip + __stave_thread_start]` (AArch64: `adrp`/`add`): the stub's address, passed to `pthread_create` through `ccall4` |

**CPU threads and atomics**. Static binaries create threads with raw `clone3`/`clone`, whose child
resumes after the system call on a new stack, so the child path is part of the intrinsic. The
atomics are sequentially consistent; `p` is the address (naturally aligned) as `usize`.

| Intrinsic | Arch | Lowering |
|---|---|---|
| `thread_clone(nr, a1, a2, a3, a4, a5, arg)`, all `usize`, `-> isize` | x86-64 | `rax=nr, rdi rsi rdx r10 r8 = a1..a5, r9 = arg; syscall; test rax, rax; jnz 1f`; child: `mov rdi, r9; xor ebp, ebp; and rsp, -16; call __stave_thread_start; mov edi, eax; mov eax, 60 (exit); syscall; ud2`; `1:` result `rax` |
| same | AArch64 | `x8=nr, x0..x4 = a1..a5, x5 = arg; svc #0; cbnz x0, 1f`; child: `mov x16, sp; and x16, x16, #-16; mov sp, x16; mov x0, x5; mov x29, xzr; mov x30, xzr; bl __stave_thread_start; mov x8, #93 (exit); svc #0; brk #0`; `1:` result `x0` |
| `atomic_load_u32/u64(p) -> u32/u64` | x86-64 / AArch64 | `mov` / `ldar` |
| `atomic_xchg_u32/u64(p, v) -> old` | x86-64 / AArch64 | `xchg` / `ldaxr`+`stlxr` loop |
| `atomic_add_u32/u64(p, v) -> old` (wrapping) | x86-64 / AArch64 | `lock xadd` / `ldaxr`+`add`+`stlxr` loop |
| `atomic_cas_u32/u64(p, expected, new) -> old` | x86-64 / AArch64 | `lock cmpxchg` / `ldaxr`+`cmp`+`b.ne`+`stlxr` loop (`clrex` on failure) |

`nr` is the arch's `clone3` (`a1` = `struct clone_args`, `a2` = its size) or `clone` number
(runtime `sys.stv` passes the arch's argument order); the `exit` system call of the child path is
in runtime/syscalls.list. `__stave_thread_start` (above) is emitted for static binaries too. The IR
interpreter returns `-ENOSYS` for `thread_clone` (runtime.thread then runs the body inline).
No IR calls `runtime.thread.entry`, so a lowered `thread_clone` or `thread_start_addr` requests
it, in debug and release alike (checked-ir.md "Instance root set"); a program that lowers neither
has no `runtime.thread.entry` and no `__stave_thread_start`.

**Runtime start and traps** (CPU). The compiler emits two
C-ABI stubs and three read-only tables per binary, and these intrinsics give their addresses or
switch stacks. Each lowers inline (no call except the ones of `call_on_stack` and `cpu_init`).

| Intrinsic | Arch | Lowering |
|---|---|---|
| `main_addr() -> usize` | x86-64 | `lea rax, [rip + __stave_main]`: address of the stub `__stave_main(arg: usize) -> usize` (C ABI; `arg` ignored), which calls the program's `main` and returns its exit status: `main`'s `i32` sign-extended to 64 bits, 0 for a unit `main` |
| same | AArch64 | `adrp x0, __stave_main; add x0, x0, :lo12:__stave_main` |
| `cpu_init()` | x86-64 / AArch64 | `call` / `bl runtime.cpu.init` (Stave convention, no arguments) if the program has that fn, else no code |
| `cpu_baseline() -> u64` | x86-64 / AArch64 | none: lowering replaces the call by the constant `--target` baseline feature word ("CPU feature detection") |
| `call_on_stack(f: usize, arg: usize, top: usize) -> usize` | x86-64 | `push rbx; mov rbx, rsp; mov rsp, top` (`top` 16-byte aligned, the caller's obligation); `mov rdi, arg; call f` (System V); `mov rsp, rbx; pop rbx`; result `rax`. Clobbers the caller-saved registers, like `ccall1` |
| same | AArch64 | `stp x19, x30, [sp, #-16]!; mov x19, sp; mov sp, top; mov x0, arg; blr f` (AAPCS64, `f` in a scratch register, `x16`); `mov sp, x19; ldp x19, x30, [sp], #16`; result `x0` |
| `signal_handler_addr() -> usize` | x86-64 / AArch64 | address (`lea` / `adrp`+`add`) of the stub `__stave_signal(sig: int, info: *siginfo_t, uctx: *ucontext_t)` (C ABI, SA_SIGINFO handler), which zero-extends `sig` and calls `runtime.panic.on_signal(sig: usize, info: usize, uctx: usize) -> never` |
| `site_table() -> usize` | x86-64 / AArch64 | address of `__stave_site_table` (`.rodata`, 8-byte aligned), layout below |
| `pc_table() -> usize` | x86-64 / AArch64 | address of `__stave_pc_table` (`.rodata`, 8-byte aligned), layout below |
| `stack_table() -> usize` | x86-64 / AArch64 | address of `__stave_stack_table` (`.rodata`, 8-byte aligned), layout below; the IR interpreter returns 0 (no table) |

Table layouts (little-endian; runtime/README.md restates them, this contract is the source of
truth):

- **site table** `__stave_site_table`: `u64 count`, then `count` rows of 32 bytes, one per trap
  site (`runtime.panic.trap(kind, site)`'s `site` is a row index; `0xFFFF_FFFF` = unknown):
  `+0 file: u64` address of the UTF-8 file path (the path diagnostics use; for `std`, `runtime`, and the
  packages they depend on `<package>/<path in the package>`, e.g. `std/vec.stv`), `+8 file_len: u32`,
  `+12 line: u32` (1-based), `+16 fn: u64` address of the UTF-8 checked-IR path of the enclosing
  fn, `+24 fn_len: u32`, `+28 col: u32` (1-based, in bytes). Rows are in canonical order (by fn
  instance id, then source offset), so the table is identical in stage0 and stage1.
- **PC table** `__stave_pc_table`: `u64 count`, then `count` rows of 16 bytes sorted by `pc_lo`
  ascending, one per emitted fn: `+0 pc_lo: u64` (address of the fn's first instruction),
  `+8 site: u32` (site-table row of the fn's name token), `+12 pad: u32 = 0`. A PC belongs to the
  last row with `pc_lo <= pc`.
- **stack table** `__stave_stack_table`: `+0 default: u64` (`[run] stack` in bytes; 0: not set,
  the runtime's 8 MiB default), `+8 main: u64` (bytes of main's stack; 0: no bound), `+16 count:
  u64`, then `count` rows of 16 bytes: `+0 code: u64` (the code address a fn value holds, as
  `fn.make` and fn items store it: the jump-table stub where the image has one), `+8 bytes: u64`
  (bytes of the stack of a thread started with a fn value of that code address; 0: no bound).
  A stack with a bound gets it (rounded up to the page size), whatever `[run] stack` says; a
  stack without one (no row, or 0) gets `default`. The compiler computes the bound per root from
  the frame sizes of the generated code over the call graph: main's stack holds `__stave_main`
  and what `main` can call; a thread's holds `__stave_thread_start`, `runtime.thread.entry`, and
  what its fn value can call; a stack whose calls can reach a call cycle (recursion, or a drop
  cycle of a recursive type) has no bound; a `rec fn` on no cycle is bounded like any fn. The bytes include a margin of 4 KiB. Signal handlers
  run on the alternate signal stack, so no bound counts them.

**Paths fixed by the compiler** (no attributes or lang items exist; these names are the whole
list):
- entry points: `runtime.start.entry`, `runtime.gpu.start.entry`, `runtime.thread.entry`,
  `runtime.thread.scope_spawn(list: &mut usize, f: fn())` and `runtime.thread.scope_join(list:
  &mut usize)` (the `scope { spawn { } }` lowering, checked-ir.md "Closures"),
  `runtime.thread.start(f: fn()) -> usize` (every fn value it can be given gets a stack-table
  row; it calls `f` itself where no thread can be created),
  `runtime.panic.trap`, `runtime.panic.trap_msg`, `runtime.panic.on_signal`,
  `runtime.panic.set_user_site`, `runtime.panic.requires_failed(clause: &str)` and
  `runtime.panic.ensures_failed(site: u32, clause: &str)` (a false contract clause),
  `runtime.panic.test_input()` and `runtime.panic.test_skip()` (the stdout markers of a `test fn`
  with parameters, cli.md "Test fn inputs"), `runtime.cpu.WORD` (a `u64` static; dispatch bodies load it),
  `runtime.cpu.features` (a request of it requests `init`), `runtime.cpu.init` (`cpu_init`),
  `runtime.cpu.trace` (dispatch bodies, trace bit), and the allocator
  `runtime.alloc.{try_alloc, try_realloc, free}` (`try_alloc(size, alignment) -> *mut u8`,
  `unsafe try_realloc(p, old_size, alignment, new_size) -> *mut u8`, both null on failure, a
  failed `try_realloc` leaving `p` unchanged; `unsafe free(p, size, alignment)`), replaced by the
  same three fns of an `allocator` declaration's module; the trapping `runtime.alloc.{alloc,
  realloc}` are derived from them (`OutOfMemory` on null) and are never replaced;
- compiler-emitted symbols: `_start`, `__stave_main`, `__stave_signal`, `__stave_thread_start`,
  `__stave_site_table`, `__stave_pc_table`, `__stave_stack_table` (and `__stave_rt` on PTX);
- `std.cell.UnsafeCell`, the only interior-mutability type: `&T` is readonly for the optimizer
  unless T contains it;
- constructors a const-expr may call: `new` of `std.cell.UnsafeCell`, `std.atomic.U32`, `U64`,
  `Usize`, `Bool`, `std.sync.Mutex`, and `std.sync.Once`. The call is evaluated as the struct
  literal the constructor returns, with its parameters bound to the (const) arguments, so each
  body is exactly `return <struct literal>;` over const-exprs and the parameters (`Mutex<T, R>`
  and `Once<T, R>` take their rank `R` from the expected type, as any type argument);
- the ranked locks of `std.sync`: `Mutex<T, R>`, `Once<T, R>`, and `MutexGuard<T, R>` (a held lock
  of rank `R`, the ADT's second argument), and the acquisitions of rank `R`, each fn's one const
  generic param: `Mutex.lock`, `Once.get_or_init` (which also runs its `f` holding `R`), and
  `lock_pair`. The rank check (E1405, E1406) knows only these; `try_lock` acquires nothing;
- `std.thread.join`, `std.thread.par_for`, `std.thread.par_for_mut`, whose fn-value arguments get
  the thread capture checks (E0802, E0806);
- language items of std: the prelude module `std.core` and its
  traits `Copy`, `Send`, `Sync`, `Drop`, `Display`, `Write`, and the operator traits (`Add` …
  `Not`); `std.string.String` (f-string values, the `String` → `str` deref through
  `String.as_str`, f-string building through `String.new` and `String.push_str`).

**PTX** (arch `ptx`): the GPU program runtime (package `runtime-gpu/`, bound as `runtime` for
`gpu-sm86` through `[target.gpu-sm86.dependencies]`) and `std` backends for
`gpu-sm86` use the intrinsics below. User kernels (`kernel fn`) get the same hardware only through the
safe `std.gpu` API (thread witnesses, collectives, atomics), never through these. `p` is a raw
pointer to global memory, and scopes are spelled in the name.

| Intrinsic | Lowering |
|---|---|
| `ld_acquire_sys_u32/u64(p) -> u32/u64`, `ld_acquire_gpu_u32/u64(p)` | `ld.acquire.sys/.gpu.global.b32/.b64` |
| `st_release_sys_u32/u64(p, v)`, `st_release_gpu_u32/u64(p, v)` | `st.release.sys/.gpu.global` |
| `ld_relaxed_gpu_u32/u64(p)`, `st_relaxed_gpu_u32/u64(p, v)` | `ld/st.relaxed.gpu.global` |
| `atom_add/or/and/min/max/exch_gpu_u32/u64(p, v) -> old`, `atom_cas_gpu_u32/u64(p, cmp, v) -> old` | `atom.<op>.relaxed.gpu.global` (`acq_rel` variants: `_acqrel` suffix) |
| `fence_sc_gpu()`, `fence_sc_sys()`, `fence_acqrel_gpu()` | `fence.sc.gpu`, `fence.sc.sys`, `fence.acq_rel.gpu` |
| `nanosleep(ns: u32)` | `nanosleep.u32` |
| `globaltimer() -> u64`, `clock64() -> u64` | `mov.u64 %globaltimer`, `mov.u64 %clock64` |
| `tid_x/y/z()`, `ntid_x/y/z()`, `ctaid_x/y/z()`, `nctaid_x/y/z()`, `laneid()`, `smid()` `-> u32` | `mov.u32` from the special register |
| `activemask() -> u32`, `match_any_u32(mask, v) -> u32`, `shfl_idx_u32(mask, v, lane) -> u32`, `vote_ballot(mask, pred) -> u32` | `activemask.b32`, `match.any.sync.b32`, `shfl.sync.idx.b32`, `vote.sync.ballot.b32` |
| `bar_sync(id: u32, n: u32)`, `bar_warp_sync(mask: u32)` | `bar.sync id, n` (id 1–15; 0 is the runtime's), `bar.warp.sync` |
| `ld_shared_u32(off: u32) -> u32`, `st_shared_u32(off: u32, v: u32)`, `atom_add/exch/cas_shared_u32(off, ..) -> old` | `ld/st/atom.shared.u32 [__stave_rt_smem + off]`: the runtime's static 256 B `.shared` area; `off` < 256, a multiple of 4 |
| `stacksave() -> usize` | `stacksave.u64` (debug-build stack checks) |
| `thread_exit()` | `exit` (only at runtime start-up for lanes 1–31 of `main`'s warp; never in virtual grids) |

PTX `trap` has no intrinsic: it poisons the CUDA context. The PTX entry `__stave_rt` is
emitted by the compiler, takes one parameter `.param .u64 ctl` (the control block,
contracts/host-channel.md) and calls `runtime.gpu.start.entry(ctl: *mut u8)` on every thread, the
GPU counterpart of `runtime.start.entry`.
