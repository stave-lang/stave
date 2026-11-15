# Contract: `stave` command line (stage0)

One binary named `stave`. Every subcommand exits with **0** on success, **1** on a user error (the
program has diagnostics), **2** on a usage error (bad flags), and **3** on an internal compiler
error. An exit code of 3 is always a bug. Environment errors that are diagnostics,
such as E1302 (`--compiler gpu` with no usable GPU), exit 1. These codes are the compiler's own; a
built program's exit codes are in LANG.md §3 (101 on a panic).

| Command | Purpose |
|---|---|
| `stave build [PKG_DIR] [--release] [--target T] [--features F,...] [-o OUT]` | Build an executable or static library |
| `stave run [PKG_DIR [ARGS...]] [--release] [--target T] [-- ARGS...]` | Build, then exec. `--target` only the host's targets (the triples of the host's arch, at any level the host CPU supports) or `gpu-sm86`; otherwise E1001 |
| `stave check [PKG_DIR]` | Parse, resolve, type check, and borrow check. No codegen |
| `stave test [PKG_DIR] [--release] [--target T] [--filter S] [--runner CMD]` | Build and run `test fn`s and the conformance files. `--runner` wraps execution (e.g. `sde64 -icx --`, `qemu-aarch64 -cpu max,...`). A target the host cannot run without a runner is E1001: reported as not run, never as passed |
| `stave unsafe-report [PKG_DIR] [--json]` | List the unsafe inventory |
| `stave caps-report [PKG_DIR] [--json]` | Planned: per package, the OS capability classes its code can reach |
| `stave bounds-report [PKG_DIR] [--target T] [--json]` | Planned: every bounds check kept in release code, with the reason |
| `stave branch-report [PKG_DIR] [--target T] [--json]` | Planned: branches in loops whose condition depends on loaded data, with a suggested fix |
| `stave fix [PKG_DIR] [--dry-run]` | Planned: apply the `exact` fixes of the diagnostics, re-check, repeat |
| `stave fmt [PKG_DIR \| FILE...] [--check]` | Planned: rewrite sources in the one canonical layout |
| `stave grammar [--format ebnf\|lark\|gbnf]` | Planned: print the grammar for constrained decoding |
| `stave lowering --target T [--op OP]` | Print the lowering-class table |
| `stave explain EXXXX` | Long-form text for a diagnostic code (registry: `diagnostic-codes.md` ∪ `gpu-diagnostics.md` ∪ `gpu-program-diagnostics.md`) |
| `stave build --watch` | Resident incremental mode (build-time targets are measured warm) |
| `stave daemon status\|stop\|evict [PKG_DIR]` | Show or stop the resident stage1 daemon, or drop its sessions for one package root ([resident-protocol.md](resident-protocol.md) `Evict`). With no daemon running, `status` and `stop` print `daemon: not running` and exit 0 |

Outputs and statuses of the code-generating commands:

- `stave build`: the static executable (ELF `ET_EXEC`, no `PT_INTERP`/`PT_DYNAMIC`) goes to `-o OUT`,
  default `./<package name>` (a single `.stv` file: `./<file stem>`). The default name must be one
  plain path component: a package `name` that is empty, `.` or `..`, contains `/`, `\`, or NUL,
  or starts with `-` is E1002 `bad-name` (exit 1, in every command, `stave check` included); such a
  file stem is refused (exit 2, nothing written) unless `-o` is given. Nothing is written when the
  build has error diagnostics (exit 1); a program without `fn main` is E0108 (exit 1). A valid
  program the compiler cannot build yet (a construct its back end does not support, `kind = "lib"`
  until the static-library output exists) exits 2 with the reason, like any unimplemented command.
- `stave run`: builds the executable in memory and writes no file (`stave build -o OUT` is the
  way to get one). The program is named as the `stave build` default (refused the same way), runs
  with `ARGS` and the inherited stdio, and `stave run` exits with the program's status (`128 + n`
  when killed by signal `n`). A target the host cannot run is E1001 (exit 1) before anything is
  built. `ARGS` are the words after `PKG_DIR` that are not flags, then everything after `--` (an
  argument starting with `-` must come after `--`).
- Scripts: `PKG_DIR` may be one `.stv` file, a program without a manifest, in every command. Its
  first line may be a shebang line (LANG.md §2), so an executable file starting with
  `#!/usr/bin/env -S stave run` runs directly: `./hello.stv a b` is `stave run ./hello.stv a b`.
- `stave test [DIR]`: the conformance runner over DIR (contracts/conformance-annotations.md);
  `--target T` applies to tests without a `TARGET` annotation, `--runner CMD` (split at
  whitespace, the binary and its arguments appended) runs every binary. When DIR is a package,
  its `test fn`s run too, each as its own executable from the package directory, passing iff it
  exits 0; their summary line is `test fns: P passed, F failed, N not run`.
- In-memory execution (`stave run`, `stave test`, on x86-64 and AArch64 Linux): the ELF image is
  held in an anonymous memory file (`memfd_create`, close-on-exec) and a child process starts it
  with `execveat(fd, "", argv, envp, AT_EMPTY_PATH)`; `argv[0]` is the program's name. With
  `--runner CMD` the child runs `CMD /dev/fd/N`, and only that child keeps descriptor `N` across
  its exec. Stdio, working directory, environment, and exit status are those of a run from a
  file. If the kernel lacks `memfd_create` or `execveat` (ENOSYS, no other error), the programs
  of that `stave` process run from files in a fresh private temporary directory instead, and one
  line on stderr says so.

- `stave unsafe-report [PKG_DIR] [--json]`: checks the program
  through type checking; with error diagnostics they go to stderr (`--json`: as JSON lines),
  nothing to stdout, exit 1. Otherwise stdout lists every site of every package of the program
  (root, dependencies, `std`, `runtime`), one per line, in canonical (file F, offset) order, then
  kind order: `block` (`unsafe("..") { }`), `expr` (`unsafe("..") e`), `fn` (unsafe fn, method, or
  trait method signature), `trait`, `impl`, `extern-call` (each use of an `extern "C"` import, with
  the innermost enclosing block/expr obligation). Human: `<file>:<line>:<col>: <kind> in <fn>:
  "<obligation>"`; `--json`: one object per line,
  `{"file":"src/main.stv","line":21,"col":17,"fn":"main.read_raw","kind":"expr","obligation":"..."}`.
  `file` is the diagnostics path inside the root package's directory, else
  `<package>/<path in package>` (`runtime/alloc.stv`, never an install path); `line`/`col` (1-based,
  col in bytes) locate the `unsafe` keyword (an `extern-call`: the callee name); `fn` is the
  checked-IR path of the enclosing fn (`trait`/`impl`: the trait's path / the impl's self-type
  head); strings are JSON-escaped in both forms. The package set follows `--target` selection of
  dependencies (the host baseline: no `--target` flag).

Planned commands (exit codes as above):

- `stave fix [PKG_DIR] [--dry-run]`: runs `stave check`, applies every fix whose
  `applicability` is `exact` to the files of the root package (never `std`, `runtime`, or a
  dependency outside the root directory), in canonical diagnostic order, skipping an edit
  that overlaps one already applied in the round; then re-checks and repeats, at most 8 rounds or
  until a round applies nothing. Prints one line per applied fix, `<file>:<line>:<col>: <code>
  fixed`, then the remaining diagnostics. Exit 0 if none remain, 1 otherwise. `--dry-run` writes
  nothing and prints the edits as a unified diff.
- `stave fmt [PKG_DIR | FILE...] [--check]`: rewrites each source file in the canonical
  layout printed by `stave-syntax`'s round-trip printer; comments are kept; the result is
  idempotent and parses to the same AST. A file with any lexical or syntax error is left unchanged
  and its diagnostics are reported (exit 1). `--check` writes nothing and exits 1 if a file would
  change, listing it.
- `stave grammar [--format ebnf|lark|gbnf]`: prints the grammar generated by
  `tools/ll1gen` from `contracts/grammar.ebnf` (default `ebnf`: the contract text with comments
  removed). `lark` (llguidance, Lark) and `gbnf` (llama.cpp, XGrammar) include the lexical rules,
  whitespace, and `//` comments. The exported grammar accepts every file stage0 parses without a
  syntax error; it may accept more (the three greedy overrides become plain alternatives, keywords
  are not excluded from identifiers in `gbnf`).
- `stave caps-report [PKG_DIR] [--json]`: checks through type checking (errors as for
  `unsafe-report`), then prints one line per package of the program, in package order: `<package>:
  <classes>`, the capability classes reachable from that package's own fns through the call graph
  (calls through fn values: every fn item coerced to a fn value in the program), space-separated in
  this order, or `none`: `fs` (read write pread64 pwrite64 openat close fstat lseek fsync fdatasync
  ftruncate fcntl flock getdents64 unlinkat mkdirat renameat2), `mem` (mmap munmap mremap mprotect
  madvise), `thread` (clone3 clone futex sched_getaffinity sched_yield), `proc` (exit_group exit
  getpid kill), `time` (clock_gettime), `random` (getrandom), `signal` (rt_sigaction
  rt_sigprocmask sigaltstack), `net` (none in v1; planned with `gpu-sm86` server ops). The classes are the
  `runtime/syscalls.list` entries a `runtime.os` fn issues; an `extern "C"` call adds `ffi`.
  `--json`: `{"package":"app","caps":["fs","time"]}` per line.
- `stave bounds-report [PKG_DIR] [--target T] [--json]`: builds in release mode and
  prints one line per bounds check kept in release code, in canonical (file, offset) order:
  `<file>:<line>:<col> <fn> kept <reason>`, reason one of `no-fact index-from-memory len-unknown
  call-arg step-limit`. Paths as in `unsafe-report`.

- `stave branch-report [PKG_DIR] [--target T] [--json]`: builds in release mode and
  prints one line per conditional branch that remains inside a loop and whose condition depends
  (through the SSA graph) on a value loaded from memory, in canonical (file, offset) order:
  `<file>:<line>:<col> <fn> branch-on-load <fix>`, fix one of `select` (both arms cheap and
  trap free), `partition` (sort or partition the data first), `mask` (a SIMD mask over the loop).
  `--target gpu-sm86` instead lists thread-varying branches in kernels.

Common flags come after the subcommand (`stave build --release`, not `stave --release build`):

- `--target T`: `x86_64-linux[-v2|-v3|-v4]` or `aarch64-linux[+sve|+sve2]`, or `gpu-sm86` (whole
  program on the GPU, [gpu-program.md](gpu-program.md); never a default). Default: host
  baseline (`-v2` or plain NEON). It is never auto-upgraded to the host's best level (no silent
  choice).
- `--features F,...`: adds features to the baseline. It may repeat; the lists accumulate. Unknown
  names are usage errors. Planned: stage0 validates the names, then refuses features beyond the
  baseline as not implemented (exit 2).
- `--alloc-check`: device allocator checks for `gpu-sm86` builds
  ([gpu-program.md](gpu-program.md) "Allocation checker"); with any other target it is a usage
  error (exit 2).
- `--compiler auto|cpu|gpu`: which compiler runs. `auto` (default): stage1 if a usable
  GPU and a matching stage1 image exist, otherwise stage0. `cpu`: always stage0. `gpu`: stage1, or
  E1302 (no usable GPU) / E1303 (target not supported by stage1). Probe order (`auto` and `gpu`),
  stopping at the first failing step, whose reason is reported:

  | Step | Check | Reason on failure | `gpu` reports |
  |---|---|---|---|
  | 1 | target supported by stage1 (code-generating commands; not AArch64 in v1) | `target <T> not supported by stage1` | E1303 |
  | 1a | the package links no C objects, judged from its declared C-link inputs before stage1 starts | `C linking not supported by stage1` | E1304 |
  | 2 | stage1 image present: `<stave install>/lib/stave/stage1-<build id>.img`, where `<stave install>` is the parent of the directory holding the `stave` binary and `<build id>` is this binary's build id (16 hex digits) | `stage1 image missing` | E1302 |
  | 3 | NVIDIA driver loads (`libcuda.so.1`) and reports a device | `no NVIDIA driver`, `no device` | E1302 |
  | 4 | device compute capability ≥ 8.6 | `compute capability <M>.<m> < 8.6` | E1302 |
  | 5 | stage1 daemon reachable, or startable (resident-protocol.md) | `daemon not reachable` | E1302 |

  Because the probe stops at the first failure, at most one of E1302, E1303, E1304 appears: when
  several would apply, E1303 (step 1) comes before E1304 (step 1a), and both before E1302. The
  memory estimate (cli-stage1.md, E1301) runs after a passing probe.
- `--dump=STAGE[,STAGE...] --dump-dir DIR`: writes `DIR/STAGE.txt` per stage (`tokens items ast
  strings names types borrow ir diags stats`). `--dump` without `--dump-dir`, or `--dump-dir`
  without `--dump`, is a usage error (exit 2). Formats and order:
  [stage-gates.md](stage-gates.md); `ir` is
  [checked-ir.md](checked-ir.md). stage0 implements every stage.
- `--diagnostics human|json`: JSON emits one object per line, following
  `diagnostics.schema.json`.
- `--timings`: per-stage wall time and throughput on stderr, in execution order, by the
  schema stage names; stage0 reports `read lex items parse intern resolve typeck` and, for builds,
  `lower` (mono and lowering fused), `codegen`, `link` (DWARF line tables fused). It also names the compiler
  (`compiler: cpu (stage0)` or `compiler: gpu (stage1, <device>)`) and, per stage, `rounds` and
  `fallbacks` where a stage has them ([cli-stage1.md](cli-stage1.md)). With `--diagnostics json` they appear as a
  `{"kind":"timings",...}` line.
- Scalar emulation of `unavailable` operations has no flag. The only opt-in is an `emulate { }`
  block in the source, and every such block is listed in the build summary.
- `--cache DIR`: incremental cache location. Default: `PKG_DIR/.stave-cache`. Planned: stage0 has
  no cache and refuses the flag as not implemented (exit 2).

**Summary line**: every compiling command prints one line on stderr naming the compiler used
([cli-stage1.md](cli-stage1.md)), e.g.
`compiled on GPU (stage1, NVIDIA RTX A6000) in 412 ms`, `compiled on CPU (--compiler cpu)`, or
`compiled on CPU (no usable GPU: <reason>)`; for steps 1 and 1a the reason stands alone,
`compiled on CPU (target aarch64-linux not supported by stage1)`. It is never part of a compared
dump.

**Shipped library**: the built-in packages `std` and `runtime` are read from a *library
directory* `D` (`D/std/`, holding `core.stv`, and `D/runtime/`). The compiler uses the first of:
(1) `$STAVE_STD_DIR` when set and not empty; it must be a library directory, otherwise usage error
(exit 2), with no fallback; (2) `<stave install>/lib/stave/` (`<stave install>` as in the probe table:
the parent of the directory holding the `stave` binary); (3) for development, the nearest ancestor
of the binary's directory that holds a `Cargo.toml` and is a library directory (the repository
root, for `target/<profile>/stave`). None found: usage error (exit 2) naming the places tried.
Every dump covers the whole program, `std` and `runtime` included.

Package layout: `stave.toml` plus `src/main.stv` (`kind = "bin"`) or
`src/lib.stv` (`kind = "lib"`: a static `.a` plus a C header for `extern "C"` exports). Module path
= file path (`src/a/b.stv` is `a.b`).

```toml
name = "app"
kind = "bin"
version = "0.1.0"

[dependencies]
prims = { path = "prims/cpu" }          # path-only in v1

[run]
stack = "8MiB"                          # main and every thread (gpu-program.md)

[target.gpu-sm86.dependencies]          # per target
prims = { path = "prims/gpu" }
```

For `--target T`, the dependencies are `[dependencies]` merged with `[target.T.dependencies]`,
matched by exact canonical target name (`x86_64-linux-v2|-v3|-v4`, `aarch64-linux`,
`aarch64-linux+sve`, `aarch64-linux+sve2`, `gpu-sm86`; `--target x86_64-linux` is
`x86_64-linux-v2`; `--features` does not change the name), except that
`[target.x86_64-linux.dependencies]` applies to every x86-64 level (v2/v3/v4) and
`[target.aarch64-linux.dependencies]` to every AArch64 target (`+sve`, `+sve2` too); a
level-specific table (`[target.x86_64-linux-v3.dependencies]`,
`[target."aarch64-linux+sve".dependencies]`) is merged after it. A name in both takes the target
entry (the specific one over the arch table). Built-in packages (`std`, `runtime`) may have a
`stave.toml` too, read the same way (the runtime selects its syscall layer this way). No `cfg` expressions. Invalid manifests (unknown key or table, wrong value type,
non-path dependency, unknown target name, TOML syntax error, duplicate key, bad `kind`, missing
`name`, missing root file, missing dependency directory, unreadable manifest) are E1002, located
in `stave.toml`; an unreadable source file is E1003. `[run] stack` (default
`"8MiB"`, at least `"4KiB"`, size grammar of gpu-program.md "Manifest keys") is the stack
size of `main` and of every runtime thread, on every target; on the CPU each stack has a guard page
below it.
