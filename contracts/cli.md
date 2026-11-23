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
| `stave check [PKG_DIR]` | Parse, resolve, type check, borrow check, and check deny clauses (`without(..)`). No codegen |
| `stave test [PKG_DIR] [--release] [--target T] [--filter S [--exact]] [--runner CMD] [--input N]` | Build and run `test fn`s and the conformance files. `--filter S` selects those whose name contains `S`; with `--exact`, the one named `S` (`main.sums`; `--exact` without `--filter`: exit 2). `--input N` runs only input N of the selected `test fn`: a selection of no `test fn`, of several, or of one without parameters is exit 2, and so is a PKG_DIR without `stave.toml`. `--runner` wraps execution (e.g. `sde64 -icx --`, `qemu-aarch64 -cpu max,...`). A target the host cannot run without a runner is E1001: reported as not run, never as passed |
| `stave unsafe-report [PKG_DIR] [--json]` | List the unsafe inventory |
| `stave effects [PKG_DIR] [--json]` | List the effect set of each fn of the program's own packages |
| `stave caps-report [PKG_DIR] [--json]` | Planned: per package, the OS capability classes its code can reach |
| `stave bounds-report [PKG_DIR] [--target T] [--json]` | Planned: every bounds check kept in release code, with the reason |
| `stave branch-report [PKG_DIR] [--target T] [--json]` | Planned: branches in loops whose condition depends on loaded data, with a suggested fix |
| `stave fix [PKG_DIR] [--dry-run] [--target T] [--diagnostics human\|json]` | Apply the `exact` fixes of the diagnostics, re-check, repeat |
| `stave fmt [PKG_DIR \| FILE...] [--check]` | Rewrite sources in the one canonical layout |
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
  exits 0; their summary line is `test fns: P passed, F failed, N not run`. A `test fn` with
  parameters runs over its inputs (below).
- **Test fn inputs.** A `test fn` with parameters is a harness: one process calls it once per
  input, in input order, and passes iff it exits 0. The runner is the caller: it passes only
  inputs within the ranges below, and the test fn evaluates all its `requires` clauses at entry,
  in order; a false one skips the input (it never panics `Contract`).
  - Parameter types (else E0214): `i8 i16 i32 i64 isize u8 u16 u32 u64 usize bool char f32 f64`,
    `[T; N]`, tuples, and non-view structs of those (nested), and, as a parameter's own type,
    `&[T]` of those (no `align`). Every struct inside, and each of its fields, must be visible from
    the test's module (`pub`, or declared there; else E0214 case `private`, or, for a struct of
    another package, case `field` naming the field that holds it and its type): so no enums,
    refs, views, `vec.Vec`, or `string.String`. A slice needs a length upper bound; a value holds at
    most 4096 scalars (a slice: bound × the element's); the return type is `()`.
  - Domain: one generator per parameter: an int's range starts as its type's, a slice's length
    range as `0..`. Each `requires` clause is split at `&&` (also inside parentheses); a conjunct
    `S op K` or `K op S` (`op` one of `< <= > >= ==`; `S` an int parameter, an int field `p.f.g`
    of a struct parameter (fields only, no index or tuple step), or `p.len()` of a slice
    parameter; `K` an int literal, a negated or parenthesized `K`, or an int `const`) intersects
    `S`'s range with the values satisfying it. Every other conjunct only filters. The domain is
    the product of the generators (finite, so it can be enumerated).
  - Edges. `int_edges(lo, hi)`: of `lo`, `lo+1`, `hi-1`, `hi`, `0`, `1`, `-1`, and for `k` in
    1..=64 `2^k-1`, `2^k`, `2^k+1`, `-2^k-1`, `-2^k`, `-2^k+1`, those in `lo..=hi`, ascending and
    distinct. `bool`: `false true`. `char`: `0 0x7F 0x80 0x7FF 0x800 0xD7FF 0xE000 0xFFFF 0x10000
    0x10FFFF`. `f32` bits: `0x00000000 0x80000000 0x3F800000 0xBF800000 0x00000001 0x00800000
    0x7F7FFFFF 0xFF7FFFFF 0x7F800000 0xFF800000 0x7FC00000` (±0, ±1, least subnormal, least
    normal, ±max, ±inf, NaN); `f64` the same eleven values. The period of a generator: an int's
    edge count, 2, 10, 11, 11; `[T; N]`: T's (1 if N = 0); a tuple or struct: the largest of its
    fields' (1 if none); a slice: the larger of its length range's edge count and T's. Edge `k`
    (cyclic): a scalar's edge `k mod period`; element `j` of an array or slice is T's edge `k + j`;
    field `i` of a tuple or struct its edge `k`; a slice's length is `int_edges(length range)[k
    mod count]`.
  - Positions: the parameters in order, a struct parameter replaced by its fields in order (a
    struct field again by its fields); a struct inside an array, tuple, or slice is not split.
  - Inputs, in order, with `n` the largest period of the positions: the diagonal (`k` = 0 to n-1:
    every position its edge `k`); with two or more positions the anti-diagonal (`k` = 0 to n-1:
    position `i` its edge `k` if `i` is even, else edge `period - 1 - (k mod period)`); then
    256 samples from SplitMix64 with state 0 (`next`: `state += 0x9E3779B97F4A7C15`; `z = state;
    z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9; z = (z ^ z >> 27) * 0x94D049BB133111EB; return
    z ^ z >> 31`, all mod 2^64), parameters in order: an int `lo + next mod (hi - lo + 1)`, `bool`
    `next & 1`, `char` `r = next mod 0x10F800` plus 0x800 if `r >= 0xD800`, `f32` the low 32 bits
    of `next`, `f64` `next`, arrays, tuples, and structs their parts in order, a slice its length (as an int
    of its length range) then its elements. An input equal to an earlier one (floats by bits) is
    dropped. Input `N` is the N-th of this list, from 0; it depends only on the parameter types and
    ranges, never on the target, the build mode, or the run.
  - Markers: the test fn writes `\x01stave-test input\n` to stdout as each input starts and
    `\x01stave-test skip\n` when a clause skips it (`runtime.panic.test_input` / `test_skip`);
    the harness counts them and never shows them.
  - Report lines: `ok        test T: R inputs run, S skipped` (`1 input run`), then `    note:
    ...` when fewer than one input in ten ran; `FAILED    test T (exit E) at input N (K of M
    started)`, the stderr lines, `    input N: p = <literal>, ...` (Stave literals: suffixed ints
    and floats, a float with no literal as `mem.f32_from_bits(0x..)` / `mem.f64_from_bits(0x..)`,
    `'c'` with `\u{..}` escapes, `[..]`, `(..)`, `&[..]`, `S.{ f: .. }` with `S` as the test's
    module names it: `S` if declared there, else through the import binding its module
    (`geo.Point`), else its full module path, generic arguments as `S::<u32>`), and `    replay: stave test DIR
    --filter T --exact --input N --target TRIPLE` (plus `--release`, `--runner CMD` when given).
    `FAILED    test T: ...` with no exit status: no input ran (an empty range; `requires` skipped
    all; `--input N` past the last input), or the process exited with status 0 before its last
    input.
- In-memory execution (`stave run`, `stave test`, on x86-64 and AArch64 Linux): the ELF image is
  held in an anonymous memory file (`memfd_create`, close-on-exec) and a child process starts it
  as `/proc/self/fd/N`; `argv[0]` is the program's name. With `--runner CMD` the child runs
  `CMD /dev/fd/N`, and only that child keeps descriptor `N` across its exec. Stdio, working
  directory, environment, and exit status are those of a run from a file. If the kernel lacks
  `memfd_create` (ENOSYS, no other error) or the system has no `/proc/self/fd`, the programs of
  that `stave` process run from files in a fresh private temporary directory instead, and one
  line on stderr says so.

- `stave unsafe-report [PKG_DIR] [--json]`: checks the program
  through type checking; with error diagnostics they go to stderr (`--json`: as JSON lines),
  nothing to stdout, exit 1. Otherwise stdout lists every site of every package of the program
  (root, dependencies, `std`, `runtime`), one per line, in canonical (file F, offset) order, then
  kind order: `block` (`unsafe("..") { }`), `expr` (`unsafe("..") e`), `fn` (unsafe fn, method, or
  trait method signature; also an `unsafe("..")` export), `trait`, `impl`, `extern` (an
  `unsafe("..") extern "C" { }` import block), `extern-call` (each use of an `extern "C"` import,
  with the innermost enclosing block/expr obligation). Human: `<file>:<line>:<col>: <kind> in <fn>:
  "<obligation>"`; `--json`: one object per line,
  `{"file":"src/main.stv","line":21,"col":17,"fn":"main.read_raw","kind":"expr","obligation":"..."}`.
  `file` is the diagnostics path inside the root package's directory, else
  `<package>/<path in package>` (`runtime/alloc.stv`, never an install path); `line`/`col` (1-based,
  col in bytes) locate the `unsafe` keyword (an `extern-call`: the callee name); `fn` is the
  checked-IR path of the enclosing fn (`trait`/`impl`: the trait's path / the impl's self-type
  head; `extern`: the module's path); strings are JSON-escaped in both forms. The package set follows `--target` selection of
  dependencies (the host baseline: no `--target` flag).

- `stave effects [PKG_DIR] [--json]`: checks the program as `stave check` does (always stage0,
  host baseline), computing `trap` from the checked IR of every fn (LANG.md §2, deny clause).
  With error diagnostics before the effect pass they go to stderr (`--json`: as JSON lines),
  nothing to stdout, exit 1. Otherwise stdout has one line per fn and closure of the root package
  and its dependencies outside `std` and `runtime` (a generic fn once per instance the program
  makes), in canonical (file F, offset) order, then by name. Human: `<file>:<line>:<col>: <fn>:
  <effects>`, the effects space-separated in the order `alloc trap recursive lock unsafe io
  block park`, or `none`; `--json`: one object per line,
  `{"file":"src/main.stv","line":9,"col":4,"fn":"main.add","effects":["trap"]}`. `file`, `line`,
  `col` as in `unsafe-report`, at the fn's name (a closure: its `fn` token); `fn` is the
  checked-IR path, with the type args of an instance (`main.id<u32>`), a closure `<fn>.<k>`.
  Deny-clause violations (E1401, E1403) are printed on stderr after the report, exit 1.

- `stave fix [PKG_DIR] [--dry-run]`: runs `stave check` (always stage0), applies every fix whose
  `applicability` is `exact` to the source files of the root package (never `std`, `runtime`, or
  another dependency, inside the root directory or not), in canonical diagnostic order, skipping a
  fix with an edit that overlaps one already applied in the round (sharing a byte, an insertion
  inside a replaced range, or two insertions at one offset); then re-checks and repeats, at most 8
  rounds or until a round applies nothing. Prints one line per applied fix on stdout,
  `<file>:<line>:<col>: <code> fixed` (the diagnostic's location in the round that reported it),
  then the remaining diagnostics on stderr (`--diagnostics` format). Exit 0 if no error remains,
  1 otherwise. The files are written once, after the last round. `--dry-run` writes nothing and
  prints, in place of the `fixed` lines, the edits as a unified diff (`--- a/<file>`,
  `+++ b/<file>`, three context lines); the remaining diagnostics and the exit status are those
  of the run without it.
- `stave fmt [PKG_DIR | FILE...] [--check]`: rewrites each source file in the canonical
  layout printed by `stave-syntax`'s round-trip printer. There are no options: the rules are the
  module documentation of `crates/stave-syntax/src/print.rs`, fixed by the golden tests in
  `crates/stave-syntax/tests/fmt.rs`. Only whitespace changes: tokens and comments are kept in
  order, so the result parses to the same AST, and formatting it again changes nothing. An
  argument is a file, or a directory standing for every `.stv` file below it (hidden directories
  and symbolic links are skipped); no argument means the current directory. A file with any
  lexical or syntax error is left unchanged and its diagnostics go to stderr (exit 1); the other
  files are still formatted. `--check` writes nothing, prints the path of each file that would
  change on stdout, one per line, and exits 1 if there is one. A path that cannot be read or
  written is exit 2.

Planned commands (exit codes as above):

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
  block in the source, and every class U operation inside one is listed in the build summary:
  one line `emulated: <file>:<line>:<col>: <op> on <type> (<target>)` each, on stderr before the
  summary line, in source order.
- The stack bound of a CPU executable is listed in the build summary of `build` and `run`, after
  the `emulated:` lines: `stack: main: <n> bytes`, then one line per function a thread can
  start with, `stack: spawn at <file>:<line>:<col>: <n> bytes` (a `spawn` body) or
  ``stack: thread `<fn>`: <n> bytes`` (a closure `std.thread` starts, e.g.
  `` `std.thread.join<u64, u64>.0` ``). `<n>` is the computed bound; the runtime maps it plus a
  4 KiB margin, rounded up to the page size. A stack without a bound reads
  ``unbounded (<why>), [run] stack <n> bytes``, `<why>` one of ``reaches rec fn `<fn>` `` (a call cycle through that `rec fn`),
  ``reaches a call cycle through `<fn>` `` (a drop cycle of a recursive type), ``calls a fn
  value with unknown targets in `<fn>` ``, ``calls C code in `<fn>` ``, `the call graph is
  incomplete`. The bound depends
  on the build mode and target (frame sizes); it never fails a build.
- `--cache DIR`: incremental cache location. Default: `PKG_DIR/.stave-cache`. Planned: stage0 has
  no cache and refuses the flag as not implemented (exit 2).

**Summary line**: every compiling command prints one line on stderr naming the compiler used
([cli-stage1.md](cli-stage1.md)), e.g.
`compiled on GPU (stage1, NVIDIA RTX A6000) in 412 ms`, `compiled on CPU (--compiler cpu)`, or
`compiled on CPU (no usable GPU: <reason>)`; for steps 1 and 1a the reason stands alone,
`compiled on CPU (target aarch64-linux not supported by stage1)`. It is never part of a compared
dump. `stave run` of a file whose first line is a shebang line prints no summary line: the stderr
of a script holds only diagnostics and what the program writes. With `--diagnostics json` the summary, the `emulated:` lines, and the stack lines are left out:
stderr holds only the JSON objects.

**Shipped library**: the built-in packages `std` and `runtime` are read from a *library
directory* `D` (`D/std/`, holding `core.stv`, and `D/runtime/`). The compiler uses the first of:
(1) `$STAVE_STD_DIR` when set and not empty; it must be a library directory, otherwise usage error
(exit 2), with no fallback; (2) `<stave install>/lib/stave/` (`<stave install>` as in the probe table:
the parent of the directory holding the `stave` binary); (3) for development, the nearest ancestor
of the binary's directory that holds a `Cargo.toml` and is a library directory (the repository
root, for `target/<profile>/stave`). None found: usage error (exit 2) naming the places tried.
Every dump covers the whole program, `std` and `runtime` included.

**Interpreter checking mode**: no `stave` command runs the stave-ir reference interpreter; the
test harness does (`stave_driver::interp::run`: the interpreter legs of the run, SIMD, dispatch,
and differential suites). `STAVE_INTERP_CHECK=1` in the environment of that process turns on its
checking mode for every run that takes the default limits (`Limits::check` in code). A memory
error then ends the run as a failure whose text starts `interp: memory error [<class>]`, class one
of `use-after-free double-free uninit-read out-of-bounds invalid-free free-size-mismatch
copy-overlap` (a `copy_bytes` whose ranges share a byte, intrinsics.md "Untyped copy"), then
the access, the allocation (id, kind `heap stack host`, size, state), and the access, allocation,
and free traces (fn, instruction index, `file:line:col`, `near` for an instruction without a
site of its own). Off, the interpreter behaves as without the switch.

**Borrow tracking** (`borrow-model.md`) is a further mode of the same interpreter, on top of the
checking mode: `Limits::borrow` in code, on a module lowered with the borrow side table ("Side
table" there). `STAVE_INTERP_CHECK=borrow` turns on both modes for every run that takes the
default limits; the pipeline then lowers each program a second time, with the side table, for the
interpreter legs (`Options::tracked`, `Output::for_interp`). Dumps, goldens, and the backends get
the canonical module in every case. A violation ends the run as a failure
whose text starts `interp: borrow violation [<class>] <rule>`, class one of `write-under-shared
two-exclusive use-after-referent second-owner`, rule `M1`..`M4`, `M9`, `M10`, or `M23`; then the
event, its bytes, the allocation, the permission and kind of the failing tag; then three lines:
`at` (the trace of the violating event), `cause` (the event that made the tag `Frozen` or `Dead`;
for `M23` the call that protects the tag; for `M9` the move that emptied the place), `tag made
at`. A `from` clause that a call did not keep (`M24`) is no violation and does not end the run: a
run that ends without a violation and has such reports is a failure whose text starts `interp:
borrow signature [signature] M24`, then what holds the reference (the result of a fn, or a param
by its index; a closure body is `the closure at <file>:<line>:<col> in <fn>`), then two lines: `at` (the trace of the return), `tag made at`; a last line counts
the further reports.

The `check` CI step runs every positive run test of `tests/conformance/` and every program of
`tests/borrowcheck/` with borrow tracking (`crates/stave-driver/tests/interp_gate.rs`). A report
on a program whose own packages have no site in the unsafe inventory (`stave unsafe-report`
without `std`, `runtime`) fails it; a program with such sites runs clean or ends with the report
it declares (`INTERP-REPORT`, [conformance-annotations.md](conformance-annotations.md)). The
fuzzer `tools/fuzz-classes` runs its interpreter leg in the mode of `STAVE_INTERP_CHECK`,
`borrow` when unset: a report on an accepted program is an `oracle` finding.

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
stack = "8MiB"                          # stacks without a computed bound (gpu-program.md)

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
`"8MiB"`, at least `"4KiB"`, size grammar of gpu-program.md "Manifest keys"; another value is
E1203) is the stack size of `main` and of every runtime thread whose calls can reach a call
cycle (recursion, or a drop cycle of a recursive type; a `rec fn` on no cycle does not count), on every target; on the CPU every other stack gets the
computed bound of what it can call (intrinsics.md "stack table"), and each stack has a guard page
below it.
