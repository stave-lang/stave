# Benchmark

Compile time and run time of the stage0 `stave` compiler against tcc, gcc, and clang, on the same
program written in both languages. `bench/run.sh` prints a markdown report on stdout.

```sh
tools/ci-step/run.sh bench                      # or: sh bench/run.sh
sh bench/run.sh --lines 100000 --seed 1 --runs 5 --reps 500   # the defaults
sh bench/run.sh --tier quick                    # stave only, small corpus: see Tiers
```

Linux only. Needs the Rust toolchain, coreutils, awk, xargs, and any of `tcc`, `gcc`, `clang`; a
compiler that is not installed is skipped and named in the report. Work files go to
`target/bench/` and `target/bench-cache/` and are never checked in.

`sh bench/run.sh --stages` prints only where `stave build` spends its time on the corpus: the
median wall time of every pipeline step (`STAVE_PROFILE`, `stave_driver::prof`) on 1 thread and on
all threads. With `--alloc-count` it builds `stave` with a counting allocator and adds allocation
calls and bytes per step (those times are slower: count and time in separate runs).

`sh bench/run.sh --simd [--runs R]` runs `bench/simd.sh` instead: explicit vector kernels built
by stave for x86_64-linux and x86_64-linux-v3 against their C99 twins (`simd/README.md`).

## Tiers

`--tier` picks how much is measured. C compiler numbers do not change between Stave commits, so
only `full` measures them.

| tier | measured | defaults | C rows | output checked against |
|---|---|---|---|---|
| `full` (default) | stave and every C configuration | 100,000 lines, 5 runs | measured, then cached | every C build |
| `stave` | stave, debug and `--release` | 100,000 lines, 5 runs | read from the cache | one C build |
| `quick` | stave, debug and `--release` | 20,000 lines, 3 runs | none | one C build |

`--lines` and `--runs` override a tier's defaults. The one C build of `stave` and `quick` is the
first installed of tcc, `gcc -O0`, `clang -O0`; the report names it.

`full` writes each C row to `target/bench-cache/c-<config>-<lines>-<seed>-<reps>.txt` with what it
was measured on: `--lines`, `--seed`, `--reps`, the compiler's command line and version string,
and a hash of the generated `single.c` and the kernels' `main.c`. `stave` prints a row only when
all of that still matches (`--runs` is not compared: the row keeps the run count of the `full` run
that wrote it); otherwise the row reads `n/a (run --tier full)`.

Every tier also keeps its stave rows in `target/bench-cache/last-<tier>-<lines>.txt` and, when the
file exists from an earlier run with the same `--seed` and `--reps`, ends the report with a table
of each stave number (compile time on 1 thread and on all threads, binary size, corpus and kernel
run times) against that run. A number more than 5% worse is marked `SLOWER`; the exit status
stays 0.

## What is measured

Inputs:

- **corpus**: `gen-corpus --pair` writes one synthetic program of about N lines (default 100,000,
  seed 1) in Stave (`stave/`, one module per unit) and in C99 (`c/single.c`, and `c/split/`: one
  `.c` file and header per unit, a shared `corpus.h`, `main.c`, and the file list `files.txt`).
  Default: 80 units, about 4,500 fns of 3 to 120 lines: `u32`/`u64`/`f64` arithmetic, `if`,
  `for`, `while`, `break`, early returns, small local arrays, calls to earlier fns. `main` runs
  every unit's driver `--reps` times and prints one checksum.
- **kernels** (`bench/kernels/`, hand-written, scalar code): `nbody` (f64 pair loop and energy
  reduction), `hashmix` (u64 hash mixing and an open-addressing table), `sieve` (sieve of
  Eratosthenes over 16M bytes). Each is a Stave package (`src/main.stv`) and its C twin `main.c`.

Columns (wall clock of the whole process, median of `--runs`, default 5):

| column | Stave | C |
|---|---|---|
| compile, 1 thread | `STAVE_THREADS=1 stave build stave/ --compiler cpu -o OUT [--release]` | `CC -std=c99 [-O0\|-O2] -o OUT single.c` (tcc: no `-O`) |
| compile, all threads | the same without `STAVE_THREADS` (one worker per hardware thread) | `xargs -P $(nproc) -n 1 CC -std=c99 [-O..] -c < files.txt`, then `CC -o OUT *.o` |
| K lines/s | Stave lines / 1-thread time | C lines of `single.c` / 1-thread time |
| binary bytes | `stave build` output: a static ELF, no libc | the single-file build: dynamically linked against glibc, not stripped |
| corpus run, kernel runs | the debug or `--release` build | the build with that compiler and `-O` level |

Every compile column produces a runnable executable: C times include the system linker, Stave
writes the ELF itself. Stave times include compiling `std` and `runtime` (about 4,800 lines,
about 15 ms), which every Stave build does.

## Why two compile columns

`stave` parallelises inside one invocation: the grammar is LL(1) and every item parses and
checks on its own, so one `stave build` uses every core. A C compiler is one single-threaded
process per translation unit; C build parallelism comes from the build system running many
processes. So the 1-thread column compares compilers on one core (Stave pinned to one worker, C on
one file), and the all-threads column compares each language's own way of using the machine: one
`stave build` against a parallel build of the same program split into translation units, plus
the link.

## How the two programs are kept equal

Both texts come from the same generator calls (`tools/gen-corpus/src/pair.rs`): the same fns,
statements, and expressions, written twice. Every operation is defined and identical in both
languages: unsigned integers only, Stave's wrapping `+% -% *%` for C's unsigned `+ - *`, constant
shifts below the width, `<<%` as `<< (b & (bits - 1))`, division and remainder by `b | 1`, f64
`+ - * /` with divisors `x * x + 1.0` (C99 mode, so no FMA contraction), `as| u32` as a
saturating helper in C, no side effects in expressions (so C's evaluation order does not
matter), bounded loops. Every local is folded into a per-fn accumulator, so an optimizer finds no
dead code, and no operation has only constant operands, so C compilers fold nothing Stave does not
also compute.

`run.sh` enforces it: it builds every configuration (Stave debug and `--release`, every C compiler
and level, single-file and split), runs each, and stops with an error unless all print the same
checksum; kernel outputs must match byte for byte. The C correctness builds use `-pedantic -Wall
-Wextra -Werror` (tcc: `-Wall -Werror`), so the C is also warning-free. Timed builds use the plain
flags above.

## Caveats

- The corpus is synthetic: dense integer expressions, small fns, shallow loops. Real programs
  have other mixes; gcc and clang at `-O2` spend unusually long on it. Lines are about 42 bytes
  long on average.
- A debug build keeps every value that crosses a block in its stack slot and optimizes nothing;
  like every build, it removes the bounds checks it has a proof for.
  `--release` lowers only what `main` reaches, reads each slice's address and length once,
  keeps values that cross blocks in registers,
  folds addresses into loads and stores, puts cold code last, and packs the image. It does not
  inline, hoist loop-invariant code, or build vector code, so it is expected to lose to `-O2`,
  and it does.
- Stave binaries are large: statically linked, with the runtime and its panic reports in every
  program. C binaries are dynamically linked, so their sizes exclude libc.
- The corpus run time measures calls and branches more than arithmetic; the kernels measure
  loops over arrays. Neither uses SIMD: stage0 cannot build vector code yet, and the C compilers
  run at their baseline x86-64 target.
- Times are wall clock of whole processes, including process start. Two runs of the script on an
  otherwise idle machine agree within 1% on compile times and within about 5% on run times,
  except `sieve` built without an optimizer (Stave, tcc, `-O0`), which varies by 10 to 15%
  between runs.
