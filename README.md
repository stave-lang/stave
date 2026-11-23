# Stave

A systems language with C/Rust-like syntax, SIMD in the type system, and memory safety without
lifetimes. Compiles to static ELF for x86-64 and AArch64 Linux; a PTX backend for NVIDIA GPUs is
designed. Goal: whole programs, and the compiler itself, run on the GPU.

**Status: pre-alpha, an experiment.** Not for production use. The stage0 compiler (Rust) builds and
runs programs on x86-64 and AArch64: front end, borrow checker, views, closures, threads, debug code
generation. SIMD code is only partly type-checked; vector code is generated for x86-64
(SSE4.2 and AVX2 levels) and AArch64 (Advanced SIMD). The optimizer is not started. GPU support is design
only. Parts of `contracts/` and `LANG.md` describe designed but unbuilt things (the GPU runtime,
optimizer rules, planned diagnostics); they say so where they do.

## Goals

- **Fast compilation.** Compile time is a feature of the language, not only of the compiler. The
  grammar is LL(1) and every item parses on its own, so the front end and the checkers run in
  parallel. There is no libc and no external linker: the compiler writes the static ELF itself.
- **SIMD first.** Vector types, chunk loops with a mandatory tail, and CPU dispatch are language
  constructs checked by the type system, not intrinsics and not a hope that a loop
  auto-vectorizes. *Status: vector code generation on x86-64 and AArch64; a `dispatch`
  function picks its arm from the CPU features read at start.*
- **The GPU as a runtime target.** Whole programs run on the GPU, not only kernels: heap,
  recursion, function values, and I/O through a thin host shim. First target: PTX for NVIDIA.
  *Status: design.*
- **The GPU as a compilation backend.** The stage1 compiler is to be written in Stave as a
  data-parallel program and run on the GPU itself; stage0 stays as the bootstrap and the reference
  it is checked against. This already constrains the language: syntax and data model must be
  parallel-parseable. *Status: design.*
- **Safe and small.** Memory safety without lifetime annotations, and a reference short enough
  for a coding agent to load whole (`LANG.md`).

## Known limits

Stave is pre-alpha.

- Memory safety is the aim, not yet a guarantee: the borrow checker is young, and soundness bugs
  are expected. A safe program that corrupts memory is a compiler bug; please report it.
- The checker is stricter than the rules need: some valid programs are rejected.
- Generated code is slow next to `-O2`: nothing is inlined, hoisted out of loops, or vectorized.
- "Safe code cannot deadlock" covers ranked locks and thread joins only.
- Linux only. No C interop yet.

The full list, with examples and rewrites, is in
[contracts/known-limits.md](contracts/known-limits.md).

## Roadmap

In order: checker soundness (adversarial testing, fuzzing) → the rest of vector code generation
(CPU dispatch) and a release back end → GPU kernels → whole programs on the GPU → the
compiler on the GPU.

## Benchmark

One generated 101,042-line program, written identically in Stave and C99 (same checksum in every
build), and three small kernels. Milliseconds, wall clock; C at 16 threads = 81 files built with
`xargs -P 16`, then linked. Details, flags, and caveats: [bench/README.md](bench/README.md).

| compiler | compile, 1 thread | compile, 16 threads | binary, MB | corpus run | nbody | hashmix | sieve |
|---|---:|---:|---:|---:|---:|---:|---:|
| stave | 1010 | 237 | 7.4 | 211 | 240 | 295 | 1106 |
| stave --release | 1147 | 269 | 5.5 | 164 | 126 | 214 | 572 |
| tcc | 83 | 34 | 3.1 | 174 | 331 | 322 | 823 |
| gcc -O0 | 6476 | 813 | 2.8 | 154 | 305 | 308 | 945 |
| gcc -O2 | 35050 | 3782 | 1.4 | 40 | 89 | 119 | 216 |
| clang -O0 | 3303 | 591 | 2.9 | 156 | 325 | 306 | 808 |
| clang -O2 | 21126 | 2371 | 2.0 | 31 | 88 | 120 | 258 |

i7-10700F (8 cores / 16 threads), Debian 13, gcc 14.2.0, clang 19.1.7, tcc 0.9.27; 101,042-line
corpus (seed 1); median of 5. On one thread Stave compiles 6.4x faster than gcc -O0 and 12x slower
than tcc. A debug build runs at about `-O0` speed (its sieve time moves between 1030 and 1280 ms
from one session to the next). A `--release` build compiles 14% slower than a debug build and runs
1.4x (nbody) to 4x (the call-heavy corpus) slower than gcc `-O2`.
C binaries exclude libc (dynamically linked).

### SIMD kernels

Nine kernels written with explicit vector ops in Stave, each against a plain C99 twin of the same
algorithm built with `-march=x86-64-v3 -ffp-contract=off`, the instruction set of
`x86_64-linux-v3`. Every build prints the same output. Milliseconds, whole process, median of 7.
Details: [bench/simd/README.md](bench/simd/README.md).

| build | sum | dot | minmax | filter | bytesearch | prefix | histogram | matmul4 | utf8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stave x86_64-linux-v3 --release | 139 | 152 | 294 | 251 | 51 | 123 | 294 | 77 | 162 |
| stave x86_64-linux --release | 142 | 185 | 585 | 732 | 100 | 156 | 248 | 142 | 245 |
| gcc -O2 | 140 | 142 | 1097 | 325 | 464 | 188 | 50 | 74 | 491 |
| gcc -O3 | 140 | 142 | 1098 | 324 | 464 | 187 | 50 | 45 | 492 |
| clang -O2 | 138 | 142 | 1119 | 292 | 282 | 173 | 50 | 95 | 499 |

What the table compares: vector code written by hand in Stave with what a C compiler makes of a
scalar loop. C with hand-written intrinsics is not measured.

- gcc vectorizes the twins of sum, dot and matmul4 by itself (and one of the two bytesearch
  loops). There Stave is within 7% of `gcc -O2`, and 1.7x behind `gcc -O3` on matmul4.
- gcc leaves minmax, filter, prefix, utf8 and the bytesearch search loop scalar. There the
  explicit vector code is 1.3x to 9x faster than `gcc -O2`. The utf8 twin is a scalar decoder, a
  different algorithm from the table-driven vector one.
- histogram is 6x slower. AVX2 has no scatter, so the vector form runs lane by lane inside
  `emulate { }`; on this target a scalar loop is the right code.
- The `x86_64-linux` row uses 128-bit SSE4.2 vectors, a smaller instruction set than the C rows.
- utf8 moves by about 10% with where its code lands in the binary: the same instructions run in
  160 or 180 ms depending on the page offset of the kernel function.

## A taste

Blocks are statements, enum variants are always qualified, arithmetic traps on overflow, and
every lossy conversion is spelled out:

```stave
import std.io;

enum Shape { Circle(f64), Rect { w: f64, h: f64 } }

fn area(s: Shape) -> f64 {
    return match s {
        Shape.Circle(r) => 3.14159 * r * r,
        Shape.Rect.{ w, h } => w * h,
    };
}

fn main() -> i32 {
    let a = area(Shape.Rect.{ w: 2.0, h: 3.0 });
    io.print(f"area = {a}\n");                      // area = 6.0
    let big: i64 = 300;
    let b: u8 = big as| u8;                         // saturating cast: 255
    return 0;
}
```

SIMD (stage0 builds this): vector types, chunk loops with a mandatory tail, and runtime CPU
dispatch:

```stave
fn sum_base(xs: &[f32]) -> f32 {
    let mut acc = f32x4.splat(0.0);
    for v: f32x4 in xs { acc = acc + v; } tail masked (v, m) { acc = acc + m.select(v, f32x4.splat(0.0)); }
    return acc.reduce_add();
}

fn sum_v3(xs: &[f32]) -> f32 needs(avx2) {
    let mut acc = f32x8.splat(0.0);
    for v: f32x8 in xs { acc = acc + v; } tail masked (v, m) { acc = acc + m.select(v, f32x8.splat(0.0)); }
    return acc.reduce_add();
}

dispatch fn sum(xs: &[f32]) -> f32 { avx2 => sum_v3, _ => sum_base, }
```

GPU (design only: not working code, there is no `std.gpu`): kernels take typed device memory,
and a launch borrows its buffers until `wait`:

```text
import std.gpu;

kernel fn saxpy(t: gpu.Thread1, a: f32, x: gpu.In<f32>, y: gpu.Out<f32>) {
    let i = t.global();
    match y.at(t) {
        Option.Some(yi) => { *yi = a.fma(x[i], *yi); }
        Option.None => {}
    }
}

fn run_gpu(hx: &[f32], hy: &mut [f32]) -> Result<(), gpu.Error> needs(cuda) {
    let dev = gpu.Device.open(0)?;
    let s = dev.stream()?;
    let dx = dev.alloc_copy(hx)?;
    let mut dy = dev.alloc_copy(hy)?;
    saxpy.launch(&s, gpu.Grid1.cover(hx.len(), 256), 2.0, &dx, &mut dy)?.wait()?;
    s.copy_to_host(&dy, hy)?.wait()?;
    return Result.Ok(());
}
```

`LANG.md` is the full language reference, written for people and coding agents.

## Build

Linux only (x86-64 / AArch64); `stave run` executes the built ELF natively. Needs the Rust
toolchain pinned in `rust-toolchain.toml` (1.100.0; rustup installs it automatically).

```sh
cargo build --release --locked -p stave-cli          # target/release/stave
target/release/stave run tests/conformance/run/fib   # a package: stave.toml + src/main.stv
tools/ci-step/run.sh check                           # fmt, clippy, tests, generated-code check
```

## Layout

- `LANG.md`: language reference. `contracts/`: grammar, diagnostic registry, file formats.
- `crates/`: stage0, the compiler in Rust (bootstrap and reference). `crates/stave-opt` holds the IR
  passes and the register assignment of `--release`. `std/`, `runtime/`: Stave sources.
- `stage1/`: the planned compiler written in Stave; only generated tables so far, no compiler yet.
- `tests/conformance/`: language conformance suite. `fuzz/`: cargo-fuzz targets for the front end.
  `tools/`: generators and CI steps.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
