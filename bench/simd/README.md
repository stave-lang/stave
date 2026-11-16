# SIMD kernels benchmark

Run time of explicit Stave vector kernels against plain C99 twins. `bench/simd.sh` (or
`bench/run.sh --simd`, `tools/ci-step/run.sh bench --simd`) prints a markdown report on stdout.

```sh
sh bench/simd.sh             # 3 runs per build
sh bench/simd.sh --runs 5
```

Linux on x86-64 only. Needs the Rust toolchain and any of `gcc`, `clang`; a compiler that is not
installed is skipped and named in the report. x86_64-linux-v3 builds run only on a CPU with
AVX2, FMA and BMI2. Work files go to `target/bench-simd/`.

## Layout

- `lanes128/`, `lanes256/`: library packages `lanes` with the kernels, written with 128-bit and
  256-bit vectors.
- `<kernel>/`: a program (`src/main.stv`) that makes its data with a fixed LCG, calls
  `lanes.<kernel>` many times, and prints one hash; `main.c` is its C twin. Each manifest binds
  `lanes` to `lanes128`, and `[target.x86_64-linux-v3.dependencies]` rebinds it to `lanes256`.

| kernel | data | per pass |
|---|---|---|
| sum, dot | 32,768 f32 | sum / dot product, eight lane accumulators |
| minmax | 32,768 f32 | least and greatest element |
| filter | 32,768 f32 | elements above a threshold, packed in order |
| bytesearch | 65,536 bytes | count of a byte and its first index |
| prefix | 32,768 u32 | wrapping inclusive prefix sums into a second array |
| histogram | 65,536 bytes | 256-bin histogram (gather, scatter) |
| matmul4 | 8,192 points of 4 f32 | 4x4 matrix times every point, in place |
| utf8 | 65,536 bytes of 1- to 4-byte UTF-8 | validation |

## Equal results

Every build of a kernel must print the same hash, or the script stops. Float results are equal
bit for bit because both sides do the same operations in the same order:

- sums keep eight lane accumulators: lane k adds elements k, k + 8, .. in order, and the lanes
  reduce as Stave's fixed tree (lane k + lane k + 4, then k + k + 2, then 0 + 1). The 128-bit
  kernels hold the eight lanes as two vectors and add them first, which is the tree's first
  step; the C twins write the eight accumulators out;
- C is built with `-std=c99 -ffp-contract=off` (no FMA contraction); Stave never contracts;
- `minmax` data has no NaN and no -0.0, so any comparison order gives the same result;
  `matmul4` computes every lane as ((m0 x0 + m1 x1) + m2 x2) + m3 x3 on both sides;
- integer kernels are exact in any order.

The C twins are the plain loops a C programmer writes (`filter` branch-free); whether the C
compiler vectorizes them is reported, not arranged.

## Columns and rows

One column per kernel, the wall time of the whole process (median of `--runs`), in ms. Rows:

| row | build |
|---|---|
| stave x86_64-linux [--release] | `stave build --target x86_64-linux [--release]`: SSE4.2, up to 128-bit vectors |
| stave x86_64-linux-v3 [--release] | `stave build --target x86_64-linux-v3 [--release]`: AVX2, up to 256-bit vectors |
| gcc -O2 | `gcc -std=c99 -march=x86-64-v3 -ffp-contract=off -O2`: the comparison |
| gcc -O3, clang -O2 | the same flags, for information |

A cell reads `not built` when stage0 stops on that kernel with "not implemented yet" (the
report lists the message), and `cannot run` when the CPU lacks the target's features. After the
table, a second table says for each twin whether gcc's `-fopt-info-vec-optimized` reports a
vectorized loop inside the kernel's functions, at `-O2` and at `-O3`.

## Caveats

- Stage0 keeps every vector value in a stack slot between operations: each vector op loads its
  operands from memory and stores its result. Expect Stave to lose to `gcc -O2`, and 256-bit
  code to gain little or nothing over 128-bit code; the numbers measure that gap, not the
  language.
- `histogram` uses gather and scatter, which lower to scalar loops on x86_64-linux (both) and
  x86_64-linux-v3 (scatter).
- Times include process start and data generation (a few ms).
