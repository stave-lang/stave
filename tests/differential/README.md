# Differential tests of SIMD kernels

Hand-written vector kernels checked against element-by-element scalar references, in the IR
interpreter and in native builds of every CPU target. Run by `cargo test -p stave-driver --test
differential` (part of `cargo test --workspace`); Linux, x86-64 host for the native legs.

## Files

- `kernels/<k>.stv`: one program per kernel family. Its vector kernels use 128-bit vectors and
  build for every target; each has a scalar reference that does the same arithmetic in the same
  order (see the file's head comment). `main` reads the cases and prints, per case, the vector
  results, ` =`, then the reference results.
- `kernels/<k>_v3.stv`: the same kernels with 256-bit vectors, x86_64-linux-v3 only. They call
  the references of `<k>.stv`, which the harness adds to the package as module `base`.
- `kernels/<k>_disp.stv`: `dispatch fn`s whose `needs(avx2)` arms are the 256-bit kernels and
  whose `_` arms are the 128-bit kernels of `base`, built for every target; the x86_64-linux
  builds run natively (the 256-bit arms on an AVX2 host) and under `qemu-x86_64 -cpu Nehalem`
  (the `_` arms). Only kernels whose results do not depend on the lane count qualify
  (`bytesearch_disp`).
- `diff.stv`: module `diff` of every kernel package: reads `cases.bin`, places each input array
  a given number of elements past a 64-byte boundary with guard elements around it, and prints
  result tokens (`u` / `i` integers, `f` / `d` float bits, `h` array hashes; with any
  command-line argument, arrays element by element).
- `gen.rs`: the case generator (SplitMix64).
- `harness.rs`: the test (`[[test]] differential` of `crates/stave-driver`).

| kernel | what | vector ops it exercises |
|---|---|---|
| sum | f32 tree and ordered sums, f64, wrapping i32, u8 widened to u32 | chunk loops, masked and scalar tails, `reduce_add`, `reduce_add_ordered`, `widen_lo/hi` |
| dot | f32, f64, i32, u8 dot products | `load`, `load_masked`, `first_n`, widening multiplies |
| minmax | f32 `min max min_num max_x86`, f64, i32, u8 | float NaN and signed-zero rules, `reduce_min/max`, `select` |
| filter | elements above a threshold, f32 / i32 / u8 | `compress`, `store_masked`, `to_bits`, `popcount` |
| bytesearch | first and last index of a byte, count | `to_bits`, scalar `ctz` / `clz`, backward loads |
| bytesearch_disp | bytesearch through `dispatch fn`s: 256-bit `needs(avx2)` arms, 128-bit `_` arms | dispatch, per-function feature sets |
| prefix | in-place inclusive prefix sums, f32 / f64 / i32 | `scan_add`, `scan_add_wrap`, `permute`, `&mut` chunk binding |
| histogram | 256-bin byte histogram | `gather`, `scatter` (inside `emulate`), lane-private bins |
| matmul4 | 4x4 matrix times 4-element points | `permute` broadcasts, `join` (v3), `select` |
| utf8 | UTF-8 validation (Keiser–Lemire), code points | `lookup`, `>>`, `-|`, `reduce_or` |
| combinators | the `std.slice` methods on f32 / f64 / i32 / u8 (no `_v3` file: std picks its loops; these slices stay below the 1 KiB where its 256-bit loops start, which `tests/conformance/simd/combinators_wide.stv` covers) | `sum min max find find_byte find_any_byte count any all eq fill copy_from` |
| stream_copy | copies f32 / f64 / i32 / u8 into a misaligned destination | `store_stream` (vector and scalar), `write_stream`, `fence_stream`, prefetches past the end |

## Cases

For each case kind of a kernel (element type and lane count N): every length from 0 to 3N + 1,
each at two misalignments from a 64-byte boundary (`STAVE_DIFF_FULL=1`: every misalignment),
then `STAVE_DIFF_RANDOM` (default 24) cases of random length up to 3N + 257 and random
misalignment. Each case draws its values from one palette: a few distinct values, edge values
(±0, ±inf, NaNs with payloads, subnormals, extremes), moderate random values, or raw bits.
`STAVE_DIFF_SEED` (default `0x5eed`) seeds the generator; `STAVE_DIFF_KERNELS=sum,dot` runs
only those programs.

## Checks

For every leg (the interpreter, then native debug and release builds for x86_64-linux,
x86_64-linux-v3 and aarch64-linux; the `_v3` programs for x86_64-linux-v3 only):

- the program exits 0 with one line per case and nothing on stderr;
- on every line the vector tokens equal the reference tokens (two NaNs are equal whatever their
  bits; NaN payloads are unspecified);
- every native line equals the interpreter's;
- for `utf8` and `bytesearch`, the reference tokens equal what the harness computes in Rust;
- outputs keep their guard elements (a flag token).

A failing line is shown with the case's head and inputs, and with the program rerun on that case
alone with element-by-element output. aarch64 runs under `qemu-aarch64`; a leg that cannot run
here is reported and skipped. A build that stops with "not implemented yet" fails the leg. Legs
that are known to fail go in `KNOWN` in `harness.rs`, with the reason; they are reported, not
failed, and an entry that starts passing fails the test until it is removed.

Default run: about 4,300 cases over the 19 programs, 99 legs, about 18,000 checked lines, about
15 s on an 8-core x86-64 host. `STAVE_DIFF_FULL=1 STAVE_DIFF_RANDOM=256`: about 377,000 checked
lines, about 5 minutes.
