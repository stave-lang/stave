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
- `kernels/<k>_disp.stv`: `dispatch fn`s with four arms: `needs(v4)` (512-bit kernels),
  `needs(avx2)` (256-bit kernels), `needs(sve)` (the 128-bit kernels again: the AArch64 backend
  emits the same code under `needs(sve)`, so these arms only give an AArch64 binary a pick at run
  time) and `_` (the 128-bit kernels of `base`); built for every target. Only kernels whose
  results do not depend on the vector width qualify (`bytesearch_disp`, the integer sums of
  `sum_disp`, and `reduce_disp`, whose float sums fix the logical lane count; the float sums
  of `sum` use the vector's lane count), so every arm must print the same. See "Dispatch runs".
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
| bytesearch_disp | bytesearch through `dispatch fn`s: 512-bit `needs(v4)`, 256-bit `needs(avx2)`, `needs(sve)` and 128-bit `_` arms | dispatch, per-function feature sets, `mask8x64` bits |
| sum_disp | wrapping i32 and widened u8 sums through `dispatch fn`s with the arms of bytesearch_disp | dispatch, 512-bit `widen_lo/hi`, `reduce_add_wrap` |
| reduce | std's f32 / f64 `sum` and `dot` against the two-level rule (blocks of 4096 by the logical-lane rule, 32 / 16 lanes from -0.0, then the block tree), written out one term at a time; the same sums from 128-bit kernels per block (last block first, combined by the tree's recursive split), from std on the root's two children under `thread.join`, and from std on two-block pieces under `thread.par_for_mut`; lengths to 6L + 1 and around and past several blocks | `load_masked`, `select` of `-0.0` into a tail, the 8-vector fold, `reduce_add`, threads |
| reduce_disp | reduce's blocks through `dispatch fn`s with the arms of bytesearch_disp (the L lanes in 2 x 512-bit and 4 x 256-bit vectors), combined by the tree's passes and by its split, and std's own one- and two-level `sum` / `dot` arms | dispatch, 512-bit masked loads and `reduce_add`, std's `v4` / `avx2` / `_` arms |
| prefix | in-place inclusive prefix sums, f32 / f64 / i32 | `scan_add`, `scan_add_wrap`, `permute`, `&mut` chunk binding |
| histogram | 256-bin byte histogram | `gather`, `scatter` (inside `emulate`), lane-private bins |
| matmul4 | 4x4 matrix times 4-element points | `permute` broadcasts, `join` (v3), `select` |
| utf8 | UTF-8 validation (Keiser–Lemire), code points | `lookup`, `>>`, `-|`, `reduce_or` |
| combinators | the `std.slice` methods on f32 / f64 / i32 / u8 (no `_v3` file: std picks its loops; these slices are mostly shorter than the 512 B to 1 KiB where its 256-bit loops start, which `tests/conformance/simd/combinators_wide.stv` covers) | `sum min max find find_byte find_any_byte count any all eq fill copy_from` |
| stream_copy | copies f32 / f64 / i32 / u8 into a misaligned destination | `store_stream` (vector and scalar), `write_stream`, `fence_stream`, prefetches past the end |

## Cases

For each case kind of a kernel (element type and lane count N): every length from 0 to 3N + 1,
each at two misalignments from a 64-byte boundary (`STAVE_DIFF_FULL=1`: every misalignment),
then `STAVE_DIFF_RANDOM` (default 24) cases of random length up to 3N + 257 and random
misalignment. Each case draws its values from one palette: a few distinct values, edge values
(±0, ±inf, NaNs with payloads, subnormals, extremes), moderate random values, or raw bits. The
`reduce` kernels add four: all `-0.0` (dot: times 1.0), cancelling values of either sign from
2^-20 to 2^21, ±MAX at every eighth element among ±1 (overflow and inf + -inf depend on the lane
count), and moderate values with NaNs; their kinds also list lengths at and past one block of
4096, each from the cancelling palette and from a step palette (2^12 in the first block, 2^-12
after it) whose block sums round differently in any other block order. `harness.rs` checks
that their cases tell the rule from L/2 and 2L lanes, the sequential sum, `+0.0` lane starts,
the one-level rule, half and twice the block, and block sums added left to right, and NaN-ness
between lane counts.
`STAVE_DIFF_SEED` (default `0x5eed`) seeds the generator; `STAVE_DIFF_KERNELS=sum,dot` runs
only those programs.

## Checks

For every leg (the interpreter, then native debug and release builds for x86_64-linux,
x86_64-linux-v3 and aarch64-linux; the `_v3` programs for x86_64-linux-v3 only):

- the program exits 0 with one line per case and nothing on stderr;
- on every line the vector tokens equal the reference tokens: two NaNs of one float type are
  equal whatever their sign and payload (unspecified, LANG.md §3), every other token compares
  bit for bit (`-0.0` is not `0.0`), and an `h` hash replaces each NaN by one canonical NaN first;
- every native line equals the interpreter's;
- for `utf8`, `bytesearch`, `sum_disp` and the `reduce` kernels, the reference tokens equal
  what the harness computes in Rust;
- in `reduce_disp` runs on x86-64, std's `sum` / `dot` dispatch fns pick the run's arm too
  (`std.slice.<f>_v4`, `_v3`, or `_`);
- outputs keep their guard elements (a flag token).

A failing line is shown with the case's head and inputs, and with the program rerun on that case
alone with element-by-element output. aarch64 runs under `qemu-aarch64`; a leg that cannot run
here is reported and skipped. A build that stops with "not implemented yet" fails the leg. Legs
that are known to fail go in `KNOWN` in `harness.rs`, with the reason; they are reported, not
failed, and an entry that starts passing fails the test until it is removed.

Default run: about 8,200 cases over the 24 programs, 186 legs, about 87,000 checked lines, about
3 min on an 8-core x86-64 host with Intel SDE (most of it the `reduce` cases past one block,
element by element in the interpreter and under SDE). `STAVE_DIFF_FULL=1 STAVE_DIFF_RANDOM=256` checks
several hundred thousand lines in several minutes.

## Dispatch runs

A `_disp` kernel's binary for a target runs several times, with `STAVE_DISPATCH_TRACE=1`: every
trace line of its dispatch fns must name the arm the run's CPU allows, and the output must equal
the interpreter's (the `_` arms) and the references. The same x86_64-linux binary runs natively
(the widest arm of the host CPU), natively with `STAVE_CPU=-avx2` (`_`), on the v4 leg (natively
on an AVX-512 CPU or under Intel SDE: `v4`), there with `STAVE_CPU` removing the five AVX-512
names or only `avx512cd` (`avx2`), and under `qemu-x86_64 -cpu Nehalem` (`_`; skipped without
qemu-x86_64). The x86_64-linux-v3 binary picks `avx2` natively and `v4` on the v4 leg. The
aarch64-linux binary (the x86-64 arms are left out of it) picks `sve` under
`qemu-aarch64 -cpu max` and `_` there with `STAVE_CPU=-sve` and under `-cpu cortex-a76`.
