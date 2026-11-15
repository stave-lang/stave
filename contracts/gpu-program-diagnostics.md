# Contract: GPU program diagnostic codes (v0.3)

New range **E12xx** in the shared registry
([diagnostic-codes.md](diagnostic-codes.md)). Codes are stable and
never reused. Every error carries a suggested fix. Warnings follow the registry's
convention: an `E` code with `severity: "warning"` ([diagnostics.schema.json](diagnostics.schema.json)). They carry a
fix too, and never fail a build.

Reused codes:
- E0704 (missing feature);
- E0801, E0802 (structured threads, Send/Sync): same rules on the GPU;
- E1001 (target not runnable on this host);
- E1101–E1111 (GPU kernels). **E1101 and E1102 apply to `kernel fn` bodies and the device fns
  they reach on every target, including `gpu-sm86`**. They do not apply to `main`'s code or
  region bodies, which get the general features.

| Code | Severity | Rule | Suggested fix |
|---|---|---|---|
| E1201 | error | `extern "C"` function declared or called in a program built for `gpu-sm86`. The call chain from `main` is listed | "GPU programs cannot call C code: all user code runs on the GPU. Write the function in Stave, or build for a CPU target" |
| E1202 | error | A kernel launched from a GPU program with a constant block size B that breaks the virtual-grid rules: B not a multiple of 32, B > R, R % B ≠ 0, or R/B > 15 when B > 32 (named barriers 1–15) | Shows R and the valid sizes, and suggests `[gpu] block = <value>` in `stave.toml` |
| E1203 | error | Invalid `[run]` or `[gpu]` manifest value per the grammar in [gpu-program.md](gpu-program.md) "Manifest keys": a size or percentage that does not parse, `stack` below 4 KiB, `heap_max` below 64 MiB, a percentage outside 1–100, `block` not a multiple of 32 or above 1024, unknown `poll` mode. The primary span is the value in `stave.toml` | Shows the accepted form, for example `stack = "8MiB"` |
| E1204 | error | Shared memory per resident block exceeds the limit: 99 KiB opt-in on sm_86 (101,376 B) minus the runtime's own 256 B, so 101,120 B. The amount is the maximum over the program's kernels of (block-shared bytes × R/B), where B is that kernel's constant block size (R when B is not constant; a launch whose run-time B needs more than the carve-out is then a launch-precondition `Err`). The largest kernel and its share are listed | Lower `[gpu] block` (fewer virtual blocks per resident block), use a larger kernel block size, or reduce the kernel's shared memory |
| E1205 | warning | A recursion cycle (call-graph SCC) is reachable from a `par_for`, `join`, or `scope` body. It runs with explicit frames on the Stave frame stack, per lane. The cycle is listed. Runtime-internal cycles through region `wait` are not reported | Rewrite as a loop, with an explicit stack (`Vec`) if needed |
| E1206 | warning | An indirect call (through a fn value) inside a `par_for`, `join`, or `scope` body. The call site and the fn type are shown | Use `match` over an enum of the cases, or a generic parameter, instead of a fn value |

Conformance programs expect these with the annotations of
[conformance-annotations.md](conformance-annotations.md): `//~ ERROR E12xx` and
`//~ WARNING E1205`/`E1206` in `.stv` files, and `#~ ERROR E1203` on the offending `stave.toml`
line.

Not diagnostics: runtime failures are reported by the shim as described in
[gpu-program.md](gpu-program.md) (exit 101, 125, or a signal). CPU-architecture vector code (wider
than 128 bits, `needs(avx2)` and the like) is not reachable on this target: `dispatch` drops its
arms, and the existing E0704 rejects a direct call, so it needs no extra code.
