# Contract: CLI additions for stage1 (extends [cli.md](cli.md))

| Flag / command | Meaning |
|---|---|
| `--compiler auto\|cpu\|gpu` | `auto` (default): stage1 if a usable GPU and a matching stage1 image exist, otherwise stage0. `gpu` without a usable GPU → E1302, exit 1. `cpu` always runs stage0 |
| `--dump=STAGE,...` `--dump-dir DIR` | Stage dumps ([stage-gates.md](stage-gates.md)), implemented identically by both compilers |
| `stave daemon status\|stop\|evict [PKG_DIR]` | Show or stop the resident stage1 daemon, or drop its sessions for one package root ([resident-protocol.md](resident-protocol.md)). Run `stave daemon stop` before GPU benchmarks: an idle daemon still holds device memory |
| `--timings` | Adds `compiler: gpu (stage1, <device>)` or `compiler: cpu (stage0)`, and per stage `ms`, `items`, `rounds`, `fallbacks`. Fallbacks are the number of times a stage left its parallel path for a slower one, always on the device (dense lexer rerun, chunk-local error re-parse, forward-backward SCC, warp worklist, match-task split) |

**`stage1 ir check|fmt FILE`** (the stage1 binary's IR text verbs, run directly; `stave ir` stays
stage0's): load FILE as `stave ir` does (checked-ir.md "Loading" steps 1-3: lines, structure,
verification; not the removal of proven checks or the bounds audit) with stage1's IR library
(`stage1/ir`), and for `fmt` print the module's text on stdout. Faults print as for `stave ir`
(`FILE:LINE:COL: error: MSG`, `FILE:LINE: verify: MSG`, at most 100) and exit 1; an unreadable
FILE or a bad command line exits 2. The stage1 IR gate (`tools/ci-step/run.sh stage1-ir-gate`)
holds `fmt` to stage0's printed text byte for byte and both verbs to stage0's verdict on mutated
text.

**`stage1 ir opt [--passes=base|full] [--dump=inline] [--dump-phase=inline|clean|base]
[--verify-phases] [--limit NAME=N]... FILE`** (the stage1 binary's optimizer, run directly; `stave ir opt` stays
stage0's base passes): loads FILE with stage1's IR library (checked-ir.md "Loading" steps 1-3);
the header must say `release` and level `checked` (exit 1 otherwise). FILE's module is made
canonical (checked-ir.md "Canonical module") before the first round. It runs the release passes
of [opt-passes.md](opt-passes.md), `full` (the default: rule P1) or `base` (rule P3), and prints
the module at level `opt` on stdout in the file's form: the stack section of an emit-form file is
printed unchanged, so `stave ir emit --release`, `stage1 ir opt`, `stave ir build` builds with
the front end's stack bounds.

- `--dump=inline`: prints the decision tables (opt-passes.md "The `inline` dump") on stdout
  instead of the module. With `--passes=base` (no decision runs) it is a usage error.
- `--dump-phase=P`: prints, instead of the output, the canonical module after phase P of the
  last round, at level `opt` and in the file's form: `inline` after the expansion, `clean` after
  the clean-up, `base` after the base passes (the output itself). A phase that `--passes` does not
  run is a usage error.
- `--verify-phases`: stage1's IR verifier also runs on the canonical module after every phase of
  every round, not only on the module printed (the output, or the `--dump-phase` module).
- `--limit NAME=N`: for this run, the `[limits]` entry `NAME` of `opt.toml` (`threshold`, `cap`,
  `call_saved`, `slot_callee`, `slot_caller`, `rounds`) is the decimal `N` (opt-passes.md K1);
  for goldens and tuning sweeps. A value that breaks opt-passes.md K3, or `rounds=0`, is a usage
  error.

Faults in FILE print as for `stave ir` and exit 1; an unreadable FILE or a bad command line
(an unknown flag or value, `--dump` with `--dump-phase`) exits 2. A verifier fault in a phase's
output is a compiler bug: `FILE: internal error: <phase>: @<inst>: MSG` on stderr (`<phase>` one
of `inline clean base`; `@<inst>: ` names the fn the fault is in, and is absent for a fault outside
every fn), nothing on stdout, exit 3.

**Commands that stage1 does not run**: `stave unsafe-report`, `stave lowering`, and `stave explain` always
run in stage0, whatever `--compiler` says. They compile no program and print no summary line.
`stave fix` checks in stage0 too (its rounds check edits that are not written yet); it takes no
`--compiler` and prints no summary line.

**Summary line** (always printed, except under `--diagnostics json`):
- `compiled on GPU (stage1, NVIDIA RTX A6000) in 412 ms`
- `compiled on CPU (no usable GPU: <reason>)`, where `<reason>` is one of `stage1 image missing`,
  `no NVIDIA driver`, `no device`, `compute capability 7.5 < 8.6`, `daemon not reachable`, or
  `out of device memory`
- `compiled on CPU (--compiler cpu)`
- `compiled on CPU (target <T> not supported by stage1)`, e.g. `target aarch64-linux`
- `compiled on CPU (C linking not supported by stage1)`

**Selection under `auto`** (client side, before any `Build` is sent):
1. Probe in the order of [cli.md](cli.md) (authoritative): target supported by stage1
   (code-generating commands `build`, `run`, `test`; not AArch64 in v1) → the package links no C
   objects (its declared C-link inputs) → stage1 image present with this build id →
   NVIDIA driver loads and reports a device → compute capability ≥ 8.6 → daemon reachable or
   startable. The first failing step is the reason, and no stage1 work starts. Under `--compiler
   gpu` the same steps report E1303, E1304, and E1302, so E1303 comes before E1302.
2. Memory (after a passing probe): estimate = the E1301 formula applied to the package's total
   source bytes (the `size` fields of `Build.files`). If estimate > `heap_max` (from `HelloOk`; with
   no daemon running, the value a new daemon would get: `[compiler] gpu_mem` resolved against free
   memory) → stage0 with reason `out of device memory`. Under `--compiler gpu` no such probe runs:
   the build goes to stage1, which reports E1301 after lexing with the exact estimate.

Once a `Build` is sent, every failure is a diagnostic or exit 3, never a stage0 run.

**`[compiler]` config**: `$XDG_CONFIG_HOME/stave/config.toml` (default `~/.config/stave/config.toml`),
table `[compiler]`, keys `gpu_mem` (bytes such as `"24GiB"`, or a percentage of free device memory
such as `"50%"`; default `"50%"`) and `idle_exit` (duration such as `"30m"`; default `"30m"`). A
`[compiler]` table in the project's `stave.toml` overrides the user file key by key. The values take
effect when a daemon starts. A running daemon keeps its values (`stave daemon status` shows them);
`stave daemon stop` applies a change at the next build.

**`stave daemon status`**: exit 0. Not running: one line `daemon: not running`. Running: these lines,
in this order (`key: value`, one space after the colon):

```text
daemon: running
build id: 3f2a9c0d17e4b852
pid: 41733
device: NVIDIA RTX A6000 (cc 8.6)
heap_max: 25769803776 bytes
heap_used: 1073741824 bytes
sessions: 2
idle: 12 s
gpu_mem: 50%
idle_exit: 30m
socket: /run/user/1000/stave/stage1-3f2a9c0d17e4b852.sock
log: /run/user/1000/stave/stage1-3f2a9c0d17e4b852.log
```

`pid` comes from `SO_PEERCRED` on the socket; the other values from `StatusOk`.

**`stave daemon evict [PKG_DIR]`**: drops every session of that package root (default: the current
package). The next build of it is a clean build in a warm daemon (how compile times are
measured). Exit 0, also when no daemon or session exists.

**Diagnostics (new range E13xx, GPU compiler)**: registered in the shared registry,
[diagnostic-codes.md](diagnostic-codes.md). Each is a diagnostic, so the exit code is 1. "No
location" means the diagnostic has no `primary` span and sorts first ([diagnostics.schema.json](diagnostics.schema.json)).

| Code | Location | Rule | Fix |
|---|---|---|---|
| E1301 | none | project needs more device memory than the daemon may use (estimate after lexing: needed vs the daemon's `heap_max`, bounded by free memory from `mem_info`, [host-channel.md](host-channel.md)). Shows needed and available bytes | "use `--compiler cpu`, raise `[compiler] gpu_mem`, or free GPU memory (`stave daemon stop` frees the daemon's own)" |
| E1302 | none | `--compiler gpu` but no usable GPU (the reason as in the summary line) | "use `--compiler auto` or `--compiler cpu`; or make the GPU usable (<reason>): install the NVIDIA driver, check the device, or reinstall Stave with the GPU compiler image"; for compute capability below 8.6, "use `--compiler auto` or `--compiler cpu`: stage1 needs compute capability 8.6 or higher" |
| E1303 | none | `--compiler gpu` with a code-generating command (`build`, `run`, `test`) for a target stage1 does not support (AArch64 in v1). `stave check` is not affected: it runs the target-independent stages in stage1 for any target | "use `--compiler auto` or `--compiler cpu`: stage0 builds this target" |
| E1304 | the package's first C-link input in `stave.toml` (or the first `extern "C"` import block if none is declared) | `--compiler gpu` and the package links C objects (through mold/lld), which stage1 v1 does not support. `stave check` is not affected. Under `auto` such packages go to stage0 before stage1 starts, judged from the declared C-link inputs (summary reason `C linking not supported by stage1`); the C-link mechanism itself is still open (LANG.md §8) | "use `--compiler auto` or `--compiler cpu`: stage0 links C objects" |

A device-side panic of the compiler is exit code 3 with the device panic report, and no artifact is
written ([resident-protocol.md](resident-protocol.md) "Errors").
