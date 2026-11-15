# Contract: `stave` CLI ↔ resident stage1 daemon (internal, versioned by build id)

**Transport**: served by the daemon through the `gpu-sm86` Unix-socket server service
([host-channel.md](host-channel.md) opcodes 50–55). The stage1 image uses the `gpu-sm86` shim variant
with `ptx_compile`. A Unix stream socket at `$XDG_RUNTIME_DIR/stave/stage1-<build id>.sock` (fallback
`/tmp/stave-<uid>/…`). The directory has mode 0700, and only the owning uid may connect, checked with
`SO_PEERCRED`. Messages are length-prefixed (u32 little-endian) CBOR maps. The x86-64 build of
stage1 (used to cross-check stage1 on the GPU against the same code on the CPU) speaks the same messages over stdin/stdout, one
connection, started with `--stdio`.

**Build id and image**:
- build id = the first 16 lowercase hex digits of XXH3-128 over, in this order: the sorted list of
  (path, XXH3-128 of contents) of every file under `crates/`, `stage1/`, `contracts/`, `tools/`
  plus `Cargo.toml` and `Cargo.lock`; the `rustc -V` line; the pinned `libnvptxcompiler` version
  string. Compiler sources, toolchain, and PTX compiler version therefore all change it.
- The image is `<stave install>/lib/stave/stage1-<build id>.img`, where `<stave install>` is the
  directory above the `stave` binary's `bin/`. It is built by stage0 in release mode:
  `stave build stage1 --target gpu-sm86 --release --compiler cpu`.

**Daemon lifetime**:
- `stave` starts the daemon (`stave __stage1-daemon`, which runs the `gpu-sm86` shim with the stage1
  image) when it is absent and the GPU is usable.
- When no request is active, the daemon parks its grid in the shim's idle mode (no busy polling). It exits
  after `[compiler] idle_exit` idle (default 30 minutes), on `Shutdown` (`stave daemon stop`), or when
  its build id differs from the client's. In that last case the client sends `Shutdown` and starts a
  matching daemon. A mismatched daemon is never used.
- **Config**: `[compiler]` in `$XDG_CONFIG_HOME/stave/config.toml`, overridden per project by
  `[compiler]` in `stave.toml` ([cli-stage1.md](cli-stage1.md) "`[compiler]` config"). Read by the launcher when it
  starts the daemon.
- **Memory**: the launcher sets the run-time override `STAVE_GPU_HEAP_MAX=<bytes|N%>`
  ([gpu-program.md](gpu-program.md)) in the shim's environment from `[compiler] gpu_mem` (default
  `50%` of free device memory), so the daemon never takes the `gpu-sm86` default of 90%. The shim
  resolves it before launch. The daemon reads the resolved cap from the runtime config
  (`RuntimeConfig.heap_max`) and free memory through the `mem_info` service (opcode 24) for E1301,
  `HelloOk`, and `StatusOk`. An evicted session's memory is returned through the non-blocking
  release path (opcode 31), so other sessions keep running. Run `stave daemon stop` before GPU benchmarks.
- One daemon per user per GPU. Concurrent requests are serialized per session and may overlap
  across sessions, if device memory allows. Two builds at the same time both stay correct,
  because sessions share no mutable state.

| Message | Direction | Fields |
|---|---|---|
| `Hello` | client → daemon | `build_id`, `protocol: 1` |
| `HelloOk` / `Mismatch` | daemon → client | `build_id`, `device` (name, cc, free memory from `mem_info`), `heap_max` |
| `Build` | client → daemon | `root`, `command: build\|check\|test\|run-build`, `options` (target, release, features, dump list, diagnostics mode), `files: [{path, mtime, size, hash128}]` for every package file. `hash128` = XXH3-128 of the file's bytes, as a 16-byte CBOR byte string in canonical (big-endian) order |
| `Progress` | daemon → client | `stage`, `ms` (only with `--timings`) |
| `Diag` | daemon → client | one diagnostic JSON object, in canonical order ([diagnostics.schema.json](diagnostics.schema.json)) |
| `Done` | daemon → client | `exit: 0\|1\|3`, `artifact` path, `summary` line, `timings` (per stage: ms, items, rounds, fallbacks), `ran_on: gpu` |
| `Fail` | daemon → client | `exit: 3`, `report` (device panic report, [gpu-program.md](gpu-program.md) "Exit status"). No partial artifact is ever written. Sent by the shim's panic sink (below) |
| `Status` | client → daemon | — |
| `StatusOk` | daemon → client | `build_id`, `device`, `heap_max`, `heap_used`, `sessions`, `idle_s`, `gpu_mem`, `idle_exit` (the values the daemon started with) |
| `Evict` | client → daemon | `root`. Drops every session of that root (memory through the non-blocking release path); answered with `Done{exit: 0, summary: "evicted <n> sessions"}` |
| `Shutdown` | client → daemon | — |

While stage1 implements no stage yet, `Build` is answered with
`Done{exit: 0, summary: "stage1: no stages", timings: [], ran_on: gpu}`, no `artifact`, and no
`Diag`.

**File access**: the daemon reads sources and writes artifacts itself, through its shim, with the
client's paths. The client sends hashes so the daemon can skip unchanged files without reading
them. A file whose hash differs from the daemon's copy is reread.

**Errors**:
- **Panic**: at the start of each build the daemon registers the client's open socket with the shim's
  panic sink (opcode 42). The registered frame prefix is the CBOR encoding of the map
  header `{kind: "Fail", exit: 3, report:` (3 entries); the shim appends the report as a CBOR text
  string, so the frame is a well-formed `Fail` message. On a device panic the shim sends the report as `Fail` on that socket,
  then exits 101 (the panic exit code). The client reports exit 3 with the report.
- **EOF** without `Done` or `Fail` (device fault, signal, shim crash): the client reports exit 3,
  "stage1 daemon died", and the daemon log path (`$XDG_RUNTIME_DIR/stave/stage1-<build id>.log`).
- Never a silent fallback to stage0. The next `stave build` starts a fresh daemon.

**Test hooks** (CI only): honoured only when `STAVE_STAGE1_TEST=1` is in the environment of the
client that runs the command and, for daemon-side hooks, of the client that started the daemon
(the launcher passes it on). Without it the client ignores every variable below, and the daemon
ignores every `test_` option and writes one log line per ignored option.

| Variable (client env) | Effect |
|---|---|
| `STAVE_STAGE1_TEST_PANIC_AT=<stage>` | Sent as `Build.options.test_panic_at`: the daemon panics on the device when that stage starts (schema stage name) |
| `STAVE_STAGE1_TEST_LEX_SPARSE_SLOTS=<k>` | Sent as `Build.options.test_lex_sparse_slots`: the lexer's sparse composed functions keep at most k slots instead of 8, which forces the dense rerun |
| `STAVE_STAGE1_IMAGE=<path>` | The launcher starts the daemon from this image instead of `lib/stave/stage1-<build id>.img`, with the build id read from the image (bootstrap) |
| `STAVE_STAGE1_TEST_PROBE=<reason>` | Selection probes report this reason (`no NVIDIA driver`, `no device`, `compute capability 7.5 < 8.6`, `out of device memory`, `stage1 image missing`) without touching the device |
