# Stave

C/Rust-like, SIMD-first systems language. Compiles to static ELF for x86-64 and AArch64 Linux;
PTX for NVIDIA GPUs is designed. Goal: whole programs and the compiler itself run on the GPU.

- Stave is new: read `LANG.md` before writing Stave code. Don't guess syntax from Rust.
- `crates/`: stage0 compiler (Rust). `std/`, `runtime/`: Stave sources linked into every program.
- `contracts/`: the one copy of machine-read contracts (grammar, diagnostic registry, formats).
  `*/generated/` comes from them via `tools/ll1gen` and `tools/diaggen`: run the `gen` step,
  never edit generated files by hand.
- A language change updates `LANG.md`, the grammar, the diagnostic registry, and
  `tests/conformance/` together.
- Build/test through `tools/ci-step/run.sh <step>` (`check`, `test`, `gen`, ...), Linux only.
- `std/`, `runtime/`, and `stage1/src/` stay `stave fmt`-clean: `check` runs `stave fmt --check`
  on them.
- Goldens: `STAVE_BLESS=1` rewrites them. Review the diff.

## Comments

- Comment only what the code can't say: invariants, reasons, non-obvious constraints.
- Never reference tasks, issues, specs, requirement ids, milestones, decisions, dates, or who
  decided. State the rule itself.
- No commented-out code, no change history.
