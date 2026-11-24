# Contracts

Machine-read and cross-compiler contracts for Stave. This is the only copy: tools (`tools/ll1gen`,
`tools/diaggen`), tests, and CI read these files directly, and stage0 (Rust, CPU) and stage1 (Stave,
GPU) both implement them.

| File | What |
|---|---|
| `grammar.ebnf` | surface grammar, LL(1)-checked; source of the parse tables |
| `ir-text.ebnf` | syntax of the checked-IR text (one line, one statement), LL(1) |
| `opt-rules.txt` | release optimizer rewrite rules (planned, to be read by `tools/rulegen`) |
| `diagnostic-codes.md` | the one diagnostic registry, human table (E0001–E13xx) |
| `diagnostics.toml` | the same registry, machine-readable: message, label, fix templates (read by `tools/diaggen`) |
| `parse-errors.md` | when E0011–E0014 and the Rust forms E0019–E0028 replace E0004; the E0004 expected set and found token (both parsers) |
| `conformance-annotations.md` | `//~` / `#~` test annotations read by `stave test` |
| `diagnostics.schema.json` | JSON diagnostics, timings, and the `gpu-sm86` parallelism report |
| `cli.md` | `stave` command line |
| `simd-ops.md` | portable vector ops and lowering classes (stave-target's tests check `crates/stave-target/tables/` against it) |
| `checked-ir.md` | checked IR: levels, the meaning of every line of its text, canonical order, loading |
| `lang-details.md` | fine print behind one-sentence rules of `LANG.md`: output formats, case lists, reasons |
| `soundness.md` | the argument for each static rule whose failure would let an accepted program misbehave |
| `known-limits.md` | what the compiler does not guarantee, what it rejects without need, and what it rejects on purpose, with examples and rewrites |
| `borrow-model.md` | what a program may do with references at run time (tags, permissions per byte); the static borrow rules must imply it, and the interpreter's borrow tracking reports against it; the side table the tracking mode's lowering adds to a body |
| `intrinsics.md` | `extern "stave-intrinsic"` operations and program entry |
| `gpu-lang.md`, `gpu-ops.md`, `gpu-diagnostics.md` | GPU kernels (PTX) |
| `gpu-program.md`, `host-channel.md`, `gpu-program-diagnostics.md` | whole programs on the GPU (`gpu-sm86`) |
| `cli-stage1.md`, `resident-protocol.md`, `stage-gates.md` | stage1 GPU-resident compiler, dumps, equivalence gates |
