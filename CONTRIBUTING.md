# Contributing

## Environment

Build and test on Linux (x86-64 or AArch64): the tests run the ELF binaries the compiler
produces, and `tools/ci-step/run.sh` refuses to run elsewhere. The Rust toolchain is pinned in
`rust-toolchain.toml`; rustup installs it automatically.

## Before a pull request

```sh
tools/ci-step/run.sh check
```

`check` runs, in order:

- `fmt`: `cargo fmt --all --check`
- `clippy`: `cargo clippy --workspace --all-targets -- -D warnings`
- `test`: `cargo test --workspace`
- `gen-check`: the generated tables are up to date

Each step can be run alone (`run.sh test -p stave-cli`; extra arguments go to cargo).

## Rules

- Language changes update `LANG.md`, the grammar (`contracts/grammar.ebnf`), the diagnostic
  registry (`contracts/diagnostics.toml` and `contracts/diagnostic-codes.md`), and
  `tests/conformance/` together.
- Files under `*/generated/` are never edited by hand. Change the source in `contracts/` and run
  `tools/ci-step/run.sh gen`.
- Golden files: `STAVE_BLESS=1` rewrites them. Review the diff before committing it.
- Comments say only what the code cannot: invariants, reasons, non-obvious constraints. They do
  not refer to tasks, issues, milestones, or history, and there is no commented-out code.

## Licence

Contributions are licensed under the same terms as the project: [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option, with no additional terms.
