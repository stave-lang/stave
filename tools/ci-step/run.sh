#!/bin/sh
# Single entry point for every CI step. Runs on Linux only: tests execute the ELF binaries the
# compiler produces.
#   tools/ci-step/run.sh <step> [args...]
# Env: STAVE_REQUIRE_LEGS=v2,v3,aarch64 makes a native test leg that cannot run here (CPU feature,
# qemu-aarch64) fail instead of being skipped; CI sets it, unset skips (tests/support/legs.rs).
# Leg v4 (AVX-512) runs natively or under the Intel SDE `sde64` that STAVE_SDE names; CI does
# not list it.
set -eu
if [ "$(uname -s)" != Linux ]; then
  echo "run.sh: refusing to run on $(uname -s): steps run on Linux" >&2
  exit 2
fi
cd "$(dirname "$0")/../.."
export PATH="$HOME/.cargo/bin:$PATH"
step="${1:-}"; [ $# -gt 0 ] && shift
case "$step" in
  fmt)     cargo fmt --all --check ;;
  stave-fmt) cargo run -q --locked -p stave-cli -- fmt --check std runtime stage1/src stage1/prims stage1/ir stage1/opt ;;
  clippy)  cargo clippy --locked --workspace --all-targets -- -D warnings ;;
  build)   cargo build --locked --workspace "$@" ;;
  test)    cargo test --locked --workspace "$@" ;;
  # The accepted programs under borrow tracking (minutes);
  # `test` leaves this ignored test out.
  interp-gate)
           cargo test --locked -p stave-driver --test interp_gate -- --ignored --nocapture --exact \
             accepted_programs_run_clean_under_borrow_tracking ;;
  # The same with the slow programs too (nightly).
  interp-gate-slow)
           cargo test --locked -p stave-driver --test interp_gate -- --ignored --nocapture --exact \
             with_the_slow_programs ;;
  gen-check)
           cargo run -q --locked -p ll1gen -- --check
           cargo run -q --locked -p diaggen -- --check
           cargo run -q --locked -p gen-tables -- --check ;;
  gen)     cargo run -q --locked -p ll1gen
           cargo run -q --locked -p diaggen
           cargo run -q --locked -p gen-tables ;;
  # stage1 dumps and diagnostics against stage0's on the conformance corpus and stage1 itself.
  stage1-gate) sh tools/ci-step/stage1-gate.sh ;;
  # stage1's IR library against stage0's IR text: round trip of every dumped text, mutant verdicts.
  stage1-ir-gate) sh tools/ci-step/stage1-ir-gate.sh ;;
  # The unit tests of the stage1 library packages and of stage1 itself.
  stage1-test) cargo run -q --locked -p stave-cli -- test stage1/prims/seq && cargo run -q --locked -p stave-cli -- test stage1/ir && cargo run -q --locked -p stave-cli -- test stage1/opt && cargo run -q --locked -p stave-cli -- test stage1 ;;
  lang-budget)
           cargo run -q --locked -p lang-budget -- LANG.md --limit 15000 ;;
  # No new class of fns with the same code in the shipped trees (stage1 with ir, prims, std, runtime
  # for x86_64-linux) beyond tools/ci-step/dup-baseline.txt; STAVE_BLESS=1 rewrites the baseline
  # (review the diff: a new line is a new duplicate).
  dup-gate)
           set -- dup stage1 --all --target x86_64-linux
           if [ -n "${STAVE_BLESS:-}" ]; then
             cargo run -q --locked -p stave-cli -- "$@" --write-baseline tools/ci-step/dup-baseline.txt
           else
             cargo run -q --locked -p stave-cli -- "$@" --baseline tools/ci-step/dup-baseline.txt
           fi ;;
  # `check` in two halves that need nothing from each other, so they can run on two machines.
  check-test)
           "$0" fmt; "$0" stave-fmt; "$0" clippy; "$0" test ;;
  check-gates)
           "$0" interp-gate; "$0" stage1-gate; "$0" stage1-ir-gate; "$0" stage1-test; "$0" gen-check; "$0" lang-budget; "$0" dup-gate ;;
  check)   "$0" check-test; "$0" check-gates ;;
  unsafe-diff) sh tools/ci-step/unsafe-diff.sh ;;
  # Scores a directory of answers (`--answers DIR`, `DIR/<task>/answer.stv`) against
  # tests/agent-eval with the release compiler; the result goes to target/agent-eval/. `test`
  # runs the self-test over the reference answers.
  agent-eval)
           cargo build -q --locked --release -p stave-cli -p agent-eval
           target/release/agent-eval --stave target/release/stave --out target/agent-eval/result.json \
             --work target/agent-eval/work "$@" ;;
  bench)   sh bench/run.sh "$@" ;;
  cargo)   cargo "$@" ;;
  *)       echo "usage: run.sh fmt|stave-fmt|clippy|build|test|interp-gate|interp-gate-slow|stage1-gate|stage1-ir-gate|stage1-test|gen|gen-check|lang-budget|dup-gate|check|check-test|check-gates|unsafe-diff|agent-eval ARGS|bench [ARGS]|cargo ARGS" >&2; exit 2 ;;
esac
