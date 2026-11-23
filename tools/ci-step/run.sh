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
  stave-fmt) cargo run -q --locked -p stave-cli -- fmt --check std runtime stage1/src ;;
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
  lang-budget)
           cargo run -q --locked -p lang-budget -- LANG.md --limit 15000 ;;
  check)   "$0" fmt; "$0" stave-fmt; "$0" clippy; "$0" test; "$0" interp-gate; "$0" gen-check; "$0" lang-budget ;;
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
  *)       echo "usage: run.sh fmt|stave-fmt|clippy|build|test|interp-gate|interp-gate-slow|gen|gen-check|lang-budget|check|unsafe-diff|agent-eval ARGS|bench [ARGS]|cargo ARGS" >&2; exit 2 ;;
esac
