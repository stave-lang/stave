#!/bin/sh
# Single entry point for every CI step. Runs on Linux only: tests execute the ELF binaries the
# compiler produces.
#   tools/ci-step/run.sh <step> [args...]
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
  clippy)  cargo clippy --locked --workspace --all-targets -- -D warnings ;;
  build)   cargo build --locked --workspace "$@" ;;
  test)    cargo test --locked --workspace "$@" ;;
  gen-check)
           cargo run -q --locked -p ll1gen -- --check
           cargo run -q --locked -p diaggen -- --check ;;
  gen)     cargo run -q --locked -p ll1gen
           cargo run -q --locked -p diaggen ;;
  check)   "$0" fmt; "$0" clippy; "$0" test; "$0" gen-check ;;
  unsafe-diff) sh tools/ci-step/unsafe-diff.sh ;;
  bench)   sh bench/run.sh "$@" ;;
  cargo)   cargo "$@" ;;
  *)       echo "usage: run.sh fmt|clippy|build|test|gen|gen-check|check|unsafe-diff|bench [ARGS]|cargo ARGS" >&2; exit 2 ;;
esac
