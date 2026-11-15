#!/bin/sh
# Unsafe-report diff gate: lists every unsafe site of the shipped
# std/runtime and the repository's packages (`stave unsafe-report --json`, contracts/cli.md) and
# prints the added and removed sites against the base revision ($STAVE_UNSAFE_BASE, default
# origin/main), which CI shows in the run log. Fails if a report cannot be produced (a site
# without an obligation is E0013 and fails the report). Without a base revision (first run,
# shallow clone without main, a tree copied without .git) the full head report is printed.
# Run through tools/ci-step/run.sh unsafe-diff.
set -eu
cd "$(dirname "$0")/../.."
BASE="${STAVE_UNSAFE_BASE:-origin/main}"
OUT=target/unsafe-diff
PKGS="tests/conformance/borrow/unsafe_report"
mkdir -p "$OUT"

report() { # report <stave binary> <tree root> <file>
  : > "$3"
  for p in $PKGS; do
    if [ -e "$2/$p" ]; then
      STAVE_STD_DIR="$2" "$1" unsafe-report "$2/$p" --json >> "$3"
    fi
  done
  sort -u -o "$3" "$3"
}

ROOT="$PWD"
cargo build -q --locked -p stave-cli
report "$ROOT/target/debug/stave" "$ROOT" "$OUT/head.json"
echo "unsafe-report: $(wc -l < "$OUT/head.json") sites at HEAD"

if [ -d .git ] && { git rev-parse -q --verify "$BASE^{commit}" >/dev/null 2>&1 \
     || git fetch -q --depth=1 origin main:refs/remotes/origin/main 2>/dev/null; } \
   && git rev-parse -q --verify "$BASE^{commit}" >/dev/null 2>&1; then
  WT="$ROOT/$OUT/base"
  git worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
  git worktree add -q --detach "$WT" "$BASE"
  BIN="$ROOT/target/unsafe-base-build/debug/stave"
  if (cd "$WT" && cargo build -q --locked -p stave-cli --target-dir "$ROOT/target/unsafe-base-build") \
     && report "$BIN" "$WT" "$OUT/base.json" 2>/dev/null; then
    echo "unsafe-report diff against $BASE (- removed, + added):"
    diff "$OUT/base.json" "$OUT/head.json" | sed -n 's/^< /- /p; s/^> /+ /p' || true
  else
    echo "unsafe-report: no report at $BASE (no \`stave unsafe-report\` there); full HEAD report:"
    cat "$OUT/head.json"
  fi
  git worktree remove --force "$WT" 2>/dev/null || true
else
  echo "unsafe-report: no base revision $BASE; full HEAD report:"
  cat "$OUT/head.json"
fi
