#!/bin/sh
# stage1 `tokens`, `items`, `ast`, and `strings` equivalence gate (contracts/stage-gates.md): builds stage0 (the
# debug build `test` leaves, or STAVE_GATE_PROFILE=release), builds stage1 with it, then runs both
# with `build --dump=tokens,items,ast,strings` on every root of the corpus (one run each) and compares
# the two `tokens.txt`, `items.txt`, `ast.txt`, and `strings.txt` byte for byte. Every dump covers the whole program, so
# each root also compares std, runtime, and the arch package.
#
# Roots: every package directory (a `stave.toml`, not descending into it) and every other `.stv`
# file below tests/conformance (fixtures and negative lexical cases included), the stage1 package
# itself, and the stage1 package for aarch64-linux (the other arch package). Paths are absolute:
# stage1 orders packages of one name by their normalized path, stage0 by the canonical one.
# Fails on any difference, a missing dump, or a stage1 exit other than 0.
#   tools/ci-step/run.sh stage1-gate
# Internal: stage1-gate.sh --one N ROOT [TARGET] compares one root (run in parallel by xargs).
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"
OUT="$ROOT/target/stage1-gate"
PROFILE="${STAVE_GATE_PROFILE:-debug}"
S0="$ROOT/target/$PROFILE/stave"
S1="$OUT/stage1"
export STAVE_STD_DIR="$ROOT"

if [ "${1:-}" = "--one" ]; then
  d="$OUT/w/$2"
  mkdir -p "$d"
  rm -f "$d/0/tokens.txt" "$d/1/tokens.txt" "$d/0/items.txt" "$d/1/items.txt" "$d/0/ast.txt" "$d/1/ast.txt" "$d/0/strings.txt" "$d/1/strings.txt"
  if [ -n "${4:-}" ]; then set -- "$2" "$3" --target "$4"; else set -- "$2" "$3"; fi
  root="$2"
  shift 2
  "$S0" build "$root" "$@" -o "$d/exe" --dump=tokens,items,ast,strings --dump-dir "$d/0" >/dev/null 2>&1 || true
  if ! "$S1" build "$root" "$@" --dump=tokens,items,ast,strings --dump-dir "$d/1" >"$d/stage1.err" 2>&1; then
    echo "FAIL stage1 exit: $root $* ($(head -c 200 "$d/stage1.err"))"
  elif [ ! -f "$d/0/tokens.txt" ] || [ ! -f "$d/0/items.txt" ] || [ ! -f "$d/0/ast.txt" ] || [ ! -f "$d/0/strings.txt" ]; then
    echo "FAIL no stage0 dump: $root $*"
  elif ! cmp -s "$d/0/tokens.txt" "$d/1/tokens.txt"; then
    echo "DIFF tokens $root $* (first: $(cmp "$d/0/tokens.txt" "$d/1/tokens.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/items.txt" "$d/1/items.txt"; then
    echo "DIFF items $root $* (first: $(cmp "$d/0/items.txt" "$d/1/items.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/ast.txt" "$d/1/ast.txt"; then
    echo "DIFF ast $root $* (first: $(cmp "$d/0/ast.txt" "$d/1/ast.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/strings.txt" "$d/1/strings.txt"; then
    echo "DIFF strings $root $* (first: $(cmp "$d/0/strings.txt" "$d/1/strings.txt" 2>&1 | head -1))"
  else
    echo "SAME $(wc -l < "$d/0/tokens.txt") $(wc -l < "$d/0/items.txt") $(wc -l < "$d/0/ast.txt") $(wc -l < "$d/0/strings.txt")"
  fi
  exit 0
fi

if [ "$PROFILE" = release ]; then cargo build -q --locked --release -p stave-cli; else cargo build -q --locked -p stave-cli; fi
mkdir -p "$OUT"
"$S0" build "$ROOT/stage1" -o "$S1" >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log"; exit 1; }

# The corpus: one line `N ROOT [TARGET]` per run.
LIST="$OUT/roots.txt"
CONF="$ROOT/tests/conformance"
find "$CONF" -name stave.toml | sed 's|/stave.toml$||' | sort > "$OUT/pkgs.txt"
{
  # a package nested in another one is that package's path dependency, not a root
  awk '{ inside = 0; for (i = 1; i <= n; i++) if (index($0, roots[i] "/") == 1) { inside = 1; break }
         if (!inside) { print; roots[++n] = $0 } }' "$OUT/pkgs.txt"
  find "$CONF" -name '*.stv' | sort | awk -v pk="$OUT/pkgs.txt" '
    BEGIN { while ((getline p < pk) > 0) pkgs[++n] = p }
    { inside = 0; for (i = 1; i <= n; i++) if (index($0, pkgs[i] "/") == 1) { inside = 1; break }
      if (!inside) print }'
  echo "$ROOT/stage1"
  echo "$ROOT/stage1 aarch64-linux"
} | awk '{ print NR, $0 }' > "$LIST"

start=$(date +%s)
xargs -P "$(nproc)" -L 1 sh "$0" --one < "$LIST" > "$OUT/result.txt"
end=$(date +%s)
runs=$(wc -l < "$LIST")
same=$(grep -c '^SAME' "$OUT/result.txt" || true)
tlines=$(awk '$1 == "SAME" { s += $2 } END { print s + 0 }' "$OUT/result.txt")
ilines=$(awk '$1 == "SAME" { s += $3 } END { print s + 0 }' "$OUT/result.txt")
alines=$(awk '$1 == "SAME" { s += $4 } END { print s + 0 }' "$OUT/result.txt")
slines=$(awk '$1 == "SAME" { s += $5 } END { print s + 0 }' "$OUT/result.txt")
echo "stage1-gate: $same of $runs roots equal (tokens $tlines lines, items $ilines lines, ast $alines lines, strings $slines lines), $((end - start)) s"
if grep -v '^SAME' "$OUT/result.txt"; then
  echo "stage1-gate: FAILED" >&2
  exit 1
fi
