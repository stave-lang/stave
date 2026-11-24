#!/bin/sh
# stage1 `tokens`, `items`, `ast`, and `diags` equivalence gate (contracts/stage-gates.md): builds
# stage0 (the debug build `test` leaves, or STAVE_GATE_PROFILE=release), builds stage1 with it, then
# runs both with `build --dump=tokens,items,ast,diags` on every root of the corpus (one run each) and
# compares the two `tokens.txt`, `items.txt`, and `ast.txt` byte for byte, and stage1's `diags.txt`
# with stage0's restricted to the codes of the stages stage1 has (S1_CODES: lex, read). Every dump
# covers the whole program, so each root also compares std, runtime, and the arch package.
#
# Exit status: stage1 exits 1 exactly when it printed a diagnostic, else 0 (a root with an error
# of a stage stage1 lacks still exits 0 there). Where stage0's diagnostics are all in S1_CODES and
# there is one, stage0's exit status must be stage1's.
#
# Roots: every package directory (a `stave.toml`, not descending into it) and every other `.stv`
# file below tests/conformance (fixtures and negative lexical cases included), the stage1 package
# itself, and the stage1 package for aarch64-linux (the other arch package). Paths are absolute:
# stage1 orders packages of one name by their normalized path, stage0 by the canonical one.
# Fails on any difference, a missing dump, or a stage1 exit that does not follow its diagnostics.
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
# Codes of the stages stage1 has so far; add a stage's codes with its dump.
S1_CODES='E0001|E0002|E0003|E0006|E0015|E0017|E0018|E0024|E0029|E1003'

if [ "${1:-}" = "--one" ]; then
  d="$OUT/w/$2"
  mkdir -p "$d"
  rm -f "$d/0/tokens.txt" "$d/1/tokens.txt" "$d/0/items.txt" "$d/1/items.txt" "$d/0/ast.txt" "$d/1/ast.txt" "$d/0/diags.txt" "$d/1/diags.txt"
  if [ -n "${4:-}" ]; then set -- "$2" "$3" --target "$4"; else set -- "$2" "$3"; fi
  root="$2"
  shift 2
  s0=0
  s1=0
  "$S0" build "$root" "$@" -o "$d/exe" --dump=tokens,items,ast,diags --dump-dir "$d/0" >/dev/null 2>&1 || s0=$?
  "$S1" build "$root" "$@" --dump=tokens,items,ast,diags --dump-dir "$d/1" >"$d/stage1.err" 2>&1 || s1=$?
  if [ "$s1" -gt 1 ]; then
    echo "FAIL stage1 exit $s1: $root $* ($(head -c 200 "$d/stage1.err"))"
  elif [ ! -f "$d/0/tokens.txt" ] || [ ! -f "$d/0/items.txt" ] || [ ! -f "$d/0/ast.txt" ] || [ ! -f "$d/0/diags.txt" ]; then
    echo "FAIL no stage0 dump: $root $*"
  elif [ ! -f "$d/1/diags.txt" ]; then
    echo "FAIL no stage1 diags.txt: $root $* ($(head -c 200 "$d/stage1.err"))"
  elif ! cmp -s "$d/0/tokens.txt" "$d/1/tokens.txt"; then
    echo "DIFF tokens $root $* (first: $(cmp "$d/0/tokens.txt" "$d/1/tokens.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/items.txt" "$d/1/items.txt"; then
    echo "DIFF items $root $* (first: $(cmp "$d/0/items.txt" "$d/1/items.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/ast.txt" "$d/1/ast.txt"; then
    echo "DIFF ast $root $* (first: $(cmp "$d/0/ast.txt" "$d/1/ast.txt" 2>&1 | head -1))"
  else
    grep -E "^\{\"kind\":\"diagnostic\",\"code\":\"($S1_CODES)\"" "$d/0/diags.txt" > "$d/0/diags.in" || true
    ours=$(wc -l < "$d/0/diags.in")
    all=$(wc -l < "$d/0/diags.txt")
    if ! cmp -s "$d/0/diags.in" "$d/1/diags.txt"; then
      echo "DIFF diags $root $* (first: $(cmp "$d/0/diags.in" "$d/1/diags.txt" 2>&1 | head -1))"
    elif [ "$s1" != "$([ "$ours" -gt 0 ] && echo 1 || echo 0)" ]; then
      echo "FAIL stage1 exit $s1 with $ours diagnostics: $root $*"
    elif [ "$ours" -gt 0 ] && [ "$ours" = "$all" ] && [ "$s0" != "$s1" ]; then
      echo "FAIL exit stage0 $s0 stage1 $s1: $root $*"
    else
      echo "SAME $(wc -l < "$d/0/tokens.txt") $(wc -l < "$d/0/items.txt") $(wc -l < "$d/0/ast.txt") $ours"
    fi
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
dlines=$(awk '$1 == "SAME" { s += $5 } END { print s + 0 }' "$OUT/result.txt")
echo "stage1-gate: $same of $runs roots equal (tokens $tlines lines, items $ilines lines, ast $alines lines, diags $dlines lines), $((end - start)) s"
if grep -v '^SAME' "$OUT/result.txt"; then
  echo "stage1-gate: FAILED" >&2
  exit 1
fi
