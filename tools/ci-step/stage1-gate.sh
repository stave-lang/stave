#!/bin/sh
# stage1 `tokens`, `items`, `ast`, `strings`, `names`, `sigs`, and `diags` equivalence gate
# (contracts/stage-gates.md): builds stage0 (the debug build `test` leaves, or
# STAVE_GATE_PROFILE=release), builds stage1 with it, then runs both once per root of the corpus
# with `build --dump=tokens,items,ast,strings,names,sigs,diags` and compares the `tokens.txt`,
# `items.txt`, `ast.txt`, `strings.txt`, `names.txt`, and `sigs.txt` of the two byte for byte,
# and stage1's `diags.txt` with stage0's restricted to the codes of the stages stage1 has
# (S1_CODES: read, lex, item cut, parse, modules, decls, resolve, signatures). Every dump covers
# the whole program, so each root
# also compares std, runtime, and the arch package.
#
# Stage0 raises some of those codes in later stages too (a type or trait expected: E0101; a field
# given twice in a literal or pattern: E0105; a private field or column: E0106). The shapes with
# a message of their own (LATER) are cut from stage0's lines; a private field or method reached
# by a value has E0106's plain message, so the E0106 lines of stage1 need only be among stage0's.
#
# `sigs`: stage1 writes no `sigs.txt` for a root whose signatures need a type layout (`size_of` /
# `align_of` in an array length or const argument; layouts are a later stage), and says so with
# one fixed stderr line. Those roots are counted, and the count must be SIGS_LAYOUT_SKIPS exactly,
# so it cannot grow unnoticed; every other root's `sigs.txt` must be stage0's.
#
# Exit status: stage1 exits 1 exactly when it printed a diagnostic, else 0 (a root with an error
# of a stage stage1 lacks still exits 0 there). Where stage0's diagnostics are all stage1's and
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
# Codes of the stages stage1 has so far; add a stage's codes with its dump. E0005 is stage1's only
# as the bracket depth form (the item cut); its expression and type forms are later stages'.
S1_CODES='E0001|E0002|E0003|E0004|E0006|E0011|E0012|E0013|E0014|E0015|E0016|E0017|E0018|E0019|E0020|E0021|E0022|E0023|E0024|E0025|E0026|E0027|E0028|E0029|E0101|E0102|E0103|E0104|E0105|E0106|E0107|E0410|E0514|E0515|E0520|E0521|E0708|E0709|E0811|E1003|E1404'
# Messages of the shapes of those codes that later stages raise.
# Roots whose `sigs` stage1 does not write because a signature needs a layout.
SIGS_LAYOUT_SKIPS=1
SIGS_LAYOUT_LINE='stage1: sigs.txt not written: a signature needs a type layout'
LATER='"code":"E0101".*"message":"[^"]*` is not a (type|trait)"|"code":"E0105".*"message":"field `[^"]*` of `[^"]*` is (given twice in one struct literal|named twice in one pattern)"|"code":"E0106".*"message":"(column `|struct `)'

if [ "${1:-}" = "--one" ]; then
  d="$OUT/w/$2"
  mkdir -p "$d"
  for k in 0 1; do
    rm -f "$d/$k/tokens.txt" "$d/$k/items.txt" "$d/$k/ast.txt" "$d/$k/strings.txt" "$d/$k/names.txt" "$d/$k/sigs.txt" "$d/$k/diags.txt"
  done
  if [ -n "${4:-}" ]; then set -- "$2" "$3" --target "$4"; else set -- "$2" "$3"; fi
  root="$2"
  shift 2
  s0=0
  s1=0
  dumps=tokens,items,ast,strings,names,sigs,diags
  "$S0" build "$root" "$@" -o "$d/exe" --dump=$dumps --dump-dir "$d/0" >/dev/null 2>&1 || s0=$?
  "$S1" build "$root" "$@" --dump=$dumps --dump-dir "$d/1" >"$d/stage1.err" 2>&1 || s1=$?
  if [ "$s1" -gt 1 ]; then
    echo "FAIL stage1 exit $s1: $root $* ($(head -c 200 "$d/stage1.err"))"
  elif [ ! -f "$d/0/tokens.txt" ] || [ ! -f "$d/0/items.txt" ] || [ ! -f "$d/0/ast.txt" ] || [ ! -f "$d/0/strings.txt" ] || [ ! -f "$d/0/names.txt" ] || [ ! -f "$d/0/sigs.txt" ] || [ ! -f "$d/0/diags.txt" ]; then
    echo "FAIL no stage0 dump: $root $*"
  elif [ ! -f "$d/1/strings.txt" ] || [ ! -f "$d/1/names.txt" ] || [ ! -f "$d/1/diags.txt" ]; then
    echo "FAIL no stage1 dump: $root $* ($(head -c 200 "$d/stage1.err"))"
  elif ! cmp -s "$d/0/tokens.txt" "$d/1/tokens.txt"; then
    echo "DIFF tokens $root $* (first: $(cmp "$d/0/tokens.txt" "$d/1/tokens.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/items.txt" "$d/1/items.txt"; then
    echo "DIFF items $root $* (first: $(cmp "$d/0/items.txt" "$d/1/items.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/ast.txt" "$d/1/ast.txt"; then
    echo "DIFF ast $root $* (first: $(cmp "$d/0/ast.txt" "$d/1/ast.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/strings.txt" "$d/1/strings.txt"; then
    echo "DIFF strings $root $* (first: $(cmp "$d/0/strings.txt" "$d/1/strings.txt" 2>&1 | head -1))"
  elif ! cmp -s "$d/0/names.txt" "$d/1/names.txt"; then
    echo "DIFF names $root $* (first: $(cmp "$d/0/names.txt" "$d/1/names.txt" 2>&1 | head -1))"
  elif [ ! -f "$d/1/sigs.txt" ] && ! grep -qxF "$SIGS_LAYOUT_LINE" "$d/stage1.err"; then
    echo "FAIL no stage1 sigs: $root $* ($(grep '^stage1: ' "$d/stage1.err" | head -c 200))"
  elif [ -f "$d/1/sigs.txt" ] && ! cmp -s "$d/0/sigs.txt" "$d/1/sigs.txt"; then
    echo "DIFF sigs $root $* (first: $(cmp "$d/0/sigs.txt" "$d/1/sigs.txt" 2>&1 | head -1))"
  else
    if [ -f "$d/1/sigs.txt" ]; then sigs="$(wc -l < "$d/0/sigs.txt") -"; else sigs="0 L"; fi
    grep -E "^\{\"kind\":\"diagnostic\",\"code\":\"(($S1_CODES)\"|E0005\",\"severity\":\"error\",\"message\":\"brackets)" "$d/0/diags.txt" | grep -vE "$LATER" > "$d/0/diags.in" || true
    priv='^{"kind":"diagnostic","code":"E0106"'
    grep -v "$priv" "$d/0/diags.in" > "$d/0/diags.rest" || true
    grep -v "$priv" "$d/1/diags.txt" > "$d/1/diags.rest" || true
    grep "$priv" "$d/0/diags.in" > "$d/0/diags.priv" || true
    grep "$priv" "$d/1/diags.txt" > "$d/1/diags.priv" || true
    extra=$(grep -cvxFf "$d/0/diags.priv" "$d/1/diags.priv" || true)
    ours=$(wc -l < "$d/1/diags.txt")
    all=$(wc -l < "$d/0/diags.txt")
    if ! cmp -s "$d/0/diags.rest" "$d/1/diags.rest"; then
      echo "DIFF diags $root $* (first: $(cmp "$d/0/diags.rest" "$d/1/diags.rest" 2>&1 | head -1))"
    elif [ "$extra" -gt 0 ]; then
      echo "DIFF diags E0106 of stage1 not among stage0's: $root $*"
    elif [ "$s1" != "$([ "$ours" -gt 0 ] && echo 1 || echo 0)" ]; then
      echo "FAIL stage1 exit $s1 with $ours diagnostics: $root $*"
    elif [ "$ours" -gt 0 ] && [ "$ours" = "$all" ] && [ "$s0" != "$s1" ]; then
      echo "FAIL exit stage0 $s0 stage1 $s1: $root $*"
    else
      echo "SAME $(wc -l < "$d/0/tokens.txt") $(wc -l < "$d/0/items.txt") $(wc -l < "$d/0/ast.txt") $(wc -l < "$d/0/strings.txt") $(wc -l < "$d/0/names.txt") $ours $sigs"
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
slines=$(awk '$1 == "SAME" { s += $5 } END { print s + 0 }' "$OUT/result.txt")
nlines=$(awk '$1 == "SAME" { s += $6 } END { print s + 0 }' "$OUT/result.txt")
dlines=$(awk '$1 == "SAME" { s += $7 } END { print s + 0 }' "$OUT/result.txt")
glines=$(awk '$1 == "SAME" { s += $8 } END { print s + 0 }' "$OUT/result.txt")
gsame=$(awk '$1 == "SAME" && $9 == "-" { s += 1 } END { print s + 0 }' "$OUT/result.txt")
gskip=$(awk '$1 == "SAME" && $9 == "L" { s += 1 } END { print s + 0 }' "$OUT/result.txt")
echo "stage1-gate: $same of $runs roots equal (tokens $tlines lines, items $ilines lines, ast $alines lines, strings $slines lines, names $nlines lines, sigs $gsame roots $glines lines, sigs skipped (layout) $gskip, diags $dlines lines), $((end - start)) s"
if grep -v '^SAME' "$OUT/result.txt"; then
  echo "stage1-gate: FAILED" >&2
  exit 1
fi
if [ "$gskip" != "$SIGS_LAYOUT_SKIPS" ]; then
  echo "stage1-gate: sigs skipped (layout) $gskip, expected $SIGS_LAYOUT_SKIPS" >&2
  exit 1
fi
