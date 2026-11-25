#!/bin/sh
# stage1 IR text gate (contracts/checked-ir.md; stage-gates.md `lir`, `ir`, `opt`): stage1's IR
# library reads, verifies, and prints the IR text stage0 prints, byte for byte, and on mutated text
# its verdict is stage0's.
#
# Round trip: builds stage0 (the debug build `test` leaves, or STAVE_GATE_PROFILE=release) and
# stage1 with it, then per root of the corpus and per target (x86_64-linux, aarch64-linux) runs
# `stave check --dump=lir,ir` and `stave check --release --dump=lir,ir,opt`, and `stage1 ir fmt`
# on every text written (a root with error diagnostics writes none): its output must be the input.
# Counts per target, mode, and level. An empty file is no text: stage0 writes one for a program
# whose lowering it does not implement; such files are counted apart.
#
# Mutants: STAVE_IR_MUTANTS (default 4000) mutants of the round trip's texts of at most 64 KiB,
# written with stage0's verdicts by the ignored stave-ir test `ir_text_mutants` (the reader fuzz
# test's mutator); `stage1 ir check` on each must agree: accepted (stage0: accepted, or refused
# only by the bounds audit, a step stage1 does not have), refused in lines or structure (an
# `error:` line), or refused by the verifier (a `verify:` line).
#
# Optimizer goldens: `stage1 ir opt --dump=inline` on every tests/opt case with an `inline.txt`,
# under the case's `limits` (each line one `--limit`), on its input and on the input's
# `stage1 ir fmt` text: both equal `inline.txt` byte for byte.
#
# Corpus: the roots of the stage1 gate (every package and other `.stv` file below
# tests/conformance), the bench programs, and the stage1 package itself (the largest text: its
# read + print time is printed against stage0's `stave ir fmt`).
#   tools/ci-step/run.sh stage1-ir-gate
# Internal: stage1-ir-gate.sh --one N ROOT compares one root (run in parallel by xargs).
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"
OUT="$ROOT/target/stage1-ir-gate"
PROFILE="${STAVE_GATE_PROFILE:-debug}"
S0="$ROOT/target/$PROFILE/stave"
S1="$OUT/stage1"
export STAVE_STD_DIR="$ROOT"
MUTANTS="${STAVE_IR_MUTANTS:-4000}"

if [ "${1:-}" = "--one" ]; then
  d="$OUT/w/$2"
  root="$3"
  for t in x86_64-linux aarch64-linux; do
    for m in debug release; do
      mkdir -p "$d/$t-$m"
      rm -f "$d/$t-$m/lir.txt" "$d/$t-$m/ir.txt" "$d/$t-$m/opt.txt"
    done
    "$S0" check "$root" --target "$t" --dump=lir,ir --dump-dir "$d/$t-debug" >/dev/null 2>&1 || true
    "$S0" check "$root" --target "$t" --release --dump=lir,ir,opt --dump-dir "$d/$t-release" >/dev/null 2>&1 || true
    for m in debug release; do
      for lv in lir ir opt; do
        f="$d/$t-$m/$lv.txt"
        [ -f "$f" ] || continue
        if [ ! -s "$f" ]; then
          echo "EMPTY $t $m $lv $f"
          continue
        fi
        if "$S1" ir fmt "$f" > "$f.s1" 2> "$f.err"; then
          if cmp -s "$f" "$f.s1"; then
            echo "SAME $t $m $lv $(wc -c < "$f") $f"
          else
            echo "DIFF $t $m $lv $f (first: $(cmp "$f" "$f.s1" 2>&1 | head -1))"
          fi
        else
          echo "FAIL $t $m $lv $f ($(head -c 300 "$f.err"))"
        fi
      done
    done
  done
  exit 0
fi

if [ "$PROFILE" = release ]; then cargo build -q --locked --release -p stave-cli; else cargo build -q --locked -p stave-cli; fi
mkdir -p "$OUT"
"$S0" build "$ROOT/stage1" -o "$S1" >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log"; exit 1; }

fail=0
ocases=0
osame=0
orounds=0
for c in "$ROOT"/tests/opt/*/; do
  [ -f "$c/inline.txt" ] || continue
  ocases=$((ocases + 1))
  lims=""
  if [ -f "$c/limits" ]; then lims=$(sed 's/^/--limit /' "$c/limits" | tr '\n' ' '); fi
  o="$OUT/opt-$(basename "$c")"
  # Only round 1 is built: a case with `rounds` above 1 is skipped, and only when stage1 refuses
  # it for that reason (exit 2 and the one-round message).
  if [ -f "$c/limits" ] && grep -Eq '^rounds=0*([2-9]|[1-9][0-9]+)$' "$c/limits"; then
    st=0
    # shellcheck disable=SC2086
    "$S1" ir opt --dump=inline $lims "$c/input.ir" > "$o.inline" 2> "$o.err" || st=$?
    if [ "$st" = 2 ] && grep -q 'only --dump=inline of one round is built' "$o.err"; then
      orounds=$((orounds + 1))
    else
      echo "OPT SKIP $(basename "$c") (exit $st: $(head -c 200 "$o.err"))"
      fail=1
    fi
    continue
  fi
  # shellcheck disable=SC2086
  if "$S1" ir opt --dump=inline $lims "$c/input.ir" > "$o.inline" 2> "$o.err" &&
     "$S1" ir fmt "$c/input.ir" > "$o.fmt" &&
     "$S1" ir opt --dump=inline $lims "$o.fmt" > "$o.fmt.inline" &&
     cmp -s "$c/inline.txt" "$o.inline" && cmp -s "$c/inline.txt" "$o.fmt.inline"; then
    osame=$((osame + 1))
  else
    echo "OPT DIFF $(basename "$c") ($(diff "$c/inline.txt" "$o.inline" 2>&1 | head -3 | tr '\n' ' ')$(head -c 200 "$o.err"))"
    fail=1
  fi
done
echo "stage1-ir-gate: opt inline dumps $osame / $ocases equal, $orounds need more rounds"

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
  for b in "$ROOT"/bench/kernels/* "$ROOT"/bench/simd/*; do
    if [ -f "$b/stave.toml" ] || { [ -f "$b" ] && [ "${b%.stv}" != "$b" ]; }; then echo "$b"; fi
  done
  echo "$ROOT/stage1"
} | awk '{ print NR, $0 }' > "$LIST"

start=$(date +%s)
xargs -P "$(nproc)" -L 1 sh "$0" --one < "$LIST" > "$OUT/result.txt"
end=$(date +%s)
runs=$(wc -l < "$LIST")
for t in x86_64-linux aarch64-linux; do
  for m in debug release; do
    for lv in lir ir opt; do
      n=$(awk -v t="$t" -v m="$m" -v l="$lv" '$1 != "EMPTY" && $2 == t && $3 == m && $4 == l' "$OUT/result.txt" | wc -l)
      [ "$n" -gt 0 ] || continue
      s=$(awk -v t="$t" -v m="$m" -v l="$lv" '$1 == "SAME" && $2 == t && $3 == m && $4 == l' "$OUT/result.txt" | wc -l)
      echo "stage1-ir-gate: $t $m $lv: $s / $n texts equal"
    done
  done
done
total=$(grep -vc '^EMPTY' "$OUT/result.txt" || true)
empty=$(grep -c '^EMPTY' "$OUT/result.txt" || true)
same=$(grep -c '^SAME' "$OUT/result.txt" || true)
bytes=$(awk '$1 == "SAME" { s += $5 } END { print s + 0 }' "$OUT/result.txt")
echo "stage1-ir-gate: round trip $same / $total texts equal ($bytes bytes) from $runs roots, $((end - start)) s; $empty empty files stage0 wrote (no text)"
if grep -v '^SAME\|^EMPTY' "$OUT/result.txt"; then fail=1; fi

# the largest text: stage1 read + print against stage0's
big=$(awk '$1 == "SAME" { print $5, $6 }' "$OUT/result.txt" | sort -n | tail -1 | cut -d' ' -f2)
if [ -n "$big" ]; then
  t0=$(date +%s%N)
  "$S0" ir fmt "$big" > "$OUT/big.s0" 2>/dev/null || true
  t1=$(date +%s%N)
  "$S1" ir fmt "$big" > "$OUT/big.s1"
  t2=$(date +%s%N)
  echo "stage1-ir-gate: largest text $(wc -c < "$big") bytes ($big): stage0 ir fmt $(( (t1 - t0) / 1000000 )) ms, stage1 ir fmt $(( (t2 - t1) / 1000000 )) ms"
fi

# mutants
MDIR="$OUT/mutants"
mkdir -p "$MDIR"
awk '$1 == "SAME" && $5 <= 65536 { print $6 }' "$OUT/result.txt" | sort > "$OUT/mutant-inputs.txt"
STAVE_IR_MUTANTS_LIST="$OUT/mutant-inputs.txt" STAVE_IR_MUTANTS_OUT="$MDIR" STAVE_IR_MUTANTS="$MUTANTS" \
  cargo test -q --locked -p stave-ir --test ir_text_mutants -- --ignored >"$OUT/mutants.log" 2>&1 \
  || { cat "$OUT/mutants.log"; exit 1; }
seq 0 $((MUTANTS - 1)) | xargs -P "$(nproc)" -I K sh -c '
  if "$1" ir check "$2/K.ir" >/dev/null 2>"$2/K.err"; then v=accept
  elif grep -q ": verify: " "$2/K.err"; then v=verify
  elif grep -q ": error: " "$2/K.err"; then v=error
  else v="crash"; fi
  echo "K $v"' sh "$S1" "$MDIR" | sort > "$MDIR/stage1.txt"
awk '{ v = $2; if (v == "audit") v = "accept"; if (v == "syntax" || v == "structure") v = "error"; print $1, v }' \
  "$MDIR/verdicts.txt" | sort > "$MDIR/stage0.txt"
agree=$(join "$MDIR/stage0.txt" "$MDIR/stage1.txt" | awk '$2 == $3' | wc -l)
acc0=$(awk '$2 == "accept"' "$MDIR/stage0.txt" | wc -l)
acc1=$(awk '$2 == "accept"' "$MDIR/stage1.txt" | wc -l)
echo "stage1-ir-gate: mutants $agree / $MUTANTS verdicts agree (accepted: stage0 $acc0, stage1 $acc1; $(wc -l < "$OUT/mutant-inputs.txt") inputs)"
if [ "$agree" != "$MUTANTS" ]; then
  join "$MDIR/stage0.txt" "$MDIR/stage1.txt" | awk '$2 != $3 { print "MUTANT " $1 ".ir stage0 " $2 " stage1 " $3 }' | head -20
  fail=1
fi
if [ "$fail" != 0 ]; then
  echo "stage1-ir-gate: FAILED" >&2
  exit 1
fi
