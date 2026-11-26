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
# Optimizer goldens, each run under the case's `limits` (one `--limit` per line), on `input.ir`
# and on its `stage1 ir fmt` text, both byte-equal to the expected file: `--dump=inline` against
# `inline.txt`; `--dump-phase=inline --verify-phases` against `inline.ir`; and on every case without
# a `passes` file `--dump-phase=clean --verify-phases` against `clean.ir`, else `output.ir` (a
# decision-only case has neither: its module is verified, and the two runs must agree). On the same
# cases the plain run (all passes, no phase flag) equals `output.ir` the same way. stage0's
# `stave ir check` loads every module after a phase and every output. No case is skipped. The
# command line: usage errors exit 2, an input other than a `release checked` module exits 1, each
# with its message. (Exit 3, a phase module the verifier refuses, is `stave test stage1`'s, by a
# fault put into one phase.)
#
# Pipeline: on every release `checked` text of the round trip, under the table's limits and under
# `rounds=3 threshold=100 cap=20000`: `stage1 ir opt --dump-phase=clean --verify-phases` (the
# module after C*) and plain `stage1 ir opt` (the output, every pass), each exits 0 and stage0
# loads its module. stage0 runs the base passes only, so the output has no stage0 text to equal.
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

# Optimizer goldens (opt-passes.md "Gates").
fail=0
lims_of() { if [ -f "$1/limits" ]; then sed 's/^/--limit /' "$1/limits" | tr '\n' ' '; fi; }
ocases=0
osame=0
for c in "$ROOT"/tests/opt/*/; do
  [ -f "$c/inline.txt" ] || continue
  ocases=$((ocases + 1))
  lims=$(lims_of "$c")
  o="$OUT/opt-$(basename "$c")"
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
echo "stage1-ir-gate: opt inline dumps $osame / $ocases equal"

# phase_ok CASE PHASE WANT: the module after PHASE of the last round, verified by stage1, on the
# input and on its fmt text, equals WANT (empty: the run on the input); stage0 loads it.
phase_ok() {
  lims=$(lims_of "$1")
  o="$OUT/opt-$(basename "$1").$2"
  exp="${3:-$o.ir}"
  # shellcheck disable=SC2086
  "$S1" ir opt --dump-phase="$2" --verify-phases $lims "$1/input.ir" > "$o.ir" 2> "$o.err" &&
    "$S1" ir fmt "$1/input.ir" > "$o.fmt" &&
    "$S1" ir opt --dump-phase="$2" --verify-phases $lims "$o.fmt" > "$o.fmt.ir" 2>> "$o.err" &&
    cmp -s "$exp" "$o.ir" && cmp -s "$exp" "$o.fmt.ir" &&
    "$S0" ir check "$o.ir" 2>> "$o.err"
}
phase_fail() {
  echo "OPT $2 DIFF $(basename "$1") ($(diff "$3" "$OUT/opt-$(basename "$1").$2.ir" 2>&1 | head -3 | tr '\n' ' ')$(head -c 200 "$OUT/opt-$(basename "$1").$2.err"))"
}
xcases=0
xsame=0
ccases=0
csame=0
cver=0
for c in "$ROOT"/tests/opt/*/; do
  if [ -f "$c/inline.ir" ]; then
    xcases=$((xcases + 1))
    if phase_ok "$c" inline "$c/inline.ir"; then xsame=$((xsame + 1)); else phase_fail "$c" inline "$c/inline.ir"; fail=1; fi
  fi
  [ -f "$c/passes" ] && continue
  ccases=$((ccases + 1))
  want=""
  if [ -f "$c/clean.ir" ]; then want="$c/clean.ir"; elif [ -f "$c/output.ir" ]; then want="$c/output.ir"; fi
  if phase_ok "$c" clean "$want"; then
    if [ -n "$want" ]; then csame=$((csame + 1)); else cver=$((cver + 1)); fi
  else
    phase_fail "$c" clean "${want:-/dev/null}"
    fail=1
  fi
done
echo "stage1-ir-gate: opt modules after X $xsame / $xcases equal, verified, loaded by stage0"
echo "stage1-ir-gate: opt modules after C* $csame / $((ccases - cver)) equal, $cver of $ccases without an expected text verified, all loaded by stage0"

# plain_ok CASE: the plain output (every pass, rule P1), verified by stage1, on the input and on its
# fmt text, equals output.ir (none: the two runs agree); stage0 loads it.
plain_ok() {
  lims=$(lims_of "$1")
  o="$OUT/opt-$(basename "$1").plain"
  exp="$o.ir"
  [ -f "$1/output.ir" ] && exp="$1/output.ir"
  # shellcheck disable=SC2086
  "$S1" ir opt $lims "$1/input.ir" > "$o.ir" 2> "$o.err" &&
    "$S1" ir fmt "$1/input.ir" > "$o.fmt" &&
    "$S1" ir opt $lims "$o.fmt" > "$o.fmt.ir" 2>> "$o.err" &&
    cmp -s "$exp" "$o.ir" && cmp -s "$exp" "$o.fmt.ir" &&
    "$S0" ir check "$o.ir" 2>> "$o.err"
}
pcases=0
psame=0
pnone=0
for c in "$ROOT"/tests/opt/*/; do
  [ -f "$c/passes" ] && continue
  pcases=$((pcases + 1))
  if plain_ok "$c"; then
    if [ -f "$c/output.ir" ]; then psame=$((psame + 1)); else pnone=$((pnone + 1)); fi
  else
    o="$OUT/opt-$(basename "$c").plain"
    echo "OPT plain DIFF $(basename "$c") ($(diff "$c/output.ir" "$o.ir" 2>&1 | head -3 | tr '\n' ' ')$(head -c 200 "$o.err"))"
    fail=1
  fi
done
echo "stage1-ir-gate: opt plain outputs $psame / $((pcases - pnone)) equal output.ir, $pnone of $pcases without one verified, all loaded by stage0"

# The command line of `stage1 ir opt` (cli-stage1.md): each case `EXIT|MESSAGE|ARGS` exits EXIT
# with MESSAGE in its stderr (none: empty stderr, a dump on stdout). @IN@ is a `release checked`
# case input, @OPT@ a `release opt` text, @DIR@ holds that input relabelled `debug checked` and
# `release lowered`, a text that does not load, and no file `none.ir`.
CLI="$OUT/cli"
mkdir -p "$CLI"
cin="$ROOT/tests/opt/10.1/input.ir"
sed '1s/ release checked$/ debug checked/' "$cin" > "$CLI/debug.ir"
sed '1s/ release checked$/ release lowered/' "$cin" > "$CLI/lowered.ir"
printf 'stave.ir 4 "x86_64-linux-v2" release checked\nfn @0\n' > "$CLI/bad.ir"
rm -f "$CLI/none.ir"
clicases=0
cliok=0
while IFS='|' read -r want msg args; do
  clicases=$((clicases + 1))
  args=$(echo "$args" | sed "s|@IN@|$cin|g; s|@OPT@|$ROOT/tests/opt/10.1/inline.ir|g; s|@DIR@|$CLI|g")
  got=0
  # shellcheck disable=SC2086
  "$S1" ir opt $args > "$CLI/out" 2> "$CLI/err" < /dev/null || got=$?
  if [ "$got" = "$want" ] && { if [ -n "$msg" ]; then grep -qF -- "$msg" "$CLI/err"; else [ ! -s "$CLI/err" ] && [ -s "$CLI/out" ]; fi; }; then
    cliok=$((cliok + 1))
  else
    echo "OPT CLI DIFF \`ir opt $args\`: exit $got, want $want with \`$msg\` ($(head -c 200 "$CLI/err" | tr '\n' ' '))"
    fail=1
  fi
done <<'EOF'
2|bad --limit `rounds=0`|--limit rounds=0 @IN@
2|bad --limit `call_saved=4`|--limit call_saved=4 @IN@
0||--limit call_saved=3 --limit rounds=1 --limit threshold=0 --dump=inline @IN@
0||--limit cap=18446744073709551615 --limit slot_callee=0 --limit slot_caller=0 --dump=inline @IN@
2|bad --limit `threshold=-1`|--limit threshold=-1 @IN@
2|bad --limit `threshold=+1`|--limit threshold=+1 @IN@
2|bad --limit `cap=18446744073709551616`|--limit cap=18446744073709551616 @IN@
2|bad --limit `depth=3`|--limit depth=3 @IN@
2|bad --limit `THRESHOLD=3`|--limit THRESHOLD=3 @IN@
2|bad --limit `cap`|--limit cap @IN@
2|unknown or repeated flag `--limit`|--limit @IN@
2|--dump and --dump-phase exclude each other|--dump=inline --dump-phase=clean @IN@
2|unknown or repeated flag `--dump-phase=inline`|--dump-phase=clean --dump-phase=inline @IN@
2|--passes=base runs no decision|--passes=base --dump=inline @IN@
2|--passes=base runs no decision|--passes=base --dump-phase=clean @IN@
2|unknown or repeated flag `--passes=full`|--passes=base --passes=full @IN@
2|unknown or repeated flag `--passes=none`|--passes=none @IN@
2|unknown or repeated flag `--verify`|--verify @IN@
2|stage1: usage: stage1 ir opt|--verify-phases
2|unknown or repeated flag `|@IN@ @IN@
2|cannot read|@DIR@/none.ir
1|error: ir opt reads a `release checked` module|@DIR@/debug.ir
1|error: ir opt reads a `release checked` module|@DIR@/lowered.ir
1|error: ir opt reads a `release checked` module|--passes=base @OPT@
1|bad.ir:|@DIR@/bad.ir
EOF
echo "stage1-ir-gate: opt command line $cliok / $clicases cases exit and report as cli-stage1.md says"

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

# Base passes (opt-passes.md section 9 "Base"): `stage1 ir opt --passes=base` on every `release`
# `ir` text of the round trip that has stage0's `opt` dump beside it equals that dump byte for
# byte; on every tests/opt case with a `passes` file it equals the case's `output.ir`.
find "$OUT/w" -path '*-release/ir.txt' -size +0 | sort > "$OUT/base-inputs.txt"
xargs -P "$(nproc)" -I '{}' sh -c '
  o="$(dirname "$2")/opt.txt"
  [ -s "$o" ] || exit 0
  t=$(basename "$(dirname "$2")" | sed "s/-release\$//")
  if "$1" ir opt --passes=base "$2" > "$2.base" 2> "$2.base.err"; then
    if cmp -s "$o" "$2.base"; then echo "SAME $t $(wc -c < "$2") $2"; else echo "DIFF $t $2 (first: $(cmp "$o" "$2.base" 2>&1 | head -1))"; fi
  else
    echo "FAIL $t $2 ($(head -c 300 "$2.base.err"))"
  fi' sh "$S1" '{}' < "$OUT/base-inputs.txt" > "$OUT/base.txt"
bcases=0
bsame=0
for c in "$ROOT"/tests/opt/*/; do
  [ -f "$c/passes" ] || continue
  bcases=$((bcases + 1))
  o="$OUT/base-$(basename "$c").ir"
  if grep -qx base "$c/passes" && "$S1" ir opt --passes=base "$c/input.ir" > "$o" 2>&1 && cmp -s "$c/output.ir" "$o"; then
    bsame=$((bsame + 1))
  else
    echo "BASE DIFF $(basename "$c") ($(diff "$c/output.ir" "$o" 2>&1 | head -3 | tr '\n' ' '))"
    fail=1
  fi
done
for t in x86_64-linux aarch64-linux; do
  n=$(awk -v t="$t" '$2 == t' "$OUT/base.txt" | wc -l)
  s=$(awk -v t="$t" '$1 == "SAME" && $2 == t' "$OUT/base.txt" | wc -l)
  echo "stage1-ir-gate: base passes $t: $s / $n release texts equal stage0's opt dump"
done
echo "stage1-ir-gate: base passes cases $bsame / $bcases equal output.ir"
if grep -v '^SAME' "$OUT/base.txt"; then fail=1; fi
bbig=$(awk '$1 == "SAME" { print $3, $4 }' "$OUT/base.txt" | sort -n | tail -1 | cut -d' ' -f2)
if [ -n "$bbig" ]; then
  t0=$(date +%s%N)
  "$S0" ir opt "$bbig" > "$OUT/bbig.s0" 2>/dev/null || true
  t1=$(date +%s%N)
  "$S1" ir opt --passes=base "$bbig" > "$OUT/bbig.s1"
  t2=$(date +%s%N)
  echo "stage1-ir-gate: base passes on the largest release text $(wc -c < "$bbig") bytes ($bbig): stage0 ir opt $(( (t1 - t0) / 1000000 )) ms, stage1 ir opt --passes=base $(( (t2 - t1) / 1000000 )) ms"
fi

# Pipeline on the corpus (opt-passes.md section 9 "Gates"): on every `release` `checked` text of
# the round trip, under the table's limits and under a second set that inlines more and over
# several rounds, `stage1 ir opt --dump-phase=clean --verify-phases` (it stops after C*) and plain
# `stage1 ir opt` (the output) exit 0 and their modules load in stage0 (`stave ir check`).
for set in table "rounds=3 threshold=100 cap=20000"; do
  lims=""
  if [ "$set" != table ]; then for l in $set; do lims="$lims --limit $l"; done; fi
  for module in "after C*" output; do
    flags="$lims"
    checked="exit 0, loaded by stage0"
    if [ "$module" != output ]; then
      flags="--dump-phase=clean --verify-phases$lims"
      checked="exit 0, verified at every phase, loaded by stage0"
    fi
    start=$(date +%s)
    xargs -P "$(nproc)" -I '{}' sh -c '
      o="$2.pipe"
      # shellcheck disable=SC2086
      if "$1" ir opt $4 "$2" > "$o" 2> "$o.err" && "$3" ir check "$o" 2>> "$o.err"; then
        echo "OK $2"
      else
        echo "FAIL $2 (ir opt $4: $(head -c 300 "$o.err" | tr "\n" " "))"
      fi
      rm -f "$o"' sh "$S1" '{}' "$S0" "$flags" < "$OUT/base-inputs.txt" > "$OUT/pipe.txt"
    end=$(date +%s)
    echo "stage1-ir-gate: opt pipeline ($set limits) on the corpus, module $module: $(grep -c '^OK' "$OUT/pipe.txt" || true) / $(wc -l < "$OUT/base-inputs.txt") release texts $checked, $((end - start)) s"
    if grep -v '^OK' "$OUT/pipe.txt"; then fail=1; fi
  done
done

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
