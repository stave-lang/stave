#!/bin/sh
# stage1 optimizer, end to end (contracts/cli-stage1.md `stage1 ir opt`; opt-passes.md section 9
# "Behaviour"): builds stage0 (the debug build `test` leaves, or STAVE_GATE_PROFILE=release) and
# stage1 with it, then:
#
# Self build: `stave ir emit --release` of the stage1 package, `stage1 ir opt`, `stave ir build`
# make `stage1-fast`. A link of that chain stage0 refuses stops the gate with the command and its
# message.
#
# Fixed point: `stage1-fast ir opt` on the same emitted text prints the bytes the normal stage1
# prints.
#
# Behaviour: every program of tests/conformance with an expected stdout (`//~ STDOUT FILE` of a
# single-file program that is no part of a package, `expected.stdout` of a package), built through
# `ir emit --release`, `stage1 ir opt --verify-phases`, `stave ir build`, prints that stdout and
# exits and writes stderr (a trap: its kind and site) as the plain `stave build --release` binary
# does (run in the program's directory, stdin empty), once per `TARGET` (`aarch64-linux` under
# qemu-aarch64; a program of tests/conformance/run without one: the host and `aarch64-linux`).
# Not run: the runner's `_selftest` fixtures, programs with `CHECK-ONLY` or `RESOURCE-LIMIT`, an
# `ARGS` other than `--release`, or a `TARGET` the host does not run (`-v4`, SVE, `aarch64-linux`
# without qemu-aarch64); counted by reason. Every run is under STAVE_OPT_GATE_LIMIT seconds
# (default 120). Prints the instruction totals (body lines that are not terminators) before and
# after, and the inlined rows of the `inline` dump.
#
# Times (no threshold): the normal stage1 (built by stage0), stage0's `--release` build of stage1,
# and stage1-fast, each on `build --dump=tokens,items,ast` of the stage1 sources, best of three.
#   tools/ci-step/run.sh stage1-opt-gate
# Internal: stage1-opt-gate.sh --one N KIND SRC TARGET EXPECTED compares one program (run in
# parallel by xargs).
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"
OUT="$ROOT/target/stage1-opt-gate"
PROFILE="${STAVE_GATE_PROFILE:-debug}"
S0="$ROOT/target/$PROFILE/stave"
S1="$OUT/stage1"
FAST="$OUT/stage1-fast"
LIMIT="${STAVE_OPT_GATE_LIMIT:-120}"
JOBS="${STAVE_OPT_GATE_JOBS:-6}"
export STAVE_STD_DIR="$ROOT"

# lim SECONDS CMD...: the command under an alarm.
lim() { s="$1"; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }
# ins FILE: the body lines that are no terminator.
ins() { grep '^  ' "$1" | grep -cvE '^  (br|cbr|switch|ret|unreachable)( |$)' || true; }

if [ "${1:-}" = "--one" ]; then
  n="$2"; kind="$3"; src="$4"; target="$5"; want="$6"
  d="$OUT/w/$n"
  rm -rf "$d"
  mkdir -p "$d"
  name=$(echo "$src" | sed "s|^$ROOT/tests/conformance/||")
  [ "$target" = "-" ] || name="$name@$target"
  tflag=""
  if [ "$target" != "-" ]; then tflag="--target $target"; fi
  # shellcheck disable=SC2086
  lim "$LIMIT" "$S0" ir emit --release $tflag "$src" -o "$d/in.ir" > /dev/null 2> "$d/emit.err" || { echo "FAIL $name: emit refused ($(head -c 200 "$d/emit.err" | tr '\n' ' '))"; exit 0; }
  lim "$LIMIT" "$S1" ir opt --verify-phases "$d/in.ir" > "$d/out.ir" 2> "$d/opt.err" || { echo "FAIL $name: stage1 ir opt exit $? ($(head -c 300 "$d/opt.err" | tr '\n' ' '))"; exit 0; }
  lim "$LIMIT" "$S0" ir build "$d/out.ir" -o "$d/bin" > /dev/null 2> "$d/build.err" || { echo "FAIL $name: ir build refused ($(head -c 300 "$d/build.err" | tr '\n' ' '))"; exit 0; }
  # shellcheck disable=SC2086
  lim "$LIMIT" "$S0" build --release $tflag "$src" -o "$d/plain" > /dev/null 2> "$d/plain.err" || { echo "FAIL $name: plain release build refused ($(head -c 200 "$d/plain.err" | tr '\n' ' '))"; exit 0; }
  inl=$("$S1" ir opt --dump=inline "$d/in.ir" | grep -c ' inline$' || true)
  ps=0; bs=0
  cwd="$src"
  [ "$kind" = pkg ] || cwd=$(dirname "$src")
  run=""
  [ "$target" = aarch64-linux ] && run=qemu-aarch64
  # shellcheck disable=SC2086
  (cd "$cwd" && lim "$LIMIT" $run "$d/plain" > "$d/plain.out" 2> "$d/plain.errout" < /dev/null) || ps=$?
  # shellcheck disable=SC2086
  (cd "$cwd" && lim "$LIMIT" $run "$d/bin" > "$d/bin.out" 2> "$d/bin.errout" < /dev/null) || bs=$?
  if ! cmp -s "$want" "$d/bin.out"; then
    echo "FAIL $name: stdout differs from $(basename "$want") (first: $(cmp "$want" "$d/bin.out" 2>&1 | head -1)); plain build $(if cmp -s "$want" "$d/plain.out"; then echo equal; else echo differs too; fi)"
  elif [ "$ps" != "$bs" ]; then
    echo "FAIL $name: exit $bs, plain release build $ps"
  elif ! cmp -s "$d/plain.errout" "$d/bin.errout"; then
    echo "FAIL $name: stderr differs from the plain release build's ($(head -c 200 "$d/bin.errout" | tr '\n' ' '))"
  else
    echo "OK $name $inl $(ins "$d/in.ir") $(ins "$d/out.ir")"
  fi
  exit 0
fi

if [ "$PROFILE" = release ]; then cargo build -q --locked --release -p stave-cli; else cargo build -q --locked -p stave-cli; fi
rm -rf "$OUT"
mkdir -p "$OUT"
fail=0
"$S0" build "$ROOT/stage1" -o "$S1" > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log"; exit 1; }

ms() { echo $(( ($2 - $1) / 1000000 )); }

# Self build.
t0=$(date +%s%N)
lim 1800 "$S0" ir emit --release "$ROOT/stage1" -o "$OUT/stage1.ir" > /dev/null 2> "$OUT/emit.err" || { echo "stage1-opt-gate: stave ir emit --release stage1 failed: $(head -c 400 "$OUT/emit.err")"; exit 1; }
t1=$(date +%s%N)
lim 1800 "$S1" ir opt "$OUT/stage1.ir" > "$OUT/stage1.opt.ir" 2> "$OUT/opt.err" || { echo "stage1-opt-gate: stage1 ir opt failed on the stage1 text (exit $?): $(head -c 400 "$OUT/opt.err")"; exit 1; }
t2=$(date +%s%N)
lim 1800 "$S0" ir build "$OUT/stage1.opt.ir" -o "$FAST" > /dev/null 2> "$OUT/fast.err" || { echo "stage1-opt-gate: stave ir build refused the optimized stage1 text: $(head -c 400 "$OUT/fast.err")"; exit 1; }
t3=$(date +%s%N)
echo "stage1-opt-gate: stage1-fast built: emit $(ms "$t0" "$t1") ms, stage1 ir opt $(ms "$t1" "$t2") ms, ir build $(ms "$t2" "$t3") ms; text $(ins "$OUT/stage1.ir") -> $(ins "$OUT/stage1.opt.ir") instructions"

# Fixed point.
lim 1800 "$FAST" ir opt "$OUT/stage1.ir" > "$OUT/stage1.opt.fast.ir" 2> "$OUT/fast-opt.err" || { echo "stage1-opt-gate: stage1-fast ir opt failed (exit $?): $(head -c 400 "$OUT/fast-opt.err")"; exit 1; }
if cmp -s "$OUT/stage1.opt.ir" "$OUT/stage1.opt.fast.ir"; then
  echo "stage1-opt-gate: fixed point: stage1-fast prints the $(wc -c < "$OUT/stage1.opt.ir") bytes the normal stage1 prints"
else
  echo "stage1-opt-gate: FIXED POINT DIFF: stage1-fast and the normal stage1 print different text (first: $(cmp "$OUT/stage1.opt.ir" "$OUT/stage1.opt.fast.ir" 2>&1 | head -1))"
  fail=1
fi

# Behaviour.
CONF="$ROOT/tests/conformance"
find "$CONF" -name stave.toml | sed 's|/stave.toml$||' | sort > "$OUT/pkgs.txt"
: > "$OUT/skips.txt"
: > "$OUT/list.txt"
# todo KIND SRC ANNOTATED_FILE WANT: one program, or a skip with its reason.
todo() {
  a="$3"
  case "$2" in "$CONF"/_selftest/*) echo "runner self-test" >> "$OUT/skips.txt"; return ;; esac
  if grep -q '^//~ \(CHECK-ONLY\|RESOURCE-LIMIT\)' "$a"; then echo "check-only or resource-limit" >> "$OUT/skips.txt"; return; fi
  if grep '^//~ ARGS' "$a" | grep -qv '^//~ ARGS --release$'; then echo "other build flags" >> "$OUT/skips.txt"; return; fi
  tgs=$(sed -n 's|^//~ TARGET ||p' "$a")
  # a program of the run suite without a TARGET runs on the host and on AArch64, as stage0's
  # suite runs it
  if [ -z "$tgs" ]; then
    tgs="-"
    case "$2" in "$CONF"/run/*) tgs="- aarch64-linux" ;; esac
  fi
  for tg in $tgs; do
    case "$tg" in
      -|x86_64-linux|x86_64-linux-v2|x86_64-linux-v3) ;;
      aarch64-linux) if [ -z "$QEMU" ]; then echo "target $tg without qemu-aarch64" >> "$OUT/skips.txt"; continue; fi ;;
      *) echo "target $tg" >> "$OUT/skips.txt"; continue ;;
    esac
    echo "$1 $2 $tg $4" >> "$OUT/list.txt"
  done
}
QEMU=$(command -v qemu-aarch64 || true)
find "$CONF" -name '*.stv' | sort | while read -r f; do
  # a file of a package is that package's, not a program of its own
  if awk -v f="$f" '{ if (index(f, $0 "/") == 1) found = 1 } END { exit !found }' "$OUT/pkgs.txt"; then continue; fi
  w=$(sed -n 's|^//~ STDOUT ||p' "$f" | head -1)
  [ -n "$w" ] || continue
  todo file "$f" "$f" "$(dirname "$f")/$w"
done
while read -r p; do
  [ -f "$p/expected.stdout" ] || continue
  m="$p/src/main.stv"
  [ -f "$m" ] || m="$p/main.stv"
  [ -f "$m" ] || m="$p/src/lib.stv"
  [ -f "$m" ] || { echo "no root module" >> "$OUT/skips.txt"; continue; }
  todo pkg "$p" "$m" "$p/expected.stdout"
done < "$OUT/pkgs.txt"
awk '{ print NR, $0 }' "$OUT/list.txt" > "$OUT/list.n"
total=$(wc -l < "$OUT/list.n")
xargs -P "$JOBS" -L 1 sh "$0" --one < "$OUT/list.n" > "$OUT/result.txt"
ok=$(grep -c '^OK ' "$OUT/result.txt" || true)
echo "stage1-opt-gate: behaviour $ok / $total program runs print their expected stdout, exit, and write stderr as the plain release build"
if [ -s "$OUT/skips.txt" ]; then
  echo "stage1-opt-gate: not run: $(sort "$OUT/skips.txt" | uniq -c | awk '{ c = $1; $1 = ""; sub(/^ /, ""); printf "%s%s (%d)", sep, $0, c; sep = "; " }')"
fi
awk '$1 == "OK" { inl += $3; b += $4; a += $5 } END { print "stage1-opt-gate: inlined rows " inl + 0 ", instructions " b + 0 " -> " a + 0 " over those programs" }' "$OUT/result.txt"
if grep -v '^OK ' "$OUT/result.txt"; then fail=1; fi
[ "$ok" = "$total" ] || fail=1

# Times.
SRCS="$ROOT/stage1"
best() {
  b=""
  for i in 1 2 3; do
    s=$(date +%s%N)
    "$@" --dump=tokens,items,ast --dump-dir "$OUT/dump" > /dev/null 2>&1 || true
    e=$(date +%s%N)
    t=$(ms "$s" "$e")
    if [ -z "$b" ] || [ "$t" -lt "$b" ]; then b="$t"; fi
  done
  echo "$b"
}
mkdir -p "$OUT/dump"
"$S0" build --release "$SRCS" -o "$OUT/stage1-release" > "$OUT/release.log" 2>&1 || { echo "stage1-opt-gate: stave build --release stage1 failed: $(head -c 300 "$OUT/release.log")"; exit 1; }
n1=$(best "$S1" build "$SRCS")
nr=$(best "$OUT/stage1-release" build "$SRCS")
nf=$(best "$FAST" build "$SRCS")
echo "stage1-opt-gate: tokens+items+ast of the stage1 sources ($(wc -c < "$OUT/dump/tokens.txt") bytes of tokens): normal stage1 $n1 ms, stage0 --release stage1 $nr ms, stage1-fast $nf ms"
if [ "$fail" != 0 ]; then
  echo "stage1-opt-gate: FAILED" >&2
  exit 1
fi
