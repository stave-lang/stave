#!/bin/sh
# Compile-time and run-time benchmark: stave (stage0) against tcc, gcc and clang on one generated
# program written in both languages, plus three hand-written kernels. Prints a markdown report on
# stdout; progress goes to stderr. Linux only. See bench/README.md.
#   bench/run.sh [--lines N] [--seed S] [--runs R] [--reps P]
set -eu
cd "$(dirname "$0")/.."

LINES=100000
SEED=1
RUNS=5
REPS=500
while [ $# -gt 0 ]; do
  case "$1" in
    --lines) LINES=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    *) echo "usage: bench/run.sh [--lines N] [--seed S] [--runs R] [--reps P]" >&2; exit 2 ;;
  esac
done

unset STAVE_THREADS
WORK=target/bench
BIN=$WORK/bin
CORPUS=$WORK/corpus
NPROC=$(nproc)
KERNELS="nbody hashmix sieve"
C_CONFIGS="tcc|gcc -O0|gcc -O2|clang -O0|clang -O2"

say() { echo "bench: $*" >&2; }
die() { echo "bench: error: $*" >&2; exit 1; }

# `label` -> file-name slug: "gcc -O2" -> gcc-O2.
slug() { echo "$1" | tr -d ' '; }

have() { command -v "$1" >/dev/null 2>&1; }

# Median wall time in ms (one decimal) of $RUNS runs of the command; its output is discarded.
measure() {
  : > "$WORK/times"
  i=0
  while [ "$i" -lt "$RUNS" ]; do
    t0=$(date +%s%N)
    "$@" >/dev/null 2>"$WORK/err" || { cat "$WORK/err" >&2; die "failed: $*"; }
    t1=$(date +%s%N)
    echo $((t1 - t0)) >> "$WORK/times"
    i=$((i + 1))
  done
  sort -n "$WORK/times" | awk '{ a[NR] = $1 }
    END { m = (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2; printf "%.1f\n", m / 1e6 }'
}

size_of() { wc -c < "$1" | tr -d ' '; }

# C compiler command for a config label, flags included. Plain C99: no FMA contraction, no
# extensions.
c_cmd() {
  set -- $1
  echo "$1 -std=c99 ${2:-}"
}

# The same, with every warning an error (correctness builds only, never timed).
c_strict() {
  case "$1" in
    tcc*) echo "$(c_cmd "$1") -Wall -Werror" ;;
    *) echo "$(c_cmd "$1") -pedantic -Wall -Wextra -Werror" ;;
  esac
}

stave_build() { # stave_build OUT PKG [--release]
  out=$1; pkg=$2; shift 2
  "$STAVE" build "$pkg" --compiler cpu -o "$out" "$@"
}

stave_build_1t() { # stave_build_1t OUT PKG [--release]
  out=$1; pkg=$2; shift 2
  STAVE_THREADS=1 "$STAVE" build "$pkg" --compiler cpu -o "$out" "$@"
}

c_single() { # c_single CMD OUT SRC
  $1 -o "$2" "$3"
}

# Parallel split build: every unit with `xargs -P`, then one link.
c_split() { # c_split CMD OUT DIR (OUT relative to the repository root)
  out=$(pwd)/$2
  (cd "$3" && rm -f ./*.o && xargs -P "$NPROC" -n 1 $1 -c < files.txt && $1 -o "$out" $(sed 's/\.c$/.o/' files.txt))
}

# ---------- build the tools ----------
say "building stave and gen-corpus (cargo --release)"
cargo build -q --release --locked -p stave-cli -p gen-corpus >&2
STAVE=target/release/stave
GEN=target/release/gen-corpus
rm -rf "$WORK"
mkdir -p "$BIN"

CCS=""
SKIPPED=""
IFS='|'
for cfg in $C_CONFIGS; do
  unset IFS
  set -- $cfg
  if have "$1"; then CCS="$CCS|$cfg"; else
    case "$SKIPPED" in *"$1"*) ;; *) SKIPPED="$SKIPPED $1" ;; esac
  fi
  IFS='|'
done
unset IFS
CCS=${CCS#|}
[ -n "$SKIPPED" ] && say "not installed, skipped:$SKIPPED"

# ---------- generate and check the corpus ----------
say "generating the corpus: $LINES lines, seed $SEED"
"$GEN" --pair --lines "$LINES" --seed "$SEED" --reps "$REPS" -o "$CORPUS" --force 2> "$WORK/gen.txt" \
  || { cat "$WORK/gen.txt" >&2; die "gen-corpus failed"; }
S_LINES=$(cat "$CORPUS"/stave/src/*.stv | wc -l | tr -d ' ')
C_LINES=$(wc -l < "$CORPUS/c/single.c" | tr -d ' ')
UNITS=$(wc -l < "$CORPUS/c/split/files.txt" | tr -d ' ')
UNITS=$((UNITS - 1))
FNS=$(sed -n 's/.* \([0-9]*\) fns$/\1/p' "$WORK/gen.txt")

say "correctness builds"
stave_build "$BIN/corpus-stave" "$CORPUS/stave" || die "stave build failed on the corpus"
stave_build "$BIN/corpus-stave-release" "$CORPUS/stave" --release || die "stave build --release failed on the corpus"
want=$("$BIN/corpus-stave") || die "the Stave corpus program failed"
got=$("$BIN/corpus-stave-release") || die "the Stave corpus program (--release) failed"
[ "$got" = "$want" ] || die "stave --release prints '$got', debug prints '$want'"
IFS='|'
for cfg in $CCS; do
  unset IFS
  s=$(slug "$cfg")
  $(c_strict "$cfg") -o "$BIN/corpus-$s" "$CORPUS/c/single.c" || die "$cfg: the C corpus does not compile cleanly"
  got=$("$BIN/corpus-$s") || die "$cfg: the C corpus program failed"
  [ "$got" = "$want" ] || die "checksum mismatch: stave '$want', $cfg '$got'"
  c_split "$(c_strict "$cfg")" "$BIN/corpus-split-$s" "$CORPUS/c/split" || die "$cfg: the split C corpus does not compile cleanly"
  got=$("$BIN/corpus-split-$s") || die "$cfg: the split C corpus program failed"
  [ "$got" = "$want" ] || die "checksum mismatch: stave '$want', split $cfg '$got'"
  IFS='|'
done
unset IFS
say "corpus: $want in every build"

for k in $KERNELS; do
  d=bench/kernels/$k
  stave_build "$BIN/$k-stave" "$d" || die "stave build failed on $k"
  stave_build "$BIN/$k-stave-release" "$d" --release || die "stave build --release failed on $k"
  "$BIN/$k-stave" > "$WORK/$k.want" || die "$k: the Stave program failed"
  "$BIN/$k-stave-release" > "$WORK/$k.got" || die "$k: the Stave program (--release) failed"
  cmp -s "$WORK/$k.want" "$WORK/$k.got" || die "$k: stave --release output differs from debug"
  IFS='|'
  for cfg in $CCS; do
    unset IFS
    s=$(slug "$cfg")
    $(c_strict "$cfg") -o "$BIN/$k-$s" "$d/main.c" || die "$k: $cfg: does not compile cleanly"
    "$BIN/$k-$s" > "$WORK/$k.got" || die "$k: $cfg: the program failed"
    cmp -s "$WORK/$k.want" "$WORK/$k.got" || { diff "$WORK/$k.want" "$WORK/$k.got" >&2; die "$k: output mismatch: stave vs $cfg"; }
    IFS='|'
  done
  unset IFS
done
say "kernels: identical output in every build"

# ---------- measure ----------
ROWS=""
row() { ROWS="$ROWS$1
"; }

kernel_runs() { # kernel_runs SUFFIX -> " | t1 | t2 | t3"
  r=""
  for k in $KERNELS; do
    t=$(measure "$BIN/$k-$1") || exit 1
    r="$r | $t"
  done
  echo "$r"
}

klps() { awk -v l="$1" -v ms="$2" 'BEGIN { printf "%.0f", l / ms }'; }

say "measuring stave (median of $RUNS)"
t1=$(measure stave_build_1t "$WORK/x" "$CORPUS/stave")
tn=$(measure stave_build "$WORK/x" "$CORPUS/stave")
run=$(measure "$BIN/corpus-stave")
kr=$(kernel_runs stave)
row "| stave | $t1 | $tn | $(klps "$S_LINES" "$t1") | $(size_of "$BIN/corpus-stave") | $run$kr |"
t1=$(measure stave_build_1t "$WORK/x" "$CORPUS/stave" --release)
tn=$(measure stave_build "$WORK/x" "$CORPUS/stave" --release)
run=$(measure "$BIN/corpus-stave-release")
kr=$(kernel_runs stave-release)
row "| stave --release | $t1 | $tn | $(klps "$S_LINES" "$t1") | $(size_of "$BIN/corpus-stave-release") | $run$kr |"

IFS='|'
for cfg in $CCS; do
  unset IFS
  say "measuring $cfg (median of $RUNS)"
  s=$(slug "$cfg")
  cmd=$(c_cmd "$cfg")
  t1=$(measure c_single "$cmd" "$WORK/x" "$CORPUS/c/single.c")
  tn=$(measure c_split "$cmd" "$WORK/x" "$CORPUS/c/split")
  run=$(measure "$BIN/corpus-$s")
  kr=$(kernel_runs "$s")
  row "| $cfg | $t1 | $tn | $(klps "$C_LINES" "$t1") | $(size_of "$BIN/corpus-$s") | $run$kr |"
  IFS='|'
done
unset IFS

# ---------- report ----------
cpu=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)
cores=$(sed -n 's/^cpu cores[[:space:]]*: //p' /proc/cpuinfo | head -1)
os=$(sed -n 's/^PRETTY_NAME="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' /etc/os-release 2>/dev/null | head -1)
ver() { if have "$1"; then "$@" 2>&1 | head -1; fi; }

cat <<EOF
## Stave stage0 vs C compilers

- CPU: $cpu, ${cores:-?} cores / $NPROC threads
- OS: ${os:-unknown}, kernel $(uname -sr)
- stave: this tree, built by $(rustc --version | cut -d' ' -f1-2) with \`cargo build --release\`
EOF
for c in gcc clang; do
  v=$(ver "$c" --version)
  [ -n "$v" ] && echo "- $c: $v"
done
v=$(ver tcc -v)
[ -n "$v" ] && echo "- tcc: $v"
[ -n "$SKIPPED" ] && echo "- not installed, skipped:$SKIPPED"
cat <<EOF
- corpus: gen-corpus --pair, seed $SEED: $S_LINES Stave lines, $C_LINES C lines, $UNITS units, $FNS fns + drivers, $REPS passes at run time
- output check: corpus checksum \`${want#checksum }\` and kernel outputs identical in every build
- times: wall clock, median of $RUNS runs, in ms

| compiler | compile, 1 thread | compile, $NPROC threads | K lines/s, 1 thread | binary bytes | corpus run | nbody run | hashmix run | sieve run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
$ROWS
"compile, 1 thread": stave with STAVE_THREADS=1; C as one process on single.c.
"compile, $NPROC threads": stave with default threads; C as $UNITS+1 files with xargs -P $NPROC, then one link.
EOF
