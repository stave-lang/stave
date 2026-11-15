#!/bin/sh
# Compile-time and run-time benchmark: stave (stage0) against tcc, gcc and clang on one generated
# program written in both languages, plus three hand-written kernels. Prints a markdown report on
# stdout; progress goes to stderr. Linux only. See bench/README.md.
#   bench/run.sh [--tier quick|stave|full] [--lines N] [--seed S] [--runs R] [--reps P]
#   bench/run.sh --stages [--alloc-count] [--lines N] [--seed S] [--runs R]
# --stages: only the step profile of `stave build` on the corpus (STAVE_PROFILE, stave_driver::prof):
# median ms per step on 1 thread and on all threads. --alloc-count builds stave with its counting
# allocator (in target/alloc-count) and adds allocation calls and bytes; its times are slower.
# --tier: full (default) measures every compiler and caches each C row in target/bench-cache;
# stave measures stave alone and prints the C rows from that cache; quick is stave alone on 20000
# lines, 3 runs, no C rows. stave and quick check the output against one C build only. After the
# report every tier compares its stave numbers with its previous run at the same --lines.
set -eu
cd "$(dirname "$0")/.."

TIER=full
LINES=""
SEED=1
RUNS=""
REPS=500
STAGES=0
ALLOC_COUNT=0
usage() {
  echo "usage: bench/run.sh [--tier quick|stave|full | --stages [--alloc-count]] [--lines N] [--seed S] [--runs R] [--reps P]" >&2
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER=$2; shift 2 ;;
    --lines) LINES=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --stages) STAGES=1; shift ;;
    --alloc-count) ALLOC_COUNT=1; shift ;;
    *) usage ;;
  esac
done
case "$TIER" in
  quick) : "${LINES:=20000}" "${RUNS:=3}" ;;
  stave|full) : "${LINES:=100000}" "${RUNS:=5}" ;;
  *) usage ;;
esac

unset STAVE_THREADS
WORK=target/bench
# Outlives $WORK, which every run starts by deleting.
CACHE=target/bench-cache
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

ver() { if have "$1"; then "$@" 2>&1 | head -1; fi; }

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

# What a cached C row was measured on: the row is reused only while all of it is the same.
c_key() { # c_key CFG
  cmd=$(c_cmd "$1")
  case "$1" in tcc*) v=$(ver tcc -v) ;; *) v=$(ver "${1%% *}" --version) ;; esac
  echo "key: lines=$LINES seed=$SEED reps=$REPS src=$SRC_HASH cmd=$cmd cc=$v"
}

# Parallel split build: every unit with `xargs -P`, then one link.
c_split() { # c_split CMD OUT DIR (OUT relative to the repository root)
  out=$(pwd)/$2
  (cd "$3" && rm -f ./*.o && xargs -P "$NPROC" -n 1 $1 -c < files.txt && $1 -o "$out" $(sed 's/\.c$/.o/' files.txt))
}

# Step profile: per step the median over $RUNS builds, 1 thread and all threads; `process` is the
# wall time of the whole process, `total` the profiled part of it.
stages() {
  : > "$WORK/prof"
  for t in 1 "$NPROC"; do
    i=0
    while [ "$i" -lt "$RUNS" ]; do
      t0=$(date +%s%N)
      STAVE_PROFILE=1 STAVE_THREADS=$t "$STAVE" build "$CORPUS/stave" --compiler cpu -o "$WORK/x" \
        >/dev/null 2>"$WORK/err" || { cat "$WORK/err" >&2; die "stave build failed on the corpus"; }
      t1=$(date +%s%N)
      sed -n "s/^profile: /$t /p" "$WORK/err" >> "$WORK/prof"
      echo "$t process $((t1 - t0))" | awk '{ printf "%s %s %.3f 0 0\n", $1, $2, $3 / 1e6 }' >> "$WORK/prof"
      i=$((i + 1))
    done
  done
  awk -v n="$NPROC" -v lines="$S_LINES" -v runs="$RUNS" '
    function med(t, name,   c, i, j, x, a) {
      c = cnt[t, name]
      if (c == 0) return 0
      for (i = 1; i <= c; i++) a[i] = v[t, name, i]
      for (i = 2; i <= c; i++) {
        x = a[i]
        for (j = i - 1; j >= 1 && a[j] > x; j--) a[j + 1] = a[j]
        a[j + 1] = x
      }
      return (c % 2) ? a[(c + 1) / 2] : (a[c / 2] + a[c / 2 + 1]) / 2
    }
    {
      if (!($2 in seen)) { seen[$2] = 1; order[++no] = $2 }
      cnt[$1, $2]++
      v[$1, $2, cnt[$1, $2]] = $3
      if ($1 == 1) { allocs[$2] = $4; bytes[$2] = $5 }
    }
    END {
      printf "- corpus: %d Stave lines; times: median of %d runs, ms\n\n", lines, runs
      printf "| step | 1 thread | %d threads | allocations | bytes |\n|---|---:|---:|---:|---:|\n", n
      for (i = 1; i <= no; i++) {
        s = order[i]
        printf "| %s | %.1f | %.1f | %.0f | %.0f |\n", s, med(1, s), med(n, s), allocs[s], bytes[s]
      }
      printf "\nK lines/s, 1 thread (process): %.0f\n", lines / med(1, "process")
    }' "$WORK/prof"
}

# ---------- build the tools ----------
say "building stave and gen-corpus (cargo --release)"
if [ "$ALLOC_COUNT" = 1 ]; then
  cargo build -q --release --locked --target-dir target/alloc-count -p stave-cli --features stave-cli/alloc-count >&2
  cargo build -q --release --locked -p gen-corpus >&2
  STAVE=target/alloc-count/release/stave
else
  cargo build -q --release --locked -p stave-cli -p gen-corpus >&2
  STAVE=target/release/stave
fi
GEN=target/release/gen-corpus
rm -rf "$WORK"
mkdir -p "$BIN" "$CACHE"

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

# The C builds the output is checked against: all of them when they are timed anyway, otherwise
# the first one without an optimizer, which is the cheapest to build.
CHECK_CCS=$CCS
CHECKED="every build"
if [ "$TIER" != full ]; then
  CHECK_CCS=""
  CHECKED="both stave builds; no C build was checked (no tcc, gcc or clang)"
  IFS='|'
  for cfg in $CCS; do
    case "$cfg" in
      *-O2) ;;
      *) CHECK_CCS=$cfg; CHECKED="both stave builds and $cfg"; break ;;
    esac
  done
  unset IFS
fi

# ---------- generate and check the corpus ----------
say "generating the corpus: $LINES lines, seed $SEED"
"$GEN" --pair --lines "$LINES" --seed "$SEED" --reps "$REPS" -o "$CORPUS" --force 2> "$WORK/gen.txt" \
  || { cat "$WORK/gen.txt" >&2; die "gen-corpus failed"; }
S_LINES=$(cat "$CORPUS"/stave/src/*.stv | wc -l | tr -d ' ')
C_LINES=$(wc -l < "$CORPUS/c/single.c" | tr -d ' ')
UNITS=$(wc -l < "$CORPUS/c/split/files.txt" | tr -d ' ')
UNITS=$((UNITS - 1))
FNS=$(sed -n 's/.* \([0-9]*\) fns$/\1/p' "$WORK/gen.txt")
SRC_HASH=$(cat "$CORPUS/c/single.c" bench/kernels/*/main.c | sha256sum | cut -d' ' -f1)

if [ "$STAGES" = 1 ]; then
  stages
  exit 0
fi

say "correctness builds"
stave_build "$BIN/corpus-stave" "$CORPUS/stave" || die "stave build failed on the corpus"
stave_build "$BIN/corpus-stave-release" "$CORPUS/stave" --release || die "stave build --release failed on the corpus"
want=$("$BIN/corpus-stave") || die "the Stave corpus program failed"
got=$("$BIN/corpus-stave-release") || die "the Stave corpus program (--release) failed"
[ "$got" = "$want" ] || die "stave --release prints '$got', debug prints '$want'"
IFS='|'
for cfg in $CHECK_CCS; do
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
say "corpus: $want in $CHECKED"

for k in $KERNELS; do
  d=bench/kernels/$k
  stave_build "$BIN/$k-stave" "$d" || die "stave build failed on $k"
  stave_build "$BIN/$k-stave-release" "$d" --release || die "stave build --release failed on $k"
  "$BIN/$k-stave" > "$WORK/$k.want" || die "$k: the Stave program failed"
  "$BIN/$k-stave-release" > "$WORK/$k.got" || die "$k: the Stave program (--release) failed"
  cmp -s "$WORK/$k.want" "$WORK/$k.got" || die "$k: stave --release output differs from debug"
  IFS='|'
  for cfg in $CHECK_CCS; do
    unset IFS
    s=$(slug "$cfg")
    $(c_strict "$cfg") -o "$BIN/$k-$s" "$d/main.c" || die "$k: $cfg: does not compile cleanly"
    "$BIN/$k-$s" > "$WORK/$k.got" || die "$k: $cfg: the program failed"
    cmp -s "$WORK/$k.want" "$WORK/$k.got" || { diff "$WORK/$k.want" "$WORK/$k.got" >&2; die "$k: output mismatch: stave vs $cfg"; }
    IFS='|'
  done
  unset IFS
done
say "kernels: identical output in $CHECKED"

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

[ "$TIER" = quick ] && CCS=""
IFS='|'
for cfg in $CCS; do
  unset IFS
  s=$(slug "$cfg")
  f=$CACHE/c-$s-$LINES-$SEED-$REPS.txt
  key=$(c_key "$cfg")
  if [ "$TIER" = full ]; then
    say "measuring $cfg (median of $RUNS)"
    cmd=$(c_cmd "$cfg")
    t1=$(measure c_single "$cmd" "$WORK/x" "$CORPUS/c/single.c")
    tn=$(measure c_split "$cmd" "$WORK/x" "$CORPUS/c/split")
    run=$(measure "$BIN/corpus-$s")
    kr=$(kernel_runs "$s")
    r="| $cfg | $t1 | $tn | $(klps "$C_LINES" "$t1") | $(size_of "$BIN/corpus-$s") | $run$kr |"
    { echo "$key"; echo "$r"; } > "$f"
  elif [ -f "$f" ] && [ "$(head -1 "$f")" = "$key" ]; then
    r=$(sed -n 2p "$f")
  else
    r="| $cfg | n/a (run --tier full) | | | | | | | |"
  fi
  row "$r"
  IFS='|'
done
unset IFS

# ---------- report ----------
cpu=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)
cores=$(sed -n 's/^cpu cores[[:space:]]*: //p' /proc/cpuinfo | head -1)
os=$(sed -n 's/^PRETTY_NAME="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' /etc/os-release 2>/dev/null | head -1)

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
case "$TIER" in
  quick) echo "- tier quick: stave only, no C compiler is timed" ;;
  stave) echo "- tier stave: stave only; the C rows are the ones \`--tier full\` cached for these compilers and this corpus" ;;
esac
cat <<EOF
- corpus: gen-corpus --pair, seed $SEED: $S_LINES Stave lines, $C_LINES C lines, $UNITS units, $FNS fns + drivers, $REPS passes at run time
- output check: corpus checksum \`${want#checksum }\` and kernel outputs identical in $CHECKED
- times: wall clock, median of $RUNS runs, in ms

| compiler | compile, 1 thread | compile, $NPROC threads | K lines/s, 1 thread | binary bytes | corpus run | nbody run | hashmix run | sieve run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
$ROWS
"compile, 1 thread": stave with STAVE_THREADS=1; C as one process on single.c.
"compile, $NPROC threads": stave with default threads; C as $UNITS+1 files with xargs -P $NPROC, then one link.
EOF

# ---------- change since the previous run of this tier ----------
# Informational: a slower run is marked, never an error.
LAST=$CACHE/last-$TIER-$LINES.txt
{ echo "# seed=$SEED reps=$REPS"; echo "$ROWS" | sed -n '/^| stave/p'; } > "$WORK/last"
if [ -f "$LAST" ]; then
  awk -F ' *[|] *' -v kernels="$KERNELS" -v tier="$TIER" -v lines="$LINES" '
    BEGIN {
      m[3] = "compile, 1 thread"; m[4] = "compile, all threads"; m[6] = "binary bytes"; m[7] = "corpus run"
      last = 7 + split(kernels, k, " ")
      for (i = 8; i <= last; i++) m[i] = k[i - 7] " run"
      printf "\n## stave against the previous `--tier %s` run at --lines %d\n\n", tier, lines
    }
    /^#/ {
      if (NR == FNR) was = $0
      else if ($0 != was) { printf "Not compared: that run had %s, this one %s.\n", substr(was, 3), substr($0, 3); exit }
      else printf "More than 5%% worse is marked SLOWER.\n\n| metric | previous | now | change | |\n|---|---:|---:|---:|---|\n"
      next
    }
    NR == FNR { for (i = 3; i <= last; i++) old[$2, i] = $i; next }
    {
      for (i = 3; i <= last; i++) {
        if (!(i in m)) continue
        o = old[$2, i]
        if (o + 0 == 0) { printf "| %s: %s | - | %s | | |\n", $2, m[i], $i; continue }
        d = ($i - o) / o * 100
        printf "| %s: %s | %s | %s | %+.1f%% | %s |\n", $2, m[i], o, $i, d, (d > 5) ? "SLOWER" : ""
      }
    }' "$LAST" "$WORK/last"
fi
mv "$WORK/last" "$LAST"
