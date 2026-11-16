#!/bin/sh
# SIMD benchmark: the kernels of bench/simd/ built by stave for x86_64-linux (128-bit vectors)
# and x86_64-linux-v3 (256-bit), debug and --release, against their plain C99 twins built by
# gcc -O2 and, for information, gcc -O3 and clang -O2, all with -march=x86-64-v3
# -ffp-contract=off. Every build must print the same output. Prints a markdown report on stdout;
# progress goes to stderr. Linux x86-64 only. See bench/simd/README.md.
#   bench/simd.sh [--runs R]
set -eu
cd "$(dirname "$0")/.."

RUNS=3
while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS=$2; shift 2 ;;
    *) echo "usage: bench/simd.sh [--runs R]" >&2; exit 2 ;;
  esac
done

WORK=target/bench-simd
BIN=$WORK/bin
KERNELS="sum dot minmax filter bytesearch prefix histogram matmul4 utf8"
STAVE_CONFIGS="x86_64-linux|x86_64-linux --release|x86_64-linux-v3|x86_64-linux-v3 --release"
C_CONFIGS="gcc -O2|gcc -O3|clang -O2"
C_FLAGS="-std=c99 -march=x86-64-v3 -ffp-contract=off"

say() { echo "bench: $*" >&2; }
die() { echo "bench: error: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
slug() { echo "$1" | tr -d ' '; }

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

# The C functions of a kernel's twin that hold its timed loops.
kernel_fns() {
  case "$1" in
    bytesearch) echo "count_byte find_byte" ;;
    matmul4) echo "transform" ;;
    utf8) echo "valid_utf8" ;;
    *) echo "$1" ;;
  esac
}

# "yes (N-byte vectors)" when gcc reports a vectorized loop inside one of the kernel's functions,
# else "no". -fopt-info-vec-optimized names the line of every vectorized loop.
vectorized() { # vectorized LEVEL KERNEL
  src=bench/simd/$2/main.c
  gcc $C_FLAGS "$1" -fopt-info-vec-optimized -c "$src" -o /dev/null 2> "$WORK/vec.txt" || die "$2: gcc $1 failed"
  ranges=""
  for f in $(kernel_fns "$2"); do
    r=$(awk -v f="$f" '
      $0 ~ "^static [^(]* \\**" f "\\(" { s = NR }
      s && !e && /^}/ { e = NR }
      END { if (s) print s ":" e }' "$src")
    ranges="$ranges $r"
  done
  awk -F: -v ranges="$ranges" '
    BEGIN { n = split(ranges, r, " "); for (i = 1; i <= n; i++) { split(r[i], p, ":"); lo[i] = p[1]; hi[i] = p[2] } }
    /optimized: loop vectorized/ {
      for (i = 1; i <= n; i++) if ($2 >= lo[i] && $2 <= hi[i]) {
        w = $0; sub(/.*using /, "", w); sub(/ vectors.*/, "", w); found[w] = 1
      }
    }
    END { s = ""; for (w in found) s = s (s ? ", " : "") w; print (s ? "yes (" s ")" : "no") }' "$WORK/vec.txt"
}

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "needs Linux on x86-64"
say "building stave (cargo --release)"
cargo build -q --release --locked -p stave-cli >&2
STAVE=target/release/stave
rm -rf "$WORK"
mkdir -p "$BIN"
V3=0
grep -q ' avx2 ' /proc/cpuinfo && grep -q ' fma ' /proc/cpuinfo && grep -q ' bmi2 ' /proc/cpuinfo && V3=1

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

# ---------- build and check ----------
# Cell states, kept in files: $BIN/<kernel>-<slug>.state is "ok", "not built: why" or "cannot
# run". The first build that runs gives the output every other build must print.
NOTES=""
for k in $KERNELS; do
  d=bench/simd/$k
  : > "$WORK/$k.want"
  IFS='|'
  for cfg in $STAVE_CONFIGS; do
    unset IFS
    set -- $cfg
    s=$(slug "stave $cfg")
    out=$BIN/$k-$s
    if "$STAVE" build "$d" --compiler cpu --target "$1" ${2:-} -o "$out" 2> "$WORK/err"; then
      echo ok > "$out.state"
    elif grep -q 'not implemented yet' "$WORK/err"; then
      why=$(grep -m1 'not implemented yet' "$WORK/err")
      echo "not built: $why" > "$out.state"
      NOTES="$NOTES
- $k, stave $cfg: not built: \`$why\`"
      IFS='|'
      continue
    else
      cat "$WORK/err" >&2
      die "$k: stave build --target $cfg failed"
    fi
    if [ "$1" = x86_64-linux-v3 ] && [ "$V3" = 0 ]; then
      echo "cannot run" > "$out.state"
      IFS='|'
      continue
    fi
    "$out" > "$WORK/got" || die "$k: stave $cfg: the program failed"
    if [ -s "$WORK/$k.want" ]; then
      cmp -s "$WORK/$k.want" "$WORK/got" || die "$k: stave $cfg prints $(cat "$WORK/got"), expected $(cat "$WORK/$k.want")"
    else
      cp "$WORK/got" "$WORK/$k.want"
    fi
    IFS='|'
  done
  IFS='|'
  for cfg in $CCS; do
    unset IFS
    set -- $cfg
    s=$(slug "$cfg")
    out=$BIN/$k-$s
    "$1" $C_FLAGS "$2" -pedantic -Wall -Wextra -Werror -o "$out" "$d/main.c" || die "$k: $cfg: does not compile cleanly"
    echo ok > "$out.state"
    "$out" > "$WORK/got" || die "$k: $cfg: the program failed"
    if [ -s "$WORK/$k.want" ]; then
      cmp -s "$WORK/$k.want" "$WORK/got" || die "$k: $cfg prints $(cat "$WORK/got"), expected $(cat "$WORK/$k.want")"
    else
      cp "$WORK/got" "$WORK/$k.want"
    fi
    IFS='|'
  done
  unset IFS
done
say "every build that ran printed its kernel's expected output"

# ---------- measure ----------
ROWS=""
row() { # row LABEL: one table row, a cell per kernel
  label=$1
  s=$(slug "$label")
  r="| $label |"
  for k in $KERNELS; do
    st=$(cat "$BIN/$k-$s.state")
    case "$st" in
      ok) r="$r $(measure "$BIN/$k-$s") |" ;;
      "not built"*) r="$r not built |" ;;
      *) r="$r $st |" ;;
    esac
  done
  ROWS="$ROWS$r
"
}
IFS='|'
for cfg in $STAVE_CONFIGS; do
  unset IFS
  say "measuring stave $cfg (median of $RUNS)"
  row "stave $cfg"
  IFS='|'
done
for cfg in $CCS; do
  unset IFS
  say "measuring $cfg (median of $RUNS)"
  row "$cfg"
  IFS='|'
done
unset IFS

VEC=""
if have gcc; then
  for k in $KERNELS; do
    VEC="$VEC| $k | $(vectorized -O2 "$k") | $(vectorized -O3 "$k") |
"
  done
fi

# ---------- report ----------
cpu=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)
header="| build |"
sep="|---|"
for k in $KERNELS; do
  header="$header $k |"
  sep="$sep---:|"
done
cat <<EOF
## SIMD kernels: stave stage0 vs C99 twins

- CPU: $cpu
- stave: this tree, built with \`cargo build --release\`; targets x86_64-linux (vectors up to 128 bits) and x86_64-linux-v3 (up to 256 bits)
- C: \`CC $C_FLAGS -O2|-O3\`; gcc -O3 and clang -O2 are for information
EOF
for c in gcc clang; do
  have "$c" && echo "- $c: $("$c" --version | head -1)"
done
[ -n "$SKIPPED" ] && echo "- not installed, skipped:$SKIPPED"
cat <<EOF
- output check: every build that ran printed the same output per kernel
- times: wall clock of the whole process, median of $RUNS runs, in ms

$header
$sep
$ROWS
EOF
[ -n "$NOTES" ] && printf 'Not built (stage0 cannot compile the kernel for that target yet):%s\n\n' "$NOTES"
if [ -n "$VEC" ]; then
  cat <<EOF
Loops gcc vectorized inside each twin's kernel functions (\`-fopt-info-vec-optimized\`):

| kernel | gcc -O2 | gcc -O3 |
|---|---|---|
$VEC
EOF
fi
