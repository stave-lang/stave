#!/bin/sh
# cargo's runner for `run.sh test-shard K`: called as `test-shard.sh BINARY [ARGS]` for every test
# binary, it starts the binary only in its shard (STAVE_TEST_SHARD, 1 or 2). Shard 1 is the
# binaries named below, which take about half of the run time; shard 2 is every other one, so a
# new or renamed test binary runs without an edit here.
set -eu
shard="${STAVE_TEST_SHARD:?}"
name="${1##*/}"
# cargo names a test binary NAME-HASH
name="${name%-*}"
case "$name" in
  borrow_suite|interp_vs_native|ir_roundtrip|stage0_suite) own=1 ;;
  *) own=2 ;;
esac
if [ "$own" != "$shard" ]; then
  echo "test-shard $shard: $name is shard $own's"
  exit 0
fi
exec "$@"
