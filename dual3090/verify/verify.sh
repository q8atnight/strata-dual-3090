#!/bin/sh
# Speed check against the numbers in the README. The engine must be running
# (./dual3090/start.sh).  ./verify.sh quick = ~1 minute; ./verify.sh = ~10.
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8080}"
if [ "${1:-quick}" = quick ]; then
  exec python3 flashbench.py run --url "http://127.0.0.1:$PORT/v1" --label dual3090 --quick
fi
exec python3 flashbench.py run --url "http://127.0.0.1:$PORT/v1" --label dual3090 --modes greedy
