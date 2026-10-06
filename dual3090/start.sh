#!/bin/sh
# Start the dual-3090 engine in its container.
#
# Settings come from the environment (see dual3090/entrypoint-dual.sh):
#   MODEL=IQ3_S|IQ2_XS|Q2_0  (default IQ3_S)   CONTEXT=262144 (default)
#   VISION=yes|no|cpu        (default yes - the encoder runs on the helper card)
#   GPUS_ORDER=0,1           put the card WITHOUT the desktop first on a PC with a monitor
#   PORT=8080  HOST=0.0.0.0  API_KEY=...       ALLOWED_HOSTS="name ip ..."
#   REINSTALL=1 re-runs the setup (model choices), RECONFIG=1 rewrites the dual config
#
#   ./start.sh            detached, then waits until /health answers
#   ./start.sh -f         foreground (logs in this terminal)
set -e
cd "$(dirname "$0")"
IMAGE=strata-dual3090
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image yet - run ./dual3090/build.sh first"; exit 1; }

FORWARD=""
[ "${1:-}" = "-f" ] && FORWARD="-it" && shift
docker rm -f strata-dual3090 >/dev/null 2>&1

if [ -z "$API_KEY" ] && [ "${HOST:-0.0.0.0}" != "127.0.0.1" ]; then
  echo "note: the server is reachable from your LAN without an API key."
  echo "      Set API_KEY=... or run HOST=127.0.0.1 ./start.sh to keep it to this PC."
fi

exec docker run $FORWARD --name strata-dual3090 --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 --ipc host \
  -p "${PORT:-8080}:${PORT:-8080}" \
  -v strata-dual-data:/data \
  -e MODEL="${MODEL:-IQ3_S}" -e CONTEXT="${CONTEXT:-262144}" -e VISION="${VISION:-yes}" \
  -e HOST="${HOST:-0.0.0.0}" -e PORT="${PORT:-8080}" -e API_KEY="$API_KEY" \
  -e GPUS_ORDER="${GPUS_ORDER:-0,1}" -e ALLOWED_HOSTS="${ALLOWED_HOSTS:-}" \
  -e LOW_RAM="${LOW_RAM:-auto}" -e KV="${KV:-}" \
  -e REINSTALL="${REINSTALL:-0}" -e RECONFIG="${RECONFIG:-0}" \
  -d "$IMAGE" ./dual3090/entrypoint-dual.sh >/dev/null

if [ -n "$FORWARD" ]; then docker attach strata-dual3090; exit 0; fi
echo "Container started (docker logs -f strata-dual3090 shows the log)."
echo "First start: the model download runs here - it is done when this says READY."
i=0
while [ $i -lt 360 ]; do
  sleep 5; i=$((i+1))
  if curl -s -m 2 "http://127.0.0.1:${PORT:-8080}/health" | grep -q ok; then
    echo "READY: API http://$(hostname -I 2>/dev/null | cut -d' ' -f1 || echo 127.0.0.1):${PORT:-8080}/v1   web chat http://127.0.0.1:${PORT:-8080}/"
    curl -s "http://127.0.0.1:${PORT:-8080}/health"
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -q '^strata-dual3090$' || {
    echo "The container stopped. Last log lines:"; docker logs --tail 20 strata-dual3090; exit 1; }
done
echo "Still working after 30 minutes (a first-start model download can take this long)."
echo "Check: docker logs -f strata-dual3090"
exit 1
