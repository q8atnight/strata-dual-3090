#!/bin/sh
# Entry point for the dual-3090 build (inside the container).
#
# First start: the upstream setup downloads the model and prepares the pack and
# the MTP layer on the /data volume (~70-90 GB, takes a while). Then
# dual3090/mkconfig.py turns the config setup recorded into the tuned two-card
# peer config, and serve/server.py starts with it.
#
# Everything the upstream docker-entrypoint.sh reads from the environment works
# here too (FAMILY, MODEL, CONTEXT, VISION, HOST, PORT, API_KEY, LOW_RAM,
# REINSTALL, KV), plus one of our own:
#
#   GPUS_ORDER   the two cards, numbered as nvidia-smi numbers them, in the
#                order the engine should see them. Default "0,1". On a desktop
#                PC put the card WITHOUT the monitor first: GPUS_ORDER=1,0
#
# The container needs: --gpus all --ulimit memlock=-1 --ulimit stack=67108864
# and (for 262K contexts) a generous --shm-size or --ipc host.
set -e
cd /opt/strata || exit 1

STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ3_S}"
CONTEXT="${CONTEXT:-262144}"
VISION="${VISION:-yes}"          # the encoder runs on the helper card
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
KV="${KV:-}"
LOW_RAM="${LOW_RAM:-auto}"
GPUS_ORDER="${GPUS_ORDER:-0,1}"

case "$FAMILY" in qwen) prefix="" ;; *) prefix="${FAMILY}-" ;; esac
tag="${prefix}$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')"
base_cfg="$STRATA_DATA/config/strata-$tag.json"
dual_cfg="$STRATA_DATA/config/strata-$tag-dual.json"
mkdir -p "$STRATA_DATA/config"

if [ "${REINSTALL:-0}" = "1" ] || [ ! -f "$base_cfg" ]; then
  echo "Setting up $tag: downloading the model (the engine is already in the image)."
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision "$VISION" \
    --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
    --port "$PORT" --no-start --low-ram "$LOW_RAM"
  # no --gpus here on purpose: the peer build takes both cards itself, the
  # setup's layer split is a different mode (docs/MULTI_GPU.md).
  if [ -n "$KV" ]; then set -- "$@" --kv "$KV"; fi
  .venv/bin/python setup.py --setup --yes "$@"
  [ -e "/opt/strata/strata-$tag.json" ] && { cmp -s "/opt/strata/strata-$tag.json" "$base_cfg" || cp -f "/opt/strata/strata-$tag.json" "$base_cfg"; }
else
  [ -e "/opt/strata/strata-$tag.json" ] || ln -s "$base_cfg" "/opt/strata/strata-$tag.json"
fi

# the helper card is the LAST one in GPUS_ORDER: the engine's primary card is
# the first, the image encoder and the peer tier live on the other.
helper=$(printf '%s' "$GPUS_ORDER" | awk -F, '{print $NF}')
if [ "${RECONFIG:-0}" = "1" ] || [ ! -f "$dual_cfg" ]; then
  .venv/bin/python dual3090/mkconfig.py "$base_cfg" -o "$dual_cfg" --helper-device "$helper"
fi

# The cards exactly as nvidia-smi numbers them (CUDA's own order can swap
# them). The peer tier and the encoder's cuda_device use these numbers too.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS_ORDER"
# Strata refuses Host names it is not told to answer to (DNS-rebinding
# protection). With an API key there is no Host check; without one, add your
# LAN names/IPs: ALLOWED_HOSTS="mine.lan 192.168.1.20"
if [ -n "${ALLOWED_HOSTS:-}" ]; then export STRATA_ALLOWED_HOSTS="$ALLOWED_HOSTS"; fi

echo "Starting the dual-3090 build: $dual_cfg on cards $GPUS_ORDER (helper $helper), port $PORT."
echo "Loading takes 1-3 minutes; then: http://<this PC>:$PORT/ for the web chat."
exec .venv/bin/python serve/server.py --engine strata --config "$dual_cfg" --port "$PORT"
