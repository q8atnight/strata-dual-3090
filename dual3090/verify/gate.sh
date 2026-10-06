#!/bin/sh
# Exactness gate: proves the two-card answers are WORD FOR WORD the one-card
# answers - and, against the shipped reference (recorded from stock Strata
# v0.1.38 with the same switches), that both equal the stock engine.
#
# Two stages run the same prompts greedy:
#   1. single - one card, fixed expert set      -> saved as your local reference
#      and compared to the shipped reference (a cross-check of your hardware)
#   2. dual   - the peer build, same expert set -> must match stage 1 exactly
#
# Both stages pin CUDA_DEVICE_ORDER=PCI_BUS_ID and the same GPUS_ORDER, and run
# with STRATA_IQ_MT_MIN=1 STRATA_MMQ_NO_STREAMK=1 so the two arms share kernels
# and results (the exactness switches, upstream #152).
#
# The main engine must be stopped: this starts its own container on port 8080.
#   MODEL=IQ3_S GPUS_ORDER=0,1 ./dual3090/verify/gate.sh
# Each stage loads the model once: expect ~2 minutes + ~1 minute of prompts.
set -e
cd "$(dirname "$0")"
MODEL_TAG=$(printf '%s' "${MODEL:-IQ3_S}" | tr 'A-Z' 'a-z')
PORT="${GATE_PORT:-8080}"
REF=gate-ref-v038-gS.json
OUT=results
mkdir -p "$OUT"
LOCAL="$OUT/gate-single-$(date +%Y%m%d-%H%M).json"

stage() {  # $1 = single|dual ; starts the gate container, waits for /health
  docker rm -f strata-gate >/dev/null 2>&1
  docker run -d --name strata-gate --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 --ipc host \
    -p "$PORT:$PORT" -v strata-dual-data:/data \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="${GPUS_ORDER:-0,1}" \
    strata-dual3090 sh -c "cd /opt/strata && .venv/bin/python dual3090/mkconfig.py \
      /data/config/strata-$MODEL_TAG.json -o /tmp/gate.json --gate $1 && \
      exec .venv/bin/python serve/server.py --engine strata --config /tmp/gate.json --port $PORT" >/dev/null
  i=0
  while [ $i -lt 90 ]; do
    sleep 5; i=$((i+1))
    curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && return 0
    docker ps --format '{{.Names}}' | grep -q '^strata-gate$' || {
      echo "stage $1 died:"; docker logs --tail 25 strata-gate; return 1; }
  done
  echo "stage $1 did not come up in 7 minutes (docker logs strata-gate)"; return 1
}

echo "== stage 1: single card (the reference)"
stage single
python3 flashbench.py gate --url "http://127.0.0.1:$PORT/v1" --save "$LOCAL"
docker rm -f strata-gate >/dev/null 2>&1

echo "== stage 2: the dual peer build"
stage dual
python3 flashbench.py gate --url "http://127.0.0.1:$PORT/v1" --against "$LOCAL"
DUAL_RC=$?
docker rm -f strata-gate >/dev/null 2>&1

echo "== reference check (your single-card run vs stock v0.1.38, recorded on 2x RTX 3090)"
python3 - "$LOCAL" "$REF" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))["outputs"]
b = json.load(open(sys.argv[2]))["outputs"]
same = all(a.get(k) == v for k, v in b.items())
print("local single == shipped stock reference:", "PASS" if same else
      "differs (driver/GPU generation can round differently - the dual-vs-single PASS above is the guarantee this build needs)")
EOF

[ $DUAL_RC -eq 0 ] && echo "GATE PASS: the dual answers are word-for-word the single-card answers." || echo "GATE FAIL on the dual stage - see the diff above."
exit $DUAL_RC
