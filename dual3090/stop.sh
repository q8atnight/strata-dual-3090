#!/bin/sh
# Stop and remove the engine container (the model files stay on the volume).
docker rm -f strata-dual3090 >/dev/null 2>&1 && echo "stopped." || echo "nothing was running."
