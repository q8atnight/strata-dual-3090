#!/bin/sh
# Build the dual-3090 image. One RTX 3090's CUDA arch (86) is compiled by
# default, which keeps the build to a few minutes; no GPU is needed to build.
#   ARCHS=86_110 ./build.sh        (also compile sm_86 with the 110a variants)
#   ARCHS="86;89" ./build.sh       (several card generations)
set -e
cd "$(dirname "$0")/.."
ARCHS="${ARCHS:-86}"
docker build -t strata-dual3090 --build-arg CUDA_ARCHITECTURES="$ARCHS" .
echo
echo "Built strata-dual3090. Next: ./dual3090/start.sh   (first start downloads the model, ~70-90 GB)"
