#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WORK_DIR="work_dirs/d3pm_hybrid_uniform_resnet101_512x512_100k"
CONFIG="configs/denoiser/d3pm_hybrid_uniform_resnet101_512x512_100k.py"
DATA_ROOT="data/test_oem_raw/test_oem_raw"
OUT_ROOT="${WORK_DIR}/inference_test_oem_raw_iter15000"
OUT_DIR="${OUT_ROOT}/refined_labels_steps10_tile384"
METRICS_OUT="${OUT_ROOT}/metrics_steps10_tile384.log"

CKPT="${1:-}"
if [[ -z "${CKPT}" ]]; then
  CKPT="$(ls -1v "${WORK_DIR}"/iter_*.pth | tail -n 1)"
fi

echo "Using checkpoint: ${CKPT}"
mkdir -p "${OUT_DIR}"

OPENCV_LOG_LEVEL=ERROR python tools/inference.py \
  "${CONFIG}" \
  "${CKPT}" \
  --img-dir "${DATA_ROOT}/images" \
  --pseudo-dir "${DATA_ROOT}/pseudolabels_binary" \
  --gt-dir "${DATA_ROOT}/labels" \
  --out-dir "${OUT_DIR}" \
  --metrics-out "${METRICS_OUT}" \
  --class-names background building \
  --num-classes 2 \
  --num-steps 10 \
  --img-suffix .tif \
  --pseudo-suffix .tif \
  --split-file "${DATA_ROOT}/test1.txt" \
  --tile-size 384 \
  --tile-stride 384 \
  --device cuda:0

echo
echo "Final metrics:"
cat "${METRICS_OUT}"
