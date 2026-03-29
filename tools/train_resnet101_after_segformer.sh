#!/bin/bash
# =================================================================
# Auto-launch ResNet-101 Hybrid training after SegFormer finishes
# =================================================================
# Usage:  nohup bash tools/train_resnet101_after_segformer.sh &
#
# This script:
#   1. Waits for the running SegFormer train.py process to finish
#   2. Launches the ResNet-101 Hybrid training in its place
# =================================================================

set -e

PROJECT_DIR="/home/ubuntu/vuong_denoiser/BUILDING/DifusionDenoiser/DifusionDenoiser"
CONFIG="configs/denoiser/d3pm_hybrid_uniform_resnet101_512x512_100k.py"
LOG_DIR="work_dirs/d3pm_hybrid_uniform_resnet101_512x512_100k"

cd "$PROJECT_DIR"

echo "[$(date)] Waiting for SegFormer training to finish..."

# Wait for segformer train.py to complete (PID of the main process)
while pgrep -f "train.py.*segformer" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] SegFormer training finished. Sleeping 30s for GPU cleanup..."
sleep 30

# Ensure work dir exists
mkdir -p "$LOG_DIR"

echo "[$(date)] Starting ResNet-101 Hybrid training..."
echo "[$(date)] Config: $CONFIG"
echo "[$(date)] Log dir: $LOG_DIR"

# Launch training (single GPU)
python tools/train.py "$CONFIG" 2>&1 | tee "$LOG_DIR/train_console.log"

echo "[$(date)] ResNet-101 Hybrid training complete."
