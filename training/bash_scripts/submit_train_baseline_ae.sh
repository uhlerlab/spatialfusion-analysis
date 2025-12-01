#!/bin/bash
# ==============================================================
# Script: run_train_baseline_ae.sh
# Purpose: Launch baseline AE training with timestamped logs and GPU control
# ==============================================================

set -euo pipefail

# --------- Settings ----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SPATIALFUSION_ROOT='../../../../SpatialFusion/results/'
LOG_DIR="${LOGS:-$SPATIALFUSION_ROOT/logs}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/baseline_ae_train_${TIMESTAMP}.log"

SCRIPT='../scripts/train_baseline_ae.py'

# --------- GPU Settings ----------
GPU_ID=${1:-0}  # Default to GPU 0 if not provided
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# --------- Run Training ----------
echo "--------------------------------------------------"
echo "Starting baseline AE training on GPU ${GPU_ID} at ${TIMESTAMP}"
echo "Logging to: ${LOG_FILE}"
echo "--------------------------------------------------"

# Allow passing extra Hydra overrides (after GPU arg)
shift || true
python "${SCRIPT}" \
  training=training_baseline_ae \
  dataset=dataset_hest_test \
  "$@" 2>&1 | tee "$LOG_FILE"

rc=${PIPESTATUS[0]}
echo "✓ Baseline AE training complete. Exit code: $rc"
echo "Logs saved to ${LOG_FILE}"
exit $rc
