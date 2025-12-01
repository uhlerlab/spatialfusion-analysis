#!/bin/bash
# ==============================================================
# Script: run_train_ae_align0.sh
# Purpose: Launch AE training (align & cross losses = 0) with timestamped logs and GPU control
# ==============================================================

set -euo pipefail

# --------- Settings ----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SPATIALFUSION_ROOT='../../../../SpatialFusion/results/'
# Default log directory: use global SPATIALFUSION_ROOT unless LOGS is set
LOG_DIR="${LOGS:-$SPATIALFUSION_ROOT/logs}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/ae_train_${TIMESTAMP}.log"

SCRIPT='../scripts/train_multi_ae.py'

# --------- GPU Settings ----------
GPU_ID=${1:-0}  # Default to GPU 0 if not provided
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# --------- Run Training ----------
echo "--------------------------------------------------"
echo "Starting AE training on GPU ${GPU_ID} at ${TIMESTAMP}"
echo "Logging to: ${LOG_FILE}"
echo "--------------------------------------------------"

# Allow passing extra Hydra overrides (after GPU arg)
shift || true
python "${SCRIPT}" \
  training=training_ae \
  training.loss_weights.align=0 \
  training.loss_weights.cross12=0 \
  training.loss_weights.cross21=0 \
  dataset=dataset_full_hest \
  "$@" 2>&1 | tee "$LOG_FILE"

rc=${PIPESTATUS[0]}
echo "✓ Training complete. Exit code: $rc"
echo "Logs saved to ${LOG_FILE}"
exit $rc
