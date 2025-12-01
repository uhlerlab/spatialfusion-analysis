#!/usr/bin/env bash
# ==============================================================
# Script: run_embed_ae.sh
# Purpose: Extract AE embeddings with timestamped logs and GPU control
# ==============================================================

set -euo pipefail

# --------- GPU (positional arg) ----------
GPU_ID=${1:-0}                   # e.g., bash run_embed_ae.sh 3
export CUDA_VISIBLE_DEVICES="$GPU_ID"
shift || true                    # consume the GPU arg so it won't reach Hydra

# --------- Paths / Logging ----------
SPATIALFUSION_ROOT='../../../../SpatialFusion/results/'
SPATIALFUSION_ROOT="${SPATIALFUSION_ROOT:-$HOME/spatialfusion_runs}"  # default outside repo
LOG_DIR="${LOGS:-$SPATIALFUSION_ROOT/logs}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/ae_embed_${TIMESTAMP}.log"
SCRIPT="../scripts/embed_AE.py"

# --------- Env niceties ----------
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled OPENBLAS_MAIN_FREE=1
export MPLBACKEND=Agg
ulimit -c unlimited

echo "=== SpatialFusion AE Embedding Extraction ==="
echo "When:                $TIMESTAMP"
echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
echo "SPATIALFUSION_ROOT:  $SPATIALFUSION_ROOT"
echo "Log file:            $LOG_FILE"
echo "PWD:                 $(pwd)"
echo "Script:              $SCRIPT"
echo "============================================="

# --------- Run ----------
set +e
/usr/bin/time -v python -X faulthandler "${SCRIPT}" \
  training=training_ae \
  training.loss_weights.align=0 \
  training.loss_weights.cross12=0 \
  training.loss_weights.cross21=0 \
  dataset=dataset_full_hest \
  eval=eval_onlyrecon \
  "$@" 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

echo "✓ Embedding extraction complete. Exit code: $rc"
echo "Logs saved to $LOG_FILE"
exit $rc