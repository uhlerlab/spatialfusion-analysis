#!/usr/bin/env bash
set -euo pipefail

# --------- GPU Settings ----------
GPU_ID=${1:-0}                 # Default to GPU 0 if not provided
export CUDA_VISIBLE_DEVICES="$GPU_ID"
shift || true 

# Put all run artifacts outside the repo by default
SPATIALFUSION_ROOT='../../../../SpatialFusion/results/'
LOG_DIR="${LOGS:-$SPATIALFUSION_ROOT/logs}"
# ========================================================


# Timestamp + ensure log dir
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/gcn_train_${TIMESTAMP}.log"

# After cd'ing into training/, the script is here:
SCRIPT="../scripts/train_gcn_pw.py"

# Prefer env’s newer libstdc++ if using conda-forge Python
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

# Thread caps before Python starts
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled OPENBLAS_MAIN_FREE=1
export MPLBACKEND=Agg
ulimit -c unlimited

echo "=== SpatialFusion GCN Training ==="
echo "When:                $TIMESTAMP"
echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
echo "SPATIALFUSION_ROOT:  $SPATIALFUSION_ROOT"
echo "Log file:            $LOG_FILE"
echo "PWD:                 $(pwd)"
echo "=================================="

# Run training with logging
set +e
/usr/bin/time -v python -X faulthandler "${SCRIPT}" \
  training=training_gcn training.combine_mode=z2 dataset=dataset_full_hest eval=eval \
  "$@" 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

echo "Exit code: $rc" | tee -a "$LOG_FILE"
echo "Logs saved to $LOG_FILE"
exit $rc
