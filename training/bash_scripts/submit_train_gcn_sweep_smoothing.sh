#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
shift || true

LOG_DIR="../../../../SpatialFusion/results/gcn_logs/"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

mkdir -p "$LOG_DIR"

SCRIPT="../scripts/train_gcn_pw.py"

LOG_FILE="${LOG_DIR}/smoothing_sweep_${TIMESTAMP}.log"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export OPENBLAS_MAIN_FREE=1
export MPLBACKEND=Agg

echo "=== Smoothing Sweep ==="

set +e

/usr/bin/time -v python -X faulthandler "${SCRIPT}" \
  --multirun \
  training=training_gcn_hyper_sweep \
  dataset=dataset_full_hest \
  eval=eval \
  eval.embedding_root=../../../../SpatialFusion/results/ae_embeddings \
  training.model_type=smoothing \
  training.combine_mode=concat,average \
  training.use_cls_loss=false \
  training.he_encoder=uni,virchow \
  training.rna_encoder=scgpt,nicheformer \
  training.alignment_mode=full,recon_only \
  training.pathway_mode=regression \
  training.checkpoint_dir=../../../../SpatialFusion/results/smoothing_fusion_sweep/ \
  "$@" 2>&1 | tee "$LOG_FILE"

rc=${PIPESTATUS[0]}
set -e

echo "Exit code: $rc"
echo "Logs saved to $LOG_FILE"

exit $rc
