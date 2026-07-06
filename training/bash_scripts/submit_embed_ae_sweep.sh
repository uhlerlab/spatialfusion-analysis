#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
shift || true

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

SPATIALFUSION_ROOT='../../../../SpatialFusion/results/'
LOG_DIR="${LOGS:-$SPATIALFUSION_ROOT/logs}"

mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/ae_embed_sweep_${TIMESTAMP}.log"

SCRIPT='../scripts/embed_AE.py'

echo "--------------------------------------------------"
echo "Starting AE embedding sweep on GPU ${GPU_ID}"
echo "Timestamp: ${TIMESTAMP}"
echo "Log file: ${LOG_FILE}"
echo "--------------------------------------------------"

set +e

/usr/bin/time -v python -X faulthandler "${SCRIPT}" \
  --multirun \
  hydra.job.name=ae_embed_sweep \
  hydra.sweep.dir=../../../../SpatialFusion/results/hydra_sweep/ \
  training=training_ae_hyper_sweep \
  training.checkpoint_dir=../../../../SpatialFusion/results/ae_encoder_sweep \
  dataset=dataset_full_hest \
  eval=eval \
  eval.embedding_root=../../../../SpatialFusion/results/ae_embeddings \
  training.he_encoder=uni,virchow \
  training.rna_encoder=scgpt,nicheformer \
  training.alignment_mode=full,recon_only \
  "$@" 2>&1 | tee "$LOG_FILE"

rc=${PIPESTATUS[0]}
set -e

echo "✓ Embedding sweep complete. Exit code: $rc"
echo "Logs saved to ${LOG_FILE}"

exit $rc