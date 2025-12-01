#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Configuration
# ==============================
HF_TOKEN="XXX"   # Replace with your Hugging Face token or export HUGGINGFACE_TOKEN
GPU_ID=2

# Paths
BASE_DIR="/ewsc/yatesjos/Broad_SpatialFoundation"
HESD_DATA_DIR="${BASE_DIR}/hest_data"
META_CSV="${BASE_DIR}/HEST_v1_1_0.csv"
UNI_MODEL="${BASE_DIR}/UNI/pytorch_model.bin"
OUT_ROOT="${BASE_DIR}/hest_processed_data"
SCGPT_MODEL_DIR="${BASE_DIR}/scGPT_model"

# ==============================
# Environment setup
# ==============================
export HUGGINGFACE_TOKEN="${HF_TOKEN}"

# Optional: activate your conda or venv environment
# source ~/miniconda3/bin/activate myenv

# ==============================
# Run the pipeline
# ==============================
python download-preprocess-hest1k.py \
  --hf-token "${HF_TOKEN}" \
  --meta-csv "${META_CSV}" \
  --local-dir "${HESD_DATA_DIR}" \
  --filter-st-technology Xenium \
  --filter-species "Homo sapiens" \
  --make-plots \
  --base-dir-hest "${HESD_DATA_DIR}" \
  --uni-model "${UNI_MODEL}" \
  --out-root "${OUT_ROOT}" \
  --gpu "${GPU_ID}" \
  --run-download \
  --run-uni \
  --run-scgpt \
  --scgpt-model-dir "${SCGPT_MODEL_DIR}"
