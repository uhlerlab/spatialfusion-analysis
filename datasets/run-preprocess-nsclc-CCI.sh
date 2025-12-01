#!/usr/bin/env bash
set -euo pipefail

# Resolve this script's directory (so we can call the Python script reliably)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths (adjust if needed)
BASE_DIR="VisiumHD-LUAD-processed/"
LR_PAIRS_CSV="CellChat_LR_pairs.csv"   # must exist where you run the script or use an absolute path

# Optional: activate your environment
# source ~/miniconda3/bin/activate myenv

python "${SCRIPT_DIR}/preprocess-nsclc-CCI.py" \
  --base-dir "${BASE_DIR}" \
  --lr-pairs-csv "${LR_PAIRS_CSV}" \
  --exclude "full_cohort" \
  --normalize \
  --target-sum 10000 \
  --copy-layer-name "norm" \
  --smoothed-layer "smoothed" \
  --knn-bw 30 \
  --knn-cutoff 0.1 \
  --scale-by-mpp \
  --build-mode knn \
  --n-neighbors 30 \
  --sigma 50 \
  --include-self \
  --use-mean \
  --batch-size 128 \
  --output-suffix "_LR_scores.parquet" \
  --log-level INFO
