#!/usr/bin/env bash
set -euo pipefail

# Optional: activate your env
# source ~/miniconda3/bin/activate myenv

# Paths (adjust if your script lives elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="VisiumHD-CRC"
STARDIST_DIR="${SCRIPT_DIR}/stardist"

# Run on all five samples, write adata.h5ad back into each sample folder.
python "${SCRIPT_DIR}/preprocess-CRC-bin2cell.py" \
  --root "${ROOT}" \
  --stardist-dir "${STARDIST_DIR}" \
  --samples "P1CRC,P2CRC,P5CRC,P3NAT,P5NAT" \
  --write-adata \
  --log-level INFO
