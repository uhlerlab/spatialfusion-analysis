#!/usr/bin/env bash
set -euo pipefail

python preprocess-CRC-embed.py \
  --run-scgpt --run-uni \
  --skip-if-scgpt-exists --skip-if-uni-exists \
  --log-level INFO \
  --uni-device cuda:0
