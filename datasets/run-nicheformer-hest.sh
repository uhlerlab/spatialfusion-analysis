#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=4

LOG_DIR="logs"
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python embed-nicheformer.py 2>&1 | tee ${LOG_DIR}/run_${TIMESTAMP}.log