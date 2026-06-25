#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=6

LOG_DIR="logs"
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python embed-Virchow2.py 2>&1 | tee ${LOG_DIR}/run_${TIMESTAMP}.log