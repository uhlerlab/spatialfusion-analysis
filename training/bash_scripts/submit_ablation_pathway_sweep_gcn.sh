#!/usr/bin/env bash
# Ablation sweep: train GCN once per pathway, each time excluding that pathway
# from the regression targets.
#
# Usage:
#   bash submit_ablation_sweep_gcn.sh [GPU_ID] [SAMPLE_FOR_DISCOVERY]
#
# Arguments:
#   GPU_ID               GPU to use (default: 0)
#   SAMPLE_FOR_DISCOVERY Sample name whose pathway_activation.parquet is used
#                        to discover pathway names (default: first in dataset)
#
# Extra Hydra overrides can be appended after the two positional args, e.g.:
#   bash submit_ablation_sweep_gcn.sh 0 TENX149 training.epochs=10
#
# Each run is saved to a separate checkpoint subdirectory named:
#   ..._ablation_<pathway_name>_<uuid>/

set -euo pipefail

GPU_ID=${1:-0}
SAMPLE_FOR_DISCOVERY=${2:-TENX149}
shift 2 || true   # remaining args are passed through to the training script

export CUDA_VISIBLE_DEVICES="$GPU_ID"

SPATIALFUSION_ROOT='../../../../SpatialFusion/results/'
LOG_DIR="${LOGS:-$SPATIALFUSION_ROOT/logs/ablation}"
DATA_ROOT="${DATA_ROOT:-../../../../Broad_SpatialFoundation}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"

SCRIPT="../scripts/train_gcn_pw.py"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled OPENBLAS_MAIN_FREE=1
export MPLBACKEND=Agg
ulimit -c unlimited

# ---- Discover pathway names from one sample's parquet ----
PATHWAY_FILE="${DATA_ROOT}/hest_processed_data/${SAMPLE_FOR_DISCOVERY}/pathway_activation.parquet"

echo "=== Discovering pathways from: $PATHWAY_FILE ==="
PATHWAYS=$(python - <<PYEOF
import pandas as pd, sys
try:
    df = pd.read_parquet("$PATHWAY_FILE")
    for col in df.columns:
        print(col)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

N_PATHWAYS=$(echo "$PATHWAYS" | wc -l)
echo "Found $N_PATHWAYS pathways. Starting sweep..."
echo "========================================="

IDX=0
while IFS= read -r PATHWAY; do
    IDX=$((IDX + 1))
    echo ""
    echo "--- Ablation $IDX/$N_PATHWAYS: excluding '$PATHWAY' ---"

    LOG_FILE="${LOG_DIR}/ablation_${TIMESTAMP}_${IDX}_gcn.log"

    # Hydra list override: wrap in brackets, escape special chars for shell
    OVERRIDE="training.excluded_pathways=[\"${PATHWAY}\"]"

    set +e
    /usr/bin/time -v python -X faulthandler "${SCRIPT}" \
        training=training_gcn_hyper_sweep \
        dataset=dataset_full_hest \
        eval=eval \
        eval.embedding_root=../../../../SpatialFusion/results/ae_embeddings \
        training.model_type=gcn \
        training.checkpoint_dir=../../../../SpatialFusion/results/gcn_pathway_ablation_sweep/ \
        "${OVERRIDE}" \
        "$@" 2>&1 | tee "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e

    echo "Exit code: $rc — run $IDX logged to $LOG_FILE"

    if [ "$rc" -ne 0 ]; then
        echo "WARNING: run $IDX failed (excluded='$PATHWAY'). Continuing sweep."
    fi

done <<< "$PATHWAYS"

echo ""
echo "=== Ablation sweep complete: $IDX runs, logs in $LOG_DIR ==="
