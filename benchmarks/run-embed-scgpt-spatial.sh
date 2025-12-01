#!/usr/bin/env bash
set -e

SCRIPT="embed-scgpt-spatial.py"

python "$SCRIPT" \
  --adata_path test_data/10X_Xenium_Ovarian_5k/adata.h5ad \
  --output_dir test_data/10X_Xenium_Ovarian_5k/embeddings/ \
  --model_dir scGPT_spatial_v1/ \
  --batch_size 16 \
  --max_seq_len 1200

echo "scGPT-spatial pipeline finished successfully!"

python "$SCRIPT" \
  --adata_path hest_processed_data/TENX157/adata.h5ad \
  --output_dir hest_processed_data/TENX157/embeddings/ \
  --model_dir scGPT_spatial_v1/ \
  --use_vocab_filter \
  --batch_size 16 \
  --max_seq_len 1200

echo "scGPT-spatial pipeline finished successfully!"

python "$SCRIPT" \
  --adata_path test_data/10X_Xenium_Breast_FFPE/adata.h5ad \
  --output_dir test_data/10X_Xenium_Breast_FFPE/embeddings/ \
  --model_dir scGPT_spatial_v1/ \
  --batch_size 16 \
  --max_seq_len 1200

echo "scGPT-spatial pipeline finished successfully!"

python "$SCRIPT" \
  --adata_path test_data/10X_VisiumHD_LUAD_FFPE/adata.h5ad \
  --output_dir test_data/10X_VisiumHD_LUAD_FFPE/embeddings/ \
  --model_dir scGPT_spatial_v1/ \
  --batch_size 16 \
  --max_seq_len 1200

echo "scGPT-spatial pipeline finished successfully!"
