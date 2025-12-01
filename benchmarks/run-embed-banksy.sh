#!/usr/bin/env bash
set -e

# Path to your Python script
SCRIPT="embed-banksy.py"

python "$SCRIPT" \
  --adata_path test_data/10X_Xenium_Ovarian_5k/adata.h5ad \
  --output_dir test_data/10X_Xenium_Ovarian_5k/embeddings/ \
  --sample_id TENXOv5k \
  --lambda_param 0.8 \
  --pca_dim 20 \
  --spatial_key spatial_px

echo "BANKSY run completed successfully!"

python "$SCRIPT" \
  --adata_path test_data/10X_Xenium_Breast_FFPE/adata.h5ad \
  --output_dir test_data/10X_Xenium_Breast_FFPE/embeddings/ \
  --sample_id TENXBreast \
  --lambda_param 0.8 \
  --pca_dim 20 \
  --spatial_key spatial_px

echo "BANKSY run completed successfully!"

python "$SCRIPT" \
  --adata_path hest_processed_data/TENX157/adata.h5ad \
  --output_dir hest_processed_data/TENX157/embeddings/ \
  --sample_id TENX157 \
  --lambda_param 0.8 \
  --pca_dim 20 \
  --spatial_key spatial_he

echo "BANKSY run completed successfully!"

python "$SCRIPT" \
  --adata_path test_data/10X_VisiumHD_LUAD_FFPE/adata.h5ad \
  --output_dir test_data/10X_VisiumHD_LUAD_FFPE/embeddings/ \
  --sample_id VisiumHD_LUAD \
  --lambda_param 0.8 \
  --pca_dim 20 \
  --spatial_key spatial

echo "BANKSY run completed successfully!"