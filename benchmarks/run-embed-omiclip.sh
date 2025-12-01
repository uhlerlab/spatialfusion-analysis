#!/usr/bin/env bash
set -e

SCRIPT="embed-omiclip.py"

python "$SCRIPT" \
  --data_dir "OmiCLIP_model/" \
  --adata_path "test_data/10X_Xenium_Ovarian_5k/adata.h5ad" \
  --housekeeping_gmt "OmiCLIP_model/HSIAO_HOUSEKEEPING_GENES.v2025.1.Hs.gmt" \
  --output_dir "test_data/10X_Xenium_Ovarian_5k/embeddings/" \
  --wsi_path "test_data/10X_Xenium_Ovarian_5k/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif" \
  --spatial_key "spatial_px" \
  --device "cuda"

echo "OmiCLIP pipeline finished successfully!"

python "$SCRIPT" \
  --data_dir "OmiCLIP_model/" \
  --adata_path "test_data/10X_Xenium_Breast_FFPE/adata.h5ad" \
  --housekeeping_gmt "OmiCLIP_model/HSIAO_HOUSEKEEPING_GENES.v2025.1.Hs.gmt" \
  --output_dir "test_data/10X_Xenium_Breast_FFPE/embeddings/" \
  --wsi_path "test_data/10X_Xenium_Breast_FFPE/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif" \
  --spatial_key "spatial_px" \
  --device "cuda"

echo "OmiCLIP pipeline finished successfully!"

python "$SCRIPT" \
  --data_dir "OmiCLIP_model/" \
  --adata_path "hest_processed_data/TENX157/adata.h5ad" \
  --housekeeping_gmt "OmiCLIP_model/HSIAO_HOUSEKEEPING_GENES.v2025.1.Hs.gmt" \
  --output_dir "hest_processed_data/TENX157/embeddings/" \
  --wsi_path "hest_processed_data/TENX157/Xenium_Prime_Human_Prostate_FFPE_he_image.ome.tif" \
  --spatial_key "spatial_he" \
  --device "cuda"

echo "OmiCLIP pipeline finished successfully!"

python "$SCRIPT" \
  --data_dir "OmiCLIP_model/" \
  --adata_path "test_data/10X_VisiumHD_LUAD_FFPE/adata.h5ad" \
  --housekeeping_gmt "OmiCLIP_model/HSIAO_HOUSEKEEPING_GENES.v2025.1.Hs.gmt" \
  --output_dir "test_data/10X_VisiumHD_LUAD_FFPE/embeddings/" \
  --wsi_path "test_data/10X_VisiumHD_LUAD_FFPE/Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_tissue_image.btf" \
  --spatial_key "spatial" \
  --device "cuda"

echo "OmiCLIP pipeline finished successfully!"