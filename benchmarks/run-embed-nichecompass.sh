#!/usr/bin/env bash
set -e

# Path to your Python script
SCRIPT="embed-nichecompass.py"

# --------- ORIGINAL ARGUMENT VALUES (converted to argparse) ---------

python "$SCRIPT" \
  --adata_path "test_data/10X_Xenium_Ovarian_5k/adata.h5ad" \
  --ga_data_folder "nichecompass-data/gene_annotations" \
  --gp_data_folder "nichecompass-data/gene_programs" \
  --so_data_folder "nichecompass-data/spatial_omics" \
  --artifacts_folder "nichecompass-data/" \
  --embedding_output_dir "test_data/10X_Xenium_Ovarian_5k/embeddings/" \
  --species "human" \
  --dataset "ovarian_cancer" \
  --spatial_key "spatial" \
  --n_neighbors 4

echo "NicheCompass run completed successfully!"

python "$SCRIPT" \
  --adata_path "test_data/10X_Xenium_Breast_FFPE/adata.h5ad" \
  --ga_data_folder "nichecompass-data/gene_annotations" \
  --gp_data_folder "nichecompass-data/gene_programs" \
  --so_data_folder "nichecompass-data/spatial_omics" \
  --artifacts_folder "nichecompass-data/" \
  --embedding_output_dir "test_data/10X_Xenium_Breast_FFPE/embeddings/" \
  --species "human" \
  --dataset "breast_cancer" \
  --spatial_key "spatial" \
  --n_neighbors 4

echo "NicheCompass run completed successfully!"

python "$SCRIPT" \
  --adata_path "hest_processed_data/TENX157/adata.h5ad" \
  --ga_data_folder "nichecompass-data/gene_annotations" \
  --gp_data_folder "nichecompass-data/gene_programs" \
  --so_data_folder "nichecompass-data/spatial_omics" \
  --artifacts_folder "nichecompass-data/" \
  --embedding_output_dir "hest_processed_data/TENX157/embeddings/" \
  --species "human" \
  --dataset "prostate_cancer" \
  --spatial_key "spatial_he" \
  --n_neighbors 4

echo "NicheCompass run completed successfully!"

python "$SCRIPT" \
  --adata_path "test_data/10X_VisiumHD_LUAD_FFPE/adata.h5ad" \
  --ga_data_folder "nichecompass-data/gene_annotations" \
  --gp_data_folder "nichecompass-data/gene_programs" \
  --so_data_folder "nichecompass-data/spatial_omics" \
  --artifacts_folder "nichecompass-data/" \
  --embedding_output_dir "test_data/10X_VisiumHD_LUAD_FFPE/embeddings/" \
  --species "human" \
  --dataset "luad_hd" \
  --spatial_key "spatial" \
  --n_neighbors 4

echo "NicheCompass run completed successfully!"
