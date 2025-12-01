#!/usr/bin/env bash
set -e

# Path to the Python script
SCRIPT="embed-nicheformer.py"

###############################
# Dataset 1
###############################
python "$SCRIPT" \
  --dataset_path "test_data/10X_Xenium_Ovarian_5k/adata.h5ad" \
  --model_path "/nicheformer_model/nicheformer.ckpt" \
  --vocab_path "nicheformer_model/model.h5ad" \
  --tech_mean_path "nicheformer_model/xenium_mean_script.npy" \
  --output_dir "test_data/10X_Xenium_Ovarian_5k/embeddings/" \
  --modality "spatial" \
  --species "human" \
  --technology "Xenium" \
  --max_seq_len 1500 \
  --batch_size 32 \
  --chunk_size 1000 \
  --num_workers 0


###############################
# Dataset 2
###############################
python "$SCRIPT" \
  --dataset_path "test_data/10X_Xenium_Breast_FFPE/adata.h5ad" \
  --model_path "nicheformer_model/nicheformer.ckpt" \
  --vocab_path "nicheformer_model/model.h5ad" \
  --tech_mean_path "nicheformer_model/xenium_mean_script.npy" \
  --output_dir "test_data/10X_Xenium_Breast_FFPE/embeddings/" \
  --modality "spatial" \
  --species "human" \
  --technology "Xenium" \
  --max_seq_len 1500 \
  --batch_size 32 \
  --chunk_size 1000 \
  --num_workers 0


###############################
# Dataset 3
###############################
python "$SCRIPT" \
  --dataset_path "hest_processed_data/TENX157/adata.h5ad" \
  --model_path "nicheformer_model/nicheformer.ckpt" \
  --vocab_path "nicheformer_model/model.h5ad" \
  --tech_mean_path "nicheformer_model/xenium_mean_script.npy" \
  --output_dir "hest_processed_data/TENX157/embeddings/" \
  --modality "spatial" \
  --species "human" \
  --technology "Xenium" \
  --max_seq_len 1500 \
  --batch_size 16 \
  --chunk_size 1000 \
  --num_workers 0
