#!/usr/bin/env bash
set -euo pipefail

# 1) Build AnnData from SpaceRanger outs, write to VisiumHD-LUAD-processed/<sample>/adata.h5ad
python preprocess-nsclc.py \
  --build-anndata \
  --basedir VisiumHD-LUAD \
  --savedir VisiumHD-LUAD-processed \
  --exclude LIB-065293st1

# 2) Run scGPT embeddings for all processed samples
python preprocess-nsclc.py \
  --run-scgpt \
  --savedir VisiumHD-LUAD-processed \
  --scgpt-model-dir scGPT_model \
  --scgpt-n-hvg 1200 --scgpt-batch-size 16 --scgpt-seed 42

# 3) Run UNI embeddings on a subset and GPU 5
python preprocess-nsclc.py \
  --run-uni \
  --savedir VisiumHD-LUAD-processed \
  --wsi-root VisiumHD-LUAD \
  --uni-model UNI/pytorch_model.bin \
  --uni-device cuda:5 \
  --uni-only LIB-064885st1,LIB-064887st1,LIB-064889st1,LIB-064890st1,LIB-065290st1,LIB-065291st1,LIB-065292st1,LIB-065294st1,LIB-065295st1
