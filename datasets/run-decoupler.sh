#!/usr/bin/env bash
set -euo pipefail

# HEST processed (spatial_he)
python decoupler.py \
  --dataset hest_processed_data:spatial_he

# test_data (spatial_px)
python decoupler.py \
  --dataset test_data:spatial_px

# VisiumHD-LUAD-processed (spatial), skip 'full_cohort'
python decoupler.py \
  --dataset VisiumHD-LUAD-processed:spatial

# VisiumHD-CRC (spatial), only selected samples
python decoupler.py \
  --dataset VisiumHD-CRC:spatial \
  --include P1CRC,P2CRC,P5CRC,P3NAT,P5NAT
