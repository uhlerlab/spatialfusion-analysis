#!/usr/bin/env python3
from banksy_utils.umap_pca import pca_umap
from banksy.embed_banksy import generate_banksy_matrix
from banksy.initialize_banksy import initialize_banksy
from banksy.main import median_dist_to_nearest_neighbour, concatenate_all
from banksy_utils.filter_utils import normalize_total, filter_hvg
from banksy_utils.plot_utils import plot_cell_positions
import matplotlib
import matplotlib.pyplot as plt
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import argparse
import sys
from pathlib import Path

# Add BANKSY package to path
sys.path.append("../../../Broad_SpatialFoundation/Banksy_py/")

matplotlib.rcParams["svg.fonttype"] = "none"

# BANKSY imports


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run BANKSY embedding pipeline.")

    p.add_argument("--adata_path", type=str, required=True,
                   help="Path to input .h5ad file.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory where outputs will be written.")
    p.add_argument("--sample_id", type=str, default="SAMPLE")
    p.add_argument("--lambda_param", type=float, default=0.8)
    p.add_argument("--pca_dim", type=int, default=20)

    # NEW ARGUMENT
    p.add_argument("--spatial_key", type=str, default="spatial_px",
                   help="Key in .obsm containing spatial coordinates.")

    return p.parse_args()


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------
def main():
    args = parse_args()

    adata_path = Path(args.adata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset: {adata_path}")
    rawdata = sc.read_h5ad(adata_path)

    # Validate spatial key
    if args.spatial_key not in rawdata.obsm:
        raise KeyError(
            f"Spatial key '{args.spatial_key}' not found in .obsm. "
            f"Available keys: {list(rawdata.obsm.keys())}"
        )

    # -----------------------------------------------------
    # Add X_coord / Y_coord columns from the chosen spatial key
    # -----------------------------------------------------
    spatial_coords = rawdata.obsm[args.spatial_key]
    rawdata.obs = pd.concat(
        [
            rawdata.obs,
            pd.DataFrame(
                spatial_coords,
                index=rawdata.obs_names,
                columns=["X_coord", "Y_coord"],
            ),
        ],
        axis=1,
    )

    coord_keys = ("X_coord", "Y_coord", args.spatial_key)

    # Visualize raw positions
    plot_cell_positions(
        rawdata,
        rawdata.obs["X_coord"],
        rawdata.obs["Y_coord"],
        coord_keys=coord_keys,
        fig_size=(6, 6),
    )

    # Normalize and HVG
    print("[INFO] Normalizing")
    adata = normalize_total(rawdata.copy())

    print("[INFO] Selecting HVGs")
    adata, adata_allgenes = filter_hvg(
        adata,
        n_top_genes=2000,
        flavor="seurat",
    )

    # Nearest neighbour radius
    print("[INFO] Calculating median NN distance")
    nbrs = median_dist_to_nearest_neighbour(adata, key=coord_keys[2])

    # BANKSY Init
    banksy_dict = initialize_banksy(
        adata,
        coord_keys,
        k_geom=15,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=True,
        plt_nbr_weights=True,
        plt_agf_angles=False,
        plt_theta=True,
    )

    # BANKSY matrix
    lambda_list = [args.lambda_param]
    banksy_dict, banksy_matrix = generate_banksy_matrix(
        adata,
        banksy_dict,
        lambda_list,
        max_m=1,
    )

    # Add non-spatial matrix
    banksy_dict["nonspatial"] = {
        0.0: {"adata": concatenate_all([adata.X], 0, adata=adata)}
    }

    # PCA + UMAP
    pca_umap(
        banksy_dict,
        pca_dims=[args.pca_dim],
        add_umap=False,
        plt_remaining_var=False,
    )

    # Save embeddings
    emb = banksy_dict["scaled_gaussian"][args.lambda_param]["adata"].obsm[
        f"reduced_pc_{args.pca_dim}"
    ]

    emb_df = pd.DataFrame(
        emb,
        index=banksy_dict["scaled_gaussian"][args.lambda_param]["adata"].obs_names,
    )

    out_file = output_dir / f"banksy_{args.lambda_param}.parquet"
    emb_df.to_parquet(out_file)

    print(f"[INFO] BANKSY embeddings saved → {out_file}")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
