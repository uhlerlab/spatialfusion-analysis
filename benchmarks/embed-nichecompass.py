#!/usr/bin/env python3
import argparse
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler

from nichecompass.models import NicheCompass
from nichecompass.utils import (
    add_gps_from_gp_dict_to_adata,
    compute_communication_gp_network,
    visualize_communication_gp_network,
    create_new_color_dict,
    extract_gp_dict_from_mebocost_ms_interactions,
    extract_gp_dict_from_nichenet_lrt_interactions,
    extract_gp_dict_from_omnipath_lr_interactions,
    filter_and_combine_gp_dict_gps_v2,
    generate_enriched_gp_info_plots,
)


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run NicheCompass pipeline")

    # Input data
    p.add_argument("--adata_path", type=str, required=True,
                   help="Path to input .h5ad")

    # Folder paths
    p.add_argument("--ga_data_folder", type=str, required=True)
    p.add_argument("--gp_data_folder", type=str, required=True)
    p.add_argument("--so_data_folder", type=str, required=True)
    p.add_argument("--artifacts_folder", type=str, required=True)
    p.add_argument("--embedding_output_dir", type=str, required=True,
                   help="Directory where the nichecompass_latent embeddings will be saved")

    # Metadata
    p.add_argument("--species", type=str, default="human")
    p.add_argument("--dataset", type=str, default="ovarian_cancer")
    p.add_argument("--spatial_key", type=str, default="spatial")
    p.add_argument("--n_neighbors", type=int, default=4)

    return p.parse_args()


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------
def main():
    args = parse_args()

    # Unpack
    species = args.species
    spatial_key = args.spatial_key
    n_neighbors = args.n_neighbors

    # Timestamp
    current_timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Build paths
    ga_data_folder_path = Path(args.ga_data_folder)
    gp_data_folder_path = Path(args.gp_data_folder)
    so_data_folder_path = Path(args.so_data_folder)
    artifacts_folder_path = Path(args.artifacts_folder)

    omnipath_lr_network_file_path = gp_data_folder_path / "omnipath_lr_network.csv"
    collectri_tf_network_file_path = gp_data_folder_path / \
        f"collectri_tf_network_{species}.csv"
    nichenet_lr_network_file_path = gp_data_folder_path / \
        f"nichenet_lr_network_v2_{species}.csv"
    nichenet_ligand_target_matrix_file_path = gp_data_folder_path / \
        f"nichenet_ligand_target_matrix_v2_{species}.csv"
    mebocost_interactions_folder = gp_data_folder_path / "metabolite_enzyme_sensor_gps"
    gene_orthologs_mapping_file_path = ga_data_folder_path / \
        "human_mouse_gene_orthologs.csv"

    # Output dirs
    model_folder_path = artifacts_folder_path / \
        "single_sample" / current_timestamp / "model"
    figure_folder_path = artifacts_folder_path / \
        "single_sample" / current_timestamp / "figures"

    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)
    os.makedirs(so_data_folder_path, exist_ok=True)

    # -------------------------------------------------
    # Retrieve Gene Programs
    # -------------------------------------------------
    print("Extracting OmniPath gene programs...")
    omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
        species=species,
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=str(omnipath_lr_network_file_path),
        gene_orthologs_mapping_file_path=str(gene_orthologs_mapping_file_path),
        plot_gp_gene_count_distributions=True,
        gp_gene_count_distributions_save_path=str(
            figure_folder_path / "omnipath_gp_gene_count_distributions.svg"
        ),
    )

    # Show example OmniPath GP
    omnipath_gp_names = list(omnipath_gp_dict.keys())
    random.shuffle(omnipath_gp_names)
    ex_omni = omnipath_gp_names[0]
    print(f"\nExample OmniPath GP:\n{ex_omni}: {omnipath_gp_dict[ex_omni]}")

    print("Extracting NicheNet gene programs...")
    nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
        species=species,
        version="v2",
        keep_target_genes_ratio=1.0,
        max_n_target_genes_per_gp=250,
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=str(nichenet_lr_network_file_path),
        ligand_target_matrix_file_path=str(
            nichenet_ligand_target_matrix_file_path),
        gene_orthologs_mapping_file_path=str(gene_orthologs_mapping_file_path),
        plot_gp_gene_count_distributions=True,
    )

    # Show example NicheNet GP
    nichenet_gp_names = list(nichenet_gp_dict.keys())
    random.shuffle(nichenet_gp_names)
    ex_nn = nichenet_gp_names[0]
    print(f"\nExample NicheNet GP:\n{ex_nn}: {nichenet_gp_dict[ex_nn]}")

    print("Extracting MEBOCOST gene programs...")
    mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
        dir_path=str(mebocost_interactions_folder),
        species=species,
        plot_gp_gene_count_distributions=True,
    )

    mebocost_gp_names = list(mebocost_gp_dict.keys())
    random.shuffle(mebocost_gp_names)
    ex_mebo = mebocost_gp_names[0]
    print(f"\nExample MEBOCOST GP:\n{ex_mebo}: {mebocost_gp_dict[ex_mebo]}")

    print("\nFiltering & combining GPs...")
    gp_dicts = [omnipath_gp_dict, nichenet_gp_dict, mebocost_gp_dict]
    combined_gp_dict = filter_and_combine_gp_dict_gps_v2(
        gp_dicts, verbose=True)
    print(f"Final number of combined gene programs: {len(combined_gp_dict)}")

    # -------------------------------------------------
    # Read dataset
    # -------------------------------------------------
    print("\nReading AnnData...")
    adata = sc.read_h5ad(args.adata_path)
    adata.layers["counts"] = adata.X.copy()

    # Spatial neighbors
    print("Computing spatial neighbors...")
    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        spatial_key=spatial_key,
        n_neighs=n_neighbors,
    )

    # Symmetrize adjacency matrix
    adj_key = "spatial_connectivities"
    adata.obsp[adj_key] = adata.obsp[adj_key].maximum(adata.obsp[adj_key].T)

    # -------------------------------------------------
    # Add Gene Programs
    # -------------------------------------------------
    print("Adding gene programs to AnnData...")
    add_gps_from_gp_dict_to_adata(
        gp_dict=combined_gp_dict,
        adata=adata,
        gp_targets_mask_key="nichecompass_gp_targets",
        gp_targets_categories_mask_key="nichecompass_gp_targets_categories",
        gp_sources_mask_key="nichecompass_gp_sources",
        gp_sources_categories_mask_key="nichecompass_gp_sources_categories",
        gp_names_key="nichecompass_gp_names",
        min_genes_per_gp=2,
        min_source_genes_per_gp=1,
        min_target_genes_per_gp=1,
    )

    print(f"Nodes: {adata.layers['counts'].shape[0]}")
    print(f"Genes: {adata.layers['counts'].shape[1]}")

    # -------------------------------------------------
    # Train NicheCompass
    # -------------------------------------------------
    print("\nInitializing NicheCompass model...")
    model = NicheCompass(
        adata,
        counts_key="counts",
        adj_key=adj_key,
        gp_names_key="nichecompass_gp_names",
        active_gp_names_key="nichecompass_active_gp_names",
        gp_targets_mask_key="nichecompass_gp_targets",
        gp_targets_categories_mask_key="nichecompass_gp_targets_categories",
        gp_sources_mask_key="nichecompass_gp_sources",
        gp_sources_categories_mask_key="nichecompass_gp_sources_categories",
        latent_key="nichecompass_latent",
        conv_layer_encoder="gcnconv",
        active_gp_thresh_ratio=0.01,
    )

    print("Training model...")
    model.train(
        n_epochs=40,
        n_epochs_all_gps=25,
        lr=0.001,
        lambda_edge_recon=500000.0,
        lambda_gene_expr_recon=300.0,
        lambda_l1_masked=0.0,
        edge_batch_size=1024,
        n_sampled_neighbors=4,
        use_cuda_if_available=True,
        verbose=False,
    )

    print("Saving model + AnnData...")
    model.save(
        dir_path=str(model_folder_path),
        overwrite=True,
        save_adata=True,
        adata_file_name="adata.h5ad",
    )

    # Compute neighbors in latent space
    print("Computing latent neighbors + UMAP...")
    sc.pp.neighbors(model.adata, use_rep="nichecompass_latent",
                    key_added="nichecompass_latent")
    sc.tl.umap(model.adata, neighbors_key="nichecompass_latent")

    # Save final model again
    model.save(
        dir_path=str(model_folder_path),
        overwrite=True,
        save_adata=True,
        adata_file_name="adata.h5ad",
    )

    embedding_output_dir = Path(args.embedding_output_dir)
    embedding_output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        model.adata.obsm["nichecompass_latent"],
        index=model.adata.obs_names
    ).to_parquet(embedding_output_dir / "nichecompass.parquet")

    print(
        f"Saved latent embeddings to: {embedding_output_dir/'nichecompass.parquet'}")


# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
