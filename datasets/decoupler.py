#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sc.set_figure_params(figsize=(3, 3), frameon=False)


# ----------------------------
# Core functions
# ----------------------------

def get_pathway_score(
    adata: sc.AnnData,
    spatial_key: str = "spatial_he",
    bw: int = 100,
    cutoff: float = 0.1,
    net: pd.DataFrame = None,
) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Compute pathway activation using decoupler (ULM), after spatial smoothing with KNN on coordinates.

    Steps:
      1) normalize_total + log1p
      2) store normalized counts in adata.layers["norm"]
      3) build spatial KNN graph via dc.pp.knn(..., key=spatial_key)
      4) smooth adata.X with graph connectivities
      5) run dc.mt.ulm to compute scores -> stored in adata.obsm['score_ulm']

    Returns the modified AnnData and the same object for convenience.
    """
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.layers["norm"] = adata.X.copy()
    dc.pp.knn(adata, key=spatial_key, bw=bw, cutoff=cutoff)
    adata.X = adata.obsp[f"{spatial_key}_connectivities"].dot(adata.X)

    dc.mt.ulm(data=adata, net=net)
    _ = dc.pp.get_obsm(adata=adata, key="score_ulm")
    return adata, adata


def plot_spatial_gex(
    adata: sc.AnnData,
    gene_list: List[str],
    layer: str = None,
    spatial_col: str = "spatial_he",
    point_size: int = 5,
):
    """
    Optional helper to visualize spatial expression or pathway scores.
    layer: None -> raw .X, 'norm' -> adata.layers['norm'], 'score_ulm' -> adata.obsm['score_ulm']
    """
    spatial_arr = pd.DataFrame(
        adata.obsm[spatial_col], index=adata.obs_names, columns=["X_coord", "Y_coord"]
    )
    if layer is None:
        plot_df = adata[:, gene_list].copy().to_df()
    elif layer == "norm":
        plot_df = pd.DataFrame(
            adata[:, gene_list].copy().layers[layer],
            index=adata.obs_names,
            columns=gene_list,
        )
    elif layer == "score_ulm":
        plot_df = adata.obsm["score_ulm"][gene_list]
    else:
        raise ValueError("layer must be None, 'norm', or 'score_ulm'.")

    plot_df = pd.concat([plot_df, spatial_arr], axis=1)

    for gene in gene_list:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=plot_df, x="X_coord", y="Y_coord", hue=gene, palette="vlag", s=point_size
        )
        plt.gca().invert_yaxis()
        plt.title(f"{gene}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


# ----------------------------
# Orchestration
# ----------------------------

def parse_dataset_arg(arg: str) -> Tuple[Path, str]:
    """
    Parse a --dataset argument of the form:
        /path/to/root:spatial_key
    """
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            "Each --dataset must be 'ROOT:SPATIAL_KEY', e.g. /data/hest_processed_data:spatial_he"
        )
    root, key = arg.split(":", 1)
    p = Path(root)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"Dataset root does not exist: {p}")
    return p, key


def parse_list_arg(arg: str) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def collect_targets(
    root: Path,
    include: List[str],
    exclude: List[str],
    skip_names: List[str],
) -> List[Path]:
    """
    Collect subdirectories in root to process.
    If include is provided -> use only those names (must exist under root).
    Then drop any in exclude and skip_names.
    """
    if include:
        dirs = [root / name for name in include]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]

    # Apply filters
    selected = []
    for d in dirs:
        name = d.stem
        if name in skip_names:
            continue
        if exclude and name in exclude:
            continue
        if not (d / "adata.h5ad").exists():
            # quietly skip if no adata exists
            continue
        selected.append(d)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Compute pathway activation scores (ULM via decoupler) for spatial transcriptomics datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Datasets (repeatable)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_arg,
        help="Dataset spec 'ROOT:SPATIAL_KEY'. Repeat for multiple roots. "
             "Example: /ewsc/.../hest_processed_data:spatial_he",
    )

    # Selection
    parser.add_argument(
        "--include",
        type=parse_list_arg,
        default="",
        help="Comma-separated list of sample folder names to include (defaults to all).",
    )
    parser.add_argument(
        "--exclude",
        type=parse_list_arg,
        default="",
        help="Comma-separated list of sample folder names to exclude.",
    )
    parser.add_argument(
        "--skip-names",
        type=parse_list_arg,
        default="move_embeddings,full_cohort",
        help="Comma-separated list of folder names to always skip.",
    )

    # decoupler / spatial KNN params
    parser.add_argument("--bw", type=int, default=100, help="Bandwidth for dc.pp.knn.")
    parser.add_argument("--cutoff", type=float, default=0.1, help="Cutoff for dc.pp.knn.")

    # Pathway network options
    parser.add_argument(
        "--net",
        choices=["progeny", "hallmark"],
        default="progeny",
        help="Which decoupler operator to use for pathway network.",
    )
    parser.add_argument(
        "--organism",
        choices=["human", "mouse"],
        default="human",
        help="Organism for decoupler operator.",
    )
    parser.add_argument(
        "--pathways",
        type=parse_list_arg,
        default="Androgen,EGFR,Estrogen,JAK-STAT,MAPK,NFkB,PI3K,TGFb,TNFa,VEGF",
        help="Comma-separated subset of pathways to keep in the output parquet. "
             "If omitted, a default actionable set is used (for PROGENy).",
    )

    # I/O
    parser.add_argument(
        "--outfile-name",
        type=str,
        default="pathway_activation.parquet",
        help="Filename to save per-sample scores under each sample directory.",
    )

    # Optional plotting (for quick checks)
    parser.add_argument(
        "--plot-genes",
        type=parse_list_arg,
        default="",
        help="Comma-separated genes or pathways to scatter-plot (uses layer 'score_ulm' if they match columns there).",
    )
    parser.add_argument(
        "--plot-layer",
        choices=[None, "norm", "score_ulm"],
        default=None,
        help="Layer to plot if --plot-genes is used.",
    )
    parser.add_argument(
        "--plot-point-size",
        type=int,
        default=5,
        help="Point size for scatter plots.",
    )

    args = parser.parse_args()

    # Load pathway network
    if args.net == "progeny":
        net_df = dc.op.progeny(organism=args.organism)
    else:
        net_df = dc.op.hallmark(organism=args.organism)

    keep_cols = args.pathways

    # Process each dataset root
    for root, spatial_key in args.dataset:
        targets = collect_targets(
            root=root,
            include=args.include,
            exclude=args.exclude,
            skip_names=args.skip_names,
        )

        pbar = tqdm(targets, desc=f"Processing: {root}", total=len(targets))
        for sample_dir in pbar:
            sample_name = sample_dir.stem
            pbar.set_postfix_str(sample_name)
            try:
                ad_path = sample_dir / "adata.h5ad"
                adata = sc.read_h5ad(ad_path)

                # Compute scores
                _, adata_scored = get_pathway_score(
                    adata, spatial_key=spatial_key, bw=args.bw, cutoff=args.cutoff, net=net_df
                )

                # Keep only requested pathways (if present)
                score_df = adata_scored.obsm["score_ulm"]
                existing = [p for p in keep_cols if p in score_df.columns]
                if not existing:
                    # If none of the requested pathways exist, keep all
                    existing = list(score_df.columns)
                score_df = score_df.loc[:, existing]

                # Save
                out_path = sample_dir / args.outfile_name
                score_df.to_parquet(out_path)

                # Optional plot for quick inspection
                if args.plot_genes:
                    # choose layer based on whether plotting pathways (in score_ulm) or genes
                    layer = args.plot_layer
                    # if user specified genes that are in score_ulm, layer defaults to score_ulm
                    if layer is None:
                        if all(g in adata_scored.obsm.get("score_ulm", pd.DataFrame()).columns for g in args.plot_genes):
                            layer = "score_ulm"
                    plot_spatial_gex(
                        adata_scored, args.plot_genes, layer=layer, spatial_col=spatial_key, point_size=args.plot_point_size
                    )

            except Exception as e:
                tqdm.write(f"❌ Failed {sample_name}: {e}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
