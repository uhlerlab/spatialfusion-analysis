#!/usr/bin/env python3
import argparse
import pathlib as pl
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import sys

# Add scGPT-spatial to PYTHONPATH
sys.path.append("/ewsc/yatesjos/Broad_SpatialFoundation/scGPT-spatial/")
import scgpt_spatial  # noqa: E402


# ---------------------------------------------------------
# Original embedding function (logic unchanged)
# ---------------------------------------------------------
def run_embed_scGPTspatial(
    dataset_path: pl.Path,
    model_dir: pl.Path,
    output_dir: pl.Path,
    n_hvg: int,
    gene_col: str = "index",
    layer_key: str = "X",
    log_norm: bool = False,
    seed: int = 42,
    max_seq_len: int = 1200,
    batch_size: int = 32,
    input_bins: int = 51,
    model_run: str = "pretrained",
    num_workers: int = 0,
):
    embeddings = scgpt_spatial.tasks.embed_data(
        dataset_path,
        model_dir=model_dir,
        gene_col=gene_col,
        max_length=max_seq_len,
        batch_size=batch_size,
        use_fast_transformer=True,
        return_new_adata=False,
    )

    if "cell_id" in embeddings.obs:
        index_vals = embeddings.obs["cell_id"]
    else:
        index_vals = embeddings.obs_names

    out_path = output_dir / "scGPTspatial.parquet"
    pd.DataFrame(
        embeddings.obsm["X_scGPT"],
        index=index_vals,
    ).to_parquet(out_path)

    print(f"[scGPT-spatial] Saved embeddings → {out_path}")


# ---------------------------------------------------------
# Filtering logic (unchanged)
# ---------------------------------------------------------
def filter_cells_for_vocab(dataset_path: pl.Path, model_dir: pl.Path) -> pl.Path:
    print(f"[Filtering] Checking vocabulary overlap for {dataset_path}")

    adata = ad.read_h5ad(dataset_path)

    vocab_path = model_dir / "vocab.json"
    vocab = set(json.load(open(vocab_path)).keys())

    genes = adata.var_names
    genes_norm = [
        g.split(".")[0].upper() if g.startswith(
            ("ENSG", "ENSMUSG")) else g.upper()
        for g in genes
    ]

    keep_mask = np.array([g in vocab for g in genes_norm])
    X = adata.X

    if hasattr(X, "toarray"):
        pos_per_cell = (X[:, keep_mask] > 0).sum(axis=1).A.ravel()
    else:
        pos_per_cell = (X[:, keep_mask] > 0).sum(axis=1)

    good = pos_per_cell > 0
    bad_idx = np.where(~good)[0]

    print(
        f"[Filtering] Removing {bad_idx.size} cells with no vocab-positive genes.")

    adata_filt = adata[good].copy()
    tmp_path = pl.Path("/tmp/adata_scgpt_ready.h5ad")
    adata_filt.write_h5ad(tmp_path)

    return tmp_path


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run scGPT-spatial embedding on a single dataset.")

    p.add_argument("--adata_path", type=str, required=True,
                   help="Path to input .h5ad file")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save embeddings")
    p.add_argument("--model_dir", type=str, required=True,
                   help="Directory containing scGPT-spatial model files")

    p.add_argument("--use_vocab_filter", action="store_true",
                   help="Filter cells with zero gene overlap before embedding")

    # Model params (same defaults you used)
    p.add_argument("--n_hvg", type=int, default=1200)
    p.add_argument("--gene_col", type=str, default="index")
    p.add_argument("--layer_key", type=str, default="X")
    p.add_argument("--log_norm", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_len", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--input_bins", type=int, default=51)
    p.add_argument("--model_run", type=str, default="pretrained")
    p.add_argument("--num_workers", type=int, default=0)

    return p.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()

    adata_path = pl.Path(args.adata_path)
    output_dir = pl.Path(args.output_dir)
    model_dir = pl.Path(args.model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # Optional Filtering
    if args.use_vocab_filter:
        print("[INFO] Filtering cells before embedding...")
        dataset_to_use = filter_cells_for_vocab(adata_path, model_dir)
    else:
        dataset_to_use = adata_path

    # Run embedding
    run_embed_scGPTspatial(
        dataset_path=dataset_to_use,
        model_dir=model_dir,
        output_dir=output_dir,
        n_hvg=args.n_hvg,
        gene_col=args.gene_col,
        layer_key=args.layer_key,
        log_norm=args.log_norm,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        input_bins=args.input_bins,
        model_run=args.model_run,
        num_workers=args.num_workers,
    )

    print("\n=== Finished scGPT-spatial embedding ===\n")


if __name__ == "__main__":
    main()
