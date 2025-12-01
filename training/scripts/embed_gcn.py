"""
Script for extracting and saving GCN embeddings for all samples using a trained GCNAutoencoder model.

Main steps:
- Loads z1/z2 AE embeddings and sample metadata.
- Combines z1/z2 into a joint embedding (concat/average/z1/z2).
- Builds kNN graphs for each sample.
- Loads trained GCN model and applies it to each graph.
- Extracts GCN embeddings and merges with cell type, spatial, and ligand-receptor metadata.
- Saves embeddings to disk.
"""
# embed_gcn.py

import os
import torch
import pandas as pd
import numpy as np
import pathlib as pl
from pathlib import Path
from tqdm import tqdm
import scanpy as sc
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra
import dgl

from spatialfusion.models.gcn import GCNAutoencoder
from spatialfusion.utils.gcn_utils import build_knn_graph, split_index
from spatialfusion.utils.pkg_ckpt import resolve_pkg_ckpt  # to load packaged ckpts


# ----------------------------
# Helpers
# ----------------------------
def _combine_embeddings(z1: pd.DataFrame, z2: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Build the joint embedding according to `mode`:
      - 'average': (z1 + z2)/2  (on shared columns)
      - 'concat' : [z1 | z2] with prefixed columns
      - 'z1'     : z1 only
      - 'z2'     : z2 only
    Assumes z1 and z2 will be aligned on their common cell index.
    """
    mode = mode.lower()
    if mode not in {"average", "concat", "z1", "z2"}:
        raise ValueError(
            "combine_mode must be one of: 'average', 'concat', 'z1', 'z2'")

    if mode == "z1":
        return z1.copy()
    if mode == "z2":
        return z2.copy()

    common_idx = z1.index.intersection(z2.index)
    if len(common_idx) == 0:
        raise ValueError(
            f"{mode} mode: z1 and z2 have no overlapping cells (index).")

    z1c = z1.loc[common_idx]
    z2c = z2.loc[common_idx]

    if mode == "concat":
        z1c = z1c.copy()
        z2c = z2c.copy()
        z1c.columns = [f"z1_{c}" for c in z1c.columns]
        z2c.columns = [f"z2_{c}" for c in z2c.columns]
        return pd.concat([z1c, z2c], axis=1)

    # average
    shared_cols = [c for c in z1c.columns if c in set(z2c.columns)]
    if len(shared_cols) == 0:
        raise ValueError(
            "average mode requires overlapping columns between z1 and z2.")
    return (z1c[shared_cols] + z2c[shared_cols]) / 2.0


def _safe_standardize(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Column-wise standardization with numerical safety."""
    mu = df.mean(axis=0)
    sigma = df.std(axis=0).replace(0, np.nan)
    out = (df - mu) / (sigma + eps)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _get_gcn_ckpt_path(cfg: DictConfig) -> Path:
    """
    Resolve GCN checkpoint:
      1) If cfg.pretrained.gcn_path (or env GCN_CKPT) is set -> use it
      2) Else load packaged file via importlib.resources using relpath:
         cfg.pretrained.gcn_full_relpath (or gcn_he_relpath).
    """
    if hasattr(cfg, "pretrained") and getattr(cfg.pretrained, "gcn_path", None):
        return Path(cfg.pretrained.gcn_path)

    rel = getattr(cfg.pretrained, "gcn_full_relpath", None) or getattr(
        cfg.pretrained, "gcn_he_relpath", None
    )
    if not rel:
        raise ValueError(
            "No GCN checkpoint specified: set pretrained.gcn_path or pretrained.gcn_full_relpath/gcn_he_relpath in config."
        )
    return resolve_pkg_ckpt(rel)


@torch.no_grad()
def extract_gcn_embeddings_with_metadata(
    model,
    graphs,
    sample_list,
    base_path,
    z_joint,
    device="cuda",
    spatial_key="spatial_he",
    celltype_key="celltypes",
):
    """
    Extracts GCN embeddings along with metadata such as spatial coordinates and cell types.
    """
    model.eval()
    all_dfs = []

    for g, sample in tqdm(list(zip(graphs, sample_list)), total=len(sample_list)):
        datapath = pl.Path(base_path) / sample
        adata = sc.read_h5ad(datapath / "adata.h5ad")

        celltypes_path = datapath / "celltypes.csv"
        celltypes_df = pd.read_csv(
            celltypes_path, index_col=0) if celltypes_path.exists() else None

        # Embeddings
        g = g.to(device)
        x = g.ndata["feat"].to(device)
        z = model.encode(dgl.add_self_loop(g), x)
        z = F.dropout(z, p=model.dropout, training=False)
        z_np = z.cpu().numpy()

        valid_idx = z_joint.index.intersection(adata.obs_names)
        df = pd.DataFrame(z_np, index=valid_idx)
        df["sample_id"] = sample
        df["cell_id"] = valid_idx

        # Cell type info
        if celltypes_df is not None:
            overlapping_ids = df.index.intersection(celltypes_df.index)
            df["celltype"] = (
                celltypes_df.loc[overlapping_ids].reindex(
                    df.index)["celltypes"].values
            )
            if "cellsubtypes" in celltypes_df.columns:
                df["cellsubtype"] = (
                    celltypes_df.loc[overlapping_ids].reindex(
                        df.index)["cellsubtypes"].values
                )
        else:
            if celltype_key in adata.obs.columns:
                df["celltype"] = adata.obs[celltype_key].values
            elif "final_lineage" in adata.obs.columns:
                df["celltype"] = adata.obs["final_lineage"].values

        for key in ["CNiche", "TNiche"]:
            if key in adata.obs.columns:
                df[key] = adata.obs[key].values

        # Spatial coords
        skey = spatial_key if spatial_key in adata.obsm_keys() else (
            "spatial" if "spatial" in adata.obsm_keys() else None
        )
        if skey is None:
            raise KeyError("No 'spatial_he' or 'spatial' found in adata.obsm.")
        spatial_df = pd.DataFrame(
            adata.obsm[skey], index=adata.obs_names, columns=[
                "X_coord", "Y_coord"]
        )
        df[["X_coord", "Y_coord"]] = spatial_df.loc[df.index][[
            "X_coord", "Y_coord"]].values

        # LR matrix (optional)
        lr_path = datapath / "training_LR_matrix.csv"
        if lr_path.exists():
            lr_df = pd.read_csv(lr_path, index_col=0)
            lr_df = lr_df.loc[df.index]
            df = pd.concat([df, lr_df], axis=1)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# ----------------------------
# Main
# ----------------------------
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for GCN embedding extraction and saving.
    Uses z1/z2 combination logic (concat/average/z1/z2) instead of a precomputed zfile.
    Also resolves the GCN checkpoint via cfg.pretrained.* or packaged weights.
    """
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu")
    base_path = pl.Path(cfg.dataset.datapath)
    out_dir = pl.Path(cfg.eval.embedding_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Select samples
    if cfg.eval.sample_mode == "train":
        sample_list = cfg.dataset.samples
    elif cfg.eval.sample_mode == "test":
        sample_list = cfg.dataset.test_samples
    else:
        raise ValueError("cfg.eval.sample_mode must be 'train' or 'test'")

    # -------- Load z1 / z2 and combine into z_joint ----------
    z1_path = pl.Path(cfg.eval.embedding_dir) / cfg.eval.z1file
    z2_path = pl.Path(cfg.eval.embedding_dir) / cfg.eval.z2file

    def _load_df(p: pl.Path) -> pd.DataFrame:
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        elif p.suffix == ".csv":
            return pd.read_csv(p, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {p.name}")

    if not z1_path.exists() or not z2_path.exists():
        raise FileNotFoundError(f"Missing z1/z2 files: {z1_path} or {z2_path}")

    z1 = _load_df(z1_path).loc[lambda df: ~df.index.duplicated()].copy()
    z2 = _load_df(z2_path).loc[lambda df: ~df.index.duplicated()].copy()

    combine_mode = getattr(cfg.training, "combine_mode", "concat").lower()
    z_joint = _combine_embeddings(z1, z2, combine_mode)
    z_joint = z_joint.loc[~z_joint.index.duplicated()]
    # _, corrected_index = split_index(z_joint.index)  # keep if you later need to normalize IDs

    # -------- Load GCN model ----------
    model_path = _get_gcn_ckpt_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"GCN checkpoint not found: {model_path}")

    in_dim = z_joint.shape[1]
    model = GCNAutoencoder(
        in_dim=in_dim,
        hidden_dim=cfg.training.hidden_dim,
        out_dim=in_dim,
        node_mask_ratio=cfg.training.node_mask_ratio,
        num_layers=cfg.training.num_layers,
        n_classes=0,  # inference: no classifier head needed
    ).to(device)

    state_dict = torch.load(str(model_path), map_location=device)
    # drop classifier weights if present in checkpoint
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
    model.load_state_dict(filtered_state_dict, strict=False)

    # -------- Build graphs ----------
    full_graphs = []
    for sample in tqdm(sample_list, desc="Building graphs"):
        datapath = base_path / sample
        adata = sc.read_h5ad(datapath / "adata.h5ad")

        new_idx = z_joint.index.intersection(adata.obs_names)
        if len(new_idx) == 0:
            print(f"[{sample}] No index overlap. Skipping.")
            continue

        adata = adata[new_idx].copy()
        joint_emb = z_joint.loc[new_idx]

        # numerically safe standardization
        joint_emb = _safe_standardize(
            joint_emb, eps=1e-6).astype(np.float32).values

        # spatial coords: prefer 'spatial_he', fallback to 'spatial'
        if "spatial_he" in adata.obsm_keys():
            coords = adata.obsm["spatial_he"]
        elif "spatial" in adata.obsm_keys():
            coords = adata.obsm["spatial"]
        else:
            print(
                f"[{sample}] No 'spatial_he' or 'spatial' in adata.obsm. Skipping.")
            continue

        coords = coords.astype(np.float32)
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6)

        full_graph = build_knn_graph(coords, k=cfg.dataset.knn_k)
        full_graph.ndata["feat"] = torch.tensor(joint_emb, dtype=torch.float32)

        label_path = datapath / "training_LR_matrix.csv"
        if label_path.exists():
            df_labels = pd.read_csv(label_path, index_col=0)
            df_labels = df_labels.loc[adata.obs_names]
            full_graph.ndata["label"] = torch.tensor(
                df_labels.values, dtype=torch.float32)

        full_graphs.append(full_graph)

    if not full_graphs:
        raise RuntimeError(
            "No graphs were constructed. Check dataset paths and z1/z2 embeddings.")

    # -------- Extract embeddings ----------
    emb_df = extract_gcn_embeddings_with_metadata(
        model,
        full_graphs,
        sample_list,
        base_path,
        z_joint,
        device=device,
        spatial_key="spatial_he",
        celltype_key="celltypes",
    )

    # -------- Save ----------
    out_path = out_dir / f"gcn_embeddings_{cfg.eval.sample_mode}.parquet"
    emb_df.to_parquet(out_path)
    print(f"✓ Embeddings saved to {out_path}")


if __name__ == "__main__":
    main()
