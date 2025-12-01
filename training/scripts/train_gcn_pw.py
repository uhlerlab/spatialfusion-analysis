"""
Train a GCN model on pathway embeddings using the spatialfusion package.
+ This script runs from the analysis repo with Hydra configs:
+     python path/to/train_gcn_pw.py --config-path <path-to-conf> --config-name config training=training_gcn ...

Main steps:
- Loads joint pathway embeddings and spatial coordinates for each sample.
- Builds kNN graphs and overlapping subgraphs for each sample.
- Optionally loads pathway activation labels for supervised training.
- Trains a GCN autoencoder model on the graphs.
- Saves the trained model and training loss plots.
"""

import torch
import sys
import gc
import uuid
import pathlib as pl
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
import dgl
import hydra
from omegaconf import DictConfig, OmegaConf
from collections.abc import Mapping
from spatialfusion.models.gcn import GCNAutoencoder
from spatialfusion.utils.gcn_utils import (
    build_knn_graph,
    generate_overlapping_subgraphs,
    split_index,
    plot_training_losses
)
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OPENBLAS_MAIN_FREE", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---- Local imports


def resolve_sample(sample_info, default_base) -> tuple[str, pl.Path]:
    """
    Resolve sample information to a name and base path.

    Args:
        sample_info (str or Mapping): Sample identifier or mapping with 'name' and optional 'path'.
        default_base (str or Path): Default base path for samples.

    Returns:
        tuple[str, Path]: (sample_name, sample_base_path)
    """
    if isinstance(sample_info, Mapping):
        name = str(sample_info["name"])
        base = pl.Path(str(sample_info.get("path", default_base)))
    else:
        name = str(sample_info)
        base = pl.Path(str(default_base))
    return name, base


def get_run_dir(cfg) -> tuple[str, str]:
    """
    Create a unique run directory for saving outputs and checkpoints.

    Args:
        cfg (DictConfig): Hydra configuration object with training.checkpoint_dir.

    Returns:
        tuple[str, str]: (run_dir, run_id)
    """
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"gcn_{timestamp}_{run_id}"
    run_dir = os.path.join(cfg.training.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


def standardize_pathways(df: pd.DataFrame, method: str = "robust_z", eps: float = 1e-6, tol: float = 1e-3) -> pd.DataFrame:
    """
    Standardize pathway scores column-wise.

    Args:
        df (pd.DataFrame): Pathway activation scores.
        method (str): 'robust_z' for median/IQR, 'z' for mean/std.
        eps (float): Small value to avoid division by zero.
        tol (float): Threshold for near-zero columns.

    Returns:
        pd.DataFrame: Standardized pathway scores.
    """
    df = df.copy()
    all_near_zero = (df.abs().max(axis=0) < tol)
    if method == "z":
        mu = df.mean(axis=0)
        sigma = df.std(axis=0).replace(0, np.nan)
        out = (df - mu) / (sigma + eps)
    else:  # robust z-score
        med = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)
        iqr = (q3 - q1).replace(0, np.nan)
        out = (df - med) / (iqr + eps)
    out.loc[:, all_near_zero] = 0.0
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def train_gcn_full_graphs(
    graphs, in_dim, hidden_dim, epochs, lr,
    lambda_cls, lambda_reg, batch_size,
    node_mask_ratio, num_layers, use_cls_loss, device,
    use_huber: bool = True
):
    """
    Train a GCN autoencoder model on a list of DGL graphs.

    Args:
        graphs (list): List of DGLGraph objects.
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        lambda_cls (float): Weight for pathway regression loss.
        lambda_reg (float): Weight for latent regularization.
        batch_size (int): Batch size for training.
        node_mask_ratio (float): Ratio of nodes to mask for reconstruction.
        num_layers (int): Number of GCN layers.
        use_cls_loss (bool): Whether to use pathway regression loss.
        device (torch.device): Device to train on.
        use_huber (bool): Use Huber loss for regression if True.

    Returns:
        model (GCNAutoencoder): Trained model.
        loss_history (dict): Training loss history.
    """
    loader = GraphDataLoader(
        graphs, batch_size=batch_size, shuffle=True, num_workers=0)

    # Infer number of pathway targets from the first graph that has labels
    if use_cls_loss and "label" in graphs[0].ndata:
        n_classes = graphs[0].ndata["label"].shape[1]
    else:
        n_classes = 0

    model = GCNAutoencoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=in_dim,
        node_mask_ratio=node_mask_ratio,
        num_layers=num_layers,
        n_classes=n_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=0.5)  # Huber with small transition point

    loss_history = {"total": [], "feat": [], "reg": [],
                    "cls": []}  # 'cls' now means pathway loss

    model.train()
    for epoch in range(epochs):
        epoch_total = epoch_feat = epoch_reg = epoch_cls = 0.0
        print(f"\nEpoch {epoch + 1}/{epochs}")
        batch_bar = tqdm(loader, desc="Batches", leave=False)

        for batch in batch_bar:
            batch = batch.to(device)
            x_recon, x_true, node_mask, z, logits = model(batch)

            # Reconstruction on masked nodes
            loss_feat = mse_loss(x_recon[node_mask], x_true[node_mask])

            # Latent L2 regularization
            loss_reg = (z ** 2).mean()

            # Pathway regression loss (continuous targets)
            if use_cls_loss and logits is not None and "label" in batch.ndata:
                targets = batch.ndata["label"]
                loss_cls = huber(logits, targets) if use_huber else mse_loss(
                    logits, targets)
            else:
                loss_cls = torch.tensor(0.0, device=device)

            loss = loss_feat + lambda_cls * loss_cls + lambda_reg * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += float(loss.item())
            epoch_feat += float(loss_feat.item())
            epoch_cls += float(loss_cls.item())
            epoch_reg += float(loss_reg.item())

        n_batches = len(loader)
        loss_history["total"].append(epoch_total / n_batches)
        loss_history["feat"].append(epoch_feat / n_batches)
        loss_history["cls"].append(epoch_cls / n_batches)
        loss_history["reg"].append(epoch_reg / n_batches)

        print(f"Epoch {epoch+1:02d} Summary | "
              f"Total: {epoch_total / n_batches:.4f} | "
              f"Feat: {epoch_feat / n_batches:.4f} | "
              f"Path: {epoch_cls / n_batches:.4f} | "
              f"Reg: {epoch_reg / n_batches:.4f}")

    return model, loss_history


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for GCN training. Loads data, builds graphs, trains model, and saves outputs.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu")
    run_dir, run_id = get_run_dir(cfg)

    with open(os.path.join(run_dir, f"config_{run_id}.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # ============================
    # combine_mode: "concat" | "average" | "z1" | "z2"
    # ============================
    combine_mode = getattr(cfg.training, "combine_mode", "concat").lower()
    if combine_mode not in {"concat", "average", "z1", "z2"}:
        raise ValueError(
            "cfg.training.combine_mode must be one of: 'concat', 'average', 'z1', 'z2'.")

    def _load_df(pathlike: pl.Path) -> pd.DataFrame:
        if pathlike.suffix == ".parquet":
            return pd.read_parquet(pathlike)
        elif pathlike.suffix == ".csv":
            return pd.read_csv(pathlike, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {pathlike.name}")

    # If a precomputed joint embedding is supplied, use it as-is
    if hasattr(cfg.eval, "zfile") and cfg.eval.zfile:
        zpath = pl.Path(cfg.eval.embedding_dir) / cfg.eval.zfile
        if zpath.suffix == ".parquet":
            z_joint = pd.read_parquet(zpath)
        elif zpath.suffix == ".csv":
            z_joint = pd.read_csv(zpath, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {zpath.name}")
        z_joint = z_joint.loc[~z_joint.index.duplicated()]
        print(f"Loaded precomputed joint embedding with shape {z_joint.shape}")

    else:
        # Build from z1 and/or z2 according to combine_mode
        need_z1 = combine_mode in {"concat", "average", "z1"}
        need_z2 = combine_mode in {"concat", "average", "z2"}

        if need_z1:
            assert hasattr(
                cfg.eval, "z1file") and cfg.eval.z1file, "combine_mode requires eval.z1file"
        if need_z2:
            assert hasattr(
                cfg.eval, "z2file") and cfg.eval.z2file, "combine_mode requires eval.z2file"

        z1 = None
        z2 = None

        if need_z1:
            z1_path = pl.Path(cfg.eval.embedding_dir) / cfg.eval.z1file
            z1 = _load_df(z1_path).loc[lambda df: ~
                                       df.index.duplicated()].copy()
        if need_z2:
            z2_path = pl.Path(cfg.eval.embedding_dir) / cfg.eval.z2file
            z2 = _load_df(z2_path).loc[lambda df: ~
                                       df.index.duplicated()].copy()

        if combine_mode == "z1":
            z_joint = z1
            print(f"Using z1 only → z_joint shape: {z_joint.shape}")

        elif combine_mode == "z2":
            z_joint = z2
            print(f"Using z2 only → z_joint shape: {z_joint.shape}")

        elif combine_mode == "concat":
            # Align on common cells
            common_idx = z1.index.intersection(z2.index)
            if len(common_idx) == 0:
                raise ValueError(
                    "concat mode: z1 and z2 have no overlapping cells (index).")
            z1c = z1.loc[common_idx]
            z2c = z2.loc[common_idx]
            # Ensure unique column names after concat (optional but safer)
            z1c.columns = [f"z1_{c}" for c in z1c.columns]
            z2c.columns = [f"z2_{c}" for c in z2c.columns]
            z_joint = pd.concat([z1c, z2c], axis=1)
            print(f"Concatenated z1 and z2 → z_joint shape: {z_joint.shape}")

        else:  # combine_mode == "average"
            # Align on common cells and shared dimensions
            common_idx = z1.index.intersection(z2.index)
            if len(common_idx) == 0:
                raise ValueError(
                    "average mode: z1 and z2 have no overlapping cells (index).")
            z1c = z1.loc[common_idx]
            z2c = z2.loc[common_idx]
            # Only shared columns; preserve z1's order for stability
            shared_cols = [c for c in z1c.columns if c in set(z2c.columns)]
            if len(shared_cols) == 0:
                raise ValueError(
                    "average mode requires overlapping latent dimensions (columns) between z1 and z2.")
            z_joint = (z1c[shared_cols] + z2c[shared_cols]) / 2.0
            print(
                f"Averaged z1 and z2 over {len(shared_cols)} shared dims → z_joint shape: {z_joint.shape}")

    graphs = []

    for sample_info in tqdm(cfg.dataset.samples, desc="Processing Samples"):
        sample, sample_base = resolve_sample(sample_info, cfg.dataset.datapath)
        datapath = sample_base / sample

        # ---- adata ----
        adata_path = datapath / "adata.h5ad"
        if not adata_path.exists():
            print(f"[{sample}] Missing {adata_path}. Skipping.")
            continue
        adata = sc.read_h5ad(adata_path)

        # ---- index overlap with z_joint ----
        new_idx = z_joint.index.intersection(adata.obs_names)
        if len(new_idx) == 0:
            print(f"[{sample}] No index overlap. Skipping.")
            continue

        adata = adata[new_idx].copy()
        joint_emb_df = z_joint.loc[new_idx]

        # standardize joint_emb safely (column-wise z-score with eps)
        eps = 1e-6
        joint_emb = ((joint_emb_df - joint_emb_df.mean())
                     / (joint_emb_df.std().replace(0, np.nan) + eps)).fillna(0.0).astype(np.float32).values

        # ---- coords: support either key ----
        if "spatial_he" in adata.obsm_keys():
            coords = adata.obsm["spatial_he"]
        elif "spatial" in adata.obsm_keys():
            coords = adata.obsm["spatial"]
        else:
            print(
                f"[{sample}] No 'spatial_he' or 'spatial' in adata.obsm. Skipping.")
            continue

        coords = coords.astype(np.float32)
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + eps)

        full_graph = build_knn_graph(coords, k=cfg.dataset.knn_k)
        full_graph.ndata["feat"] = torch.tensor(joint_emb, dtype=torch.float32)

        subgraphs = generate_overlapping_subgraphs(
            full_graph, coords,
            subgraph_size=cfg.dataset.subgraph_size,
            stride=cfg.dataset.stride
        )

        # ---- optional pathway targets (continuous) ----
        if cfg.training.use_cls_loss:
            path_candidates = [
                sample_base / sample / "pathway_activation.parquet",
                pl.Path(str(cfg.dataset.datapath)) /
                sample / "pathway_activation.parquet"
            ]
            path_candidates = [p for p in path_candidates if p is not None]
            label_path = next((p for p in path_candidates if p.exists()), None)

            if label_path is None:
                print(
                    f"[{sample}] No pathway_activation.parquet found. Skipping labels for this sample.")
                full_labels = None
            else:
                df_labels = pd.read_parquet(label_path)
                df_labels = df_labels.loc[adata.obs_names]
                df_labels = standardize_pathways(df_labels, method="robust_z")
                full_labels = torch.tensor(
                    df_labels.values, dtype=torch.float32)
        else:
            full_labels = None

        for sg in subgraphs:
            if cfg.training.use_cls_loss and full_labels is not None:
                orig_node_ids = sg.ndata[dgl.NID].numpy()
                sg.ndata["label"] = full_labels[orig_node_ids]
            sg = dgl.add_self_loop(sg)
            graphs.append(sg)

    model, loss_history = train_gcn_full_graphs(
        graphs=graphs,
        in_dim=graphs[0].ndata["feat"].shape[1],
        hidden_dim=cfg.training.hidden_dim,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        lambda_cls=cfg.training.lambda_cls,
        lambda_reg=cfg.training.lambda_reg,
        batch_size=cfg.training.batch_size,
        node_mask_ratio=cfg.training.node_mask_ratio,
        num_layers=cfg.training.num_layers,
        use_cls_loss=cfg.training.use_cls_loss,
        device=device
    )

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    plot_training_losses(loss_history, os.path.join(
        run_dir, f"gcn_losses_{run_id}.png"))
    print(f"✓ Model and losses saved to {run_dir}")


if __name__ == "__main__":
    # Allow running as a module: python -m spatialfusion.train_gcn_pw
    main()
