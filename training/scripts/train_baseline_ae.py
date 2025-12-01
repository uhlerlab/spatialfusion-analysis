"""
Train a baseline autoencoder model using the spatialfusion package.
This script is intended to be run as part of the spatialfusion package:
    python -m spatialfusion.train_baseline_ae

Main steps:
- Computes a soft union of genes present in at least 50% of samples.
- Loads and preprocesses baseline features for each sample.
- Trains a paired autoencoder model on the baseline data.
- Saves the trained model and gene list used.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import uuid
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from spatialfusion.utils.baseline_ae_data_loader import load_and_preprocess_sample_baseline
from spatialfusion.models.baseline_multi_ae import PairedAE, PairedDatasetBaseline
import scanpy as sc
from collections import Counter


def get_soft_union(base_path, sample_list):
    """
    Compute the soft union of genes present in at least 50% of samples, excluding unwanted keywords.

    Args:
        base_path (str or Path): Base directory containing sample folders.
        sample_list (list): List of sample names.

    Returns:
        list: Sorted list of genes present in >=50% of samples and not containing unwanted keywords.
    """
    unwanted_keywords = [
        "NegControlCodeword",
        "NegControlProbe",
        "UnassignedCodeword",
        "DeprecatedCodeword",
        "BLANK"
    ]

    sample_gene_lists = [
        set(sc.read_h5ad(f"{base_path}/{s}/adata.h5ad").var_names)
        for s in sample_list
    ]

    # Count how often each gene appears across samples
    gene_counter = Counter(g for genes in sample_gene_lists for g in genes)

    # Threshold: keep genes present in >= 50% of samples
    threshold = len(sample_list) // 2 + 1  # >=50%

    SOFT_UNION_GENE_LIST = sorted([
        g for g, count in gene_counter.items()
        if count >= threshold and not any(bad in g for bad in unwanted_keywords)
    ])

    print(
        f"Keeping {len(SOFT_UNION_GENE_LIST)} genes that are present in at least 50% of samples")
    return SOFT_UNION_GENE_LIST


def get_run_dir(cfg):
    """
    Create a unique run directory for saving outputs and checkpoints.

    Args:
        cfg (DictConfig): Hydra configuration object with training.log_dir.

    Returns:
        tuple[str, str]: (run_dir, run_id)
    """
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}_{run_id}"
    run_dir = os.path.join(cfg.training.log_dir, run_name)
    return run_dir, run_id


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for baseline autoencoder training. Loads data, builds datasets, trains model, and saves outputs.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    run_dir, run_id = get_run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    config_path = os.path.join(run_dir, f"config_{run_id}.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"✓ Saved config to {config_path}")

    SOFT_UNION_GENE_LIST = get_soft_union(
        cfg.dataset.datapath, cfg.dataset.samples)
    print("Saving gene list")
    with open(os.path.join(run_dir, "used_genes.txt"), "w") as f:
        f.write("\n".join(SOFT_UNION_GENE_LIST))

    datasets = []
    d1_dim, d2_dim = None, None

    for sample in tqdm(cfg.dataset.samples):
        std_feat_1, std_feat_2, _ = load_and_preprocess_sample_baseline(
            sample, cfg.dataset.datapath, cfg.dataset.rawpath, SOFT_UNION_GENE_LIST, max_cells=cfg.dataset.max_cells, image_size=224)

        if d2_dim is None:
            d2_dim = std_feat_2.shape[1]
        else:
            assert std_feat_2.shape[1] == d2_dim, f"{sample} d2 dim mismatch"

        datasets.append(PairedDatasetBaseline(std_feat_1, std_feat_2))

    full_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        full_dataset, batch_size=cfg.training.batch_size, num_workers=4, shuffle=True)

    model = PairedAE(
        d2_dim=d2_dim,
        latent_dim=cfg.training.latent_dim,
        resnet_backbone=cfg.training.resnet_backbone,       # e.g., 'resnet18'
        freeze_resnet=cfg.training.freeze_resnet             # True or False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.MSELoss(reduction='mean')

    lambdas = cfg.training.loss_weights

    model.train()
    model_path = os.path.join(
        cfg.training.checkpoint_dir, f"paired_model_{run_id}.pt")
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    # Initialize AMP scaler
    scaler = GradScaler()

    for epoch in tqdm(range(cfg.training.epochs), desc="Training"):
        epoch_losses = {k: 0 for k in [
            "total", "recon1", "recon2", "cross12", "cross21", "align"]}

        for d1, d2 in dataloader:
            d1, d2 = d1.to(device), d2.to(device)

            optimizer.zero_grad()

            with autocast():  # Enable mixed precision for forward + loss
                out = model(d1, d2)

                z1, z2 = out["z1"], out["z2"]
                recon1, recon2 = out["recon1"], out["recon2"]
                cross12, cross21 = out["cross12"], out["cross21"]

                loss_recon1 = criterion(recon1, d1)
                loss_recon2 = criterion(recon2, d2)
                loss_cross12 = criterion(cross12, d2)
                loss_cross21 = criterion(cross21, d1)
                loss_align = criterion(z1, z2)

                loss = (
                    lambdas.recon1 * loss_recon1 +
                    lambdas.recon2 * loss_recon2 +
                    lambdas.cross12 * loss_cross12 +
                    lambdas.cross21 * loss_cross21 +
                    lambdas.align * loss_align
                )

            # Backprop with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses["total"] += loss.item()
            epoch_losses["recon1"] += loss_recon1.item()
            epoch_losses["recon2"] += loss_recon2.item()
            epoch_losses["cross12"] += loss_cross12.item()
            epoch_losses["cross21"] += loss_cross21.item()
            epoch_losses["align"] += loss_align.item()

        for k in epoch_losses:
            epoch_losses[k] /= len(full_dataset)
            writer.add_scalar(f"Loss/{k}", epoch_losses[k], epoch)

        print(
            f"Epoch {epoch+1:03d} | Total: {epoch_losses['total']:.4f} | "
            f"Recon1: {epoch_losses['recon1']:.4f} | Recon2: {epoch_losses['recon2']:.4f} | "
            f"Cross12: {epoch_losses['cross12']:.4f} | Cross21: {epoch_losses['cross21']:.4f} | "
            f"Align: {epoch_losses['align']:.4f}"
        )

    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved model to {model_path}")

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    # Allow running as a module: python -m spatialfusion.train_baseline_ae
    main()
