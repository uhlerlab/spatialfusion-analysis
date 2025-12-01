"""
Train a multi-modal autoencoder model using the NicheFinder package.
This script is intended to be run as part of the NicheFinder package:
    python -m NicheFinder.train_multi_ae

Main steps:
- Loads and preprocesses multi-modal features for each sample.
- Builds paired datasets for training.
- Trains a paired autoencoder model on the multi-modal data.
- Saves the trained model and training loss plots.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import os
import uuid
from datetime import datetime
from tqdm import tqdm

from spatialfusion.utils.ae_data_loader import load_and_preprocess_sample
from spatialfusion.models.multi_ae import PairedDataset, PairedAE


def get_device():
    """
    Get the best available torch device (MPS, CUDA, or CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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
    Main entry point for multi-modal autoencoder training. Loads data, builds datasets, trains model, and saves outputs.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    device = get_device()
    print(f"Running on {device}")

    run_dir, run_id = get_run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    config_path = os.path.join(run_dir, f"config_{run_id}.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"✓ Saved config to {config_path}")

    datasets = []
    d1_dim, d2_dim = None, None

    for sample_info in tqdm(cfg.dataset.samples):
        if isinstance(sample_info, dict) or isinstance(sample_info, DictConfig):
            sample_name = str(sample_info["name"])
            sample_path = str(sample_info.get("path", cfg.dataset.datapath))
        else:
            sample_name = str(sample_info)
            sample_path = str(cfg.dataset.datapath)

        std_feat_1, std_feat_2, _ = load_and_preprocess_sample(
            sample_name, sample_path, cfg.dataset.max_cells
        )

        if d1_dim is None:
            d1_dim, d2_dim = std_feat_1.shape[1], std_feat_2.shape[1]
        else:
            assert std_feat_1.shape[1] == d1_dim, f"{sample_name} d1 dim mismatch"
            assert std_feat_2.shape[1] == d2_dim, f"{sample_name} d2 dim mismatch"

        datasets.append(PairedDataset(std_feat_1, std_feat_2))

    full_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        full_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )

    model = PairedAE(
        d1_dim=d1_dim,
        d2_dim=d2_dim,
        latent_dim=cfg.training.latent_dim,
        enc_hidden_dims=cfg.training.enc_hidden_dims,
        dec_hidden_dims=cfg.training.dec_hidden_dims,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.MSELoss(reduction='mean')

    lambdas = cfg.training.loss_weights

    model.train()
    model_path = os.path.join(
        cfg.training.checkpoint_dir, f"paired_model_{run_id}.pt")
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(cfg.training.epochs), desc="Training"):
        epoch_losses = {k: 0.0 for k in [
            "total", "recon1", "recon2", "cross12", "cross21", "align"]}
        num_batches = 0

        for d1, d2 in dataloader:
            d1, d2 = d1.to(device), d2.to(device)
            num_batches += 1

            optimizer.zero_grad()
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

            loss.backward()
            optimizer.step()

            epoch_losses["total"] += float(loss.item())
            epoch_losses["recon1"] += float(loss_recon1.item())
            epoch_losses["recon2"] += float(loss_recon2.item())
            epoch_losses["cross12"] += float(loss_cross12.item())
            epoch_losses["cross21"] += float(loss_cross21.item())
            epoch_losses["align"] += float(loss_align.item())

        # average over batches
        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)
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
    # Allow running as a module: python -m NicheFinder.train_multi_ae
    main()
