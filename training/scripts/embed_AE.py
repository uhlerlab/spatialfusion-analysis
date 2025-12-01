"""
Script for extracting and saving AE embeddings for all samples using a trained PairedAE model.

Main steps:
- Infers input dimensions from sample embeddings.
- Loads model checkpoint and configuration.
- Extracts embeddings for all samples and saves them to disk.
"""
# embed_AE.py

import os
import pathlib as pl
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc  # noqa: F401 (kept if used inside your utils)

from omegaconf import DictConfig
import hydra
from spatialfusion.models.multi_ae import PairedAE
from spatialfusion.utils.embed_ae_utils import (
    extract_embeddings_for_all_samples,
    save_embeddings_separately,
)

from spatialfusion.utils.pkg_ckpt import resolve_pkg_ckpt


def get_ae_ckpt_path(cfg) -> Path:
    # 1) explicit absolute path
    if getattr(cfg, "pretrained", None) and getattr(cfg.pretrained, "ae_path", ""):
        p = Path(cfg.pretrained.ae_path)
        if p.exists():
            return p

    # 2) paths.checkpoints + ae_relpath
    rel = getattr(cfg.pretrained, "ae_relpath",
                  "checkpoint_dir_ae/spatialfusion-multimodal-ae.pt")
    chk_root = getattr(cfg.paths, "checkpoints", "")
    if chk_root:
        cand = Path(chk_root) / rel
        if cand.exists():
            return cand

    # 3) packaged fallback (editable/site-packages)
    return resolve_pkg_ckpt(rel)


def infer_input_dims(sample_list, base_path: pl.Path) -> Tuple[int, int]:
    """
    Return (UNI_dim, scGPT_dim) by peeking at the first valid sample.
    Supports both .csv and .parquet for each embedding.

    Args:
        sample_list (list): List of sample info (str or dict).
        base_path (Path): Base directory for samples.
    Returns:
        Tuple[int, int]: (UNI_dim, scGPT_dim)
    """
    for sample_info in sample_list:
        # Support legacy string or new dict format
        if isinstance(sample_info, dict):
            sample = str(sample_info["name"])
            datapath = pl.Path(sample_info.get("path", base_path)) / sample
        else:
            sample = str(sample_info)
            datapath = pl.Path(base_path) / sample

        embeddings_path = datapath / "embeddings"

        uni_path = None
        scgpt_path = None
        for ext in (".csv", ".parquet"):
            maybe_uni = embeddings_path / f"UNI{ext}"
            maybe_scgpt = embeddings_path / f"scGPT{ext}"
            if maybe_uni.exists() and maybe_scgpt.exists():
                uni_path = maybe_uni
                scgpt_path = maybe_scgpt
                break

        if not (uni_path and scgpt_path):
            continue

        try:
            if uni_path.suffix == ".csv":
                uni = pd.read_csv(uni_path, index_col=0, nrows=1)
            else:
                uni = pd.read_parquet(uni_path).iloc[:1]

            if scgpt_path.suffix == ".csv":
                scgpt = pd.read_csv(scgpt_path, index_col=0, nrows=1)
            else:
                scgpt = pd.read_parquet(scgpt_path).iloc[:1]

            return uni.shape[1], scgpt.shape[1]
        except Exception as e:
            print(f"⚠️ Skipping {sample} due to read error: {e}")
            continue

    raise ValueError(
        "❌ No valid samples found with both UNI and scGPT embeddings.")


def get_device(cfg: DictConfig) -> torch.device:
    """
    Get the best available torch device, preferring explicit config if provided.

    Args:
        cfg (DictConfig): Hydra configuration object.
    Returns:
        torch.device: Selected device.
    """
    # Prefer explicit cfg if provided; otherwise auto-detect
    if hasattr(cfg, "training") and getattr(cfg.training, "device", None):
        return torch.device(cfg.training.device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for AE embedding extraction and saving.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    device = get_device(cfg)
    print(f"Running embed on device: {device}")

    base_path = pl.Path(cfg.dataset.datapath)

    # Choose samples based on eval.sample_mode
    if cfg.eval.sample_mode == "train":
        sample_list = cfg.dataset.samples
    elif cfg.eval.sample_mode == "test":
        sample_list = cfg.dataset.test_samples
    else:
        raise ValueError("cfg.eval.sample_mode must be 'train' or 'test'")

    out_dir = cfg.eval.embedding_dir
    os.makedirs(out_dir, exist_ok=True)

    # model_path = cfg.eval.checkpoint_path
    model_path = get_ae_ckpt_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    latent_dim = cfg.training.latent_dim

    # Infer input dims (UNI first, scGPT second, per your loader)
    d1_dim, d2_dim = infer_input_dims(sample_list, base_path)

    model = PairedAE(
        d1_dim=d1_dim,                         # UNI dim
        d2_dim=d2_dim,                         # scGPT dim
        latent_dim=latent_dim,
        enc_hidden_dims=cfg.training.enc_hidden_dims,
        dec_hidden_dims=cfg.training.dec_hidden_dims,
    )
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Extract embeddings for all samples (expects model.forward to return z1,z2,...)
    with torch.no_grad():
        z1, z2, z_joint, celltypes, samples = extract_embeddings_for_all_samples(
            model, sample_list, base_path, device
        )

    # Persist per-sample outputs
    save_embeddings_separately(
        z1, z2, z_joint, celltypes, samples, out_dir, cfg.eval.sample_mode
    )
    print(f"✓ Saved embeddings to {out_dir}")


if __name__ == "__main__":
    main()
