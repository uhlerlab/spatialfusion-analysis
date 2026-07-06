"""
Script for extracting and saving AE embeddings for all samples using a trained PairedAE model.

Main steps:
- Infers input dimensions from sample embeddings.
- Loads model checkpoint and configuration.
- Extracts embeddings for all samples and saves them to disk.
"""
# embed_AE.py

import re
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

HE_EMBEDDINGS = {
    "uni": "UNI",
    "virchow": "Virchow2",
}

RNA_EMBEDDINGS = {
    "scgpt": "scGPT",
    "nicheformer": "nicheformer",
}


RUN_PATTERN = re.compile(
    r"^(?P<he>uni|virchow)_"
    r"(?P<rna>scgpt|nicheformer)_"
    r"(?P<align>full|recon_only)_"
    r"(?P<timestamp>\d{8}-\d{6})_"
    r"(?P<runid>[a-z0-9]{8})$"
)


def parse_run_name(run_dir: Path):
    """
    Parse a checkpoint directory name.

    Example
    -------
    virchow_nicheformer_recon_only_20260529-114521_ef56gh78

    Returns
    -------
    dict or None
    """

    match = RUN_PATTERN.match(run_dir.name)

    if match is None:
        return None

    return match.groupdict()


def get_model_name(cfg):
    """
    Return the model identifier without timestamp/run-id.
    """

    return (
        f"{cfg.training.he_encoder}_"
        f"{cfg.training.rna_encoder}_"
        f"{cfg.training.alignment_mode}"
    )


def find_matching_checkpoint_dir(cfg) -> Path:
    """
    Find the newest checkpoint directory matching the requested:

    - he_encoder
    - rna_encoder
    - alignment_mode

    Example directory names:

        uni_scgpt_full_20260529-101532_ab12cd34
        virchow_nicheformer_recon_only_20260529-114521_ef56gh78

    Returns
    -------
    Path
        Matching checkpoint directory.
    """

    root = Path(cfg.training.checkpoint_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"Checkpoint root does not exist: {root}"
        )

    matches = []

    for run_dir in root.iterdir():

        if not run_dir.is_dir():
            continue

        parsed = parse_run_name(run_dir)

        if parsed is None:
            continue

        if (
            parsed["he"] == cfg.training.he_encoder
            and parsed["rna"] == cfg.training.rna_encoder
            and parsed["align"] == cfg.training.alignment_mode
        ):
            matches.append(run_dir)

    if len(matches) == 0:

        available = sorted(
            [x.name for x in root.iterdir() if x.is_dir()]
        )

        raise RuntimeError(
            f"No checkpoint found for:\n"
            f"  he_encoder={cfg.training.he_encoder}\n"
            f"  rna_encoder={cfg.training.rna_encoder}\n"
            f"  alignment_mode={cfg.training.alignment_mode}\n\n"
            f"Available checkpoint dirs:\n"
            + "\n".join(available)
        )

    matches.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    chosen = matches[0]

    print(
        f"✓ Using checkpoint directory:\n"
        f"  {chosen}"
    )

    return chosen


def get_embedding_dir(cfg) -> Path:
    """
    Create an embedding output directory that matches the
    discovered AE checkpoint run.

    Example
    -------
    Checkpoint:
        uni_scgpt_full_20260529-080720_27cd4408

    Embeddings:
        <embedding_root>/
            uni_scgpt_full_20260529-080720_27cd4408
    """

    ckpt_dir = find_matching_checkpoint_dir(cfg)

    out_dir = (
        Path(cfg.eval.embedding_root)
        / ckpt_dir.name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return out_dir


def get_ae_ckpt_path(cfg) -> Path:
    """
    Resolve AE checkpoint path.

    Resolution order
    ----------------
    1. Automatically find newest checkpoint matching:
           he_encoder
           rna_encoder
           alignment_mode

    2. Explicit checkpoint:
           cfg.pretrained.ae_path

    3. Explicit relative checkpoint:
           cfg.paths.checkpoints / cfg.pretrained.ae_relpath

    4. Packaged fallback checkpoint.
    """

    # --------------------------------------------------
    # 1. Auto-discover matching AE checkpoint
    # --------------------------------------------------

    try:

        ckpt_dir = find_matching_checkpoint_dir(cfg)

        model_path = ckpt_dir / "model.pt"

        if model_path.exists():

            print(
                f"✓ Using discovered checkpoint:\n"
                f"  {model_path}"
            )

            return model_path

    except Exception as e:

        print(
            f"⚠️ Automatic checkpoint discovery failed:\n"
            f"  {e}"
        )

    # --------------------------------------------------
    # 2. Explicit absolute checkpoint path
    # --------------------------------------------------

    if (
        getattr(cfg, "pretrained", None)
        and getattr(cfg.pretrained, "ae_path", "")
    ):

        p = Path(cfg.pretrained.ae_path)

        if p.exists():

            print(
                f"✓ Using explicit checkpoint:\n"
                f"  {p}"
            )

            return p

    # --------------------------------------------------
    # 3. Explicit relative checkpoint path
    # --------------------------------------------------

    pretrained_cfg = getattr(cfg, "pretrained", None)

    rel = (
        getattr(pretrained_cfg, "ae_relpath", None)
        if pretrained_cfg is not None
        else None
    )

    if rel is None:
        rel = "checkpoint_dir_ae/spatialfusion-multimodal-ae.pt"

    chk_root = getattr(cfg.paths, "checkpoints", "")

    if chk_root:

        cand = Path(chk_root) / rel

        if cand.exists():

            print(
                f"✓ Using relative checkpoint:\n"
                f"  {cand}"
            )

            return cand

    # --------------------------------------------------
    # 4. Packaged fallback checkpoint
    # --------------------------------------------------

    packaged = resolve_pkg_ckpt(rel)

    print(
        f"✓ Using packaged fallback checkpoint:\n"
        f"  {packaged}"
    )

    return packaged


def infer_input_dims(
    sample_list,
    base_path: pl.Path,
    he_encoder: str,
    rna_encoder: str,
):
    """
    Infer HE and RNA input dimensions from the first sample
    containing the requested embedding pair.

    Parameters
    ----------
    sample_list
        List of samples from the Hydra config.
    base_path
        Root dataset directory.
    he_encoder
        One of: "uni", "virchow".
    rna_encoder
        One of: "scgpt", "nicheformer".

    Returns
    -------
    tuple[int, int]
        (he_dim, rna_dim)
    """

    he_file = HE_EMBEDDINGS[he_encoder.lower()]
    rna_file = RNA_EMBEDDINGS[rna_encoder.lower()]

    print(
        f"Using HE encoder: {he_encoder} ({he_file}) | "
        f"RNA encoder: {rna_encoder} ({rna_file})"
    )

    for sample_info in sample_list:

        if isinstance(sample_info, dict):
            sample = str(sample_info["name"])
            datapath = pl.Path(
                sample_info.get("path", base_path)
            ) / sample
        else:
            sample = str(sample_info)
            datapath = pl.Path(base_path) / sample

        embeddings_path = datapath / "embeddings"

        print(
            f"🔍 [{sample}] Searching for "
            f"{he_file} + {rna_file}"
        )

        he_path = None
        rna_path = None

        for ext in (".csv", ".parquet"):
            candidate = embeddings_path / f"{he_file}{ext}"
            if candidate.exists():
                he_path = candidate
                break

        for ext in (".csv", ".parquet"):
            candidate = embeddings_path / f"{rna_file}{ext}"
            if candidate.exists():
                rna_path = candidate
                break

        print(
            f"HE:  {he_path}"
        )
        print(
            f"RNA: {rna_path}"
        )

        if he_path is None or rna_path is None:
            print(
                f"✗ Missing embedding file(s) for {sample}"
            )
            continue

        print(
            f"✓ Found pair: "
            f"{he_path.name} + {rna_path.name}"
        )

        try:

            if he_path.suffix == ".csv":
                he_df = pd.read_csv(
                    he_path,
                    index_col=0,
                    nrows=1,
                )
            else:
                he_df = pd.read_parquet(
                    he_path
                ).iloc[:1]

            if rna_path.suffix == ".csv":
                rna_df = pd.read_csv(
                    rna_path,
                    index_col=0,
                    nrows=1,
                )
            else:
                rna_df = pd.read_parquet(
                    rna_path
                ).iloc[:1]

            print(
                f"✓ Inferred dimensions from {sample}: "
                f"HE={he_df.shape[1]}, RNA={rna_df.shape[1]}"
            )

            return (
                he_df.shape[1],
                rna_df.shape[1],
            )

        except Exception as e:

            print(
                f"⚠️ Failed reading {sample}: {e}"
            )

    raise RuntimeError(
        f"Could not infer dimensions for "
        f"{he_encoder} + {rna_encoder}."
    )


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

    ckpt_dir = find_matching_checkpoint_dir(cfg)

    model_path = ckpt_dir / "model.pt"

    out_dir = (
        Path(cfg.eval.embedding_root)
        / ckpt_dir.name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    latent_dim = cfg.training.latent_dim

    # Infer input dims (HE first, RNA second, per your loader)
    d1_dim, d2_dim = infer_input_dims(
        sample_list,
        base_path,
        cfg.training.he_encoder,
        cfg.training.rna_encoder,
    )

    model = PairedAE(
        d1_dim=d1_dim,                         # HE dim
        d2_dim=d2_dim,                         # RNA dim
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
            model,
            sample_list,
            base_path,
            device,
            he_encoder=cfg.training.he_encoder,
            rna_encoder=cfg.training.rna_encoder,
        )

    # Persist per-sample outputs
    save_embeddings_separately(
        z1, z2, z_joint, celltypes, samples, out_dir, cfg.eval.sample_mode
    )
    print(f"✓ Saved embeddings to {out_dir}")


if __name__ == "__main__":
    main()
