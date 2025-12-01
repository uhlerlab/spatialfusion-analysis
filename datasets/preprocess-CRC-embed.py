#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python 3.9+
from __future__ import annotations

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

# vision / ML
import torch
from torchvision import transforms
import timm
import tifffile
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger("crc_scgpt_uni")


# -----------------------------
# scGPT
# -----------------------------
def run_embed_scGPT(
    dataset_path: Path,
    model_dir: Path,
    output_dir: Path,
    n_hvg: int = 1200,
    gene_col: str = "index",
    layer_key: str = "X",
    log_norm: bool = False,
    seed: int = 42,
    max_seq_len: int = 1200,
    batch_size: int = 16,
    input_bins: int = 51,
    model_run: str = "pretrained",
    num_workers: int = 0,
    repo_root: Path | None = None,
    out_name: str = "scGPT.parquet",
) -> Path:
    """
    Runs scGPT embedding using your local sc_foundation_evals package.
    Writes <output_dir>/<out_name> (parquet).
    """
    if repo_root:
        sys.path.append(str(repo_root))

    from sc_foundation_evals import scgpt_forward, data  # type: ignore
    from sc_foundation_evals.helpers.custom_logging import log as sc_log  # type: ignore

    sc_log.setLevel(logging.INFO)

    scgpt_model = scgpt_forward.scGPT_instance(
        saved_model_path=str(model_dir),
        model_run=model_run,
        batch_size=batch_size,
        save_dir=str(output_dir),
        num_workers=num_workers,
        explicit_save_dir=True,
    )

    scgpt_model.create_configs(seed=seed, max_seq_len=max_seq_len, n_bins=input_bins)
    scgpt_model.load_pretrained_model()

    input_data = data.InputData(adata_dataset_path=str(dataset_path))
    vocab_list = scgpt_model.vocab.get_stoi().keys()

    adata_obj = input_data.adata
    genes_in_vocab = adata_obj.var_names.intersection(vocab_list)
    if len(genes_in_vocab) / max(len(adata_obj.var_names), 1) < 0.5:
        sc_log.warning("Fewer than 50%% of genes are in vocab — continuing anyway.")
    adata_obj._inplace_subset_var(genes_in_vocab)
    input_data.adata = adata_obj

    input_data.preprocess_data(
        gene_vocab=vocab_list,
        model_type="scGPT",
        gene_col=gene_col,
        data_is_raw=not log_norm,
        counts_layer=layer_key,
        n_bins=input_bins,
        n_hvg=n_hvg,
    )

    scgpt_model.tokenize_data(data=input_data, input_layer_key="X_binned", include_zero_genes=False)
    scgpt_model.extract_embeddings(data=input_data)

    out_path = Path(output_dir) / out_name
    pd.DataFrame(
        input_data.adata.obsm["X_scGPT"],
        index=(input_data.adata.obs["cell_id"] if "cell_id" in input_data.adata.obs.columns else input_data.adata.obs.index),
    ).to_parquet(out_path)
    log.info("[scGPT] Wrote %s", out_path)
    return out_path


# -----------------------------
# UNI (WSI embeddings)
# -----------------------------
def load_UNI_model(model_path: Path, device: str = "cuda"):
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True,
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    sd = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, transform


def run_embed_UNI(
    adata: sc.AnnData,
    wsi_path: Path,
    sample_id: str,
    model_path: Path,
    out_root: Path,
    device: str = "cuda",
    batch_size: int = 128,
    patch_radius: int = 128,
    out_name: str = "UNI.parquet",
) -> Path:
    with tifffile.TiffFile(str(wsi_path)) as tif:
        wsi = tif.pages[0].asarray()

    model, transform = load_UNI_model(model_path, device)

    he_coords = np.asarray(adata.obsm["spatial"])
    out_dir = Path(out_root) / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings: List[np.ndarray] = []
    cell_ids: List[str] = []
    batch_imgs: List[torch.Tensor] = []
    batch_ids: List[str] = []

    H, W = wsi.shape[:2]
    r, side = patch_radius, patch_radius * 2

    log.info("[UNI] %s: %d patches, batch=%d, side=%d", sample_id, len(he_coords), batch_size, side)
    iterator = zip(adata.obs_names, he_coords)
    iterator = tqdm(list(iterator), total=len(adata), desc=f"UNI {sample_id}")

    for cid, (x, y) in iterator:
        x, y = int(x), int(y)
        x0, x1 = x - r, x + r
        y0, y1 = y - r, y + r

        pad_x0 = max(0, -x0)
        pad_x1 = max(0, x1 - W)
        pad_y0 = max(0, -y0)
        pad_y1 = max(0, y1 - H)

        core = wsi[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        if core.ndim == 2:
            core = np.stack([core] * 3, axis=-1)
        patch = np.pad(core, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode="constant")

        if patch.shape[:2] != (side, side):
            continue
        if patch.shape[2] > 3:
            patch = patch[:, :, :3]

        img = Image.fromarray(patch).convert("RGB")
        tensor_img = transform(img)
        batch_imgs.append(tensor_img)
        batch_ids.append(cid)

        if len(batch_imgs) == batch_size:
            img_tensor = torch.stack(batch_imgs).to(device)
            with torch.inference_mode():
                if str(device).startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        batch_embs = model(img_tensor).to(torch.float16).cpu().numpy()
                else:
                    batch_embs = model(img_tensor).cpu().numpy()
            embeddings.extend(batch_embs)
            cell_ids.extend(batch_ids)
            batch_imgs.clear()
            batch_ids.clear()

    if batch_imgs:
        img_tensor = torch.stack(batch_imgs).to(device)
        with torch.inference_mode():
            if str(device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch_embs = model(img_tensor).to(torch.float16).cpu().numpy()
            else:
                batch_embs = model(img_tensor).cpu().numpy()
        embeddings.extend(batch_embs)
        cell_ids.extend(batch_ids)

    df = pd.DataFrame(embeddings, index=cell_ids)
    out_path = out_dir / out_name
    df.to_parquet(out_path, index=True)
    log.info("[UNI] %s: wrote %s", sample_id, out_path)
    return out_path


# -----------------------------
# CLI
# -----------------------------
DEFAULT_WSI_MAP: Dict[str, str] = {
    'P1CRC': 'Visium_HD_Human_Colon_Cancer_P1_tissue_image.btf',
    'P2CRC': 'Visium_HD_Human_Colon_Cancer_P2_tissue_image.btf',
    'P5CRC': 'Visium_HD_Human_Colon_Cancer_P5_tissue_image.btf',
    'P3NAT': 'Visium_HD_Human_Colon_Normal_P3_tissue_image.btf',
    'P5NAT': 'Visium_HD_Human_Colon_Normal_P5_tissue_image.btf',
}


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="CRC pipeline: scGPT and UNI embeddings over VisiumHD-CRC samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # General
    p.add_argument("--base-dir", type=Path, default=Path("../../../Broad_SpatialFoundation/VisiumHD-CRC"))
    p.add_argument("--samples", type=str, default="P1CRC,P2CRC,P5CRC,P3NAT,P5NAT")
    p.add_argument("--include-only", type=str, default="", help="Optional subset, comma-separated.")
    p.add_argument("--exclude", type=str, default="", help="Comma-separated names to skip.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # scGPT toggles/params
    p.add_argument("--run-scgpt", action="store_true")
    p.add_argument("--scgpt-model-dir", type=Path, default=Path("../../../Broad_SpatialFoundation/scGPT_model"))
    p.add_argument("--repo-root", type=Path, default=Path("../../../Broad_SpatialFoundation"),
                   help="Parent dir so sc_foundation_evals is importable.")
    p.add_argument("--scgpt-n-hvg", type=int, default=1200)
    p.add_argument("--scgpt-batch-size", type=int, default=16)
    p.add_argument("--scgpt-seed", type=int, default=42)
    p.add_argument("--scgpt-out", type=str, default="scGPT.parquet")
    p.add_argument("--skip-if-scgpt-exists", action="store_true")

    # UNI toggles/params
    p.add_argument("--run-uni", action="store_true")
    p.add_argument("--uni-model", type=Path, default=Path("../../../Broad_SpatialFoundation/UNI/pytorch_model.bin"))
    p.add_argument("--wsi-mapping-json", type=Path, default=None, help="JSON of {sample: filename}")
    p.add_argument("--uni-device", type=str, default="cuda")
    p.add_argument("--uni-batch-size", type=int, default=128)
    p.add_argument("--uni-patch-radius", type=int, default=128)
    p.add_argument("--uni-out", type=str, default="UNI.parquet")
    p.add_argument("--skip-if-uni-exists", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    samples = [s.strip() for s in (args.include_only or args.samples).split(",") if s.strip()]
    if args.exclude:
        ex = {s.strip() for s in args.exclude.split(",") if s.strip()}
        samples = [s for s in samples if s not in ex]

    log.info("Samples: %s", ", ".join(samples))

    # load mapping
    wsi_map = dict(DEFAULT_WSI_MAP)
    if args.wsi_mapping_json and args.wsi_mapping_json.exists():
        wsi_map.update(json.loads(args.wsi_mapping_json.read_text()))

    # iterate samples
    for sample in tqdm(samples, desc="Samples"):
        sample_dir = args.base_dir / sample
        adata_path = sample_dir / "adata.h5ad"
        if not adata_path.exists():
            tqdm.write(f"❌ {sample}: missing {adata_path}")
            continue

        # ---------- scGPT ----------
        if args.run_scgpt:
            scgpt_path = sample_dir / args.scgpt_out
            if args.skip_if_scgpt_exists and scgpt_path.exists():
                tqdm.write(f"[scGPT] Skip existing for {sample}")
            else:
                try:
                    run_embed_scGPT(
                        dataset_path=adata_path,
                        model_dir=args.scgpt_model_dir,
                        output_dir=sample_dir,
                        n_hvg=args.scgpt_n_hvg,
                        batch_size=args.scgpt_batch_size,
                        seed=args.scgpt_seed,
                        repo_root=args.repo_root,
                        out_name=args.scgpt_out,
                    )
                except Exception as e:
                    tqdm.write(f"❌ scGPT {sample}: {e}")

        # ---------- UNI ----------
        if args.run_uni:
            uni_path = sample_dir / args.uni_out
            if args.skip_if_uni_exists and uni_path.exists():
                tqdm.write(f"[UNI] Skip existing for {sample}")
            else:
                wsi_name = wsi_map.get(sample)
                if not wsi_name:
                    tqdm.write(f"❌ {sample}: no WSI mapping defined")
                    continue
                wsi_path = sample_dir / wsi_name  # your notebook kept WSIs inside the sample folder
                if not wsi_path.exists():
                    tqdm.write(f"❌ {sample}: WSI not found at {wsi_path}")
                    continue

                try:
                    adata = sc.read_h5ad(adata_path)
                    run_embed_UNI(
                        adata=adata,
                        wsi_path=wsi_path,
                        sample_id=sample,
                        model_path=args.uni_model,
                        out_root=args.base_dir,
                        device=args.uni_device,
                        batch_size=args.uni_batch_size,
                        patch_radius=args.uni_patch_radius,
                        out_name=args.uni_out,
                    )
                except Exception as e:
                    tqdm.write(f"❌ UNI {sample}: {e}")

    log.info("✅ Done.")


if __name__ == "__main__":
    main()
