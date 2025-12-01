#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python 3.9+
from __future__ import annotations

import os
import json
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm

# plotting imports (used only if you later add plotting flags)
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401

# ML / vision stack for embeddings
import torch
from torchvision import transforms
import timm
import tifffile
from PIL import Image

# --- quiet some warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
sc.set_figure_params(figsize=(3, 3), frameon=False)

logger = logging.getLogger("visiumhd")


# ---------------------------------------------------------------------
# Helpers for SpaceRanger -> AnnData
# ---------------------------------------------------------------------

def _choose_xy_columns(tp_df: pd.DataFrame) -> tuple[str, str] | None:
    """Pick reasonable x/y column names from a 2µm tissue_positions parquet."""
    candidates = [
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),  # common 10x naming
        ("pxl_col", "pxl_row"),
        ("x", "y"),
        ("x_fullres", "y_fullres"),
    ]
    for xcol, ycol in candidates:
        if xcol in tp_df.columns and ycol in tp_df.columns:
            return xcol, ycol
    return None


def _centroids_from_2um(outs: Path, obs_index: pd.Index) -> np.ndarray:
    """
    Compute cell centroids as the mean of member **2 µm** barcode coordinates.

    Requires:
      - outs/barcode_mappings.parquet
      - outs/binned_outputs/square_002um/spatial/tissue_positions.parquet
    """
    logger.info("Computing centroids from 2µm bins under %s", outs)

    # Read mapping
    bm = outs / "barcode_mappings.parquet"
    if not bm.exists():
        raise FileNotFoundError(f"Required mapping not found: {bm}")
    map_df = pd.read_parquet(bm)
    bcol = "barcode" if "barcode" in map_df.columns else map_df.columns[0]
    ccol = "cell_id" if "cell_id" in map_df.columns else next(
        (c for c in map_df.columns if "cell" in c), None)
    if ccol is None:
        raise KeyError(
            f"Could not identify cell id column in {bm} (columns: {list(map_df.columns)})")

    map_df = map_df[[bcol, ccol]].dropna()
    map_df[bcol] = map_df[bcol].astype(str)
    map_df[ccol] = map_df[ccol].astype(str)
    logger.info("Mapping rows: %d | unique cells: %d | unique barcodes: %d",
                len(map_df), map_df[ccol].nunique(), map_df[bcol].nunique())

    # Read 2µm tissue positions
    tp = outs / "binned_outputs" / "square_002um" / \
        "spatial" / "tissue_positions.parquet"
    if not tp.exists():
        raise FileNotFoundError(
            f"Required 2µm tissue positions not found: {tp}")
    tpdf = pd.read_parquet(tp)
    idx_name = "barcode" if "barcode" in tpdf.columns else tpdf.columns[0]
    tpdf = tpdf.set_index(idx_name)

    xy = _choose_xy_columns(tpdf)
    if xy is None:
        raise KeyError(
            f"Could not find x/y columns in {tp}. Available columns: {list(tpdf.columns)}")
    xcol, ycol = xy
    logger.info("Using 2µm coordinate columns: %s, %s", xcol, ycol)

    pos = (map_df.merge(tpdf[[xcol, ycol]], left_on=bcol, right_index=True, how="left")
                 .dropna(subset=[xcol, ycol]))

    if pos.empty:
        logger.warning(
            "No overlapping barcodes between mapping and tissue_positions; centroids will be NaN")
        centroids = pd.DataFrame(index=pd.Index(
            obs_index, name="cell_id"), columns=[xcol, ycol], dtype=float)
    else:
        centroids = pos.groupby(ccol)[[xcol, ycol]].mean().reindex(obs_index)

    n_missing = int(centroids.isna().any(axis=1).sum())
    if n_missing:
        logger.warning("Centroids missing for %d/%d cells.",
                       n_missing, len(obs_index))

    return centroids.values  # (n_cells, 2)


def _find_spatial_dir(outs: Path) -> Optional[Path]:
    cand1, cand2 = outs / "segmented_outputs" / "spatial", outs / "spatial"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def _read_visium_image_and_scales(spatial_dir: Optional[Path]) -> Optional[dict]:
    """Optional: load images & scalefactors for visualization."""
    if spatial_dir is None or not spatial_dir.exists():
        return None
    scales_fp = spatial_dir / "scalefactors_json.json"
    lowres = next((str(p) for p in [spatial_dir / "tissue_lowres_image.png",
                                    spatial_dir / "tissue_lowres_image.jpg",
                                    spatial_dir / "detected_tissue_image.jpg"] if p.exists()), None)
    hires = next((str(p) for p in [spatial_dir / "tissue_hires_image.png",
                                   spatial_dir / "tissue_hires_image.jpg"] if p.exists()), None)
    scales = None
    if scales_fp.exists():
        try:
            scales = json.loads(scales_fp.read_text())
            logger.info("Loaded scalefactors from %s", scales_fp)
        except Exception as e:
            logger.warning("Failed to parse scalefactors_json.json: %s", e)
    return {"library_id": "VisiumHD", "images": {"lowres": lowres, "hires": hires}, "scalefactors": scales, "metadata": {}}


def _make_from_segmented_outputs(outs: Path) -> Optional[ad.AnnData]:
    """
    Build AnnData using counts from segmented_outputs/filtered_feature_cell_matrix.h5,
    and ALWAYS compute adata.obsm['spatial'] from the 2µm bins.
    """
    seg = outs / "segmented_outputs"
    if not seg.exists():
        logger.info("No segmented_outputs directory found in %s", outs)
        return None

    h5 = seg / "filtered_feature_cell_matrix.h5"
    if not h5.exists():
        logger.error("Expected segmented cell matrix not found: %s", h5)
        return None

    logger.info("Reading segmented cell-by-gene matrix: %s", h5)
    adata = sc.read_10x_h5(str(h5))
    logger.info("Matrix loaded: n_obs=%d, n_vars=%d",
                adata.n_obs, adata.n_vars)

    cells_parq = seg / "cells.parquet"
    if cells_parq.exists():
        logger.info("Joining per-cell metadata from %s", cells_parq)
        cells_df = pd.read_parquet(cells_parq)
        if "cell_id" in cells_df.columns:
            cells_df = cells_df.set_index("cell_id")
        adata.obs = adata.obs.join(cells_df, how="left")
        logger.info("obs columns now: %d", adata.obs.shape[1])

    logger.info("Computing spatial centroids strictly from 2µm bin positions…")
    adata.obsm["spatial"] = _centroids_from_2um(outs, adata.obs_names)

    bm = outs / "barcode_mappings.parquet"
    if bm.exists():
        adata.uns["barcode_mappings_parquet"] = str(bm)

    spatial_dir = _find_spatial_dir(outs)
    visium_uns = _read_visium_image_and_scales(spatial_dir)
    if visium_uns:
        adata.uns.setdefault("spatial", {})[
            visium_uns["library_id"]] = visium_uns

    adata.var_names_make_unique()
    logger.info("Finished building AnnData.")
    return adata


def build_anndata_from_spaceranger(outs_dir: Path) -> ad.AnnData:
    outs = Path(outs_dir)
    if not outs.exists():
        raise FileNotFoundError(f"{outs} does not exist.")
    adata = _make_from_segmented_outputs(outs)
    if adata is None:
        raise RuntimeError(
            "Could not build AnnData: segmented matrix missing.")
    return adata


# ---------------------------------------------------------------------
# scGPT embedding
# ---------------------------------------------------------------------

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
) -> None:
    """
    Run scGPT embedding using your local sc_foundation_evals package & model dir.
    Writes <output_dir>/scGPT.parquet
    """
    import sys
    sys.path.append(str(model_dir.parent))
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

    scgpt_model.create_configs(
        seed=seed, max_seq_len=max_seq_len, n_bins=input_bins)
    scgpt_model.load_pretrained_model()

    input_data = data.InputData(adata_dataset_path=str(dataset_path))
    vocab_list = scgpt_model.vocab.get_stoi().keys()

    # keep model-supported genes (your guardrail)
    adata_obj = input_data.adata
    genes_in_vocab = adata_obj.var_names.intersection(vocab_list)
    if len(genes_in_vocab) / max(len(adata_obj.var_names), 1) < 0.5:
        sc_log.warning(
            "Fewer than 50%% of genes are found in the model vocab — continuing anyway.")
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

    scgpt_model.tokenize_data(
        data=input_data, input_layer_key="X_binned", include_zero_genes=False
    )
    scgpt_model.extract_embeddings(data=input_data)

    out_path = Path(output_dir) / "scGPT.parquet"
    pd.DataFrame(
        input_data.adata.obsm["X_scGPT"],
        index=(
            input_data.adata.obs["cell_id"]
            if "cell_id" in input_data.adata.obs.columns
            else input_data.adata.obs.index
        ),
    ).to_parquet(out_path)
    logger.info("Wrote scGPT embeddings: %s", out_path)


# ---------------------------------------------------------------------
# UNI embedding (WSI patches)
# ---------------------------------------------------------------------

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
        'dynamic_img_size': True
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    sd = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Lambda(lambda im: im.convert('RGB')),  # ensure 3 channels
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, transform


def process_HDsample_and_embed_UNI(
    adata: ad.AnnData,
    wsi_path: Path,
    sample_id: str,
    model_path: Path,
    out_root: Path,
    device: str = "cuda",
    patch_radius: int = 128,
    batch_size: int = 128,
) -> Path:
    logger.info("Loading WSI from %s", wsi_path)
    with tifffile.TiffFile(str(wsi_path)) as tif:
        wsi = tif.pages[0].asarray()

    logger.info("Loading UNI model from %s on %s", model_path, device)
    model, transform = load_UNI_model(model_path, device)

    he_coords = np.asarray(adata.obsm["spatial"])
    out_dir = Path(out_root) / sample_id / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings: List[np.ndarray] = []
    cell_ids: List[str] = []
    batch_imgs: List[torch.Tensor] = []
    batch_ids: List[str] = []

    H, W = wsi.shape[:2]
    r, side = patch_radius, patch_radius * 2

    logger.info("Embedding %d patches (batch_size=%d, side=%d)",
                len(he_coords), batch_size, side)
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

        # slice & pad
        core = wsi[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        if core.ndim == 2:
            core = np.stack([core] * 3, axis=-1)
        patch = np.pad(
            core,
            ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
            mode="constant"
        )

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
                if device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        batch_embs = model(img_tensor).to(
                            torch.float16).cpu().numpy()
                else:
                    batch_embs = model(img_tensor).cpu().numpy()
            embeddings.extend(batch_embs)
            cell_ids.extend(batch_ids)
            batch_imgs.clear()
            batch_ids.clear()

    if batch_imgs:
        img_tensor = torch.stack(batch_imgs).to(device)
        with torch.inference_mode():
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch_embs = model(img_tensor).to(
                        torch.float16).cpu().numpy()
            else:
                batch_embs = model(img_tensor).cpu().numpy()
        embeddings.extend(batch_embs)
        cell_ids.extend(batch_ids)

    df = pd.DataFrame(embeddings, index=cell_ids)
    out_path = out_dir / "UNI.parquet"
    df.to_parquet(out_path, index=True)
    logger.info("Saved %d embeddings to %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Visium-HD processing: build AnnData from SpaceRanger outs, scGPT embeddings, UNI WSI embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Logging
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # Phase toggles
    p.add_argument("--build-anndata", action="store_true",
                   help="Build AnnData from SpaceRanger outs -> savedir/<sample>/adata.h5ad")
    p.add_argument("--run-scgpt", action="store_true",
                   help="Run scGPT embeddings for each sample with adata.h5ad")
    p.add_argument("--run-uni", action="store_true",
                   help="Run UNI WSI embeddings for selected samples")

    # Build AnnData
    p.add_argument("--basedir", type=Path, default=Path("../../../Broad_SpatialFoundation/VisiumHD-LUAD/"),
                   help="Directory containing per-sample folders with SpaceRanger outputs under <sample>/outs")
    p.add_argument("--savedir", type=Path, default=Path("../../../Broad_SpatialFoundation/VisiumHD-LUAD-processed/"),
                   help="Where to write <sample>/adata.h5ad")

    # Sample selection
    p.add_argument("--include", type=str, default="",
                   help="Comma-separated sample names to include (defaults to all in basedir).")
    p.add_argument("--exclude", type=str, default="LIB-065293st1",
                   help="Comma-separated sample names to exclude.")
    p.add_argument("--exists-skip", action="store_true",
                   help="Skip writing adata if <savedir>/<sample>/adata.h5ad already exists.")

    # scGPT
    p.add_argument("--scgpt-model-dir", type=Path,
                   default=Path("../../../Broad_SpatialFoundation/scGPT_model/"))
    p.add_argument("--scgpt-n-hvg", type=int, default=1200)
    p.add_argument("--scgpt-batch-size", type=int, default=16)
    p.add_argument("--scgpt-seed", type=int, default=42)

    # UNI
    p.add_argument("--uni-model", type=Path,
                   default=Path("../../../Broad_SpatialFoundation/UNI/pytorch_model.bin"))
    p.add_argument("--wsi-root", type=Path, default=Path("../../../Broad_SpatialFoundation/VisiumHD-LUAD/"),
                   help="Root where raw WSIs live (sample subdirs).")
    p.add_argument("--wsi-mapping-json", type=Path, default=None,
                   help="JSON mapping of sample -> WSI filename. If omitted, uses built-in mapping from your snippet.")
    p.add_argument("--uni-device", type=str, default="cuda:5",
                   help="Torch device for UNI (e.g., cuda:0, cuda:5, or cpu).")
    p.add_argument("--uni-batch-size", type=int, default=128)
    p.add_argument("--uni-patch-radius", type=int, default=128)
    p.add_argument("--uni-only", type=str, default="",
                   help="Comma-separated subset of samples to run UNI on.")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    include = [s.strip() for s in args.include.split(",") if s.strip()]
    exclude = set(s.strip() for s in args.exclude.split(",") if s.strip())

    # Discover samples in basedir
    samples = [p.name for p in args.basedir.iterdir() if p.is_dir()]
    if include:
        samples = [s for s in samples if s in include]
    if exclude:
        samples = [s for s in samples if s not in exclude]
    samples = sorted(samples)

    logger.info("Samples selected (%d): %s", len(samples),
                ", ".join(samples) if samples else "NONE")

    # ---------------- Build AnnData ----------------
    if args.build_anndata:
        for sample in tqdm(samples, desc="Building AnnData"):
            try:
                outs_dir = args.basedir / sample / "outs"
                out_dir = args.savedir / sample
                out_dir.mkdir(parents=True, exist_ok=True)
                adata_path = out_dir / "adata.h5ad"

                if args.exists-skip and adata_path.exists():  # noqa
                    tqdm.write(f"Skip existing {sample}")
                    continue

                adata = build_anndata_from_spaceranger(outs_dir)
                adata.write_h5ad(adata_path, compression="lzf")
                tqdm.write(f"Wrote {adata_path}")
            except Exception as e:
                tqdm.write(f"❌ {sample}: {e}")

    # ---------------- scGPT embeddings ----------------
    if args.run_scgpt:
        for sample in tqdm(samples, desc="scGPT"):
            try:
                sample_dir = args.savedir / sample
                adata_path = sample_dir / "adata.h5ad"
                emb_dir = sample_dir / "embeddings"
                emb_dir.mkdir(parents=True, exist_ok=True)

                if not adata_path.exists():
                    tqdm.write(f"❌ {sample}: missing {adata_path}")
                    continue

                run_embed_scGPT(
                    dataset_path=adata_path,
                    model_dir=args.scgpt_model_dir,
                    output_dir=emb_dir,
                    n_hvg=args.scgpt_n_hvg,
                    batch_size=args.scgpt_batch_size,
                    seed=args.scgpt_seed,
                )
            except Exception as e:
                tqdm.write(f"❌ scGPT {sample}: {e}")

    # ---------------- UNI WSI embeddings ----------------
    if args.run_uni:
        # mapping
        if args.wsi_mapping_json and Path(args.wsi_mapping_json).exists():
            wsi_mapping: Dict[str, str] = json.loads(
                Path(args.wsi_mapping_json).read_text())
        else:
            # fallback to hard-coded mapping
            wsi_mapping = {
                'LIB-064885st1': '8554_A1.tiff',
                'LIB-064886st1': '8552_A1.tiff',
                'LIB-064887st1': '8551_A1.tiff',
                'LIB-064888st1': '8555_A1.tiff',
                'LIB-064889st1': '8556_A1.tiff',
                'LIB-064890st1': '8553_A1.tiff',
                'LIB-065290st1': '8558_A1.tiff',
                'LIB-065291st1': '8563_A1.tiff',
                'LIB-065292st1': '8562_A1.tiff',
                'LIB-065294st1': '8559_A1.tiff',
                'LIB-065295st1': '8560_A1.tiff',
            }

        uni_subset = [s.strip() for s in args.uni_only.split(",") if s.strip()]
        run_samples = [s for s in samples if (
            not uni_subset or s in uni_subset)]

        for sample in tqdm(run_samples, desc="UNI"):
            try:
                adata_path = args.savedir / sample / "adata.h5ad"
                if not adata_path.exists():
                    tqdm.write(f"❌ {sample}: missing {adata_path}")
                    continue

                wsi_name = wsi_mapping.get(sample)
                if not wsi_name:
                    tqdm.write(f"❌ {sample}: no WSI mapping available")
                    continue

                wsi_path = args.wsi_root / sample / wsi_name
                if not wsi_path.exists():
                    tqdm.write(f"❌ {sample}: WSI not found at {wsi_path}")
                    continue

                adata = sc.read_h5ad(adata_path)
                process_HDsample_and_embed_UNI(
                    adata=adata,
                    wsi_path=wsi_path,
                    sample_id=sample,
                    model_path=args.uni_model,
                    out_root=args.savedir,
                    device=args.uni_device,
                    patch_radius=args.uni_patch_radius,
                    batch_size=args.uni_batch_size,
                )
            except Exception as e:
                tqdm.write(f"❌ UNI {sample}: {e}")

    logger.info("✅ Done.")


if __name__ == "__main__":
    main()
