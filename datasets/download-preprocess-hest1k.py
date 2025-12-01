#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HEST pipeline script:
- (Optional) Login to Hugging Face
- (Optional) Filter HEST metadata, export a CSV, and (optionally) make plots
- (Optional) Download HEST subsets from HF using patterns derived from filtered IDs
- (Optional) Process samples to create AnnData and UNI image embeddings
- (Optional) Run scGPT embeddings on created AnnData

Author: Josephine Yates
"""

import os
import sys
import json
import logging
import warnings
import argparse
from typing import List, Optional
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hest")

import os
import sys
import json
import logging
import warnings
import argparse
from typing import List, Optional
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from huggingface_hub import login as hf_login
import datasets

import numpy as np
import scanpy as sc
import tifffile
import shapely.wkb  # access as shapely.wkb.loads(...)
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import timm

# ---------------------------
# Utility / I/O
# ---------------------------

def ensure_imports_for(step: str):
    return None


def read_metadata(meta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    if "id" not in df.columns:
        raise ValueError("Metadata CSV must contain an 'id' column.")
    return df


def filter_metadata(
    df: pd.DataFrame,
    st_technology: Optional[str],
    species: Optional[str],
    extra_filters: List[str],
) -> pd.DataFrame:
    """Apply simple equality filters: 'col==value'. Extra filters are 'col=value' strings."""
    if st_technology:
        if "st_technology" not in df.columns:
            raise ValueError("Metadata missing 'st_technology' column.")
        df = df[df["st_technology"] == st_technology]
    if species:
        if "species" not in df.columns:
            raise ValueError("Metadata missing 'species' column.")
        df = df[df["species"] == species]
    # Extra filters like organ=Breast or disease_state=Tumor
    for f in extra_filters:
        if "=" not in f:
            log.warning(f"Skipping malformed filter '{f}'. Expected 'col=value'.")
            continue
        col, val = f.split("=", 1)
        col = col.strip()
        val = val.strip()
        if col not in df.columns:
            log.warning(f"Filter column '{col}' not in metadata. Skipping.")
            continue
        df = df[df[col].astype(str) == val]
    return df


def export_meta_subset(df: pd.DataFrame, out_csv: Path, cols: Optional[List[str]] = None):
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            log.warning(f"Some requested columns not in metadata: {missing}. Exporting existing columns only.")
        ecols = [c for c in cols if c in df.columns]
        df[ecols].to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False)
    log.info(f"Wrote filtered metadata to {out_csv}")


def maybe_make_plots(
    df: pd.DataFrame,
    category_columns: List[str],
    plot_dir: Path,
    show: bool,
):
    plot_dir.mkdir(parents=True, exist_ok=True)
    for col in category_columns:
        if col not in df.columns:
            log.warning(f"Plot column '{col}' not in metadata. Skipping plot.")
            continue
        counts = df[col].value_counts()
        if counts.empty:
            log.warning(f"No values to plot for '{col}'. Skipping.")
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
            startangle=90,
            wedgeprops={"width": 0.4},
        )
        ax.legend(wedges, counts.index.astype(str), title=col, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"{col.replace('_',' ').capitalize()} Distribution")
        plt.tight_layout()
        out_path = plot_dir / f"{col}_distribution.png"
        plt.savefig(out_path, dpi=150)
        log.info(f"Saved plot: {out_path}")
        if show:
            plt.show()
        plt.close(fig)


# ---------------------------
# Hugging Face download
# ---------------------------

def hf_auth(token: Optional[str]):
    token = token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        log.warning("No Hugging Face token provided (env HUGGINGFACE_TOKEN/HF_TOKEN or --hf-token). "
                    "Proceeding unauthenticated (may fail for private repos).")
        return
    if hf_login is None:
        raise ImportError("huggingface_hub not installed.")
    hf_login(token=token)
    log.info("Authenticated to Hugging Face.")


def download_hest_subset(
    dataset_repo: str,
    cache_dir: Path,
    ids: List[str],
    pattern_template: str,
):
    ensure_imports_for("download")
    cache_dir.mkdir(parents=True, exist_ok=True)
    list_patterns = [pattern_template.format(id=i) for i in ids]
    log.info(f"Requesting dataset '{dataset_repo}' with {len(list_patterns)} pattern(s)...")
    ds = datasets.load_dataset(
        dataset_repo,
        cache_dir=str(cache_dir),
        patterns=list_patterns,
    )
    # This call materializes data in cache_dir. You can further inspect or copy as needed.
    log.info(f"Loaded dataset splits: {list(ds.keys())} into cache {cache_dir}")


# ---------------------------
# UNI embedding
# ---------------------------

def load_UNI_model(model_path: Path, device: str = "cuda"):
    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
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


def clean_transcript_df(df_tx: pd.DataFrame) -> pd.DataFrame:
    if "feature_name" not in df_tx.columns or "cell_id" not in df_tx.columns:
        raise ValueError("Transcript parquet must contain 'feature_name' and 'cell_id' columns.")

    if isinstance(df_tx["feature_name"].iloc[0], (bytes, bytearray)):
        df_tx["feature_name"] = df_tx["feature_name"].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
    if isinstance(df_tx["cell_id"].iloc[0], (bytes, bytearray)):
        df_tx["cell_id"] = df_tx["cell_id"].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)

    df_tx = df_tx[df_tx["cell_id"] != -1].copy()
    df_tx = df_tx[df_tx["cell_id"] != "UNASSIGNED"].copy()
    df_tx["cell_id"] = df_tx["cell_id"].astype(str)
    return df_tx


def process_sample_and_embed_UNI(
    sample_id: str,
    base_dir: Path,
    model_path: Path,
    out_root: Path,
    device: str = "cuda",
    patch_radius: int = 128,
    batch_size: int = 128,
):
    ensure_imports_for("uni")
    log.info(f"[UNI] Loading model from {model_path} on {device}")
    model, transform = load_UNI_model(model_path, device)

    out_dir = out_root / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    transcript_path = base_dir / "transcripts" / f"{sample_id}_transcripts.parquet"
    seg_path = base_dir / "xenium_seg" / f"{sample_id}_xenium_cell_seg.parquet"
    wsi_path = base_dir / "wsis" / f"{sample_id}.tif"

    for p in [transcript_path, seg_path, wsi_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    log.info(f"[UNI] Loading data: {sample_id}")
    df_tx = pd.read_parquet(transcript_path)
    df_seg = pd.read_parquet(seg_path)
    with tifffile.TiffFile(str(wsi_path)) as tif:
        wsi = tif.pages[0].asarray()

    df_tx = clean_transcript_df(df_tx)

    log.info("[UNI] Parsing geometry and centroids")
    if "geometry" not in df_seg.columns:
        raise ValueError("Segmentation parquet must contain a 'geometry' column with polygons.")
    df_seg["geometry"] = df_seg["geometry"].apply(lambda g: g if hasattr(g, "centroid") else shapely.wkb.loads(g))
    df_seg["he_x"] = df_seg["geometry"].apply(lambda g: g.centroid.x)
    df_seg["he_y"] = df_seg["geometry"].apply(lambda g: g.centroid.y)
    df_seg.index = df_seg.index.astype(str)

    # Count matrix
    counts = pd.crosstab(df_tx["cell_id"], df_tx["feature_name"])
    df_seg = df_seg.loc[counts.index]
    he_coords = df_seg[["he_x", "he_y"]].to_numpy()
    morph_coords = df_seg[["x", "y"]].to_numpy() if all(c in df_seg.columns for c in ["x", "y"]) else np.zeros_like(he_coords)

    # AnnData
    log.info("[UNI] Creating AnnData")
    adata = sc.AnnData(X=counts.values)
    adata.obs_names = [f"{sample_id}_{cid}" for cid in counts.index]
    adata.var_names = counts.columns.astype(str)
    adata.obsm["spatial_he"] = he_coords
    adata.obsm["spatial"] = morph_coords
    # If X is a numpy array, sum(axis=1) returns 1D np.ndarray
    adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata_path = out_dir / "adata.h5ad"
    adata.write_h5ad(str(adata_path))
    log.info(f"[UNI] Wrote AnnData: {adata_path}")

    # Embeddings
    embeddings: List[np.ndarray] = []
    cell_ids: List[str] = []
    batch_imgs: List[torch.Tensor] = []
    batch_ids: List[str] = []

    H, W = wsi.shape[:2]
    r = patch_radius
    side = 2 * r

    total = len(he_coords)
    it = zip(adata.obs_names, he_coords)
    if tqdm:
        it = tqdm(it, total=total, desc=f"[UNI] {sample_id} patches")

    log.info(f"[UNI] Embedding {total} patches with batch_size={batch_size}, side={side}")
    for cid, (x, y) in it:
        x, y = int(x), int(y)
        x0, x1 = x - r, x + r
        y0, y1 = y - r, y + r

        pad_x0 = max(0, -x0)
        pad_x1 = max(0, x1 - W)
        pad_y0 = max(0, -y0)
        pad_y1 = max(0, y1 - H)

        patch = np.pad(
            wsi[max(0, y0):min(H, y1), max(0, x0):min(W, x1)],
            ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
            mode="constant"
        )

        if patch.shape[:2] != (side, side):
            # unlikely with our padding, but guard anyway
            continue

        tensor_img = transform(Image.fromarray(patch))
        batch_imgs.append(tensor_img)
        batch_ids.append(cid)

        if len(batch_imgs) == batch_size:
            img_tensor = torch.stack(batch_imgs).to(device)
            if device == "cuda":
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch_embs = model(img_tensor).to(torch.float16).cpu().numpy()
            else:
                with torch.inference_mode():
                    batch_embs = model(img_tensor).cpu().numpy()
            embeddings.extend(batch_embs)
            cell_ids.extend(batch_ids)
            batch_imgs.clear()
            batch_ids.clear()

    if batch_imgs:
        img_tensor = torch.stack(batch_imgs).to(device)
        if device == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                batch_embs = model(img_tensor).to(torch.float16).cpu().numpy()
        else:
            with torch.inference_mode():
                batch_embs = model(img_tensor).cpu().numpy()
        embeddings.extend(batch_embs)
        cell_ids.extend(batch_ids)

    df_emb = pd.DataFrame(embeddings, index=cell_ids)
    emb_path = out_dir / "UNI.csv"
    df_emb.to_csv(emb_path)
    log.info(f"[UNI] Saved {len(df_emb)} embeddings to {emb_path}")


# ---------------------------
# scGPT embedding
# ---------------------------

def run_embed_scGPT(
    dataset_path: Path,
    model_dir: Path,  # dir containing best_model.pt, args.json, vocab.json
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
):
    # Lazy imports from user repo
    sys.path.append(str(model_dir.parent))  # allow e.g., /.../Broad_SpatialFoundation/
    try:
        from sc_foundation_evals import cell_embeddings, scgpt_forward, data, model_output  # noqa: F401
        from sc_foundation_evals.helpers.custom_logging import log as sc_log  # noqa: F401
    except Exception as e:
        raise ImportError(
            "Could not import sc_foundation_evals from your codebase. "
            "Ensure the repository is available and model_dir parent is in sys.path."
        ) from e

    # Create model
    scgpt_model = scgpt_forward.scGPT_instance(
        saved_model_path=str(model_dir),
        model_run=model_run,
        batch_size=batch_size,
        save_dir=str(output_dir),
        num_workers=num_workers,
        explicit_save_dir=True,
    )

    # Configs and weights
    scgpt_model.create_configs(seed=seed, max_seq_len=max_seq_len, n_bins=input_bins)
    scgpt_model.load_pretrained_model()

    # Data prep
    input_data = data.InputData(adata_dataset_path=str(dataset_path))
    vocab_list = scgpt_model.vocab.get_stoi().keys()

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

    out_csv = output_dir / "scGPT.csv"
    pd.DataFrame(
        input_data.adata.obsm["X_scGPT"],
        index=input_data.adata.obs["cell_id"] if "cell_id" in input_data.adata.obs.columns else input_data.adata.obs.index,
    ).to_csv(out_csv)
    log.info(f"[scGPT] Wrote embeddings: {out_csv}")


# ---------------------------
# Orchestrator
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="HEST processing pipeline (download → UNI embeddings → scGPT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General
    p.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (or set HUGGINGFACE_TOKEN env).")
    p.add_argument("--dataset-repo", type=str, default="MahmoodLab/hest", help="HF dataset repo for HEST.")
    p.add_argument("--local-dir", type=Path, default=Path("./hest_data"), help="Local cache/working dir for HEST dataset.")
    p.add_argument("--meta-csv", type=Path, required=True, help="Path to HEST metadata CSV.")
    p.add_argument("--meta-out-csv", type=Path, default=Path("./Xenium_dataset_meta.csv"), help="Path to write filtered metadata CSV.")
    p.add_argument("--category-columns", type=str, default="organ,disease_state", help="Comma-separated columns to plot.")
    p.add_argument("--make-plots", action="store_true", help="Generate distribution plots for category columns.")
    p.add_argument("--plot-dir", type=Path, default=Path("./plots"), help="Directory to save plots.")
    p.add_argument("--show-plots", action="store_true", help="Show plots interactively (if running with GUI).")

    # Filters
    p.add_argument("--filter-st-technology", type=str, default="Xenium", help="Filter st_technology equality.")
    p.add_argument("--filter-species", type=str, default="Homo sapiens", help="Filter species equality.")
    p.add_argument("--filter", action="append", default=[], help="Extra filters like 'organ=Breast'. Can repeat.")

    # Download
    p.add_argument("--run-download", action="store_true", help="Run the dataset download step.")
    p.add_argument("--pattern-template", type=str, default="*{id}[_.]*", help="Pattern template to select sample files.")
    p.add_argument("--ids-csv-col", type=str, default="id", help="Column name that holds sample IDs.")
    p.add_argument("--ids-list", type=str, default=None, help="Optional comma-separated IDs to override metadata IDs.")

    # UNI processing
    p.add_argument("--run-uni", action="store_true", help="Run the UNI processing step.")
    p.add_argument("--base-dir-hest", type=Path, default=Path("./hest_data"), help="Base dir for raw HEST data (transcripts/xenium_seg/wsis).")
    p.add_argument("--uni-model", type=Path, required=False, help="Path to UNI model weights (pytorch_model.bin).")
    p.add_argument("--out-root", type=Path, default=Path("./hest_processed_data"), help="Output root for processed HEST data.")
    p.add_argument("--gpu", type=int, default=None, help="GPU index to select with torch.cuda.set_device().")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Compute device for UNI model.")
    p.add_argument("--uni-batch-size", type=int, default=128, help="Batch size for UNI embeddings.")
    p.add_argument("--patch-radius", type=int, default=128, help="Half patch size (pixels). Full patch side = 2*radius.")
    p.add_argument("--samples-file", type=Path, default=None, help="Optional text file with one sample_id per line to process.")
    p.add_argument("--samples", type=str, default=None, help="Optional comma-separated sample IDs to process.")

    # scGPT
    p.add_argument("--run-scgpt", action="store_true", help="Run the scGPT embedding step.")
    p.add_argument("--scgpt-model-dir", type=Path, default=None, help="Directory with scGPT model files.")
    p.add_argument("--scgpt-n-hvg", type=int, default=1200)
    p.add_argument("--scgpt-gene-col", type=str, default="index")
    p.add_argument("--scgpt-layer-key", type=str, default="X")
    p.add_argument("--scgpt-log-norm", action="store_true", help="If set, data are already log-normalized.")
    p.add_argument("--scgpt-seed", type=int, default=42)
    p.add_argument("--scgpt-max-seq-len", type=int, default=1200)
    p.add_argument("--scgpt-batch-size", type=int, default=16)
    p.add_argument("--scgpt-input-bins", type=int, default=51)
    p.add_argument("--scgpt-model-run", type=str, default="pretrained")
    p.add_argument("--scgpt-num-workers", type=int, default=0)

    return p.parse_args()


def resolve_ids(args, filtered_df: pd.DataFrame) -> List[str]:
    if args.ids_list:
        return [s.strip() for s in args.ids_list.split(",") if s.strip()]
    if args.samples:
        return [s.strip() for s in args.samples.split(",") if s.strip()]
    if args.samples_file and Path(args.samples_file).exists():
        return [line.strip() for line in Path(args.samples_file).read_text().splitlines() if line.strip()]
    if args.ids_csv_col not in filtered_df.columns:
        raise ValueError(f"Column '{args.ids_csv_col}' not found in metadata.")
    return filtered_df[args.ids_csv_col].astype(str).tolist()


def main():
    args = parse_args()

    # Auth
    if args.run_download:
        hf_auth(args.hf_token)

    # Read + filter metadata
    meta_df = read_metadata(args.meta_csv)
    meta_df = filter_metadata(
        meta_df,
        args.filter_st_technology,
        args.filter_species,
        args.filter,
    )

    # Export a clean subset for reference
    export_cols = [
        "id", "organ", "disease_state", "oncotree_code", "patient",
        "pixel_size_um_embedded", "pixel_size_um_estimated", "magnification", "disease_comment",
    ]
    export_meta_subset(meta_df, args.meta_out_csv, export_cols)

    # Optional plots
    if args.make_plots:
        cats = [c.strip() for c in args.category_columns.split(",") if c.strip()]
        maybe_make_plots(meta_df, cats, args.plot_dir, show=args.show_plots)

    # IDs / samples list
    try:
        ids_to_query = resolve_ids(args, meta_df)
    except Exception as e:
        log.error(f"Failed to resolve sample IDs: {e}")
        sys.exit(1)

    # Download (optional)
    if args.run_download:
        try:
            download_hest_subset(
                dataset_repo=args.dataset_repo,
                cache_dir=args.local_dir,
                ids=ids_to_query,
                pattern_template=args.pattern_template,
            )
        except Exception as e:
            log.error(f"Download step failed: {e}")
            sys.exit(1)

    # Device / GPU selection for torch
    if args.run_uni:
        if torch is None:
            raise ImportError("torch is required for UNI step. pip install torch torchvision timm")
        if args.gpu is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(args.gpu)
                log.info(f"Using GPU index {args.gpu}")
            except Exception as e:
                log.warning(f"Could not set GPU device {args.gpu}: {e}")
        if args.device == "auto":
            device = "cuda" if (torch.cuda.is_available()) else "cpu"
        else:
            device = args.device
        if args.uni_model is None:
            log.error("--uni-model is required when --run-uni is set.")
            sys.exit(1)

        # Determine sample list for UNI
        if args.samples or args.samples_file:
            sample_list = resolve_ids(args, meta_df)
        else:
            # By default, try to infer from base_dir_hest/st/* folder names if present
            st_dir = args.base_dir_hest / "st"
            if st_dir.exists():
                sample_list = [p.stem for p in st_dir.iterdir() if p.is_dir() or p.is_file()]
            else:
                # Fall back to IDs from metadata
                sample_list = ids_to_query

        failed_uni_log = Path("failed_samples_uni.log")
        for sample_id in (tqdm(sample_list, desc="[UNI] samples") if tqdm else sample_list):
            try:
                log.info(f"\n🔄 Processing {sample_id}")
                process_sample_and_embed_UNI(
                    sample_id=sample_id,
                    base_dir=args.base_dir_hest,
                    model_path=args.uni_model,
                    out_root=args.out_root,
                    device=device,
                    patch_radius=args.patch_radius,
                    batch_size=args.uni_batch_size,
                )
            except Exception as e:
                log.error(f"❌ Failed to process {sample_id}: {e}")
                with open(failed_uni_log, "a") as f:
                    f.write(f"{sample_id}: {str(e)}\n")

    # scGPT (optional)
    if args.run_scgpt:
        ensure_imports_for("scgpt")

        if args.scgpt_model_dir is None:
            log.error("--scgpt-model-dir is required when --run-scgpt is set.")
            sys.exit(1)

        base_dir_processed = args.out_root
        if not base_dir_processed.exists():
            log.error(f"Processed data directory not found: {base_dir_processed}")
            sys.exit(1)

        # Sample list derived from processed out_root
        sample_list = [p.stem for p in base_dir_processed.iterdir() if p.is_dir()]
        if args.samples or args.samples_file:
            # Restrict to requested
            requested = set(resolve_ids(args, meta_df))
            sample_list = [s for s in sample_list if s in requested]

        failed_scgpt_log = Path("failed_samples_scgpt.log")
        for sample_name in (tqdm(sample_list, desc="[scGPT] samples") if tqdm else sample_list):
            try:
                log.info(f"\n🔄 scGPT on {sample_name}")
                run_embed_scGPT(
                    dataset_path=base_dir_processed / sample_name / "adata.h5ad",
                    model_dir=args.scgpt_model_dir,
                    output_dir=base_dir_processed / sample_name,
                    n_hvg=args.scgpt_n_hvg,
                    gene_col=args.scgpt_gene_col,
                    layer_key=args.scgpt_layer_key,
                    log_norm=args.scgpt_log_norm,
                    seed=args.scgpt_seed,
                    max_seq_len=args.scgpt_max_seq_len,
                    batch_size=args.scgpt_batch_size,
                    input_bins=args.scgpt_input_bins,
                    model_run=args.scgpt_model_run,
                    num_workers=args.scgpt_num_workers,
                )
            except Exception as e:
                log.error(f"❌ Failed to process {sample_name}: {e}")
                with open(failed_scgpt_log, "a") as f:
                    f.write(f"{sample_name}: {str(e)}\n")

    log.info("Done.")


if __name__ == "__main__":
    main()
