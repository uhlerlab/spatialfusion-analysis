#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python 3.9+
from __future__ import annotations

import os
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc

# heavy libs used by the pipeline
import matplotlib.pyplot as plt  # only used if --plot
from tqdm import tqdm
import tifffile

# bin2cell toolkit
import bin2cell as b2c

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
sc.set_figure_params(figsize=(3, 3), frameon=False)

log = logging.getLogger("crc_bin2cell")

# ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleCfg:
    rel_dir: str                    # sample folder under --root
    source_image: str               # *.btf used in spaceranger --image
    mask_row: Tuple[int, int]       # (row_min, row_max)
    mask_col: Tuple[int, int]       # (col_min, col_max)

# Per-sample defaults (you can override via --config or CLI)
DEFAULT_SAMPLES: Dict[str, SampleCfg] = {
    # CRCs
    "P1CRC": SampleCfg(
        rel_dir="P1CRC",
        source_image="Visium_HD_Human_Colon_Cancer_P1_tissue_image.btf",
        mask_row=(1450, 1550), mask_col=(250, 450)
    ),
    "P2CRC": SampleCfg(
        rel_dir="P2CRC",
        source_image="Visium_HD_Human_Colon_Cancer_P2_tissue_image.btf",
        mask_row=(1350, 1450), mask_col=(250, 450)
    ),
    "P5CRC": SampleCfg(
        rel_dir="P5CRC",
        source_image="Visium_HD_Human_Colon_Cancer_P5_tissue_image.btf",
        mask_row=(1350, 1450), mask_col=(250, 450)
    ),
    # NATs
    "P3NAT": SampleCfg(
        rel_dir="P3NAT",
        source_image="Visium_HD_Human_Colon_Normal_P3_tissue_image.btf",
        mask_row=(1650, 1750), mask_col=(250, 450)
    ),
    "P5NAT": SampleCfg(
        rel_dir="P5NAT",
        source_image="Visium_HD_Human_Colon_Normal_P5_tissue_image.btf",
        mask_row=(1350, 1450), mask_col=(250, 450)
    ),
}


# ------------------------------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="VisiumHD-CRC: StarDist cell labels + bin-to-cell aggregation (bin2cell).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--root", type=Path,
                   default=Path("/ewsc/yatesjos/Broad_SpatialFoundation/VisiumHD-CRC/"),
                   help="Root folder containing sample subfolders (e.g., P1CRC/).")
    p.add_argument("--stardist-dir", type=Path, default=Path("./stardist"),
                   help="Working directory to write StarDist inputs/outputs.")
    p.add_argument("--write-adata", action="store_true",
                   help="Write <sample>/adata.h5ad after bin_to_cell (same location as input sample).")

    # Which samples
    p.add_argument("--samples", type=str, default="P1CRC,P2CRC,P5CRC,P3NAT,P5NAT",
                   help="Comma-separated sample names to run from defaults/config.")
    p.add_argument("--config", type=Path, default=None,
                   help="Optional JSON with sample configs overriding defaults. "
                        "Format: { 'P1CRC': {'rel_dir': 'P1CRC', 'source_image': 'foo.btf', "
                        "'mask_row': [a,b], 'mask_col':[c,d]}, ... }")

    # Processing params
    p.add_argument("--mpp", type=float, default=0.3, help="Microns-per-pixel for b2c.scaled_he_image.")
    p.add_argument("--min-cells-gene", type=int, default=3, help="scanpy filter_genes min_cells.")
    p.add_argument("--min-counts-cell", type=int, default=1, help="scanpy filter_cells min_counts.")
    p.add_argument("--min-total-counts", type=int, default=30,
                   help="Final filter: drop cells with total counts < this threshold.")
    p.add_argument("--prob-thresh", type=float, default=0.01, help="StarDist probability threshold.")
    p.add_argument("--stardist-model", type=str, default="2D_versatile_he",
                   help="StarDist pretrained model name.")
    p.add_argument("--labels-key", type=str, default="labels_he",
                   help="Obs column to put initial labels into.")
    p.add_argument("--expanded-labels-key", type=str, default="labels_he_expanded",
                   help="Obs column to put expanded labels into.")

    # Spatial keys / plotting
    p.add_argument("--spatial-key", type=str, default="spatial_cropped_150_buffer",
                   help="Spatial basis where cropped image is registered (used in insert_labels/plots).")
    p.add_argument("--img-key", type=str, default="0.3_mpp_150_buffer",
                   help="Key under .uns['spatial'] used by scanpy plotting to select background image.")
    p.add_argument("--plot", action="store_true", help="Show scanpy spatial plots (disabled by default).")

    # Logging
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def load_sample_cfgs(args) -> Dict[str, SampleCfg]:
    cfgs = dict(DEFAULT_SAMPLES)
    if args.config and args.config.exists():
        user_cfg = json.loads(args.config.read_text())
        for k, v in user_cfg.items():
            cfgs[k] = SampleCfg(
                rel_dir=v["rel_dir"],
                source_image=v["source_image"],
                mask_row=tuple(v["mask_row"]),
                mask_col=tuple(v["mask_col"]),
            )
    # filter to requested set
    wanted = [s.strip() for s in args.samples.split(",") if s.strip()]
    return {k: v for k, v in cfgs.items() if k in wanted}


def mask_from_cfg(adata: sc.AnnData, cfg: SampleCfg) -> np.ndarray:
    rr0, rr1 = cfg.mask_row
    cc0, cc1 = cfg.mask_col
    return (
        (adata.obs["array_row"] >= rr0)
        & (adata.obs["array_row"] <= rr1)
        & (adata.obs["array_col"] >= cc0)
        & (adata.obs["array_col"] <= cc1)
    ).to_numpy()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------
# Per-sample pipeline
# ------------------------------------------------------------------------------------

def run_one_sample(sample_name: str, cfg: SampleCfg, args) -> Optional[Path]:
    """
    Returns path to final adata.h5ad if --write-adata, else None.
    """
    sample_root = args.root / cfg.rel_dir
    path_2um = sample_root / "binned_outputs" / "square_002um"
    spaceranger_spatial = sample_root / "spatial"
    source_image_path = sample_root / cfg.source_image

    # quick sanity checks
    if not path_2um.exists():
        log.error("[%s] Missing 2µm output folder: %s", sample_name, path_2um)
        return None
    if not spaceranger_spatial.exists():
        log.error("[%s] Missing spaceranger 'spatial' folder: %s", sample_name, spaceranger_spatial)
        return None
    if not source_image_path.exists():
        log.error("[%s] Missing source image (btf): %s", sample_name, source_image_path)
        return None

    # optional: read the WSI to confirm it's readable (matches your notebook)
    try:
        with tifffile.TiffFile(str(source_image_path)) as tif:
            _ = tif.series[0].asarray()
    except Exception as e:
        log.warning("[%s] Could not read source image: %s (continuing)", sample_name, e)

    # read visium 2µm data
    adata = b2c.read_visium(
        str(path_2um),
        source_image_path=str(source_image_path),
        spaceranger_image_path=str(spaceranger_spatial),
    )
    adata.var_names_make_unique()

    # QC filters
    sc.pp.filter_genes(adata, min_cells=args.min_cells_gene)
    sc.pp.filter_cells(adata, min_counts=args.min_counts_cell)

    # HE scaled image + destripe
    ensure_dir(args.stardist_dir / sample_name)
    he_tiff = args.stardist_dir / sample_name / "he.tiff"
    he_npz = args.stardist_dir / sample_name / "he.npz"

    b2c.scaled_he_image(adata, mpp=args.mpp, save_path=str(he_tiff))
    b2c.destripe(adata)

    # Optional region-of-interest preview (before labels)
    if args.plot:
        bmask = mask_from_cfg(adata, cfg)
        bdata = adata[bmask].copy()
        sc.pl.spatial(
            bdata,
            color=[None, "n_counts", "n_counts_adjusted"],
            color_map="OrRd",
            img_key=args.img_key,
            basis=args.spatial_key,
        )

    # StarDist inference
    b2c.stardist(
        image_path=str(he_tiff),
        labels_npz_path=str(he_npz),
        stardist_model=args.stardist_model,
        prob_thresh=args.prob_thresh,
    )

    # Insert labels into obs
    b2c.insert_labels(
        adata,
        labels_npz_path=str(he_npz),
        basis="spatial",
        spatial_key=args.spatial_key,
        mpp=args.mpp,
        labels_key=args.labels_key,
    )

    # visualize labels (optional)
    if args.plot:
        bmask = mask_from_cfg(adata, cfg)
        bdata = adata[bmask].copy()
        bdata = bdata[bdata.obs[args.labels_key] > 0]
        bdata.obs[args.labels_key] = bdata.obs[args.labels_key].astype(str)
        sc.pl.spatial(
            bdata, color=[None, args.labels_key], img_key=args.img_key, basis=args.spatial_key
        )

    # Expand labels (graph-based dilation)
    b2c.expand_labels(
        adata,
        labels_key=args.labels_key,
        expanded_labels_key=args.expanded_labels_key,
    )

    # visualize expanded (optional)
    if args.plot:
        bmask = mask_from_cfg(adata, cfg)
        bdata = adata[bmask].copy()
        bdata = bdata[bdata.obs[args.expanded_labels_key] > 0]
        bdata.obs[args.expanded_labels_key] = bdata.obs[args.expanded_labels_key].astype(str)
        sc.pl.spatial(
            bdata, color=[None, args.expanded_labels_key], img_key=args.img_key, basis=args.spatial_key
        )

    # Bin → Cell aggregation
    cdata = b2c.bin_to_cell(
        adata,
        labels_key=args.expanded_labels_key,
        spatial_keys=["spatial", args.spatial_key],
    )

    # Final light cleaning 
    tot = np.asarray(cdata.X.sum(axis=1)).ravel()
    keep = (pd.Series(tot) >= args.min_total_counts).to_numpy()
    cdata = cdata[keep].copy()

    # small region check plot (optional)
    if args.plot:
        cell_mask = mask_from_cfg(cdata, cfg)
        ddata = cdata[cell_mask].copy()
        sc.pl.spatial(ddata, color=["bin_count"], img_key=args.img_key, basis=args.spatial_key)

    # Write .h5ad next to sample 
    out_path = (args.root / cfg.rel_dir) / "adata.h5ad"
    if args.write_adata:
        cdata.write_h5ad(out_path)
        log.info("[%s] Wrote %s", sample_name, out_path)
        return out_path
    else:
        log.info("[%s] Completed (not writing adata; pass --write-adata to save).", sample_name)
        return None


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    ensure_dir(args.stardist_dir)

    sample_cfgs = load_sample_cfgs(args)
    if not sample_cfgs:
        log.error("No samples selected. Check --samples or --config.")
        return

    log.info("Running on samples: %s", ", ".join(sample_cfgs.keys()))
    for name, cfg in tqdm(sample_cfgs.items(), desc="Samples"):
        try:
            run_one_sample(name, cfg, args)
        except Exception as e:
            tqdm.write(f"❌ {name}: {e}")

    log.info("✅ Done.")


if __name__ == "__main__":
    main()
