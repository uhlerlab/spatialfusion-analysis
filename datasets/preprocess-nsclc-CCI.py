#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from tqdm import tqdm
from scipy.sparse import issparse, csr_matrix, eye
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger("lr_scores")


# ---------------------------
# Gene vector utilities
# ---------------------------

def _obs_vec(adata: sc.AnnData, gene: str, layer: Optional[str] = None) -> np.ndarray:
    return adata.obs_vector(gene, layer=layer).astype(float)

def _split_any(s: str, toks: Tuple[str, ...]) -> List[str]:
    out = [s]
    for t in toks:
        out = sum([chunk.split(t) for chunk in out], [])
    return [x.strip() for x in out if x.strip()]

def _combine_vectors(vs: List[np.ndarray], rule: str) -> np.ndarray:
    if len(vs) == 1:
        return vs[0]
    if rule == 'min':
        return np.minimum.reduce(vs)
    if rule == 'geom_mean':
        eps = 1e-12
        return np.exp(np.mean([np.log(v + eps) for v in vs], axis=0)) - eps
    if rule == 'max':
        return np.maximum.reduce(vs)
    raise ValueError(f"Unknown combine rule: {rule}")

def _make_complex_vector(
    adata: sc.AnnData, spec: str, layer: Optional[str] = None,
    and_delims: Tuple[str, ...] = ('_', '+'), or_delims: Tuple[str, ...] = ('|',),
    rule_and: str = 'geom_mean', rule_or: str = 'max',
    missing: str = 'skip'  # 'skip' -> return None, 'zero' -> zero vector
) -> Optional[np.ndarray]:
    present = set(adata.var_names)
    or_groups = _split_any(spec, or_delims)
    group_vecs: List[np.ndarray] = []
    for group in or_groups:
        and_genes = _split_any(group, and_delims)
        vecs: List[np.ndarray] = []
        for g in and_genes:
            if g not in present:
                if missing == 'skip':
                    return None
                elif missing == 'zero':
                    return np.zeros(adata.n_obs, dtype=float)
            vecs.append(_obs_vec(adata, g, layer))
        group_vecs.append(_combine_vectors(vecs, rule_and))
    return _combine_vectors(group_vecs, rule_or)


# ---------------------------
# Spatial weight matrix
# ---------------------------

def _get_weight_matrix(
    adata: sc.AnnData,
    obsp_distance_key: str = 'spatial_distance',
    obsp_connect_key: str = 'spatial_connectivities',
    build_if_missing: bool = False,
    build_mode: str = 'radius',     # 'radius' or 'knn'
    radius: Optional[float] = None, # required if radius
    n_neighbors: Optional[int] = None, # required if knn
    coord_key: str = 'spatial',
    sigma: float = 50.0,
    include_self: bool = True,
    use_binary_if_no_distance: bool = False,
) -> csr_matrix:
    n = adata.n_obs
    if obsp_distance_key in adata.obsp:
        D = adata.obsp[obsp_distance_key]
        if not issparse(D): D = csr_matrix(D)
        if D.shape != (n, n):
            raise ValueError(f"{obsp_distance_key} shape {D.shape} != ({n},{n}).")
        W = D.tocsr(copy=True)
        W.data = np.exp(-(W.data**2) / (2.0 * sigma**2))

    elif obsp_connect_key in adata.obsp:
        C = adata.obsp[obsp_connect_key]
        if not issparse(C): C = csr_matrix(C)
        if C.shape != (n, n):
            raise ValueError(f"{obsp_connect_key} shape {C.shape} != ({n},{n}).")
        if use_binary_if_no_distance and not build_if_missing:
            W = C.tocsr(copy=True)
        else:
            if coord_key not in adata.obsm:
                raise ValueError(f"Missing adata.obsm['{coord_key}'] to build distances.")
            X = adata.obsm[coord_key]
            if build_mode == 'radius':
                if radius is None:
                    raise ValueError("Set --radius for radius mode.")
                D = radius_neighbors_graph(X, radius=radius, mode='distance', include_self=False, n_jobs=-1).tocsr()
            elif build_mode == 'knn':
                if n_neighbors is None:
                    raise ValueError("Set --n-neighbors for knn mode.")
                D = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False, n_jobs=-1).tocsr()
            else:
                raise ValueError("build_mode must be 'radius' or 'knn'")
            adata.obsp[obsp_distance_key] = D
            W = D.tocsr(copy=True)
            W.data = np.exp(-(W.data**2) / (2.0 * sigma**2))

    else:
        if not build_if_missing:
            raise ValueError("No spatial graph found and build_if_missing=False.")
        if coord_key not in adata.obsm:
            raise ValueError(f"Missing adata.obsm['{coord_key}'] to build distances.")
        X = adata.obsm[coord_key]
        if build_mode == 'radius':
            if radius is None:
                raise ValueError("Set --radius for radius mode.")
            D = radius_neighbors_graph(X, radius=radius, mode='distance', include_self=False, n_jobs=-1).tocsr()
        elif build_mode == 'knn':
            if n_neighbors is None:
                raise ValueError("Set --n-neighbors for knn mode.")
            D = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False, n_jobs=-1).tocsr()
        else:
            raise ValueError("build_mode must be 'radius' or 'knn'")
        adata.obsp[obsp_distance_key] = D
        W = D.tocsr(copy=True)
        W.data = np.exp(-(W.data**2) / (2.0 * sigma**2))

    if include_self:
        W = W + eye(n, format='csr')
    return W


# ---------------------------
# LR scoring (single AnnData)
# ---------------------------

def compute_spatial_lr_scores(
    adata: sc.AnnData,
    df_ligrec: pd.DataFrame,
    ligand_col: str = 'ligand',
    receptor_col: str = 'receptor',
    layer: Optional[str] = 'smoothed',
    complex_and_delims: Tuple[str, ...] = ('_', '+'),
    complex_or_delims: Tuple[str, ...] = ('|',),
    combine_rule_and: str = 'geom_mean',
    combine_rule_or: str = 'max',
    missing_handling: str = 'skip',
    sigma: float = 50.0,
    include_self: bool = True,
    use_mean: bool = True,
    obsp_distance_key: str = 'spatial_distance',
    obsp_connect_key: str = 'spatial_connectivities',
    build_if_missing: bool = False,
    build_mode: str = 'knn',
    radius: Optional[float] = None,
    n_neighbors: Optional[int] = 30,
    coord_key: str = 'spatial',
    use_binary_if_no_distance: bool = False,
    batch_size: int = 128,
    return_skipped: bool = True,
) -> Tuple[pd.DataFrame, Optional[List[Tuple[str, str]]]]:
    pairs = df_ligrec[[ligand_col, receptor_col]].drop_duplicates().reset_index(drop=True)

    W = _get_weight_matrix(
        adata,
        obsp_distance_key=obsp_distance_key,
        obsp_connect_key=obsp_connect_key,
        build_if_missing=build_if_missing,
        build_mode=build_mode,
        radius=radius,
        n_neighbors=n_neighbors,
        coord_key=coord_key,
        sigma=sigma,
        include_self=include_self,
        use_binary_if_no_distance=use_binary_if_no_distance,
    )

    n = adata.n_obs
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    if use_mean:
        row_sum[row_sum == 0] = 1e-12

    uniq_receptors = pairs[receptor_col].unique().tolist()
    Rproj: Dict[str, np.ndarray] = {}
    skipped_receptors: set[str] = set()
    tqdm.write(f"[{adata.uns.get('sample_id','adata')}] Precomputing receptor neighborhoods for {len(uniq_receptors)} receptors…")
    for rec in tqdm(uniq_receptors, desc="Receptor complexes", leave=False):
        rv = _make_complex_vector(
            adata, rec, layer=layer,
            and_delims=complex_and_delims, or_delims=complex_or_delims,
            rule_and=combine_rule_and, rule_or=combine_rule_or,
            missing=missing_handling
        )
        if rv is None:
            skipped_receptors.add(rec)
            continue
        r_near = W.dot(rv)
        if use_mean:
            r_near = r_near / row_sum
        Rproj[rec] = r_near.astype(np.float32)

    tqdm.write(f"[{adata.uns.get('sample_id','adata')}] Scoring {len(pairs)} ligand–receptor pairs…")
    batch_frames: List[pd.DataFrame] = []
    colbuf: Dict[str, np.ndarray] = {}
    skipped_pairs: List[Tuple[str, str]] = []
    count = 0

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="LR pairs"):
        lig, rec = row[ligand_col], row[receptor_col]
        lv = _make_complex_vector(
            adata, lig, layer=layer,
            and_delims=complex_and_delims, or_delims=complex_or_delims,
            rule_and=combine_rule_and, rule_or=combine_rule_or,
            missing=missing_handling
        )
        if lv is None or (rec in skipped_receptors) or (rec not in Rproj):
            skipped_pairs.append((lig, rec))
            continue
        colname = f"{lig}__{rec}"
        colbuf[colname] = (lv.astype(np.float32) * Rproj[rec])
        count += 1
        if count >= batch_size:
            batch_frames.append(pd.DataFrame(colbuf, index=adata.obs_names))
            colbuf, count = {}, 0

    if colbuf:
        batch_frames.append(pd.DataFrame(colbuf, index=adata.obs_names))

    scores_df = pd.concat(batch_frames, axis=1) if batch_frames else pd.DataFrame(index=adata.obs_names)
    return (scores_df, skipped_pairs) if return_skipped else (scores_df, None)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Compute spatial ligand–receptor scores for Visium/VisiumHD samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--base-dir", type=Path, default=Path("/ewsc/yatesjos/Broad_SpatialFoundation/VisiumHD-LUAD-processed"))
    p.add_argument("--lr-pairs-csv", type=Path, default=Path("CellChat_LR_pairs.csv"))
    p.add_argument("--include", type=str, default="", help="Comma-separated list of sample names to include (defaults to all).")
    p.add_argument("--exclude", type=str, default="full_cohort", help="Comma-separated list of names to skip.")
    p.add_argument("--output-suffix", type=str, default="_LR_scores.parquet")

    # Preprocess & smoothing
    p.add_argument("--normalize", action="store_true", help="Run normalize_total(target_sum=10000) + log1p.")
    p.add_argument("--target-sum", type=float, default=10000)
    p.add_argument("--copy-layer-name", type=str, default="norm", help="Save pre-smoothed counts into this layer name.")
    p.add_argument("--smoothed-layer", type=str, default="smoothed", help="Name for smoothed layer to compute on.")
    p.add_argument("--knn-bw", type=float, default=30.0, help="Bandwidth for dc.pp.knn (used with cutoff).")
    p.add_argument("--knn-cutoff", type=float, default=0.1)

    # Spatial scaling
    p.add_argument("--scale-by-mpp", action="store_true",
                   help="If set, replace obsm['spatial'] by pixels*microns_per_pixel (VisiumHD uns).")

    # Distance graph & scoring
    p.add_argument("--build-mode", choices=["knn", "radius"], default="knn")
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--sigma", type=float, default=50.0)
    p.add_argument("--include-self", action="store_true")
    p.add_argument("--use-mean", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)

    # Complex combination rules
    p.add_argument("--and-delims", type=str, default="_,+")
    p.add_argument("--or-delims", type=str, default="|")
    p.add_argument("--rule-and", choices=["geom_mean", "min"], default="geom_mean")
    p.add_argument("--rule-or", choices=["max", "min", "geom_mean"], default="max")
    p.add_argument("--missing", choices=["skip", "zero"], default="skip")

    # Misc
    p.add_argument("--save-skipped-json", action="store_true", help="Also save skipped pairs to JSON per sample.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load LR table
    if not args.lr_pairs_csv.exists():
        raise FileNotFoundError(f"LR pairs CSV not found: {args.lr_pairs_csv}")
    df_ligrec = pd.read_csv(args.lr_pairs_csv)

    # Sample discovery
    all_dirs = [p for p in Path(args.base_dir).iterdir() if p.is_dir()]
    includes = set(s.strip() for s in args.include.split(",") if s.strip()) or {p.name for p in all_dirs}
    excludes = set(s.strip() for s in args.exclude.split(",") if s.strip())
    samples = sorted([p for p in all_dirs if (p.name in includes and p.name not in excludes)])
    log.info("Samples: %s", ", ".join(p.name for p in samples) if samples else "NONE")

    # Delimiters
    and_delims = tuple(d for d in (x.strip() for x in args.and_delims.split(",")) if d)
    or_delims = tuple(d for d in (x.strip() for x in args.or_delims.split(",")) if d)

    for sample_dir in tqdm(samples, desc="Samples"):
        sample = sample_dir.name
        try:
            adata_path = sample_dir / "adata.h5ad"
            if not adata_path.exists():
                tqdm.write(f"❌ {sample}: missing {adata_path}")
                continue

            adata = sc.read_h5ad(adata_path)

            # optional normalize/log
            if args.normalize:
                sc.pp.normalize_total(adata, target_sum=args.target_sum)
                sc.pp.log1p(adata)

            # optional microns-per-pixel scaling into .obsm['spatial']
            if args.scale_by_mpp:
                try:
                    mpp = adata.uns['spatial']['VisiumHD']['scalefactors']['microns_per_pixel']
                    adata.obsm['spatial_px'] = adata.obsm['spatial']
                    adata.obsm['spatial'] = adata.obsm['spatial_px'] * mpp
                except Exception:
                    tqdm.write(f"⚠️ {sample}: could not scale by microns_per_pixel; proceeding with original spatial coords.")

            # keep a copy of pre-smoothed counts
            adata.layers[args.copy_layer_name] = adata.X.copy()

            # build spatial KNN for smoothing and apply
            dc.pp.knn(adata, key="spatial", bw=args.knn_bw, cutoff=args.knn_cutoff)
            adata.layers[args.smoothed_layer] = adata.obsp["spatial_connectivities"].dot(adata.layers[args.copy_layer_name])

            # also create a pure distance graph for scoring (knn default)
            if args.build_mode == "knn":
                G = kneighbors_graph(adata.obsm['spatial'], n_neighbors=args.n_neighbors, mode='distance', include_self=False, n_jobs=-1).tocsr()
                adata.obsp['spatial_distance'] = G
            else:
                if args.radius is None:
                    raise ValueError("Set --radius when --build-mode=radius")
                G = radius_neighbors_graph(adata.obsm['spatial'], radius=args.radius, mode='distance', include_self=False, n_jobs=-1).tocsr()
                adata.obsp['spatial_distance'] = G

            scores_df, skipped = compute_spatial_lr_scores(
                adata, df_ligrec,
                ligand_col='ligand', receptor_col='receptor',
                layer=args.smoothed_layer,
                sigma=args.sigma,
                include_self=args.include_self,
                use_mean=args.use_mean,
                build_if_missing=True,
                build_mode=args.build_mode,
                n_neighbors=args.n_neighbors,
                radius=args.radius,
                coord_key='spatial',
                use_binary_if_no_distance=False,
                complex_and_delims=and_delims,
                complex_or_delims=or_delims,
                combine_rule_and=args.rule_and,
                combine_rule_or=args.rule_or,
                missing_handling=args.missing,
                batch_size=args.batch_size,
            )

            out_path = sample_dir / f"{sample}{args.output_suffix}"
            scores_df.to_parquet(out_path, index=True)

            skipped_count = len(skipped) if skipped else 0
            tqdm.write(f"[{sample}] saved: {out_path} | shape={scores_df.shape} | skipped={skipped_count}")

            if args.save_skipped_json:
                (sample_dir / f"{sample}_LR_skipped.json").write_text(json.dumps(skipped or [], indent=2))

        except Exception as e:
            tqdm.write(f"❌ {sample}: {e}")

    log.info("✅ Done.")


if __name__ == "__main__":
    main()
