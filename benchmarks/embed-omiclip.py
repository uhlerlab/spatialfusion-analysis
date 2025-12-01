#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import List, Union, Optional, Iterable, Tuple

import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from PIL import Image
import tifffile
import shapely.wkb

import loki.utils
import loki.preprocess

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run OmiCLIP model workflow")

    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing model and support files.")
    p.add_argument("--adata_path", type=str, required=True,
                   help="Path to input .h5ad dataset.")
    p.add_argument("--housekeeping_gmt", type=str, required=True,
                   help="Path to housekeeping genes .gmt file.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory where embeddings and outputs are saved.")
    p.add_argument("--wsi_path", type=str, required=True,
                   help="Path to H&E / WSI image.")
    p.add_argument("--spatial_key", type=str, required=True,
                   help="the name of the spatial key in adata.obsm containing spatial coordinates.")

    p.add_argument("--device", type=str, default="cuda",
                   help="Device: cuda or cpu.")

    return p.parse_args()


# ---------------------------------------------------------
# Utilities (logic unchanged)
# ---------------------------------------------------------
def read_gmt(file_path):
    gene_sets = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            set_name = parts[0]
            description = parts[1]
            genes = parts[2:]
            gene_sets[set_name] = {
                "description": description,
                "genes": genes
            }
    return gene_sets


@torch.no_grad()
def encode_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: Union[str, torch.device],
    batch_size: int = 256,
    max_length: Optional[int] = None,
    use_amp: bool = True
) -> torch.Tensor:

    device = torch.device(device)
    model = model.to(device).eval()

    amp_dtype = torch.bfloat16 if torch.cuda.is_available(
    ) and torch.cuda.is_bf16_supported() else torch.float16
    autocast_ctx = torch.cuda.amp.autocast if (
        use_amp and device.type == "cuda") else torch.cpu.amp.autocast

    all_feats = []
    N = len(texts)

    for start in tqdm(range(0, N, batch_size)):
        batch_texts = texts[start:start + batch_size]
        if max_length is not None:
            batch_texts = [t[:max_length] for t in batch_texts]

        # Try HuggingFace tokenizer
        try:
            tok_out = tokenizer(batch_texts, return_tensors="pt",
                                padding=True, truncation=True)
        except TypeError:
            tok_out = tokenizer(batch_texts)

        if isinstance(tok_out, dict):
            tok_out = {k: (v.to(device) if torch.is_tensor(v) else v)
                       for k, v in tok_out.items()}
        else:
            tok_out = tok_out.to(device)

        try:
            with autocast_ctx(dtype=amp_dtype) if device.type == "cuda" else torch.no_grad():
                feats = model.encode_text(tok_out)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1 and device.type == "cuda":
                torch.cuda.empty_cache()
                return encode_texts(model, tokenizer, texts, device,
                                    batch_size=max(1, batch_size // 2),
                                    max_length=max_length,
                                    use_amp=use_amp)
            raise

        feats = F.normalize(feats, p=2, dim=-1).cpu()
        all_feats.append(feats)

        if device.type == "cuda":
            del feats, tok_out
            torch.cuda.empty_cache()

    return torch.cat(all_feats, dim=0)


def encode_text_df(model, tokenizer, df, col_name, device):
    texts = df[col_name].astype(str).tolist()
    return encode_texts(model, tokenizer, texts, device)


@torch.no_grad()
def encode_cell_patches(
    model: torch.nn.Module,
    preprocess,
    wsi: np.ndarray,
    coords: Iterable[Tuple[float, float]],
    ids: Optional[Iterable[str]] = None,
    device: Union[str, torch.device] = "cuda",
    patch_size: int = 256,
    batch_size: int = 64,
    use_amp: bool = True,
):

    device = torch.device(device)
    model = model.to(device).eval()

    H, W = wsi.shape[:2]

    if wsi.ndim == 2:
        wsi = np.stack([wsi] * 3, axis=-1)
    elif wsi.shape[2] == 1:
        wsi = np.repeat(wsi, 3, axis=2)

    coords = list(coords)
    out_ids = list(ids) if ids is not None else [
        str(i) for i in range(len(coords))]

    half = patch_size // 2

    def extract_patch(x, y):
        x, y = int(x), int(y)
        x0, x1 = x - half, x + half
        y0, y1 = y - half, y + half

        pad_x0 = max(0, -x0)
        pad_x1 = max(0, x1 - W)
        pad_y0 = max(0, -y0)
        pad_y1 = max(0, y1 - H)

        crop = wsi[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        patch = np.pad(
            crop, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode="constant")

        if patch.shape[:2] != (patch_size, patch_size):
            patch = np.resize(patch, (patch_size, patch_size, 3))
        return patch

    probe_patch = preprocess(Image.fromarray(extract_patch(*coords[0])))
    probe_imgs = probe_patch.unsqueeze(0).to(device)
    amp_dtype = torch.bfloat16 if (
        device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            D = model.encode_image(probe_imgs).shape[-1]
    else:
        D = model.encode_image(probe_imgs).shape[-1]
    del probe_imgs

    if device.type == "cuda":
        torch.cuda.empty_cache()

    N = len(coords)
    embs = torch.zeros((N, D), dtype=torch.float32)
    failures = 0

    start_idx = 0
    pbar = tqdm(total=N, desc="Embedding cells (batched)")
    while start_idx < N:
        end_idx = min(start_idx + batch_size, N)
        batch_coords = coords[start_idx:end_idx]

        try:
            batch_tensors = [preprocess(Image.fromarray(
                extract_patch(x, y))) for (x, y) in batch_coords]
        except Exception:
            failures += 1
            black = Image.fromarray(
                np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
            batch_tensors = [preprocess(black) for _ in batch_coords]

        imgs = torch.stack(batch_tensors).to(device, non_blocking=True)

        try:
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    feats = model.encode_image(imgs)
            else:
                feats = model.encode_image(imgs)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda" and (end_idx - start_idx) > 1:
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            failures += (end_idx - start_idx)
            feats = torch.zeros((end_idx - start_idx, D),
                                dtype=torch.float32, device=device)

        feats = F.normalize(feats, p=2, dim=-1).detach().cpu()
        embs[start_idx:end_idx] = feats

        if device.type == "cuda":
            del imgs, feats
            torch.cuda.empty_cache()

        pbar.update(end_idx - start_idx)
        start_idx = end_idx

    pbar.close()
    if failures:
        print(
            f"[encode_cell_patches] Completed with {failures} fallback zero embeddings.")

    return embs, out_ids


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------
def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading OmiCLIP model...")
    model_path = data_dir / "checkpoint.pt"
    model, preprocess, tokenizer = loki.utils.load_model(
        str(model_path), args.device)
    model.eval()

    # Load data
    adata = sc.read_h5ad(args.adata_path)

    # Read housekeeping genes
    housekeeping = read_gmt(args.housekeeping_gmt)
    housekeeping = pd.DataFrame(
        housekeeping["HSIAO_HOUSEKEEPING_GENES"]["genes"], columns=["genesymbol"])

    # Compute top-k strings
    top_k_genes_str = loki.preprocess.generate_gene_df(adata, housekeeping)
    top_k_genes_str.to_parquet(output_dir / "OmiCLIP_kstring.parquet")

    # Encode text embeddings
    print("Encoding text embeddings...")
    text_embs = encode_text_df(
        model, tokenizer, top_k_genes_str, "label", args.device)
    pd.DataFrame(text_embs.cpu().numpy(), index=adata.obs_names).to_parquet(
        output_dir / "OmiCLIP_text_emb.parquet"
    )

    # Image embeddings
    he_coords = adata.obsm[args.spatial_key]

    print("Loading WSI image...")
    with tifffile.TiffFile(args.wsi_path) as tif:
        wsi = tif.pages[0].asarray()

    print("Encoding cell image patches...")
    embs, cell_ids = encode_cell_patches(
        model=model,
        preprocess=preprocess,
        wsi=wsi,
        coords=he_coords,
        ids=adata.obs_names,
        device=args.device,
        patch_size=256,
        batch_size=64,
        use_amp=True,
    )

    df = pd.DataFrame(embs.cpu().numpy(), index=list(cell_ids))
    out_file = output_dir / "OmiCLIP_image_emb.parquet"
    df.to_parquet(out_file)
    print(f"Saved {len(df)} embeddings to {out_file}")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
