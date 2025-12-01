#!/usr/bin/env python3

import argparse
import pathlib as pl
from typing import Any

import anndata as ad
import nicheformer
import nptyping as npt
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader


# -----------------------------
# Dicts (unchanged)
# -----------------------------
MODALITY_DICT = {
    "dissociated": 3,
    "spatial": 4,
}

SPECIES_DICT = {
    "human": 5,
    "Homo sapiens": 5,
    "Mus musculus": 6,
    "mouse": 6,
}

TECHNOLOGY_DICT = {
    "merfish": 7,
    "MERFISH": 7,
    "cosmx": 8,
    "NanoString digital spatial profiling": 8,
    "Xenium": 9,
    "10x 5' v2": 10,
    "10x 3' v3": 11,
    "10x 3' v2": 12,
    "10x 5' v1": 13,
    "10x 3' v1": 14,
    "10x 3' transcription profiling": 15,
    "10x transcription profiling": 15,
    "10x 5' transcription profiling": 16,
    "CITE-seq": 17,
    "Smart-seq v4": 18,
}


# -----------------------------
# Functions (unchanged)
# -----------------------------
def fix_tech_mean(x: npt.NDArray[Any, Any]) -> npt.NDArray[Any, Any]:
    rounded = np.round(np.nan_to_num(x))
    return np.where(rounded == 0, 1, rounded)


def subset_genes_and_mean(
    adata: ad.AnnData, vocab: ad.AnnData, tech_mean: npt.NDArray[Any, Any]
) -> ad.AnnData:

    adata.var["orig_idx"] = adata.var.index
    if "gene_ids" in adata.var:
        adata.var.index = adata.var.gene_ids

    vocab.var["tech_mean"] = tech_mean
    merged = ad.concat([vocab, adata], join="inner", axis=0, merge="unique")
    return merged[1:].copy()


def run_embed_nicheformer(
    dataset_path: str | pl.Path,
    model_path: str | pl.Path,
    vocab_path: str | pl.Path,
    tech_mean_path: str | pl.Path,
    output_dir: str | pl.Path,
    modality: str,
    species: str,
    technology: str,
    max_seq_len: int = 1500,
    batch_size: int = 32,
    chunk_size: int = 1000,
    num_workers: int = 0,
) -> None:

    if modality not in MODALITY_DICT:
        raise ValueError(f"invalid modality '{modality}'")
    if species not in SPECIES_DICT:
        raise ValueError(f"invalid species '{species}'")
    if technology not in TECHNOLOGY_DICT:
        raise ValueError(f"invalid technology '{technology}'")

    adata = ad.read_h5ad(dataset_path)
    vocab = ad.read_h5ad(vocab_path)
    full_tech_mean = fix_tech_mean(np.load(tech_mean_path))

    if "cell_id" in adata.obs:
        cell_ids = adata.obs["cell_id"]
    else:
        cell_ids = adata.obs_names.astype(str)

    adata = subset_genes_and_mean(adata, vocab, full_tech_mean)
    tech_mean = adata.var.tech_mean.to_numpy()

    adata.obs["modality"] = MODALITY_DICT[modality]
    adata.obs["species"] = SPECIES_DICT[species]
    adata.obs["assay"] = TECHNOLOGY_DICT[technology]
    if "nicheformer_split" not in adata.obs.columns:
        adata.obs["nicheformer_split"] = "train"

    dataset = nicheformer.data.NicheformerDataset(
        adata=adata,
        technology_mean=tech_mean,
        split="train",
        max_seq_len=max_seq_len,
        aux_tokens=30,
        chunk_size=chunk_size,
        metadata_fields={"obs": ["modality", "species", "assay"]},
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = nicheformer.models.Nicheformer.load_from_checkpoint(
        checkpoint_path=model_path,
        strict=False,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embeddings = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            emb = model.get_embeddings(batch=batch, layer=-1)
            embeddings.append(emb.cpu())

    embeddings = np.concatenate(embeddings, axis=0)

    output_dir = pl.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(embeddings, index=cell_ids).to_parquet(
        output_dir / "nicheformer.parquet"
    )


# -----------------------------
# CLI / main entrypoint
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Nicheformer embedding on an AnnData file."
    )

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--tech_mean_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--modality", type=str, required=True)
    parser.add_argument("--species", type=str, required=True)
    parser.add_argument("--technology", type=str, required=True)

    parser.add_argument("--max_seq_len", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    run_embed_nicheformer(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        tech_mean_path=args.tech_mean_path,
        output_dir=args.output_dir,
        modality=args.modality,
        species=args.species,
        technology=args.technology,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
