import gc
import gzip
import logging
import pathlib as pl
from datetime import datetime

import anndata as ad
import nicheformer
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tqdm

from torch.utils.data import DataLoader

# ============================================================
# Paths
# ============================================================

BASE_DIR = pl.Path(
    "../../../Broad_SpatialFoundation/hest_processed_data/"
)

MODEL_PATH = (
    "../../../Broad_SpatialFoundation/"
    "nicheformer_model/nicheformer.ckpt"
)

VOCAB_PATH = (
    "../../../Broad_SpatialFoundation/"
    "nicheformer_model/model.h5ad"
)

TECH_MEAN_PATH = (
    "../../../Broad_SpatialFoundation/"
    "nicheformer_model/xenium_mean_script.npy"
)

GTF_PATH = (
    "../../../Broad_SpatialFoundation/"
    "gencode.v48.basic.annotation.gtf.gz"
)

# ============================================================
# Config
# ============================================================

CONFIG = {
    "batch_size": 32,
    "max_seq_len": 1500,
    "aux_tokens": 30,
    "chunk_size": 1000,
    "num_workers": 0,
    "embedding_layer": -1,
}

# ============================================================
# Logging
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = BASE_DIR / f"nicheformer_batch_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)

logging.info("Starting Nicheformer batch embedding")

# ============================================================
# Utilities
# ============================================================

def set_seed(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Build symbol -> Ensembl mapping
# ============================================================

def build_symbol_to_ensembl_map(gtf_path):

    logging.info("Parsing GTF")

    records = []

    with gzip.open(gtf_path, "rt") as f:

        for line in f:

            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")

            if fields[2] != "gene":
                continue

            attrs = fields[8]

            attr_dict = {}

            for item in attrs.split(";"):

                item = item.strip()

                if item == "":
                    continue

                key, value = item.split(" ", 1)

                attr_dict[key] = value.strip('"')

            gene_id = attr_dict.get("gene_id")
            gene_name = attr_dict.get("gene_name")

            if gene_id and gene_name:

                gene_id = gene_id.split(".")[0]

                records.append(
                    (
                        gene_name.upper(),
                        gene_id,
                    )
                )

    mapping_df = pd.DataFrame(
        records,
        columns=["gene_symbol", "ensembl_id"],
    ).drop_duplicates()

    symbol_to_ens = mapping_df.set_index(
        "gene_symbol"
    )["ensembl_id"].to_dict()

    logging.info(
        f"Parsed {len(symbol_to_ens)} mappings"
    )

    return symbol_to_ens


# ============================================================
# Remove Xenium controls
# ============================================================

def remove_control_probes(adata):

    mask = ~adata.var_names.str.upper().str.startswith(
        (
            "BLANK_",
            "NEGCONTROLCODEWORD",
            "NEGCONTROLPROBE",
            "ANTISENSE_",
        )
    )

    return adata[:, mask].copy()


# ============================================================
# Map genes to Ensembl
# ============================================================

def map_genes_to_ensembl(
    adata,
    symbol_to_ens,
):

    adata.var_names = (
        adata.var_names
        .str.strip()
        .str.upper()
    )

    adata.var_names_make_unique()

    adata.var["ensembl_id"] = [
        symbol_to_ens.get(g, None)
        for g in adata.var_names
    ]

    mapped = adata.var["ensembl_id"].notnull().sum()

    logging.info(
        f"Mapped genes: {mapped}/{adata.n_vars}"
    )

    adata = adata[
        :,
        adata.var["ensembl_id"].notnull()
    ].copy()

    adata.var_names = (
        adata.var["ensembl_id"]
        .astype(str)
    )

    adata.var_names_make_unique()

    return adata


# ============================================================
# Align to vocab
# ============================================================

def align_to_vocab(
    adata,
    vocab,
):

    vocab = vocab[
        :,
        vocab.var_names.str.startswith("ENSG")
    ].copy()

    common_genes = vocab.var_names.intersection(
        adata.var_names
    )

    logging.info(
        f"Common genes: {len(common_genes)}"
    )

    adata = adata[:, common_genes].copy()

    ordered_genes = [
        g for g in vocab.var_names
        if g in adata.var_names
    ]

    adata = adata[:, ordered_genes].copy()

    return adata, ordered_genes, vocab


# ============================================================
# Align technology mean
# ============================================================

def align_technology_mean(
    ordered_genes,
    vocab,
    tech_mean_path,
):

    technology_mean_full = np.load(
        tech_mean_path
    )

    tech_mean_map = dict(
        zip(vocab.var_names, technology_mean_full)
    )

    technology_mean = np.array([
        tech_mean_map[g]
        for g in ordered_genes
    ])

    return technology_mean.astype(np.float32)


# ============================================================
# Add metadata
# ============================================================

def add_metadata(adata):

    adata.obs["modality"] = 4
    adata.obs["species"] = 5
    adata.obs["assay"] = 9

    if "nicheformer_split" not in adata.obs.columns:

        adata.obs["nicheformer_split"] = "train"

    return adata


# ============================================================
# Main embedding function
# ============================================================

def run_embed_nicheformer(
    dataset_path,
    output_dir,
    symbol_to_ens,
):

    logging.info(f"Loading {dataset_path}")

    adata = ad.read_h5ad(dataset_path)

    logging.info(f"Original shape: {adata.shape}")

    if "cell_id" in adata.obs:

        cell_ids = (
            adata.obs["cell_id"]
            .astype(str)
        )

    else:

        cell_ids = (
            adata.obs_names
            .astype(str)
        )

    # --------------------------------------------------------
    # Remove controls
    # --------------------------------------------------------

    adata = remove_control_probes(
        adata
    )

    logging.info(
        f"After removing controls: {adata.shape}"
    )

    # --------------------------------------------------------
    # Map to Ensembl
    # --------------------------------------------------------

    adata = map_genes_to_ensembl(
        adata,
        symbol_to_ens,
    )

    logging.info(
        f"After mapping: {adata.shape}"
    )

    # --------------------------------------------------------
    # Load vocab
    # --------------------------------------------------------

    vocab = sc.read_h5ad(
        VOCAB_PATH
    )

    # --------------------------------------------------------
    # Align genes
    # --------------------------------------------------------

    adata, ordered_genes, vocab = align_to_vocab(
        adata,
        vocab,
    )

    logging.info(
        f"Final aligned shape: {adata.shape}"
    )

    # --------------------------------------------------------
    # Align tech mean
    # --------------------------------------------------------

    technology_mean = align_technology_mean(
        ordered_genes,
        vocab,
        TECH_MEAN_PATH,
    )

    # --------------------------------------------------------
    # Convert counts to float32
    # --------------------------------------------------------

    adata.X = adata.X.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    adata = add_metadata(
        adata
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = nicheformer.data.NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split="train",
        max_seq_len=CONFIG["max_seq_len"],
        aux_tokens=CONFIG["aux_tokens"],
        chunk_size=CONFIG["chunk_size"],
        metadata_fields={
            "obs": [
                "modality",
                "species",
                "assay",
            ]
        },
    )

    logging.info(
        f"Token shape: {dataset.tokens.shape}"
    )

    # --------------------------------------------------------
    # Dataloader
    # --------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = (
        nicheformer.models.Nicheformer
        .load_from_checkpoint(
            checkpoint_path=MODEL_PATH,
            strict=False,
        )
    )

    model.eval()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = []

    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):

            batch = {
                k: v.to(device)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            emb = model.get_embeddings(
                batch=batch,
                layer=CONFIG["embedding_layer"],
            )

            embeddings.append(
                emb.cpu().numpy()
            )

            gc.collect()

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    # --------------------------------------------------------
    # Save parquet
    # --------------------------------------------------------

    outfile = (
        pl.Path(output_dir)
        / "nicheformer.parquet"
    )

    pd.DataFrame(
        embeddings,
        index=cell_ids,
    ).to_parquet(outfile)

    logging.info(
        f"Saved embeddings to {outfile}"
    )


# ============================================================
# Main batch loop
# ============================================================

def main():

    set_seed(42)

    symbol_to_ens = (
        build_symbol_to_ensembl_map(
            GTF_PATH
        )
    )

    for sample_dir in sorted(
        BASE_DIR.iterdir()
    ):

        if not sample_dir.is_dir():
            continue

        sample_name = sample_dir.name

        logging.info(
            f"Processing {sample_name}"
        )

        adata_path = (
            sample_dir / "adata.h5ad"
        )

        if not adata_path.exists():

            logging.warning(
                f"Skipping {sample_name}: "
                f"missing adata.h5ad"
            )

            continue

        output_dir = (
            sample_dir / "embeddings"
        )

        output_dir.mkdir(
            exist_ok=True
        )

        outfile = (
            output_dir
            / "nicheformer.parquet"
        )

        # ----------------------------------------------------
        # Skip completed
        # ----------------------------------------------------

        if outfile.exists():

            logging.info(
                f"Skipping {sample_name}: "
                f"nicheformer.parquet exists"
            )

            continue

        # ----------------------------------------------------
        # Run embedding
        # ----------------------------------------------------

        try:

            run_embed_nicheformer(
                dataset_path=adata_path,
                output_dir=output_dir,
                symbol_to_ens=symbol_to_ens,
            )

            logging.info(
                f"Finished {sample_name}"
            )

        except Exception as e:

            logging.exception(
                f"FAILED {sample_name}: {e}"
            )

    logging.info(
        "All samples processed"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()