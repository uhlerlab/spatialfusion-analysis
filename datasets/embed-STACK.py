import pathlib as pl
import logging
from datetime import datetime

import pandas as pd
import anndata as ad

from stack.model_loading import load_model_from_checkpoint

# ============================================================
# Paths
# ============================================================

# Root directory containing all sample folders
BASE_DIR = pl.Path(
    "../../../Broad_SpatialFoundation/hest_processed_data/"
)

# STACK model resources
MODEL_CHECKPOINT = "STACK-model/bc_large.ckpt"
GENELIST_PATH = "STACK-model/basecount_1000per_15000max.pkl"

# ============================================================
# Load model once
# ============================================================

print("Loading STACK model...")
model = load_model_from_checkpoint(MODEL_CHECKPOINT)
print("Model loaded.")

# ============================================================
# Logging setup
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = BASE_DIR / f"stack_batch_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logging.info("Starting STACK batch embedding job")
logging.info(f"Log file: {log_file}")

# ============================================================
# Process all samples
# ============================================================

for sample_dir in sorted(BASE_DIR.iterdir()):

    # Skip non-directories
    if not sample_dir.is_dir():
        continue

    sample_name = sample_dir.name

    adata_path = sample_dir / "adata.h5ad"

    # Skip folders without adata.h5ad
    if not adata_path.exists():
        logging.warning(f"Skipping {sample_name}: no adata.h5ad found")
        continue

    # ========================================================
    # Output paths
    # ========================================================

    output_dir = sample_dir / "embeddings"
    output_dir.mkdir(exist_ok=True)

    outfile = output_dir / "STACK.parquet"

    # ========================================================
    # Skip completed samples
    # ========================================================

    if outfile.exists():
        logging.info(
            f"Skipping {sample_name}: STACK.parquet already exists"
        )
        continue

    # ========================================================
    # Run embedding
    # ========================================================

    logging.info(f"Processing {sample_name}")

    try:

        # Load adata just to get cell IDs
        adata = ad.read_h5ad(adata_path)

        if "cell_id" in adata.obs:
            cell_ids = adata.obs["cell_id"]
        else:
            cell_ids = adata.obs_names.astype(str)

        # Get embeddings
        embeddings, _ = model.get_latent_representation(
            adata_path=str(adata_path),
            genelist_path=GENELIST_PATH,
            gene_name_col="feature_name",
            batch_size=16,
            num_workers=4,
        )

        # Save embeddings
        pd.DataFrame(
            embeddings,
            index=cell_ids,
        ).to_parquet(outfile)

        logging.info(f"Finished {sample_name}")

    except Exception as e:
        logging.exception(f"FAILED processing {sample_name}: {e}")

logging.info("All samples processed")
