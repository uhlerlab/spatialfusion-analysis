import os
import pathlib as pl
import logging
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
import timm
import torch

from PIL import Image
from tqdm import tqdm

from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# ============================================================
# Paths
# ============================================================

# Folder containing sample directories with adata.h5ad
ADATA_BASE_DIR = pl.Path(
    "../../../Broad_SpatialFoundation/hest_processed_data/"
)

# Folder containing WSI .tif files
WSI_DIR = pl.Path(
    "../../../Broad_SpatialFoundation/hest_data/wsis"
)

# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Logging setup
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = ADATA_BASE_DIR / f"virchow2_batch_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logging.info("Starting Virchow2 batch embedding job")
logging.info(f"Using device: {device}")
logging.info(f"Log file: {log_file}")

# ============================================================
# Load Virchow2 model ONCE
# ============================================================

logging.info("Loading Virchow2 model...")

model = timm.create_model(
    "hf-hub:paige-ai/Virchow2",
    pretrained=True,
    mlp_layer=SwiGLUPacked,
    act_layer=torch.nn.SiLU,
)

model.eval().to(device)

# Official Virchow2 transforms
transform = create_transform(
    **resolve_data_config(model.pretrained_cfg, model=model)
)

logging.info("Virchow2 model loaded")

# ============================================================
# Embedding function
# ============================================================

def load_wsi(path):
    try:
        logging.info("Trying tifffile.imread()")
        wsi = tifffile.imread(path)
        logging.info(f"Loaded via imread: shape={wsi.shape}")
        return wsi

    except Exception as e:
        logging.warning(
            f"tifffile.imread failed ({type(e).__name__}: {e}); "
            "falling back to TiffFile.pages[0]"
        )

        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            wsi = page.asarray()

        logging.info(f"Loaded via page[0]: shape={wsi.shape}")
        return wsi


def embed_virchow2(
    wsi,
    cell_names,
    he_coords,
    output_file,
    batch_size=128,
):

    embeddings = []
    cell_ids = []

    batch_imgs = []
    batch_ids = []

    logging.info(
        f"Embedding {len(he_coords)} image patches "
        f"in batches of {batch_size}"
    )

    for cid, (x, y) in tqdm(
        zip(cell_names, he_coords),
        total=len(cell_names)
    ):

        x, y = int(x), int(y)

        # ====================================================
        # Extract centered 256x256 patch
        # ====================================================

        x0, x1 = x - 128, x + 128
        y0, y1 = y - 128, y + 128

        pad_x0 = max(0, -x0)
        pad_x1 = max(0, x1 - wsi.shape[1])

        pad_y0 = max(0, -y0)
        pad_y1 = max(0, y1 - wsi.shape[0])

        patch = np.pad(
            wsi[
                max(0, y0):min(wsi.shape[0], y1),
                max(0, x0):min(wsi.shape[1], x1)
            ],
            ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
            mode="constant"
        )

        # Skip malformed patches
        if patch.shape[:2] != (256, 256):
            continue

        # Convert to PIL image
        patch_pil = Image.fromarray(patch)

        # Apply official transforms
        tensor_img = transform(patch_pil)

        batch_imgs.append(tensor_img)
        batch_ids.append(cid)

        # ====================================================
        # Run batch inference
        # ====================================================

        if len(batch_imgs) == batch_size:

            img_tensor = torch.stack(batch_imgs).to(device)

            with torch.inference_mode(), torch.autocast(
                device_type=device,
                dtype=torch.float16
            ):

                # Forward pass
                output = model(img_tensor)

                # Virchow2 token extraction
                class_token = output[:, 0]
                patch_tokens = output[:, 5:]

                # Final 2560-dim embedding
                batch_embs = torch.cat(
                    [
                        class_token,
                        patch_tokens.mean(1)
                    ],
                    dim=-1
                )

                batch_embs = batch_embs.cpu().numpy()

            embeddings.extend(batch_embs)
            cell_ids.extend(batch_ids)

            batch_imgs.clear()
            batch_ids.clear()

    # ========================================================
    # Final partial batch
    # ========================================================

    if batch_imgs:

        img_tensor = torch.stack(batch_imgs).to(device)

        with torch.inference_mode(), torch.autocast(
            device_type=device,
            dtype=torch.float16
        ):

            output = model(img_tensor)

            class_token = output[:, 0]
            patch_tokens = output[:, 5:]

            batch_embs = torch.cat(
                [
                    class_token,
                    patch_tokens.mean(1)
                ],
                dim=-1
            )

            batch_embs = batch_embs.cpu().numpy()

        embeddings.extend(batch_embs)
        cell_ids.extend(batch_ids)

    # ========================================================
    # Save embeddings
    # ========================================================

    df = pd.DataFrame(
        embeddings,
        index=cell_ids,
    )

    df.to_parquet(output_file)

    logging.info(
        f"Saved {len(df)} embeddings to {output_file}"
    )


# ============================================================
# Process all samples
# ============================================================

for sample_dir in sorted(ADATA_BASE_DIR.iterdir()):

    # Skip non-directories
    if not sample_dir.is_dir():
        continue

    sample_name = sample_dir.name

    logging.info(f"Processing {sample_name}")

    # ========================================================
    # Input paths
    # ========================================================

    adata_path = sample_dir / "adata.h5ad"

    wsi_path = WSI_DIR / f"{sample_name}.tif"

    # ========================================================
    # Validate inputs
    # ========================================================

    if not adata_path.exists():

        logging.warning(
            f"Skipping {sample_name}: missing adata.h5ad"
        )

        continue

    if not wsi_path.exists():

        logging.warning(
            f"Skipping {sample_name}: missing WSI {wsi_path.name}"
        )

        continue

    # ========================================================
    # Output paths
    # ========================================================

    output_dir = sample_dir / "embeddings"

    output_dir.mkdir(exist_ok=True)

    outfile = output_dir / "Virchow2.parquet"

    # ========================================================
    # Skip completed samples
    # ========================================================

    if outfile.exists():

        logging.info(
            f"Skipping {sample_name}: "
            f"Virchow2.parquet already exists"
        )

        continue

    # ========================================================
    # Run embedding
    # ========================================================

    try:

        # ----------------------------------------------------
        # Load AnnData
        # ----------------------------------------------------

        logging.info(f"Loading adata: {adata_path}")

        adata = ad.read_h5ad(adata_path)

        # ----------------------------------------------------
        # Cell IDs
        # ----------------------------------------------------

        if "cell_id" in adata.obs:

            cell_names = (
                adata.obs["cell_id"]
                .astype(str)
                .values
            )

        else:

            cell_names = (
                adata.obs_names
                .astype(str)
            )

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        he_coords = adata.obsm['spatial_he']

        # ----------------------------------------------------
        # Load WSI
        # ----------------------------------------------------

        logging.info(f"Loading WSI: {wsi_path}")

        wsi = load_wsi(wsi_path)

        logging.info(f"WSI shape: {wsi.shape}")

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        embed_virchow2(
            wsi=wsi,
            cell_names=cell_names,
            he_coords=he_coords,
            output_file=outfile,
            batch_size=128,
        )

        logging.info(f"Finished {sample_name}")

    except Exception as e:

        logging.exception(
            f"FAILED processing {sample_name}: {e}"
        )

logging.info("All samples processed")
