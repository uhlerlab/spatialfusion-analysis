# Segmentation Comparison Folder README

This folder contains notebooks for evaluating how alternative cell segmentation (Cellpose) affects SpatialFusion embedding quality and spatial domain benchmarking on the Xenium ovarian cancer dataset, as a supplementary analysis in the SpatialFusion project.

## Contents & Workflow

The recommended order to run the notebooks is:

1. **cellpose.ipynb**
2. **rerun-embeddings.ipynb**
3. **rerun-benchmark.ipynb**

### 1. cellpose.ipynb
- **Purpose:** Performs cell segmentation on the 10x Xenium Ovarian 5k morphology image using Cellpose and assembles a gene-by-cell count matrix.
- **Main Steps:**
  - Tiles the large OME-TIFF image with overlap and runs the Cellpose model on each tile.
  - Stitches tile masks back into a global label image with seam reconciliation via union-find.
  - Assigns Xenium transcripts (from `transcripts.zarr.zip`) to segmented cells by coordinate lookup.
  - Filters and saves the resulting AnnData object.
- **Outputs:** `xenium_cellpose.h5ad` — a filtered AnnData file with cell spatial coordinates and a sparse count matrix.

### 2. rerun-embeddings.ipynb
- **Purpose:** Generates all required single-modality and fused embeddings for the Cellpose-segmented Xenium ovarian cancer cells.
- **Main Steps:**
  - Runs scGPT on the Cellpose count matrix.
  - Extracts UNI2 and Virchow2 image patch embeddings by transforming Xenium pixel coordinates to H&E space using the 10X alignment matrix.
  - Generates Nicheformer RNA embeddings.
  - Runs the SpatialFusion autoencoder+GCN pipeline for all four encoder combinations (UNI+scGPT, UNI+Nicheformer, Virchow+scGPT, Virchow+Nicheformer).
- **Outputs:** Parquet files for each embedding type, used as input to the benchmark notebook.

### 3. rerun-benchmark.ipynb
- **Purpose:** Runs the spatial domain benchmark on the Cellpose-segmented dataset and compares against the default 10X segmentation results.
- **Main Steps:**
  - Loads the Cellpose-derived AnnData and assigns pathologist region annotations via geospatial join.
  - Loads pre-computed SpatialFusion embeddings and competitor embeddings (NicheCompass, Nicheformer, BANKSY).
  - Runs Leiden clustering at matched resolutions and computes SDMBench metrics (ARI, NMI, HOM, COM, PAS, CHAOS).
- **Outputs:** `CELLPOSE_OVCA_benchmark.svg` — a benchmark heatmap comparing performance across methods under Cellpose segmentation.

## How to Run

1. **Prepare Data and Models**: Ensure all required data files (Xenium OME-TIFF, `transcripts.zarr.zip`, H&E image, alignment matrix, model checkpoints) are downloaded and available at the paths referenced in the notebooks. Update file paths as needed.
2. **Install Dependencies**: The `cellpose.ipynb` notebook requires the Cellpose package in addition to the standard environment. Install the required Python packages (`scanpy`, `torch`, `cellpose`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, etc.). See the main project README for environment setup instructions.

## Notes

- This analysis is designed to evaluate segmentation robustness and complements the main OVCA benchmark in `Fig2/`.
- Outputs such as plots and CSVs are saved to the `results/figures_segmentation/` directory by default.
- For troubleshooting, refer to the comments in each notebook and the main project documentation.
