# Fig4 Folder README

This folder contains analysis notebooks for benchmarking prostate cancer (PRAD) and lung adenocarcinoma Visium HD (LUAD) spatial transcriptomics embeddings, as used in Figure 4 of the SpatialFusion project.

## Contents

- `Benchmark-PRAD.ipynb`: Jupyter notebook for benchmarking and evaluating spatial transcriptomics methods on prostate cancer dataset. The notebook loads precomputed embeddings and metadata, runs performance metrics, and generates figures for comparison across methods.
- `Benchmark-PRAD_afterreview.ipynb`: Updated benchmark notebook for a Visium prostate adenocarcinoma (PRAD) sample from the HEST dataset (TENX157), applying the same SpatialFusion vs. competitor benchmarking pipeline. Loads pathologist region annotations from a GeoJSON, generates multiple SpatialFusion embeddings, runs Leiden clustering, computes SDMBench metrics, and additionally produces cross-tabulation heatmaps showing correspondence between discovered clusters and annotated tissue regions and cell type compositions.
- `Benchmark-LungHD.ipynb`: Jupyter notebook for benchmarking and evaluating spatial transcriptomics methods on lung cancer dataset. The notebook loads precomputed embeddings and metadata, runs performance metrics, and generates figures for comparison across methods.
- `Benchmark-LungHD_afterreview.ipynb`: Updated benchmark notebook for a Visium HD lung adenocarcinoma (LUAD) FFPE dataset, following the same benchmarking pattern as the BRCA notebook. Assigns pathologist region annotations from a QuPath GeoJSON via geospatial join, generates SpatialFusion embeddings for four encoder combinations plus H&E-only and finetuned variants, evaluates against 11 competing methods with SDMBench metrics, and outputs `afterreview_LUAD_HD_benchmark.svg` and a spatial cluster panel PNG.

## How to Run

1. **Prepare Data and Models**: Ensure all required data files (e.g., `.h5ad`, `.parquet`, model checkpoints) referenced in the notebook are downloaded and available at the specified paths. You may need to update file paths in the notebook to match your local setup.
2. **Install Dependencies**: Install the required Python packages (e.g., `scanpy`, `torch`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, etc.) in your environment. See the main project README for environment setup instructions.

## Notes

- The notebook is designed for reproducibility and publication-quality figure generation. You may need to adjust parameters or file paths for your own data.
- Outputs such as plots and CSVs are saved to the `results/figures_Fig4/` directory by default.
- For troubleshooting, refer to the comments in the notebook and the main project documentation.
