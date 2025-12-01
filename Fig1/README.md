# Fig1 Folder README

This folder contains analysis notebooks for generating and comparing spatial transcriptomics embeddings, as used in Figure 1 of the SpatialFusion project.

## Contents

- `compare-baseline-ae.ipynb`: Jupyter notebook for extracting, evaluating, and visualizing embeddings from baseline autoencoder (AE) models and multimodal AEs. It includes code to load trained models, extract cell embeddings, compare classification and batch mixing metrics, and generate publication-quality plots.
- `compare-pathway-gcn.ipynb`: Jupyter notebook for comparing GCN-based embeddings with and without pathway information. It loads precomputed embeddings, runs PCA, visualizes pathway activation, and quantifies organization and performance using metrics such as Moran's I and cross-validated R².

## How to Run

1. **Prepare Data and Models**: Ensure all required data files (e.g., `.h5ad`, `.parquet`, model checkpoints) referenced in the notebooks are downloaded and available at the specified paths. You may need to update file paths in the notebooks to match your local setup.
2. **Install Dependencies**: Install the required Python packages (e.g., `scanpy`, `torch`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, etc.) in your environment. See the main project README for environment setup instructions.

## Notes

- The notebooks are designed for reproducibility and publication-quality figure generation. You may need to adjust parameters or file paths for your own data.
- Outputs such as plots and CSVs are saved to the `results/figures_Fig1/` directory by default.
- For troubleshooting, refer to the comments in each notebook and the main project documentation.
