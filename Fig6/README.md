# Fig6 Folder README

This folder contains three Jupyter notebooks for the NSCLC (non-small cell lung cancer) spatial transcriptomics analysis and prediction workflow. The recommended order to run the notebooks is:

1. **NSCLC-embed-SpatialFusion.ipynb**
2. **NSCLC-ABMIL-prediction.ipynb**
3. **Lung-Novartis-analysis.ipynb**

## Contents & Workflow

### 1. NSCLC-embed-SpatialFusion.ipynb
- **Purpose:** Generates cell-level spatial transcriptomics embeddings for NSCLC samples using the SpatialFusion pipeline.
- **Main Steps:**
  - Loads raw AnnData objects and metadata for NSCLC samples.
  - Runs the embedding model to produce cell-level latent representations.
  - Saves embeddings for downstream analysis and prediction.
- **Outputs:** Parquet files containing cell embeddings for each sample.

### 2. NSCLC-ABMIL-prediction.ipynb
- **Purpose:** Trains and evaluates an Attention-based Multiple Instance Learning (ABMIL) model to predict clinical outcomes (e.g., tumor stage) from cell-level spatial embeddings.
- **Main Steps:**
  - Loads cell embeddings and clinical metadata.
  - Aggregates cell embeddings into patient-level bags, focusing on malignant niches.
  - Implements and trains an ABMIL neural network with attention pooling and dropout.
  - Performs leave-one-out cross-validation (LOO-CV) and ensemble averaging for robust prediction.
  - Computes and visualizes metrics (AUC, balanced accuracy, F1), ROC curves, and confusion matrices.
  - Extracts and saves attention scores for each cell, enabling interpretation of model focus.
  - Performs enrichment analysis to identify clusters and cell types most associated with high attention.
- **Outputs:**
  - CSV files with ensemble predictions and per-cell attention scores.
  - Publication-quality figures (ROC, confusion matrix, volcano/enrichment plots, attention maps).

### 3. Lung-Novartis-analysis.ipynb
- **Purpose:** Performs downstream biological and clinical analysis of NSCLC prediction results and attention maps.
- **Main Steps:**
  - Loads ensemble predictions and cell-level attention scores.
  - Integrates attention with cell metadata, clusters, and spatial coordinates.
  - Visualizes attention maps and cluster enrichments across samples.
  - Performs statistical analysis (e.g., Fisher's exact test, FDR correction) to identify significant associations.
  - Generates publication-quality figures for reporting and interpretation.
- **Outputs:**
  - Figures and tables summarizing model interpretation and biological findings.

## How to Run

1. **Prepare Data:** Download all required data files (AnnData `.h5ad`, embeddings `.parquet`, clinical metadata, palettes, etc.) and place them at the paths referenced in the notebooks. Update file paths if needed.
2. **Install Dependencies:** Install required Python packages (`scanpy`, `torch`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, etc.). See the main project README for environment setup.
## Notes

- Outputs such as figures and CSVs are saved to the `results/figures_Fig6/` directory by default.
- You may need to adjust parameters, sample lists, or file paths for your own data.
- For troubleshooting, refer to notebook comments and the main project documentation.
