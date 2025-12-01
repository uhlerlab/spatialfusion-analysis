# Fig5 Folder README

This folder contains the `CRC-analysis.ipynb` notebook for advanced spatial transcriptomics analysis of colorectal cancer (CRC) samples, as used in Figure 5 of the SpatialFusion project.

## Contents

- `CRC-analysis.ipynb`: Jupyter notebook for comprehensive CRC spatial analysis. The workflow includes:
  - Loading and harmonizing multiple CRC and normal colon samples, including AnnData objects and precomputed embeddings.
  - Cell type and cell subtype annotation, cluster grouping, and palette generation for consistent visualization.
  - High-resolution spatial visualization of cell types, subtypes, and clusters across samples, including publication-quality scatterplots and legends.
  - Quantification and visualization of niche (cluster) proportions and cell type composition using stacked barplots and heatmaps.
  - Pathway activation analysis: loading pathway scores, summarizing and plotting pathway activity per cluster and per sample, including statistical significance testing.
  - Integration and overlay of spatial data with H&E images, including non-convex niche outlines and zoomed-in regions of interest.
  - UMAP embedding and batch correction for selected cell populations (e.g., epithelial, lymphoid), with multi-panel visualizations.
  - Differential gene expression analysis between niches, dotplots grouped by gene function, and removal of contaminants.
  - Alluvial plots to visualize cell subtype transitions between niches.

## How to Run

1. **Prepare Data and Models**: Download all required data files (e.g., `.h5ad`, `.parquet`, H&E images, pathway scores) and place them at the paths referenced in the notebook. Update file paths if needed.
2. **Install Dependencies**: Install required Python packages (`scanpy`, `torch`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tifffile`, etc.). See the main project README for environment setup.

## Notes

- The notebook generates publication-quality figures and CSVs in `results/figures_Fig5/`.
- You may need to adjust parameters, sample lists, or file paths for your own data.
- For troubleshooting, refer to notebook comments and the main project documentation.
