# Fig2 Folder README

This folder contains analysis notebooks for benchmarking ovarian cancer (OVCA) spatial transcriptomics embeddings, as used in Figure 2 of the SpatialFusion project.

## Contents

- `Benchmark-OVCA.ipynb`: Jupyter notebook for benchmarking and evaluating spatial transcriptomics methods on ovarian cancer dataset. The notebook loads precomputed embeddings and metadata, runs performance metrics, and generates figures for comparison across methods.
- `Benchmark-OVCA-afterreview.ipynb`: Updated benchmark notebook for the Xenium ovarian cancer dataset, comparing SpatialFusion model variants (joint UNI+scGPT, concat, H&E-only, RNA-only) against competing methods including NicheCompass, BANKSY, Nicheformer, scGPT-spatial, OmiCLIP, Novae, and Scanpy. Computes SDMBench metrics (ARI, NMI, HOM, COM, PAS, CHAOS) against pathologist-annotated regions and outputs a benchmark heatmap SVG and spatial cluster panel plots.
- `ablation-comparison.ipynb`: Jupyter notebook performing a model architecture ablation study on the Xenium ovarian cancer dataset. It generates Leiden clusterings for a large grid of SpatialFusion architectural configurations — varying graph type (GCN vs. spatial smoothing), H&E encoder (UNI vs. Virchow), RNA encoder (scGPT vs. Nicheformer), reconstruction mode (full vs. recon-only), fusion mode (gated vs. average vs. concat), and use of the pathway auxiliary regression loss — across three target cluster counts (k=9, 11, 13). SDMBench metrics are computed for each configuration and the effect of each design choice is summarized using paired effect tables and bar charts showing mean paired Δ in bio-score, ARI, NMI, HOM, and COM.
- `analyze-pathway-ablation.ipynb`: Jupyter notebook that loads the per-pathway ablation embeddings and evaluates how much each removed pathway affects spatial clustering quality on the ovarian cancer Xenium sample. Produces a heatmap and bar-chart summary quantifying each pathway's contribution, with outputs including `benchmark_ovarian_pathway_ablation.csv` and `pathway_ablation.svg`.

## How to Run

1. **Prepare Data and Models**: Ensure all required data files (e.g., `.h5ad`, `.parquet`, model checkpoints) referenced in the notebook are downloaded and available at the specified paths. You may need to update file paths in the notebook to match your local setup.
2. **Install Dependencies**: Install the required Python packages (e.g., `scanpy`, `torch`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, etc.) in your environment. See the main project README for environment setup instructions.

## Notes

- The notebook is designed for reproducibility and publication-quality figure generation. You may need to adjust parameters or file paths for your own data.
- Outputs such as plots and CSVs are saved to the `results/figures_Fig2/` directory by default.
- For troubleshooting, refer to the comments in the notebook and the main project documentation.
