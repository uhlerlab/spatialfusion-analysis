# Benchmarks Folder README

This folder contains example scripts for running spatial transcriptomics embedding benchmarks using several state-of-the-art methods. The scripts are provided as templates to help you run analyses on your own data. **You must download the required datasets and update the file paths in the scripts before running them.**

Please follow instructions in the original repositories to install the different packages to be able to run this analysis. For convenience, we also provide the .yml files of the environments these methods were run with. 

## Contents

- `run-embed-banksy.sh` / `embed-banksy.py`: Run the BANKSY embedding workflow on several example datasets. BANKSY integrates spatial and gene expression information to produce cell embeddings. Arguments control input data, output location, sample ID, regularization, PCA dimension, and spatial coordinate key.
- `run-embed-nichecompass.sh` / `embed-nichecompass.py`: Run the NicheCompass pipeline, which uses gene programs and spatial omics data to generate cell embeddings. Arguments specify input AnnData, gene annotation/program folders, spatial omics folder, output directory, species, dataset name, spatial key, and number of neighbors.
- `run-embed-omiclip.sh` / `embed-omiclip.py`: Run the OmiCLIP pipeline, which combines gene expression and image data (H&E/WSI) for cell embedding. Arguments specify model/data directories, input AnnData, housekeeping gene set, output directory, WSI image path, spatial key, and device (CPU/GPU).
- `run-embed-scgpt-spatial.sh` / `embed-scgpt-spatial.py`: Run the scGPT-spatial pipeline, which uses a transformer-based model for spatial transcriptomics. Arguments specify input AnnData, output directory, model directory, batch size, max sequence length, and optional vocabulary filtering.
- `run-embed-nicheformer.sh` / `embed-nicheformer.py`: Run the Nicheformer pipeline, which uses deep learning to embed spatial transcriptomics data. Arguments specify input AnnData, model checkpoint, vocabulary, technology mean file, output directory, modality, species, technology, and model parameters.

## Instructions to Run

1. **Download the required data**: The scripts reference datasets (e.g., `test_data/10X_Xenium_Ovarian_5k/adata.h5ad`). You must obtain these datasets and place them in the correct locations, or update the paths in the scripts to match your data locations.
2. **Update paths**: Edit the bash scripts to ensure all file and directory paths point to your local copies of the data and models.
3. **Run the scripts**: You can execute the bash scripts from the command line. For example:
   ```bash
   bash run-embed-banksy.sh
   bash run-embed-nichecompass.sh
   bash run-embed-omiclip.sh
   bash run-embed-scgpt-spatial.sh
   bash run-embed-nicheformer.sh
   ```
   Ensure you have the required Python environment and dependencies installed (see the main project README for setup instructions).

## Description of Arguments in Bash Scripts

### BANKSY (`embed-banksy.py`)
- `--adata_path`: Path to the input AnnData `.h5ad` file.
- `--output_dir`: Directory for output embeddings/results.
- `--sample_id`: Sample identifier.
- `--lambda_param`: Regularization parameter (spatial/gene expression balance).
- `--pca_dim`: Number of principal components for dimensionality reduction.
- `--spatial_key`: Key in AnnData `.obsm` for spatial coordinates.

### NicheCompass (`embed-nichecompass.py`)
- `--adata_path`: Path to input AnnData `.h5ad` file.
- `--ga_data_folder`: Gene annotation data folder.
- `--gp_data_folder`: Gene program data folder.
- `--so_data_folder`: Spatial omics data folder.
- `--artifacts_folder`: Folder for model artifacts.
- `--embedding_output_dir`: Output directory for embeddings.
- `--species`: Species name (e.g., human).
- `--dataset`: Dataset name.
- `--spatial_key`: Key for spatial coordinates.
- `--n_neighbors`: Number of spatial neighbors.

### OmiCLIP (`embed-omiclip.py`)
- `--data_dir`: Directory with model/support files.
- `--adata_path`: Path to input AnnData `.h5ad` file.
- `--housekeeping_gmt`: Path to housekeeping genes `.gmt` file.
- `--output_dir`: Output directory for embeddings.
- `--wsi_path`: Path to H&E/WSI image file.
- `--spatial_key`: Key for spatial coordinates.
- `--device`: Device to use (cuda/cpu).

### scGPT-spatial (`embed-scgpt-spatial.py`)
- `--adata_path`: Path to input AnnData `.h5ad` file.
- `--output_dir`: Output directory for embeddings.
- `--model_dir`: Directory with model files.
- `--use_vocab_filter`: Filter cells with zero gene overlap before embedding.
- `--batch_size`: Batch size for embedding.
- `--max_seq_len`: Maximum sequence length.

### Nicheformer (`embed-nicheformer.py`)
- `--dataset_path`: Path to input AnnData `.h5ad` file.
- `--model_path`: Path to model checkpoint.
- `--vocab_path`: Path to vocabulary AnnData file.
- `--tech_mean_path`: Path to Nicheformer technology mean file.
- `--output_dir`: Output directory for embeddings.
- `--modality`: Data modality (e.g., spatial).
- `--species`: Species name.
- `--technology`: Technology/platform name.
- `--max_seq_len`, `--batch_size`, `--chunk_size`, `--num_workers`: Model parameters.

## Notes

- The provided bash scripts are **examples**. You may need to modify them to suit your data and analysis needs.
- Ensure all dependencies for the Python scripts are installed and available in your environment.
- If you encounter errors, check that all paths are correct and that the input files exist.
- For each method, refer to the main project documentation for further details and troubleshooting.
