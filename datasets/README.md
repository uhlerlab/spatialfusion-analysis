# SpatialFusion Datasets: Preprocessing and Analysis

This folder contains scripts and notebooks for preprocessing and analyzing spatial transcriptomics datasets for the SpatialFusion project. The workflow is designed to process multiple datasets, extract cell type information, and compute pathway activation scores.

Note: run-preprocess-CRC-bin2cell.sh will require setting up the bin2cell environment.

## Workflow Order

Run the following files in order:

1. **run-download-preprocess-hest1k.sh**  
   Downloads and preprocesses the HEST1k dataset.

2. **Preprocess-celltype-HEST1k.ipynb**  
   Extracts cell type information from the preprocessed HEST1k data.

3. **run-STACK-hest.sh**  
   Generates STACK RNA embeddings for all HEST1k samples (required for Fig1 embedding benchmark).

4. **run-embed-Virchow2.sh**  
   Generates Virchow2 image patch embeddings for all HEST1k samples.

5. **run-nicheformer-hest.sh**  
   Generates Nicheformer RNA embeddings for all HEST1k samples.

6. **run-preprocess-CRC-bin2cell.sh**  
   Preprocesses colorectal cancer (CRC) data, converting bin-level data to cell-level data.

7. **run-preprocess-CRC-embed.sh**  
   Embeds CRC data for downstream analysis.

8. **Preprocess-celltype-CRC.ipynb**  
   Extracts cell type information from the CRC dataset.

9. **run-preprocess-nsclc.sh**  
   Preprocesses non-small cell lung cancer (NSCLC) data.

10. **Preprocess-celltype-Lung-Novartis.ipynb**  
    Extracts cell types from the Novartis lung dataset.

11. **run-preprocess-nsclc-CCI.sh**  
    Computes cell-cell interaction (CCI) ligand-receptor scores for the NSCLC Visium HD dataset.

12. **run-decoupler.sh**  
    Computes pathway activation scores using the decoupler framework.

## File Descriptions

- **download_filter.py**: Python utility module containing helper functions for downloading and filtering Xenium spatial data using `spatialdata` and `spatialdata-io`. Provides functions for loading raw Xenium outputs, aligning H&E images, and applying quality-control filters. Used as a dependency by other preprocessing scripts.
- **run-download-preprocess-hest1k.sh**: Shell script to download and preprocess the HEST1k dataset.
- **Preprocess-celltype-HEST1k.ipynb**: Jupyter notebook for cell type extraction from HEST1k data.
- **embed-STACK.py**: Python script that batch-generates STACK RNA embeddings for all samples in the HEST processed data directory. Loads the STACK model once, iterates over sample folders, and saves per-sample embeddings as parquet files.
- **run-STACK-hest.sh**: Shell script launcher for `embed-STACK.py`.
- **embed-Virchow2.py**: Python script that batch-extracts Virchow2 pathology foundation model image patch embeddings for all HEST1k samples, transforming spatial coordinates to align with whole-slide images and saving results as parquet files.
- **run-embed-Virchow2.sh**: Shell script launcher for `embed-Virchow2.py`.
- **embed-nicheformer.py**: Python script that batch-generates Nicheformer RNA embeddings for all samples in the HEST processed data directory, handling gene vocabulary alignment and chunked inference before saving embeddings as parquet files.
- **run-nicheformer-hest.sh**: Shell script launcher for `embed-nicheformer.py`.
- **run-preprocess-CRC-bin2cell.sh**: Shell script to convert CRC bin-level data to cell-level data.
- **run-preprocess-CRC-embed.sh**: Shell script to embed CRC data for further analysis.
- **Preprocess-celltype-CRC.ipynb**: Jupyter notebook for extracting cell types from CRC data.
- **run-preprocess-nsclc.sh**: Shell script to preprocess NSCLC data.
- **Preprocess-celltype-Lung-Novartis.ipynb**: Jupyter notebook for extracting cell types from the Novartis lung dataset.
- **preprocess-nsclc-CCI.py**: Python script that computes cell-cell interaction (CCI) ligand-receptor scores for the NSCLC Visium HD cohort using CellChat LR pairs. Normalizes expression, builds spatial KNN neighbor graphs, and computes smoothed LR scores for each sample, saving results as parquet files.
- **run-preprocess-nsclc-CCI.sh**: Shell script launcher for `preprocess-nsclc-CCI.py`.
- **run-decoupler.sh**: Shell script to compute pathway activation scores using the decoupler method.
- **decoupler.py**: Python script containing core functions for pathway activation analysis.

## Important Notes

- **Path Configuration**: Many scripts and especially the Jupyter notebooks require you to set or update file paths to match your local environment. Please review and adjust paths as needed before running each step.
- **Dependencies**: Ensure all required Python packages are installed (see code comments for typical requirements such as `scanpy`, `decoupler`, `pandas`, etc.).
- **Execution**: Shell scripts can be run from the command line (e.g., `bash run-download-preprocess-hest1k.sh`). Jupyter notebooks should be run interactively, making sure to update any hardcoded paths.

For further details on each step, refer to the comments and documentation within each script or notebook.
