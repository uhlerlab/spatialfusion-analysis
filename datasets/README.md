# SpatialFusion Datasets: Preprocessing and Analysis

This folder contains scripts and notebooks for preprocessing and analyzing spatial transcriptomics datasets for the SpatialFusion project. The workflow is designed to process multiple datasets, extract cell type information, and compute pathway activation scores.

Note: run-preprocess-CRC-bin2cell.sh will require setting up the bin2cell environment.

## Workflow Order

Run the following files in order:

1. **run-download-preprocess-hest1k.sh**  
   Downloads and preprocesses the HEST1k dataset.

2. **Preprocess-celltype-HEST1k.ipynb**  
   Extracts cell type information from the preprocessed HEST1k data.

3. **run-preprocess-CRC-bin2cell.sh**  
   Preprocesses colorectal cancer (CRC) data, converting bin-level data to cell-level data.

4. **run-preprocess-CRC-embed.sh**  
   Embeds CRC data for downstream analysis.

5. **Preprocess-celltype-CRC.ipynb**  
   Extracts cell type information from the CRC dataset.

6. **run-preprocess-nsclc.sh**  
   Preprocesses non-small cell lung cancer (NSCLC) data.

7. **Preprocess-celltype-Lung-Novartis.ipynb**  
   Extracts cell type information from the Novartis lung dataset.

8. **run-decoupler.sh**  
   Computes pathway activation scores using the decoupler framework.

## File Descriptions

- **run-download-preprocess-hest1k.sh**: Shell script to download and preprocess the HEST1k dataset.
- **Preprocess-celltype-HEST1k.ipynb**: Jupyter notebook for cell type extraction from HEST1k data.
- **run-preprocess-CRC-bin2cell.sh**: Shell script to convert CRC bin-level data to cell-level data.
- **run-preprocess-CRC-embed.sh**: Shell script to embed CRC data for further analysis.
- **Preprocess-celltype-CRC.ipynb**: Jupyter notebook for extracting cell types from CRC data.
- **run-preprocess-nsclc.sh**: Shell script to preprocess NSCLC data.
- **Preprocess-celltype-Lung-Novartis.ipynb**: Jupyter notebook for extracting cell types from the Novartis lung dataset.
- **run-decoupler.sh**: Shell script to compute pathway activation scores using the decoupler method.
- **decoupler.py**: Python script containing core functions for pathway activation analysis.

## Important Notes

- **Path Configuration**: Many scripts and especially the Jupyter notebooks require you to set or update file paths to match your local environment. Please review and adjust paths as needed before running each step.
- **Dependencies**: Ensure all required Python packages are installed (see code comments for typical requirements such as `scanpy`, `decoupler`, `pandas`, etc.).
- **Execution**: Shell scripts can be run from the command line (e.g., `bash run-download-preprocess-hest1k.sh`). Jupyter notebooks should be run interactively, making sure to update any hardcoded paths.

For further details on each step, refer to the comments and documentation within each script or notebook.
