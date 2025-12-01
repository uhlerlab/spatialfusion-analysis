# SpatialFusion-Analysis

SpatialFusion-Analysis contains all scripts, configuration files, and Jupyter notebooks used for training, benchmarking, and downstream analyses within the SpatialFusion pipeline. It provides a fully reproducible framework for multimodal spatial transcriptomics analysis, including model training, embedding extraction, benchmarking, and figure generation for manuscripts.

If you are looking for the **SpatialFusion package** to run the framework on your own data, please visit the main SpatialFusion repository instead: https://github.com/uhlerlab/spatialfusion. 

---

## Repository Structure

### **benchmarks/**

Workflows for benchmarking model performance and comparing approaches.
Includes scripts, notebooks, and a detailed README with environment and usage instructions.

### **training/**

Resources for training AE/GCN models and extracting embeddings.
Contains:

* `bash_scripts/` — Shell scripts to launch training and embedding jobs
* `conf/` — Hydra configuration files for training, evaluation, and datasets
* `scripts/` — Python scripts for model training and embedding generation

See `training/README.md` for full documentation, example commands, and workflow details.

### **datasets/**

Scripts for data preparation and preprocessing, including sample lists for all supported datasets.
Run these workflows **before** any training, benchmarking, or figure generation to ensure data is correctly formatted.

### **Fig1/ ... Fig6/**

Notebooks and scripts used to generate the figures in the SpatialFusion manuscript.
Each folder includes a README describing inputs, outputs, and the analysis workflow for that figure.

---

## Environment Setup

Most workflows use the **`spatialfusion-env`** conda environment:

```bash
conda env create -f spatialfusion_env.yml
conda activate spatialfusion_env
```

This environment has been validated on a compute cluster using NVIDIA A6000 GPUs with CUDA 12.1. Depending on your local hardware and CUDA setup, you may need to adjust package versions accordingly.

Some scripts-particularly those requiring **bin2cell**—use a separate **`bin2cell-env`** environment.
Refer to the README in the corresponding figure directory (e.g., `datasets/README.md`) for installation instructions.

Benchmarking workflows may require additional dependencies. See `benchmarks/README.md` for the full environment specification.

---

## Usage Guide

1. **Set up the environment**

   * Use `spatialfusion-env` for most training, embedding, and analysis workflows
   * Use `bin2cell-env` for specific figure notebooks that depend on bin2cell

2. **Prepare your data**

   * Follow the data organization and preprocessing steps in `datasets/`
   * Update configuration files as needed for your dataset

3. **Train models**

   * Run the AE/GCN training scripts in `training/bash_scripts/`
   * Consult `training/README.md` for example commands and detailed explanations

4. **Extract embeddings**

   * Use the embedding generation scripts in `training/bash_scripts/`

5. **Benchmark and visualize**

   * Use the workflows in `benchmarks/` to evaluate models
   * Reproduce manuscript figures or perform custom visualizations using the `Fig*/` notebooks

---

## Documentation

* Each subdirectory includes its own detailed README
* Training workflows: see `training/README.md`
* Benchmarking workflows: see `benchmarks/README.md`
* Figure reproduction workflows: see each `Fig*/README.md`

---

## Citation

If you use SpatialFusion-Analysis in your work, please cite the corresponding SpatialFusion manuscript (citation details to be added).

---

## License

SpatialFusion-Analysis is released under the **MIT License**.
See the `LICENSE` file for full details.

---

For questions, bug reports, or contributions, please open an issue or submit a pull request on GitHub.
