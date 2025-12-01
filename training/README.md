# SpatialFusion Training Folder

This folder contains all scripts, configuration files, and utilities for training and embedding models in the SpatialFusion pipeline. It is organized for reproducible, modular training and embedding of autoencoders (AE) and graph convolutional networks (GCN).

## Folder Structure

- **bash_scripts/**  
  Shell scripts for launching training and embedding jobs. These handle environment setup, GPU selection, logging, and call the main Python scripts with appropriate arguments.

- **conf/**  
  Hydra configuration files for all training, evaluation, and dataset settings.  
  - `conf/training/` — AE and GCN training configs  
  - `conf/eval/` — Evaluation and embedding configs  
  - `conf/dataset/` — Dataset sample lists and paths  
  - `conf/paths.yaml` — Root paths for logs, checkpoints, embeddings, and data  
  - `conf/pretrained.yaml` — Pretrained model paths

- **scripts/**  
  Main Python scripts for training and embedding:  
  - `train_multi_ae.py` — Train multi-modal AE  
  - `train_baseline_ae.py` — Train baseline AE  
  - `train_gcn_pw.py` — Train GCN on pathway embeddings  
  - `embed_AE.py` — Extract AE embeddings  
  - `embed_gcn.py` — Extract GCN embeddings

## Subfolder Details

### bash_scripts/
Contains ready-to-use shell scripts for launching jobs.  
- **AE Training:**  
  - `submit_full_train_ae.sh`, `submit_subset_train_ae.sh`, `submit_full_train_ae_onlyrecon.sh`, `submit_train_baseline_ae.sh`
- **GCN Training:**  
  - `submit_train_gcn_alllosses.sh`, `submit_train_gcn_alllosses_concat.sh`, `submit_train_gcn_alllosses_z1.sh`, `submit_train_gcn_alllosses_z2.sh`, `submit_train_gcn_nopathway.sh`, `submit_train_gcn_onlyrecon.sh`, etc.
- **AE Embedding:**  
  - `submit_embed_ae.sh`, `submit_embed_ae_test.sh`, `submit_embed_ae_subset.sh`, `submit_embed_ae_onlyrecon.sh`

Each script sets up the environment, selects the GPU, and runs the corresponding Python script with logging.

### conf/
Hydra configuration files for all aspects of training and evaluation.
- **training/** — AE and GCN training hyperparameters (epochs, batch size, loss weights, etc.)
- **eval/** — Embedding extraction configs (checkpoint paths, output directories)
- **dataset/** — Sample lists and data paths for train/test splits
- **paths.yaml** — Centralized paths for logs, checkpoints, embeddings, and data
- **pretrained.yaml** — Paths to pretrained AE and GCN models

### scripts/
Main Python scripts for model training and embedding. Below are detailed descriptions of each script and their required inputs:

- **train_multi_ae.py**
  - *Purpose*: Trains a multi-modal autoencoder (AE) using paired datasets of gene expression and spatial features.
  - *Inputs*: Uses Hydra config files for training parameters (`conf/training/training_ae.yaml`), dataset sample lists (`conf/dataset/dataset_full_hest.yaml`), and output paths. Accepts command-line overrides for epochs, batch size, device, etc.
  - *Workflow*: Loads and preprocesses multi-modal features for each sample, builds paired datasets, trains the AE, and saves the trained model and loss plots.

- **train_baseline_ae.py**
  - *Purpose*: Trains a baseline autoencoder model on basic features (gene expression and image only).
  - *Inputs*: Hydra config (`conf/training/training_baseline_ae.yaml`), dataset sample lists, and output paths. Accepts overrides for device, epochs, etc.
  - *Workflow*: Computes a soft union of genes present in at least 50% of samples, loads and preprocesses baseline features, trains the AE, and saves the model and gene list used.

- **train_gcn_pw.py**
  - *Purpose*: Trains a graph convolutional network (GCN) on pathway embeddings and spatial graphs.
  - *Inputs*: Hydra config (`conf/training/training_gcn.yaml`), dataset sample lists, AE embeddings, and output paths. Accepts combine mode (e.g., `concat`, `z1`, `z2`), device, and other overrides.
  - *Workflow*: Loads joint pathway embeddings and spatial coordinates, builds kNN graphs and subgraphs, optionally loads pathway activation labels, trains the GCN autoencoder, and saves the model and training loss plots.

- **embed_AE.py**
  - *Purpose*: Extracts and saves AE embeddings for all samples using a trained PairedAE model.
  - *Inputs*: Hydra config (`conf/eval/eval.yaml`), AE checkpoint path, sample lists, and output embedding directory. Accepts overrides for checkpoint and output paths.
  - *Workflow*: Infers input dimensions, loads model checkpoint and config, extracts embeddings for all samples, and saves them to disk.

- **embed_gcn.py**
  - *Purpose*: Extracts and saves GCN embeddings for all samples using a trained GCNAutoencoder model.
  - *Inputs*: Hydra config, AE embeddings (z1/z2), sample metadata, GCN checkpoint path, and output directory. Accepts combine mode and other overrides.
  - *Workflow*: Loads z1/z2 AE embeddings and metadata, combines embeddings, builds kNN graphs, loads trained GCN, extracts GCN embeddings, merges with cell type/spatial/ligand-receptor metadata, and saves to disk.

Each script is designed to be run with Hydra configuration management, allowing flexible overrides and reproducible experiments. For more details, see the script docstrings and the relevant YAML config files in `conf/`.

## Minimal Example Commands

Below are minimal commands for the three main workflows. All commands should be run from the `training/` directory.

### (a) Train the AE

```bash
# Train the multi-modal AE on the full HEST dataset (default GPU 0)
bash bash_scripts/submit_full_train_ae.sh 0
```
- Uses `train_multi_ae.py` with config `training=training_ae` and dataset `dataset_full_hest`.
- Logs and checkpoints are saved to the paths specified in `conf/paths.yaml`.

### (b) Train the GCN

```bash
# Train the GCN with all losses, using the full HEST dataset (default GPU 0)
bash bash_scripts/submit_train_gcn_alllosses.sh 0
```
- Uses `train_gcn_pw.py` with config `training=training_gcn` and dataset `dataset_full_hest`.
- You can select combine modes (e.g., `concat`, `z1`, `z2`) by using the corresponding script (e.g., `submit_train_gcn_alllosses_concat.sh`).

### (c) Embed the AE

```bash
# Extract AE embeddings using a trained AE checkpoint (default GPU 0)
bash bash_scripts/submit_embed_ae.sh 0
```
- Uses `embed_AE.py` to extract and save AE embeddings for all samples.
- Output embeddings are saved to the directory specified in the config (see `conf/eval/eval.yaml`).

## Notes

- All scripts accept additional Hydra overrides after the GPU argument, e.g.:
  ```bash
  bash bash_scripts/submit_full_train_ae.sh 0 training.epochs=50
  ```
- Environment variables (e.g., `SPATIALFUSION_ROOT`, `LOGS`) can be set to customize output locations.
- For more details on arguments and configuration, see the comments in each script and the YAML config files in `conf/`.

---

This README provides a comprehensive overview of the training folder, its organization, and how to run the main workflows. For more details or documentation of additional scripts, see the script comments and config files.
