import spatialdata
import os
import h5py
import json
import warnings

import pandas as pd
import numpy as np
import pathlib as pl
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import tifffile as tff

from spatialdata_io import xenium, xenium_aligned_image
from spatialdata.models import TableModel, Image2DModel
from spatialdata.transformations import set_transformation, Identity
from scipy.stats import median_abs_deviation
from PIL import Image
import xarray as xr
from typing import Optional, Tuple, List, Dict, Union
from tqdm import tqdm

import logging

# Define logger
logger = logging.getLogger(__name__)


def pretty_ax(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=False,
        labelbottom=True,
        labelleft=True,
    )
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)


def download_data(
    datapath: pl.Path,
    aligned_hande: bool,
    hande_name: str,
    cell_id_col: str = "cell_id",
) -> spatialdata._core.spatialdata.SpatialData:
    adata = xenium(datapath)

    if aligned_hande:
        he_path = None
        alignment_matrix_path = None
        for f in (datapath / hande_name).iterdir():
            if f.suffix == ".tif":
                he_path = f
                image_name = f.stem.split(".")[0]
            elif f.suffix == ".csv":
                alignment_matrix_path = f
        assert (
            he_path is not None
        ), "No H&E image found, please provide it. H&E image must be a .tif (or ome.tif) file."

        if alignment_matrix_path is None:
            warnings.warn(
                "Warning: No alignment file provided. Assuming the H&E and IF images are already aligned."
            )

        image = xenium_aligned_image(he_path, alignment_matrix_path)

        adata.images[image_name] = image

    adata["table"].obs = adata["table"].obs.set_index(cell_id_col)

    # where the pixel size in µm is saved
    experiment_path = str(datapath / "experiment.xenium")
    with open(experiment_path) as f:
        dict_exp = json.load(f)
    # This is the size of a pixel in Xenium, used to convert coordinates
    _PIXEL_SIZE_XENIUM = dict_exp["pixel_size"]
    # Why do we need this? The output of Xenium is in µm, whereas the coordinates in the H&E and morphology image are in pixels
    # We thus save the coordinates of the cells in the morphology in pixels
    # This will be useful to use the transformation matrix to obtain the coordinates in pixels in the H&E
    # description of Xenium output: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs
    adata["table"].obsm["spatial_px"] = (
        adata["table"].obsm["spatial"] / _PIXEL_SIZE_XENIUM
    )

    return adata


def filter_lowquality(
    adata: spatialdata._core.spatialdata.SpatialData, total_counts_col: str, nmad: int
):
    median = adata["table"].obs[total_counts_col].quantile(0.5)
    mad = median_abs_deviation(adata["table"].obs[total_counts_col])

    logger.info(
        f"Remove cells with less than {max(0,median-nmad*mad)} transcripts detected."
    )

    selected_cells = adata["table"].copy()
    selected_cells = selected_cells[
        selected_cells.obs[total_counts_col] >= max(0, median - nmad * mad)
    ].copy()

    return selected_cells


def add_cell_types(celltype_file: pl.Path, selected_cells: sc.AnnData) -> sc.AnnData:
    celltypes = pd.read_csv(celltype_file, index_col=0)
    celltypes.index = celltypes.index.astype(str)

    annotated_cells = selected_cells.obs_names.intersection(celltypes.index)

    celltypes = celltypes.loc[annotated_cells]
    selected_cells = selected_cells[annotated_cells].copy()

    selected_cells.obs = pd.concat([selected_cells.obs, celltypes], axis=1)
    return selected_cells


def _transform_x(aff_transf: pd.DataFrame, coords: np.ndarray) -> np.ndarray:
    """Why do we need this? The H&E image is not naturally aligned to the Xenium output. This can be done through the
    Xenium
    """

    inv_transf = np.linalg.inv(aff_transf.values)
    transformed_coords = (inv_transf @ np.vstack((coords.T, np.ones(len(coords))))).T[
        :, :-1
    ]

    return transformed_coords


def create_cell_patches(
    selected_cells: sc.AnnData,
    datapath: pl.Path,
    hande_name: str,
    n_tile: int,
    hande_image_name: Optional[pl.Path] = None,
) -> Tuple[List, List]:

    alignment_matrix_path = None
    if hande_image_name is None:

        for f in (datapath / hande_name).iterdir():
            if f.suffix in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                hande_image_name = f
            elif f.suffix == ".csv":
                alignment_matrix_path = f
        assert (
            hande_image_name is not None
        ), "No H&E image found. H&E image must be a .tif (or ome.tif), .png, .jpg, or .jpeg file."

        if alignment_matrix_path is None:
            warnings.warn(
                "Warning: No alignment file provided. Assuming the H&E and IF images are already aligned."
            )

    # Load image
    if hande_image_name.suffix == ".tif":
        image = tff.imread(hande_image_name)
    else:
        image = Image.open(hande_image_name)
        img_rgb = image.convert("RGB")
        image = np.array(img_rgb)

    if alignment_matrix_path is not None:
        aff_transf = pd.read_csv(alignment_matrix_path, header=None)
    else:
        aff_transf = pd.DataFrame(np.identity(3))

    coords = selected_cells.obsm["spatial_px"]
    cell_names = selected_cells.obs_names.to_numpy()

    transformed_coords = _transform_x(aff_transf=aff_transf, coords=coords)
    # Clip at 0 bc sometimes the transformation bugs a little bit, this should be minor though
    # (ex: 1 of 150,000 cells had this in a dataset I am evaluating)
    print(
        f"There are {((transformed_coords<0).sum(axis=1)>0).sum()} cells with negative coordinates, clipping at 0."
    )
    transformed_coords = transformed_coords.clip(0)

    # get the boundaries
    max_x = image.shape[1]
    max_y = image.shape[0]

    # there might some rotation going on here, but given all methods are rotation invariant shoudln't change anything
    all_tiles, labels = [], []
    for i, cell in tqdm(enumerate(transformed_coords)):
        x, y = cell
        tile = image[
            max(0, int(y - n_tile / 2)) : min(max_y, int(y + n_tile / 2)),
            max(0, int(x - n_tile / 2)) : min(max_x, int(x + n_tile / 2)),
        ]
        # pad the image if not the right shape
        if tile.shape != (n_tile, n_tile, 3):
            nrows, ncols, _ = tile.shape
            xpad = ((n_tile - nrows) // 2, (n_tile - nrows) // 2 + (n_tile - nrows) % 2)
            ypad = ((n_tile - ncols) // 2, (n_tile - ncols) // 2 + (n_tile - ncols) % 2)
            tile = np.pad(tile, [xpad, ypad, (0, 0)], "constant")

        all_tiles.append(tile)
        labels.append(cell_names[i])
    return all_tiles, labels


# from https://realpython.com/storing-images-in-python/
def store_many_hdf5(images: List, labels: List, savedir: pl.Path, name: str):
    """Stores an array of images to HDF5.
    Parameters:
    ---------------
    images       images array, (N, n_patch, n_patch, 3) to be stored
    labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    os.makedirs(savedir / "patches", exist_ok=True)

    # Create a new HDF5 file
    file = h5py.File(savedir / "patches" / f"{name}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset("meta", np.shape(labels), data=labels)
    file.close()


def save_transcriptomics(
    selected_cells: sc.AnnData, savedir: pl.Path, name: str
) -> None:
    selected_cells.obs["cell_id"] = selected_cells.obs_names
    selected_cells.write(savedir / f"{name}.h5ad")


def prepare_data(
    datapath: pl.Path,
    aligned_hande: bool,
    hande_name: str,
    total_counts_col: str,
    nmad: int,
    celltype_file: Optional[pl.Path],
    savedir_base: pl.Path,
    n_tile: int,
    minor_celltype_str: str = "minor_celltype",
    major_celltype_str: str = "major_celltype",
    coord_type: str = "generic",
    cell_id_col: str = "cell_id",
) -> None:

    savedir = savedir_base / datapath.stem
    os.makedirs(savedir, exist_ok=True)

    logger.info("Downloading data.")
    adata = download_data(
        datapath=datapath,
        aligned_hande=aligned_hande,
        hande_name=hande_name,
        cell_id_col=cell_id_col,
    )

    logger.info("Filtering low quality cells.")
    adata = filter_lowquality(adata=adata, total_counts_col=total_counts_col, nmad=nmad)

    if celltype_file is not None:
        logger.info("Adding cell type information.")
        adata = add_cell_types(celltype_file=celltype_file, selected_cells=adata)

    logger.info("Saving transcriptomic information.")
    save_transcriptomics(
        selected_cells=adata,
        savedir=savedir,
        name=datapath.stem,
    )

    logger.info("Creating cell image patches.")

    all_tiles, labels = create_cell_patches(
        selected_cells=adata,
        datapath=datapath,
        hande_name=hande_name,
        n_tile=n_tile,
    )

    logger.info("Saving cell image patches.")

    store_many_hdf5(
        images=all_tiles, labels=labels, savedir=savedir, name=datapath.stem
    )