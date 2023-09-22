import logging
from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np
import tifffile

from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.tiling import (
    extract_patches_random,
    extract_patches_sequential,
    extract_tiles,
)


def list_files(data_path, data_format) -> List[Path]:
    """Creates a list of paths to source tiff files from path string.

    Parameters
    ----------
    data_path : str
        path to the folder containing the data
    data_format : str
        data format, e.g. tif

    Returns
    -------
    List[Path]
        List of pathlib.Path objects
    """
    files = sorted(Path(data_path).rglob(f"*.{data_format}*"))
    return files


def fix_axes(sample: np.ndarray, axes: str) -> np.ndarray:
    """Fixes axes of the sample to match the config axes.

    Parameters
    ----------
    sample : np.ndarray
        array containing the image

    Returns
    -------
    np.ndarray
        reshaped array
    """
    # concatenate ST axes to N, return NCZYX
    if ("S" in axes or "T" in axes) and sample.dtype != "O":
        new_axes_len = len(axes.replace("Z", "").replace("YX", ""))
        # TODO test reshape, replace with moveaxis ?
        sample = sample.reshape(-1, *sample.shape[new_axes_len:]).astype(np.float32)

    elif sample.dtype == "O":
        for i in range(len(sample)):
            sample[i] = np.expand_dims(sample[i], axis=0).astype(np.float32)

    else:
        sample = np.expand_dims(sample, axis=0).astype(np.float32)

    return sample


def read_tiff(file_path: Path, axes: str) -> np.ndarray:
    """Reads a file and returns a numpy array.

    Parameters
    ----------
    file_path : Path
        pathlib.Path object containing a path to a file

    Returns
    -------
    np.ndarray
        array containing the image

    Raises
    ------
    ValueError, OSError
        if a file is not a valid tiff or damaged
    ValueError
        if data dimensions are not 2, 3 or 4
    ValueError
        if axes parameter from config is not consistent with data dimensions
    """
    if file_path.suffix[:4] == ".tif":
        try:
            sample = tifffile.imread(file_path)
        except (ValueError, OSError) as e:
            logging.exception(f"Exception in file {file_path}: {e}, skipping")
            raise e
    else:
        raise ValueError(f"File {file_path} is not a valid tiff.")

    sample = sample.squeeze()

    if len(sample.shape) < 2 or len(sample.shape) > 4:
        raise ValueError(
            f"Incorrect data dimensions. Must be 2, 3 or 4 (got {sample.shape} for"
            f"file {file_path})."
        )

    # check number of axes
    if len(axes) != len(sample.shape):
        raise ValueError(f"Incorrect axes length (got {axes} for file {file_path}).")
    sample = fix_axes(sample, axes)
    return sample


def generate_patches(
    sample: np.ndarray,
    patch_extraction_method: str,
    patch_size: Union[List[int], Tuple[int]],
    patch_overlap: Union[List[int], Tuple[int]],
) -> Generator[np.ndarray, None, None]:
    """Generate patches from a sample.

    Parameters
    ----------
    sample : np.ndarray
        array containing the image

    Yields
    ------
    Generator[np.ndarray, None, None]
        Generator function yielding patches/tiles

    Raises
    ------
    ValueError
        if no patches are generated
    """
    patches = None
    assert patch_size is not None, "Patch size must be provided"

    if patch_extraction_method == ExtractionStrategies.TILED:
        assert patch_overlap is not None, "Patch overlap must be provided"
        patches = extract_tiles(
            arr=sample, tile_size=patch_size, overlaps=patch_overlap
        )

    elif patch_extraction_method == ExtractionStrategies.SEQUENTIAL:
        patches = extract_patches_sequential(sample, patch_size=patch_size)

    elif patch_extraction_method == ExtractionStrategies.RANDOM:
        patches = extract_patches_random(sample, patch_size=patch_size)

    if patches is None:
        raise ValueError("No patches generated")

    return patches
