"""Convenience methods for datasets."""
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import tifffile


def list_files(data_path: Union[str, Path], data_format: str) -> List[Path]:
    """
    Return a list of path to files in a directory.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the data.
    data_format : str
        Extension of the files to load, without period, e.g. `tif`.

    Returns
    -------
    List[Path]
        List of pathlib.Path objects.
    """
    files = sorted(Path(data_path).rglob(f"*.{data_format}*"))
    return files


def _update_axes(array: np.ndarray, axes: str) -> np.ndarray:
    """
    Update axes of the sample to match the config axes.

    This method concatenate the S and T axes.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    axes : str
        Description of axes in format STCZYX.

    Returns
    -------
    np.ndarray
        Updated array.
    """
    # concatenate ST axes to N, return NCZYX
    if ("S" in axes or "T" in axes) and array.dtype != "O":
        new_axes_len = len(axes.replace("Z", "").replace("YX", ""))
        # TODO test reshape as it can scramble data, moveaxis is probably better
        array = array.reshape(-1, *array.shape[new_axes_len:]).astype(np.float32)

    elif array.dtype == "O":
        for i in range(len(array)):
            array[i] = np.expand_dims(array[i], axis=0).astype(np.float32)

    else:
        array = np.expand_dims(array, axis=0).astype(np.float32)

    return array


def read_tiff(file_path: Path, axes: str) -> np.ndarray:
    """
    Read a tiff file and return a numpy array.

    Parameters
    ----------
    file_path : Path
        Path to a file.
    axes : str
        Description of axes in format STCZYX.

    Returns
    -------
    np.ndarray
        Resulting array.

    Raises
    ------
    ValueError
        If the file failed to open.
    OSError
        If the file failed to open.
    ValueError
        If the file is not a valid tiff.
    ValueError
        If the data dimensions are incorrect.
    ValueError
        If the axes length is incorrect.
    """
    if file_path.suffix[:4] == ".tif":
        try:
            sample = tifffile.imread(file_path)
        except (ValueError, OSError) as e:
            logging.exception(f"Exception in file {file_path}: {e}, skipping it.")
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
    sample = _update_axes(sample, axes)

    return sample
