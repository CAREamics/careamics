"""Convenience methods for datasets."""
import logging
from pathlib import Path
from typing import Callable, List, Tuple, Union

import albumentations as A
import numpy as np
import tifffile
import zarr

from ..config.transforms import ALL_TRANSFORMS
from ..utils.logging import get_logger

logger = get_logger(__name__)


def list_files(
    data_path: Union[str, Path, List[Union[str, Path]]],
    data_format: str,
    return_list: bool = True,
) -> List[Path]:
    """Creates a list of paths to source tiff files from path string.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the data.
    data_format : str
        data format, e.g. tif
    return_list : bool, optional
        Whether to return a list of paths or str, by default True

    Returns
    -------
    List[Path]
        List of pathlib.Path objects.
    """
    data_path = Path(data_path) if not isinstance(data_path, list) else data_path

    if isinstance(data_path, list):
        files = []
        for path in data_path:
            files.append(list_files(path, data_format, return_list=False))
        if len(files) == 0:
            raise ValueError(f"Data path {data_path} is empty.")
        return files

    elif data_path.is_dir():
        if return_list:
            files = sorted(Path(data_path).rglob(f"*.{data_format}*"))
            if len(files) == 0:
                raise ValueError(f"Data path {data_path} is empty.")
        else:
            files = sorted(Path(data_path).rglob(f"*.{data_format}*"))[0]
        return files

    elif data_path.is_file():
        return [data_path] if return_list else data_path

    else:
        raise ValueError(
            f"Data path {data_path} is not a valid directory or a list of filenames."
        )


def _update_axes(array: np.ndarray, axes: str) -> np.ndarray:
    """
    Update axes of the array to match the config axes.

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
        array = array.reshape(-1, *array.shape[new_axes_len:]).astype(np.float32)
        array.reshape(-1, *array.shape[new_axes_len:]).astype(np.float32)

    elif "C" in axes:
        if len(axes) != len(array.shape):
            array = np.expand_dims(array, axis=0)
        if axes[-1] == "C":
            array = np.moveaxis(array, -1, 0)
        else:
            array = array.astype(np.float32)

    elif array.dtype == "O":
        for i in range(len(array)):
            array[i] = np.expand_dims(array[i], axis=0).astype(np.float32)

    else:
        array = np.expand_dims(array, axis=0).astype(np.float32)

    return array


def validate_files(train_files: List[Path], target_files: List[Path]) -> None:
    """
    Validate that the train and target folders are consistent.

    Parameters
    ----------
    train_files : List[Path]
        List of paths to train files.
    target_files : List[Path]
        List of paths to target files.

    Raises
    ------
    ValueError
        If the number of files in train and target folders is not the same.
    """
    if len(train_files) != len(target_files):
        raise ValueError(
            f"Number of train files ({len(train_files)}) is not equal to the number of"
            f"target files ({len(target_files)})."
        )
    if {f.name for f in train_files} != {f.name for f in target_files}:
        raise ValueError("Some filenames in Train and target folders are not the same.")


def expand_dims(
    arr: Union[np.ndarray, Tuple[np.ndarray]]
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Expand the dimensions of each array in the input.

    Parameters
    ----------
    arr : Union[np.ndarray, Tuple[np.ndarray]]
        Array to expand.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray]]
        Expanded array.
    """
    if isinstance(arr, np.ndarray):
        return np.expand_dims(arr, axis=0)
    elif isinstance(arr, tuple):
        return tuple(np.expand_dims(a, axis=0) for a in arr)
    else:
        raise ValueError(f"Unsupported type {type(arr)}.")


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
            array = tifffile.imread(file_path)
        except (ValueError, OSError) as e:
            logging.exception(f"Exception in file {file_path}: {e}, skipping it.")
            raise e
    else:
        raise ValueError(f"File {file_path} is not a valid tiff.")

    array = array.squeeze()

    if len(array.shape) < 2 or len(array.shape) > 4:
        raise ValueError(
            f"Incorrect data dimensions. Must be 2, 3 or 4 (got {array.shape} for"
            f"file {file_path})."
        )

    # check number of axes
    # if len(axes) != len(array.shape):
    #     raise ValueError(f"Incorrect axes length (got {axes} for file {file_path}).")
    array = _update_axes(array, axes)

    return array


def read_zarr(
    zarr_source: zarr.Group, axes: str
) -> Union[zarr.core.Array, zarr.storage.DirectoryStore, zarr.hierarchy.Group]:
    """Reads a file and returns a pointer.

    Parameters
    ----------
    file_path : Path
        pathlib.Path object containing a path to a file

    Returns
    -------
    np.ndarray
        Pointer to zarr storage

    Raises
    ------
    ValueError, OSError
        if a file is not a valid tiff or damaged
    ValueError
        if data dimensions are not 2, 3 or 4
    ValueError
        if axes parameter from config is not consistent with data dimensions
    """
    if isinstance(zarr_source, zarr.hierarchy.Group):
        array = zarr_source[0]

    elif isinstance(zarr_source, zarr.storage.DirectoryStore):
        raise NotImplementedError("DirectoryStore not supported yet")

    elif isinstance(zarr_source, zarr.core.Array):
        # array should be of shape (S, (C), (Z), Y, X), iterating over S ?
        if zarr_source.dtype == "O":
            raise NotImplementedError("Object type not supported yet")
        else:
            array = zarr_source
    else:
        raise ValueError(f"Unsupported zarr object type {type(zarr_source)}")

    # sanity check on dimensions
    if len(array.shape) < 2 or len(array.shape) > 4:
        raise ValueError(
            f"Incorrect data dimensions. Must be 2, 3 or 4 (got {array.shape})."
        )

    # sanity check on axes length
    if len(axes) != len(array.shape):
        raise ValueError(f"Incorrect axes length (got {axes}).")

    # arr = fix_axes(arr, axes)
    return array


def get_patch_transform(
    patch_transform: str, normalize_mask: bool, target: bool
) -> Union[None, Callable]:
    """Return a pixel manipulation function.

    Used in N2V family of algorithms.

    Parameters
    ----------
    patch_transform_type : str
        Type of patch transform.
    target : bool
        Whether the transform is applied to the target(if the target is present).

    Returns
    -------
    Union[None, Callable]
        Patch transform function.
    """
    if patch_transform is None:
        return A.NoOp()
    elif isinstance(patch_transform, dict):
        return A.Compose(
            [
                ALL_TRANSFORMS[transform](**parameters)
                if parameters
                else ALL_TRANSFORMS[transform]()
                for transform, parameters in patch_transform.items()
            ],
            additional_targets={"target": "image"}
            if (target and normalize_mask)
            else {},
        )
    else:
        raise ValueError(
            f"Incorrect patch transform type {patch_transform}. "
            f"Please refer to the documentation."  # TODO add link to documentation
        )
