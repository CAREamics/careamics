import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import tifffile
import zarr

from careamics.config.training import ExtractionStrategies
from careamics.dataset.tiling import (
    extract_patches_random,
    extract_patches_sequential,
    extract_tiles,
)


def list_files(
    data_path: Union[str, Path, List[Union[str, Path]]],
    data_format: str,
    return_list=True,
) -> List[Path]:
    """Creates a list of paths to source tiff files from path string.

    Parameters
    ----------
    data_path : str
        path to the folder containing the data
    data_format : str
        data format, e.g. tif
    return_list : bool, optional
        Whether to return a list of paths or str, by default True

    Returns
    -------
    List[Path]
        List of pathlib.Path objects
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
        # TODO This doesn't work for ZARR !
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


def read_zarr(
    file_path: Path, axes: str
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
    zarr_source = zarr.open(Path(file_path), mode="r")
    if isinstance(zarr_source, zarr.hierarchy.Group):
        raise NotImplementedError("Group not supported yet")

    elif isinstance(zarr_source, zarr.storage.DirectoryStore):
        raise NotImplementedError("DirectoryStore not supported yet")

    elif isinstance(zarr_source, zarr.core.Array):
        # array should be of shape (S, (C), (Z), Y, X), iterating over S ?
        # TODO what if array is not of that shape and/or chunks aren't defined and
        if zarr_source.dtype == "O":
            raise NotImplementedError("Object type not supported yet")
        else:
            arr = zarr_source
    else:
        raise ValueError(f"Unsupported zarr object type {type(zarr_source)}")

    # TODO how to fix dimensions? Or just raise error?
    # sanity check on dimensions
    if len(arr.shape) < 2 or len(arr.shape) > 4:
        raise ValueError(
            f"Incorrect data dimensions. Must be 2, 3 or 4 (got {arr.shape})."
        )

    # sanity check on axes length
    if len(axes) != len(arr.shape):
        raise ValueError(f"Incorrect axes length (got {axes}).")

    # FIXME !
    # arr = fix_axes(arr, axes)
    return arr


def generate_patches(
    sample: np.ndarray,
    patch_extraction_method: ExtractionStrategies,
    patch_size: Optional[Union[List[int], Tuple[int]]] = None,
    patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
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
        if no patch has been generated
    """
    patches = None

    if patch_size is not None:
        patches = None

        if patch_extraction_method == ExtractionStrategies.TILED:
            if patch_overlap is None:
                raise ValueError(
                    "Overlaps must be specified when using tiling (got None)."
                )
            patches = extract_tiles(
                arr=sample, tile_size=patch_size, overlaps=patch_overlap
            )

        elif patch_extraction_method == ExtractionStrategies.SEQUENTIAL:
            patches = extract_patches_sequential(sample, patch_size=patch_size)

        elif patch_extraction_method == ExtractionStrategies.RANDOM:
            patches = extract_patches_random(sample, patch_size=patch_size)

        if patches is None:
            raise ValueError("No patch generated")

        return patches
    else:
        # no patching
        return (sample for _ in range(1))
