"""Convenience methods for datasets."""
import logging
from pathlib import Path
from typing import Callable, List, Tuple, Union

import albumentations as Aug
import numpy as np
import tifffile
import zarr

from ..config.transform import ALL_TRANSFORMS
from ..utils.logging import get_logger

logger = get_logger(__name__)


def approximate_file_size(filename: Path) -> int:
    """
    Approximate file size.

    Parameters
    ----------
    filename : Path
        Path to a file.

    Returns
    -------
    int
        Approximate file size in mbytes.
    """
    try:
        pointer = tifffile.TiffFile(filename)
        return pointer.filehandle.size / 1024 ** 2
    except (tifffile.TiffFileError, StopIteration, FileNotFoundError):
        logger.warning(f"File {filename} is not a valid tiff file or is empty.")
        return 0


def get_file_sizes(files: List[Path]) -> List[int]:
    """
    Get file sizes.

    Parameters
    ----------
    files : List[Path]
        List of paths to files.

    Returns
    -------
    List[int]
        List of file sizes in mbytes.
    """
    return sum([approximate_file_size(file) for file in files])


def list_files(
    data_path: Union[str, Path, List[Union[str, Path]]],
    data_format: str,
    return_list: bool = True,
) -> Tuple[List[Path], int]:
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
    int
        Approximate size of the files in mbytes.
    """
    data_path = Path(data_path) if not isinstance(data_path, list) else data_path

    if isinstance(data_path, list):
        files = []
        for path in data_path:
            files.append(list_files(path, data_format, return_list=False))
        if len(files) == 0:
            raise ValueError(f"Data path {data_path} is empty.")
        approx_size = get_file_sizes(files)
        return files, approx_size

    elif data_path.is_dir():
        if return_list:
            files = sorted(Path(data_path).rglob(f"*.{data_format}*"))
            if len(files) == 0:
                raise ValueError(f"Data path {data_path} is empty.")
            approx_size = get_file_sizes(files)
            return files, approx_size
        else:
            files = sorted(Path(data_path).rglob(f"*.{data_format}*"))[0]
        return files

    elif data_path.is_file():
        approx_size = approximate_file_size(data_path)
        return [data_path] if return_list else data_path, approx_size

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


def get_shape_order(shape_in: Tuple, axes_in: str, ref_axes: str = "STCZYX"):
    """Return the new shape and axes of x, ordered according to the reference axes.

    Parameters
    ----------
    shape_in : Tuple
        Input shape.
    ref_axes : str
        Reference axes.
    axes_in : str
        Input axes.

    Returns
    -------
    Tuple
        New shape.
    str
        New axes.
    Tuple
        Indices of axes in the new axes order.
    """
    indices = [axes_in.find(k) for k in ref_axes]

    # remove all non-existing axes (index == -1)
    indices = tuple(filter(lambda k: k != -1, indices))

    # find axes order and get new shape
    new_axes = [axes_in[ind] for ind in indices]
    new_shape = tuple([shape_in[ind] for ind in indices])

    return new_shape, "".join(new_axes), indices


def list_diff(list1: List, list2: List) -> List:
    """Return the difference between two lists.

    Parameters
    ----------
    list1 : List
        First list.
    list2 : List
        Second list.

    Returns
    -------
    List
        Difference between the two lists.
    """
    return list(set(list1) - set(list2))


def reshape_data(x: np.ndarray, axes: str):
    """Reshape the data to 'SZYXC' or 'SYXC', merging 'S' and 'T' channels if necessary.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axes : str
        Description of axes in format STCZYX.

    Returns
    -------
    np.ndarray
        Reshaped array.
    str
        New axes string.
    """
    _x = x
    _axes = axes

    # sanity checks
    if len(_axes) != len(_x.shape):
        raise ValueError(f"Incompatible data ({_x.shape}) and axes ({_axes}).")

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x.shape, _axes)

    # if S is not in the list of axes, then add a singleton S
    if "S" not in new_axes:
        new_axes = "S" + new_axes
        _x = _x[np.newaxis, ...]
        new_x_shape = (1,) + new_x_shape

        # need to change the array of indices
        indices = [0] + [1 + i for i in indices]

    # reshape by moving axes
    destination = list(range(len(indices)))
    _x = np.moveaxis(_x, indices, destination)

    # remove T if necessary
    if "T" in new_axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_axes = new_axes.replace("T", "")

        # reshape S and T together
        _x = _x.reshape(new_x_shape)

    # add channel
    if "C" not in new_axes:
        # Add channel axis after S
        _x = np.expand_dims(_x, new_axes.index("S") + 1)
        #get the location of the 1st spatial axis
        c_coord = len(new_axes.replace("Z", "").replace("YX", ""))
        new_axes = new_axes[:c_coord] + "C" + new_axes[c_coord:]

    return _x, new_axes


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

    if len(array.shape) < 2 or len(array.shape) > 6:
        raise ValueError(
            f"Incorrect data dimensions. Must be 2, 3 or 4 (got {array.shape} for"
            f"file {file_path})."
        )

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
    patch_transforms: List, target: bool, normalize_mask: bool = True
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
    if patch_transforms is None:
        return Aug.NoOp()
    elif isinstance(patch_transforms, list):
        # TODO not very readable
        return Aug.Compose(
            [
                ALL_TRANSFORMS[transform["name"]](**transform["parameters"])
                if "parameters" in transform
                else ALL_TRANSFORMS[transform["name"]]()
                for transform in patch_transforms
            ],
            additional_targets={"target": "image"}
            if (target and normalize_mask)
            else {},
        )
    else:
        raise ValueError(
            f"Incorrect patch transform type {patch_transforms}. "
            f"Please refer to the documentation."  # TODO add link to documentation
        )
