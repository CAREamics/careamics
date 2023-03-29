import os
import torch
import logging
import itertools
import tifffile
import numpy as np

from functools import partial
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path
from skimage.util import view_as_windows
from typing import Callable, List, Optional, Sequence, Union, Tuple

from .n2v import n2v_manipulate


############################################
#   ETL pipeline #TODO add description to all modules
############################################


def open_input_source(path: Union[str, Path], num_files: Union[int, None] = None) -> List:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    path : Union[str, Path]
        _description_
    n_files : int
        _description_

    Returns
    -------
    List
        _description_
    """
    # Basic function to open input source
    #TODO add support for reading a subset of files 
    return itertools.islice(Path(path).rglob('*.tif*'), num_files) if num_files else (Path(path).rglob('*.tif*'))


def extract_patches_sequential(
    arr, patch_size, num_patches=None
) -> np.ndarray:  # TODO add support for shapes
    """Generate patches from ND array
    #TODO add time series, image or volume. Patches will be generated differently
    Crop an array into patches deterministically covering the whole array.

    Parameters
    ----------
    arr : np.ndarray
        Input array. Possible shapes are (C, Z, Y, X), (C, Y, X), (Z, Y, X) or (Y, X)
    patch_size : Tuple
        Patch dimensions for 'ZYX' or 'YX'
    num_patches : int or None (Currently not implemented)
        If None, function will calculate the overlap required to cover the whole array with patches. This might increase memory usage.
        If num_patches is less than calculated value, patches are taken from random locations with no guarantee of covering the whole array. (Currently not implemented)
        If num_patches is greater than calculated value, overlap is increased to produce required number of patches. Memory usage may be increased significantly.

    Returns
    -------
    patches : np.ndarray of shape (n_patches, C, Z, Y, X) or (n_patches, C, Y, X)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted given the overlap.
    """

    z_patch_size = None if len(patch_size) == 2 else patch_size[0]
    y_patch_size = patch_size[-2]
    x_patch_size = patch_size[-1]

    #TODO put asserts in separate function in init 
    # Asserts
    assert len(patch_size) == len(arr.shape[1:]), "Number of patch dimensions must match image dimensions"
    assert (
        z_patch_size is None or z_patch_size <= arr.shape[1]
    ), "Z patch size is incosistent with image shape"
    assert (
        y_patch_size <= arr.shape[-2] and x_patch_size <= arr.shape[-1]
    ), "At least one of XY patch dimensions is incosistent with image shape"

    # Calculate total number of patches for each dimension
    z_total_patches = (
        np.ceil(arr.shape[1] / z_patch_size) if z_patch_size is not None else None
    )
    y_total_patches = np.ceil(arr.shape[-2] / y_patch_size)
    x_total_patches = np.ceil(arr.shape[-1] / x_patch_size)

    # Calculate overlap for each dimension #TODO add more thorough explanation
    overlap_z = (
        np.ceil(
            (z_patch_size * z_total_patches - arr.shape[1])
            / max(1, z_total_patches - 1)
        ).astype(int)
        if z_patch_size is not None
        else None
    )
    overlap_y = np.ceil(
        (y_patch_size * y_total_patches - arr.shape[-2]) / max(1, y_total_patches - 1)
    ).astype(int)
    overlap_x = np.ceil(
        (x_patch_size * x_total_patches - arr.shape[-1]) / max(1, x_total_patches - 1)
    ).astype(int)

    if z_patch_size is not None:
        window_shape = (arr.shape[0], z_patch_size, y_patch_size, x_patch_size)
        step = (
            arr.shape[0],
            min(z_patch_size - overlap_z, z_patch_size),
            min(y_patch_size - overlap_y, y_patch_size),
            min(x_patch_size - overlap_x, x_patch_size),
        )
        output_shape = (-1, arr.shape[0], y_patch_size, x_patch_size) if z_patch_size == 1 \
        else (-1, arr.shape[0], z_patch_size, y_patch_size, x_patch_size)
    else:
        window_shape = (arr.shape[0], y_patch_size, x_patch_size)
        step = (
            arr.shape[0],
            min(y_patch_size - overlap_y, y_patch_size),
            min(x_patch_size - overlap_x, x_patch_size),
        )
        output_shape = (-1, arr.shape[0], y_patch_size, x_patch_size)
    # Generate a view of the input array containing pre-calculated number of patches in each dimension with overlap.
    # Resulting array is resized to (n_patches, C, Z, Y, X) or (n_patches,C, Y, X)
    # TODO add possibility to remove empty or almost empty patches ?
    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )

    # Yield single patch #TODO view_as_windows might be inefficient
    for patch_ixd in range(patches.shape[0]):
        yield patches[patch_ixd].astype(np.float32)


def extract_patches_random(arr, patch_size, num_patches=None) -> np.ndarray:

    # TODO either num_patches are all unique or output exact number of patches
    crop_coords = np.random.default_rng().integers(
        np.subtract(arr.shape, (0, *patch_size)), size=(num_patches, len(arr.shape))
    )

    # TODO add multiple arrays support, add possibility to remove empty or almost empty patches ?
    for i in range(crop_coords.shape[1]):
        yield arr[
            (
                ...,
                *[
                    slice(c, c + patch_size[j])
                    for j, c in enumerate(crop_coords[:, i, ...])
                ],
            )
        ].copy().astype(np.float32)


class PatchDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a list of images and apply transforms to the patches."""

    # TODO add napari style axes params, add asserts
    def __init__(
        self,
        data_path: str,
        num_files: int,
        data_reader: Callable,
        patch_size: Union[List[int], Tuple[int]],
        patch_generator: Union[np.ndarray, Callable],
        image_level_transform: Optional[Callable] = None,
        patch_level_transform: Optional[Callable] = None,
    ) -> None:
        """
        Parameters
        ----------
        data_path : str
            Path to data, must be a directory.
        data_reader : Callable
            Function that reads the image data from the file. Returns an iterable of image filenames.
        patch_size : Tuple[int]
            The size of the patch to extract from the image. Must be a tuple of len either 2 or 3 depending on number of spatial dimension in the data.
        patch_generator : Union[np.ndarray, Callable]
            Function that converts an input image (item from dataset) into a iterable of image patches.
            `patch_iter(dataset[idx])` must yield a tuple: (patches, coordinates).
        image_level_transform : Optional[Callable], optional
            _description_, by default None
        patch_level_transform : Optional[Callable], optional
            _description_, by default None
        """
        
        # Assert input data
        assert isinstance(
            data_path, str
        ), f"Incorrect data_path type. Must be a str, given{type(data_path)}"

        # Assert patch_size
        assert isinstance(
            patch_size, (list, tuple)
        ), f"Incorrect patch_size. Must be a tuple, given{type(patch_size)}"
        assert len(patch_size) in (
            2,
            3,
        ), f"Incorrect patch_size. Must be a 2 or 3, given{len(patch_size)}"

        self.data_reader = data_reader(data_path, num_files)
        self.source_iter = iter(self.data_reader)
        self.patch_size = patch_size
        self.patch_generator = patch_generator
        self.image_transform = image_level_transform
        self.patch_transform = patch_level_transform

        assert any(True for _ in self.data_reader), "Data source is empty"

    @staticmethod
    def read_data_source(self, data_source: str):
        """
        Read data source and correct dimensions

        Parameters
        ----------
        data_source : str
            Path to data source

        Returns
        -------
        image volume : np.ndarray
        """

        if not os.path.exists(data_source):
            raise ValueError(f"Data source {data_source} does not exist")

        arr = tifffile.imread(data_source)
        # Assert data dimensions are correct
        assert len(arr.shape) in (
            2,
            3,
            4,
        ), f"Incorrect data dimensions. Must be 2, 3 or 4, given {arr.shape} for file {data_source}"
        
        #TODO improve shape asserts
        # Adding channel dimension if necessary. If present, check correctness
        if len(arr.shape) == 2 or (len(arr.shape) == 3 and len(self.patch_size) == 3):
            arr = np.expand_dims(arr, axis=0)
            updated_patch_size = self.patch_size
        elif len(arr.shape) > 3 and len(self.patch_size) == 2:
            raise ValueError(
                f"Incorrect data dimensions {arr.shape} for given dimensionality {len(self.patch_size)}D in file {data_source}"
            )
        elif len(arr.shape) == 3 and len(self.patch_size) == 2 and arr.shape[0] > 4:
            logging.warning(f"Number of channels is {arr.shape[0]} for 2D data. Assuming time series.")
            arr = np.expand_dims(arr, axis=0)
            updated_patch_size = (1, *self.patch_size)
        return arr, updated_patch_size

    def __iter_source__(self):
        """
        Iterate over data source and yield whole image. Optional transform is applied to the images.

        Yields
        ------
        np.ndarray
        """
        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        id = info.id if info is not None else 0
        # TODO check for mem leaks, explicitly gc the arr after iterator is exhausted
        for i, filename in enumerate(self.data_reader):
            try:
                #TODO add buffer, several images up to some memory limit?
                arr, patch_size = self.read_data_source(self, filename)
            except (ValueError, FileNotFoundError, OSError) as e:
                logging.exception(f"Exception in file {filename}, skipping")
                raise e
            if i % num_workers == id:
                yield self.image_transform(
                    (arr, patch_size)
                ) if self.image_transform is not None else (arr, patch_size)

    def __iter__(self):
        """
        Iterate over data source and yield single patch. Optional transform is applied to the patches.

        Yields
        ------
        np.ndarray
        """
        for image, updated_patch_size in self.__iter_source__():
            for patch in self.patch_generator(image, updated_patch_size):
                # TODO add augmentations, multiple functions. 
                yield self.patch_transform(
                    patch
                ) if self.patch_transform is not None else patch
