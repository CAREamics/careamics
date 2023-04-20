import itertools
import logging
import os
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import tifffile
import torch
from skimage.util import view_as_windows

from .dataloader_utils import (
    compute_overlap,
    compute_reshaped_view,
    compute_patch_steps,
)

############################################
#   ETL pipeline #TODO add description to all modules
############################################


def list_input_source_tiff(
    path: Union[str, Path], num_files: Union[int, None] = None
) -> List:
    """_summary_.

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
    # TODO add support for reading a subset of files
    return (
        list(itertools.islice(Path(path).rglob("*.tif*"), num_files))
        if num_files
        else list(Path(path).rglob("*.tif*"))
    )
    # return ['/home/igor.zubarev/data/paris_chunk/wt_N10Division2988shift[0, 0].tif',
    #  '/home/igor.zubarev/data/paris_chunk/wt_N10Division2993shift[0, 0].tif']


# TODO: number of patches?
# formerly :
# https://github.com/juglab-torch/n2v/blob/00d536cdc5f5cd4bb34c65a777940e6e453f4a93/src/n2v/dataloader.py#L52
def extract_patches_sequential(
    arr: np.ndarray,
    patch_sizes: Tuple[int],
    overlaps: Union[Tuple[int], None] = None,
) -> Generator[np.ndarray, None, None]:
    """Generate patches from an array of dimensions C(Z)YX, where C can
    be a singleton dimension.

    The patches are generated sequentially and cover the whole array.
    """
    if len(arr.shape) < 3 or len(arr.shape) > 4:
        raise ValueError(
            f"Input array must have dimensions CZYX or CYX (got length {len(arr.shape)})."
        )

    # Patches sanity check
    if len(patch_sizes) != len(arr.shape[1:]):
        raise ValueError(
            f"There must be a patch size for each spatial dimensions "
            f"(got {patch_sizes} patches for dims {arr.shape[1:]})."
        )

    # Sanity checks on patch sizes versus array dimension
    is_3d_patch = len(patch_sizes) == 3
    if is_3d_patch and patch_sizes[-3] > arr.shape[-3]:
        raise ValueError(
            f"Z patch size is inconsistent with image shape "
            f"(got {patch_sizes[-3]} patches for dim {arr.shape[1]})."
        )

    if patch_sizes[-2] > arr.shape[-2] or patch_sizes[-1] > arr.shape[-1]:
        raise ValueError(
            f"At least one of YX patch dimensions is inconsistent with image shape "
            f"(got {patch_sizes} patches for dims {arr.shape[-2:]})."
        )

    # Overlaps sanity check
    if overlaps is not None and len(overlaps) != len(arr.shape[1:]):
        raise ValueError(
            f"There must be an overlap for each spatial dimensions "
            f"(got {overlaps} overlaps for dims {arr.shape[1:]})."
        )
    elif overlaps is None:
        overlaps = compute_overlap(arr=arr, patch_sizes=patch_sizes)

    for o, p in zip(overlaps, patch_sizes):
        if o >= p:
            raise ValueError(
                f"Overlaps must be smaller than patch sizes "
                f"(got {o} overlap for patch size {p})."
            )

    # Create view window and overlaps
    window_steps = compute_patch_steps(patch_sizes=patch_sizes, overlaps=overlaps)

    # Correct for first dimension
    window_shape = (arr.shape[0], *patch_sizes)
    window_steps = (arr.shape[0], *window_steps)

    if is_3d_patch and patch_sizes[-3] == 1:
        output_shape = (-1,) + window_shape[1:]
    else:
        output_shape = (-1, *window_shape)

    # Generate a view of the input array containing pre-calculated number of patches
    # in each dimension with overlap.
    # Resulting array is resized to (n_patches, C, Z, Y, X) or (n_patches,C, Y, X)
    # TODO add possibility to remove empty or almost empty patches ?
    patches = compute_reshaped_view(
        arr, window_shape=window_shape, step=window_steps, output_shape=output_shape
    )

    # Yield single patch #TODO view_as_windows might be inefficient
    for patch_ixd in range(patches.shape[0]):
        yield (patches[patch_ixd].astype(np.float32))


def extract_patches_random(arr, patch_size, num_patches=None, *args) -> np.ndarray:
    # TODO either num_patches are all unique or output exact number of patches
    crop_coords = np.random.default_rng().integers(
        np.subtract(arr.shape, (0, *patch_size)), size=(num_patches, len(arr.shape))
    )
    # TODO test random patching
    # TODO add multiple arrays support, add possibility to remove empty or almost empty patches ?
    for i in range(crop_coords.shape[1]):
        yield (
            arr[
                (
                    ...,
                    *[
                        slice(c, c + patch_size[j])
                        for j, c in enumerate(crop_coords[:, i, ...])
                    ],
                )
            ]
            .copy()
            .astype(np.float32)
        )


class PatchDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a list of images and apply transforms to the patches."""

    # TODO add napari style axes params, add asserts
    def __init__(
        self,
        data_path: str,
        num_files: int,
        data_reader: Callable,
        patch_size: Union[List[int], Tuple[int]],
        patch_generator: Optional[Callable],
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
        # TODO make this a class method?
        self.data_path = data_path
        self.num_files = num_files
        self.data_reader = data_reader
        self.patch_size = patch_size
        self.patch_generator = patch_generator
        self.add_channel = patch_generator is not None
        self.image_transform = image_level_transform
        self.patch_transform = patch_level_transform

    @staticmethod
    def read_data_source(self, data_source: str, add_channel: True):
        """
        Read data source and correct dimensions.

        Parameters
        ----------
        data_source : str
            Path to data source

        add_channel : bool
            If True, add channel dimension to data source

        Returns
        -------
        image volume : np.ndarray
        """
        if not os.path.exists(data_source):
            raise ValueError(f"Data source {data_source} does not exist")

        arr = tifffile.imread(data_source)
        is_time_series = False
        # Assert data dimensions are correct
        assert len(arr.shape) in (
            2,
            3,
            4,
        ), f"Incorrect data dimensions. Must be 2, 3 or 4, given {arr.shape} for file {data_source}"
        # TODO test add_channel, ugly?
        # TODO updated patch_size can be not defined, fix !
        # TODO improve shape asserts
        # Adding channel dimension if necessary. If present, check correctness
        if len(arr.shape) == 2 or (len(arr.shape) == 3 and len(self.patch_size) == 3):
            arr = np.expand_dims(arr, axis=0)
            updated_patch_size = self.patch_size
        elif len(arr.shape) > 3 and len(self.patch_size) == 2:
            raise ValueError(
                f"Incorrect data dimensions {arr.shape} for given dimensionality {len(self.patch_size)}D in file {data_source}"
            )
        elif len(arr.shape) == 3 and len(self.patch_size) == 2 and arr.shape[0] > 4:
            logging.warning(
                f"Number of channels is {arr.shape[0]} for 2D data. Assuming time series."
            )
            # TODO check if time series
            is_time_series = True
            arr = np.expand_dims(arr, axis=0)
            updated_patch_size = (1, *self.patch_size)
        if not add_channel:
            arr = np.squeeze(arr, axis=0)
            # TODO also update overlap ?
        # TODO time/n_samples dim should come first, not channel ?
        return arr, updated_patch_size, is_time_series

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
        self.source = (
            itertools.islice(Path(self.data_path).rglob("*.tif*"), self.num_files)
            if self.num_files
            else Path(self.data_path).rglob("*.tif*")
        )

        # TODO check for mem leaks, explicitly gc the arr after iterator is exhausted
        for i, filename in enumerate(self.source):
            try:
                # TODO add buffer, several images up to some memory limit?
                arr, patch_size, is_time_series = self.read_data_source(
                    self, filename, self.add_channel
                )
            except (ValueError, FileNotFoundError, OSError) as e:
                logging.exception(f"Exception in file {filename}, skipping")
                raise e
            if i % num_workers == id:
                # TODO add iterator inside
                yield self.image_transform(
                    (
                        arr,
                        patch_size,
                        is_time_series,
                    )  # TODO add is_time_series inside im transform
                ) if self.image_transform is not None else (
                    arr,
                    patch_size,
                    is_time_series,
                )

    def __iter__(self):
        """
        Iterate over data source and yield single patch. Optional transform is applied to the patches.

        Yields
        ------
        np.ndarray
        """
        for image, updated_patch_size, is_time_series in self.__iter_source__():
            # TODO add no patch generator option !
            if self.patch_generator is None:
                yield image, updated_patch_size, is_time_series
            else:
                for patch_data in self.patch_generator(image, updated_patch_size):
                    # TODO add augmentations, multiple functions.
                    # TODO Works incorrectly if patch transform is NONE
                    yield self.patch_transform(
                        patch_data
                    ) if self.patch_transform is not None else (patch_data)
