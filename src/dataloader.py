import os
import torch
import logging
import tifffile
import numpy as np

from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path
from skimage.util import view_as_windows
from typing import Callable, List, Optional, Sequence, Union, Tuple


############################################
#   ETL pipeline #TODO add description to all modules
############################################


def apply_struct_n2v_mask(patch, coords, dims, mask):
    """
    each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value
    """
    coords = np.array(coords).astype(np.int32)
    ndim = mask.ndim
    center = np.array(mask.shape) // 2
    ## leave the center value alone
    mask[tuple(center.T)] = 0j
    ## displacements from center
    dx = np.indices(mask.shape)[:, mask == 1] - center[:, None]
    ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = dx.T[..., None] + coords[None]
    mix = mix.transpose([1, 0, 2]).reshape([ndim, -1]).T
    ## stay within patch boundary
    mix = mix.clip(min=np.zeros(ndim), max=np.array(patch.shape) - 1).astype(np.uint)
    ## replace neighbouring pixels with random values from flat dist
    patch[tuple(mix.T)] = np.random.rand(mix.shape[0]) * 4 - 2
    #TODO finish, test
    return patch

def augment_single(image, seed=1, channel_dim=True):
    """Augment single data object(2D or 3D) before batching by rotating and flipping patches.

    Parameters
    ----------
    patches : np.ndarray
        array containing single image or patch, 2D or 3D
    seed : int, optional
        seed for random number generator, controls the rotation and flipping
    channel_dim : bool, optional
        Set to True if the channel dimension is present, by default True

    Returns
    -------
    np.ndarray
        _description_
    """
    rotate_state = np.random.randint(0, 5)
    flip_state = np.random.randint(0, 2)
    rotated = np.rot90(image, k=rotate_state, axes=(-2, -1))
    flipped = np.flip(rotated, axis=-1) if flip_state == 1 else rotated
    # TODO check for memory leak
    return flipped.copy()


def open_input_source(path: Union[str, Path]) -> List:
    # Basic function to open input source
    # TODO add support for other cases
    #TODO add support for reading a subset of files 
    return [Path(p).rglob('*.tif*') for p in path]


def extract_patches_sequential(
    arr, patch_size, num_patches=None
) -> np.ndarray:  # TODO add support for shapes
    """Generate patches from ND array

    Crop an array into patches deterministically covering the whole array.

    Parameters
    ----------
    arr : np.ndarray
        Input array. Possible shapes are (C, Z, Y, X) or (C, Y, X)
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

    # Asserts
    assert (
        z_patch_size is None or z_patch_size <= arr.shape[1]
    ), "Patch size must be of length 2 or 3"
    assert (
        y_patch_size <= arr.shape[-2] and x_patch_size <= arr.shape[-1]
    ), "Patch size must be of length 2 or 3"

    # Calculate total number of patches for each dimension
    z_total_patches = (
        np.ceil(arr.shape[1] / z_patch_size) if z_patch_size is not None else None
    )
    y_total_patches = np.ceil(arr.shape[-2] / y_patch_size)
    x_total_patches = np.ceil(arr.shape[-1] / x_patch_size)

    # TODO make patch size and overlap calc shorter ?
    total_patches = [np.ceil()]

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
        output_shape = (-1, arr.shape[0], z_patch_size, y_patch_size, x_patch_size)
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
        data_reader: Callable,
        patch_size: Union[List[int], Tuple[int]],
        patch_generator: Union[np.ndarray, Callable],
        image_level_transform: Optional[Callable] = None,
        patch_level_transform: Union[Callable, Optional[Callable]] = None,
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
        # super().__init__(data=data, transform=None)
        
        # Assert input data
        assert isinstance(
            data_path, list
        ), f"Incorrect data_path type. Must be a list, given{type(data_path)}"

        # Assert patch_size
        assert isinstance(
            patch_size, (list, tuple)
        ), f"Incorrect patch_size. Must be a tuple, given{type(patch_size)}"
        assert len(patch_size) in (
            2,
            3,
        ), f"Incorrect patch_size. Must be a 2 or 3, given{len(patch_size)}"

        self.data_reader = data_reader(data_path)
        self.source_iter = iter(self.data_reader)
        self.patch_size = patch_size
        self.patch_generator = patch_generator
        self.image_transform = image_level_transform
        self.patch_transform = patch_level_transform

        assert len(self.data_reader) > 0, "Data source is empty"

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

        # Adding channel dimension if necessary. If present, check correctness
        if len(arr.shape) == 2 or (len(arr.shape) == 3 and len(self.patch_size) == 3):
            arr = np.expand_dims(arr, axis=0)
        elif len(arr.shape) == 3 and len(self.patch_size) == 2 and arr.shape[0] > 4:
            raise ValueError(f"Incorrect number of channels {arr.shape[0]}")
        elif len(arr.shape) > 3 and len(self.patch_size) == 2:
            raise ValueError(
                f"Incorrect data dimensions {arr.shape} for given dimensionality {len(self.patch_size)}D in file {data_source}"
            )
        # TODO add other asserts
        return arr

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
        for i, filename in enumerate(self.source):
            try:
                arr = self.read_data_source(self, filename)
            except (ValueError, FileNotFoundError, OSError) as e:
                logging.exception(f"Exception in file {filename}, skipping")
                raise e
            if i % num_workers == id:
                yield self.image_transform(
                    arr
                ) if self.image_transform is not None else arr

    def __iter__(self):
        """
        Iterate over data source and yield single patch. Optional transform is applied to the patches.

        Yields
        ------
        np.ndarray
        """
        for image in self.__iter_source__():
            for patch in self.patch_generator(image, self.patch_size):
                # TODO add augmentations, multiple functions
                for func in self.patch_transform:
                    patch = func(patch)
                yield self.patch_transform(
                    patch
                ) if self.patch_transform is not None else patch
