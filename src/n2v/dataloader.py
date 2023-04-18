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

from .pixel_manipulation import n2v_manipulate


############################################
#   ETL pipeline #TODO add description to all modules
############################################


def list_input_source_tiff(
    path: Union[str, Path], num_files: Union[int, None] = None
) -> List:
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
    # TODO add support for reading a subset of files
    return (
        list(itertools.islice(Path(path).rglob("*.tif*"), num_files))
        if num_files
        else list(Path(path).rglob("*.tif*"))
    )
    # return ['/home/igor.zubarev/data/paris_chunk/wt_N10Division2988shift[0, 0].tif',
    #  '/home/igor.zubarev/data/paris_chunk/wt_N10Division2993shift[0, 0].tif']


def extract_patches_sequential(
    arr,
    patch_size,
    num_patches=None,
    overlap=None,  # TODO add support for overlap. This is slighly ugly
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
    # TODO 2D/3D separately ?
    # TODO put asserts in separate function in init
    # Asserts
    assert len(patch_size) == len(
        arr.shape[1:]
    ), "Number of patch dimensions must match image dimensions"
    assert (
        z_patch_size is None or z_patch_size <= arr.shape[1]
    ), "Z patch size is incosistent with image shape"
    assert (
        y_patch_size <= arr.shape[-2] and x_patch_size <= arr.shape[-1]
    ), "At least one of XY patch dimensions is incosistent with image shape"

    # Calculate total number of patches for each dimension
    z_total_patches = (
        np.ceil(arr.shape[1] / z_patch_size) if z_patch_size is not None else None
    ) #TODO might need to be changed for prediction case and moved into a separate function
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
        output_shape = (
            (-1, arr.shape[0], y_patch_size, x_patch_size)
            if z_patch_size == 1
            else (-1, arr.shape[0], z_patch_size, y_patch_size, x_patch_size)
        )
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
        yield (patches[patch_ixd].astype(np.float32))


def extract_patches_random(arr, patch_size, num_patches=None, *args) -> np.ndarray:

    # TODO either num_patches are all unique or output exact number of patches
    crop_coords = np.random.default_rng().integers(
        np.subtract(arr.shape, (0, *patch_size)), size=(num_patches, len(arr.shape))
    )

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


def extract_patches_predict_old(
    arr: np.ndarray, patch_size: Tuple[int], overlap: Tuple[int]
) -> np.ndarray:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    arr : _type_
        _description_
    patch_size : _type_
        _description_

    Yields
    ------
    Iterator[np.ndarray]
        _description_
    """
    arr = arr[:, 0, ...][np.newaxis, ...]

    z_patch_size = (
        None if len(patch_size) == 2 else patch_size[0]
    )  # TODO Should it be None in some cases ?
    y_patch_size, x_patch_size = patch_size[-2:]
    z_overlap = 0 if len(overlap) == 2 else overlap[0]
    y_overlap, x_overlap = overlap[-2:]

    num_samples = 1

    # TODO add asserts

    z_min = 0
    y_min = 0
    x_min = 0
    z_max = z_patch_size
    y_max = y_patch_size
    x_max = x_patch_size
    pred = np.zeros(arr.shape)

    tiles = []
    check_coords = []
    pred_coords = []
    # TODO Refactor, this is extremely ugly. 2D/3D separately?
    z_running_overlap = 0
    while z_min < arr.shape[1]:
        # TODO Rename
        overlap_left = 0
        while x_min < arr.shape[2]:
            overlap_top = 0
            while y_min < arr.shape[3]:
                # TODO hardcoded dimensions ? arr size always 4 ? assert ?


                # start coordinates of the new patch
                z_min_ = min(arr.shape[1], z_max) - z_patch_size
                y_min_ = min(arr.shape[2], y_max) - y_patch_size
                x_min_ = min(arr.shape[3], x_max) - x_patch_size
                
                # difference between the start coordinates of the new patch and the patch given the border
                # this is non zero only at the borders
                last_patch_shift_z = z_min - z_min_
                last_patch_shift_y = y_min - y_min_
                last_patch_shift_x = x_min - x_min_
                
                
                
                
                if (
                    (z_min_, y_min_, y_max),
                    (z_min_, x_min_, x_max),
                ) not in check_coords:
                    coords = ((z_min_, y_min_, y_max), (z_min_, x_min_, x_max))
                    tile = arr[:, z_min_:z_max, y_min_:y_max, x_min_:x_max]
                    check_coords.append(coords)
                    tiles.append(tile)
                    # TODO add proper description
                    pred_coords.append(
                        [
                            last_patch_shift_z,
                            last_patch_shift_y,
                            last_patch_shift_x,
                            z_running_overlap,
                            overlap_top,
                            overlap_left,
                        ]
                    )
                    # pred[:, z_min:z_max, y_min:y_max, x_min:x_max] = tile[:, last_patch_shift_z:, last_patch_shift_y:, last_patch_shift_x:][:, z_running_overlap:, overlap_top:, overlap_left:]
                y_min = y_min - y_overlap + y_patch_size
                y_max = y_min + y_patch_size
                overlap_top = y_overlap // 2
            y_min = 0
            y_max = y_patch_size
            x_min = x_min - x_overlap + x_patch_size
            x_max = x_min + x_patch_size
            overlap_left = x_overlap // 2
        x_min = 0
        x_max = x_patch_size
        z_min = z_min - z_overlap + z_patch_size
        z_max = z_min + z_patch_size
        z_running_overlap = z_overlap // 2

    # crops = [crop for crop in crops if all(crop.shape) > 0]
    # TODO assert len tiles == len coords
    for tile, crop in zip(tiles, pred_coords):
        yield (tile.astype(np.float32), crop)


def extract_patches_predict(arr: np.ndarray, patch_size: Tuple[int], overlap=Tuple[int]) -> List[np.ndarray]:
    # TODO remove hard coded vals
    # Overlap is half of the value mentioned in original N2V #TODO must be even. It's like this because of current N2V notation
    #TODO range start from 1, because 0 is the channel dimension
    #TODO check if patch size == image size 
    actual_overlap = [arr.shape[0],
        *[patch_size[i] - overlap[i-1] for i in range(1, len(patch_size))
    ]]

    if len(patch_size) + 1 != len(actual_overlap):
        #TODO ugly fix for incosistent overlap shape
        actual_overlap.insert(0, 1)

    #TODO add asserts
    all_tiles = view_as_windows(arr, window_shape=[arr.shape[0], *patch_size], step=actual_overlap) #shape (tiles in y, tiles in x, Y, X)
    #TODO properly handle 2d/3d, copy from sequential patch extraction
    #TODO questo e una grande cazzata !!!
    output_shape = (arr.shape[0], arr.shape[1], all_tiles.shape[2], all_tiles.shape[3], *patch_size[1:])
    all_tiles = all_tiles.reshape(*output_shape)
    for tile_coords in itertools.product(*map(range, all_tiles.shape[:len(patch_size)])): #TODO add 2/3d automatic selection of axes  
        #TODO test for number of tiles in each category
        tile = all_tiles[(*[c for c in tile_coords], ...)]
        yield (tile.astype(np.float32), tile_coords, all_tiles.shape[:len(patch_size)], overlap)


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
        # TODO make this a class method?
        self.data_path = data_path
        self.num_files = num_files
        self.data_reader = data_reader
        self.patch_size = patch_size
        self.patch_generator = patch_generator
        self.image_transform = image_level_transform
        self.patch_transform = patch_level_transform

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
            arr = np.expand_dims(arr, axis=0)
            updated_patch_size = (1, *self.patch_size)
            #TODO also update overlap ?
        #TODO time/n_samples dim should come first, not channel ?
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
        self.source = (
            itertools.islice(Path(self.data_path).rglob("*.tif*"), self.num_files)
            if self.num_files
            else Path(self.data_path).rglob("*.tif*")
        )

        # TODO check for mem leaks, explicitly gc the arr after iterator is exhausted
        for i, filename in enumerate(self.source):
            try:
                # TODO add buffer, several images up to some memory limit?
                arr, patch_size = self.read_data_source(self, filename)
            except (ValueError, FileNotFoundError, OSError) as e:
                logging.exception(f"Exception in file {filename}, skipping")
                raise e
            if i % num_workers == id:
                # TODO add iterator inside
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
            for patch_data in self.patch_generator(image, updated_patch_size):
                # TODO add augmentations, multiple functions.
                # TODO Works incorrectly if patch transform is NONE
                yield self.patch_transform(
                    patch_data
                ) if self.patch_transform is not None else (patch_data)
