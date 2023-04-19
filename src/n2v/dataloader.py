import os
import torch
import logging
import itertools
import tifffile
import numpy as np

from typing import Generator

from functools import partial
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path
from skimage.util import view_as_windows
from typing import Callable, List, Optional, Sequence, Union, Tuple

from .pixel_manipulation import n2v_manipulate
from .dataloader_utils import(
    compute_view_windows,
    compute_overlap,
    compute_reshaped_view
)


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


# TODO: number of patches?
# TODO: overlap?
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
            f"Z patch size is inconsistent with image shape " \
            f"(got {patch_sizes[-3]} patches for dim {arr.shape[1]})."
            )
    
    if patch_sizes[-2] > arr.shape[-2] or patch_sizes[-1] > arr.shape[-1]:
        raise ValueError(
            f"At least one of YX patch dimensions is inconsistent with image shape " \
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
    window_shape, window_steps = compute_view_windows(
        patch_sizes=patch_sizes, overlaps=overlaps
    )

    # Correct for first dimension
    window_shape = (arr.shape[0],) + window_shape
    window_steps = (arr.shape[0],) + window_steps
    
    if is_3d_patch and patch_sizes[-3] == 1:
        output_shape = (-1,) + window_shape[1:]
    else:
        output_shape = (-1,) + window_shape
        
    # Generate a view of the input array containing pre-calculated number of patches 
    # in each dimension with overlap.
    # Resulting array is resized to (n_patches, C, Z, Y, X) or (n_patches,C, Y, X)
    # TODO add possibility to remove empty or almost empty patches ?
    patches = compute_reshaped_view(
        arr, 
        window_shape=window_shape, 
        step=window_steps, 
        output_shape=output_shape
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


def extract_patches_predict(
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


def _calculate_stitching_coords(tile_coords: Tuple[int], last_tile_coord: Tuple[int], overlap: Tuple[int]) -> Tuple[slice]:
    
   
    #TODO add 2/3d support
    # TODO different overlaps for each dimension
    # TODO different patch sizes for each dimension
    list_coord = []

    for i, coord in enumerate(tile_coords): 
        if coord == 0:
            list_coord.append(slice(0, -overlap[i]//2))
        elif coord == last_tile_coord[i] - 1:
            list_coord.append(slice(overlap[i]//2, None))
        else:
            list_coord.append(slice(overlap[i]//2, -overlap[i]//2))
    return list_coord


def extract_patches_predict_new(arr: np.ndarray, patch_size: Tuple[int], overlap=32) -> List[np.ndarray]:
    # TODO remove hard coded vals
    # Overlap is half of the value mentioned in original N2V #TODO must be even. It's like this because of current N2V notation
    actual_overlap = [
        patch_size[i] - overlap[i] for i in range(len(patch_size))
    ]
    
    #TODO add asserts
    all_tiles = view_as_windows(arr, window_shape=patch_size, step=actual_overlap) #shape (tiles in y, tiles in x, Y, X)
    pred = []

    for tile_coords in itertools.product(*map(range, all_tiles.shape[:len(patch_size)])): #TODO add 2/3d automatic selection of axes  
        #TODO test for number of tiles in each category
        tile = all_tiles[(*[c for c in tile_coords], ...)]

        coords = _calculate_stitching_coords(tile_coords, all_tiles.shape[:len(patch_size)], overlap)
        pred.append(tile[(*[c for c in coords], ...)]) #TODO add proper last tile coord ! Should be list !)

    return pred 


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
