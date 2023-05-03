import itertools
import logging
import os
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import tifffile
import torch
from skimage.util import view_as_windows

from .dataloader_utils import (
    compute_overlap,
    compute_reshaped_view,
    compute_view_windows,
    compute_overlap_predict,
)

from .utils import normalize

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
    mean: int = None,
    std: int = None,
) -> Generator[np.ndarray, None, None]:
    """Generate patches from an array of dimensions C(Z)YX, where C can
    be a singleton dimension.

    The patches are generated sequentially and cover the whole array.
    """
    if len(arr.shape) < 3 or len(arr.shape) > 4:
        raise ValueError(
            f"Input array must have dimensions SZYX or SYX (got length {len(arr.shape)})."
        )

    # Patches sanity check
    if len(patch_sizes) != len(arr.shape[1:]):
        raise ValueError(
            f"There must be a patch size for each spatial dimensions "
            f"(got {patch_sizes} patches for dims {arr.shape})."
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
    window_shape, window_steps = compute_view_windows(
        patch_sizes=patch_sizes, overlaps=overlaps
    )

    # Correct for first dimension for computing windowed views
    window_shape = (1, *window_shape)
    window_steps = (1, *window_steps)

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
        patch = patches[patch_ixd].astype(np.float32)
        yield (normalize(patch, mean, std)) if mean and std else (patch)


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


def extract_patches_predict(
    arr: np.ndarray, patch_size: Tuple[int], overlap: Tuple[int]
) -> List[np.ndarray]:
    # Overlap is half of the value mentioned in original N2V. must be even. It's like this because of current N2V notation

    last_overlap = compute_overlap_predict(
        arr=arr, patch_size=patch_size, overlap=overlap
    )
    step = tuple([p - o for p, o in zip(patch_size, overlap)])
    all_tiles = view_as_windows(
        arr, window_shape=[1, *patch_size], step=[1, *step]
    )  # shape (tiles in y, tiles in x, Y, X)

    output_shape = (
        arr.shape[0],
        *all_tiles.shape[1 : 1 + len(patch_size)],
        *patch_size,
    )
    # Save number of tiles in each dimension
    all_tiles = all_tiles.reshape(*output_shape)

    # Iterate over num samples (S)
    # TODO add +1 to all tiles shape in each dim -> when all_tiles is exhausted -> yield last_tile
    for sample in range(all_tiles.shape[0]):
        # Adding 1 to all dimensions to ensure capturing the last tile
        full_coverage_shape = [d + 1 for d in all_tiles.shape[1 : len(patch_size) + 1]]
        for tile_level_coords in itertools.product(*map(range, full_coverage_shape)):
            if all([c1 < c2 for c1, c2 in zip(tile_level_coords, full_coverage_shape)]):
                tile = all_tiles[sample][(*[c for c in tile_level_coords], ...)]
            else:
                last_tile_dims = np.zeros_like(arr.shape)
                # Id of dimension where the tile pointer is currently at the border
                idx = np.where(
                    [c1 >= c2 for c1, c2 in zip(tile_level_coords, full_coverage_shape)]
                )[0]
                # TODO fix case where len(idx) > 1
                last_tile_dims[idx]
                last_tile_coords = [slice(None)] + [
                    slice(
                        (arr.shape[i + 1] - patch_size[i]) * tile_level_coords[i]
                        - overlap[i],
                        arr.shape[i + 1] * tile_level_coords[i] - overlap[i],
                        None,
                    )
                    for i in range(len(patch_size))
                ]
                tile = arr[sample][(*[c for c in last_tile_coords], ...)]
                # TODO

            yield (
                tile.astype(np.float32),
                sample,
                tile_level_coords,
                all_tiles.shape[1 : len(patch_size) + 1],
                arr.shape[1:],
            )


class PatchDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a list of images and apply transforms to the patches."""

    def __init__(
        self,
        data_path: str,
        ext: str,
        axes: str,
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
        self.ext = ext
        self.axes = axes
        self.num_files = num_files
        self.data_reader = data_reader
        self.patch_size = patch_size
        self.patch_generator = patch_generator
        self.add_channel = patch_generator is not None
        self.image_transform = image_level_transform
        self.patch_transform = patch_level_transform

    # @staticmethod
    # def read_alter(self):

    #     somearray = read_image_whatever(config)

    #     # read configuration
    #     check_axis_adn_raise_error(somearray, config.axes)

    # def check_axis_adn_raise_error():
    #     if config.axes is None and axes!= C(Z)YX:
    #         raise Error
    #     else:
    #         # sanity check on the axes
    #         if axes != config.axes:
    #             raise Error

    #         new_arr = move_axes(array)

    #         return new_arr

    @staticmethod
    def read_data_source(
        data_source: Union[str, Path], axes: str, patch_size: Tuple[int]
    ):
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
        if not Path(data_source).exists():
            raise ValueError(f"Data source {data_source} does not exist")

        # TODO separate this into a function with other formats
        if data_source.suffix == ".npy":
            try:
                arr = np.load(data_source)
            except ValueError:
                arr = np.load(data_source, allow_pickle=True)[0].astype(
                    np.float32
                )  # TODO this is a hack to deal with the fact that we are saving a list of arrays
        elif data_source.suffix[:4] == ".tif":
            arr = tifffile.imread(data_source)

        # Assert data dimensions are correct
        assert len(arr.shape) in (
            2,
            3,
            4,
        ), f"Incorrect data dimensions. Must be 2, 3 or 4, given {arr.shape} for file {data_source}"

        assert len(arr.shape) == len(axes), f"Incorrect axes. Must be {len(arr.shape)}"

        # TODO add axes shuffling and reshapes. so far assuming correct order
        if "S" in axes or "T" in axes:
            # TODO use re?
            arr = arr.reshape(
                -1, *arr.shape[len(axes.replace("Z", "").replace("YX", "")) :]
            )
        else:
            arr = np.expand_dims(arr, axis=0)
            # TODO do we need to update patch size?
        return arr

    def calculate_stats(self):
        mean = 0
        std = 0
        for i, image in tqdm(enumerate(self.__iter_source__())):
            mean += image.mean()
            std += np.std(image)
        self.mean = mean / (i + 1)
        self.std = std / (i + 1)

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
            itertools.islice(
                Path(self.data_path).rglob(f"*.{self.ext}*"), self.num_files
            )
            if self.num_files
            else Path(self.data_path).rglob(f"*.{self.ext}*")
        )

        # TODO check for mem leaks, explicitly gc the arr after iterator is exhausted
        for i, filename in enumerate(self.source):
            try:
                # TODO add buffer, several images up to some memory limit?
                arr = self.read_data_source(filename, self.axes, self.patch_size)
            except (ValueError, FileNotFoundError, OSError) as e:
                logging.exception(f"Exception in file {filename}, skipping")
                raise e
            if i % num_workers == id:
                # TODO add iterator inside
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
            if self.patch_generator is None:
                yield normalize(
                    image, self.mean, self.std
                ) if self.mean and self.std else image
            else:
                for patch_data in self.patch_generator(
                    image, self.patch_size, overlaps=None, mean=self.mean, std=self.std
                ):
                    # TODO add augmentations, multiple functions.
                    # TODO Works incorrectly if patch transform is NONE
                    yield self.patch_transform(
                        patch_data
                    ) if self.patch_transform is not None else (patch_data)
