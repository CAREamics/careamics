import itertools
import logging
import os
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import tifffile
import torch

from .dataloader_utils import (
    compute_overlap,
    compute_reshaped_view,
    compute_patch_steps,
    are_axes_valid,
    compute_crop_and_stitch_coords_1d,
)

from .utils import normalize

############################################
#   ETL pipeline #TODO add description to all modules
############################################


logger = logging.getLogger(__name__)


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

    for p in patch_sizes:
        # check if 1
        if p < 2:
            raise ValueError(f"Invalid patch value (got {p}).")

        # check if power of two
        if not (p & (p - 1) == 0):
            raise ValueError(f"Patch size must be a power of two (got {p}).")

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

    # Compute overlap
    overlaps = compute_overlap(arr=arr, patch_sizes=patch_sizes)

    # Create view window and overlaps
    window_steps = compute_patch_steps(patch_sizes=patch_sizes, overlaps=overlaps)

    # Correct for first dimension for computing windowed views
    window_shape = (1, *patch_sizes)
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
    logger.info(f"Extracted {patches.shape[0]} patches from input array.")
    # Yield single patch #TODO view_as_windows might be inefficient
    for patch_ixd in range(patches.shape[0]):
        patch = patches[patch_ixd].astype(np.float32).squeeze()
        yield (normalize(patch, mean, std)) if (mean and std) else (patch)


def extract_patches_random(
    arr,
    patch_size,
    mean: int = None,
    std: int = None,
    overlaps=None,
    num_patches=None,
    *args,
) -> Generator[np.ndarray, None, None]:
    # TODO either num_patches are all unique or output exact number of patches
    rng = np.random.default_rng()
    y = rng.integers(0, arr.shape[1] - patch_size[0])
    x = rng.integers(0, arr.shape[2] - patch_size[1])
    rng.shuffle(arr, axis=0)
    if num_patches is None:
        num_patches = arr.shape[0]
    # crop_coords = rng.integers(
    #     np.subtract(arr.shape, (0, *patch_size)), size=(num_patches, len(arr.shape))
    # )
    # TODO test random patching
    # TODO add multiple arrays support, add possibility to remove empty or almost empty patches ?
    for i in range(arr.shape[0]):
        # patch = arr[(...,*[slice(c, c + patch_size[j]) for j, c in enumerate(crop_coords[:, i, ...])],)].copy().astype(np.float32)
        patch = (
            arr[i, y : y + patch_size[0], x : x + patch_size[1]]
            .copy()
            .astype(np.float32)
        )
        yield (normalize(patch, mean, std)) if (mean and std) else (patch)


def extract_patches_predict(
    arr: np.ndarray,
    patch_size: Tuple[int],
    overlaps: Tuple[int],
    mean: int = None,
    std: int = None,
) -> List[np.ndarray]:
    # Overlap is half of the value mentioned in original N2V. must be even. It's like this because of current N2V notation
    arr = arr[0, 99:, :141][np.newaxis]
    # Iterate over num samples (S)
    for sample_idx in range(arr.shape[0]):
        sample = arr[sample_idx]
        # Create an array of coordinates for cropping and stitching for all axes.
        # Shape: (axes, type_of_coord, tile_num, start/end coord)
        crop_and_stitch_coords = np.array(
            [
                compute_crop_and_stitch_coords_1d(
                    sample.shape[i], patch_size[i], overlaps[i]
                )
                for i in range(len(patch_size))
            ]
        )
        # TODO check this ugly swapaxes
        # Swap axes to regroup initial array and get arrays of all crop coords, all stitch coords and all overlap crop coords
        all_crop_coords, all_stitch_coords, all_overlap_crop_coords = np.swapaxes(
            crop_and_stitch_coords, 0, 1
        )
        # Iterate over generated coordinate pairs:
        for crop_coords, stitch_coords, overlap_crop_coords in zip(
            itertools.product(*all_crop_coords),
            itertools.product(*all_stitch_coords),
            itertools.product(*all_overlap_crop_coords),
        ):
            tile = sample[(..., *[slice(c[0], c[1]) for c in list(crop_coords)])]
            yield (
                np.expand_dims(tile.astype(np.float32), 0),
                sample_idx,
                arr.shape[1:],
                overlap_crop_coords,
                stitch_coords,
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
        # #TODO Assert should all be done in Configuration.validate_wordir. Check

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
                arr_num_dims = len(arr.shape)
            except ValueError:
                arr = np.load(
                    data_source, allow_pickle=True
                )  # TODO this is a hack to deal with the fact that we are saving a list of arrays
                arr_num_dims = (
                    len(arr[0].shape) + 1
                )  # TODO check all arrays have the same or compliant shape ?
        elif data_source.suffix[:4] == ".tif":
            arr = tifffile.imread(data_source)
            arr_num_dims = len(arr.shape)

        # remove any singleton dimensions
        arr = arr.squeeze()

        # sanity check on dimensions
        if len(arr.shape) < 2 or len(arr.shape) > 4:
            raise ValueError(
                f"Incorrect data dimensions. Must be 2, 3 or 4 (got {arr.shape} for file {data_source})."
            )

        # sanity check on axes length
        if len(axes) != len(arr.shape):
            raise ValueError(
                f"Incorrect axes length (got {axes} for file {data_source})."
            )

        # check axes validity
        are_axes_valid(axes)  # this raises errors

        # patch sanity check
        if len(patch_size) != len(arr.shape) and len(patch_size) != len(arr.shape) - 1:
            raise ValueError(
                f"Incorrect patch size (got {patch_size} for file {data_source} with shape {arr.shape})."
            )

        for p in patch_size:
            # check if power of 2
            if not (p & (p - 1) == 0):
                raise ValueError(
                    f"Incorrect patch size, should be power of 2 (got {patch_size} for file {data_source})."
                )

        # TODO add axes shuffling and reshapes. so far assuming correct order
        if ("S" in axes or "T" in axes) and arr.dtype != "O":
            # TODO use re?
            arr = arr.reshape(
                -1, *arr.shape[len(axes.replace("Z", "").replace("YX", "")) :]
            )
        elif arr.dtype == "O":
            for i in range(len(arr)):  # TODO add check for dimenstions of each array
                arr[i] = np.expand_dims(arr[i], axis=0)
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
        logger.info(f"Calculated mean and std for {i + 1} images")
        logger.info(f"Mean: {self.mean}, std: {self.std}")

    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.num_files if self.num_files else len(self.source)

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
                for idx in range(image.shape[0]):
                    sample = np.expand_dims(image[idx], (0, 1)).astype(
                        np.float32
                    )  # TODO check explanddims !!
                    yield normalize(sample, self.mean, self.std) if (
                        self.mean and self.std
                    ) else image
            else:
                for patch_data in self.patch_generator(
                    image, self.patch_size, mean=self.mean, std=self.std
                ):
                    yield self.patch_transform(
                        patch_data
                    ) if self.patch_transform is not None else (patch_data)
