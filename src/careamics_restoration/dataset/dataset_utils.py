import itertools
import logging
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
from skimage.util import view_as_windows

from ..utils import normalize

logger = logging.getLogger(__name__)


def _compute_number_of_patches(
    arr: np.ndarray, patch_sizes: Tuple[int]
) -> Tuple[int, ...]:
    """Compute a number of patches in each dimension in order to covert the whole
    array.

    Array must be of dimensions C(Z)YX, and patches must be of dimensions YX or ZYX.

    Parameters
    ----------
    arr : np.ndarray
        Input array 3 or 4 dimensions.
    patche_sizes : Tuple[int]
        Size of the patches

    Returns
    -------
    Tuple[int]
        Number of patches in each dimension
    """
    n_patches = [
        np.ceil(arr.shape[i + 1] / patch_sizes[i]).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(n_patches)


def compute_overlap(arr: np.ndarray, patch_sizes: Tuple[int]) -> Tuple[int, ...]:
    """Compute the overlap between patches in each dimension.

    Array must be of dimensions C(Z)YX, and patches must be of dimensions YX or ZYX.
    If the array dimensions are divisible by the patch sizes, then the overlap is 0.
    Otherwise, it is the result of the division rounded to the upper value.

    Parameters
    ----------
    arr : np.ndarray
        Input array 3 or 4 dimensions.
    patche_sizes : Tuple[int]
        Size of the patches

    Returns
    -------
    Tuple[int]
        Overlap between patches in each dimension
    """
    n_patches = _compute_number_of_patches(arr, patch_sizes)

    overlap = [
        np.ceil(
            np.clip(n_patches[i] * patch_sizes[i] - arr.shape[i + 1], 0, None)
            / max(1, (n_patches[i] - 1))
        ).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(overlap)


def compute_crop_and_stitch_coords_1d(
    axis_size: int, tile_size: int, overlap: int
) -> Tuple[List[Tuple[int, int]], ...]:  # TODO mypy must be wrong here
    """Compute the coordinates for cropping image into tiles, cropping the overlap from predictions and stitching
    the tiles back together across one axis.

    Parameters
    ----------
    axis_size : int
        Length of the axis
    tile_size : int
        size of the tile for the given axis
    overlap : int
        size of the overlap for the given axis

    Returns
    -------
    Tuple[Tuple[int]]
        Tuple of all coordinates for given axis
    """
    # Compute the step between tiles
    step = tile_size - overlap
    crop_coords = []
    stitch_coords = []
    overlap_crop_coords = []
    # Iterate over the axis with a certain step
    for i in range(0, axis_size - overlap, step):
        # Check if the tile fits within the axis
        if i + tile_size <= axis_size:
            # Add the coordinates to crop one tile
            crop_coords.append((i, i + tile_size))
            # Add the pixel coordinates of the cropped tile in the image space
            stitch_coords.append(
                (
                    i + overlap // 2 if i > 0 else 0,
                    i + tile_size - overlap // 2
                    if crop_coords[-1][1] < axis_size
                    else axis_size,
                )
            )
            # Add the coordinates to crop the overlap from the prediction
            overlap_crop_coords.append(
                (
                    overlap // 2 if i > 0 else 0,
                    tile_size - overlap // 2
                    if crop_coords[-1][1] < axis_size
                    else tile_size,
                )
            )
        # If the tile does not fit within the axis, perform the abovementioned operations starting from the end of the axis
        else:
            # if (axis_size - tile_size, axis_size) not in crop_coords:
            crop_coords.append((axis_size - tile_size, axis_size))
            last_tile_end_coord = stitch_coords[-1][1]
            stitch_coords.append((last_tile_end_coord, axis_size))
            overlap_crop_coords.append(
                (tile_size - (axis_size - last_tile_end_coord), tile_size)
            )
            break
    return crop_coords, stitch_coords, overlap_crop_coords


def compute_patch_steps(
    patch_sizes: Tuple[int, ...], overlaps: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute steps between patches.

    Parameters
    ----------
    patch_size : Tuple[int]
        Size of the patches
    overlaps : Tuple[int]
        Overlap between patches

    Returns
    -------
    Tuple[int]
        Steps between patches
    """
    steps = [
        min(patch_sizes[i] - overlaps[i], patch_sizes[i])
        for i in range(len(patch_sizes))
    ]
    return tuple(steps)


def compute_reshaped_view(
    arr: np.ndarray,
    window_shape: Tuple[int, ...],
    step: Tuple[int, ...],
    output_shape: Tuple[int, ...],
) -> np.ndarray:
    """Compute the reshaped views of an array.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the views are extracted
    window_shape : Tuple[int]
        Shape of the views
    step : Tuple[int]
        Steps between views
    output_shape : Tuple[int]
        Shape of the output array
    """
    rng = np.random.default_rng()
    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )
    rng.shuffle(patches, axis=0)
    return patches


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
    return (
        list(itertools.islice(Path(path).rglob("*.tif*"), num_files))
        if num_files
        else list(
            Path(path).rglob("*.tif*")
        )  # TODO Igor: move this functionality to the dataloader. Sort the files by name
    )


# formerly :
# https://github.com/juglab-torch/n2v/blob/00d536cdc5f5cd4bb34c65a777940e6e453f4a93/src/n2v/dataloader.py#L52
def extract_patches_sequential(
    arr: np.ndarray,
    patch_sizes: Tuple[int],
    mean: Optional[int] = None,
    std: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """Generate patches from an array of dimensions C(Z)YX, where C can
    be a singleton dimension.

    The patches are generated sequentially and cover the whole array.
    """
    # TODO document
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

    for patch_ixd in range(patches.shape[0]):
        patch = patches[patch_ixd].astype(np.float32).squeeze()
        yield (normalize(patch, mean, std)) if (mean and std) else (patch)


def extract_patches_random(
    arr,
    patch_size,
    mean: Optional[int] = None,
    std: Optional[int] = None,
    *args,
) -> Generator[np.ndarray, None, None]:
    rng = np.random.default_rng()
    # shuffle the array along the first axis TODO do we need shuffling?
    rng.shuffle(arr, axis=0)

    for sample_idx in range(arr.shape[0]):
        sample = arr[sample_idx]
        # calculate how many number of patches can image area be divided into
        # TODO add possibility to remove empty or almost empty patches ? Recalculate?
        n_patches = np.ceil(np.prod(sample.shape) / np.prod(patch_size)).astype(int)
        for _ in range(n_patches):
            crop_coords = [
                rng.integers(0, arr.shape[i + 1] - patch_size[i])
                for i in range(len(patch_size))
            ]
            patch = (
                sample[
                    (
                        ...,
                        *[
                            slice(c, c + patch_size[i])
                            for i, c in enumerate(crop_coords)
                        ],
                    )
                ]
                .copy()
                .astype(np.float32)
            )
            yield (normalize(patch, mean, std)) if (mean and std) else (patch)


def extract_patches_predict(
    arr: np.ndarray,
    patch_size: Tuple[int],
    overlaps: Tuple[int],
    mean: Optional[int] = None,
    std: Optional[int] = None,
) -> Iterable[List[np.ndarray]]:
    # Overlap is half of the value mentioned in original N2V. must be even. It's like this because of current N2V notation
    # Iterate over num samples (S)
    for sample_idx in range(arr.shape[0]):
        sample = arr[sample_idx]
        # Create an array of coordinates for cropping and stitching for all axes.
        # Shape: (axes, type_of_coord, tile_num, start/end coord)
        crop_and_stitch_coords_list = [
            compute_crop_and_stitch_coords_1d(
                sample.shape[i], patch_size[i], overlaps[i]
            )
            for i in range(len(patch_size))
        ]
        crop_and_stitch_coords = np.array(crop_and_stitch_coords_list)
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
            tile = (normalize(tile, mean, std)) if (mean and std) else (tile)
            yield (
                np.expand_dims(tile.astype(np.float32), 0),
                sample_idx,
                arr.shape[1:],
                overlap_crop_coords,
                stitch_coords,
            )
