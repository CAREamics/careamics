"""
Tiling submodule.

These functions are used to tile images into patches or tiles.
"""
import itertools
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import zarr
from skimage.util import view_as_windows

from ..utils.logging import get_logger
from .extraction_strategy import ExtractionStrategy

logger = get_logger(__name__)


def _compute_number_of_patches(
    arr: np.ndarray, patch_sizes: Union[List[int], Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Compute the number of patches that fit in each dimension.

    Array must have one dimension more than the patches (C dimension).

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    patch_sizes : Tuple[int]
        Size of the patches.

    Returns
    -------
    Tuple[int]
        Number of patches in each dimension.
    """
    n_patches = [
        np.ceil(arr.shape[i + 1] / patch_sizes[i]).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(n_patches)


def _compute_overlap(
    arr: np.ndarray, patch_sizes: Union[List[int], Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Compute the overlap between patches in each dimension.

    Array must be of dimensions C(Z)YX, and patches must be of dimensions YX or ZYX.
    If the array dimensions are divisible by the patch sizes, then the overlap is 0.
    Otherwise, it is the result of the division rounded to the upper value.

    Parameters
    ----------
    arr : np.ndarray
        Input array 3 or 4 dimensions.
    patch_sizes : Tuple[int]
        Size of the patches.

    Returns
    -------
    Tuple[int]
        Overlap between patches in each dimension.
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


def _compute_crop_and_stitch_coords_1d(
    axis_size: int, tile_size: int, overlap: int
) -> Tuple[List[Tuple[int, int]], ...]:
    """
    Compute the coordinates of each tile along an axis, given the overlap.

    Parameters
    ----------
    axis_size : int
        Length of the axis.
    tile_size : int
        Size of the tile for the given axis.
    overlap : int
        Size of the overlap for the given axis.

    Returns
    -------
    Tuple[Tuple[int]]
        Tuple of all coordinates for given axis.
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
            # Add the pixel coordinates of the cropped tile in the original image space
            stitch_coords.append(
                (
                    i + overlap // 2 if i > 0 else 0,
                    i + tile_size - overlap // 2
                    if crop_coords[-1][1] < axis_size
                    else axis_size,
                )
            )
            # Add the coordinates to crop the overlap from the prediction.
            overlap_crop_coords.append(
                (
                    overlap // 2 if i > 0 else 0,
                    tile_size - overlap // 2
                    if crop_coords[-1][1] < axis_size
                    else tile_size,
                )
            )
        # If the tile does not fit within the axis, perform the abovementioned
        # operations starting from the end of the axis
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


def _compute_patch_steps(
    patch_sizes: Union[List[int], Tuple[int, ...]], overlaps: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Compute steps between patches.

    Parameters
    ----------
    patch_sizes : Tuple[int]
        Size of the patches.
    overlaps : Tuple[int]
        Overlap between patches.

    Returns
    -------
    Tuple[int]
        Steps between patches.
    """
    steps = [
        min(patch_sizes[i] - overlaps[i], patch_sizes[i])
        for i in range(len(patch_sizes))
    ]
    return tuple(steps)


def _compute_reshaped_view(
    arr: np.ndarray,
    window_shape: Tuple[int, ...],
    step: Tuple[int, ...],
    output_shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Compute reshaped views of an array, where views correspond to patches.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the views are extracted.
    window_shape : Tuple[int]
        Shape of the views.
    step : Tuple[int]
        Steps between views.
    output_shape : Tuple[int]
        Shape of the output array.

    Returns
    -------
    np.ndarray
        Array with views dimension.
    """
    rng = np.random.default_rng()
    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )
    rng.shuffle(patches, axis=0)
    return patches


def _patches_sanity_check(
    arr: np.ndarray,
    patch_size: Union[List[int], Tuple[int, ...]],
    is_3d_patch: bool,
) -> None:
    """
    Check patch size and array compatibility.

    This method validates the patch sizes with respect to the array dimensions:
    - The patch sizes must have one dimension fewer than the array (C dimension).
    - Chack that patch sizes are smaller than array dimensions.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    patch_size : Union[List[int], Tuple[int, ...]]
        Size of the patches along each dimension of the array, except the first.
    is_3d_patch : bool
        Whether the patch is 3D or not.

    Raises
    ------
    ValueError
        If the patch size is not consistent with the array shape (one more array
        dimension).
    ValueError
        If the patch size in Z is larger than the array dimension.
    ValueError
        If either of the patch sizes in X or Y is larger than the corresponding array
        dimension.
    """
    if len(patch_size) != len(arr.shape[1:]):
        raise ValueError(
            f"There must be a patch size for each spatial dimensions "
            f"(got {patch_size} patches for dims {arr.shape})."
        )

    # Sanity checks on patch sizes versus array dimension
    if is_3d_patch and patch_size[0] > arr.shape[-3]:
        raise ValueError(
            f"Z patch size is inconsistent with image shape "
            f"(got {patch_size[0]} patches for dim {arr.shape[1]})."
        )

    if patch_size[-2] > arr.shape[-2] or patch_size[-1] > arr.shape[-1]:
        raise ValueError(
            f"At least one of YX patch dimensions is inconsistent with image shape "
            f"(got {patch_size} patches for dims {arr.shape[-2:]})."
        )


# formerly :
# in dataloader.py#L52, 00d536c
def _extract_patches_sequential(
    arr: np.ndarray, patch_size: Union[List[int], Tuple[int]]
) -> Generator[np.ndarray, None, None]:
    """
    Generate patches from an array in a sequential manner.

    Array dimensions should be C(Z)YX, where C can be a singleton dimension. The patches
    are generated sequentially and cover the whole array.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.
    patch_size : Tuple[int]
        Patch sizes in each dimension.

    Returns
    -------
    Generator[np.ndarray, None, None]
        Generator of patches.
    """
    # Patches sanity check
    is_3d_patch = len(patch_size) == 3

    _patches_sanity_check(arr, patch_size, is_3d_patch)

    # Compute overlap
    overlaps = _compute_overlap(arr=arr, patch_sizes=patch_size)

    # Create view window and overlaps
    window_steps = _compute_patch_steps(patch_sizes=patch_size, overlaps=overlaps)

    # Correct for first dimension for computing windowed views
    window_shape = (1, *patch_size)
    window_steps = (1, *window_steps)

    if is_3d_patch and patch_size[0] == 1:
        output_shape = (-1,) + window_shape[1:]
    else:
        output_shape = (-1, *window_shape)

    # Generate a view of the input array containing pre-calculated number of patches
    # in each dimension with overlap.
    # Resulting array is resized to (n_patches, C, Z, Y, X) or (n_patches,C, Y, X)
    patches = _compute_reshaped_view(
        arr, window_shape=window_shape, step=window_steps, output_shape=output_shape
    )
    logger.info(f"Extracted {patches.shape[0]} patches from input array.")

    # return a generator of patches
    return (patches[i, ...] for i in range(patches.shape[0]))


def _extract_patches_random(
    arr: np.ndarray, patch_size: Union[List[int], Tuple[int]]
) -> Generator[np.ndarray, None, None]:
    """
    Generate patches from an array in a random manner.

    The method calculates how many patches the image can be divided into and then
    extracts an equal number of random patches.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.
    patch_size : Tuple[int]
        Patch sizes in each dimension.

    Yields
    ------
    Generator[np.ndarray, None, None]
        Generator of patches.
    """
    is_3d_patch = len(patch_size) == 3

    # Patches sanity check
    _patches_sanity_check(arr, patch_size, is_3d_patch)

    rng = np.random.default_rng()

    for sample_idx in range(arr.shape[0]):
        sample = arr[sample_idx]
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
            yield patch


def _extract_patches_random_chunks(
    arr: np.ndarray,
    patch_size: Union[List[int], Tuple[int, ...]],
    chunk_size: Union[List[int], Tuple[int, ...]],
    chunk_limit: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generate patches from an array in a random manner.

    The method calculates how many patches the image can be divided into and then
    extracts an equal number of random patches.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.
    patch_size : Tuple[int]
        Patch sizes in each dimension.
    chunk_size : Tuple[int]
        Chunk sizes to load from the.

    Yields
    ------
    Generator[np.ndarray, None, None]
        Generator of patches.
    """
    is_3d_patch = len(patch_size) == 3

    # Patches sanity check
    _patches_sanity_check(arr, patch_size, is_3d_patch)

    rng = np.random.default_rng()
    num_chunks = chunk_limit if chunk_limit else np.prod(arr._cdata_shape)

    # Iterate over num chunks in the array
    for _ in range(num_chunks):
        chunk_crop_coords = [
            rng.integers(0, max(0, arr.shape[i] - chunk_size[i]), endpoint=True)
            for i in range(len(chunk_size))
        ]
        chunk = arr[
            (
                ...,
                *[slice(c, c + chunk_size[i]) for i, c in enumerate(chunk_crop_coords)],
            )
        ].squeeze()

        # Add a singleton dimension if the chunk does not have a sample dimension
        if len(chunk.shape) == len(patch_size):
            chunk = np.expand_dims(chunk, axis=0)
        # Iterate over num samples (S)
        for sample_idx in range(chunk.shape[0]):
            spatial_chunk = chunk[sample_idx]
            assert len(spatial_chunk.shape) == len(
                patch_size
            ), "Requested chunk shape is not equal to patch size"

            n_patches = np.ceil(
                np.prod(spatial_chunk.shape) / np.prod(patch_size)
            ).astype(int)

            # Iterate over the number of patches
            for _ in range(n_patches):
                patch_crop_coords = [
                    rng.integers(
                        0, spatial_chunk.shape[i] - patch_size[i], endpoint=True
                    )
                    for i in range(len(patch_size))
                ]
                patch = (
                    spatial_chunk[
                        (
                            ...,
                            *[
                                slice(c, c + patch_size[i])
                                for i, c in enumerate(patch_crop_coords)
                            ],
                        )
                    ]
                    .copy()
                    .astype(np.float32)
                )
                yield patch


def _extract_tiles(
    arr: np.ndarray,
    tile_size: Union[List[int], Tuple[int]],
    overlaps: Union[List[int], Tuple[int]],
) -> Generator:
    """
    Generate tiles from the input array with specified overlap.

    The tiles cover the whole array.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (S, (Z), Y, X).
    tile_size : Union[List[int], Tuple[int]]
        Tile sizes in each dimension, of length 2 or 3.
    overlaps : Union[List[int], Tuple[int]]
        Overlap values in each dimension, of length 2 or 3.

    Yields
    ------
    Generator
        Tile generator that yields the tile with corresponding coordinates to stitch
        back the tiles together.
    """
    # Iterate over num samples (S)
    for sample_idx in range(arr.shape[0]):
        sample = arr[sample_idx]

        # Create an array of coordinates for cropping and stitching all axes.
        # Shape: (axes, type_of_coord, tile_num, start/end coord)
        crop_and_stitch_coords_list = [
            _compute_crop_and_stitch_coords_1d(
                sample.shape[i], tile_size[i], overlaps[i]
            )
            for i in range(len(tile_size))
        ]

        # Rearrange crop coordinates from a list of coordinate pairs per axis to a list
        # grouped by type.
        # For axis of size 35 and patch size of 32 compute_crop_and_stitch_coords_1d
        # will output ([(0, 32), (3, 35)], [(0, 20), (20, 35)], [(0, 20), (17, 32)]),
        # where the first list is crop coordinates for 1st axis.
        all_crop_coords, all_stitch_coords, all_overlap_crop_coords = zip(
            *crop_and_stitch_coords_list
        )

        # Iterate over generated coordinate pairs:
        for tile_idx, (crop_coords, stitch_coords, overlap_crop_coords) in enumerate(
            zip(
                itertools.product(*all_crop_coords),
                itertools.product(*all_stitch_coords),
                itertools.product(*all_overlap_crop_coords),
            )
        ):
            tile = sample[(..., *[slice(c[0], c[1]) for c in list(crop_coords)])]

            # Check if we are at the end of the sample.
            # To check that we compute the length of the array that contains all the
            # tiles
            if tile_idx == np.prod([len(axis) for axis in all_crop_coords]) - 1:
                last_tile = True
            else:
                last_tile = False
            yield (
                np.expand_dims(tile.astype(np.float32), 0),
                last_tile,
                arr.shape[1:],
                overlap_crop_coords,
                stitch_coords,
            )


def generate_patches(
    sample: Union[np.ndarray, zarr.Array],
    patch_extraction_method: ExtractionStrategy,
    patch_size: Optional[Union[List[int], Tuple[int]]] = None,
    patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generate patches from a sample.

    Parameters
    ----------
    sample : np.ndarray
        Input array.
    patch_extraction_method : ExtractionStrategies
        Patch extraction method, as defined in extraction_strategy.ExtractionStrategy.
    patch_size : Optional[Union[List[int], Tuple[int]]]
        Size of the patches along each dimension of the array, except the first.
    patch_overlap : Optional[Union[List[int], Tuple[int]]]
        Overlap between patches.

    Returns
    -------
    Generator[np.ndarray, None, None]
        Generator yielding patches/tiles.

    Raises
    ------
    ValueError
        If overlap is not specified when using tiling.
    ValueError
        If patches is None.
    """
    patches = None

    if patch_size is not None:
        patches = None

        if patch_extraction_method == ExtractionStrategy.TILED:
            if patch_overlap is None:
                raise ValueError(
                    "Overlaps must be specified when using tiling (got None)."
                )
            patches = _extract_tiles(
                arr=sample, tile_size=patch_size, overlaps=patch_overlap
            )

        elif patch_extraction_method == ExtractionStrategy.SEQUENTIAL:
            patches = _extract_patches_sequential(sample, patch_size=patch_size)

        elif patch_extraction_method == ExtractionStrategy.RANDOM:
            patches = _extract_patches_random(sample, patch_size=patch_size)

        elif patch_extraction_method == ExtractionStrategy.RANDOM_ZARR:
            patches = _extract_patches_random_chunks(
                sample, patch_size=patch_size, chunk_size=sample.chunks
            )

        if patches is None:
            raise ValueError("No patch generated")

        return patches
    else:
        # no patching
        return (sample for _ in range(1))
