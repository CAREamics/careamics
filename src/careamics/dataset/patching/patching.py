"""
Tiling submodule.

These functions are used to tile images into patches or tiles.
"""
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import zarr

from ...config.support.supported_extraction_strategies import (
    SupportedExtractionStrategy,
)
from ...utils.logging import get_logger
from ..dataset_utils import reshape_array
from .random_patching import extract_patches_random, extract_patches_random_from_chunks
from .sequential_patching import extract_patches_sequential
from .tiled_patching import extract_tiles

logger = get_logger(__name__)


# TODO: several issues that require refactoring
# - some patching return array, others generator
# - in iterable and in memory, the reshaping happens at different moment
# - return type is not consistent (ndarray, ndarray or ndarray, None or just ndarray)

# called by in memory dataset
def prepare_patches_supervised(
    train_files: List[Path],
    target_files: List[Path],
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
    read_source_func: Callable,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Iterate over data source and create an array of patches and corresponding targets.

    Returns
    -------
    np.ndarray
        Array of patches.
    """
    train_files.sort()
    target_files.sort()

    means, stds, num_samples = 0, 0, 0
    all_patches, all_targets = [], []
    for train_filename, target_filename in zip(train_files, target_files):
        try:
            sample: np.ndarray = read_source_func(train_filename, axes)
            target: np.ndarray = read_source_func(target_filename, axes)
            means += sample.mean()
            stds += sample.std()
            num_samples += 1

            # reshape array
            sample = reshape_array(sample, axes)
            target = reshape_array(target, axes)

            # generate patches, return a generator
            patches, targets = extract_patches_sequential(
                sample, patch_size=patch_size, target=target
            )

            # convert generator to list and add to all_patches
            all_patches.append(patches)

            # ensure targets are not None (type checking)
            if targets is not None:
                all_targets.append(targets)
            else:
                raise ValueError(
                    f"No target found for {target_filename}."
                )

        except Exception as e:
            # emit warning and continue
            logger.error(f"Failed to read {train_filename} or {target_filename}: {e}")

    # raise error if no valid samples found
    if num_samples == 0 or len(all_patches) == 0:
        raise ValueError(
            f"No valid samples found in the input data: {train_files} and "
            f"{target_files}."
        )

    result_mean, result_std = means / num_samples, stds / num_samples

    patch_array: np.ndarray = np.concatenate(all_patches, axis=0)
    target_array: np.ndarray = np.concatenate(all_targets, axis=0)
    logger.info(f"Extracted {patch_array.shape[0]} patches from input array.")

    return (
        patch_array,
        target_array,
        result_mean,
        result_std,
    )


# called by in_memory_dataset
def prepare_patches_unsupervised(
    train_files: List[Path],
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
    read_source_func: Callable,
) -> Tuple[np.ndarray, None, float, float]:
    """
    Iterate over data source and create an array of patches.

    Returns
    -------
    np.ndarray
        Array of patches.
    """
    means, stds, num_samples = 0, 0, 0
    all_patches = []
    for filename in train_files:
        try:
            sample: np.ndarray = read_source_func(filename, axes)
            means += sample.mean()
            stds += sample.std()
            num_samples += 1

            # reshape array
            sample = reshape_array(sample, axes)

            # generate patches, return a generator
            patches, _ = extract_patches_sequential(sample, patch_size=patch_size)

            # convert generator to list and add to all_patches
            all_patches.append(patches)
        except Exception as e:
            # emit warning and continue
            logger.error(f"Failed to read {filename}: {e}")

    # raise error if no valid samples found
    if num_samples == 0:
        raise ValueError(f"No valid samples found in the input data: {train_files}.")

    result_mean, result_std = means / num_samples, stds / num_samples

    patch_array: np.ndarray = np.concatenate(all_patches)
    logger.info(f"Extracted {patch_array.shape[0]} patches from input array.")

    return patch_array, _, result_mean, result_std # TODO return object?


# called on arrays by in memory dataset
def prepare_patches_supervised_array(
    data: np.ndarray,
    axes: str,
    data_target: np.ndarray,
    patch_size: Union[List[int], Tuple[int]],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # compute statistics
    mean = data.mean()
    std = data.std()

    # reshape array
    sample = reshape_array(data, axes)

    # generate patches, return a generator
    patches, patch_targets = extract_patches_sequential(
        sample, patch_size=patch_size, target=data_target
    )

    if patch_targets is None:
        raise ValueError("No target extracted.")

    logger.info(f"Extracted {patches.shape[0]} patches from input array.")

    return (
        patches,
        patch_targets,
        mean,
        std,
    )


# called by in memory dataset
def prepare_patches_unsupervised_array(
    data: np.ndarray,
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
) -> Tuple[np.ndarray, None, float, float]:
    """
    Iterate over data source and create an array of patches.

    This method expects an array of shape SC(Z)YX, where S and C can be singleton
    dimensions.

    Patches returned are of shape SC(Z)YX, where S is now the patches dimension.
    
    Returns
    -------
    np.ndarray
        Array of patches.
    """
    # calculate mean and std
    mean = data.mean()
    std = data.std()

    # reshape array
    sample = reshape_array(data, axes)

    # generate patches, return a generator
    patches, _ = extract_patches_sequential(sample, patch_size=patch_size)

    return patches, _, mean, std # TODO inelegant, replace  by dataclass?


# prediction, both in memory and iterable
def generate_patches_predict(
    sample: np.ndarray,
    tile_size: Union[List[int], Tuple[int, ...]],
    tile_overlap: Union[List[int], Tuple[int, ...]],
) -> List[
    Tuple[np.ndarray, bool, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]
]:
    """
    Iterate over data source and create an array of patches.

    Returns
    -------
    np.ndarray
        Array of patches.
    """
    # generate patches, return a generator
    patches = extract_tiles(
        arr=sample, tile_size=tile_size, overlaps=tile_overlap
    )
    patches_list = list(patches)
    if len(patches_list) == 0:
        raise ValueError("No patch generated")

    return patches_list


# iterator over files
def generate_patches_supervised(
    sample: Union[np.ndarray, zarr.Array],
    patch_extraction_method: SupportedExtractionStrategy,
    patch_size: Union[List[int], Tuple[int, ...]],
    patch_overlap: Optional[Union[List[int], Tuple[int, ...]]] = None,
    target: Optional[Union[np.ndarray, zarr.Array]] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Creates an iterator with patches and corresponding targets from a sample.

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
    targets = None

    if patch_size is not None:
        patches = None

        if patch_extraction_method == SupportedExtractionStrategy.TILED:
            if patch_overlap is None:
                raise ValueError(
                    "Overlaps must be specified when using tiling (got None)."
                )
            patches = extract_tiles(
                arr=sample, tile_size=patch_size, overlaps=patch_overlap
            )

        elif patch_extraction_method == SupportedExtractionStrategy.SEQUENTIAL:
            patches, targets = extract_patches_sequential(
                arr=sample, patch_size=patch_size, target=target
            )

        elif patch_extraction_method == SupportedExtractionStrategy.RANDOM:
            # Returns a generator of patches and targets(if present)
            patches = extract_patches_random(
                arr=sample, patch_size=patch_size, target=target
            )

        elif patch_extraction_method == SupportedExtractionStrategy.RANDOM_ZARR:
            # Returns a generator of patches and targets(if present)
            patches = extract_patches_random_from_chunks(
                sample, patch_size=patch_size, chunk_size=sample.chunks
            )

        if patches is None:
            raise ValueError("No patch generated")

        return patches, targets
    else:
        # no patching
        return (sample for _ in range(1)), target


# iterator over files
def generate_patches_unsupervised(
    sample: Union[np.ndarray, zarr.Array],
    patch_extraction_method: SupportedExtractionStrategy,
    patch_size: Union[List[int], Tuple[int, ...]],
    patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Creates an iterator over patches from a sample.

    Parameters
    ----------
    sample : np.ndarray
        Input array.
    patch_extraction_method : SupportedExtractionStrategy
        Patch extraction methods (see `config.support`).
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
    # if tiled (patches with overlaps)
    if patch_extraction_method == SupportedExtractionStrategy.TILED:
        if patch_overlap is None:
            patch_overlap = [48] * len(patch_size)  # TODO pass overlap instead

        # return a Generator of the following:
        # - patch: np.ndarray, dims C(Z)YX
        # - last_tile: bool
        # - shape: Tuple[int], shape of a tile, excluding the S dimension
        # - overlap_crop_coords: coordinates used to crop the patch during stitching
        # - stitch_coords: coordinates used to stitch the tiles back to the full image
        patches = extract_tiles(
            arr=sample, tile_size=patch_size, overlaps=patch_overlap
        )

    # random extraction
    elif patch_extraction_method == SupportedExtractionStrategy.RANDOM:
        # return a Generator that yields the following:
        # - patch: np.ndarray, dimension C(Z)YX
        # - target_patch: np.ndarray, dimension C(Z)YX, or None
        patches = extract_patches_random(sample, patch_size=patch_size)

    # zarr specific random extraction
    elif patch_extraction_method == SupportedExtractionStrategy.RANDOM_ZARR:
        # # Returns a generator of patches and targets(if present)
        # patches = extract_patches_random_from_chunks(
        #     sample, patch_size=patch_size, chunk_size=sample.chunks
        # )
        raise NotImplementedError("Random zarr extraction not implemented yet.")

    # no patching, return sample
    elif patch_extraction_method == SupportedExtractionStrategy.NONE:
        patches = (sample for _ in range(1))

    # no extraction method
    else:
        raise ValueError("Invalid patch extraction method.")

    return patches
