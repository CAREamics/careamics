"""
Tiling submodule.

These functions are used to tile images into patches or tiles.
"""
from pathlib import Path
from typing import Callable, Generator, List, Iterator, Optional, Tuple, Union

import numpy as np
import zarr

from ...utils.logging import get_logger
from ...config.support.supported_extraction_strategies import SupportedExtractionStrategy

from .sequential_patching import extract_patches_sequential
from .random_patching import extract_patches_random, extract_patches_random_from_chunks
from .tiled_patching import extract_tiles

logger = get_logger(__name__)

# TODO should we overload or singlepatch the functions?

def prepare_patches_supervised(
    train_files: List[Path],
    target_files: List[Path],
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
    read_source_func: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, float]:
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
        sample: np.ndarray = read_source_func(train_filename, axes)
        target: np.ndarray = read_source_func(target_filename, axes)
        means += sample.mean()
        stds += sample.std()
        num_samples += 1

        # generate patches, return a generator
        patches, targets = extract_patches_sequential(
            sample, axes, patch_size=patch_size, target=target
        )

        # convert generator to list and add to all_patches
        all_patches.append(patches)
        all_targets.append(targets)

    result_mean, result_std = means / num_samples, stds / num_samples

    all_patches = np.concatenate(all_patches, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    logger.info(f"Extracted {all_patches.shape[0]} patches from input array.")

    return (
        all_patches,
        all_targets,
        result_mean,
        result_std,
    )

def prepare_patches_supervised_array(
    data: np.ndarray,
    data_target: np.ndarray,
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
) -> Tuple[np.ndarray, float, float]:

    # compute statistics
    mean = data.mean()
    std = data.std()

    # generate patches, return a generator
    patches, patch_targets = extract_patches_sequential(
        data, axes, patch_size=patch_size, target=data_target
    )

    logger.info(f"Extracted {patches.shape[0]} patches from input array.")

    return (
        patches,
        patch_targets,
        mean,
        std,
    )


def prepare_patches_unsupervised(
    train_files: List[Path],
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
    read_source_func: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, float]:
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
        sample = read_source_func(filename, axes)
        means += sample.mean()
        stds += np.std(sample)
        num_samples += 1

        # generate patches, return a generator
        patches, _ = extract_patches_sequential(sample, axes, patch_size=patch_size)

        # convert generator to list and add to all_patches
        all_patches.append(patches)

        result_mean, result_std = means / num_samples, stds / num_samples
    return np.concatenate(all_patches), _, result_mean, result_std


def prepare_patches_unsupervised_array(
    data: np.ndarray,
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
) -> Tuple[np.ndarray, float, float]:
    """
    Iterate over data source and create an array of patches.

    Returns
    -------
    np.ndarray
        Array of patches.
    """
    # calculate mean and std
    means = data.mean()
    stds = data.std()

    # generate patches, return a generator
    patches, _ = extract_patches_sequential(data, axes, patch_size=patch_size)

    return np.concatenate(patches), _, means, stds


def generate_patches_predict(
    sample: np.ndarray,
    axes: str,
    tile_size: Union[List[int], Tuple[int]],
    tile_overlap: Union[List[int], Tuple[int]],
) -> Tuple[np.ndarray, float, float]:
    """
    Iterate over data source and create an array of patches.

    Returns
    -------
    np.ndarray
        Array of patches.
    """
    # generate patches, return a generator
    patches = extract_tiles(
        arr=sample, axes=axes, tile_size=tile_size, overlaps=tile_overlap
    )
    patches_list = list(patches)
    if len(patches_list) == 0:
        raise ValueError("No patch generated")

    return patches_list


def generate_patches_supervised(
    sample: Union[np.ndarray, zarr.Array],
    axes: str,
    patch_extraction_method: SupportedExtractionStrategy,
    patch_size: Optional[Union[List[int], Tuple[int]]] = None,
    patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
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
                arr=sample, axes=axes, tile_size=patch_size, overlaps=patch_overlap
            )

        elif patch_extraction_method == SupportedExtractionStrategy.SEQUENTIAL:
            patches, targets = extract_patches_sequential(
                arr=sample, axes=axes, patch_size=patch_size, target=target
            )

        elif patch_extraction_method == SupportedExtractionStrategy.RANDOM:
            # Returns a generator of patches and targets(if present)
            patches = extract_patches_random(
                arr=sample, axes=axes, patch_size=patch_size, target=target
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


def generate_patches_unsupervised(
    sample: Union[np.ndarray, zarr.Array],
    axes: str,
    patch_extraction_method: SupportedExtractionStrategy,
    patch_size: Optional[Union[List[int], Tuple[int]]] = None,
    patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Creates an iterator over patches from a sample.

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

    if patch_extraction_method is not None:
        patches = None

        if patch_extraction_method == SupportedExtractionStrategy.TILED:
            if patch_overlap is None:
                patch_overlap = [48] * len(patch_size)# TODO calculate OL from model
            patches = extract_tiles(
                arr=sample, axes=axes, tile_size=patch_size, overlaps=patch_overlap
            )
        # TODO split so there's no extraciton strat param
        elif patch_extraction_method == SupportedExtractionStrategy.RANDOM:
            # Returns a generator of patches and targets(if present)
            patches = extract_patches_random(sample, patch_size=patch_size)

        elif patch_extraction_method == SupportedExtractionStrategy.RANDOM_ZARR:
            # Returns a generator of patches and targets(if present)
            patches = extract_patches_random_from_chunks(
                sample, patch_size=patch_size, chunk_size=sample.chunks
            )

        else:
            raise ValueError("Invalid patch extraction method")

        if patches is None:
            raise ValueError("No patch generated")

        return patches
    else:
        # no patching. sample should have channel dimension
        return (sample for _ in range(1))

