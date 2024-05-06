"""
Tiling submodule.

These functions are used to tile images into patches or tiles.
"""

from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np

from ...utils.logging import get_logger
from ..dataset_utils import reshape_array
from .sequential_patching import extract_patches_sequential

logger = get_logger(__name__)


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
                raise ValueError(f"No target found for {target_filename}.")

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

    return patch_array, _, result_mean, result_std  # TODO return object?


# called on arrays by in memory dataset
def prepare_patches_supervised_array(
    data: np.ndarray,
    axes: str,
    data_target: np.ndarray,
    patch_size: Union[List[int], Tuple[int]],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Iterate over data source and create an array of patches.

    This method expects an array of shape SC(Z)YX, where S and C can be singleton
    dimensions.

    Patches returned are of shape SC(Z)YX, where S is now the patches dimension.

    Returns
    -------
    np.ndarray
        Array of patches.
    """
    # compute statistics
    mean = data.mean()
    std = data.std()

    # reshape array
    reshaped_sample = reshape_array(data, axes)
    reshaped_target = reshape_array(data_target, axes)

    # generate patches, return a generator
    patches, patch_targets = extract_patches_sequential(
        reshaped_sample, patch_size=patch_size, target=reshaped_target
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
    reshaped_sample = reshape_array(data, axes)

    # generate patches, return a generator
    patches, _ = extract_patches_sequential(reshaped_sample, patch_size=patch_size)

    return patches, _, mean, std  # TODO inelegant, replace  by dataclass?
