"""Patching functions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np

from ...utils.logging import get_logger
from ..dataset_utils import reshape_array
from ..dataset_utils.running_stats import compute_normalization_stats
from .sequential_patching import extract_patches_sequential

logger = get_logger(__name__)


@dataclass
class Stats:
    """Dataclass to store statistics."""

    means: Union[np.ndarray, tuple, list, None]
    stds: Union[np.ndarray, tuple, list, None]


@dataclass
class PatchedOutput:
    """Dataclass to store patches and statistics."""

    patches: Union[np.ndarray]
    targets: Union[np.ndarray, None]
    image_stats: Stats
    target_stats: Stats


@dataclass
class StatsOutput:
    """Dataclass to store patches and statistics."""

    image_stats: Stats
    target_stats: Stats


# called by in memory dataset
def prepare_patches_supervised(
    train_files: List[Path],
    target_files: List[Path],
    axes: str,
    patch_size: Union[List[int], Tuple[int, ...]],
    read_source_func: Callable,
) -> PatchedOutput:
    """
    Iterate over data source and create an array of patches and corresponding targets.

    The lists of Paths should be pre-sorted.

    Parameters
    ----------
    train_files : List[Path]
        List of paths to training data.
    target_files : List[Path]
        List of paths to target data.
    axes : str
        Axes of the data.
    patch_size : Union[List[int], Tuple[int]]
        Size of the patches.
    read_source_func : Callable
        Function to read the data.

    Returns
    -------
    np.ndarray
        Array of patches.
    """
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

    image_means, image_stds = compute_normalization_stats(np.concatenate(all_patches))
    target_means, target_stds = compute_normalization_stats(np.concatenate(all_targets))

    patch_array: np.ndarray = np.concatenate(all_patches, axis=0)
    target_array: np.ndarray = np.concatenate(all_targets, axis=0)
    logger.info(f"Extracted {patch_array.shape[0]} patches from input array.")

    return PatchedOutput(
        patch_array,
        target_array,
        Stats(image_means, image_stds),
        Stats(target_means, target_stds),
    )


# called by in_memory_dataset
def prepare_patches_unsupervised(
    train_files: List[Path],
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
    read_source_func: Callable,
) -> PatchedOutput:
    """Iterate over data source and create an array of patches.

    This method returns the mean and standard deviation of the image.

    Parameters
    ----------
    train_files : List[Path]
        List of paths to training data.
    axes : str
        Axes of the data.
    patch_size : Union[List[int], Tuple[int]]
        Size of the patches.
    read_source_func : Callable
        Function to read the data.

    Returns
    -------
    Tuple[np.ndarray, None, float, float]
        Source and target patches, mean and standard deviation.
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

    image_means, image_stds = compute_normalization_stats(np.concatenate(all_patches))

    patch_array: np.ndarray = np.concatenate(all_patches)
    logger.info(f"Extracted {patch_array.shape[0]} patches from input array.")

    return PatchedOutput(
        patch_array, None, Stats(image_means, image_stds), Stats((), ())
    )


# called on arrays by in memory dataset
def prepare_patches_supervised_array(
    data: np.ndarray,
    axes: str,
    data_target: np.ndarray,
    patch_size: Union[List[int], Tuple[int]],
) -> PatchedOutput:
    """Iterate over data source and create an array of patches.

    This method expects an array of shape SC(Z)YX, where S and C can be singleton
    dimensions.

    Patches returned are of shape SC(Z)YX, where S is now the patches dimension.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    axes : str
        Axes of the data.
    data_target : np.ndarray
        Target data array.
    patch_size : Union[List[int], Tuple[int]]
        Size of the patches.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, float]
        Source and target patches, mean and standard deviation.
    """
    # reshape array
    reshaped_sample = reshape_array(data, axes)
    reshaped_target = reshape_array(data_target, axes)

    # compute statistics
    image_means, image_stds = compute_normalization_stats(reshaped_sample)
    target_means, target_stds = compute_normalization_stats(reshaped_target)

    # generate patches, return a generator
    patches, patch_targets = extract_patches_sequential(
        reshaped_sample, patch_size=patch_size, target=reshaped_target
    )

    if patch_targets is None:
        raise ValueError("No target extracted.")

    logger.info(f"Extracted {patches.shape[0]} patches from input array.")

    return PatchedOutput(
        patches,
        patch_targets,
        Stats(image_means, image_stds),
        Stats(target_means, target_stds),
    )


# called by in memory dataset
def prepare_patches_unsupervised_array(
    data: np.ndarray,
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
) -> PatchedOutput:
    """
    Iterate over data source and create an array of patches.

    This method expects an array of shape SC(Z)YX, where S and C can be singleton
    dimensions.

    Patches returned are of shape SC(Z)YX, where S is now the patches dimension.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    axes : str
        Axes of the data.
    patch_size : Union[List[int], Tuple[int]]
        Size of the patches.

    Returns
    -------
    Tuple[np.ndarray, None, float, float]
        Source patches, mean and standard deviation.
    """
    # reshape array
    reshaped_sample = reshape_array(data, axes)

    # calculate mean and std
    means, stds = compute_normalization_stats(reshaped_sample)

    # generate patches, return a generator
    patches, _ = extract_patches_sequential(reshaped_sample, patch_size=patch_size)

    return PatchedOutput(patches, None, Stats(means, stds), Stats((), ()))
