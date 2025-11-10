"""Patching functions."""

from collections.abc import Callable
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ...utils.logging import get_logger
from ..dataset_utils import reshape_array
from ..dataset_utils.running_stats import compute_normalization_stats
from .sequential_patching import extract_patches_sequential

logger = get_logger(__name__)


@dataclass
class Stats:
    """Dataclass to store statistics."""

    means: Union[NDArray, tuple, list, None]
    """Mean of the data across channels."""

    stds: Union[NDArray, tuple, list, None]
    """Standard deviation of the data across channels."""

    def get_statistics(self) -> tuple[list[float], list[float]]:
        """Return the means and standard deviations.

        Returns
        -------
        tuple of two lists of floats
            Means and standard deviations.
        """
        if self.means is None or self.stds is None:
            return [], []

        return list(self.means), list(self.stds)


@dataclass
class PatchedOutput:
    """Dataclass to store patches and statistics."""

    patches: Union[NDArray]
    """Image patches."""

    targets: Union[NDArray, None]
    """Target patches."""

    image_stats: Stats
    """Statistics of the image patches."""

    target_stats: Stats
    """Statistics of the target patches."""


def prepare_patches_supervised(
    train_files: list[Path],
    target_files: list[Path],
    axes: str,
    patch_size: Union[list[int], tuple[int, ...]],
    read_source_func: Callable,
) -> PatchedOutput:
    """
    Iterate over data source and create an array of patches and corresponding targets.

    The lists of Paths should be pre-sorted.

    Parameters
    ----------
    train_files : list of pathlib.Path
        List of paths to training data.
    target_files : list of pathlib.Path
        List of paths to target data.
    axes : str
        Axes of the data.
    patch_size : list or tuple of int
        Size of the patches.
    read_source_func : Callable
        Function to read the data.

    Returns
    -------
    PatchedOutput
        Patched output with supervised patches and targets.
    """
    means, stds, num_samples = 0, 0, 0
    all_patches, all_targets = [], []
    for train_filename, target_filename in zip(train_files, target_files, strict=False):
        try:
            sample: np.ndarray = read_source_func(train_filename, axes)
            target: np.ndarray = read_source_func(target_filename, axes)
            means += sample.mean()
            stds += sample.std()
            num_samples += 1

            # reshape array
            sample = reshape_array(sample, axes)
            target = reshape_array(target, axes)

            # generate patches with axes parameter
            patches, targets = extract_patches_sequential(
                sample, patch_size=patch_size, target=target, axes=axes
            )

            all_patches.append(patches)

            if targets is not None:
                all_targets.append(targets)
            else:
                raise ValueError(f"No target found for {target_filename}.")

        except Exception as e:
            logger.error(f"Failed to read {train_filename} or {target_filename}: {e}")

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


def prepare_patches_unsupervised(
    train_files: list[Path],
    axes: str,
    patch_size: Union[list[int], tuple[int]],
    read_source_func: Callable,
) -> PatchedOutput:
    """Iterate over data source and create an array of patches.

    This method returns the mean and standard deviation of the image.

    Parameters
    ----------
    train_files : list of pathlib.Path
        List of paths to training data.
    axes : str
        Axes of the data.
    patch_size : list or tuple of int
        Size of the patches.
    read_source_func : Callable
        Function to read the data.

    Returns
    -------
    PatchedOutput
        Dataclass holding patches and their statistics.
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

            # generate patches - use axes parameter for 1D compatibility
            patches, _ = extract_patches_sequential(
                sample, patch_size=patch_size, axes=axes
            )

            all_patches.append(patches)
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")

    if num_samples == 0:
        raise ValueError(f"No valid samples found in the input data: {train_files}.")

    image_means, image_stds = compute_normalization_stats(np.concatenate(all_patches))

    patch_array: np.ndarray = np.concatenate(all_patches)
    logger.info(f"Extracted {patch_array.shape[0]} patches from input array.")

    return PatchedOutput(
        patch_array, None, Stats(image_means, image_stds), Stats((), ())
    )


def prepare_patches_supervised_array(
    data: NDArray,
    axes: str,
    data_target: NDArray,
    patch_size: Union[list[int], tuple[int]],
) -> PatchedOutput:
    """
    Prepare patches for supervised training from arrays.

    Updated to support 1D, 2D, and 3D data through axes parameter.

    Parameters
    ----------
    data : NDArray
        Input data.
    axes : str
        Axes string describing data dimensions.
    data_target : NDArray
        Target data.
    patch_size : Union[list[int], tuple[int]]
        Patch size for spatial dimensions.

    Returns
    -------
    PatchedOutput
        Patched output with supervised patches and targets.
    """
    # reshape the data
    reshaped_sample = reshape_array(data, axes)
    reshaped_target = reshape_array(data_target, axes)

    # extract patches with axes parameter
    patches, targets = extract_patches_sequential(
        reshaped_sample, patch_size=patch_size, target=reshaped_target, axes=axes
    )

    # compute statistics
    means, stds = compute_normalization_stats(patches)
    target_means, target_stds = compute_normalization_stats(targets)

    logger.info(f"Extracted {patches.shape[0]} patches from input array.")

    return PatchedOutput(
        patches=patches,
        targets=targets,
        image_stats=Stats(means=means, stds=stds),
        target_stats=Stats(means=target_means, stds=target_stds),
    )


def prepare_patches_unsupervised_array(
    data: NDArray,
    axes: str,
    patch_size: Union[list[int], tuple[int]],
) -> PatchedOutput:
    """
    Prepare patches from array for unsupervised training.

    Parameters
    ----------
    data : NDArray
        Input array.
    axes : str
        Axes description.
    patch_size : list of int or tuple of int
        Patch size.

    Returns
    -------
    PatchedOutput
        Patches output.
    """
    # reshape array

    reshaped_sample = reshape_array(data, axes)

    means, stds = compute_normalization_stats(reshaped_sample)

    patches, _ = extract_patches_sequential(
        reshaped_sample, patch_size=patch_size, axes=axes
    )

    logger.info(f"Final: {patches.shape[0]} patches from input array.")
    return PatchedOutput(patches, None, Stats(means, stds), Stats((), ()))