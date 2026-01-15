"""Patching functions."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ...utils.logging import get_logger
from ..dataset_utils import reshape_array
from ..dataset_utils.running_stats import compute_normalization_stats
from .random_patching import extract_patches_random
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
    patching_strategy: str = "sequential",
    patching_seed: int | None = None,
    num_patches_per_sample: int | None = None,
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
    patching_strategy : str, default="sequential"
        Patching strategy to use. Options are "sequential" or "random".
    patching_seed : int or None, default=None
        Random seed for random patching. Only used when patching_strategy is "random".
    num_patches_per_sample : int or None, default=None
        Number of patches to extract per sample when using random patching. If None,
        automatically calculated. Only used when patching_strategy is "random".

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
            if patching_strategy == "random":
                # extract_patches_random returns a generator, collect results
                patch_generator = extract_patches_random(
                    sample,
                    patch_size=patch_size,
                    target=target,
                    seed=patching_seed,
                    num_patches_per_sample=num_patches_per_sample,
                )
                patches_list = []
                targets_list = []
                for patch, target_patch in patch_generator:
                    patches_list.append(patch)
                    if target_patch is not None:
                        targets_list.append(target_patch)
                patches = np.array(patches_list)
                targets = np.array(targets_list) if targets_list else None
            else:
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
    patching_strategy: str = "sequential",
    patching_seed: int | None = None,
    num_patches_per_sample: int | None = None,
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
    patching_strategy : str, default="sequential"
        Patching strategy to use. Options are "sequential" or "random".
    patching_seed : int or None, default=None
        Random seed for random patching. Only used when patching_strategy is "random".
    num_patches_per_sample : int or None, default=None
        Number of patches to extract per sample when using random patching. If None,
        automatically calculated. Only used when patching_strategy is "random".

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
            if patching_strategy == "random":
                # extract_patches_random returns a generator, so we need to collect
                # results
                patch_generator = extract_patches_random(
                    sample,
                    patch_size=patch_size,
                    seed=patching_seed,
                    num_patches_per_sample=num_patches_per_sample,
                )
                patches = np.array([patch for patch, _ in patch_generator])
            else:
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
    patching_strategy: str = "sequential",
    patching_seed: int | None = None,
    num_patches_per_sample: int | None = None,
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
    patching_strategy : str, default="sequential"
        Patching strategy to use. Options are "sequential" or "random".
    patching_seed : int or None, default=None
        Random seed for random patching. Only used when patching_strategy is "random".
    num_patches_per_sample : int or None, default=None
        Number of patches to extract per sample when using random patching. If None,
        automatically calculated. Only used when patching_strategy is "random".

    Returns
    -------
    PatchedOutput
        Patched output with supervised patches and targets.
    """
    # reshape the data
    reshaped_sample = reshape_array(data, axes)
    reshaped_target = reshape_array(data_target, axes)

    # extract patches with axes parameter
    if patching_strategy == "random":
        # extract_patches_random returns a generator, collect results
        patch_generator = extract_patches_random(
            reshaped_sample,
            patch_size=patch_size,
            target=reshaped_target,
            seed=patching_seed,
            num_patches_per_sample=num_patches_per_sample,
        )
        patches_list = []
        targets_list = []
        for patch, target_patch in patch_generator:
            patches_list.append(patch)
            if target_patch is not None:
                targets_list.append(target_patch)
        patches = np.array(patches_list)
        targets = np.array(targets_list) if targets_list else None
    else:
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
    patching_strategy: str = "sequential",
    patching_seed: int | None = None,
    num_patches_per_sample: int | None = None,
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
    patching_strategy : str, default="sequential"
        Patching strategy to use. Options are "sequential" or "random".
    patching_seed : int or None, default=None
        Random seed for random patching. Only used when patching_strategy is "random".
    num_patches_per_sample : int or None, default=None
        Number of patches to extract per sample when using random patching. If None,
        automatically calculated. Only used when patching_strategy is "random".

    Returns
    -------
    PatchedOutput
        Patches output.
    """
    # reshape array
    reshaped_sample = reshape_array(data, axes)

    means, stds = compute_normalization_stats(reshaped_sample)

    # generate patches based on strategy
    if patching_strategy == "random":
        # extract_patches_random returns a generator, so we need to collect results
        patch_generator = extract_patches_random(
            reshaped_sample,
            patch_size=patch_size,
            seed=patching_seed,
            num_patches_per_sample=num_patches_per_sample,
        )
        patches = np.array([patch for patch, _ in patch_generator])
    else:
        patches, _ = extract_patches_sequential(
            reshaped_sample, patch_size=patch_size, axes=axes
        )

    logger.info(f"Final: {patches.shape[0]} patches from input array.")
    return PatchedOutput(patches, None, Stats(means, stds), Stats((), ()))
