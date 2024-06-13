"""Iterable dataset used to load data file by file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np
from torch.utils.data import IterableDataset

from careamics.config import DataConfig
from careamics.config.transformations import NormalizeModel
from careamics.transforms import Compose

from ..utils.logging import get_logger
from .dataset_utils import compute_normalization_stats, iterate_over_files, read_tiff
from .patching.patching import Stats, StatsOutput
from .patching.random_patching import extract_patches_random

logger = get_logger(__name__)


class PathIterableDataset(IterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    src_files : list[Path]
        List of data files.
    target_files : list[Path] or None, optional
        Optional list of target files, by default None.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.

    Attributes
    ----------
    data_path : list[Path]
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    patch_size : list[int] or tuple[int] or None, optional
        Size of the patches in each dimension, by default None.
    patch_overlap : list[int] or tuple[int] or None, optional
        Overlap of the patches in each dimension, by default None.
    mean : float or None, optional
        Expected mean of the dataset, by default None.
    std : float or None, optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Callable or None, optional
        Patch transform callable, by default None.
    """

    def __init__(
        self,
        data_config: DataConfig,
        src_files: list[Path],
        target_files: Optional[list[Path]] = None,
        read_source_func: Callable = read_tiff,
    ) -> None:
        """Constructors.

        Parameters
        ----------
        data_config : DataConfig
            Data configuration.
        src_files : list[Path]
            List of data files.
        target_files : list[Path] or None, optional
            Optional list of target files, by default None.
        read_source_func : Callable, optional
            Read source function for custom types, by default read_tiff.
        """
        self.data_config = data_config
        self.data_files = src_files
        self.target_files = target_files
        self.data_config = data_config
        self.read_source_func = read_source_func

        # compute mean and std over the dataset
        # Only checking the image_mean because the DataConfig class ensures that
        # if image_mean is provided, image_std is also provided
        if not self.data_config.image_mean:
            self.patches_data = self._calculate_mean_and_std()
            logger.info(
                f"Computed dataset mean: {self.patches_data.image_stats.means},"
                f"std: {self.patches_data.image_stats.stds}"
            )

        else:
            self.patches_data = StatsOutput(
                Stats(self.data_config.image_mean, self.data_config.image_std),
                Stats(self.data_config.target_mean, self.data_config.target_std),
            )

        if hasattr(self.data_config, "set_mean_and_std"):
            self.data_config.set_mean_and_std(
                image_mean=self.patches_data.image_stats.means,
                image_std=self.patches_data.image_stats.stds,
                target_mean=(
                    tuple(self.patches_data.target_stats.means)
                    if self.patches_data.target_stats.means is not None
                    else []
                ),
                target_std=(
                    tuple(self.patches_data.target_stats.stds)
                    if self.patches_data.target_stats.stds is not None
                    else []
                ),
            )

        # get transforms
        self.patch_transform = Compose(
            transform_list=[
                NormalizeModel(mean=self.mean, std=self.std),
            ]
            + data_config.transforms
        )

    def _calculate_mean_and_std(self) -> StatsOutput:
        """
        Calculate mean and std of the dataset.

        Returns
        -------
        PatchedOutput
            Data class containing the image statistics.
        """
        image_means = image_stds = target_means = target_stds = []
        num_samples = 0

        for sample, target in iterate_over_files(
            self.data_config, self.data_files, self.target_files, self.read_source_func
        ):
            sample_mean, sample_std = compute_normalization_stats(sample)
            image_means.append(sample_mean)
            image_stds.append(sample_std)

            if target is not None:
                target_mean, target_std = compute_normalization_stats(target)
                target_means.append(target_mean)
                target_stds.append(target_std)

            num_samples += 1

        if num_samples == 0:
            raise ValueError("No samples found in the dataset.")

        # Average the means and stds per sample
        image_means = np.mean(image_means, axis=0)
        image_stds = np.mean([std**2 for std in image_stds], axis=0)

        if target is not None:
            target_means = np.mean(target_means, axis=0)
            target_stds = np.mean([std**2 for std in image_stds], axis=0)

        logger.info(f"Calculated mean and std for {num_samples} images")
        logger.info(f"Mean: {image_means}, std: {image_stds}")
        return StatsOutput(
            Stats(image_means, image_stds),
            Stats(
                np.array(target_means) if target is not None else None,
                np.array(target_stds) if target is not None else None,
            ),
        )

    def __iter__(
        self,
    ) -> Generator[tuple[np.ndarray, ...], None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
            Single patch.
        """
        assert (
            self.patches_data.image_stats.means is not None
            and self.patches_data.image_stats.stds is not None
        ), "Mean and std must be provided"

        # iterate over files
        for sample_input, sample_target in iterate_over_files(
            self.data_config, self.data_files, self.target_files, self.read_source_func
        ):
            patches = extract_patches_random(
                arr=sample_input,
                patch_size=self.data_config.patch_size,
                target=sample_target,
            )

            # iterate over patches
            # patches are tuples of (patch, target) if target is available
            # or (patch, None) only if no target is available
            # patch is of dimensions (C)ZYX
            for patch_data in patches:
                yield self.patch_transform(
                    patch=patch_data[0],
                    target=patch_data[1],
                )

    def get_number_of_files(self) -> int:
        """
        Return the number of files in the dataset.

        Returns
        -------
        int
            Number of files in the dataset.
        """
        return len(self.data_files)

    def split_dataset(
        self,
        percentage: float = 0.1,
        minimum_number: int = 5,
    ) -> PathIterableDataset:
        """Split up dataset in two.

        Splits the datest sing a percentage of the data (files) to extract, or the
        minimum number of the percentage is less than the minimum number.

        Parameters
        ----------
        percentage : float, optional
            Percentage of files to split up, by default 0.1.
        minimum_number : int, optional
            Minimum number of files to split up, by default 5.

        Returns
        -------
        IterableDataset
            Dataset containing the split data.

        Raises
        ------
        ValueError
            If the percentage is smaller than 0 or larger than 1.
        ValueError
            If the minimum number is smaller than 1 or larger than the number of files.
        """
        if percentage < 0 or percentage > 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}.")

        if minimum_number < 1 or minimum_number > self.get_number_of_files():
            raise ValueError(
                f"Minimum number of files must be between 1 and "
                f"{self.get_number_of_files()} (number of files), got "
                f"{minimum_number}."
            )

        # compute number of files
        total_files = self.get_number_of_files()
        n_files = max(round(percentage * total_files), minimum_number)

        # get random indices
        indices = np.random.choice(total_files, n_files, replace=False)

        # extract files
        val_files = [self.data_files[i] for i in indices]

        # remove patches from self.patch
        data_files = []
        for i, file in enumerate(self.data_files):
            if i not in indices:
                data_files.append(file)
        self.data_files = data_files

        # same for targets
        if self.target_files is not None:
            val_target_files = [self.target_files[i] for i in indices]

            data_target_files = []
            for i, file in enumerate(self.target_files):
                if i not in indices:
                    data_target_files.append(file)
            self.target_files = data_target_files

        # clone the dataset
        dataset = copy.deepcopy(self)

        # reassign patches
        dataset.data_files = val_files

        # reassign targets
        if self.target_files is not None:
            dataset.target_files = val_target_files

        return dataset
