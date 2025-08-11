"""Iterable dataset used to load data file by file."""

from __future__ import annotations

import copy
from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset

from careamics.config import DataConfig
from careamics.config.transformations import NormalizeModel
from careamics.file_io.read import read_tiff
from careamics.transforms import Compose

from ..utils.logging import get_logger
from .dataset_utils import iterate_over_files
from .dataset_utils.running_stats import WelfordStatistics
from .patching.patching import Stats
from .patching.random_patching import extract_patches_random

logger = get_logger(__name__)


class PathIterableDataset(IterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    src_files : list of pathlib.Path
        List of data files.
    target_files : list of pathlib.Path, optional
        Optional list of target files, by default None.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.

    Attributes
    ----------
    data_path : list of pathlib.Path
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    """

    def __init__(
        self,
        data_config: DataConfig,
        src_files: list[Path],
        target_files: list[Path] | None = None,
        read_source_func: Callable = read_tiff,
    ) -> None:
        """Constructors.

        Parameters
        ----------
        data_config : GeneralDataConfig
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
        self.read_source_func = read_source_func

        # compute mean and std over the dataset
        # only checking the image_mean because the DataConfig class ensures that
        # if image_mean is provided, image_std is also provided
        if not self.data_config.image_means:
            self.image_stats, self.target_stats = self._calculate_mean_and_std()
            logger.info(
                f"Computed dataset mean: {self.image_stats.means},"
                f"std: {self.image_stats.stds}"
            )

            # update the mean in the config
            self.data_config.set_means_and_stds(
                image_means=self.image_stats.means,
                image_stds=self.image_stats.stds,
                target_means=(
                    list(self.target_stats.means)
                    if self.target_stats.means is not None
                    else None
                ),
                target_stds=(
                    list(self.target_stats.stds)
                    if self.target_stats.stds is not None
                    else None
                ),
            )

        else:
            # if mean and std are provided in the config, use them
            self.image_stats, self.target_stats = (
                Stats(self.data_config.image_means, self.data_config.image_stds),
                Stats(self.data_config.target_means, self.data_config.target_stds),
            )

        # create transform composed of normalization and other transforms
        self.patch_transform = Compose(
            transform_list=[
                NormalizeModel(
                    image_means=self.image_stats.means,
                    image_stds=self.image_stats.stds,
                    target_means=self.target_stats.means,
                    target_stds=self.target_stats.stds,
                )
            ]
            + list(data_config.transforms)
        )

    def _calculate_mean_and_std(self) -> tuple[Stats, Stats]:
        """
        Calculate mean and std of the dataset.

        Returns
        -------
        tuple of Stats and optional Stats
            Data classes containing the image and target statistics.
        """
        num_samples = 0
        image_stats = WelfordStatistics()
        if self.target_files is not None:
            target_stats = WelfordStatistics()

        for sample, target in iterate_over_files(
            self.data_config, self.data_files, self.target_files, self.read_source_func
        ):
            # update the image statistics
            image_stats.update(sample, num_samples)

            # update the target statistics if target is available
            if target is not None:
                target_stats.update(target, num_samples)

            num_samples += 1

        if num_samples == 0:
            raise ValueError("No samples found in the dataset.")

        # Average the means and stds per sample
        image_means, image_stds = image_stats.finalize()

        if target is not None:
            target_means, target_stds = target_stats.finalize()

            return (
                Stats(image_means, image_stds),
                Stats(np.array(target_means), np.array(target_stds)),
            )
        else:
            return Stats(image_means, image_stds), Stats(None, None)

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
            self.image_stats.means is not None and self.image_stats.stds is not None
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

    def get_data_statistics(self) -> tuple[list[float], list[float]]:
        """Return training data statistics.

        Returns
        -------
        tuple of list of floats
            Means and standard deviations across channels of the training data.
        """
        return self.image_stats.get_statistics()

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
