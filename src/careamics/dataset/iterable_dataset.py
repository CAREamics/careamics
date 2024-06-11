"""Iterable dataset used to load data file by file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np
from torch.utils.data import IterableDataset

from careamics.config import DataConfig
from careamics.transforms import Compose

from ..utils.logging import get_logger
from .dataset_utils import compute_normalization_stats, iterate_over_files, read_tiff
from .patching.patching import PatchedOutput, Stats
from .patching.random_patching import extract_patches_random

logger = get_logger(__name__)


class PathIterableDataset(IterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    src_files : List[Path]
        List of data files.
    target_files : Optional[List[Path]], optional
        Optional list of target files, by default None.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.

    Attributes
    ----------
    data_path : List[Path]
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    patch_extraction_method : Union[ExtractionStrategies, None]
        Patch extraction strategy, as defined in extraction_strategy.
    patch_size : Optional[Union[List[int], Tuple[int]]], optional
        Size of the patches in each dimension, by default None.
    patch_overlap : Optional[Union[List[int], Tuple[int]]], optional
        Overlap of the patches in each dimension, by default None.
    mean : Optional[float], optional
        Expected mean of the dataset, by default None.
    std : Optional[float], optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Optional[Callable], optional
        Patch transform callable, by default None.
    """

    def __init__(
        self,
        data_config: DataConfig,
        src_files: List[Path],
        target_files: Optional[List[Path]] = None,
        read_source_func: Callable = read_tiff,
    ) -> None:
        """Constructors.

        Parameters
        ----------
        data_config : DataConfig
            Data configuration.
        src_files : List[Path]
            List of data files.
        target_files : Optional[List[Path]], optional
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
        # Only checking the image_mean because the DatasetConfig class ensures that
        # if image_mean is provided, image_std is also provided
        if not self.data_config.image_mean:
            self.patches_data = self._calculate_mean_and_std()
            logger.info(
                f"Computed dataset mean: {self.patches_data.image_stats.means},"
                f"std: {self.patches_data.image_stats.stds}"
            )

        else:
            self.patches_data = PatchedOutput(
                None,
                None,
                Stats(self.data_config.image_mean, self.data_config.image_std),
                Stats(self.data_config.target_mean, self.data_config.target_std),
            )

        if hasattr(self.data_config, "set_mean_and_std"):
            self.data_config.set_mean_and_std(
                image_mean=list(self.patches_data.image_stats.means),
                image_std=list(self.patches_data.image_stats.stds),
                target_mean=(
                    list(self.patches_data.target_stats.means)
                    if self.patches_data.target_stats.means is not None
                    else []
                ),
                target_std=(
                    list(self.patches_data.target_stats.stds)
                    if self.patches_data.target_stats.stds is not None
                    else []
                ),
            )

        # get transforms
        self.patch_transform = Compose(transform_list=data_config.transforms)

    def _calculate_mean_and_std(self) -> PatchedOutput:
        """
        Calculate mean and std of the dataset.

        Returns
        -------
        PatchedOutput
            Data class containing the image statistics.
        """
        image_means, image_stds, target_means, target_stds = 0, 0, 0, 0
        num_samples = 0

        for sample, target in _iterate_over_files(
            self.data_config, self.data_files, self.target_files, self.read_source_func
        ):
            sample = reshape_array(sample, self.data_config.axes)
            target = (
                None if target is None else reshape_array(target, self.data_config.axes)
            )

            sample_mean, sample_std = compute_normalization_stats(sample)
            image_means += sample_mean
            image_stds += sample_std

            if target is not None:
                target_mean, target_std = compute_normalization_stats(target)
                target_means += target_mean
                target_stds += target_std

            num_samples += 1

        if num_samples == 0:
            raise ValueError("No samples found in the dataset.")

        # Average the means and stds per channel
        for ch in range(sample.shape[0]):
            image_means[ch] /= num_samples
            image_stds[ch] /= num_samples
            if target is not None:
                target_means[ch] /= num_samples
                target_stds[ch] /= num_samples

        logger.info(f"Calculated mean and std for {num_samples} images")
        logger.info(f"Mean: {image_means}, std: {image_stds}")
        return PatchedOutput(
            None,
            None,
            Stats(image_means, image_stds),
            Stats(
                target_means if target is not None else None,
                target_stds if target is not None else None,
            ),
        )

    def __iter__(
        self,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
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


class IterablePredictionDataset(IterableDataset):
    """
    Prediction dataset.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Inference configuration.
    src_files : List[Path]
        List of data files.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.
    **kwargs : Any
        Additional keyword arguments, unused.

    Attributes
    ----------
    data_path : Union[str, Path]
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    mean : Optional[float], optional
        Expected mean of the dataset, by default None.
    std : Optional[float], optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Optional[Callable], optional
        Patch transform callable, by default None.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        src_files: List[Path],
        read_source_func: Callable = read_tiff,
        **kwargs: Any,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Inference configuration.
        src_files : List[Path]
            List of data files.
        read_source_func : Callable, optional
            Read source function for custom types, by default read_tiff.
        **kwargs : Any
            Additional keyword arguments, unused.

        Raises
        ------
        ValueError
            If mean and std are not provided in the inference configuration.
        """
        self.prediction_config = prediction_config
        self.data_files = src_files
        self.axes = prediction_config.axes
        self.tile_size = self.prediction_config.tile_size
        self.tile_overlap = self.prediction_config.tile_overlap
        self.read_source_func = read_source_func
        self.image_means = self.prediction_config.image_mean
        self.image_stds = self.prediction_config.image_std

        # tile only if both tile size and overlaps are provided
        self.tile = self.tile_size is not None and self.tile_overlap is not None

        # check mean and std and create normalize transform
        if (
            self.prediction_config.image_mean is None
            or self.prediction_config.image_std is None
        ):
            raise ValueError("Mean and std must be provided for prediction.")
        else:
            self.mean = self.prediction_config.image_mean
            self.std = self.prediction_config.image_std

            # instantiate normalize transform
            self.patch_transform = Compose(
                transform_list=[
                    NormalizeModel(
                        image_means=prediction_config.image_mean,
                        image_stds=prediction_config.image_std,
                    )
                ],
            )

    def __iter__(
        self,
    ) -> Generator[Tuple[np.ndarray, TileInformation], None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
            Single patch.
        """
        assert (
            self.image_means is not None and self.image_stds is not None
        ), "Mean and std must be provided"

        for sample, _ in _iterate_over_files(
            self.prediction_config,
            self.data_files,
            read_source_func=self.read_source_func,
        ):
            # reshape array
            reshaped_sample = reshape_array(sample, self.axes)

            if (
                self.tile
                and self.tile_size is not None
                and self.tile_overlap is not None
            ):
                # generate patches, return a generator
                patch_gen = extract_tiles(
                    arr=reshaped_sample,
                    tile_size=self.tile_size,
                    overlaps=self.tile_overlap,
                )
            else:
                # just wrap the sample in a generator with default tiling info
                array_shape = reshaped_sample.squeeze().shape
                patch_gen = (
                    (reshaped_sample, TileInformation(array_shape=array_shape))
                    for _ in range(1)
                )

            # apply transform to patches
            for patch_array, tile_info in patch_gen:
                transformed_patch, _ = self.patch_transform(patch=patch_array)

                yield transformed_patch, tile_info
