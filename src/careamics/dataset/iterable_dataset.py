from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from careamics.transforms import Compose

from ..config import DataConfig, InferenceConfig
from ..config.tile_information import TileInformation
from ..utils.logging import get_logger
from .dataset_utils import read_tiff, reshape_array
from .patching.random_patching import extract_patches_random
from .patching.tiled_patching import extract_tiles

logger = get_logger(__name__)


class PathIterableDataset(IterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
    ----------
    data_path : Union[str, Path]
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
        data_config: Union[DataConfig, InferenceConfig],
        src_files: List[Path],
        target_files: Optional[List[Path]] = None,
        read_source_func: Callable = read_tiff,
    ) -> None:
        self.data_config = data_config
        self.data_files = src_files
        self.target_files = target_files
        self.data_config = data_config
        self.read_source_func = read_source_func

        # compute mean and std over the dataset
        if not data_config.mean or not data_config.std:
            self.mean, self.std = self._calculate_mean_and_std()

            # update mean and std in configuration
            # the object is mutable and should then be recorded in the CAREamist
            data_config.set_mean_and_std(self.mean, self.std)
        else:
            self.mean = data_config.mean
            self.std = data_config.std

        # get transforms
        self.patch_transform = Compose(transform_list=data_config.transforms)

    def _calculate_mean_and_std(self) -> Tuple[float, float]:
        """
        Calculate mean and std of the dataset.

        Returns
        -------
        Tuple[float, float]
            Tuple containing mean and standard deviation.
        """
        means, stds = 0, 0
        num_samples = 0

        for sample, _ in self._iterate_over_files():
            means += sample.mean()
            stds += sample.std()
            num_samples += 1

        if num_samples == 0:
            raise ValueError("No samples found in the dataset.")

        result_mean = means / num_samples
        result_std = stds / num_samples

        logger.info(f"Calculated mean and std for {num_samples} images")
        logger.info(f"Mean: {result_mean}, std: {result_std}")
        return result_mean, result_std

    def _iterate_over_files(
        self,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
        """
        Iterate over data source and yield whole image.

        Yields
        ------
        np.ndarray
            Image.
        """
        # When num_workers > 0, each worker process will have a different copy of the
        # dataset object
        # Configuring each copy independently to avoid having duplicate data returned
        # from the workers
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # iterate over the files
        for i, filename in enumerate(self.data_files):
            # retrieve file corresponding to the worker id
            if i % num_workers == worker_id:
                try:
                    # read data
                    sample = self.read_source_func(filename, self.data_config.axes)

                    # read target, if available
                    if self.target_files is not None:
                        if filename.name != self.target_files[i].name:
                            raise ValueError(
                                f"File {filename} does not match target file "
                                f"{self.target_files[i]}. Have you passed sorted "
                                f"arrays?"
                            )

                        # read target
                        target = self.read_source_func(
                            self.target_files[i], self.data_config.axes
                        )

                        yield sample, target
                    else:
                        yield sample, None

                except Exception as e:
                    logger.error(f"Error reading file {filename}: {e}")

    def __iter__(
        self,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
            Single patch.
        """
        assert (
            self.mean is not None and self.std is not None
        ), "Mean and std must be provided"

        # iterate over files
        for sample_input, sample_target in self._iterate_over_files():
            reshaped_sample = reshape_array(sample_input, self.data_config.axes)
            reshaped_target = (
                None
                if sample_target is None
                else reshape_array(sample_target, self.data_config.axes)
            )

            patches = extract_patches_random(
                arr=reshaped_sample,
                patch_size=self.data_config.patch_size,
                target=reshaped_target,
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
            Percentage of files to split up, by default 0.1
        minimum_number : int, optional
            Minimum number of files to split up, by default 5

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


class IterablePredictionDataset(PathIterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
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
        super().__init__(
            data_config=prediction_config,
            src_files=src_files,
            read_source_func=read_source_func,
        )

        self.prediction_config = prediction_config
        self.axes = prediction_config.axes
        self.tile_size = self.prediction_config.tile_size
        self.tile_overlap = self.prediction_config.tile_overlap
        self.read_source_func = read_source_func

        # tile only if both tile size and overlaps are provided
        self.tile = self.tile_size is not None and self.tile_overlap is not None

        # get tta transforms
        self.patch_transform = Compose(
            transform_list=prediction_config.transforms,
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
            self.mean is not None and self.std is not None
        ), "Mean and std must be provided"

        for sample, _ in self._iterate_over_files():
            # reshape array
            reshaped_sample = reshape_array(sample, self.axes)

            if self.tile:
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
