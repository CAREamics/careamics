from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from ..config.data_model import DataModel
from ..config.support import SupportedExtractionStrategy
from ..utils.logging import get_logger
from .dataset_utils import read_tiff, reshape_array
from .patching import (
    generate_patches_predict,
    generate_patches_supervised,
    generate_patches_unsupervised,
    get_patch_transform,
)

logger = get_logger(__name__)


class IterableDataset(IterableDataset):
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
        data_config: DataModel,
        src_files: List[Path],
        target_files: Optional[List[Path]] = None,
        read_source_func: Callable = read_tiff,
    ) -> None:
        if target_files is not None:
            raise NotImplementedError("Targets are not yet supported.")

        self.data_files = src_files
        self.target_files = target_files
        self.axes = data_config.axes
        self.patch_size = data_config.patch_size
        self.read_source_func = read_source_func

        # compute mean and std over the dataset
        if not data_config.mean or not data_config.std:
            self.mean, self.std = self._calculate_mean_and_std()

            # if the transforms are not an instance of Compose
            if data_config.has_tranform_list():
                # update mean and std in configuration
                # the object is mutable and should then be recorded in the CAREamist obj
                data_config.set_mean_and_std(self.mean, self.std)

        # get transforms
        self.patch_transform = get_patch_transform(
            patch_transforms=data_config.transforms,
            with_target=target_files is not None,
        )

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

        for sample in self._iterate_over_files():
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

    def _iterate_over_files(self) -> Generator[Tuple[np.ndarray, ...], None, None]:
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
                    sample = self.read_source_func(filename, self.axes)

                    # reshape data
                    reshaped_sample = reshape_array(sample, self.axes)

                    # read target, if available
                    if self.target_files is not None:
                        if filename.name != self.target_files[i].name:
                            raise ValueError(
                                f"File {filename} does not match target file "
                                f"{self.target_files[i]}. Have you passed sorted "
                                f"arrays?"
                            )

                        # read target
                        target = self.read_source_func(self.target_files[i], self.axes)

                        # reshape target
                        reshaped_target = reshape_array(target, self.axes)

                        yield reshaped_sample, reshaped_target
                    else:
                        yield reshaped_sample
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {e}")

    def __iter__(self) -> Generator[np.ndarray, None, None]:
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
        for sample in self._iterate_over_files():
            if self.target_files is not None:
                sample_input, sample_target = sample
                patches = generate_patches_supervised(
                    sample=sample_input,
                    axes=self.axes,
                    patch_extraction_method=SupportedExtractionStrategy.RANDOM,
                    patch_size=self.patch_size,
                    target=sample_target,
                )

            else:
                patches = generate_patches_unsupervised(
                    sample=sample,
                    axes=self.axes,
                    patch_extraction_method=SupportedExtractionStrategy.RANDOM,
                    patch_size=self.patch_size,
                )

            # iterate over patches
            # patches are tuples of (patch, target) if target is available
            # or (patch, None) only if no target is available
            # patch is of dimensions (C)ZYX
            for patch_data in patches:
                # if there is a target
                if self.target_files is not None:
                    # Albumentations expects the channel dimension to be last
                    c_patch = np.moveaxis(patch_data[0], 0, -1)
                    c_target = np.moveaxis(patch_data[1], 0, -1)

                    # apply the transform to the patch and the target
                    transformed = self.patch_transform(
                        image=c_patch,
                        target=c_target,
                    )

                    # TODO if ManipulateN2V, then we get a tuple not an array!
                    # TODO if "target" string is used, then make it a co or enum

                    # move the axes back to the original position
                    c_patch = np.moveaxis(transformed["image"], -1, 0)
                    c_target = np.moveaxis(transformed["target"], -1, 0)

                    yield (c_patch, c_target)
                else:
                    # Albumentations expects the channel dimension to be last
                    patch = np.moveaxis(patch_data[0], 0, -1)

                    # apply transform
                    transformed = self.patch_transform(image=patch)

                    # TODO is there a chance that ManipulateN2V is not in transforms?
                    # retrieve the output of ManipulateN2V
                    masked_patch, patch, mask = transformed["image"]

                    # move C axes back
                    masked_patch = np.moveaxis(masked_patch, -1, 0)
                    patch = np.moveaxis(patch, -1, 0)
                    mask = np.moveaxis(mask, -1, 0)

                    yield (masked_patch, patch, mask)

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
    ) -> IterableDataset:
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


# TODO: why was this calling transforms on prediction patches?
class IterablePredictionDataset(IterableDataset):
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
        data_config: DataModel,
        files: List[Path],
        tile_size: Union[List[int], Tuple[int]],
        tile_overlap: Optional[Union[List[int], Tuple[int]]] = None,
        read_source_func: Callable = read_tiff,
        **kwargs,
    ) -> None:
        super().__init__(
            data_config=data_config, src_files=files, read_source_func=read_source_func
        )

        self.patch_size = data_config.patch_size
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.read_source_func = read_source_func

        # check that mean and std are provided
        if not self.mean or not self.std:
            raise ValueError(
                "Mean and std must be provided to the configuration in order to "
                " perform prediction."
            )

    def __iter__(self) -> Generator[np.ndarray, None, None]:
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

        for sample in self._iterate_over_files():
            patches = generate_patches_predict(
                sample, self.axes, self.tile_size, self.tile_overlap
            )

            for patch_data in patches:
                if isinstance(patch_data, tuple):
                    transformed = self.patch_transform(
                        image=np.moveaxis(patch_data[0], 0, -1)
                    )
                    yield (np.moveaxis(transformed["image"], -1, 0), *patch_data[1:])
                else:
                    yield self.patch_transform(image=patch_data)["image"]
