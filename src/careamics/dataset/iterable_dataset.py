"""
Tiff dataset module.

This module contains the implementation of the TiffDataset class, which allows loading
tiff files.
"""
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import torch

from ..config.data import DataModel
from ..utils.logging import get_logger
from .dataset_utils import get_patch_transform, get_patch_transform_predict, read_tiff
from .patching import generate_patches_supervised, generate_patches_unsupervised

logger = get_logger(__name__)


class IterableDataset(torch.utils.data.IterableDataset):
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
        files: List[Path],
        config: DataModel,
        target_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        read_source_func: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        self.data_files = files
        self.target_files = target_files
        self.axes = config.axes
        self.patch_size = config.patch_size
        self.patch_extraction_method = "random"
        self.read_source_func = read_source_func if read_source_func else read_tiff

        if not config.mean or not config.std:
            self.mean, self.std = self._calculate_mean_and_std()

        self.patch_transform = get_patch_transform(
            patch_transforms=config.transforms,
            mean=self.mean,
            std=self.std,
            target=target_files is not None,
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

        for sample in self._iterate_files():
            means += sample.mean()
            stds += np.std(sample)
            num_samples += 1

        result_mean = means / num_samples
        result_std = stds / num_samples

        logger.info(f"Calculated mean and std for {num_samples} images")
        logger.info(f"Mean: {result_mean}, std: {result_std}")
        return result_mean, result_std

    def _iterate_files(self) -> Generator:
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
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        for i, filename in enumerate(self.data_files):
            if i % num_workers == worker_id:
                sample = self.read_source_func(filename, self.axes)
                if self.target_files is not None:
                    if filename.name != self.target_files[i].name:
                        raise ValueError(
                            f"File {filename} does not match target file "
                            f"{self.target_files[i]}"
                        )
                    target = self.read_source_func(self.target_files[i], self.axes)
                    yield sample, target
                else:
                    yield sample

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

        for sample in self._iterate_files():
            if self.target_files is not None:
                patches = generate_patches_supervised(
                    sample,
                    self.axes,
                    self.patch_extraction_method,
                    self.patch_size,
                )

            else:
                patches = generate_patches_unsupervised(
                    sample,
                    self.axes,
                    self.patch_extraction_method,
                    self.patch_size,
                )

            for patch_data in patches:
                if isinstance(patch_data, tuple):
                    if self.target_files is not None:
                        target = patch_data[1:]
                        transformed = self.patch_transform(
                            image=np.moveaxis(patch_data[0], 0, -1),
                            target=np.moveaxis(target, 0, -1),
                        )
                        yield (transformed["image"], transformed["mask"])
                        # TODO fix dimensions
                    else:
                        transformed = self.patch_transform(
                            image=np.moveaxis(patch_data[0], 0, -1)
                        )
                        yield (transformed["image"], *patch_data[1:])
                else:
                    yield self.patch_transform(image=patch_data)["image"]

            # else:
            #     # if S or T dims are not empty - assume every image is a separate
            #     # sample in dim 0
            #     for i in range(sample.shape[0]):
            #         item = np.expand_dims(sample[i], (0, 1))
            #         item = normalize(img=item, mean=self.mean, std=self.std)
            #         yield item


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
        files: List[Path],
        config: DataModel,
        read_source_func: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(files=files, config=config, read_source_func=read_source_func)
        self.data_files = files
        self.axes = config.axes
        self.patch_size = config.patch_size
        self.patch_extraction_method = "tiled"
        self.read_source_func = read_source_func if read_source_func else read_tiff

        if not config.mean or not config.std:
            self.mean, self.std = self._calculate_mean_and_std()

        self.patch_transform = get_patch_transform_predict(
            patch_transforms=config.transforms,
            mean=self.mean,
            std=self.std,
            target=False,
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

        for sample in self._iterate_files():
            patches = generate_patches_unsupervised(
                sample,
                self.axes,
                self.patch_extraction_method,
                self.patch_size,
            )

            for patch_data in patches:
                if isinstance(patch_data, tuple):
                    transformed = self.patch_transform(
                        image=np.moveaxis(patch_data[0], 0, -1)
                    )
                    yield (np.moveaxis(transformed["image"], -1, 0), *patch_data[1:])
                else:
                    yield self.patch_transform(image=patch_data)["image"]
