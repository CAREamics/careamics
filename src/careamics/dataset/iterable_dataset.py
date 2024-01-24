"""
Tiff dataset module.

This module contains the implementation of the TiffDataset class, which allows loading
tiff files.
"""
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import torch

from ..utils import normalize
from ..utils.logging import get_logger
from .dataset_utils import get_patch_transform, list_files, read_tiff, validate_files
from .extraction_strategy import ExtractionStrategy
from .patching import generate_patches_supervised, generate_patches_unsupervised

logger = get_logger(__name__)


class IterableDataset(torch.utils.data.IterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data, must be a directory.
    data_format : str
        Extension of the files to load, without the period.
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
        data_path: Union[str, Path],
        data_format: str,
        axes: str,
        patch_extraction_method: Union[ExtractionStrategy, None],
        patch_size: Optional[Union[List[int], Tuple[int]]] = None,
        patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        patch_transform: Optional[Callable] = None,
        target_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        target_format: Optional[str] = None,
        read_source_func: Optional[Callable] = None,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError("Path to data should be an existing folder.")
        self.target_path = target_path
        self.data_format = data_format
        self.target_format = target_format
        self.axes = axes

        if not self.data_path.is_dir():
            raise ValueError("Path to data should be an existing folder.")

        self.data_files = list_files(data_path, self.data_format)
        if self.target_path is not None:
            if not self.target_path.is_dir():
                raise ValueError("Path to targets should be an existing folder.")
            if self.target_format is None:
                raise ValueError("Target format must be specified.")
            self.target_files = list_files(self.target_path, self.target_format)
            validate_files(self.data_files, self.target_files)

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_extraction_method = patch_extraction_method
        self.source_read_func = read_source_func if read_source_func else read_tiff

        self.mean = mean
        self.std = std

        if not mean or not std:
            self.mean, self.std = self._calculate_mean_and_std()

        self.patch_transform = get_patch_transform(patch_transform)

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
                sample = self.source_read_func(filename, self.axes)
                if self.target_path is not None:
                    if filename.name != self.target_files[i].name:
                        raise ValueError(
                            f"File {filename} does not match target file "
                            f"{self.target_files[i]}"
                        )
                    target = self.source_read_func(self.target_files[i], self.axes)
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
            if self.target_path is not None:
                patches = generate_patches_supervised(
                    sample,
                    self.axes,
                    self.patch_extraction_method,
                    self.patch_size,
                    self.patch_overlap,
                )

            else:
                patches = generate_patches_unsupervised(
                    sample,
                    self.axes,
                    self.patch_extraction_method,
                    self.patch_size,
                    self.patch_overlap,
                )

            for patch_data in patches:
                if isinstance(patch_data, tuple):
                    if self.target_path is not None:
                        patch = normalize(
                            img=patch_data[0], mean=self.mean, std=self.std
                        )
                        target = patch_data[1:]
                        transformed = self.patch_transform(image=patch, mask=target)
                        yield (transformed["image"], transformed["mask"])
                    else:
                        patch = normalize(
                            img=patch_data[0], mean=self.mean, std=self.std
                        )
                        transformed = self.patch_transform(image=patch)
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
