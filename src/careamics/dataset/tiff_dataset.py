from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch

from careamics.config.training import ExtractionStrategies
from careamics.dataset.dataset_utils import (
    generate_patches,
    list_files,
    read_tiff,
)
from careamics.utils import normalize
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


class TiffDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a tiff image(s).

    Parameters
    ----------
    data_path : str
        Path to data, must be a directory.

    axes: str
        Description of axes in format STCZYX

    patch_extraction_method: ExtractionStrategies
        Patch extraction strategy, one of "sequential", "random", "tiled"

    patch_size : Tuple[int]
        The size of the patch to extract from the image. Must be a tuple of len either
        2 or 3 depending on number of spatial dimension in the data.

    patch_overlap: Tuple[int]
        Size of the overlaps. Used for "tiled" tiling strategy.

    mean: float
        Expected mean of the samples

    std: float
        Expected std of the samples

    patch_transform: Optional[Callable], optional
        Transform to apply to patches.

    patch_transform_params: Optional[Dict], optional
        Additional parameters to pass to patch transform function
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        data_format: str,
        axes: str,
        patch_extraction_method: Union[ExtractionStrategies, None],
        patch_size: Optional[Union[List[int], Tuple[int]]] = None,
        patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        patch_transform: Optional[Callable] = None,
        patch_transform_params: Optional[Dict] = None,
    ) -> None:

        self.data_format = data_format
        self.axes = axes

        self.patch_transform = patch_transform

        self.files = list_files(data_path, self.data_format)

        self.mean = mean
        self.std = std
        if not mean or not std:
            self.mean, self.std = self.calculate_mean_and_std()

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_extraction_method = patch_extraction_method
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params

    def calculate_mean_and_std(self) -> Tuple[float, float]:
        """Calculates mean and std of the dataset.

        Returns
        -------
        Tuple[float, float]
            Tuple containing mean and std
        """
        means, stds = 0, 0
        num_samples = 0

        for sample in self.iterate_files():
            means += sample.mean()
            stds += np.std(sample)
            num_samples += 1

        result_mean = means / num_samples
        result_std = stds / num_samples

        logger.info(f"Calculated mean and std for {num_samples} images")
        logger.info(f"Mean: {result_mean}, std: {result_std}")
        return result_mean, result_std

    def iterate_files(self) -> Generator:
        """
        Iterate over data source and yield whole image.

        Yields
        ------
        np.ndarray
        """
        # When num_workers > 0, each worker process will have a different copy of the
        # dataset object
        # Configuring each copy independently to avoid having duplicate data returned
        # from the workers
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        for i, filename in enumerate(self.files):
            if i % num_workers == worker_id:
                sample = read_tiff(filename, self.axes)
                yield sample

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
        """
        assert (
            self.mean is not None and self.std is not None
        ), "Mean and std must be provided"
        for sample in self.iterate_files():
            # TODO patch_extraction_method should never be None!
            if self.patch_extraction_method:
                # TODO: move S and T unpacking logic from patch generator
                patches = generate_patches(
                    sample,
                    self.patch_extraction_method,
                    self.patch_size,
                    self.patch_overlap,
                )

                for patch in patches:
                    if isinstance(patch, tuple):
                        normalized_patch = normalize(
                            img=patch[0], mean=self.mean, std=self.std
                        )
                        patch = (normalized_patch, *patch[1:])
                    else:
                        patch = normalize(img=patch, mean=self.mean, std=self.std)

                    if self.patch_transform is not None:
                        assert self.patch_transform_params is not None
                        patch = self.patch_transform(
                            patch, **self.patch_transform_params
                        )

                    yield patch

            else:
                # if S or T dims are not empty - assume every image is a separate
                # sample in dim 0
                for i in range(sample.shape[0]):
                    item = np.expand_dims(sample[i], (0, 1))
                    item = normalize(img=item, mean=self.mean, std=self.std)
                    yield item
