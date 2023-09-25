from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.dataset_utils import (
    generate_patches,
    list_files,
    read_tiff,
)
from careamics_restoration.utils import normalize
from careamics_restoration.utils.logging import get_logger

logger = get_logger(__name__)


class InMemoryDataset(torch.utils.data.Dataset):
    """Dataset to extract patches from image(s) tat can be stored in memory.

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
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError("Path to data should be an existing folder.")

        self.data_format = data_format
        self.axes = axes

        self.patch_transform = patch_transform

        self.files = list_files(self.data_path, self.data_format)

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_extraction_method = patch_extraction_method
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params

        self.mean = mean
        self.std = std

        # Generate patches
        self.data, computed_mean, computed_std = self.prepare_patches()

        if not mean or not std:
            self.mean, self.std = computed_mean, computed_std
            logger.info(f"Computed dataset mean: {self.mean}, std: {self.std}")

        assert self.mean is not None
        assert self.std is not None

    def prepare_patches(self) -> Tuple[np.ndarray, float, float]:
        """Iterate over data source and create array of patches.

        Returns
        -------
        np.ndarray
            Array of patches
        """
        means, stds, num_samples = 0, 0, 0
        self.all_patches = []
        for filename in self.files:
            sample = read_tiff(filename, self.axes)
            means += sample.mean()
            stds += np.std(sample)
            num_samples += 1

            patches = generate_patches(
                sample,
                self.patch_extraction_method,
                self.patch_size,
                self.patch_overlap,
            )
            self.all_patches.append(patches)
            result_mean, result_std = means / num_samples, stds / num_samples
        return np.concatenate(self.all_patches), result_mean, result_std

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return sum(s.shape[0] for s in self.all_patches)

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        """Returns the patch."""
        patch = self.data[index].squeeze()

        if isinstance(patch, tuple):
            patch = normalize(img=patch[0], mean=self.mean, std=self.std)  # type: ignore
            patch = (patch, *patch[1:])
        else:
            patch = normalize(img=patch, mean=self.mean, std=self.std)  # type: ignore

        if self.patch_transform is not None:
            patch = self.patch_transform(patch, **self.patch_transform_params)
        return patch
