from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from careamics.utils import RunningStats, normalize
from careamics.utils.logging import get_logger

from .dataset_utils import read_zarr
from .extraction_strategy import ExtractionStrategy
from .patching import generate_patches

logger = get_logger(__name__)


class ZarrDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a zarr storage."""

    def __init__(
        self,
        data_path: Union[str, Path],
        axes: str,
        patch_extraction_method: Union[ExtractionStrategy, None],
        patch_size: Optional[Union[List[int], Tuple[int]]] = None,
        num_patches: Optional[int] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        patch_transform: Optional[Callable] = None,
        patch_transform_params: Optional[Dict] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.axes = axes
        self.patch_extraction_method = patch_extraction_method
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mean = mean
        self.std = std
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params

        self.sample = read_zarr(self.data_path, self.axes)
        self.running_stats = RunningStats()

    def _generate_patches(self):
        patches = generate_patches(
            self.sample,
            self.patch_extraction_method,
            self.patch_size,
        )

        for idx, patch in enumerate(patches):
            if self.mean is None or self.std is None:
                self.running_stats.update_mean(patch.mean())
                self.running_stats.update_std(patch.std())

            if isinstance(patch, tuple):
                normalized_patch = normalize(
                    img=patch[0],
                    mean=self.running_stats.avg_mean,
                    std=self.running_stats.avg_std,
                )
                patch = (normalized_patch, *patch[1:])
            else:
                patch = normalize(
                    img=patch,
                    mean=self.running_stats.avg_mean,
                    std=self.running_stats.avg_std,
                )

            if self.patch_transform is not None:
                assert self.patch_transform_params is not None
                patch = self.patch_transform(patch, **self.patch_transform_params)
            if self.num_patches is not None and idx >= self.num_patches:
                return
            else:
                yield patch

        self.mean = self.running_stats.avg_mean
        self.std = self.running_stats.avg_std

    def __iter__(self):
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
        """
        # TODO: add support for multiple files/zarr groups

        yield from self._generate_patches()
