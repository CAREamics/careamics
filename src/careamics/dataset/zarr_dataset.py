from itertools import islice
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import zarr

from careamics.utils import RunningStats
from careamics.utils.logging import get_logger

from ..utils import normalize
from .dataset_utils import read_zarr
from .extraction_strategy import ExtractionStrategy
from .patching import generate_patches

logger = get_logger(__name__)


class ZarrDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a zarr storage."""

    def __init__(
        self,
        data_source: Union[zarr.Group, zarr.Array],
        axes: str,
        patch_extraction_method: Union[ExtractionStrategy, None],
        patch_size: Optional[Union[List[int], Tuple[int]]] = None,
        num_patches: Optional[int] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        patch_transform: Optional[Callable] = None,
        patch_transform_params: Optional[Dict] = None,
        running_stats_window: float = 0.01,
        mode: str = "train",
    ) -> None:
        self.data_source = data_source
        self.axes = axes
        self.patch_extraction_method = patch_extraction_method
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mean = mean
        self.std = std
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params
        self.running_stats_window = running_stats_window
        self.mode = mode

        self.sample = read_zarr(self.data_source, self.axes)
        self.running_stats = RunningStats()

        self._calculate_initial_mean_std()

    def _calculate_initial_mean_std(self):
        if self.mean is None and self.std is None:
            idxs = np.random.randint(
                0,
                np.prod(self.sample._cdata_shape),
                size=max(
                    1,
                    int(np.prod(self.sample._cdata_shape) * self.running_stats_window),
                ),
            )
            random_chunks = self.sample[idxs]
            self.running_stats.update_mean(random_chunks.mean())
            self.running_stats.update_std(random_chunks.std())

    def _generate_patches(self):
        patches = generate_patches(
            self.sample,
            self.patch_extraction_method,
            self.patch_size,
        )

        for idx, patch in enumerate(patches):
            if self.mode != "predict":
                self.running_stats.update_mean(patch.mean())
                self.running_stats.update_std(patch.std())
            if isinstance(patch, tuple):
                normalized_patch = normalize(
                    img=patch[0],
                    mean=self.running_stats.avg_mean.value,
                    std=self.running_stats.avg_std.value,
                )
                patch = (normalized_patch, *patch[1:])
            else:
                patch = normalize(
                    img=patch,
                    mean=self.running_stats.avg_mean.value,
                    std=self.running_stats.avg_std.value,
                )

            if self.patch_transform is not None:
                assert self.patch_transform_params is not None
                patch = self.patch_transform(patch, **self.patch_transform_params)
            if self.num_patches is not None and idx >= self.num_patches:
                return
            else:
                print("mean", self.running_stats.avg_mean.value, "std", self.running_stats.avg_std.value)
                yield patch
        self.mean = self.running_stats.avg_mean.value
        self.std = self.running_stats.avg_std.value

    def __iter__(self):
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
        """
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        yield from islice(self._generate_patches(), 0, None, num_workers)
