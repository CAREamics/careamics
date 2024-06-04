"""Zarr dataset."""

# from itertools import islice
# from typing import Callable, Dict, List, Optional, Tuple, Union

# import numpy as np
# import torch
# import zarr

# from careamics.utils import RunningStats
# from careamics.utils.logging import get_logger

# from ..utils import normalize
# from .dataset_utils import read_zarr
# from .patching.patching import (
#     generate_patches_unsupervised,
# )

# logger = get_logger(__name__)


# class ZarrDataset(torch.utils.data.IterableDataset):
#     """Dataset to extract patches from a zarr storage.

#     Parameters
#     ----------
#     data_source : Union[zarr.Group, zarr.Array]
#         Zarr storage.
#     axes : str
#         Description of axes in format STCZYX.
#     patch_extraction_method : Union[ExtractionStrategies, None]
#         Patch extraction strategy, as defined in extraction_strategy.
#     patch_size : Optional[Union[List[int], Tuple[int]]], optional
#         Size of the patches in each dimension, by default None.
#     num_patches : Optional[int], optional
#         Number of patches to extract, by default None.
#     mean : Optional[float], optional
#         Expected mean of the dataset, by default None.
#     std : Optional[float], optional
#         Expected standard deviation of the dataset, by default None.
#     patch_transform : Optional[Callable], optional
#         Patch transform callable, by default None.
#     patch_transform_params : Optional[Dict], optional
#         Patch transform parameters, by default None.
#     running_stats_window_perc : float, optional
#         Percentage of the dataset to use for calculating the initial mean and standard
#         deviation, by default 0.01.
#     mode : str, optional
#         train/predict, controls running stats calculation.
#     """

#     def __init__(
#         self,
#         data_source: Union[zarr.Group, zarr.Array],
#         axes: str,
#         patch_extraction_method: Union[SupportedExtractionStrategy, None],
#         patch_size: Optional[Union[List[int], Tuple[int]]] = None,
#         num_patches: Optional[int] = None,
#         mean: Optional[float] = None,
#         std: Optional[float] = None,
#         patch_transform: Optional[Callable] = None,
#         patch_transform_params: Optional[Dict] = None,
#         running_stats_window_perc: float = 0.01,
#         mode: str = "train",
#     ) -> None:
#         self.data_source = data_source
#         self.axes = axes
#         self.patch_extraction_method = patch_extraction_method
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#         self.mean = mean
#         self.std = std
#         self.patch_transform = patch_transform
#         self.patch_transform_params = patch_transform_params
#         self.sample = read_zarr(self.data_source, self.axes)
#         self.running_stats_window = int(
#             np.prod(self.sample._cdata_shape) * running_stats_window_perc
#         )
#         self.mode = mode
#         self.running_stats = RunningStats()

#         self._calculate_initial_mean_std()

#     def _calculate_initial_mean_std(self):
#         """Calculate initial mean and std of the dataset."""
#         if self.mean is None and self.std is None:
#             idxs = np.random.randint(
#                 0,
#                 np.prod(self.sample._cdata_shape),
#                 size=max(1, self.running_stats_window),
#             )
#             random_chunks = self.sample[idxs]
#             self.running_stats.init(random_chunks.mean(), random_chunks.std())

#     def _generate_patches(self):
#         """Generate patches from the dataset and calculates running stats.

#         Yields
#         ------
#         np.ndarray
#             Patch.
#         """
#         patches = generate_patches_unsupervised(
#             self.sample,
#             self.patch_extraction_method,
#             self.patch_size,
#         )

#         # num_patches = np.ceil(
#         #     np.prod(self.sample.chunks)
#         #     / (np.prod(self.patch_size) * self.running_stats_window)
#         # ).astype(int)

#         for idx, patch in enumerate(patches):
#             if self.mode != "predict":
#                 self.running_stats.update(patch.mean())
#             if isinstance(patch, tuple):
#                 normalized_patch = normalize(
#                     img=patch[0],
#                     mean=self.running_stats.avg_mean.value,
#                     std=self.running_stats.avg_std.value,
#                 )
#                 patch = (normalized_patch, *patch[1:])
#             else:
#                 patch = normalize(
#                     img=patch,
#                     mean=self.running_stats.avg_mean.value,
#                     std=self.running_stats.avg_std.value,
#                 )

#             if self.patch_transform is not None:
#                 assert self.patch_transform_params is not None
#                 patch = self.patch_transform(patch, **self.patch_transform_params)
#             if self.num_patches is not None and idx >= self.num_patches:
#                 return
#             else:
#                 yield patch
#         self.mean = self.running_stats.avg_mean.value
#         self.std = self.running_stats.avg_std.value

#     def __iter__(self):
#         """
#         Iterate over data source and yield single patch.

#         Yields
#         ------
#         np.ndarray
#         """
#         worker_info = torch.utils.data.get_worker_info()
#         num_workers = worker_info.num_workers if worker_info is not None else 1
#         yield from islice(self._generate_patches(), 0, None, num_workers)
