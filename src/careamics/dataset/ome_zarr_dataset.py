"""Example ome zarr dataset using Dask."""

import numpy as np
import torch
from dask.array.core import Array
from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.config.data_model import DataConfig
from careamics.dataset.dataset_utils.running_stats import WelfordStatistics


class PatchDaskDataset(Dataset):
    """
    Dataset returning patches from ome zarr dataset.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    inputs : Array
        Input data.
    """

    def __init__(self, data_config: DataConfig, inputs: Array):
        """
        Constructor.

        Parameters
        ----------
        data_config : DataConfig
            Data configuration.
        inputs : Array
            Input data.
        """
        self.data_config = data_config
        self.inputs = inputs
        self.chunk_size = np.array(inputs.chunksize)
        self.data_shape = np.array(inputs.shape)
        self.patch_size = np.array(self.data_config.patch_size)

        if len(self.data_shape) != len(self.patch_size):
            new_dims = len(self.data_shape) - len(self.patch_size)
            for _ in range(new_dims):
                self.patch_size = np.insert(self.patch_size, 0, 1, axis=0)

        assert len(self.patch_size) == len(self.data_shape)

        self.patches_per_dimension = np.ceil(self.data_shape / self.patch_size)
        self.patches_per_dimension = self.patches_per_dimension.astype(np.uint32)

        if np.any(self.patch_size > self.chunk_size):
            print("Larger patch than chunk size")

        self.len = np.prod(self.patches_per_dimension)
        self.indices = np.arange(self.len, dtype=np.uint32)

        self.running_stats = WelfordStatistics()

    def __len__(self) -> int:
        """
        Dataset length.

        Returns
        -------
        int
            Length of dataset.
        """
        return self.len

    def get_data_statistics(self) -> tuple[NDArray, NDArray]:
        """
        Dataset statistics.

        Returns
        -------
        tuple[NDArray, NDArray]
            Data statistics.
        """
        return self.running_stats.finalize()

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Dataset getitem.

        Parameters
        ----------
        index : int
            Dataset index.

        Returns
        -------
        torch.Tensor
            Output data.
        """
        data_index = self.indices[index]
        index_location = np.unravel_index(data_index, self.patches_per_dimension)

        patch_start = np.array(index_location) * self.patch_size
        patch_end = patch_start + self.patch_size

        if np.any(patch_end >= self.data_shape):
            patch_end = np.where(
                patch_end > self.data_shape, self.data_shape, patch_end
            )
            patch_start = patch_end - self.patch_size

        slices = tuple(slice(s, e) for s, e in zip(patch_start, patch_end))
        chunk = self.inputs[slices].compute()
        chunk = chunk.astype(np.float32)
        self.running_stats.update(chunk, sample_idx=index)

        running_mean, running_std = self.running_stats.finalize()
        chunk = (chunk - running_mean) / running_std

        return torch.tensor(chunk, dtype=torch.float32)
