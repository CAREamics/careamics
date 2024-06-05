"""In-memory prediction dataset."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.transformations import NormalizeModel
from .dataset_utils import read_tiff, reshape_array


class InMemoryPredDataset(Dataset):
    """Simple prediction dataset returning images along the sample axis.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Prediction configuration.
    inputs : np.ndarray
        Input data.
    data_target : Optional[np.ndarray], optional
        Target data, by default None.
    read_source_func : Optional[Callable], optional
        Read source function for custom types, by default read_tiff.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        inputs: np.ndarray,
        data_target: Optional[np.ndarray] = None,
        read_source_func: Optional[Callable] = read_tiff,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Prediction configuration.
        inputs : np.ndarray
            Input data.
        data_target : Optional[np.ndarray], optional
            Target data, by default None.
        read_source_func : Optional[Callable], optional
            Read source function for custom types, by default read_tiff.

        Raises
        ------
        ValueError
            If data_path is not a directory.
        """
        self.pred_config = prediction_config
        self.input_array = inputs
        self.axes = self.pred_config.axes
        self.tile_size = self.pred_config.tile_size
        self.tile_overlap = self.pred_config.tile_overlap
        self.mean = self.pred_config.mean
        self.std = self.pred_config.std
        self.data_target = data_target
        self.mean, self.std = self.pred_config.mean, self.pred_config.std

        # tiling only if both tile size and overlap are provided
        self.tiling = self.tile_size is not None and self.tile_overlap is not None

        # read function
        self.read_source_func = read_source_func

        # Reshape data
        self.data = reshape_array(self.input_array, self.axes)

        # get transforms
        self.patch_transform = Compose(
            transform_list=[NormalizeModel(mean=self.mean, std=self.std)],
        )

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        np.ndarray
            Transformed patch.
        """
        transformed_patch, _ = self.patch_transform(patch=self.data[[index]])

        return transformed_patch
