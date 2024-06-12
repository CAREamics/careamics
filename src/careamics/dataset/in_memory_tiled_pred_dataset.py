"""In-memory tiled prediction dataset."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.tile_information import TileInformation
from ..config.transformations import NormalizeModel
from .dataset_utils import reshape_array
from .tiling import extract_tiles


class InMemoryTiledPredDataset(Dataset):
    """Prediction dataset storing data in memory and returning tiles of each image.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Prediction configuration.
    inputs : np.ndarray
        Input data.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        inputs: np.ndarray,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Prediction configuration.
        inputs : np.ndarray
            Input data.

        Raises
        ------
        ValueError
            If data_path is not a directory.
        """
        if (
            prediction_config.tile_size is None
            or prediction_config.tile_overlap is None
        ):
            raise ValueError(
                "Tile size and overlap must be provided to use the tiled prediction "
                "dataset."
            )

        self.pred_config = prediction_config
        self.input_array = inputs
        self.axes = self.pred_config.axes
        self.tile_size = prediction_config.tile_size
        self.tile_overlap = prediction_config.tile_overlap
        self.mean = self.pred_config.mean
        self.std = self.pred_config.std

        # Generate patches
        self.data = self._prepare_tiles()
        self.mean, self.std = self.pred_config.mean, self.pred_config.std

        # get transforms
        self.patch_transform = Compose(
            transform_list=[NormalizeModel(mean=self.mean, std=self.std)],
        )

    def _prepare_tiles(self) -> List[Tuple[np.ndarray, TileInformation]]:
        """
        Iterate over data source and create an array of patches.

        Returns
        -------
        List[XArrayTile]
            List of tiles.
        """
        # reshape array
        reshaped_sample = reshape_array(self.input_array, self.axes)

        # generate patches, which returns a generator
        patch_generator = extract_tiles(
            arr=reshaped_sample,
            tile_size=self.tile_size,
            overlaps=self.tile_overlap,
        )
        patches_list = list(patch_generator)

        if len(patches_list) == 0:
            raise ValueError("No tiles generated, ")

        return patches_list

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, TileInformation]:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        Tuple[np.ndarray, TileInformation]
            Transformed patch.
        """
        tile_array, tile_info = self.data[index]

        # Apply transforms
        transformed_tile, _ = self.patch_transform(patch=tile_array)

        return transformed_tile, tile_info
