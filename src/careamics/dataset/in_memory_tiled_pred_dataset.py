"""In-memory tiled prediction dataset."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from numpy.typing import NDArray

from careamics.config.inference_model import InferenceConfig
from careamics.config.tile_information import TileInformation
from careamics.config.transformations import NormalizeModel
from careamics.dataset.dataset_utils import reshape_array
from careamics.dataset.tiling.tiled_patching import extract_tiles
from careamics.transforms import Compose


class InMemoryTiledPredDataset:
    """
    In-memory prediction dataset for tiled prediction.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Prediction configuration.
    inputs : Union[str, Path, NDArray]
        Path to the folder containing the images or numpy array.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        inputs: Union[str, Path, NDArray],
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Prediction configuration.
        inputs : Union[str, Path, NDArray]
            Path to the folder containing the images or numpy array.

        Raises
        ------
        ValueError
            If data_path is not a directory.
        """
        # config
        self.pred_config = prediction_config

        # Save original data before reshaping for stitching
        self.original_data = inputs

        # data
        reshaped_data = reshape_array(
            inputs,
            self.pred_config.axes,  
        )
        self.reshaped_data = reshaped_data

        # Tile size and overlap
        self.tile_size = self.pred_config.tile_size
        self.tile_overlap = self.pred_config.tile_overlap

        # Mean and std
        self.image_means = self.pred_config.image_means
        self.image_stds = self.pred_config.image_stds

        # Generate patches
        self.data = self._prepare_tiles()

        # get transforms
        self.patch_transform = Compose(
            transform_list=[
                NormalizeConfig(
                    image_means=self.image_means, image_stds=self.image_stds
                )
            ],
        )

    def _prepare_tiles(self) -> list[tuple[NDArray, TileInformation]]:
        """
        Prepare tiles for prediction.

        Returns
        -------
        list of tuples of NDArray and TileInformation
            List of tuples containing the tiles and their information.
        """
        # iterate over all samples
        reshaped_sample = self.reshaped_data

        # generate patches, which returns a generator
        patch_generator = extract_tiles(
            arr=reshaped_sample,
            tile_size=self.tile_size,
            overlaps=self.tile_overlap,
            axes=self.pred_config.axes,
        )
        patches_list = list(patch_generator)

        if len(patches_list) == 0:
            raise ValueError("No tiles generated")

        return patches_list

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[tuple[NDArray, ...], TileInformation]:
        """
        Return a sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        tuple of NDArray and TileInformation
            Sample and tile information.
        """
        sample, tile_info = self.data[index]

        # Ensure the sample is a tensor with correct dtype before transformation
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).float()
        elif isinstance(sample, torch.Tensor):
            sample = sample.float()

        # Apply normalization transform - wrap in tuple as expected by transform
        if self.patch_transform is not None:
            # The transform expects a tuple of tensors
            transformed_result = self.patch_transform((sample,))
            # Extract the transformed tensor from the result
            if isinstance(transformed_result, tuple):
                sample = transformed_result[0]
            else:
                sample = transformed_result
    
        # Ensure sample has correct shape for 1D data: (C, X)
        if len(sample.shape) != 2:
            raise ValueError(f"Expected 2D tensor (C, X) for 1D data, got shape {sample.shape}")

        return (sample,), tile_info
