"""In-memory tiled prediction dataset."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.config.data.inference_config import InferenceConfig
from careamics.config.data.tile_information import TileInformation
from careamics.config.transformations import NormalizeConfig
from careamics.dataset.dataset_utils import reshape_array
from careamics.dataset.tiling.tiled_patching import extract_tiles
from careamics.transforms import Compose


class InMemoryTiledPredDataset(Dataset):
    """Prediction dataset storing data in memory and returning tiles of each image.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Prediction configuration.
    inputs : NDArray
        Input data.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        inputs: NDArray,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Prediction configuration.
        inputs : NDArray
            Input data array.

        Raises
        ------
        ValueError
            If no tiles are generated.
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

        Raises
        ------
        ValueError
            If tile_size or tile_overlap are not specified, or if no tiles generated.
        """
        # iterate over all samples
        reshaped_sample = self.reshaped_data

        # Ensure tile_size and tile_overlap are specified
        if self.tile_size is None or self.tile_overlap is None:
            raise ValueError(
                "tile_size and tile_overlap must be specified for tiled prediction"
            )

        # generate patches, which returns a generator
        patch_generator = extract_tiles(
            arr=reshaped_sample,
            tile_size=self.tile_size,
            overlaps=self.tile_overlap,
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

        # Keep sample as numpy array for transform compatibility
        # The transforms expect numpy arrays and handle tensor conversion internally
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()

        # Ensure it's a numpy array with correct dtype
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample)
        # logger.info(sample.shape)
        # logger.info(sample.shape)
        # if sample.shape[0] != len(self.image_means):
        #     sample = sample.T
        # Apply normalization transform - keep as numpy array
        if self.patch_transform is not None:
            # Call transform with patch keyword argument
            transformed_result = self.patch_transform(patch=sample)
            # Extract the transformed result from the tuple
            if isinstance(transformed_result, tuple):
                sample = transformed_result[0]
            else:
                sample = transformed_result

        # Validate sample shape based on axes
        # Determine expected number of dimensions from axes
        spatial_axes = [ax for ax in self.pred_config.axes if ax in "XYZ"]
        expected_ndims = len(spatial_axes) + 1  # +1 for channel dimension

        if len(sample.shape) != expected_ndims:
            raise ValueError(
                f"Expected {expected_ndims}D tensor for axes "
                f"'{self.pred_config.axes}', got shape {sample.shape} "
                f"({len(sample.shape)}D)"
            )

        return (sample,), tile_info
