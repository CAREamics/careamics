"""In-memory dataset module."""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..config.data_model import DataModel
from ..utils import normalize
from ..utils.logging import get_logger
from .dataset_utils import read_tiff
from .patching.patch_transform import get_patch_transform
from .patching.patching import (
    generate_patches_predict,
    prepare_patches_supervised,
    prepare_patches_unsupervised,
    prepare_patches_supervised_array,
    prepare_patches_unsupervised_array,
)

logger = get_logger(__name__)



# TODO dataset which sets appart some data for validation?


class InMemoryDataset(torch.utils.data.Dataset):
    """
    Dataset storing data in memory and allowing generating patches from it.


    """

    def __init__(
        self,
        data_config: DataModel,
        data: Union[np.ndarray, List[Path]],
        data_target: Optional[Union[np.ndarray, List[Path]]] = None,
        read_source_func: Callable = read_tiff,
        **kwargs,
    ) -> None:
        """
        Constructor.

        # TODO
        """
        self.data = data
        self.data_target = data_target
        self.axes = data_config.axes
        self.patch_size = data_config.patch_size

        # read function
        self.read_source_func = read_source_func

        # Generate patches
        supervised = self.data_target is not None
        patches = self._prepare_patches(supervised)

        # Add results to members
        self.patches, self.patch_targets, computed_mean, computed_std = patches

        if not data_config.mean or not data_config.std:
            self.mean, self.std = computed_mean, computed_std
            logger.info(f"Computed dataset mean: {self.mean}, std: {self.std}")

            # update mean and std in configuration
            # the object is mutable and should then be recorded at the CAREamist level
            data_config.set_mean_and_std(self.mean, self.std)
        else:
            self.mean, self.std = data_config.mean, data_config.std

        self.patch_transform = get_patch_transform(
            patch_transforms=data_config.transforms,
            mean=self.mean,
            std=self.std,
            target=self.data_target is not None,
        )

    def _prepare_patches(self, supervised: bool) -> Tuple[np.ndarray, float, float]:
        """
        Iterate over data source and create an array of patches.

        Parameters
        ----------
        supervised : bool
            Whether the dataset is supervised or not.

        Returns
        -------
        np.ndarray
            Array of patches.
        """
        # if numpy array
        if isinstance(self.data, np.ndarray):
            # supervised case: CARE, N2N, segmentation etc.
            if supervised:
                return prepare_patches_supervised_array(
                    self.data,
                    self.data_target,
                    self.axes,
                    self.patch_size,
                )
            # unsupervised: N2V, PN2V, etc.
            else:
                return prepare_patches_unsupervised_array(
                    self.data,
                    self.axes,
                    self.patch_size,
                )
        # else it is a list of paths
        else:
            # supervised case: CARE, N2N, segmentation etc.
            if supervised:
                return prepare_patches_supervised(
                    self.data,
                    self.data_target,
                    self.axes,
                    self.patch_size,
                    self.read_source_func,
                )
            # unsupervised: N2V, PN2V, etc.
            else:
                return prepare_patches_unsupervised(
                    self.data,
                    self.axes,
                    self.patch_size,
                    self.read_source_func,
                )
        
    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return self.patches.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        Tuple[np.ndarray]
            Patch.

        Raises
        ------
        ValueError
            If dataset mean and std are not set.
        """
        patch = self.patches[index]

        if self.mean is not None and self.std is not None:
            if self.data_target is not None:
                # Splitting targets into a list. 1st dim is the number of targets
                target = self.patch_targets[index, ...]
                # Move channels to the last dimension for the transform
                transformed = self.patch_transform(
                    image=np.moveaxis(patch, 0, -1), target=np.moveaxis(target, 0, -1)
                )
                patch, target = np.moveaxis(transformed["image"], -1, 0), np.moveaxis(
                    transformed["target"], -1, 0
                )  # TODO check if this is correct!
                return patch, target
            else:
                patch = self.patch_transform(image=np.moveaxis(patch, 0, -1))["image"]
                return patch
        else:
            raise ValueError("Dataset mean and std must be set before using it.")


# TODO add tile size
class InMemoryPredictionDataset(torch.utils.data.Dataset):
    """
    Dataset storing data in memory and allowing generating patches from it.

    # TODO
    """

    def __init__(
        self,
        data_config: DataModel,
        data: np.ndarray,
        tile_size: Union[List[int], Tuple[int]],
        tile_overlap: Optional[Union[List[int], Tuple[int]]] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        array : np.ndarray
            Array containing the data.
        axes : str
            Description of axes in format STCZYX.
        patch_size : Union[List[int], Tuple[int]]
            Size of the patches along each axis, must be of dimension 2 or 3.
        mean : Optional[float], optional
            Expected mean of the dataset, by default None.
        std : Optional[float], optional
            Expected standard deviation of the dataset, by default None.

        Raises
        ------
        ValueError
            If data_path is not a directory.
        """
        self.data_config = data_config
        self.axes = data_config.axes
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.mean = data_config.mean
        self.std = data_config.std

        # check that mean and std are provided
        if not self.mean or not self.std:
            raise ValueError(
                f"Mean and std must be provided to the configuration in order to "
                f" perform prediction."
            )


        self.input_array = data
        self.tile = tile_size and tile_overlap

        # Generate patches
        self.data = self._prepare_patches()

    def _prepare_patches(self) -> Callable:
        """
        Iterate over data source and create an array of patches.

        Calls consecutive function for supervised and unsupervised learning.

        Returns
        -------
        np.ndarray
            Array of patches.
        """
        if self.tile:
            return generate_patches_predict(
                self.input_array, self.axes, self.tile_size, self.tile_overlap
            )
        else:
            return self.input_array

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        # convert to numpy array to convince mypy that it is not a generator
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        Tuple[np.ndarray]
            Patch.

        Raises
        ------
        ValueError
            If dataset mean and std are not set.
        """
        if self.tile:
            (
                tile,
                last_tile,
                arr_shape,
                overlap_crop_coords,
                stitch_coords,
            ) = self.data[index]

            return (
                normalize(img=tile, mean=self.mean, std=self.std),
                last_tile,
                arr_shape,
                overlap_crop_coords,
                stitch_coords,
            )
        else:
            return normalize(img=self.data, mean=self.mean, std=self.std).astype(
                np.float32
            )
