"""In-memory dataset module."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from ..config.data_model import DataModel
from ..utils.logging import get_logger
from .dataset_utils import read_tiff
from .patching.patch_transform import get_patch_transform
from .patching.patching import (
    generate_patches_predict,
    prepare_patches_supervised,
    prepare_patches_supervised_array,
    prepare_patches_unsupervised,
    prepare_patches_unsupervised_array,
)

logger = get_logger(__name__)


class InMemoryDataset(Dataset):
    """Dataset storing data in memory and allowing generating patches from it."""

    def __init__(
        self,
        data_config: DataModel,
        data: Union[np.ndarray, List[Path]],
        data_target: Optional[Union[np.ndarray, List[Path]]] = None,
        read_source_func: Callable = read_tiff,
        **kwargs: Any,
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

            # if the transforms are not an instance of Compose
            if data_config.has_transform_list():
                # update mean and std in configuration
                # the object is mutable and should then be recorded in the CAREamist obj
                data_config.set_mean_and_std(self.mean, self.std)
        else:
            self.mean, self.std = data_config.mean, data_config.std

        # get transforms
        self.patch_transform = get_patch_transform(
            patch_transforms=data_config.transforms,
            with_target=self.data_target is not None,
        )

    def _prepare_patches(
        self, supervised: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
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
        if supervised:
            if isinstance(self.data, np.ndarray) and \
                isinstance(self.data_target, np.ndarray):
                return prepare_patches_supervised_array(
                    self.data,
                    self.axes,
                    self.data_target,
                    self.patch_size,
                )
            elif isinstance(self.data, list) and \
                isinstance(self.data_target, list):
                return prepare_patches_supervised(
                    self.data,
                    self.data_target,
                    self.axes,
                    self.patch_size,
                    self.read_source_func,
                )
            else:
                raise ValueError(
                    f"Data and target must be of the same type, either both numpy "
                    f"arrays or both lists of paths, got {type(self.data)} (data) and "
                    f"{type(self.data_target)} (target)."
                )
        else:
            if isinstance(self.data, np.ndarray):
                return prepare_patches_unsupervised_array(
                    self.data,
                    self.axes,
                    self.patch_size,
                )
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

        # if there is a target
        if self.data_target is not None:
            # get target
            target = self.patch_targets[index]

            # Albumentations requires Channel last
            c_patch = np.moveaxis(patch, 0, -1)
            c_target = np.moveaxis(target, 0, -1)

            # Apply transforms
            transformed = self.patch_transform(image=c_patch, target=c_target)

            # move axes back
            patch = np.moveaxis(transformed["image"], -1, 0)
            target = np.moveaxis(transformed["target"], -1, 0)

            return patch, target
        else:
            # Albumentations requires Channel last
            patch = np.moveaxis(patch, 0, -1)

            # Apply transforms
            transformed_patch = self.patch_transform(image=patch)["image"]
            manip_patch, patch, mask = transformed_patch

            # move C axes back
            manip_patch = np.moveaxis(manip_patch, -1, 0)
            patch = np.moveaxis(patch, -1, 0)
            mask = np.moveaxis(mask, -1, 0)

            return (manip_patch, patch, mask)

    def get_number_of_patches(self) -> int:
        """
        Return the number of patches in the dataset.

        Returns
        -------
        int
            Number of patches in the dataset.
        """
        return self.patches.shape[0]

    def split_dataset(
        self,
        percentage: float = 0.1,
        minimum_patches: int = 5,
    ) -> InMemoryDataset:
        """Split a new dataset away from the current one.

        This method is used to extract random validation patches from the dataset.

        Parameters
        ----------
        percentage : float, optional
            Percentage of patches to extract, by default 0.1.
        minimum_patches : int, optional
            Minimum number of patches to extract, by default 5.

        Returns
        -------
        InMemoryDataset
            New dataset with the extracted patches.

        Raises
        ------
        ValueError
            If `percentage` is not between 0 and 1.
        ValueError
            If `minimum_number` is not between 1 and the number of patches.
        """
        if percentage < 0 or percentage > 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}.")

        if minimum_patches < 1 or minimum_patches > self.get_number_of_patches():
            raise ValueError(
                f"Minimum number of patches must be between 1 and "
                f"{self.get_number_of_patches()} (number of patches), got {minimum_patches}."
            )

        total_patches = self.get_number_of_patches()

        # number of patches to extract (either percentage rounded or minimum number)
        n_patches = max(round(total_patches * percentage), minimum_patches)

        # get random indices
        indices = np.random.choice(total_patches, n_patches, replace=False)

        # extract patches
        val_patches = self.patches[indices]

        # remove patches from self.patch
        self.patches = np.delete(self.patches, indices, axis=0)

        # same for targets
        if self.patch_targets is not None:
            val_targets = self.patch_targets[indices]
            self.patch_targets = np.delete(self.patch_targets, indices, axis=0)

        # clone the dataset
        dataset = copy.deepcopy(self)

        # reassign patches
        dataset.patches = val_patches

        # reassign targets
        if self.patch_targets is not None:
            dataset.patch_targets = val_targets

        return dataset


class InMemoryPredictionDataset(Dataset):
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
        data_target: Optional[np.ndarray] = None,
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
        self.data_target = data_target

        # check that mean and std are provided
        if not self.mean or not self.std:
            raise ValueError(
                "Mean and std must be provided to the configuration in order to "
                " perform prediction."
            )
        # TODO this needs to be restructured
        self.input_array = data
        self.tile = tile_size and tile_overlap

        # Generate patches
        self.data = self._prepare_patches()

        # get tta transforms
        self.patch_transform = get_patch_transform(
            patch_transforms=data_config.prediction_transforms,
            with_target=self.data_target is not None,
        )

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
                self.input_array, self.tile_size, self.tile_overlap
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

            # Albumentations requires Channel last
            tile = np.moveaxis(tile, 0, -1)

            # Apply transforms
            transformed_tile = self.patch_transform(image=tile)["image"]
            tile = transformed_tile

            # move C axes back
            tile = np.moveaxis(tile, -1, 0)

            return (
                tile,
                last_tile,
                arr_shape,
                overlap_crop_coords,
                stitch_coords,
            )
        # else:
        #     return normalize(img=self.data, mean=self.mean, std=self.std).astype(
        #         np.float32
        #     )
