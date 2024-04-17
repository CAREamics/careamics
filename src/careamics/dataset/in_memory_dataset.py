"""In-memory dataset module."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from ..config import DataModel, InferenceModel
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
        inputs: Union[np.ndarray, List[Path]],
        data_target: Optional[Union[np.ndarray, List[Path]]] = None,
        read_source_func: Callable = read_tiff,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        # TODO
        """
        self.data_config = data_config
        self.inputs = inputs
        self.data_target = data_target
        self.axes = self.data_config.axes
        self.patch_size = self.data_config.patch_size

        # read function
        self.read_source_func = read_source_func

        # Generate patches
        supervised = self.data_target is not None
        patches = self._prepare_patches(supervised)

        # Add results to members
        self.data, self.data_targets, computed_mean, computed_std = patches

        if not self.data_config.mean or not self.data_config.std:
            self.mean, self.std = computed_mean, computed_std
            logger.info(f"Computed dataset mean: {self.mean}, std: {self.std}")

            # if the transforms are not an instance of Compose
            if self.data_config.has_transform_list():
                # update mean and std in configuration
                # the object is mutable and should then be recorded in the CAREamist obj
                self.data_config.set_mean_and_std(self.mean, self.std)
        else:
            self.mean, self.std = self.data_config.mean, self.data_config.std

        # get transforms
        self.patch_transform = get_patch_transform(
            patch_transforms=self.data_config.transforms,
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
            if isinstance(self.inputs, np.ndarray) and isinstance(
                self.data_target, np.ndarray
            ):
                return prepare_patches_supervised_array(
                    self.inputs,
                    self.axes,
                    self.data_target,
                    self.patch_size,
                )
            elif isinstance(self.inputs, list) and isinstance(self.data_target, list):
                return prepare_patches_supervised(
                    self.inputs,
                    self.data_target,
                    self.axes,
                    self.patch_size,
                    self.read_source_func,
                )
            else:
                raise ValueError(
                    f"Data and target must be of the same type, either both numpy "
                    f"arrays or both lists of paths, got {type(self.inputs)} (data) "
                    f"and {type(self.data_target)} (target)."
                )
        else:
            if isinstance(self.inputs, np.ndarray):
                return prepare_patches_unsupervised_array(
                    self.inputs,
                    self.axes,
                    self.patch_size,
                )
            else:
                return prepare_patches_unsupervised(
                    self.inputs,
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
        patch = self.data[index]

        # if there is a target
        if self.data_target is not None:
            # get target
            target = self.data_targets[index]

            # Albumentations requires Channel last
            c_patch = np.moveaxis(patch, 0, -1)
            c_target = np.moveaxis(target, 0, -1)

            # Apply transforms
            transformed = self.patch_transform(image=c_patch, target=c_target)

            # move axes back
            patch = np.moveaxis(transformed["image"], -1, 0)
            target = np.moveaxis(transformed["target"], -1, 0)

            return patch, target

        elif self.data_config.has_n2v_manipulate():
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
        else:
            raise ValueError(
                "Something went wrong! No target provided (not supervised training) "
                "and no N2V manipulation (no N2V training)."
            )

    def split_dataset(
        self,
        percentage: float = 0.1,
        minimum_patches: int = 1,
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

        if minimum_patches < 1 or minimum_patches > len(self):
            raise ValueError(
                f"Minimum number of patches must be between 1 and "
                f"{len(self)} (number of patches), got "
                f"{minimum_patches}. Adjust the patch size or the minimum number of "
                f"patches."
            )

        total_patches = len(self)

        # number of patches to extract (either percentage rounded or minimum number)
        n_patches = max(round(total_patches * percentage), minimum_patches)

        # get random indices
        indices = np.random.choice(total_patches, n_patches, replace=False)

        # extract patches
        val_patches = self.data[indices]

        # remove patches from self.patch
        self.data = np.delete(self.data, indices, axis=0)

        # same for targets
        if self.data_targets is not None:
            val_targets = self.data_targets[indices]
            self.data_targets = np.delete(self.data_targets, indices, axis=0)

        # clone the dataset
        dataset = copy.deepcopy(self)

        # reassign patches
        dataset.data = val_patches

        # reassign targets
        if self.data_targets is not None:
            dataset.data_targets = val_targets

        return dataset


class InMemoryPredictionDataset(InMemoryDataset):
    """
    Dataset storing data in memory and allowing generating patches from it.

    # TODO
    """

    def __init__(
        self,
        prediction_config: InferenceModel,
        inputs: np.ndarray,
        data_target: Optional[np.ndarray] = None,
        read_source_func: Optional[Callable] = read_tiff,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        array : np.ndarray
            Array containing the data.
        axes : str
            Description of axes in format STCZYX.

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
        self.tiling = self.tile_size and self.tile_overlap
        self.mean = self.pred_config.mean
        self.std = self.pred_config.std
        self.data_target = data_target

        # read function
        self.read_source_func = read_source_func

        # Generate patches
        tiles = self._prepare_tiles()

        # Add results to members
        self.data, computed_mean, computed_std = tiles

        if not self.pred_config.mean or not self.pred_config.std:
            self.mean, self.std = computed_mean, computed_std
            logger.info(f"Computed dataset mean: {self.mean}, std: {self.std}")

            # if the transforms are not an instance of Compose
            if hasattr(self.pred_config, "has_transform_list"):
                if self.pred_config.has_transform_list():
                    # update mean and std in configuration
                    # the object is mutable and should then be recorded in the CAREamist
                    self.pred_config.set_mean_and_std(self.mean, self.std)
            else:
                self.pred_config.set_mean_and_std(self.mean, self.std)
        else:
            self.mean, self.std = self.pred_config.mean, self.pred_config.std

        # get transforms
        self.patch_transform = get_patch_transform(
            patch_transforms=self.pred_config.transforms,
            with_target=self.data_target is not None,
        )

    def _prepare_tiles(self) -> Callable:
        """
        Iterate over data source and create an array of patches.

        Calls consecutive function for supervised and unsupervised learning.

        Returns
        -------
        np.ndarray
            Array of patches.
        """
        if self.tiling:
            return generate_patches_predict(
                self.input_array, self.axes, self.tile_size, self.tile_overlap
            ), self.input_array.mean(), self.input_array.std()
        else:
            return self.input_array, self.input_array.mean(), self.input_array.std()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Any, Any, Any, Any]:
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
        if self.tiling:
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
            )  # TODO can we wrap this into an object?
