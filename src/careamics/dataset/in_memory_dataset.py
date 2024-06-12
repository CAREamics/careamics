"""In-memory dataset module."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from careamics.transforms import Compose

from ..config import DataConfig
from ..config.transformations import NormalizeModel
from ..utils.logging import get_logger
from .dataset_utils import read_tiff
from .patching.patching import (
    prepare_patches_supervised,
    prepare_patches_supervised_array,
    prepare_patches_unsupervised,
    prepare_patches_unsupervised_array,
)

logger = get_logger(__name__)


class InMemoryDataset(Dataset):
    """Dataset storing data in memory and allowing generating patches from it.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    inputs : Union[np.ndarray, List[Path]]
        Input data.
    input_target : Optional[Union[np.ndarray, List[Path]]], optional
        Target data, by default None.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.
    **kwargs : Any
        Additional keyword arguments, unused.
    """

    def __init__(
        self,
        data_config: DataConfig,
        inputs: Union[np.ndarray, List[Path]],
        input_target: Optional[Union[np.ndarray, List[Path]]] = None,
        read_source_func: Callable = read_tiff,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_config : DataConfig
            Data configuration.
        inputs : Union[np.ndarray, List[Path]]
            Input data.
        input_target : Optional[Union[np.ndarray, List[Path]]], optional
            Target data, by default None.
        read_source_func : Callable, optional
            Read source function for custom types, by default read_tiff.
        **kwargs : Any
            Additional keyword arguments, unused.
        """
        self.data_config = data_config
        self.inputs = inputs
        self.input_targets = input_target
        self.axes = self.data_config.axes
        self.patch_size = self.data_config.patch_size

        # read function
        self.read_source_func = read_source_func

        # Generate patches
        supervised = self.input_targets is not None
        patch_data = self._prepare_patches(supervised)

        # Add results to members
        self.patches, self.patch_targets, computed_mean, computed_std = patch_data

        if not self.data_config.mean or not self.data_config.std:
            self.mean, self.std = computed_mean, computed_std
            logger.info(f"Computed dataset mean: {self.mean}, std: {self.std}")

            # update mean and std in configuration
            # the object is mutable and should then be recorded in the CAREamist obj
            self.data_config.set_mean_and_std(self.mean, self.std)
        else:
            self.mean, self.std = self.data_config.mean, self.data_config.std

        # add normalization to transforms and create a compose object
        self.patch_transform = Compose(
            transform_list=[NormalizeModel(mean=self.mean, std=self.std)]
            + self.data_config.transforms,
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
                self.input_targets, np.ndarray
            ):
                return prepare_patches_supervised_array(
                    self.inputs,
                    self.axes,
                    self.input_targets,
                    self.patch_size,
                )
            elif isinstance(self.inputs, list) and isinstance(self.input_targets, list):
                return prepare_patches_supervised(
                    self.inputs,
                    self.input_targets,
                    self.axes,
                    self.patch_size,
                    self.read_source_func,
                )
            else:
                raise ValueError(
                    f"Data and target must be of the same type, either both numpy "
                    f"arrays or both lists of paths, got {type(self.inputs)} (data) "
                    f"and {type(self.input_targets)} (target)."
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
        return len(self.patches)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
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
        if self.patch_targets is not None:
            # get target
            target = self.patch_targets[index]

            return self.patch_transform(patch=patch, target=target)

        elif self.data_config.has_n2v_manipulate():
            return self.patch_transform(patch=patch)
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
