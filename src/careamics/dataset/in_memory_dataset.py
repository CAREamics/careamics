"""In-memory dataset module."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

import numpy as np
from torch.utils.data import Dataset

from careamics.config import DataConfig
from careamics.config.transformations import NormalizeModel
from careamics.dataset.patching.patching import (
    PatchedOutput,
    Stats,
    prepare_patches_supervised,
    prepare_patches_supervised_array,
    prepare_patches_unsupervised,
    prepare_patches_unsupervised_array,
)
from careamics.file_io.read import read_tiff
from careamics.transforms import Compose
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


class InMemoryDataset(Dataset):
    """Dataset storing data in memory and allowing generating patches from it.

    Parameters
    ----------
    data_config : CAREamics DataConfig
        (see careamics.config.data_model.DataConfig)
        Data configuration.
    inputs : numpy.ndarray or list[pathlib.Path]
        Input data.
    input_target : numpy.ndarray or list[pathlib.Path], optional
        Target data, by default None.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.
    **kwargs : Any
        Additional keyword arguments, unused.
    """

    def __init__(
        self,
        data_config: DataConfig,
        inputs: Union[np.ndarray, list[Path]],
        input_target: Union[np.ndarray, list[Path]] | None = None,
        read_source_func: Callable = read_tiff,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_config : GeneralDataConfig
            Data configuration.
        inputs : numpy.ndarray or list[pathlib.Path]
            Input data.
        input_target : numpy.ndarray or list[pathlib.Path], optional
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

        # generate patches
        supervised = self.input_targets is not None
        patches_data = self._prepare_patches(supervised)

        # unpack the dataclass
        self.data = patches_data.patches
        self.data_targets = patches_data.targets

        # set image statistics
        if self.data_config.image_means is None:
            self.image_stats = patches_data.image_stats
            logger.info(
                f"Computed dataset mean: {self.image_stats.means}, "
                f"std: {self.image_stats.stds}"
            )
        else:
            self.image_stats = Stats(
                self.data_config.image_means, self.data_config.image_stds
            )

        # set target statistics
        if self.data_config.target_means is None:
            self.target_stats = patches_data.target_stats
        else:
            self.target_stats = Stats(
                self.data_config.target_means, self.data_config.target_stds
            )

        # update mean and std in configuration
        # the object is mutable and should then be recorded in the CAREamist obj
        self.data_config.set_means_and_stds(
            image_means=self.image_stats.means,
            image_stds=self.image_stats.stds,
            target_means=self.target_stats.means,
            target_stds=self.target_stats.stds,
        )
        # get transforms
        self.patch_transform = Compose(
            transform_list=[
                NormalizeModel(
                    image_means=self.image_stats.means,
                    image_stds=self.image_stats.stds,
                    target_means=self.target_stats.means,
                    target_stds=self.target_stats.stds,
                )
            ]
            + list(self.data_config.transforms),
        )

    def _prepare_patches(self, supervised: bool) -> PatchedOutput:
        """
        Iterate over data source and create an array of patches.

        Parameters
        ----------
        supervised : bool
            Whether the dataset is supervised or not.

        Returns
        -------
        numpy.ndarray
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
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        tuple of numpy.ndarray
            Patch.

        Raises
        ------
        ValueError
            If dataset mean and std are not set.
        """
        patch = self.data[index]

        # if there is a target
        if self.data_targets is not None:
            # get target
            target = self.data_targets[index]
            return self.patch_transform(patch=patch, target=target)

        return self.patch_transform(patch=patch)

    def get_data_statistics(self) -> tuple[list[float], list[float]]:
        """Return training data statistics.

        This does not return the target data statistics, only those of the input.

        Returns
        -------
        tuple of list of floats
            Means and standard deviations across channels of the training data.
        """
        return self.image_stats.get_statistics()

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
        CAREamics InMemoryDataset
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
