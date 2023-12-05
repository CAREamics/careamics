"""In-memory dataset module."""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..utils import normalize
from ..utils.logging import get_logger
from .dataset_utils import (
    expand_dims,
    get_patch_transform,
    list_files,
    prepare_patches_supervised,
    prepare_patches_unsupervised,
    validate_files,
)
from .extraction_strategy import ExtractionStrategy

logger = get_logger(__name__)


class InMemoryDataset(torch.utils.data.Dataset):
    """
    Dataset storing data in memory and allowing generating patches from it.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data, must be a directory.
    data_format : str
        Extension of the data files, without period.
    axes : str
        Description of axes in format STCZYX.
    patch_extraction_method : ExtractionStrategies
        Patch extraction strategy, as defined in extraction_strategy.
    patch_size : Union[List[int], Tuple[int]]
        Size of the patches along each axis, must be of dimension 2 or 3.
    patch_overlap : Optional[Union[List[int], Tuple[int]]], optional
        Overlap of the patches, must be of dimension 2 or 3, by default None.
    mean : Optional[float], optional
        Expected mean of the dataset, by default None.
    std : Optional[float], optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Optional[Callable], optional
        Patch transform to apply, by default None. Contains type and parameters in a
        dict. Used in N2V family of algorithms, or any custom patch
        manipulation/augmentation.
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        data_format: str,
        axes: str,
        patch_extraction_method: ExtractionStrategy,
        patch_size: Union[List[int], Tuple[int]],
        patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        patch_transform: Optional[Callable] = None,
        target_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        target_format: Optional[str] = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_path : Union[str, Path]
            Path to the data, must be a directory.
        data_format : str
            Extension of the data files, without period.
        axes : str
            Description of axes in format STCZYX.
        patch_extraction_method : ExtractionStrategies
            Patch extraction strategy, as defined in extraction_strategy.
        patch_size : Union[List[int], Tuple[int]]
            Size of the patches along each axis, must be of dimension 2 or 3.
        patch_overlap : Optional[Union[List[int], Tuple[int]]], optional
            Overlap of the patches, must be of dimension 2 or 3, by default None.
        mean : Optional[float], optional
            Expected mean of the dataset, by default None.
        std : Optional[float], optional
            Expected standard deviation of the dataset, by default None.
        patch_transform : Optional[Callable], optional
            Patch transform to apply, by default None. Could be any augmentation
            function, or algorithm specific pixel manipulation (N2V family).
            Please refer to the documentation for more details.
            # TODO add link

        Raises
        ------
        ValueError
            If data_path is not a directory.
        """
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError("Path to data should be an existing folder.")

        self.data_format = data_format
        self.target_path = target_path
        self.target_format = target_format

        self.axes = axes
        # TODO Just 2 for now, add more later
        self.patch_preparation_method = (
            prepare_patches_unsupervised
            if self.target_path is None
            else prepare_patches_supervised
        )
        self.train_files = list_files(data_path, self.data_format)
        if self.target_path is not None:
            if not self.target_path.is_dir():
                raise ValueError("Path to targets should be an existing folder.")
            if self.target_format is None:
                raise ValueError("Target format must be specified.")
            self.target_files = list_files(self.target_path, self.target_format)
            validate_files(self.train_files, self.target_files)

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_extraction_method = patch_extraction_method
        self.patch_transform = get_patch_transform(patch_transform)
        self.patch_transform_params = (
            {} if patch_transform is None else patch_transform["parameters"]
        )

        self.mean = mean
        self.std = std

        # Generate patches
        self.data, self.targets, computed_mean, computed_std = self._prepare_patches()

        if not mean or not std:
            self.mean, self.std = computed_mean, computed_std
            logger.info(f"Computed dataset mean: {self.mean}, std: {self.std}")

        assert self.mean is not None
        assert self.std is not None

    def _prepare_patches(self) -> Callable:
        """
        Iterate over data source and create an array of patches.

        Calls consecutive function for supervised and unsupervised learning.

        Returns
        -------
        np.ndarray
            Array of patches.
        """
        if self.target_path is not None:
            return prepare_patches_supervised(
                self.train_files,
                self.target_files,
                self.axes,
                self.patch_extraction_method,
                self.patch_size,
                self.patch_overlap,
            )
        else:
            return prepare_patches_unsupervised(
                self.train_files,
                self.axes,
                self.patch_extraction_method,
                self.patch_size,
                self.patch_overlap,
            )

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        # convert to numpy array to convince mypy that it is not a generator
        return self.data.shape[0]

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
        if self.target_path is not None:
            # Splitting targets into a list. 1st dim is the number of targets
            target = self.targets[:, index, ...]

        if self.mean is not None and self.std is not None:
            patch = normalize(img=patch, mean=self.mean, std=self.std)
            if self.target_path is not None:
                target = normalize(img=target, mean=self.mean, std=self.std)

            patch = self.patch_transform(patch, **self.patch_transform_params)

            # Needed to add channel dimension in case input image is single channel
            if len(patch.shape) < len(self.patch_size) + 1:
                patch = expand_dims(patch)
            return patch, target
        else:
            raise ValueError("Dataset mean and std must be set before using it.")
