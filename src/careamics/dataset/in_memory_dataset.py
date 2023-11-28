"""In-memory dataset module."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..utils import normalize
from ..utils.logging import get_logger
from .dataset_utils import (
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
        Patch transform to apply, by default None.
    patch_transform_params : Optional[Dict], optional
        Patch transform parameters, by default None.
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
        patch_transform_params: Optional[Dict] = None,
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
            Patch transform to apply, by default None.
        patch_transform_params : Optional[Dict], optional
            Patch transform parameters, by default None.

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

        self.patch_transform = patch_transform

        self.train_files = list_files(data_path, self.data_format)
        if self.target_path is not None:
            self.target_files = list_files(self.target_path, self.target_format)
            validate_files(self.train_files, self.target_files)

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_extraction_method = patch_extraction_method
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params

        self.mean = mean
        self.std = std

        # Generate patches(
        self.data, computed_mean, computed_std = self._prepare_patches()

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
                self.patch_overlap,)
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
        return sum(np.array(s).shape[0] for s in self.all_patches)

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
        patch = self.data[index].squeeze()

        if self.mean is not None and self.std is not None:
            patch = normalize(img=patch, mean=self.mean, std=self.std)

            if self.patch_transform is not None:
                # replace None self.patch_transform_params with empty dict
                if self.patch_transform_params is None:
                    self.patch_transform_params = {}

                patch = self.patch_transform(patch, **self.patch_transform_params)
            return patch
        else:
            raise ValueError("Dataset mean and std must be set before using it.")
