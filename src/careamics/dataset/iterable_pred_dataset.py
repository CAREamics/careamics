"""Iterable prediction dataset used to load data file by file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator, List

import numpy as np
from torch.utils.data import IterableDataset

from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.transformations import NormalizeModel
from .dataset_utils import iterate_over_files, read_tiff


class IterablePredictionDataset(IterableDataset):
    """Simple iterable prediction dataset.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Inference configuration.
    src_files : List[Path]
        List of data files.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.
    **kwargs : Any
        Additional keyword arguments, unused.

    Attributes
    ----------
    data_path : Union[str, Path]
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    mean : Optional[float], optional
        Expected mean of the dataset, by default None.
    std : Optional[float], optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Optional[Callable], optional
        Patch transform callable, by default None.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        src_files: List[Path],
        read_source_func: Callable = read_tiff,
        **kwargs: Any,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Inference configuration.
        src_files : List[Path]
            List of data files.
        read_source_func : Callable, optional
            Read source function for custom types, by default read_tiff.
        **kwargs : Any
            Additional keyword arguments, unused.

        Raises
        ------
        ValueError
            If mean and std are not provided in the inference configuration.
        """
        self.prediction_config = prediction_config
        self.data_files = src_files
        self.axes = prediction_config.axes
        self.read_source_func = read_source_func

        # check mean and std and create normalize transform
        if self.prediction_config.mean is None or self.prediction_config.std is None:
            raise ValueError("Mean and std must be provided for prediction.")
        else:
            self.mean = self.prediction_config.mean
            self.std = self.prediction_config.std

            # instantiate normalize transform
            self.patch_transform = Compose(
                transform_list=[
                    NormalizeModel(
                        mean=prediction_config.mean, std=prediction_config.std
                    )
                ],
            )

    def __iter__(
        self,
    ) -> Generator[np.ndarray, None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
            Single patch.
        """
        assert (
            self.mean is not None and self.std is not None
        ), "Mean and std must be provided"

        for sample, _ in iterate_over_files(
            self.prediction_config,
            self.data_files,
            read_source_func=self.read_source_func,
        ):
            # TODO what if S dimensions > 1, should we yield each sample independently?
            transformed_sample, _ = self.patch_transform(patch=sample)

            yield transformed_sample
