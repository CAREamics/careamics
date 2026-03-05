"""Next-Generation CAREamics DataModule."""

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

import numpy as np
import pytorch_lightning as L
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.dataset import CareamicsDataset, ImageRegionData
from careamics.dataset_ng.factory import (
    ImageStackLoading,
    Loading,
    PredData,
    ReadFuncLoading,
    TrainValData,
    TrainValSplitData,
    create_pred_dataset,
    create_train_val_datasets,
    create_val_split_datasets,
)
from careamics.dataset_ng.grouped_index_sampler import GroupedIndexSampler
from careamics.dataset_ng.image_stack import ImageStack
from careamics.dataset_ng.patching_strategies import (
    PatchSpecs,
    TileSpecs,
)
from careamics.utils import get_logger

from .data_module_utils import initialize_data_pair

logger = get_logger(__name__)


InputVar = TypeVar(
    "InputVar", NDArray[Any], Path, str, Sequence[NDArray[Any]], Sequence[Path | str]
)
"""
Data source types, numpy arrays or paths or sequences of either.

(Paths can be `str` or `pathlib.Path`).
"""


_Data = TrainValData[Any] | TrainValSplitData[Any] | PredData[Any]
"""Data for training with validation or validation splitting or data for prediction."""


class CareamicsDataModule(L.LightningDataModule):
    """Data module for Careamics dataset.

    Parameters
    ----------
    data_config : NGDataConfig
        Pydantic model for CAREamics data configuration.
    train_data : Any, default=None
        Training data. If custom `loading` is provided it can be any type, otherwise
        it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these,
        or None.
    train_data_target : Any, default=None
        Training data target. If custom `loading` is provided it can be any type,
        otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
        of these, or None.
    train_data_mask : Any, default=None.
        Training data mask, an optional mask that can be provided to filter regions
        of the data during training, such as large areas of background. The mask
        should be a binary image where a 1 indicates a pixel should be included in
        the training data.
        If custom `loading` is provided it can be any type, otherwise
        it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these,
        or None.
    val_data : Any, default=None
        Validation data. If custom `loading` is provided it can be any type,
        otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
        of these, or None.
    val_data_target : Any, default=None
        Validation data target. If custom `loading` is provided it can be any type,
        otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
        of these, or None.
    val_percentage : float | None, default=None
        Percentage of the training data to use for validation. Only
        used if `val_data` is None.
    val_minimum_split : int
        Minimum number of patches or files to split from the training data for
        validation, by default 5. Only used if `val_data` is None.
    pred_data : Any, default=None
        Prediction data. If custom `loading` is provided it can be any type,
        otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
        of these, or None.
    pred_data_target : Any, default=None
        Prediction data target, this may be used for calculating metrics. If custom
        `loading` is provided it can be any type, otherwise it must be a
        `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these, or None.
    loading : ReadFuncLoading | ImageStackLoading | None, default=None
        The type of loading used for custom data. `ReadFuncLoading` is the use of
        a simple function that will load full images into memory.
        `ImageStackLoading` is for custom chunked or memory-mapped next-generation
        file formats enabling  single patches to be read from disk at a time.
        If the data type is not custom `loading` should be `None`.

    Attributes
    ----------
    config : DataConfig
        Pydantic model for CAREamics data configuration.
    data_type : str
        Type of data, one of SupportedData.
    batch_size : int
        Batch size for the dataloaders.

    Raises
    ------
    ValueError
        If at least one of train_data, val_data or pred_data is not provided.
    ValueError
        If input and target data types are not consistent.
    """

    # if not using ImageStackLoading the input type should be array or path or sequence
    @overload
    def __init__(
        self,
        data_config: NGDataConfig | dict[str, Any],
        *,
        train_data: InputVar | None = None,
        train_data_target: InputVar | None = None,
        train_data_mask: InputVar | None = None,
        val_data: InputVar | None = None,
        val_data_target: InputVar | None = None,
        n_val_patches: int | None = None,
        pred_data: InputVar | None = None,
        pred_data_target: InputVar | None = None,
        loading: ReadFuncLoading | None = None,
    ) -> None: ...

    # if using ImageStackLoading the input data can be anything.
    @overload
    def __init__(
        self,
        data_config: NGDataConfig | dict[str, Any],
        *,
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        train_data_mask: Any | None = None,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        n_val_patches: int | None = None,
        pred_data: Any | None = None,
        pred_data_target: Any | None = None,
        loading: ImageStackLoading = ...,
    ) -> None: ...
    def __init__(
        self,
        data_config: NGDataConfig | dict[str, Any],
        *,
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        train_data_mask: Any | None = None,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        n_val_patches: int | None = None,
        pred_data: Any | None = None,
        pred_data_target: Any | None = None,
        loading: Loading = None,
    ) -> None:
        """
        Data module for Careamics dataset initialization.

        Create a lightning datamodule that handles creating datasets for training,
        validation, and prediction.

        Parameters
        ----------
        data_config : NGDataConfig
            Pydantic model for CAREamics data configuration.
        train_data : Any, default=None
            Training data. If custom `loading` is provided it can be any type, otherwise
            it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these,
            or None.
        train_data_target : Any, default=None
            Training data target. If custom `loading` is provided it can be any type,
            otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
            of these, or None.
        train_data_mask : Any, default=None.
            Training data mask, an optional mask that can be provided to filter regions
            of the data during training, such as large areas of background. The mask
            should be a binary image where a 1 indicates a pixel should be included in
            the training data.
            If custom `loading` is provided it can be any type, otherwise
            it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these,
            or None.
        val_data : Any, default=None
            Validation data. If custom `loading` is provided it can be any type,
            otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
            of these, or None.
        val_data_target : Any, default=None
            Validation data target. If custom `loading` is provided it can be any type,
            otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
            of these, or None.
        val_percentage : float | None, default=None
            Percentage of the training data to use for validation. Only
            used if `val_data` is None.
        val_minimum_split : int
            Minimum number of patches or files to split from the training data for
            validation, by default 5. Only used if `val_data` is None.
        pred_data : Any, default=None
            Prediction data. If custom `loading` is provided it can be any type,
            otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
            of these, or None.
        pred_data_target : Any, default=None
            Prediction data target, this may be used for calculating metrics. If custom
            `loading` is provided it can be any type, otherwise it must be a
            `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these, or None.
        loading : ReadFuncLoading | ImageStackLoading | None, default=None
            The type of loading used for custom data. `ReadFuncLoading` is the use of
            a simple function that will load full images into memory.
            `ImageStackLoading` is for custom chunked or memory-mapped next-generation
            file formats enabling  single patches to be read from disk at a time.
            If the data type is not custom `loading` should be `None`.
        """
        super().__init__()

        if isinstance(data_config, NGDataConfig):
            self.config = data_config
        else:
            self.config = NGDataConfig.model_validate(data_config)
        self.save_hyperparameters(
            {"data_config": self.config.model_dump(mode="json")},
            ignore=[
                "train_data",
                "train_data_target",
                "train_data_mask",
                "val_data",
                "val_data_target",
                "pred_data",
                "pred_data_target",
                "read_source_func",
                "read_kwargs",
                "image_stack_loader",
                "image_stack_loader_kwargs",
                "extension_filter",
                "val_percentage",
                "val_minimum_split",
            ],
        )

        self.rng = np.random.default_rng(seed=self.config.seed)

        self.data_type: SupportedData = SupportedData(self.config.data_type)
        self.batch_size: int = self.config.batch_size

        self._data: _Data = _validate_data(
            self.data_type,
            train_data=train_data,
            train_data_target=train_data_target,
            train_data_mask=train_data_mask,
            val_data=val_data,
            val_data_target=val_data_target,
            n_val_patches=n_val_patches,
            pred_data=pred_data,
            pred_data_target=pred_data_target,
            loading=loading,
        )

        self.loading: Loading = loading

        self.train_dataset: CareamicsDataset[ImageStack] | None = None
        self.val_dataset: CareamicsDataset[ImageStack] | None = None
        self.predict_dataset: CareamicsDataset[ImageStack] | None = None

    def setup(self, stage: str) -> None:
        """
        Setup datasets.

        Lightning hook that is called at the beginning of fit (train + validate),
        validate, test, or predict. Creates the datasets for a given stage.

        Parameters
        ----------
        stage : str
            The stage to set up datasets for.
            Is either 'fit', 'validate', 'test', or 'predict'.

        Raises
        ------
        NotImplementedError
            If stage is not one of "fit", "validate" or "predict".
        """
        if stage == "fit" or stage == "validate":
            if (self.train_dataset is not None) and (self.val_dataset is not None):
                return

            if isinstance(self._data, TrainValSplitData):
                self.train_dataset, self.val_dataset = create_val_split_datasets(
                    self.config, self._data, self.loading, self.rng
                )
            elif isinstance(self._data, TrainValData):
                self.train_dataset, self.val_dataset = create_train_val_datasets(
                    self.config, self._data, self.loading
                )
            else:
                raise ValueError("Training and validation data has not been provided.")
        elif stage == "predict":
            if not isinstance(self._data, PredData):
                raise ValueError("No data has been provided for prediction.")

            self.predict_dataset = create_pred_dataset(
                self.config, self._data, self.loading
            )
        else:
            raise NotImplementedError(f"Stage {stage} not implemented")

    def _sampler(self, dataset: Literal["train", "val", "predict"]) -> Sampler | None:
        sampler: GroupedIndexSampler | None
        rng = np.random.default_rng(self.config.seed)
        if not self.config.in_memory and self.config.data_type == SupportedData.TIFF:
            match dataset:
                case "train":
                    ds = self.train_dataset
                case "val":
                    ds = self.val_dataset
                case "predict":
                    ds = self.predict_dataset
                case _:
                    raise (
                        f"Unrecognized dataset '{dataset}', should be one of 'train', "
                        "'val' or 'predict'."
                    )
            assert ds is not None
            sampler = GroupedIndexSampler.from_dataset(ds, rng=rng)
        else:
            sampler = None
        return sampler

    def train_dataloader(self) -> DataLoader[ImageRegionData[PatchSpecs]]:
        """
        Create a dataloader for training.

        Returns
        -------
        DataLoader
            Training dataloader.
        """
        sampler = self._sampler("train")
        dataloader_params = copy.deepcopy(self.config.train_dataloader_params)
        # have to remove shuffle with sampler because of torch error:
        #   ValueError: sampler option is mutually exclusive with shuffle
        # TODO: there might be other parameters mutually exclusive with sampler
        if (sampler is not None) and ("shuffle" in dataloader_params):
            del dataloader_params["shuffle"]
        assert self.train_dataset is not None
        return DataLoader[ImageRegionData[PatchSpecs]](
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            sampler=sampler,
            **dataloader_params,
        )

    def val_dataloader(self) -> DataLoader[ImageRegionData[PatchSpecs]]:
        """
        Create a dataloader for validation.

        Returns
        -------
        DataLoader
            Validation dataloader.
        """
        sampler = self._sampler("val")
        dataloader_params = copy.deepcopy(self.config.val_dataloader_params)
        if (sampler is not None) and ("shuffle" in dataloader_params):
            del dataloader_params["shuffle"]
        assert self.val_dataset is not None
        return DataLoader[ImageRegionData[PatchSpecs]](
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            sampler=sampler,
            **dataloader_params,
        )

    def predict_dataloader(self) -> DataLoader[ImageRegionData[TileSpecs]]:
        """
        Create a dataloader for prediction.

        Returns
        -------
        DataLoader
            Prediction dataloader.
        """
        assert self.predict_dataset is not None
        return DataLoader[ImageRegionData[TileSpecs]](
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            **self.config.pred_dataloader_params,
        )


def _validate_data(
    data_type: SupportedData,
    train_data: Any | None = None,
    train_data_target: Any | None = None,
    train_data_mask: Any | None = None,
    val_data: Any | None = None,
    val_data_target: Any | None = None,
    n_val_patches: int | None = None,
    pred_data: Any | None = None,
    pred_data_target: Any | None = None,
    loading: Loading = None,
) -> TrainValData[Any] | TrainValSplitData[Any] | PredData[Any]:
    """Validate the combination of input arguments and their types.

    Parameters
    ----------
    data_type : SupportedData
        The type of the data to validate against.
    train_data : Any, default=None
        Training data. If custom `loading` is provided it can be any type, otherwise
        it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these,
        or None.
    train_data_target : Any, default=None
        Training data target. If custom `loading` is provided it can be any type,
        otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
        of these, or None.
    train_data_mask : Any, default=None.
        Training data mask, an optional mask that can be provided to filter regions
        of the data during training, such as large areas of background. The mask
        should be a binary image where a 1 indicates a pixel should be included in
        the training data.
        If custom `loading` is provided it can be any type, otherwise
        it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these,
        or None.
    val_data : Any, default=None
        Validation data. If custom `loading` is provided it can be any type,
        otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
        of these, or None.
    val_data_target : Any, default=None
        Validation data target. If custom `loading` is provided it can be any type,
        otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
        of these, or None.
    val_percentage : float | None, default=None
        Percentage of the training data to use for validation. Only
        used if `val_data` is None.
    val_minimum_split : int
        Minimum number of patches or files to split from the training data for
        validation, by default 5. Only used if `val_data` is None.
    pred_data : Any, default=None
        Prediction data. If custom `loading` is provided it can be any type,
        otherwise it must be a `pathlib.Path`, `str`, `numpy.ndarray` or a sequence
        of these, or None.
    pred_data_target : Any, default=None
        Prediction data target, this may be used for calculating metrics. If custom
        `loading` is provided it can be any type, otherwise it must be a
        `pathlib.Path`, `str`, `numpy.ndarray` or a sequence of these, or None.
    loading : ReadFuncLoading | ImageStackLoading | None, default=None
        The type of loading used for custom data. `ReadFuncLoading` is the use of
        a simple function that will load full images into memory.
        `ImageStackLoading` is for custom chunked or memory-mapped next-generation
        file formats enabling  single patches to be read from disk at a time.
        If the data type is not custom `loading` should be `None`.

    Returns
    -------
    data : _TrainVal[Any] | _TrainValSplit[Any] | _PredData[Any]
        The validated data wrapped in a dataclass. The `_TrainVal` class is for training
        with validation data provided; the `_TrainValSplit` class is used for training
        with automatic validation splitting, and the `_PredData` class is used for
        prediction.

    Raises
    ------
    ValueError
        In the case of incompatible combinations of arguments.
    """
    match train_data, val_data, n_val_patches, pred_data:
        case train_data, val_data, None, None if (
            train_data is not None and val_data is not None
        ):
            train_data, train_data_target = initialize_data_pair(
                data_type, train_data, train_data_target, loading
            )
            if train_data_mask is not None:
                train_data_mask, _ = initialize_data_pair(
                    data_type, train_data_mask, None, loading
                )
            val_data, val_data_target = initialize_data_pair(
                data_type, val_data, val_data_target, loading
            )
            return TrainValData(
                train_data=train_data,
                train_data_target=train_data_target,
                train_data_mask=train_data_mask,
                val_data=val_data,
                val_data_target=val_data_target,
            )
        case train_data, None, n_val_patches, None if (
            train_data is not None and n_val_patches is not None
        ):
            train_data, train_data_target = initialize_data_pair(
                data_type, train_data, train_data_target, loading
            )
            if train_data_mask is not None:
                train_data_mask, _ = initialize_data_pair(
                    data_type, train_data_mask, None, loading
                )
            return TrainValSplitData(
                train_data=train_data,
                train_data_target=train_data_target,
                train_data_mask=train_data_mask,
                n_val_patches=n_val_patches,
            )
        case None, None, None, pred_data if pred_data is not None:
            pred_data, pred_data_target = initialize_data_pair(
                data_type, pred_data, pred_data_target, loading
            )
            return PredData(pred_data=pred_data, pred_data_target=pred_data_target)
        case _:
            raise ValueError(
                "Incompatible combination of arguments for CAREamicsDataModule. "
                "Please only provide, training data with validation data OR "
                "training data with validation splitting arguments OR "
                "prediction data."
            )
