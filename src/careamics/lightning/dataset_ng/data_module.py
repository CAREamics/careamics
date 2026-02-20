"""Next-Generation CAREamics DataModule."""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, overload

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
    ReadFuncLoading,
    create_dataset,
)
from careamics.dataset_ng.grouped_index_sampler import GroupedIndexSampler
from careamics.dataset_ng.image_stack import ImageStack
from careamics.dataset_ng.patching_strategies import PatchSpecs, TileSpecs
from careamics.lightning.dataset_ng.data_module_utils import initialize_data_pair
from careamics.utils import get_logger

logger = get_logger(__name__)

ItemType = Path | str | NDArray[Any]
"""Type of input items passed to the dataset."""

InputType = ItemType | Sequence[ItemType] | None
"""Type of input data passed to the dataset."""

T = TypeVar("T")
InputVar = TypeVar(
    "InputVar", NDArray[Any], Path, str, Sequence[NDArray[Any]], Sequence[Path | str]
)


@dataclass
class TrainVal(Generic[T]):
    train_data: T
    val_data: T
    train_data_target: T | None = None
    val_data_target: T | None = None
    train_data_mask: T | None = None


@dataclass
class TrainValSplit(Generic[T]):
    train_data: T
    val_percentage: float
    val_minimum_split: int
    train_data_target: T | None = None
    train_data_mask: T | None = None


@dataclass
class PredData(Generic[T]):
    pred_data: T
    pred_data_target: T | None = None


class CareamicsDataModule(L.LightningDataModule):
    """Data module for Careamics dataset.

    Parameters
    ----------
    data_config : DataConfig
        Pydantic model for CAREamics data configuration.
    train_data : Optional[InputType]
        Training data, can be a path to a folder, a list of paths, or a numpy array.
    train_data_target : Optional[InputType]
        Training data target, can be a path to a folder,
        a list of paths, or a numpy array.
    train_data_mask : InputType (when filtering is needed)
        Training data mask, can be a path to a folder,
        a list of paths, or a numpy array. Used for coordinate filtering.
        Only required when using coordinate-based patch filtering.
    val_data : Optional[InputType]
        Validation data, can be a path to a folder,
        a list of paths, or a numpy array.
    val_data_target : Optional[InputType]
        Validation data target, can be a path to a folder,
        a list of paths, or a numpy array.
    pred_data : Optional[InputType]
        Prediction data, can be a path to a folder, a list of paths,
        or a numpy array.
    pred_data_target : Optional[InputType]
        Prediction data target, can be a path to a folder,
        a list of paths, or a numpy array.
    read_source_func : Optional[Callable], default=None
        Function to read the source data. Only used for `custom`
        data type (see DataModel).
    read_kwargs : Optional[dict[str, Any]]
        The kwargs for the read source function.
    image_stack_loader : Optional[ImageStackLoader]
        The image stack loader.
    image_stack_loader_kwargs : Optional[dict[str, Any]]
        The image stack loader kwargs.
    extension_filter : str, default=""
        Filter for file extensions. Only used for `custom` data types
        (see DataModel).
    val_percentage : Optional[float]
        Percentage of the training data to use for validation. Only
        used if `val_data` is None.
    val_minimum_split : int, default=5
        Minimum number of patches or files to split from the training data for
        validation. Only used if `val_data` is None.


    Attributes
    ----------
    config : DataConfig
        Pydantic model for CAREamics data configuration.
    data_type : str
        Type of data, one of SupportedData.
    batch_size : int
        Batch size for the dataloaders.
    extension_filter : str
        Filter for file extensions, by default "".
    read_source_func : Optional[Callable], default=None
        Function to read the source data.
    read_kwargs : Optional[dict[str, Any]], default=None
        The kwargs for the read source function.
    val_percentage : Optional[float]
        Percentage of the training data to use for validation.
    val_minimum_split : int, default=5
        Minimum number of patches or files to split from the training data for
        validation.
    train_data : Optional[Any]
        Training data, can be a path to a folder, a list of paths, or a numpy array.
    train_data_target : Optional[Any]
        Training data target, can be a path to a folder, a list of paths, or a numpy
        array.
    train_data_mask : Optional[Any]
        Training data mask, can be a path to a folder, a list of paths, or a numpy
        array.
    val_data : Optional[Any]
        Validation data, can be a path to a folder, a list of paths, or a numpy array.
    val_data_target : Optional[Any]
        Validation data target, can be a path to a folder, a list of paths, or a numpy
        array.
    pred_data : Optional[Any]
        Prediction data, can be a path to a folder, a list of paths, or a numpy array.
    pred_data_target : Optional[Any]
        Prediction data target, can be a path to a folder, a list of paths, or a numpy
        array.

    Raises
    ------
    ValueError
        If at least one of train_data, val_data or pred_data is not provided.
    ValueError
        If input and target data types are not consistent.
    """

    # standard use (no mask)
    # TODO: remove pred data from overloads?
    @overload
    def __init__(
        self,
        data_config: NGDataConfig | dict[str, Any],
        *,
        data: TrainVal[InputVar] | TrainValSplit[InputVar] | PredData[InputVar],
        loading: ReadFuncLoading | None = None,
    ): ...

    @overload
    def __init__(
        self,
        data_config: NGDataConfig | dict[str, Any],
        *,
        data: TrainVal[Any] | TrainValSplit[Any] | PredData[Any],
        loading: ImageStackLoading,
    ): ...

    def __init__(
        self,
        data_config: NGDataConfig | dict[str, Any],
        *,
        data: TrainVal[Any] | TrainValSplit[Any] | PredData[Any],
        loading: ReadFuncLoading | ImageStackLoading | None = None,
    ):
        """
        Data module for Careamics dataset initialization.

        Create a lightning datamodule that handles creating datasets for training,
        validation, and prediction.

        Parameters
        ----------
        data_config : NGDataConfig
            Pydantic model for CAREamics data configuration.
        train_data : Optional[InputType]
            Training data, can be a path to a folder, a list of paths, or a numpy array.
        train_data_target : Optional[InputType]
            Training data target, can be a path to a folder,
            a list of paths, or a numpy array.
        train_data_mask : InputType (when filtering is needed)
            Training data mask, can be a path to a folder,
            a list of paths, or a numpy array. Used for coordinate filtering.
            Only required when using coordinate-based patch filtering.
        val_data : Optional[InputType]
            Validation data, can be a path to a folder,
            a list of paths, or a numpy array.
        val_data_target : Optional[InputType]
            Validation data target, can be a path to a folder,
            a list of paths, or a numpy array.
        pred_data : Optional[InputType]
            Prediction data, can be a path to a folder, a list of paths,
            or a numpy array.
        pred_data_target : Optional[InputType]
            Prediction data target, can be a path to a folder,
            a list of paths, or a numpy array.
        read_source_func : Optional[Callable]
            Function to read the source data, by default None. Only used for `custom`
            data type (see DataModel).
        read_kwargs : Optional[dict[str, Any]]
            The kwargs for the read source function.
        image_stack_loader : Optional[ImageStackLoader]
            The image stack loader.
        image_stack_loader_kwargs : Optional[dict[str, Any]]
            The image stack loader kwargs.
        extension_filter : str
            Filter for file extensions, by default "". Only used for `custom` data types
            (see DataModel).
        val_percentage : Optional[float]
            Percentage of the training data to use for validation. Only
            used if `val_data` is None.
        val_minimum_split : int
            Minimum number of patches or files to split from the training data for
            validation, by default 5. Only used if `val_data` is None.
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

        self.data_type: SupportedData = SupportedData(self.config.data_type)
        self.batch_size: int = self.config.batch_size

        self.data: TrainVal[Any] | TrainValSplit[Any] | PredData[Any] = data
        self.loading = loading

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

            if not isinstance(self.data, TrainVal):
                raise ValueError

            self.train_dataset, self.val_dataset = create_train_val_datasets(
                self.config, self.data, self.loading
            )
        elif stage == "predict":
            if not isinstance(self.data, PredData):
                raise ValueError("No data has been provided for prediction.")

            self.predict_dataset = create_pred_dataset(
                self.config, self.data, self.loading
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


def create_train_val_datasets(
    config: NGDataConfig,
    data: TrainVal[Any],
    loading: ReadFuncLoading | ImageStackLoading | None,
):

    if config.mode != "training":
        raise ValueError(
            f"CAREamicsDataModule configured for {config.mode} cannot be "
            f"used for training. Please create a new CareamicsDataModule with "
            f"a configuration with mode='training'."
        )
    train_input = data.train_data
    train_target = data.train_data_target
    train_mask = data.train_data_mask
    val_input = data.val_data
    val_target = data.val_data_target

    train_input, train_target = initialize_data_pair(
        config.data_type, train_input, train_target, loading
    )
    if train_mask is not None:
        train_mask, _ = initialize_data_pair(
            config.data_type, train_mask, None, loading
        )
    val_input, val_target = initialize_data_pair(
        config.data_type, val_input, val_target, loading
    )

    train_dataset = create_dataset(
        config=config,
        inputs=train_input,
        targets=train_target,
        masks=train_mask,
        loading=loading,
    )

    validation_config = config.convert_mode("validating")

    val_dataset = create_dataset(
        config=validation_config,
        inputs=val_input,
        targets=val_target,
        loading=loading,
    )

    return train_dataset, val_dataset


def create_pred_dataset(
    config: NGDataConfig,
    data: PredData[Any],
    loading: ReadFuncLoading | ImageStackLoading | None,
):
    if config.mode == "validating":
        raise ValueError(
            "CAREamicsDataModule configured for validating cannot be used for "
            "prediction. Please create a new CareamicsDataModule with a "
            "configuration with mode='predicting'."
        )
    pred_input = data.pred_data
    pred_target = data.pred_data_target
    pred_input, pred_target = initialize_data_pair(
        SupportedData(config.data_type), pred_input, pred_target, loading
    )
    return create_dataset(
        config=(
            config.convert_mode("predicting") if config.mode == "training" else config
        ),
        inputs=pred_input,
        targets=pred_target,
        loading=loading,
    )
