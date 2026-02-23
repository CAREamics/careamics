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
    create_patching_strategy,
    init_patch_extractor,
    select_image_stack_loader,
    select_patch_extractor_type,
)
from careamics.dataset_ng.grouped_index_sampler import GroupedIndexSampler
from careamics.dataset_ng.image_stack import ImageStack
from careamics.dataset_ng.patching_strategies import (
    PatchSpecs,
    StratifiedPatchingStrategy,
    TileSpecs,
)
from careamics.dataset_ng.val_split import create_val_split
from careamics.utils import get_logger

from .data_module_utils import initialize_data_pair

logger = get_logger(__name__)

T = TypeVar("T")
InputVar = TypeVar(
    "InputVar", NDArray[Any], Path, str, Sequence[NDArray[Any]], Sequence[Path | str]
)

Loading = ReadFuncLoading | ImageStackLoading | None


@dataclass
class _TrainVal(Generic[T]):
    train_data: T
    val_data: T
    train_data_target: T | None = None
    val_data_target: T | None = None
    train_data_mask: T | None = None


@dataclass
class _TrainValSplit(Generic[T]):
    train_data: T
    val_percentage: float
    val_minimum_split: int
    train_data_target: T | None = None
    train_data_mask: T | None = None


@dataclass
class _PredData(Generic[T]):
    pred_data: T
    pred_data_target: T | None = None


_Data = _TrainVal[Any] | _TrainValSplit[Any] | _PredData[Any]


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
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
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
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
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
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
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
            val_percentage=val_percentage,
            val_minimum_split=val_minimum_split,
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

            if isinstance(self._data, _TrainValSplit):
                self.train_dataset, self.val_dataset = _create_val_split_datasets(
                    self.config, self._data, self.loading, self.rng
                )
            elif isinstance(self._data, _TrainVal):
                self.train_dataset, self.val_dataset = _create_train_val_datasets(
                    self.config, self._data, self.loading
                )
            else:
                raise ValueError("Training and validation data has not been provided.")
        elif stage == "predict":
            if not isinstance(self._data, _PredData):
                raise ValueError("No data has been provided for prediction.")

            self.predict_dataset = _create_pred_dataset(
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
    val_percentage: float | None = None,
    val_minimum_split: int = 5,
    pred_data: Any | None = None,
    pred_data_target: Any | None = None,
    loading: Loading = None,
) -> _TrainVal[Any] | _TrainValSplit[Any] | _PredData[Any]:
    match train_data, val_data, val_percentage, pred_data:
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
            return _TrainVal(
                train_data=train_data,
                train_data_target=train_data_target,
                train_data_mask=train_data_mask,
                val_data=val_data,
                val_data_target=val_data_target,
            )
        case train_data, None, val_percentage, None if (
            train_data is not None and val_percentage is not None
        ):
            train_data, train_data_target = initialize_data_pair(
                data_type, train_data, train_data_target, loading
            )
            if train_data_mask is not None:
                train_data_mask, _ = initialize_data_pair(
                    data_type, train_data_mask, None, loading
                )
            return _TrainValSplit(
                train_data=train_data,
                train_data_target=train_data_target,
                train_data_mask=train_data_mask,
                val_percentage=val_percentage,
                val_minimum_split=val_minimum_split,
            )
        case None, None, None, pred_data if pred_data is not None:
            pred_data, pred_data_target = initialize_data_pair(
                data_type, pred_data, pred_data_target, loading
            )
            return _PredData(pred_data=pred_data, pred_data_target=pred_data_target)
        case _:
            raise ValueError(
                "Incompatible combination of arguments for CAREamicsDataModule. "
                "Please only provide, training data with validation data OR "
                "training data with validation splitting arguments OR "
                "prediction data."
            )


def _create_train_val_datasets(
    config: NGDataConfig,
    data: _TrainVal[Any],
    loading: Loading,
):

    if config.mode != "training":
        raise ValueError(
            f"CAREamicsDataModule configured for {config.mode} cannot be "
            f"used for training. Please create a new CareamicsDataModule with "
            f"a configuration with mode='training'."
        )

    train_dataset = create_dataset(
        config=config,
        inputs=data.train_data,
        targets=data.train_data_target,
        masks=data.train_data_mask,
        loading=loading,
    )

    validation_config = config.convert_mode("validating")

    val_dataset = create_dataset(
        config=validation_config,
        inputs=data.val_data,
        targets=data.val_data_target,
        loading=loading,
    )

    return train_dataset, val_dataset


def _create_val_split_datasets(
    config: NGDataConfig,
    data: _TrainValSplit[Any],
    loading: Loading,
    rng: np.random.Generator,
) -> tuple[CareamicsDataset[ImageStack], CareamicsDataset[ImageStack]]:
    if config.mode != "training":
        raise ValueError(
            f"CAREamicsDataModule configured for {config.mode} cannot be "
            f"used for training. Please create a new CareamicsDataModule with "
            f"a configuration with mode='training'."
        )
    if config.patching.name != "stratified":
        # TODO: we could optionally split by samples instead.
        raise ValueError(
            "Validation split is only compatible with stratified patching."
        )

    train_input = data.train_data
    train_target = data.train_data_target
    train_mask = data.train_data_mask

    # init dataset components
    image_stack_loader = select_image_stack_loader(
        data_type=SupportedData(config.data_type),
        in_memory=config.in_memory,
        loading=loading,
    )
    patch_extractor_type = select_patch_extractor_type(
        data_type=SupportedData(config.data_type), in_memory=config.in_memory
    )
    input_extractor = init_patch_extractor(
        patch_extractor_type, image_stack_loader, train_input, config.axes
    )
    if train_target is not None:
        target_extractor = init_patch_extractor(
            patch_extractor_type, image_stack_loader, train_target, config.axes
        )
    else:
        target_extractor = None
    if train_mask is not None:
        mask_extractor = init_patch_extractor(
            patch_extractor_type, image_stack_loader, train_mask, config.axes
        )
    else:
        mask_extractor = None

    train_patching = create_patching_strategy(input_extractor.shapes, config.patching)
    # ensured by guard on config at the start of function
    assert isinstance(train_patching, StratifiedPatchingStrategy)

    # calculate n val patches
    n_patches = train_patching.n_patches
    if data.val_minimum_split > n_patches:
        raise RuntimeError(
            f"`val_minimum_split` has been set to {data.val_minimum_split}, which is "
            f"greater than the total available patches, {n_patches}."
        )
    n_val_patches = int(n_patches * data.val_percentage)
    if n_val_patches < data.val_minimum_split:
        n_val_patches = data.val_minimum_split

    # val split applied to patching strat
    train_patching, val_patching = create_val_split(
        train_patching, n_val_patches, rng=rng
    )

    train_dataset = CareamicsDataset(
        data_config=config,
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        mask_extractor=mask_extractor,
        patching_strategy=train_patching,
    )
    val_dataset = CareamicsDataset(
        data_config=config.convert_mode("validating"),
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        mask_extractor=None,
        patching_strategy=val_patching,
    )
    return train_dataset, val_dataset


def _create_pred_dataset(
    config: NGDataConfig,
    data: _PredData[Any],
    loading: Loading,
):
    if config.mode == "validating":
        raise ValueError(
            "CAREamicsDataModule configured for validating cannot be used for "
            "prediction. Please create a new CareamicsDataModule with a "
            "configuration with mode='predicting'."
        )
    return create_dataset(
        config=(
            config.convert_mode("predicting") if config.mode == "training" else config
        ),
        inputs=data.pred_data,
        targets=data.pred_data_target,
        loading=loading,
    )
