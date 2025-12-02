"""Next-Generation CAREamics DataModule."""

import copy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Union, overload

import numpy as np
import pytorch_lightning as L
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.factory import create_dataset
from careamics.dataset_ng.grouped_index_sampler import GroupedIndexSampler
from careamics.dataset_ng.image_stack_loader import ImageStackLoader
from careamics.lightning.dataset_ng.data_module_utils import initialize_data_pair
from careamics.utils import get_logger

logger = get_logger(__name__)

ItemType = Union[Path, str, NDArray[Any]]
"""Type of input items passed to the dataset."""

InputType = Union[ItemType, Sequence[ItemType], None]
"""Type of input data passed to the dataset."""


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
    @overload
    def __init__(
        self,
        data_config: NGDataConfig,
        *,
        train_data: InputType | None = None,
        train_data_target: InputType | None = None,
        val_data: InputType | None = None,
        val_data_target: InputType | None = None,
        pred_data: InputType | None = None,
        pred_data_target: InputType | None = None,
        extension_filter: str = "",
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
    ) -> None: ...

    # with training mask for filtering
    @overload
    def __init__(
        self,
        data_config: NGDataConfig,
        *,
        train_data: InputType | None = None,
        train_data_target: InputType | None = None,
        train_data_mask: InputType,
        val_data: InputType | None = None,
        val_data_target: InputType | None = None,
        pred_data: InputType | None = None,
        pred_data_target: InputType | None = None,
        extension_filter: str = "",
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
    ) -> None: ...

    # custom read function (no mask)
    @overload
    def __init__(
        self,
        data_config: NGDataConfig,
        *,
        train_data: InputType | None = None,
        train_data_target: InputType | None = None,
        val_data: InputType | None = None,
        val_data_target: InputType | None = None,
        pred_data: InputType | None = None,
        pred_data_target: InputType | None = None,
        read_source_func: Callable,
        read_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
    ) -> None: ...

    # custom read function with training mask
    @overload
    def __init__(
        self,
        data_config: NGDataConfig,
        *,
        train_data: InputType | None = None,
        train_data_target: InputType | None = None,
        train_data_mask: InputType,
        val_data: InputType | None = None,
        val_data_target: InputType | None = None,
        pred_data: InputType | None = None,
        pred_data_target: InputType | None = None,
        read_source_func: Callable,
        read_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
    ) -> None: ...

    # image stack loader (no mask)
    @overload
    def __init__(
        self,
        data_config: NGDataConfig,
        *,
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        pred_data: Any | None = None,
        pred_data_target: Any | None = None,
        image_stack_loader: ImageStackLoader,
        image_stack_loader_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
    ) -> None: ...

    # image stack loader with training mask
    @overload
    def __init__(
        self,
        data_config: NGDataConfig,
        *,
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        train_data_mask: Any,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        pred_data: Any | None = None,
        pred_data_target: Any | None = None,
        image_stack_loader: ImageStackLoader,
        image_stack_loader_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
    ) -> None: ...

    def __init__(
        self,
        data_config: NGDataConfig,
        *,
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        train_data_mask: Any | None = None,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        pred_data: Any | None = None,
        pred_data_target: Any | None = None,
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        image_stack_loader: ImageStackLoader | None = None,
        image_stack_loader_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
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

        if train_data is None and val_data is None and pred_data is None:
            raise ValueError(
                "At least one of train_data, val_data or pred_data must be provided."
            )
        elif train_data is None != val_data is None:
            raise ValueError(
                "If one of train_data or val_data is provided, both must be provided."
            )

        self.config: NGDataConfig = data_config
        self.data_type: str = data_config.data_type
        self.batch_size: int = data_config.batch_size

        self.extension_filter: str = (
            extension_filter  # list_files pulls the correct ext
        )
        self.read_source_func = read_source_func
        self.read_kwargs = read_kwargs
        self.image_stack_loader = image_stack_loader
        self.image_stack_loader_kwargs = image_stack_loader_kwargs

        # TODO: implement the validation split logic
        self.val_percentage = val_percentage
        self.val_minimum_split = val_minimum_split
        if self.val_percentage is not None:
            raise NotImplementedError("Validation split is not implemented.")

        custom_loader = self.image_stack_loader is not None
        self.train_data, self.train_data_target = initialize_data_pair(
            self.data_type,
            train_data,
            train_data_target,
            extension_filter,
            custom_loader,
        )
        self.train_data_mask, _ = initialize_data_pair(
            self.data_type, train_data_mask, None, extension_filter, custom_loader
        )

        self.val_data, self.val_data_target = initialize_data_pair(
            self.data_type, val_data, val_data_target, extension_filter, custom_loader
        )

        # The pred_data_target can be needed to count metrics on the prediction
        self.pred_data, self.pred_data_target = initialize_data_pair(
            self.data_type, pred_data, pred_data_target, extension_filter, custom_loader
        )

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
        if stage == "fit":
            if self.config.mode != "training":
                raise ValueError(
                    f"CAREamicsDataModule configured for {self.config.mode} cannot be "
                    f"used for training. Please create a new CareamicsDataModule with "
                    f"a configuration with mode='training'."
                )

            self.train_dataset = create_dataset(
                config=self.config,
                inputs=self.train_data,
                targets=self.train_data_target,
                masks=self.train_data_mask,
                read_func=self.read_source_func,
                read_kwargs=self.read_kwargs,
                image_stack_loader=self.image_stack_loader,
                image_stack_loader_kwargs=self.image_stack_loader_kwargs,
            )
            # TODO: ugly, need to find a better solution
            self.stats = self.train_dataset.input_stats
            self.config.set_means_and_stds(
                self.train_dataset.input_stats.means,
                self.train_dataset.input_stats.stds,
                self.train_dataset.target_stats.means,
                self.train_dataset.target_stats.stds,
            )

            validation_config = self.config.convert_mode("validating")
            self.val_dataset = create_dataset(
                config=validation_config,
                inputs=self.val_data,
                targets=self.val_data_target,
                read_func=self.read_source_func,
                read_kwargs=self.read_kwargs,
                image_stack_loader=self.image_stack_loader,
                image_stack_loader_kwargs=self.image_stack_loader_kwargs,
            )
        elif stage == "validate":
            validation_config = self.config.convert_mode("validating")
            self.val_dataset = create_dataset(
                config=validation_config,
                inputs=self.val_data,
                targets=self.val_data_target,
                read_func=self.read_source_func,
                read_kwargs=self.read_kwargs,
                image_stack_loader=self.image_stack_loader,
                image_stack_loader_kwargs=self.image_stack_loader_kwargs,
            )
            self.stats = self.val_dataset.input_stats
        elif stage == "predict":
            if self.config.mode == "validating":
                raise ValueError(
                    "CAREamicsDataModule configured for validating cannot be used for "
                    "prediction. Please create a new CareamicsDataModule with a "
                    "configuration with mode='predicting'."
                )

            self.predict_dataset = create_dataset(
                config=(
                    self.config.convert_mode("predicting")
                    if self.config.mode == "training"
                    else self.config
                ),
                inputs=self.pred_data,
                targets=self.pred_data_target,
                read_func=self.read_source_func,
                read_kwargs=self.read_kwargs,
                image_stack_loader=self.image_stack_loader,
                image_stack_loader_kwargs=self.image_stack_loader_kwargs,
            )
            self.stats = self.predict_dataset.input_stats
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
            sampler = GroupedIndexSampler.from_dataset(ds, rng=rng)
        else:
            sampler = None
        return sampler

    def train_dataloader(self) -> DataLoader:
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            sampler=sampler,
            **dataloader_params,
        )

    def val_dataloader(self) -> DataLoader:
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            sampler=sampler,
            **dataloader_params,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Create a dataloader for prediction.

        Returns
        -------
        DataLoader
            Prediction dataloader.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            **self.config.pred_dataloader_params,
        )
