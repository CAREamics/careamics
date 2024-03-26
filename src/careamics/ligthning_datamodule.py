from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as L
from albumentations import Compose
from torch.utils.data import DataLoader

from careamics.config import DataModel, PredictionModel
from careamics.config.support import SupportedData, SupportedTransform
from careamics.dataset.dataset_utils import (
    get_files_size,
    get_read_func,
    list_files,
    reshape_array,
    validate_source_target_files,
)
from careamics.dataset.in_memory_dataset import (
    InMemoryDataset,
    InMemoryPredictionDataset,
)
from careamics.dataset.iterable_dataset import (
    IterablePredictionDataset,
    PathIterableDataset,
)
from careamics.utils import get_logger, get_ram_size

DatasetType = Union[InMemoryDataset, PathIterableDataset]
PredictDatasetType = Union[InMemoryPredictionDataset, IterablePredictionDataset]

logger = get_logger(__name__)

class CAREamicsWood(L.LightningDataModule):
    def __init__(
        self,
        data_config: DataModel,
        train_data: Union[Path, str, np.ndarray],
        val_data: Optional[Union[Path, str, np.ndarray]] = None,
        train_data_target: Optional[Union[Path, str, np.ndarray]] = None,
        val_data_target: Optional[Union[Path, str, np.ndarray]] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        val_percentage: float = 0.1,
        val_minimum_split: int = 5,
        use_in_memory: bool = True,
        num_workers: int = 0,
        #dataloader_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """LightningDataModule for CAREamics training, including training and validation
        datasets.

        The data module can be used with Path, str or numpy arrays. In the case of
        numpy arrays, it loads and computes all the patches in memory. For Path and str
        inputs, it calculates the total file size and estimate whether it can fit in
        memory. If it does not, it iterates through the files. This behaviour can be
        deactivated by setting `use_in_memory` to False, in which case it will
        always use the iterating dataset to train on a Path or str.

        The data can be either a folder containing images or a single file.

        Validation can be omitted, in which case the validation data is extracted from
        the training data. The percentage of the training data to use for validation,
        as well as the minimum number of patches or files to split from the training
        data can be set using `val_percentage` and `val_minimum_split`, respectively.

        To read custom data types, you can set `data_type` to `custom` in `data_config`
        and provide a function that returns a numpy array from a path as
        `read_source_func` parameter. The function will receive a Path object and
        an axies string as arguments, the axes being derived from the `data_config`.

        You can also provide a `fnmatch` and `Path.rglob` compatible expression (e.g.
        "*.czi") to filter the files extension using `extension_filter`.

        Parameters
        ----------
        data_config : DataModel
            Pydantic model for CAREamics data configuration.
        train_data : Union[Path, str, np.ndarray]
            Training data, can be a path to a folder, a file or a numpy array.
        val_data : Optional[Union[Path, str, np.ndarray]], optional
            Validation data, can be a path to a folder, a file or a numpy array, by
            default None.
        train_data_target : Optional[Union[Path, str, np.ndarray]], optional
            Training target data, can be a path to a folder, a file or a numpy array, by
            default None.
        val_data_target : Optional[Union[Path, str, np.ndarray]], optional
            Validation target data, can be a path to a folder, a file or a numpy array,
            by default None.
        read_source_func : Optional[Callable], optional
            Function to read the source data, by default None. Only used for `custom`
            data type (see DataModel).
        extension_filter : str, optional
            Filter for file extensions, by default "". Only used for `custom` data types
            (see DataModel).
        val_percentage : float, optional
            Percentage of the training data to use for validation, by default 0.1. Only
            used if `val_data` is None.
        val_minimum_split : int, optional
            Minimum number of patches or files to split from the training data for
            validation, by default 5. Only used if `val_data` is None.

        Raises
        ------
        NotImplementedError
            Raised if target data is provided.
        ValueError
            If the input types are mixed (e.g. Path and np.ndarray).
        ValueError
            If the data type is `custom` and no `read_source_func` is provided.
        ValueError
            If the data type is `array` and the input is not a numpy array.
        ValueError
            If the data type is `tiff` and the input is neither a Path nor a str.
        """
        super().__init__()
        # self.save_hyperparameters(data_config.model_dump())
        if train_data_target is not None:
            raise NotImplementedError(
                "Training with target data is not yet implemented."
            )

        # check input types coherence (no mixed types)
        inputs = [train_data, val_data, train_data_target, val_data_target]
        types_set = {type(i) for i in inputs}
        if len(types_set) > 2:  # None + expected type
            raise ValueError(
                f"Inputs for `train_data`, `val_data`, `train_data_target` and "
                f"`val_data_target` must be of the same type or None. Got "
                f"{types_set}."
            )

        # check that a read source function is provided for custom types
        if data_config.data_type == SupportedData.CUSTOM and read_source_func is None:
            raise ValueError(
                f"Data type {SupportedData.CUSTOM} is not allowed without "
                f"specifying a `read_source_func`."
            )

        # and that arrays are passed, if array type specified
        elif data_config.data_type == SupportedData.ARRAY and not isinstance(
            train_data, np.ndarray
        ):
            raise ValueError(
                f"Expected array input (see configuration.data.data_type), but got "
                f"{type(train_data)} instead."
            )

        # and that Path or str are passed, if tiff file type specified
        elif data_config.data_type == SupportedData.TIFF and (
            not isinstance(train_data, Path) and not isinstance(train_data, str)
        ):
            raise ValueError(
                f"Expected Path or str input (see configuration.data.data_type), "
                f"but got {type(train_data)} instead."
            )

        # configuration
        self.data_config = data_config
        self.data_type = data_config.data_type
        self.batch_size = data_config.batch_size
        self.num_workers = num_workers
        self.use_in_memory = use_in_memory

        # data
        self.train_data = train_data
        self.val_data = val_data

        self.train_data_target = train_data_target
        self.val_data_target = val_data_target
        self.val_percentage = val_percentage
        self.val_minimum_split = val_minimum_split

        # read source function corresponding to the requested type
        if data_config.data_type == SupportedData.CUSTOM:
            self.read_source_func: Callable = read_source_func
        else:
            self.read_source_func = get_read_func(data_config.data_type)
        self.extension_filter = extension_filter

    def prepare_data(self) -> None:
        """Hook used to prepare the data before calling `setup` and creating
        the dataloader.

        Here, we only need to examine the data if it was provided as a str or a Path.
        """
        # if the data is a Path or a str
        if not isinstance(self.train_data, np.ndarray) and \
            not isinstance(self.val_data, np.ndarray) and \
                not isinstance(self.train_data_target, np.ndarray) and \
                    not isinstance(self.val_data_target, np.ndarray):
            # list training files
            self.train_files = list_files(
                self.train_data, self.data_type, self.extension_filter
            )
            self.train_files_size = get_files_size(self.train_files)

            # list validation files
            if self.val_data is not None:
                self.val_files = list_files(
                    self.val_data, self.data_type, self.extension_filter
                )

            # same for target data
            if self.train_data_target is not None:
                self.train_target_files: List[Path] = list_files(
                    self.train_data_target, self.data_type, self.extension_filter
                )

                # verify that they match the training data
                validate_source_target_files(self.train_files, self.train_target_files)

            if self.val_data_target is not None:
                self.val_target_files = list_files(
                    self.val_data_target, self.data_type, self.extension_filter
                )

                # verify that they match the validation data
                validate_source_target_files(self.val_files, self.val_target_files)

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """
        Hook called at the beginning of fit (train + validate), validate, test, or
        predict.
        """
        # if numpy array
        if self.data_type == SupportedData.ARRAY:
            # train dataset
            self.train_dataset: DatasetType = InMemoryDataset(
                data_config=self.data_config,
                data=self.train_data,
                data_target=self.train_data_target,
            )

            # validation dataset
            if self.val_data is not None:
                # create its own dataset
                self.val_dataset: DatasetType = InMemoryDataset(
                    data_config=self.data_config,
                    data=self.val_data,
                    data_target=self.val_data_target,
                )
            else:
                # extract validation from the training patches
                self.val_dataset = self.train_dataset.split_dataset(
                    percentage=self.val_percentage,
                    minimum_patches=self.val_minimum_split,
                )

        # else we read files
        else:
            # Heuristics, if the file size is smaller than 80% of the RAM,
            # we run the training in memory, otherwise we switch to iterable dataset
            # The switch is deactivated if use_in_memory is False
            if self.use_in_memory and self.train_files_size < get_ram_size() * 0.8:
                # train dataset
                self.train_dataset = InMemoryDataset(
                    data_config=self.data_config,
                    data=self.train_files,
                    data_target=self.train_target_files
                    if self.train_data_target
                    else None,
                    read_source_func=self.read_source_func,
                )

                # validation dataset
                if self.val_data is not None:
                    self.val_dataset = InMemoryDataset(
                        data_config=self.data_config,
                        data=self.val_files,
                        data_target=self.val_target_files
                        if self.val_data_target
                        else None,
                        read_source_func=self.read_source_func,
                    )
                else:
                    # split dataset
                    self.val_dataset = self.train_dataset.split_dataset(
                        percentage=self.val_percentage,
                        minimum_patches=self.val_minimum_split,
                    )

            # else if the data is too large, load file by file during training
            else:
                # create training dataset
                self.train_dataset = PathIterableDataset(
                    data_config=self.data_config,
                    src_files=self.train_files,
                    target_files=self.train_target_files
                    if self.train_data_target
                    else None,
                    read_source_func=self.read_source_func,
                )

                # create validation dataset
                if self.val_files is not None:
                    # create its own dataset
                    self.val_dataset = PathIterableDataset(
                        data_config=self.data_config,
                        src_files=self.val_files,
                        target_files=self.val_target_files
                        if self.val_data_target
                        else None,
                        read_source_func=self.read_source_func,
                    )
                elif len(self.train_files) <= self.val_minimum_split:
                    raise ValueError(
                        f"Not enough files to split a minimum of "
                        f"{self.val_minimum_split} files, got {len(self.train_files)} "
                        f"files."
                    )
                else:
                    # extract validation from the training patches
                    self.val_dataset = self.train_dataset.split_dataset(
                        percentage=self.val_percentage,
                        minimum_files=self.val_minimum_split,
                    )

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class CAREamicsClay(L.LightningDataModule):
    def __init__(
        self,
        data_config: DataModel,
        pred_data: Union[Path, str, np.ndarray],
        tile_size: Union[List[int], Tuple[int, ...]],
        tile_overlap: Union[List[int], Tuple[int, ...]],
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        # check that a read source function is provided for custom types
        if data_config.data_type == SupportedData.CUSTOM and read_source_func is None:
            raise ValueError(
                f"Data type {SupportedData.CUSTOM} is not allowed without "
                f"specifying a `read_source_func`."
            )

        # and that arrays are passed, if array type specified
        elif data_config.data_type == SupportedData.ARRAY and not isinstance(
            pred_data, np.ndarray
        ):
            raise ValueError(
                f"Expected array input (see configuration.data.data_type), but got "
                f"{type(pred_data)} instead."
            )

        # and that Path or str are passed, if tiff file type specified
        elif data_config.data_type == SupportedData.TIFF and not (
            isinstance(pred_data, Path) or isinstance(pred_data, str)
        ):
            raise ValueError(
                f"Expected Path or str input (see configuration.data.data_type), "
                f"but got {type(pred_data)} instead."
            )

        # configuration data
        self.data_config = data_config
        self.data_type = data_config.data_type
        self.batch_size = data_config.batch_size
        self.num_workers = num_workers

        self.pred_data = pred_data
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        # read source function
        if data_config.data_type == SupportedData.CUSTOM:
            self.read_source_func: Callable = read_source_func
        else:
            self.read_source_func = get_read_func(data_config.data_type)
        self.extension_filter = extension_filter

    def prepare_data(self) -> None:
        # if the data is a Path or a str
        if not isinstance(self.pred_data, np.ndarray):
            self.pred_files = list_files(
                self.pred_data, self.data_type, self.extension_filter
            )
        else:
            # reshape array
            self.pred_data = reshape_array(self.pred_data, self.data_config.axes)

    def setup(self, stage: Optional[str] = None) -> None:
        # if numpy array
        if self.data_type == SupportedData.ARRAY:
            # prediction dataset
            self.predict_dataset: PredictDatasetType = InMemoryPredictionDataset(
                data_config=self.data_config,
                data=self.pred_data,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )
        else:
            self.predict_dataset = IterablePredictionDataset(
                src_files=self.pred_files,
                data_config=self.data_config,
                read_source_func=self.read_source_func,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class CAREamicsTrainDataModule(CAREamicsWood):
    def __init__(
        self,
        train_path: Union[str, Path],
        data_type: Union[str, SupportedData],
        patch_size: List[int],
        axes: str,
        batch_size: int,
        val_path: Optional[Union[str, Path]] = None,
        transforms: Optional[Union[List, Compose]] = None,
        train_target_path: Optional[Union[str, Path]] = None,
        val_target_path: Optional[Union[str, Path]] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        val_percentage: float = 0.1,
        val_minimum_patches: int = 5,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        data_config = {
            "mode": "train",
            "data_type": data_type,
            "patch_size": patch_size,
            "axes": axes,
            "batch_size": batch_size,
            "pin_memory": pin_memory,
        }

        # if transforms are passed (otherwise it will use the default ones)
        if transforms is not None:
            data_config["transforms"] = transforms

        super().__init__(
            data_config=DataModel(**data_config),
            train_data=train_path,
            val_data=val_path,
            train_data_target=train_target_path,
            val_data_target=val_target_path,
            read_source_func=read_source_func,
            extension_filter=extension_filter,
            val_percentage=val_percentage,
            val_minimum_split=val_minimum_patches,
            num_workers=num_workers,
        )


class CAREamicsPredictDataModule(CAREamicsClay):
    def __init__(
        self,
        pred_path: Union[str, Path],
        data_type: Union[str, SupportedData],
        tile_size: List[int],
        axes: str,
        batch_size: int,
        prediction_transforms: Optional[Union[List, Compose]] = None,
        tta_transforms: Optional[bool] = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        data_config = {
            "mode": "predict",
            "data_type": data_type,
            "tile_size": tile_size,
            "axes": axes,
            "mean": mean,
            "std": std,
            "tta": tta_transforms,
            "batch_size": batch_size,
        }
        # Pred
        # TODO this needs to be reorganized
        # if transforms are passed
        if mean is not None and std is not None:
            data_config["prediction_transforms"] = [
                {
                    "name": SupportedTransform.NORMALIZE.value,
                    "parameters": {
                        "mean": mean,
                        "std": std,
                        "max_pixel_value": 1
                    },
                },
            ]
        elif prediction_transforms is not None:
            data_config["transforms"] = prediction_transforms

        else:
            logger.info(
                "No transform defined for prediction. "
                "Prediction will apply default normalization only."
            )


        super().__init__(
            data_config=PredictionModel(**data_config),
            pred_data=pred_path,
            tile_size=tile_size,
            tile_overlap=(48, 48),
            read_source_func=read_source_func,
            extension_filter=extension_filter,
            num_workers=num_workers,
        )
