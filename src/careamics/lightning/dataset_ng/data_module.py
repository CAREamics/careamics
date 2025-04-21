from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pytorch_lightning as L
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from careamics.config.data import DataConfig
from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import list_files, validate_source_target_files
from careamics.dataset_ng.dataset import Mode
from careamics.dataset_ng.factory import create_dataset
from careamics.dataset_ng.patch_extractor import ImageStackLoader
from careamics.utils import get_logger

logger = get_logger(__name__)


InputDataType = Union[Path, str, NDArray]
SupportedDataType = Union[InputDataType, list[InputDataType]]


class CareamicsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        train_data: Optional[SupportedDataType] = None,
        train_data_target: Optional[SupportedDataType] = None,
        val_data: Optional[SupportedDataType] = None,
        val_data_target: Optional[SupportedDataType] = None,
        pred_data: Optional[SupportedDataType] = None,
        pred_data_target: Optional[SupportedDataType] = None,
        read_source_func: Optional[Callable] = None,
        read_kwargs: Optional[dict[str, Any]] = None,
        image_stack_loader: Optional[ImageStackLoader] = None,
        image_stack_loader_kwargs: Optional[dict[str, Any]] = None,
        extension_filter: str = "",
        val_percentage: Optional[float] = None,
        val_minimum_split: int = 5,
        use_in_memory: bool = True,
    ) -> None:
        super().__init__()

        if train_data is None and val_data is None and pred_data is None:
            raise ValueError(
                "At least one of train_data, val_data or pred_data must be provided."
            )

        self.config: DataConfig = data_config
        self.data_type: str = data_config.data_type
        self.batch_size: int = data_config.batch_size
        self.use_in_memory: bool = use_in_memory
        self.extension_filter: str = extension_filter
        self.read_source_func = read_source_func
        self.read_kwargs = read_kwargs
        self.image_stack_loader = image_stack_loader
        self.image_stack_loader_kwargs = image_stack_loader_kwargs

        # TODO: implement the validation split logic
        self.val_percentage = val_percentage
        self.val_minimum_split = val_minimum_split
        if self.val_percentage is not None:
            raise NotImplementedError("Validation split not implemented")

        self.train_data, self.train_data_target = self._initialize_data_pair(
            train_data, train_data_target
        )
        self.val_data, self.val_data_target = self._initialize_data_pair(
            val_data, val_data_target
        )
        self.pred_data, self.pred_data_target = self._initialize_data_pair(
            pred_data, pred_data_target
        )

    def _validate_input_target_type_consistency(
        self,
        input_data: SupportedDataType,
        target_data: Optional[SupportedDataType],
    ) -> None:
        """Validate if the input and target data types are consistent."""
        if input_data is not None and target_data is not None:
            if type(input_data) != type(target_data):
                raise ValueError(
                    f"Inputs for input and target must be of the same type or None. "
                    f"Got {type(input_data)} and {type(target_data)}."
                )
        if isinstance(input_data, list) and isinstance(target_data, list):
            if len(input_data) != len(target_data):
                raise ValueError(
                    f"Inputs and targets must have the same length. "
                    f"Got {len(input_data)} and {len(target_data)}."
                )
            if not isinstance(input_data[0], type(target_data[0])):
                raise ValueError(
                    f"Inputs and targets must have the same type. "
                    f"Got {type(input_data[0])} and {type(target_data[0])}."
                )

    def _validate_input_type_matches_config(
        self, input_data: SupportedDataType
    ) -> None:
        """Validate if the input data type matches the configuration."""
        if isinstance(input_data, np.ndarray):
            if self.data_type != SupportedData.ARRAY:
                raise ValueError(
                    f"Received a numpy array as input, but the data type was set to "
                    f"{self.data_type}. Set the data type in the configuration "
                    f"to {SupportedData.ARRAY} to train on numpy arrays."
                )
        if isinstance(input_data, list) and self.data_type == SupportedData.ARRAY:
            if not isinstance(input_data[0], np.ndarray):
                raise ValueError(
                    f"Received a list of {type(input_data[0])} as input, but the data type was set to "
                    f"{self.data_type}. Set the data type in the configuration "
                    f"to {SupportedData.ARRAY} to train on numpy arrays."
                )

        if isinstance(input_data, Path) or isinstance(input_data, str):
            if (
                self.data_type != SupportedData.TIFF
                and self.data_type != SupportedData.CUSTOM
            ):
                raise ValueError(
                    f"Received a path as input, but the data type was neither set to "
                    f"{SupportedData.TIFF} nor {SupportedData.CUSTOM}. "
                    f"Set the data type in the configuration to {SupportedData.TIFF} or"
                    f" {SupportedData.CUSTOM} to train on files"
                )

        if isinstance(input_data, list) and self.data_type == SupportedData.TIFF:
            if not isinstance(input_data[0], Path) and not isinstance(
                input_data[0], str
            ):
                raise ValueError(
                    f"Received a list of {type(input_data[0])} as input, but the data type was set to "
                    f"{self.data_type}. Set the data type in the configuration to {SupportedData.TIFF} to train on files"
                )

    def _list_files_in_directory(
        self,
        input_data: Union[Path, str],
        target_data: Optional[Union[Path, str]] = None,
    ) -> tuple[list[Path], Optional[list[Path]]]:
        """List files from input and target directories."""
        input_data = Path(input_data)
        input_files = list_files(input_data, self.data_type, self.extension_filter)
        if target_data is None:
            return input_files, None
        else:
            target_data = Path(target_data)
            target_files = list_files(
                target_data, self.data_type, self.extension_filter
            )
            validate_source_target_files(input_files, target_files)
            return input_files, target_files

    def _convert_paths_to_pathlib(
        self,
        input_data: list[Union[Path, str]],
        target_data: Optional[list[Union[Path, str]]] = None,
    ) -> tuple[list[Path], Optional[list[Path]]]:
        """Create a list of file paths from the input and target data."""
        input_files = [
            Path(item) if isinstance(item, str) else item for item in input_data
        ]
        if target_data is None:
            return input_files, None
        else:
            target_files = [
                Path(item) if isinstance(item, str) else item for item in target_data
            ]
            validate_source_target_files(input_files, target_files)
            return input_files, target_files

    def _initialize_data_pair(
        self,
        input_data: SupportedDataType,
        target_data: Optional[SupportedDataType],
    ) -> tuple[
         Optional[Union[list[NDArray], list[Path]]],
         Optional[Union[list[NDArray], list[Path]]]
    ]:
        """
        Initialize a pair of input and target data.

        Returns
        -------
        tuple[Union[list[NDArray], list[Path]], Optional[Union[list[NDArray], list[Path]]]]
            A tuple containing the initialized input and target data.
            For file paths, returns lists of Path objects.
            For numpy arrays, returns the arrays directly.
        """
        if input_data is None:
            return None, None

        self._validate_input_type_matches_config(input_data)
        self._validate_input_target_type_consistency(input_data, target_data)

        # TODO: figure out the correct type annotations for target data
        if self.data_type == SupportedData.ARRAY:
            if isinstance(input_data, np.ndarray):
                input_array = [input_data]
                target_array = [target_data] if target_data is not None else None
                return input_array, target_array
            elif isinstance(input_data, list):
                return input_data, target_data
            else:
                raise ValueError(f"Unsupported input type for {self.data_type}: {type(input_data)}")
        elif (
            self.data_type == SupportedData.TIFF
            or self.data_type == SupportedData.CUSTOM
        ):
            if isinstance(input_data, (str, Path)):
                input_list, target_list = self._list_files_in_directory(
                    input_data, target_data
                )
            elif isinstance(input_data, list):
                input_list, target_list = self._convert_paths_to_pathlib(
                    input_data, target_data
                )
            else:
                raise ValueError(f"Unsupported input type for {self.data_type}: {type(input_data)}")
            return input_list, target_list
        else:
            raise NotImplementedError(f"Unsupported data type: {self.data_type}")


    def setup(self, stage: str) -> None:
        """
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
            self.train_dataset = create_dataset(
                mode=Mode.TRAINING,
                inputs=self.train_data,
                targets=self.train_data_target,
                config=self.config,
                in_memory=self.use_in_memory,
                read_func=self.read_source_func,
                read_kwargs=self.read_kwargs,
                image_stack_loader=self.image_stack_loader,
                image_stack_loader_kwargs=self.image_stack_loader_kwargs,
            )
            self.val_dataset = create_dataset(
                mode=Mode.VALIDATING,
                inputs=self.val_data,
                targets=self.val_data_target,
                config=self.config,
                in_memory=self.use_in_memory,
                read_func=self.read_source_func,
                read_kwargs=self.read_kwargs,
                image_stack_loader=self.image_stack_loader,
                image_stack_loader_kwargs=self.image_stack_loader_kwargs,
            )
            # TODO: ugly, need to find a better solution
            self.stats = self.train_dataset.input_stats
        elif stage == "validate":
            self.val_dataset = create_dataset(
                mode=Mode.VALIDATING,
                inputs=self.val_data,
                targets=self.val_data_target,
                config=self.config,
                in_memory=self.use_in_memory,
                read_func=self.read_source_func,
                read_kwargs=self.read_kwargs,
                image_stack_loader=self.image_stack_loader,
                image_stack_loader_kwargs=self.image_stack_loader_kwargs,
            )
            self.stats = self.val_dataset.input_stats
        elif stage == "predict":
            self.predict_dataset = create_dataset(
                mode=Mode.PREDICTING,
                inputs=self.pred_data,
                targets=self.pred_data_target,
                config=self.config,
                in_memory=self.use_in_memory,
                read_func=self.read_source_func,
                read_kwargs=self.read_kwargs,
                image_stack_loader=self.image_stack_loader,
                image_stack_loader_kwargs=self.image_stack_loader_kwargs,
            )
            self.stats = self.predict_dataset.input_stats
        else:
            raise NotImplementedError(f"Stage {stage} not implemented")

    def train_dataloader(self) -> DataLoader:
        """
        Create a dataloader for training.

        Returns
        -------
        DataLoader
            Training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            **self.config.train_dataloader_params,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create a dataloader for validation.

        Returns
        -------
        DataLoader
            Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            **self.config.val_dataloader_params,
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
            # TODO: set appropriate key for params once config changes are merged
        )
