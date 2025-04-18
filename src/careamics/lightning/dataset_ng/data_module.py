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


class CareamicsDataModule(L.LightningDataModule):
    """
    CAREamics Lightning data module for training, validation and prediction.

    The data module can be used with Path, str or numpy arrays. In the case of
    numpy arrays, it loads and computes all the patches in memory.

    The data can be either a path to a folder containing images or a single file.

    Validation can be omitted, in which case the validation data is extracted from
    the training data. The percentage of the training data to use for validation,
    as well as the minimum number of patches or files to split from the training
    data can be set using `val_percentage` and `val_minimum_split`, respectively.

    You can provide a `fnmatch` and `Path.rglob` compatible expression (e.g.
    "*.czi") to filter the files extension using `extension_filter`.


    Parameters
    ----------
    data_config : DataConfig
        Pydantic model for CAREamics data configuration.
    train_data : Path or str or numpy.ndarray, optional
        Training data, can be a path to a folder, a file or a numpy array.
    train_data_target : Path or str or numpy.ndarray, optional
        Training target data, can be a path to a folder, a file or a numpy array.
    val_data : Path or str or numpy.ndarray, optional
        Validation data, can be a path to a folder, a file or a numpy array.
    val_data_target : Path or str or numpy.ndarray, optional
        Validation target data, can be a path to a folder, a file or a numpy array.
    pred_data : Path or str or numpy.ndarray, optional
        Prediction data, can be a path to a folder, a file or a numpy array.
    pred_data_target : Path or str or numpy.ndarray, optional
        Prediction target data, can be a path to a folder, a file or a numpy array.
    read_source_func : Callable, optional
        Function to read the source data, by default None. Only used for `custom`
        data type (see DataModel).
    read_kwargs : dict, optional
        Additional keyword arguments passed to the read function.
    image_stack_loader : ImageStackLoader, optional
        Custom loader for handling image stacks.
    image_stack_loader_kwargs : dict, optional
        Additional keyword arguments passed to the image stack loader.
    extension_filter : str, optional
        Filter for file extensions, by default "". Only used for `custom` data types
        (see DataModel).
    val_percentage : float, optional
        Percentage of training data to use for validation if
        no validation data provided.
    val_minimum_split : int, optional
        Minimum number of samples to use for validation split.
    use_in_memory : bool, optional
        Whether to load all data into memory. Default is True.

    Raises
    ------
    ValueError
        If no data is provided for training, validation or prediction.
    """

    def __init__(
        self,
        data_config: DataConfig,
        train_data: Optional[Union[Path, str, NDArray]] = None,
        train_data_target: Optional[Union[Path, str, NDArray]] = None,
        val_data: Optional[Union[Path, str, NDArray]] = None,
        val_data_target: Optional[Union[Path, str, NDArray]] = None,
        pred_data: Optional[Union[Path, str, NDArray]] = None,
        pred_data_target: Optional[Union[Path, str, NDArray]] = None,
        read_source_func: Optional[Callable] = None,
        read_kwargs: Optional[dict[str, Any]] = None,
        image_stack_loader: Optional[ImageStackLoader] = None,
        image_stack_loader_kwargs: Optional[dict[str, Any]] = None,
        extension_filter: str = "",
        val_percentage: Optional[float] = None,
        val_minimum_split: int = 5,
        use_in_memory: bool = True,
    ) -> None:
        """
        Initialize a CAREamics Lightning data module.

        Parameters
        ----------
        data_config : DataConfig
            Configuration for the data module including batch size, data type, etc.
        train_data : Path or str or numpy.ndarray, optional
            Training input data
        train_data_target : Path or str or numpy.ndarray, optional
            Training target data
        val_data : Path or str or numpy.ndarray, optional
            Validation input data
        val_data_target : Path or str or numpy.ndarray, optional
            Validation target data
        pred_data : Path or str or numpy.ndarray, optional
            Prediction input data
        pred_data_target : Path or str or numpy.ndarray, optional
            Prediction target data
        read_source_func : Callable, optional
            Custom function for reading data files. Required if data_type is 'custom'
        read_kwargs : dict, optional
            Additional keyword arguments passed to read functions
        image_stack_loader : ImageStackLoader, optional
            Custom loader for handling image stacks
        image_stack_loader_kwargs : dict, optional
            Additional keyword arguments passed to image stack loader
        extension_filter : str, optional
            File extension filter pattern (e.g. "*.tif")
        val_percentage : float, optional
            Percentage of training data to use for validation if
            no validation data provided
        val_minimum_split : int, optional
            Minimum number of samples to use for validation split
        use_in_memory : bool, optional
            Whether to load all data into memory. Default is True

        Raises
        ------
        ValueError
            If no data is provided for training, validation or prediction
            If input and target data types don't match
            If data type doesn't match configuration
            If using custom data type without read_source_func
        """
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
        input_data: Union[Path, str, NDArray, None],
        target_data: Optional[Union[Path, str, NDArray]],
    ) -> None:
        """Validate if the input and target data types are consistent."""
        if input_data is not None and target_data is not None:
            if type(input_data) != type(target_data):
                raise ValueError(
                    f"Inputs for input and target must be of the same type or None. "
                    f"Got {type(input_data)} and {type(target_data)}."
                )

    def _validate_input_type_matches_config(
        self, input_data: Union[Path, str, NDArray, None]
    ) -> None:
        """Validate if the input data type matches the configuration."""
        if isinstance(input_data, np.ndarray):
            if self.data_type != SupportedData.ARRAY:
                raise ValueError(
                    f"Received a numpy array as input, but the data type was set to "
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

    def _list_files(
        self, input_data: Path, target_data: Optional[Path] = None
    ) -> tuple[list[Path], Optional[list[Path]]]:
        """List files from input and target directories."""
        input_files = list_files(input_data, self.data_type, self.extension_filter)
        if target_data is not None:
            target_files = list_files(
                target_data, self.data_type, self.extension_filter
            )
            validate_source_target_files(input_files, target_files)
            return input_files, target_files
        return input_files, None

    def _initialize_data_pair(
        self,
        input_data: Union[Path, str, NDArray, None],
        target_data: Optional[Union[Path, str, NDArray]],
    ) -> tuple[Union[Path, NDArray, str, None], Union[Path, str, NDArray, None]]:
        """
        Initialize a pair of input and target data.

        Returns
        -------
        tuple[Union[Path, NDArray, None], Union[Path, str, NDArray]]
            A tuple containing the initialized input and target data.
            For file paths, returns lists of Path objects.
            For numpy arrays, returns the arrays directly.
        """
        self._validate_input_type_matches_config(input_data)
        self._validate_input_target_type_consistency(input_data, target_data)

        if isinstance(input_data, str) or isinstance(input_data, Path):
            input_path = Path(input_data)
            target_path = Path(target_data) if target_data is not None else None
            input_data, target_data = self._list_files(input_path, target_path)

        if isinstance(input_data, np.ndarray):
            input_data = [input_data]
            if target_data is not None:
                target_data = [target_data]

        return input_data, target_data

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
        dataset_kwargs = {
            "config": self.config,
            "in_memory": self.use_in_memory,
            "read_func": self.read_source_func,
            "read_kwargs": self.read_kwargs,
            "image_stack_loader": self.image_stack_loader,
            "image_stack_loader_kwargs": self.image_stack_loader_kwargs,
        }

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
