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

from careamics.config.data.ng_data_model import NGDataConfig
from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import list_files, validate_source_target_files
from careamics.dataset_ng.dataset import Mode
from careamics.dataset_ng.factory import create_dataset
from careamics.dataset_ng.grouped_index_sampler import GroupedIndexSampler
from careamics.dataset_ng.patch_extractor import ImageStackLoader
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
    use_in_memory : bool
        Load data in memory dataset if possible, by default True.


    Attributes
    ----------
    config : DataConfig
        Pydantic model for CAREamics data configuration.
    data_type : str
        Type of data, one of SupportedData.
    batch_size : int
        Batch size for the dataloaders.
    use_in_memory : bool
        Whether to load data in memory if possible.
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
        use_in_memory: bool = True,
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
        use_in_memory: bool = True,
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
        use_in_memory: bool = True,
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
        use_in_memory: bool = True,
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
        use_in_memory: bool = True,
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
        use_in_memory: bool = True,
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
        use_in_memory: bool = True,
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
        use_in_memory : bool
            Load data in memory dataset if possible, by default True.
        """
        super().__init__()

        if train_data is None and val_data is None and pred_data is None:
            raise ValueError(
                "At least one of train_data, val_data or pred_data must be provided."
            )

        self.config: NGDataConfig = data_config
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
        self.train_data_mask, _ = self._initialize_data_pair(train_data_mask, None)

        self.val_data, self.val_data_target = self._initialize_data_pair(
            val_data, val_data_target
        )

        # The pred_data_target can be needed to count metrics on the prediction
        self.pred_data, self.pred_data_target = self._initialize_data_pair(
            pred_data, pred_data_target
        )

    def _validate_input_target_type_consistency(
        self,
        input_data: InputType,
        target_data: InputType | None,
    ) -> None:
        """Validate if the input and target data types are consistent.

        Parameters
        ----------
        input_data : InputType
            Input data, can be a path to a folder, a list of paths, or a numpy array.
        target_data : Optional[InputType]
            Target data, can be None, a path to a folder, a list of paths, or a numpy
            array.
        """
        if input_data is not None and target_data is not None:
            if not isinstance(input_data, type(target_data)):
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

    def _list_files_in_directory(
        self,
        input_data,
        target_data=None,
    ) -> tuple[list[Path], list[Path] | None]:
        """List files from input and target directories.

        Parameters
        ----------
        input_data : InputType
            Input data, can be a path to a folder, a list of paths, or a numpy array.
        target_data : Optional[InputType]
            Target data, can be None, a path to a folder, a list of paths, or a numpy
            array.

        Returns
        -------
        (list[Path], Optional[list[Path]])
            A tuple containing lists of file paths for input and target data.
            If target_data is None, the second element will be None.
        """
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
        input_data,
        target_data=None,
    ) -> tuple[list[Path], list[Path] | None]:
        """Create a list of file paths from the input and target data.

        Parameters
        ----------
        input_data : InputType
            Input data, can be a path to a folder, a list of paths, or a numpy array.
        target_data : Optional[InputType]
            Target data, can be None, a path to a folder, a list of paths, or a numpy
            array.

        Returns
        -------
        (list[Path], Optional[list[Path]])
            A tuple containing lists of file paths for input and target data.
            If target_data is None, the second element will be None.
        """
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

    def _validate_array_input(
        self,
        input_data: InputType,
        target_data: InputType | None,
    ) -> tuple[Any, Any]:
        """Validate if the input data is a numpy array.

        Parameters
        ----------
        input_data : InputType
            Input data, can be a path to a folder, a list of paths, or a numpy array.
        target_data : Optional[InputType]
            Target data, can be None, a path to a folder, a list of paths, or a numpy
            array.

        Returns
        -------
        (Any, Any)
            A tuple containing the input and target.
        """
        if isinstance(input_data, np.ndarray):
            input_array = [input_data]
            target_array = [target_data] if target_data is not None else None
            return input_array, target_array
        elif isinstance(input_data, list):
            return input_data, target_data
        else:
            raise ValueError(
                f"Unsupported input type for {self.data_type}: {type(input_data)}"
            )

    def _validate_path_input(
        self, input_data: InputType, target_data: InputType | None
    ) -> tuple[list[Path], list[Path] | None]:
        """Validate if the input data is a path or a list of paths.

        Parameters
        ----------
        input_data : InputType
            Input data, can be a path to a folder, a list of paths, or a numpy array.
        target_data : Optional[InputType]
            Target data, can be None, a path to a folder, a list of paths, or a numpy
            array.

        Returns
        -------
        (list[Path], Optional[list[Path]])
            A tuple containing lists of file paths for input and target data.
            If target_data is None, the second element will be None.
        """
        if isinstance(input_data, str | Path):
            if target_data is not None:
                assert isinstance(target_data, str | Path)
            input_list, target_list = self._list_files_in_directory(
                input_data, target_data
            )
            return input_list, target_list
        elif isinstance(input_data, list):
            if target_data is not None:
                assert isinstance(target_data, list)
            input_list, target_list = self._convert_paths_to_pathlib(
                input_data, target_data
            )
            return input_list, target_list
        else:
            raise ValueError(
                f"Unsupported input type for {self.data_type}: {type(input_data)}"
            )

    def _validate_custom_input(self, input_data, target_data) -> tuple[Any, Any]:
        """Convert custom input data to a list of file paths.

        Parameters
        ----------
        input_data : InputType
            Input data, can be a path to a folder, a list of paths, or a numpy array.
        target_data : Optional[InputType]
            Target data, can be None, a path to a folder, a list of paths, or a numpy
            array.

        Returns
        -------
        (Any, Any)
            A tuple containing lists of file paths for input and target data.
            If target_data is None, the second element will be None.
        """
        if self.image_stack_loader is not None:
            return input_data, target_data
        elif isinstance(input_data, str | Path):
            if target_data is not None:
                assert isinstance(target_data, str | Path)
            input_list, target_list = self._list_files_in_directory(
                input_data, target_data
            )
            return input_list, target_list
        elif isinstance(input_data, list):
            if isinstance(input_data[0], str | Path):
                if target_data is not None:
                    assert isinstance(target_data, list)
                input_list, target_list = self._convert_paths_to_pathlib(
                    input_data, target_data
                )
                return input_list, target_list
        else:
            raise ValueError(
                f"If using {self.data_type}, pass a custom "
                f"image_stack_loader or read_source_func"
            )
        return input_data, target_data

    def _initialize_data_pair(
        self,
        input_data: InputType | None,
        target_data: InputType | None,
    ) -> tuple[Any, Any]:
        """
        Initialize a pair of input and target data.

        Parameters
        ----------
        input_data : InputType
            Input data, can be None, a path to a folder, a list of paths, or a numpy
            array.
        target_data : Optional[InputType]
            Target data, can be None, a path to a folder, a list of paths, or a numpy
            array.

        Returns
        -------
        (list of numpy.ndarray or list of pathlib.Path, None or list of numpy.ndarray or
        list of pathlib.Path)
            A tuple containing the initialized input and target data. For file paths,
            returns lists of Path objects. For numpy arrays, returns the arrays
            directly.
        """
        if input_data is None:
            return None, None

        self._validate_input_target_type_consistency(input_data, target_data)

        if self.data_type == SupportedData.ARRAY:
            if isinstance(input_data, np.ndarray):
                return self._validate_array_input(input_data, target_data)
            elif isinstance(input_data, list):
                if isinstance(input_data[0], np.ndarray):
                    return self._validate_array_input(input_data, target_data)
                else:
                    raise ValueError(
                        f"Unsupported input type for {self.data_type}: "
                        f"{type(input_data[0])}"
                    )
            else:
                raise ValueError(
                    f"Unsupported input type for {self.data_type}: {type(input_data)}"
                )
        elif self.data_type in (SupportedData.TIFF, SupportedData.CZI):
            if isinstance(input_data, str | Path):
                return self._validate_path_input(input_data, target_data)
            elif isinstance(input_data, list):
                if isinstance(input_data[0], str | Path):
                    return self._validate_path_input(input_data, target_data)
                else:
                    raise ValueError(
                        f"Unsupported input type for {self.data_type}: "
                        f"{type(input_data[0])}"
                    )
            else:
                raise ValueError(
                    f"Unsupported input type for {self.data_type}: {type(input_data)}"
                )
        elif self.data_type == SupportedData.CUSTOM:
            return self._validate_custom_input(input_data, target_data)
        else:
            raise NotImplementedError(f"Unsupported data type: {self.data_type}")

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
            self.train_dataset = create_dataset(
                mode=Mode.TRAINING,
                inputs=self.train_data,
                targets=self.train_data_target,
                masks=self.train_data_mask,
                config=self.config,
                in_memory=self.use_in_memory,
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

    def _sampler(self, dataset: Literal["train", "val", "predict"]) -> Sampler | None:
        sampler: GroupedIndexSampler | None
        rng = np.random.default_rng(self.config.seed)
        if not self.use_in_memory and self.config.data_type == SupportedData.TIFF:
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
            **self.config.test_dataloader_params,
        )
