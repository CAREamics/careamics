from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pytorch_lightning as L
from albumentations import Compose
from torch.utils.data import DataLoader

from careamics.config import DataModel, InferenceModel
from careamics.config.data_model import TRANSFORMS_UNION
from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import (
    get_files_size,
    get_read_func,
    list_files,
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
    """LightningDataModule for training and validation datasets.

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
    """

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
        dataloader_params: Optional[dict] = None,
    ) -> None:
        """Constructor.

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
        if dataloader_params is None:
            dataloader_params = {}
        super().__init__()

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
            # mypy check
            assert read_source_func is not None

            self.read_source_func: Callable = read_source_func
        elif data_config.data_type != SupportedData.ARRAY:
            self.read_source_func = get_read_func(data_config.data_type)

        self.extension_filter = extension_filter

        # Pytorch dataloader parameters
        self.dataloader_params = dataloader_params

    def prepare_data(self) -> None:
        """Hook used to prepare the data before calling `setup`.

        Here, we only need to examine the data if it was provided as a str or a Path.
        """
        # if the data is a Path or a str
        if (
            not isinstance(self.train_data, np.ndarray)
            and not isinstance(self.val_data, np.ndarray)
            and not isinstance(self.train_data_target, np.ndarray)
            and not isinstance(self.val_data_target, np.ndarray)
        ):
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
        """Hook called at the beginning of fit, validate, or predict."""
        # if numpy array
        if self.data_type == SupportedData.ARRAY:
            # train dataset
            self.train_dataset: DatasetType = InMemoryDataset(
                data_config=self.data_config,
                inputs=self.train_data,
                data_target=self.train_data_target,
            )

            # validation dataset
            if self.val_data is not None:
                # create its own dataset
                self.val_dataset: DatasetType = InMemoryDataset(
                    data_config=self.data_config,
                    inputs=self.val_data,
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
                    inputs=self.train_files,
                    data_target=self.train_target_files
                    if self.train_data_target
                    else None,
                    read_source_func=self.read_source_func,
                )

                # validation dataset
                if self.val_data is not None:
                    self.val_dataset = InMemoryDataset(
                        data_config=self.data_config,
                        inputs=self.val_files,
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
        """Create a dataloader for training.

        Returns
        -------
        Any
            Training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            **self.dataloader_params,
        )

    def val_dataloader(self) -> Any:
        """Create a dataloader for validation.

        Returns
        -------
        Any
            Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )


class CAREamicsClay(L.LightningDataModule):
    """LightningDataModule for prediction dataset.

    The data module can be used with Path, str or numpy arrays. The data can be either
    a folder containing images or a single file.

    To read custom data types, you can set `data_type` to `custom` in `data_config`
    and provide a function that returns a numpy array from a path as
    `read_source_func` parameter. The function will receive a Path object and
    an axies string as arguments, the axes being derived from the `data_config`.

    You can also provide a `fnmatch` and `Path.rglob` compatible expression (e.g.
    "*.czi") to filter the files extension using `extension_filter`.
    """

    def __init__(
        self,
        prediction_config: InferenceModel,
        pred_data: Union[Path, str, np.ndarray],
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        dataloader_params: Optional[dict] = None,
    ) -> None:
        """Constructor.

        The data module can be used with Path, str or numpy arrays. The data can be
        either a folder containing images or a single file.

        To read custom data types, you can set `data_type` to `custom` in `data_config`
        and provide a function that returns a numpy array from a path as
        `read_source_func` parameter. The function will receive a Path object and
        an axies string as arguments, the axes being derived from the `data_config`.

        You can also provide a `fnmatch` and `Path.rglob` compatible expression (e.g.
        "*.czi") to filter the files extension using `extension_filter`.

        Parameters
        ----------
        prediction_config : InferenceModel
            Pydantic model for CAREamics prediction configuration.
        pred_data : Union[Path, str, np.ndarray]
            Prediction data, can be a path to a folder, a file or a numpy array.
        read_source_func : Optional[Callable], optional
            Function to read custom types, by default None
        extension_filter : str, optional
            Filter to filter file extensions for custom types, by default ""
        dataloader_params : dict, optional
            Dataloader parameters, by default {}

        Raises
        ------
        ValueError
            If the data type is `custom` and no `read_source_func` is provided.
        ValueError
            If the data type is `array` and the input is not a numpy array.
        ValueError
            If the data type is `tiff` and the input is neither a Path nor a str.
        """
        if dataloader_params is None:
            dataloader_params = {}
        if dataloader_params is None:
            dataloader_params = {}
        super().__init__()

        # check that a read source function is provided for custom types
        if (
            prediction_config.data_type == SupportedData.CUSTOM
            and read_source_func is None
        ):
            raise ValueError(
                f"Data type {SupportedData.CUSTOM} is not allowed without "
                f"specifying a `read_source_func`."
            )

        # and that arrays are passed, if array type specified
        elif prediction_config.data_type == SupportedData.ARRAY and not isinstance(
            pred_data, np.ndarray
        ):
            raise ValueError(
                f"Expected array input (see configuration.data.data_type), but got "
                f"{type(pred_data)} instead."
            )

        # and that Path or str are passed, if tiff file type specified
        elif prediction_config.data_type == SupportedData.TIFF and not (
            isinstance(pred_data, Path) or isinstance(pred_data, str)
        ):
            raise ValueError(
                f"Expected Path or str input (see configuration.data.data_type), "
                f"but got {type(pred_data)} instead."
            )

        # configuration data
        self.prediction_config = prediction_config
        self.data_type = prediction_config.data_type
        self.batch_size = prediction_config.batch_size
        self.dataloader_params = dataloader_params

        self.pred_data = pred_data
        self.tile_size = prediction_config.tile_size
        self.tile_overlap = prediction_config.tile_overlap

        # read source function
        if prediction_config.data_type == SupportedData.CUSTOM:
            # mypy check
            assert read_source_func is not None

            self.read_source_func: Callable = read_source_func
        elif prediction_config.data_type != SupportedData.ARRAY:
            self.read_source_func = get_read_func(prediction_config.data_type)

        self.extension_filter = extension_filter

    def prepare_data(self) -> None:
        """Hook used to prepare the data before calling `setup`."""
        # if the data is a Path or a str
        if not isinstance(self.pred_data, np.ndarray):
            self.pred_files = list_files(
                self.pred_data, self.data_type, self.extension_filter
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Hook called at the beginning of predict.

        Parameters
        ----------
        stage : Optional[str], optional
            Stage, by default None
        """
        # if numpy array
        if self.data_type == SupportedData.ARRAY:
            # prediction dataset
            self.predict_dataset: PredictDatasetType = InMemoryPredictionDataset(
                prediction_config=self.prediction_config,
                inputs=self.pred_data,
            )
        else:
            self.predict_dataset = IterablePredictionDataset(
                prediction_config=self.prediction_config,
                src_files=self.pred_files,
                read_source_func=self.read_source_func,
            )

    def predict_dataloader(self) -> DataLoader:
        """Create a dataloader for prediction.

        Returns
        -------
        DataLoader
            Prediction dataloader.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            **self.dataloader_params,
        )


class CAREamicsTrainDataModule(CAREamicsWood):
    """LightningDataModule wrapper for training and validation datasets.

    Since the lightning datamodule has no access to the model, make sure that the
    parameters passed to the datamodule are consistent with the model's requirements and
    are coherent.

    The data module can be used with Path, str or numpy arrays. In the case of
    numpy arrays, it loads and computes all the patches in memory. For Path and str
    inputs, it calculates the total file size and estimate whether it can fit in
    memory. If it does not, it iterates through the files. This behaviour can be
    deactivated by setting `use_in_memory` to False, in which case it will
    always use the iterating dataset to train on a Path or str.

    To use array data, set `data_type` to `array` and pass a numpy array to
    `train_data`.

    In particular, N2V requires a specific transformation (N2V manipulates), which is
    not compatible with supervised training. The default transformations applied to the
    training patches are defined in `careamics.config.data_model`. To use different
    transformations, pass a list of transforms or an albumentation `Compose` as
    `transforms` parameter. See examples for more details.

    By default, CAREamics only supports types defined in
    `careamics.config.support.SupportedData`. To read custom data types, you can set
    `data_type` to `custom` and provide a function that returns a numpy array from a
    path. Additionally, pass a `fnmatch` and `Path.rglob` compatible expression (e.g.
    "*.jpeg") to filter the files extension using `extension_filter`.

    In the absence of validation data, the validation data is extracted from the
    training data. The percentage of the training data to use for validation, as well as
    the minimum number of patches to split from the training data for validation can be
    set using `val_percentage` and `val_minimum_patches`, respectively.

    In `dataloader_params`, you can pass any parameter accepted by PyTorch dataloaders,
    except for `batch_size`, which is set by the `batch_size` parameter.

    Finally, if you intend to use N2V family of algorithms, you can set `use_n2v2` to
    use N2V2, and set the `struct_n2v_axis` and `struct_n2v_span` parameters to define
    the axis and span of the structN2V mask. These parameters are without effect if
    a `train_target_data` or if `transforms` are provided.

    Examples
    --------
    Create a CAREamicsTrainDataModule with default transforms with a numpy array:
    >>> import numpy as np
    >>> from careamics import CAREamicsTrainDataModule
    >>> my_array = np.arange(100).reshape(10, 10)
    >>> data_module = CAREamicsTrainDataModule(
    ...     train_data=my_array,
    ...     data_type="array",
    ...     patch_size=(2, 2),
    ...     axes='YX',
    ...     batch_size=2,
    ... )

    For custom data types (those not supported by CAREamics), then one can pass a read
    function and a filter for the files extension:
    >>> import numpy as np
    >>> from careamics import CAREamicsTrainDataModule
    >>> def read_npy(path):
    ...     return np.load(path)
    >>> data_module = CAREamicsTrainDataModule(
    ...     train_data="path/to/data",
    ...     data_type="custom",
    ...     patch_size=(2, 2),
    ...     axes='YX',
    ...     batch_size=2,
    ...     read_source_func=read_npy,
    ...     extension_filter="*.npy",
    ... )

    If you want to use a different set of transformations, you can pass a list of
    transforms:
    >>> import numpy as np
    >>> from careamics import CAREamicsTrainDataModule
    >>> from careamics.config.support import SupportedTransform
    >>> my_array = np.arange(100).reshape(10, 10)
    >>> my_transforms = [
    ...     {
    ...         "name": SupportedTransform.NORMALIZE.value,
    ...         "parameters": {"mean": 0, "std": 1},
    ...     },
    ...     {
    ...         "name": "PixelDropout",
    ...         "parameters": {"dropout_prob": 0.05, "per_channel": True},
    ...     },
    ...     {
    ...         "name": SupportedTransform.N2V_MANIPULATE.value,
    ...     }
    ... ]
    >>> data_module = CAREamicsTrainDataModule(
    ...     train_data=my_array,
    ...     data_type="array",
    ...     patch_size=(2, 2),
    ...     axes='YX',
    ...     batch_size=2,
    ...     transforms=my_transforms,
    ... )
    """

    def __init__(
        self,
        train_data: Union[str, Path, np.ndarray],
        data_type: Union[Literal["array", "tiff", "custom"], SupportedData],
        patch_size: List[int],
        axes: str,
        batch_size: int,
        val_data: Optional[Union[str, Path]] = None,
        transforms: Optional[Union[List[TRANSFORMS_UNION], Compose]] = None,
        train_target_data: Optional[Union[str, Path]] = None,
        val_target_data: Optional[Union[str, Path]] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        val_percentage: float = 0.1,
        val_minimum_patches: int = 5,
        dataloader_params: Optional[dict] = None,
        use_in_memory: bool = True,
        use_n2v2: bool = False,
        struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
        struct_n2v_span: int = 5,
        **kwargs: Any,
    ) -> None:
        """LightningDataModule wrapper for training and validation datasets.

        Since the lightning datamodule has no access to the model, make sure that the
        parameters passed to the datamodule are consistent with the model's requirements
        and are coherent.

        The data module can be used with Path, str or numpy arrays. In the case of
        numpy arrays, it loads and computes all the patches in memory. For Path and str
        inputs, it calculates the total file size and estimate whether it can fit in
        memory. If it does not, it iterates through the files. This behaviour can be
        deactivated by setting `use_in_memory` to False, in which case it will
        always use the iterating dataset to train on a Path or str.

        To use array data, set `data_type` to `array` and pass a numpy array to
        `train_data`.

        In particular, N2V requires a specific transformation (N2V manipulates), which
        is not compatible with supervised training. The default transformations applied
        to the training patches are defined in `careamics.config.data_model`. To use
        different transformations, pass a list of transforms or an albumentation
        `Compose` as `transforms` parameter. See examples for more details.

        By default, CAREamics only supports types defined in
        `careamics.config.support.SupportedData`. To read custom data types, you can set
        `data_type` to `custom` and provide a function that returns a numpy array from a
        path. Additionally, pass a `fnmatch` and `Path.rglob` compatible expression
        (e.g. "*.jpeg") to filter the files extension using `extension_filter`.

        In the absence of validation data, the validation data is extracted from the
        training data. The percentage of the training data to use for validation, as
        well as the minimum number of patches to split from the training data for
        validation can be set using `val_percentage` and `val_minimum_patches`,
        respectively.

        In `dataloader_params`, you can pass any parameter accepted by PyTorch
        dataloaders, except for `batch_size`, which is set by the `batch_size`
        parameter.

        Finally, if you intend to use N2V family of algorithms, you can set `use_n2v2`
        to use N2V2, and set the `struct_n2v_axis` and `struct_n2v_span` parameters to
        define the axis and span of the structN2V mask. These parameters are without
        effect if a `train_target_data` or if `transforms` are provided.

        Parameters
        ----------
        train_data : Union[str, Path, np.ndarray]
            Training data.
        data_type : Union[str, SupportedData]
            Data type, see `SupportedData` for available options.
        patch_size : List[int]
            Patch size, 2D or 3D patch size.
        axes : str
            Axes of the data, choosen amongst SCZYX.
        batch_size : int
            Batch size.
        val_data : Optional[Union[str, Path]], optional
            Validation data, by default None.
        transforms : Optional[Union[List[TRANSFORMS_UNION], Compose]], optional
            List of transforms to apply to training patches. If None, default transforms
            are applied.
        train_target_data : Optional[Union[str, Path]], optional
            Training target data, by default None.
        val_target_data : Optional[Union[str, Path]], optional
            Validation target data, by default None.
        read_source_func : Optional[Callable], optional
            Function to read the source data, used if `data_type` is `custom`, by
            default None.
        extension_filter : str, optional
            Filter for file extensions, used if `data_type` is `custom`, by default "".
        val_percentage : float, optional
            Percentage of the training data to use for validation if no validation data
            is given, by default 0.1.
        val_minimum_patches : int, optional
            Minimum number of patches to split from the training data for validation if
            no validation data is given, by default 5.
        dataloader_params : dict, optional
            Pytorch dataloader parameters, by default {}.
        use_n2v2 : bool, optional
            Use N2V2 transformation during training, by default False.
        struct_n2v_axis : Literal["horizontal", "vertical", "none"], optional
            Axis for the structN2V mask, only applied if `struct_n2v_axis` is `none`, by
            default "none".
        struct_n2v_span : int, optional
        Span for the structN2V mask, by default 5.

        Raises
        ------
        ValueError
            If a target is set and N2V manipulation is present in the transforms.
        """
        if dataloader_params is None:
            dataloader_params = {}
        data_dict: Dict[str, Any] = {
            "mode": "train",
            "data_type": data_type,
            "patch_size": patch_size,
            "axes": axes,
            "batch_size": batch_size,
        }

        # if transforms are passed (otherwise it will use the default ones)
        if transforms is not None:
            data_dict["transforms"] = transforms

        # validate configuration
        self.data_config = DataModel(**data_dict)

        # N2V specific checks, N2V, structN2V, and transforms
        if (
            self.data_config.has_transform_list()
            and self.data_config.has_n2v_manipulate()
        ):
            # there is not target, n2v2 and structN2V can be changed
            if train_target_data is None:
                self.data_config.set_N2V2(use_n2v2)
                self.data_config.set_structN2V_mask(struct_n2v_axis, struct_n2v_span)
            else:
                raise ValueError(
                    "Cannot have both supervised training (target data) and "
                    "N2V manipulation in the transforms. Pass a list of transforms "
                    "that is compatible with your supervised training."
                )

        # sanity check on the dataloader parameters
        if "batch_size" in dataloader_params:
            # remove it
            del dataloader_params["batch_size"]

        super().__init__(
            data_config=self.data_config,
            train_data=train_data,
            val_data=val_data,
            train_data_target=train_target_data,
            val_data_target=val_target_data,
            read_source_func=read_source_func,
            extension_filter=extension_filter,
            val_percentage=val_percentage,
            val_minimum_split=val_minimum_patches,
            dataloader_params=dataloader_params,
            use_in_memory=use_in_memory,
        )


class CAREamicsPredictDataModule(CAREamicsClay):
    """LightningDataModule wrapper of an inference dataset.

    Since the lightning datamodule has no access to the model, make sure that the
    parameters passed to the datamodule are consistent with the model's requirements
    and are coherent.

    The data module can be used with Path, str or numpy arrays. To use array data, set
    `data_type` to `array` and pass a numpy array to `train_data`.

    The default transformations applied to the images are defined in
    `careamics.config.inference_model`. To use different transformations, pass a list
    of transforms or an albumentation `Compose` as `transforms` parameter. See examples
    for more details.

    The `mean` and `std` parameters are only used if Normalization is defined either
    in the default transformations or in the `transforms` parameter, but not with
    a `Compose` object. If you pass a `Normalization` transform in a list as
    `transforms`, then the mean and std parameters will be overwritten by those passed
    to this method.

    By default, CAREamics only supports types defined in
    `careamics.config.support.SupportedData`. To read custom data types, you can set
    `data_type` to `custom` and provide a function that returns a numpy array from a
    path. Additionally, pass a `fnmatch` and `Path.rglob` compatible expression
    (e.g. "*.jpeg") to filter the files extension using `extension_filter`.

    In `dataloader_params`, you can pass any parameter accepted by PyTorch
    dataloaders, except for `batch_size`, which is set by the `batch_size`
    parameter.
    """

    def __init__(
        self,
        pred_data: Union[str, Path, np.ndarray],
        data_type: Union[Literal["array", "tiff", "custom"], SupportedData],
        tile_size: List[int],
        tile_overlap: List[int] = (48, 48), #TODO replace with calculator
        axes: str = "YX",
        batch_size: int = 1,
        tta_transforms: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        transforms: Optional[Union[List, Compose]] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        dataloader_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        pred_path : Union[str, Path, np.ndarray]
            Prediction data.
        data_type : Union[Literal["array", "tiff", "custom"], SupportedData]
            Data type, see `SupportedData` for available options.
        tile_size : List[int]
            Tile size, 2D or 3D tile size.
        tile_overlap : List[int]
            Tile overlap, 2D or 3D tile overlap.
        axes : str
            Axes of the data, choosen amongst SCZYX.
        batch_size : int
            Batch size.
        tta_transforms : bool, optional
            Use test time augmentation, by default True.
        mean : Optional[float], optional
            Mean value for normalization, only used if Normalization is defined, by
            default None.
        std : Optional[float], optional
            Standard deviation value for normalization, only used if Normalization is
            defined, by default None.
        transforms : Optional[Union[List[TRANSFORMS_UNION], Compose]], optional
            List of transforms to apply to prediction patches. If None, default
            transforms are applied.
        read_source_func : Optional[Callable], optional
            Function to read the source data, used if `data_type` is `custom`, by
            default None.
        extension_filter : str, optional
            Filter for file extensions, used if `data_type` is `custom`, by default "".
        dataloader_params : dict, optional
            Pytorch dataloader parameters, by default {}.
        """
        if dataloader_params is None:
            dataloader_params = {}
        prediction_dict = {
            "data_type": data_type,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "axes": axes,
            "mean": mean,
            "std": std,
            "tta": tta_transforms,
            "batch_size": batch_size,
        }

        # if transforms are passed (otherwise it will use the default ones)
        if transforms is not None:
            prediction_dict["transforms"] = transforms

        # validate configuration
        self.prediction_config = InferenceModel(**prediction_dict)

        # sanity check on the dataloader parameters
        if "batch_size" in dataloader_params:
            # remove it
            del dataloader_params["batch_size"]

        super().__init__(
            prediction_config=self.prediction_config,
            pred_data=pred_data,
            read_source_func=read_source_func,
            extension_filter=extension_filter,
            dataloader_params=dataloader_params,
        )
