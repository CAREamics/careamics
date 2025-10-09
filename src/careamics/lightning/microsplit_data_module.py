"""MicroSplit data module for training and validation."""

from collections.abc import Callable
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as L
import tifffile
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from careamics.dataset.dataset_utils.dataset_utils import reshape_array
from careamics.lvae_training.dataset import (
    DataSplitType,
    DataType,
    LCMultiChDloader,
    MicroSplitDataConfig,
)
from careamics.lvae_training.dataset.types import TilingMode


# TODO refactor
def load_one_file(fpath):
    """Load a single 2D image file.

    Parameters
    ----------
    fpath : str or Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Reshaped image data.
    """
    data = tifffile.imread(fpath)
    if len(data.shape) == 2:
        axes = "YX"
    elif len(data.shape) == 3:
        axes = "SYX"
    elif len(data.shape) == 4:
        axes = "STYX"
    else:
        raise ValueError(f"Invalid data shape: {data.shape}")
    data = reshape_array(data, axes)
    data = data.reshape(-1, data.shape[-2], data.shape[-1])
    return data


# TODO refactor
def load_data(datadir):
    """Load data from a directory containing channel subdirectories with image files.

    Parameters
    ----------
    datadir : str or Path
        Path to the data directory containing channel subdirectories.

    Returns
    -------
    numpy.ndarray
        Stacked array of all channels' data.
    """
    data_path = Path(datadir)

    channel_dirs = sorted(p for p in data_path.iterdir() if p.is_dir())
    channels_data = []

    for channel_dir in channel_dirs:
        image_files = sorted(f for f in channel_dir.iterdir() if f.is_file())
        channel_images = [load_one_file(image_path) for image_path in image_files]

        channel_stack = np.concatenate(
            channel_images, axis=0
        )  # FIXME: this line works iff images have
        # a singleton channel dimension. Specify in the notebook or change with
        # `torch.stack`??
        channels_data.append(channel_stack)

    final_data = np.stack(channels_data, axis=-1)
    return final_data


# TODO refactor
def get_datasplit_tuples(val_fraction, test_fraction, data_length):
    """Get train/val/test indices for data splitting.

    Parameters
    ----------
    val_fraction : float or None
        Fraction of data to use for validation.
    test_fraction : float or None
        Fraction of data to use for testing.
    data_length : int
        Total length of the dataset.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Training, validation, and test indices.
    """
    indices = np.arange(data_length)
    np.random.shuffle(indices)

    if val_fraction is None:
        val_fraction = 0.0
    if test_fraction is None:
        test_fraction = 0.0

    val_size = int(data_length * val_fraction)
    test_size = int(data_length * test_fraction)
    train_size = data_length - val_size - test_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    return train_idx, val_idx, test_idx


# TODO refactor
def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    allow_generation=None,
    **kwargs,
):
    """Load and split data according to configuration.

    Parameters
    ----------
    data_config : MicroSplitDataConfig
        Data configuration object.
    datadir : str or Path
        Path to the data directory.
    datasplit_type : DataSplitType
        Type of data split to return.
    val_fraction : float, optional
        Fraction of data to use for validation.
    test_fraction : float, optional
        Fraction of data to use for testing.
    allow_generation : bool, optional
        Whether to allow data generation.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    numpy.ndarray
        Split data array.
    """
    data = load_data(datadir)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )

    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float64)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float64)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float64)
    elif datasplit_type == DataSplitType.Test:
        # TODO this is only used for prediction, and only because old dataset uses it
        data = data[test_idx].astype(np.float64)
    else:
        raise Exception("invalid datasplit")

    return data


class MicroSplitDataModule(L.LightningDataModule):
    """Lightning DataModule for MicroSplit-style datasets.

    Matches the interface of TrainDataModule, but internally uses original MicroSplit
    dataset logic.

    Parameters
    ----------
    data_config : MicroSplitDataConfig
        Configuration for the MicroSplit dataset.
    train_data : str
        Path to training data directory.
    val_data : str, optional
        Path to validation data directory.
    train_data_target : str, optional
        Path to training target data.
    val_data_target : str, optional
        Path to validation target data.
    read_source_func : Callable, optional
        Function to read source data.
    extension_filter : str, optional
        File extension filter.
    val_percentage : float, optional
        Percentage of data to use for validation, by default 0.1.
    val_minimum_split : int, optional
        Minimum number of samples for validation split, by default 5.
    use_in_memory : bool, optional
        Whether to use in-memory dataset, by default True.
    """

    def __init__(
        self,
        # Should be compatible with microSplit DatasetConfig
        data_config: MicroSplitDataConfig,
        train_data: str,
        val_data: str | None = None,
        train_data_target: str | None = None,
        val_data_target: str | None = None,
        read_source_func: Callable | None = None,
        extension_filter: str = "",
        val_percentage: float = 0.1,
        val_minimum_split: int = 5,
        use_in_memory: bool = True,
    ):
        """Initialize MicroSplitDataModule.

        Parameters
        ----------
        data_config : MicroSplitDataConfig
            Configuration for the MicroSplit dataset.
        train_data : str
            Path to training data directory.
        val_data : str, optional
            Path to validation data directory.
        train_data_target : str, optional
            Path to training target data.
        val_data_target : str, optional
            Path to validation target data.
        read_source_func : Callable, optional
            Function to read source data.
        extension_filter : str, optional
            File extension filter.
        val_percentage : float, optional
            Percentage of data to use for validation, by default 0.1.
        val_minimum_split : int, optional
            Minimum number of samples for validation split, by default 5.
        use_in_memory : bool, optional
            Whether to use in-memory dataset, by default True.
        """
        super().__init__()
        # Dataset selection logic (adapted from create_train_val_datasets)
        self.train_config = data_config  # SHould configs be separated?
        self.val_config = data_config
        self.test_config = data_config

        datapath = train_data
        load_data_func = read_source_func

        dataset_class = LCMultiChDloader  # TODO hardcoded for now

        # Create datasets
        self.train_dataset = dataset_class(
            self.train_config,
            datapath,
            load_data_fn=load_data_func,
            val_fraction=val_percentage,
            test_fraction=0.1,
        )
        max_val = self.train_dataset.get_max_val()
        self.val_config.max_val = max_val
        if self.train_config.datasplit_type == DataSplitType.All:
            self.val_config.datasplit_type = DataSplitType.All
            self.test_config.datasplit_type = DataSplitType.All
        self.val_dataset = dataset_class(
            self.val_config,
            datapath,
            load_data_fn=load_data_func,
            val_fraction=val_percentage,
            test_fraction=0.1,
        )
        self.test_config.max_val = max_val
        self.test_dataset = dataset_class(
            self.test_config,
            datapath,
            load_data_fn=load_data_func,
            val_fraction=val_percentage,
            test_fraction=0.1,
        )
        mean_val, std_val = self.train_dataset.compute_mean_std()
        self.train_dataset.set_mean_std(mean_val, std_val)
        self.val_dataset.set_mean_std(mean_val, std_val)
        self.test_dataset.set_mean_std(mean_val, std_val)
        data_stats = self.train_dataset.get_mean_std()

        # Store data statistics
        self.data_stats = (
            data_stats[0],
            data_stats[1],
        )  # TODO repeats old logic, revisit

    def train_dataloader(self):
        """Create a dataloader for training.

        Returns
        -------
        DataLoader
            Training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            # TODO should be inside dataloader params?
            batch_size=self.train_config.batch_size,
            **self.train_config.train_dataloader_params,
        )

    def val_dataloader(self):
        """Create a dataloader for validation.

        Returns
        -------
        DataLoader
            Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_config.batch_size,
            **self.val_config.val_dataloader_params,  # TODO duplicated
        )

    def get_data_stats(self):
        """Get data statistics.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing two dictionaries:
            - data_mean: mean values for input and target
            - data_std: standard deviation values for input and target
        """
        return self.data_stats, self.val_config.max_val  # TODO should be in the config?


def create_microsplit_train_datamodule(
    train_data: str,
    patch_size: tuple,
    data_type: DataType,
    axes: str,  # TODO should be there after refactoring
    batch_size: int,
    val_data: str | None = None,
    num_channels: int = 2,
    depth3D: int = 1,
    grid_size: tuple | None = None,
    multiscale_count: int | None = None,
    tiling_mode: TilingMode = TilingMode.ShiftBoundary,
    read_source_func: Callable | None = None,  # TODO should be there after refactoring
    extension_filter: str = "",
    val_percentage: float = 0.1,
    val_minimum_split: int = 5,
    use_in_memory: bool = True,
    transforms: list | None = None,  # TODO should it be here?
    train_dataloader_params: dict | None = None,
    val_dataloader_params: dict | None = None,
    **dataset_kwargs,
) -> MicroSplitDataModule:
    """
    Create a MicroSplitDataModule for microSplit-style datasets.

    This includes config creation.

    Parameters
    ----------
    train_data : str
        Path to training data.
    patch_size : tuple
        Size of one patch of data.
    data_type : DataType
        Type of the dataset (must be a DataType enum value).
    axes : str
        Axes of the data (e.g., 'SYX').
    batch_size : int
        Batch size for dataloaders.
    val_data : str, optional
        Path to validation data.
    num_channels : int, default=2
        Number of channels in the input.
    depth3D : int, default=1
        Number of slices in 3D.
    grid_size : tuple, optional
        Grid size for patch extraction.
    multiscale_count : int, optional
        Number of LC scales.
    tiling_mode : TilingMode, default=ShiftBoundary
        Tiling mode for patch extraction.
    read_source_func : Callable, optional
        Function to read the source data.
    extension_filter : str, optional
        File extension filter.
    val_percentage : float, default=0.1
        Percentage of training data to use for validation.
    val_minimum_split : int, default=5
        Minimum number of patches/files for validation split.
    use_in_memory : bool, default=True
        Use in-memory dataset if possible.
    transforms : list, optional
        List of transforms to apply.
    train_dataloader_params : dict, optional
        Parameters for training dataloader.
    val_dataloader_params : dict, optional
        Parameters for validation dataloader.
    **dataset_kwargs :
        Additional arguments passed to DatasetConfig.

    Returns
    -------
    MicroSplitDataModule
        Configured MicroSplitDataModule instance.
    """
    # Create dataset configs with only valid parameters
    dataset_config_params = {
        "data_type": data_type,
        "image_size": patch_size,
        "num_channels": num_channels,
        "depth3D": depth3D,
        "grid_size": grid_size,
        "multiscale_lowres_count": multiscale_count,
        "tiling_mode": tiling_mode,
        "batch_size": batch_size,
        "train_dataloader_params": train_dataloader_params,
        "val_dataloader_params": val_dataloader_params,
        **dataset_kwargs,
    }

    train_config = MicroSplitDataConfig(
        **dataset_config_params,
        datasplit_type=DataSplitType.Train,
    )
    # val_config = MicroSplitDataConfig(
    #     **dataset_config_params,
    #     datasplit_type=DataSplitType.Val,
    # )
    # TODO, data config is duplicated here and in configuration

    return MicroSplitDataModule(
        data_config=train_config,
        train_data=train_data,
        val_data=val_data or train_data,
        train_data_target=None,
        val_data_target=None,
        read_source_func=get_train_val_data,  # Use our wrapped function
        extension_filter=extension_filter,
        val_percentage=val_percentage,
        val_minimum_split=val_minimum_split,
        use_in_memory=use_in_memory,
    )


class MicroSplitPredictDataModule(L.LightningDataModule):
    """Lightning DataModule for MicroSplit-style prediction datasets.

    Matches the interface of PredictDataModule, but internally uses MicroSplit
    dataset logic for prediction.

    Parameters
    ----------
    pred_config : MicroSplitDataConfig
        Configuration for MicroSplit prediction.
    pred_data : str or Path or numpy.ndarray
        Prediction data, can be a path to a folder, a file or a numpy array.
    read_source_func : Callable, optional
        Function to read custom types.
    extension_filter : str, optional
        Filter to filter file extensions for custom types.
    dataloader_params : dict, optional
        Dataloader parameters.
    """

    def __init__(
        self,
        pred_config: MicroSplitDataConfig,
        pred_data: Union[str, Path, NDArray],
        read_source_func: Callable | None = None,
        extension_filter: str = "",
        dataloader_params: dict | None = None,
    ) -> None:
        """
        Constructor for MicroSplit prediction data module.

        Parameters
        ----------
        pred_config : MicroSplitDataConfig
            Configuration for MicroSplit prediction.
        pred_data : str or Path or numpy.ndarray
            Prediction data, can be a path to a folder, a file or a numpy array.
        read_source_func : Callable, optional
            Function to read custom types, by default None.
        extension_filter : str, optional
            Filter to filter file extensions for custom types, by default "".
        dataloader_params : dict, optional
            Dataloader parameters, by default {}.
        """
        super().__init__()

        if dataloader_params is None:
            dataloader_params = {}
        self.pred_config = pred_config
        self.pred_data = pred_data
        self.read_source_func = read_source_func or get_train_val_data
        self.extension_filter = extension_filter
        self.dataloader_params = dataloader_params

    def prepare_data(self) -> None:
        """Hook used to prepare the data before calling `setup`."""
        # # TODO currently data preparation is handled in dataset creation, revisit!
        pass

    def setup(self, stage: str | None = None) -> None:
        """
        Hook called at the beginning of predict.

        Parameters
        ----------
        stage : Optional[str], optional
            Stage, by default None.
        """
        # Create prediction dataset using LCMultiChDloader
        self.predict_dataset = LCMultiChDloader(
            self.pred_config,
            self.pred_data,
            load_data_fn=self.read_source_func,
            val_fraction=0.0,  # No validation split for prediction
            test_fraction=1.0,  # No test split for prediction
        )
        self.predict_dataset.set_mean_std(*self.pred_config.data_stats)

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
            batch_size=self.pred_config.batch_size,
            **self.dataloader_params,
        )


def create_microsplit_predict_datamodule(
    pred_data: Union[str, Path, NDArray],
    tile_size: tuple,
    data_type: DataType,
    axes: str,
    batch_size: int = 1,
    num_channels: int = 2,
    depth3D: int = 1,
    grid_size: int | None = None,
    multiscale_count: int | None = None,
    data_stats: tuple | None = None,
    tiling_mode: TilingMode = TilingMode.ShiftBoundary,
    read_source_func: Callable | None = None,
    extension_filter: str = "",
    dataloader_params: dict | None = None,
    **dataset_kwargs,
) -> MicroSplitPredictDataModule:
    """
    Create a MicroSplitPredictDataModule for microSplit-style prediction datasets.

    Parameters
    ----------
    pred_data : str or Path or numpy.ndarray
        Prediction data, can be a path to a folder, a file or a numpy array.
    tile_size : tuple
        Size of one tile of data.
    data_type : DataType
        Type of the dataset (must be a DataType enum value).
    axes : str
        Axes of the data (e.g., 'SYX').
    batch_size : int, default=1
        Batch size for prediction dataloader.
    num_channels : int, default=2
        Number of channels in the input.
    depth3D : int, default=1
        Number of slices in 3D.
    grid_size : tuple, optional
        Grid size for patch extraction.
    multiscale_count : int, optional
        Number of LC scales.
    data_stats : tuple, optional
        Data statistics, by default None.
    tiling_mode : TilingMode, default=ShiftBoundary
        Tiling mode for patch extraction.
    read_source_func : Callable, optional
        Function to read the source data.
    extension_filter : str, optional
        File extension filter.
    dataloader_params : dict, optional
        Parameters for prediction dataloader.
    **dataset_kwargs :
        Additional arguments passed to MicroSplitDataConfig.

    Returns
    -------
    MicroSplitPredictDataModule
        Configured MicroSplitPredictDataModule instance.
    """
    if dataloader_params is None:
        dataloader_params = {}

    # Create prediction config with only valid parameters
    prediction_config_params = {
        "data_type": data_type,
        "image_size": tile_size,
        "num_channels": num_channels,
        "depth3D": depth3D,
        "grid_size": grid_size,
        "multiscale_lowres_count": multiscale_count,
        "data_stats": data_stats,
        "tiling_mode": tiling_mode,
        "batch_size": batch_size,
        "datasplit_type": DataSplitType.Test,  # For prediction, use all data
        **dataset_kwargs,
    }

    pred_config = MicroSplitDataConfig(**prediction_config_params)

    # Remove batch_size from dataloader_params if present
    if "batch_size" in dataloader_params:
        del dataloader_params["batch_size"]

    return MicroSplitPredictDataModule(
        pred_config=pred_config,
        pred_data=pred_data,
        read_source_func=(
            read_source_func if read_source_func is not None else get_train_val_data
        ),
        extension_filter=extension_filter,
        dataloader_params=dataloader_params,
    )
