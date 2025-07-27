"""MicroSplit data module for training and validation."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import tifffile
import torch
from torch.utils.data import DataLoader

from careamics.lvae_training.dataset import (
    DataSplitType,
    DataType,
    LCMultiChDloader,
    MicroSplitDataConfig,
)
from careamics.lvae_training.dataset.types import TilingMode
from careamics.dataset.dataset_utils.dataset_utils import reshape_array


# TODO refactor
def load_one_file(fpath):
    """Load a single 2D image file."""
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
        # a singleton channel dimension. Specify in the notebook or change with `torch.stack`??
        channels_data.append(channel_stack)

    final_data = np.stack(channels_data, axis=-1)
    return final_data


# TODO refactor
def get_datasplit_tuples(val_fraction, test_fraction, data_length):
    """Get train/val/test indices for data splitting."""
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
    """Load and split data according to configuration."""
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
        data = data[test_idx].astype(np.float64)
    else:
        raise Exception("invalid datasplit")

    return data


class MicroSplitDataModule(L.LightningDataModule):
    """Lightning DataModule for MicroSplit-style datasets.

    Matches the interface of TrainDataModule, but internally uses original MicroSplit
    dataset logic.
    """

    def __init__(
        self,
        data_config: MicroSplitDataConfig,  # Should be compatible with microSplit DatasetConfig
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
        super().__init__()
        # Dataset selection logic (adapted from create_train_val_datasets)
        train_config = data_config if hasattr(data_config, "data_type") else None
        val_config = data_config if hasattr(data_config, "data_type") else None
        test_config = data_config if hasattr(data_config, "data_type") else None
        datapath = train_data
        load_data_func = read_source_func

        dataset_class = LCMultiChDloader  # TODO hardcoded for now

        # Create datasets
        self.train_dataset = dataset_class(
            train_config,
            datapath,
            load_data_fn=load_data_func,
            val_fraction=val_percentage,
            test_fraction=0.1,
        )
        max_val = self.train_dataset.get_max_val()
        val_config.max_val = max_val
        if train_config.datasplit_type == DataSplitType.All:
            val_config.datasplit_type = DataSplitType.All
            test_config.datasplit_type = DataSplitType.All
        self.val_dataset = dataset_class(
            val_config,
            datapath,
            load_data_fn=load_data_func,
            val_fraction=val_percentage,
            test_fraction=0.1,
        )
        test_config.max_val = max_val
        self.test_dataset = dataset_class(
            test_config,
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
        self.data_mean = {
            "input": torch.tensor(data_stats[0]["input"]),
            "target": torch.tensor(data_stats[0]["target"]),
        }
        self.data_std = {
            "input": torch.tensor(data_stats[1]["input"]),
            "target": torch.tensor(data_stats[1]["target"]),
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def get_data_stats(self):
        """Get data statistics.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing two dictionaries:
            - data_mean: mean values for input and target
            - data_std: standard deviation values for input and target
        """
        return self.data_mean, self.data_std


def create_microsplit_train_datamodule(
    train_data: str,
    patch_size: tuple,
    data_type: DataType,
    axes: str,
    batch_size: int,
    val_data: str = None,
    num_channels: int = 2,
    depth3D: int = 1,
    grid_size: tuple = None,
    multiscale_count: int = None,
    tiling_mode: TilingMode = TilingMode.ShiftBoundary,
    read_source_func: Callable = None,
    extension_filter: str = "",
    val_percentage: float = 0.1,
    val_minimum_split: int = 5,
    use_in_memory: bool = True,
    transforms: list = None,
    train_dataloader_params: dict = None,
    val_dataloader_params: dict = None,
    **dataset_kwargs,
) -> MicroSplitDataModule:
    """
    Create a MicroSplitDataModule for microSplit-style datasets, including config creation.

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
    val_config = MicroSplitDataConfig(
        **dataset_config_params,
        datasplit_type=DataSplitType.Val,
    )
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
