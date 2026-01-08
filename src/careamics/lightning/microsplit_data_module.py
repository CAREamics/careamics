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
def get_train_val_data_tiff(
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
    data = load_data(datadir).astype(np.float64)
    val_index = 18
    test_index = 19 # TODO hardcoded for testing
    data_length = data.shape[0]
    if datasplit_type == DataSplitType.All:
        return data
    if datasplit_type == DataSplitType.Train:
        reserved_indices = {val_index, test_index}
        train_indices = [
            idx for idx in range(data_length) if idx not in reserved_indices
        ]
        if not train_indices:
            raise ValueError("Empty train split.")
        return data[train_indices]
    if datasplit_type == DataSplitType.Val:
        return data[[val_index]]
    if datasplit_type == DataSplitType.Test:
        return data[[test_index]]


from typing import Callable, Union

from numpy.typing import NDArray

from careamics.lvae_training.dataset import MicroSplitDataConfig
from careamics.lvae_training.dataset import DataType, DataSplitType
from careamics.lvae_training.dataset import (
    LCMultiChDloader,
    MultiChDloader,
    MultiChDloaderRef,
    MultiFileDset,
    MultiCropDset
)

SplittingDataset = Union[LCMultiChDloader, MultiChDloader, MultiFileDset, MultiCropDset]


def create_train_val_datasets(
    datapath: str,
    train_config: MicroSplitDataConfig,
    val_config: MicroSplitDataConfig,
    test_config: MicroSplitDataConfig,
    load_data_func: Callable[..., NDArray],
) -> tuple[SplittingDataset, SplittingDataset, SplittingDataset, tuple[dict, dict]]:
    if train_config.data_type in [
        DataType.TavernaSox2Golgi,
        DataType.Dao3Channel,
        DataType.Dao3ChannelWithInput,
        # DataType.ExpMicroscopyV1,
        DataType.ExpMicroscopyV2,
        DataType.TavernaSox2GolgiV2,
    ]:
        dataset_class = MultiFileDset
    elif train_config.multiscale_lowres_count > 1:
        dataset_class = LCMultiChDloader
    elif train_config.data_type in [
        DataType.HTH23BData]:
        dataset_class = MultiChDloaderRef
    else:
        dataset_class = MultiChDloader

    train_data = dataset_class(
        train_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    max_val = train_data.get_max_val()
    val_config.max_val = max_val
    if train_config.datasplit_type == DataSplitType.All:
        val_config.datasplit_type = DataSplitType.All
        test_config.datasplit_type = DataSplitType.All # TODO temporary hack
    val_data = dataset_class(
        val_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    test_config.max_val = max_val
    test_data = dataset_class(
        test_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    mean_val, std_val = train_data.compute_mean_std()
    train_data.set_mean_std(mean_val, std_val)
    val_data.set_mean_std(mean_val, std_val)
    test_data.set_mean_std(mean_val, std_val)
    data_stats = train_data.get_mean_std()

    assert isinstance(data_stats, tuple)
    assert isinstance(data_stats[0], dict)

    return train_data, val_data, test_data, data_stats


def get_target_images(test_dset: SplittingDataset) -> NDArray:
    """Get the target images."""
    if test_dset.data_type in [
        DataType.HTIba1Ki67,
    ]:
        return test_dset._data


from typing import Literal

from careamics.lvae_training.dataset import MicroSplitDataConfig, DataSplitType, DataType

CH_IDX_LIST = [1, 2, 3, 17]


class HTLIF24DataConfig(MicroSplitDataConfig):
    dset_type: Literal[
        "high", "mid", "low", "verylow", "2ms", "3ms", "5ms", "20ms", "500ms"
    ]
    # TODO: add description
    
    channel_idx_list: list
    # TODO: add description


def get_data_configs(
    dset_type: Literal["high", "mid", "low", "verylow", "2ms", "3ms", "5ms", "20ms", "500ms"],
    channel_idx_list: list = CH_IDX_LIST,
) -> tuple[HTLIF24DataConfig, HTLIF24DataConfig, HTLIF24DataConfig]:
    """Get the data configurations to use at training time.
    
    Parameters
    ----------
    dset_type : Literal["high", "mid", "low", "verylow", "2ms", "3ms", "5ms", "20ms", "500ms"]
        The dataset type to use.
    channel_idx_list : list[Literal[1, 2, 3, 17]]
        The channel indices to use.
    
    Returns
    -------
    tuple[HTLIF24DataConfig, HTLIF24DataConfig]
        The train, validation and test data configurations.
    """
    train_data_config = HTLIF24DataConfig(
        data_type=DataType.HTLIF24Data,
        dset_type=dset_type,
        datasplit_type=DataSplitType.Train,
        image_size=[64, 64],
        grid_size=32,
        channel_idx_list=channel_idx_list,
        num_channels=len(channel_idx_list),
        input_idx=len(channel_idx_list) - 1,
        target_idx_list=list(range(len(channel_idx_list) - 1)),
        multiscale_lowres_count=3,
        poisson_noise_factor=-1,
        enable_gaussian_noise=False,
        synthetic_gaussian_scale=100,
        input_has_dependant_noise=True,
        use_one_mu_std=True,
        train_aug_rotate=True,
        target_separate_normalization=True,
        input_is_sum=False,
        padding_kwargs={"mode": "reflect"},
        overlapping_padding_kwargs={"mode": "reflect"},
    )
    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    test_data_config = val_data_config.model_copy(
        update=dict(datasplit_type=DataSplitType.Test,)
    )
    return train_data_config, val_data_config, test_data_config


import os

import numpy as np

import nd2

from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples


def get_ms_based_datafiles(ms: str):
    return [f"Set{i}/uSplit_{ms}.nd2" for i in range(1, 7)]


def get_raw_files_dict():
    files_dict = {
        "high": [
            "uSplit_14022025_highSNR.nd2",
            "uSplit_20022025_highSNR.nd2",
            "uSplit_20022025_001_highSNR.nd2",
        ],
        "mid": [
            "uSplit_14022025_midSNR.nd2",
            "uSplit_20022025_midSNR.nd2",
            "uSplit_20022025_001_midSNR.nd2",
        ],
        "low": [
            "uSplit_14022025_lowSNR.nd2",
            "uSplit_20022025_lowSNR.nd2",
            "uSplit_20022025_001_lowSNR.nd2",
        ],
        "verylow": [
            "uSplit_14022025_verylowSNR.nd2",
            "uSplit_20022025_verylowSNR.nd2",
            "uSplit_20022025_001_verylowSNR.nd2",
        ],
        "2ms": get_ms_based_datafiles("2ms"),
        "3ms": get_ms_based_datafiles("3ms"),
        "5ms": get_ms_based_datafiles("5ms"),
        "20ms": get_ms_based_datafiles("20ms"),
        "500ms": get_ms_based_datafiles("500ms"),
    }
    # check that the order is correct
    keys = ["high", "mid", "low", "verylow"]
    for key1 in keys:
        filetokens1 = list(map(lambda x: x.replace(key1, ""), files_dict[key1]))
        for key2 in keys:
            filetokens2 = list(map(lambda x: x.replace(key2, ""), files_dict[key2]))
            assert np.array_equal(
                filetokens1, filetokens2
            ), f"File tokens are not equal for {key1} and {key2}"
    return files_dict


def load_one_fpath(fpath, channel_list):
    with nd2.ND2File(fpath) as nd2file:
        # axes = "".join(nd2_file.sizes.keys())  # PCYX [4 of possible 7 "TPZCYXS"]
        data = nd2file.asarray()                 # shape: (20, 19, 1608, 1608)
    # Here, 20 are different locations and 19 are different channels.
    data = data[:, channel_list, ...]
    # swap the second and fourth axis
    data = np.swapaxes(data[..., None], 1, 4)[:, 0]

    fname_prefix = "_".join(os.path.basename(fpath).split(".")[0].split("_")[:-1])
    if fname_prefix == "uSplit_20022025_001":
        data = np.delete(data, 2, axis=0)
    elif fname_prefix == "uSplit_14022025":
        data = np.delete(data, [17, 19], axis=0)

    # shape: (20, 1608, 1608, C)
    return data


def load_data(datadir, channel_list, dataset_type):
    files_dict = get_raw_files_dict()[dataset_type]
    data_list = []
    for i, fname in enumerate(files_dict):
        fpath = os.path.join(datadir, fname)
        print(f"Loading file {i + 1}/{len(files_dict)}: {fname}")
        data = load_one_fpath(fpath, channel_list)
        data_list.append(data)
        print(f"  Loaded shape: {data.shape}")
    if len(data_list) > 1:
        data = np.concatenate(data_list, axis=0)
    else:
        data = data_list[0]
    return data


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    data = load_data(
        datadir,
        channel_list=data_config.channel_idx_list,
        dataset_type=data_config.dset_type,
    )
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
    print(f"train_idx: {train_idx}")
    print(f"val_idx: {val_idx}")
    print(f"test_idx: {test_idx}")
    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float32)
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
        # Should be compatible with microSplit MicroSplitDataConfig
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
        self.train_config = data_config
        self.val_config = data_config.model_copy(deep=True)

        train_datapath = train_data
        val_datapath = val_data or train_data
        load_data_func = read_source_func

        # dataset_class = LCMultiChDloader  # TODO hardcoded for now

        # # Create datasets
        # self.train_dataset = dataset_class(
        #     self.train_config,
        #     train_datapath,
        #     load_data_fn=load_data_func,
        #     val_fraction=val_percentage,
        #     test_fraction=0.1,
        # )
        # max_val = self.train_dataset.get_max_val()
        # self.val_config.max_val = max_val
        # if self.train_config.datasplit_type == DataSplitType.All:
        #     self.val_config.datasplit_type = DataSplitType.All
        # else:
        #     self.val_config.datasplit_type = DataSplitType.Val
        # self.val_dataset = dataset_class(
        #     self.val_config,
        #     val_datapath,
        #     load_data_fn=load_data_func,
        #     val_fraction=val_percentage,
        #     test_fraction=0.1,
        # )
        # mean_val, std_val = self.train_dataset.compute_mean_std()
        # self.train_dataset.set_mean_std(mean_val, std_val)
        # self.val_dataset.set_mean_std(mean_val, std_val)
        # data_stats = self.train_dataset.get_mean_std()

        # # Store data statistics
        # self.data_stats = (
        #     data_stats[0],
        #     data_stats[1],
        # )  # TODO repeats old logic, revisit
        EXPOSURE_DURATION = "5ms"
        train_data_config, val_data_config, test_data_config = get_data_configs(
            dset_type=EXPOSURE_DURATION, channel_idx_list=[0, 1, 8],
        )
        self.train_config = train_data_config
        self.val_config = val_data_config
        # start the download of required files
        self.train_dataset, self.val_dataset, self.test_dataset, self.data_stats = create_train_val_datasets(
            datapath=Path("/home/igor.zubarev/projects/microSplit-reproducibility/examples/2D/HT_LIF24/data") / f"ht_lif24_{EXPOSURE_DURATION}.zip.unzip/{EXPOSURE_DURATION}",
            train_config=train_data_config,
            val_config=val_data_config,
            test_config=test_data_config,
            load_data_func=get_train_val_data,
            )
        self.train_dataset.set_mean_std(*self.data_stats)
        self.val_dataset.set_mean_std(*self.data_stats)

        self.train_dataset.reduce_data(list(range(10)))
        self.val_dataset.reduce_data([0])

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
            batch_size=32,#self.train_config.batch_size,
            #**self.train_config.train_dataloader_params,
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
            batch_size=32,#self.train_config.batch_size,
            #**self.val_config.val_dataloader_params,  # TODO duplicated
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
        
        EXPOSURE_DURATION = "5ms"
        _, _, test_data_config = get_data_configs(
            dset_type=EXPOSURE_DURATION, channel_idx_list=[0, 1, 8],
        )
        test_data_config.data_stats = pred_config.data_stats
        test_data_config.max_val = pred_config.max_val
        
        self.predict_dataset = LCMultiChDloader(
            test_data_config,
            Path("/home/igor.zubarev/projects/microSplit-reproducibility/examples/2D/HT_LIF24/data") / f"ht_lif24_{EXPOSURE_DURATION}.zip.unzip/{EXPOSURE_DURATION}",
            load_data_fn=self.read_source_func,
            val_fraction=0.0,
            test_fraction=0.1,
        )
        self.predict_dataset.set_mean_std(*self.pred_config.data_stats)
        # self.predict_dataset.reduce_data([0])

    # def prepare_data(self) -> None:
    #     """Hook used to prepare the data before calling `setup`."""
    #     # # TODO currently data preparation is handled in dataset creation, revisit!
    #     pass

    # def setup(self, stage: str | None = None) -> None:
    #     """
    #     Hook called at the beginning of predict.

    #     Parameters
    #     ----------
    #     stage : Optional[str], optional
    #         Stage, by default None.
    #     """
    #     # Create prediction dataset using LCMultiChDloader
    #     self.predict_dataset = LCMultiChDloader(
    #         self.pred_config,
    #         self.pred_data,
    #         load_data_fn=self.read_source_func,
    #         val_fraction=0.0,  # No validation split for prediction
    #         test_fraction=1.0,  # No test split for prediction
        # )
        # self.predict_dataset.set_mean_std(*self.pred_config.data_stats)

    def predict_dataloader(self) -> DataLoader:
        """
        Create a dataloader for prediction.

        Returns
        -------
        DataLoader
            Prediction dataloader.
        """
        params = {**self.dataloader_params}
        params["shuffle"] = False
        return DataLoader(
            self.predict_dataset,
            batch_size=self.pred_config.batch_size,
            **params,
        )


def create_datamodules(
    train_data: str,
    pred_data: Union[str, Path, NDArray],
    patch_size: tuple,
    tile_size: tuple,
    data_type: DataType,
    axes: str,
    batch_size: int,
    pred_batch_size: int | None = None,
    val_data: str | None = None,
    num_channels: int = 2,
    depth3D: int = 1,
    train_grid_size: tuple | None = None,
    pred_grid_size: int | tuple[int, int, int] | None = None,
    multiscale_count: int | None = None,
    tiling_mode: TilingMode = TilingMode.ShiftBoundary,
    read_source_func: Callable | None = None,
    extension_filter: str = "",
    val_percentage: float = 0.1,
    val_minimum_split: int = 5,
    use_in_memory: bool = True,
    transforms: list | None = None,
    train_dataloader_params: dict | None = None,
    val_dataloader_params: dict | None = None,
    predict_dataloader_params: dict | None = None,
    **dataset_kwargs,
) -> tuple[MicroSplitDataModule, MicroSplitPredictDataModule]:
    dataset_config_params = {
        "data_type": data_type,
        "image_size": patch_size,
        "num_channels": num_channels,
        "depth3D": depth3D,
        "grid_size": train_grid_size,
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
    train_config.batch_size = batch_size
    train_config.train_dataloader_params = train_dataloader_params
    train_config.val_dataloader_params = val_dataloader_params
    train_module = MicroSplitDataModule(
        data_config=train_config,
        train_data=train_data,
        val_data=val_data or train_data,
        train_data_target=None,
        val_data_target=None,
        read_source_func=get_train_val_data,
        extension_filter=extension_filter,
        val_percentage=val_percentage,
        val_minimum_split=val_minimum_split,
        use_in_memory=use_in_memory,
    )
    data_stats, max_val = train_module.get_data_stats()
    if pred_batch_size is None:
        pred_batch_size = batch_size
    prediction_config_params = {
        "data_type": data_type,
        "image_size": tile_size,
        "num_channels": num_channels,
        "depth3D": depth3D,
        "grid_size": pred_grid_size,
        "multiscale_lowres_count": multiscale_count,
        "data_stats": data_stats,
        "max_val": max_val,
        "tiling_mode": tiling_mode,
        "batch_size": pred_batch_size,
        "enable_random_cropping": False,
        "datasplit_type": DataSplitType.Test,
        **dataset_kwargs,
    }
    pred_config = MicroSplitDataConfig(**prediction_config_params)
    params = predict_dataloader_params or {}
    if "batch_size" in params:
        del params["batch_size"]
    predict_module = MicroSplitPredictDataModule(
        pred_config=pred_config,
        pred_data=pred_data,
        read_source_func=(
            read_source_func if read_source_func is not None else get_train_val_data
        ),
        extension_filter=extension_filter,
        dataloader_params=params,
    )
    return train_module, predict_module
