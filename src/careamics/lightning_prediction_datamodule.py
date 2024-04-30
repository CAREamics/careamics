from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as L
from albumentations import Compose
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from careamics.config import InferenceModel
from careamics.config.support import SupportedData
from careamics.config.tile_information import TileInformation
from careamics.dataset.dataset_utils import (
    get_read_func,
    list_files,
)
from careamics.dataset.in_memory_dataset import (
    InMemoryPredictionDataset,
)
from careamics.dataset.iterable_dataset import (
    IterablePredictionDataset,
)
from careamics.utils import get_logger

PredictDatasetType = Union[InMemoryPredictionDataset, IterablePredictionDataset]

logger = get_logger(__name__)


def _collate_tiles(batch: List[Tuple[np.ndarray, TileInformation]]) -> Any:
    """
    Collate tiles received from CAREamics prediction dataloader.

    CAREamics prediction dataloader returns tuples of arrays and TileInformation. In
    case of non-tiled data, this function will return the arrays. In case of tiled data,
    it will return the arrays, the last tile flag, the overlap crop coordinates and the
    stitch coordinates.

    Parameters
    ----------
    batch : Tuple[Tuple[np.ndarray, TileInformation], ...]
        Batch of tiles.

    Returns
    -------
    Any
        Collated batch.
    """
    first_tile_info: TileInformation = batch[0][1]
    # if not tiled, then return arrays
    if not first_tile_info.tiled:
        arrays, _ = zip(*batch)

        return default_collate(arrays)
    # else we explicit the last_tile flag and coordinates
    else:
        new_batch = [
            (tile, t.last_tile, t.array_shape, t.overlap_crop_coords, t.stitch_coords)
            for tile, t in batch
        ]

        return default_collate(new_batch)


class CAREamicsPredictData(L.LightningDataModule):
    """
    CAREamics Lightning prediction data module.

    The data module can be used with Path, str or numpy arrays. The data can be either
    a folder containing images or a single file.

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
        Function to read custom types, by default None.
    extension_filter : str, optional
        Filter to filter file extensions for custom types, by default "".
    dataloader_params : dict, optional
        Dataloader parameters, by default {}.
    """

    def __init__(
        self,
        prediction_config: InferenceModel,
        pred_data: Union[Path, str, np.ndarray],
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        dataloader_params: Optional[dict] = None,
    ) -> None:
        """
        Constructor.

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
            Function to read custom types, by default None.
        extension_filter : str, optional
            Filter to filter file extensions for custom types, by default "".
        dataloader_params : dict, optional
            Dataloader parameters, by default {}.

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
        """
        Hook called at the beginning of predict.

        Parameters
        ----------
        stage : Optional[str], optional
            Stage, by default None.
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
            collate_fn=_collate_tiles,
            **self.dataloader_params,
        )  # TODO check workers are used


class PredictDataWrapper(CAREamicsPredictData):
    """
    Wrapper around the CAREamics inference Lightning data module.

    This class is used to explicitely pass the parameters usually contained in a
    `inference_model` configuration.

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

    Parameters
    ----------
    pred_data : Union[str, Path, np.ndarray]
        Prediction data.
    data_type : Union[Literal["array", "tiff", "custom"], SupportedData]
        Data type, see `SupportedData` for available options.
    mean : float
        Mean value for normalization, only used if Normalization is defined in the
        transforms.
    std : float
        Standard deviation value for normalization, only used if Normalization is
        defined in the transform.
    tile_size : Tuple[int, ...]
        Tile size, 2D or 3D tile size.
    tile_overlap : Tuple[int, ...]
        Tile overlap, 2D or 3D tile overlap.
    axes : str
        Axes of the data, choosen amongst SCZYX.
    batch_size : int
        Batch size.
    tta_transforms : bool, optional
        Use test time augmentation, by default True.
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

    def __init__(
        self,
        pred_data: Union[str, Path, np.ndarray],
        data_type: Union[Literal["array", "tiff", "custom"], SupportedData],
        mean: float,
        std: float,
        tile_size: Optional[Tuple[int, ...]] = None,
        tile_overlap: Optional[Tuple[int, ...]] = None,
        axes: str = "YX",
        batch_size: int = 1,
        tta_transforms: bool = True,
        transforms: Optional[Union[List, Compose]] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        dataloader_params: Optional[dict] = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        pred_data : Union[str, Path, np.ndarray]
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
