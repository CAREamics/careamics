"""Module containing functions to create `CAREamicsPredictData`."""

from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from careamics.config import Configuration, create_inference_configuration
from careamics.utils import check_path_exists

from ..lightning_prediction_datamodule import CAREamicsPredictData


def create_pred_datamodule(
    source: Union[CAREamicsPredictData, Path, str, NDArray],
    config: Configuration,
    batch_size: Optional[int] = None,
    tile_size: Optional[Tuple[int, ...]] = None,
    tile_overlap: Tuple[int, ...] = (48, 48),
    axes: Optional[str] = None,
    data_type: Optional[Literal["array", "tiff", "custom"]] = None,
    tta_transforms: bool = True,
    dataloader_params: Optional[Dict] = None,
    read_source_func: Optional[Callable] = None,
    extension_filter: str = "",
) -> CAREamicsPredictData:
    """
    Create a `CAREamicsPredictData` module.

    Parameters
    ----------
    source : CAREamicsPredData, pathlib.Path, str or numpy.ndarray
        Data to predict on.
    config : Configuration
        Global configuration.
    batch_size : int, default=1
        Batch size for prediction.
    tile_size : tuple of int, optional
        Size of the tiles to use for prediction.
    tile_overlap : tuple of int, default=(48, 48)
        Overlap between tiles.
    axes : str, optional
        Axes of the input data, by default None.
    data_type : {"array", "tiff", "custom"}, optional
        Type of the input data.
    tta_transforms : bool, default=True
        Whether to apply test-time augmentation.
    dataloader_params : dict, optional
        Parameters to pass to the dataloader.
    read_source_func : Callable, optional
        Function to read the source data.
    extension_filter : str, default=""
        Filter for the file extension.

    Returns
    -------
    prediction datamodule: CAREamicsPredictData
        Subclass of `pytorch_lightning.LightningDataModule` for creating predictions.

    Raises
    ------
    ValueError
        If the input is not a CAREamicsPredData instance, a path or a numpy array.
    """
    # Reuse batch size if not provided explicitly
    if batch_size is None:
        batch_size = config.data_config.batch_size

    # create predict config, reuse training config if parameters missing
    prediction_config = create_inference_configuration(
        configuration=config,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        data_type=data_type,
        axes=axes,
        tta_transforms=tta_transforms,
        batch_size=batch_size,
    )

    # remove batch from dataloader parameters (priority given to config)
    if dataloader_params is None:
        dataloader_params = {}
    if "batch_size" in dataloader_params:
        del dataloader_params["batch_size"]

    if isinstance(source, CAREamicsPredictData):
        pred_datamodule = source
    elif isinstance(source, Path) or isinstance(source, str):
        pred_datamodule = _create_from_path(
            source=source,
            pred_config=prediction_config,
            read_source_func=read_source_func,
            extension_filter=extension_filter,
            dataloader_params=dataloader_params,
        )
    elif isinstance(source, np.ndarray):
        pred_datamodule = _create_from_array(
            source=source,
            pred_config=prediction_config,
            dataloader_params=dataloader_params,
        )
    else:
        raise ValueError(
            f"Invalid input. Expected a CAREamicsPredData instance, paths or "
            f"NDArray (got {type(source)})."
        )

    return pred_datamodule


def _create_from_path(
    source: Union[Path, str],
    pred_config: Configuration,
    read_source_func: Optional[Callable] = None,
    extension_filter: str = "",
    dataloader_params: Optional[Dict] = None,
    **kwargs,
) -> CAREamicsPredictData:
    """
    Create `CAREamicsPredictData` from path.

    Parameters
    ----------
    source : Path or str
        _Data to predict on.
    pred_config : Configuration
        Prediction configuration.
    read_source_func : Callable, optional
        Function to read the source data.
    extension_filter : str, default=""
        Function to read the source data.
    dataloader_params : Optional[Dict], optional
        Parameters to pass to the dataloader.
    **kwargs
        Unused.

    Returns
    -------
    prediction datamodule: CAREamicsPredictData
        Subclass of `pytorch_lightning.LightningDataModule` for creating predictions.
    """
    source_path = check_path_exists(source)

    datamodule = CAREamicsPredictData(
        pred_config=pred_config,
        pred_data=source_path,
        read_source_func=read_source_func,
        extension_filter=extension_filter,
        dataloader_params=dataloader_params,
    )
    return datamodule


def _create_from_array(
    source: NDArray,
    pred_config: Configuration,
    dataloader_params: Optional[Dict] = None,
    **kwargs,
) -> CAREamicsPredictData:
    """
    Create `CAREamicsPredictData` from array.

    Parameters
    ----------
    source : Path or str
        _Data to predict on.
    pred_config : Configuration
        Prediction configuration.
    dataloader_params : Optional[Dict], optional
        Parameters to pass to the dataloader.
    **kwargs
        Unused. Added for compatible function signature with `_create_from_path`.

    Returns
    -------
    prediction datamodule: CAREamicsPredictData
        Subclass of `pytorch_lightning.LightningDataModule` for creating predictions.
    """
    datamodule = CAREamicsPredictData(
        pred_config=pred_config,
        pred_data=source,
        dataloader_params=dataloader_params,
    )
    return datamodule
