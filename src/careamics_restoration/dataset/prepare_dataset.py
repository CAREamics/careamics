from pathlib import Path
from typing import List, Optional, Union

from careamics_restoration.config import Configuration
from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.in_memory_dataset import InMemoryDataset
from careamics_restoration.dataset.tiff_dataset import TiffDataset
from careamics_restoration.manipulation import default_manipulate
from careamics_restoration.utils import check_tiling_validity


def get_train_dataset(
    config: Configuration, train_path: str
) -> Union[TiffDataset, InMemoryDataset]:
    """Create Dataset instance from configuration.

    Parameters
    ----------
    config : Configuration
        Configuration object
    train_path : Union[str, Path]
        Pathlike object with a path to training data

    Returns
    -------
        Dataset object

    Raises
    ------
    ValueError
        No training configuration found
    """
    if config.training is None:
        raise ValueError("Training configuration is not defined.")

    if config.data.in_memory:
        dataset = InMemoryDataset(
            data_path=train_path,
            data_format=config.data.data_format,
            axes=config.data.axes,
            mean=config.data.mean,
            std=config.data.std,
            patch_extraction_method=ExtractionStrategies.SEQUENTIAL,
            patch_size=config.training.patch_size,
            patch_transform=default_manipulate,
            patch_transform_params={
                "mask_pixel_percentage": config.algorithm.masked_pixel_percentage,
                "roi_size": config.algorithm.roi_size,
            },
        )
    else:
        dataset = TiffDataset(
            data_path=train_path,
            data_format=config.data.data_format,
            axes=config.data.axes,
            mean=config.data.mean,
            std=config.data.std,
            patch_extraction_method=ExtractionStrategies.RANDOM,
            patch_size=config.training.patch_size,
            patch_transform=default_manipulate,
            patch_transform_params={
                "mask_pixel_percentage": config.algorithm.masked_pixel_percentage,
                "roi_size": config.algorithm.roi_size,
            },
        )
    return dataset


def get_validation_dataset(config: Configuration, val_path: str) -> InMemoryDataset:
    """Create Dataset instance from configuration.

    Parameters
    ----------
    config : Configuration
        Configuration object
    val_path : Union[str, Path]
        Pathlike object with a path to validation data

    Returns
    -------
    TiffDataset
        Dataset object

    Raises
    ------
    ValueError
        No validation configuration found
    """
    if config.training is None:
        raise ValueError("Training configuration is not defined.")

    data_path = val_path

    dataset = InMemoryDataset(
        data_path=data_path,
        data_format=config.data.data_format,
        axes=config.data.axes,
        mean=config.data.mean,
        std=config.data.std,
        patch_extraction_method=ExtractionStrategies.SEQUENTIAL,
        patch_size=config.training.patch_size,
        patch_transform=default_manipulate,
        patch_transform_params={
            "mask_pixel_percentage": config.algorithm.masked_pixel_percentage
        },
    )

    return dataset


def get_prediction_dataset(
    config: Configuration,
    pred_path: Union[str, Path],
    *,
    tile_shape: Optional[List[int]] = None,
    overlaps: Optional[List[int]] = None,
    axes: Optional[str] = None,
) -> TiffDataset:
    """Create Dataset instance from configuration.

    To use tiling, both `tile_shape` and `overlaps` must be specified, have same
    length, be divisible by 2 and greater than 0. Finally, the overlaps must be
    smaller than the tiles.

    Parameters
    ----------
    config : Configuration
        Configuration object
    pred_path : Union[str, Path]
        Pathlike object with a path to prediction data
    tile_shape : Optional[List[int]], optional
        2D or 3D shape of the tiles to be predicted, by default None
    overlaps : Optional[List[int]], optional
        2D or 3D overlaps between tiles, by default None
    axes : Optional[str], optional
        Axes of the data, by default None

    Returns
    -------
    TiffDataset
        Dataset object

    Raises
    ------
    ValueError

    """
    use_tiling = False  # default value

    # Validate tiles and overlaps
    if tile_shape is not None and overlaps is not None:
        check_tiling_validity(tile_shape, overlaps)

        # Use tiling
        use_tiling = True

    # Extraction strategy
    if use_tiling:
        patch_extraction_method = ExtractionStrategies.TILED
    else:
        patch_extraction_method = None

    # Create dataset
    if config.data.in_memory:
        dataset = InMemoryDataset(
            data_path=pred_path,
            data_format=config.data.data_format,
            axes=config.data.axes if axes is None else axes,  # supersede axes
            mean=config.data.mean,
            std=config.data.std,
            patch_size=tile_shape,
            patch_overlap=overlaps,
            patch_extraction_method=patch_extraction_method,
            patch_transform=None,
        )
    else:
        dataset = TiffDataset(
            data_path=pred_path,
            data_format=config.data.data_format,
            axes=config.data.axes if axes is None else axes,  # supersede axes
            mean=config.data.mean,
            std=config.data.std,
            patch_size=tile_shape,
            patch_overlap=overlaps,
            patch_extraction_method=patch_extraction_method,
            patch_transform=None,
        )

    return dataset
