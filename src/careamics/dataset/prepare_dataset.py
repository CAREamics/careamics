"""
Dataset preparation module.

Methods to set up the datasets for training, validation and prediction.
"""
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import zarr

from ..config import Configuration
from ..manipulation import default_manipulate
from ..utils import check_external_array_validity, check_tiling_validity
from .extraction_strategy import ExtractionStrategy
from .in_memory_dataset import InMemoryDataset, InMemoryPredictionDataset
from .iterable_dataset import IterableDataset
from .zarr_dataset import ZarrDataset


def get_train_dataset(
    config: Configuration,
    train_path: str,
    train_target_path: Optional[str] = None,
    read_source_func: Optional[Callable] = None,
) -> Union[IterableDataset, InMemoryDataset, ZarrDataset]:
    """
    Create training dataset.

    Depending on the configuration, this methods return either a TiffDataset or an
    InMemoryDataset.

    Parameters
    ----------
    config : Configuration
        Configuration.
    train_path : Union[str, Path]
        Path to training data.

    Returns
    -------
    Union[TiffDataset, InMemoryDataset]
        Dataset.
    """
    if config.data.in_memory:
        dataset = InMemoryDataset(
            data_path=train_path,
            data_format=config.data.data_format,
            axes=config.data.axes,
            mean=config.data.mean,
            std=config.data.std,
            patch_size=config.training.patch_size,
            algorithm=config.algorithm.algorithm_type,
            patch_transform=config.algorithm.transforms,
            target_path=train_target_path,
            target_format=config.data.data_format,
            read_source_func=read_source_func,
        )
    else:
        if config.data.data_format in ["tif", "tiff"]:
            dataset = IterableDataset(
                data_path=train_path,
                data_format=config.data.data_format,
                axes=config.data.axes,
                mean=config.data.mean,
                std=config.data.std,
                patch_extraction_method=ExtractionStrategy.RANDOM,
                patch_size=config.training.patch_size,
                patch_transform=config.algorithm.transforms,
                target_path=train_target_path,
                target_format=config.data.data_format,
                read_source_func=read_source_func,
            )
        elif config.data.data_format == "zarr":
            if ".zarray" in os.listdir(train_path):
                zarr_source = zarr.open(train_path, mode="r")
            else:
                source = zarr.DirectoryStore(train_path)
                cache = zarr.LRUStoreCache(source, max_size=None)
                zarr_source = zarr.group(store=cache, overwrite=False)

            dataset = ZarrDataset(
                data_source=zarr_source,
                axes=config.data.axes,
                patch_extraction_method=ExtractionStrategy.RANDOM_ZARR,
                patch_size=config.training.patch_size,
                mean=config.data.mean,
                std=config.data.std,
                patch_transform=default_manipulate,
                patch_transform_params={
                    "mask_pixel_percentage": config.algorithm.masked_pixel_percentage,
                    "roi_size": config.algorithm.roi_size,
                },
            )
    return dataset


def get_validation_dataset(
    config: Configuration,
    val_path: str,
    val_target_path: Optional[str] = None,
    read_source_func: Optional[Callable] = None,
) -> Union[InMemoryDataset, ZarrDataset]:
    """
    Create validation dataset.

    Validation dataset is kept in memory.

    Parameters
    ----------
    config : Configuration
        Configuration.
    val_path : Union[str, Path]
        Path to validation data.

    Returns
    -------
    TiffDataset
        In memory dataset.
    """
    if config.data.data_format in ["tif", "tiff"]:
        dataset = InMemoryDataset(
            data_path=val_path,
            data_format=config.data.data_format,
            axes=config.data.axes,
            mean=config.data.mean,
            std=config.data.std,
            patch_size=config.training.patch_size,
            algorithm=config.algorithm.algorithm_type,
            patch_transform=config.algorithm.transforms,
            target_path=val_target_path,
            target_format=config.data.data_format,
            read_source_func=read_source_func,
        )
    elif config.data.data_format == "zarr":
        if ".zarray" in os.listdir(val_path):
            zarr_source = zarr.open(val_path, mode="r")
        else:
            source = zarr.DirectoryStore(val_path)
            cache = zarr.LRUStoreCache(source, max_size=None)
            zarr_source = zarr.group(store=cache, overwrite=False)

        dataset = ZarrDataset(
            data_source=zarr_source,
            axes=config.data.axes,
            patch_extraction_method=ExtractionStrategy.RANDOM_ZARR,
            patch_size=config.training.patch_size,
            num_patches=10,
            mean=config.data.mean,
            std=config.data.std,
            patch_transform=default_manipulate,
            patch_transform_params={
                "mask_pixel_percentage": config.algorithm.masked_pixel_percentage,
                "roi_size": config.algorithm.roi_size,
            },
        )

    return dataset


def get_prediction_dataset(
    config: Configuration,
    pred_source: Union[str, Path, np.ndarray],
    *,
    tile_shape: Optional[List[int]] = None,
    overlaps: Optional[List[int]] = None,
    axes: Optional[str] = None,
    read_source_func: Optional[Callable] = None,
) -> Union[IterableDataset, ZarrDataset]:
    """
    Create prediction dataset.

    To use tiling, both `tile_shape` and `overlaps` must be specified, have same
    length, be divisible by 2 and greater than 0. Finally, the overlaps must be
    smaller than the tiles.

    By default, axes are extracted from the configuration. To use images with
    different axes, set the `axes` parameter. Note that the difference between
    configuration and parameter axes must be S or T, but not any of the spatial
    dimensions (e.g. 2D vs 3D).

    Parameters
    ----------
    config : Configuration
        Configuration.
    pred_path : Union[str, Path]
        Path to prediction data.
    tile_shape : Optional[List[int]], optional
        2D or 3D shape of the tiles, by default None.
    overlaps : Optional[List[int]], optional
        2D or 3D overlaps between tiles, by default None.
    axes : Optional[str], optional
        Axes of the data, by default None.

    Returns
    -------
    TiffDataset
        Dataset.
    """
    use_tiling = False  # default value

    # Validate tiles and overlaps
    if tile_shape is not None and overlaps is not None:
        check_tiling_validity(tile_shape, overlaps)

        # Use tiling
        use_tiling = True

    # Extraction strategy
    if use_tiling:
        patch_extraction_method = ExtractionStrategy.TILED
    else:
        patch_extraction_method = None

    # Create dataset
    if isinstance(pred_source, np.ndarray):
        check_external_array_validity(pred_source, axes, use_tiling)

        dataset = InMemoryPredictionDataset(
            array=pred_source,
            axes=axes if axes is not None else config.data.axes,
            tile_size=tile_shape,
            tile_overlap=overlaps,
            mean=config.data.mean,
            std=config.data.std,
            read_source_func=read_source_func,
        )
    elif isinstance(pred_source, str) or isinstance(pred_source, Path):
        if config.data.data_format in ["tif", "tiff"]:
            dataset = IterableDataset(
                data_path=pred_source,
                data_format=config.data.data_format,
                axes=axes if axes is not None else config.data.axes,
                mean=config.data.mean,
                std=config.data.std,
                patch_extraction_method=patch_extraction_method,
                patch_size=tile_shape,
                patch_overlap=overlaps,
                patch_transform=None,
                read_source_func=read_source_func,
            )
        elif config.data.data_format == "zarr":
            if ".zarray" in os.listdir(pred_source):
                zarr_source = zarr.open(pred_source, mode="r")
            else:
                source = zarr.DirectoryStore(pred_source)
                cache = zarr.LRUStoreCache(source, max_size=None)
                zarr_source = zarr.group(store=cache, overwrite=False)

            dataset = ZarrDataset(
                data_source=zarr_source,
                axes=axes if axes is not None else config.data.axes,
                patch_extraction_method=ExtractionStrategy.RANDOM_ZARR,
                patch_size=tile_shape,
                num_patches=10,
                mean=config.data.mean,
                std=config.data.std,
                patch_transform=default_manipulate,
                patch_transform_params={
                    "mask_pixel_percentage": config.algorithm.masked_pixel_percentage,
                    "roi_size": config.algorithm.roi_size,
                },
                mode="predict",
            )

    return dataset
