"""Convenience functions to create NG data configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.data import NGDataConfig
from careamics.config.transformations import (
    SPATIAL_TRANSFORMS_UNION,
    XYFlipConfig,
    XYRandomRotate90Config,
)


def list_spatial_augmentations(
    augmentations: list[SPATIAL_TRANSFORMS_UNION] | None = None,
) -> list[SPATIAL_TRANSFORMS_UNION]:
    """
    List the augmentations to apply.

    Parameters
    ----------
    augmentations : list of transforms, optional
        List of transforms to apply, either both or one of XYFlipConfig and
        XYRandomRotate90Config.

    Returns
    -------
    list of transforms
        List of transforms to apply.

    Raises
    ------
    ValueError
        If the transforms are not XYFlipConfig or XYRandomRotate90Config.
    ValueError
        If there are duplicate transforms.
    """
    if augmentations is None:
        transform_list: list[SPATIAL_TRANSFORMS_UNION] = [
            XYFlipConfig(),
            XYRandomRotate90Config(),
        ]
    else:
        # throw error if not all transforms are pydantic models
        if not all(
            isinstance(t, XYFlipConfig) or isinstance(t, XYRandomRotate90Config)
            for t in augmentations
        ):
            raise ValueError(
                "Accepted transforms are either XYFlipConfig or "
                "XYRandomRotate90Config."
            )

        # check that there is no duplication
        aug_types = [t.__class__ for t in augmentations]
        if len(set(aug_types)) != len(aug_types):
            raise ValueError("Duplicate transforms are not allowed.")

        transform_list = augmentations

    return transform_list


def create_ng_data_configuration(
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    augmentations: list[SPATIAL_TRANSFORMS_UNION] | None = None,
    channels: Sequence[int] | None = None,
    in_memory: bool | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    pred_dataloader_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> NGDataConfig:
    """
    Create a training NGDatasetConfig.

    Parameters
    ----------
    data_type : {"array", "tiff", "zarr", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data.
    patch_size : list of int
        Size of the patches along the spatial dimensions.
    batch_size : int
        Batch size.
    augmentations : list of transforms
        List of transforms to apply.
    channels : Sequence of int, default=None
        List of channels to use. If `None`, all channels are used.
    in_memory : bool, default=None
        Whether to load all data into memory. This is only supported for 'array',
        'tiff' and 'custom' data types. If `None`, defaults to `True` for 'array',
        'tiff' and `custom`, and `False` for 'zarr' and 'czi' data types. Must be `True`
        for `array`.
    augmentations : list of transforms or None, default=None
        List of transforms to apply. If `None`, default augmentations are applied
        (flip in X and Y, rotations by 90 degrees in the XY plane).
    train_dataloader_params : dict
        Parameters for the training dataloader, see PyTorch notes, by default None.
    val_dataloader_params : dict
        Parameters for the validation dataloader, see PyTorch notes, by default None.
    pred_dataloader_params : dict
        Parameters for the test dataloader, see PyTorch notes, by default None.
    seed : int, default=None
        Random seed for reproducibility. If `None`, no seed is set.

    Returns
    -------
    NGDataConfig
        Next-Generation Data model with the specified parameters.
    """
    if augmentations is None:
        augmentations = list_spatial_augmentations()

    # data model
    data: dict[str, Any] = {
        "mode": "training",
        "data_type": data_type,
        "axes": axes,
        "batch_size": batch_size,
        "channels": channels,
        "transforms": augmentations,
        "seed": seed,
    }

    if in_memory is not None:
        data["in_memory"] = in_memory

    # don't override defaults set in DataConfig class
    if train_dataloader_params is not None:
        # the presence of `shuffle` key in the dataloader parameters is enforced
        # by the NGDataConfig class
        if "shuffle" not in train_dataloader_params:
            train_dataloader_params["shuffle"] = True

        data["train_dataloader_params"] = train_dataloader_params

    if val_dataloader_params is not None:
        data["val_dataloader_params"] = val_dataloader_params

    if pred_dataloader_params is not None:
        data["pred_dataloader_params"] = pred_dataloader_params

    # add training patching
    data["patching"] = {
        "name": "random",
        "patch_size": patch_size,
    }

    return NGDataConfig(**data)
