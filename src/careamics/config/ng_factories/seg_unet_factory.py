"""Convenience function to create N2V configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.ng_configs import SegConfiguration
from careamics.config.transformations import (
    XYFlipConfig,
    XYRandomRotate90Config,
)

from .algorithm_factory import create_algorithm_configuration
from .data_factory import create_ng_data_configuration, list_spatial_augmentations
from .training_factory import create_training_configuration, update_trainer_params


def create_seg_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    n_classes: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: list[XYFlipConfig | XYRandomRotate90Config] | None = None,
    in_memory: bool | None = None,
    n_input_channels: int = 1,
    loss: Literal["ce", "dice", "dice_ce"] = "dice",
    trainer_params: dict | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
) -> SegConfiguration:
    """
    Create a configuration for training segmentation using a UNet model.

    If "Z" is present in `axes`, then `patch_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_input_channels` to the number
    of channels.

    By default, the transformations applied are a random flip along X or Y, and a random
    90 degrees rotation in the XY plane. Normalization is always applied, as well as the
    N2V manipulation.

    By setting `augmentations` to `None`, the default transformations (flip in X and Y,
    rotations by 90 degrees in the XY plane) are applied. Rather than the default
    transforms, a list of transforms can be passed to the `augmentations` parameter. To
    disable the transforms, simply pass an empty list.

    The parameters of the UNet can be specified in the `model_params` (passed as a
    parameter-value dictionary).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    n_classes : int
        Number of segmentation classes.
    num_epochs : int, default=100
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int, optional
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipConfig and
        XYRandomRotate90Config. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    in_memory : bool, optional
        Whether to load all data into memory. This is only supported for 'array',
        'tiff' and 'custom' data types. If `None`, defaults to `True` for 'array',
        'tiff' and `custom`, and `False` for 'zarr' and 'czi' data types. Must be `True`
        for `array`.
    n_input_channels : int, default=1
        Number of input channels.
    loss : {"ce", "dice", "dice_ce"}, default="dice"
        Loss function to use.
    trainer_params : dict, optional
        Parameters for the trainer, see the relevant documentation.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_params : dict, default=None
        UNetModel parameters.
    optimizer : Literal["Adam", "Adamax", "SGD"], default="Adam"
        Optimizer to use.
    optimizer_params : dict, default=None
        Parameters for the optimizer, see PyTorch documentation for more details.
    lr_scheduler : Literal["ReduceLROnPlateau", "StepLR"], default="ReduceLROnPlateau"
        Learning rate scheduler to use.
    lr_scheduler_params : dict, default=None
        Parameters for the learning rate scheduler, see PyTorch documentation for more
        details.
    train_dataloader_params : dict, optional
        Parameters for the training dataloader, see the PyTorch docs for `DataLoader`.
        If left as `None`, the dict `{"shuffle": True}` will be used, this is set in
        the `GeneralDataConfig`.
    val_dataloader_params : dict, optional
        Parameters for the validation dataloader, see PyTorch the docs for `DataLoader`.
        If left as `None`, the empty dict `{}` will be used, this is set in the
        `GeneralDataConfig`.
    checkpoint_params : dict, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters.

    Returns
    -------
    SegConfiguration
        Configuration for training a segmentation model.
    """
    # if there are channels, we need to specify their number
    channels_present = "C" in axes

    if not channels_present and n_input_channels > 1:
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_input_channels} channel)."
        )

    # augmentations
    spatial_transforms = list_spatial_augmentations(augmentations)

    # data
    data_config = create_ng_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=spatial_transforms,
        channels=None,
        in_memory=in_memory,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
    )

    # algorithm
    algorithm_params = create_algorithm_configuration(
        dimensions=3 if data_config.is_3D() else 2,
        algorithm="seg",
        loss=loss,
        independent_channels=False,
        n_channels_in=n_input_channels,
        n_channels_out=n_classes,
        use_n2v2=False,
        model_params=model_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        lr_scheduler_params=lr_scheduler_params,
    )

    # training
    final_trainer_params = update_trainer_params(
        trainer_params=trainer_params,
        num_epochs=num_epochs,
        num_steps=num_steps,
    )
    training_params = create_training_configuration(
        trainer_params=final_trainer_params,
        logger=logger,
        checkpoint_params=checkpoint_params,
    )

    return SegConfiguration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_params,
        data_config=data_config,
        training_config=training_params,
    )
