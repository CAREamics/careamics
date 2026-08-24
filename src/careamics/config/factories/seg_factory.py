"""Convenience function to create UNet-based segmentation configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.seg_configuration import SegConfiguration
from careamics.utils import get_logger

from .algorithm_factory import create_algorithm_configuration
from .data_factory import (
    SupportedPatchFilterConfig,
    create_data_configuration,
)
from .factory_utils import assemble_augmentations, validate_input_channels
from .training_factory import create_training_configuration, update_trainer_params

logging = get_logger("Segmentation factory")


def _get_expected_target_axes(axes: str) -> str:
    """Return expected target axes from input axes.

    Parameters
    ----------
    axes : str
        Expected target axes.

    Returns
    -------
    str
        Expected target axes given inputs.
    """
    return "".join([ax for ax in axes if ax != "C"])


def _get_input_size(
    axes: str,
    channels: Sequence[int] | None,
    n_channels_in: int | None,
) -> int:
    """Validate channel dimensions and return model input size.

    Parameters
    ----------
    axes : str
        Axes of the data (e.g. YX).
    channels : Sequence[int] or None
        Indices of the channels to use.
    n_channels_in : int or None
        Number of input channels.

    Returns
    -------
    int
        Adjusted number of input channels.
    """
    validate_input_channels(
        axes=axes, channels=channels, n_channels=n_channels_in, attr_name="n_channels"
    )

    # resolve number of input channels
    if n_channels_in is None and channels is None:
        resolved_n_channels_in = 1
    elif n_channels_in is not None:
        resolved_n_channels_in = n_channels_in
    else:
        assert channels is not None
        resolved_n_channels_in = len(channels)

    return resolved_n_channels_in


def create_seg_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    n_classes: int,
    # optional parameters
    num_epochs: int = 30,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    n_channels_in: int | None = None,
) -> SegConfiguration:
    """
    Create a configuration for training a UNet for semantic segmentation.

    The `axes` parameters must reflect the actual axes and axis order from the data,
    and should be the same throughout all images. The accepted axes are STCZYX. If "C"
    is in `axes`, then you need to set `n_channels_in` to the number of channels
    expected in the input.

    By default, CAREamics will go through the entire training data once per epoch. For
    large datasets, this can lead to very long epochs. To limit the number of batches
    per epoch, set the `num_steps` parameter to the desired number of batches.

    If the content of your data is expected to always have the same orientation,
    consider disabling certain augmentations. By default `augmentations=None` will apply
    random flips along X and Y, and random 90 degrees rotations in the XY plane. To
    disable augmentations, set `augmentations=[]`.

    See `create_advanced_seg_config` for more parameters.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment. A valid experiment name is a non-empty string that only
        contains letters, numbers, underscores, dashes and spaces.
    data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. YX).
    patch_size : Sequence[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    n_classes : int
        Number of foreground segmentation classes.
    num_epochs : int, default=30
        Number of epochs to train for.
    num_steps : int, default=None
        Number of batches in 1 epoch.
    augmentations : Sequence of {"x_flip", "y_flip", "rotate_90"}, default=None
        List of augmentations to apply. If `None`, all augmentations are applied.
    n_val_patches : int, default=8,
        The number of patches to set aside for validation during training. This
        parameter will be ignored if separate validation data is specified for training.
    n_channels_in : int or None, default=None
        Number of input channels.

    Returns
    -------
    SegConfiguration
        Configuration for training a UNet for semantic segmentation.
    """
    return create_advanced_seg_config(**locals())


def create_advanced_seg_config(
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    n_classes: int,
    # optional parameters
    num_epochs: int = 30,
    num_steps: int | None = None,
    n_channels_in: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    # advanced parameters
    in_memory: bool | None = None,
    channels: Sequence[int] | None = None,
    normalization: Literal["mean_std", "min_max", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
    patch_filter_config: SupportedPatchFilterConfig | None = None,
    # lightning parameters
    num_workers: int = -1,
    loss: Literal["dice", "ce", "dice_ce"] = "dice",
    trainer_params: dict | None = None,
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
    early_stopping_params: dict[str, Any] | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    # reproducibility
    seed: int | None = None,
) -> SegConfiguration:
    """
    Create a configuration for training segmentation using a UNet model.

    If "Z" is present in `axes`, then `patch_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` to the number
    of input channels.

    By default, the transformations applied are a random flip along X or Y, and a random
    90 degrees rotation in the XY plane. Normalization is always applied.

    The parameters of the UNet can be specified in the `model_params` (passed as a
    parameter-value dictionary).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment. A valid experiment name is a non-empty string that only
        contains letters, numbers, underscores, dashes and spaces.
    data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    n_classes : int
        Number of foreground segmentation classes.
    num_epochs : int, default=30
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int | None, default=None
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.
    n_channels_in : int | None, default=None
        Number of input channels. If `channels` is specified, then the number of
        channels is inferred from its length and this parameter is ignored.
    augmentations : Sequence[{"x_flip", "y_flip", "rotate_90"}] | None, default=None
        List of transforms to apply, either both or one of XYFlipConfig and
        XYRandomRotate90Config. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    n_val_patches : int, default=8,
        The number of patches to set aside for validation during training. This
        parameter will be ignored if separate validation data is specified for training.
    in_memory : bool | None, default=None
        Whether to load all data into memory. This is only supported for 'array',
        'tiff' and 'custom' data types. If `None`, defaults to `True` for 'array',
        'tiff' and `custom`, and `False` for 'zarr' and 'czi' data types. Must be `True`
        for `array`.
    channels : Sequence[int] | None, default=None
        List of channels to use. If `None`, all channels are used.
    normalization : {"mean_std", "min_max", "quantile", "none"}, default="mean_std"
        Normalization strategy to use.
    normalization_params : dict[str, Any] | None, default=None
        Strategy-specific normalization parameters. If None, default values are used.
        For "mean_std": {"input_means": [...], "input_stds": [...]} (optional)
        For "min_max": {"input_mins": [...], "input_maxes": [...]} (optional)
        For "quantile": {"lower_quantiles": 0.01, "upper_quantiles": 0.99} (optional)
        For "none": No parameters needed.
    patch_filter_config : SupportedPatchFilterConfig | None, default=None
        Specify the configuration for patch filtering. Patch filtering reduces the
        probability of background patches being selected during training. If `None`,
        no patch filter is applied.
    num_workers : int, default=-1
        Number of workers for data loading. Use `-1` to automatically choose based
        on the number of available CPUs. Unless explicitly overridden in
        `train_dataloader_params` and `val_dataloader_params`, this will be applied to
        all dataloaders.
    loss : Literal["dice", "ce", "dice_ce"], default="dice"
        Loss function to use for training.
    trainer_params : dict | None, default=None
        Parameters for the trainer, see the relevant documentation.
    model_params : dict | None, default=None
        UNetModel parameters.
    optimizer : Literal["Adam", "Adamax", "SGD"], default="Adam"
        Optimizer to use.
    optimizer_params : dict[str, Any] | None, default=None
        Parameters for the optimizer, see PyTorch documentation for more details.
    lr_scheduler : Literal["ReduceLROnPlateau", "StepLR"], default="ReduceLROnPlateau"
        Learning rate scheduler to use.
    lr_scheduler_params : dict[str, Any] | None, default=None
        Parameters for the learning rate scheduler, see PyTorch documentation for more
        details.
    train_dataloader_params : dict[str, Any] | None, default=None
        Parameters for the training dataloader, see the PyTorch docs for `DataLoader`.
        If left as `None`, `{"shuffle": True}` will be used.
    val_dataloader_params : dict[str, Any] | None, default=None
        Parameters for the validation dataloader, see PyTorch the docs for `DataLoader`.
    checkpoint_params : dict[str, Any] | None, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters.
    early_stopping_params : dict[str, Any] | None, default=None
        Parameters for the early stopping callback, see PyTorch Lightning documentation
        (`EarlyStopping`) for the list of available parameters.
    logger : Literal["wandb", "tensorboard", "none"], default="none"
        Logger to use.
    seed : int | None, default=None
        Random seed for reproducibility.

    Returns
    -------
    SegConfiguration
        Configuration for training a segmentation model.
    """
    n_channels_in = _get_input_size(
        axes=axes,
        channels=channels,
        n_channels_in=n_channels_in,
    )

    # normalization
    norm_config = {"name": normalization, "skip_target": True}
    if normalization_params is not None:
        if (
            "skip_target" in normalization_params
            and not normalization_params["skip_target"]
        ):
            logging.warning(
                msg=(
                    "Parameter `skip_target` in `normalization_params` must be `True`. "
                    "Current value will be ignored."
                ),
                stacklevel=2,
            )
            del normalization_params["skip_target"]
        norm_config.update(normalization_params)

    # data
    data_config = create_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        target_axes=_get_expected_target_axes(axes),
        augmentations=assemble_augmentations(augmentations, seed),
        n_val_patches=n_val_patches,
        normalization=norm_config,
        patch_filter_config=patch_filter_config,
        channels=channels,
        in_memory=in_memory,
        num_workers=num_workers,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
        seed=seed,
    )

    # algorithm
    algorithm_params = create_algorithm_configuration(
        dimensions=3 if data_config.is_3D() else 2,
        algorithm="seg",
        loss=loss,
        independent_channels=False,
        n_channels_in=n_channels_in,
        n_channels_out=n_classes + 1,  # add background channel
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
        algorithm="seg",
        trainer_params=final_trainer_params,
        logger=logger,
        checkpoint_params=checkpoint_params,
        early_stopping_params=early_stopping_params,
        monitor_metric="val_loss",
    )

    return SegConfiguration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_params,
        data_config=data_config,
        training_config=training_params,
    )
