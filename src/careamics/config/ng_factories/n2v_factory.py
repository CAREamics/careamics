"""Convenience function to create N2V configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.ng_configs import N2VConfiguration
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedTransform,
)
from careamics.config.transformations import (
    N2VManipulateConfig,
    XYFlipConfig,
    XYRandomRotate90Config,
)

from .algorithm_factory import create_algorithm_configuration
from .data_factory import create_ng_data_configuration, list_spatial_augmentations
from .training_factory import create_training_configuration, update_trainer_params


def create_n2v_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: list[XYFlipConfig | XYRandomRotate90Config] | None = None,
    channels: Sequence[int] | None = None,
    in_memory: bool | None = None,
    independent_channels: bool = True,
    use_n2v2: bool = False,
    n_channels: int | None = None,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int = 5,
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
) -> N2VConfiguration:
    """
    Create a configuration for training Noise2Void.

    N2V uses a UNet model to denoise images in a self-supervised manner. To use its
    variants structN2V and N2V2, set the `struct_n2v_axis` and `struct_n2v_span`
    (structN2V) parameters, or set `use_n2v2` to True (N2V2).

    N2V2 modifies the UNet architecture by adding blur pool layers and removes the skip
    connections, thus removing checkboard artefacts. StructN2V is used when vertical
    or horizontal correlations are present in the noise; it applies an additional mask
    to the manipulated pixel neighbors.

    If "Z" is present in `axes`, then `patch_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels` to the number of
    channels.

    By default, all channels are trained independently. To train all channels together,
    set `independent_channels` to False.

    By default, the transformations applied are a random flip along X or Y, and a random
    90 degrees rotation in the XY plane. Normalization is always applied, as well as the
    N2V manipulation.

    By setting `augmentations` to `None`, the default transformations (flip in X and Y,
    rotations by 90 degrees in the XY plane) are applied. Rather than the default
    transforms, a list of transforms can be passed to the `augmentations` parameter. To
    disable the transforms, simply pass an empty list.

    The `roi_size` parameter specifies the size of the area around each pixel that will
    be manipulated by N2V. The `masked_pixel_percentage` parameter specifies how many
    pixels per patch will be manipulated.

    The parameters of the UNet can be specified in the `model_params` (passed as a
    parameter-value dictionary). Note that `use_n2v2` and 'n_channels' override the
    corresponding parameters passed in `model_params`.

    If you pass "horizontal" or "vertical" to `struct_n2v_axis`, then structN2V mask
    will be applied to each manipulated pixel.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
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
    channels : Sequence of int, optional
        List of channels to use. If `None`, all channels are used.
    in_memory : bool, optional
        Whether to load all data into memory. This is only supported for 'array',
        'tiff' and 'custom' data types. If `None`, defaults to `True` for 'array',
        'tiff' and `custom`, and `False` for 'zarr' and 'czi' data types. Must be `True`
        for `array`.
    independent_channels : bool, optional
        Whether to train all channels together, by default True.
    use_n2v2 : bool, optional
        Whether to use N2V2, by default False.
    n_channels : int or None, default=None
        Number of channels (in and out). If `channels` is specified, then the number of
        channels is inferred from its length.
    roi_size : int, optional
        N2V pixel manipulation area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels masked in each patch, by default 0.2.
    struct_n2v_axis : Literal["horizontal", "vertical", "none"], optional
        Axis along which to apply structN2V mask, by default "none".
    struct_n2v_span : int, optional
        Span of the structN2V mask, by default 5.
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
    N2VConfiguration
        Configuration for training N2V.
    """
    # if there are channels, we need to specify their number
    channels_present = "C" in axes

    if channels_present and (n_channels is None and channels is None):
        raise ValueError(
            "`n_channels` or `channels` must be specified when using channels."
        )
    elif not channels_present and (n_channels is not None and n_channels > 1):
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels} channel)."
        )

    if n_channels is not None and channels is not None:
        if n_channels != len(channels):
            raise ValueError(
                f"Number of channels ({n_channels}) does not match length of "
                f"`channels` ({len(channels)}). Only specify `channels`."
            )

    if n_channels is None:
        n_channels = 1 if channels is None else len(channels)

    # augmentations
    spatial_transforms = list_spatial_augmentations(augmentations)

    # data
    data_config = create_ng_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=spatial_transforms,
        channels=channels,
        in_memory=in_memory,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
    )

    # algorithm
    algorithm_params = create_algorithm_configuration(
        dimensions=3 if data_config.is_3D() else 2,
        algorithm="n2v",
        loss="n2v",
        independent_channels=independent_channels,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
        use_n2v2=use_n2v2,
        model_params=model_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        lr_scheduler_params=lr_scheduler_params,
    )

    # create the N2VManipulate transform using the supplied parameters
    n2v_transform = N2VManipulateConfig(
        name=SupportedTransform.N2V_MANIPULATE.value,
        strategy=(
            SupportedPixelManipulation.MEDIAN.value
            if use_n2v2
            else SupportedPixelManipulation.UNIFORM.value
        ),
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        struct_mask_axis=struct_n2v_axis,
        struct_mask_span=struct_n2v_span,
    )
    algorithm_params["n2v_config"] = n2v_transform

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

    return N2VConfiguration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_params,
        data_config=data_config,
        training_config=training_params,
    )
