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


def create_n2v_config(
    # mandatory parameters
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    # optional parameters
    num_epochs: int = 30,  # not too high, in case data is very large
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    use_n2v2: bool = False,
    n_channels: int | None = None,
) -> N2VConfiguration:
    """
    Create a configuration for training N2V.

    To activate N2V2, set `use_n2v2` to True.

    The `axes` parameters must reflect the actual axes and axis order from the data,
    and should be the same throughout all images. The accepted axes are STCZYX. If "C"
    is in `axes`, then you need to set `n_channels` to the number of channels.

    By default, CAREamics will go through the entire training data once per epoch. For
    large datasets, this can lead to very long epochs. To limit the number of batches
    per epoch, set the `num_steps` parameter to the desired number of batches.

    If the content of your data is expected to always have the same orientation,
    consider disabling certain augmentations. By default `augmentations=None` will apply
    random flips along X and Y, and random 90 degrees rotations in the XY plane. To
    disable augmentations, set `augmentations=[]`.

    See `create_advanced_n2v_config` for more parameters.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int, default=30
        Number of epochs to train for.
    num_steps : int, default=None
        Number of batches in 1 epoch.
    augmentations : Sequence of {"x_flip", "y_flip", "rotate_90"}, default=None
        List of augmentations to apply. If `None`, all augmentations are applied.
    use_n2v2 : bool, default=False
        Whether to use N2V2.
    n_channels : int or None, default=None
        Number of channels (in and out).

    Returns
    -------
    N2VConfiguration
        Configuration for training N2V.
    """
    return create_advanced_n2v_config(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_steps=num_steps,
        augmentations=augmentations,
        use_n2v2=use_n2v2,
        n_channels=n_channels,
    )


def create_structn2v_config(
    # mandatory parameters
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    # struct n2v
    struct_n2v_axis: Literal["horizontal", "vertical"],
    struct_n2v_span: int = 5,
    # optional parameters
    num_epochs: int = 30,
    num_steps: int | None = None,
    # TODO no rotation until we support 2D masks for structN2V
    use_n2v2: bool = False,
    n_channels: int | None = None,
) -> N2VConfiguration:
    """
    Create a configuration for training structN2V.

    The structN2V mask is applied a horizontal or vertical axis, with extent defined by
    `struct_n2v_span` (default=5, leading to a mask of size 11). For structN2V,
    augmentations are disabled.

    To activate N2V2, set `use_n2v2` to True.

    The `axes` parameters must reflect the actual axes and axis order from the data,
    and should be the same throughout all images. The accepted axes are STCZYX. If "C"
    is in `axes`, then you need to set `n_channels` to the number of channels.

    `patch_size` is only along the spatial dimensions and should be of length 3 if "Z"
    is present in `axes`, otherwise of length 2.

    By default, CAREamics will go through the entire training data once per epoch. For
    large datasets, this can lead to very long epochs. To limit the number of batches
    per epoch, set the `num_steps` parameter to the desired number of batches.

    See `create_advanced_n2v_config` for more parameters.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    struct_n2v_axis : Literal["horizontal", "vertical"]
        Axis along which to apply structN2V mask.
    struct_n2v_span : int, default=5
        Span of the structN2V mask.
    num_epochs : int, default=30
        Number of epochs to train for.
    num_steps : int, default=None
        Number of batches in 1 epoch.
    use_n2v2 : bool, default=False
        Whether to use N2V2.
    n_channels : int or None, default=None
        Number of channels (in and out).

    Returns
    -------
    N2VConfiguration
        Configuration for training structN2V.
    """
    return create_advanced_n2v_config(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_steps=num_steps,
        augmentations=[],
        use_n2v2=use_n2v2,
        n_channels=n_channels,
        struct_n2v_axis=struct_n2v_axis,
        struct_n2v_span=struct_n2v_span,
    )


# TODO reorganize docstring parameters
def create_advanced_n2v_config(
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    # optional parameters
    num_epochs: int = 30,
    num_steps: int | None = None,
    n_channels: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    # advanced parameters
    in_memory: bool | None = None,
    channels: Sequence[int] | None = None,
    independent_channels: bool = True,
    normalization: Literal["mean_std", "minmax", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
    # - N2V specific
    use_n2v2: bool = False,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    # - structN2V specific
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int = 5,
    # - Lightning parameters
    num_workers: int = 0,
    trainer_params: dict | None = None,
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    # - reproducibility
    seed: int | None = None,
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

    By default, the augmentations applied are random flips along X or Y, and random
    90 degrees rotations in the XY plane. To disable the augmentations, simply pass an
    empty list.

    The `roi_size` parameter specifies the size of the area around each pixel that will
    be manipulated by N2V. The `masked_pixel_percentage` parameter specifies how many
    pixels per patch will be manipulated.

    If you pass "horizontal" or "vertical" to `struct_n2v_axis`, then structN2V mask
    will be applied to each manipulated pixel.

    The parameters of the UNet can be specified in the `model_params` (passed as a
    parameter-value dictionary). Note that `use_n2v2` and 'n_channels' override the
    corresponding parameters passed in `model_params`.

    Note that `num_workers` is applied to all dataloaders unless explicitly overridden
    in the respective dataloader parameters.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int, default=30
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int | None, default=None
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.
    n_channels : int | None, default=None
        Number of channels (in and out). If `channels` is specified, then the number of
        channels is inferred from its length and this parameter is ignored.
    augmentations : Sequence[{"x_flip", "y_flip", "rotate_90"}] | None, default=None
        List of transforms to apply, either both or one of XYFlipConfig and
        XYRandomRotate90Config. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    in_memory : bool | None, default=None
        Whether to load all data into memory. This is only supported for 'array',
        'tiff' and 'custom' data types. If `None`, defaults to `True` for 'array',
        'tiff' and `custom`, and `False` for 'zarr' and 'czi' data types. Must be `True`
        for `array`.
    channels : Sequence[int] | None, default=None
        List of channels to use. If `None`, all channels are used.
    independent_channels : bool, default=True
        Whether to train all channels independently.
    normalization : {"mean_std", "minmax", "quantile", "none"}, default="mean_std"
        Normalization strategy to use.
    normalization_params : dict[str, Any] | None, default=None
        Strategy-specific normalization parameters. If None, default values are used.
        For "mean_std": {"input_means": [...], "input_stds": [...]} (optional)
        For "minmax": {"input_mins": [...], "input_maxes": [...]} (optional)
        For "quantile": {"lower_quantile": 0.01, "upper_quantile": 0.99} (optional)
        For "none": No parameters needed.
    use_n2v2 : bool, default=False
        Whether to use N2V2.
    roi_size : int, default=11
        N2V pixel manipulation area.
    masked_pixel_percentage : float, default=0.2
        Percentage of pixels masked in each patch.
    struct_n2v_axis : Literal["horizontal", "vertical", "none"], default="none"
        Axis along which to apply structN2V mask.
    struct_n2v_span : int, default=5
        Span of the structN2V mask.
    num_workers : int, default=0
        Number of workers for data loading. Unless explicitly overridden in
        `train_dataloader_params` and `val_dataloader_params`, this will be applied to
        all dataloaders.
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
    logger : Literal["wandb", "tensorboard", "none"], default="none"
        Logger to use.
    seed : int | None, default=None
        Random seed for reproducibility.

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

    # normalization
    norm_config = {"name": normalization}
    if normalization_params is not None:
        norm_config.update(normalization_params)

    # augmentations
    augs: list[XYFlipConfig | XYRandomRotate90Config] | None = None
    if augmentations is not None:
        augs = []

        x_flip_present = "x_flip" in augmentations
        y_flip_present = "y_flip" in augmentations
        rotate_90_present = "rotate_90" in augmentations

        if x_flip_present or y_flip_present:
            augs.append(
                XYFlipConfig(
                    flip_x=x_flip_present,
                    flip_y=y_flip_present,
                    seed=seed,
                )
            )
        if rotate_90_present:
            augs.append(XYRandomRotate90Config(seed=seed))
    spatial_transforms = list_spatial_augmentations(augs)

    # data
    data_config = create_ng_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=spatial_transforms,
        normalization=norm_config,
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
    n2v_transform = N2VManipulateConfig(  # TODO should be seeded
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


# Backward compatibility alias (deprecated)
def create_n2v_configuration(*args, **kwargs):
    """
    Deprecated: Use `create_advanced_n2v_config` instead.

    This function is provided for backward compatibility and will be removed
    in a future version.
    """
    import warnings

    warnings.warn(
        "create_n2v_configuration is deprecated, use create_advanced_n2v_config "
        "instead. Note: the 'augmentations' parameter now accepts a list of strings "
        "(e.g., ['x_flip', 'y_flip', 'rotate_90']) instead of config objects.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_advanced_n2v_config(*args, **kwargs)
