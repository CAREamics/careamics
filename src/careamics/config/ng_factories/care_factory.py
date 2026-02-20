"""Convenience functions to create CARE configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.ng_configs import CAREConfiguration
from careamics.config.transformations import (
    XYFlipConfig,
    XYRandomRotate90Config,
)

from .algorithm_factory import create_algorithm_configuration
from .data_factory import create_ng_data_configuration, list_spatial_augmentations
from .training_factory import create_training_configuration, update_trainer_params


def create_care_config(
    # mandatory parameters
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    # optional parameters
    num_epochs: int = 30,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_channels_in: int | None = None,
    n_channels_out: int | None = None,
) -> CAREConfiguration:
    """
    Create a configuration for training CARE.

    The `axes` parameters must reflect the actual axes and axis order from the data,
    and should be the same throughout all images. The accepted axes are STCZYX. If "C"
    is in `axes`, then you need to set `n_channels_in` and `n_channels_out` to the
    number of channels expected in the input and output, respectively.

    By default, CAREamics will go through the entire training data once per epoch. For
    large datasets, this can lead to very long epochs. To limit the number of batches
    per epoch, set the `num_steps` parameter to the desired number of batches.

    If the content of your data is expected to always have the same orientation,
    consider disabling certain augmentations. By default `augmentations=None` will apply
    random flips along X and Y, and random 90 degrees rotations in the XY plane. To
    disable augmentations, set `augmentations=[]`.

    See `create_advanced_care_config` for more parameters.

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
    n_channels_in : int or None, default=None
        Number of input channels.
    n_channels_out : int or None, default=None
        Number of output channels.

    Returns
    -------
    CAREConfiguration
        Configuration for training CARE.
    """
    return create_advanced_care_config(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_steps=num_steps,
        augmentations=augmentations,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )


def create_advanced_care_config(
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    # optional parameters
    num_epochs: int = 30,
    num_steps: int | None = None,
    n_channels_in: int | None = None,
    n_channels_out: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    # advanced parameters
    in_memory: bool | None = None,
    channels: Sequence[int] | None = None,
    independent_channels: bool = True,
    normalization: Literal["mean_std", "minmax", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
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
) -> CAREConfiguration:
    """
    Create a configuration for training CARE.

    If "Z" is present in `axes`, then `patch_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` and
    `n_channels_out` to the number of input and output channels, respectively.

    By default, all channels are trained independently. To train all channels together,
    set `independent_channels` to False.

    By default, the augmentations applied are random flips along X or Y, and random
    90 degrees rotations in the XY plane. To disable the augmentations, simply pass an
    empty list.

    The parameters of the UNet can be specified in the `model_params` (passed as a
    parameter-value dictionary).

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
    n_channels_in : int | None, default=None
        Number of input channels. If `channels` is specified, then the number of
        channels is inferred from its length and this parameter is ignored.
    n_channels_out : int | None, default=None
        Number of output channels. If not specified, but n_channels_in is specified,
        it will default to the same number as n_channels_in.
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
    CAREConfiguration
        Configuration for training CARE.
    """
    return CAREConfiguration(
        **_create_advanced_supervised_config(
            algorithm="care",
            experiment_name=experiment_name,
            data_type=data_type,
            axes=axes,
            patch_size=patch_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_steps=num_steps,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            augmentations=augmentations,
            in_memory=in_memory,
            channels=channels,
            independent_channels=independent_channels,
            normalization=normalization,
            normalization_params=normalization_params,
            num_workers=num_workers,
            trainer_params=trainer_params,
            model_params=model_params,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            train_dataloader_params=train_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            checkpoint_params=checkpoint_params,
            logger=logger,
            seed=seed,
        )
    )


def _create_advanced_supervised_config(
    algorithm: Literal["care", "n2n"],
    # mandatory parameters
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    # optional parameters
    num_epochs: int = 30,
    num_steps: int | None = None,
    n_channels_in: int | None = None,
    n_channels_out: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    # advanced parameters
    in_memory: bool | None = None,
    channels: Sequence[int] | None = None,
    independent_channels: bool = True,
    normalization: Literal["mean_std", "minmax", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
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
) -> dict[str, Any]:
    """
    Create a configuration for training CARE or N2N.

    If "Z" is present in `axes`, then `patch_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` and
    `n_channels_out` to the number of input and output channels, respectively.

    By default, all channels are trained independently. To train all channels together,
    set `independent_channels` to False.

    By default, the augmentations applied are random flips along X or Y, and random
    90 degrees rotations in the XY plane. To disable the augmentations, simply pass an
    empty list.

    The parameters of the UNet can be specified in the `model_params` (passed as a
    parameter-value dictionary).

    Note that `num_workers` is applied to all dataloaders unless explicitly overridden
    in the respective dataloader parameters.

    Parameters
    ----------
    algorithm : Literal["care", "n2n"]
        Algorithm for which to create the configuration.
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
    n_channels_in : int | None, default=None
        Number of input channels. If `channels` is specified, then the number of
        channels is inferred from its length and this parameter is ignored.
    n_channels_out : int | None, default=None
        Number of output channels. If not specified, but n_channels_in is specified,
        it will default to the same number as n_channels_in.
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
    dict[str, Any]
        Parameters required to instantiate a CAREamics configuration.
    """
    # if there are channels, we need to specify their number
    channels_present = "C" in axes

    if channels_present and (n_channels_in is None and channels is None):
        raise ValueError(
            "`n_channels_in` or `channels` must be specified when using channels."
        )
    elif not channels_present and (n_channels_in is not None and n_channels_in > 1):
        raise ValueError(
            f"C is not present in the axes, but number of input channels is specified "
            f"(got {n_channels_in} channel)."
        )

    if not channels_present and (n_channels_out is not None and n_channels_out > 1):
        raise ValueError(
            f"C is not present in the axes, but number of output channels is specified "
            f"(got {n_channels_out} channel)."
        )

    if n_channels_in is not None and channels is not None:
        if n_channels_in != len(channels):
            raise ValueError(
                f"Number of input channels ({n_channels_in}) does not match length of "
                f"`channels` ({len(channels)}). Only specify `channels`."
            )

    if n_channels_in is None:
        n_channels_in = 1 if channels is None else len(channels)

    if n_channels_out is None:
        n_channels_out = n_channels_in

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
        algorithm=algorithm,
        loss="mae",
        independent_channels=independent_channels,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
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

    return {
        "experiment_name": experiment_name,
        "algorithm_config": algorithm_params,
        "data_config": data_config,
        "training_config": training_params,
    }
