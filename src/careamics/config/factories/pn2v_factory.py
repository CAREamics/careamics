"""Convenience functions to create PN2V configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.algorithms import PN2VAlgorithm
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.config.pn2v_configuration import PN2VConfiguration
from careamics.config.utils.random import generate_random_seed

from .algorithm_factory import create_algorithm_configuration
from .data_factory import (
    SupportedPatchFilterConfig,
    create_data_configuration,
)
from .factory_utils import assemble_augmentations
from .n2v_factory import _create_n2v_transform, _validate_n2v_channel_dim
from .training_factory import create_training_configuration, update_trainer_params


def create_pn2v_config(
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    noise_model: GaussianMixtureNMConfig,
    num_epochs: int = 30,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    use_n2v2: bool = False,
    n_channels: int | None = None,
    num_classes: int | None = None,
) -> PN2VConfiguration:
    """
    Create a configuration for training PN2V.

    `num_classes` is the number of prior samples predicted per pixel. It is independent
    of `n_channels`. The Convallaria GMM protocol uses `num_classes=800`.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "zarr", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence of int
        Size of the patches along the spatial dimensions.
    batch_size : int
        Batch size.
    noise_model : GaussianMixtureNMConfig
        Trained Gaussian mixture noise model.
    num_epochs : int, default=30
        Number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in one epoch (`limit_train_batches`).
    augmentations : sequence of {"x_flip", "y_flip", "rotate_90"} or None, default=None
        Augmentations to apply. `None` applies the N2V defaults.
    n_val_patches : int, default=8
        Number of patches held out for validation.
    use_n2v2 : bool, default=False
        Whether to use the N2V2 manipulation strategy and architecture changes.
    n_channels : int or None, default=None
        Number of input channels.
    num_classes : int or None, default=None
        Number of prior samples. Defaults to the number of input channels.

    Returns
    -------
    PN2VConfiguration
        Configuration for training PN2V.
    """
    return create_advanced_pn2v_config(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        noise_model=noise_model,
        num_epochs=num_epochs,
        num_steps=num_steps,
        augmentations=augmentations,
        n_val_patches=n_val_patches,
        use_n2v2=use_n2v2,
        n_channels=n_channels,
        num_classes=num_classes,
    )


def create_advanced_pn2v_config(
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    noise_model: GaussianMixtureNMConfig,
    num_epochs: int = 30,
    num_steps: int | None = None,
    n_channels: int | None = None,
    num_classes: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    in_memory: bool | None = None,
    channels: Sequence[int] | None = None,
    independent_channels: bool = True,
    normalization: Literal["mean_std", "min_max", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
    patch_filter_config: SupportedPatchFilterConfig | None = None,
    use_n2v2: bool = False,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    struct_n2v_axes: Literal[
        "horizontal", "vertical", "cross", "square", "none"
    ] = "none",
    struct_n2v_span: int = 5,
    sample_scale: float = 10.0,
    num_workers: int = -1,
    trainer_params: dict | None = None,
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    monitor_metric: Literal["train_loss", "train_loss_epoch", "val_loss"] = "val_loss",
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    seed: int | None = None,
) -> PN2VConfiguration:
    """
    Create a PN2V configuration.

    UNet depth and initial filter count are set through `model_params`
    (`depth`, `num_channels_init`). `num_classes` is the prior-sample head and is not
    required to match the number of input channels.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "zarr", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence of int
        Size of the patches along the spatial dimensions.
    batch_size : int
        Batch size.
    noise_model : GaussianMixtureNMConfig
        Trained Gaussian mixture noise model.
    num_epochs : int, default=30
        Number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in one epoch (`limit_train_batches`).
    n_channels : int or None, default=None
        Number of input channels.
    num_classes : int or None, default=None
        Number of prior samples. Defaults to the number of input channels.
    augmentations : sequence of {"x_flip", "y_flip", "rotate_90"} or None, default=None
        Augmentations to apply. `None` applies the N2V defaults.
    n_val_patches : int, default=8
        Number of patches held out for validation.
    in_memory : bool or None, default=None
        Whether to load all data into memory.
    channels : sequence of int or None, default=None
        Channel indices to use.
    independent_channels : bool, default=True
        Whether to process input channels independently. With one input channel this
        does not change the UNet.
    normalization : {"mean_std", "min_max", "quantile", "none"}, default="mean_std"
        Normalization strategy.
    normalization_params : dict or None, default=None
        Strategy-specific normalization parameters.
    patch_filter_config : SupportedPatchFilterConfig or None, default=None
        Patch filtering configuration.
    use_n2v2 : bool, default=False
        Whether to use the N2V2 manipulation strategy and architecture changes.
    roi_size : int, default=11
        N2V pixel manipulation area.
    masked_pixel_percentage : float, default=0.2
        Percentage of pixels masked in each patch.
    struct_n2v_axes : {"horizontal", "vertical", "cross", "square", "none"}
        StructN2V mask axes, default "none".
    struct_n2v_span : int, default=5
        Span of the structN2V mask.
    sample_scale : float, default=10.0
        Scale applied to normalized outputs before denormalization.
    num_workers : int, default=-1
        Number of dataloader workers.
    trainer_params : dict or None, default=None
        Extra Lightning Trainer parameters, such as `accumulate_grad_batches`.
    model_params : dict or None, default=None
        Extra UNet parameters, such as `depth` and `num_channels_init`.
    optimizer : {"Adam", "Adamax", "SGD"}, default="Adam"
        Optimizer name.
    optimizer_params : dict or None, default=None
        Optimizer parameters.
    lr_scheduler : {"ReduceLROnPlateau", "StepLR"}, default="ReduceLROnPlateau"
        Learning rate scheduler name.
    lr_scheduler_params : dict or None, default=None
        Learning rate scheduler parameters.
    monitor_metric : {"train_loss", "train_loss_epoch", "val_loss"}, default="val_loss"
        Metric monitored by the scheduler.
    train_dataloader_params : dict or None, default=None
        Training dataloader parameters.
    val_dataloader_params : dict or None, default=None
        Validation dataloader parameters.
    checkpoint_params : dict or None, default=None
        Checkpoint callback parameters.
    logger : {"wandb", "tensorboard", "none"}, default="none"
        Logger to use.
    seed : int or None, default=None
        Random seed for the pixel manipulation.

    Returns
    -------
    PN2VConfiguration
        Configuration for training PN2V.
    """
    if seed is None:
        seed = generate_random_seed()

    n_channels = _validate_n2v_channel_dim(axes, channels, n_channels)
    n_output = n_channels if num_classes is None else num_classes

    norm_config = {"name": normalization}
    if normalization_params is not None:
        norm_config.update(normalization_params)

    data_config = create_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=assemble_augmentations(augmentations=augmentations, seed=seed),
        normalization=norm_config,
        patch_filter_config=patch_filter_config,
        channels=channels,
        in_memory=in_memory,
        n_val_patches=n_val_patches,
        num_workers=num_workers,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
        seed=seed,
    )

    algorithm_params = create_algorithm_configuration(
        dimensions=3 if data_config.is_3D() else 2,
        algorithm="pn2v",
        loss="pn2v",
        independent_channels=independent_channels,
        n_channels_in=n_channels,
        n_channels_out=n_output,
        use_n2v2=use_n2v2,
        model_params=model_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        lr_scheduler_params=lr_scheduler_params,
    )
    algorithm_params["n2v_config"] = _create_n2v_transform(
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        use_n2v2=use_n2v2,
        struct_n2v_axes=struct_n2v_axes,
        struct_n2v_span=struct_n2v_span,
        seed=seed,
    )
    algorithm_params["monitor_metric"] = monitor_metric
    algorithm_params["sample_scale"] = sample_scale
    algorithm_params["noise_model"] = noise_model

    final_trainer_params = update_trainer_params(
        trainer_params=trainer_params,
        num_epochs=num_epochs,
        num_steps=num_steps,
    )
    training_params = create_training_configuration(
        algorithm="pn2v",
        trainer_params=final_trainer_params,
        logger=logger,
        checkpoint_params=checkpoint_params,
    )

    return PN2VConfiguration(
        experiment_name=experiment_name,
        algorithm_config=PN2VAlgorithm(**algorithm_params),
        data_config=data_config,
        training_config=training_params,
    )
