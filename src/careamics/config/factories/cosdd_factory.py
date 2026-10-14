"""Convenience functions to create COSDD configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.algorithms import COSDDAlgorithm
from careamics.config.architectures import COSDDConfig
from careamics.config.cosdd_configuration import COSDDConfiguration
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)

from .data_factory import SupportedPatchFilterConfig, create_data_configuration
from .factory_utils import assemble_augmentations, validate_input_channels
from .training_factory import create_training_configuration, update_trainer_params


def create_cosdd_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int = 4,
    num_epochs: int = 1000,
    num_steps: int | None = None,
    n_val_patches: int = 8,
    noise_direction: Literal["x", "y", "z", "none"] = "x",
    use_direct_denoiser: bool = True,
    n_channels: int | None = None,
) -> COSDDConfiguration:
    """Create a configuration for training COSDD.

    COSDD is an unsupervised denoiser for noise that is signal-dependent and
    correlated along one axis (`noise_direction`), e.g. rows in scanning-based
    microscopy. It is trained on the noisy images only.

    If "C" is in `axes`, then `n_channels` must be set to the number of channels.

    COSDD can take a long time to converge, consider limiting the training time with
    the Lightning Trainer `max_time` parameter (see `create_advanced_cosdd_config`).
    Early stopping on the validation loss is enabled by default.

    No augmentations are applied by default, see `create_advanced_cosdd_config`.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "zarr", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence of int
        Size of the patches along the spatial dimensions (e.g. [256, 256]).
    batch_size : int, default=4
        Batch size.
    num_epochs : int, default=1000
        Maximum number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in 1 epoch.
    n_val_patches : int, default=8
        Number of patches to set aside for validation during training. Ignored if
        separate validation data is specified for training.
    noise_direction : {"x", "y", "z", "none"}, default="x"
        Axis along which the noise is correlated.
    use_direct_denoiser : bool, default=True
        Whether to train a direct denoiser, making prediction deterministic and
        fast.
    n_channels : int or None, default=None
        Number of channels.

    Returns
    -------
    COSDDConfiguration
        Configuration for training COSDD.
    """
    return create_advanced_cosdd_config(**locals())


def create_advanced_cosdd_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int = 4,
    num_epochs: int = 1000,
    num_steps: int | None = None,
    n_val_patches: int = 8,
    noise_direction: Literal["x", "y", "z", "none"] = "x",
    use_direct_denoiser: bool = True,
    n_channels: int | None = None,
    # advanced data parameters
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] = (),
    in_memory: bool | None = None,
    channels: Sequence[int] | None = None,
    normalization: Literal["mean_std", "min_max", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
    patch_filter_config: SupportedPatchFilterConfig | None = None,
    # model parameters
    model_params: dict[str, Any] | None = None,
    n_grad_batches: int = 4,
    # lightning parameters
    num_workers: int = -1,
    trainer_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adamax",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
    early_stopping_params: dict[str, Any] | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    seed: int | None = None,
) -> COSDDConfiguration:
    """Create an advanced configuration for training COSDD.

    The COSDD network parameters can be set in `model_params` (see `COSDDConfig`),
    e.g. `s_code_channels`, `n_layers`, `n_gaussians`, `scale_initialisation`,
    `direct_denoiser_loss` or `checkpointed`. `noise_direction`,
    `use_direct_denoiser` and the channels take precedence over `model_params`.

    COSDD trains the Ladder VAE with the autoregressive decoder, the signal decoder
    and the direct denoiser with separate optimizers, all built from `optimizer` and
    `lr_scheduler`. Gradients are accumulated over `n_grad_batches` batches.

    Random 90 degree rotations are only allowed if the noise is not correlated along
    X or Y.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "zarr", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence of int
        Size of the patches along the spatial dimensions (e.g. [256, 256]).
    batch_size : int, default=4
        Batch size.
    num_epochs : int, default=1000
        Maximum number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in 1 epoch.
    n_val_patches : int, default=8
        Number of patches to set aside for validation during training. Ignored if
        separate validation data is specified for training.
    noise_direction : {"x", "y", "z", "none"}, default="x"
        Axis along which the noise is correlated.
    use_direct_denoiser : bool, default=True
        Whether to train a direct denoiser, making prediction deterministic and
        fast.
    n_channels : int or None, default=None
        Number of channels. If `channels` is specified, the number of channels is
        inferred from its length.
    augmentations : Sequence of {"x_flip", "y_flip", "rotate_90"}, default=()
        Augmentations to apply. No augmentation is applied by default.
    in_memory : bool or None, default=None
        Whether to load all data into memory.
    channels : Sequence of int or None, default=None
        List of channels to use. If `None`, all channels are used.
    normalization : {"mean_std", "min_max", "quantile", "none"}, default="mean_std"
        Normalization strategy.
    normalization_params : dict or None, default=None
        Strategy-specific normalization parameters.
    patch_filter_config : SupportedPatchFilterConfig or None, default=None
        Patch filter configuration, reducing the probability of selecting
        background patches during training.
    model_params : dict or None, default=None
        Additional `COSDDConfig` parameters.
    n_grad_batches : int, default=4
        Number of batches over which gradients are accumulated.
    num_workers : int, default=-1
        Number of workers for data loading.
    trainer_params : dict or None, default=None
        Parameters for the PyTorch Lightning Trainer, e.g. `max_time` or
        `precision`.
    optimizer : {"Adam", "Adamax", "SGD"}, default="Adamax"
        Optimizer name.
    optimizer_params : dict or None, default=None
        Optimizer parameters.
    lr_scheduler : {"ReduceLROnPlateau", "StepLR"}, default="ReduceLROnPlateau"
        Learning rate scheduler.
    lr_scheduler_params : dict or None, default=None
        Learning rate scheduler parameters. If `None` and `lr_scheduler` is
        `ReduceLROnPlateau`, `{"factor": 0.1, "patience": 25}` is used.
    train_dataloader_params : dict or None, default=None
        Parameters for the training dataloader.
    val_dataloader_params : dict or None, default=None
        Parameters for the validation dataloader.
    checkpoint_params : dict or None, default=None
        Parameters for the checkpoint callback.
    early_stopping_params : dict or None, default=None
        Parameters for the early stopping callback. If `None`, training stops when
        the validation loss has not improved for 50 epochs.
    logger : {"wandb", "tensorboard", "none"}, default="none"
        Logger to use.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    COSDDConfiguration
        Configuration for training COSDD.
    """
    validate_input_channels(
        axes=axes, channels=channels, n_channels=n_channels, attr_name="n_channels"
    )
    if channels is not None:
        n_channels = len(channels)
    elif n_channels is None:
        n_channels = 1

    model = COSDDConfig(
        **{
            **(model_params or {}),
            "architecture": "COSDD",
            "input_shape": tuple(patch_size),
            "in_channels": n_channels,
            "noise_direction": noise_direction,
            "use_direct_denoiser": use_direct_denoiser,
        }
    )

    if lr_scheduler_params is None and lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler_params = {"factor": 0.1, "patience": 25}

    algorithm_config = COSDDAlgorithm(
        model=model,
        optimizer=OptimizerConfig(name=optimizer, parameters=optimizer_params or {}),
        lr_scheduler=LrSchedulerConfig(
            name=lr_scheduler, parameters=lr_scheduler_params or {}
        ),
        n_grad_batches=n_grad_batches,
    )

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

    if early_stopping_params is None:
        early_stopping_params = {"monitor": "val_loss", "mode": "min", "patience": 50}

    training_config = create_training_configuration(
        algorithm="cosdd",
        trainer_params=update_trainer_params(
            trainer_params=trainer_params,
            num_epochs=num_epochs,
            num_steps=num_steps,
        ),
        logger=logger,
        checkpoint_params=checkpoint_params,
        early_stopping_params=early_stopping_params,
    )

    return COSDDConfiguration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_config,
        data_config=data_config,
        training_config=training_config,
    )
