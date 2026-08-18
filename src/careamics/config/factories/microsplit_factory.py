"""Convenience functions to create MicroSplit configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.augmentations import XYFlipConfig, XYRandomRotate90Config
from careamics.config.data import MicroSplitDataConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.microsplit_configuration import MicroSplitConfiguration
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

from .data_factory import create_data_configuration, list_spatial_augmentations
from .training_factory import create_training_configuration, update_trainer_params


def create_microsplit_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    output_channels: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    multiscale_count: int = 3,
    noise_model: MultiChannelNMConfig | None = None,
) -> MicroSplitConfiguration:
    """Create a configuration for training MicroSplit.

    See `create_advanced_microsplit_config` for more parameters.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "zarr", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : sequence of int
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    output_channels : int
        Number of target channels to split into.
    num_epochs : int, default=100
        Number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in 1 epoch.
    augmentations : sequence of {"x_flip", "y_flip", "rotate_90"} or None, default=None
        List of augmentations to apply. If `None`, all augmentations are applied.
    n_val_patches : int, default=8
        Number of patches to set aside for validation during training.
    multiscale_count : int, default=3
        Number of lateral-context scales.
    noise_model : MultiChannelNMConfig or None, default=None
        Trained noise model, required for denoiSplit training.

    Returns
    -------
    MicroSplitConfiguration
        Configuration for training MicroSplit.
    """
    return create_advanced_microsplit_config(**locals())


def create_advanced_microsplit_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    output_channels: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    # advanced data parameters
    channels: Sequence[int] | None = None,
    normalization: Literal["mean_std", "min_max", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
    in_memory: bool | None = None,
    multiscale_count: int = 3,
    padding_mode: Literal["reflect", "wrap"] = "reflect",
    alpha_ranges: Sequence[tuple[float, float]] | None = None,
    uncorrelated_channel_prob: float = 0.0,
    # model parameters
    model_params: dict[str, Any] | None = None,
    predict_logvar: bool = True,
    logvar_lowerbound: float | None = -5.0,
    # loss parameters
    reconstruction_weight: float = 1.0,
    kl_weight: float = 1.0,
    musplit_weight: float = 0.1,
    denoisplit_weight: float = 0.9,
    # algorithm parameters
    noise_model: MultiChannelNMConfig | None = None,
    mmse_count: int = 10,
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
) -> MicroSplitConfiguration:
    """Create an advanced configuration for training MicroSplit.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "zarr", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : sequence of int
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    output_channels : int
        Number of target channels to split into.
    num_epochs : int, default=100
        Number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in 1 epoch.
    augmentations : sequence of {"x_flip", "y_flip", "rotate_90"} or None, default=None
        List of augmentations to apply. If `None`, all augmentations are applied.
    n_val_patches : int, default=8
        Number of patches to set aside for validation during training.
    channels : sequence of int or None, default=None
        List of channels to use. If `None`, all channels are used.
    normalization : {"mean_std", "min_max", "quantile", "none"}, default="mean_std"
        Normalization strategy.
    normalization_params : dict or None, default=None
        Additional normalization parameters.
    in_memory : bool or None, default=None
        Whether to load all data into memory.
    multiscale_count : int, default=3
        Number of lateral-context scales.
    padding_mode : {"reflect", "wrap"}, default="reflect"
        Padding mode for lateral-context patches extending beyond image borders.
    alpha_ranges : sequence of tuple of float or None, default=None
        Ranges used to sample channel mixing weights for synthetic inputs.
    uncorrelated_channel_prob : float, default=0.0
        Probability of sampling uncorrelated channels for synthetic inputs.
    model_params : dict or None, default=None
        LVAE model parameters overriding the MicroSplit defaults (`z_dims=[128, 128]`,
        `n_filters=32`, `encoder_dropout=0.1`,
        `decoder_dropout=0.1`). Structural parameters
        (`architecture`, `input_shape`, `output_channels`, `multiscale_count`,
        `encoder_conv_strides`, `decoder_conv_strides`, `predict_logvar`) are set from
        the dedicated arguments and cannot be overridden here.
    predict_logvar : bool, default=True
        Whether to predict the pixelwise log-variance.
    logvar_lowerbound : float or None, default=-5.0
        Lower bound on the predicted log-variance.
    reconstruction_weight : float, default=1.0
        Weight of the reconstruction term.
    kl_weight : float, default=1.0
        Weight of the KL term.
    musplit_weight : float, default=0.1
        Weight of the Gaussian likelihood (muSplit).
    denoisplit_weight : float, default=0.9
        Weight of the noise model likelihood (denoiSplit).
    noise_model : MultiChannelNMConfig or None, default=None
        Trained noise model, required for denoiSplit training.
    mmse_count : int, default=10
        Number of samples used for MMSE prediction.
    num_workers : int, default=-1
        Number of workers for data loading.
    trainer_params : dict or None, default=None
        Parameters for the PyTorch Lightning Trainer.
    optimizer : {"Adam", "Adamax", "SGD"}, default="Adamax"
        Optimizer name.
    optimizer_params : dict or None, default=None
        Optimizer parameters. If `None`, `{"lr": 1e-3, "weight_decay": 0}` is used.
    lr_scheduler : {"ReduceLROnPlateau", "StepLR"}, default="ReduceLROnPlateau"
        Learning rate scheduler.
    lr_scheduler_params : dict or None, default=None
        Learning rate scheduler parameters. If `None`, a `ReduceLROnPlateau` preset
        (`mode="min"`, `factor=0.5`, `patience=30`, `min_lr=1e-12`) is used.
    train_dataloader_params : dict or None, default=None
        Parameters for the training dataloader.
    val_dataloader_params : dict or None, default=None
        Parameters for the validation dataloader.
    checkpoint_params : dict or None, default=None
        Parameters for the checkpoint callback.
    early_stopping_params : dict or None, default=None
        Parameters for the early stopping callback.
    logger : {"wandb", "tensorboard", "none"}, default="none"
        Logger to use.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    MicroSplitConfiguration
        Configuration for training MicroSplit.
    """
    conv_strides = [2] * len(patch_size)

    # TODO consider accepting an LVAELossConfig directly instead of individual weights
    # (see PR #1007 discussion); to be addressed in a follow-up PR.
    loss = LVAELossConfig(
        loss_type="microsplit",
        reconstruction_weight=reconstruction_weight,
        kl_weight=kl_weight,
        musplit_weight=musplit_weight,
        denoisplit_weight=denoisplit_weight,
        predict_logvar=predict_logvar,
        logvar_lowerbound=logvar_lowerbound,
    )

    # MicroSplit-specific LVAE defaults; user-supplied `model_params` override these,
    # while structural parameters (set below) always take precedence.
    lvae_params: dict[str, Any] = {
        "z_dims": [128, 128],
        "n_filters": 32,
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.1,
        **(model_params or {}),
        "architecture": "LVAE",
        "input_shape": tuple(patch_size),
        "output_channels": output_channels,
        "multiscale_count": multiscale_count,
        "encoder_conv_strides": conv_strides,
        "decoder_conv_strides": conv_strides,
        "predict_logvar": predict_logvar,
    }
    model = LVAEConfig(**lvae_params)

    algorithm_config = MicroSplitAlgorithm(
        algorithm="microsplit",
        loss=loss,
        model=model,
        noise_model=noise_model,
        mmse_count=mmse_count,
        optimizer=OptimizerConfig(
            name=optimizer,
            parameters=optimizer_params or {"lr": 1e-3, "weight_decay": 0},
        ),
        lr_scheduler=LrSchedulerConfig(
            name=lr_scheduler,
            parameters=lr_scheduler_params
            or {"mode": "min", "factor": 0.5, "patience": 30, "min_lr": 1e-12},
        ),
    )

    norm_config = {"name": normalization}
    if normalization_params is not None:
        norm_config.update(normalization_params)

    # TODO refactor when #1005 will be merged
    augs: list[XYFlipConfig | XYRandomRotate90Config] | None = None
    if augmentations is not None:
        augs = []
        if "x_flip" in augmentations or "y_flip" in augmentations:
            augs.append(
                XYFlipConfig(
                    flip_x="x_flip" in augmentations,
                    flip_y="y_flip" in augmentations,
                    seed=seed,
                )
            )
        if "rotate_90" in augmentations:
            augs.append(XYRandomRotate90Config(seed=seed))

    base_data_config = create_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=list_spatial_augmentations(augs),
        normalization=norm_config,
        channels=channels,
        in_memory=in_memory,
        n_val_patches=n_val_patches,
        num_workers=num_workers,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
        seed=seed,
    )
    data_config = MicroSplitDataConfig(
        **base_data_config.model_dump(),
        multiscale_count=multiscale_count,
        padding_mode=padding_mode,
        alpha_ranges=alpha_ranges,
        uncorrelated_channel_prob=uncorrelated_channel_prob,
    )

    training_config = create_training_configuration(
        algorithm="microsplit",
        trainer_params=update_trainer_params(
            trainer_params=trainer_params,
            num_epochs=num_epochs,
            num_steps=num_steps,
        ),
        logger=logger,
        checkpoint_params=checkpoint_params,
        early_stopping_params=early_stopping_params,
    )

    return MicroSplitConfiguration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_config,
        data_config=data_config,
        training_config=training_config,
    )
