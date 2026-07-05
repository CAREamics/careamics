"""Convenience functions to create HDN configurations."""

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal

from careamics.config.algorithms import HDNAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.augmentations import XYFlipConfig, XYRandomRotate90Config
from careamics.config.hdn_configuration import HDNConfiguration
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.lightning.training_configuration import (
    SelfSupervisedCheckpointing,
    TrainingConfig,
)
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

from .data_factory import create_ng_data_configuration, list_spatial_augmentations
from .training_factory import update_trainer_params

NonLinearity = Literal["None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"]


def create_hdn_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    noise_model: MultiChannelNMConfig | None = None,
) -> HDNConfiguration:
    """Create a configuration for training HDN.

    The reconstruction likelihood is selected from `noise_model`: pass one to use the
    noise model likelihood, omit it to learn a Gaussian likelihood (DivNoising).

    See `create_advanced_hdn_config` for more parameters.

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
    num_epochs : int, default=100
        Number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in 1 epoch.
    augmentations : sequence of {"x_flip", "y_flip", "rotate_90"} or None, default=None
        List of augmentations to apply. If `None`, all augmentations are applied.
    n_val_patches : int, default=8
        Number of patches to set aside for validation during training.
    noise_model : MultiChannelNMConfig or None, default=None
        Trained noise model. If `None`, the Gaussian (DivNoising) pathway is used.

    Returns
    -------
    HDNConfiguration
        Configuration for training HDN.
    """
    return create_advanced_hdn_config(**locals())


def create_advanced_hdn_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    noise_model: MultiChannelNMConfig | None = None,
    # advanced data parameters
    channels: Sequence[int] | None = None,
    normalization: Literal["mean_std", "min_max", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
    in_memory: bool | None = None,
    # model parameters
    output_channels: int = 1,
    z_dims: Sequence[int] = (128, 128),
    encoder_n_filters: int = 32,
    decoder_n_filters: int = 32,
    encoder_dropout: float = 0.0,
    decoder_dropout: float = 0.0,
    nonlinearity: NonLinearity = "ReLU",
    analytical_kl: bool = False,
    # loss parameters
    reconstruction_weight: float = 1.0,
    kl_weight: float = 1.0,
    logvar_lowerbound: float | None = -5.0,
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
) -> HDNConfiguration:
    """Create an advanced configuration for training HDN.

    `predict_logvar` is derived from `noise_model`: it is enabled when no noise model
    is provided (DivNoising Gaussian likelihood) and disabled otherwise.

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
    num_epochs : int, default=100
        Number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in 1 epoch.
    augmentations : sequence of {"x_flip", "y_flip", "rotate_90"} or None, default=None
        List of augmentations to apply. If `None`, all augmentations are applied.
    n_val_patches : int, default=8
        Number of patches to set aside for validation during training.
    noise_model : MultiChannelNMConfig or None, default=None
        Trained noise model. If `None`, the Gaussian (DivNoising) pathway is used.
    channels : sequence of int or None, default=None
        List of channels to use. If `None`, all channels are used.
    normalization : {"mean_std", "min_max", "quantile", "none"}, default="mean_std"
        Normalization strategy.
    normalization_params : dict or None, default=None
        Additional normalization parameters.
    in_memory : bool or None, default=None
        Whether to load all data into memory.
    output_channels : int, default=1
        Number of target channels (HDN uses 1).
    z_dims : sequence of int, default=(128, 128)
        Latent channels per hierarchy level; its length sets the number of LVAE layers.
    encoder_n_filters : int, default=32
        Number of encoder convolution filters.
    decoder_n_filters : int, default=32
        Number of decoder convolution filters.
    encoder_dropout : float, default=0.0
        Encoder dropout rate.
    decoder_dropout : float, default=0.0
        Decoder dropout rate.
    nonlinearity : str, default="ReLU"
        Activation function.
    analytical_kl : bool, default=False
        Whether to use the analytical KL divergence.
    reconstruction_weight : float, default=1.0
        Weight of the reconstruction term.
    kl_weight : float, default=1.0
        Weight of the KL term.
    logvar_lowerbound : float or None, default=-5.0
        Lower bound on the predicted log-variance (Gaussian pathway only).
    num_workers : int, default=-1
        Number of workers for data loading.
    trainer_params : dict or None, default=None
        Parameters for the PyTorch Lightning Trainer.
    optimizer : {"Adam", "Adamax", "SGD"}, default="Adamax"
        Optimizer name.
    optimizer_params : dict or None, default=None
        Optimizer parameters. If `None`, `{"lr": 3e-4}` is used.
    lr_scheduler : {"ReduceLROnPlateau", "StepLR"}, default="ReduceLROnPlateau"
        Learning rate scheduler.
    lr_scheduler_params : dict or None, default=None
        Learning rate scheduler parameters.
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
    HDNConfiguration
        Configuration for training HDN.
    """
    predict_logvar = noise_model is None
    conv_strides = [2] * len(patch_size)

    loss = LVAELossConfig(
        loss_type="hdn",
        reconstruction_weight=reconstruction_weight,
        kl_weight=kl_weight,
        musplit_weight=0.0,
        denoisplit_weight=1.0,
        predict_logvar=predict_logvar,
        logvar_lowerbound=logvar_lowerbound,
    )

    model = LVAEConfig(
        architecture="LVAE",
        input_shape=tuple(patch_size),
        output_channels=output_channels,
        multiscale_count=1,
        z_dims=list(z_dims),
        encoder_n_filters=encoder_n_filters,
        decoder_n_filters=decoder_n_filters,
        encoder_conv_strides=conv_strides,
        decoder_conv_strides=conv_strides,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        nonlinearity=nonlinearity,
        analytical_kl=analytical_kl,
        predict_logvar=predict_logvar,
    )

    algorithm_config = HDNAlgorithm(
        algorithm="hdn",
        loss=loss,
        model=model,
        noise_model=noise_model,
        is_supervised=False,
        optimizer=OptimizerConfig(
            name=optimizer,
            parameters=optimizer_params or {"lr": 3e-4},
        ),
        lr_scheduler=LrSchedulerConfig(
            name=lr_scheduler,
            parameters=lr_scheduler_params or {},
        ),
    )

    norm_config = {"name": normalization}
    if normalization_params is not None:
        norm_config.update(normalization_params)

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

    data_config = create_ng_data_configuration(
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

    training_config = TrainingConfig(
        trainer_params=update_trainer_params(
            trainer_params=trainer_params,
            num_epochs=num_epochs,
            num_steps=num_steps,
        ),
        logger=None if logger == "none" else logger,
        checkpoint_params=(
            checkpoint_params
            if checkpoint_params is not None
            else asdict(SelfSupervisedCheckpointing())
        ),
        early_stopping_params=early_stopping_params,
    )

    return HDNConfiguration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_config,
        data_config=data_config,
        training_config=training_config,
    )
