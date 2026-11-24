"""Convenience functions to create MicroSplit configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.data import MicroSplitDataConfig
from careamics.config.data.normalization_config import NormalizationConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.losses import MicroSplitLossConfig
from careamics.config.microsplit_configuration import MicroSplitConfiguration

from .config_discriminators import instantiate_norm_config
from .data_factory import create_data_configuration
from .factory_utils import assemble_augmentations
from .training_factory import create_training_configuration, update_trainer_params


def create_microsplit_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    n_channels_out: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    multiscale_count: int = 3,
    use_noise_model: bool = True,
    seed: int | None = None,
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
    n_channels_out : int
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
    use_noise_model : bool, default=True
        Whether to use a noise model during training. If `True`, a noise model must be
        supplied at train time.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    MicroSplitConfiguration
        Configuration for training MicroSplit.

    Notes
    -----
    Setting `use_noise_model=True` configures MicroSplit to use two reconstruction terms
    in the loss, Gaussian (weight 0.1) and noise model likelihoods (0.9), while
    `use_noise_model=False` leads to only the Gaussian likelihood term.

    If `use_noise_model` is set to `True`, then a noise model must be trained using
    `train_noise_model` and the resulting model be passed at MicroSplit training time
    to `CAREamist.train(noise_model=...)` or directly set into the MicroSplit Lightning
    module if using the Lightning API.
    """
    return create_advanced_microsplit_config(**locals())


def create_advanced_microsplit_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    n_channels_out: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
    n_val_patches: int = 8,
    # advanced data parameters
    channels: Sequence[int] | None = None,
    normalization: (
        Literal["mean_std", "min_max", "quantile", "none"] | NormalizationConfig
    ) = "mean_std",
    in_memory: bool | None = None,
    multiscale_count: int = 3,
    alpha_ranges: Sequence[tuple[float, float]] | None = None,
    uncorrelated_channel_prob: float = 0.0,
    # advanced algorithm parameters
    use_noise_model: bool | None = None,
    model_params: dict[str, Any] | None = None,
    loss: MicroSplitLossConfig | None = None,
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

    `use_noise_model` controls default weights of the MicroSplit loss. If set to `True`,
    then a noise model must be provided at training time. `use_noise_model` is not
    compatible with explicitly setting `loss`, which may or may not set up a loss that
    requires a noise model.

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
    n_channels_out : int
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
    normalization : {"mean_std", "min_max", "quantile", "none"} or NormalizationConfig
        Normalization strategy or complete normalization configuration. Default is
        `"mean_std"`.
    in_memory : bool or None, default=None
        Whether to load all data into memory.
    multiscale_count : int, default=3
        Number of lateral-context scales.
    alpha_ranges : sequence of tuple of float or None, default=None
        Ranges used to sample channel mixing weights for synthetic inputs.
    uncorrelated_channel_prob : float, default=0.0
        Probability of sampling uncorrelated channels for synthetic inputs.
    use_noise_model : bool | None, default=None
        Whether to use a noise model during training. Cannot be passed together with
        `loss` as the latter may define a noise model-dependent loss. If neither
        `use_noise_model` nor `loss` are passed, then defaults to `True`.
    model_params : dict or None, default=None
        LVAE model parameters overriding the MicroSplit defaults (`z_dims=[128, 128]`,
        `n_filters=32`, `encoder_dropout=0.1`,
        `decoder_dropout=0.1`). Structural parameters
        (`architecture`, `input_shape`, `output_channels`, `multiscale_count`,
        `encoder_conv_strides`, `decoder_conv_strides`, `predict_logvar`,
        `analytical_kl`, `enable_topdown_normalize_factor`,
        `encoder_first_conv_kernel`) are set by the algorithm and cannot be overridden
        here.
    loss : MicroSplitLossConfig | None, default=None
        MicroSplit loss configuration, allowing fine control over the weight of
        the loss terms. Cannot be passed together with `use_noise_model` as `loss` may
        define a noise model-dependent loss.
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

    Raises
    ------
    ValueError
        If both `use_noise_model` and `loss` are supplied.

    Notes
    -----
    Setting `use_noise_model=True` configures MicroSplit to use two reconstruction terms
    in the loss, Gaussian (weight 0.1) and noise model likelihoods (0.9), while
    `use_noise_model=False` leads to only the Gaussian likelihood term.

    If `use_noise_model` is set to `True`, then a noise model must be trained using
    `train_noise_model` and the resulting model be passed at MicroSplit training time
    to `CAREamist.train(noise_model=...)` or directly set into the MicroSplit Lightning
    module if using the Lightning API.

    `loss` allows setting all loss weights, including the reconstruction terms and
    whether a noise model is required.
    """
    conv_strides = [2] * len(patch_size)

    predict_logvar = True  # default
    if loss is None:
        if use_noise_model is None:
            denoising_weight = 0.9
        else:
            denoising_weight = 0.9 if use_noise_model else 0

        loss = MicroSplitLossConfig(
            reconstruction_weight=1.0,
            kl_weight=1.0,
            gaussian_likelihood_weight=1.0 - denoising_weight,
            noise_model_likelihood_weight=denoising_weight,
            predict_logvar=predict_logvar,
            logvar_lowerbound=-5.0,
        )
    else:
        if use_noise_model is not None:
            raise ValueError(
                "Parameters `use_noise_model` and `loss` cannot be passed conjointly. "
                "Use `use_noise_model` for simple switch between using a noise model "
                "or not. Use `loss` for fine setting of the loss terms, including "
                "whether to use a noise model loss term."
            )

        # read `predict_logvar` from the supplied loss
        predict_logvar = loss.predict_logvar

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
        "output_channels": n_channels_out,
        "multiscale_count": multiscale_count,
        "encoder_conv_strides": conv_strides,
        "decoder_conv_strides": conv_strides,
        "predict_logvar": predict_logvar,
        "analytical_kl": False,
        "enable_topdown_normalize_factor": True,
        "encoder_first_conv_kernel": 3,
    }
    model = LVAEConfig(**lvae_params)

    algorithm_config = MicroSplitAlgorithm(
        algorithm="microsplit",
        loss=loss,
        model=model,
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

    norm_config = (
        instantiate_norm_config({"name": normalization})
        if isinstance(normalization, str)
        else normalization
    )

    base_data_config = create_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=assemble_augmentations(augmentations, seed),
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
        padding_mode="reflect",  # TODO is it ever changed in our experiments?
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
