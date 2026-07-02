"""Convenience functions to create HDN configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.algorithms import HDNAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

NonLinearity = Literal["None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"]


def create_hdn_config(
    # mandatory parameters
    patch_size: Sequence[int],
    # optional noise model
    noise_model: MultiChannelNMConfig | None = None,
) -> HDNAlgorithm:
    """Create a configuration for training HDN.

    HDN (Hierarchical DivNoising) denoises images with an LVAE. The reconstruction
    likelihood is selected automatically:

    - pass a ``noise_model`` to use the noise model likelihood (HDN paper);
    - omit it to learn a Gaussian likelihood with a per-pixel variance (DivNoising).

    See ``create_advanced_hdn_config`` for the full set of parameters.

    Parameters
    ----------
    patch_size : Sequence[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]). Length 2 for
        2D, length 3 for 3D. Minimum spatial size is 64.
    noise_model : MultiChannelNMConfig or None, default=None
        Trained noise model. If ``None``, the Gaussian (DivNoising) pathway is used.

    Returns
    -------
    HDNAlgorithm
        Configuration for training HDN.
    """
    return create_advanced_hdn_config(
        patch_size=patch_size,
        noise_model=noise_model,
    )


def create_advanced_hdn_config(
    # mandatory parameters
    patch_size: Sequence[int],
    # optional noise model
    noise_model: MultiChannelNMConfig | None = None,
    output_channels: int = 1,
    # architecture parameters
    z_dims: Sequence[int] = (32, 32, 32, 32, 32, 32),
    n_filters: int = 64,
    blocks_per_layer: int = 5,
    dropout: float = 0.2,
    nonlinearity: NonLinearity = "ELU",
    # --- posterior-collapse / loss knobs -------------------------------------
    reconstruction_weight: float = 1.0,
    kl_weight: float = 1.0,
    logvar_lowerbound: float | None = -5.0,  # Gaussian-path variance floor
    # --- optimization --------------------------------------------------------
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adamax",
    optimizer_params: dict[str, Any] | None = None,  # None -> HDN default lr 3e-4
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    # --- supervision ---------------------------------------------------------
    is_supervised: bool = False,
) -> HDNAlgorithm:
    """Create an advanced configuration for training HDN.

    The likelihood pathway (and therefore ``predict_logvar``) is derived from
    ``noise_model``; the user never sets ``predict_logvar`` directly.

    Parameters
    ----------
    patch_size : Sequence[int]
        Spatial patch size (length 2 for 2D, 3 for 3D); becomes the LVAE
        ``input_shape``.
    noise_model : MultiChannelNMConfig or None, default=None
        Trained noise model. ``None`` selects the Gaussian (DivNoising) pathway.
    output_channels : int, default=1
        Number of target channels (HDN uses 1).
    z_dims : Sequence[int], default=(32, 32, 32, 32, 32, 32)
        Latent channels per hierarchy level; its length sets the number of LVAE layers
        (HDN: 6 levels of 32).
    n_filters : int, default=64
        Convolution width, shared by encoder and decoder (they must match because the
        LVAE merges their features).
    blocks_per_layer : int, default=5
        Number of residual blocks per hierarchy level (HDN uses 5).
    dropout : float, default=0.2
        Dropout rate, shared by encoder and decoder.
    nonlinearity : str, default="ELU"
        Activation function, one of "None", "Sigmoid", "Softmax", "Tanh", "ReLU",
        "LeakyReLU", "ELU".
    reconstruction_weight : float, default=1.0
        Weight of the reconstruction term.
    kl_weight : float, default=1.0
        Weight of the KL term (beta). Annealing is not used.
    logvar_lowerbound : float or None, default=-5.0
        Lower bound on the predicted log-variance (Gaussian pathway only); prevents the
        predicted variance from collapsing toward zero. Ignored on the noise model path.
    optimizer : {"Adam","Adamax","SGD"}, default="Adamax"
        Optimizer name.
    optimizer_params : dict or None, default=None
        Optimizer parameters. If ``None``, the HDN default ``{"lr": 3e-4}`` is used.
    lr_scheduler : {"ReduceLROnPlateau","StepLR"}, default="ReduceLROnPlateau"
        Learning rate scheduler.
    lr_scheduler_params : dict or None, default=None
        Scheduler parameters.
    is_supervised : bool, default=False
        Whether a supervised target is provided via the second batch element.

    Returns
    -------
    HDNAlgorithm
        Configuration for training HDN.
    """
    predict_logvar = noise_model is None

    n_spatial = len(patch_size)
    conv_strides = [2] * n_spatial

    loss = LVAELossConfig(
        loss_type="hdn",
        reconstruction_weight=reconstruction_weight,
        kl_weight=kl_weight,
        predict_logvar=predict_logvar,
        logvar_lowerbound=logvar_lowerbound,
    )

    model = LVAEConfig(
        architecture="LVAE",
        input_shape=tuple(patch_size),
        output_channels=output_channels,
        multiscale_count=1,
        z_dims=list(z_dims),
        encoder_n_filters=n_filters,
        decoder_n_filters=n_filters,
        encoder_conv_strides=conv_strides,
        decoder_conv_strides=conv_strides,
        encoder_dropout=dropout,
        decoder_dropout=dropout,
        encoder_blocks_per_layer=blocks_per_layer,
        decoder_blocks_per_layer=blocks_per_layer,
        nonlinearity=nonlinearity,
        predict_logvar=predict_logvar,
    )

    return HDNAlgorithm(
        algorithm="hdn",
        loss=loss,
        model=model,
        noise_model=noise_model,
        is_supervised=is_supervised,
        optimizer=OptimizerConfig(
            name=optimizer,
            parameters=optimizer_params or {"lr": 3e-4},
        ),
        lr_scheduler=LrSchedulerConfig(
            name=lr_scheduler,
            parameters=lr_scheduler_params or {},
        ),
    )
