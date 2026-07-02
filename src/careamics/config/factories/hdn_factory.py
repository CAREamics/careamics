"""Convenience functions to create HDN configurations.

SCAFFOLD — the default values below are placeholders marked ``TODO(defaults)`` and
are meant to be reviewed together before this is considered final.

Design (agreed):
- Two likelihood pathways, selected automatically by the presence of a noise model.
  ``predict_logvar`` is *derived* here (``noise_model is None``) and never exposed to
  the user, matching the "auto, strict" coupling enforced by ``HDNModule``:
    * noise model present -> noise model likelihood, ``predict_logvar=False`` (HDN);
    * no noise model      -> Gaussian, learned variance, ``predict_logvar=True``
      (DivNoising).
- A minimal ``create_hdn_config`` for regular users delegating to a full
  ``create_advanced_hdn_config`` for experts (same pattern as ``create_n2v_config`` /
  ``create_advanced_n2v_config``).

TODO(scope): decide the return type. N2V returns a full ``N2VConfiguration``
(algorithm + data + training). The base ``Configuration`` is currently constrained to
UNet algorithms (its ``TypeVar`` and validators call UNet-only model methods), so a
proper ``HDNConfiguration(Configuration)`` bundling data + training is a follow-up.
For now these factories return only the ``HDNAlgorithm``; the data config is built
separately with ``create_ng_data_configuration``.
"""

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
    # common optional parameters
    mmse_count: int = 20,
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
    mmse_count : int, default=20
        Number of stochastic samples averaged into the MMSE estimate at prediction time.

    Returns
    -------
    HDNAlgorithm
        Configuration for training HDN.
    """
    return create_advanced_hdn_config(
        patch_size=patch_size,
        noise_model=noise_model,
        mmse_count=mmse_count,
    )


def create_advanced_hdn_config(
    # mandatory parameters
    patch_size: Sequence[int],
    # optional noise model
    noise_model: MultiChannelNMConfig | None = None,
    # prediction
    mmse_count: int = 20,
    output_channels: int = 1,
    # --- architecture (LVAE) -------------------------------------------------
    z_dims: Sequence[int] = (128, 128, 128, 128),  # TODO: defaults
    encoder_n_filters: int = 32,
    decoder_n_filters: int = 32,
    encoder_dropout: float = 0.1,
    decoder_dropout: float = 0.1,
    nonlinearity: NonLinearity = "ELU",
    # --- posterior-collapse / loss knobs -------------------------------------
    reconstruction_weight: float = 1.0,
    kl_weight: float = 1.0,
    logvar_lowerbound: float | None = -5.0,  # TODO(defaults): Gaussian-path var floor
    # --- optimization --------------------------------------------------------
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adamax",
    optimizer_params: dict[str, Any] | None = None,
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
    mmse_count : int, default=1
        Number of MMSE samples at prediction time.
    output_channels : int, default=1
        Number of target channels (HDN uses 1).
    z_dims : Sequence[int], default=(128, 128, 128, 128)
        Latent dimension per hierarchy level; its length sets the number of LVAE layers.
    encoder_n_filters, decoder_n_filters : int, default=64
        Convolution width for encoder/decoder.
    encoder_dropout, decoder_dropout : float, default=0.1
        Dropout rates.
    nonlinearity : {"None","Sigmoid","Softmax","Tanh","ReLU","LeakyReLU","ELU"}
        Activation function.
    reconstruction_weight : float, default=1.0
        Weight of the reconstruction term.
    kl_weight : float, default=1.0
        Weight of the KL term (beta). Annealing is not used.
    logvar_lowerbound : float or None, default=-5.0
        Lower bound on the predicted log-variance (Gaussian pathway only); guards
        against variance-explosion posterior collapse. Ignored on the noise model path.
    optimizer : {"Adam","Adamax","SGD"}, default="Adamax"
        Optimizer name.
    optimizer_params : dict or None, default=None
        Optimizer parameters (e.g. ``{"lr": ...}``).
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
        encoder_n_filters=encoder_n_filters,
        decoder_n_filters=decoder_n_filters,
        encoder_conv_strides=conv_strides,
        decoder_conv_strides=conv_strides,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        nonlinearity=nonlinearity,
        predict_logvar=predict_logvar,
    )

    return HDNAlgorithm(
        algorithm="hdn",
        loss=loss,
        model=model,
        noise_model=noise_model,
        mmse_count=mmse_count,
        is_supervised=is_supervised,
        optimizer=OptimizerConfig(
            name=optimizer,
            parameters=optimizer_params or {},
        ),
        lr_scheduler=LrSchedulerConfig(
            name=lr_scheduler,
            parameters=lr_scheduler_params or {},
        ),
    )
