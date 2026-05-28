"""Pydantic-config factories for MicroSplit inference scripts.

Holds the pkl-driven factories that build the configs feeding a
`MicroSplitModule`:

- :func:`pkl_load` — load a legacy training-config dump.
- :func:`get_predict_config` — `MicroSplitDataConfig` (data side; picks tiled vs
  sliding-window patching based on whether `stride` is given).
- :func:`get_model_config` — `LVAEConfig` (architecture).
- :func:`get_loss_config` — `LVAELossConfig`. Loss is not actually used at
  inference time, but :class:`VAEBasedAlgorithm` requires it to validate. We
  hardcode the loss type to ``"denoisplit_musplit"`` (the value used by every
  experiment we predict on); kl-type is read from the pkl for completeness.
- :func:`get_likelihood_config` — `GaussianLikelihoodConfig`. Like the loss, not
  consumed at predict time, but supplied so :class:`VAEBasedAlgorithm` matches
  what the checkpoint was trained with.
- :func:`create_algorithm_config` — `VAEBasedAlgorithm` assembled from the
  three above.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from careamics.config.algorithms.vae_algorithm_config import VAEBasedAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.data.data_config import (
    SlidingWindowTiledPatchingConfig,
    TiledPatchingConfig,
)
from careamics.config.data.microsplit_data_config import MicroSplitDataConfig
from careamics.config.data.normalization_config import MeanStdConfig
from careamics.config.losses.loss_config import KLLossConfig, LVAELossConfig
from careamics.config.noise_model.likelihood_config import GaussianLikelihoodConfig


# Legacy `nonlin` strings are lowercase; `LVAEConfig.nonlinearity` is a
# capitalised Literal.
_NONLIN_MAP: dict[str, str] = {
    "elu": "ELU",
    "relu": "ReLU",
    "leakyrelu": "LeakyReLU",
    "leaky_relu": "LeakyReLU",
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "tanh": "Tanh",
    "none": "None",
}


def pkl_load(path: str | Path) -> dict:
    """Load a legacy MicroSplit training config dump."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_predict_config(
    pkl_data: dict,
    *,
    overlap: list[int],
    stride: list[int] | None = None,
    input_means: list[float],
    input_stds: list[float],
    target_means: list[float],
    target_stds: list[float],
    batch_size: int = 1,
) -> MicroSplitDataConfig:
    """Build a prediction `MicroSplitDataConfig` from a legacy training-config dump.

    Parameters
    ----------
    pkl_data : dict
        Legacy MicroSplit training config (loaded with :func:`pkl_load`).
        Must carry `image_size`, `multiscale_lowres_count`, `padding_mode`, and
        optionally `mode_3D` / `depth3D` for 3D experiments.
    overlap : list of int
        Overlap per spatial dimension (length 2 for 2D, length 3 for 3D).
    stride : list of int or None, default=None
        If `None`, a classical `TiledPatchingConfig` is used (inner tiling, no
        overlap on the kept region). If provided, a
        `SlidingWindowTiledPatchingConfig` is used (dense overlap averaging — see
        :class:`careamics.dataset.patching.SlidingWindowTiledPatching`).
    input_means, input_stds : list of float
        Per-input-channel normalization stats.
    target_means, target_stds : list of float
        Per-target-channel normalization stats (used to denormalize predictions).
    batch_size : int, default=1
        Prediction batch size.

    Returns
    -------
    MicroSplitDataConfig
        Configuration ready to be passed to
        :func:`careamics.dataset.factory.create_microsplit_pred_dataset`.
    """
    is_3d = pkl_data.get("mode_3D", False)
    axes = "CZYX" if is_3d else "CYX"
    img = pkl_data["image_size"]
    patch_size = [pkl_data["depth3D"], img, img] if is_3d else [img, img]

    patching = (
        SlidingWindowTiledPatchingConfig(
            patch_size=patch_size, overlaps=overlap, stride=stride
        )
        if stride is not None
        else TiledPatchingConfig(
            patch_size=patch_size, overlaps=overlap
        )
    )

    return MicroSplitDataConfig(
        mode="predicting",
        data_type="tiff",
        axes=axes,
        patching=patching,
        normalization=MeanStdConfig(
            input_means=input_means,
            input_stds=input_stds,
            target_means=target_means,
            target_stds=target_stds,
        ),
        multiscale_count=pkl_data["multiscale_lowres_count"],
        padding_mode=pkl_data["padding_mode"],
        batch_size=batch_size,
        augmentations=[],  # predict mode: no augs
    )


def get_model_config(pkl_data: dict) -> LVAEConfig:
    """Build an `LVAEConfig` from a legacy MicroSplit training-config dump.

    Architecture fields come from `pkl_data["model"]`; spatial / multiscale
    fields come from `pkl_data["data"]`. Output channels are resolved by trying,
    in order: `model.num_targets`, `len(data.target_idx_list)`,
    `data.num_channels` — covers both paired and multiplexed experiments.
    """
    data = pkl_data["data"]
    model = pkl_data["model"]

    is_3d = bool(data.get("mode_3D", False))
    img = int(data["image_size"])
    if is_3d:
        input_shape = (int(data["depth3D"]), img, img)
        strides = [1, 2, 2]
    else:
        input_shape = (img, img)
        strides = [2, 2]

    nonlin_raw = str(model.get("nonlin", "ELU"))
    nonlinearity = _NONLIN_MAP.get(nonlin_raw.lower(), nonlin_raw)

    return LVAEConfig(
        architecture="LVAE",
        input_shape=input_shape,
        encoder_conv_strides=strides,
        decoder_conv_strides=strides,
        multiscale_count=int(data["multiscale_lowres_count"]),
        z_dims=list(model.get("z_dims", [128, 128, 128, 128])),
        output_channels=_resolve_output_channels(pkl_data),
        nonlinearity=nonlinearity,
        predict_logvar=model.get("predict_logvar"),
        analytical_kl=bool(model.get("analytical_kl", False)),
    )


def get_loss_config(pkl_data: dict) -> LVAELossConfig:
    """Build an `LVAELossConfig` from a legacy training-config dump.

    Loss type is hardcoded to ``"denoisplit_musplit"`` — every pre-trained
    MicroSplit checkpoint we predict on was trained with it, and the loss isn't
    consumed at predict time anyway; this just satisfies
    :class:`VAEBasedAlgorithm`'s cross-validator. The kl-type is read from the
    pkl in case anything downstream needs it.
    """
    loss = pkl_data["loss"]
    kl_type = "kl_restricted" if bool(loss.get("restricted_kl", False)) else "kl"
    return LVAELossConfig(
        loss_type="denoisplit_musplit",
        kl_params=KLLossConfig(loss_type=kl_type),
    )


def get_likelihood_config(pkl_data: dict) -> GaussianLikelihoodConfig:
    """Build a `GaussianLikelihoodConfig` from a legacy training-config dump.

    Read from the `model` section of the pkl. Not consumed at predict time
    (chunking into mean/logvar is driven by :attr:`LVAEConfig.predict_logvar`),
    but supplied so :class:`VAEBasedAlgorithm` matches what the checkpoint was
    trained with.

    TODO (v2): build `MultiChannelNMConfig` + `NMLikelihoodConfig` once we
    decide how to stage / rewrite the noise-model paths from the legacy pkl
    (they currently point at the original training host).
    """
    model = pkl_data["model"]
    return GaussianLikelihoodConfig(
        predict_logvar=model.get("predict_logvar"),
        logvar_lowerbound=float(model.get("logvar_lowerbound", -5.0)),
    )


def create_algorithm_config(pkl_data: dict) -> VAEBasedAlgorithm:
    """Assemble a `VAEBasedAlgorithm` from a legacy training-config dump.

    Composes :func:`get_model_config`, :func:`get_loss_config` and
    :func:`get_likelihood_config`. Algorithm is always ``"microsplit"``
    (CAREamics's umbrella label for muSplit / denoiSplit / denoiSplit-muSplit
    training).
    """
    return VAEBasedAlgorithm(
        algorithm="microsplit",
        model=get_model_config(pkl_data),
        loss=get_loss_config(pkl_data),
        gaussian_likelihood=get_likelihood_config(pkl_data),
    )


def _resolve_output_channels(pkl_data: dict) -> int:
    """Resolve the number of output (target) channels from a legacy pkl dump.

    Tries, in order: `model.num_targets`, `len(data.target_idx_list)`,
    `data.num_channels`. Covers both paired (HT_LIF24) and multiplexed
    (CARE3D / PaviaATN) experiments.
    """
    model = pkl_data.get("model", {})
    if model.get("num_targets") is not None:
        return int(model["num_targets"])
    data = pkl_data.get("data", {})
    target_idx_list = data.get("target_idx_list")
    if target_idx_list is not None:
        return len(target_idx_list)
    if data.get("num_channels") is not None:
        return int(data["num_channels"])
    raise KeyError(
        "Could not resolve output channels from pkl: none of "
        "`model.num_targets`, `data.target_idx_list`, `data.num_channels` "
        "is present."
    )
