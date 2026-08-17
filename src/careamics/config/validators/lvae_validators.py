"""LVAE-based algorithm validators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import LVAELossConfig, MicroSplitLossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig


def model_without_multiscale(model: LVAEConfig) -> LVAEConfig:
    """Validate that the LVAE model does not use multiscale inputs.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.

    Returns
    -------
    LVAEConfig
        The validated model configuration.

    Raises
    ------
    ValueError
        If the model has a multiscale count greater than 1.
    """
    if model.multiscale_count > 1:
        raise ValueError("Algorithm `hdn` does not support multiscale models.")
    return model


def model_with_single_output_channel(model: LVAEConfig) -> LVAEConfig:
    """Validate that the LVAE model has a single output channel.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.

    Returns
    -------
    LVAEConfig
        The validated model configuration.

    Raises
    ------
    ValueError
        If the model has more than one output channel.
    """
    if model.output_channels != 1:
        raise ValueError(
            f"Number of output channels ({model.output_channels}) must be 1 "
            "for algorithm `hdn`."
        )
    return model


if TYPE_CHECKING:
    from careamics.config.data import DataConfig, MicroSplitDataConfig


def predict_logvar_consistent(model: LVAEConfig, loss: LVAELossConfig) -> None:
    """Validate the consistency of `predict_logvar` between model and loss.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.
    loss : LVAELossConfig
        Loss configuration.

    Raises
    ------
    ValueError
        If the model and loss `predict_logvar` do not match.
    """
    if model.predict_logvar != loss.predict_logvar:
        raise ValueError(
            f"Model `predict_logvar` ({model.predict_logvar}) "
            f"must match loss `predict_logvar` ({loss.predict_logvar})."
        )


def noise_models_match_output_channels(
    model: LVAEConfig, noise_model: MultiChannelNMConfig | None
) -> None:
    """Validate that the number of noise models matches the output channels.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.
    noise_model : MultiChannelNMConfig or None
        Noise model configuration.

    Raises
    ------
    ValueError
        If the number of output channels does not match the number of noise models.
    """
    if noise_model is not None and model.output_channels != len(
        noise_model.noise_models
    ):
        raise ValueError(
            f"Number of output channels ({model.output_channels}) must match "
            f"the number of noise models ({len(noise_model.noise_models)})."
        )


def lvae_conv_strides_valid(model: LVAEConfig) -> LVAEConfig:
    """Validate the encoder and decoder convolutional strides.

    The strides must be 2D or 3D, match the input dimensionality, and the decoder
    cannot be 3D when the encoder is 2D.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.

    Returns
    -------
    LVAEConfig
        The validated model configuration.

    Raises
    ------
    ValueError
        If the strides are inconsistent with each other or the input shape.
    """
    if len(model.encoder_conv_strides) < 2 or len(model.encoder_conv_strides) > 3:
        raise ValueError(
            f"Strides must be 2 or 3 (got {len(model.encoder_conv_strides)})."
        )
    if len(model.decoder_conv_strides) < 2 or len(model.decoder_conv_strides) > 3:
        raise ValueError(
            f"Strides must be 2 or 3 (got {len(model.decoder_conv_strides)})."
        )
    if len(model.input_shape) != len(model.encoder_conv_strides):
        raise ValueError(
            f"Input dimensions must be equal to the number of encoder conv strides "
            f"(got {len(model.input_shape)} and {len(model.encoder_conv_strides)})."
        )
    if len(model.encoder_conv_strides) < len(model.decoder_conv_strides):
        raise ValueError(
            f"Decoder can't be 3D when encoder is 2D (got "
            f"{len(model.encoder_conv_strides)} and {len(model.decoder_conv_strides)})."
        )
    if any(s < 1 for s in model.encoder_conv_strides) or any(
        s < 1 for s in model.decoder_conv_strides
    ):
        raise ValueError(
            f"All strides must be greater or equal to 1 (got "
            f"{model.encoder_conv_strides} and {model.decoder_conv_strides})."
        )
    return model


def lvae_multiscale_count_valid(model: LVAEConfig) -> LVAEConfig:
    """Validate the multiscale (lateral-context) count against the hierarchy depth.

    The count must be 1 (lateral context off) or at most ``len(z_dims) + 1``.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.

    Returns
    -------
    LVAEConfig
        The validated model configuration.

    Raises
    ------
    ValueError
        If the multiscale count is out of range.
    """
    if model.multiscale_count < 1 or model.multiscale_count > len(model.z_dims) + 1:
        raise ValueError(
            f"Multiscale count must be 1 for LC off or less or equal to the number "
            f"of Z dims + 1 (got {model.multiscale_count} and {len(model.z_dims)})."
        )
    return model


def lvae_spatial_shape_valid(model: LVAEConfig) -> LVAEConfig:
    """Validate that the input shape is compatible with the downsampling.

    Each spatial dimension is downsampled once per hierarchy level (there are
    ``len(z_dims)`` levels) by its convolutional stride, so it must be divisible by
    ``stride ** len(z_dims)``. Dimensions with a stride of 1 (e.g. Z in a 2.5D model)
    are unconstrained. Invalid shapes otherwise raise obscure errors deep in the model
    forward pass.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.

    Returns
    -------
    LVAEConfig
        The validated model configuration.

    Raises
    ------
    ValueError
        If a downsampled input dimension is not divisible by the cumulative
        downsampling factor.
    """
    n_levels = len(model.z_dims)
    for dim, stride in zip(model.input_shape, model.encoder_conv_strides, strict=True):
        factor = stride**n_levels
        if dim % factor != 0:
            raise ValueError(
                f"Input shape {tuple(model.input_shape)} is not compatible with the "
                f"encoder strides {list(model.encoder_conv_strides)} and {n_levels} "
                f"hierarchy levels: dimension {dim} must be divisible by "
                f"stride ** n_levels = {stride} ** {n_levels} = {factor}."
            )
    return model


def lvae_depth_valid(model: LVAEConfig) -> LVAEConfig:
    """Validate the depth of a 3D encoder feeding a 2D decoder.

    When a 3D encoder is paired with a 2D decoder (the depth is "squished" away before
    decoding), the model requires an odd depth. This mirrors the runtime assertion in
    the LVAE model, surfaced here at configuration time.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.

    Returns
    -------
    LVAEConfig
        The validated model configuration.

    Raises
    ------
    ValueError
        If the input depth is even for a 3D-encoder/2D-decoder model.
    """
    if len(model.input_shape) == 3 and len(model.decoder_conv_strides) == 2:
        if model.input_shape[0] % 2 == 0:
            raise ValueError(
                f"A 3D encoder with a 2D decoder requires an odd depth, got depth "
                f"{model.input_shape[0]} in input shape {tuple(model.input_shape)}."
            )
    return model


def at_least_one_likelihood(loss: MicroSplitLossConfig) -> MicroSplitLossConfig:
    """Validate that at least one likelihood term is active.

    The reconstruction likelihood is a weighted sum of the Gaussian (muSplit) and
    noise-model (denoiSplit) terms; if both weights are 0 there is no data term.

    Parameters
    ----------
    loss : MicroSplitLossConfig
        Loss configuration.

    Returns
    -------
    MicroSplitLossConfig
        The validated loss configuration.

    Raises
    ------
    ValueError
        If both `gaussian_likelihood_weight` and `noise_model_likelihood_weight` are 0.
    """
    if loss.gaussian_likelihood_weight == 0 and loss.noise_model_likelihood_weight == 0:
        raise ValueError(
            "At least one of `gaussian_likelihood_weight` or "
            "`noise_model_likelihood_weight` must be greater than 0; both are 0 so no "
            "likelihood term is active."
        )
    return loss


def predict_logvar_required_for_musplit(
    loss: MicroSplitLossConfig,
) -> MicroSplitLossConfig:
    """Validate that `predict_logvar` is enabled when the muSplit likelihood is active.

    The Gaussian (muSplit) likelihood consumes the pixelwise predicted log-variance, so
    `predict_logvar` must be ``True`` whenever `gaussian_likelihood_weight` > 0. This
    mirrors the runtime check in the LVAE loss, surfaced here at configuration time.
    Pure denoiSplit (`gaussian_likelihood_weight` == 0) may use either value.

    Parameters
    ----------
    loss : MicroSplitLossConfig
        Loss configuration.

    Returns
    -------
    MicroSplitLossConfig
        The validated loss configuration.

    Raises
    ------
    ValueError
        If `gaussian_likelihood_weight` > 0 but `predict_logvar` is False.
    """
    if loss.gaussian_likelihood_weight > 0 and not loss.predict_logvar:
        raise ValueError(
            "`predict_logvar` must be True when the muSplit Gaussian likelihood is "
            f"active (`gaussian_likelihood_weight` > 0, got "
            f"{loss.gaussian_likelihood_weight})."
        )
    return loss


def multiscale_counts_match(model: LVAEConfig, data: MicroSplitDataConfig) -> None:
    """Validate that the model and data agree on the lateral-context count.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.
    data : MicroSplitDataConfig
        Data configuration.

    Raises
    ------
    ValueError
        If the model and data `multiscale_count` do not match.
    """
    if model.multiscale_count != data.multiscale_count:
        raise ValueError(
            f"Model `multiscale_count` ({model.multiscale_count}) must match data "
            f"`multiscale_count` ({data.multiscale_count})."
        )


def input_shape_matches_patch_size(model: LVAEConfig, data: DataConfig) -> None:
    """Validate that the model input shape matches the data patch size.

    Skipped for whole-image patching, which has no patch size.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.
    data : DataConfig
        Data configuration.

    Raises
    ------
    ValueError
        If the model `input_shape` does not match the data `patch_size`.
    """
    patching = data.patching
    if not hasattr(patching, "patch_size"):
        return
    patch_size = tuple(patching.patch_size)
    if tuple(model.input_shape) != patch_size:
        raise ValueError(
            f"Model `input_shape` {tuple(model.input_shape)} must match the data "
            f"`patch_size` {patch_size}."
        )


def alpha_ranges_match_output_channels(
    model: LVAEConfig, data: MicroSplitDataConfig
) -> None:
    """Validate that there is one channel-mixing range per output channel.

    Parameters
    ----------
    model : LVAEConfig
        Model configuration.
    data : MicroSplitDataConfig
        Data configuration.

    Raises
    ------
    ValueError
        If `alpha_ranges` is set and its length does not match `output_channels`.
    """
    alpha_ranges = data.alpha_ranges
    if alpha_ranges is not None and len(alpha_ranges) != model.output_channels:
        raise ValueError(
            f"Number of `alpha_ranges` ({len(alpha_ranges)}) must match the number of "
            f"output channels ({model.output_channels})."
        )


def normalization_supported(data: MicroSplitDataConfig) -> MicroSplitDataConfig:
    """Validate that the normalization is compatible with MicroSplit.

    MicroSplit assumes standardized inputs (the LVAE runs on normalized data and the
    noise-model likelihood is defined against mean/std statistics), so disabling
    normalization is not supported.

    Parameters
    ----------
    data : MicroSplitDataConfig
        Data configuration.

    Returns
    -------
    MicroSplitDataConfig
        The validated data configuration.

    Raises
    ------
    ValueError
        If normalization is disabled (`normalization='none'`).
    """
    if data.normalization.name == "none":
        raise ValueError(
            "MicroSplit requires normalized inputs; `normalization='none'` is not "
            "supported. Use 'mean_std' (recommended)."
        )
    return data


def alpha_ranges_wellformed(
    alpha_ranges: Sequence[tuple[float, float]] | None,
) -> Sequence[tuple[float, float]] | None:
    """Validate that each channel-mixing range is a well-formed ``(low, high)`` pair.

    Parameters
    ----------
    alpha_ranges : sequence of tuple of float, or None
        Ranges used to sample channel mixing weights.

    Returns
    -------
    sequence of tuple of float, or None
        The validated alpha ranges.

    Raises
    ------
    ValueError
        If any range does not satisfy ``0 <= low <= high``.
    """
    if alpha_ranges is not None:
        for alpha_range in alpha_ranges:
            low, high = alpha_range
            if not 0.0 <= low <= high:
                raise ValueError(
                    f"Each alpha range must satisfy 0 <= low <= high, got "
                    f"{tuple(alpha_range)}."
                )
    return alpha_ranges
