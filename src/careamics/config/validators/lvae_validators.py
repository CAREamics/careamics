"""LVAE-based algorithm validators."""

from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
from careamics.config.support import SupportedLoss


def loss_type_is_hdn(loss: LVAELossConfig) -> LVAELossConfig:
    """Validate that the loss type is `hdn`.

    Parameters
    ----------
    loss : LVAELossConfig
        Loss configuration.

    Returns
    -------
    LVAELossConfig
        The validated loss configuration.

    Raises
    ------
    ValueError
        If the loss type is not `hdn`.
    """
    if loss.loss_type != SupportedLoss.HDN:
        raise ValueError("HDN only supports loss `hdn`.")
    return loss


def loss_type_is_microsplit(loss: LVAELossConfig) -> LVAELossConfig:
    """Validate that the loss type is `microsplit`.

    Parameters
    ----------
    loss : LVAELossConfig
        Loss configuration.

    Returns
    -------
    LVAELossConfig
        The validated loss configuration.

    Raises
    ------
    ValueError
        If the loss type is not `microsplit`.
    """
    if loss.loss_type != SupportedLoss.MICROSPLIT:
        raise ValueError("Algorithm `microsplit` only supports loss `microsplit`.")
    return loss


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
