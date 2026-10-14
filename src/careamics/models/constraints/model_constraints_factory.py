"""Model constraints factory."""

from careamics.config.architectures import COSDDConfig, LVAEConfig, UNetConfig

from .cosdd_constraints import COSDDConstraints
from .lvae_constraints import LVAEConstraints
from .model_constraints import ModelConstraints
from .unet_constraints import UNetConstraints


def get_model_constraints(
    model_config: UNetConfig | LVAEConfig | COSDDConfig,
) -> ModelConstraints:
    """Get the model constraints for the given model configuration.

    Parameters
    ----------
    model_config : UNetConfig, LVAEConfig or COSDDConfig
        The model configuration.

    Returns
    -------
    ModelConstraints
        The model constraints for the given model configuration.

    Raises
    ------
    ValueError
        If the model type is not supported.
    """
    match model_config.architecture:
        case "UNet":
            return UNetConstraints(model_config)
        case "LVAE":
            return LVAEConstraints(model_config)
        case "COSDD":
            return COSDDConstraints(model_config)
        case _:
            raise ValueError(
                f"Model type {model_config.architecture} is not supported."
            )
