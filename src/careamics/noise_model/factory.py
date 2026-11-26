"""Factories for constructing configured noise models."""

from careamics.config.noise_model import MultiChannelNMConfig

from .noise_models import GaussianMixtureNoiseModel, MultiChannelNoiseModel


def multichannel_noise_model_factory(
    model_config: MultiChannelNMConfig | None,
) -> MultiChannelNoiseModel | None:
    """Build a multi-channel noise model from its configuration.

    Parameters
    ----------
    model_config : MultiChannelNMConfig or None
        Multi-channel noise model configuration.

    Returns
    -------
    MultiChannelNoiseModel or None
        A multi-channel noise model, or `None` when `model_config` is `None`.

    Raises
    ------
    NotImplementedError
        If a configured noise model type is not supported.
    """
    if model_config:
        noise_models = []
        for nm in model_config.noise_models:
            if nm.model_type == "GaussianMixtureNoiseModel":
                noise_models.append(GaussianMixtureNoiseModel(nm))
            else:
                raise NotImplementedError(f"Model {nm.model_type} is not implemented")
        return MultiChannelNoiseModel(noise_models)
    return None
