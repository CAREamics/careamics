from typing import Type, Union

from ..config.noise_models import NoiseModel, NoiseModelType
from .noise_models import GaussianMixtureNoiseModel, HistogramNoiseModel


def noise_model_factory(
    noise_config: NoiseModel,
) -> Type[Union[HistogramNoiseModel, GaussianMixtureNoiseModel, None]]:
    """Create loss model based on Configuration.

    Parameters
    ----------
    config : Configuration
        Configuration.

    Returns
    -------
    Noise model

    Raises
    ------
    NotImplementedError
        If the noise model is unknown.
    """
    noise_model_type = noise_config.model_type if noise_config else None

    if noise_model_type == NoiseModelType.HIST:
        return HistogramNoiseModel

    elif noise_model_type == NoiseModelType.GMM:
        return GaussianMixtureNoiseModel

    elif noise_model_type is None:
        return None

    else:
        raise NotImplementedError(
            f"Noise model {noise_model_type} is not yet supported."
        )
