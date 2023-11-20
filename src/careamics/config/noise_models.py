from __future__ import annotations

from enum import Enum
from typing import Dict, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class NoiseModelType(str, Enum):
    """
    Available noise models.

    Currently supported noise models:

        - hist: Histogram noise model.
        - gmm: Gaussian mixture model noise model.
    """

    HIST = "hist"
    GMM = "gmm"

    @classmethod
    def validate_noise_model_type(
        cls, noise_model: Union[str, NoiseModel], parameters: dict
    ) -> None:
        """_summary_.

        Parameters
        ----------
        noise_model : Union[str, NoiseModel]
            _description_
        parameters : dict
            _description_

        Returns
        -------
        BaseModel
            _description_
        """
        if noise_model == NoiseModelType.HIST.value:
            HistogramNoiseModel(**parameters)
        elif noise_model == NoiseModelType.GMM.value:
            GaussianMixtureNoiseModel(**parameters)


class NoiseModel(BaseModel):
    """_summary_.

    Parameters
    ----------
    BaseModel : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    model_config = ConfigDict(
        use_enum_values=True,
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    model_type: NoiseModelType
    parameters: Dict = {}

    @field_validator("parameters")
    def validate_parameters(cls, parameters: Dict, info: ValidationInfo) -> Dict:
        """_summary_.

        Parameters
        ----------
        parameters : Dict
            _description_

        Returns
        -------
        Dict
            _description_
        """
        if "model_type" not in info.data:
            raise ValueError("Noise model type is missing.")

        noise_model_type = info.data["model_type"]

        NoiseModelType.validate_noise_model_type(noise_model_type, parameters)

        return parameters


class HistogramNoiseModel(BaseModel):
    """
    Histogram noise model.

    Attributes
    ----------
    min_value : float
        Minimum value in the input.
    max_value : float
        Maximum value in the input.
    bins : int
        Number of bins of the histogram.
    """

    min_value: float = Field(default=350.0, ge=0.0, le=65535.0)
    max_value: float = Field(default=6500.0, ge=0.0, le=65535.0)
    bins: int = Field(default=256, ge=1)


class GaussianMixtureNoiseModel(BaseModel):
    """
    Gaussian mixture model noise model.

    Attributes
    ----------
    min_signal : float
    Minimum signal intensity expected in the image.
    max_signal : float
    Maximum signal intensity expected in the image.
    weight : array
    A [3*n_gaussian, n_coeff] sized array containing the values of the weights
    describing the noise model.
    Each gaussian contributes three parameters (mean, standard deviation and weight),
    hence the number of rows in `weight` are 3*n_gaussian.
    If `weight = None`, the weight array is initialized using the `min_signal` and
    `max_signal` parameters.
    n_gaussian: int
    Number of gaussians.
    n_coeff: int
    Number of coefficients to describe the functional relationship between gaussian
    parameters and the signal.
    2 implies a linear relationship, 3 implies a quadratic relationship and so on.
    device: device
    GPU device.
    min_sigma: int
    """

    num_components: int = Field(default=3, ge=1)
    min_value: float = Field(default=350.0, ge=0.0, le=65535.0)
    max_value: float = Field(default=6500.0, ge=0.0, le=65535.0)
    n_gaussian: int = Field(default=3, ge=1)
    n_coeff: int = Field(default=2, ge=1)
    min_sigma: int = Field(default=50, ge=1)
