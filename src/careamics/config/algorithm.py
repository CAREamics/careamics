"""Algorithm configuration."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .config_filter import remove_default_optionals
from .models import Model
from .noise_models import NoiseModel

# python 3.11: https://docs.python.org/3/library/enum.html


class AlgorithmType(str, Enum):
    """
    Available types of algorithms.

    Currently supported algorithms:
        - CARE: CARE. https://www.nature.com/articles/s41592-018-0216-7
        - n2v: Noise2Void. https://arxiv.org/abs/1811.10980
        - n2n: Noise2Noise. https://arxiv.org/abs/1803.04189
        - pn2v: Probabilistic Noise2Void. https://arxiv.org/abs/1906.00651
        - hdn: Hierarchical DivNoising. https://arxiv.org/abs/2104.01374
    """

    CARE = "care"
    N2V = "n2v"
    N2N = "n2n"
    PN2V = "pn2v"
    HDN = "hdn"
    CUSTOM = "custom"


class Loss(str, Enum):
    """
    Available loss functions.

    Currently supported losses:

        - n2v: Noise2Void loss.
        - n2n: Noise2Noise loss.
        - pn2v: Probabilistic Noise2Void loss.
        - hdn: Hierarchical DivNoising loss.
    """

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    PN2V = "pn2v"
    HDN = "hdn"
    CUSTOM = "custom"


class MaskingStrategy(str, Enum):
    """
    Available masking strategy.

    Currently supported strategies:

    - default: default masking strategy of Noise2Void (uniform sampling of neighbors).
    - median: median masking strategy of N2V2.
    """

    NONE = "none"
    DEFAULT = "default"
    MEDIAN = "median"


class ModelParameters(BaseModel):
    """
    Deep-learning model parameters.

    The number of filters (base) must be even and minimum 8.

    Attributes
    ----------
    depth : int
        Depth of the model, between 1 and 10 (default 2).
    num_channels_init : int
        Number of filters of the first level of the network, should be even
        and minimum 8 (default 96).
    """

    model_config = ConfigDict(validate_assignment=True)

    depth: int = Field(default=2, ge=1, le=10)
    num_channels_init: int = Field(default=32, ge=8)

    # TODO revisit the constraints on num_channels_init
    @field_validator("num_channels_init")
    def even(cls, num_channels: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        num_channels : int
            Number of channels.

        Returns
        -------
        int
            Validated number of channels.

        Raises
        ------
        ValueError
            If the number of channels is odd.
        """
        # if odd
        if num_channels % 2 != 0:
            raise ValueError(
                f"Number of channels (init) must be even (got {num_channels})."
            )

        return num_channels


class Algorithm(BaseModel):
    """
    Algorithm configuration.

    The minimum algorithm configuration is composed of the following fields:
        - loss:
            Loss to use, currently only supports n2v.
        - model:
            Model to use, currently only supports UNet.
        - is_3D:
            Whether to use a 3D model or not, this should be coherent with the
            data configuration (axes).

    Other optional fields are:
        - masking_strategy:
            Masking strategy to use, currently only supports default masking.
        - masked_pixel_percentage:
            Percentage of pixels to be masked in each patch.
        - roi_size:
            Size of the region of interest to use in the masking algorithm.
        - model_parameters:
            Model parameters, see ModelParameters for more details.

    Attributes
    ----------
    loss : List[Losses]
        List of losses to use, currently only supports n2v.
    model : Models
        Model to use, currently only supports UNet.
    is_3D : bool
        Whether to use a 3D model or not.
    masking_strategy : MaskingStrategies
        Masking strategy to use, currently only supports default masking.
    masked_pixel_percentage : float
        Percentage of pixels to be masked in each patch.
    roi_size : int
        Size of the region of interest used in the masking scheme.
    model_parameters : ModelParameters
        Model parameters, see ModelParameters for more details.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    # Mandatory fields
    algorithm_type: AlgorithmType
    loss: Loss
    model: Model
    is_3D: bool

    # Optional fields, define a default value
    noise_model: Optional[NoiseModel] = None
    masking_strategy: MaskingStrategy = MaskingStrategy.NONE
    masked_pixel_percentage: float = Field(default=None, ge=0.1, le=20)
    roi_size: int = Field(default=None, ge=3, le=21)
    model_parameters: ModelParameters = ModelParameters()

    def get_conv_dim(self) -> int:
        """
        Get the convolution layers dimension (2D or 3D).

        Returns
        -------
        int
            Dimension (2 or 3).
        """
        return 3 if self.is_3D else 2

    def get_noise_model(self, noise_model: Dict, info: ValidationInfo) -> Dict:
        """
        Validate noise model.

        Returns
        -------
        Dict
            Validated noise model.
        """
        # TODO validate noise model
        if "noise_model_type" not in info.data:
            raise ValueError("Noise model is missing.")

        noise_model_type = info.data["noise_model_type"]

        if noise_model is not None:
            _ = NoiseModel.get_noise_model(noise_model_type, noise_model)

        return noise_model

    @model_validator(mode="after")
    def algorithm_loss_cross_validation(cls, data: Algorithm) -> Algorithm:
        """Validate loss.

        Returns
        -------
        Loss
            Validated loss.

        Raises
        ------
        ValueError
            If the loss is not supported or inconsistent with the noise model.
        """
        if data.algorithm_type in [
            AlgorithmType.CARE,
            AlgorithmType.N2N,
        ] and data.loss not in [
            Loss.MSE,
            Loss.MAE,
            Loss.CUSTOM,
        ]:
            raise ValueError(
                f"Algorithm {data.type} does not support {data.loss} loss."
            )

        if (
            data.algorithm_type in [AlgorithmType.CARE, AlgorithmType.N2N]
            and data.noise_model is not None
        ):
            raise ValueError(f"Algorithm {data.type} does not support a noise model.")

        if (
            data.loss == Loss.PN2V or data.loss == Loss.HDN
        ) and data.noise_model is None:
            raise ValueError(f"Loss {data.loss} requires a noise model.")

        if data.loss in [Loss.N2V, Loss.MAE, Loss.MSE] and data.noise_model is not None:
            raise ValueError(f"Loss {data.loss} does not support a noise model.")

        if data.loss == Loss.N2V and data.masking_strategy != "none":
            raise ValueError(
                f"Algorithm {data.loss} does not require masking strategy "
                f"{data.masking_strategy}."
            )

        return data

    @model_validator(mode="after")
    def masking_strategy_validation(cls, data: Algorithm) -> Algorithm:
        """Validate masking strategy and its parameters.

        Returns
        -------
        MaskingStrategy
            Validated masking strategy.

        Raises
        ------
        ValueError
            If the masking strategy is not supported.
        """
        if data.masking_strategy not in [
            MaskingStrategy.NONE,
            MaskingStrategy.DEFAULT,
            MaskingStrategy.MEDIAN,
        ]:
            raise ValueError(
                f"Masking strategy {data.masking_strategy} is not supported."
            )
        if (
            data.masking_strategy
            in [
                MaskingStrategy.DEFAULT,
                MaskingStrategy.MEDIAN,
            ]
            and data.masked_pixel_percentage is None
            or data.roi_size is None
        ):
            raise ValueError(
                f"Masking strategy {data.masking_strategy} requires a masked pixel "
                f"percentage and a ROI size. Please refer to the documentation"
            )  # TODO add link to documentation
        return data

    @field_validator("roi_size")
    def even(cls, roi_size: int) -> int:
        """
        Validate that roi_size is odd.

        Parameters
        ----------
        roi_size : int
            Size of the region of interest in the masking scheme.

        Returns
        -------
        int
            Validated size of the region of interest.

        Raises
        ------
        ValueError
            If the size of the region of interest is even.
        """
        # if even
        if roi_size % 2 == 0:
            raise ValueError(f"ROI size must be odd (got {roi_size}).")

        return roi_size

    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """
        Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value.
            - remove optional values if they have the default value.

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.
        *args : List
            Positional arguments, unused.
        **kwargs : Dict
            Keyword arguments, unused.

        Returns
        -------
        Dict
            Dictionary representation of the model.
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals is True:
            # remove optional arguments if they are default
            defaults = {
                "masking_strategy": MaskingStrategy.DEFAULT.value,
                "masked_pixel_percentage": 0.2,
                "roi_size": 11,
                "model_parameters": ModelParameters().model_dump(exclude_none=True),
            }

            remove_default_optionals(dictionary, defaults)

        return dictionary
