"""Algorithm configuration."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
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
    SEGM = "segmentation"


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
    CE = "ce"
    DICE = "dice"
    CUSTOM = "custom"


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
    transforms: Optional[Dict] = None

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
    def algorithm_cross_validation(cls, data: Algorithm) -> Algorithm:
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
                f"Algorithm {data.algorithm_type} does not support"
                f" {data.loss.upper()} loss. Please refer to the documentation"
                # TODO add link to documentation
            )

        if (
            data.algorithm_type in [AlgorithmType.CARE, AlgorithmType.N2N]
            and data.noise_model is not None
        ):
            raise ValueError(
                f"Algorithm {data.algorithm_type} isn't compatible with a noise model."
            )

        if data.algorithm_type in [AlgorithmType.N2V, AlgorithmType.PN2V]:
            if data.transforms is None:
                raise ValueError(
                    f"Algorithm {data.algorithm_type} requires a masking strategy."
                    "Please add ManipulateN2V to transforms."
                )
            else:
                if "ManipulateN2V" not in data.transforms:
                    raise ValueError(
                        f"Algorithm {data.algorithm_type} requires a masking strategy."
                        "Please add ManipulateN2V to transforms."
                    )
        elif "ManipulateN2V" in data.transforms:
            raise ValueError(
                f"Algorithm {data.algorithm_type} doesn't require a masking strategy."
                "Please remove ManipulateN2V from the image or patch_transform."
            )
        if (
            data.loss == Loss.PN2V or data.loss == Loss.HDN
        ) and data.noise_model is None:
            raise ValueError(f"Loss {data.loss.upper()} requires a noise model.")

        if data.loss in [Loss.N2V, Loss.MAE, Loss.MSE] and data.noise_model is not None:
            raise ValueError(
                f"Loss {data.loss.upper()} does not support a noise model."
            )
        if data.loss == Loss.N2V and data.algorithm_type != AlgorithmType.N2V:
            raise ValueError(
                f"Loss {data.loss.upper()} is only supported by "
                f"{AlgorithmType.N2V}."
            )
        # TODO Check conditions and add tests

        return data

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
                "model": {
                    # "architecture": "UNet",
                    "parameters": {"depth": 2, "num_channels_init": 32},
                },
                # TODO don't kmow how to drop nested defaults and don't know why we need this ?!
                "masking_strategy": {
                    # "strategy_type": "default",
                    "parameters": {"masked_pixel_percentage": 0.2, "roi_size": 11},
                },
            }

            remove_default_optionals(dictionary, defaults)

        return dictionary
