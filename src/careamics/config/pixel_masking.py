from __future__ import annotations

from enum import Enum
from typing import Dict, Union

from pydantic import BaseModel, ConfigDict, model_validator


class MaskingStrategyType(str, Enum):
    """
    Available masking strategy.

    Currently supported strategies:

    - default: default masking strategy of Noise2Void (uniform sampling of neighbors).
    - median: median masking strategy of N2V2.
    """

    NONE = "none"
    DEFAULT = "default"
    MEDIAN = "median"

    @classmethod
    def validate_masking_strategy_type(
        cls, masking_strategy: Union[str, MaskingStrategy], parameters: dict
    ) -> None:
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
        if masking_strategy not in [
            MaskingStrategyType.NONE,
            MaskingStrategyType.DEFAULT,
            MaskingStrategyType.MEDIAN,
        ]:
            raise ValueError(
                f"Incorrect value for asking strategy {masking_strategy}."
                f"Please refer to the documentation"  # TODO add link to documentation
            )
        if masking_strategy in [
            MaskingStrategyType.DEFAULT,
            MaskingStrategyType.MEDIAN,
        ] and (
            parameters["masked_pixel_percentage"] is None
            or parameters["roi_size"] is None
        ):
            raise ValueError(
                f"Masking strategy {masking_strategy} requires a masked pixel "
                f"percentage and a ROI size. Please refer to the documentation"
            )  # TODO add link to documentation


class MaskingStrategy(BaseModel):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    BaseModel : _type_
        _description_
    """

    model_config = ConfigDict(
        use_enum_values=True,
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    strategy_type: MaskingStrategyType
    parameters: Dict = {}

    @model_validator(mode="after")
    def validate_parameters(cls, data: MaskingStrategy) -> MaskingStrategy:
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
        MaskingStrategyType.validate_masking_strategy_type(
            data.strategy_type, data.parameters
        )

        if data.parameters["roi_size"] % 2 == 0:
            raise ValueError(f"ROI size must be odd, got {data.parameters['roi_size']}")
        if data.parameters["roi_size"] < 3 or data.parameters["roi_size"] > 21:
            raise ValueError(
                f"ROI size must be between 3 and 21, got {data.parameters['roi_size']}"
            )
        if data.parameters["masked_pixel_percentage"] < 0.1:
            raise ValueError(
                "Masked pixel percentage must be at least 0.1"
            )
        return data

    #TODO finish modifying this class
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
                    "architecture": "UNet",
                    "parameters": {"depth": 2, "num_channels_init": 32},
                },
                # TODO don't kmow how to drop nested defaults and don't know why we need this ?!
                MaskingStrategy()
                "masking_strategy": {
                    "strategy_type": "default",
                    "parameters": {"masked_pixel_percentage": 0.2, "roi_size": 11},
                },
            }

            remove_default_optionals(dictionary, defaults)

        return dictionary
