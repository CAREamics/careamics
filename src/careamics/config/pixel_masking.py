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
        return data
