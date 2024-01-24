from __future__ import annotations

from enum import Enum
from typing import Dict, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# TODO where is all this used?
class MaskingStrategyType(str, Enum):
    """Available masking strategy.

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
        if masking_strategy == MaskingStrategyType.DEFAULT:
            DefaultMaskingStrategy(**parameters)
            return (
                DefaultMaskingStrategy().model_dump() if not parameters else parameters
            )
        elif masking_strategy == MaskingStrategyType.MEDIAN:
            MedianMaskingStrategy(**parameters)
            return (
                MedianMaskingStrategy().model_dump() if not parameters else parameters
            )


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
    parameters: Dict = Field(default_factory=dict, validate_default=True)

    @field_validator("parameters")
    def validate_parameters(cls, data, values) -> Dict:
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
        if values.data["strategy_type"] not in [
            MaskingStrategyType.DEFAULT,
            MaskingStrategyType.MEDIAN,
        ]:
            raise ValueError(
                f"Incorrect masking strategy {values.data['strategy_type']}."
                f"Please refer to the documentation"  # TODO add link to documentation
            )
        parameters = MaskingStrategyType.validate_masking_strategy_type(
            values.data["strategy_type"], data
        )
        return parameters

    # def model_dump(
    #     self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    # ) -> Dict:
    #     """
    #     Override model_dump method.

    #     The purpose is to ensure export smooth import to yaml. It includes:
    #         - remove entries with None value.
    #         - remove optional values if they have the default value.

    #     Parameters
    #     ----------
    #     exclude_optionals : bool, optional
    #         Whether to exclude optional arguments if they are default, by default True.
    #     *args : List
    #         Positional arguments, unused.
    #     **kwargs : Dict
    #         Keyword arguments, unused.

    #     Returns
    #     -------
    #     Dict
    #         Dictionary representation of the model.
    #     """
    #     dictionary = super().model_dump(exclude_none=True)

    #     if exclude_optionals is True:
    #         # remove optional arguments if they are default
    #         defaults = {
    #             "model": {
    #                 "architecture": "UNet",
    #                 "parameters": {"depth": 2, "num_channels_init": 32},
    #             },
    #             # MaskingStrategy()
    #             "masking_strategy": {
    #                 "strategy_type": "default",
    #                 "parameters": {"masked_pixel_percentage": 0.2, "roi_size": 11},
    #             },
    #         }

    #         remove_default_optionals(dictionary, defaults)

    #     return dictionary


class DefaultMaskingStrategy(BaseModel):
    """Default masking strategy of Noise2Void.

    Parameters
    ----------
    masked_pixel_percentage : float
        Percentage of pixels to be masked.
    roi_size : float
        Size of the region of interest (ROI).
    """

    masked_pixel_percentage: float = Field(default=0.2, ge=0.01, le=21.0)
    roi_size: float = Field(default=11, ge=3, le=21)


class MedianMaskingStrategy(BaseModel):
    """Median masking strategy of N2V2.

    Parameters
    ----------
    masked_pixel_percentage : float
        Percentage of pixels to be masked.
    roi_size : float
        Size of the region of interest (ROI).
    """

    masked_pixel_percentage: float = Field(default=0.2, ge=0.01, le=21.0)
    roi_size: float = Field(default=11, ge=3, le=21)
