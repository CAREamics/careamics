from __future__ import annotations

from inspect import getmembers, isclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationInfo
import albumentations as Aug
import careamics.utils.transforms as custom_transforms
from careamics.utils.torch_utils import filter_parameters

ALL_TRANSFORMS = dict(getmembers(Aug, isclass) + getmembers(custom_transforms, isclass))


class TransformType:
    """Available transforms.

    Can be applied both to an image and to a patch

    """

    @classmethod
    def validate_transform_type(cls, transform: str, parameters: dict) -> None:
        """_summary_.

        Parameters
        ----------
        transform : Union[str, Transform]
            _description_
        parameters : dict
            _description_

        Returns
        -------
        BaseModel
            _description_
        """
        if transform not in ALL_TRANSFORMS.keys():
            raise ValueError(
                f"Incorrect transform name {transform}."
                f"Please refer to the documentation"  # TODO add link to documentation
            )
        # TODO validate provided params against default params
        # TODO validate no duplicates
        return transform, parameters


class TransformModel(BaseModel):
    """Whole image transforms.

    Parameters
    ----------
    BaseModel : _type_
        _description_
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal[
        "FLIP",
        "RANDOM_ROTATE90",
        "NORMALIZE_WO_TARGET",
        "MANIPULATE_N2V",
        "CUSTOM"
    ]
    parameters: dict = Field(default={}, validate_default=True)


    # TODO remove this
    @field_validator("parameters")
    def validate_transform(cls, params: dict, value: ValidationInfo) -> dict:
        """Validate transform parameters."""


        transform_name = value.data["name"]

        # filter the user parameters according to the scheduler's signature
        parameters, missing_mandatory = filter_parameters(
            ALL_TRANSFORMS[transform_name], params
        )

        # if there are missing parameters, raise an error
        if len(missing_mandatory) > 0:
            raise ValueError(
                f"Optimizer {transform_name} requires the following parameters: "
                f"{missing_mandatory}."
            )

        return parameters