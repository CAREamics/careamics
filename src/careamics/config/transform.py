from __future__ import annotations

from inspect import getmembers, isclass, signature
from typing import Dict, Literal

import albumentations as Aug
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import careamics.utils.transforms as custom_transforms
from careamics.utils.torch_utils import get_parameters

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

    name: str
    parameters: dict = Field(default={}, validate_default=True)

    @field_validator("name")
    def name_valid(cls, name: str) -> str:
        """Validate transform name.

        Parameters
        ----------
        name : str
            Transform name.

        Returns
        -------
        str
            Validated transform name.

        Raises
        ------
        ValueError
            If transform name is not valid.
        """
        if name not in ALL_TRANSFORMS.keys():
            raise ValueError(
                f"Incorrect transform name {name}."
                f"Please refer to the documentation"  # TODO add link to documentation
            )
        return name


    @model_validator(mode="after")
    def validate_transform(cls, data: TransformModel) -> TransformModel:
        """Validate transform parameters."""
        filtered_params, mandatory_params = get_parameters(
            ALL_TRANSFORMS[data.name], data.parameters
        )

        # check if none of the parameters are accepted
        if len(filtered_params) == 0 and len(data.parameters) > 0:
            raise ValueError(
                f"Transform {data.name} does not accept parameters"
                f" {data.parameters.keys()}."
                f"Please refer to the documentation"  # TODO add link to doc
            )
        # check if all mandatory parameters are provided
        if len(set(mandatory_params) - filtered_params.keys()) > 0:
            missing_params = set(mandatory_params) - filtered_params.keys() 
            raise ValueError(
                f"Transform {data.name} requires the following parameters: "
                f"{missing_params}."
                f"Please refer to the documentation"  # TODO add link to doc
            )

        data.parameters = filtered_params

        return data
