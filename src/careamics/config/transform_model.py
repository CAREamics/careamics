from __future__ import annotations

from inspect import getmembers, isclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationInfo
import albumentations as Aug
from  careamics import transforms
from careamics.utils.torch_utils import filter_parameters

ALL_TRANSFORMS = dict(getmembers(Aug, isclass) + getmembers(transforms, isclass))


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
    def validate_name(cls, transform_name: str) -> str:
        """Validate transform name based on `ALL_TRANSFORMS`."""
        if transform_name not in ALL_TRANSFORMS.keys():
            raise ValueError(
                f"Incorrect transform name {transform_name}. Accepted transforms "
                f"are ManipulateN2V, NormalizeWithoutTarget, and all transformations "
                f"in Albumentations (see https://albumentations.ai/)."
            )
        return transform_name


    @field_validator("parameters")
    def validate_transform(cls, params: dict, value: ValidationInfo) -> dict:
        """Validate transform parameters based on the transform signature."""
        transform_name = value.data["name"]

        # filter the user parameters according to the scheduler's signature
        parameters = filter_parameters(
            ALL_TRANSFORMS[transform_name], params
        )

        # try to instantiate the transform with the filtered parameters
        try:
            ALL_TRANSFORMS[transform_name](**parameters)
        except Exception as e:
            raise ValueError(
                f"Error while trying to instantiate the transform {transform_name} "
                f"with the provided parameters: {parameters}. The error is: {e}."
            )
        
        return parameters