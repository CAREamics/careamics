from __future__ import annotations

from inspect import getmembers, isclass, signature
from typing import Dict

import albumentations as A
from pydantic import BaseModel, ConfigDict, Field, model_validator

import careamics.utils.transforms as custom_transforms

ALL_TRANSFORMS = dict(getmembers(A, isclass) + getmembers(custom_transforms, isclass))


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
        # TOD) validate provided params against default params
        return transform, parameters


class Transform(BaseModel):
    """Whole image transforms.

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
    name: str
    parameters: Dict = Field(default_factory=dict, validate_default=True)

    @model_validator(mode="after")
    def validate_transform(cls, data: Transform) -> Transform:
        """Validate transform parameters."""
        try:
            if data.name in ALL_TRANSFORMS.keys():
                if data.parameters:
                    if (
                        not signature(
                            ALL_TRANSFORMS[data.name]
                        ).parameters.keys()
                        & data.parameters.keys()
                    ):
                        raise ValueError(
                            f"Transform {data.name} does not accept parameters"
                            f" {data.parameters.keys()}."
                            f"Please refer to the documentation"  # TODO add link to doc
                        )

            else:
                raise ValueError(
                    f"Incorrect transform name {data.name}."
                    f"Please refer to the documentation"  # TODO add link to doc
                )
        except AttributeError as e:
            raise ValueError(
                "Missing transform name."
                "Please refer to the documentation"  # TODO add link to documentation
            ) from e

        return data
