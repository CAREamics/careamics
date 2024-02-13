from __future__ import annotations

from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator, ValidationInfo
)

from careamics.utils.torch_utils import filter_parameters
from .support import get_all_transforms

class TransformModel(BaseModel):

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: str
    parameters: dict = Field(default={}, validate_default=True)


    @field_validator("name", mode="plain")
    @classmethod
    def validate_name(cls, transform_name: str) -> str:
        """Validate transform name based on the list of all accepted transforms."""
        if not transform_name in get_all_transforms().keys():
            raise ValueError(
                f"Incorrect transform name {transform_name}. Accepted transforms "
                f"are ManipulateN2V, NormalizeWithoutTarget, and all transformations "
                f"in Albumentations (see https://albumentations.ai/)."
            )
        return transform_name


    @model_validator(mode="after")
    def validate_transform(self) -> 'TransformModel':
        """Validate transform parameters based on the transform's signature."""
        # filter the user parameters according to the transform's signature
        parameters = filter_parameters(
            get_all_transforms()[self.name], self.parameters
        )

        # try to instantiate the transform with the filtered parameters
        try:
            get_all_transforms()[self.name](**parameters)
        except Exception as e:
            raise ValueError(
                f"Error while trying to instantiate the transform {self.name} "
                f"with the provided parameters: {parameters}. Are you missing some "
                f"mandatory parameters? The error is: {e}."
            )
    
        # update the parameters with the filtered ones
        # note: assigment would trigger an infinite recursion
        self.parameters.clear()
        self.parameters.update(parameters)
        
        return self
    