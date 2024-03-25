from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from careamics.utils.torch_utils import filter_parameters
from careamics.transforms import get_all_transforms


class TransformParameters(BaseModel):

    model_config = ConfigDict(
        extra="allow",
    )


class TransformModel(BaseModel):
    """Pydantic model used to represent an image transformation.

    Accepted transformations are ManipulateN2V, NormalizeWithoutTarget, and all
    transformations in Albumentations (see https://albumentations.ai/).
    """
    name: str
    parameters: TransformParameters = TransformParameters()

    @field_validator("name", mode="plain")
    @classmethod
    def validate_name(cls, transform_name: str) -> str:
        """Validate transform name based on the list of all accepted transforms."""
        if transform_name not in get_all_transforms().keys():
            raise ValueError(
                f"Incorrect transform name {transform_name}. Accepted transforms "
                f"are ManipulateN2V, NormalizeWithoutTarget, and all transformations "
                f"in Albumentations (see https://albumentations.ai/)."
            )
        return transform_name

    @model_validator(mode="after")
    def validate_transform(self) -> TransformModel:
        """Validate transform parameters based on the transform's signature."""
        # filter the user parameters according to the transform's signature
        parameters = filter_parameters(
            get_all_transforms()[self.name], 
            self.parameters.model_dump())

        # try to instantiate the transform with the filtered parameters
        try:
            get_all_transforms()[self.name](**parameters)
        except Exception as e:
            raise ValueError(
                f"Error while trying to instantiate the transform {self.name} "
                f"with the provided parameters: {parameters}. Are you missing some "
                f"mandatory parameters?"
            ) from e

        # update the parameters with the filtered ones
        self.parameters = TransformParameters(**parameters)

        return self
