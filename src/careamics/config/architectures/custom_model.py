from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict
)

# TODO: decorator to register custom model, problem is the architecture Literal 
# as it is used as a discriminator in algorithm
class CustomModel(BaseModel):

    # pydantic model config
    model_config = ConfigDict(
        validate_assignment=True
    )

    # discriminator used for choosing the pydantic model in Model
    architecture: Literal["custom"]

    # parameters
    parameters: dict = {}
