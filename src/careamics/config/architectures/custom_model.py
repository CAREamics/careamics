from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict
)

# TODO: decorator to register custom model 
# https://stackoverflow.com/questions/3054372/auto-register-class-methods-using-decorator
class CustomModel(BaseModel):

    # pydantic model config
    model_config = ConfigDict(
        validate_assignment=True
    )

    # discriminator used for choosing the pydantic model in Model
    architecture: Literal["Custom"]

    # parameters
    parameters: dict = {}
