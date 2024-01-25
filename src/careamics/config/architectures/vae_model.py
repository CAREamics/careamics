from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


class VAEModel(BaseModel):

    model_config = ConfigDict(
        use_enum_values=True, protected_namespaces=(), validate_assignment=True
    )

    architecture: Literal["VAE"]