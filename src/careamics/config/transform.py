from typing import Dict, Optional

from pydantic import BaseModel, model_validator


class Transform(BaseModel):

    name: str
    parameters: Optional[Dict[str, float]] = None


    @model_validator
    def validate_exists(cls, v):
        # TODO check if in the transform package or is Manipulate
        return v