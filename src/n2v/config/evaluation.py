from pydantic import BaseModel, Field, validator

from .data import Data


class Evaluation(BaseModel):
    data: Data
    metric: str  # TODO add enum

    class Config:
        allow_mutation = False  # model is immutable
