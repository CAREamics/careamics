from pydantic import BaseModel

from .stage import Stage


class Evaluation(Stage):
    metric: str  # TODO add enum

    class Config:
        allow_mutation = False  # model is immutable
