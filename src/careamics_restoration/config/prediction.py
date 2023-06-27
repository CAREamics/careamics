from typing import List

from pydantic import Field

from .stage import Stage


class Prediction(Stage):
    overlap: List[int] = Field(
        ..., min_items=2, max_items=3
    )  # TODO ge image size, check consistency with image size

    class Config:
        allow_mutation = False  # model is immutable
