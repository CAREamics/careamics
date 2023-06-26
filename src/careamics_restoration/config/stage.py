from pydantic import BaseModel

from .data import Data


class Stage(BaseModel):
    data: Data
