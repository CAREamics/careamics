from dataclasses import dataclass
from typing import Literal

@dataclass
class StructMaskParameters:

    axis: Literal[0, 1]
    span: int