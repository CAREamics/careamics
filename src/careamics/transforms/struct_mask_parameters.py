from dataclasses import dataclass
from typing import Literal


@dataclass
class StructMaskParameters:
    """Parameters of structN2V masks.

    Parameters
    ----------
    axis : Literal[0, 1]
        Axis along which to apply the mask, horizontal (0) or vertical (1).
    span : int
        Span of the mask.
    """

    axis: Literal[0, 1]
    span: int
