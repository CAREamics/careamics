"""Class representing the parameters of structN2V masks."""

from dataclasses import dataclass
from typing import Literal

from careamics.config.support import SupportedStructAxis


@dataclass
class StructMaskParameters:
    """Parameters of structN2V masks.

    Attributes
    ----------
    axis : int
        Axis along which to apply the mask, horizontal (0), vertical (1) or cross (2).
    span : int
        Span of the mask, must be odd.

    Parameters
    ----------
    axis : Literal[0, 1, 2, "horizontal", "vertical", "cross"]
        Axis along which to apply the mask, horizontal (0), vertical (1) or
        cross (2).
    span : int
        Span of the mask, must be odd.
    """

    axis: int
    span: int

    def __init__(
        self,
        axis: Literal[0, 1, 2, "horizontal", "vertical", "cross"],
        span: int,
    ):
        """Constructor.

        Parameters
        ----------
        axis : Literal[0, 1, 2, "horizontal", "vertical", "cross"]
            Axis along which to apply the mask, horizontal (0), vertical (1) or
            cross (2).
        span : int
            Span of the mask, must be odd.

        Raises
        ------
        ValueError
            If span is not odd.
        ValueError
            If axis is not 0, 1, or 2.
        """
        if span % 2 == 0:
            raise ValueError(f"Span must be odd, got {span}.")

        if not isinstance(axis, int):
            index = list(SupportedStructAxis).index(SupportedStructAxis(axis))
            axis = index  # type: ignore

        if axis not in [0, 1, 2]:
            raise ValueError(f"Axis must be 0, 1 or 2, got {axis}.")

        self.axis = axis  # type: ignore
        self.span = span
