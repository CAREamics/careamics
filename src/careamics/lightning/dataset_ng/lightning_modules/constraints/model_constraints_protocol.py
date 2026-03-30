"""Model constraints protocol."""

from collections.abc import Sequence
from typing import Protocol


class ModelConstraints(Protocol):
    """Protocol for model constraints on input tensors spatial shape."""

    def validate_input_shape(self, shape: Sequence[int]) -> None:
        """Whether the given spatial shape is compatible with the model constraints.

        Parameters
        ----------
        shape : Sequence[int]
            The spatial shape of the input tensor to validate (length 2 or 3).
        """
        ...
