"""Model constraints protocol."""

from collections.abc import Sequence
from typing import Protocol


class ModelConstraints(Protocol):
    """Protocol for model constraints on input and output tensors."""

    def validate_input_channels(self, n_channels: int) -> None:
        """Whether the given channel size is compatible with the model constraints.

        Parameters
        ----------
        n_channels : int
            The number of channels in the input tensor to validate.

        Raises
        ------
        ValueError
            If the number of channels is not compatible with the model constraints.
        """
        ...

    def validate_target_channels(self, n_channels: int) -> None:
        """Whether the given channel size is compatible with the model constraints.

        Parameters
        ----------
        n_channels : int
            The number of channels in the target tensor to validate.

        Raises
        ------
        ValueError
            If the number of channels is not compatible with the model constraints.
        """
        ...

    def validate_spatial_shape(self, shape: Sequence[int]) -> None:
        """Whether the given spatial shape is compatible with the model constraints.

        Shape must be of length 2 (YX) or 3 (ZYX). To validate channel dimension, use
        `validate_input_channels` or `validate_target_channels` instead.

        Parameters
        ----------
        shape : Sequence[int]
            The spatial shape of the input tensor to validate (length 2 or 3).

        Raises
        ------
        ValueError
            If the spatial shape is not compatible with the model constraints.
        """
        ...
