"""A protocol for patch filtering."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class PatchFilterProtocol(Protocol):
    """
    An interface for implementing patch filtering strategies.
    """

    def filter_out(self, patch: np.ndarray) -> bool:
        """
        Determine whether to filter out a given patch.

        Parameters
        ----------
        patch : numpy.NDArray
            The image patch to evaluate.

        Returns
        -------
        bool
            True if the patch should be filtered out (excluded), False otherwise.
        """
        ...

    @staticmethod
    def filter_map(
        image: np.ndarray,
        patch_size: Sequence[int],
    ) -> np.ndarray:
        """
        Compute a filter map for the entire image based on the patch filtering criteria.

        Parameters
        ----------
        image : numpy.NDArray
            The full image to evaluate.
        patch_size : Sequence[int]
            The size of the patches to consider.

        Returns
        -------
        numpy.NDArray
            A map where each element is the .
        """
        ...
