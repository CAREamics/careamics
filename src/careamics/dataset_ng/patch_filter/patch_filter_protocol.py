"""A protocol for patch filtering."""

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
