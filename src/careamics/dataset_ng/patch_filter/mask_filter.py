"""Filter using an image mask."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from careamics.dataset_ng.patch_filter.patch_filter_protocol import (
    PatchFilterProtocol,
)


# TODO is it more intuitive to have a negative mask? (mask of what to avoid)
class MaskFilter(PatchFilterProtocol):
    """
    Filter patches based on a boolean image mask.

    Evaluates patches by computing the fraction of True pixels in a boolean mask
    and filters out patches that fall below a coverage threshold.

    Parameters
    ----------
    coverage : float
        Minimum percentage of masked pixels (True values) required to keep a patch.
        Must be between 0 and 1.

    Attributes
    ----------
    coverage : float
        Minimum percentage of masked pixels required to keep a patch.
    """

    def __init__(self, coverage: float) -> None:
        """
        Create a MaskFilter.

        This filter removes patches that fall below a threshold of masked pixels
        (True values) in a boolean mask. The mask is expected to be a boolean array
        where True represents regions of interest and False represents background.

        Parameters
        ----------
        coverage : float
            Minimum percentage of masked pixels (True values) required to keep a patch.
            Must be between 0 and 1.

        Raises
        ------
        ValueError
            If coverage is not between 0 and 1.
        """
        if not (0 <= coverage <= 1):
            raise ValueError("Coverage must be between 0 and 1.")

        self.coverage = coverage

    def filter_out(self, patch: NDArray[np.bool_]) -> bool:
        """
        Determine whether to filter out a patch based on mask coverage.

        Parameters
        ----------
        patch : NDArray[np.bool_]
            A boolean mask patch to evaluate. Expected to have dtype bool where
            True indicates regions of interest and False indicates background.

        Returns
        -------
        bool
            True if the patch should be filtered out (masked fraction < coverage),
            False otherwise.
        """
        masked_fraction = np.sum(patch) / patch.size
        return bool(masked_fraction < self.coverage)

    @staticmethod
    def filter_map(
        image: NDArray[np.bool_],
        patch_size: Sequence[int],
    ) -> NDArray[np.bool_]:
        """
        Return the mask image as the filter map.

        For a mask image, the filter map is simply the mask itself since it
        already indicates regions of interest.

        Parameters
        ----------
        image : NDArray[np.bool_]
            The boolean mask image to evaluate.
        patch_size : Sequence[int]
            The size of the patches to consider (unused for mask filter).

        Returns
        -------
        NDArray[np.bool_]
            The mask image itself.
        """
        return image
