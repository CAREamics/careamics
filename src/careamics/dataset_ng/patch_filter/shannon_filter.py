"""Filter patches based on Shannon entropy threshold."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from skimage.measure import shannon_entropy

from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol

from .filtermap_utils import create_filter_map


class ShannonPatchFilter(PatchFilterProtocol):
    """Filter patches based on Shannon entropy threshold.

    Parameters
    ----------
    threshold : float
        Shannon entropy threshold; patches below are filtered out.

    Attributes
    ----------
    threshold : float
        Threshold for the Shannon entropy of the patch.
    """

    def __init__(self, threshold: float) -> None:
        """Create a ShannonPatchFilter.

        This filter removes patches whose Shannon entropy is below a specified
        threshold.

        Parameters
        ----------
        threshold : float
            Threshold for the Shannon entropy of the patch.

        Raises
        ------
        ValueError
            If threshold is negative.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative.")

        self.threshold = threshold

    def filter_out(self, patch: np.ndarray) -> bool:
        """
        Determine whether to filter out a patch based on its Shannon entropy.

        Parameters
        ----------
        patch : numpy.NDArray
            The patch to evaluate.

        Returns
        -------
        bool
            True if the patch should be filtered out, False otherwise.
        """
        return bool(shannon_entropy(patch) < self.threshold)

    @staticmethod
    def filter_map(
        image: NDArray[np.int_ | np.floating],
        patch_size: Sequence[int],
    ) -> np.ndarray:
        """
        Compute a filter map for the entire image based on the patch filtering criteria.

        The filter map will show the threshold that, above which, will result in regions
        of the image being excluded from training.

        Parameters
        ----------
        image : numpy.ndarray
            A 2D or 3D image.
        patch_size : sequence of int
            The patch size intended to be used for training.

        Returns
        -------
        numpy.ndarray
            The filter map, which has the same shape as the input image.
        """
        filtermap = create_filter_map(
            image,
            # using lambda just for converting to float for mypy
            lambda patch: float(shannon_entropy(patch)),
            patch_size,
            direction="greater",
        )
        return filtermap
