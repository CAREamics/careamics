"""Filter using an image mask."""

import numpy as np

from careamics.dataset_ng.image_stack import GenericImageStack
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patch_filter.coordinate_filter_protocol import (
    CoordinateFilterProtocol,
)
from careamics.dataset_ng.patching_strategies import PatchSpecs


# TODO is it more intuitive to have a negative mask? (mask of what to avoid)
class MaskCoordFilter(CoordinateFilterProtocol):
    """
    Filter patch coordinates based on an image mask.

    Parameters
    ----------
    mask_extractor : PatchExtractor[GenericImageStack]
        Patch extractor for the binary mask to use for filtering.
    coverage : float
        Minimum percentage of masked pixels required to keep a patch. Must be
        between 0 and 1.

    Attributes
    ----------
    mask_extractor : PatchExtractor[GenericImageStack]
        Patch extractor for the binary mask to use for filtering.
    coverage : float
        Minimum percentage of masked pixels required to keep a patch.
    """

    def __init__(
        self,
        mask_extractor: PatchExtractor[GenericImageStack],
        coverage: float,
    ) -> None:
        """
        Create a MaskCoordFilter.

        This filter removes patches who fall below a threshold of masked pixels
        percentage. The mask is expected to be a positive mask where masked pixels
        correspond to regions of interest.

        Parameters
        ----------
        mask_extractor : PatchExtractor[GenericImageStack]
            The patch extractor for the mask used for filtering.
        coverage : float
            Minimum percentage of masked pixels required to keep a patch. Must be
            between 0 and 1.

        Raises
        ------
        ValueError
            If coverage is not between 0 and 1.
        """
        if not (0 <= coverage <= 1):
            raise ValueError("Probability p must be between 0 and 1.")

        self.mask_extractor = mask_extractor
        self.coverage = coverage

    def filter_out(self, patch_specs: PatchSpecs) -> bool:
        """
        Determine whether to filter out a patch based an image mask.

        Parameters
        ----------
        patch_specs : PatchSpecs
            The patch coordinates to evaluate.

        Returns
        -------
        bool
            True if the patch should be filtered out, False otherwise.
        """
        mask_patch = self.mask_extractor.extract_patch(**patch_specs)

        masked_fraction = np.sum(mask_patch) / mask_patch.size
        return masked_fraction < self.coverage
