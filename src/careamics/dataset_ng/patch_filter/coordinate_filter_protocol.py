"""A protocol for patch filtering."""

from typing import Protocol

from careamics.dataset_ng.patching_strategies import PatchSpecs


class CoordinateFilterProtocol(Protocol):
    """
    An interface for implementing coordinate filtering strategies.
    """

    def filter_out(self, patch: PatchSpecs) -> bool:
        """
        Determine whether to filter out a given patch based on its coordinates.

        Parameters
        ----------
        patch : PatchSpecs
            The patch coordinates to evaluate.

        Returns
        -------
        bool
            True if the patch should be filtered out (excluded), False otherwise.
        """
        ...
