"""A protocol for patch filtering."""

from collections.abc import Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np


class PatchFilterProtocol(Protocol):
    """Interface for implementing patch filtering strategies."""

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
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute a filter map for the entire image based on the patch filtering criteria.

        Parameters
        ----------
        image : numpy.ndarray
            The full image to evaluate.
        patch_size : Sequence[int]
            The size of the patches to consider.
        *args : Any
            Concrete implementations may have additional positional arguments.
        **kwargs : Any
            Concrete implementations may have additional key-word arguments.

        Returns
        -------
        numpy.NDArray
            A map where each element is the .
        """
        ...

    @staticmethod
    def plot_filter_map(
        image: np.ndarray, filter_map: np.ndarray, z_idx: int | None = None
    ) -> plt.Figure:
        """
        Plot the filter map over an image.

        Parameters
        ----------
        image : numpy.ndarray
            The image that has been evaluated.
        filter_map : numpy.ndarray
            The filter map that has been evaluated using the method `filter_map`.
        z_idx : int | None, default=None
            If the image is 3D, `z_idx` selects the slice to display. If `None` the
            central slice will be selected.

        Returns
        -------
        matplotlib.pyplot.Figure
            The figure object displaying the filter map.
        """
        ...
