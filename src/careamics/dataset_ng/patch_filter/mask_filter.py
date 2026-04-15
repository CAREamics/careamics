"""Filter using an image mask."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .filtermap_utils import create_filter_map
from .patch_filter_protocol import PatchFilterProtocol


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

    def filter_out(self, patch: NDArray[np.bool_ | np.int_]) -> bool:
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
            True if the patch should be filtered out, False otherwise.
        """
        masked_fraction = np.count_nonzero(patch) / patch.size
        return bool(masked_fraction < self.coverage)

    @staticmethod
    def filter_map(
        image: NDArray[np.bool_ | np.int_],
        patch_size: Sequence[int],
    ) -> NDArray[np.bool_]:
        """
        Compute a filter map for the entire image based on the patch filtering criteria.

        The filter map will show the percentage coverage of the mask in a patch.

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
        return create_filter_map(
            image, MaskFilter._filter_value, patch_size, direction="greater"
        )

    @staticmethod
    def apply_filter(filter_map: np.ndarray, threshold: float) -> NDArray[np.bool_]:
        """
        Apply the max filter to a filter map.

        The filter map is the output of the `filter_map` method.

        Parameters
        ----------
        filter_map : numpy.ndarray
            The max filter map of the image.
        threshold : float
            The threshold to apply to the filter map.

        Returns
        -------
        numpy.typing.NDArray[np.bool_]
           A binary map where True indicates patches that pass the filter, i.e. they
           should be kept for training.
        """
        return filter_map > threshold

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
        if image.ndim == 3:
            # take the middle z slice if not specified
            z_idx == image.shape[0] // 2 if z_idx is None else z_idx
            image = image[z_idx]

        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax.imshow(image, "gray")
        m = ax.imshow(filter_map, "magma", alpha=0.5)
        cbar = plt.colorbar(m, ax=ax)
        cbar.ax.set_ylabel("coverage")
        fig.suptitle("Mask Filter Map")
        return fig

    @staticmethod
    def _filter_value(patch: NDArray[np.bool_]) -> float:
        """
        Get the filter value of the MaskFilter.

        This is the coverage of the mask in a patch.

        Parameters
        ----------
        patch : numpy.ndarray
            A patch of the mask to evaluate.

        Returns
        -------
        float
            The coverage of the mask in the patch.
        """
        return np.count_nonzero(patch) / patch.size
