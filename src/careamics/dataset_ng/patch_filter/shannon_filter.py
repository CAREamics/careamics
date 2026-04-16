"""Filter patches based on Shannon entropy threshold."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
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

    @staticmethod
    def apply_filter(filter_map: np.ndarray, threshold: float) -> NDArray[np.bool_]:
        """
        Apply the Shannon entropy filter to a precomputed filter map.

        The filter map is the output of the `filter_map` method.

        Parameters
        ----------
        filter_map : numpy.NDArray
            The precomputed Shannon entropy map of the image.
        threshold : float
            The Shannon entropy threshold for filtering.

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
        cbar.ax.set_ylabel("threshold")
        fig.suptitle("Shannon Entropy Filter Map")
        return fig
