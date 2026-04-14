"""Filter patch using a maximum filter."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol
from careamics.utils import get_logger

from .filtermap_utils import FilterValueFunc, create_filter_map

logger = get_logger(__name__)


class MaxPatchFilter(PatchFilterProtocol):
    """Patch filter based on thresholding the maximum filter (CSBDeep-inspired).

    Parameters
    ----------
    threshold : float
        Maximum-filter threshold; patches below are filtered out.
    coverage : float, default=0.25
        Ratio of pixels below threshold to filter out (0-1).

    Attributes
    ----------
    threshold : float
        Threshold for the maximum filter of the patch.
    """

    def __init__(
        self,
        threshold: float,
        coverage: float = 0.25,
    ) -> None:
        """Create a MaxPatchFilter. Removes patches below max-filter threshold.

        Parameters
        ----------
        threshold : float
            Maximum-filter threshold.
        coverage : float, default=0.25
            If the ratio of pixels above the threshold is below this value, the patch
            will be filtered out.
        """
        self.threshold = threshold
        self.coverage = coverage

    def filter_out(self, patch: np.ndarray) -> bool:
        """Return True if patch should be filtered out by max-filter criteria.

        Parameters
        ----------
        patch : numpy.ndarray
            Image patch to evaluate.

        Returns
        -------
        bool
            True if patch should be filtered out, False otherwise.
        """
        if np.max(patch) < self.threshold:
            return True

        patch_shape = [(p // 2 if p > 1 else 1) for p in patch.shape]
        filtered = maximum_filter(patch, patch_shape, mode="constant")
        return (np.mean(filtered > self.threshold) < self.coverage).item()

    @staticmethod
    def filter_map(
        image: np.ndarray,
        patch_size: Sequence[int],
        coverage: float | None = None,
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
        coverage : float | None, default=None
            If the ratio of pixels above the threshold is below this value, the patch
            will be filtered out. If `None`, for 2D it will be 0.25 and for 3D it will
            be 0.125.

        Returns
        -------
        numpy.ndarray
            The filter map, which has the same shape as the input image.
        """
        filter_value_func = MaxPatchFilter._get_filter_value_func(patch_size, coverage)
        filter_map = create_filter_map(
            image, filter_value_func, patch_size, direction="greater"
        )
        return filter_map

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
        plt.colorbar(m, ax=ax)
        fig.suptitle("Maximum Filter Map")
        return fig

    @staticmethod
    def _get_filter_value_func(
        patch_size: Sequence[int],
        coverage: float | None = None,
    ) -> FilterValueFunc:
        """
        Get a function that returns the filter value of a patch.

        Parameters
        ----------
        patch_size : sequence of int
            The patch size intended to be used for training.
        coverage : float | None, default=None
            If the ratio of pixels above the threshold is below this value, the patch
            will be filtered out. If `None`, for 2D it will be 0.25 and for 3D it will
            be 0.125.

        Returns
        -------
        FilterValueFunc
            A function that outputs a value to determine whether a patch should be
            filtered.
        """
        if coverage is None:
            n_dims = len(patch_size)
            coverage = 0.5**n_dims

        def filter_value_func(patch: np.ndarray) -> float:  # numpydoc ignore=GL08
            patch_filt = maximum_filter(patch.squeeze(0), patch_size)
            total_pixels = patch.size
            # we want to the maximum value that meets the coverage requirements
            unique_values, counts = np.unique(
                patch_filt, return_counts=True, sorted=True
            )
            # reverse order of counts (now ordered by largest to smallest filter value)
            valid = np.where(np.cumsum(counts[::-1]) / total_pixels >= coverage)[0]
            # now we find the first value which has more counts than the coverage ratio
            v_idx = valid[0]
            # use the index of the counts to extract the value
            value = unique_values[::-1][v_idx]
            return float(value)

        return filter_value_func
