"""Filter using mean and standard deviation thresholds."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol

from .filtermap_utils import create_filter_map


class MeanStdPatchFilter(PatchFilterProtocol):
    """
    Filter patches based on mean and standard deviation thresholds.

    Parameters
    ----------
    mean_threshold : float
        Threshold for the mean of the patch.
    std_threshold : float, optional
        Threshold for the standard deviation of the patch. If None, then no
        standard deviation filtering is applied.

    Attributes
    ----------
    mean_threshold : float
        Threshold for the mean of the patch.
    std_threshold : float
        Threshold for the standard deviation of the patch.
    """

    def __init__(
        self,
        mean_threshold: float,
        std_threshold: float | None = None,
    ) -> None:
        """
        Create a MeanStdPatchFilter.

        This filter removes patches whose mean and standard deviation are both below
        specified thresholds. The filtering is applied with a probability `p`, allowing
        for stochastic filtering.

        Parameters
        ----------
        mean_threshold : float
            Threshold for the mean of the patch.
        std_threshold : float | None, default=None
            Threshold for the standard deviation of the patch. If None, then no
            standard deviation filtering is applied.

        Raises
        ------
        ValueError
            If mean_threshold or std_threshold is negative.
        ValueError
            If std_threshold is negative.
        """
        if mean_threshold < 0:
            raise ValueError("Mean threshold must be non-negative.")
        if std_threshold is not None and std_threshold < 0:
            raise ValueError("Std threshold must be non-negative.")

        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold

    def filter_out(self, patch: np.ndarray) -> bool:
        """
        Determine whether to filter out a patch based on mean and std thresholds.

        Parameters
        ----------
        patch : numpy.NDArray
            The image patch to evaluate.

        Returns
        -------
        bool
            True if the patch should be filtered out, False otherwise.
        """
        patch_mean = np.mean(patch)
        patch_std = np.std(patch)

        return (patch_mean < self.mean_threshold).item() or (
            self.std_threshold is not None and (patch_std < self.std_threshold).item()
        )

    @staticmethod
    def filter_map(image: np.ndarray, patch_size: Sequence[int]) -> np.ndarray:
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
        filtermap_mean = create_filter_map(
            image,
            # using lambda just for converting to float for mypy
            lambda patch: float(np.mean(patch)),
            patch_size,
            direction="greater",
        )
        filtermap_std = create_filter_map(
            image,
            # using lambda just for converting to float for mypy
            lambda patch: float(np.std(patch)),
            patch_size,
            direction="greater",
        )
        return np.stack([filtermap_mean, filtermap_std])

    @staticmethod
    def plot_filter_map(
        image: np.ndarray, filter_map: np.ndarray, z_idx: int | None = None
    ) -> plt.Figure:
        """
        Plot both the mean and standard-deviation filter over an image.

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
            The figure object displaying the filter maps.
        """
        if image.ndim == 4:
            # take the middle z slice if not specified
            z_idx == image.shape[0] // 2 if z_idx is None else z_idx
            image = image[:, z_idx]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        for i, ax in enumerate(axes):
            ax.imshow(image, "gray")
            m = ax.imshow(filter_map[i], "magma", alpha=0.5)
            plt.colorbar(m, ax=ax)
        axes[0].set_title("Mean Filter Map")
        axes[1].set_title("Standard-Deviation Filter Map")
        fig.suptitle("Mean-Std Filter Map")
        return fig
