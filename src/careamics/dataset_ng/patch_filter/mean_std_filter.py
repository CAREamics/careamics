"""Filter using mean and standard deviation thresholds."""

from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_array_extractor,
)
from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol
from careamics.dataset_ng.patching_strategies import TilingStrategy


class MeanStdPatchFilter(PatchFilterProtocol):
    """
    Filter patches based on mean and standard deviation thresholds.

    Attributes
    ----------
    mean_threshold : float
        Threshold for the mean of the patch.
    std_threshold : float
        Threshold for the standard deviation of the patch.
    p : float
        Probability of applying the filter to a patch.
    rng : np.random.Generator
        Random number generator for stochastic filtering.
    """

    def __init__(
        self,
        mean_threshold: float,
        std_threshold: float | None = None,
        p: float = 1.0,
        seed: int | None = None,
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
        p : float, default=1
            Probability of applying the filter to a patch. Must be between 0 and 1.
        seed : int | None, default=None
            Seed for the random number generator for reproducibility.

        Raises
        ------
        ValueError
            If mean_threshold or std_threshold is negative.
        ValueError
            If std_threshold is negative.
        ValueError
            If p is not between 0 and 1.
        """

        if mean_threshold < 0:
            raise ValueError("Mean threshold must be non-negative.")
        if std_threshold is not None and std_threshold < 0:
            raise ValueError("Std threshold must be non-negative.")
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")

        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold

        self.p = p
        self.rng = np.random.default_rng(seed)

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

        if self.rng.uniform(0, 1) < self.p:
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)

            return (patch_mean < self.mean_threshold) or (
                self.std_threshold is not None and patch_std < self.std_threshold
            )
        return False

    @staticmethod
    def filter_map(image: np.ndarray, patch_size: Sequence[int]) -> np.ndarray:
        """
        Compute the mean and std map of an image.

        The mean and std are computed over non-overlapping patches. This method can be
        used to assess a useful threshold for the MeanStd filter.

        Parameters
        ----------
        image : numpy.NDArray
            The full image to evaluate.
        patch_size : Sequence[int]
            The size of the patches to consider.

        Returns
        -------
        np.ndarray
            Stacked mean and std maps of the image.

        Raises
        ------
        ValueError
            If the image is not 2D or 3D.

        Example
        -------
        The `filter_map` method can be used to assess useful thresholds for the
        MeanStd filter.
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from careamics.dataset_ng.patch_filter import MeanStdPatchFilter
        >>> rng = np.random.default_rng(42)
        >>> image = rng.binomial(20, 0.1, (256, 256)).astype(np.float32)
        >>> image[64:192, 64:192] = rng.normal(50, 3, (128, 128))
        >>> image[96:160, 96:160] = rng.poisson(image[96:160, 96:160])
        >>> patch_size = (16, 16)
        >>> meanstd_map = MeanStdPatchFilter.filter_map(image, patch_size)
        >>> fig, ax = plt.subplots(3, 3, figsize=(10, 10)) # doctest: +SKIP
        >>> for i, mean_thresh in enumerate([48 + i for i in range(3)]):
        ...     for j, std_thresh in enumerate([5 + i for i in range(3)]):
        ...         ax[i, j].imshow(
        ...             (meanstd_map[0, ...] > mean_thresh)
        ...             & (meanstd_map[1, ...] > std_thresh),
        ...             cmap="gray", vmin=0, vmax=1
        ...         ) # doctest: +SKIP
        ...         ax[i, j].set_title(
        ...             f"Mean: {mean_thresh}, Std: {std_thresh}"
        ...         ) # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
        """
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError("Image must be 2D or 3D.")

        axes = "YX" if len(patch_size) == 2 else "ZYX"

        mean = np.zeros_like(image, dtype=float)
        std = np.zeros_like(image, dtype=float)

        extractor = create_array_extractor(source=[image], axes=axes)
        tiling = TilingStrategy(
            data_shapes=[(1, 1, *image.shape)],
            tile_size=patch_size,
            overlaps=(0,) * len(patch_size),  # no overlap
        )

        for idx in tqdm(range(tiling.n_patches), desc="Computing Mean/STD map"):
            patch_spec = tiling.get_patch_spec(idx)
            patch = extractor.extract_patch(
                data_idx=0,
                sample_idx=0,
                coords=patch_spec["coords"],
                patch_size=patch_size,
            )

            coordinates = tuple(
                slice(patch_spec["coords"][i], patch_spec["coords"][i] + p)
                for i, p in enumerate(patch_size)
            )
            mean[coordinates] = np.mean(patch)
            std[coordinates] = np.std(patch)

        return np.stack([mean, std], axis=0)

    @staticmethod
    def apply_filter(
        filter_map: np.ndarray,
        mean_threshold: float,
        std_threshold: float | None = None,
    ) -> np.ndarray:
        """
        Apply mean and std thresholds to a filter map.

        The filter map is the output of the `filter_map` method.

        Parameters
        ----------
        filter_map : np.ndarray
            Stacked mean and std maps of the image.
        mean_threshold : float
            Threshold for the mean of the patch.
        std_threshold : float | None, default=None
            Threshold for the standard deviation of the patch. If None, then no
            standard deviation filtering is applied.

        Returns
        -------
        np.ndarray
            A binary map where True indicates patches that pass the filter.
        """
        if std_threshold is not None:
            return (filter_map[0, ...] > mean_threshold) & (
                filter_map[1, ...] > std_threshold
            )
        return filter_map[0, ...] > mean_threshold
