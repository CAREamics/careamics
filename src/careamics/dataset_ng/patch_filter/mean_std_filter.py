"""Filter using mean and standard deviation thresholds."""

import numpy as np

from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol


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
