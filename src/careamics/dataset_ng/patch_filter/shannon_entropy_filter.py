import numpy as np
from skimage.measure import shannon_entropy

from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol


class ShannonEntropyFilter(PatchFilterProtocol):
    """
    Filter patches based on Shannon entropy threshold.

    Attributes
    ----------
    threshold : float
        Threshold for the Shannon entropy of the patch.
    p : float
        Probability of applying the filter to a patch.
    rng : np.random.Generator
        Random number generator for stochastic filtering.
    """

    def __init__(
        self, threshold: float, p: float = 1.0, seed: int | None = None
    ) -> None:
        """
        Create a ShannonEntropyFilter.

        This filter removes patches whose Shannon entropy is below a specified
        threshold.

        Parameters
        ----------
        threshold : float
            Threshold for the Shannon entropy of the patch.
        p : float, default=1
            Probability of applying the filter to a patch. Must be between 0 and 1.
        seed : int | None, default=None
            Seed for the random number generator for reproducibility.

        Raises
        ------
        ValueError
            If threshold is negative.
        ValueError
            If p is not between 0 and 1.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative.")
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")

        self.threshold = threshold

        self.p = p
        self.rng = np.random.default_rng(seed)

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
        if self.rng.uniform(0, 1) < self.p:
            return shannon_entropy(patch) < self.threshold
        return False
