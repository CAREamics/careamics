import numpy as np
from scipy.ndimage import maximum_filter

from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol


class MaxPercentilePatchFilter(PatchFilterProtocol):
    """
    A patch filter based on thresholding the maximum filter of the patch.

    Inspired by CSBDeep approach.

    Attributes
    ----------
    threshold : float
        Threshold for the maximum filter of the patch.
    p : float
        Probability of applying the filter to a patch.
    rng : np.random.Generator
        Random number generator for stochastic filtering.
    """

    def __init__(
        self,
        max_value: float,
        weight: float = 0.4,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:

        if not (0 <= weight <= 100):
            raise ValueError("Weight must be between 0 and 1.")

        self.threshold = max_value * weight

        self.p = p
        self.rng = np.random.default_rng(seed)

    def filter_out(self, patch: np.ndarray) -> bool:
        if self.rng.uniform(0, 1) < self.p:

            patch_shape = [(p // 2 if p > 1 else 1) for p in patch.shape]
            filtered = maximum_filter(patch, patch_shape, mode="constant")

            return (filtered < self.threshold).any()
        return False
