import numpy as np

from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol


class PercentilePatchFilter(PatchFilterProtocol):

    def __init__(
        self,
        low_p: float,
        high_p: float,
        threshold: float,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:

        if not (0 <= low_p <= 100):
            raise ValueError("Low percentile must be between 0 and 100.")
        if not (0 <= high_p <= 100):
            raise ValueError("High percentile must be between 0 and 100.")
        if low_p >= high_p:
            raise ValueError("Low percentile must be less than high percentile.")
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")

        self.low_percentile = low_p
        self.high_percentile = high_p
        self.threshold = threshold

        self.p = p
        self.rng = np.random.default_rng(seed)

    def filter_out(self, patch: np.ndarray) -> bool:
        if self.rng.uniform(0, 1) < self.p:
            p_low = np.percentile(patch, self.low_percentile)
            p_high = np.percentile(patch, self.high_percentile)

            return (p_high - p_low) < self.threshold
        return False
