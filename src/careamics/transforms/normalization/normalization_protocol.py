from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class NormalizationProtocol(Protocol):
    """Protocol for normalization strategies."""

    def __call__(
        self,
        patch: np.ndarray,
        target: np.ndarray | None = None,
        **additional_arrays: np.ndarray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]: ...

    def denormalize(self, patch: np.ndarray) -> NDArray: ...
