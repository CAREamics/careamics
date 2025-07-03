from typing import Any

from numpy.typing import NDArray

from ..transform import Transform
from .normalization_protocol import NormalizationProtocol


class NoNormalization(NormalizationProtocol, Transform):
    """No normalization transform. Returns the patch as is."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(
        self,
        patch: NDArray,
        target: NDArray | None = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        return patch, target, additional_arrays

    def denormalize(self, patch: NDArray) -> NDArray:
        return patch
