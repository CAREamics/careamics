import numpy as np
from numpy.typing import NDArray

from ..transform import Transform
from .normalization_protocol import NormalizationProtocol
from .standardization import _reshape_stats


class RangeNormalization(NormalizationProtocol, Transform):
    """Normalize an image or image patch.

    Normalization is a range normalization. This transform expects C(Z)YX
    dimensions. Nomalizes to [0, 1].
    """

    def __init__(self, image_mins: list[float], image_maxs: list[float]):
        self.min_val = image_mins
        self.max_val = image_maxs 

    def __call__(
        self,
        patch: NDArray,
        target: NDArray | None = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        if len(self.min_val) != patch.shape[0]:
            raise ValueError(
                f"Number of mins (got a list of size {len(self.min_val)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )

        if len(self.max_val) != patch.shape[0]:
            raise ValueError(
                f"Number of max values (got a list of size {len(self.max_val)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )

        if len(additional_arrays) != 0:
            raise NotImplementedError(
                "Transforming additional arrays is currently not supported for "
                "`Normalize`."
            )
            
        min_val = _reshape_stats(self.min_val, patch.ndim)
        max_val = _reshape_stats(self.max_val, patch.ndim)

        norm_patch = (patch - min_val) / (max_val - min_val)

        norm_target = None
        if target is not None:  
            norm_target = (target - min_val) / (max_val - min_val)

        return norm_patch, norm_target, additional_arrays

    
    # TODO: check axes CZYX vs BCZYX! 
    # TODO: check if we need to swap axes for CZYX case!
    def denormalize(self, patch: NDArray) -> NDArray:
        min_val = _reshape_stats(self.min_val, patch.ndim)
        max_val = _reshape_stats(self.max_val, patch.ndim)
        return patch * (max_val - min_val) + min_val
