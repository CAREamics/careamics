from numpy.typing import NDArray

from .normalization_protocol import NormalizationProtocol
from .standardization import _reshape_stats


class RangeNormalization(NormalizationProtocol):
    """Normalize an image or image patch.

    Normalization is a range normalization. This transform expects C(Z)YX
    dimensions. Normalizes to [0, 1].
    """

    def __init__(
        self,
        input_mins: list[float],
        input_maxes: list[float],
        target_mins: list[float] | None = None,
        target_maxes: list[float] | None = None,
    ):
        self.input_mins = input_mins
        self.input_maxes = input_maxes
        self.target_mins = target_mins
        self.target_maxes = target_maxes

    def __call__(
        self, patch: NDArray, target: NDArray | None = None
    ) -> tuple[NDArray, NDArray | None]:
        if len(self.input_mins) != patch.shape[0]:
            raise ValueError(
                f"Number of mins (got a list of size {len(self.input_mins)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )

        if len(self.input_maxes) != patch.shape[0]:
            raise ValueError(
                f"Number of max values (got a list of size {len(self.input_maxes)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )

        min_val = _reshape_stats(self.input_mins, patch.ndim)
        max_val = _reshape_stats(self.input_maxes, patch.ndim)

        norm_patch = (patch - min_val) / (max_val - min_val)

        norm_target = None
        if target is not None:
            if self.target_mins is None or self.target_maxes is None:
                raise ValueError(
                    "Target mins and maxs must be provided if target is not None."
                )
            if len(self.target_mins) != target.shape[0]:
                raise ValueError(
                    f"Number of mins (got a list of size {len(self.target_mins)}) and "
                    f"number of channels (got shape {target.shape} for C(Z)YX) do not match."
                )
            if len(self.target_maxes) != target.shape[0]:
                raise ValueError(
                    f"Number of max values (got a list of size {len(self.target_maxes)}) and "
                    f"number of channels (got shape {target.shape} for C(Z)YX) do not match."
                )
            target_mins = _reshape_stats(self.target_mins, target.ndim)
            target_maxes = _reshape_stats(self.target_maxes, target.ndim)
            norm_target = (target - target_mins) / (target_maxes - target_mins)

        return norm_patch, norm_target

    # TODO: check axes CZYX vs BCZYX!
    # TODO: check if we need to swap axes for CZYX case!
    def denormalize(self, patch: NDArray) -> NDArray:
        input_mins = _reshape_stats(self.input_mins, patch.ndim)
        input_maxes = _reshape_stats(self.input_maxes, patch.ndim)
        return patch * (input_maxes - input_mins) + input_mins
