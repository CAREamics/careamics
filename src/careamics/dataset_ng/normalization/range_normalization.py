from numpy.typing import NDArray

from .normalization_protocol import NormalizationProtocol
from .standardization import _reshape_stats


class RangeNormalization(NormalizationProtocol):
    """Normalize an image or image patch.

    Normalization is a range normalization. This transform expects C(Z)YX
    dimensions. Nomalizes to [0, 1].
    """

    def __init__(
        self,
        input_mins: list[float],
        input_maxs: list[float],
        target_mins: list[float] | None = None,
        target_maxs: list[float] | None = None,
    ):
        self.input_mins = input_mins
        self.input_maxs = input_maxs
        self.target_mins = target_mins
        self.target_maxs = target_maxs

    def __call__(
        self, patch: NDArray, target: NDArray | None = None
    ) -> tuple[NDArray, NDArray | None]:
        if len(self.input_mins) != patch.shape[0]:
            raise ValueError(
                f"Number of mins (got a list of size {len(self.input_mins)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )

        if len(self.input_maxs) != patch.shape[0]:
            raise ValueError(
                f"Number of max values (got a list of size {len(self.input_maxs)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )

        min_val = _reshape_stats(self.input_mins, patch.ndim)
        max_val = _reshape_stats(self.input_maxs, patch.ndim)

        norm_patch = (patch - min_val) / (max_val - min_val)

        norm_target = None
        if target is not None:
            if self.target_mins is None or self.target_maxs is None:
                raise ValueError(
                    "Target mins and maxs must be provided if target is not None."
                )
            if len(self.target_mins) != target.shape[0]:
                raise ValueError(
                    f"Number of mins (got a list of size {len(self.target_mins)}) and "
                    f"number of channels (got shape {target.shape} for C(Z)YX) do not match."
                )
            if len(self.target_maxs) != target.shape[0]:
                raise ValueError(
                    f"Number of max values (got a list of size {len(self.target_maxs)}) and "
                    f"number of channels (got shape {target.shape} for C(Z)YX) do not match."
                )
            target_mins = _reshape_stats(self.target_mins, target.ndim)
            target_maxs = _reshape_stats(self.target_maxs, target.ndim)
            norm_target = (target - target_mins) / (target_maxs - target_mins)

        return norm_patch, norm_target

    # TODO: check axes CZYX vs BCZYX!
    # TODO: check if we need to swap axes for CZYX case!
    def denormalize(self, patch: NDArray) -> NDArray:
        input_mins = _reshape_stats(self.input_mins, patch.ndim)
        input_maxs = _reshape_stats(self.input_maxs, patch.ndim)
        return patch * (input_maxs - input_mins) + input_mins
