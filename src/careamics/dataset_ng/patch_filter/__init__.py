"""Patch filtering strategies."""

__all__ = [
    "CoordinateFilterProtocol",
    "MaskPatchFilter",
    "MaxPercentilePatchFilter",
    "MeanStdPatchFilter",
    "PatchFilterProtocol",
    "ShannonEntropyFilter",
]

from .coordinate_filter_protocol import CoordinateFilterProtocol
from .mask_filter import MaskPatchFilter
from .max_percentile_filter import MaxPercentilePatchFilter
from .mean_std_filter import MeanStdPatchFilter
from .patch_filter_protocol import PatchFilterProtocol
from .shannon_entropy_filter import ShannonEntropyFilter


# TODO:
# - Benchmark the different filters (CZI data, microssim, toy dataset)
# - Implement tests
# - Add config entries + all the way to careamics dataloader
# - Convenience functions to help decide on filter parameters or which filter
