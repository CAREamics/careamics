"""Patch filtering strategies."""

__all__ = [
    "CoordinateFilterProtocol",
    "MaskPatchFilter",
    "MaxPercentilePatchFilter",
    "MeanStdPatchFilter",
    "PatchFilterProtocol",
    "PercentilePatchFilter",
    "ShannonEntropyFilter",
]

from .coordinate_filter_protocol import CoordinateFilterProtocol
from .mask_filter import MaskPatchFilter
from .max_percentile_filter import MaxPercentilePatchFilter
from .mean_std_filter import MeanStdPatchFilter
from .patch_filter_protocol import PatchFilterProtocol
from .percentile_filter import PercentilePatchFilter
from .shannon_entropy_filter import ShannonEntropyFilter


# TODO:
# - Data mask? out of scoper here
# - Sampler for non random access with mask
# - Benchmark the different filters (CZI data, microssim, toy dataset)
# - Implement tests
# - Add config entries + all the way to careamics dataloader
# - Why not sampler?
# - Convenience functions to help decide on filter parameters or which filter
