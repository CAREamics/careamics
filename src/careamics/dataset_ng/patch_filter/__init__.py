"""Patch filtering strategies."""

__all__ = [
    "MeanStdPatchFilter",
    "PatchFilterProtocol",
    "PercentilePatchFilter",
    "ShannonEntropyFilter",
]

from .mean_std_filter import MeanStdPatchFilter
from .patch_filter_protocol import PatchFilterProtocol
from .percentile_filter import PercentilePatchFilter
from .shannon_entropy_filter import ShannonEntropyFilter


# TODO:
# - Data mask?
# - Sampler for non random access with mask
# - Benchmark the different filters (CZI data, microssim, toy dataset)
# - Implement tests
# - Add config entries + all the way to careamics dataloader
# - Why not sampler?
# - Convenience functions to help decide on filter parameters or which filter