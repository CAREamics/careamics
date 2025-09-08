"""Patch filtering strategies."""

__all__ = [
    "CoordinateFilterProtocol",
    "MaskPatchFilter",
    "MaxPatchFilter",
    "MeanStdPatchFilter",
    "PatchFilterProtocol",
    "ShannonEntropyFilter",
]

from .coordinate_filter_protocol import CoordinateFilterProtocol
from .mask_filter import MaskPatchFilter
from .max_patch_filter import MaxPatchFilter
from .mean_std_filter import MeanStdPatchFilter
from .patch_filter_protocol import PatchFilterProtocol
from .shannon_entropy_filter import ShannonEntropyFilter


# TODO:
# - Add config entries + all the way to careamics dataloader
# - Entry point for mask
