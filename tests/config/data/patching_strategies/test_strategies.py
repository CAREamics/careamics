import pytest

from careamics.config.data.patching_strategies._overlapping_patched_model import (
    _OverlappingPatchedModel,
)
from careamics.config.data.patching_strategies._patched_model import _PatchedModel


@pytest.mark.parametrize(
    "patch_size",
    [
        [8, 14],  # not a power of 2
        [16, 32, 6],  # smaller than 8
        [64],  # less than 2 elements
        [64, 32, 32, 32],  # more than 3 elements
    ],
)
def test_patch_sizes_error(patch_size):
    """Test that the random patching"""
    with pytest.raises(ValueError):
        _PatchedModel(patch_size=patch_size)


@pytest.mark.parametrize(
    "patch_size, overlap",
    [
        ([32, 32], [16, 16, 16]),  # different lengths
        ([32, 32], [16, 48]),  # larger in one dims
        ([32, 32], [16, 32]),  # smaller in one dims
    ],
)
def test_patch_overlaps_error(patch_size, overlap):
    """Test that the random patching"""
    with pytest.raises(ValueError):
        _OverlappingPatchedModel(patch_size=patch_size, overlaps=overlap)
