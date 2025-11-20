import numpy as np

from careamics.dataset_ng.image_stack_loader import load_arrays
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patch_filter import MaskCoordFilter
from careamics.dataset_ng.patching_strategies import PatchSpecs

# TODO test probability of application


def test_filter():
    """Test MaskPatchFilter functionality."""
    size = 16
    mask = np.zeros((size, size))
    mask[size // 4 : -size // 4, size // 4 : -size // 4] = 1

    image_stacks = load_arrays(source=[mask], axes="YX")
    mask_filter = MaskCoordFilter(
        mask_extractor=PatchExtractor(image_stacks),
        coverage=0.50,
    )
    assert mask_filter.filter_out(
        PatchSpecs(data_idx=0, sample_idx=0, coords=[0, 0], patch_size=[4, 4])
    )  # corner, no overlap with mask
    assert mask_filter.filter_out(
        PatchSpecs(data_idx=0, sample_idx=0, coords=[2, 2], patch_size=[4, 4])
    )  # 25% overlap
    assert not mask_filter.filter_out(
        PatchSpecs(data_idx=0, sample_idx=0, coords=[2, 4], patch_size=[4, 4])
    )  # 50% overlap
    assert not mask_filter.filter_out(
        PatchSpecs(data_idx=0, sample_idx=0, coords=[4, 4], patch_size=[4, 4])
    )  # 100% overlap
