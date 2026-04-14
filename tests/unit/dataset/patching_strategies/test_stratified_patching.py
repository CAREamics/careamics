import itertools

import numpy as np
import pytest

from careamics.dataset.patching_strategies import StratifiedPatchingStrategy
from careamics.dataset.patching_strategies.stratified_patching import (
    _boxes_overlap,
    _ImageStratifiedPatching,
    _region_bin_packing,
)


# TODO: add unit tests for all StratifiedPatching methods.
class TestStratifiedPatching:

    @pytest.fixture(
        params=[
            [(1, 1, 512, 512), (2, 1, 531, 591)],
            [(1, 1, 512, 512, 512), (2, 1, 531, 591, 554)],
        ]
    )
    def stratified_patching(self, request) -> StratifiedPatchingStrategy:
        data_shapes: list[tuple[int, ...]] = request.param
        ndims = len(data_shapes[0]) - 2
        patch_size = (64,) * ndims
        stratified_patching = StratifiedPatchingStrategy(data_shapes, patch_size, 42)
        return stratified_patching

    @pytest.mark.parametrize("rand_seed", [42, 666, 1])
    def test_set_region_probs(
        self, stratified_patching: StratifiedPatchingStrategy, rand_seed: int
    ):
        n_patches = stratified_patching.n_patches
        grid_coords = stratified_patching.get_all_grid_coords()
        # choose random probabilities for each region
        rng = np.random.default_rng(rand_seed)
        expected_patch_reduction = 0
        for (data_idx, sample_idx), coords in grid_coords.items():
            probs_array = rng.random(len(coords))
            expected_patch_reduction += np.sum(1 - probs_array)
            probs = dict(zip(coords, probs_array, strict=True))
            stratified_patching.set_region_probs(data_idx, sample_idx, probs)

        reduced_patches = stratified_patching.n_patches
        # not exact due to rounding on n_patches
        assert abs(reduced_patches - (n_patches - expected_patch_reduction)) < 1


# TODO: add unit tests for all _ImageStratifiedPatching methods.
class TestImageStratifiedPatch:

    @pytest.fixture(params=[(512, 512), (531, 591), (512, 512, 512), (531, 591, 554)])
    def image_patching_grid_coords(
        self,
        request,
    ) -> tuple[_ImageStratifiedPatching, list[tuple[int, ...]]]:
        data_shape: tuple[int, ...] = request.param
        patch_size = (64,) * len(data_shape)
        rng = np.random.default_rng(42)
        image_patching = _ImageStratifiedPatching(data_shape, patch_size, rng)
        grid_shape = image_patching.grid_shape
        grid_coords = list(itertools.product(*[range(gs) for gs in grid_shape]))
        return image_patching, grid_coords

    @pytest.mark.parametrize("rand_seed", [42, 666, 1])
    def test_set_region_probs(
        self,
        image_patching_grid_coords: tuple[
            _ImageStratifiedPatching, list[tuple[int, ...]]
        ],
        rand_seed: int,
    ):
        image_patching, grid_coords = image_patching_grid_coords
        n_patches = image_patching.n_patches

        # choose random probabilities for each region
        rng = np.random.default_rng(rand_seed)
        probs_array = rng.random(len(grid_coords))
        expected_patch_reduction = np.sum(1 - probs_array)
        probs = dict(zip(grid_coords, probs_array, strict=True))

        # set region probabilities and get the reduced n_patches
        image_patching.set_region_probs(probs)
        reduced_patches = image_patching.n_patches

        # not exact due to rounding on n_patches
        assert abs(reduced_patches - (n_patches - expected_patch_reduction)) < 1


@pytest.mark.parametrize(
    "box_a,box_b,overlapping",
    [
        (np.array([[0, 1]]), np.array([[0, 1]]), True),
        (np.array([[0, 1]]), np.array([[0.25, 0.75]]), True),
        (np.array([[0, 1]]), np.array([[0.5, 1.5]]), True),
        (np.array([[0, 1], [0, 1]]), np.array([[0.5, 1.5], [0.5, 1.5]]), True),
        (np.array([[0, 1]]), np.array([[2, 3]]), False),
        (np.array([[0, 1], [0, 1]]), np.array([[2, 3], [2, 3]]), False),
    ],
)
def test_boxes_overlap(
    box_a: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    box_b: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    overlapping: bool,
):
    assert _boxes_overlap(box_a, box_b) == overlapping


@pytest.mark.parametrize(
    "areas,n_patches,expected_bin_size",
    [
        [
            [2048, 2048, 2048, 1096, 1096, 1096, 1024, 1024, 1024, 1024],  # 10 areas
            12,  # n_patches greater than n_areas
            2048,
        ],
        [
            [2048, 2048, 2048, 1096, 1096, 1096, 1024, 1024, 1024, 1024],  # 10 areas
            8,
            2048,
        ],
        [
            [2048, 2048, 2048, 1096, 1096, 1096, 1024, 1024, 1024, 1024],  # 10 areas
            7,
            1096 + 1024,
        ],
        [
            [2048, 2048, 2048, 1096, 1096, 1096, 1024, 1024, 1024, 1024],  # 10 areas
            0,
            0,
        ],
    ],
)
def test_region_bin_packing(
    areas: list[int],
    n_patches: int,
    expected_bin_size: float,
):
    """Test that the bin packing algorithm produces the expected bin size."""
    areas_dict = dict(enumerate(areas))
    bin_size, bins = _region_bin_packing(areas_dict, n_patches)
    assert len(bins) == n_patches
    assert bin_size == expected_bin_size
