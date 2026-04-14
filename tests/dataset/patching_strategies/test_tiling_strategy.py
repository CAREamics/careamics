import numpy as np
import pytest

from careamics.dataset.patching_strategies import TilingStrategy


@pytest.mark.parametrize("overlaps", [(2, 2), (3, 4), (6, 3)])
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 8)],
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 5)],
    ],
)
def test_whole_image_covered_2d(
    data_shapes: list[tuple[int, int, int, int]],
    patch_size: tuple[int, int],
    overlaps: tuple[int, int],
):
    patching_strategy = TilingStrategy(data_shapes, patch_size, overlaps)
    patch_specs = patching_strategy.tile_specs

    # track where patches have been sampled from
    tracking_arrays = [np.zeros(data_shape, dtype=bool) for data_shape in data_shapes]
    for patch_spec in patch_specs:
        tracking_array = tracking_arrays[patch_spec["data_idx"]]
        spatial_slice = tuple(
            slice(c, c + ps)
            for c, ps in zip(
                patch_spec["coords"], patch_spec["patch_size"], strict=False
            )
        )
        # set to true where the patches would be sampled from
        tracking_array[(patch_spec["sample_idx"], slice(None), *spatial_slice)] = True

    for tracking_array in tracking_arrays:
        # if the patch specs covered all the image all the values should be true
        assert tracking_array.all()


@pytest.mark.parametrize("overlaps", [(2, 2, 2), (3, 4, 2), (6, 3, 5)])
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 8, 8)],
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 5, 7)],
    ],
)
def test_whole_image_covered_3d(
    data_shapes: list[tuple[int, int, int, int]],
    patch_size: tuple[int, int],
    overlaps: tuple[int, int],
):
    patching_strategy = TilingStrategy(data_shapes, patch_size, overlaps)
    patch_specs = patching_strategy.tile_specs

    # track where patches have been sampled from
    tracking_arrays = [np.zeros(data_shape, dtype=bool) for data_shape in data_shapes]
    for patch_spec in patch_specs:
        tracking_array = tracking_arrays[patch_spec["data_idx"]]
        spatial_slice = tuple(
            slice(c, c + ps)
            for c, ps in zip(
                patch_spec["coords"], patch_spec["patch_size"], strict=False
            )
        )
        # set to true where the patches would be sampled from
        tracking_array[(patch_spec["sample_idx"], slice(None), *spatial_slice)] = True

    for tracking_array in tracking_arrays:
        # if the patch specs covered all the image all the values should be true
        assert tracking_array.all()
