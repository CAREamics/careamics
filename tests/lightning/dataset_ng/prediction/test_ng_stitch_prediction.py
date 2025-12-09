import random

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.image_stack_loader import load_arrays
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patching_strategies import TileSpecs, TilingStrategy
from careamics.lightning.dataset_ng.prediction.stitch_prediction import (
    group_tiles_by_key,
    stitch_prediction,
    stitch_single_prediction,
    stitch_single_sample,
)

# TODO will be redundant once tiling_strategy tests will not use the old stitching


# TODO should be in conftest now, it needs to be merged
@pytest.fixture
def tiles(n_data, shape, axes) -> tuple[NDArray, list[ImageRegionData]]:
    # create data
    array = np.arange(n_data * np.prod(shape)).reshape((n_data, *shape))

    # create tiling strategy
    if "Z" in axes:
        tile_size = (8, 16, 16)
        overlaps = (2, 4, 4)
    else:
        tile_size = (16, 16)
        overlaps = (4, 4)

    if "S" in axes:
        if "C" in axes:
            shape_with_sc = shape
        else:
            shape_with_sc = (shape[0], 1, *shape[1:])
    else:
        if "C" in axes:
            shape_with_sc = (1, *shape)
        else:
            shape_with_sc = (1, 1, *shape)

    tiling_strategy = TilingStrategy(
        data_shapes=[shape_with_sc] * n_data, patch_size=tile_size, overlaps=overlaps
    )
    n_tiles = tiling_strategy.n_patches

    # create patch extractor
    image_stacks = load_arrays(source=[array[i] for i in range(n_data)], axes=axes)
    patch_extractor = PatchExtractor(image_stacks)

    # extract tiles
    tiles: list[ImageRegionData] = []
    for i in range(n_tiles):
        tile_spec: TileSpecs = tiling_strategy.get_patch_spec(i)
        tile = patch_extractor.extract_patch(
            data_idx=tile_spec["data_idx"],
            sample_idx=tile_spec["sample_idx"],
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )
        tiles.append(
            ImageRegionData(
                data=tile,
                source=str(tile_spec["data_idx"]),  # for testing purposes
                dtype=str(tile.dtype),
                data_shape=shape,
                axes=axes,
                region_spec=tile_spec,
                additional_metadata={},
            )
        )
    return array, tiles


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1, 3])
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((32, 32), "YX", None),
    ],
)
def test_group_tiles_by_data_index(tiles):
    array, tile_list = tiles
    n_data = array.shape[0]

    sorted_tiles = group_tiles_by_key(tile_list, key="data_idx")
    assert len(sorted_tiles.keys()) == n_data
    n_tiles_per_data = [len(sorted_tiles[data_idx]) for data_idx in sorted_tiles.keys()]
    assert all(n_tiles_per_data[0] == n_tiles_per_data[i] for i in range(1, n_data))


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize(
    "n_data, shape, axes, channels",
    [
        (1, (5, 32, 32), "SYX", None),
    ],
)
def test_group_tiles_by_sample_index(tiles):
    array, tile_list = tiles
    n_data = array.shape[1]

    sorted_tiles = group_tiles_by_key(tile_list, key="sample_idx")
    assert len(sorted_tiles.keys()) == n_data
    n_tiles_per_data = [len(sorted_tiles[data_idx]) for data_idx in sorted_tiles.keys()]
    assert all(n_tiles_per_data[0] == n_tiles_per_data[i] for i in range(1, n_data))


@pytest.mark.parametrize("n_data", [1])
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((32, 32), "YX", None),
        ((3, 32, 32), "CYX", None),
        ((1, 32, 32), "CYX", None),
        ((8, 32, 32), "ZYX", None),
        ((1, 8, 32, 32), "CZYX", None),
        ((3, 8, 32, 32), "CZYX", None),
    ],
)
def test_stitching_single_sample(tiles):
    array, tile_list = tiles
    array = array[0]  # remove sample dim

    stitched_array = stitch_single_sample(tile_list)
    np.testing.assert_array_equal(stitched_array, array)


@pytest.mark.parametrize("n_data", [1])
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((1, 32, 32), "CYX", [0]),
        ((3, 32, 32), "CYX", [1]),
        ((3, 32, 32), "CYX", [0, 2]),
        ((1, 8, 32, 32), "CZYX", [0]),
        ((3, 8, 32, 32), "CZYX", [1]),
        ((3, 8, 32, 32), "CZYX", [0, 2]),
    ],
)
def test_stitching_single_sample_w_channels(tiles, channels):
    array, tile_list = tiles
    array = array[0][channels]  # remove sample dim and adjust channels

    stitched_array = stitch_single_sample(tile_list)
    np.testing.assert_array_equal(stitched_array, array)


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1])
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((32, 32), "YX", None),
        ((3, 32, 32), "CYX", None),
        ((5, 32, 32), "SYX", None),
        ((1, 32, 32), "CYX", None),
        ((1, 32, 32), "SYX", None),
        ((8, 32, 32), "ZYX", None),
        ((3, 8, 32, 32), "CZYX", None),
        ((5, 3, 32, 32), "SCYX", None),
        ((1, 8, 32, 32), "CZYX", None),
        ((1, 3, 32, 32), "SCYX", None),
        ((5, 8, 32, 32), "SZYX", None),
        ((5, 3, 8, 32, 32), "SCZYX", None),
    ],
)
def test_stitching_single_prediction(tiles):
    array, tile_list = tiles

    stitched_array = stitch_single_prediction(tile_list)
    np.testing.assert_array_equal(stitched_array.squeeze(), array.squeeze())


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1])
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((1, 32, 32), "CYX", [0]),
        ((3, 32, 32), "CYX", [1]),
        ((3, 32, 32), "CYX", [0, 2]),
        ((5, 1, 32, 32), "SCYX", [0]),
        ((5, 3, 32, 32), "SCYX", [1]),
        ((5, 3, 32, 32), "SCYX", [0, 2]),
        ((1, 8, 32, 32), "CZYX", [0]),
        ((3, 8, 32, 32), "CZYX", [1]),
        ((3, 8, 32, 32), "CZYX", [0, 2]),
        ((5, 1, 8, 32, 32), "SCZYX", [0]),
        ((5, 3, 8, 32, 32), "SCZYX", [1]),
        ((5, 3, 8, 32, 32), "SCZYX", [0, 2]),
    ],
)
def test_stitching_single_prediction_w_channels(tiles, axes, channels):
    array, tile_list = tiles

    # remove data dim, adjust channels
    array = array[0]
    if "S" in axes:
        array_w_channels = np.take(array, indices=channels, axis=1)
    else:
        array_w_channels = np.take(array, indices=channels, axis=0)

    stitched_array = stitch_single_prediction(tile_list)
    np.testing.assert_array_equal(stitched_array, array_w_channels.squeeze())


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [4])
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((32, 32), "YX", None),
        ((3, 32, 32), "CYX", None),
        ((1, 32, 32), "CYX", None),
        ((5, 32, 32), "SYX", None),
        ((1, 32, 32), "SYX", None),
        ((8, 32, 32), "ZYX", None),
        ((3, 8, 32, 32), "CZYX", None),
        ((5, 3, 32, 32), "SCYX", None),
        ((1, 1, 32, 32), "SCYX", None),
        ((5, 8, 32, 32), "SZYX", None),
        ((5, 3, 8, 32, 32), "SCZYX", None),
    ],
)
def test_stitching_prediction(tiles):
    array, tile_list = tiles

    stitched_arrays, data_indices = stitch_prediction(tile_list)

    prediction = np.stack(stitched_arrays)
    np.testing.assert_array_equal(prediction, array.squeeze())

    assert data_indices == [str(i) for i in range(array.shape[0])]


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1, 4])
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((1, 32, 32), "CYX", [0]),
        ((3, 32, 32), "CYX", [0, 2]),
        ((5, 1, 32, 32), "SCYX", [0]),
        ((1, 1, 32, 32), "SCYX", [0]),
        ((5, 3, 32, 32), "SCYX", [0, 2]),
        ((1, 3, 32, 32), "SCYX", [0, 2]),
        ((1, 8, 32, 32), "CZYX", [0]),
        ((3, 8, 32, 32), "CZYX", [0, 2]),
        ((5, 1, 8, 32, 32), "SCZYX", [0]),
        ((5, 3, 8, 32, 32), "SCZYX", [0, 2]),
        ((1, 1, 8, 32, 32), "SCZYX", [0]),
        ((1, 3, 8, 32, 32), "SCZYX", [0, 2]),
    ],
)
def test_stitching_prediction_w_channels(tiles, axes, channels):
    array, tile_list = tiles

    if "S" in axes:
        array_w_channels = np.take(array, indices=channels, axis=2)
    else:
        array_w_channels = np.take(array, indices=channels, axis=1)

    stitched_arrays, _ = stitch_prediction(tile_list)

    prediction = np.stack(stitched_arrays)
    np.testing.assert_array_equal(prediction.squeeze(), array_w_channels.squeeze())


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [4])
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((32, 32), "YX", None),
    ],
)
def test_stitching_prediction_ordering(tiles):
    array, tile_list = tiles

    # shuffle tiles to test ordering
    random.shuffle(tile_list)

    stitched_arrays, sources = stitch_prediction(tile_list)

    # check that data indices are in order
    for i, data_idx_str in enumerate(sources):
        assert data_idx_str == str(i)
        np.testing.assert_array_equal(stitched_arrays[i], array[i])
