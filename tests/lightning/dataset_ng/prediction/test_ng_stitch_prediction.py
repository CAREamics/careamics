import random

import numpy as np
import pytest

from careamics.lightning.dataset_ng.prediction.stitch_prediction import (
    group_tiles_by_key,
    stitch_prediction,
    stitch_single_prediction,
    stitch_single_sample,
)


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
