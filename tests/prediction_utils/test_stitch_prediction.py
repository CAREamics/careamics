import numpy as np
import pytest

from careamics.dataset.tiling import extract_tiles
from careamics.prediction_utils import stitch_prediction, stitch_prediction_single


@pytest.mark.parametrize(
    "input_shape, tile_size, overlaps",
    [
        ((1, 1, 8, 8), (4, 4), (2, 2)),
        ((1, 2, 8, 8), (4, 4), (2, 2)),
        ((2, 1, 8, 8), (4, 4), (2, 2)),
        ((2, 2, 8, 8), (4, 4), (2, 2)),
        ((1, 1, 7, 9), (4, 4), (2, 2)),
        ((1, 3, 7, 9), (4, 4), (2, 2)),
        ((1, 1, 9, 7, 8), (4, 4, 4), (2, 2, 2)),
        ((1, 1, 321, 481), (256, 256), (48, 48)),
        ((2, 1, 321, 481), (256, 256), (48, 48)),
        ((1, 4, 321, 481), (256, 256), (48, 48)),
        ((4, 3, 321, 481), (256, 256), (48, 48)),
    ],
)
def test_stitch_tiles_single(ordered_array, input_shape, tile_size, overlaps):
    """Test stitching tiles back together."""
    arr = ordered_array(input_shape, dtype=int)
    n_samples = input_shape[0]

    # extract tiles
    all_tiles = list(extract_tiles(arr, tile_size, overlaps))

    tiles = []
    tile_infos = []
    sample_id = 0
    for tile, tile_info in all_tiles:
        # create lists mimicking the output of the prediction loop
        tiles.append(tile)
        tile_infos.append(tile_info)

        # if we reached the last tile
        if tile_info.last_tile:
            result = stitch_prediction_single(tiles, tile_infos)

            # check equality with the correct sample
            assert np.array_equal(result, arr[[sample_id]])
            sample_id += 1

            # clear the lists
            tiles.clear()
            tile_infos.clear()

    assert sample_id == n_samples


@pytest.mark.parametrize(
    "input_shape, tile_size, overlaps",
    [
        ((1, 1, 8, 8), (4, 4), (2, 2)),
        ((1, 2, 8, 8), (4, 4), (2, 2)),
        ((2, 1, 8, 8), (4, 4), (2, 2)),
        ((2, 2, 8, 8), (4, 4), (2, 2)),
        ((1, 1, 7, 9), (4, 4), (2, 2)),
        ((1, 3, 7, 9), (4, 4), (2, 2)),
        ((1, 1, 9, 7, 8), (4, 4, 4), (2, 2, 2)),
        ((1, 1, 321, 481), (256, 256), (48, 48)),
        ((2, 1, 321, 481), (256, 256), (48, 48)),
        ((1, 4, 321, 481), (256, 256), (48, 48)),
        ((4, 3, 321, 481), (256, 256), (48, 48)),
    ],
)
def test_stitch_tiles_multi(ordered_array, input_shape, tile_size, overlaps):
    """Test stitching tiles back together."""
    arr = ordered_array(input_shape, dtype=int)
    n_samples = input_shape[0]

    # extract tiles
    all_tiles = list(extract_tiles(arr, tile_size, overlaps))

    tiles = []
    tile_infos = []
    for tile, tile_info in all_tiles:
        # create lists mimicking the output of the prediction loop
        tiles.append(tile)
        tile_infos.append(tile_info)

    stitched = stitch_prediction(tiles, tile_infos)
    for sample_id, result in enumerate(stitched):
        assert np.array_equal(result, arr[[sample_id]])

    assert len(stitched) == n_samples
