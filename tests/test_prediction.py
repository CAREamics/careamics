import numpy as np
import pytest

# TODO: unused impots and incomplete tests


@pytest.mark.parametrize(
    "n_tiles, tile_size",
    [
        (4, (4, 4)),
    ],
)
@pytest.mark.parametrize("input_shape", [(8, 8)])
def test_stitch_prediction(n_tiles, tile_size, input_shape):
    """Test calculating stitching coordinates"""
    tile_coords = []
    np.zeros(input_shape, dtype=int)

    # create dummy tiles
    for y in range(0, input_shape[0] // tile_size[0]):
        for x in range(input_shape[1] // tile_size[1]):
            tile = np.ones(tile_size, dtype=int)
            ((y, y + tile_size[0]), (x, x + tile_size[0]))
            tile_coords.append((tile,))
    # TODO finish this test...........

    # # compute stitching coordinates
    # result = stitch_prediction(tile_coords, input_shape)
