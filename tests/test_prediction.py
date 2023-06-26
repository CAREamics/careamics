import pytest
import itertools
import numpy as np

from careamics_restoration.prediction_utils import stitch_prediction
from careamics_restoration.dataloader.dataloader_utils import (
    compute_crop_and_stitch_coords_1d,
)

# TODO: unused impots and incomplete tests

[
    (48, 48),
    (48, 32),
    (48, 32),
    (48, 29),
    (32, 48),
    (32, 32),
    (32, 32),
    (32, 29),
    (32, 48),
    (32, 32),
    (32, 32),
    (32, 29),
    (45, 48),
    (45, 32),
    (),
]


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
    out = np.zeros(input_shape, dtype=int)

    # create dummy tiles
    for y in range(0, input_shape[0] // tile_size[0]):
        for x in range(input_shape[1] // tile_size[1]):
            tile = np.ones(tile_size, dtype=int)
            stitch_coords = ((y, y + tile_size[0]), (x, x + tile_size[0]))
            tile_coords.append((tile,))
    # TODO finish this test...........

    # # compute stitching coordinates
    # result = stitch_prediction(tile_coords, input_shape)
