import numpy as np
import pytest
from torch import from_numpy

from careamics.dataset.patching import _extract_tiles
from careamics.prediction.prediction_utils import (
    stitch_prediction,
    tta_backward,
    tta_forward,
)


@pytest.mark.parametrize(
    "input_shape, tile_size, overlaps",
    [
        ((1, 8, 8), (4, 4), (2, 2)),
        ((1, 7, 9), (4, 4), (2, 2)),
        ((1, 9, 7, 8), (4, 4, 4), (2, 2, 2)),
    ],
)
def test_stitch_prediction(input_shape, ordered_array, tile_size, overlaps):
    """Test calculating stitching coordinates.

    Test cases include only valid inputs.
    """
    arr = ordered_array(input_shape, dtype=int)
    tiles = []
    stitching_data = []

    # extract tiles
    tiling_outputs = _extract_tiles(arr, tile_size, overlaps)

    # Assemble all tiles as it's done during the prediction stage
    for tile_data in tiling_outputs:
        tile, _, input_shape, overlap_crop_coords, stitch_coords = tile_data

        tiles.append(tile)
        stitching_data.append(
            (
                input_shape,
                overlap_crop_coords,
                stitch_coords,
            )
        )
    # compute stitching coordinates
    result = stitch_prediction(tiles, stitching_data)
    assert (result == arr).all()


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (1, 1, 8, 8, 8)])
def test_tta_forward(shape):
    """Test TTA forward."""
    n = np.prod(shape)
    x = np.arange(n).reshape(shape)

    # tta forward
    x_aug = tta_forward(from_numpy(x))

    # check output
    assert len(x_aug) == 8
    for i, x_ in enumerate(x_aug):
        # check correct shape
        assert x_.shape == shape

        # arrays different (at least from the previous one)
        if i > 0:
            assert (x_ != x_aug[i - 1]).any()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 4, 4),
        # (1, 1, 4, 4, 4)
    ],
)
def test_tta_backward(shape):
    """Test TTA backward."""
    n = np.prod(shape)
    x = np.arange(n).reshape(shape)

    # tta forward
    x_aug = tta_forward(from_numpy(x))

    # tta backward
    x_back = tta_backward(x_aug)

    # check that it returns the same array
    assert (x_back == x).all()
