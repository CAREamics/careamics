import numpy as np
import pytest

from careamics_restoration.dataset.tiling import extract_tiles
from careamics_restoration.prediction_utils import stitch_prediction


@pytest.mark.parametrize(
    "tile_size, overlaps",
    [
        ((4, 4), (2, 2)),
    ],
)
@pytest.mark.parametrize("input_shape", [(1, 8, 8), (1, 7, 9)])
def test_stitch_prediction(input_shape, tile_size, overlaps):
    """Test calculating stitching coordinates.

    Test cases include only valid inputs.
    """
    arr = np.zeros(input_shape, dtype=int)
    stitching_data = []
    # extract tiles
    tiling_outputs = extract_tiles(arr, tile_size, overlaps)
    # Assemble all tiles as it's done during the prediction stage
    for tile_data in tiling_outputs:
        tile, _, _, overlap_crop_coords, stitch_coords = tile_data
        stitching_data.append(
            (
                tile,
                overlap_crop_coords,
                stitch_coords,
            )
        )
    # compute stitching coordinates
    result = stitch_prediction(stitching_data, input_shape)
    assert result.shape == input_shape
