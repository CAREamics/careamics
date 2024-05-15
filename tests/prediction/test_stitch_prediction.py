import pytest
from torch import from_numpy, tensor

from careamics.dataset.patching.tiled_patching import extract_tiles
from careamics.prediction.stitch_prediction import stitch_prediction


@pytest.mark.parametrize(
    "input_shape, tile_size, overlaps",
    [
        ((1, 1, 8, 8), (4, 4), (2, 2)),
        ((1, 1, 8, 8), (4, 4), (2, 2)),
        ((1, 1, 7, 9), (4, 4), (2, 2)),
        ((1, 1, 9, 7, 8), (4, 4, 4), (2, 2, 2)),
        ((1, 1, 321, 481), (256, 256), (48, 48)),
    ],
)
def test_stitch_prediction(ordered_array, input_shape, tile_size, overlaps):
    """Test calculating stitching coordinates."""
    arr = ordered_array(input_shape, dtype=int)
    tiles = []
    stitching_data = []

    # extract tiles
    tile_generator = extract_tiles(arr, tile_size, overlaps)

    # Assemble all tiles as it is done during the prediction stage
    for tile_data, tile_info in tile_generator:
        tiles.append(from_numpy(tile_data))  # need to convert to torch.Tensor
        stitching_data.append(
            (  # this is way too wacky
                [tensor(i) for i in input_shape],  # need to convert to torch.Tensor
                [[tensor([j]) for j in i] for i in tile_info.overlap_crop_coords],
                [[tensor([j]) for j in i] for i in tile_info.stitch_coords],
            )
        )

    # compute stitching coordinates, it returns a torch.Tensor
    result = stitch_prediction(tiles, stitching_data)

    assert (result.numpy() == arr).all()
