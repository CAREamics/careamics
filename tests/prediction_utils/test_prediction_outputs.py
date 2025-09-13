import numpy as np
import pytest

from careamics.dataset.tiling import extract_tiles
from careamics.prediction_utils import convert_outputs
from careamics.prediction_utils.prediction_outputs import _combine_tiled_batches


@pytest.mark.parametrize("n_samples", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_convert_outputs_tiled(ordered_array, batch_size, n_samples):
    """Test conversion of output for when prediction is tiled"""
    # --- simulate outputs from trainer.predict
    tile_size = (4, 4)
    overlaps = (2, 2)
    arr = ordered_array((n_samples, 1, 16, 16))
    all_tiles = list(extract_tiles(arr, tile_size, overlaps))
    # combine into batches to match output of trainer.predict
    prediction_batches = []
    for i in range(0, len(all_tiles), batch_size):
        tiles = np.concatenate(
            [tile[np.newaxis] for tile, _ in all_tiles[i : i + batch_size]], axis=0
        )
        tile_infos = [tile_info for _, tile_info in all_tiles[i : i + batch_size]]
        prediction_batches.append((tiles, tile_infos))

    predictions = convert_outputs(prediction_batches, tiled=True)
    assert np.array_equal(np.stack(predictions, axis=0).squeeze(), arr.squeeze())


def test_combine_tiled_batches_out_of_order(ordered_array):
    """Test that tile ordering is preserved even when batches come back out of order"""
    # Create a simple 2x2 tile case
    tile_size = (4, 4)
    overlaps = (2, 2)
    arr = ordered_array((1, 1, 8, 8))  # Small array that creates 4 tiles (2x2 grid)
    all_tiles = list(extract_tiles(arr, tile_size, overlaps))

    # Create batches normally
    batch1_tiles = np.concatenate(
        [all_tiles[0][0][np.newaxis], all_tiles[1][0][np.newaxis]], axis=0
    )
    batch1_infos = [all_tiles[0][1], all_tiles[1][1]]

    batch2_tiles = np.concatenate(
        [all_tiles[2][0][np.newaxis], all_tiles[3][0][np.newaxis]], axis=0
    )
    batch2_infos = [all_tiles[2][1], all_tiles[3][1]]

    # Simulate out-of-order processing by swapping batch order
    predictions_out_of_order = [
        (batch2_tiles, batch2_infos),
        (batch1_tiles, batch1_infos),
    ]

    # Test that _combine_tiled_batches correctly reorders
    prediction_tiles, tile_infos = _combine_tiled_batches(predictions_out_of_order)

    # Verify that tiles are in correct order by checking stitch coordinates
    for i in range(len(tile_infos) - 1):
        current_coords = tile_infos[i].stitch_coords
        next_coords = tile_infos[i + 1].stitch_coords

        # For a 2x2 grid, tiles should be ordered: (0,0), (0,X), (Y,0), (Y,X)
        # where the first coordinate is Y and second is X
        current_key = (current_coords[0][0], current_coords[1][0])  # (Y_start, X_start)
        next_key = (next_coords[0][0], next_coords[1][0])

        # Should be in row-major order
        assert (
            current_key <= next_key
        ), f"Tiles not in correct order: {current_key} vs {next_key}"


@pytest.mark.parametrize("batch_size, n_samples", [(1, 1), (1, 2), (2, 2)])
def test_convert_outputs_not_tiled(ordered_array, batch_size, n_samples):
    """Test conversion of output for when prediction is not tiled"""
    # --- simulate outputs from trainer.predict
    prediction_batches = [
        ordered_array((batch_size, 1, 16, 16)) for _ in range(n_samples // batch_size)
    ]
    predictions = convert_outputs(prediction_batches, tiled=False)
    assert np.array_equal(
        # stack predictions because there is no S axis
        # squeeze to remove singleton S or C axes
        np.stack(predictions, axis=0).squeeze(),
        np.concatenate(prediction_batches, axis=0).squeeze(),
    )
