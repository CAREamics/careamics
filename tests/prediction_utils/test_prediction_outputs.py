import numpy as np
import pytest

from careamics.dataset.tiling import extract_tiles
from careamics.prediction_utils import convert_outputs


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
