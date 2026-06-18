import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.config.data import DataConfig, MeanStdConfig, TiledPatchingConfig
from careamics.dataset.image_region_data import ImageRegionData
from careamics.lightning.data import (
    CareamicsDataModule,
)
from careamics.lightning.prediction import (
    convert_prediction,
)


@pytest.fixture
def data_arrays(num_arrays: int, shape: tuple[int, ...]) -> list[NDArray[np.float32]]:
    """Fixture providing data arrays to test stitching tiled predictions."""
    data_arrays = [np.random.rand(*shape).astype(np.float32) for _ in range(num_arrays)]

    return data_arrays


@pytest.mark.parametrize(
    ["num_arrays", "shape", "axes", "pred_channels"],
    [
        (2, (64, 64), "YX", [0]),
        (2, (5, 64, 64), "CYX", [0, 1, 2, 3, 4]),
        (2, (5, 64, 64), "CYX", [0, 2]),
    ],
)
def test_stitch_tiled_prediction(
    data_arrays: list[NDArray[np.float32]],
    axes: str,
    pred_channels: list[int],
):
    """Test stitching of tiled predictions."""

    config = DataConfig(
        mode="predicting",
        data_type="array",
        axes=axes,
        patching=TiledPatchingConfig(
            patch_size=(16, 16),
            overlaps=(4, 4),
        ),
        batch_size=4,
        in_memory=True,
        seed=42,
        normalization=MeanStdConfig(
            input_means=[0],
            input_stds=[1],
        ),
    )

    datamodule = CareamicsDataModule(data_config=config, pred_data=data_arrays)
    datamodule.setup(stage="predict")

    predictions = []
    for i, tiles in enumerate(datamodule.predict_dataloader()):
        print(i, len(tiles), tiles[0].data.shape)
        predictions.append(
            ImageRegionData(
                source=tiles[0].source,
                data=tiles[0].data.numpy()[:, pred_channels],
                data_shape=tiles[0].data_shape,
                dtype=tiles[0].dtype,
                axes=tiles[0].axes,
                region_spec=tiles[0].region_spec,
                additional_metadata=tiles[0].additional_metadata,
                original_data_shape=tiles[0].original_data_shape,
            )
        )

    stitched_prediction, _ = convert_prediction(
        predictions, tiled=True, restore_shape=True
    )

    assert len(stitched_prediction) == len(data_arrays)

    for i in range(len(data_arrays)):
        if len(data_arrays[i].shape) > 2:
            # there is a channel dimension
            assert stitched_prediction[i].shape == data_arrays[i][pred_channels].shape
            assert np.allclose(stitched_prediction[i], data_arrays[i][pred_channels])
        else:
            assert stitched_prediction[i].shape == data_arrays[i].shape
            assert np.allclose(stitched_prediction[i], data_arrays[i])
