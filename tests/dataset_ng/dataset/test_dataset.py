from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.config import InferenceConfig, create_n2n_configuration
from careamics.dataset_ng.dataset.dataset import CareamicsDataset, Mode


# TODO: these tests not deterministic anymore - DataConfig doesn't have random seed
# could make a mock? temporarily
@pytest.mark.parametrize(
    "data_shape, patch_size, expected_dataset_len",
    [
        ((256, 256), (32, 32), 64),
        ((512, 512), (64, 64), 64),
        ((128, 128), (32, 32), 16),
    ],
)
def test_from_array(data_shape, patch_size, expected_dataset_len):
    rng = np.random.default_rng(42)
    example_input = rng.random(data_shape)
    example_target = rng.random(data_shape)

    train_data_config = create_n2n_configuration(
        "test_exp",
        data_type="array",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        num_epochs=1,
    ).data_config

    train_dataset = CareamicsDataset(
        data_config=train_data_config,
        mode=Mode.TRAINING,
        inputs=[example_input],
        targets=[example_target],
    )

    assert len(train_dataset) == expected_dataset_len
    sample, target = train_dataset[0]
    assert sample.data.shape == (1, *patch_size)
    assert target.data.shape == (1, *patch_size)


@pytest.mark.parametrize(
    "data_shape, patch_size, expected_dataset_len",
    [
        ((256, 256), (32, 32), 64),
        ((512, 512), (64, 64), 64),
        ((128, 128), (32, 32), 16),
    ],
)
def test_from_tiff(tmp_path: Path, data_shape, patch_size, expected_dataset_len):
    rng = np.random.default_rng(42)
    example_input = rng.random(data_shape)
    example_target = rng.random(data_shape)

    input_file_path = tmp_path / "input.tiff"
    target_file_path = tmp_path / "target.tiff"

    tifffile.imwrite(input_file_path, example_input)
    tifffile.imwrite(target_file_path, example_target)

    train_data_config = create_n2n_configuration(
        "test_exp",
        data_type="tiff",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        num_epochs=1,
    ).data_config

    train_dataset = CareamicsDataset(
        data_config=train_data_config,
        mode=Mode.TRAINING,
        inputs=[input_file_path],
        targets=[target_file_path],
    )

    assert len(train_dataset) == expected_dataset_len
    sample, target = train_dataset[0]
    assert sample.data.shape == (1, *patch_size)
    assert target.data.shape == (1, *patch_size)


@pytest.mark.skip("Prediction not fully implemented")
@pytest.mark.parametrize(
    "data_shape, tile_size, tile_overlap",
    [
        ((256, 256), (32, 32), (16, 16)),
        ((512, 512), (64, 64), (32, 32)),
        ((128, 128), (32, 32), (8, 8)),
    ],
)
def test_prediction_from_array(data_shape, tile_size, tile_overlap):
    rng = np.random.default_rng(42)
    example_data = rng.random(data_shape)

    prediction_config = InferenceConfig(
        data_type="array",
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        axes="YX",
        image_means=(example_data.mean(),),
        image_stds=(example_data.std(),),
        tta_transforms=False,
        batch_size=1,
    )

    prediction_dataset = CareamicsDataset(
        data_config=prediction_config,
        mode=Mode.PREDICTING,
        inputs=[example_data],
    )

    assert len(prediction_dataset) > 0
    sample, target = prediction_dataset[0]
    assert sample.data.shape == (1, *tile_size)
    assert target is None


@pytest.mark.parametrize(
    "patch_size, data_shape",
    [
        ((32, 32), (256, 256)),
        ((64, 64), (512, 512)),
        ((16, 16), (128, 128)),
    ],
)
def test_from_custom_data_type(patch_size, data_shape):
    rng = np.random.default_rng(42)
    example_data = rng.random(data_shape)
    example_target = rng.random(data_shape)

    train_data_config = create_n2n_configuration(
        "test_exp",
        data_type="custom",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        num_epochs=1,
    ).data_config

    def read_data_func_test(data):
        return 1 - data

    train_dataset = CareamicsDataset(
        data_config=train_data_config,
        mode=Mode.TRAINING,
        inputs=[example_data],
        targets=[example_target],
        read_func=read_data_func_test,
        read_kwargs={},
    )

    assert len(train_dataset) > 0
    sample, target = train_dataset[0]
    assert sample.data.shape == (1, *patch_size)
    assert target.data.shape == (1, *patch_size)
