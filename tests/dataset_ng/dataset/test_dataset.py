from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.config.configuration_factories import (
    _create_ng_data_configuration,
    _list_spatial_augmentations,
)
from careamics.config.data import NGDataConfig
from careamics.dataset_ng.dataset import Mode
from careamics.dataset_ng.factory import (
    create_array_dataset,
    create_custom_file_dataset,
    create_tiff_dataset,
)


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

    train_data_config = _create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=_list_spatial_augmentations(),
        seed=42,
    )

    train_data_config.set_means_and_stds(
        [example_input.mean()],
        [example_input.std()],
        [example_target.mean()],
        [example_target.std()],
    )

    train_dataset = create_array_dataset(
        config=train_data_config,
        mode=Mode.TRAINING,
        inputs=[example_input],
        targets=[example_target],
    )

    assert len(train_dataset) == expected_dataset_len
    output = train_dataset[0]
    assert len(output) == 2
    sample, target = output
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

    train_data_config = _create_ng_data_configuration(
        data_type="tiff",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=_list_spatial_augmentations(),
        seed=42,
    )

    train_data_config.set_means_and_stds(
        [example_input.mean()],
        [example_input.std()],
        [example_target.mean()],
        [example_target.std()],
    )

    train_dataset = create_tiff_dataset(
        config=train_data_config,
        mode=Mode.TRAINING,
        inputs=[input_file_path],
        targets=[target_file_path],
    )

    assert len(train_dataset) == expected_dataset_len
    output = train_dataset[0]
    assert len(output) == 2
    sample, target = output
    assert sample.data.shape == (1, *patch_size)
    assert target.data.shape == (1, *patch_size)


@pytest.mark.parametrize(
    "data_shape, tile_size, tile_overlap",
    [
        ((256, 256), (32, 32), (16, 16)),
        ((512, 512), (64, 64), (32, 32)),
        ((128, 128), (32, 32), (8, 8)),
        ((24, 24), (32, 32), (8, 8)),  # data smaller than patch
    ],
)
def test_prediction_from_array(data_shape, tile_size, tile_overlap):
    rng = np.random.default_rng(42)
    example_data = rng.random(data_shape)

    prediction_config = NGDataConfig(
        data_type="array",
        patching={
            "name": "tiled",
            "patch_size": tile_size,
            "overlaps": tile_overlap,
        },
        axes="YX",
        image_means=[example_data.mean()],
        image_stds=[example_data.std()],
        transforms=_list_spatial_augmentations(),
        batch_size=1,
        seed=42,
    )

    prediction_dataset = create_array_dataset(
        config=prediction_config,
        mode=Mode.PREDICTING,
        inputs=[example_data],
        targets=None,
    )

    assert len(prediction_dataset) > 0
    output = prediction_dataset[0]
    assert len(output) == 1
    (sample,) = output
    assert sample.data.shape == (1, *tile_size)


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

    train_data_config = _create_ng_data_configuration(
        data_type="custom",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=_list_spatial_augmentations(),
        seed=42,
    )

    train_data_config.set_means_and_stds(
        [example_data.mean()],
        [example_data.std()],
        [example_target.mean()],
        [example_target.std()],
    )

    def read_data_func_test(data):
        return 1 - data

    train_dataset = create_custom_file_dataset(
        config=train_data_config,
        mode=Mode.TRAINING,
        inputs=[example_data],
        targets=[example_target],
        read_func=read_data_func_test,
        read_kwargs={},
    )

    assert len(train_dataset) > 0
    output = train_dataset[0]
    assert len(output) == 2
    sample, target = output
    assert sample.data.shape == (1, *patch_size)
    assert target.data.shape == (1, *patch_size)


def test_array_coordinate_filtering():
    """Test that coordinate filtering is applied correctly when creating a dataset from
    an array."""
    size = 16
    img = np.zeros((size, size))
    mask = np.zeros((size, size))

    # create a square and mask it
    coords = (slice(8, 24), slice(8, 24))
    mask[coords] = 1
    img[coords] = 255

    train_data_config = _create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=1,
        augmentations=[],
        seed=42,
    )

    train_data_config.patch_filter_patience = 100
    train_data_config.coord_filter = {
        "name": "mask",
        "coverage": 0.5,
    }

    train_dataset = create_array_dataset(
        config=train_data_config,
        mode=Mode.TRAINING,
        inputs=[img],
        targets=None,
        masks=[mask],
    )

    # check that we only get patches with at least half of 255 pixels
    threshold = 255 // 2
    stats = train_dataset.input_stats
    normed_thresh = (threshold - stats.means[0]) / stats.stds[0]
    for i in range(len(train_dataset)):
        (sample,) = train_dataset[i]
        assert sample.data.mean() > normed_thresh


def test_array_patch_filtering():
    """Test that patch filtering is applied correctly when creating a dataset from
    an array."""
    size = 32
    img = np.zeros((size, size))

    # create a square
    coords = (slice(8, 24), slice(8, 24))
    img[coords] = 255

    train_data_config = _create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=1,
        augmentations=[],
        seed=42,
    )
    threshold = 255 // 2
    train_data_config.patch_filter_patience = 10
    train_data_config.patch_filter = {
        "name": "mean_std",
        "mean_threshold": threshold,
    }

    train_dataset = create_array_dataset(
        config=train_data_config,
        mode=Mode.TRAINING,
        inputs=[img],
        targets=None,
    )

    # check that we only get the full 255 patch (in normalized units)
    stats = train_dataset.input_stats
    normed_thresh = (threshold - stats.means[0]) / stats.stds[0]
    for i in range(len(train_dataset)):
        (sample,) = train_dataset[i]
        assert sample.data.mean() >= normed_thresh


def test_error_data_smaller_than_patch():
    """
    In training mode, initializing the dataset with data smaller than the patch size
    should result in an error.
    """

    data_shape = (24, 24)
    patch_size = (32, 32)

    rng = np.random.default_rng(42)
    example_input = rng.random(data_shape)
    example_target = rng.random(data_shape)

    train_data_config = _create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=_list_spatial_augmentations(),
        seed=42,
    )

    with pytest.raises(ValueError):
        _ = create_array_dataset(
            config=train_data_config,
            mode=Mode.TRAINING,
            inputs=[example_input],
            targets=[example_target],
        )
