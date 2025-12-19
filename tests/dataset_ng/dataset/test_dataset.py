from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.config import create_ng_data_configuration
from careamics.config.configuration_factories import (
    _list_spatial_augmentations,
)
from careamics.config.data import NGDataConfig
from careamics.dataset_ng.dataset import _adjust_shape_for_channels
from careamics.dataset_ng.factory import create_dataset


@pytest.mark.parametrize(
    "shape, channels, expected_shape",
    [
        ((1, 1, 32, 32), None, (1, 1, 32, 32)),
        ((1, 1, 32, 32), [0], (1, 1, 32, 32)),
        ((5, 4, 32, 32), None, (5, 4, 32, 32)),
        ((5, 4, 32, 32), [1], (5, 1, 32, 32)),
        ((5, 4, 32, 32), [1, 3], (5, 2, 32, 32)),
    ],
)
def test_adjust_shape_for_channels(shape, channels, expected_shape):
    adjusted_shape = _adjust_shape_for_channels(shape, channels)
    assert adjusted_shape == expected_shape


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

    train_data_config = create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=_list_spatial_augmentations(),
        in_memory=True,
        seed=42,
    )

    train_data_config.set_means_and_stds(
        [example_input.mean()],
        [example_input.std()],
        [example_target.mean()],
        [example_target.std()],
    )

    train_dataset = create_dataset(
        config=train_data_config,
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
    "data_shape, patch_size, channels",
    [
        ((3, 32, 32), (8, 8), None),
        ((3, 32, 32), (8, 8), [1]),
        ((3, 32, 32), (8, 8), [0, 2]),
    ],
)
def test_from_array_with_channels(data_shape, patch_size, channels):
    rng = np.arange(np.prod(data_shape)).reshape(data_shape)
    for i in range(data_shape[0]):
        rng[0] *= i * 1_000

    train_data_config = create_ng_data_configuration(
        data_type="array",
        axes="CYX",
        patch_size=patch_size,
        batch_size=1,
        channels=channels,
        seed=42,
    )

    n_channels = len(channels) if channels is not None else data_shape[0]
    train_data_config.set_means_and_stds(
        [0 for _ in range(n_channels)],
        [1 for _ in range(n_channels)],
        [0 for _ in range(n_channels)],
        [1 for _ in range(n_channels)],
    )

    train_dataset = create_dataset(
        config=train_data_config,
        inputs=[rng],
        targets=[rng],
    )

    sample, target = train_dataset[0]
    assert sample.data.shape[0] == data_shape[0] if channels is None else len(channels)
    assert target.data.shape[0] == data_shape[0] if channels is None else len(channels)

    if channels is not None:
        for sample, target in train_dataset:
            for i, ch in enumerate(channels):
                assert np.all(ch * 1000 <= sample.data[i])
                assert np.all((ch + 1) * 1000 >= sample.data[i])
                assert np.all(ch * 1000 <= target.data[i])
                assert np.all((ch + 1) * 1000 >= target.data[i])

            # test that channels are properly adjusted in data_shape
            assert sample.data_shape[1] == len(channels)


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

    train_data_config = create_ng_data_configuration(
        data_type="tiff",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=_list_spatial_augmentations(),
        in_memory=True,
        seed=42,
    )

    train_data_config.set_means_and_stds(
        [example_input.mean()],
        [example_input.std()],
        [example_target.mean()],
        [example_target.std()],
    )

    train_dataset = create_dataset(
        config=train_data_config,
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
        mode="predicting",
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

    prediction_dataset = create_dataset(
        config=prediction_config,
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

    train_data_config = create_ng_data_configuration(
        data_type="custom",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=_list_spatial_augmentations(),
        in_memory=True,
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

    train_dataset = create_dataset(
        config=train_data_config,
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

    train_data_config = create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=1,
        augmentations=[],
        in_memory=True,
        seed=42,
    )

    train_data_config.patch_filter_patience = 100
    train_data_config.coord_filter = {
        "name": "mask",
        "coverage": 0.5,
    }

    train_dataset = create_dataset(
        config=train_data_config,
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

    train_data_config = create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=1,
        augmentations=[],
        in_memory=True,
        seed=42,
    )
    threshold = 255 // 2
    train_data_config.patch_filter_patience = 10
    train_data_config.patch_filter = {
        "name": "mean_std",
        "mean_threshold": threshold,
    }

    train_dataset = create_dataset(
        config=train_data_config,
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

    train_data_config = create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=_list_spatial_augmentations(),
        in_memory=True,
        seed=42,
    )

    with pytest.raises(ValueError):
        _ = create_dataset(
            config=train_data_config,
            inputs=[example_input],
            targets=[example_target],
        )
