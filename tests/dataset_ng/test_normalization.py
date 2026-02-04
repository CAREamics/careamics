import numpy as np
import torch

from careamics.config.data import NGDataConfig
from careamics.dataset_ng.factory import create_dataset


def test_mean_std_with_known_stats():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=25.0, size=(64, 64)).astype(np.float32)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={
            "name": "mean_std",
            "input_means": [100.0],
            "input_stds": [25.0],
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    assert np.abs(sample.data.mean()) < 0.5
    assert np.abs(sample.data.std() - 1.0) < 0.2

    normalized_tensor = torch.from_numpy(sample.data).unsqueeze(0)
    denormalized = dataset.normalization.denormalize(normalized_tensor)
    assert np.allclose(denormalized[0, 0], data, atol=1e-4)


def test_mean_std_auto_computes_stats():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, size=(64, 64), dtype=np.uint8).astype(np.float32)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={"name": "mean_std"},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert config.normalization.input_means is not None
    assert config.normalization.input_stds is not None
    assert np.isclose(config.normalization.input_means[0], data.mean(), atol=0.1)
    assert np.isclose(config.normalization.input_stds[0], data.std(), atol=0.1)

    sample, *_ = dataset[0]
    assert np.abs(sample.data.mean()) < 0.1
    assert np.abs(sample.data.std() - 1.0) < 0.1


def test_minmax_with_known_range():
    rng = np.random.default_rng(42)
    data = rng.integers(10, 200, size=(64, 64), dtype=np.uint8).astype(np.float32)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={
            "name": "minmax",
            "input_mins": [0.0],
            "input_maxes": [255.0],
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    assert sample.data.min() >= 0.0
    assert sample.data.max() <= 1.0

    normalized_tensor = torch.from_numpy(sample.data).unsqueeze(0)
    denormalized = dataset.normalization.denormalize(normalized_tensor)
    assert np.allclose(denormalized[0, 0], data, atol=1e-4)


def test_minmax_auto_computes_range():
    rng = np.random.default_rng(42)
    data = rng.integers(50, 200, size=(64, 64), dtype=np.uint8).astype(np.float32)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={"name": "minmax"},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert config.normalization.input_mins is not None
    assert config.normalization.input_maxes is not None
    assert config.normalization.input_mins[0] == data.min()
    assert config.normalization.input_maxes[0] == data.max()

    sample, *_ = dataset[0]
    assert np.isclose(sample.data.min(), 0.0, atol=0.01)
    assert np.isclose(sample.data.max(), 1.0, atol=0.01)


def test_quantile_clips_outliers():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=20.0, size=(128, 128)).astype(np.float32)
    data[0, 0] = 0.0
    data[0, 1] = 255.0

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={
            "name": "quantile",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert config.normalization.input_lower_quantile_values is not None
    assert config.normalization.input_upper_quantile_values is not None

    sample, *_ = dataset[0]
    central_values = sample.data[(sample.data > 0.01) & (sample.data < 0.99)]
    assert len(central_values) > 0.9 * sample.data.size


def test_no_normalization_preserves_values():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, size=(64, 64), dtype=np.uint8).astype(np.float32)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={"name": "none"},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    assert np.allclose(sample.data[0], data)

    normalized_tensor = torch.from_numpy(sample.data).unsqueeze(0)
    denormalized = dataset.normalization.denormalize(normalized_tensor)
    assert np.allclose(denormalized[0, 0], data)


def test_mean_std_per_channel():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=50, scale=10, size=(64, 64)).astype(np.float32)
    ch1 = rng.normal(loc=100, scale=20, size=(64, 64)).astype(np.float32)
    ch2 = rng.normal(loc=200, scale=5, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1, ch2], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={
            "name": "mean_std",
            "input_means": [50.0, 100.0, 200.0],
            "input_stds": [10.0, 20.0, 5.0],
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    for ch in range(3):
        assert np.abs(sample.data[ch].mean()) < 0.5
        assert np.abs(sample.data[ch].std() - 1.0) < 0.2


def test_minmax_per_channel_different_ranges():
    rng = np.random.default_rng(42)
    ch0 = rng.integers(0, 100, size=(64, 64)).astype(np.float32)
    ch1 = rng.integers(100, 1000, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={
            "name": "minmax",
            "input_mins": [0.0, 100.0],
            "input_maxes": [100.0, 1000.0],
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    for ch in range(2):
        assert sample.data[ch].min() >= 0.0
        assert sample.data[ch].max() <= 1.0


def test_input_and_target_have_independent_stats():
    rng = np.random.default_rng(42)
    input_data = rng.normal(loc=50, scale=10, size=(64, 64)).astype(np.float32)
    target_data = rng.normal(loc=150, scale=30, size=(64, 64)).astype(np.float32)

    config = NGDataConfig(
        mode="training",
        data_type="array",
        axes="YX",
        patching={"name": "random", "patch_size": (32, 32)},
        normalization={
            "name": "mean_std",
            "input_means": [50.0],
            "input_stds": [10.0],
            "target_means": [150.0],
            "target_stds": [30.0],
        },
    )
    dataset = create_dataset(
        config=config,
        inputs=[input_data],
        targets=[target_data],
    )
    input_sample, target_sample = dataset[0]

    assert np.abs(input_sample.data.mean()) < 1.0
    assert np.abs(target_sample.data.mean()) < 1.0
