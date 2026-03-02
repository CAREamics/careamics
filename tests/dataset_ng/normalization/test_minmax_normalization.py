import numpy as np
import torch

from careamics.config.data import NGDataConfig
from careamics.dataset_ng.factory import create_dataset


def test_with_known_range():
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


def test_auto_computes_range():
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


def test_per_channel_different_ranges():
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


def test_per_channel_auto_computes_range():
    rng = np.random.default_rng(42)
    ch0 = rng.integers(0, 100, size=(64, 64)).astype(np.float32)
    ch1 = rng.integers(500, 1000, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={"name": "minmax", "per_channel": True},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_mins) == 2
    assert len(config.normalization.input_maxes) == 2
    assert config.normalization.input_mins[0] == ch0.min()
    assert config.normalization.input_maxes[1] == ch1.max()

    sample, *_ = dataset[0]
    for ch in range(2):
        assert np.isclose(sample.data[ch].min(), 0.0, atol=0.01)
        assert np.isclose(sample.data[ch].max(), 1.0, atol=0.01)


def test_global_auto_computes_range():
    rng = np.random.default_rng(42)
    ch0 = rng.integers(0, 100, size=(64, 64)).astype(np.float32)
    ch1 = rng.integers(500, 1000, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={"name": "minmax", "per_channel": False},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_mins) == 1
    assert len(config.normalization.input_maxes) == 1
    assert config.normalization.input_mins[0] == data.min()
    assert config.normalization.input_maxes[0] == data.max()

    sample, *_ = dataset[0]
    assert sample.data.shape[0] == 2


def test_scalar_config_values():
    rng = np.random.default_rng(42)
    data = rng.integers(10, 200, size=(64, 64), dtype=np.uint8).astype(np.float32)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={
            "name": "minmax",
            "input_mins": 0.0,
            "input_maxes": 255.0,
        },
    )
    assert config.normalization.input_mins == [0.0]
    assert config.normalization.input_maxes == [255.0]

    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    assert sample.data.min() >= 0.0
    assert sample.data.max() <= 1.0


def test_single_value_broadcast_to_multichannel():
    rng = np.random.default_rng(42)
    ch0 = rng.integers(0, 255, size=(64, 64)).astype(np.float32)
    ch1 = rng.integers(0, 255, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={
            "name": "minmax",
            "input_mins": [0.0],
            "input_maxes": [255.0],
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    for ch in range(2):
        assert sample.data[ch].min() >= 0.0
        assert sample.data[ch].max() <= 1.0


def test_global_stats_denormalization_round_trip():
    rng = np.random.default_rng(42)
    ch0 = rng.integers(0, 255, size=(64, 64)).astype(np.float32)
    ch1 = rng.integers(0, 255, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={"name": "minmax", "per_channel": False},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    normalized_tensor = torch.from_numpy(sample.data).unsqueeze(0)
    denormalized = dataset.normalization.denormalize(normalized_tensor)
    assert np.allclose(denormalized[0].numpy(), data, atol=1e-3)


def test_global_stats_pools_across_channels():
    ch0 = np.full((64, 64), 10.0, dtype=np.float32)
    ch1 = np.full((64, 64), 990.0, dtype=np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={"name": "minmax", "per_channel": False},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_mins) == 1
    assert config.normalization.input_mins[0] == 10.0
    assert config.normalization.input_maxes[0] == 990.0

    sample, *_ = dataset[0]
    assert np.isclose(sample.data[0].mean(), 0.0, atol=0.01)
    assert np.isclose(sample.data[1].mean(), 1.0, atol=0.01)
