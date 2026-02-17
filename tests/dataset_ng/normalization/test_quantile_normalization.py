import numpy as np

from careamics.config.data import NGDataConfig
from careamics.dataset_ng.factory import create_dataset


def test_clips_outliers():
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


def test_per_channel_auto_computes_quantiles():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=50, scale=10, size=(128, 128)).astype(np.float32)
    ch1 = rng.normal(loc=200, scale=40, size=(128, 128)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={
            "name": "quantile",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "per_channel": True,
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_lower_quantile_values) == 2
    assert len(config.normalization.input_upper_quantile_values) == 2

    assert (
        config.normalization.input_lower_quantile_values[0]
        < config.normalization.input_lower_quantile_values[1]
    )

    sample, *_ = dataset[0]
    assert sample.data.shape[0] == 2


def test_global_auto_computes_quantiles():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=50, scale=10, size=(128, 128)).astype(np.float32)
    ch1 = rng.normal(loc=200, scale=40, size=(128, 128)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={
            "name": "quantile",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "per_channel": False,
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_lower_quantile_values) == 1
    assert len(config.normalization.input_upper_quantile_values) == 1

    sample, *_ = dataset[0]
    assert sample.data.shape[0] == 2


def test_scalar_config_values():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=20.0, size=(128, 128)).astype(np.float32)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={
            "name": "quantile",
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
        },
    )
    assert config.normalization.lower_quantile == [0.05]
    assert config.normalization.upper_quantile == [0.95]

    _ = create_dataset(config=config, inputs=[data], targets=None)

    assert config.normalization.input_lower_quantile_values is not None
    assert config.normalization.input_upper_quantile_values is not None


def test_global_stats_pools_across_channels():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=0, scale=1, size=(128, 128)).astype(np.float32)
    ch1 = rng.normal(loc=1000, scale=1, size=(128, 128)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={
            "name": "quantile",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "per_channel": False,
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_lower_quantile_values) == 1
    lower = config.normalization.input_lower_quantile_values[0]
    upper = config.normalization.input_upper_quantile_values[0]

    assert lower < 10.0
    assert upper > 990.0

    sample, *_ = dataset[0]
    assert sample.data.shape[0] == 2


def test_per_channel_quantile_levels():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=100, scale=20, size=(128, 128)).astype(np.float32)
    ch1 = rng.normal(loc=100, scale=20, size=(128, 128)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = NGDataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={
            "name": "quantile",
            "lower_quantile": [0.01, 0.10],
            "upper_quantile": [0.99, 0.90],
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    lv = config.normalization.input_lower_quantile_values
    uv = config.normalization.input_upper_quantile_values

    assert lv[0] < lv[1]
    assert uv[0] > uv[1]

    sample, *_ = dataset[0]
    assert sample.data.shape[0] == 2
