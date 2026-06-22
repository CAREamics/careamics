import numpy as np
import torch

from careamics.config.data import DataConfig
from careamics.dataset.factory import create_dataset


def test_with_known_stats():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=25.0, size=(64, 64)).astype(np.float32)

    config = DataConfig(
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


def test_auto_computes_stats():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, size=(64, 64), dtype=np.uint8).astype(np.float32)

    config = DataConfig(
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


def test_input_and_target_have_independent_stats():
    rng = np.random.default_rng(42)
    input_data = rng.normal(loc=50, scale=10, size=(64, 64)).astype(np.float32)
    target_data = rng.normal(loc=150, scale=30, size=(64, 64)).astype(np.float32)

    config = DataConfig(
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


def test_per_channel_auto_computes_stats():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=50, scale=10, size=(64, 64)).astype(np.float32)
    ch1 = rng.normal(loc=200, scale=40, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = DataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={"name": "mean_std", "per_channel": True},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_means) == 2
    assert len(config.normalization.input_stds) == 2
    assert np.isclose(config.normalization.input_means[0], 50.0, atol=1.0)
    assert np.isclose(config.normalization.input_means[1], 200.0, atol=2.0)

    sample, *_ = dataset[0]
    for ch in range(2):
        assert np.abs(sample.data[ch].mean()) < 0.5
        assert np.abs(sample.data[ch].std() - 1.0) < 0.2


def test_global_auto_computes_stats():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=50, scale=10, size=(64, 64)).astype(np.float32)
    ch1 = rng.normal(loc=200, scale=40, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = DataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={"name": "mean_std", "per_channel": False},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_means) == 1
    assert len(config.normalization.input_stds) == 1

    global_mean = data.mean()
    assert np.isclose(config.normalization.input_means[0], global_mean, atol=1.0)

    sample, *_ = dataset[0]
    assert sample.data.shape[0] == 2


def test_scalar_config_values():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=25.0, size=(64, 64)).astype(np.float32)

    config = DataConfig(
        mode="predicting",
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={
            "name": "mean_std",
            "input_means": 100.0,
            "input_stds": 25.0,
        },
    )
    assert config.normalization.input_means == [100.0]
    assert config.normalization.input_stds == [25.0]

    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    assert np.abs(sample.data.mean()) < 0.5
    assert np.abs(sample.data.std() - 1.0) < 0.2


def test_single_value_broadcast_to_multichannel():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=100, scale=25, size=(64, 64)).astype(np.float32)
    ch1 = rng.normal(loc=100, scale=25, size=(64, 64)).astype(np.float32)
    ch2 = rng.normal(loc=100, scale=25, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1, ch2], axis=0)

    config = DataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={
            "name": "mean_std",
            "input_means": [100.0],
            "input_stds": [25.0],
        },
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    for ch in range(3):
        assert np.abs(sample.data[ch].mean()) < 0.5
        assert np.abs(sample.data[ch].std() - 1.0) < 0.2


def test_global_stats_denormalization_round_trip():
    rng = np.random.default_rng(42)
    ch0 = rng.normal(loc=100, scale=25, size=(64, 64)).astype(np.float32)
    ch1 = rng.normal(loc=100, scale=25, size=(64, 64)).astype(np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = DataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={"name": "mean_std", "per_channel": False},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)
    sample, *_ = dataset[0]

    normalized_tensor = torch.from_numpy(sample.data).unsqueeze(0)
    denormalized = dataset.normalization.denormalize(normalized_tensor)
    assert np.allclose(denormalized[0].numpy(), data, atol=1e-4)


def test_global_stats_pools_across_channels():
    ch0 = np.full((64, 64), 0.0, dtype=np.float32)
    ch1 = np.full((64, 64), 1000.0, dtype=np.float32)
    data = np.stack([ch0, ch1], axis=0)

    config = DataConfig(
        mode="predicting",
        data_type="array",
        axes="CYX",
        patching={"name": "whole"},
        normalization={"name": "mean_std", "per_channel": False},
    )
    dataset = create_dataset(config=config, inputs=[data], targets=None)

    assert len(config.normalization.input_means) == 1
    assert np.isclose(config.normalization.input_means[0], 500.0, atol=1.0)

    sample, *_ = dataset[0]
    assert sample.data[0].mean() < 0.0
    assert sample.data[1].mean() > 0.0


def test_denormalize_uses_target_stats_not_input_stats():
    """Regression test for #967: denormalize() must use target stats when
    they exist, not input stats. Real failure case: input and target have
    a different number of channels (e.g. denoising task), so reusing
    input's per-channel stats on the model's (target-shaped) output either
    crashes or silently applies the wrong values."""
    rng = np.random.default_rng(42)
    input_data = rng.normal(loc=50, scale=10, size=(64, 64)).astype(np.float32)
    target_data = rng.normal(loc=150, scale=30, size=(64, 64)).astype(np.float32)

    config = DataConfig(
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
        config=config, inputs=[input_data], targets=[target_data]
    )

    # Real failure scenario from the issue: model output has a different
    # channel count than the input. Force target stats to be multi-channel
    # while input stats stay single-channel (global), the exact mismatch
    # described in #967.
    norm = dataset.normalization
    norm.target_means = [150.0, 300.0]
    norm.target_stds = [30.0, 50.0]

    # Simulate model output: batch=1, 2 channels (matches target, not input)
    model_output = torch.zeros((1, 2, 8, 8))
    result = norm.denormalize(model_output)

    assert result.shape == (1, 2, 8, 8)
    # With model output all zeros, denormalized result should equal the
    # target means exactly -- proves target stats were actually used,
    # not input stats (which would give [50.0, 50.0] instead).
    assert torch.allclose(result[0, 0], torch.tensor(150.0), atol=1e-3)
    assert torch.allclose(result[0, 1], torch.tensor(300.0), atol=1e-3)
