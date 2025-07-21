import numpy as np

from careamics.config.data import NGDataConfig
from careamics.dataset_ng.dataset import Mode
from careamics.dataset_ng.factory import create_array_dataset
from careamics.transforms.normalization import build_normalization_transform


def test_dataset_mean_std_normalization_and_denormalization():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 65535, size=(64, 64), dtype=np.uint16).astype(np.float32)
    config = NGDataConfig(
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={"name": "mean_std"},
        batch_size=1,
        seed=42,
    )
    config.set_means_and_stds([data.mean()], [data.std()])
    dataset = create_array_dataset(
        config=config, mode=Mode.PREDICTING, inputs=[data], targets=None
    )
    sample, *_ = dataset[0]

    # Test normalization
    assert np.abs(sample.data.mean()) < 0.1
    assert np.abs(sample.data.std() - 1) < 0.1

    # Test denormalization
    normalization_config = dataset.config.normalization
    normalization_transform = build_normalization_transform(normalization_config)
    sample_data = sample.data[None, ...]  # add batch dimension
    denormalized_output = normalization_transform.denormalize(sample_data)
    assert np.allclose(denormalized_output.round(), data.round())


def test_dataset_no_normalization():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 65535, size=(64, 64), dtype=np.uint16).astype(np.float32)
    config = NGDataConfig(
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={"name": "none"},
        batch_size=1,
        seed=42,
    )
    dataset = create_array_dataset(
        config=config, mode=Mode.PREDICTING, inputs=[data], targets=None
    )
    sample, *_ = dataset[0]

    # Test that no normalization returns data unchanged
    assert np.allclose(sample.data, data)

    # Test denormalization
    normalization_config = dataset.config.normalization
    normalization_transform = build_normalization_transform(normalization_config)
    sample_data = sample.data[None, ...]  # add batch dimension
    denormalized_output = normalization_transform.denormalize(sample_data)
    assert np.allclose(denormalized_output, data)
