import numpy as np

from careamics.config.data import NGDataConfig
from careamics.dataset_ng.dataset import Mode
from careamics.dataset_ng.factory import create_dataset


def test_standardization():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 100, size=(64, 64), dtype=np.uint16)
    data = data.astype(np.float32)
    config = NGDataConfig(
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={
            "name": "standardize",
            "input_means": [data.mean()],
            "input_stds": [data.std()],
        },
    )
    dataset = create_dataset(
        config=config, mode=Mode.PREDICTING, inputs=[data], targets=None, in_memory=True
    )
    sample, *_ = dataset[0]

    assert np.abs(sample.data.mean()) < 0.1
    assert np.abs(sample.data.std() - 1) < 0.1

    denormalized = dataset.normalization.denormalize(sample.data[np.newaxis, ...])

    assert np.allclose(denormalized.round(), data.round())


def test_no_normalization():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 100, size=(64, 64), dtype=np.uint16)
    data = data.astype(np.float32)
    config = NGDataConfig(
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization={"name": "none"},
    )
    dataset = create_dataset(
        config=config, mode=Mode.PREDICTING, inputs=[data], targets=None, in_memory=True
    )
    sample, *_ = dataset[0]

    assert np.allclose(sample.data.round(), data.round())

    denormalized = dataset.normalization.denormalize(sample.data)

    assert np.allclose(denormalized.round(), data.round())


# def test_quantile_normalization():
#     rng = np.random.default_rng(42)
#     data = rng.integers(0, 100, size=(64, 64), dtype=np.float32)
#     config = NGDataConfig(
#         data_type="array",
#         axes="YX",
#         patching={"name": "whole"},
#         normalization={"name": "quantile", "lower_quantiles": [0.01], "upper_quantiles": [0.99]},
#     )
#     dataset = create_dataset(
#         config=config, mode=Mode.TRAINING, inputs=[data], targets=None, in_memory=True
#     )
#     sample, *_ = dataset[0]

#     assert np.allclose(sample.data.round(), data.round())


# def test_minmax_normalization():
#     rng = np.random.default_rng(42)
#     data = rng.integers(0, 100, size=(64, 64), dtype=np.float32)
#     config = NGDataConfig(
#         data_type="array",
#         axes="YX",
#         patching={"name": "whole"},
#         normalization={"name": "minmax", "image_mins": [0], "image_maxs": [100]},
#     )
#     dataset = create_dataset(
#         config=config, mode=Mode.TRAINING, inputs=[data], targets=None, in_memory=True
#     )
#     sample, *_ = dataset[0]

#     assert np.allclose(sample.data.round(), data.round())


# def test_error_invalid_normalization_strategy():
#     with pytest.raises(ValueError):
#         NGDataConfig(
#             data_type="array",
#             axes="YX",
#             patching={"name": "whole"},
#             normalization={"name": "invalid"},
#         )
