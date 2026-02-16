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
