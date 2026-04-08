import numpy as np
import torch

from careamics.config.data import NGDataConfig
from careamics.dataset_ng.factory import create_dataset


def test_preserves_values():
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
