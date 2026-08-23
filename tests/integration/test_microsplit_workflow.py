"""Integration tests for the MicroSplit noise-model workflow.

The noise model is a runtime, loss-side artifact (see
`tests/lightning/modules/test_set_noise_model.py`), not part of the configuration.
`CAREamist.train_noise_model` is the end-user entry point that fits it and attaches
it to the underlying module.

# TODO: full end-to-end MicroSplit training through `CAREamist.train` once the
# datamodule dispatches `MicroSplitDataConfig` to the MicroSplit-specific dataset
# construction (`careamics.dataset.factory.microsplit_factory`); today
# `CareamicsDataModule.setup` always goes through the generic
# `create_train_val_datasets` / `create_pred_dataset` factories.
"""

from pathlib import Path

import numpy as np
import pytest

from careamics.careamist import CAREamist
from careamics.config.factories import (
    create_advanced_n2v_config,
    create_microsplit_config,
)
from careamics.config.noise_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.models.lvae.noise_models import (
    MultiChannelNoiseModel,
    multichannel_noise_model_factory,
)

pytestmark = pytest.mark.lvae


def _microsplit_careamist(tmp_path: Path, output_channels: int = 2) -> CAREamist:
    config = create_microsplit_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=2,
        output_channels=output_channels,
        multiscale_count=1,
    )
    return CAREamist(config=config, work_dir=tmp_path)


def test_train_noise_model_fits_saves_and_attaches(tmp_path: Path):
    """`CAREamist.train_noise_model` fits, saves under `work_dir/noise_models`,
    attaches the result to the module, and returns it."""
    careamist = _microsplit_careamist(tmp_path, output_channels=2)

    rng = np.random.default_rng(42)
    signal = rng.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + rng.normal(0, 10, signal.shape).astype(np.float32)

    noise_model = careamist.train_noise_model(
        signal=signal, observation=observation, n_epochs=5, save=True
    )

    assert isinstance(noise_model, MultiChannelNoiseModel)
    assert len(noise_model) == 2
    assert careamist.model._raw_noise_model is noise_model

    saved_dir = tmp_path / "noise_models"
    assert saved_dir.exists()
    assert list(saved_dir.glob("*.npz"))


def test_train_noise_model_unsupported_algorithm_raises(tmp_path: Path):
    """Only MicroSplit and HDN support a noise model; N2V does not."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    careamist = CAREamist(config=config, work_dir=tmp_path)

    rng = np.random.default_rng(42)
    signal = rng.uniform(0, 255, (4, 16, 16)).astype(np.float32)
    observation = signal + rng.normal(0, 10, signal.shape).astype(np.float32)

    with pytest.raises(ValueError, match="does not support a noise model"):
        careamist.train_noise_model(signal=signal, observation=observation, n_epochs=5)


def test_train_input_unsupported_algorithm_raises(tmp_path: Path):
    """`train(noise_model=...)` guards the same way for algorithms without one."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    careamist = CAREamist(config=config, work_dir=tmp_path)
    train_array = np.ones((32, 32), dtype=np.float32)

    gmm_config = GaussianMixtureNMConfig(
        n_gaussian=1, n_coeff=2, min_signal=0.0, max_signal=1.0, min_sigma=0.1
    )
    noise_model = multichannel_noise_model_factory(
        MultiChannelNMConfig(noise_models=[gmm_config])
    )

    with pytest.raises(ValueError, match="does not support one"):
        careamist.train(train_data=train_array, noise_model=noise_model)
