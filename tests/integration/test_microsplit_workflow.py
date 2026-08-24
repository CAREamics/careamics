"""Integration tests for the MicroSplit workflows through `CAREamist`.

Two workflows are covered. The noise model is a runtime, loss-side artifact (see
`tests/lightning/modules/test_set_noise_model.py`), not part of the configuration;
`CAREamist.train_noise_model` is the end-user entry point that fits it and attaches
it to the underlying module. Training and prediction go through
`CareamicsDataModule`, which dispatches a `MicroSplitDataConfig` to the
MicroSplit-specific dataset construction in
`careamics.dataset.factory.microsplit_factory`.
"""

from pathlib import Path

import numpy as np
import pytest

from careamics.careamist import CAREamist
from careamics.config.factories import (
    create_advanced_microsplit_config,
    create_advanced_n2v_config,
    create_microsplit_config,
)
from careamics.config.noise_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.dataset.patch_constructor.microsplit_patch_constructors import (
    PairedInputTargetMsPatchConstr,
)
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


def _e2e_careamist(tmp_path: Path, output_channels: int = 2) -> CAREamist:
    """Return a CAREamist with a small MicroSplit configuration for training.

    The LVAE is shrunk and training is limited to a couple of batches, so that the
    whole training and prediction chain can be exercised cheaply. The muSplit
    likelihood is used alone, so that no noise model is required.
    """
    config = create_advanced_microsplit_config(
        experiment_name="test_e2e",
        data_type="array",
        axes="SCYX",
        patch_size=[64, 64],  # the LVAE requires at least 64 pixels in XY
        batch_size=2,
        output_channels=output_channels,
        num_epochs=1,
        num_steps=2,
        multiscale_count=1,
        augmentations=[],
        gaussian_likelihood_weight=1.0,
        noise_model_likelihood_weight=0.0,
        model_params={"z_dims": [32, 32], "n_filters": 8},
        seed=42,
    )
    return CAREamist(config=config, work_dir=tmp_path)


def _paired_arrays(
    seed: int, output_channels: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return a single-channel input and the target channels it superimposes."""
    rng = np.random.default_rng(seed)
    shape = (2, output_channels, 128, 128)
    target = rng.uniform(0, 1, shape).astype(np.float32)
    noise = rng.normal(0, 0.05, (2, 1, 128, 128))
    return (target.sum(axis=1, keepdims=True) + noise).astype(np.float32), target


def test_train_predict_end_to_end(tmp_path: Path):
    """`CAREamist` trains and predicts with MicroSplit datasets end to end."""
    # Arrange
    careamist = _e2e_careamist(tmp_path)
    train_input, train_target = _paired_arrays(seed=1)
    val_input, val_target = _paired_arrays(seed=2)

    # Act
    careamist.train(
        train_data=train_input,
        train_data_target=train_target,
        val_data=val_input,
        val_data_target=val_target,
    )

    # Assert: the data module built MicroSplit datasets, not the generic ones
    assert isinstance(
        careamist.train_datamodule.train_dataset.patch_constructor,
        PairedInputTargetMsPatchConstr,
    )
    # statistics are computed on the principal input, without the lateral context
    normalization = careamist.config.data_config.normalization
    assert len(normalization.input_means) == 1
    assert len(normalization.target_means) == 2

    # Act: tiles must match the training patch size, the LVAE input shape is validated
    predictions, _ = careamist.predict(
        val_input, tile_size=(64, 64), tile_overlap=(32, 32)
    )

    # Assert
    assert np.asarray(predictions[0]).shape == (2, 2, 128, 128)


def test_predict_whole_image(tmp_path: Path):
    """MicroSplit prediction works without tiling."""
    # Arrange
    careamist = _e2e_careamist(tmp_path)
    train_input, train_target = _paired_arrays(seed=1)
    careamist.train(
        train_data=train_input,
        train_data_target=train_target,
        val_data=train_input,
        val_data_target=train_target,
    )

    # Act
    predictions, _ = careamist.predict(train_input)

    # Assert
    assert np.asarray(predictions[0]).shape == (2, 2, 128, 128)


def test_train_without_validation_data_raises(tmp_path: Path):
    """MicroSplit does not support automatic validation splitting."""
    careamist = _e2e_careamist(tmp_path)
    train_input, train_target = _paired_arrays(seed=1)

    with pytest.raises(NotImplementedError, match="Automatic validation splitting"):
        careamist.train(train_data=train_input, train_data_target=train_target)
