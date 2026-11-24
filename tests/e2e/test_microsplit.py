"""End-to-end tests for the MicroSplit workflow."""

import numpy as np

from careamics.careamist import CAREamist
from careamics.config import (
    create_advanced_microsplit_config,
    create_advanced_n2v_config,
)
from careamics.config.losses import MicroSplitLossConfig
from careamics.noise_model import train_noise_model


def test_n2v_noise_model_microsplit_workflow(tmp_path):
    """Train N2V, fit a noise model, and use it to train and predict with MicroSplit."""
    rng = np.random.default_rng(42)

    # Keep two target channels so that the N2V estimate can be used directly to fit
    # the two-channel noise model required by MicroSplit.
    target = rng.uniform(0.1, 0.9, (2, 2, 128, 128)).astype(np.float32)
    observation = (
        target + rng.normal(0.0, 0.03, target.shape).astype(np.float32)
    ).clip(0.0, 1.0)

    n2v_config = create_advanced_n2v_config(
        experiment_name="e2e_n2v_noise_model",
        data_type="array",
        axes="SCYX",
        patch_size=(16, 16),
        batch_size=2,
        n_channels=2,
        num_epochs=1,
        num_steps=1,
        augmentations=[],
        num_workers=0,
        masked_pixel_percentage=1.0,
        model_params={
            "depth": 1,
            "num_channels_init": 8,
            "use_batch_norm": False,
        },
        seed=42,
    )
    n2v = CAREamist(config=n2v_config, work_dir=tmp_path / "n2v")
    n2v.train(train_data=observation, val_data=observation)

    n2v_predictions, _ = n2v.predict(observation)
    signal = np.asarray(n2v_predictions[0])
    assert signal.shape == observation.shape

    noise_model = train_noise_model(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_gaussian=1,
        n_coeff=2,
        min_sigma=0.01,
        n_epochs=1,
        save_dir=tmp_path / "noise_models",
    )
    assert len(noise_model) == 2

    microsplit_config = create_advanced_microsplit_config(
        experiment_name="e2e_microsplit",
        data_type="array",
        axes="SCYX",
        patch_size=(64, 64),
        batch_size=2,
        n_channels_out=2,
        num_epochs=1,
        num_steps=1,
        augmentations=[],
        num_workers=0,
        multiscale_count=1,
        loss=MicroSplitLossConfig(
            gaussian_likelihood_weight=0.0,
            noise_model_likelihood_weight=1.0,
        ),
        model_params={"z_dims": [8, 8], "n_filters": 8},
        seed=42,
    )
    microsplit = CAREamist(config=microsplit_config, work_dir=tmp_path / "microsplit")
    microsplit.train(
        train_data=observation[:, :1],
        train_data_target=target,
        val_data=observation[:, :1],
        val_data_target=target,
        noise_model=noise_model,
    )

    predictions, _ = microsplit.predict(observation[:, :1])

    assert np.asarray(predictions[0]).shape == target.shape
