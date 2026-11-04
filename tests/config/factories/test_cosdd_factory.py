"""Tests for the COSDD configuration factories and configuration."""

import pytest

from careamics.config import (
    COSDDConfiguration,
    create_advanced_cosdd_config,
    create_cosdd_config,
)
from careamics.config.augmentations import XYFlipConfig
from careamics.config.factories.config_discriminators import instantiate_config


def _create(**kwargs) -> COSDDConfiguration:
    params = {
        "experiment_name": "test",
        "data_type": "array",
        "axes": "SYX",
        "patch_size": [64, 64],
    }
    params.update(kwargs)
    return create_advanced_cosdd_config(**params)


def test_create_cosdd_config():
    """Test the default COSDD configuration."""
    config = create_cosdd_config(
        experiment_name="test", data_type="array", axes="SYX", patch_size=[64, 64]
    )

    assert isinstance(config, COSDDConfiguration)
    model = config.algorithm_config.model
    assert model.architecture == "COSDD"
    assert model.input_shape == (64, 64)
    assert model.in_channels == 1
    assert config.data_config.batch_size == 4
    assert len(config.data_config.augmentations) == 0
    assert not config.is_supervised()

    # validation loss is used for checkpointing and early stopping
    training_config = config.training_config
    assert training_config.checkpoint_params["monitor"] == "val_loss"
    assert training_config.early_stopping_params == {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 50,
    }
    assert training_config.trainer_params["max_epochs"] == 1000


def test_3D_config():
    """Test a 3D configuration with noise correlated along Z."""
    config = _create(axes="ZYX", patch_size=[8, 64, 64], noise_direction="z")

    assert config.algorithm_config.model.is_3D()
    assert config.data_config.is_3D()


@pytest.mark.parametrize(
    "axes, n_channels, channels, expected",
    [("SCYX", 3, None, 3), ("SCYX", None, [0, 2], 2), ("SYX", None, None, 1)],
)
def test_channels(axes, n_channels, channels, expected):
    """Test that the number of channels is propagated to the model."""
    config = _create(axes=axes, n_channels=n_channels, channels=channels)

    assert config.algorithm_config.model.in_channels == expected


def test_channels_required_with_C_axis():
    """Test that the number of channels is required with a channel axis."""
    with pytest.raises(ValueError):
        _create(axes="SCYX")


def test_model_params_precedence():
    """Test that explicit parameters take precedence over `model_params`."""
    config = _create(
        noise_direction="y",
        use_direct_denoiser=False,
        model_params={
            "noise_direction": "x",
            "use_direct_denoiser": True,
            "n_gaussians": 5,
            "input_shape": (32, 32),
        },
    )
    model = config.algorithm_config.model

    assert model.noise_direction == "y"
    assert not model.use_direct_denoiser
    assert model.n_gaussians == 5
    assert model.input_shape == (64, 64)


def test_optimization_parameters():
    """Test the optimizer, scheduler and gradient accumulation parameters."""
    config = _create(
        optimizer="Adam",
        optimizer_params={"lr": 1e-4},
        lr_scheduler="StepLR",
        lr_scheduler_params={"step_size": 10},
        n_grad_batches=2,
    )
    algorithm_config = config.algorithm_config

    assert algorithm_config.optimizer.name == "Adam"
    assert algorithm_config.optimizer.parameters == {"lr": 1e-4}
    assert algorithm_config.lr_scheduler.name == "StepLR"
    assert algorithm_config.lr_scheduler.parameters == {"step_size": 10}
    assert algorithm_config.n_grad_batches == 2


def test_augmentations():
    """Test that flips are allowed and rotations only for uncorrelated noise."""
    config = _create(augmentations=["x_flip", "y_flip"])
    assert isinstance(config.data_config.augmentations[0], XYFlipConfig)

    with pytest.raises(ValueError, match="rotation"):
        _create(augmentations=["rotate_90"], noise_direction="x")

    config = _create(augmentations=["rotate_90"], noise_direction="none")
    assert len(config.data_config.augmentations) == 1


def test_patch_size_must_match_input_shape():
    """Test that the training patch size must match the model input shape."""
    config = _create()
    config_dict = config.model_dump()
    config_dict["algorithm_config"]["model"]["input_shape"] = (32, 32)

    with pytest.raises(ValueError, match="input shape"):
        COSDDConfiguration(**config_dict)


def test_target_axes_not_supported():
    """Test that COSDD rejects target axes."""
    config_dict = _create().model_dump()
    config_dict["data_config"]["target_axes"] = "SYX"

    with pytest.raises(ValueError, match="target_axes"):
        COSDDConfiguration(**config_dict)


def test_prediction_data_config():
    """Test that prediction is allowed on tiles of any size."""
    config = _create()
    pred_config = config.data_config.convert_mode(
        "predicting", new_patch_size=[96, 96], overlap_size=[16, 16]
    )

    config.data_config = pred_config
    assert config.data_config.mode == "predicting"


def test_set_3D():
    """Test switching the configuration to 3D."""
    config = _create()
    config.set_3D(True, "ZYX", [8, 64, 64])

    assert config.algorithm_config.model.input_shape == (8, 64, 64)
    assert config.data_config.is_3D()

    with pytest.raises(ValueError):
        config.set_3D(False, "YX", [8, 64, 64])


def test_instantiate_config():
    """Test that the configuration discriminator returns a COSDD configuration."""
    config = _create()

    assert isinstance(instantiate_config(config.model_dump()), COSDDConfiguration)
