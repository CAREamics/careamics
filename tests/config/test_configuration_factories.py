from pathlib import Path

import numpy as np
import pytest

from careamics.config import (
    CAREAlgorithm,
    Configuration,
    MicroSplitAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
    algorithm_factory,
    create_care_configuration,
    create_microsplit_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
)
from careamics.config.configuration_factories import (
    _create_data_configuration,
    _create_supervised_config_dict,
    _create_unet_configuration,
    _list_spatial_augmentations,
)
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedPixelManipulation,
    SupportedStructAxis,
    SupportedTransform,
)
from careamics.config.transformations import (
    N2VManipulateModel,
    XYFlipModel,
    XYRandomRotate90Model,
)


def test_careamics_config_n2v(minimum_n2v_configuration):
    """Test that the N2V configuration is created correctly."""
    config = Configuration(**minimum_n2v_configuration)
    assert config.algorithm_config.algorithm == SupportedAlgorithm.N2V.value


@pytest.mark.parametrize(
    "algorithm", [SupportedAlgorithm.N2N.value, SupportedAlgorithm.CARE.value]
)
def test_careamics_config_supervised(minimum_supervised_configuration, algorithm):
    """Test that the supervised configuration is created correctly."""
    min_config = minimum_supervised_configuration
    min_config["algorithm_config"]["algorithm"] = algorithm

    config = Configuration(**min_config)

    assert config.algorithm_config.algorithm == algorithm


def test_algorithm_factory_n2v(minimum_algorithm_n2v):
    """Test that the N2V configuration is created correctly."""
    algorithm = algorithm_factory(minimum_algorithm_n2v)
    assert isinstance(algorithm, N2VAlgorithm)


@pytest.mark.parametrize("algorithm", ["n2n", "care"])
def test_algorithm_factory_supervised(minimum_algorithm_supervised, algorithm):
    """Test that the supervised configuration is created correctly."""
    min_config = minimum_algorithm_supervised
    min_config["algorithm"] = algorithm

    algorithm_config = algorithm_factory(min_config)

    exp_class = N2NAlgorithm if algorithm == "n2n" else CAREAlgorithm
    assert isinstance(algorithm_config, exp_class)


def test_list_aug_default():
    """Test that the default augmentations are present."""
    list_aug = _list_spatial_augmentations(augmentations=None)

    assert len(list_aug) == 2
    assert list_aug[0].name == SupportedTransform.XY_FLIP.value
    assert list_aug[1].name == SupportedTransform.XY_RANDOM_ROTATE90.value


def test_list_aug_no_aug():
    """Test that disabling augmentation results in empty transform list."""
    list_aug = _list_spatial_augmentations(augmentations=[])
    assert len(list_aug) == 0


def test_list_aug_error_duplicate_transforms():
    """Test that an error is raised when there are duplicate transforms."""
    with pytest.raises(ValueError):
        _list_spatial_augmentations(
            augmentations=[XYFlipModel(), XYRandomRotate90Model(), XYFlipModel()],
        )


def test_list_aug_error_wrong_transform():
    """Test that an error is raised when the wrong transform is passed."""
    with pytest.raises(ValueError):
        _list_spatial_augmentations(
            augmentations=[XYFlipModel(), N2VManipulateModel()],
        )


def test_create_data_configuration_train_dataloader_params(minimum_data):
    """Test that shuffle is added silently to the train_dataloader_params."""
    config_dict = minimum_data
    config_dict["train_dataloader_params"] = {"num_workers": 4}

    config = _create_data_configuration(batch_size=1, augmentations=[], **config_dict)
    assert "shuffle" in config.train_dataloader_params
    assert config.train_dataloader_params["shuffle"]


def test_supervised_configuration_passing_transforms():
    """Test that transforms can be passed to the configuration."""
    config_dict = _create_supervised_config_dict(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        augmentations=[XYFlipModel()],
    )
    config = Configuration(**config_dict)

    assert len(config.data_config.transforms) == 1
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value


def test_model_creation():
    """Test that the correct parameters are passed to the model."""
    model_kwargs = {
        "depth": 4,
        "conv_dims": 2,
        "n2v2": False,
        "in_channels": 2,
        "num_classes": 5,
        "independent_channels": False,
    }

    # choose different parameters
    axes = "XYZ"
    conv_dims = 3
    in_channels = 3
    num_classes = 4
    independent_channels = True
    use_n2v2 = True

    model = _create_unet_configuration(
        axes=axes,
        n_channels_in=in_channels,
        n_channels_out=num_classes,
        independent_channels=independent_channels,
        use_n2v2=use_n2v2,
        model_params=model_kwargs,
    )

    assert model.depth == model_kwargs["depth"]
    assert model.conv_dims == conv_dims
    assert model.n2v2 == use_n2v2
    assert model.in_channels == in_channels
    assert model.num_classes == num_classes
    assert model.independent_channels == independent_channels


def test_create_configuration():
    """Test that the methods correctly passes all parameters."""
    algorithm = "care"
    experiment_name = "test"
    data_type = "tiff"
    axes = "CYX"
    patch_size = [128, 128]
    batch_size = 8
    transform_list = [XYFlipModel(), XYRandomRotate90Model()]
    independent_channels = False
    loss = "mse"
    n_channels_in = 2
    n_channels_out = 3
    logger = "tensorboard"
    model_params = {
        "depth": 5,
    }
    optimizer = "SGD"
    optimizer_params = {
        "lr": 0.07,
    }
    lr_scheduler = "StepLR"
    lr_scheduler_params = {
        "step_size": 19,
    }
    train_dataloader_params = {
        "num_workers": 4,
        "shuffle": True,
    }
    val_dataloader_params = {
        "num_workers": 4,
    }
    checkpoint_params = {
        "every_n_epochs": 9,
    }

    # instantiate config
    config = Configuration(
        **_create_supervised_config_dict(
            algorithm=algorithm,
            experiment_name=experiment_name,
            data_type=data_type,
            axes=axes,
            patch_size=patch_size,
            batch_size=batch_size,
            augmentations=transform_list,
            independent_channels=independent_channels,
            loss=loss,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            logger=logger,
            model_params=model_params,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            train_dataloader_params=train_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            checkpoint_params=checkpoint_params,
        )
    )

    assert config.algorithm_config.algorithm == algorithm
    assert config.experiment_name == experiment_name
    assert config.data_config.data_type == data_type
    assert config.data_config.axes == axes
    assert config.data_config.patch_size == patch_size
    assert config.data_config.batch_size == batch_size
    assert config.data_config.transforms == transform_list
    assert config.algorithm_config.model.independent_channels == independent_channels
    assert config.algorithm_config.loss == loss
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_out
    assert config.training_config.logger == logger
    assert config.algorithm_config.model.depth == model_params["depth"]
    assert config.algorithm_config.optimizer.name == optimizer
    assert config.algorithm_config.optimizer.parameters == optimizer_params
    assert config.algorithm_config.lr_scheduler.name == lr_scheduler
    assert config.algorithm_config.lr_scheduler.parameters == lr_scheduler_params
    assert config.data_config.train_dataloader_params == train_dataloader_params
    assert config.data_config.val_dataloader_params == val_dataloader_params
    assert (
        config.training_config.checkpoint_callback.every_n_epochs
        == checkpoint_params["every_n_epochs"]
    )


def test_supervised_configuration_error_with_channel_axes():
    """Test that an error is raised if channels are in axes, but the input channel
    number is not specified."""
    with pytest.raises(ValueError):
        _create_supervised_config_dict(
            algorithm="n2n",
            experiment_name="test",
            data_type="tiff",
            axes="CYX",
            patch_size=[64, 64],
            batch_size=8,
        )


def test_supervised_configuration_singleton_channel():
    """Test that no error is raised if channels are in axes, and the input channel is
    1."""
    _create_supervised_config_dict(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        n_channels_in=1,
    )


def test_supervised_configuration_no_channel():
    """Test that no error is raised without channel and number of inputs."""
    _create_supervised_config_dict(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
    )


def test_supervised_configuration_error_without_channel_axes():
    """Test that an error is raised if channels are not in axes, but the input channel
    number is specified and greater than 1."""
    with pytest.raises(ValueError):
        _create_supervised_config_dict(
            algorithm="n2n",
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            n_channels_in=2,
        )


def test_supervised_configuration_channels():
    """Test that no error is raised if channels are in axes and the input channel
    are specified."""
    _create_supervised_config_dict(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        n_channels_in=4,
    )


def test_n2n_configuration():
    """Test that N2N configuration can be created."""
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        train_dataloader_params={"num_workers": 2},
    )
    assert config.algorithm_config.algorithm == "n2n"


def test_n2n_configuration_n_channels():
    """Test the behaviour of the number of channels in and out."""
    n_channels_in = 4
    n_channels_out = 5

    # n_channels_out not specified
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        n_channels_in=n_channels_in,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_in

    # specify n_channels_out
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_out


def test_n2n_configuration_limit_train_batches():
    """Num_steps parameter is passed to trainer config as limit_train_batches."""
    num_steps = 15

    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_steps=num_steps,
    )
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps
    )

    # Test with float value
    num_steps_float = 0.3
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_steps=num_steps_float,
    )
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps_float
    )

    # Test without num_steps (should not be in config)
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
    )
    assert "limit_train_batches" not in config.training_config.lightning_trainer_config


def test_care_configuration():
    """Test that CARE configuration can be created."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        train_dataloader_params={"num_workers": 2},
    )
    assert config.algorithm_config.algorithm == "care"


def test_care_configuration_n_channels():
    """Test the behaviour of the number of channels in and out."""
    n_channels_in = 4
    n_channels_out = 5

    # n_channels_out not specified
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        n_channels_in=n_channels_in,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_in

    # specify n_channels_out
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_out


def test_care_configuration_limit_train_batches():
    """Num_steps parameter is passed to trainer config as limit_train_batches."""
    num_steps = 10

    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_steps=num_steps,
    )
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps
    )

    # Test with float value
    num_steps_float = 0.5
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_steps=num_steps_float,
    )
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps_float
    )

    # Test without num_steps (should not be in config)
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
    )
    assert "limit_train_batches" not in config.training_config.lightning_trainer_config


def test_care_configuration_trainer_params():
    """Test that trainer_params are correctly passed to CARE trainer config."""
    num_epochs = 40
    trainer_params = {
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        "gradient_clip_val": 0.8,
        "check_val_every_n_epoch": 3,
        "deterministic": True,
    }

    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=num_epochs,
        trainer_params=trainer_params,
    )

    assert config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
    for key, value in trainer_params.items():
        assert config.training_config.lightning_trainer_config[key] == value


def test_care_configuration_trainer_params_override():
    """Test that explicit parameters override trainer_params in CARE configuration."""
    num_epochs = 35
    num_steps = 750

    trainer_params = {
        "max_epochs": 200,
        "limit_train_batches": 3000,
        "precision": "32-true",
    }

    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=num_epochs,
        num_steps=num_steps,
        trainer_params=trainer_params,
    )

    # Explicit parameters should override
    assert config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps
    )
    # Non-conflicting should be preserved
    assert config.training_config.lightning_trainer_config["precision"] == "32-true"


def test_n2v_configuration():
    """Test that N2V configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        train_dataloader_params={"num_workers": 2},
    )
    assert isinstance(config.algorithm_config, N2VAlgorithm)


def test_n2v_configuration_no_aug():
    """Test the default n2v transforms."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        augmentations=[],
    )
    assert config.data_config.transforms == []


def test_n2v_configuration_n2v2_structn2v():
    """Test that N2V manipulate is ignored when explicitely passed."""
    use_n2v2 = True
    roi_size = 15
    masked_pixel_percentage = 0.5
    struct_mask_axis = SupportedStructAxis.HORIZONTAL.value
    struct_n2v_span = 15

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        use_n2v2=use_n2v2,  # median strategy
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        struct_n2v_axis=struct_mask_axis,
        struct_n2v_span=struct_n2v_span,
    )
    assert (
        config.algorithm_config.n2v_config.strategy
        == SupportedPixelManipulation.MEDIAN.value
    )
    assert config.algorithm_config.n2v_config.roi_size == roi_size
    assert (
        config.algorithm_config.n2v_config.masked_pixel_percentage
        == masked_pixel_percentage
    )
    assert config.algorithm_config.n2v_config.struct_mask_axis == struct_mask_axis
    assert config.algorithm_config.n2v_config.struct_mask_span == struct_n2v_span


def test_n2v_configuration_limit_train_batches():
    """Num_steps parameter is passed to trainer config as limit_train_batches."""
    num_steps = 20

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_steps=num_steps,
    )
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps
    )

    # Test with float value
    num_steps_float = 0.7
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_steps=num_steps_float,
    )
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps_float
    )

    # Test without num_steps (should not be in config)
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
    )
    assert "limit_train_batches" not in config.training_config.lightning_trainer_config


def test_n2v_configuration_num_epochs():
    """Test that num_epochs parameter is correctly passed to trainer config."""
    num_epochs = 50

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=num_epochs,
    )
    assert config.training_config.lightning_trainer_config["max_epochs"] == num_epochs

    # Test with num_epochs=None (should not be in config)
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=None,
    )
    assert "max_epochs" not in config.training_config.lightning_trainer_config


def test_n2v_configuration_num_steps():
    """Test that num_steps parameter is correctly passed to trainer config."""
    num_steps = 1000

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_steps=num_steps,
    )
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps
    )

    # Test without num_steps (should not be in config)
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
    )
    assert "limit_train_batches" not in config.training_config.lightning_trainer_config


def test_n2v_configuration_num_epochs_and_num_steps():
    """Test that both num_epochs and num_steps can be set simultaneously."""
    num_epochs = 25
    num_steps = 500

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=num_epochs,
        num_steps=num_steps,
    )
    assert config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps
    )


def test_n2v_configuration_trainer_params():
    """Test that various trainer_params are correctly passed to trainer config."""
    trainer_params = {
        "accelerator": "gpu",
        "devices": 2,
        "precision": 16,
        "gradient_clip_val": 0.5,
        "accumulate_grad_batches": 4,
        "check_val_every_n_epoch": 5,
        "log_every_n_steps": 10,
        "enable_checkpointing": True,
        "enable_progress_bar": False,
        "enable_model_summary": True,
    }

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        trainer_params=trainer_params,
    )

    for key, value in trainer_params.items():
        assert config.training_config.lightning_trainer_config[key] == value


def test_n2v_configuration_trainer_params_with_timing():
    """Test trainer_params with timing-related parameters."""
    trainer_params = {
        "min_epochs": 5,
        "min_steps": 100,
        "max_time": "00:01:00:00",  # 1 hour
        "val_check_interval": 0.25,
    }

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        trainer_params=trainer_params,
    )

    for key, value in trainer_params.items():
        assert config.training_config.lightning_trainer_config[key] == value


def test_n2v_configuration_trainer_params_override():
    """Test that explicit parameters override trainer_params."""
    num_epochs = 30
    num_steps = 800

    # trainer_params has conflicting values
    trainer_params = {
        "max_epochs": 100,
        "limit_train_batches": 0.5,
        "accelerator": "cpu",
    }

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=num_epochs,
        num_steps=num_steps,
        trainer_params=trainer_params,
    )

    # Explicit parameters should override trainer_params
    assert config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps
    )

    # Non-conflicting trainer_params should be preserved
    assert config.training_config.lightning_trainer_config["accelerator"] == "cpu"


def test_n2v_configuration_trainer_params_profiler():
    """Test trainer_params with profiler settings."""
    trainer_params = {
        "profiler": "simple",
        "detect_anomaly": True,
        "benchmark": False,
        "deterministic": True,
    }

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        trainer_params=trainer_params,
    )

    for key, value in trainer_params.items():
        assert config.training_config.lightning_trainer_config[key] == value


def test_n2v_configuration_trainer_params_distributed():
    """Test trainer_params with distributed training settings."""
    trainer_params = {
        "strategy": "ddp",
        "num_nodes": 2,
        "sync_batchnorm": True,
        "use_distributed_sampler": True,
    }

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        trainer_params=trainer_params,
    )

    for key, value in trainer_params.items():
        assert config.training_config.lightning_trainer_config[key] == value


def test_n2v_configuration_all_trainer_combinations():
    """Test comprehensive combination of all trainer parameters."""
    num_epochs = 15
    num_steps = 300

    trainer_params = {
        "accelerator": "auto",
        "devices": "auto",
        "precision": "32-true",
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "accumulate_grad_batches": 2,
        "check_val_every_n_epoch": 2,
        "val_check_interval": 1.0,
        "log_every_n_steps": 50,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "deterministic": False,
        "benchmark": True,
        "inference_mode": True,
        "use_distributed_sampler": True,
        "detect_anomaly": False,
        "barebones": False,
        "reload_dataloaders_every_n_epochs": 0,
    }

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=num_epochs,
        num_steps=num_steps,
        trainer_params=trainer_params,
    )

    # Check explicit parameters
    assert config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
    assert (
        config.training_config.lightning_trainer_config["limit_train_batches"]
        == num_steps
    )

    # Check trainer_params
    for key, value in trainer_params.items():
        assert config.training_config.lightning_trainer_config[key] == value


def test_n2v_configuration_empty_trainer_params():
    """Test that empty trainer_params dict works correctly."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=None,
        trainer_params={},
    )

    # Should have empty dict (no trainer params)
    assert config.training_config.lightning_trainer_config == {}


def test_n2v_configuration_trainer_params_none():
    """Test that trainer_params=None works correctly (default behavior)."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=None,
        trainer_params=None,
    )

    # Should have empty dict when trainer_params is None
    assert config.training_config.lightning_trainer_config == {}


def test_checkpoint_model_save_top_k_default():
    """CheckpointModel's default save_top_k=3 doesn't conflict with trainer params."""
    from careamics.config.callback_model import CheckpointModel

    # Test default save_top_k value
    checkpoint_model = CheckpointModel()
    assert checkpoint_model.save_top_k == 3

    # Test that default works with standard trainer configuration
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=10,
        checkpoint_params={"save_top_k": 3},  # explicit default
    )

    assert config.training_config.checkpoint_callback.save_top_k == 3


def test_checkpoint_model_comprehensive_checkpoint_params():
    """Test comprehensive checkpoint parameters with save_top_k=3 default."""
    checkpoint_params = {
        "save_top_k": 3,  # explicit default
        "monitor": "val_loss",
        "mode": "min",
        "save_weights_only": False,
        "save_last": True,
        "every_n_epochs": 1,
        "verbose": True,
    }

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=10,
        checkpoint_params=checkpoint_params,
    )

    # Verify all checkpoint parameters are set correctly
    checkpoint_callback = config.training_config.checkpoint_callback
    assert checkpoint_callback.save_top_k == 3
    assert checkpoint_callback.monitor == "val_loss"
    assert checkpoint_callback.mode == "min"
    assert checkpoint_callback.save_weights_only is False
    assert checkpoint_callback.save_last is True
    assert checkpoint_callback.every_n_epochs == 1
    assert checkpoint_callback.verbose is True


def test_checkpoint_model_save_top_k_with_resource_constraints():
    """Test checkpoint save_top_k behavior with resource constraint scenarios."""
    # Test with limited training steps - checkpointing should still work
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_steps=50,
        checkpoint_params={"save_top_k": 3, "every_n_train_steps": 10},
    )

    assert config.training_config.lightning_trainer_config["limit_train_batches"] == 50
    assert config.training_config.checkpoint_callback.save_top_k == 3
    assert config.training_config.checkpoint_callback.every_n_train_steps == 10


def test_checkpoint_model_save_top_k_edge_cases():
    """Test edge cases for save_top_k parameter."""
    import pytest

    from careamics.config.callback_model import CheckpointModel

    # Test that save_top_k accepts valid range (-1 to 100)
    checkpoint_model = CheckpointModel(save_top_k=-1)  # Save all
    assert checkpoint_model.save_top_k == -1

    checkpoint_model = CheckpointModel(save_top_k=0)  # Save none
    assert checkpoint_model.save_top_k == 0

    checkpoint_model = CheckpointModel(save_top_k=100)  # Maximum allowed
    assert checkpoint_model.save_top_k == 100

    # Test that invalid values are rejected (assuming validation exists)
    # This depends on the Pydantic field constraints
    with pytest.raises(ValueError):
        CheckpointModel(save_top_k=101)  # Above maximum

    with pytest.raises(ValueError):
        CheckpointModel(save_top_k=-2)  # Below minimum


def test_checkpoint_model_save_top_k_with_development_modes():
    """Test checkpoint save_top_k with various development and debugging modes."""
    # Test with overfit_batches (overfitting mode)
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        trainer_params={"overfit_batches": 1},
        checkpoint_params={"save_top_k": 3},
    )
    assert config.training_config.lightning_trainer_config["overfit_batches"] == 1
    assert config.training_config.checkpoint_callback.save_top_k == 3

    # Test with limit_val_batches
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        trainer_params={"limit_val_batches": 0.1},
        checkpoint_params={"save_top_k": 3},
    )
    assert config.training_config.lightning_trainer_config["limit_val_batches"] == 0.1
    assert config.training_config.checkpoint_callback.save_top_k == 3

    # Test with num_sanity_val_steps=0 (disable sanity checks)
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        trainer_params={"num_sanity_val_steps": 0},
        checkpoint_params={"save_top_k": 3},
    )
    assert config.training_config.lightning_trainer_config["num_sanity_val_steps"] == 0
    assert config.training_config.checkpoint_callback.save_top_k == 3


def test_checkpoint_model_save_top_k_cross_algorithm_consistency():
    """Test that save_top_k behavior is consistent across all three algorithms."""
    checkpoint_params = {"save_top_k": 5, "monitor": "val_loss"}

    # Test N2V
    n2v_config = create_n2v_configuration(
        experiment_name="test_n2v",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=10,
        checkpoint_params=checkpoint_params,
    )

    # Test CARE
    care_config = create_care_configuration(
        experiment_name="test_care",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=10,
        checkpoint_params=checkpoint_params,
    )

    # Test N2N
    n2n_config = create_n2n_configuration(
        experiment_name="test_n2n",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=10,
        checkpoint_params=checkpoint_params,
    )

    # All should have the same checkpoint configuration
    assert n2v_config.training_config.checkpoint_callback.save_top_k == 5
    assert care_config.training_config.checkpoint_callback.save_top_k == 5
    assert n2n_config.training_config.checkpoint_callback.save_top_k == 5

    assert n2v_config.training_config.checkpoint_callback.monitor == "val_loss"
    assert care_config.training_config.checkpoint_callback.monitor == "val_loss"
    assert n2n_config.training_config.checkpoint_callback.monitor == "val_loss"


def test_microsplit_configuration(tmp_path: Path, create_dummy_noise_model):
    """Test that MicroSplit configuration can be created."""
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    config = create_microsplit_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        predict_logvar="pixelwise",
        nm_paths=[tmp_path / "dummy_noise_model.npz"],
        data_stats=[0, 0],
        train_dataloader_params={"num_workers": 0},
    )
    assert config.algorithm_config.algorithm == "microsplit"
    assert isinstance(config.algorithm_config, MicroSplitAlgorithm)
    assert config.algorithm_config.model.architecture == "LVAE"
    assert config.algorithm_config.noise_model_likelihood is not None
