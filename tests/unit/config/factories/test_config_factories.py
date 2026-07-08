"""Shared tests between CARE, N2N and N2V factories."""

import itertools

import pytest

from careamics.config.configuration import Configuration
from careamics.config.data.patch_filter import MeanStdPatchFilterConfig
from careamics.config.factories import (
    create_advanced_care_config,
    create_advanced_n2n_config,
    create_advanced_n2v_config,
)
from careamics.config.support import (
    SupportedData,
    SupportedLogger,
    SupportedNormalization,
    SupportedOptimizer,
    SupportedScheduler,
)

# --- Utility functions


def is_subdict(subdict, dictionary):
    """Check if subdict is a subset of dictionary."""
    return all(item in dictionary.items() for item in subdict.items())


def test_is_subdict():
    """Test the is_subdict utility function."""
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"a": 1, "b": 2}
    dict3 = {"a": 1, "b": 3}
    dict4 = {"d": 4}

    assert is_subdict(dict2, dict1) is True
    assert is_subdict(dict3, dict1) is False
    assert is_subdict(dict4, dict1) is False


# --- List of parameters re-used in tests

FACTORIES = [
    create_advanced_care_config,
    create_advanced_n2n_config,
    create_advanced_n2v_config,
]

DATA_TYPE = [d.value for d in SupportedData if d.value != "czi"]  # separate czi tests

SIMPLE_AXES_2D = ["YX"]
"""2D axes that should not interact with other parameters. Presence of C tested with
another set of values."""

SIMPLE_AXES_3D = ["ZYX"]
"""3D axes that should not interact with other parameters. Presence of C tested with
another set of values."""

PATCH_SIZE_2D = [(32, 32)]

PATCH_SIZE_3D = [(16, 32, 32)]


# --- Unit tests


@pytest.mark.parametrize(
    "factory, data_type, axes, patch_size",
    # 2D
    list(
        itertools.product(
            FACTORIES,
            DATA_TYPE,
            SIMPLE_AXES_2D,
            PATCH_SIZE_2D,
        )
    )
    # 3D
    + list(
        itertools.product(
            FACTORIES,
            DATA_TYPE,
            SIMPLE_AXES_3D,
            PATCH_SIZE_3D,
        )
    ),
)
def test_orthogonal_params(
    factory,
    data_type,
    axes,
    patch_size,
):
    """Test that orthogal parameters are correctly passed to the configuration.

    Note that these parameters should not interact (except axes and patch size), and are
    internally passed directly to other configuration functions.
    """
    exp_name = "test_orthogonal"
    batch_size = 8
    num_epochs = 50
    num_steps = 200
    n_val = 25
    in_memory = True if data_type not in ["czi", "zarr"] else False
    patch_filter = MeanStdPatchFilterConfig(mean_threshold=0.5)
    norm = SupportedNormalization.QUANTILE.value
    norm_params = {"lower_quantiles": [0.2]}
    trainer_params = {"check_val_every_n_epoch": 5}
    model_params = {"depth": 4}
    optimizer = SupportedOptimizer.SGD.value
    optimizer_params = {"lr": 0.01}
    scheduler = SupportedScheduler.STEP_LR.value
    scheduler_params = {"step_size": 10}
    num_workers = 4
    train_dataloader_params = {"pin_memory": True}
    val_dataloader_params = {"drop_last": True}
    checkpoint_params = {"save_top_k": 8}
    logger = SupportedLogger.WANDB.value
    seed = 42

    config: Configuration = factory(
        experiment_name=exp_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_steps=num_steps,
        n_val_patches=n_val,
        in_memory=in_memory,
        patch_filter_config=patch_filter,
        normalization=norm,
        normalization_params=norm_params,
        trainer_params=trainer_params,
        model_params=model_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        lr_scheduler=scheduler,
        lr_scheduler_params=scheduler_params,
        num_workers=num_workers,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
        checkpoint_params=checkpoint_params,
        logger=logger,
        seed=seed,
    )

    assert config.experiment_name == exp_name
    assert config.data_config.data_type == data_type
    assert config.data_config.axes == axes
    assert config.data_config.patching.patch_size == patch_size

    assert config.data_config.batch_size == batch_size
    assert config.training_config.trainer_params["max_epochs"] == num_epochs
    assert config.training_config.trainer_params["limit_train_batches"] == num_steps
    assert config.data_config.n_val_patches == n_val
    assert config.data_config.in_memory == in_memory
    assert config.data_config.num_workers == num_workers

    assert config.data_config.patch_filter == patch_filter
    assert config.data_config.normalization.name == norm

    assert is_subdict(norm_params, config.data_config.normalization.model_dump())
    assert is_subdict(trainer_params, config.training_config.trainer_params)
    assert is_subdict(model_params, config.algorithm_config.model.model_dump())

    assert config.algorithm_config.optimizer.name == optimizer
    assert is_subdict(optimizer_params, config.algorithm_config.optimizer.parameters)

    assert config.algorithm_config.lr_scheduler.name == scheduler
    assert is_subdict(scheduler_params, config.algorithm_config.lr_scheduler.parameters)

    assert config.training_config.checkpoint_params == checkpoint_params

    assert is_subdict(
        train_dataloader_params, config.data_config.train_dataloader_params
    )
    assert is_subdict(val_dataloader_params, config.data_config.val_dataloader_params)
    assert config.training_config.logger == logger
    assert config.data_config.seed == seed


@pytest.mark.parametrize("factory", FACTORIES)
def test_no_augmentation(factory):
    """Test that no augmentation is correctly passed to the configuration."""
    config: Configuration = factory(
        experiment_name="test_no_aug",
        data_type="tiff",
        axes="YX",
        patch_size=[32, 32],
        batch_size=8,
        augmentations=[],
    )
    assert len(config.data_config.augmentations) == 0
