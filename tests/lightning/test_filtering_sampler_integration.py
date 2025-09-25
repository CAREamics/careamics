import numpy as np

# TODO: filter is applied before normalization, what to do with the thresholds?
# TODO: test if random seed propagates correctly to the filters
# TODO: tests for the sampler itself
# TODO: should the tiling be changed in validation?
from pytorch_lightning import Trainer

from careamics.config.configuration_factories import create_n2v_configuration
from careamics.config.data import NGDataConfig
from careamics.lightning.callbacks import (
    DatasetReshuffleCallback,
    HyperParametersCallback,
)
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule
from careamics.lightning.dataset_ng.lightning_modules import N2VModule


def create_default_configuration():
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(64, 64),
        batch_size=8,
    )
    ng_data_config = NGDataConfig(
        data_type="array",
        patching={
            "name": "random",
            "patch_size": (64, 64),
        },
        axes="YX",
        batch_size=1,
        seed=42,
    )
    return config, ng_data_config


def test_callback_initialization(tmp_path):
    """Test if default DatasetReshuffleCallback callback is added by default and is not overriding optional ones"""
    image = np.ones((512, 512))
    config, ng_data_config = create_default_configuration()

    data_module = CareamicsDataModule(
        data_config=ng_data_config,
        train_data=image,
        val_data=image,
    )

    additional_callback = HyperParametersCallback(config)
    model = N2VModule(config.algorithm_config)
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[additional_callback],
        log_every_n_steps=1,
    )
    # calling fit is nessesary to append the callbacks and datamodule
    trainer.fit(model, datamodule=data_module)

    print(trainer.callbacks)

    assert len(trainer.callbacks) == 5  # 3 are default, 1 custom default, 1 additional
    # check that DatasetReshuffleCallback is added
    assert any(
        isinstance(callback, DatasetReshuffleCallback) for callback in trainer.callbacks
    )
    # check that HyperParametersCallback is not overridden
    assert any(
        isinstance(callback, HyperParametersCallback) for callback in trainer.callbacks
    )


def test_callback_integration(tmp_path):
    image = np.ones((512, 512))
    config, ng_data_config = create_default_configuration()

    data_module = CareamicsDataModule(
        data_config=ng_data_config,
        train_data=image,
        val_data=image,
    )
    model = N2VModule(config.algorithm_config)
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
    )
    # calling fit is nessesary to append the callbacks and datamodule
    trainer.fit(model, datamodule=data_module)

    first_sample = next(iter(data_module.train_dataloader()))

    trainer.fit(model, datamodule=data_module)

    second_sample = next(iter(data_module.train_dataloader()))

    # samples are different
    assert not np.array_equal(first_sample, second_sample)

    # no patches are filtered
    assert len(data_module.train_dataset) == len(data_module.train_dataloader())


def test_sampler_integration_with_coord_filter(tmp_path):
    image = np.ones((512, 512))
    mask = np.zeros_like(image, dtype=bool)
    # make a mask in the bottom right corner
    mask[image.shape[0] // 2 :, image.shape[1] // 2 :] = True

    config, ng_data_config = create_default_configuration()

    model = N2VModule(config.algorithm_config)
    mask = np.zeros_like(image, dtype=bool)
    mask[image.shape[0] // 2 :, image.shape[1] // 2 :] = True
    ng_data_config.coord_filter = {"name": "mask", "coverage": 1}

    data_module = CareamicsDataModule(
        data_config=ng_data_config,
        train_data=image,
        val_data=image,
        train_data_mask=mask,
    )
    data_module.setup("fit")
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)

    # check that some patches are filtered
    assert len(data_module.train_dataset) != len(data_module.train_dataloader())


def test_sampler_integration_with_patch_filter(tmp_path):
    image = np.ones((512, 512))
    config, ng_data_config = create_default_configuration()

    model = N2VModule(config.algorithm_config)
    ng_data_config.patch_filter = {"name": "max", "threshold": 0.5}

    data_module = CareamicsDataModule(
        data_config=ng_data_config, train_data=image, val_data=image
    )
    data_module.setup("fit")
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)

    # the length of the dataset should still be the same after filtering
    assert len(data_module.train_dataset) == len(data_module.train_dataloader())
