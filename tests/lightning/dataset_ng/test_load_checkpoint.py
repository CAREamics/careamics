from pathlib import Path

import numpy as np
import pytest
from pytorch_lightning import Callback, Trainer

from careamics.config import TrainingConfig
from careamics.config.ng_configs import NGConfiguration
from careamics.config.ng_factories import (
    create_advanced_care_config,
    create_advanced_n2v_config,
)
from careamics.lightning.callbacks.careamics_checkpoint_info_callback import (
    CareamicsCheckpointInfo,
)
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule
from careamics.lightning.dataset_ng.lightning_modules import CAREModule, N2VModule
from careamics.lightning.dataset_ng.load_checkpoint import (
    load_config_from_checkpoint,
    load_module_from_checkpoint,
)


@pytest.fixture(params=[True, False], ids=["w_ckpt_info", "w/o_ckpt_info"])
def checkpoint_trainer(request):
    if request.param:
        info_callback = CareamicsCheckpointInfo(
            careamics_version="0.1.0",
            experiment_name="testing",
            training_config=TrainingConfig(),
        )
        callbacks: list[Callback] = [info_callback]
    else:
        info_callback = None
        callbacks = []
    return Trainer(max_epochs=1, callbacks=callbacks), info_callback


@pytest.fixture(params=["n2v", "care"])
def checkpoint(
    request,
    checkpoint_trainer: tuple[Trainer, CareamicsCheckpointInfo | None],
    tmp_path: Path,
):

    train_data = np.random.rand(32, 32).astype(np.float32)
    val_data = np.random.rand(16, 16).astype(np.float32)
    train_data_target = np.random.rand(32, 32).astype(np.float32)
    val_data_target = np.random.rand(16, 16).astype(np.float32)

    if request.param == "n2v":
        module_cls = N2VModule
        config = create_advanced_n2v_config(
            experiment_name="checkpoint_fixture_n2v",
            data_type="array",
            axes="YX",
            patch_size=[16, 16],
            batch_size=2,
            masked_pixel_percentage=0.4,
            normalization="mean_std",
            normalization_params={"input_means": [0.5], "input_stds": [0.3]},
        )
        data = {"train_data": train_data, "val_data": val_data}

    elif request.param == "care":
        module_cls = CAREModule
        config = create_advanced_care_config(
            experiment_name="checkpoint_fixture_care",
            data_type="array",
            axes="YX",
            patch_size=[16, 16],
            batch_size=2,
            normalization="mean_std",
            normalization_params={
                "input_means": [0.5],
                "input_stds": [0.3],
                "target_means": [0.5],
                "target_stds": [0.3],
            },
        )
        data = {
            "train_data": train_data,
            "val_data": val_data,
            "train_data_target": train_data_target,
            "val_data_target": val_data_target,
        }
    else:
        raise ValueError(f"Unexpected algorithm value: {request.param}")

    module = module_cls(config.algorithm_config)
    dmodule = CareamicsDataModule(data_config=config.data_config, **data)
    # trainer = Trainer(max_epochs=1)
    trainer, info_callback = checkpoint_trainer
    trainer.fit(model=module, datamodule=dmodule)
    ckpt_path = tmp_path / "checkpoint_test_fixture.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path, module_cls, config, info_callback


def test_load_module_from_checkpoint(checkpoint):
    checkpoint_path, expected_module_cls, _, _ = checkpoint
    loaded_module = load_module_from_checkpoint(checkpoint_path)
    assert isinstance(loaded_module, expected_module_cls)


def test_load_config_from_checkpoint(
    checkpoint: tuple[Path, type, NGConfiguration, CareamicsCheckpointInfo | None],
):
    checkpoint_path, _, expected_config, info_callback = checkpoint
    config = load_config_from_checkpoint(checkpoint_path)

    assert config.algorithm_config == expected_config.algorithm_config
    assert config.data_config == expected_config.data_config
    if info_callback is not None:
        assert config.experiment_name == info_callback.experiment_name
        assert config.version == info_callback.careamics_version
        assert config.training_config == info_callback.training_config
