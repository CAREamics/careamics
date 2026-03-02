from pathlib import Path

from careamics.config.ng_configs import NGConfiguration
from careamics.lightning.dataset_ng.load_checkpoint import (
    load_config_from_checkpoint,
    load_module_from_checkpoint,
)


def test_load_module_from_checkpoint(checkpoint):
    checkpoint_path, expected_module_cls, _ = checkpoint
    loaded_module = load_module_from_checkpoint(checkpoint_path)
    assert isinstance(loaded_module, expected_module_cls)


def test_load_config_from_checkpoint(
    checkpoint: tuple[Path, type, NGConfiguration],
):
    checkpoint_path, _, expected_config = checkpoint
    config = load_config_from_checkpoint(checkpoint_path)
    assert config == expected_config

    # assert config.algorithm_config == expected_config.algorithm_config
    # assert config.data_config == expected_config.data_config
    # if info_callback is not None:
    #     assert config.experiment_name == info_callback.experiment_name
    #     assert config.version == info_callback.careamics_version
    #     assert config.training_config == info_callback.training_config
