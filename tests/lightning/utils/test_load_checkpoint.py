from pathlib import Path

from careamics.config.configuration import Configuration
from careamics.lightning.utils.load_checkpoint import (
    load_config_from_checkpoint,
    load_module_from_checkpoint,
)


def test_load_module_from_checkpoint(checkpoint):
    checkpoint_path, expected_module_cls, _ = checkpoint
    loaded_module = load_module_from_checkpoint(checkpoint_path)
    assert isinstance(loaded_module, expected_module_cls)


def test_load_config_from_checkpoint(
    checkpoint: tuple[Path, type, Configuration],
):
    checkpoint_path, _, expected_config = checkpoint
    config = load_config_from_checkpoint(checkpoint_path)
    assert config == expected_config
    assert config.data_config.normalization.input_means is not None
