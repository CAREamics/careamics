from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from careamics.config.lightning.callbacks.callback_config import (
    CheckpointConfig,
    EarlyStoppingConfig,
)


def test_defaults_checkpoint():
    """Test that a default CheckpointConfig can initialize a ModelCheckpoint
    callback."""
    config = CheckpointConfig()
    ModelCheckpoint(**config.model_dump())


def test_defaults_early_stopping():
    """Test that a default EarlyStoppingConfig can initialize an EarlyStopping
    callback."""
    config = EarlyStoppingConfig()
    EarlyStopping(**config.model_dump())
