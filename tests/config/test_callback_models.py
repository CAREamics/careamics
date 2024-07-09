from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from careamics.config.callback_model import CheckpointModel, EarlyStoppingModel


def test_defaults_checkpoint():
    """Test that a default CheckpointModel can initialize a ModelCheckpoint callback."""
    config = CheckpointModel()
    ModelCheckpoint(**config.model_dump())


def test_defaults_early_stopping():
    """Test that a default EarlyStoppingModel can initialize an EarlyStopping
    callback."""
    config = EarlyStoppingModel()
    EarlyStopping(**config.model_dump())
