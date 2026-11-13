"""Tests for the checkpoint record callback."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from careamics.lightning.callbacks import CheckpointRecordCallback
from careamics.lightning.callbacks.checkpoint_record_callback import (
    CHECKPOINT_RECORD_BACKUP_FILENAME,
    CHECKPOINT_RECORD_FILENAME,
)


@pytest.fixture
def model_checkpoint(tmp_path: Path) -> Mock:
    """Create a mocked model checkpoint callback."""
    checkpoint_dir = tmp_path / "checkpoints" / "experiment_0"
    callback = Mock(spec=ModelCheckpoint)
    callback.dirpath = checkpoint_dir
    callback.last_model_path = str(checkpoint_dir / "experiment_last.ckpt")
    callback.best_model_path = str(checkpoint_dir / "experiment_epoch_02.ckpt")
    callback.monitor = "val_loss"
    callback.mode = "min"
    callback.save_top_k = 2
    callback.best_k_models = {
        str(checkpoint_dir / "experiment_epoch_01.ckpt"): 0.4,
        str(checkpoint_dir / "experiment_epoch_02.ckpt"): 0.2,
    }

    k_best_model_names = list(callback.best_k_models.keys())
    # create empty mock checkpoint files
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(k_best_model_names[0], "w"):
        ...
    with open(k_best_model_names[1], "w"):
        ...
    with open(callback.last_model_path, "w"):
        ...

    return callback


@pytest.fixture
def trainer(model_checkpoint: Mock) -> Mock:
    """Create a mocked trainer with checkpointing enabled."""
    trainer = Mock(spec=Trainer)
    trainer.is_global_zero = True
    trainer.checkpoint_callback = model_checkpoint
    return trainer


def test_on_save_checkpoint_creates_record(trainer: Mock) -> None:
    """Test that the checkpoint hook creates a JSON record."""
    callback = CheckpointRecordCallback("best")

    callback.on_save_checkpoint(trainer, Mock(), {})

    checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)
    with (checkpoint_dir / CHECKPOINT_RECORD_FILENAME).open() as file:
        record = json.load(file)
    assert record == {
        "last": "experiment_last.ckpt",
        "best": "experiment_epoch_02.ckpt",
        "monitor": "val_loss",
        "mode": "min",
        "best_k": [
            {"name": "experiment_epoch_02.ckpt", "score": 0.2},
            {"name": "experiment_epoch_01.ckpt", "score": 0.4},
        ],
        "checkpoints": [
            "experiment_epoch_01.ckpt",
            "experiment_epoch_02.ckpt",
            "experiment_last.ckpt",
        ],
        "selected": "experiment_epoch_02.ckpt",
    }


def test_on_save_checkpoint_backs_up_existing_record(
    trainer: Mock,
) -> None:
    """Test that a subsequent checkpoint hook backs up the previous record."""
    callback = CheckpointRecordCallback("best")
    callback.on_save_checkpoint(trainer, Mock(), {})
    checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)
    record_path = checkpoint_dir / CHECKPOINT_RECORD_FILENAME
    original_record = record_path.read_text()

    callback.on_save_checkpoint(trainer, Mock(), {})

    backup_path = checkpoint_dir / CHECKPOINT_RECORD_BACKUP_FILENAME
    assert backup_path.read_text() == original_record
