"""Tests for the checkpoint record logger."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from careamics.lightning.loggers import CheckpointRecordLogger
from careamics.lightning.loggers.checkpoint_record_logger import (
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

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint_path in (
        *callback.best_k_models,
        callback.last_model_path,
    ):
        Path(checkpoint_path).touch()

    return callback


def test_after_save_checkpoint_creates_record(model_checkpoint: Mock) -> None:
    """Test that the post-save hook creates a JSON record."""
    logger = CheckpointRecordLogger("best")

    logger.after_save_checkpoint(model_checkpoint)

    checkpoint_dir = Path(model_checkpoint.dirpath)
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


def test_after_save_checkpoint_backs_up_existing_record(
    model_checkpoint: Mock,
) -> None:
    """Test that a subsequent checkpoint save backs up the previous record."""
    logger = CheckpointRecordLogger("best")
    logger.after_save_checkpoint(model_checkpoint)
    checkpoint_dir = Path(model_checkpoint.dirpath)
    record_path = checkpoint_dir / CHECKPOINT_RECORD_FILENAME
    original_record = record_path.read_text()
    new_checkpoint = checkpoint_dir / "experiment_epoch_03.ckpt"
    new_checkpoint.touch()

    logger.after_save_checkpoint(model_checkpoint)

    backup_path = checkpoint_dir / CHECKPOINT_RECORD_BACKUP_FILENAME
    assert backup_path.read_text() == original_record
    with record_path.open() as file:
        updated_record = json.load(file)
    assert new_checkpoint.name in updated_record["checkpoints"]


def test_finalize_removes_deleted_checkpoint_from_record(
    model_checkpoint: Mock,
) -> None:
    """Test that finalization removes checkpoint paths deleted after saving."""
    logger = CheckpointRecordLogger("best")
    logger.after_save_checkpoint(model_checkpoint)
    deleted_checkpoint = Path(next(iter(model_checkpoint.best_k_models)))
    deleted_checkpoint.unlink()

    logger.finalize("success")

    record_path = Path(model_checkpoint.dirpath) / CHECKPOINT_RECORD_FILENAME
    with record_path.open() as file:
        finalized_record = json.load(file)
    assert deleted_checkpoint.name not in finalized_record["checkpoints"]
