"""Tests for the checkpoint record logger."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from careamics.lightning.loggers import CheckpointRecordLogger
from careamics.lightning.loggers import (
    checkpoint_record_logger as checkpoint_record_logger_module,
)
from careamics.lightning.loggers.checkpoint_record_logger import (
    CHECKPOINT_RECORD_BACKUP_FILENAME,
    CHECKPOINT_RECORD_FILENAME,
    get_checkpoints,
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


def test_after_save_checkpoint_creates_record(
    model_checkpoint: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the post-save hook creates a JSON record."""
    logger = CheckpointRecordLogger("best")
    checkpoint_dir = Path(model_checkpoint.dirpath)
    checkpoint_paths = [
        checkpoint_dir / "experiment_epoch_01.ckpt",
        checkpoint_dir / "experiment_epoch_02.ckpt",
        checkpoint_dir / "experiment_last.ckpt",
    ]
    # using monkey patch to ensure the checkpoints are always returned in this order
    # creation time ordering is not reliable on all machines.
    monkeypatch.setattr(
        checkpoint_record_logger_module,
        "get_checkpoints",
        lambda *_args, **_kwargs: checkpoint_paths,
    )

    logger.after_save_checkpoint(model_checkpoint)

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


def test_get_checkpoints_updates_checkpoint_list(tmp_path: Path) -> None:
    """Test that the checkpoint list is updated correctly."""

    (tmp_path / "experiment_epoch_01.ckpt").touch()
    (tmp_path / "experiment_epoch_02.ckpt").touch()
    (tmp_path / "experiment_last.ckpt").touch()
    # create dummy record
    record = {
        "last": "experiment_last.ckpt",
        "checkpoints": [
            "experiment_epoch_01.ckpt",
            "deleted_checkpoint.ckpt",
            "experiment_epoch_02.ckpt",
        ],
    }
    (tmp_path / CHECKPOINT_RECORD_FILENAME).write_text(json.dumps(record))

    # simulate another saving event
    (tmp_path / "experiment_epoch_03.ckpt").touch()
    (tmp_path / "experiment_last.ckpt").touch()

    # the deleted checkpoint should be removed and last should always remain last
    checkpoint_paths = get_checkpoints(checkpoint_dir=tmp_path)
    assert [path.name for path in checkpoint_paths] == [
        "experiment_epoch_01.ckpt",
        "experiment_epoch_02.ckpt",
        "experiment_epoch_03.ckpt",
        "experiment_last.ckpt",
    ]


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
