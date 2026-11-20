"""Tests for the checkpoint record logger."""

import json
from contextlib import nullcontext as no_error
from pathlib import Path
from unittest.mock import Mock

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from careamics.lightning import checkpoint_record_logger
from careamics.lightning.checkpoint_record_logger import (
    CHECKPOINT_RECORD_BACKUP_FILENAME,
    CHECKPOINT_RECORD_FILENAME,
    CheckpointRecordLogger,
    create_checkpoint_record_data,
    get_checkpoints,
    save_checkpoint_record,
)


def create_model_checkpoint(root_save_dir: Path, mode: str) -> Mock:
    """Create a mocked model checkpoint callback."""
    checkpoint_dir = root_save_dir / "checkpoints" / "experiment_0"
    callback = Mock(spec=ModelCheckpoint)
    callback.dirpath = checkpoint_dir
    callback.last_model_path = str(checkpoint_dir / "experiment_last.ckpt")
    callback.best_model_path = str(checkpoint_dir / "experiment_epoch_02.ckpt")
    callback.monitor = "val_loss"
    callback.mode = mode
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


@pytest.mark.parametrize("mode", ["min", "max", "unrecognized"])
def test_after_save_checkpoint_creates_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Test that the post-save hook creates a JSON record."""
    model_checkpoint = create_model_checkpoint(tmp_path, mode)
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
        checkpoint_record_logger,
        "get_checkpoints",
        lambda *_args, **_kwargs: checkpoint_paths,
    )
    best_k = [
        {"name": "experiment_epoch_02.ckpt", "score": 0.2},
        {"name": "experiment_epoch_01.ckpt", "score": 0.4},
    ]

    expected_error = no_error()
    if model_checkpoint.mode == "max":
        best_k = best_k[::-1]
    elif model_checkpoint.mode not in ["min", "max"]:
        expected_error = pytest.raises(
            ValueError, match="Unrecognized checkpoint monitoring mode"
        )

    with expected_error:
        logger.after_save_checkpoint(model_checkpoint)

        with (checkpoint_dir / CHECKPOINT_RECORD_FILENAME).open() as file:
            record = json.load(file)
        assert record == {
            "last": "experiment_last.ckpt",
            "best": "experiment_epoch_02.ckpt",
            "monitor": "val_loss",
            "mode": model_checkpoint.mode,
            "best_k": best_k,
            "checkpoints": [
                "experiment_epoch_01.ckpt",
                "experiment_epoch_02.ckpt",
                "experiment_last.ckpt",
            ],
            "selected": "experiment_epoch_02.ckpt",
        }


def test_after_save_checkpoint_backs_up_existing_record(tmp_path: Path) -> None:
    """Test that a subsequent checkpoint save backs up the previous record."""
    model_checkpoint = create_model_checkpoint(tmp_path, mode="min")
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


def test_no_checkpoints_record(tmp_path: Path):
    model_checkpoint = create_model_checkpoint(tmp_path, "min")
    model_checkpoint.best_k_models = {}
    model_checkpoint.best_model_path = ""

    data = create_checkpoint_record_data(model_checkpoint, "best")
    assert data["selected"] is None


def test_finalize_removes_deleted_checkpoint_from_record(tmp_path: Path) -> None:
    """Test that finalization removes checkpoint paths deleted after saving."""
    model_checkpoint = create_model_checkpoint(tmp_path, mode="min")
    logger = CheckpointRecordLogger("best")
    logger.after_save_checkpoint(model_checkpoint)
    deleted_checkpoint = Path(next(iter(model_checkpoint.best_k_models)))
    deleted_checkpoint.unlink()

    logger.finalize("success")

    record_path = Path(model_checkpoint.dirpath) / CHECKPOINT_RECORD_FILENAME
    with record_path.open() as file:
        finalized_record = json.load(file)
    assert deleted_checkpoint.name not in finalized_record["checkpoints"]


def test_dirpath_not_configured_error(tmp_path: Path):
    """Test there is an error if ModelCheckpoint.dirpath is None."""
    model_checkpoint = create_model_checkpoint(root_save_dir=tmp_path, mode="min")
    model_checkpoint.dirpath = None
    with pytest.raises(ValueError, match="`dirpath` configured"):
        save_checkpoint_record(model_checkpoint, "best")


# just to increase test coverage
def test_logger_identity() -> None:
    """Test required abstract properties of logger."""
    logger = CheckpointRecordLogger("best")
    assert (logger.name, logger.version) == ("checkpoint_record", 0)
