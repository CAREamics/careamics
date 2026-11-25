"""Unit tests for CAREamist."""

import json
from pathlib import Path
from typing import Literal

import pytest

from careamics import CAREamist
from careamics.config.factories import create_advanced_n2v_config
from careamics.lightning.checkpoint_record_logger import CheckpointRecordLogger
from careamics.utils.checkpointing import CHECKPOINT_RECORD_FILENAME


def test_checkpoint_record_logger_is_configured(tmp_path):
    """Test that CAREamist configures its checkpoint record logger."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=[32, 32],
        batch_size=8,
    )

    careamist = CAREamist(config=config, work_dir=tmp_path)

    record_loggers = [
        logger
        for logger in careamist.trainer.loggers
        if isinstance(logger, CheckpointRecordLogger)
    ]
    assert len(record_loggers) == 1
    assert record_loggers[0].selected_checkpoint == "last"


@pytest.mark.parametrize("init_from", ["config", "pretrained"])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("choose_checkpoint", [True, False])
def test_select_default_checkpoint(
    tmp_path: Path,
    pre_trained: Path,
    init_from: Literal["config", "pretrained"],
    train: bool,
    choose_checkpoint: bool,
):
    """
    Test default checkpoint loading for prediction.

    Covers cases for initializing from a config loading a pre-trained checkpoint,
    with or without having a training run that produced saved checkpoints, and whether
    a specific checkpoint is chosen.
    """
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=[32, 32],
        batch_size=8,
    )
    if init_from == "config":
        careamist = CAREamist(config=config, work_dir=tmp_path)
    else:
        careamist = CAREamist(checkpoint_path=pre_trained, work_dir=tmp_path)

    selected_checkpoint: Path | None = None
    if train:
        selected_checkpoint = _simulate_saved_checkpoints(
            careamist.get_checkpoint_root()
        )

    if choose_checkpoint:
        chosen_checkpoint = tmp_path / "selected.ckpt"
        chosen_checkpoint.touch()
    else:
        chosen_checkpoint = None

    result = careamist._select_default_ckpt(chosen_checkpoint)
    match (init_from, train, chosen_checkpoint):
        case (_, _, chosen_checkpoint) if chosen_checkpoint is not None:
            assert result == chosen_checkpoint
        case ("config", False, None):
            assert result is None
        case ("pretrained", False, None):
            assert result == pre_trained
        case (_, True, None):
            assert result == selected_checkpoint
        case _:
            raise ValueError("Uncovered case in test.")


def _simulate_saved_checkpoints(checkpoint_root: Path) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = checkpoint_root / "selected.ckpt"
    selected_checkpoint.touch()
    record = {
        "last": selected_checkpoint.name,
        "best": None,
        "monitor": None,
        "mode": "min",
        "best_k": [],
        "checkpoints": [selected_checkpoint.name],
        "selected": selected_checkpoint.name,
    }
    (checkpoint_root / CHECKPOINT_RECORD_FILENAME).write_text(json.dumps(record))
    return selected_checkpoint
