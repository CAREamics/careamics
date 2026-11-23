"""Unit tests for CAREamist."""

import json

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


def test_select_default_checkpoint(tmp_path):
    """Test explicit, loaded, and recorded checkpoint selection precedence."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=[32, 32],
        batch_size=8,
    )
    careamist = CAREamist(config=config, work_dir=tmp_path)

    assert careamist._select_default_ckpt(None) is None

    checkpoint_root = careamist.get_checkpoint_root()
    checkpoint_root.mkdir(parents=True)
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

    assert careamist._select_default_ckpt(None) == selected_checkpoint

    loaded_checkpoint = tmp_path / "loaded.ckpt"
    careamist.checkpoint_path = loaded_checkpoint
    assert careamist._select_default_ckpt(None) == loaded_checkpoint
    assert careamist._select_default_ckpt("last") == "last"
