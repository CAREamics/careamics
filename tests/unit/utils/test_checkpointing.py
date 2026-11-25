"""Tests for checkpointing utilities."""

import json

import pytest

from careamics.utils.checkpointing import (
    CHECKPOINT_RECORD_FILENAME,
    get_default_checkpoint_type,
    read_checkpoint_record,
)


@pytest.mark.parametrize(
    "checkpoint_params, expected",
    [
        ({}, "best"),
        ({"save_top_k": 3}, "best"),
        ({"save_top_k": 0}, "last"),
        ({"save_top_k": -1}, "last"),
    ],
)
def test_get_default_checkpoint_type(checkpoint_params, expected):
    """Test default checkpoint selection from ModelCheckpoint parameters."""
    assert get_default_checkpoint_type(checkpoint_params) == expected


def test_read_checkpoint_record(tmp_path):
    """Test reading a checkpoint record when it is present."""
    expected = {
        "last": "last.ckpt",
        "best": "best.ckpt",
        "monitor": "val_loss",
        "mode": "min",
        "best_k": [{"name": "best.ckpt", "score": 0.1}],
        "checkpoints": ["best.ckpt", "last.ckpt"],
        "selected": "best.ckpt",
    }
    record_path = tmp_path / CHECKPOINT_RECORD_FILENAME
    record_path.write_text(json.dumps(expected))

    assert read_checkpoint_record(tmp_path) == expected


def test_read_missing_checkpoint_record(tmp_path):
    """Test that a missing checkpoint record returns None."""
    assert read_checkpoint_record(tmp_path) is None
