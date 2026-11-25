"""Module for utilities related to reading the checkpoint record."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

from torch.types import Number

CHECKPOINT_RECORD_FILENAME = "checkpoint_record.json"


class CheckpointRecord(TypedDict):
    """
    A structured record of the checkpoints saved during training.

    Attributes
    ----------
    last : str | None
        The file name of the last checkpoint. `None` if last checkpoint saving is not
        enabled.
    best : str | None
        The file name of the best checkpoint. `None` if best checkpoint saving is not
        enabled.
    monitor : str | None
        The metric that was monitored to determine the best checkpoint. Can be None if
        best checkpoint saving is not enabled.
    best_k : Sequence[TopKRecord]
        A sequence of dictionaries recording the top k best checkpoints. Each dictionary
        contains a "name" and a "score" key. The "name" records the file name and the
        "score" records the value of the monitored metric. The checkpoints are ordered
        from best to worst. If saving top k checkpoints is not enabled the sequence
        will be empty.
    checkpoints : Sequence[str]
        A sequence containing the file name of all saved checkpoints ordered by their
        creation time.
    selected : str
        The file name of the checkpoint that should be selected for inference. Can be
        `None` if no checkpoints have been saved.
    """

    last: str | None
    best: str | None
    monitor: str | None
    mode: Literal["min", "max"]
    best_k: Sequence["TopKRecord"]
    checkpoints: Sequence[str]
    selected: str | None


class TopKRecord(TypedDict):
    """
    A structured record to represent one of the top k saved checkpoints.

    Attributes
    ----------
    name : str
        The file stem of the checkpoint.
    score : Number
        The score the checkpoint achieved on the monitored metric.
    """

    name: str
    score: Number


def read_checkpoint_record(checkpoint_dir: Path) -> CheckpointRecord | None:
    """Read a checkpoint record.

    Parameters
    ----------
    checkpoint_dir : Path
        The directory where the checkpoints and the record file are saved.

    Returns
    -------
    CheckpointRecord | None
        A structured record of the checkpoints saved during training. Returns None if
        there is no checkpoint record file.
    """
    record_path = checkpoint_dir / CHECKPOINT_RECORD_FILENAME
    if record_path.exists():
        with record_path.open() as file:
            record: CheckpointRecord = json.load(file)
        return record
    else:
        return None


def get_default_checkpoint_type(
    checkpoint_params: Mapping[str, Any],
) -> Literal["best", "last"]:
    """Return the default checkpoint type for ModelCheckpoint parameters.

    Parameters
    ----------
    checkpoint_params : Mapping[str, Any]
        Parameters used to instantiate a Lightning ModelCheckpoint.

    Returns
    -------
    {"best", "last"}
        `"best"` when `save_top_k` is positive, otherwise `"last"`.
    """
    save_top_k_lightning_default = 1
    saves_best = checkpoint_params.get("save_top_k", save_top_k_lightning_default) > 0
    return "best" if saves_best else "last"
