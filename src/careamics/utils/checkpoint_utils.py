"""Utilities for inspecting checkpoint files."""

from pathlib import Path
from typing import TypedDict

from .lightning_utils import _epoch_to_val_loss, read_csv_logger


class CheckpointInfo(TypedDict):
    """Structured information about a single checkpoint file."""

    name: str
    epoch: int | None
    val_loss: float | None
    path: Path


def get_checkpoint_info(
    experiment_name: str,
    checkpoint_dir: Path,
    log_dir: Path,
) -> list[CheckpointInfo]:
    """Scan a checkpoint directory and return structured checkpoint info.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment, used to parse filenames and read CSV logs.
    checkpoint_dir : Path
        Path to the directory containing checkpoint files.
    log_dir : Path
        Path to the CSV logs directory.

    Returns
    -------
    list of CheckpointInfo
        Each entry contains "name" (str), "epoch" (int or None),
        "val_loss" (float or None), and "path" (Path). Epoch-based
        checkpoints are sorted by epoch number, with the last checkpoint
        appended at the end.
    """
    try:
        losses = read_csv_logger(experiment_name, log_dir)
        epoch_to_loss = _epoch_to_val_loss(losses)
    except Exception:
        epoch_to_loss = {}

    checkpoints: list[CheckpointInfo] = []
    last_checkpoint: CheckpointInfo | None = None

    for ckpt_path in sorted(checkpoint_dir.glob("*.ckpt")):
        name = ckpt_path.stem

        if name.endswith("_last"):
            last_checkpoint = CheckpointInfo(
                name=ckpt_path.name,
                epoch=None,
                val_loss=None,
                path=ckpt_path,
            )
            continue

        suffix = name[len(experiment_name) + 1 :]
        try:
            epoch = int(suffix.split("_")[0])
        except ValueError:
            continue

        checkpoints.append(
            CheckpointInfo(
                name=ckpt_path.name,
                epoch=epoch,
                val_loss=epoch_to_loss.get(epoch),
                path=ckpt_path,
            )
        )

    checkpoints.sort(key=lambda x: x["epoch"] if x["epoch"] is not None else -1)

    if last_checkpoint is not None:
        checkpoints.append(last_checkpoint)

    return checkpoints
