"""Lightning logger for recording checkpoints produced during training."""

import json
import shutil
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

from careamics.utils.checkpoint_record_io import (
    CHECKPOINT_RECORD_FILENAME,
    CheckpointRecord,
    TopKRecord,
    read_checkpoint_record,
)

CHECKPOINT_RECORD_BACKUP_FILENAME = f"{CHECKPOINT_RECORD_FILENAME}.bak"


class CheckpointRecordLogger(Logger):
    """Save a JSON record after checkpointing and refresh it after successful training.

    Parameters
    ----------
    selected_checkpoint : {"best", "last"}
        Checkpoint type selected by default for prediction.
    """

    def __init__(self, selected_checkpoint: Literal["best", "last"]) -> None:
        """Initialize the checkpoint record logger.

        Parameters
        ----------
        selected_checkpoint : {"best", "last"}
            Checkpoint type selected by default for prediction.
        """
        super().__init__()
        self.selected_checkpoint: Literal["best", "last"] = selected_checkpoint
        self._checkpoint_callback: ModelCheckpoint | None = None

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Save the checkpoint record after a checkpoint is written.

        Parameters
        ----------
        checkpoint_callback : lightning.pytorch.callbacks.ModelCheckpoint
            Callback that saved the model checkpoint.
        """
        save_checkpoint_record(checkpoint_callback, self.selected_checkpoint)
        self._checkpoint_callback = checkpoint_callback

    # important to implement
    # (checkpoints may have been deleted since the last `after_save_checkpoints` call)
    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Refresh the checkpoint record after training.

        Parameters
        ----------
        status : str
            Training completion status supplied by Lightning.
        """
        checkpoint_callback = self._checkpoint_callback
        self._checkpoint_callback = None
        if checkpoint_callback is not None:
            save_checkpoint_record(checkpoint_callback, self.selected_checkpoint)

    # required abstract property
    @property
    def name(self) -> str:
        """Return the logger name.

        Returns
        -------
        str
            Logger name.
        """
        return "checkpoint_record"

    # required abstract property
    @property
    def version(self) -> int:
        """Return the logger version.

        Returns
        -------
        int
            Logger version.
        """
        return 0

    # required abstract method
    def log_hyperparams(self, params: dict[str, Any] | Namespace) -> None:
        """Ignore hyperparameters.

        Parameters
        ----------
        params : dict[str, Any] or argparse.Namespace
            Hyperparameters supplied by Lightning.
        """

    # required abstract method
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Ignore metrics.

        Parameters
        ----------
        metrics : dict[str, float]
            Metrics supplied by Lightning.
        step : int | None, default=None
            Training step associated with the metrics.
        """


def save_checkpoint_record(
    model_checkpoint: ModelCheckpoint,
    selected_checkpoint: Literal["best", "last"],
) -> None:
    """Save a checkpoint record, backing up an existing record first.

    Parameters
    ----------
    model_checkpoint : lightning.pytorch.callbacks.ModelCheckpoint
        Callback managing the training checkpoints.
    selected_checkpoint : {"best", "last"}
        Checkpoint type selected by default for prediction.
    """
    if model_checkpoint.dirpath is None:
        # Error is only relevant for users of the lightning API
        raise ValueError(
            "Lightning's `ModelCheckpoint` must have `dirpath` configured to log a "
            "checkpoint record."
        )

    checkpoint_dir = Path(model_checkpoint.dirpath)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record_path = checkpoint_dir / CHECKPOINT_RECORD_FILENAME
    if record_path.exists():
        shutil.copy2(record_path, checkpoint_dir / CHECKPOINT_RECORD_BACKUP_FILENAME)

    data = create_checkpoint_record_data(model_checkpoint, selected_checkpoint)
    with record_path.open("w") as file:
        json.dump(data, file, indent=2)


def create_checkpoint_record_data(
    model_checkpoint: ModelCheckpoint,
    selected_checkpoint: Literal["best", "last"],
) -> CheckpointRecord:
    """Create a record of checkpoints produced during training.

    If an existing checkpoint record exists it is read and then updated with any new
    checkpoints saved or deleted.

    Parameters
    ----------
    model_checkpoint : lightning.pytorch.callbacks.ModelCheckpoint
        Callback managing the training checkpoints.
    selected_checkpoint : {"best", "last"}
        Checkpoint type selected by default for prediction.

    Returns
    -------
    dict[str, Any]
        Checkpoint paths, scores, monitoring settings, and default selection.

    Raises
    ------
    ValueError
        If the monitoring mode is unsupported.
    """
    if model_checkpoint.dirpath is None:
        raise ValueError("ModelCheckpoint.dirpath must be set.")

    last = (
        Path(last_path).name
        if (last_path := model_checkpoint.last_model_path) != ""
        else None
    )
    best = (
        Path(best_path).name
        if (best_path := model_checkpoint.best_model_path) != ""
        else None
    )
    mode = model_checkpoint.mode

    best_k: list[TopKRecord] = [
        TopKRecord(name=Path(path).name, score=score.item())
        for path, score in model_checkpoint.best_k_models.items()
    ]
    if mode == "min":
        best_k.sort(key=lambda checkpoint: checkpoint["score"])
    elif mode == "max":
        best_k.sort(key=lambda checkpoint: checkpoint["score"], reverse=True)
    else:
        raise ValueError(f"Unrecognized checkpoint monitoring mode: {mode}")

    checkpoint_paths = get_checkpoints(Path(model_checkpoint.dirpath))
    checkpoints = [path.name for path in checkpoint_paths]

    match (len(checkpoints), selected_checkpoint):
        case 0, _:
            selected: str | None = None
        case _, "last":
            selected = checkpoints[-1]
        case _, "best":
            selected = best
        case _, _:
            # should not be reachable
            raise ValueError("Error selecting checkpoint.")

    return {
        "last": last,
        "best": best,
        "monitor": model_checkpoint.monitor,
        "mode": mode,
        "best_k": best_k,
        "checkpoints": checkpoints,
        "selected": selected,
    }


def get_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Return existing checkpoint paths in the order that they were added.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory containing checkpoint files.

    Returns
    -------
    list of pathlib.Path
        Existing recorded checkpoints followed by new checkpoints.
    """
    recorded_paths: list[Path] = []
    last_path: Path | None = None
    record = read_checkpoint_record(checkpoint_dir)
    if record is not None:
        # extract recorded checkpoints; remove those that do not exist
        # (checkpoints bumped out of the top k may be deleted.)
        recorded_names: Sequence[str] = record.get("checkpoints", [])
        recorded_paths = [
            checkpoint_path
            for name in recorded_names
            if (checkpoint_path := (checkpoint_dir / name)).is_file()
        ]
        last_name = record.get("last", None)
        if last_name is not None:
            last_path = checkpoint_dir / last_name

    # find any new paths not present in the record
    existing_paths = sorted(
        checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_ctime_ns
    )
    new_paths = [path for path in existing_paths if path not in recorded_paths]

    # update the record
    updated_paths = recorded_paths + new_paths

    # ensure last checkpoint is always last
    # (it won't appear as a new checkpoint if it is overwritten with the same name)
    if last_path is not None:
        try:
            last_idx = updated_paths.index(last_path)
        except ValueError as e:
            raise ValueError(
                f"Last checkpoint path '{last_path}' not found in checkpoint directory."
            ) from e
        updated_paths.pop(last_idx)
        updated_paths.append(last_path)

    return updated_paths
