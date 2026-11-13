"""Lightning logger for recording checkpoints produced during training."""

import json
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

CHECKPOINT_RECORD_FILENAME = "checkpoint_record.json"
CHECKPOINT_RECORD_BACKUP_FILENAME = "checkpoint_record.json.bak"


def get_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Return checkpoint paths ordered by creation time.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory containing checkpoint files.

    Returns
    -------
    list of pathlib.Path
        Checkpoint paths ordered from oldest to newest.
    """
    return sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_ctime)


def _score_value(score: Any) -> Any:
    """Return a JSON-serializable checkpoint score.

    Parameters
    ----------
    score : Any
        Checkpoint score, which may be a scalar tensor.

    Returns
    -------
    Any
        JSON-serializable score.
    """
    return score.item() if hasattr(score, "item") else score


def get_checkpoint_record_data(
    model_checkpoint: ModelCheckpoint,
    selected_checkpoint: Literal["best", "last"],
) -> dict[str, Any]:
    """Get a record of checkpoints produced during training.

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

    last = model_checkpoint.last_model_path
    best = model_checkpoint.best_model_path
    data: dict[str, Any] = {
        "last": Path(last).name if last else None,
        "best": Path(best).name if best else None,
        "monitor": model_checkpoint.monitor,
        "mode": model_checkpoint.mode,
    }

    best_k = [
        {"name": Path(path).name, "score": _score_value(score)}
        for path, score in model_checkpoint.best_k_models.items()
    ]
    if data["mode"] == "min":
        best_k.sort(key=lambda checkpoint: checkpoint["score"])
    elif data["mode"] == "max":
        best_k.sort(key=lambda checkpoint: checkpoint["score"], reverse=True)
    else:
        raise ValueError(f"Unrecognized checkpoint monitoring mode: {data['mode']}")
    data["best_k"] = best_k

    checkpoint_paths = get_checkpoints(Path(model_checkpoint.dirpath))
    data["checkpoints"] = [path.name for path in checkpoint_paths]

    match (len(data["checkpoints"]), selected_checkpoint):
        case 0, _:
            data["selected"] = None
        case _, "last":
            data["selected"] = data["checkpoints"][-1]
        case _, "best":
            data["selected"] = data[selected_checkpoint]
        case _, _:
            raise ValueError("Error selecting checkpoint.")

    return data


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
        raise ValueError("ModelCheckpoint.dirpath must be set.")

    checkpoint_dir = Path(model_checkpoint.dirpath)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record_path = checkpoint_dir / CHECKPOINT_RECORD_FILENAME
    if record_path.exists():
        shutil.copy2(record_path, checkpoint_dir / CHECKPOINT_RECORD_BACKUP_FILENAME)

    data = get_checkpoint_record_data(model_checkpoint, selected_checkpoint)
    with record_path.open("w") as file:
        json.dump(data, file, indent=2)


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

    @property
    def name(self) -> str:
        """Return the logger name.

        Returns
        -------
        str
            Logger name.
        """
        return "checkpoint_record"

    @property
    def version(self) -> int:
        """Return the logger version.

        Returns
        -------
        int
            Logger version.
        """
        return 0

    def log_hyperparams(self, params: dict[str, Any] | Namespace) -> None:
        """Ignore hyperparameters.

        Parameters
        ----------
        params : dict[str, Any] or argparse.Namespace
            Hyperparameters supplied by Lightning.
        """

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Ignore metrics.

        Parameters
        ----------
        metrics : dict[str, float]
            Metrics supplied by Lightning.
        step : int | None, default=None
            Training step associated with the metrics.
        """

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
