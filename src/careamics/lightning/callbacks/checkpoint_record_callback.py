"""Lightning callback for recording checkpoints produced during training."""

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

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


def _checkpoint_names(model_checkpoint: ModelCheckpoint) -> list[str]:
    """Return all known checkpoint names in creation order.

    Parameters
    ----------
    model_checkpoint : lightning.pytorch.callbacks.ModelCheckpoint
        Callback managing the training checkpoints.

    Returns
    -------
    list of str
        Names of all known checkpoints.
    """
    if model_checkpoint.dirpath is None:
        raise ValueError("ModelCheckpoint.dirpath must be set.")

    checkpoints = get_checkpoints(Path(model_checkpoint.dirpath))

    if model_checkpoint.monitor is not None:
        known_paths = list(model_checkpoint.best_k_models)
    elif model_checkpoint.save_top_k == 1:
        known_paths = [model_checkpoint.best_model_path]
    else:
        known_paths = [model_checkpoint.best_model_path]
        known_paths.extend(str(path) for path in checkpoints)

    known_paths.append(model_checkpoint.last_model_path)
    paths = list(dict.fromkeys(Path(path) for path in known_paths if path))
    existing_paths = sorted(
        (path for path in paths if path.exists()),
        key=lambda path: path.stat().st_ctime,
    )
    pending_paths = [path for path in paths if not path.exists()]
    return [path.name for path in existing_paths + pending_paths]


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
        If no checkpoints are known or the monitoring mode is unsupported.
    """
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

    data["checkpoints"] = _checkpoint_names(model_checkpoint)
    if not data["checkpoints"]:
        raise ValueError(
            "No checkpoints found. Ensure that model checkpointing has been "
            "configured correctly."
        )

    data["selected"] = data[selected_checkpoint]
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
    record_path = checkpoint_dir / CHECKPOINT_RECORD_FILENAME
    if record_path.exists():
        shutil.copy2(record_path, checkpoint_dir / CHECKPOINT_RECORD_BACKUP_FILENAME)

    with record_path.open("w") as file:
        json.dump(
            get_checkpoint_record_data(model_checkpoint, selected_checkpoint),
            file,
            indent=2,
        )


class CheckpointRecordCallback(Callback):
    """Save a JSON record whenever Lightning saves a checkpoint.

    Parameters
    ----------
    selected_checkpoint : {"best", "last"}
        Checkpoint type selected by default for prediction.
    """

    def __init__(self, selected_checkpoint: Literal["best", "last"]) -> None:
        """Initialize the checkpoint record callback.

        Parameters
        ----------
        selected_checkpoint : {"best", "last"}
            Checkpoint type selected by default for prediction.
        """
        super().__init__()
        self.selected_checkpoint = selected_checkpoint

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        """Save the checkpoint record alongside the model checkpoints.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            Trainer saving the checkpoint.
        pl_module : lightning.pytorch.LightningModule
            Module being trained.
        checkpoint : dict[str, Any]
            Checkpoint data being saved.
        """
        if not trainer.is_global_zero:
            return

        model_checkpoint = trainer.checkpoint_callback
        if not isinstance(model_checkpoint, ModelCheckpoint):
            raise TypeError("Trainer checkpoint callback must be a ModelCheckpoint.")

        save_checkpoint_record(model_checkpoint, self.selected_checkpoint)
