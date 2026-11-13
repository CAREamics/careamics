"""Lightning callback for recording checkpoints produced during training."""

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

CHECKPOINT_RECORD_FILENAME = "checkpoint_record.json"
CHECKPOINT_RECORD_BACKUP_FILENAME = "checkpoint_record.json.bak"
CHECKPOINTS_RECORD_FILENAME = "checkpoints_record.json"


def get_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Return checkpoint paths from a record or ordered by creation time.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory containing checkpoint files.

    Returns
    -------
    list of pathlib.Path
        Recorded checkpoint paths, or checkpoint paths ordered from oldest to newest.
    """
    for filename in (CHECKPOINT_RECORD_FILENAME, CHECKPOINTS_RECORD_FILENAME):
        record_path = checkpoint_dir / filename
        if record_path.exists():
            with record_path.open() as file:
                record = json.load(file)
            return [checkpoint_dir / name for name in record["checkpoints"]]

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

    # select checkpoint
    match (len(data["checkpoints"]), selected_checkpoint):
        case 0, _:
            data["selected"] = None
        case _, "last":
            # selecting last saved in-case saving each epoch has not been configured.
            data["selected"] = data["checkpoints"][-1]
        case _, "best":
            data["selected"] = data[selected_checkpoint]
        case _, _:
            # should not be reachable
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
        self.selected_checkpoint: Literal["best", "last"] = selected_checkpoint

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
