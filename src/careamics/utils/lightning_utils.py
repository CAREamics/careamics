"""PyTorch lightning utilities."""

from pathlib import Path
from typing import Union


def read_csv_logger(experiment_name: str, log_folder: Union[str, Path]) -> dict:
    """Return the loss curves from the csv logs.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    log_folder : Path or str
        Path to the folder containing the csv logs.

    Returns
    -------
    dict
        Dictionary containing the loss curves, with keys "train_epoch", "val_epoch",
        "train_loss" and "val_loss".
    """
    path = Path(log_folder) / experiment_name

    # find the most recent of version_* folders
    versions = [int(v.name.split("_")[-1]) for v in path.iterdir() if v.is_dir()]
    version = max(versions)

    path_log = path / f"version_{version}" / "metrics.csv"

    epochs = []
    train_losses_tmp = []
    val_losses_tmp = []
    with open(path_log) as f:
        lines = f.readlines()

        for single_line in lines[1:]:
            epoch, _, train_loss, _, val_loss = single_line.strip().split(",")

            epochs.append(epoch)
            train_losses_tmp.append(train_loss)
            val_losses_tmp.append(val_loss)

    # train and val are not logged on the same row and can have different lengths
    train_epoch = [
        int(epochs[i]) for i in range(len(epochs)) if train_losses_tmp[i] != ""
    ]
    val_epoch = [int(epochs[i]) for i in range(len(epochs)) if val_losses_tmp[i] != ""]
    train_losses = [float(loss) for loss in train_losses_tmp if loss != ""]
    val_losses = [float(loss) for loss in val_losses_tmp if loss != ""]

    return {
        "train_epoch": train_epoch,
        "val_epoch": val_epoch,
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
