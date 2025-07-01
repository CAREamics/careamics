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

    with open(path_log) as f:
        lines = f.readlines()

        header = lines[0].strip().split(",")
        metrics: dict[str, list] = {value: [] for value in header}
        print(metrics)

        for single_line in lines[1:]:
            values = single_line.strip().split(",")

            for k, v in zip(header, values, strict=False):
                metrics[k].append(v)

    # train and val are not logged on the same row and can have different lengths
    train_epoch = [
        int(metrics["epoch"][i])
        for i in range(len(metrics["epoch"]))
        if metrics["train_loss_epoch"][i] != ""
    ]
    val_epoch = [
        int(metrics["epoch"][i])
        for i in range(len(metrics["epoch"]))
        if metrics["val_loss"][i] != ""
    ]
    train_losses = [
        float(metrics["train_loss_epoch"][i])
        for i in range(len(metrics["train_loss_epoch"]))
        if metrics["train_loss_epoch"][i] != ""
    ]
    val_losses = [
        float(metrics["val_loss"][i])
        for i in range(len(metrics["val_loss"]))
        if metrics["val_loss"][i] != ""
    ]

    return {
        "train_epoch": train_epoch,
        "val_epoch": val_epoch,
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
