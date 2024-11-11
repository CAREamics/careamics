"""PyTorch lightning utilities."""

from pathlib import Path


def read_csv_logger(experiment_name: str, log_folder: Path) -> dict:
    """Return the loss curves from the csv logs.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    log_folder : Path
        Path to the folder containing the csv logs.

    Returns
    -------
    dict
        Dictionary containing the loss curves, with keys "epoch", "step", "train" and
        "val".
    """
    path = log_folder / experiment_name

    # find the most recent of version_* folders
    versions = [int(v.name.split("_")[-1]) for v in path.iterdir() if v.is_dir()]
    version = max(versions)

    path_log = path / f"version_{version}" / "metrics.csv"

    metrics: dict[str, list] = {
        "epoch": [],
        "train": [],
        "val": [],
    }

    with open(path_log) as f:
        lines = f.readlines()

        for single_line in lines[1:]:
            epoch, _, train_loss, _, val_loss = single_line.strip().split(",")

            metrics["epoch"].append(int(epoch))

            if train_loss != "":
                metrics["train"].append(float(train_loss))

            if val_loss != "":
                metrics["val"].append(float(val_loss))

    # remove dupicates from epoch
    metrics["epoch"] = list(set(metrics["epoch"]))

    assert len(metrics["epoch"]) == len(metrics["train"]) == len(metrics["val"])

    return metrics
