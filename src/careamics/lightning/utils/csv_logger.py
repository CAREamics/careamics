"""PyTorch lightning utilities."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class Series:
    """An easy to plot series extracting from training log."""

    epoch: list[int]
    """X-axis values representing training epochs."""

    value: list[int | float]
    """Y-axis values representing the quantity recorded over the epoch."""


@dataclass
class TrainingReport:
    """Series extracted from training logs."""

    train_loss: Series
    """Training loss."""

    val_loss: Series
    """Validation loss."""

    learning_rate: Series
    """Learning rate."""

    metrics: dict[str, Series]
    """Metrics, can be empty."""


def _extract_series(rows: list[dict[str, str]], column: str) -> Series:
    """Extract an epoch-aligned plottable series from sparse CSV logger rows.

    Parameters
    ----------
    rows : list[dict[str, str]]
        List of rows as extracted from `csv.DictReader`.
    column : str
        Name of the column to extract as a series.

    Returns
    -------
    Series
        Series `column` extracted from `rows`.
    """
    values_by_epoch: dict[int, float] = {}

    for row in rows:
        raw_epoch = row.get("epoch", "")
        raw_value = row.get(column, "")

        if raw_epoch == "" or raw_value == "":
            continue

        values_by_epoch[int(float(raw_epoch))] = float(raw_value)

    epochs = sorted(values_by_epoch)
    return Series(
        epoch=epochs,
        value=[values_by_epoch[epoch] for epoch in epochs],
    )


def read_csv_logger(
    log_folder: Union[str, Path], experiment_name: str, version: int | None = None
) -> TrainingReport:
    """Return plottable training curves from Lightning CSV logs.

    Parameters
    ----------
    log_folder : Path or str
        Path to the folder containing the csv logs.
    experiment_name : str
        Name of the experiment.
    version : int or None, default = None
        Version number to load, if `None` then the latest version is loaded.

    Returns
    -------
    TrainingReport
        Dataclass containing train and validation loss, learning rate, and any
        discovered validation metrics.

    Raises
    ------
    ValueError
        If `version` is specified but not found.
    """
    path = Path(log_folder) / experiment_name

    # find the most recent of version_* folders
    versions = [int(v.name.split("_")[-1]) for v in path.iterdir() if v.is_dir()]

    if version is None:
        version = max(versions)
    else:
        if version not in versions:
            raise ValueError(
                f"Version {version} not found in {path}. Existing versions are "
                f"{versions}."
            )

    path_log = path / f"version_{version}" / "metrics.csv"

    with open(path_log, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = [name for name in (reader.fieldnames or []) if name is not None]

    reserved_columns = {
        "epoch",
        "step",
        "train_loss_step",
        "train_loss_epoch",
        "val_loss",
        "learning_rate",
    }
    metrics = {
        column: _extract_series(rows, column)
        for column in fieldnames
        if column not in reserved_columns
        and any(row.get(column, "") != "" for row in rows)
    }

    return TrainingReport(
        train_loss=_extract_series(rows, "train_loss_epoch"),
        val_loss=_extract_series(rows, "val_loss"),
        learning_rate=_extract_series(rows, "learning_rate"),
        metrics=metrics,
    )
