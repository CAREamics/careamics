"""Tests for Lightning CSV logger utilities."""

from pathlib import Path

from careamics.lightning.utils.csv_logger import Series, TrainingReport, read_csv_logger


def test_read_csv_logger_extracts_plottable_vectors(tmp_path: Path) -> None:
    """Read sparse Lightning CSV logs into epoch-aligned plottable vectors."""
    metrics_dir = tmp_path / "experiment" / "version_0"
    metrics_dir.mkdir(parents=True)

    metrics_csv = metrics_dir / "metrics.csv"
    metrics_csv.write_text(
        "\n".join(
            [
                (
                    "epoch,learning_rate,step,train_loss_epoch,train_loss_step,"
                    "val_dice_class_0,val_dice_class_1,val_loss"
                ),
                "0,,49,,0.48,,,,",
                "0,,99,,0.27,,,,",
                "0,,149,,0.31,,,,",
                "0,,199,,0.15,,,,",
                "0,,236,,,0.98,0.82,0.15",
                "0,0.001,236,0.33,,,,",
                "1,,249,,0.15,,,,",
                "1,,299,,0.15,,,,",
                "1,,473,,,0.99,0.83,0.14",
                "1,0.001,473,0.16,,,,",
            ]
        )
    )

    history = read_csv_logger(tmp_path, "experiment")

    assert history == TrainingReport(
        train_loss=Series(epoch=[0, 1], value=[0.33, 0.16]),
        val_loss=Series(epoch=[0, 1], value=[0.15, 0.14]),
        learning_rate=Series(epoch=[0, 1], value=[0.001, 0.001]),
        metrics={
            "val_dice_class_0": Series(epoch=[0, 1], value=[0.98, 0.99]),
            "val_dice_class_1": Series(epoch=[0, 1], value=[0.82, 0.83]),
        },
    )
