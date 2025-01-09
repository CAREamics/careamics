"""
Module for CLI functionality and entrypoint.

Contains the CLI entrypoint, the `run` function; and first level subcommands `train`
and `predict`. The `conf` subcommand is added through the `app.add_typer` function, and
its implementation is contained in the conf.py file.
"""

from pathlib import Path
from typing import Annotated, Optional

import click
import typer

from ..careamist import CAREamist
from . import conf
from .utils import handle_2D_3D_callback

app = typer.Typer(
    help="Run CAREamics algorithms from the command line, including Noise2Void "
    "and its many variants and cousins",
    pretty_exceptions_show_locals=False,
)
app.add_typer(conf.app, name="conf")


@app.command()
def train(  # numpydoc ignore=PR01
    source: Annotated[
        Path,
        typer.Argument(
            help="Path to a configuration file or a trained model.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    train_source: Annotated[
        Path,
        typer.Option(
            "--train-source",
            "-ts",
            help="Path to the training data.",
            exists=True,
            file_okay=True,
            dir_okay=True,
        ),
    ],
    train_target: Annotated[
        Optional[Path],
        typer.Option(
            "--train-target",
            "-tt",
            help="Path to train target data.",
            exists=True,
            file_okay=True,
            dir_okay=True,
        ),
    ] = None,
    val_source: Annotated[
        Optional[Path],
        typer.Option(
            "--val-source",
            "-vs",
            help="Path to validation data.",
            exists=True,
            file_okay=True,
            dir_okay=True,
        ),
    ] = None,
    val_target: Annotated[
        Optional[Path],
        typer.Option(
            "--val-target",
            "-vt",
            help="Path to validation target data.",
            exists=True,
            file_okay=True,
            dir_okay=True,
        ),
    ] = None,
    use_in_memory: Annotated[
        bool,
        typer.Option(
            "--use-in-memory/--not-in-memory",
            "-m/-M",
            help="Use in memory dataset if possible.",
        ),
    ] = True,
    val_percentage: Annotated[
        float,
        typer.Option(help="Percentage of files to use for validation."),
    ] = 0.1,
    val_minimum_split: Annotated[
        int,
        typer.Option(help="Minimum number of files to use for validation,"),
    ] = 1,
    work_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--work-dir",
            "-wd",
            help=("Path to working directory in which to save checkpoints and logs"),
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
):
    """Train CAREamics models."""
    engine = CAREamist(source=source, work_dir=work_dir)
    engine.train(
        train_source=train_source,
        val_source=val_source,
        train_target=train_target,
        val_target=val_target,
        use_in_memory=use_in_memory,
        val_percentage=val_percentage,
        val_minimum_split=val_minimum_split,
    )


@app.command()
def predict(  # numpydoc ignore=PR01
    model: Annotated[
        Path,
        typer.Argument(
            help="Path to a configuration file or a trained model.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    source: Annotated[
        Path,
        typer.Argument(
            help="Path to the training data. Can be a directory or single file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
        ),
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 1,
    tile_size: Annotated[
        Optional[click.Tuple],
        typer.Option(
            help=(
                "Size of the tiles to use for prediction, (if the data "
                "is not 3D pass the last value as -1 e.g. --tile_size 64 64 -1)."
            ),
            click_type=click.Tuple([int, int, int]),
            callback=handle_2D_3D_callback,
        ),
    ] = None,
    tile_overlap: Annotated[
        click.Tuple,
        typer.Option(
            help=(
                "Overlap between tiles, (if the data is not 3D pass the last value as "
                "-1 e.g. --tile_overlap 64 64 -1)."
            ),
            click_type=click.Tuple([int, int, int]),
            callback=handle_2D_3D_callback,
        ),
    ] = (48, 48, -1),
    axes: Annotated[
        Optional[str],
        typer.Option(
            help="Axes of the input data. If unused the data is assumed to have the "
            "same axes as the original training data."
        ),
    ] = None,
    data_type: Annotated[
        click.Choice,
        typer.Option(click_type=click.Choice(["tiff"]), help="Type of the input data."),
    ] = "tiff",
    tta_transforms: Annotated[
        bool,
        typer.Option(
            "--tta-transforms/--no-tta-transforms",
            "-t/-T",
            help="Whether to apply test-time augmentation.",
        ),
    ] = False,
    write_type: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["tiff"]), help="Type of the output data."
        ),
    ] = "tiff",
    # TODO: could make dataloader_params as json, necessary?
    work_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--work-dir",
            "-wd",
            help=("Path to working directory."),
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    prediction_dir: Annotated[
        Path,
        typer.Option(
            "--prediction-dir",
            "-pd",
            help=(
                "Directory to save predictions to. If not an abosulte path it will be "
                "relative to the set working directory."
            ),
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("predictions"),
):
    """Create and save predictions from CAREamics models."""
    engine = CAREamist(source=model, work_dir=work_dir)
    engine.predict_to_disk(
        source=source,
        batch_size=batch_size,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        axes=axes,
        data_type=data_type,
        tta_transforms=tta_transforms,
        write_type=write_type,
        prediction_dir=prediction_dir,
    )


def run():
    """CLI Entry point."""
    app()
