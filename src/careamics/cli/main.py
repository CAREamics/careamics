"""
Module for CLI functionality and entrypoint.

Contains the CLI entrypoint, the `run` function; and first level subcommands `train`
and `predict`. The `conf` subcommand is added through the `app.add_typer` function, and
its implementation is contained in the conf.py file.
"""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from ..careamist import CAREamist
from . import conf

app = typer.Typer(
    help="Run CAREamics algorithms from the command line, including Noise2Void "
    "and its many variants and cousins"
)
app.add_typer(
    conf.app,
    name="conf",
    # callback=conf.conf_options
)


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
            help=("Path to working directory in which to save checkpoints and " "logs"),
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
def predict():  # numpydoc ignore=PR01
    """Create and save predictions from CAREamics models."""
    # TODO: Need a save predict to workdir function
    raise NotImplementedError


def run():
    """CLI Entry point."""
    app()
