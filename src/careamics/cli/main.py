import os
from typing import Optional

import typer
from typing_extensions import Annotated

from careamics import CAREamist

app = typer.Typer()


@app.command()
def train(
    source: Annotated[
        str,
        typer.Argument(help="Path to a configuration file or a trained model."),
    ],
    train_source: Annotated[
        str,
        typer.Option(
            "--train_source", "-ts", help="Path to the training data."
        ),
    ],
    train_target: Annotated[
        Optional[str],
        typer.Option(
            "--train_target", "-tt", help="Path to train target data."
        ),
    ] = None,
    val_source: Annotated[
        Optional[str],
        typer.Option("--val_source", "-vs", help="Path to validation data."),
    ] = None,
    val_target: Annotated[
        Optional[str],
        typer.Option(
            "--val_target", "-vt", help="Path to validation target data."
        ),
    ] = None,
    # TODO: better to do opposite i.e. on_disk=False ?
    # use_in_memory: Annotated[
    #     bool,
    #     typer.Option(
    #         "--use_in_memory", "-m", help="Use in memory dataset if possible."
    #     ),
    # ] = True,
    val_percentage: Annotated[
        float,
        typer.Option(help="Percentage of files to use for validation."),
    ] = 0.1,
    val_minimum_split: Annotated[
        int,
        typer.Option(help="Minimum number of files to use for validation,"),
    ] = 1,
    work_dir: Annotated[
        Optional[str],
        typer.Option(
            "--work_dir",
            "-wd",
            help=(
                "Path to working directory in which to save checkpoints and "
                "logs"
            ),
        ),
    ] = None,
):
    engine = CAREamist(source=source, work_dir=work_dir)
    engine.train(
        train_source=train_source,
        val_source=val_source,
        train_target=train_target,
        val_target=val_target,
        # use_in_memory=use_in_memory,
        val_percentage=val_percentage,
        val_minimum_split=val_minimum_split,
    )


@app.command()
def predict():
    raise NotImplementedError


@app.command()
def conf():
    raise NotImplementedError


def run():
    app()
