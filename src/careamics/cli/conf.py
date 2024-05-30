import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from typing_extensions import Annotated
import yaml

import typer
import click
from typing_extensions import Annotated

from ..config import (
    Configuration,
    create_care_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
)

app = typer.Typer()

def _save_config(config: Configuration, fp: Path) -> None:
    with open(fp, "w") as file:
        yaml.dump(config.model_dump(), file, indent=2)

@dataclass
class ConfigOptions:
    dir: Path
    name: Path
    force: bool
    print: bool


@app.callback()
def config_options(
    ctx: typer.Context,
    dir: Annotated[
        Path, typer.Option("--dir", "-d", exists=True)
    ] = os.getcwd(),
    name: Annotated[str, typer.Option("--name", "-n")] = "config",
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    print: Annotated[bool, typer.Option("--print", "-p")] = False,
):
    conf_path = (dir / name).with_suffix(".yaml")
    if conf_path.exists() and not force:
        raise FileExistsError(
            f"To overwrite '{conf_path}' use flag --force/-f."
        )

    ctx.obj = ConfigOptions(dir, name, force, print)


@app.command()
def care(
    ctx: typer.Context,
    experiment_name: Annotated[str, typer.Option()],
    axes: Annotated[str, typer.Option()],
    patch_size: Annotated[
        click.Tuple, typer.Option(click_type=click.Tuple([int, int]))
    ],
    batch_size: Annotated[int, typer.Option()],
    num_epochs: Annotated[int, typer.Option()],
    data_type: Annotated[
        click.Choice, typer.Option(click_type=click.Choice(["tiff"]))
    ] = "tiff",
    use_augmentations: Annotated[bool, typer.Option()] = True,
    independent_channels: Annotated[bool, typer.Option()] = False,
    loss: Annotated[
        click.Choice, typer.Option(click_type=click.Choice(["mae", "mse"]))
    ] = "mae",
    n_channels_in: Annotated[int, typer.Option()] = 1,
    n_channels_out: Annotated[int, typer.Option()] = -1,
    logger: Annotated[
        click.Choice,
        typer.Option(click_type=click.Choice(["wandb", "tensorboard", "none"])),
    ] = "none",
    # TODO: How to address model kwargs
):
    config = create_care_configuration(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_augmentations=use_augmentations,
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger
    )
    conf_path = (ctx.obj.dir / ctx.obj.name).with_suffix(".yaml")
    _save_config(config, conf_path)
    if ctx.obj.print:
        print(yaml.dump(config.model_dump(), indent=2))
