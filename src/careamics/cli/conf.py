import os
from pathlib import Path
from dataclasses import dataclass
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


def _config_builder_exit(ctx: typer.Context, config: Configuration) -> None:
    conf_path = (ctx.obj.dir / ctx.obj.name).with_suffix(".yaml")
    _save_config(config, conf_path)
    if ctx.obj.print:
        print(yaml.dump(config.model_dump(), indent=2))


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
    experiment_name: Annotated[
        str, typer.Option(help="Name of the experiment.")
    ],
    axes: Annotated[str, typer.Option(help="Axes of the data (e.g. SYX).")],
    patch_size: Annotated[
        click.Tuple,
        typer.Option(
            click_type=click.Tuple([int, int]),
            help="Size of the patches along the spatial dimensions (e.g. --patch-size 64 64).",
        ),
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size.")],
    num_epochs: Annotated[int, typer.Option(help="Number of epochs.")],
    data_type: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["tiff"]), help="Type of the data."
        ),
    ] = "tiff",
    use_augmentations: Annotated[
        bool, typer.Option(help="Whether to use augmentations.")
    ] = True,
    independent_channels: Annotated[
        bool, typer.Option(help="Whether to train all channels independently.")
    ] = False,
    loss: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["mae", "mse"]),
            help="Loss function to use.",
        ),
    ] = "mae",
    n_channels_in: Annotated[
        int, typer.Option(help="Number of channels in")
    ] = 1,
    n_channels_out: Annotated[
        int, typer.Option(help="Number of channels out")
    ] = -1,
    logger: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["wandb", "tensorboard", "none"]),
            help="Logger to use.",
        ),
    ] = "none",
    # TODO: How to address model kwargs
) -> None:
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
        logger=logger,
    )
    _config_builder_exit(ctx, config)


@app.command()
def n2n(
    ctx: typer.Context,
    experiment_name: Annotated[
        str, typer.Option(help="Name of the experiment.")
    ],
    axes: Annotated[str, typer.Option(help="Axes of the data (e.g. SYX).")],
    patch_size: Annotated[
        click.Tuple,
        typer.Option(
            click_type=click.Tuple([int, int]),
            help="Size of the patches along the spatial dimensions (e.g. --patch-size 64 64).",
        ),
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size.")],
    num_epochs: Annotated[int, typer.Option(help="Number of epochs.")],
    data_type: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["tiff"]), help="Type of the data."
        ),
    ] = "tiff",
    use_augmentations: Annotated[
        bool, typer.Option(help="Whether to use augmentations.")
    ] = True,
    independent_channels: Annotated[
        bool, typer.Option(help="Whether to train all channels independently.")
    ] = False,
    loss: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["mae", "mse"]),
            help="Loss function to use.",
        ),
    ] = "mae",
    n_channels: Annotated[
        int, typer.Option(help="Number of channels (in and out)")
    ] = 1,
    logger: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["wandb", "tensorboard", "none"]),
            help="Logger to use.",
        ),
    ] = "none",
    # TODO: How to address model kwargs
) -> None:
    config = create_n2n_configuration(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_augmentations=use_augmentations,
        independent_channels=independent_channels,
        loss=loss,
        n_channels=n_channels,
        logger=logger,
    )
    _config_builder_exit(ctx, config)


@app.command()
def n2v(
    ctx: typer.Context,
    experiment_name: Annotated[
        str, typer.Option(help="Name of the experiment.")
    ],
    axes: Annotated[str, typer.Option(help="Axes of the data (e.g. SYX).")],
    patch_size: Annotated[
        click.Tuple,
        typer.Option(
            click_type=click.Tuple([int, int]),
            help="Size of the patches along the spatial dimensions (e.g. --patch-size 64 64).",
        ),
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size.")],
    num_epochs: Annotated[int, typer.Option(help="Number of epochs.")],
    data_type: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["tiff"]), help="Type of the data."
        ),
    ] = "tiff",
    use_augmentations: Annotated[
        bool, typer.Option(help="Whether to use augmentations.")
    ] = True,
    independent_channels: Annotated[
        bool, typer.Option(help="Whether to train all channels independently.")
    ] = True,
    use_n2v2: Annotated[bool, typer.Option(help="Whether to use N2V2")] = False,
    n_channels: Annotated[
        int, typer.Option(help="Number of channels (in and out)")
    ] = 1,
    roi_size: Annotated[
        int, typer.Option(help="N2V pixel manipulation area.")
    ] = 11,
    masked_pixel_percentage: Annotated[
        float, typer.Option(help="Percentage of pixels masked in each patch.")
    ] = 0.2,
    struct_n2v_axis: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["horizontal", "vertical", "none"])
        ),
    ] = "none",
    struct_n2v_span: Annotated[
        int, typer.Option(help="Span of the structN2V mask.")
    ] = 5,
    logger: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["wandb", "tensorboard", "none"]),
            help="Logger to use.",
        ),
    ] = "none",
    # TODO: How to address model kwargs
) -> None:
    config = create_n2v_configuration(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_augmentations=use_augmentations,
        independent_channels=independent_channels,
        use_n2v2=use_n2v2,
        n_channels=n_channels,
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        struct_n2v_axis=struct_n2v_axis,
        struct_n2v_span=struct_n2v_span,
        logger=logger,
    )
    _config_builder_exit(ctx, config)
