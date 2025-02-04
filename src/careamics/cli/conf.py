"""Configuration building convenience functions for the CAREamics CLI."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import click
import typer
import yaml

from ..config import (
    Configuration,
    create_care_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
    save_configuration,
)
from .utils import handle_2D_3D_callback

WORK_DIR = Path.cwd()

app = typer.Typer()


def _config_builder_exit(ctx: typer.Context, config: Configuration) -> None:
    """
    Function to be called at the end of a CLI configuration builder.

    Saves the `config` object and performs other functionality depending on the command
    context.

    Parameters
    ----------
    ctx : typer.Context
        Typer Context.
    config : Configuration
        CAREamics configuration.
    """
    conf_path = (ctx.obj.dir / ctx.obj.name).with_suffix(".yaml")
    save_configuration(config, conf_path)
    if ctx.obj.print:
        print(yaml.dump(config.model_dump(), indent=2))


@dataclass
class ConfOptions:
    """Data class for containing CLI `conf` command option values."""

    dir: Path
    name: str
    force: bool
    print: bool


@app.callback()
def conf_options(  # numpydoc ignore=PR01
    ctx: typer.Context,
    dir: Annotated[
        Path,
        typer.Option(
            "--dir", "-d", exists=True, help="Directory to save the config file to."
        ),
    ] = WORK_DIR,
    name: Annotated[
        str, typer.Option("--name", "-n", help="The config file name.")
    ] = "config",
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Whether to overwrite existing config files."
        ),
    ] = False,
    print: Annotated[
        bool,
        typer.Option(
            "--print",
            "-p",
            help="Whether to print the config file to the console.",
        ),
    ] = False,
):
    """Build and save CAREamics configuration files."""
    # Callback is called still on --help command
    # If a config exists it will complain that you need to use the -f flag
    if "--help" in sys.argv:
        return
    conf_path = (dir / name).with_suffix(".yaml")
    if conf_path.exists() and not force:
        raise FileExistsError(f"To overwrite '{conf_path}' use flag --force/-f.")

    ctx.obj = ConfOptions(dir, name, force, print)


# TODO: Need to decide how to parse model kwargs
#   - Could be json style string to be loaded as dict e.g. {"depth": 3}
#        - Cons: Annoying to type, easily have syntax errors
#   - Could parse all unknown options as model kwargs
#       - Cons: There could be argument name clashes


@app.command()
def care(  # numpydoc ignore=PR01
    ctx: typer.Context,
    experiment_name: Annotated[str, typer.Option(help="Name of the experiment.")],
    axes: Annotated[str, typer.Option(help="Axes of the data (e.g. SYX).")],
    patch_size: Annotated[
        click.Tuple,
        typer.Option(
            help=(
                "Size of the patches along the spatial dimensions (if the data "
                "is not 3D pass the last value as -1 e.g. --patch-size 64 64 -1)."
            ),
            click_type=click.Tuple([int, int, int]),
            callback=handle_2D_3D_callback,
        ),
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size.")],
    num_epochs: Annotated[int, typer.Option(help="Number of epochs.")],
    data_type: Annotated[
        click.Choice,
        typer.Option(click_type=click.Choice(["tiff"]), help="Type of the data."),
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
        Optional[int], typer.Option(help="Number of channels in")
    ] = None,
    n_channels_out: Annotated[
        Optional[int], typer.Option(help="Number of channels out")
    ] = None,
    logger: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["wandb", "tensorboard", "none"]),
            help="Logger to use.",
        ),
    ] = "none",
    # TODO: How to address model kwargs
) -> None:
    """
    Create a configuration for training CARE.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` to the number of
    channels. Likewise, if you set the number of channels, then "C" must be present in
    `axes`.

    To set the number of output channels, use the `n_channels_out` parameter. If it is
    not specified, it will be assumed to be equal to `n_channels_in`.

    By default, all channels are trained together. To train all channels independently,
    set `independent_channels` to True.

    By setting `use_augmentations` to False, the only transformation applied will be
    normalization.
    """
    config = create_care_configuration(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        # TODO: fix choosing augmentations
        augmentations=None if use_augmentations else [],
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger,
    )
    _config_builder_exit(ctx, config)


@app.command()
def n2n(  # numpydoc ignore=PR01
    ctx: typer.Context,
    experiment_name: Annotated[str, typer.Option(help="Name of the experiment.")],
    axes: Annotated[str, typer.Option(help="Axes of the data (e.g. SYX).")],
    patch_size: Annotated[
        click.Tuple,
        typer.Option(
            help=(
                "Size of the patches along the spatial dimensions (if the data "
                "is not 3D pass the last value as -1 e.g. --patch-size 64 64 -1)."
            ),
            click_type=click.Tuple([int, int, int]),
            callback=handle_2D_3D_callback,
        ),
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size.")],
    num_epochs: Annotated[int, typer.Option(help="Number of epochs.")],
    data_type: Annotated[
        click.Choice,
        typer.Option(click_type=click.Choice(["tiff"]), help="Type of the data."),
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
        Optional[int], typer.Option(help="Number of channels in")
    ] = None,
    n_channels_out: Annotated[
        Optional[int], typer.Option(help="Number of channels out")
    ] = None,
    logger: Annotated[
        click.Choice,
        typer.Option(
            click_type=click.Choice(["wandb", "tensorboard", "none"]),
            help="Logger to use.",
        ),
    ] = "none",
    # TODO: How to address model kwargs
) -> None:
    """
    Create a configuration for training Noise2Noise.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels` to the number of
    channels. Likewise, if you set the number of channels, then "C" must be present in
    `axes`.

    By default, all channels are trained together. To train all channels independently,
    set `independent_channels` to True.

    By setting `use_augmentations` to False, the only transformation applied will be
    normalization.
    """
    config = create_n2n_configuration(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        # TODO: fix choosing augmentations
        augmentations=None if use_augmentations else [],
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger,
    )
    _config_builder_exit(ctx, config)


@app.command()
def n2v(  # numpydoc ignore=PR01
    ctx: typer.Context,
    experiment_name: Annotated[str, typer.Option(help="Name of the experiment.")],
    axes: Annotated[str, typer.Option(help="Axes of the data (e.g. SYX).")],
    patch_size: Annotated[
        click.Tuple,
        typer.Option(
            help=(
                "Size of the patches along the spatial dimensions (if the data "
                "is not 3D pass the last value as -1 e.g. --patch-size 64 64 -1)."
            ),
            click_type=click.Tuple([int, int, int]),
            callback=handle_2D_3D_callback,
        ),
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size.")],
    num_epochs: Annotated[int, typer.Option(help="Number of epochs.")],
    data_type: Annotated[
        click.Choice,
        typer.Option(click_type=click.Choice(["tiff"]), help="Type of the data."),
    ] = "tiff",
    use_augmentations: Annotated[
        bool, typer.Option(help="Whether to use augmentations.")
    ] = True,
    independent_channels: Annotated[
        bool, typer.Option(help="Whether to train all channels independently.")
    ] = True,
    use_n2v2: Annotated[bool, typer.Option(help="Whether to use N2V2")] = False,
    n_channels: Annotated[
        Optional[int], typer.Option(help="Number of channels (in and out)")
    ] = None,
    roi_size: Annotated[int, typer.Option(help="N2V pixel manipulation area.")] = 11,
    masked_pixel_percentage: Annotated[
        float, typer.Option(help="Percentage of pixels masked in each patch.")
    ] = 0.2,
    struct_n2v_axis: Annotated[
        click.Choice,
        typer.Option(click_type=click.Choice(["horizontal", "vertical", "none"])),
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
    """
    Create a configuration for training Noise2Void.

    N2V uses a UNet model to denoise images in a self-supervised manner. To use its
    variants structN2V and N2V2, set the `struct_n2v_axis` and `struct_n2v_span`
    (structN2V) parameters, or set `use_n2v2` to True (N2V2).

    N2V2 modifies the UNet architecture by adding blur pool layers and removes the skip
    connections, thus removing checkboard artefacts. StructN2V is used when vertical
    or horizontal correlations are present in the noise; it applies an additional mask
    to the manipulated pixel neighbors.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels` to the number of
    channels.

    By default, all channels are trained independently. To train all channels together,
    set `independent_channels` to False.

    By setting `use_augmentations` to False, the only transformations applied will be
    normalization and N2V manipulation.

    The `roi_size` parameter specifies the size of the area around each pixel that will
    be manipulated by N2V. The `masked_pixel_percentage` parameter specifies how many
    pixels per patch will be manipulated.

    The parameters of the UNet can be specified in the `model_kwargs` (passed as a
    parameter-value dictionary). Note that `use_n2v2` and 'n_channels' override the
    corresponding parameters passed in `model_kwargs`.

    If you pass "horizontal" or "vertical" to `struct_n2v_axis`, then structN2V mask
    will be applied to each manipulated pixel.
    """
    config = create_n2v_configuration(
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        # TODO: fix choosing augmentations
        augmentations=None if use_augmentations else [],
        independent_channels=independent_channels,
        use_n2v2=use_n2v2,
        n_channels=n_channels,
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        struct_n2v_axis=struct_n2v_axis,
        struct_n2v_span=struct_n2v_span,
        logger=logger,
        # TODO: Model kwargs
    )
    _config_builder_exit(ctx, config)
