#!/usr/bin/env python3

"""Create an HDN configuration from command-line arguments."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from careamics.config import create_hdn_config
from careamics.config.support import SupportedData
from careamics.config.utils.configuration_io import save_configuration

from careamics.cli.cli_utils import without_none


def create_config_hdn(argv: Sequence[str] | None = None) -> None:
    """Create and save an HDN configuration.

    Parameters
    ----------
    argv : Sequence[str]
        Arguments to be parsed by argparse.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to save the output files."
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument("--data_type", type=SupportedData, required=True)
    parser.add_argument("--axes", type=str, required=True)
    parser.add_argument(
        "--patch_size", nargs="+", type=int, required=True, help="2D or 3D"
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument(
        "--augmentations",
        nargs="+",
        choices=["x_flip", "y_flip", "rotate_90"],
        default=None,
    )
    parser.add_argument("--n_val_patches", type=int, default=None)
    parser.add_argument("--use_noise_model", action="store_true", default=None)
    parser.add_argument("--supervised", action="store_true", default=None)
    parser.add_argument("--channels", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(argv)

    config = create_hdn_config(
        experiment_name=args.experiment_name,
        data_type=args.data_type.value,
        axes=args.axes,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        **without_none(
            {
                "num_epochs": args.num_epochs,
                "num_steps": args.num_steps,
                "augmentations": args.augmentations,
                "n_val_patches": args.n_val_patches,
                "use_noise_model": args.use_noise_model,
                "supervised": args.supervised,
                "channels": args.channels,
                "seed": args.seed,
            }
        ),
    )
    save_configuration(config, args.output_path / "careamics.yaml")


if __name__ == "__main__":
    create_config_hdn()
