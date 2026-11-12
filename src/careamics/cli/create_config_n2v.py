#!/usr/bin/env python3

"""Script for the CLI. Creates a N2V configuration file."""

import argparse
import os
from pathlib import Path

from careamics.cli.cli_utils import add_subcommand_parser, without_none
from careamics.config import create_n2v_config
from careamics.config.support import SupportedData
from careamics.config.utils.configuration_io import save_configuration


def register_config_n2v_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the config-n2v subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers to which the config-n2v parser is added.
    """
    parser = add_subcommand_parser(
        subparsers, "config-n2v", help="Create an N2V configuration."
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to save the output files."
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="name of the experiment."
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
    parser.add_argument("--use_n2v2", action="store_true", default=None)
    parser.add_argument("--n_channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)


def create_config_n2v(args: argparse.Namespace) -> None:
    """Create and save an N2V configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    config = create_n2v_config(
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
                "use_n2v2": args.use_n2v2,
                "n_channels": args.n_channels,
                "seed": args.seed,
            }
        ),
    )
    save_configuration(config, os.path.join(args.output_path, "careamics.yaml"))
