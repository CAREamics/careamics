#!/usr/bin/env python3

"""Script for the CLI. Runs prediction."""

import argparse
from pathlib import Path
from typing import Literal

from careamics import CAREamist
from careamics.cli.cli_utils import Subcommand, SubcommandFunc, without_none
from careamics.config.support import SupportedData


def register_command_predict(
    subparsers: argparse._SubParsersAction,
    func_registry: dict[Subcommand, SubcommandFunc],
) -> None:
    """Register the predict subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers to which the predict parser is added.
    func_registry : dict[Subcommand, SubcommandFunc]
        The registry that will be updated to contain the function that should be called
        by the subcommand.
    """
    command_name: Literal["predict"] = "predict"
    func_registry[command_name] = predict

    parser = subparsers.add_parser(command_name, help="Run prediction.")
    parser.add_argument(
        "--ckpt_path", type=Path, required=True, help="Path to pretrained model"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to folder with images"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the output files.",
        default=Path("."),
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--tile_size", nargs="+", type=int, default=None, help="2D or 3D"
    )
    parser.add_argument(
        "--tile_overlap", nargs="+", type=int, default=None, help="2D or 3D"
    )
    parser.add_argument("--axes", type=str, default=None)
    parser.add_argument("--data_type", type=SupportedData, default=None)
    parser.add_argument(
        "--write_type", choices=["tiff", "zarr", "custom"], default=None
    )


def predict(args: argparse.Namespace) -> None:
    """Run prediction from the command line.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    careamics_pretrained = CAREamist(checkpoint_path=args.ckpt_path)
    careamics_pretrained.predict_to_disk(
        pred_data=args.data,
        prediction_dir=args.output_path / "predictions",
        **without_none(
            {
                "batch_size": args.batch_size,
                "tile_size": args.tile_size,
                "tile_overlap": args.tile_overlap,
                "axes": args.axes,
                "data_type": (
                    args.data_type if args.data_type is None else args.data_type.value
                ),
                "write_type": args.write_type,
            }
        ),
    )
