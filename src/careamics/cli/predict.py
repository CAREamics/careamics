#!/usr/bin/env python3

"""Script for the CLI. Runs prediction."""

import argparse
from pathlib import Path

from careamics import CAREamist
from careamics.cli.cli_utils import without_none
from careamics.config.support import SupportedData


def add_args_predict(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the predict subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which the predict arguments are added.
    """
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
