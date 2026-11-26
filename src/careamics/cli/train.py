#!/usr/bin/env python3

"""Script for the CLI. Runs training."""

import argparse
from pathlib import Path

from careamics import CAREamist
from careamics.cli.cli_utils import without_none
from careamics.config.utils.configuration_io import load_configuration


def add_args_train(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the train subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which the train arguments are added.
    """
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to save the output files."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to train data."
    )
    parser.add_argument(
        "--train_target", type=str, default=None, help="Path to target data."
    )
    parser.add_argument(
        "--val_data", type=str, default=None, help="Path to validation data."
    )
    parser.add_argument(
        "--val_target", type=str, default=None, help="Path to validation target data."
    )
    parser.add_argument(
        "--noise_model",
        type=str,
        default=None,
        help="Path to trained noise model, only used by MicroSpilt and HDN.",
    )
    # TODO: add filtering mask?


def train(args: argparse.Namespace) -> None:
    """Run training from the command line.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    config = load_configuration(args.config)
    # TODO: have to make sure that checkpoint callback is saving last checkpoint?
    # TODO: export last/best checkpoint name?

    careamist = CAREamist(config=config, work_dir=args.output_path)
    careamist.train(
        train_data=args.train_data,
        train_data_target=args.train_target,
        **without_none(
            {
                "val_data": args.val_data,
                "val_data_target": args.val_target,
                "noise_model": args.noise_model,
            }
        ),
    )
