#!/usr/bin/env python3

"""Script for the CLI. Trains a noise model."""

import argparse
from pathlib import Path

import numpy as np

from careamics.noise_model import train_noise_model


def add_args_train_noise_model(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the train-noise-model subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which the noise model training arguments are added.
    """
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the trained noise models.",
    )
    parser.add_argument(
        "--signal", type=Path, required=True, help="Path to the signal NumPy array."
    )
    parser.add_argument(
        "--observation",
        type=Path,
        required=True,
        help="Path to the observation NumPy array.",
    )
    parser.add_argument("--axes", type=str, required=True)
    parser.add_argument("--n_gaussian", type=int, default=3)
    parser.add_argument("--n_coeff", type=int, default=3)
    parser.add_argument("--min_sigma", type=float, default=125.0)
    parser.add_argument("--n_epochs", type=int, default=2000)


def run_train_noise_model(args: argparse.Namespace) -> None:
    """Train a noise model from the command line.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    signal = np.load(args.signal)
    observation = np.load(args.observation)
    train_noise_model(
        signal=signal,
        observation=observation,
        axes=args.axes,
        n_gaussian=args.n_gaussian,
        n_coeff=args.n_coeff,
        min_sigma=args.min_sigma,
        n_epochs=args.n_epochs,
        save_dir=args.output_path,
    )
