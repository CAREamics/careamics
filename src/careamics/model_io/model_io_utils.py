"""Utility functions to load pretrained models."""

from pathlib import Path

import torch

from careamics.config.configuration import Configuration
from careamics.config.support import SupportedArchitecture
from careamics.lightning.modules import CAREamicsModule, create_module
from careamics.model_io.bmz_io import load_from_bmz


def load_pretrained(
    path: Path | str,
) -> tuple[Configuration, CAREamicsModule]:
    """
    Load a pretrained model from a checkpoint or a BioImage Model Zoo model.

    Expected formats are .ckpt or .zip files.

    Parameters
    ----------
    path : Path | str
        Path to the pretrained model.

    Returns
    -------
    tuple[Configuration, CAREamicsModule]
        tuple of CAREamics model and its configuration.

    Raises
    ------
    ValueError
        If the model format is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data path {path} is incorrect or does not exist.")

    if path.suffix == ".ckpt":
        return _load_checkpoint(path)
    elif path.suffix == ".zip":
        return load_from_bmz(path)
    else:
        raise ValueError(
            f"Invalid model format. Expected .ckpt or .zip, got {path.suffix}."
        )


def _load_checkpoint(
    path: Path | str,
) -> tuple[Configuration, CAREamicsModule]:
    """
    Load a model from a checkpoint and return both model and configuration.

    Parameters
    ----------
    path : Path | str
        Path to the checkpoint.

    Returns
    -------
    tuple[Configuration, CAREamicsModule]
        tuple of CAREamics model and its configuration.

    Raises
    ------
    ValueError
        If the checkpoint file does not contain hyper parameters (configuration).
    """
    # load checkpoint
    # here we might run into issues between devices
    # see https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint: dict = torch.load(path, map_location=device)

    # attempt to load configuration
    try:
        cfg_dict = checkpoint["hyper_parameters"]
    except KeyError as e:
        raise ValueError(
            f"Invalid checkpoint file. No `hyper_parameters` found in the "
            f"checkpoint: {checkpoint.keys()}"
        ) from e

    config = Configuration(**cfg_dict)

    # create careamics lightning module
    if config.algorithm_config.model.architecture == SupportedArchitecture.UNET:
        model = create_module(config.algorithm_config)

    else:
        raise ValueError(
            f"Unsupported architecture {config.algorithm_config.model.architecture}"
        )

    # loading model state_dict
    model.model.load_state_dict(checkpoint)

    return config, model
