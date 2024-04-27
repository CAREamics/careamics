"""Utility functions to load pretrained models."""
from pathlib import Path
from typing import Tuple, Union

from torch import __version__, load, save

from careamics.config import Configuration
from careamics.lightning_module import CAREamicsKiln
from careamics.utils import check_path_exists


def load_pretrained(path: Union[Path, str]) -> Tuple[CAREamicsKiln, Configuration]:
    """
    Load a pretrained model from a checkpoint or a BioImage Model Zoo model.

    Expected formats are .ckpt, .zip, .pth or .pt files.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the pretrained model.

    Returns
    -------
    CAREamicsKiln
        CAREamics model loaded from the checkpoint.

    Raises
    ------
    ValueError
        If the model format is not supported.
    """
    path = check_path_exists(path)

    if path.suffix == ".ckpt":
        # load checkpoint
        checkpoint: dict = load(path)

        # attempt to load algorithm parameters
        try:
            cfg_dict = checkpoint["hyper_parameters"]
        except KeyError as e:
            raise ValueError(
                f"Invalid checkpoint file. No `hyper_parameters` found in the "
                f"checkpoint: {checkpoint.keys()}"
            ) from e

        model = _load_from_checkpoint(path)

        return model, Configuration(**cfg_dict)

    elif path.suffix == ".zip":
        return _load_from_bmz(path)
    else:
        raise ValueError(
            f"Invalid model format. Expected .ckpt or .zip, " f"got {path.suffix}."
        )


def _load_from_checkpoint(path: Union[Path, str]) -> CAREamicsKiln:
    """
    Load a model from a checkpoint.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the checkpoint.

    Returns
    -------
    CAREamicsKiln
        CAREamics model loaded from the checkpoint.
    """
    return CAREamicsKiln.load_from_checkpoint(path)


def _load_from_torch_dict(
    path: Union[Path, str]
) -> Tuple[CAREamicsKiln, Configuration]:
    """
    Load a model from a PyTorch dictionary.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the PyTorch dictionary.

    Returns
    -------
    Tuple[CAREamicsKiln, Configuration]
        CAREamics model and Configuration loaded from the BioImage Model Zoo.
    """
    raise NotImplementedError(
        "Loading a model from a PyTorch dictionary is not implemented yet."
    )


def _load_from_bmz(
    path: Union[Path, str],
) -> Tuple[CAREamicsKiln, Configuration]:
    """
    Load a model from BioImage Model Zoo.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the BioImage Model Zoo model.

    Returns
    -------
    Tuple[CAREamicsKiln, Configuration]
        CAREamics model and Configuration loaded from the BioImage Model Zoo.

    Raises
    ------
    NotImplementedError
        If the method is not implemented yet.
    """
    raise NotImplementedError(
        "Loading a model from BioImage Model Zoo is not implemented yet."
    )

    # load BMZ archive
    # extract model and call _load_from_torch_dict


def export_state_dict(model: CAREamicsKiln, path: Union[Path, str]) -> Path:
    """
    Export the model state dictionary to a file.

    Parameters
    ----------
    model : CAREamicsKiln
        CAREamics model to export.
    path : Union[Path, str]
        Path to the file where to save the model state dictionary.

    Returns
    -------
    Path
        Path to the saved model state dictionary.
    """
    path = Path(path)

    # make sure it has the correct suffix
    if path.suffix not in ".pth":
        path = path.with_suffix(".pth")

    # save model state dictionary
    save(model.model.state_dict(), path)

    return path