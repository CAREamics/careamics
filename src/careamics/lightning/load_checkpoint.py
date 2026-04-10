"""Module for loading CAREamics models and configs from a checkpoint."""

from pathlib import Path
from typing import Any

import torch

from careamics.config.configuration import AlgorithmConfig, Configuration
from careamics.config.factories.config_discriminators import instantiate_config
from careamics.config.support import SupportedAlgorithm

from .lightning_modules import CAREModule, N2VModule
from .lightning_modules.get_module import get_module_cls

CAREamicsModuleCls = type[N2VModule] | type[CAREModule]
CAREamicsModule = N2VModule | CAREModule


def load_module_from_checkpoint(checkpoint_path: Path) -> CAREamicsModule:
    """
    Load a trained CAREamics module from checkpoint.

    Automatically detects the algorithm type from the checkpoint and loads
    the appropriate module with trained weights.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the PyTorch Lightning checkpoint file.

    Returns
    -------
    CAREamicsModule
        Lightning module with loaded weights.

    Raises
    ------
    ValueError
        If the algorithm type cannot be determined from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    try:
        algorithm = checkpoint["hyper_parameters"]["algorithm_config"]["algorithm"]
        algorithm = SupportedAlgorithm(algorithm)
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"Could not determine algorithm type from checkpoint at: {checkpoint_path}"
        ) from e

    ModuleClass = get_module_cls(algorithm)
    return ModuleClass.load_from_checkpoint(checkpoint_path)


def load_config_from_checkpoint(
    checkpoint_path: Path,
) -> Configuration[AlgorithmConfig]:
    """
    Load a CAREamics config from a checkpoint.

    Some fields, if missing, will be populated by defaults. Namely, `version`,
    `training_config` and `experiment_name`.

    The default for `experiment_name` will be `"loaded_from_<checkpoint_filename>"`.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the PyTorch Lightning checkpoint file.

    Returns
    -------
    Configuration
        A CAREamics configuration object.

    Raises
    ------
    ValueErrors:
        If certain required information is not found in the checkpoint.
    """
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")

    # if careamics_info is not included (i.e. it was saved with the lightning API)
    # then version and training_config will be the default from the pydantic models.
    careamics_info = checkpoint.get(
        "careamics_info", {"experiment_name": _create_loaded_exp_name(checkpoint_path)}
    )

    # --- alg config
    hparams_key = "hyper_parameters"
    try:
        algorithm_config: dict[str, Any] = checkpoint[hparams_key]["algorithm_config"]
    except (KeyError, IndexError) as e:
        raise ValueError(
            "Could not determine a CAREamics supported algorithm from the provided "
            f"checkpoint at: {checkpoint_path}."
        ) from e

    # --- data config
    data_hparams_key = "datamodule_hyper_parameters"
    try:
        data_config: dict[str, Any] = checkpoint[data_hparams_key]["data_config"]
    except (KeyError, IndexError) as e:
        raise ValueError(
            "Could not determine the data configuration from the provided "
            f"checkpoint at: {checkpoint_path}."
        ) from e

    config_dict = {
        "algorithm_config": algorithm_config,
        "data_config": data_config,
    }
    config_dict.update(careamics_info)
    return instantiate_config(config_dict)


def _create_loaded_exp_name(checkpoint_path: Path):
    """
    Create an experiment name for when loading from a checkpoint.

    This should only be used if a checkpoint does not have a saved experiment name.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the PyTorch Lightning checkpoint file.

    Returns
    -------
    str
        The generated name will be `"loaded_from_<checkpoint_file_name>"`.
    """
    return f"loaded_from_{checkpoint_path.stem}"
