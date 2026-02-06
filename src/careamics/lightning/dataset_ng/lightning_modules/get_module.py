"""Factory functions for lightning modules."""

from pathlib import Path

import torch

from careamics.config import CAREAlgorithm, N2VAlgorithm
from careamics.config.algorithms.unet_algorithm_config import UNetBasedAlgorithm
from careamics.config.support import SupportedAlgorithm

from .care_module import CAREModule
from .n2v_module import N2VModule

CAREamicsModuleCls = type[N2VModule] | type[CAREModule]
CAREamicsModule = N2VModule | CAREModule


# TODO: update to accept all algorithm configs
def create_module(algorithm_config: UNetBasedAlgorithm) -> CAREamicsModule:
    """
    Initialize the correct Lightning module from an algorithm config.

    Parameters
    ----------
    algorithm_config : UNetBasedAlgorithm
        The pydantic model with algorithm specific parameters.

    Returns
    -------
    CAREamicsModule
        A lightning module for running one of the algorithms supported by CAREamics.

    Raises
    ------
    NotImplementedError
        If the chosen algorithm is not yet supported.
    """
    if isinstance(algorithm_config, CAREAlgorithm):
        return CAREModule(algorithm_config)
    elif isinstance(algorithm_config, N2VAlgorithm):
        return N2VModule(algorithm_config)
    else:
        algorithm = algorithm_config.algorithm
        raise NotImplementedError(
            f"Support for {algorithm} has not been implemented yet."
        )


def get_module_cls(algorithm: SupportedAlgorithm) -> CAREamicsModuleCls:
    """
    Get the lightning module class for the specified `algorithm`.

    Parameters
    ----------
    algorithm : SupportedAlgorithm
        One of the algorithms supported by CAREamics, e.g. `"n2v"`.

    Returns
    -------
    CAREamicsModuleCls
        A Lightning module class for running the specified `algorithm`.

    Raises
    ------
    NotImplementedError
        If the chosen algorithm is not get supported.
    """
    match algorithm:
        case SupportedAlgorithm.CARE:
            return CAREModule
        case SupportedAlgorithm.N2V:
            return N2VModule
        case _:
            raise NotImplementedError(
                f"Support for {algorithm.value} has not been implemented yet."
            )


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
