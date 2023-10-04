"""
Convenience functions using torch.

These functions are used to control certain aspects and behaviours of PyTorch.
"""
import logging
import os
import sys
from typing import Optional

import torch


def get_device() -> torch.device:
    """
    Select the device to use for training.

    Returns
    -------
    torch.device
        CUDA or CPU device, depending on availability of CUDA devices.
    """
    if torch.cuda.is_available():
        logging.info("CUDA available. Using GPU.")
        device = torch.device("cuda")
    else:
        logging.info("CUDA not available. Using CPU.")
        device = torch.device("cpu")
    return device


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Torch.compile wrapper.

    Parameters
    ----------
    model : torch.nn.Module
        Model.

    Returns
    -------
    torch.nn.Module
        Compiled model if compile is available, the model itself otherwise.
    """
    if hasattr(torch, "compile") and sys.version_info.minor <= 9:
        return torch.compile(model, mode="reduce-overhead")
    else:
        return model


def setup_cudnn_reproducibility(
    deterministic: Optional[bool] = None, benchmark: Optional[bool] = None
) -> None:
    """
    Prepare CuDNN benchmark and sets it to be deterministic/non-deterministic mode.

    https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking.

    Parameters
    ----------
    deterministic : Optional[bool]
        Deterministic mode if running in CuDNN backend.
    benchmark : Optional[bool]
        If True, uses CuDNN heuristics to figure out which algorithm will be most
        performant for your model architecture and input. False may slow down training.
    """
    if torch.cuda.is_available():
        if deterministic is None:
            deterministic = os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
        torch.backends.cudnn.deterministic = deterministic

        if benchmark is None:
            benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        torch.backends.cudnn.benchmark = benchmark