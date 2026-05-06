"""
Convenience functions using torch.

These functions are used to control certain aspects and behaviours of PyTorch.
"""

import platform

import torch


def get_device() -> str:
    """
    Get the device on which operations take place.

    Returns
    -------
    str
        The device on which operations take place, e.g. "cuda", "cpu" or "mps".
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and platform.processor() in (
        "arm",
        "arm64",
    ):
        device = "mps"
    else:
        device = "cpu"

    return device
