import logging
import os
from typing import Optional

import torch


def get_device() -> torch.device:
    """Selects the device to use for training."""
    if torch.cuda.is_available():
        logging.info("CUDA available. Using GPU.")
        device = torch.device("cuda")
    else:
        logging.info("CUDA not available. Using CPU.")
        device = torch.device("cpu")
    return device


def setup_cudnn_reproducibility(
    deterministic: Optional[bool] = None, benchmark: Optional[bool] = None
) -> None:
    """Prepares CuDNN benchmark and sets it to be deterministic/non-deterministic mode.

    https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking.

    Args:
        deterministic: deterministic mode if running in CuDNN backend.
        benchmark: If ``True`` use CuDNN heuristics to figure out
            which algorithm will be most performant
            for your model architecture and input.
            Setting it to ``False`` may slow down your training.
    """
    if torch.cuda.is_available():
        if deterministic is None:
            deterministic = os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
        torch.backends.cudnn.deterministic = deterministic

        if benchmark is None:
            benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        torch.backends.cudnn.benchmark = benchmark
