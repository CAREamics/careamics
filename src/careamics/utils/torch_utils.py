"""
Convenience functions using torch.

These functions are used to control certain aspects and behaviours of PyTorch.
"""
import logging

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


# def compile_model(model: torch.nn.Module) -> torch.nn.Module:
#     """
#     Torch.compile wrapper.

#     Parameters
#     ----------
#     model : torch.nn.Module
#         Model.

#     Returns
#     -------
#     torch.nn.Module
#         Compiled model if compile is available, the model itself otherwise.
#     """
#     if hasattr(torch, "compile") and sys.version_info.minor <= 9:
#         return torch.compile(model, mode="reduce-overhead")
#     else:
#         return model


# def seed_everything(seed: int) -> None:
#     """
#     Seed all random number generators for reproducibility.

#     Parameters
#     ----------
#     seed : int
#         Seed.
#     """
#     import random

#     import numpy as np

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def setup_cudnn_reproducibility(
#     deterministic: bool = True, benchmark: bool = True
# ) -> None:
#     """
#     Prepare CuDNN benchmark and sets it to be deterministic/non-deterministic mode.

#     Parameters
#     ----------
#     deterministic : bool
#         Deterministic mode, if running CuDNN backend.
#     benchmark : bool
#         If True, uses CuDNN heuristics to figure out which algorithm will be most
#         performant for your model architecture and input. False may slow down training
#     """
#     if torch.cuda.is_available():
#         if deterministic:
#             deterministic = os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
#         torch.backends.cudnn.deterministic = deterministic

#         if benchmark:
#             benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
#         torch.backends.cudnn.benchmark = benchmark
