############################################
#   Utility Functions
############################################

import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


class DuplicateFilter(logging.Filter):
    def filter(self, record):
        current_log = (record.module, record.levelno, record.msg)
        return True if current_log == getattr(self, "_last_log", None) else False


def set_logging(logger, default_level=logging.INFO, log_path=""):
    # TODO add log_path and level to config
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(default_level)
    logger.addFilter(DuplicateFilter())
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(1024**2 * 2), backupCount=3
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logging.addHandler(file_handler)


# TODO add EDA visualization


def config_loader(cfg_path):
    """Load a yaml config file and correct all datatypes."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    return yaml.load(Path(cfg_path).open("r"), Loader=loader)


def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean / std


def denormalize(x, mean, std):
    return x * std + mean


def get_device():
    if torch.cuda.is_available():
        logging.info("CUDA available. Using GPU.")
        device = torch.device("cuda")
    else:
        logging.info("CUDA not available. Using CPU.")
        device = torch.device("cpu")
    return device


def export_model_to_zoo(model, path):
    """Export model to Bioimage model zoo.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be exported
    path : str
        Path to save the exported model
    """
    try:
        import bioimageio.core
    except ImportError:
        raise ImportError("bioimageio.core not found. Please install it first.")
    # TODO add model name, author, etc
    bioimageio.core.build_spec.build_model()


def export_model_to_onnx(model, path):
    """Export model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be exported
    path : str
        Path to save the exported model
    """
    dummy_input = torch.randn(1, 1, 256, 256, device=getDevice())
    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
    )
