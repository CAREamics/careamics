import logging

import torch

# TODO instantiate logging?


def get_device():
    if torch.cuda.is_available():
        logging.info("CUDA available. Using GPU.")
        device = torch.device("cuda")
    else:
        logging.info("CUDA not available. Using CPU.")
        device = torch.device("cpu")
    return device
