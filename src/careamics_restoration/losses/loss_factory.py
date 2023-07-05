from typing import Callable

from ..config import Configuration
from ..config.algorithm import Losses
from .losses import n2v_loss


def create_loss_function(config: Configuration) -> Callable:
    loss_type = config.algorithm.loss

    if loss_type == Losses.N2V:
        return n2v_loss
    else:
        raise NotImplementedError(f"Loss {loss_type} is not yet supported.")
