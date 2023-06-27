from typing import Callable

from ..config import Configuration
from ..config.algorithm import LossName
from .losses import n2v_loss, pn2v_loss


def create_loss_function(config: Configuration) -> Callable:
    """Builds a model based on the model_name or load a checkpoint.

    _extended_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    """
    loss_type = config.algorithm.loss

    if len(loss_type) > 1:
        raise NotImplementedError("Multiple losses are not supported yet.")

    if loss_type[0] == LossName.n2v:
        return n2v_loss
    elif loss_type[0] == LossName.pn2v:
        return pn2v_loss
    else:
        raise NotImplementedError(f"Unknown loss ({loss_type[0]}).")
