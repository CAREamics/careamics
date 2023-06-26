from typing import Callable, Dict

from .losses import n2v_loss


def create_loss_function(config: Dict) -> Callable:
    """Builds a model based on the model_name or load a checkpoint.

    _extended_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    """
    loss_type = config.algorithm.loss
    if loss_type[0] == "n2v":
        loss_function = n2v_loss
    # TODO rewrite this ugly bullshit. registry,etc!
    # loss_func = getattr(sys.__name__, loss_type)
    return loss_function
