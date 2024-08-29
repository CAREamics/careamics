"""A script for serializers in the careamics package."""

import json
from typing import Union

import numpy as np
import torch


def array_to_json(arr: Union[np.ndarray, torch.Tensor]) -> str:
    """Convert an array to a list and then to a JSON string.
    
    Parameters
    ----------
    arr : Union[np.ndarray, torch.Tensor]
        Array to be serialized.
    
    Returns
    -------
    str
        JSON string representing the array.
    """
    return json.dumps(arr.tolist())