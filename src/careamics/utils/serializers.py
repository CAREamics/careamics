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


def list_to_numpy(lst: np.ndarray) -> str:
    """Deserialize a list into `np.array`.
    
    NOTE: this deserializer takes a list as input, since whenever a config file is
    loaded (e.g., json, yml, pkl), the strings representing arrays are loaded as lists.

    Parameters
    ----------
    arr : list
        List with the array content to be deserialized.

    Returns
    -------
    np.ndarray
        The deserialized array.
    """
    return np.asarray(lst)


def list_to_torch(lst: list) -> str:
    """Deserialize list into `torch.Tensor`.
    
    NOTE: this deserializer takes a list as input, since whenever a config file is
    loaded (e.g., json, yml, pkl), the strings representing arrays are loaded as lists.

    Parameters
    ----------
    lst : list
        List with the array content to be deserialized.

    Returns
    -------
    torch.Tensor
        The deserialized tensor.
    """
    return torch.tensor(lst)