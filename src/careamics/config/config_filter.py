"""Convenience functions to filter dictionaries resulting from a Pydantic export."""

from pathlib import Path
from typing import Dict


def paths_to_str(dictionary: dict) -> dict:
    """
    Replace Path objects in a dictionary by str.

    Parameters
    ----------
    dictionary : dict
        Dictionary to modify.

    Returns
    -------
    dict
        Modified dictionary.
    """
    for k in dictionary.keys():
        if isinstance(dictionary[k], Path):
            dictionary[k] = str(dictionary[k])

    return dictionary


def remove_default_optionals(dictionary: Dict, default: Dict) -> None:
    """
    Remove default arguments from a dictionary.

    The method removes arguments if they are equal to the provided default ones.

    Parameters
    ----------
    dictionary : dict
        Dictionary to modify.
    default : dict
        Dictionary containing the default values.
    """
    dict_copy = dictionary.copy()
    for k in dict_copy.keys():
        if k in default.keys():
            if dict_copy[k] == default[k]:
                del dictionary[k]
