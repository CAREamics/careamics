from pathlib import Path


def paths_to_str(dictionary: dict) -> dict:
    """Replace Path objects in a dictionary by str.

    Parameters
    ----------
    dictionary : dict
        Dictionary to modify.

    Returns
    -------
    dict
        Modified dictionary.
    """ """
    """
    for k in dictionary.keys():
        if isinstance(dictionary[k], Path):
            dictionary[k] = str(dictionary[k])

    return dictionary


def remove_default_optionals(dictionary: dict, default: dict) -> dict:
    """Remove default arguments from a dictionary.

    Removes arguments if they are equal to the provided default ones in place

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
