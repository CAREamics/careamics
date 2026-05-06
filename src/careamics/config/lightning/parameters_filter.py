"""Inspect function signatures to filter unknown parameters."""

import inspect


def get_unknown_parameters(
    func: type,
    user_params: dict,
) -> dict:
    """
    Return unknown parameters.

    Parameters
    ----------
    func : type
        Class object.
    user_params : dict
        User provided parameters.

    Returns
    -------
    dict
        Unknown parameters.
    """
    # Get the list of all default parameters
    default_params = list(inspect.signature(func).parameters.keys())

    # Check for unknown parameters
    unknown_params = set(user_params.keys()) - set(default_params)

    return {param: user_params[param] for param in unknown_params}
