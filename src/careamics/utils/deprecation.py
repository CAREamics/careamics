"""Deprecation utilities."""

import functools
import warnings
from collections.abc import Callable
from typing import Any


# TODO useful until py3.13 which has warnings.deprecated decorator
def deprecated(msg: str = "This function is deprecated") -> Callable:
    """Decorator to mark functions as deprecated.

    Parameters
    ----------
    msg : str
        The deprecation message to display when the function is called.

    Returns
    -------
    Callable
        The decorator that marks the function as deprecated.
    """

    def decorator(func: Callable) -> Callable:
        """Decorator.

        Parameters
        ----------
        func : Callable
            The function to be decorated.

        Returns
        -------
        Callable
            The wrapped function.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper.

            Parameters
            ----------
            *args : Any
                Positional arguments for the original function.
            **kwargs : Any
                Keyword arguments for the original function.

            Returns
            -------
            Any
                The return value of the original function.
            """
            warnings.warn(
                f"{func.__name__} is deprecated: {msg}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
