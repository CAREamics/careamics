import inspect
from functools import singledispatch, update_wrapper
from typing import Callable


def method_dispatch(method: Callable) -> Callable:
    """Single dispatch decorator for instance methods.

    Since functools.singledispatch does not support instance methods, as the dispatch
    is based on the first argument (`self` for instance methods), this decorator
    uses singledispatch to dispatch the call based on the second argument.

    (Barely) adapted from Zero Piraeus:
    https://stackoverflow.com/a/24602374

    Parameters
    ----------
    method : Callable
        Decorated method.

    Returns
    -------
    Callable
        Wrapper around the method that dispatches the call based on the second argument.
    """
    # create single dispatch from the function
    dispatcher = singledispatch(method)

    # define a wrapper to dispatch the function based on the second argument
    def wrapper(*args, **kw):
        if len(args) + len(kw) < 2:
            raise ValueError(f"Missing argument to {method}.")

        # if the second argument was passed as a keyword argument, we use its class
        # for the dispatch
        if len(args) == 1:
            # get signature of the method
            parameter = list(inspect.signature(method).parameters)[1]

            # raise error if the parameter is not in the keyword arguments
            if parameter not in kw:
                raise ValueError(f"Missing argument {parameter} to {method}.")

            return dispatcher.dispatch(kw[parameter].__class__)(*args, **kw)

        # else dispatch using the second argument
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    # copy the original method's registered methods to the wrapper
    wrapper.register = dispatcher.register

    # update wrapper to look like the original method
    update_wrapper(wrapper, method)

    return wrapper
