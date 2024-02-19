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
        _description_
    """
    # create single dispatch from the function
    dispatcher = singledispatch(method)
    
    # define a wrapper to dispatch the function based on the second argument
    def wrapper(*args, **kw):

        if len(args) < 2:
            raise ValueError(f"Missing argument to {method}.")

        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    
    # copy the original method's registered methods to the wrapper
    wrapper.register = dispatcher.register

    # update wrapper to look like the original method
    update_wrapper(wrapper, method)
    
    return wrapper
