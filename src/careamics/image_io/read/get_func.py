"""Module to get read functions."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, Union

from numpy.typing import NDArray

from careamics.config.support import SupportedData

from .tiff import read_tiff


# This is very strict, function signature has to match including arg names
# See WriteFunc notes
class ReadFunc(Protocol):
    """Protocol that defines the accepted function signature for "read functions".

    Read functions take a file path and read image data into a numpy array.

    The first argument MUST be named `file_path` and be type `pathlib.Path`, any number
    of additional arguments of any type are allowed. The return type MUST be
    `numpy.typing.NDArray[Any]`. For example:

    ```python
    def read_func_example(file_path: Path, extra_arg: int) -> NDArray[Any]: ...
    ```
    """

    def __call__(self, file_path: Path, *args: Any, **kwargs: Any) -> NDArray[Any]:
        """
        Type hinted callables must match this function signature (not including self).

        Parameters
        ----------
        file_path : pathlib.Path
            Path to file.
        *args
            Other positional arguments.
        **kwargs
            Other keyword arguments.
        """
        ...


READ_FUNCS: dict[SupportedData, ReadFunc] = {
    SupportedData.TIFF: read_tiff,
}


def get_read_func(data_type: Union[str, SupportedData]) -> Callable:
    """
    Get the read function for the data type.

    Parameters
    ----------
    data_type : SupportedData
        Data type.

    Returns
    -------
    callable
        Read function.
    """
    if data_type in READ_FUNCS:
        data_type = SupportedData(data_type)  # mypy complaining about dict key type
        return READ_FUNCS[data_type]
    else:
        raise NotImplementedError(f"Data type '{data_type}' is not supported.")
