"""Module to get write functions."""

from pathlib import Path
from typing import Protocol

from numpy.typing import NDArray

from careamics.config.support import SupportedData

from .tiff import write_tiff


# This is very strict, arguments have to be called file_path & img
# Alternative? - doesn't capture *args & **kwargs
# WriteFunc = Callable[[Path, NDArray], None]
class WriteFunc(Protocol):
    """Protocol for type hinting write functions."""

    def __call__(self, file_path: Path, img: NDArray, *args, **kwargs) -> None:
        """
        Type hinted callables must match this function signature (not including self).

        Parameters
        ----------
        file_path : pathlib.Path
            Path to file.
        img : numpy.ndarray
            Image data to save.
        *args
            Other positional arguments.
        **kwargs
            Other keyword arguments.
        """


WRITE_FUNCS: dict[SupportedData, WriteFunc] = {
    SupportedData.TIFF: write_tiff,
}


def get_write_func(data_type: SupportedData) -> WriteFunc:
    """
    Get the write function for the data type.

    Parameters
    ----------
    data_type : SupportedData
        Data type.

    Returns
    -------
    callable
        Write function.
    """
    if data_type in WRITE_FUNCS:
        return WRITE_FUNCS[data_type]
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")
