"""Module to get write functions."""

from pathlib import Path
from typing import Literal, Protocol

from numpy.typing import NDArray

from careamics.config.support import SupportedData

from .tiff import write_tiff

SupportedWriteType = Literal["tiff", "custom"]


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


def get_write_func(data_type: SupportedWriteType) -> WriteFunc:
    """
    Get the write function for the data type.

    Parameters
    ----------
    data_type : {"tiff", "custom"}
        Data type.

    Returns
    -------
    callable
        Write function.
    """
    # error raised here if not supported
    data_type_ = SupportedData(data_type)  # new variable for mypy
    # error if no write func.
    if data_type_ not in WRITE_FUNCS:
        raise NotImplementedError(f"No write function for data type '{data_type}'.")

    return WRITE_FUNCS[data_type_]
