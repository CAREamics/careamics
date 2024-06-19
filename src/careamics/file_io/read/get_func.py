"""Read function utilities."""

from pathlib import Path
from typing import Callable, Protocol, Union

from careamics.config.support import SupportedData

from .tiff import read_tiff


# This is very strict, arguments have to be called img
# See Write func notes
class ReadFunc(Protocol):
    def __call__(self, fp: Path, *args, **kwargs) -> None: ...


READ_FUNCS: dict[SupportedData, ReadFunc] = {
    SupportedData.TIFF: read_tiff,
}


def get_read_func(data_type: Union[SupportedData, str]) -> Callable:
    """
    Get the read function for the data type.

    Parameters
    ----------
    data_type : SupportedData
        Data type.

    Returns
    -------
    Callable
        Read function.
    """
    if data_type in READ_FUNCS:
        return READ_FUNCS[data_type]
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")
