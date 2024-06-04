"""Read function utilities."""

from typing import Callable, Union

from careamics.config.support import SupportedData

from .read_tiff import read_tiff


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
    if data_type == SupportedData.TIFF:
        return read_tiff
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")
