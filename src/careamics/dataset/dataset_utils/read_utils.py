from typing import Callable

from careamics.config.support import SupportedData

from .read_tiff import read_tiff


def get_read_func(data_type: SupportedData) -> Callable:
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
    if data_type == SupportedData.ARRAY:
        return None
    elif data_type == SupportedData.TIFF:
        return read_tiff
    elif data_type == SupportedData.CUSTOM:
        return None
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")
