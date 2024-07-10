"""Module containing convienience function to create WriteStrategy"""

from typing import Optional, Union, Any

from careamics.config.support import SupportedData
from careamics.file_io import WriteFunc, get_write_func

from .write_strategy import WriteStrategy, CacheTiles, WriteImage, WriteTilesZarr


def create_write_strategy(
    write_type: Union[SupportedData, str],
    tiled: bool,
    write_func: Optional[WriteFunc] = None,
    write_extension: Optional[str] = None,
    write_func_kwargs: Optional[dict[str, Any]] = None,
) -> WriteStrategy:

    if not tiled:
        write_strategy = WriteImage(
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
        )
    else:
        # select CacheTiles or WriteTilesZarr (when implemented)
        write_strategy = _create_tiled_write_strategy(
            write_type=write_type,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs
        )

    return write_strategy


def _create_tiled_write_strategy(
    write_type: Union[SupportedData, str],
    write_func: Optional[WriteFunc] = None,
    write_extension: Optional[str] = None,
    write_func_kwargs: Optional[dict[str, Any]] = None,
) -> WriteStrategy:
    # if write_type == SupportedData.ZARR:
    #    create *args, **kwargs
    #    return WriteTilesZarr(*args, **kwargs)
    # else:
    write_func = select_write_func(write_type=write_type, write_func=write_func)
    write_extension = select_write_extension(
        write_type=write_type, write_func=write_func
    )
    return CacheTiles(
        write_func=write_func,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
    )


def select_write_func(
    write_type: Union[SupportedData, str], write_func: Optional[WriteFunc] = None
) -> WriteFunc:
    if write_type == SupportedData.CUSTOM:
        if write_func is None:
            raise ValueError(
                "A save function must be provided for custom data types."
                # TODO: link to how save functions should be implemented
            )
        else:
            write_func = write_func
    else:
        write_func = get_write_func(write_type)


def select_write_extension(
    write_type: Union[SupportedData, str], write_extension: Optional[str] = None
) -> str:
    if write_type == SupportedData.CUSTOM:
        if write_extension is None:
            raise ValueError("A save extension must be provided for custom data types.")
        else:
            write_extension = write_extension
    else:
        # kind of a weird pattern -> reason to move get_extension from SupportedData
        write_extension = write_type.get_extension(write_type)
