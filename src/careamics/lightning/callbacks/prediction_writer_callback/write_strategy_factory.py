"""Module containing convenience function to create `WriteStrategy`."""

from typing import Any, Optional

from careamics.config.support import SupportedData
from careamics.file_io import SupportedWriteType, WriteFunc, get_write_func

from .write_strategy import WriteImage, WriteStrategy, WriteTiles


def create_write_strategy(
    write_type: SupportedWriteType,
    tiled: bool,
    write_func: Optional[WriteFunc] = None,
    write_extension: Optional[str] = None,
    write_func_kwargs: Optional[dict[str, Any]] = None,
    write_filenames: Optional[list[str]] = None,
    n_samples_per_file: Optional[list[int]] = None,
) -> WriteStrategy:
    """
    Create a write strategy from convenient parameters.

    Parameters
    ----------
    write_type : {"tiff", "custom"}
        The data type to save as, includes custom.
    tiled : bool
        Whether the prediction will be tiled or not.
    write_func : WriteFunc, optional
        If a known `write_type` is selected this argument is ignored. For a custom
        `write_type` a function to save the data must be passed. See notes below.
    write_extension : str, optional
        If a known `write_type` is selected this argument is ignored. For a custom
        `write_type` an extension to save the data with must be passed.
    write_func_kwargs : dict of {str: any}, optional
        Additional keyword arguments to be passed to the save function.
    write_filenames : list of str, optional
        A list of filenames in the order that predictions will be written in.
    n_samples_per_file : list of int
        The number of samples in each file, (controls which samples will be grouped
        together in each file).

    Returns
    -------
    WriteStrategy
        A strategy for writing predicions.

    Notes
    -----
    The `write_func` function signature must match that of the example below
        ```
        write_func(file_path: Path, img: NDArray, *args, **kwargs) -> None: ...
        ```

    The `write_func_kwargs` will be passed to the `write_func` doing the following:
        ```
        write_func(file_path=file_path, img=img, **kwargs)
        ```
    """
    if write_func_kwargs is None:
        write_func_kwargs = {}

    write_strategy: WriteStrategy
    if not tiled:
        write_func = select_write_func(write_type=write_type, write_func=write_func)
        write_extension = select_write_extension(
            write_type=write_type, write_extension=write_extension
        )
        write_strategy = WriteImage(
            write_func=write_func,
            write_filenames=write_filenames,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
            n_samples_per_file=n_samples_per_file,
        )
    else:
        # select CacheTiles or WriteTilesZarr (when implemented)
        write_strategy = _create_tiled_write_strategy(
            write_type=write_type,
            write_func=write_func,
            write_filenames=write_filenames,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
            n_samples_per_file=n_samples_per_file,
        )

    return write_strategy


def _create_tiled_write_strategy(
    write_type: SupportedWriteType,
    write_func: Optional[WriteFunc],
    write_filenames: Optional[list[str]],
    write_extension: Optional[str],
    write_func_kwargs: dict[str, Any],
    n_samples_per_file: Optional[list[int]],
) -> WriteStrategy:
    """
    Create a tiled write strategy.

    Either `CacheTiles` for caching tiles until a whole image is predicted or
    `WriteTilesZarr` for writing tiles directly to disk.

    Parameters
    ----------
    write_type : {"tiff", "custom"}
        The data type to save as, includes custom.
    write_func : WriteFunc, optional
        If a known `write_type` is selected this argument is ignored. For a custom
        `write_type` a function to save the data must be passed. See notes below.
    write_filenames : list of str, optional
        A list of filenames in the order that predictions will be written in.
    write_extension : str, optional
        If a known `write_type` is selected this argument is ignored. For a custom
        `write_type` an extension to save the data with must be passed.
    write_func_kwargs : dict of {str: any}
        Additional keyword arguments to be passed to the save function.

    Returns
    -------
    WriteStrategy
        A strategy for writing tiled predictions.

    Raises
    ------
    NotImplementedError
        if `write_type="zarr" is chosen.
    """
    # if write_type == SupportedData.ZARR:
    #    create *args, **kwargs
    #    return WriteTilesZarr(*args, **kwargs)
    # else:
    if write_type == "zarr":
        raise NotImplementedError("Saving to zarr is not implemented yet.")
    else:
        write_func = select_write_func(write_type=write_type, write_func=write_func)
        write_extension = select_write_extension(
            write_type=write_type, write_extension=write_extension
        )
        return WriteTiles(
            write_func=write_func,
            write_filenames=write_filenames,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
            n_samples_per_file=n_samples_per_file,
        )


def select_write_func(
    write_type: SupportedWriteType, write_func: Optional[WriteFunc] = None
) -> WriteFunc:
    """
    Return a function to write images.

    If `write_type` is "custom" then `write_func`, otherwise the known write function
    is selected.

    Parameters
    ----------
    write_type : {"tiff", "custom"}
        The data type to save as, includes custom.
    write_func : WriteFunc, optional
        If a known `write_type` is selected this argument is ignored. For a custom
        `write_type` a function to save the data must be passed. See notes below.

    Returns
    -------
    WriteFunc
        A function for writing images.

    Raises
    ------
    ValueError
        If `write_type="custom"` but `write_func` has not been given.

    Notes
    -----
    The `write_func` function signature must match that of the example below
        ```
        write_func(file_path: Path, img: NDArray, *args, **kwargs) -> None: ...
        ```
    """
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
    return write_func


def select_write_extension(
    write_type: SupportedWriteType, write_extension: Optional[str] = None
) -> str:
    """
    Return an extension to add to file paths.

    If `write_type` is "custom" then `write_extension`, otherwise the known
    write extension is selected.

    Parameters
    ----------
    write_type : {"tiff", "custom"}
        The data type to save as, includes custom.
    write_extension : str, optional
        If a known `write_type` is selected this argument is ignored. For a custom
        `write_type` an extension to save the data with must be passed.

    Returns
    -------
    str
        The extension to be added to file paths.

    Raises
    ------
    ValueError
        If `self.save_type="custom"` but `save_extension` has not been given.
    """
    write_type_: SupportedData = SupportedData(write_type)  # new variable for mypy
    if write_type_ == SupportedData.CUSTOM:
        if write_extension is None:
            raise ValueError("A save extension must be provided for custom data types.")
        else:
            write_extension = write_extension
    else:
        # kind of a weird pattern -> reason to move get_extension from SupportedData
        write_extension = write_type_.get_extension(write_type_)
    return write_extension
