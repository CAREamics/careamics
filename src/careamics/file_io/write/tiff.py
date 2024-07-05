"""Write tiff function."""

from fnmatch import fnmatch
from pathlib import Path

import tifffile
from numpy.typing import NDArray

from careamics.config.support import SupportedData


def write_tiff(file_path: Path, img: NDArray, *args, **kwargs) -> None:
    # TODO: add link to tiffile docs for args kwrgs?
    """
    Write tiff files.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to file.
    img : numpy.ndarray
        Image data to save.
    *args
        Positional arguments passed to `tifffile.imwrite`.
    **kwargs
        Keyword arguments passed to `tifffile.imwrite`.

    Raises
    ------
    ValueError
        When the file extension of `file_path` does not match the Unix shell-style
        pattern '*.tif*'.
    """
    if not fnmatch(
        file_path.suffix, SupportedData.get_extension_pattern(SupportedData.TIFF)
    ):
        raise ValueError(
            f"Unexpected extension '{file_path.suffix}' for save file type 'tiff'."
        )
    tifffile.imwrite(file_path, img, *args, **kwargs)
