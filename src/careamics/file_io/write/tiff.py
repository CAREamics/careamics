"""Write tiff function."""

from pathlib import Path

import tifffile
from numpy.typing import NDArray


def write_tiff(file_path: Path, img: NDArray, *args, **kwargs) -> None:
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
    """
    tifffile.imwrite(file_path, img, *args, **kwargs)
