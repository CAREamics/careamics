import logging
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import tifffile

from careamics.config.support import SupportedData
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def read_tiff(file_path: Path, *args: list, **kwargs: dict) -> np.ndarray:
    """
    Read a tiff file and return a numpy array.

    Parameters
    ----------
    file_path : Path
        Path to a file.
    axes : str
        Description of axes in format STCZYX.

    Returns
    -------
    np.ndarray
        Resulting array.

    Raises
    ------
    ValueError
        If the file failed to open.
    OSError
        If the file failed to open.
    ValueError
        If the file is not a valid tiff.
    ValueError
        If the data dimensions are incorrect.
    ValueError
        If the axes length is incorrect.
    """
    if fnmatch(file_path.suffix, SupportedData.get_extension(SupportedData.TIFF)):
        try:
            array = tifffile.imread(file_path)
        except (ValueError, OSError) as e:
            logging.exception(f"Exception in file {file_path}: {e}, skipping it.")
            raise e
    else:
        raise ValueError(f"File {file_path} is not a valid tiff.")

    # check dimensions
    # TODO or should this really be done here? probably in the LightningDataModule
    # TODO this should also be centralized somewhere else (validate_dimensions)
    if len(array.shape) < 2 or len(array.shape) > 6:
        raise ValueError(
            f"Incorrect data dimensions. Must be 2, 3 or 4 (got {array.shape} for"
            f"file {file_path})."
        )

    return array
