"""File utilities."""

from pathlib import Path

import numpy as np

from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def create_write_file_path(
    dirpath: Path, file_path: Path, write_extension: str
) -> Path:
    """
    Create the file name for the output file.

    Takes the original file path, changes the directory to `dirpath` and changes
    the extension to `write_extension`.

    Parameters
    ----------
    dirpath : pathlib.Path
        The output directory to write file to.
    file_path : pathlib.Path
        The original file path.
    write_extension : str
        The extension that output files should have.

    Returns
    -------
    Path
        The output file path.
    """
    file_name = Path(file_path.stem).with_suffix(write_extension)
    file_path = dirpath / file_name
    return file_path


def get_files_size(files: list[Path]) -> float:
    """Get files size in MB.

    Parameters
    ----------
    files : list of pathlib.Path
        List of files.

    Returns
    -------
    float
        Total size of the files in MB.
    """
    return np.sum([f.stat().st_size / 1024**2 for f in files])
