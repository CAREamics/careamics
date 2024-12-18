"""File utilities."""

from fnmatch import fnmatch
from pathlib import Path
from typing import Union

import numpy as np

from careamics.config.support import SupportedData
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


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


def list_files(
    data_path: Union[str, Path],
    data_type: Union[str, SupportedData],
    extension_filter: str = "",
) -> list[Path]:
    """List recursively files in `data_path` and return a sorted list.

    If `data_path` is a file, its name is validated against the `data_type` using
    `fnmatch`, and the method returns `data_path` itself.

    By default, if `data_type` is equal to `custom`, all files will be listed. To
    further filter the files, use `extension_filter`.

    `extension_filter` must be compatible with `fnmatch` and `Path.rglob`, e.g. "*.npy"
    or "*.czi".

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the folder containing the data.
    data_type : Union[str, SupportedData]
        One of the supported data type (e.g. tif, custom).
    extension_filter : str, optional
        Extension filter, by default "".

    Returns
    -------
    list[Path]
        list of pathlib.Path objects.

    Raises
    ------
    FileNotFoundError
        If the data path does not exist.
    ValueError
        If the data path is empty or no files with the extension were found.
    ValueError
        If the file does not match the requested extension.
    """
    # convert to Path
    data_path = Path(data_path)

    # raise error if does not exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist.")

    # get extension compatible with fnmatch and rglob search
    extension = SupportedData.get_extension_pattern(data_type)

    if data_type == SupportedData.CUSTOM and extension_filter != "":
        extension = extension_filter

    # search recurively
    if data_path.is_dir():
        # search recursively the path for files with the extension
        files = sorted(data_path.rglob(extension))
    else:
        # raise error if it has the wrong extension
        if not fnmatch(str(data_path.absolute()), extension):
            raise ValueError(
                f"File {data_path} does not match the requested extension "
                f'"{extension}".'
            )

        # save in list
        files = [data_path]

    # raise error if no files were found
    if len(files) == 0:
        raise ValueError(
            f'Data path {data_path} is empty or files with extension "{extension}" '
            f"were not found."
        )

    return files


def validate_source_target_files(src_files: list[Path], tar_files: list[Path]) -> None:
    """
    Validate source and target path lists.

    The two lists should have the same number of files, and the filenames should match.

    Parameters
    ----------
    src_files : list of pathlib.Path
        List of source files.
    tar_files : list of pathlib.Path
        List of target files.

    Raises
    ------
    ValueError
        If the number of files in source and target folders is not the same.
    ValueError
        If some filenames in Train and target folders are not the same.
    """
    # check equal length
    if len(src_files) != len(tar_files):
        raise ValueError(
            f"The number of source files ({len(src_files)}) is not equal to the number "
            f"of target files ({len(tar_files)})."
        )

    # check identical names
    src_names = {f.name for f in src_files}
    tar_names = {f.name for f in tar_files}
    difference = src_names.symmetric_difference(tar_names)

    if len(difference) > 0:
        raise ValueError(f"Source and target files have different names: {difference}.")
