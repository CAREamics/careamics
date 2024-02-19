from fnmatch import fnmatch
from pathlib import Path
from typing import List, Tuple, Union

from tifffile import TiffFile, TiffFileError

from careamics.config.support import SupportedData
from careamics.utils.logging import get_logger


logger = get_logger(__name__)

# TODO no difference with the normal way to get file size
def _approximate_tiff_file_size(filename: Path) -> int:
    """
    Approximate TIFF file size in MB.

    Parameters
    ----------
    filename : Path
        Path to a TIFF file.

    Returns
    -------
    int
        Approximate file size in MB.
    """
    try:
        pointer = TiffFile(filename)
        return pointer.filehandle.size / 1024**2
    except (TiffFileError, StopIteration, FileNotFoundError):
        logger.warning(f"File {filename} is not a valid tiff file or is empty.")
        return 0


def get_files_size(
        files: List[Path]
    ) -> int:
    """
    Get files size in MB.

    Parameters
    ----------
    files : List[Path]
        List of files.

    Returns
    -------
    int
        Total size of the files in MB.
    """
    return sum(f.stat().st_size / 1024**2 for f in files)


def list_files(
    data_path: Union[str, Path],
    data_type: Union[str, SupportedData],
) -> List[Path]:
    """Creates a list of paths to source tiff files from path string.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the folder containing the data.
    data_type : Union[str, SupportedData]
        One of the supported data type (e.g. tif, custom).

    Returns
    -------
    List[Path]
        List of pathlib.Path objects.

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
    extension = SupportedData.get_extension(data_type)
    
    if data_path.is_dir():
        # search recursively the path for files with the extension
        files = sorted([f for f in data_path.rglob(extension)])
    else:
        # raise error if it has the wrong extension
        if not fnmatch(data_path, extension):
            raise ValueError(
                f"File {data_path} does not match the requested extension "
                f"\"{extension}\"."
            )

        # save in list
        files = [data_path]
    
    # raise error if no files were found
    if len(files) == 0:
        raise ValueError(
            f"Data path {data_path} is empty or files with extension \"{extension}\" "
            f"were not found."
        )

    return files


def validate_source_target_files(
        src_files: List[Path], tar_files: List[Path]
    ) -> None:
    """
    Validate source and target path lists.

    The two lists should have the same number of files, and the filenames should match.

    Parameters
    ----------
    src_files : List[Path]
        List of source files.
    tar_files : List[Path]
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
    src_names = set([f.name for f in src_files])
    tar_names = set([f.name for f in tar_files])
    difference = src_names.symmetric_difference(tar_names)
    
    if len(difference) > 0:
        raise ValueError(
            f"Source and target files have different names: {difference}."
        )
    