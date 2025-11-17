"""Module containing file path utilities for `WriteStrategy` to use."""

from pathlib import Path


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
    file_path = Path(file_path)  # as a guard against str input

    file_name = Path(file_path.stem).with_suffix(write_extension)
    file_path = dirpath / file_name
    return file_path
