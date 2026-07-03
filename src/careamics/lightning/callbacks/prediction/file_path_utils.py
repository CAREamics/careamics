"""Module containing file path utilities for `WriteStrategy` to use."""

from pathlib import Path

# Extensions that CAREamics recognises as input image files (see `SupportedData`).
KNOWN_INPUT_EXTENSIONS = frozenset({".tif", ".tiff", ".czi", ".zarr"})


def create_write_file_path(
    dirpath: Path, file_path: Path, write_extension: str, postfix: str = ""
) -> Path:
    """
    Create the file name for the output file.

    Takes the original file path, changes the directory to `dirpath` and changes
    the extension to `write_extension`.

    Only a recognised input extension (see `KNOWN_INPUT_EXTENSIONS`) is replaced;
    any other dotted segments in the name are preserved. This avoids truncating
    names such as ``experiment.0001`` or ``cells.pos0.tif``, which would otherwise
    collide and raise a ``FileExistsError`` when written.

    Parameters
    ----------
    dirpath : pathlib.Path
        The output directory to write file to.
    file_path : pathlib.Path
        The original file path.
    write_extension : str
        The extension that output files should have.
    postfix : str, default=""
        Appends to filename before extension.

    Returns
    -------
    Path
        The output file path.
    """
    file_path = Path(file_path)  # as a guard against str input
    if file_path.suffix.lower() in KNOWN_INPUT_EXTENSIONS:
        stem = file_path.stem
    else:
        stem = file_path.name
    file_name = f"{stem}{postfix}{write_extension}"
    return dirpath / file_name
