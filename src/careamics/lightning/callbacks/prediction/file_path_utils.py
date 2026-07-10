"""Module containing file path utilities for `WriteStrategy` to use."""

import os
from collections.abc import Sequence
from pathlib import Path


def common_source_base(sources: Sequence[Path | str]) -> Path | None:
    """
    Find the common parent directory of the source paths.

    Parameters
    ----------
    sources : sequence of pathlib.Path or str
        Source paths of the predictions. Array sources (``"array"``) are ignored.

    Returns
    -------
    Path or None
        The common parent, or None if it cannot be determined (fewer than two file
        sources, or no shared base such as mixed absolute and relative paths).
    """
    paths = [Path(source) for source in sources if str(source) != "array"]
    if len(paths) < 2:
        return None
    try:
        return Path(os.path.commonpath([str(path) for path in paths]))
    except ValueError:
        # paths share no common base (e.g. mixing absolute and relative paths)
        return None


def create_write_file_path(
    dirpath: Path,
    file_path: Path,
    write_extension: str,
    postfix: str = "",
    source_base: Path | None = None,
) -> Path:
    """
    Create the file name for the output file.

    If `source_base` is given, the source path relative to it is preserved under
    `dirpath`, keeping outputs unique for same-named files in different directories.

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
    source_base : pathlib.Path or None, default=None
        Common parent of all sources. If set, the structure relative to it is kept.

    Returns
    -------
    Path
        The output file path.
    """
    file_path = Path(file_path)  # as a guard against str input

    relative_path = Path(file_path.name)
    if source_base is not None:
        try:
            relative_path = file_path.relative_to(source_base)
        except ValueError:
            # source is not located under `source_base`, fall back to the file name
            relative_path = Path(file_path.name)

    file_name = f"{relative_path.stem}{postfix}{write_extension}"
    return dirpath / relative_path.parent / file_name
